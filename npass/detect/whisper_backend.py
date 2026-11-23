"""faster-whisper on the GPU, behind the `Recogniser` face.

Each `transcribe()` call is one full pass over the rolling buffer the detector hands in (Whisper is
not a streaming model): beam 1, greedy (temperature 0), no carry-over between passes, English only,
word times on. Whatever `hints` the detector passes go in as `hotwords` (it passes none by
default: see `Detector._hints` for the measurements behind that).

Model files live in `<project>/models/` and are only read from disk here. Fetching them is a
separate, deliberate step (`python -m npass.detect download`), so the running program makes no
network call for it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..types import SAMPLE_RATE, HeardWord

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MAX_TOKENS_PER_S = 12       # nobody sings faster than this many word pieces a second
PARTIAL_TAIL_S = 0.15       # a final word ending this close to the end of the audio may still grow
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def add_cuda_dll_dirs() -> None:
    """ctranslate2 needs the cuBLAS + cuDNN DLLs; the nvidia-* pip wheels put them in site-packages."""
    if os.name != "nt":
        return
    for base in list(sys.path):
        nv = Path(base) / "nvidia"
        if not nv.is_dir():
            continue
        for sub in ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime"):
            b = nv / sub / "bin"
            if b.is_dir():
                if str(b) not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = str(b) + os.pathsep + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(str(b))
                except OSError:
                    pass


def _run(cmd: list[str], timeout: float = 8.0) -> str:
    # Windows looks in the current folder first: always start the System32 copy by full path
    if os.name == "nt":
        exe = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / (cmd[0] + ".exe")
        if not exe.is_file():
            return ""
        cmd = [str(exe)] + cmd[1:]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              creationflags=_NO_WINDOW).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu_total_used_mb() -> float | None:
    """Memory in use on the first GPU, all programs together."""
    out = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    try:
        return float(out.strip().splitlines()[0])
    except (IndexError, ValueError):
        return None


def process_gpu_mem_mb(pid: int | None = None) -> float | None:
    """This process's own GPU memory in MB, or None when Windows will not say.

    nvidia-smi's per-process figure is "[N/A]" for ordinary programs on Windows (WDDM), so the
    Windows counter "GPU Process Memory \\ Dedicated Usage" is the fallback.
    """
    pid = pid or os.getpid()
    out = _run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits"])
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            try:
                return float(parts[1])
            except ValueError:
                break
    if os.name == "nt":
        out = _run(["typeperf", rf"\GPU Process Memory(pid_{pid}_*)\Dedicated Usage", "-sc", "1"],
                   timeout=15.0)
        total, found = 0.0, False
        for line in out.splitlines():
            cells = [c.strip().strip('"') for c in line.split(",")]
            if len(cells) >= 2 and cells[0][:2].isdigit():
                for c in cells[1:]:
                    try:
                        total += float(c)
                        found = True
                    except ValueError:
                        pass
        if found:
            return total / (1024.0 * 1024.0)
    return None


def model_is_local(model: str, models_dir: Path = MODELS_DIR) -> bool:
    if Path(model).is_dir():
        return True
    return any(models_dir.glob(f"models--*faster-whisper-{model}/snapshots/*/model.bin"))


def download_model(model: str, models_dir: Path = MODELS_DIR) -> str:
    """One-time fetch of the model files into `models/` (the only network use in this module)."""
    from faster_whisper import download_model as fw_download
    models_dir.mkdir(parents=True, exist_ok=True)
    return fw_download(model, cache_dir=str(models_dir))


class WhisperRecogniser:
    # Whisper cannot name a word from its first ~200 ms, so the tail of every pass is "looked at"
    # but not really checked. The detector holds `verified_upto()` back by this much, which makes
    # the safety guard cover that tail inside risky lyric lines (review 2026-11-23).
    latency_ms = 200

    def __init__(self, model: str = "small.en", compute: str = "int8_float16",
                 device: str = "cuda", models_dir: Path | str = MODELS_DIR):
        self.model_name = model
        self.compute = compute
        self.device = device
        self.models_dir = Path(models_dir)
        self.name = (f"GPU (faster-whisper {model})" if device == "cuda"
                     else f"CPU (faster-whisper {model})")
        self.gpu_mem_mb: float | None = None     # this process's GPU memory after loading
        self._model = None

    def load(self) -> None:
        if not model_is_local(self.model_name, self.models_dir):
            raise FileNotFoundError(
                f"speech model '{self.model_name}' is not in {self.models_dir}; "
                f"run: uv run python -m npass.detect download")
        if self.device == "cuda":
            add_cuda_dll_dirs()
        from faster_whisper import WhisperModel
        before = gpu_total_used_mb() if self.device == "cuda" else None
        from faster_whisper.audio import pad_or_trim
        from faster_whisper.tokenizer import Tokenizer
        from faster_whisper.transcribe import get_suppressed_tokens
        self._model = WhisperModel(self.model_name, device=self.device, compute_type=self.compute,
                                   download_root=str(self.models_dir), local_files_only=True)
        self._pad = pad_or_trim
        self._tok = Tokenizer(self._model.hf_tokenizer, self._model.model.is_multilingual,
                              task="transcribe", language="en")
        self._suppress = get_suppressed_tokens(self._tok, [-1])
        # first pass is slow (CUDA start-up): get it out of the way before real audio arrives
        self._decode(np.zeros(16000, dtype=np.float32), [])
        if self.device == "cuda":
            own = process_gpu_mem_mb()
            if own is None and before is not None:
                after = gpu_total_used_mb()
                own = max(0.0, after - before) if after is not None else None
            self.gpu_mem_mb = own

    def _decode(self, audio16k: np.ndarray, hints: list[str]) -> list[dict]:
        """One pass: features -> encoder -> greedy decode -> word alignment.

        This calls faster-whisper's own building blocks directly instead of `model.transcribe()`.
        With word times on, `transcribe()` decodes a short clip twice (it re-runs the stretch after
        the last word): measured on 2026-11-23 with small.en and a 3 s spoken clip, 187 ms a pass
        that way against 120-131 ms this way. The settings are the same ones the brief asks for: English,
        beam 1, temperature 0, nothing carried over from the pass before, no VAD, hotwords.
        """
        m = self._model
        feats = m.feature_extractor(audio16k)
        frames = int(feats.shape[-1])
        enc = m.encode(self._pad(feats))
        prompt = m.get_prompt(self._tok, [], without_timestamps=True,
                              hotwords=" ".join(hints) if hints else None)
        # a 3 s buffer cannot hold many words: the cap stops a runaway decode from eating seconds
        new_tokens = int(MAX_TOKENS_PER_S * len(audio16k) / 16000.0) + 8
        res = m.model.generate(
            enc, [prompt],
            beam_size=1, patience=1.0, length_penalty=1.0, repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            max_length=min(m.max_length, len(prompt) + new_tokens),
            return_scores=True, suppress_blank=True, suppress_tokens=self._suppress,
        )[0]
        tokens = [t for t in res.sequences_ids[0] if t < self._tok.eot]
        if not tokens:
            return []
        return m.find_alignment(self._tok, [tokens], enc, frames)[0]

    def transcribe(self, audio16k: np.ndarray, start_sample48k: int,
                   hints: list[str]) -> list[HeardWord]:
        if self._model is None:
            raise RuntimeError("the speech model is not loaded")
        audio = np.ascontiguousarray(audio16k, dtype=np.float32)
        dur = len(audio) / 16000.0
        words = self._decode(audio, hints)
        out: list[HeardWord] = []
        for w in words:
            text = w["word"].strip()
            if not text:
                continue
            start = min(max(0.0, float(w["start"])), dur)
            end = min(max(start, float(w["end"])), dur)
            out.append(HeardWord(text=text,
                                 start=start_sample48k + int(round(start * SAMPLE_RATE)),
                                 end=start_sample48k + int(round(end * SAMPLE_RATE)),
                                 prob=float(w["probability"])))
        if out:
            last = out[-1]
            if (start_sample48k + int(dur * SAMPLE_RATE)) - last.end <= PARTIAL_TAIL_S * SAMPLE_RATE:
                out[-1] = HeardWord(last.text, last.start, last.end, last.prob, partial=True)
        return out

    def close(self) -> None:
        self._model = None
