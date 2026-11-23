"""Vosk on the CPU, behind the `Recogniser` face.

Vosk is a true streaming recogniser and keeps its own state, so this adapter does NOT re-read the
whole rolling buffer on every call the way the Whisper backend does. It remembers how far it has
fed the stream and gives Vosk only the NEW audio at the end of the buffer.

What comes back from a call:
- finished words (Vosk closes an utterance after a short silence) with Vosk's own word times, kept
  for as long as they are inside the buffer
- the words of the utterance still in progress, from Vosk's partial TEXT. Vosk gives no usable
  times for those (its timed partials trail by more than a second, measured 2026-11-23), so each
  one is timed from the moment it first showed up: it ends there and is assumed to have started up
  to `EST_WORD_MS` earlier. That is deliberately generous; the detector's loudness track tightens
  it where it can. The last word is marked `partial` while it is fresh.

If the buffer jumps (a hole in the audio, or the stream was left behind by more than the buffer),
the stream is reset and the whole buffer is fed again.

Limits to know about:
- Vosk names a word only 0.4 to 0.75 s after it STARTS (measured on this PC). `latency_ms` tells
  the detector, which holds `verified_upto()` back by that much so the safety guard is not told
  that audio is clear when Vosk simply has not spoken yet.
- The small model has a fixed word list. A word that is not in it can never be returned as such;
  Vosk returns the nearest words it knows and the sound-alike matcher has to catch those.
- `hints` are not used (this model cannot be biased towards extra words at run time).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..types import ASR_RATE, SAMPLE_RATE, HeardWord, ms_to_samples

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MODEL_NAME = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
_RATIO = SAMPLE_RATE // ASR_RATE

EST_WORD_MS = 750        # a word in a partial result is assumed to have started this long before
FRESH_MS = 400           # the last partial word counts as "still growing" for this long
# Vosk decodes in steps; the model's default step made partial results arrive on a 240 ms grid.
# A smaller step brought them about 100 ms sooner for about 5 ms of CPU per 80 ms of audio.
TUNE_LINE = "--frames-per-chunk=9"


def model_is_local(models_dir: Path = MODELS_DIR) -> bool:
    return (models_dir / MODEL_NAME / "am" / "final.mdl").is_file()


def tune_model(models_dir: Path = MODELS_DIR) -> None:
    """Add the faster decoding step to the model's own settings file (once)."""
    conf = models_dir / MODEL_NAME / "conf" / "model.conf"
    try:
        text = conf.read_text(encoding="utf-8")
        if "--frames-per-chunk" not in text:
            conf.write_text(text.rstrip("\n") + "\n" + TUNE_LINE + "\n", encoding="utf-8")
    except OSError:
        pass


def download_model(models_dir: Path = MODELS_DIR) -> str:
    """One-time fetch of the small English model into `models/` (the only network use here)."""
    import io
    import zipfile

    import requests

    target = models_dir / MODEL_NAME
    if not model_is_local(models_dir):
        models_dir.mkdir(parents=True, exist_ok=True)
        data = requests.get(MODEL_URL, timeout=120).content
        root = models_dir.resolve()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for member in z.namelist():           # refuse paths that would land outside models/
                dest = (root / member).resolve()
                if not dest.is_relative_to(root) or not member.startswith(MODEL_NAME + "/"):
                    raise RuntimeError("unexpected path in the model archive")
            z.extractall(root)
    tune_model(models_dir)
    return str(target)


class VoskRecogniser:
    name = "CPU (Vosk)"
    latency_ms = EST_WORD_MS     # how far behind the audio this backend's answers really are

    def __init__(self, models_dir: Path | str = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self._model = None
        self._rec = None
        self._origin = 0                 # mic-clock index of the stream's time zero
        self._fed_upto: int | None = None
        self._final: list[HeardWord] = []
        self._seen: list[tuple[str, int]] = []   # words of the open utterance: (text, first seen at)

    def load(self) -> None:
        if not model_is_local(self.models_dir):
            raise FileNotFoundError(
                f"the CPU speech model is not in {self.models_dir}; "
                f"run: uv run python -m npass.detect download")
        import vosk
        vosk.SetLogLevel(-1)
        tune_model(self.models_dir)
        self._model = vosk.Model(str(self.models_dir / MODEL_NAME))
        self._rec = None
        self._fed_upto = None
        self._final = []
        self._seen = []

    def _reset(self, origin: int) -> None:
        import vosk
        self._rec = vosk.KaldiRecognizer(self._model, ASR_RATE)
        self._rec.SetWords(True)
        self._origin = origin
        self._fed_upto = origin
        self._final = []
        self._seen = []

    def transcribe(self, audio16k: np.ndarray, start_sample48k: int,
                   hints: list[str]) -> list[HeardWord]:
        if self._model is None:
            raise RuntimeError("the CPU speech model is not loaded")
        end48 = start_sample48k + len(audio16k) * _RATIO
        if (self._rec is None or self._fed_upto is None
                or self._fed_upto < start_sample48k or self._fed_upto > end48):
            self._reset(start_sample48k)
        offset = (self._fed_upto - start_sample48k) // _RATIO
        new = audio16k[offset:]
        if len(new):
            pcm = (np.clip(new, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            if self._rec.AcceptWaveform(pcm):
                res = json.loads(self._rec.Result())
                for it in res.get("result", []):
                    text = str(it.get("word", "")).strip()
                    if text:
                        self._final.append(HeardWord(
                            text=text,
                            start=self._origin + int(round(float(it.get("start", 0.0)) * SAMPLE_RATE)),
                            end=self._origin + int(round(float(it.get("end", 0.0)) * SAMPLE_RATE)),
                            prob=float(it.get("conf", 1.0))))
                self._seen = []
            self._fed_upto = start_sample48k + (offset + len(new)) * _RATIO
        now = self._fed_upto

        words = str(json.loads(self._rec.PartialResult()).get("partial", "")).split()
        seen: list[tuple[str, int]] = []
        for k, w in enumerate(words):
            # a word keeps its first-seen time for as long as Vosk does not change its mind about it
            seen.append(self._seen[k] if k < len(self._seen) and self._seen[k][0] == w else (w, now))
        self._seen = seen

        est, fresh = ms_to_samples(EST_WORD_MS), ms_to_samples(FRESH_MS)
        floor = self._final[-1].end if self._final else self._origin
        growing: list[HeardWord] = []
        for k, (w, at) in enumerate(seen):
            last = k == len(seen) - 1
            still = last and now - at <= fresh
            growing.append(HeardWord(text=w, start=max(floor, at - est), end=now if still else at,
                                     prob=0.5, partial=still))
        self._final = [w for w in self._final if w.end > start_sample48k]
        return self._final + [w for w in growing if w.end > start_sample48k]

    def close(self) -> None:
        self._rec = None
        self._model = None
        self._fed_upto = None
        self._final = []
        self._seen = []
