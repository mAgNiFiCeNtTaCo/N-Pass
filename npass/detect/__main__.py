"""Set-up and measuring tools for the speech backends.

    uv run python -m npass.detect download [model ...]   fetch the model files into models/ (one time)
    uv run python -m npass.detect bench [--model small.en] [--compute int8_float16]
                                        [--context 3.0] [--wav clip.wav] [--passes 30]
                                                         time one recognition pass, show GPU memory
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np


def _download(models: list[str]) -> int:
    from . import vosk_backend, whisper_backend
    for m in models or ["small.en"]:
        print(f"speech model {m} ->", whisper_backend.download_model(m))
    print("CPU model ->", vosk_backend.download_model())
    return 0


def _load_clip(path: str | None, seconds: float) -> np.ndarray:
    n = int(seconds * 16000)
    if not path:
        rng = np.random.default_rng(0)
        return (rng.standard_normal(n) * 0.02).astype(np.float32)
    import soundfile as sf
    import soxr
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != 16000:
        audio = soxr.resample(audio, rate, 16000)
    if len(audio) < n:
        audio = np.tile(audio, int(np.ceil(n / len(audio))))
    return np.ascontiguousarray(audio[:n], dtype=np.float32)


def _bench(args) -> int:
    from .whisper_backend import WhisperRecogniser, gpu_total_used_mb, process_gpu_mem_mb
    if args.engine == "cpu":
        from .vosk_backend import VoskRecogniser
        rec = VoskRecogniser()
    else:
        rec = WhisperRecogniser(model=args.model, compute=args.compute)
    before = gpu_total_used_mb()
    t0 = time.perf_counter()
    rec.load()
    print(f"{rec.name}: loaded in {time.perf_counter() - t0:.1f} s")
    audio = _load_clip(args.wav, args.context)
    hints: list[str] = []      # the detector passes no hotwords by default
    times, words = [], []
    for i in range(args.passes):
        t = time.perf_counter()
        if args.engine == "cpu":
            # a streaming backend only gets the new 80 ms each call
            hop = 1280
            a, b = (i * hop) % len(audio), (i * hop) % len(audio) + hop
            words = rec.transcribe(audio[:b], 0, hints) if b <= len(audio) else words
        else:
            words = rec.transcribe(audio, 0, hints)
        times.append((time.perf_counter() - t) * 1000.0)
    times_sorted = sorted(times[1:] or times)
    p95 = times_sorted[max(0, int(len(times_sorted) * 0.95) - 1)]
    print(f"context {args.context:.1f} s, {args.passes} passes: "
          f"median {statistics.median(times_sorted):.0f} ms, p95 {p95:.0f} ms, "
          f"max {times_sorted[-1]:.0f} ms")
    print("heard:", " ".join(w.text for w in words) or "(nothing)")
    if args.engine != "cpu":
        own, after = process_gpu_mem_mb(), gpu_total_used_mb()
        print("GPU memory, this process:", "unknown" if own is None else f"{own:.0f} MB")
        if before is not None and after is not None:
            print(f"GPU memory, whole card: {before:.0f} MB before, {after:.0f} MB after "
                  f"(+{after - before:.0f} MB)")
    rec.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m npass.detect")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("models", nargs="*")
    b = sub.add_parser("bench")
    b.add_argument("--engine", default="gpu", choices=["gpu", "cpu"])
    b.add_argument("--model", default="small.en")
    b.add_argument("--compute", default="int8_float16")
    b.add_argument("--context", type=float, default=3.0)
    b.add_argument("--wav", default=None)
    b.add_argument("--passes", type=int, default=30)
    args = ap.parse_args(argv)
    return _download(args.models) if args.cmd == "download" else _bench(args)


if __name__ == "__main__":
    sys.exit(main())
