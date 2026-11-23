"""A scripted recogniser for tests and the offline harness: no model, no GPU.

Give it words with their times on the mic clock (48 kHz sample indexes). Each `transcribe()` call
answers with the scripted words that fall inside the audio it was handed. A word the audio stops in
the middle of comes back cut short and marked `partial`, the way a real recogniser hears a word
that is still being sung: "banana" half sung is returned as "ban".
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..types import ASR_RATE, SAMPLE_RATE, HeardWord

_RATIO = SAMPLE_RATE // ASR_RATE


@dataclass(frozen=True)
class ScriptedWord:
    text: str
    start: int            # mic-clock sample index (48 kHz)
    end: int
    prob: float = 0.9
    visible_from: int = 0  # the word is only "heard" once the audio reaches this index (a late answer)


class FakeRecogniser:
    name = "Fake (scripted)"

    def __init__(self, script: list[ScriptedWord] | list[tuple] | None = None,
                 delay_s: float = 0.0, fail_loads: int = 0, reveal_partials: bool = True):
        self.script: list[ScriptedWord] = [
            w if isinstance(w, ScriptedWord) else ScriptedWord(*w) for w in (script or [])
        ]
        self.delay_s = delay_s              # sleep this long in every pass (a slow model)
        self.fail_loads = fail_loads        # the next N load() calls raise
        self.reveal_partials = reveal_partials
        self.crash_now = False              # True: the next transcribe() raises (stays on until cleared)
        self.crash_once = False             # True: the next transcribe() raises, then it clears itself
        self.loaded = False
        self.load_calls = 0
        self.close_calls = 0
        self.passes: list[tuple[int, int]] = []     # (start, end) mic-clock span of every pass
        self.hints_seen: list[list[str]] = []
        self._lock = threading.Lock()

    # --- Recogniser face ---
    def load(self) -> None:
        self.load_calls += 1
        if self.fail_loads > 0:
            self.fail_loads -= 1
            raise RuntimeError("fake recogniser: scripted load failure")
        self.loaded = True

    def transcribe(self, audio16k: np.ndarray, start_sample48k: int,
                   hints: list[str]) -> list[HeardWord]:
        if not self.loaded:
            raise RuntimeError("fake recogniser: transcribe() before load()")
        if self.crash_now or self.crash_once:
            self.crash_once = False
            raise RuntimeError("fake recogniser: scripted crash")
        end48 = start_sample48k + len(audio16k) * _RATIO
        with self._lock:
            self.passes.append((start_sample48k, end48))
            self.hints_seen.append(list(hints))
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        out: list[HeardWord] = []
        for w in self.script:
            if w.end <= start_sample48k or w.start >= end48 or end48 < w.visible_from:
                continue
            if w.end <= end48:
                out.append(HeardWord(w.text, w.start, w.end, w.prob, partial=False))
            elif self.reveal_partials:
                share = (end48 - w.start) / max(1, w.end - w.start)
                letters = max(1, min(len(w.text), math.ceil(len(w.text) * share)))
                out.append(HeardWord(w.text[:letters], w.start, end48, w.prob * 0.8, partial=True))
        out.sort(key=lambda h: h.start)
        return out

    def close(self) -> None:
        self.close_calls += 1
        self.loaded = False
