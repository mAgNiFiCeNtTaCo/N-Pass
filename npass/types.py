"""Shared messages between the pieces. See docs/CONTRACT.md. Owned by the lead: do not edit in a station.

Time base: every audio time is a SAMPLE INDEX on the mic clock, counted at 48 kHz from the moment
the engine (or the offline harness) started. `int`, never wall-clock seconds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Protocol

import numpy as np

SAMPLE_RATE = 48_000   # everything the user hears or OBS gets
ASR_RATE = 16_000      # what the recognisers are fed


def ms_to_samples(ms: float) -> int:
    return int(round(ms * SAMPLE_RATE / 1000.0))


def samples_to_ms(n: int) -> float:
    return n * 1000.0 / SAMPLE_RATE


class CutReason(str, Enum):
    ASR = "asr"                    # recogniser matched outside a danger window
    LYRIC_WINDOW = "lyric window"  # matched inside a danger window (lower threshold)
    FAILSAFE = "failsafe"          # safety guard muted a danger window nobody checked


@dataclass(frozen=True)
class CutOrder:
    """Silence (or dip) the voice from `start` (inclusive) to `end` (exclusive)."""
    cut_id: int
    start: int
    end: int
    word: str = ""            # the blocked-list entry that matched ("" for failsafe)
    heard: str = ""           # what the recogniser heard
    confidence: float = 0.0   # 0..1
    reason: CutReason = CutReason.ASR


@dataclass(frozen=True)
class HeardWord:
    """One recognised word. start/end are mic-clock sample indexes (48 kHz)."""
    text: str
    start: int
    end: int
    prob: float               # recogniser's own confidence, 0..1
    partial: bool = False     # True: the word may still be growing (streaming partial)


@dataclass(frozen=True)
class DangerWindow:
    """A stretch of mic time in which a lyric line containing a blocked word is being sung."""
    start: int
    end: int
    line: str = ""
    words: tuple[str, ...] = ()   # blocked-list entries present in the line


@dataclass(frozen=True)
class TrackInfo:
    track_id: str
    artist: str
    title: str
    position_ms: int
    is_playing: bool

    def label(self) -> str:
        m, s = divmod(max(0, self.position_ms) // 1000, 60)
        return f"{self.artist} - {self.title} @ {m}:{s:02d}"


class Health(str, Enum):
    OK = "ok"
    LATE = "late"          # recogniser has not cleared audio that is about to be played
    DOWN = "down"          # crashed, or the model failed to load
    STARTING = "starting"


@dataclass(frozen=True)
class Catch:
    """One caught word, for the counter and the log."""
    wall_time: float          # time.time()
    order: CutOrder
    track: TrackInfo | None = None
    late_ms: float = 0.0      # >0: the cut arrived after its start had already played


@dataclass
class EngineStatus:
    running: bool = False
    input_ok: bool = False
    output_ok: bool = False
    input_name: str = ""
    output_name: str = ""
    lookahead_ms: float = 0.0
    measured_delay_ms: float = 0.0   # real mic-to-cable delay from the driver clocks
    level_db: float = -120.0         # mic level, dBFS peak of the last ~50 ms
    cutting: bool = False            # a cut is being applied right now
    warnings: list[str] = field(default_factory=list)   # plain-English, shown as-is


# --- callables the pieces hand each other (wired in npass/app.py and in the harness) ---

SubmitCut = Callable[[CutOrder], None]
"""Detector/guard -> audio. May be called from any thread. Same cut_id again = replace that cut
(used to extend a cut while a stretched word keeps going)."""

AudioTap = Callable[[np.ndarray, int], None]
"""Audio -> detector: (mono float32 block at 48 kHz, sample index of its first sample).
Must return at once (it only queues)."""

WindowsProvider = Callable[[int, int], list[DangerWindow]]
"""(from_sample, to_sample) -> danger windows overlapping that span. Cheap, thread-safe."""


class Recogniser(Protocol):
    """A speech backend (faster-whisper or Vosk)."""
    name: str

    def load(self) -> None: ...
    def transcribe(self, audio16k: np.ndarray, start_sample48k: int,
                   hints: list[str]) -> list[HeardWord]: ...
    def close(self) -> None: ...
