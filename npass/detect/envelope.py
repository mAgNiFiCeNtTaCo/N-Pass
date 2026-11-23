"""Loudness track of the voice, used to put cut edges on real pauses.

A recogniser's word times are rough (Whisper's words touch end to end), so the detector does not
trust them. It looks at how loud the voice is in short frames and moves a cut's start BACK and its
end FORWARD to the nearest dip: a frame quieter than a fraction of the word's own peak, or a clear
local minimum. Edges only ever move outwards (a wider cut is safer than a clipped one).

Everything here works on sample positions inside one audio array; the caller converts to the mic
clock. Pure functions, no state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FRAME_MS = 10.0          # loudness frame
SEARCH_MS = 120.0        # how far an edge may move
DIP_FRACTION = 0.25      # "quiet" = below this share of the word's peak loudness
MIN_FRACTION = 0.60      # a local minimum counts as a dip only below this share of the peak
FLOOR = 1e-4             # about -80 dBFS: below this everything is silence


@dataclass(frozen=True)
class Edges:
    start: int               # sample position in the array
    end: int
    start_snapped: bool      # False: no dip found, `start` is the recogniser's own time
    end_snapped: bool
    end_quiet: bool = False  # True: the end sits on a real pause, not just a local minimum


def rms_track(audio: np.ndarray, rate: int, frame_ms: float = FRAME_MS) -> tuple[np.ndarray, int]:
    """(RMS per frame, frame length in samples). The last, incomplete frame is dropped."""
    frame = max(1, int(round(rate * frame_ms / 1000.0)))
    n = len(audio) // frame
    if n == 0:
        return np.zeros(0, dtype=np.float32), frame
    x = np.asarray(audio[: n * frame], dtype=np.float32).reshape(n, frame)
    return np.sqrt(np.mean(x * x, axis=1)), frame


def _is_quiet(track: np.ndarray, i: int, peak: float, fraction: float) -> bool:
    return bool(track[i] <= max(FLOOR, fraction * peak))


def _is_local_min(track: np.ndarray, i: int, peak: float) -> bool:
    v = track[i]
    if not (0 < i < len(track) - 1) or v >= MIN_FRACTION * peak:
        return False
    return bool(v < track[i - 1] and v <= track[i + 1] or v <= track[i - 1] and v < track[i + 1])


def snap_edges(track: np.ndarray, frame: int, start: int, end: int, total: int,
               search_ms: float = SEARCH_MS, frame_ms: float = FRAME_MS,
               fraction: float = DIP_FRACTION) -> Edges:
    """Move `start` back and `end` forward to the nearest dip within `search_ms`.

    `start`/`end` are sample positions in the audio the track was made from, `total` its length.
    A real pause (a quiet frame) anywhere in reach wins over a mere local minimum, which may be
    the gap between two syllables of the same word. When no dip is found on a side, that side
    keeps the recogniser's time.
    """
    start = max(0, min(start, total))
    end = max(start, min(end, total))
    n = len(track)
    if n == 0:
        return Edges(start, end, False, False, False)
    reach = max(1, int(round(search_ms / frame_ms)))
    f0 = min(n - 1, start // frame)
    f1 = min(n - 1, max(f0, (end - 1) // frame if end > 0 else 0))
    peak = float(track[f0: f1 + 1].max())
    if peak <= FLOOR:
        return Edges(start, end, False, False, False)

    back = range(f0, max(-1, f0 - reach - 1), -1)
    i = next((i for i in back if _is_quiet(track, i, peak, fraction)), None)
    if i is None:
        i = next((i for i in back if _is_local_min(track, i, peak)), None)
    # the sound starts after the dip frame; never move the start later than it was
    new_start, s_ok = (min(start, (i + 1) * frame), True) if i is not None else (start, False)

    fwd = range(f1, min(n, f1 + reach + 1))
    j = next((j for j in fwd if _is_quiet(track, j, peak, fraction)), None)
    quiet = j is not None
    if j is None:
        j = next((j for j in fwd if _is_local_min(track, j, peak)), None)
    new_end, e_ok = (max(end, j * frame), True) if j is not None else (end, False)
    return Edges(new_start, new_end, s_ok, e_ok, quiet)
