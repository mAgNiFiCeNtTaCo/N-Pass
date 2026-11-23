"""The look-ahead buffer and the cutter. Pure numpy, no devices: used live and by the offline harness.

Clock: `captured` = how many mic samples have been taken in. `playhead` = index of the NEXT sample
that will be handed out = captured - look-ahead.

How the gain is made (this is what keeps it click-free whatever arrives, whenever):
  a counter k runs from 0 (open) to F (fully cut), F = fade length in samples. For every output
  sample n the counter moves ONE step: up when some cut "wants" n (start - F < n < end), down
  otherwise. gain = table[k], a raised cosine from 1 down to the floor (0, or dip_db in "dip" mode).
  An on-time cut therefore reaches the floor exactly at `start`, holds it to `end - 1`, and the
  first sample at `end` is already one step up. Cuts whose wanted spans touch or overlap behave as
  one cut. A late cut simply starts stepping from the playhead. The gain can never move more than
  one table step between two neighbouring samples.

A look-ahead change is a second counter of the same kind (fade to 0, switch the delay, fade back);
the two gains are combined with min(), so the one-step rule still holds.
"""
from __future__ import annotations

import threading

import numpy as np

from npass.types import SAMPLE_RATE, CutOrder, ms_to_samples

MAX_LOOKAHEAD_MS = 2000.0     # the buffer is sized for this; set_lookahead clamps to it
_CHUNK = 4800                 # process() works in pieces of at most this many samples


def _cos_table(fade: int) -> np.ndarray:
    """1 -> 0 raised cosine in `fade` steps (fade + 1 entries, exact 1.0 first and exact 0.0 last)."""
    t = 0.5 * (1.0 + np.cos(np.pi * np.arange(fade + 1, dtype=np.float64) / fade))
    t[0], t[-1] = 1.0, 0.0
    return t


class LookaheadProcessor:
    def __init__(self, lookahead_ms: float, fade_ms: float = 15, mode: str = "silence",
                 dip_db: float = -30.0):
        self._lock = threading.Lock()
        self._cap = ms_to_samples(MAX_LOOKAHEAD_MS) + 2 * _CHUNK
        self._buf = np.zeros(self._cap, dtype=np.float32)
        self._captured = 0
        self._look = self._clamp_look(lookahead_ms)
        self._pending_look: int | None = None
        self._cuts: dict[int, tuple[int, int]] = {}

        self._k = 0               # cut counter, 0..F
        self._j = 0               # look-ahead-change counter, 0..F
        self._j_dir = 0           # +1 fading out, -1 fading back in, 0 idle
        self._pending_mode: tuple[str, float, float] | None = None
        self._mode = "silence"
        self._dip_db = -30.0
        self._fade = 1
        self._table = np.ones(2, dtype=np.float32)
        self._ttable = np.ones(2, dtype=np.float32)
        self._apply_mode(mode, dip_db, fade_ms)

        # scratch, so process() does not allocate more than the output block
        self._ramp = np.arange(1, _CHUNK + 1, dtype=np.int64)
        self._mask = np.zeros(_CHUNK, dtype=bool)
        self._idx = np.zeros(_CHUNK, dtype=np.int64)

    # ---------------------------------------------------------------- settings

    @staticmethod
    def _clamp_look(ms: float) -> int:
        return ms_to_samples(max(0.0, min(MAX_LOOKAHEAD_MS, float(ms))))

    def _apply_mode(self, mode: str, dip_db: float, fade_ms: float) -> None:
        self._mode = "dip" if mode == "dip" else "silence"
        self._dip_db = float(dip_db)
        self._fade = max(1, ms_to_samples(max(0.0, float(fade_ms))))
        floor = 0.0
        if self._mode == "dip":
            floor = float(min(1.0, max(0.0, 10.0 ** (self._dip_db / 20.0))))
        cos = _cos_table(self._fade)
        self._table = (floor + (1.0 - floor) * cos).astype(np.float32)
        self._ttable = cos.astype(np.float32)

    def set_mode(self, mode: str, dip_db: float, fade_ms: float) -> None:
        """Takes effect at once when nothing is being cut, otherwise as soon as the gain is fully
        open again (changing the floor or the fade length in the middle of a cut would click)."""
        with self._lock:
            if self._k == 0 and self._j_dir == 0:
                self._apply_mode(mode, dip_db, fade_ms)
                self._pending_mode = None
            else:
                self._pending_mode = (mode, dip_db, fade_ms)

    def set_lookahead(self, ms: float) -> None:
        """Glitch-free: fades out over one fade length, switches the delay, fades back in."""
        new = self._clamp_look(ms)
        with self._lock:
            if self._captured == 0 and self._j_dir == 0:
                self._look = new
                self._pending_look = None
            elif new == self._look and self._j_dir == 0:
                self._pending_look = None
            else:
                self._pending_look = new

    # ---------------------------------------------------------------- cuts

    def submit_cut(self, order: CutOrder) -> float:
        """Thread-safe. Returns how late the cut is in ms (0 = its start has not played yet).
        The same cut_id again replaces the earlier order; an empty order (end <= start) removes it."""
        start, end = int(order.start), int(order.end)
        with self._lock:
            playhead = self._captured - self._look
            if end <= start:
                self._cuts.pop(order.cut_id, None)
                return 0.0
            # forget cuts that have left the buffer for good (they can never be played again)
            gone = self._captured - self._cap
            if len(self._cuts) > 8:
                for cid in [c for c, (_, e) in self._cuts.items() if e < gone]:
                    del self._cuts[cid]
            self._cuts[order.cut_id] = (start, end)
            late = playhead - start
        return late * 1000.0 / SAMPLE_RATE if late > 0 else 0.0

    # ---------------------------------------------------------------- state

    @property
    def captured(self) -> int:
        return self._captured

    @property
    def playhead(self) -> int:
        return self._captured - self._look

    @property
    def cutting(self) -> bool:
        return self._k > 0

    @property
    def lookahead_ms(self) -> float:
        return self._look * 1000.0 / SAMPLE_RATE

    @property
    def lookahead_samples(self) -> int:
        return self._look

    @property
    def fade_samples(self) -> int:
        return self._fade

    @property
    def fade_step(self) -> float:
        """The biggest gain change there can ever be between two neighbouring output samples."""
        return float(np.max(np.abs(np.diff(self._ttable))))

    # ---------------------------------------------------------------- audio

    def process(self, block: np.ndarray) -> np.ndarray:
        """Mono float32 in, same length out: the input delayed by exactly the look-ahead, cuts applied."""
        x = np.asarray(block, dtype=np.float32).reshape(-1)
        out = np.empty(x.shape[0], dtype=np.float32)
        for a in range(0, x.shape[0], _CHUNK):
            b = min(a + _CHUNK, x.shape[0])
            with self._lock:
                self._process_chunk(x[a:b], out[a:b])
        return out

    def _process_chunk(self, x: np.ndarray, out: np.ndarray) -> None:
        n = x.shape[0]
        cap0 = self._captured
        # 1. store the new samples
        w = cap0 % self._cap
        first = min(n, self._cap - w)
        self._buf[w:w + first] = x[:first]
        if first < n:
            self._buf[:n - first] = x[first:]
        self._captured = cap0 + n

        # 2. hand out the delayed samples, in pieces that each have one look-ahead and one direction
        i = 0
        while i < n:
            if self._j_dir == 0:
                if self._pending_mode is not None and self._k == 0:
                    self._apply_mode(*self._pending_mode)
                    self._pending_mode = None
                if self._pending_look is not None:
                    if self._pending_look == self._look:
                        self._pending_look = None
                    else:
                        self._j_dir = 1
            if self._j_dir > 0:
                seg = min(n - i, self._fade - self._j)
            elif self._j_dir < 0:
                seg = min(n - i, self._j)
            else:
                seg = n - i
            o = out[i:i + seg]
            first_out = cap0 + i - self._look
            self._read(first_out, o)
            gain = self._cut_gain(first_out, seg)
            if self._j_dir != 0:
                if self._j_dir > 0:
                    tg = self._ttable[self._j + 1:self._j + 1 + seg]
                    self._j += seg
                else:
                    tg = self._ttable[self._j - seg:self._j][::-1]
                    self._j -= seg
                gain = tg if gain is None else np.minimum(gain, tg)
                if self._j_dir > 0 and self._j >= self._fade:
                    # fully faded out: switch the delay here, where nothing is audible
                    if self._pending_look is not None:
                        self._look = self._pending_look
                        self._pending_look = None
                    self._j_dir = -1
                    p = cap0 + i + seg - self._look
                    self._k = self._fade if self._wanted_at(p) else 0
                elif self._j_dir < 0 and self._j <= 0:
                    self._j_dir = 0
            if gain is not None:
                np.multiply(o, gain, out=o)
            i += seg

    def _read(self, first: int, o: np.ndarray) -> None:
        """Copy buffered samples [first, first + len(o)) into o; indexes below 0 are silence."""
        m = o.shape[0]
        skip = 0
        if first < 0:
            skip = min(m, -first)
            o[:skip] = 0.0
            if skip == m:
                return
            first = 0
        r = first % self._cap
        cnt = m - skip
        head = min(cnt, self._cap - r)
        o[skip:skip + head] = self._buf[r:r + head]
        if head < cnt:
            o[skip + head:] = self._buf[:cnt - head]

    def _wanted_at(self, p: int) -> bool:
        f = self._fade
        for s, e in self._cuts.values():
            if s - f < p < e:
                return True
        return False

    def _cut_gain(self, n0: int, m: int) -> np.ndarray | None:
        """Gain for output samples [n0, n0 + m). None = fully open (the caller copies untouched)."""
        f = self._fade
        mask = self._mask[:m]
        hit = False
        for s, e in self._cuts.values():
            a = s - f + 1 - n0
            b = e - n0
            if b <= 0 or a >= m:
                continue
            if not hit:
                mask[:] = False
                hit = True
            mask[max(a, 0):min(b, m)] = True
        k = self._k
        if not hit:
            if k == 0:
                return None
            mask[:] = False
        idx = self._idx[:m]
        edges = np.flatnonzero(mask[1:] != mask[:-1]) + 1
        r0 = 0
        for r1 in (*edges.tolist(), m):
            ln = r1 - r0
            if mask[r0]:
                np.minimum(self._ramp[:ln] + k, f, out=idx[r0:r1])
            else:
                np.maximum(k - self._ramp[:ln], 0, out=idx[r0:r1])
            k = int(idx[r1 - 1])
            r0 = r1
        self._k = k
        return self._table[idx]
