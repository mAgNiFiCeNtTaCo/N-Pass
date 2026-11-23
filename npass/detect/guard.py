"""The safety guard: mutes a danger window that nobody checked in time.

Pure: no threads, no clock of its own. The app calls `tick()` once per audio block.

Rule (docs/CONTRACT.md): audio must be verified up to `playhead + margin`. The span from the
verified point (or the playhead, whichever is later) up to `playhead + margin` is "unverified and
about to play". For every danger window overlapping that span a FAILSAFE cut is ordered over
exactly the overlap, and extended tick by tick with the same cut_id while the recogniser stays
behind. Outside danger windows nothing is cut; the returned Health tells the window to warn.

A DOWN or STARTING recogniser is judged by the same span: audio it verified before it went down
stays verified, everything after it is not.
"""
from __future__ import annotations

from ..types import CutOrder, CutReason, DangerWindow, Health, SubmitCut, ms_to_samples

FAILSAFE_ID_BASE = 1_000_000_000    # keeps guard cut ids apart from the detector's (which start at 1)


class SafetyGuard:
    def __init__(self, submit_cut: SubmitCut, margin_ms: float = 60, ahead_ms: float = 0,
                 bridge_ms: float = 0):
        """ahead_ms: order each failsafe mute this much further ahead than strictly needed, so a
        guard that misses a few ticks has still covered the audio (review 2026-11-23).
        bridge_ms: a new unverified stretch this close after an open cut joins it instead of
        becoming a new cut (one mute per risky line, not one per recognition pass)."""
        self._submit = submit_cut
        self._margin = ms_to_samples(margin_ms)
        self._ahead = ms_to_samples(ahead_ms)
        self._bridge = ms_to_samples(bridge_ms)
        self._next_id = FAILSAFE_ID_BASE
        # [cut_id, cut_start, cut_end] of cuts that may still grow. Matched by position, not by
        # window identity: lyric windows shift a little whenever the song position is re-synced.
        self._open: list[list[int]] = []
        self.failsafe_cuts = 0          # how many separate failsafe cuts were ordered

    def tick(self, playhead: int, verified_upto: int, health: Health,
             windows: list[DangerWindow], captured: int | None = None) -> Health:
        need = playhead + self._margin
        span_start = max(verified_upto, playhead)
        behind = span_start < need

        if behind:
            reach = need + self._ahead
            if captured is not None:
                reach = max(need, min(reach, captured))
            for w in windows:
                lo, hi = max(span_start, w.start), min(reach, w.end)
                if lo >= hi:
                    continue
                cur = next((c for c in self._open
                            if lo <= c[2] + self._bridge and hi >= c[1]), None)
                if cur is not None:
                    # the same stretch is still unverified: grow the cut, never shrink it
                    if hi > cur[2]:
                        cur[2] = hi
                        self._send(cur)
                else:
                    cur = [self._next_id, lo, hi]
                    self._next_id += 1
                    self.failsafe_cuts += 1
                    self._open.append(cur)
                    self._send(cur)

        # forget cuts that have fully played
        if self._open:
            self._open = [c for c in self._open if c[2] + self._bridge >= playhead]

        if health in (Health.DOWN, Health.STARTING):
            return health
        return Health.LATE if behind else Health.OK

    def _send(self, cur: list[int]) -> None:
        self._submit(CutOrder(cut_id=cur[0], start=cur[1], end=cur[2], word="", heard="",
                              confidence=0.0, reason=CutReason.FAILSAFE))
