"""Fault tests: what the listener hears when the recogniser freezes, crashes or never loads.

The three live pieces are wired the way `npass/app.py` wires them (LookaheadProcessor, Detector,
SafetyGuard), with the scripted FakeRecogniser. No model, no GPU, no sound device, no network.
Stand-in blocked words only.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from npass.audio.processor import LookaheadProcessor
from npass.detect import Detector, FakeRecogniser, SafetyGuard, ScriptedWord
from npass.detect.guard import FAILSAFE_ID_BASE
from npass.settings import Settings
from npass.types import SAMPLE_RATE, CutReason, DangerWindow, Health, ms_to_samples

BLOCK = 480
WORDS = ["pineapple", "banana"]


def sec(s: float) -> int:
    return int(round(s * SAMPLE_RATE))


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NPASS_DATA_DIR", str(tmp_path))


def voice(n: int) -> np.ndarray:
    """A steady stand-in for a voice that is never zero, so gain = out / in is exact."""
    t = np.arange(n, dtype=np.float64)
    return (0.25 + 0.05 * np.sin(2 * np.pi * 220.0 * t / SAMPLE_RATE)).astype(np.float32)


class Chain:
    """Processor + detector + guard, driven one 10 ms block at a time."""

    def __init__(self, seconds: float, wins=(), script=(), rec=None, threaded=False):
        self.settings = Settings()
        self.look = ms_to_samples(self.settings.lookahead_ms)
        self.proc = LookaheadProcessor(self.settings.lookahead_ms, self.settings.fade_ms)
        self.rec = rec or FakeRecogniser(list(script))
        self.wins = list(wins)
        self.orders = []
        self.det = Detector(self.settings, lambda: list(WORDS), self.windows, self.submit,
                            lambda order: None, recogniser=self.rec)
        self.guard = SafetyGuard(self.submit, ahead_ms=250, bridge_ms=300)    # as npass/app.py
        self.audio = voice(sec(seconds))
        self.out = np.zeros_like(self.audio)
        self.pos = 0
        self.frozen = False
        self.threaded = threaded
        self.health_log: list[tuple[int, Health]] = []

    def windows(self, a, b):
        return [w for w in self.wins if w.start < b and w.end > a]

    def submit(self, order):
        self.orders.append(order)
        self.proc.submit_cut(order)

    def step(self) -> Health:
        block = self.audio[self.pos:self.pos + BLOCK]
        start = self.proc.captured
        self.out[self.pos:self.pos + BLOCK] = self.proc.process(block)
        self.det.feed(block, start)
        if not self.threaded and not self.frozen:
            self.det.process_pending()
        self.pos += BLOCK
        playhead = self.proc.playhead
        health = self.guard.tick(playhead, self.det.verified_upto(), self.det.health(),
                                 self.windows(playhead, self.proc.captured),
                                 captured=self.proc.captured)
        self.health_log.append((playhead, health))
        return health

    def run_to(self, seconds: float, realtime=False) -> None:
        while self.pos < min(sec(seconds), len(self.audio)):
            self.step()
            if realtime:
                time.sleep(0.004)

    def finish(self):
        self.run_to(len(self.audio) / SAMPLE_RATE)
        played = self.out[self.look:]                       # what came out, lined up with the input
        source = self.audio[:len(played)]
        return source, played, played / source

    def health_at(self, seconds: float) -> Health:
        """The guard's verdict at the moment this point of the audio was being played."""
        return next(h for p, h in self.health_log if p >= sec(seconds))


def assert_no_click(chain: Chain, gain: np.ndarray) -> None:
    assert float(np.max(np.abs(np.diff(gain)))) <= chain.proc.fade_step + 1e-4


# ------------------------------------------------------------------ frozen recogniser

def test_frozen_inside_a_danger_window_mutes_the_window_with_fades():
    win = DangerWindow(sec(2.0), sec(3.0), "a line with pineapple in it", ("pineapple",))
    c = Chain(5.0, wins=[win])
    c.run_to(1.5)
    assert c.health_at(0.5) == Health.OK and not c.orders
    c.frozen = True                                # the recogniser is stuck inside a pass
    source, played, gain = c.finish()

    fade = c.proc.fade_samples
    assert np.all(played[win.start:win.end] == 0.0)             # the whole line is silent
    assert np.array_equal(played[:win.start - fade], source[:win.start - fade])
    assert np.array_equal(played[win.end + fade:], source[win.end + fade:])
    ramp_in = gain[win.start - fade:win.start]
    assert ramp_in[0] > 0.9 and np.all(np.diff(ramp_in) < 0)    # a fade, not a hard edge
    ramp_out = gain[win.end:win.end + fade]
    assert np.all(np.diff(ramp_out) > 0)
    assert_no_click(c, gain)
    assert c.health_at(2.5) == Health.LATE                      # and the window is told
    assert all(o.reason == CutReason.FAILSAFE and o.cut_id >= FAILSAFE_ID_BASE for o in c.orders)
    assert c.guard.failsafe_cuts == 1


def test_frozen_part_way_through_a_window_mutes_only_what_was_not_checked():
    win = DangerWindow(sec(2.0), sec(3.0), "a line with banana in it", ("banana",))
    c = Chain(5.0, wins=[win])
    c.run_to(2.5)                                  # everything up to about 2.5 s was checked: clean
    checked = c.det.verified_upto()
    assert sec(2.4) <= checked <= sec(2.5)
    c.frozen = True
    source, played, gain = c.finish()

    fade = c.proc.fade_samples
    assert np.array_equal(played[:checked - fade], source[:checked - fade])   # checked part plays
    assert np.all(played[checked:win.end] == 0.0)                             # the rest is muted
    assert np.array_equal(played[win.end + fade:], source[win.end + fade:])
    assert_no_click(c, gain)


def test_frozen_outside_a_window_passes_audio_untouched_and_says_late():
    win = DangerWindow(sec(0.2), sec(0.6), "an early line with banana", ("banana",))
    c = Chain(4.0, wins=[win])
    c.run_to(1.5)                                  # the window went by while everything was fine
    c.frozen = True
    source, played, _ = c.finish()

    assert np.array_equal(played, source)          # nothing cut, nothing dipped
    assert not c.orders
    assert c.health_at(1.0) == Health.OK
    assert c.health_at(2.5) == Health.LATE
    assert c.det.health() == Health.OK             # the detector cannot know; the guard does


def test_a_recogniser_that_keeps_up_is_never_reported_late():
    c = Chain(3.0)
    source, played, _ = c.finish()
    assert {h for playhead, h in c.health_log if playhead > 0} == {Health.OK}
    assert np.array_equal(played, source) and not c.orders


# ------------------------------------------------------------------ crash and recovery

def test_crash_then_recovery_mutes_the_window_meanwhile_and_cuts_again_after():
    win = DangerWindow(sec(1.0), sec(2.0), "a line with pineapple", ("pineapple",))
    word = ScriptedWord("pineapple", sec(4.0), sec(4.5))
    c = Chain(6.5, wins=[win], script=[word])
    c.run_to(0.6)
    assert c.det.health() == Health.OK

    c.rec.crash_now = True
    try:
        c.det.process_pending(force=True)
    except RuntimeError as exc:
        c.det._crashed(exc)                        # what the worker thread does with a crash
    assert c.det.health() == Health.DOWN
    assert c.rec.close_calls == 1
    c.frozen = True                                # down: no passes while it is being restarted
    c.run_to(2.6)
    assert c.health_at(1.5) == Health.DOWN

    c.rec.crash_now = False                        # the restart works
    c.frozen = False
    source, played, gain = c.finish()

    fade = c.proc.fade_samples
    assert c.det.health() == Health.OK and c.det.stats["restarts"] == 2
    assert np.all(played[win.start:win.end] == 0.0)             # muted while it was down
    assert np.all(played[word.start:word.end] == 0.0)           # caught again once it was back
    quiet_from, quiet_to = win.end + fade, sec(3.5)
    assert np.array_equal(played[quiet_from:quiet_to], source[quiet_from:quiet_to])
    assert_no_click(c, gain)
    reasons = {o.reason for o in c.orders}
    assert CutReason.FAILSAFE in reasons and (CutReason.ASR in reasons)


def test_worker_thread_survives_a_crash_and_restarts_by_itself():
    rec = FakeRecogniser([])
    c = Chain(20.0, rec=rec, threaded=True)
    c.det.restart_s = 0.1
    c.det.start()
    try:
        deadline = time.monotonic() + 5.0
        while c.det.health() != Health.OK and time.monotonic() < deadline:
            c.step()
            time.sleep(0.004)
        assert c.det.health() == Health.OK

        rec.crash_once = True
        deadline = time.monotonic() + 5.0
        seen_down = False
        while time.monotonic() < deadline:
            c.step()
            time.sleep(0.004)
            seen_down = seen_down or c.det.health() == Health.DOWN
            if seen_down and c.det.health() == Health.OK:
                break
        assert seen_down
        assert c.det.health() == Health.OK
        assert c.det.stats["crashes"] == 1 and c.det.stats["restarts"] == 2
        assert c.det._thread.is_alive()
        assert "RuntimeError" in c.det.last_error
    finally:
        c.det.stop()


# ------------------------------------------------------------------ model fails to load

def test_model_that_fails_to_load_is_down_mutes_windows_and_passes_the_rest():
    win = DangerWindow(sec(1.0), sec(1.6), "a line with banana", ("banana",))
    rec = FakeRecogniser([], fail_loads=10_000)
    c = Chain(3.0, wins=[win], rec=rec, threaded=True)
    c.det.restart_s = 0.05
    c.det.start()
    try:
        deadline = time.monotonic() + 5.0
        while c.det.health() != Health.DOWN and time.monotonic() < deadline:
            time.sleep(0.01)
        assert c.det.health() == Health.DOWN
        assert "not running" in c.det.engine_name()
        source, played, gain = c.finish()
    finally:
        c.det.stop()

    fade = c.proc.fade_samples
    assert rec.load_calls >= 1 and not rec.loaded
    assert c.det.verified_upto() == 0                          # nothing was ever checked
    assert np.all(played[win.start:win.end] == 0.0)
    assert np.array_equal(played[:win.start - fade], source[:win.start - fade])
    assert np.array_equal(played[win.end + fade:], source[win.end + fade:])
    assert_no_click(c, gain)
    assert {h for _, h in c.health_log} == {Health.DOWN}


def test_model_that_loads_on_a_later_try_comes_up_by_itself():
    rec = FakeRecogniser([], fail_loads=2)
    c = Chain(20.0, rec=rec, threaded=True)
    c.det.restart_s = 0.05
    c.det.start()
    try:
        deadline = time.monotonic() + 5.0
        while c.det.health() != Health.OK and time.monotonic() < deadline:
            c.step()
            time.sleep(0.004)
        assert c.det.health() == Health.OK
        assert rec.load_calls == 3 and c.det.stats["crashes"] == 2
        assert c.det.engine_name() == rec.name
    finally:
        c.det.stop()
