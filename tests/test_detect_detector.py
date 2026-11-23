"""Detector logic with the scripted recogniser: no model, no GPU, no devices, no network.

Most tests drive the detector by hand (`process_pending()` after every block) so they are exact and
fast; the ones about the worker thread (stale hops, crash and recovery) start the real thread.
Stand-in blocked words only: `pineapple` and `banana`.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from npass.detect import Detector, FakeRecogniser, ScriptedWord
from npass.detect.guard import FAILSAFE_ID_BASE
from npass.settings import Settings
from npass.types import CutReason, DangerWindow, Health, ms_to_samples

BLOCK = 480                      # 10 ms at 48 kHz
WORDS = ["pineapple", "banana"]


def sec(s: float) -> int:
    return int(round(s * 48_000))


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NPASS_DATA_DIR", str(tmp_path))


def windows_of(wins):
    return lambda a, b: [w for w in wins if w.start < b and w.end > a]


class Rig:
    def __init__(self, script=(), words=WORDS, wins=(), rec=None, audio=None, on_catch=None,
                 hold_ms=0.0):
        self.settings = Settings()
        # the rig's default "voice" is steady noise with no pauses, so holding a cut until the
        # voice dips is off here and tested on shaped audio below
        self.settings.max_hold_ms = hold_ms
        self.rec = rec or FakeRecogniser(list(script))
        self.cuts = []
        self.cut_fed_at = []          # how much audio had been fed when each cut order arrived
        self.catches = []
        self.pos = 0
        self.audio = audio
        self._rng = np.random.default_rng(7)
        self.words = list(words)
        self.det = Detector(self.settings, lambda: list(self.words), windows_of(list(wins)),
                            self._submit, on_catch or self.catches.append, recogniser=self.rec)

    def _submit(self, order):
        self.cuts.append(order)
        self.cut_fed_at.append(self.pos)

    def _block(self):
        if self.audio is not None:
            b = self.audio[self.pos:self.pos + BLOCK]
            if len(b) < BLOCK:
                b = np.concatenate([b, np.zeros(BLOCK - len(b), dtype=np.float32)])
            return b
        return (self._rng.standard_normal(BLOCK) * 0.1).astype(np.float32)   # steady "voice"

    def feed_to(self, seconds: float, step: bool = True):
        end = sec(seconds)
        while self.pos < end:
            block = self._block()
            self.det.feed(block, self.pos)
            self.pos += BLOCK
            if step:
                self.det.process_pending()


def wait_for(cond, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.005)
    return False


# ------------------------------------------------------------------ thresholds and windows
def test_full_match_outside_a_window_is_an_asr_cut():
    rig = Rig([("banana", sec(1.0), sec(1.4)), ("then", sec(1.6), sec(1.8))])
    rig.feed_to(3.0)
    assert len(rig.catches) == 1
    c = rig.cuts[-1]
    assert c.word == "banana" and c.heard == "banana" and c.confidence == 1.0
    assert c.reason is CutReason.ASR
    assert c.cut_id < FAILSAFE_ID_BASE


def test_near_miss_is_cut_only_inside_a_danger_window():
    # "cabana" scores between window_threshold (0.50) and threshold (0.72) against "banana"
    script = [("cabana", sec(1.0), sec(1.4)), ("then", sec(1.6), sec(1.8))]
    outside = Rig(script)
    outside.feed_to(3.0)
    assert outside.cuts == [] and outside.catches == []

    win = DangerWindow(sec(0.5), sec(2.0), "sing the banana song", ("banana",))
    inside = Rig(script, wins=[win])
    inside.feed_to(3.0)
    assert len(inside.catches) == 1
    c = inside.catches[0]
    assert c.reason is CutReason.LYRIC_WINDOW and c.word == "banana" and c.heard == "cabana"
    assert 0.50 <= c.confidence < 0.72


def test_window_that_is_elsewhere_does_not_lower_the_bar():
    win = DangerWindow(sec(5.0), sec(6.0), "sing the banana song", ("banana",))
    rig = Rig([("cabana", sec(1.0), sec(1.4)), ("then", sec(1.6), sec(1.8))], wins=[win])
    rig.feed_to(3.0)
    assert rig.cuts == []


def test_full_match_inside_a_window_is_marked_lyric_window():
    win = DangerWindow(sec(0.5), sec(2.0), "sing the banana song", ("banana",))
    rig = Rig([("banana", sec(1.0), sec(1.4))], wins=[win])
    rig.feed_to(3.0)
    assert [c.reason for c in rig.catches] == [CutReason.LYRIC_WINDOW]


def test_ordinary_words_are_not_cut():
    words = "i really like to sing this song with you tonight baby".split()
    script = [(w, sec(0.5 + 0.3 * i), sec(0.5 + 0.3 * i + 0.25)) for i, w in enumerate(words)]
    rig = Rig(script)
    rig.feed_to(5.0)
    assert rig.cuts == [] and rig.catches == []
    assert len(rig.rec.passes) > 20


# ------------------------------------------------------------------ cut on suspicion
def test_prefix_cut_then_extension_with_one_cut_id():
    start, end = sec(1.0), sec(2.2)                 # one long, stretched word
    rig = Rig([("banana", start, end)])
    rig.feed_to(5.0)
    pad = ms_to_samples(rig.settings.cut_pad_ms)

    assert len(rig.catches) == 1                    # one word, one catch
    assert len({c.cut_id for c in rig.cuts}) == 1   # ... and one cut, extended
    assert len(rig.cuts) >= 3

    first = rig.cuts[0]
    assert rig.cut_fed_at[0] < end                  # ordered while the word was still being sung
    assert rig.cut_fed_at[0] <= start + (end - start) * 0.80
    assert first.heard != "banana"                  # on a beginning, not the whole word
    assert first.end > rig.cut_fed_at[0]            # a little past what had been heard
    assert rig.catches[0] == first                  # on_catch got the first order

    ends = [c.end for c in rig.cuts]
    assert ends == sorted(ends) and ends[-1] > ends[0]
    for order, fed in zip(rig.cuts, rig.cut_fed_at):
        if fed < end:                               # while the word goes on the cut stays ahead
            assert order.end > fed
    assert abs(rig.cuts[-1].start - (start - pad)) <= 3
    assert rig.cuts[-1].end >= end + pad
    assert rig.cuts[-1].end <= end + ms_to_samples(600)     # closed soon after the word ended
    assert rig.cuts[-1].heard == "banana" and rig.cuts[-1].confidence == 1.0


def _shaped(voiced, total=5.0):
    """Noise "voice" only inside the given (start, end) second spans, silence elsewhere."""
    rng = np.random.default_rng(11)
    a = np.zeros(sec(total), dtype=np.float32)
    for s0, s1 in voiced:
        a[sec(s0):sec(s1)] = (rng.standard_normal(sec(s1) - sec(s0)) * 0.1).astype(np.float32)
    return a


def test_a_held_word_is_cut_until_the_voice_pauses():
    # the recogniser says the word is 1.0-1.4 s, but the voice holds it until 2.3 s
    rig = Rig([("banana", sec(1.0), sec(1.4))], audio=_shaped([(0.2, 0.8), (1.0, 2.3), (3.0, 4.0)]),
              hold_ms=1500.0)
    rig.feed_to(4.5)
    assert len(rig.catches) == 1 and len({c.cut_id for c in rig.cuts}) == 1
    last = rig.cuts[-1]
    assert last.end >= sec(2.3)                       # the whole held word is covered
    assert last.end <= sec(2.3) + ms_to_samples(450)  # and it stops soon after the voice does
    assert last.end < sec(3.0)                        # the next phrase is not touched


def test_a_hold_gives_up_after_the_limit():
    rig = Rig([("banana", sec(1.0), sec(1.4))], audio=_shaped([(0.2, 0.8), (1.0, 5.0)]),
              hold_ms=1500.0)
    rig.feed_to(4.8)
    assert rig.cuts[-1].end <= sec(1.4) + ms_to_samples(1500)
    assert rig.cuts[-1].end >= sec(2.5)     # counted from the first (partial) sighting of the word


def test_cut_is_not_resent_once_the_word_is_over():
    rig = Rig([("banana", sec(1.0), sec(1.5))])
    rig.feed_to(2.5)
    n = len(rig.cuts)
    rig.feed_to(6.0)                                # the word slides through and out of the buffer
    assert len(rig.cuts) == n
    assert len(rig.catches) == 1


def test_half_a_word_is_not_enough_outside_a_window_but_is_inside():
    # audio stops while only "ban" has been sung
    script = [("banana", sec(1.0), sec(2.0))]
    outside = Rig(script)
    outside.feed_to(1.45)
    assert outside.cuts == []
    win = DangerWindow(sec(0.5), sec(3.0), "banana boat", ("banana",))
    inside = Rig(script, wins=[win])
    inside.feed_to(1.62)
    assert len(inside.catches) == 1 and inside.catches[0].reason is CutReason.LYRIC_WINDOW


# ------------------------------------------------------------------ one word, one catch
def test_one_catch_per_word_across_overlapping_passes():
    script = [("banana", sec(1.0), sec(1.4)), ("and", sec(1.5), sec(1.7)),
              ("pineapple", sec(2.0), sec(2.5)), ("banana", sec(2.7), sec(3.1)),
              ("banana", sec(3.15), sec(3.55)), ("yeah", sec(4.0), sec(4.2))]
    rig = Rig(script)
    rig.feed_to(8.0)
    assert len(rig.rec.passes) > 50                 # each word was seen in dozens of passes
    assert [c.word for c in rig.catches] == ["banana", "pineapple", "banana", "banana"]
    assert len({c.cut_id for c in rig.cuts}) == 4
    assert rig.det.stats["catches"] == 4
    final = {c.cut_id: c for c in rig.cuts}
    pad = ms_to_samples(rig.settings.cut_pad_ms)
    for (text, s, e), cut in zip([w for w in script if w[0] in WORDS], final.values()):
        assert cut.start <= s - pad + 3 and cut.end >= e + pad - 3


def test_a_word_the_recogniser_split_in_two_is_one_catch():
    rig = Rig([("pine", sec(1.0), sec(1.2)), ("apple", sec(1.25), sec(1.6)), ("pie", sec(1.9), sec(2.1))])
    rig.feed_to(4.0)
    assert [c.word for c in rig.catches] == ["pineapple"]
    last = rig.cuts[-1]
    pad = ms_to_samples(rig.settings.cut_pad_ms)
    assert last.start <= sec(1.0) - pad + 3 and last.end >= sec(1.6) + pad - 3


def test_on_catch_failure_does_not_stop_the_cut():
    def boom(order):
        raise RuntimeError("log is broken")
    rig = Rig([("banana", sec(1.0), sec(1.4))], on_catch=boom)
    rig.feed_to(2.5)
    assert rig.cuts and rig.det.stats["callback_errors"] == 1


def test_a_word_heard_just_too_late_is_still_ordered_and_reported():
    # the recogniser only names the word 500 ms after it ended: its start has already played
    rig = Rig([ScriptedWord("banana", sec(1.0), sec(1.4), visible_from=sec(1.9))])
    rig.feed_to(4.0)
    assert len(rig.catches) == 1
    assert rig.cut_fed_at[0] >= sec(1.9)
    assert rig.det.stats["stale_hits"] == 0


def test_a_word_first_heard_long_after_it_played_is_dropped():
    rig = Rig([ScriptedWord("banana", sec(1.0), sec(1.4), visible_from=sec(3.0))])
    rig.feed_to(5.0)
    assert rig.cuts == [] and rig.catches == []
    assert rig.det.stats["stale_hits"] >= 1


# ------------------------------------------------------------------ cut edges
def test_cut_edges_follow_the_loudness_not_the_recogniser():
    rng = np.random.default_rng(3)
    audio = np.zeros(sec(4.0), dtype=np.float32)
    for a, b in ((1.0, 1.3), (1.5, 1.8)):           # burst, pause, burst
        audio[sec(a):sec(b)] = (rng.standard_normal(sec(b) - sec(a)) * 0.2).astype(np.float32)
    # the recogniser puts the word inside the second burst, 60 ms late and 50 ms short
    rig = Rig([("banana", sec(1.56), sec(1.75))], audio=audio)
    rig.feed_to(4.0)
    assert len(rig.catches) == 1
    pad = ms_to_samples(rig.settings.cut_pad_ms)
    tol = ms_to_samples(12)                         # one loudness frame
    last = rig.cuts[-1]
    assert abs(last.start - (sec(1.5) - pad)) <= tol
    assert last.end >= sec(1.8) + pad - tol
    assert last.start > sec(1.3)                    # the first burst is left alone


def test_no_dip_keeps_the_recognisers_time_plus_pad():
    rig = Rig([("banana", sec(1.0), sec(1.4)), ("then", sec(1.6), sec(1.8))])   # steady noise
    rig.feed_to(3.0)
    pad = ms_to_samples(rig.settings.cut_pad_ms)
    assert abs(rig.cuts[-1].start - (sec(1.0) - pad)) <= 3


# ------------------------------------------------------------------ hints, silence, bookkeeping
def test_hints_are_off_unless_switched_on():
    win = DangerWindow(sec(1.0), sec(3.0), "Sing the banana song", ("banana",))
    script = [("hello", sec(1.0), sec(1.3))]

    rig = Rig(script, wins=[win])                   # default: no hotwords, window or not
    rig.feed_to(2.0)
    assert rig.rec.hints_seen and all(h == [] for h in rig.rec.hints_seen)

    rig = Rig(script, wins=[win])
    rig.settings.hotwords_in_windows = True
    rig.feed_to(2.0)
    last = rig.rec.hints_seen[-1]
    assert last[:2] == WORDS and {"sing", "the", "song"} <= set(last)

    rig = Rig(script)                               # switched on, but no window anywhere near
    rig.settings.hotwords_in_windows = True
    rig.feed_to(2.0)
    assert all(h == [] for h in rig.rec.hints_seen)

    rig = Rig(script)
    rig.settings.hotwords_always = True
    rig.feed_to(2.0)
    assert rig.rec.hints_seen[-1] == WORDS


def test_silence_and_an_empty_word_list_skip_the_recogniser():
    rig = Rig([("banana", sec(0.5), sec(0.9))], audio=np.zeros(sec(2.0), dtype=np.float32))
    rig.feed_to(2.0)
    assert rig.rec.passes == [] and rig.cuts == []
    assert rig.det.stats["silent_passes"] > 0
    assert rig.det.verified_upto() > sec(1.8)

    rig = Rig([("banana", sec(0.5), sec(0.9))], words=[])
    rig.feed_to(2.0)
    assert rig.rec.passes == [] and rig.cuts == []
    assert rig.det.verified_upto() > sec(1.8)


def test_verified_upto_is_the_end_of_a_completed_pass():
    rig = Rig([])
    assert rig.det.verified_upto() == 0 and rig.det.health() is Health.STARTING
    rig.feed_to(0.05)                               # less than one hop: no pass yet
    assert rig.det.verified_upto() == 0
    rig.feed_to(1.0)
    v = rig.det.verified_upto()
    assert v == rig.rec.passes[-1][1]               # exactly what the last pass was given
    assert sec(1.0) - ms_to_samples(100) <= v <= sec(1.0)
    assert rig.det.health() is Health.OK
    assert rig.det.last_pass_ms() >= 0.0
    assert rig.det.engine_name() == "Fake (scripted)"


def test_a_backend_that_answers_late_holds_verified_back():
    rec = FakeRecogniser([])
    rec.latency_ms = 500
    rig = Rig(rec=rec)
    rig.feed_to(2.0)
    assert rig.det.verified_upto() == rig.rec.passes[-1][1] - ms_to_samples(500)


def test_a_hole_in_the_audio_restarts_the_buffer():
    rig = Rig([("banana", sec(3.0), sec(3.4))])
    rig.feed_to(1.0)
    rig.pos += sec(1.0)                             # a second of audio never arrives
    rig.feed_to(4.5)
    assert rig.det.stats["gaps"] == 1
    assert len(rig.catches) == 1
    assert rig.det.verified_upto() > sec(4.4)


def test_rolling_buffer_is_capped_at_the_context():
    rig = Rig([])
    rig.feed_to(6.0)
    s, e = rig.rec.passes[-1]
    assert e - s == sec(rig.settings.context_s)


# ------------------------------------------------------------------ the worker thread
def test_slow_passes_skip_stale_hops_and_take_the_newest_audio():
    rec = FakeRecogniser([], delay_s=0.25)          # every pass takes 250 ms; a hop is 80 ms
    rig = Rig(rec=rec)
    rig.det.start()
    try:
        for _ in range(150):                        # 1.5 s of audio at real-time pace
            rig.feed_to((rig.pos + BLOCK) / 48_000, step=False)
            time.sleep(0.01)
        end = rig.pos
        assert wait_for(lambda: rig.det.verified_upto() >= end - ms_to_samples(20), timeout=3.0)
    finally:
        rig.det.stop()
    hops = 1500 // rig.settings.hop_ms              # 18 hops of audio went in
    assert len(rec.passes) <= hops // 2             # far fewer passes than hops: none queued up
    assert rig.det.stats["hops_skipped"] >= 5
    ends = [e for _, e in rec.passes]
    assert ends == sorted(ends)
    assert ends[-1] >= end - ms_to_samples(20)      # the last pass saw the newest audio
    assert rig.det.last_pass_ms() >= 240


def test_worker_crash_goes_down_then_recovers():
    rec = FakeRecogniser([("banana", sec(100.0), sec(100.4))])
    rig = Rig(rec=rec)
    rig.det.restart_s = 0.3
    seen = []
    rig.det.start()
    try:
        def pump(cond, seconds=5.0):
            t0 = time.monotonic()
            while time.monotonic() - t0 < seconds:
                rig.feed_to((rig.pos + BLOCK) / 48_000, step=False)
                time.sleep(0.005)
                seen.append(rig.det.health())
                if cond():
                    return True
            return False

        assert pump(lambda: rig.det.health() is Health.OK)
        v_before = rig.det.verified_upto()
        rec.crash_once = True
        assert pump(lambda: rig.det.health() is Health.DOWN)
        assert "not running" in rig.det.engine_name()
        assert "scripted crash" in rig.det.last_error
        assert pump(lambda: rig.det.health() is Health.OK)
        assert pump(lambda: rig.det.verified_upto() > v_before + sec(0.2))
        assert rig.det._thread.is_alive()
    finally:
        rig.det.stop()
    assert rig.det.stats["crashes"] == 1
    assert rec.load_calls == 2 and rec.close_calls >= 1
    assert rig.det.engine_name() == "Fake (scripted)"


def test_failed_load_is_down_and_keeps_trying():
    rec = FakeRecogniser([], fail_loads=2)
    rig = Rig(rec=rec)
    rig.det.restart_s = 0.1
    rig.det.start()
    try:
        assert wait_for(lambda: rig.det.health() is Health.DOWN, timeout=2.0)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0 and rig.det.health() is not Health.OK:
            rig.feed_to((rig.pos + BLOCK) / 48_000, step=False)
            time.sleep(0.005)
        assert rig.det.health() is Health.OK
    finally:
        rig.det.stop()
    assert rec.load_calls == 3


def test_feed_only_queues_and_stop_is_clean():
    rig = Rig([])
    t0 = time.perf_counter()
    for _ in range(300):
        rig.feed_to((rig.pos + BLOCK) / 48_000, step=False)
    assert time.perf_counter() - t0 < 0.5           # 3 s of audio queued almost instantly
    assert rig.rec.passes == []                     # nothing ran: the worker was never started
    rig.det.start()
    assert wait_for(lambda: rig.det.health() is Health.OK)
    rig.det.stop()
    assert rig.det._thread is None and rig.rec.close_calls == 1
