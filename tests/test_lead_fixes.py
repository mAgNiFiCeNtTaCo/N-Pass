"""Tests for the fixes made after the fail-safe review of 2026-11-23."""
from npass.detect import SafetyGuard
from npass.detect.whisper_backend import WhisperRecogniser
from npass.types import CutReason, DangerWindow, Health, ms_to_samples

MS = ms_to_samples


def collect():
    got = []
    return got, got.append


def test_failsafe_reaches_ahead_so_a_stalled_guard_still_covers_the_line():
    got, submit = collect()
    g = SafetyGuard(submit, margin_ms=60, ahead_ms=250)
    win = [DangerWindow(MS(1000), MS(3000), "risky", ("pineapple",))]
    # recogniser dead: nothing verified past 1.2 s; the playhead is at 1.2 s
    h = g.tick(MS(1200), MS(1200), Health.DOWN, win, captured=MS(1750))
    assert h is Health.DOWN
    assert got[-1].reason is CutReason.FAILSAFE
    assert got[-1].start == MS(1200)
    assert got[-1].end == MS(1200 + 60 + 250)      # 250 ms further than the bare margin


def test_failsafe_never_reaches_past_what_was_captured_or_the_window():
    got, submit = collect()
    g = SafetyGuard(submit, margin_ms=60, ahead_ms=250)
    win = [DangerWindow(MS(1000), MS(1250), "risky", ("pineapple",))]
    g.tick(MS(1200), MS(1200), Health.DOWN, win, captured=MS(1300))
    assert got[-1].end == MS(1250)                 # the window's end wins
    got.clear()
    g2 = SafetyGuard(submit, margin_ms=60, ahead_ms=250)
    g2.tick(MS(1200), MS(1200), Health.DOWN, [DangerWindow(MS(1000), MS(3000))], captured=MS(1290))
    assert got[-1].end == MS(1290)                 # cannot mute audio that does not exist yet


def test_a_recogniser_hovering_at_the_limit_gives_one_mute_not_many():
    got, submit = collect()
    g = SafetyGuard(submit, margin_ms=60, ahead_ms=0, bridge_ms=300)
    win = [DangerWindow(0, MS(5000))]
    playhead, verified = 0, 0
    for step in range(300):                        # 3 s in 10 ms ticks
        playhead = MS(step * 10)
        if step % 25 == 0:                         # a pass lands every 250 ms, barely ahead
            verified = playhead + MS(70)
        g.tick(playhead, verified, Health.OK, win, captured=playhead + MS(550))
    assert g.failsafe_cuts == 1
    assert len({o.cut_id for o in got}) == 1


def test_without_the_bridge_the_old_behaviour_is_unchanged():
    got, submit = collect()
    g = SafetyGuard(submit, margin_ms=60)
    win = [DangerWindow(MS(1000), MS(3000))]
    g.tick(MS(1200), MS(1200), Health.DOWN, win)
    assert (got[-1].start, got[-1].end) == (MS(1200), MS(1260))


def test_whisper_holds_verified_back_by_its_unrecognisable_tail():
    assert WhisperRecogniser.latency_ms >= 200
