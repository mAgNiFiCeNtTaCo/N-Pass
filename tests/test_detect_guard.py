"""SafetyGuard: pure rule, no threads."""
from __future__ import annotations

from npass.detect import SafetyGuard
from npass.detect.guard import FAILSAFE_ID_BASE
from npass.types import CutReason, DangerWindow, Health, ms_to_samples

MARGIN = ms_to_samples(60)


def _guard():
    cuts = []
    return SafetyGuard(cuts.append, margin_ms=60), cuts


def test_on_time_is_ok_and_cuts_nothing():
    g, cuts = _guard()
    win = [DangerWindow(0, 1_000_000, "a line with pineapple", ("pineapple",))]
    assert g.tick(48_000, 48_000 + MARGIN, Health.OK, win) is Health.OK
    assert g.tick(48_000, 48_000 + 20_000, Health.OK, win) is Health.OK
    assert cuts == []


def test_late_outside_a_window_warns_but_does_not_cut():
    g, cuts = _guard()
    assert g.tick(48_000, 40_000, Health.OK, []) is Health.LATE
    far = [DangerWindow(500_000, 600_000, "later", ("banana",))]
    assert g.tick(48_000, 40_000, Health.OK, far) is Health.LATE
    assert cuts == []


def test_late_inside_a_window_cuts_exactly_the_unverified_overlap():
    g, cuts = _guard()
    playhead = 100_000
    verified = playhead + 1_000                       # only 1000 samples ahead: behind the margin
    win = [DangerWindow(90_000, 200_000, "line", ("pineapple",))]
    assert g.tick(playhead, verified, Health.OK, win) is Health.LATE
    assert len(cuts) == 1
    c = cuts[0]
    assert c.reason is CutReason.FAILSAFE
    assert (c.start, c.end) == (verified, playhead + MARGIN)
    assert c.word == "" and c.cut_id >= FAILSAFE_ID_BASE


def test_overlap_is_clipped_to_the_window():
    g, cuts = _guard()
    playhead = 100_000
    # window opens in the middle of the unverified span and closes before its end
    win = [DangerWindow(playhead + 500, playhead + 1_500, "line", ("banana",))]
    g.tick(playhead, playhead - 5_000, Health.OK, win)
    assert (cuts[0].start, cuts[0].end) == (playhead + 500, playhead + 1_500)


def test_verified_behind_the_playhead_cuts_from_the_playhead():
    g, cuts = _guard()
    win = [DangerWindow(0, 10_000_000, "line", ("banana",))]
    g.tick(100_000, 50_000, Health.OK, win)
    assert (cuts[0].start, cuts[0].end) == (100_000, 100_000 + MARGIN)


def test_cut_is_extended_block_by_block_with_one_id():
    g, cuts = _guard()
    win = [DangerWindow(0, 10_000_000, "line", ("banana",))]
    verified = 100_500
    for k in range(5):                                # the recogniser is stuck, audio keeps playing
        g.tick(100_000 + k * 480, verified, Health.OK, win)
    assert len({c.cut_id for c in cuts}) == 1
    assert [c.start for c in cuts] == [verified] * 5
    assert [c.end for c in cuts] == [100_000 + k * 480 + MARGIN for k in range(5)]
    assert g.failsafe_cuts == 1


def test_a_shifting_window_does_not_start_a_new_cut_every_tick():
    g, cuts = _guard()
    for k in range(4):                                # lyrics re-synced: the window moves a little
        win = [DangerWindow(1_000 + k * 37, 10_000_000 + k * 37, "line", ("banana",))]
        g.tick(100_000 + k * 480, 90_000, Health.OK, win)
    assert len({c.cut_id for c in cuts}) == 1


def test_recovery_then_a_second_stall_gets_a_new_id():
    g, cuts = _guard()
    win = [DangerWindow(0, 10_000_000, "line", ("banana",))]
    g.tick(100_000, 100_100, Health.OK, win)
    assert g.tick(100_480, 200_000, Health.OK, win) is Health.OK      # caught up
    g.tick(300_000, 300_100, Health.OK, win)                         # behind again, much later
    ids = [c.cut_id for c in cuts]
    assert len(ids) == 2 and ids[0] != ids[1]


def test_down_inside_a_window_cuts_and_reports_down():
    g, cuts = _guard()
    win = [DangerWindow(0, 10_000_000, "line", ("pineapple",))]
    assert g.tick(100_000, 60_000, Health.DOWN, win) is Health.DOWN
    assert len(cuts) == 1 and cuts[0].reason is CutReason.FAILSAFE
    assert (cuts[0].start, cuts[0].end) == (100_000, 100_000 + MARGIN)


def test_down_outside_a_window_reports_down_without_a_cut():
    g, cuts = _guard()
    assert g.tick(100_000, 60_000, Health.DOWN, []) is Health.DOWN
    assert cuts == []


def test_starting_is_passed_through_and_guards_windows():
    g, cuts = _guard()
    win = [DangerWindow(0, 10_000_000, "line", ("pineapple",))]
    assert g.tick(-20_000, 0, Health.STARTING, win) is Health.STARTING   # nothing has played yet
    assert cuts == []
    assert g.tick(5_000, 0, Health.STARTING, win) is Health.STARTING
    assert len(cuts) == 1 and (cuts[0].start, cuts[0].end) == (5_000, 5_000 + MARGIN)


def test_two_windows_in_the_span_get_a_cut_each():
    cuts = []
    g = SafetyGuard(cuts.append, margin_ms=200)
    span_end = 100_000 + ms_to_samples(200)
    win = [DangerWindow(100_100, 100_900, "one", ("banana",)),
           DangerWindow(105_000, span_end + 50_000, "two", ("pineapple",))]
    g.tick(100_000, 90_000, Health.OK, win)
    assert [(c.start, c.end) for c in cuts] == [(100_100, 100_900), (105_000, span_end)]
    assert cuts[0].cut_id != cuts[1].cut_id
