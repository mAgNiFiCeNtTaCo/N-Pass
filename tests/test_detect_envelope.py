"""Cut edges from the loudness track. Synthetic signals only."""
from __future__ import annotations

import numpy as np

from npass.detect.envelope import rms_track, snap_edges

RATE = 16_000


def _ms(ms: float) -> int:
    return int(RATE * ms / 1000)


def _burst_pause_burst() -> np.ndarray:
    """silence 0-200 ms, voice 200-500, pause 500-700, voice 700-1000, silence 1000-1300."""
    rng = np.random.default_rng(1)
    x = np.zeros(_ms(1300), dtype=np.float32)
    for a, b in ((200, 500), (700, 1000)):
        x[_ms(a):_ms(b)] = (rng.standard_normal(_ms(b) - _ms(a)) * 0.2).astype(np.float32)
    return x


def test_rms_track_shape_and_levels():
    x = _burst_pause_burst()
    track, frame = rms_track(x, RATE)
    assert frame == 160
    assert len(track) == len(x) // frame
    assert track[5] == 0.0                    # 50 ms: silence
    assert 0.15 < track[30] < 0.25            # 300 ms: inside the first burst
    assert track[60] == 0.0                   # 600 ms: the pause


def test_edges_move_out_to_the_pause_on_both_sides():
    x = _burst_pause_burst()
    track, frame = rms_track(x, RATE)
    # the recogniser puts the word well inside the second burst
    e = snap_edges(track, frame, _ms(760), _ms(950), len(x))
    assert e.start_snapped and e.end_snapped and e.end_quiet
    assert e.start == _ms(700)
    assert e.end == _ms(1000)


def test_first_burst_snaps_to_its_own_edges_not_the_other_burst():
    x = _burst_pause_burst()
    track, frame = rms_track(x, RATE)
    e = snap_edges(track, frame, _ms(260), _ms(440), len(x))
    assert (e.start, e.end) == (_ms(200), _ms(500))


def test_edges_never_move_inwards():
    x = _burst_pause_burst()
    track, frame = rms_track(x, RATE)
    # the recogniser's times already reach into the pauses
    e = snap_edges(track, frame, _ms(650), _ms(1050), len(x))
    assert e.start <= _ms(650)
    assert e.end >= _ms(1050)


def test_dip_out_of_reach_keeps_the_recognisers_time():
    x = _burst_pause_burst()
    track, frame = rms_track(x, RATE)
    # 850 ms is 150 ms from either edge of the second burst: further than the 120 ms reach
    e = snap_edges(track, frame, _ms(850), _ms(860), len(x))
    assert not e.start_snapped and not e.end_snapped
    assert (e.start, e.end) == (_ms(850), _ms(860))


def test_steady_sound_has_no_dip():
    t = np.arange(_ms(1000)) / RATE
    x = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    track, frame = rms_track(x, RATE)
    e = snap_edges(track, frame, _ms(400), _ms(600), len(x))
    assert not e.start_snapped and not e.end_snapped
    assert (e.start, e.end) == (_ms(400), _ms(600))


def test_a_clear_local_minimum_counts_when_there_is_no_real_pause():
    # two loud syllables with a softer (not silent) valley between them at 480-520 ms
    rng = np.random.default_rng(2)
    x = (rng.standard_normal(_ms(1000)) * 0.2).astype(np.float32)
    x[_ms(480):_ms(520)] *= 0.4
    track, frame = rms_track(x, RATE)
    e = snap_edges(track, frame, _ms(560), _ms(700), len(x))
    assert e.start_snapped
    assert _ms(480) <= e.start <= _ms(530)
    assert not e.end_quiet


def test_empty_and_silent_input():
    track, frame = rms_track(np.zeros(0, dtype=np.float32), RATE)
    assert len(track) == 0
    e = snap_edges(track, frame, 10, 20, 0)
    assert (e.start, e.end, e.start_snapped, e.end_snapped) == (0, 0, False, False)
    silent = np.zeros(_ms(500), dtype=np.float32)
    track, frame = rms_track(silent, RATE)
    e = snap_edges(track, frame, _ms(100), _ms(200), len(silent))
    assert (e.start, e.end) == (_ms(100), _ms(200))
