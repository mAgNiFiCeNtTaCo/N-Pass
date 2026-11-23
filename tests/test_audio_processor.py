"""LookaheadProcessor: exact delay, cut shape, and the no-click rule. No devices."""
from __future__ import annotations

import sys
from pathlib import Path

# plain `uv run pytest` does not put the project root on the path (no conftest, package = false)
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import threading

import numpy as np
import pytest

from npass.audio.processor import LookaheadProcessor
from npass.types import SAMPLE_RATE, CutOrder, ms_to_samples

BLOCK = 480
L_MS = 450
L = ms_to_samples(L_MS)
F = ms_to_samples(15)


def run(proc: LookaheadProcessor, x: np.ndarray, block: int = BLOCK, events=None) -> np.ndarray:
    """Push x through in blocks. events: {block_number: callable} run just before that block."""
    out = []
    for bi, a in enumerate(range(0, len(x), block)):
        if events and bi in events:
            events[bi]()
        out.append(proc.process(x[a:a + block]))
    return np.concatenate(out)


def gain_by_sample(proc: LookaheadProcessor, n: int, lookahead: int = L, events=None) -> np.ndarray:
    """Feed ones; return g where g[i] is the gain applied to mic sample i (fixed look-ahead only)."""
    out = run(proc, np.ones(n + lookahead, dtype=np.float32), events=events)
    return out[lookahead:]


def max_step(y: np.ndarray) -> float:
    return float(np.max(np.abs(np.diff(y.astype(np.float64)))))


def cut(cid: int, start: int, end: int) -> CutOrder:
    return CutOrder(cut_id=cid, start=start, end=end, word="pineapple")


# ------------------------------------------------------------------ delay

@pytest.mark.parametrize("block", [480, 441, 1, 7000])
def test_delay_is_exact(block):
    rng = np.random.default_rng(1)
    n = 60000 if block > 1 else 23000
    x = rng.uniform(-1, 1, n).astype(np.float32)
    p = LookaheadProcessor(L_MS)
    y = run(p, x, block=block)
    assert y.dtype == np.float32 and y.shape == x.shape
    assert np.all(y[:L] == 0.0)
    assert np.array_equal(y[L:], x[:-L])
    assert p.captured == n
    assert p.playhead == n - L


def test_playhead_is_negative_at_start():
    p = LookaheadProcessor(L_MS)
    assert p.captured == 0 and p.playhead == -L
    p.process(np.zeros(BLOCK, dtype=np.float32))
    assert p.playhead == BLOCK - L


def test_long_run_wraps_the_buffer_correctly():
    rng = np.random.default_rng(2)
    x = rng.uniform(-1, 1, 5 * SAMPLE_RATE).astype(np.float32)   # longer than the internal buffer
    y = run(LookaheadProcessor(L_MS), x)
    assert np.array_equal(y[L:], x[:-L])


# ------------------------------------------------------------------ cut shape

def test_cut_silences_exactly_start_to_end_and_fades_sit_outside():
    p = LookaheadProcessor(L_MS, fade_ms=15)
    s, e = 30000, 36000
    assert p.submit_cut(cut(1, s, e)) == 0.0
    g = gain_by_sample(p, 60000)
    assert np.all(g[s:e] == 0.0)
    assert g[s - 1] > 0.0 and g[e] > 0.0                 # not one sample more than asked
    assert np.all(g[:s - F + 1] == 1.0)                  # untouched before the fade
    assert np.all(g[e + F - 1:] == 1.0)                  # and after it
    assert np.all((g[s - F + 1:s] > 0.0) & (g[s - F + 1:s] < 1.0))
    assert np.all((g[e:e + F - 1] > 0.0) & (g[e:e + F - 1] < 1.0))
    # raised cosine, ending at start / beginning at end
    k = np.arange(1, F)
    want = 0.5 * (1.0 + np.cos(np.pi * k / F))
    assert np.allclose(g[s - F + 1:s], want, atol=1e-6)
    assert np.allclose(g[e:e + F - 1], want[::-1], atol=1e-6)
    assert max_step(g) <= p.fade_step + 1e-7
    assert not p.cutting


def test_cutting_flag_is_true_during_the_cut():
    p = LookaheadProcessor(L_MS)
    p.submit_cut(cut(1, 4800, 9600))
    seen = []
    for _ in range(80):
        p.process(np.ones(BLOCK, dtype=np.float32))
        seen.append((p.playhead, p.cutting))
    assert any(c for ph, c in seen if 4800 < ph <= 9600)
    assert not any(c for ph, c in seen if ph <= 4800 - F or ph > 9600 + F)


def test_audio_outside_the_cut_is_bit_exact():
    rng = np.random.default_rng(3)
    x = rng.uniform(-1, 1, 60000).astype(np.float32)
    p = LookaheadProcessor(L_MS)
    s, e = 20000, 24000
    p.submit_cut(cut(1, s, e))
    y = run(p, x)[L:]
    assert np.array_equal(y[:s - F + 1], x[:s - F + 1])
    assert np.all(y[s:e] == 0.0)
    assert np.array_equal(y[e + F:], x[e + F:60000 - L])


def test_dip_mode_holds_the_dip_level():
    p = LookaheadProcessor(L_MS, fade_ms=15, mode="dip", dip_db=-30.0)
    s, e = 30000, 36000
    p.submit_cut(cut(1, s, e))
    g = gain_by_sample(p, 60000)
    floor = 10 ** (-30 / 20)
    assert np.allclose(g[s:e], floor, atol=1e-6)
    assert np.all(g >= floor - 1e-6) and np.all(g <= 1.0)
    assert np.all(g[:s - F + 1] == 1.0) and np.all(g[e + F - 1:] == 1.0)
    assert max_step(g) <= p.fade_step + 1e-7


# ------------------------------------------------------------------ late cuts

def test_late_cut_fades_from_the_playhead_and_reports_lateness():
    p = LookaheadProcessor(L_MS)
    late_by = 1000
    box = {}

    def submit():
        ph = p.playhead
        box["ph"] = ph
        box["late"] = p.submit_cut(cut(1, ph - late_by, ph + 6000))

    blocks_before = (L + 20000) // BLOCK
    g = gain_by_sample(p, 60000, events={blocks_before: submit})
    ph = box["ph"]
    assert box["late"] == pytest.approx(late_by * 1000 / SAMPLE_RATE)
    assert np.all(g[:ph] == 1.0)                         # what already played stays as it was
    assert g[ph] < 1.0                                   # the fade starts at once
    assert np.all(g[ph + F - 1:ph + 6000] == 0.0)        # silent one fade later
    assert max_step(g) <= p.fade_step + 1e-7


def test_cut_inside_the_fade_room_is_not_late_and_does_not_click():
    p = LookaheadProcessor(L_MS)
    box = {}

    def submit():
        box["ph"] = p.playhead
        box["late"] = p.submit_cut(cut(1, p.playhead + 100, p.playhead + 5000))

    g = gain_by_sample(p, 60000, events={(L + 20000) // BLOCK: submit})
    assert box["late"] == 0.0
    assert max_step(g) <= p.fade_step + 1e-7
    assert np.all(g[box["ph"] + F:box["ph"] + 5000] == 0.0)


def test_on_time_cut_returns_zero():
    p = LookaheadProcessor(L_MS)
    p.process(np.zeros(48000, dtype=np.float32))
    assert p.submit_cut(cut(1, p.playhead, p.playhead + 100)) == 0.0
    assert p.submit_cut(cut(2, p.playhead - 48, p.playhead + 100)) == pytest.approx(1.0)


# ------------------------------------------------------------------ merging / replacing

def test_overlapping_and_adjacent_cuts_merge_into_one():
    p = LookaheadProcessor(L_MS)
    p.submit_cut(cut(1, 20000, 24000))
    p.submit_cut(cut(2, 23000, 27000))                   # overlaps
    p.submit_cut(cut(3, 27000, 30000))                   # touches
    p.submit_cut(cut(4, 30000 + F // 2, 33000))          # gap shorter than a fade: stays shut
    g = gain_by_sample(p, 60000)
    assert np.all(g[20000:33000] == 0.0)
    assert g[19999] > 0.0 and g[33000] > 0.0
    assert max_step(g) <= p.fade_step + 1e-7


def test_cuts_a_little_apart_reopen_partly_without_a_click():
    p = LookaheadProcessor(L_MS)
    p.submit_cut(cut(1, 20000, 24000))
    p.submit_cut(cut(2, 24000 + F + F // 2, 28000))
    g = gain_by_sample(p, 60000)
    assert np.all(g[20000:24000] == 0.0) and np.all(g[24000 + F + F // 2:28000] == 0.0)
    between = g[24000:24000 + F + F // 2]
    assert 0.0 < between.max() < 1.0
    assert max_step(g) <= p.fade_step + 1e-7


def test_same_cut_id_replaces_and_extends():
    p = LookaheadProcessor(L_MS)
    p.submit_cut(cut(7, 30000, 31000))
    events = {}
    # keep extending the same cut while it is being played, like a stretched word
    for i, end in enumerate(range(32000, 40001, 1000)):
        at_block = (L + 30200 + i * 900) // BLOCK
        events[at_block] = (lambda e=end: p.submit_cut(cut(7, 30000, e)))
    g = gain_by_sample(p, 70000, events=events)
    assert np.all(g[30000:40000] == 0.0)
    assert g[40000] > 0.0
    assert np.all(g[40000 + F:] == 1.0)
    assert max_step(g) <= p.fade_step + 1e-7


def test_replace_can_move_or_cancel_a_cut():
    p = LookaheadProcessor(L_MS)
    p.submit_cut(cut(1, 20000, 21000))
    p.submit_cut(cut(1, 30000, 31000))                   # replaces: the first span is not cut
    p.submit_cut(cut(2, 40000, 41000))
    p.submit_cut(cut(2, 40000, 40000))                   # empty = cancel
    g = gain_by_sample(p, 60000)
    assert np.all(g[19000:22000] == 1.0)
    assert np.all(g[30000:31000] == 0.0)
    assert np.all(g[39000:42000] == 1.0)


# ------------------------------------------------------------------ changing things while running

def settle_blocks(n_samples: int) -> int:
    return -(-n_samples // BLOCK)


def test_set_lookahead_is_click_free_and_keeps_the_clock():
    for new_ms in (350, 600):
        rng = np.random.default_rng(4)
        x = rng.uniform(-1, 1, 144000).astype(np.float32)
        p = LookaheadProcessor(L_MS)
        new = ms_to_samples(new_ms)
        change_block = 100
        y = run(p, x, events={change_block: lambda: p.set_lookahead(new_ms)})
        assert p.captured == len(x)
        assert p.playhead == len(x) - new
        assert p.lookahead_ms == pytest.approx(new_ms)
        t0 = change_block * BLOCK
        # before the change: old delay, exact
        assert np.array_equal(y[L:t0], x[:t0 - L])
        # well within 1 s after it: new delay, exact
        t1 = t0 + 2 * F + BLOCK
        assert t1 - t0 < SAMPLE_RATE
        assert np.array_equal(y[t1:], x[t1 - new:len(x) - new])
        # and the gain went smoothly through zero in between
        g = run(_changed(new_ms, change_block), np.ones(len(x), dtype=np.float32),
                events=None)
        assert max_step(g[L:]) <= p.fade_step + 1e-7


def _changed(new_ms: float, change_block: int) -> LookaheadProcessor:
    """A processor that changes its look-ahead by itself at block `change_block`."""
    p = LookaheadProcessor(L_MS)
    real = p.process
    count = {"n": 0}

    def process(block):
        if count["n"] == change_block:
            p.set_lookahead(new_ms)
        count["n"] += 1
        return real(block)

    p.process = process  # type: ignore[method-assign]
    return p


def test_set_lookahead_fades_to_zero_at_the_switch():
    p = LookaheadProcessor(L_MS)
    y = run(p, np.ones(96000, dtype=np.float32), events={100: lambda: p.set_lookahead(600)})
    t0 = 100 * BLOCK
    assert np.all(y[L:t0] == 1.0)
    assert y[t0 + F - 1] == 0.0
    assert np.all(y[t0 + 2 * F:] == 1.0)
    assert max_step(y[L:]) <= p.fade_step + 1e-7


@pytest.mark.parametrize("new_ms", [350, 600])
@pytest.mark.parametrize("offset", [-2000, -300, 0, 300, 2000, 5000])
def test_cut_during_set_lookahead_no_click_and_no_leak(new_ms, offset):
    change_block = 100
    t0 = change_block * BLOCK                    # capture index when the change is asked for
    s = t0 - L + offset                          # a cut around what is playing at that moment
    e = s + 4000

    # gain view: ones in
    p = LookaheadProcessor(L_MS)
    p.submit_cut(cut(1, s, e))
    y = run(p, np.ones(96000, dtype=np.float32), events={change_block: lambda: p.set_lookahead(new_ms)})
    assert max_step(y[L:]) <= p.fade_step + 1e-7
    assert np.all(y >= 0.0) and np.all(y <= 1.0)

    # leak view: the signal is 1 only inside the cut, so anything non-zero coming out is a leak
    p2 = LookaheadProcessor(L_MS)
    p2.submit_cut(cut(1, s, e))
    x = np.zeros(96000, dtype=np.float32)
    x[s:e] = 1.0
    y2 = run(p2, x, events={change_block: lambda: p2.set_lookahead(new_ms)})
    assert np.all(y2 == 0.0)


def test_cut_submitted_in_the_middle_of_a_lookahead_change():
    p = LookaheadProcessor(L_MS)
    events = {
        100: lambda: p.set_lookahead(600),
        101: lambda: p.submit_cut(cut(1, p.playhead - 200, p.playhead + 3000)),
        102: lambda: p.set_lookahead(350),
        103: lambda: p.submit_cut(cut(2, p.playhead + 100, p.playhead + 900)),
    }
    y = run(p, np.ones(96000, dtype=np.float32), events=events)
    assert max_step(y[L:]) <= p.fade_step + 1e-7
    assert p.playhead == 96000 - ms_to_samples(350)
    assert np.all(y[-9600:] == 1.0)


def test_set_lookahead_before_start_applies_at_once():
    p = LookaheadProcessor(L_MS)
    p.set_lookahead(500)
    assert p.playhead == -ms_to_samples(500)
    x = np.arange(48000, dtype=np.float32)
    y = p.process(x)
    assert np.array_equal(y[ms_to_samples(500):], x[:-ms_to_samples(500)])


def test_set_mode_in_the_middle_of_a_cut_does_not_click():
    p = LookaheadProcessor(L_MS, fade_ms=15, mode="silence")
    s, e = 30000, 40000
    p.submit_cut(cut(1, s, e))
    p.submit_cut(cut(2, 50000, 52000))
    mid = (L + 35000) // BLOCK
    g = gain_by_sample(p, 80000, events={mid: lambda: p.set_mode("dip", -20.0, 10)})
    old_step = float(np.max(np.abs(np.diff(0.5 * (1 + np.cos(np.pi * np.arange(F + 1) / F))))))
    new_f = ms_to_samples(10)
    new_step = float(np.max(np.abs(np.diff(0.5 * (1 + np.cos(np.pi * np.arange(new_f + 1) / new_f))))))
    assert max_step(g[:45000]) <= old_step + 1e-7        # the running cut finishes the old way
    assert np.all(g[s:e] == 0.0)
    assert max_step(g) <= new_step + 1e-7
    assert np.allclose(g[50000:52000], 10 ** (-20 / 20), atol=1e-6)   # the next one uses the new mode
    assert p.fade_samples == new_f


def test_big_block_equals_small_blocks():
    rng = np.random.default_rng(5)
    x = rng.uniform(-1, 1, 100000).astype(np.float32)
    a, b = LookaheadProcessor(L_MS), LookaheadProcessor(L_MS)
    for p in (a, b):
        p.submit_cut(cut(1, 10000, 12000))
        p.submit_cut(cut(2, 50000, 50100))
    assert np.array_equal(run(a, x, block=len(x)), run(b, x, block=480))


# ------------------------------------------------------------------ threads

def test_thread_hammer_cuts_from_another_thread_never_click():
    p = LookaheadProcessor(L_MS)
    stop = threading.Event()
    errors = []
    lates = []

    def hammer():
        rng = np.random.default_rng(6)
        cid = 0
        try:
            while not stop.is_set():
                ph = p.playhead
                start = ph + int(rng.integers(-3000, 12000))
                length = int(rng.integers(1, 4000))
                cid = cid + 1 if rng.random() < 0.6 else max(1, cid - int(rng.integers(0, 3)))
                lates.append(p.submit_cut(cut(cid, start, start + length)))
                if rng.random() < 0.02:
                    p.set_lookahead(float(rng.choice([350, 450, 600])))
        except Exception as ex:                  # pragma: no cover
            errors.append(ex)

    t = threading.Thread(target=hammer)
    t.start()
    try:
        outs = [p.process(np.ones(BLOCK, dtype=np.float32)) for _ in range(3000)]
    finally:
        stop.set()
        t.join()
    assert not errors
    assert len(lates) > 100 and all(v >= 0.0 for v in lates)
    y = np.concatenate(outs)[ms_to_samples(600):]
    assert np.all(y >= 0.0) and np.all(y <= 1.0)
    assert max_step(y) <= p.fade_step + 1e-7
    assert p.captured == 3000 * BLOCK
