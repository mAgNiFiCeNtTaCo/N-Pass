"""AudioEngine with fake devices, and the device lookup with a mocked sounddevice.
Nothing here opens a real sound device."""
from __future__ import annotations

import sys
from pathlib import Path

# plain `uv run pytest` does not put the project root on the path (no conftest, package = false)
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time

import numpy as np
import pytest

from npass.audio import devices
from npass.audio.devices import DeviceInfo
from npass.audio.engine import CABLE_MISSING, AudioEngine, JitterFifo
from npass.settings import Settings
from npass.types import SAMPLE_RATE, CutOrder, ms_to_samples

BLOCK = 480
MIC = DeviceInfo(index=1, name="Microphone (NVIDIA Broadcast)", max_in=2, max_out=0, default_rate=48000.0)
OTHER_MIC = DeviceInfo(index=2, name="Microphone (USB Thing)", max_in=1, max_out=0, default_rate=44100.0)
LOOP = DeviceInfo(index=3, name="CABLE Output (VB-Audio Virtual Cable)", max_in=2, max_out=0, default_rate=48000.0)
SPEAKERS = DeviceInfo(index=4, name="Speakers (Realtek Audio)", max_in=0, max_out=2, default_rate=48000.0)
CABLE = DeviceInfo(index=5, name="CABLE Input (VB-Audio Virtual Cable)", max_in=0, max_out=2, default_rate=48000.0)


class FakeStream:
    def __init__(self, dev, callback, latency=0.010):
        self.dev = dev
        self.callback = callback
        self.latency = latency
        self.active = True
        self.closed = False

    def abort(self):
        self.active = False

    def close(self):
        self.active = False
        self.closed = True


class FakeBackend:
    def __init__(self, ins, outs, default_in=None):
        self.ins = list(ins)
        self.outs = list(outs)
        self.default_in = default_in
        self.fp = 0
        self.rescans = 0
        self.in_streams: list[FakeStream] = []
        self.out_streams: list[FakeStream] = []
        self.fail_input = False
        self.fail_output = False
        self.out_latency = 0.010

    def rescan(self):
        self.rescans += 1

    def fingerprint(self):
        return self.fp

    def inputs(self):
        return list(self.ins)

    def outputs(self):
        return list(self.outs)

    def default_input(self):
        return self.default_in

    def open_input(self, dev, blocksize, callback):
        if self.fail_input:
            raise RuntimeError("Error opening InputStream: Invalid device [PaErrorCode -9996]")
        s = FakeStream(dev, callback)
        self.in_streams.append(s)
        return s, abs(dev.default_rate - SAMPLE_RATE) >= 1

    def open_output(self, dev, blocksize, callback):
        if self.fail_output:
            raise RuntimeError("device busy")
        s = FakeStream(dev, callback, self.out_latency)
        self.out_streams.append(s)
        return s, abs(dev.default_rate - SAMPLE_RATE) >= 1

    # helpers for the tests
    def mic(self) -> FakeStream:
        return self.in_streams[-1]

    def cable(self) -> FakeStream:
        return self.out_streams[-1]


@pytest.fixture(autouse=True)
def _data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NPASS_DATA_DIR", str(tmp_path))


def make(backend, tap=None, **over) -> AudioEngine:
    s = Settings()
    for k, v in over.items():
        setattr(s, k, v)
    return AudioEngine(s, tap=tap, backend=backend, supervisor=False)


def pump(eng, backend, blocks, value=0.5, pull=True):
    """One mic block in, one cable block out, `blocks` times. Returns everything the cable got."""
    got = []
    for _ in range(blocks):
        backend.mic().callback(np.full(BLOCK, value, dtype=np.float32), 0.010, False)
        if pull and backend.out_streams and backend.cable().active:
            got.append(backend.cable().callback(BLOCK, 0.010))
    return np.concatenate(got) if got else np.zeros(0, dtype=np.float32)


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


# ------------------------------------------------------------------ device lookup (mocked sounddevice)

class FakeSd:
    def __init__(self, devs):
        self._devs = devs

    def query_hostapis(self, index=None):
        apis = [
            {"name": "MME", "devices": [0, 1], "default_input_device": 0, "default_output_device": 1},
            {"name": "Windows WASAPI", "devices": [2, 3, 4, 5, 6], "default_input_device": 2,
             "default_output_device": 3},
        ]
        return apis if index is None else apis[index]

    def query_devices(self):
        return self._devs


def _dev(i, name, api, cin, cout, rate=48000.0):
    return {"index": i, "name": name, "hostapi": api, "max_input_channels": cin,
            "max_output_channels": cout, "default_samplerate": rate}


NO_CABLE = [
    _dev(0, "Microphone (NVIDIA Broadcast", 0, 2, 0),
    _dev(1, "CABLE Input (VB-Audio Virtual C", 0, 0, 2),      # MME copy: must be ignored
    _dev(2, "Microphone (NVIDIA Broadcast)", 1, 2, 0),
    _dev(3, "Speakers (Realtek Audio)", 1, 0, 2),
    _dev(4, "Speakers (Realtek Audio) [Loopback]", 1, 2, 0),
]
WITH_CABLE = NO_CABLE + [
    _dev(5, "CABLE Input (VB-Audio Virtual Cable)", 1, 0, 2),
    _dev(6, "CABLE Output (VB-Audio Virtual Cable)", 1, 2, 0),
]


def test_find_cable_returns_none_when_not_installed(monkeypatch):
    monkeypatch.setattr(devices, "sd", FakeSd(NO_CABLE))
    assert devices.find_cable() is None
    assert devices.list_outputs() == ["Speakers (Realtek Audio)"]
    assert devices.list_inputs() == ["Microphone (NVIDIA Broadcast)"]   # WASAPI only, no loopback


def test_find_cable_finds_the_playback_end(monkeypatch):
    monkeypatch.setattr(devices, "sd", FakeSd(WITH_CABLE))
    assert devices.find_cable() == "CABLE Input (VB-Audio Virtual Cable)"
    assert "CABLE Output (VB-Audio Virtual Cable)" in devices.list_inputs()
    assert devices.default_input_device().name == "Microphone (NVIDIA Broadcast)"


def test_lookup_without_the_sound_library(monkeypatch):
    monkeypatch.setattr(devices, "sd", None)
    assert devices.find_cable() is None
    assert devices.list_inputs() == [] and devices.list_outputs() == []


def test_match_device_by_substring_any_case():
    devs = [MIC, OTHER_MIC]
    assert devices.match_device(devs, "nvidia broadcast") is MIC
    assert devices.match_device(devs, "USB") is OTHER_MIC
    assert devices.match_device(devs, "") is None
    assert devices.match_device(devs, "nothing like it") is None


# ------------------------------------------------------------------ no cable

def test_no_cable_reports_the_warning_and_opens_no_output():
    taps = []
    be = FakeBackend([MIC], [SPEAKERS])
    eng = make(be, tap=lambda b, s: taps.append((b.copy(), s)))
    eng.start()
    try:
        st = eng.status()
        assert st.running and st.input_ok and not st.output_ok
        assert CABLE_MISSING in st.warnings
        assert st.output_name == ""
        assert be.out_streams == []                       # never the speakers
        # the mic side still runs: meter and tap
        pump(eng, be, 20, value=0.5, pull=False)
        st = eng.status()
        assert st.level_db == pytest.approx(20 * np.log10(0.5), abs=0.1)
        assert wait_for(lambda: len(taps) == 20)
        assert [s for _, s in taps] == [i * BLOCK for i in range(20)]
        assert all(b.dtype == np.float32 and len(b) == BLOCK for b, _ in taps)
        # retried every 2 s, still nothing opened
        for k in range(1, 6):
            eng._tick(time.monotonic() + 2.1 * k)
        assert be.out_streams == []
        assert CABLE_MISSING in eng.status().warnings
    finally:
        eng.stop()


def test_cable_installed_later_is_picked_up():
    be = FakeBackend([MIC], [SPEAKERS])
    eng = make(be)
    eng.start()
    try:
        assert not eng.status().output_ok
        pump(eng, be, 10, pull=False)
        clock_before = eng.processor.captured
        be.outs.append(CABLE)
        be.fp = 1                                         # Windows' device list changed
        now = time.monotonic() + 2.1
        eng._tick(now)
        st = eng.status()
        assert st.output_ok and st.input_ok
        assert st.output_name == CABLE.name
        assert CABLE_MISSING not in st.warnings
        assert be.cable().dev is CABLE
        assert eng.processor.captured >= clock_before     # the clock was not reset
    finally:
        eng.stop()


def test_cable_open_error_is_reported_and_retried():
    be = FakeBackend([MIC], [SPEAKERS, CABLE])
    be.fail_output = True
    eng = make(be)
    eng.start()
    try:
        st = eng.status()
        assert not st.output_ok
        assert any("could not be opened" in w and "CABLE Input" in w for w in st.warnings)
        be.fail_output = False
        eng._tick(time.monotonic() + 2.1)
        assert eng.status().output_ok
        assert not any("cable" in w.lower() for w in eng.status().warnings)
    finally:
        eng.stop()


# ------------------------------------------------------------------ normal running

def test_audio_reaches_the_cable_delayed_and_cut():
    be = FakeBackend([MIC, LOOP], [SPEAKERS, CABLE])
    eng = make(be, lookahead_ms=450)
    eng.start()
    try:
        st = eng.status()
        assert st.input_ok and st.output_ok and st.warnings == []
        assert st.input_name == MIC.name and st.output_name == CABLE.name
        look = ms_to_samples(450)
        late = eng.submit_cut(CutOrder(cut_id=1, start=48000, end=52800, word="pineapple"))
        assert late == 0.0
        out = pump(eng, be, 300, value=0.5)
        assert len(out) == 300 * BLOCK
        # the only silent stretch after the start-up is the cut: exactly as long as ordered,
        # and shifted by the whole delay (look-ahead + what waits in the jitter FIFO)
        zeros = np.flatnonzero(out[look + 8 * BLOCK:] == 0.0) + look + 8 * BLOCK
        a, b = int(zeros[0]), int(zeros[-1]) + 1
        assert b - a == 4800 and len(zeros) == 4800
        fifo_delay = a - 48000 - look
        assert 0 <= fifo_delay <= 6 * BLOCK
        assert np.all(out[:look] == 0.0)                  # nothing comes out early
        assert np.allclose(out[look + 8 * BLOCK:a - 1000], 0.5)
        assert np.allclose(out[b + 2000:], 0.5)
        assert eng.processor.captured == 300 * BLOCK
        st = eng.status()
        assert st.lookahead_ms == pytest.approx(450)
        assert st.cutting is False
    finally:
        eng.stop()


def test_measured_delay_adds_up():
    be = FakeBackend([MIC], [CABLE])
    eng = make(be, lookahead_ms=450)
    eng.start()
    try:
        pump(eng, be, 200)
        now = time.monotonic()
        for i in range(60):
            eng._tick(now + 0.01 * i)
        st = eng.status()
        fifo_ms = eng._fifo.smoothed_fill * 1000 / SAMPLE_RATE
        assert 20 <= fifo_ms <= 50
        assert st.measured_delay_ms == pytest.approx(10 + 450 + fifo_ms + 10, abs=1.0)
        assert not any("delay" in w for w in st.warnings)
    finally:
        eng.stop()


def test_delay_over_the_limit_warns_and_shrinks_the_fifo():
    be = FakeBackend([MIC], [CABLE])
    be.out_latency = 0.300
    eng = make(be, lookahead_ms=600)
    eng._fifo.set_target(ms_to_samples(60))
    eng._fifo_target0 = eng._fifo.target
    eng.start()
    try:
        for _ in range(200):
            be.mic().callback(np.zeros(BLOCK, dtype=np.float32), 0.010, False)
            be.cable().callback(BLOCK, 0.300)
        now = time.monotonic()
        for i in range(60):
            eng._tick(now + 0.01 * i)
        st = eng.status()
        assert st.measured_delay_ms > 700
        assert any("700 ms limit" in w for w in st.warnings)
        assert eng._fifo.target == eng._fifo.min_target
    finally:
        eng.stop()


def test_sample_rate_mismatch_gives_a_plain_warning():
    be = FakeBackend([OTHER_MIC], [CABLE])
    eng = make(be, input_device="USB Thing")
    eng.start()
    try:
        st = eng.status()
        assert st.input_ok
        w = [x for x in st.warnings if "44100 Hz" in x]
        assert len(w) == 1
        assert OTHER_MIC.name in w[0] and "Advanced -> 48000 Hz" in w[0]
    finally:
        eng.stop()


def test_hint_not_found_falls_back_but_never_to_the_cable_loop():
    be = FakeBackend([LOOP, OTHER_MIC], [CABLE], default_in=LOOP)
    eng = make(be)                                        # hint is "NVIDIA Broadcast": not there
    eng.start()
    try:
        st = eng.status()
        assert st.input_ok and st.input_name == OTHER_MIC.name
        assert any("NVIDIA Broadcast" in w and OTHER_MIC.name in w for w in st.warnings)
    finally:
        eng.stop()


# ------------------------------------------------------------------ mic trouble

def test_mic_unplugged_then_back_keeps_the_clock():
    taps = []
    be = FakeBackend([MIC], [CABLE])
    eng = make(be, tap=lambda b, s: taps.append((s, len(b))), input_device="NVIDIA Broadcast")
    eng.start()
    try:
        pump(eng, be, 100)
        before = eng.processor.captured
        assert before == 100 * BLOCK
        now = time.monotonic()

        be.mic().active = False                           # the device dies
        be.ins.clear()
        be.fp = 1
        eng._tick(now + 0.1)
        st = eng.status()
        assert not st.input_ok and st.output_ok is not None
        assert any("NVIDIA Broadcast" in w and "not connected" in w for w in st.warnings)
        assert st.level_db == -120.0

        # the clock keeps counting on silence at real-time pace, and silence goes to the cable
        eng._tick(now + 0.6)
        mid = eng.processor.captured
        assert before + 0.4 * SAMPLE_RATE <= mid <= before + 0.6 * SAMPLE_RATE
        for k in range(2, 6):
            eng._tick(now + 0.1 + k)                      # retries keep failing quietly
        assert not eng.status().input_ok

        be.ins.append(MIC)                                # plugged back in
        be.fp = 2
        t_back = now + 7.5
        eng._tick(t_back)
        st = eng.status()
        assert st.input_ok and st.input_name == MIC.name
        assert not any("not connected" in w for w in st.warnings)
        resumed_at = eng.processor.captured
        assert resumed_at > mid                           # never back to 0
        pump(eng, be, 10)
        assert eng.processor.captured == resumed_at + 10 * BLOCK

        # every block the tap got follows the one before: no gap, no restart
        assert wait_for(lambda: sum(n for _, n in taps) == eng.processor.captured)
        pos = 0
        for s, n in taps:
            assert s == pos
            pos += n
    finally:
        eng.stop()


def test_mic_that_stops_calling_back_is_noticed():
    be = FakeBackend([MIC], [CABLE])
    eng = make(be)
    eng.start()
    try:
        pump(eng, be, 5)
        first = be.mic()
        eng._tick(time.monotonic() + 1.5)                 # no callback for 1.5 s
        assert first.closed
        assert be.rescans >= 1
    finally:
        eng.stop()


def test_start_never_raises_and_stop_is_idempotent():
    be = FakeBackend([MIC], [CABLE])
    be.fail_input = True
    be.fail_output = True
    eng = make(be)
    eng.stop()                                            # stop before start: fine
    eng.start()
    eng.start()                                           # twice: fine
    st = eng.status()
    assert st.running and not st.input_ok and not st.output_ok
    assert any("could not be opened" in w for w in st.warnings)
    eng.stop()
    eng.stop()
    assert not eng.status().running

    # no devices at all
    eng2 = make(FakeBackend([], []))
    eng2.start()
    st = eng2.status()
    assert not st.input_ok and not st.output_ok
    assert CABLE_MISSING in st.warnings
    assert any("icrophone" in w for w in st.warnings)
    eng2.stop()


def test_restart_carries_on_counting():
    be = FakeBackend([MIC], [CABLE])
    eng = make(be)
    eng.start()
    pump(eng, be, 10)
    eng.stop()
    assert all(s.closed for s in be.in_streams + be.out_streams)
    eng.start()
    pump(eng, be, 10)
    assert eng.processor.captured == 20 * BLOCK
    eng.stop()


def test_real_supervisor_thread_starts_and_stops():
    be = FakeBackend([MIC], [SPEAKERS])
    eng = AudioEngine(Settings(), tap=lambda b, s: None, backend=be)
    eng.start()
    try:
        assert wait_for(lambda: eng._sup_thread is not None and eng._sup_thread.is_alive())
        be.mic().callback(np.zeros(BLOCK, dtype=np.float32), None, False)
    finally:
        eng.stop()
    assert eng._sup_thread is None and eng._tap_thread is None
    assert be.out_streams == []


def test_set_input_switches_microphone():
    be = FakeBackend([MIC, OTHER_MIC], [CABLE])
    eng = make(be)
    eng.start()
    try:
        assert eng.status().input_name == MIC.name
        eng.set_input("USB Thing")
        st = eng.status()
        assert st.input_ok and st.input_name == OTHER_MIC.name
        assert be.in_streams[0].closed
    finally:
        eng.stop()


# ------------------------------------------------------------------ jitter FIFO

def _sine(n, start=0, freq=200.0):
    t = (np.arange(n) + start) / SAMPLE_RATE
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.mark.parametrize("push_len", [479, 481])       # mic clock slower / faster than the cable
def test_fifo_holds_its_target_when_the_clocks_drift(push_len):
    target = ms_to_samples(30)
    f = JitterFifo(target, BLOCK)
    pos = 0
    out = []
    fills = []
    for _ in range(6000):                               # one minute, 2000 ppm apart
        f.push(_sine(push_len, pos))
        pos += push_len
        fills.append(f.fill)
        out.append(f.pull(BLOCK))
    y = np.concatenate(out)
    assert f.underruns == 0 and f.overflows == 0
    assert f.corrections > 0
    late = np.array(fills[1000:])
    assert late.min() > target - BLOCK and late.max() < target + BLOCK    # the delay does not creep
    # no click: a 200 Hz sine never moves faster than this between two samples
    live = y[2 * target:]
    assert np.max(np.abs(np.diff(live))) < 2 * np.pi * 200 / SAMPLE_RATE * 1.5


def test_fifo_underrun_goes_quiet_then_recovers():
    f = JitterFifo(ms_to_samples(30), BLOCK)
    assert np.all(f.pull(BLOCK) == 0.0)                 # nothing yet
    for _ in range(3):                                  # exactly the 30 ms target
        f.push(np.ones(BLOCK, dtype=np.float32))
    a = f.pull(BLOCK)
    assert a[0] == 0.0 and a[-1] == 1.0                 # fades in
    f.pull(BLOCK), f.pull(BLOCK)
    f.push(np.ones(100, dtype=np.float32))
    b = f.pull(BLOCK)                                   # runs dry: fades out, rest silence
    assert f.underruns == 1
    assert b[0] == 1.0 and np.all(b[100:] == 0.0)
    assert np.all(f.pull(BLOCK) == 0.0)                 # refilling


def test_fifo_does_not_grow_when_nobody_pulls():
    f = JitterFifo(ms_to_samples(30), BLOCK)
    for _ in range(1000):
        f.push(np.ones(BLOCK, dtype=np.float32))
    assert f.fill <= SAMPLE_RATE
