"""Fault tests on the device side: the mic drops and comes back, the virtual cable is missing.

The real AudioEngine, Detector and SafetyGuard run together (and, further down, the real AppCore
that `N-Pass.cmd` starts), on a fake device layer and the scripted FakeRecogniser. No sound device
is opened, no model is loaded, the network is not used.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

from npass.audio import devices
from npass.audio.devices import DeviceInfo
from npass.audio.engine import CABLE_MISSING, AudioEngine
from npass.detect import Detector, FakeRecogniser, SafetyGuard, ScriptedWord
from npass.settings import Settings
from npass.types import SAMPLE_RATE, CutReason, Health, ms_to_samples

BLOCK = 480
WORDS = ["pineapple", "banana"]
MIC = DeviceInfo(index=1, name="Microphone (Test Mic)", max_in=1, max_out=0, default_rate=48000.0)
SPEAKERS = DeviceInfo(index=4, name="Speakers (Test Audio)", max_in=0, max_out=2, default_rate=48000.0)
CABLE = DeviceInfo(index=5, name="CABLE Input (VB-Audio Virtual Cable)", max_in=0, max_out=2,
                   default_rate=48000.0)


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NPASS_DATA_DIR", str(tmp_path))


class FakeStream:
    def __init__(self, dev, callback):
        self.dev, self.callback = dev, callback
        self.latency = 0.010
        self.active = True

    def abort(self):
        self.active = False

    def close(self):
        self.active = False


class FakeBackend:
    """Stands in for sounddevice: lists devices, hands out streams that never touch hardware."""

    def __init__(self, ins, outs):
        self.ins, self.outs = list(ins), list(outs)
        self.fp = 0
        self.in_streams: list[FakeStream] = []
        self.out_streams: list[FakeStream] = []

    def rescan(self):
        pass

    def fingerprint(self):
        return self.fp

    def inputs(self):
        return list(self.ins)

    def outputs(self):
        return list(self.outs)

    def default_input(self):
        return self.ins[0] if self.ins else None

    def open_input(self, dev, blocksize, callback):
        s = FakeStream(dev, callback)
        self.in_streams.append(s)
        return s, False

    def open_output(self, dev, blocksize, callback):
        s = FakeStream(dev, callback)
        self.out_streams.append(s)
        return s, False


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


class Rig:
    """Engine (fake devices, hand-driven supervisor) -> tap -> detector -> cuts -> engine."""

    def __init__(self, backend, script=()):
        self.settings = Settings()
        self.settings.input_hint = "Test Mic"
        self.be = backend
        self.rec = FakeRecogniser(list(script))
        self.orders = []
        self.lates = []
        self.det = Detector(self.settings, lambda: list(WORDS), lambda a, b: [], self.submit,
                            lambda order: None, recogniser=self.rec)
        self.eng = AudioEngine(self.settings, tap=self.det.feed, backend=backend, supervisor=False)
        self.guard = SafetyGuard(self.submit)
        self.cable_audio: list[np.ndarray] = []

    def submit(self, order):
        self.orders.append(order)
        self.lates.append(self.eng.submit_cut(order))

    def settle(self) -> None:
        """Let the tap thread hand everything to the detector, then let the recogniser run."""
        assert wait_for(lambda: self.det._captured == self.eng.processor.captured)
        self.det.process_pending()

    def speak(self, blocks: int, value: float = 0.3) -> None:
        for _ in range(blocks):
            self.be.in_streams[-1].callback(np.full(BLOCK, value, dtype=np.float32), 0.010, False)
            if self.be.out_streams and self.be.out_streams[-1].active:
                self.cable_audio.append(self.be.out_streams[-1].callback(BLOCK, 0.010))
            self.settle()


# ------------------------------------------------------------------ the mic drops and returns

def test_mic_drop_keeps_the_clock_counting_and_the_filter_working_afterwards():
    be = FakeBackend([MIC], [SPEAKERS, CABLE])
    rig = Rig(be)
    rig.eng.start()
    try:
        rig.speak(100)                                   # 1 s of voice
        assert rig.eng.status().input_ok and rig.eng.status().warnings == []
        before = rig.eng.processor.captured
        assert before == 100 * BLOCK

        # the mic is unplugged for 2 s
        be.ins.clear()
        be.in_streams[-1].active = False
        t0 = time.monotonic()
        for k in range(101):
            rig.eng._tick(t0 + 0.02 * k)
        st = rig.eng.status()
        assert not st.input_ok and st.output_ok
        assert any("is not connected" in w and "Test Mic" in w for w in st.warnings)
        during = rig.eng.processor.captured
        assert during - before == pytest.approx(2.0 * SAMPLE_RATE, abs=2 * BLOCK)   # silence, in step
        rig.settle()

        # it is plugged back in
        be.ins.append(MIC)
        rig.eng._tick(t0 + 3.2)
        st = rig.eng.status()
        assert st.input_ok and st.input_name == MIC.name
        assert not any("not connected" in w for w in st.warnings)
        assert len(be.in_streams) == 2

        # a blocked word said after the return is cut at the right place on the same clock
        now = rig.eng.processor.captured
        word = ScriptedWord("banana", now + ms_to_samples(300), now + ms_to_samples(700))
        rig.rec.script.append(word)
        rig.speak(150)
        assert rig.det.stats["gaps"] == 0                # the detector never saw a hole in the audio
        assert rig.det.health() == Health.OK
        cuts = [o for o in rig.orders if o.reason == CutReason.ASR]
        assert cuts and cuts[0].word == "banana"
        final = [o for o in rig.orders if o.cut_id == cuts[0].cut_id][-1]
        assert final.start <= word.start and final.end >= word.end
        assert rig.lates[rig.orders.index(cuts[0])] == 0.0          # ordered before it played
        assert rig.eng.processor.captured == now + 150 * BLOCK
    finally:
        rig.eng.stop()


def test_mic_drop_sends_silence_to_the_cable_not_a_frozen_buffer():
    be = FakeBackend([MIC], [CABLE])
    rig = Rig(be)
    rig.eng.start()
    try:
        rig.speak(120)
        be.ins.clear()
        be.in_streams[-1].active = False
        t0 = time.monotonic()
        got = []
        for k in range(151):                              # 3 s without a mic
            rig.eng._tick(t0 + 0.02 * k)
            for _ in range(2):
                got.append(be.out_streams[-1].callback(BLOCK, 0.010))
        out = np.concatenate(got)
        tail = out[-SAMPLE_RATE:]                         # the last second: the old voice has drained
        assert np.all(tail == 0.0)
        assert rig.eng.status().output_ok
    finally:
        rig.eng.stop()


# ------------------------------------------------------------------ the cable is missing

def test_cable_missing_never_falls_back_to_the_speakers_and_detection_still_runs():
    be = FakeBackend([MIC], [SPEAKERS])
    word = ScriptedWord("pineapple", ms_to_samples(1000), ms_to_samples(1500))
    rig = Rig(be, script=[word])
    rig.eng.start()                                       # must not raise
    try:
        st = rig.eng.status()
        assert st.running and st.input_ok and not st.output_ok
        assert CABLE_MISSING in st.warnings
        rig.speak(250)
        assert be.out_streams == []                       # no output was ever opened
        assert rig.cable_audio == []
        cuts = [o for o in rig.orders if o.reason == CutReason.ASR]
        assert cuts and cuts[0].word == "pineapple"       # the filter itself keeps working
        t0 = time.monotonic()
        for k in range(1, 6):
            rig.eng._tick(t0 + 2.1 * k)
        assert be.out_streams == []
        assert CABLE_MISSING in rig.eng.status().warnings
    finally:
        rig.eng.stop()


def test_cable_installed_while_running_is_used_without_a_restart():
    be = FakeBackend([MIC], [SPEAKERS])
    rig = Rig(be)
    rig.eng.start()
    try:
        rig.speak(20)
        clock = rig.eng.processor.captured
        be.outs.append(CABLE)
        be.fp = 1                                         # Windows' device list changed
        rig.eng._tick(time.monotonic() + 2.1)
        st = rig.eng.status()
        assert st.output_ok and st.output_name == CABLE.name
        assert CABLE_MISSING not in st.warnings
        assert be.out_streams[-1].dev is CABLE
        assert rig.eng.processor.captured >= clock        # same clock, not a new one
        rig.speak(20)
        assert rig.det.stats["gaps"] == 0
    finally:
        rig.eng.stop()


# ------------------------------------------------------------------ the whole app, cable missing

class FakeSd:
    """What `sounddevice` would list on a PC without VB-CABLE."""

    def query_hostapis(self, index=None):
        apis = [{"name": "Windows WASAPI", "devices": [0, 1], "default_input_device": 0,
                 "default_output_device": 1}]
        return apis if index is None else apis[index]

    def query_devices(self):
        return [
            {"index": 0, "name": MIC.name, "hostapi": 0, "max_input_channels": 1,
             "max_output_channels": 0, "default_samplerate": 48000.0},
            {"index": 1, "name": SPEAKERS.name, "hostapi": 0, "max_input_channels": 0,
             "max_output_channels": 2, "default_samplerate": 48000.0},
        ]


@pytest.fixture
def app_core(tmp_path, monkeypatch):
    """The real AppCore on fake devices and a scripted recogniser. Lyric mode off (no network)."""
    from npass import app as app_mod
    from npass import settings as cfg
    from npass.detect import detector as detector_mod

    (tmp_path / "settings.json").write_text(
        json.dumps({"lyrics_enabled": False, "input_hint": "Test Mic"}), encoding="utf-8")
    cfg.save_words(WORDS)
    be = FakeBackend([MIC], [SPEAKERS])
    rec = FakeRecogniser([])

    def fake_pick(settings):
        rec.load()
        return rec, rec.name

    monkeypatch.setattr(devices, "SoundBackend", lambda: be)
    monkeypatch.setattr(devices, "sd", FakeSd())
    monkeypatch.setattr(detector_mod, "pick_backend", fake_pick)
    core = app_mod.AppCore()
    core.start()
    try:
        yield core, be, rec
    finally:
        core.stop()


def _feed_app(core, be, blocks, value=0.3):
    for _ in range(blocks):
        be.in_streams[-1].callback(np.full(BLOCK, value, dtype=np.float32), 0.010, False)
        time.sleep(0.002)


def test_app_without_the_cable_starts_warns_and_opens_no_output(app_core):
    core, be, rec = app_core
    assert wait_for(lambda: bool(be.in_streams))
    _feed_app(core, be, 60)
    assert wait_for(lambda: core.snapshot()["health"] == Health.OK)
    snap = core.snapshot()
    assert snap["cable_found"] is False
    assert CABLE_MISSING in snap["engine"].warnings
    assert snap["engine"].input_ok and not snap["engine"].output_ok
    assert be.out_streams == []                           # the speakers were never opened
    assert snap["engine_name"] == rec.name
    assert snap["enabled"] is True and snap["lyrics_enabled"] is False


def test_app_catches_a_scripted_word_and_logs_it_once(app_core, tmp_path):
    core, be, rec = app_core
    assert wait_for(lambda: bool(be.in_streams))
    _feed_app(core, be, 60)
    assert wait_for(lambda: core.snapshot()["health"] == Health.OK)
    now = core.engine.processor.captured
    rec.script.append(ScriptedWord("pineapple", now + ms_to_samples(400), now + ms_to_samples(900)))
    _feed_app(core, be, 200)
    assert wait_for(lambda: core.snapshot()["caught"] == 1)
    catches = core.recent_catches()
    assert len(catches) == 1 and catches[0].order.word == "pineapple"
    assert catches[0].late_ms == 0.0
    log_lines = (tmp_path / "catches.log").read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1 and 'caught "pineapple"' in log_lines[0] and "no song" in log_lines[0]


def test_app_switched_off_makes_no_cuts_but_keeps_the_delay(app_core):
    core, be, rec = app_core
    assert wait_for(lambda: bool(be.in_streams))
    core.set_enabled(False)
    _feed_app(core, be, 60)
    now = core.engine.processor.captured
    rec.script.append(ScriptedWord("banana", now + ms_to_samples(400), now + ms_to_samples(900)))
    _feed_app(core, be, 200)
    time.sleep(0.2)
    snap = core.snapshot()
    assert snap["caught"] == 0 and core.recent_catches() == []
    assert not core.engine.processor._cuts                # nothing reached the cutter
    assert snap["engine"].lookahead_ms == pytest.approx(core.settings.lookahead_ms)
