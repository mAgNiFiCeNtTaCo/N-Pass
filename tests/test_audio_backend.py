"""SoundBackend against a fake `sounddevice` module: how streams are asked for, never a real device."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

# plain `uv run pytest` does not put the project root on the path (no conftest, package = false)
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from npass.audio import devices
from npass.audio.devices import DeviceInfo, SoundBackend


class _Stream:
    def __init__(self, owner, kind, **kw):
        if owner.refuse(kind, kw):
            raise RuntimeError("Invalid sample rate [PaErrorCode -9997]")
        self.kind = kind
        self.kw = kw
        self.started = False
        owner.opened.append(self)

    def start(self):
        self.started = True


class FakeSd:
    def __init__(self, refuse=lambda kind, kw: False):
        self.opened: list[_Stream] = []
        self.refuse = refuse
        self.terminated = 0
        self.initialized = 0

    def WasapiSettings(self, auto_convert=False):
        return SimpleNamespace(auto_convert=auto_convert)

    def InputStream(self, **kw):
        return _Stream(self, "in", **kw)

    def OutputStream(self, **kw):
        return _Stream(self, "out", **kw)

    def _terminate(self):
        self.terminated += 1

    def _initialize(self):
        self.initialized += 1


def _time(adc=0.0, now=0.0, dac=0.0):
    return SimpleNamespace(inputBufferAdcTime=adc, currentTime=now, outputBufferDacTime=dac)


def test_input_opens_at_48k_mono_float32_and_passes_blocks_on(monkeypatch):
    fake = FakeSd()
    monkeypatch.setattr(devices, "sd", fake)
    got = []
    dev = DeviceInfo(7, "Microphone (Test)", 2, 0, 48000.0)
    stream, converted = SoundBackend().open_input(dev, 480, lambda b, lat, glitch: got.append((b.copy(), lat, glitch)))
    assert converted is False and stream.started
    kw = stream.kw
    assert kw["samplerate"] == 48000 and kw["blocksize"] == 480 and kw["device"] == 7
    assert kw["channels"] == 1 and kw["dtype"] == "float32" and kw["extra_settings"] is None
    block = np.linspace(-1, 1, 480, dtype=np.float32).reshape(-1, 1)
    kw["callback"](block, 480, _time(adc=1.000, now=1.012), None)
    assert np.array_equal(got[0][0], block[:, 0])
    assert got[0][1] == pytest.approx(0.012) and got[0][2] is False


def test_other_sample_rate_uses_wasapi_auto_convert(monkeypatch):
    fake = FakeSd()
    monkeypatch.setattr(devices, "sd", fake)
    dev = DeviceInfo(7, "Microphone (Test)", 1, 0, 44100.0)
    stream, converted = SoundBackend().open_input(dev, 480, lambda *a: None)
    assert converted is True
    assert stream.kw["samplerate"] == 48000                 # still opened at 48000
    assert stream.kw["extra_settings"].auto_convert is True


def test_input_falls_back_to_two_channels_and_mixes_down(monkeypatch):
    fake = FakeSd(refuse=lambda kind, kw: kw["channels"] == 1)
    monkeypatch.setattr(devices, "sd", fake)
    got = []
    dev = DeviceInfo(7, "Microphone (Test)", 2, 0, 48000.0)
    stream, _ = SoundBackend().open_input(dev, 480, lambda b, lat, glitch: got.append(b.copy()))
    assert stream.kw["channels"] == 2
    stereo = np.stack([np.full(480, 0.2, np.float32), np.full(480, 0.6, np.float32)], axis=1)
    stream.kw["callback"](stereo, 480, _time(), None)
    assert got[0].shape == (480,) and np.allclose(got[0], 0.4)


def test_input_that_cannot_be_opened_raises_for_the_engine_to_report(monkeypatch):
    monkeypatch.setattr(devices, "sd", FakeSd(refuse=lambda kind, kw: True))
    with pytest.raises(RuntimeError):
        SoundBackend().open_input(DeviceInfo(7, "Microphone (Test)", 2, 0, 48000.0), 480, lambda *a: None)


def test_output_writes_the_same_signal_to_every_channel(monkeypatch):
    fake = FakeSd()
    monkeypatch.setattr(devices, "sd", fake)
    dev = DeviceInfo(9, "CABLE Input (VB-Audio Virtual Cable)", 0, 8, 48000.0)
    mono = np.linspace(-1, 1, 480, dtype=np.float32)
    asked = []

    def cb(frames, lat):
        asked.append((frames, lat))
        return mono

    stream, converted = SoundBackend().open_output(dev, 480, cb)
    assert converted is False and stream.started
    assert stream.kw["samplerate"] == 48000 and stream.kw["channels"] == 8 and stream.kw["device"] == 9
    out = np.zeros((480, 8), dtype=np.float32)
    stream.kw["callback"](out, 480, _time(now=2.000, dac=2.015), None)
    for ch in range(8):
        assert np.array_equal(out[:, ch], mono)
    assert asked[0][0] == 480 and asked[0][1] == pytest.approx(0.015)


def test_rescan_restarts_the_device_list(monkeypatch):
    fake = FakeSd()
    monkeypatch.setattr(devices, "sd", fake)
    SoundBackend().rescan()
    assert fake.terminated == 1 and fake.initialized == 1


def test_fingerprint_is_stable_between_two_reads():
    a = devices.endpoint_fingerprint()                      # reads the registry only, no device
    assert a == devices.endpoint_fingerprint()
