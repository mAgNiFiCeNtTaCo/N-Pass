"""Finding and opening sound devices. WASAPI only (the Windows host API OBS and VB-CABLE use).

`SoundBackend` is the one place that touches `sounddevice`; the engine talks to it through a few
small methods so the tests can swap in a fake.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from npass.types import SAMPLE_RATE

try:                                    # PortAudio missing must not break imports (tests, harness)
    import sounddevice as sd
except Exception:                       # pragma: no cover - depends on the PC
    sd = None

_pa_lock = threading.RLock()            # the device list must not be read while it is being rebuilt

CABLE_HINT = "CABLE Input"             # VB-CABLE's playback end; OBS listens on "CABLE Output"
CABLE_RECORD_END = "CABLE Output"       # never use this as the microphone: it would loop

InputCallback = Callable[[np.ndarray, "float | None", bool], None]
"""(mono float32 block, input latency in seconds or None, the driver reported a glitch)"""
OutputCallback = Callable[[int, "float | None"], np.ndarray]
"""(frames wanted, output latency in seconds or None) -> mono float32 block of that length"""


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    max_in: int
    max_out: int
    default_rate: float


def _wasapi_index() -> int | None:
    if sd is None:
        return None
    try:
        for i, api in enumerate(sd.query_hostapis()):
            if "WASAPI" in str(api.get("name", "")):
                return i
    except Exception:
        return None
    return None


def _wasapi_devices() -> list[DeviceInfo]:
    with _pa_lock:
        api = _wasapi_index()
        if api is None:
            return []
        try:
            devices = list(sd.query_devices())
        except Exception:
            return []
    out: list[DeviceInfo] = []
    for i, d in enumerate(devices):
        if d.get("hostapi") != api:
            continue
        out.append(DeviceInfo(
            index=int(d.get("index", i)),
            name=str(d.get("name", "")),
            max_in=int(d.get("max_input_channels", 0)),
            max_out=int(d.get("max_output_channels", 0)),
            default_rate=float(d.get("default_samplerate", 0.0) or 0.0),
        ))
    return out


def input_devices() -> list[DeviceInfo]:
    return [d for d in _wasapi_devices()
            if d.max_in > 0 and not d.name.rstrip().endswith("[Loopback]")]


def output_devices() -> list[DeviceInfo]:
    return [d for d in _wasapi_devices() if d.max_out > 0]


def default_input_device() -> DeviceInfo | None:
    api = _wasapi_index()
    if api is None:
        return None
    try:
        with _pa_lock:
            idx = int(sd.query_hostapis(api).get("default_input_device", -1))
    except Exception:
        return None
    for d in input_devices():
        if d.index == idx:
            return d
    return None


def match_device(devices: list[DeviceInfo], text: str) -> DeviceInfo | None:
    """Pick by name: an exact name first, then the first name that contains `text` (any case)."""
    want = (text or "").strip().lower()
    if not want:
        return None
    for d in devices:
        if d.name.lower() == want:
            return d
    for d in devices:
        if want in d.name.lower():
            return d
    return None


def list_inputs() -> list[str]:
    return [d.name for d in input_devices()]


def list_outputs() -> list[str]:
    return [d.name for d in output_devices()]


def find_cable_device(hint: str = CABLE_HINT) -> DeviceInfo | None:
    return match_device(output_devices(), hint or CABLE_HINT)


def find_cable() -> str | None:
    """Name of the virtual cable's playback end, None when VB-CABLE is not installed."""
    d = find_cable_device()
    return d.name if d else None


def endpoint_fingerprint() -> object | None:
    """A cheap "did Windows' list of sound devices change?" value, read from the registry.

    PortAudio only notices new devices after a full restart of its device list, which needs every
    stream closed. The engine compares this value first, so it only does that when something changed.
    None = could not tell.
    """
    try:
        import winreg
    except ImportError:
        return None
    found = []
    base = r"SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio"
    for kind in ("Render", "Capture"):
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base + "\\" + kind) as key:
                i = 0
                while True:
                    try:
                        name = winreg.EnumKey(key, i)
                    except OSError:
                        break
                    i += 1
                    state = None
                    try:
                        with winreg.OpenKey(key, name) as sub:
                            state = winreg.QueryValueEx(sub, "DeviceState")[0]
                    except OSError:
                        pass
                    found.append((kind, name, state))
        except OSError:
            return None
    return tuple(found)


class SoundBackend:
    """The real devices (sounddevice / PortAudio, WASAPI shared mode)."""

    def rescan(self) -> None:
        """Make PortAudio re-read the device list. Every stream must be closed first."""
        if sd is None:
            return
        with _pa_lock:
            try:
                sd._terminate()
                sd._initialize()
            except Exception:
                pass

    def fingerprint(self) -> object | None:
        return endpoint_fingerprint()

    def inputs(self) -> list[DeviceInfo]:
        return input_devices()

    def outputs(self) -> list[DeviceInfo]:
        return output_devices()

    def default_input(self) -> DeviceInfo | None:
        return default_input_device()

    @staticmethod
    def _attempts(dev_rate: float, channel_options: list[int]):
        """(auto_convert, channels) in the order to try them."""
        same_rate = abs(dev_rate - SAMPLE_RATE) < 1.0
        conv = [False, True] if same_rate else [True]
        return [(ac, ch) for ac in conv for ch in channel_options]

    def open_input(self, dev: DeviceInfo, blocksize: int, callback: InputCallback):
        """Open and start the mic at 48 kHz. Returns (stream, auto_convert_used). Raises on failure."""
        if sd is None:
            raise RuntimeError("the sound library is not available")
        options = [1] + ([2] if dev.max_in >= 2 else [])
        last: Exception | None = None
        for auto_convert, channels in self._attempts(dev.default_rate, options):
            def cb(indata, frames, t, status, _ch=channels):
                mono = indata[:, 0] if _ch == 1 else indata.mean(axis=1)
                lat = None
                try:
                    v = float(t.currentTime - t.inputBufferAdcTime)
                    if 0.0 < v < 1.0:
                        lat = v
                except Exception:
                    pass
                callback(mono, lat, bool(status))
            try:
                with _pa_lock:
                    stream = sd.InputStream(
                        samplerate=SAMPLE_RATE, blocksize=blocksize, device=dev.index,
                        channels=channels, dtype="float32", latency="low", callback=cb,
                        extra_settings=sd.WasapiSettings(auto_convert=True) if auto_convert else None)
                    stream.start()
                return stream, auto_convert
            except Exception as e:
                last = e
        raise last if last else RuntimeError("could not open the microphone")

    def open_output(self, dev: DeviceInfo, blocksize: int, callback: OutputCallback):
        """Open and start the cable at 48 kHz; the mono signal goes to every channel."""
        if sd is None:
            raise RuntimeError("the sound library is not available")
        options: list[int] = []
        for ch in (dev.max_out, 2, 1):
            if 0 < ch <= dev.max_out and ch not in options:
                options.append(ch)
        last: Exception | None = None
        for auto_convert, channels in self._attempts(dev.default_rate, options):
            def cb(outdata, frames, t, status):
                lat = None
                try:
                    v = float(t.outputBufferDacTime - t.currentTime)
                    if 0.0 < v < 1.0:
                        lat = v
                except Exception:
                    pass
                outdata[:] = callback(frames, lat).reshape(-1, 1)
            try:
                with _pa_lock:
                    stream = sd.OutputStream(
                        samplerate=SAMPLE_RATE, blocksize=blocksize, device=dev.index,
                        channels=channels, dtype="float32", latency="low", callback=cb,
                        extra_settings=sd.WasapiSettings(auto_convert=True) if auto_convert else None)
                    stream.start()
                return stream, auto_convert
            except Exception as e:
                last = e
        raise last if last else RuntimeError("could not open the virtual audio cable")
