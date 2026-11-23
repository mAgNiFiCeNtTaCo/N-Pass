"""The scripted backend, backend picking, and the real models.

The real-model tests need files in `models/` (fetch once with `uv run python -m npass.detect
download`); they are skipped when the files are missing. Only the Whisper one needs the GPU and is
marked `gpu`. No test here touches the network or an audio device.
"""
from __future__ import annotations

import numpy as np
import pytest

from npass.detect import FakeRecogniser, ScriptedWord
from npass.detect import detector as detector_mod
from npass.settings import Settings


def sec(s: float) -> int:
    return int(round(s * 48_000))


def test_fake_returns_words_inside_the_audio():
    rec = FakeRecogniser([ScriptedWord("hello", sec(0.2), sec(0.5)), ("banana", sec(1.0), sec(2.0))])
    rec.load()
    audio = np.zeros(16_000, dtype=np.float32)              # 0.0 - 1.0 s
    words = rec.transcribe(audio, 0, ["x"])
    assert [(w.text, w.partial) for w in words] == [("hello", False)]
    assert rec.passes == [(0, sec(1.0))] and rec.hints_seen == [["x"]]

    words = rec.transcribe(np.zeros(24_000, dtype=np.float32), 0, [])   # 0.0 - 1.5 s: half the word
    assert [(w.text, w.partial) for w in words] == [("hello", False), ("ban", True)]
    assert words[1].end == sec(1.5)

    words = rec.transcribe(np.zeros(16_000, dtype=np.float32), sec(1.2), [])   # 1.2 - 2.2 s
    assert [(w.text, w.partial) for w in words] == [("banana", False)]


def test_fake_crash_and_load_switches():
    rec = FakeRecogniser([], fail_loads=1)
    with pytest.raises(RuntimeError):
        rec.load()
    with pytest.raises(RuntimeError):
        rec.transcribe(np.zeros(160, dtype=np.float32), 0, [])       # not loaded
    rec.load()
    rec.crash_once = True
    with pytest.raises(RuntimeError):
        rec.transcribe(np.zeros(160, dtype=np.float32), 0, [])
    assert rec.transcribe(np.zeros(160, dtype=np.float32), 0, []) == []
    rec.crash_now = True
    for _ in range(2):
        with pytest.raises(RuntimeError):
            rec.transcribe(np.zeros(160, dtype=np.float32), 0, [])
    rec.close()
    assert not rec.loaded and rec.close_calls == 1


class _Stub:
    def __init__(self, name, fail=False, mem=None):
        self.name, self._fail, self.gpu_mem_mb, self.closed = name, fail, mem, False

    def load(self):
        if self._fail:
            raise RuntimeError("no model")

    def close(self):
        self.closed = True


def _patch(monkeypatch, gpu, cpu):
    import npass.detect.vosk_backend as vb
    import npass.detect.whisper_backend as wb
    monkeypatch.setattr(wb, "WhisperRecogniser", lambda **kw: gpu)
    monkeypatch.setattr(vb, "VoskRecogniser", lambda **kw: cpu)


def test_pick_backend_auto_prefers_the_gpu(monkeypatch):
    gpu, cpu = _Stub("GPU (faster-whisper small.en)", mem=600.0), _Stub("CPU (Vosk)")
    _patch(monkeypatch, gpu, cpu)
    rec, name = detector_mod.pick_backend(Settings())
    assert rec is gpu and name == "GPU (faster-whisper small.en)"


def test_pick_backend_auto_falls_back_and_says_so(monkeypatch):
    gpu, cpu = _Stub("GPU", fail=True), _Stub("CPU (Vosk)")
    _patch(monkeypatch, gpu, cpu)
    rec, name = detector_mod.pick_backend(Settings())
    assert rec is cpu and name.startswith("CPU (Vosk)") and "GPU model did not load" in name


def test_pick_backend_auto_respects_the_memory_limit(monkeypatch):
    gpu, cpu = _Stub("GPU", mem=2500.0), _Stub("CPU (Vosk)")
    _patch(monkeypatch, gpu, cpu)
    s = Settings()
    s.gpu_mem_limit_mb = 1500
    rec, name = detector_mod.pick_backend(s)
    assert rec is cpu and gpu.closed


def test_pick_backend_forced_modes(monkeypatch):
    gpu, cpu = _Stub("GPU", fail=True), _Stub("CPU (Vosk)")
    _patch(monkeypatch, gpu, cpu)
    s = Settings()
    s.engine = "gpu"
    with pytest.raises(RuntimeError):
        detector_mod.pick_backend(s)                 # no silent fall-back when the GPU was asked for
    s.engine = "cpu"
    rec, name = detector_mod.pick_backend(s)
    assert rec is cpu and name == "CPU (Vosk)"


def test_pick_backend_both_failing_raises(monkeypatch):
    _patch(monkeypatch, _Stub("GPU", fail=True), _Stub("CPU (Vosk)", fail=True))
    with pytest.raises(RuntimeError):
        detector_mod.pick_backend(Settings())


def test_vosk_real_model_on_silence():
    from npass.detect import vosk_backend
    if not vosk_backend.model_is_local():
        pytest.skip("the Vosk model is not in models/")
    rec = vosk_backend.VoskRecogniser()
    rec.load()
    try:
        silence = np.zeros(32_000, dtype=np.float32)
        assert rec.transcribe(silence[:16_000], 0, []) == []
        assert rec.transcribe(silence, 0, ["banana"]) == []          # second call feeds only the new second
        assert rec.transcribe(silence, sec(10.0), []) == []          # a jump resets the stream
    finally:
        rec.close()


@pytest.mark.gpu
def test_whisper_real_model_on_two_seconds_of_silence():
    from npass.detect import whisper_backend
    s = Settings()
    if not whisper_backend.model_is_local(s.whisper_model):
        pytest.skip(f"the {s.whisper_model} model is not in models/")
    rec = whisper_backend.WhisperRecogniser(model=s.whisper_model, compute=s.whisper_compute)
    rec.load()
    try:
        assert rec.name == f"GPU (faster-whisper {s.whisper_model})"
        words = rec.transcribe(np.zeros(32_000, dtype=np.float32), sec(5.0), ["pineapple", "banana"])
        assert isinstance(words, list)
        for w in words:                                              # whatever it makes up stays in range
            assert sec(5.0) <= w.start <= w.end <= sec(7.0)
        assert rec.gpu_mem_mb is None or rec.gpu_mem_mb < s.gpu_mem_limit_mb
    finally:
        rec.close()
