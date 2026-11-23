"""Blocked-word detection: the Detector, the SafetyGuard and the speech backends.

The heavy backends (faster-whisper, Vosk) are imported only when they are loaded, so importing
this package is cheap and needs no GPU.
"""
from .detector import Detector, pick_backend
from .fake_backend import FakeRecogniser, ScriptedWord
from .guard import SafetyGuard

__all__ = ["Detector", "SafetyGuard", "FakeRecogniser", "ScriptedWord", "pick_backend",
           "WhisperRecogniser", "VoskRecogniser"]


def __getattr__(name: str):
    if name == "WhisperRecogniser":
        from .whisper_backend import WhisperRecogniser
        return WhisperRecogniser
    if name == "VoskRecogniser":
        from .vosk_backend import VoskRecogniser
        return VoskRecogniser
    raise AttributeError(name)
