"""The stand-in clips in test_clips/standin/, and an end-to-end run over one of them.

The end-to-end test feeds a real recorded clip through the live classes (LookaheadProcessor,
Detector, SafetyGuard) with the FakeRecogniser scripted from the clip's own sidecar: the result
must be the input, untouched, everywhere except one clean cut over the word.
One more test (marked gpu + slow) runs a clip through the real speech model at real-time pace.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from npass.audio.processor import LookaheadProcessor
from npass.detect import FakeRecogniser, ScriptedWord
from npass.settings import Settings
from npass.types import SAMPLE_RATE, ms_to_samples
from tools import run_clips as rc

ROOT = Path(__file__).resolve().parents[1]
STANDIN = ROOT / "test_clips" / "standin"
WORDS = ["pineapple", "banana"]
CLIPS = sorted(STANDIN.glob("*.flac"))

pytestmark = pytest.mark.skipif(not CLIPS, reason="the stand-in clips are not in test_clips/standin")


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NPASS_DATA_DIR", str(tmp_path))


# ------------------------------------------------------------------ the set itself

def test_every_clip_has_a_sidecar_that_fits_it():
    with_word = 0
    for clip in CLIPS:
        side = clip.with_suffix(".txt")
        assert side.is_file(), f"{clip.name} has no .txt next to it"
        info = sf.info(str(clip))
        assert info.channels == 1 and info.samplerate == 16000
        truth = rc.read_truth(side)
        for exp in truth:
            assert exp.word in WORDS, f"{side.name}: only the stand-in words belong here"
            assert 0 <= exp.start < exp.end <= info.duration * SAMPLE_RATE + 1
            assert 0.15 <= (exp.end - exp.start) / SAMPLE_RATE <= 3.0
        if clip.name.startswith("clean_"):
            assert truth == []
        else:
            assert len(truth) == 1
            with_word += 1
    assert with_word >= 10 and len(CLIPS) - with_word >= 5
    assert any(c.name.startswith("stretched_") for c in CLIPS)


def test_the_set_is_small_and_every_clip_names_its_source_and_licence():
    total = sum(p.stat().st_size for p in STANDIN.iterdir() if p.is_file())
    assert total < 15 * 1024 * 1024
    sources = (STANDIN / "SOURCES.md").read_text(encoding="utf-8")
    assert "CC BY 4.0" in sources and "LibriSpeech" in sources and "openslr.org/12" in sources
    for clip in CLIPS:
        assert f"`{clip.name}`" in sources, f"{clip.name} is not listed in SOURCES.md"


def test_the_marked_word_is_voice_not_silence():
    for clip in CLIPS:
        truth = rc.read_truth(clip.with_suffix(".txt"))
        if not truth:
            continue
        audio = rc.load_audio(clip)
        span = audio[truth[0].start:truth[0].end]
        assert float(np.sqrt(np.mean(span ** 2))) > 0.01, clip.name


def test_stretched_clips_only_differ_from_the_spoken_one_around_the_word():
    spoken = rc.load_audio(STANDIN / "spoken_banana_01.flac")
    longer = rc.load_audio(STANDIN / "stretched_banana_01_x2.flac")
    word = rc.read_truth(STANDIN / "spoken_banana_01.txt")[0]
    head = word.start - ms_to_samples(120)
    assert np.allclose(spoken[:head], longer[:head], atol=2e-3)       # the same recording before it
    grew = len(longer) - len(spoken)
    assert grew == pytest.approx((word.end - word.start) + ms_to_samples(60), abs=ms_to_samples(15))
    tail = len(spoken) - word.end - ms_to_samples(120)
    assert np.allclose(spoken[-tail:], longer[-tail:], atol=2e-3)     # and after it


# ------------------------------------------------------------------ end to end, scripted

@pytest.mark.parametrize("name", ["spoken_pineapple_01", "spoken_banana_04", "stretched_banana_04_x1_5"])
def test_scripted_end_to_end_is_the_input_delayed_with_one_clean_cut(name):
    audio = rc.load_audio(STANDIN / f"{name}.flac")
    truth = rc.read_truth(STANDIN / f"{name}.txt")
    word = truth[0]
    s = Settings()
    rec = FakeRecogniser([ScriptedWord(word.word, word.start, word.end)])
    res, filtered = rc.run_clip(audio, truth, s, WORDS, recogniser=rec, realtime=False, name=name)

    assert (res.caught, res.part_played, res.missed) == (1, 0, 0)
    assert res.wrong_cuts == [] and res.late_cuts == 0
    assert res.words[0].removed_share == pytest.approx(1.0, abs=1e-6)

    # the cut covers the word, and it is one cut
    assert np.all(filtered[word.start:word.end] == 0.0)
    changed = np.flatnonzero(filtered != audio)
    lo, hi = int(changed[0]), int(changed[-1]) + 1
    fade = ms_to_samples(s.fade_ms)
    assert lo <= word.start - fade and hi >= word.end + fade
    # ... and reaches no further than the edge search, the padding, the hold and the fades allow
    assert lo >= word.start - ms_to_samples(120 + s.cut_pad_ms + s.fade_ms + 15)
    assert hi <= word.end + ms_to_samples(120 + s.cut_pad_ms + 2 * s.hop_ms + 600 + s.fade_ms)

    # everywhere else the output is exactly the input (run_clip already took the look-ahead out)
    assert np.array_equal(filtered[:lo], audio[:lo])
    assert np.array_equal(filtered[hi:], audio[hi:])

    # no click: between two neighbouring samples the gain never moves more than one fade step
    proc = LookaheadProcessor(s.lookahead_ms, s.fade_ms)
    loud = np.flatnonzero(np.abs(audio) > 1e-3)
    loud = loud[(loud >= lo - 10) & (loud < hi + 10)]
    gain = filtered[loud] / audio[loud]
    steps = np.abs(np.diff(gain)) / np.diff(loud)          # per sample, also across quiet samples
    assert float(steps.max()) <= proc.fade_step + 1e-3
    assert gain.min() == 0.0 and gain.max() > 0.99


def test_scripted_end_to_end_really_is_delayed_by_the_look_ahead():
    """Same chain without run_clip's re-alignment: the raw output lags by exactly the look-ahead."""
    audio = rc.load_audio(STANDIN / "clean_apple_03.flac")
    s = Settings()
    proc = LookaheadProcessor(s.lookahead_ms, s.fade_ms)
    look = ms_to_samples(s.lookahead_ms)
    padded = np.concatenate([audio, np.zeros(look, dtype=np.float32)])
    out = np.concatenate([proc.process(padded[i:i + 480]) for i in range(0, len(padded), 480)])
    assert np.all(out[:look] == 0.0)
    assert np.array_equal(out[look:look + len(audio)], audio)


def test_clean_clip_with_a_silent_recogniser_comes_out_untouched():
    audio = rc.load_audio(STANDIN / "clean_plain_01.flac")
    res, filtered = rc.run_clip(audio, [], Settings(), WORDS, recogniser=FakeRecogniser([]),
                                realtime=False)
    assert np.array_equal(filtered, audio)
    assert res.wrong_cuts == [] and res.silenced_ms == 0.0 and res.health_end == "ok"


# ------------------------------------------------------------------ the real model, real time

@pytest.mark.gpu
@pytest.mark.slow
def test_real_gpu_model_catches_a_spoken_stand_in_word_at_real_time_pace():
    from npass.detect.whisper_backend import WhisperRecogniser, model_is_local
    s = Settings()
    if not model_is_local(s.whisper_model):
        pytest.skip(f"the {s.whisper_model} model is not in models/")
    inner = WhisperRecogniser(model=s.whisper_model, compute=s.whisper_compute)
    try:
        inner.load()
    except Exception as exc:
        pytest.skip(f"the GPU model could not be loaded here ({type(exc).__name__})")
    rec = rc.KeepLoaded(inner)
    rec._loaded = True
    try:
        audio = rc.load_audio(STANDIN / "spoken_banana_08.flac")
        truth = rc.read_truth(STANDIN / "spoken_banana_08.txt")
        res, _ = rc.run_clip(audio, truth, s, WORDS, recogniser=rec, realtime=True)
        clean = rc.load_audio(STANDIN / "clean_apple_03.flac")
        res_clean, out_clean = rc.run_clip(clean, [], s, WORDS, recogniser=rec, realtime=True)
    finally:
        rec.really_close()
    assert res.missed == 0 and res.caught + res.part_played == 1
    assert res.words[0].removed_share > 0.5 and res.wrong_cuts == []
    assert res_clean.wrong_cuts == [] and np.array_equal(out_clean, clean)
