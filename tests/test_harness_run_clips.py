"""tools/run_clips.py: reading clips and sidecars, scoring, the report, and the command line.

A scripted FakeRecogniser stands in for the speech model and steady noise stands in for a voice,
so these tests need no model, no GPU, no sound device and no network. Stand-in words only.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from npass.detect import FakeRecogniser, ScriptedWord
from npass.settings import Settings
from npass.types import SAMPLE_RATE, ms_to_samples
from tools import run_clips as rc

WORDS = ["pineapple", "banana"]


def sec(s: float) -> int:
    return int(round(s * SAMPLE_RATE))


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NPASS_DATA_DIR", str(tmp_path / "data"))


def voice(seconds: float, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(sec(seconds)) * 0.1).astype(np.float32)


def truth(*spans) -> list[rc.Expected]:
    return [rc.Expected(sec(a), sec(b), w) for a, b, w in spans]


def run(audio, expected, script, **kw):
    settings = kw.pop("settings", Settings())
    rec = kw.pop("rec", None) or FakeRecogniser(list(script))
    return rc.run_clip(audio, expected, settings, WORDS, recogniser=rec, realtime=False, **kw)


# ------------------------------------------------------------------ reading

def test_read_truth_parses_lines_comments_and_commas(tmp_path):
    p = tmp_path / "clip.txt"
    p.write_text("# my notes\n3.2 3.8 Pineapple\n\n9.1, 10.4, banana   # the long one\n1.0 1.5\n",
                 encoding="utf-8")
    got = rc.read_truth(p)
    assert [(e.start, e.end, e.word) for e in got] == [
        (sec(1.0), sec(1.5), ""), (sec(3.2), sec(3.8), "pineapple"), (sec(9.1), sec(10.4), "banana")]


def test_read_truth_missing_or_empty_means_no_blocked_word(tmp_path):
    assert rc.read_truth(tmp_path / "nothing.txt") == []
    (tmp_path / "empty.txt").write_text("\n", encoding="utf-8")
    assert rc.read_truth(tmp_path / "empty.txt") == []


@pytest.mark.parametrize("line", ["three four word", "2.0", "5.0 4.0 banana"])
def test_read_truth_says_which_line_is_wrong(tmp_path, line):
    p = tmp_path / "clip.txt"
    p.write_text("1.0 2.0 banana\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clip.txt, line 2"):
        rc.read_truth(p)


def test_load_audio_makes_mono_48k_from_a_stereo_44k_file(tmp_path):
    t = np.arange(44100) / 44100.0
    stereo = np.stack([0.5 * np.sin(2 * np.pi * 440 * t), np.zeros_like(t)], axis=1)
    sf.write(str(tmp_path / "a.wav"), stereo, 44100, subtype="PCM_16")
    got = rc.load_audio(tmp_path / "a.wav")
    assert got.dtype == np.float32 and got.ndim == 1
    assert abs(len(got) - SAMPLE_RATE) <= 2
    assert float(np.max(np.abs(got[4800:-4800]))) == pytest.approx(0.25, abs=0.01)   # mean of L and R


def test_load_audio_reports_a_file_that_is_not_sound(tmp_path):
    bad = tmp_path / "not sound.mp3"
    bad.write_bytes(b"this is not an mp3 file at all" * 10)
    with pytest.raises(RuntimeError, match="could not read not sound.mp3"):
        rc.load_audio(bad)


def test_find_clips_skips_text_files_and_its_own_output(tmp_path):
    for name in ("b.wav", "a.MP3", "a.txt", "a.filtered.wav", "notes.md", "c.flac", "d.m4a"):
        (tmp_path / name).write_bytes(b"")
    assert [p.name for p in rc.find_clips(tmp_path)] == ["a.MP3", "b.wav", "c.flac", "d.m4a"]


def test_read_windows_builds_danger_windows_from_an_lrc_file(tmp_path):
    lrc = tmp_path / "song.lrc"
    assert rc.read_windows(lrc, WORDS, Settings()) is None
    lrc.write_text("[00:01.00]nothing here\n[00:03.00]one pineapple please\n[00:05.00]and on\n",
                   encoding="utf-8")
    s = Settings()
    wins = rc.read_windows(lrc, WORDS, s)
    assert len(wins) == 1
    assert wins[0].start == sec(3.0) - ms_to_samples(s.window_pre_ms)
    assert wins[0].end == sec(5.0) + ms_to_samples(s.window_post_ms)
    assert wins[0].words == ("pineapple",)


# ------------------------------------------------------------------ measuring

def test_measured_gain_reads_the_gain_back_and_carries_it_over_silence():
    audio = np.full(1000, 0.5, dtype=np.float32)
    audio[400:500] = 0.0                                  # digital silence inside a cut
    gain = np.ones(1000, dtype=np.float32)
    gain[300:600] = 0.0
    gain[600:700] = 0.25
    got = rc.measured_gain(audio, audio * gain)
    assert np.allclose(got, gain, atol=1e-6)
    assert rc.runs_of(got <= 0.001) == [(300, 600)]


def test_runs_of_edges():
    assert rc.runs_of(np.array([], dtype=bool)) == []
    assert rc.runs_of(np.array([True, True, False, True])) == [(0, 2), (3, 4)]


# ------------------------------------------------------------------ scoring one clip

def test_a_scripted_word_is_caught_and_nothing_else_is_touched():
    audio = voice(4.0)
    word = ScriptedWord("pineapple", sec(2.0), sec(2.5))
    s = Settings()
    s.max_hold_ms = 0.0     # the test "voice" is steady noise with no pause to stop a hold at
    res, filtered = run(audio, truth((2.0, 2.5, "pineapple")), [word], settings=s)
    assert (res.expected, res.caught, res.part_played, res.missed) == (1, 1, 0, 0)
    assert res.wrong_cuts == [] and res.late_cuts == 0 and res.safety_mutes == 0
    w = res.words[0]
    assert w.status == "caught" and w.removed_share == pytest.approx(1.0, abs=1e-4)
    assert w.played_ms == 0.0
    assert w.start_error_ms <= 0 and w.end_error_ms >= 0   # silence starts before, ends after
    assert w.reason == "asr" and w.heard.startswith("pine")
    assert 0 < w.heard_after_ms < Settings().lookahead_ms   # ordered before the word reached the output
    assert len(filtered) == len(audio)
    assert np.all(filtered[sec(2.0):sec(2.5)] == 0.0)
    assert np.array_equal(filtered[:sec(1.8)], audio[:sec(1.8)])
    assert np.array_equal(filtered[sec(3.2):], audio[sec(3.2):])
    assert res.silenced_ms >= 500.0 and res.unverified_ms == 0.0
    assert res.health_end == "ok" and res.passes > 0


def test_a_word_nobody_heard_is_reported_missed_with_how_long_it_played():
    audio = voice(4.0)
    res, filtered = run(audio, truth((2.0, 2.5, "banana")), [])
    assert (res.caught, res.part_played, res.missed) == (0, 0, 1)
    w = res.words[0]
    assert w.status == "MISSED" and w.removed_share == 0.0
    assert w.played_ms == pytest.approx(500.0, abs=0.1)
    assert w.start_error_ms is None and w.heard_after_ms is None
    assert np.array_equal(filtered, audio)
    assert res.silenced_ms == 0.0


def test_a_cut_over_innocent_audio_is_a_wrong_cut_with_what_was_heard():
    audio = voice(4.0)
    res, _ = run(audio, [], [ScriptedWord("banana", sec(1.0), sec(1.4))])
    assert res.expected == 0 and res.missed == 0
    assert len(res.wrong_cuts) == 1
    c = res.wrong_cuts[0]
    assert c.reason == "asr" and c.word == "banana" and c.heard == "banana"
    assert c.start_s <= 1.0 and c.end_s >= 1.4 and c.ms >= 400.0


def test_an_answer_that_comes_too_late_is_a_late_cut_and_a_part_played_word():
    audio = voice(5.0)
    s = Settings()
    arrives = sec(3.0) + ms_to_samples(50)                 # the word is over before it is named
    rec = FakeRecogniser([ScriptedWord("pineapple", sec(2.0), sec(3.0), 0.9, arrives)],
                         reveal_partials=False)
    res, filtered = run(audio, truth((2.0, 3.0, "pineapple")), [], rec=rec, settings=s)
    w = res.words[0]
    assert w.status == "part played" and res.part_played == 1 and res.caught == 0
    assert res.late_cuts == 1
    played = (3.05 - s.lookahead_ms / 1000.0) - 2.0        # how much had gone out when the order came
    assert w.played_ms == pytest.approx(played * 1000.0, abs=120.0)
    assert w.late_ms == pytest.approx(res.worst_late_ms) and w.late_ms > 400.0
    assert 1050.0 <= w.heard_after_ms <= 1160.0            # the first check after it became visible
    assert np.array_equal(filtered[:sec(2.0)], audio[:sec(2.0)])
    assert np.all(filtered[sec(2.7):sec(3.0)] == 0.0)      # the rest of the word was still cut


def test_dip_mode_counts_a_dipped_word_as_removed():
    audio = voice(4.0)
    s = Settings()
    s.removal_mode = "dip"
    res, filtered = run(audio, truth((2.0, 2.5, "banana")),
                        [ScriptedWord("banana", sec(2.0), sec(2.5))], settings=s)
    assert res.caught == 1 and res.wrong_cuts == []
    floor = 10.0 ** (s.dip_db / 20.0)
    assert np.allclose(filtered[sec(2.1):sec(2.4)], audio[sec(2.1):sec(2.4)] * floor, atol=1e-6)


def test_lyric_window_with_a_dead_recogniser_is_muted_by_the_guard():
    audio = voice(5.0)
    s = Settings()
    rec = FakeRecogniser([])
    rec.crash_now = True                                   # every pass fails
    from npass.types import DangerWindow
    win = DangerWindow(sec(2.0), sec(3.5), "a pineapple line", ("pineapple",))
    res, filtered = run(audio, truth((2.5, 3.0, "pineapple")), [], rec=rec, settings=s,
                        windows=[win])
    assert res.caught == 1 and res.words[0].reason == "failsafe"
    assert res.wrong_cuts == []                            # a safety mute is not a wrong cut
    assert np.all(filtered[sec(2.0):sec(3.5)] == 0.0)
    assert np.array_equal(filtered[:sec(1.9)], audio[:sec(1.9)])
    assert res.health_end == "down" and res.crashes > 0 and res.lyric_windows == 1

    res2, _ = run(audio, [], [], rec=rec, settings=s, windows=[win])
    assert res2.safety_mutes == 1 and res2.safety_mute_ms == pytest.approx(1500.0, abs=1.0)
    assert res2.wrong_cuts == []


def test_real_time_pace_with_the_detector_thread():
    audio = voice(1.6)
    word = ScriptedWord("banana", sec(0.7), sec(1.1))
    res, filtered = rc.run_clip(audio, truth((0.7, 1.1, "banana")), Settings(), WORDS,
                                recogniser=FakeRecogniser([word]), realtime=True)
    assert res.caught == 1 and res.missed == 0
    assert np.all(filtered[sec(0.7):sec(1.1)] == 0.0)
    assert 0 < res.worst_lag_ms < Settings().lookahead_ms


# ------------------------------------------------------------------ the recogniser wrapper

def test_keep_loaded_loads_once_and_ignores_the_detectors_close():
    inner = FakeRecogniser([])
    inner.latency_ms = 750
    keep = rc.KeepLoaded(inner)
    assert keep.name == inner.name and keep.latency_ms == 750
    keep.load()
    keep.load()
    keep.close()
    assert inner.load_calls == 1 and inner.close_calls == 0 and inner.loaded
    assert keep.transcribe(np.zeros(1600, dtype=np.float32), 0, []) == []
    keep.really_close()
    assert inner.close_calls == 1
    assert not hasattr(rc.KeepLoaded(FakeRecogniser([])), "latency_ms")


# ------------------------------------------------------------------ the report

def make_folder(tmp_path):
    folder = tmp_path / "clips"
    folder.mkdir()
    sf.write(str(folder / "one.wav"), voice(3.0, 1), SAMPLE_RATE, subtype="FLOAT")
    (folder / "one.txt").write_text("1.5 2.0 pineapple\n", encoding="utf-8")
    sf.write(str(folder / "two.wav"), voice(3.0, 2), SAMPLE_RATE, subtype="FLOAT")
    (folder / "two.txt").write_text("1.0 1.4 banana\n", encoding="utf-8")
    sf.write(str(folder / "three.wav"), voice(2.0, 3), SAMPLE_RATE, subtype="FLOAT")
    (folder / "broken.wav").write_bytes(b"nope")
    return folder


def test_run_folder_reports_every_clip_and_hides_the_words(tmp_path):
    folder = make_folder(tmp_path)
    out = tmp_path / "out"
    rec = FakeRecogniser([ScriptedWord("pineapple", sec(1.5), sec(2.0))])   # hears only clip one's word
    said = []
    report = rc.run_folder(folder, Settings(), WORDS, out, recogniser=rec, realtime=False,
                           log=said.append)
    by_name = {r.name: r for r in report["results"]}
    assert set(by_name) == {"one.wav", "two.wav", "three.wav", "broken.wav"}
    assert by_name["one.wav"].caught == 1
    assert by_name["two.wav"].missed == 1 and len(by_name["two.wav"].wrong_cuts) == 1
    assert len(by_name["three.wav"].wrong_cuts) == 1       # the same script over a clean clip
    assert by_name["broken.wav"].error
    totals = report["totals"]
    assert (totals["clips"], totals["clips_failed"]) == (4, 1)
    assert (totals["expected"], totals["caught"], totals["missed"], totals["wrong_cuts"]) == (2, 1, 1, 2)

    text = "\n".join(said + report["summary"])
    assert "pineapple" not in text and "banana" not in text
    assert "word 1" in text and "word 2" in text and "MISSED" in text and "WRONG CUT" in text
    assert "COULD NOT RUN" in text
    assert said == report["lines"]

    for name in ("one", "two", "three"):
        data, rate = sf.read(str(out / f"{name}.filtered.wav"))
        assert rate == SAMPLE_RATE
    data, _ = sf.read(str(out / "one.filtered.wav"), dtype="float32")
    assert len(data) == sec(3.0) and np.all(data[sec(1.5):sec(2.0)] == 0.0)
    assert not (out / "broken.filtered.wav").exists()


def test_show_words_puts_the_words_and_what_was_heard_in_the_report(tmp_path):
    folder = make_folder(tmp_path)
    rec = FakeRecogniser([ScriptedWord("pineapple", sec(1.5), sec(2.0))])
    report = rc.run_folder(folder, Settings(), WORDS, None, recogniser=rec, realtime=False,
                           show_words=True, log=lambda line: None)
    text = "\n".join(report["lines"])
    assert '"pineapple"' in text and '"banana"' in text and "word 1" not in text


def test_namer_never_leaks_a_word_that_is_not_in_the_list():
    namer = rc.Namer(WORDS, show=False)
    assert namer.word("banana") == "word 2"
    assert namer.word("coconut") == "a word that is not in the list"
    assert namer.word("") == "(no word given)"
    assert "coconut" not in namer.heard("coconut") and namer.heard("") == "nothing"
    assert rc.Namer(WORDS, show=True).heard("coconut") == '"coconut"'


# ------------------------------------------------------------------ the command line

def test_main_refuses_a_missing_folder_an_empty_list_and_an_empty_folder(tmp_path, capsys):
    assert rc.main([str(tmp_path / "nowhere")]) == 2
    assert "no folder" in capsys.readouterr().out
    empty = tmp_path / "empty"
    empty.mkdir()
    assert rc.main([str(empty)]) == 2                      # NPASS_DATA_DIR is new: no words yet
    assert "blocked word list is empty" in capsys.readouterr().out
    assert rc.main([str(empty), "--words", "pineapple"]) == 2
    assert "No sound files" in capsys.readouterr().out


def test_main_writes_reports_without_the_words_and_exits_1_on_a_miss(tmp_path, monkeypatch, capsys):
    folder = make_folder(tmp_path)
    (folder / "broken.wav").unlink()
    out = tmp_path / "out"
    rec = rc.KeepLoaded(FakeRecogniser([ScriptedWord("pineapple", sec(1.5), sec(2.0))]))
    monkeypatch.setattr(rc, "load_recogniser", lambda settings: rec)
    monkeypatch.setattr(rc, "gpu_memory", lambda: {"process_mb": None, "card_used_mb": None})
    real = rc.run_folder
    monkeypatch.setattr(rc, "run_folder", lambda *a, **k: real(*a, **{**k, "realtime": False}))
    code = rc.main([str(folder), "--words", "pineapple,banana", "--out", str(out),
                    "--lookahead", "600", "--defaults"])
    assert code == 1                                       # clip two's word was missed
    printed = capsys.readouterr().out
    report_txt = (out / "report.txt").read_text(encoding="utf-8")
    body = json.loads((out / "report.json").read_text(encoding="utf-8"))
    for text in (printed, report_txt, json.dumps(body)):
        assert "pineapple" not in text and "banana" not in text
    assert "look-ahead 600 ms" in printed and "NOT ALL CAUGHT: 1 missed" in printed
    assert body["lookahead_ms"] == 600 and body["word_count"] == 2
    assert body["totals"]["expected"] == 2 and body["totals"]["caught"] == 1
    assert {c["name"] for c in body["clips"]} == {"one.wav", "two.wav", "three.wav"}
    assert body["clips"][0]["words"][0]["word"] == "word 1"


def test_main_exits_0_when_everything_was_caught(tmp_path, monkeypatch, capsys):
    folder = tmp_path / "clips"
    folder.mkdir()
    sf.write(str(folder / "one.wav"), voice(3.0, 1), SAMPLE_RATE, subtype="FLOAT")
    (folder / "one.txt").write_text("1.5 2.0 pineapple\n", encoding="utf-8")
    rec = rc.KeepLoaded(FakeRecogniser([ScriptedWord("pineapple", sec(1.5), sec(2.0))]))
    monkeypatch.setattr(rc, "load_recogniser", lambda settings: rec)
    monkeypatch.setattr(rc, "gpu_memory", lambda: {"process_mb": None, "card_used_mb": None})
    real = rc.run_folder
    monkeypatch.setattr(rc, "run_folder", lambda *a, **k: real(*a, **{**k, "realtime": False}))
    assert rc.main([str(folder), "--words", "pineapple", "--out", str(tmp_path / "out"),
                    "--defaults"]) == 0
    assert "NOT ALL CAUGHT" not in capsys.readouterr().out
