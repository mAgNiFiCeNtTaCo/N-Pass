import sys
from pathlib import Path

# plain `uv run pytest` does not put the project root on the path (no conftest, package = false)
for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import threading
import time

import pytest

from npass import catchlog, settings
from npass.types import Catch, CutOrder, CutReason, TrackInfo


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NPASS_DATA_DIR", str(tmp_path))
    return tmp_path


WHEN = time.mktime((2026, 11, 22, 21, 14, 3, 0, 0, -1))   # local time, like the log


def make(word="pineapple", conf=0.71, reason=CutReason.LYRIC_WINDOW, track=None, late=0.0, when=WHEN):
    order = CutOrder(cut_id=1, start=0, end=4800, word=word, heard=word, confidence=conf, reason=reason)
    return Catch(wall_time=when, order=order, track=track, late_ms=late)


def test_format_with_song():
    track = TrackInfo("id1", "Artist", "Song", 102_000, True)
    assert catchlog.format_catch(make(track=track)) == (
        '2026-11-22 21:14:03 | caught "pineapple" (0.71, lyric window) | Artist - Song @ 1:42'
    )


def test_format_without_song():
    assert catchlog.format_catch(make(conf=0.8, reason=CutReason.ASR)) == (
        '2026-11-22 21:14:03 | caught "pineapple" (0.80, asr) | no song'
    )


def test_format_late_cut():
    line = catchlog.format_catch(make(late=35.4))
    assert line.endswith("| no song | late 35 ms")


def test_format_failsafe_has_a_label_instead_of_an_empty_word():
    line = catchlog.format_catch(make(word="", conf=0.0, reason=CutReason.FAILSAFE))
    assert 'caught "(unchecked lyric line)" (0.00, failsafe)' in line


def test_log_appends_utf8_lines(data_dir):
    track = TrackInfo("id2", "Beyoncé", "Déjà Vu", 5_000, True)
    catchlog.log_catch(make(track=track))
    catchlog.log_catch(make(word="banana", late=12))
    raw = settings.catch_log_path().read_bytes()
    assert b"\r" not in raw
    lines = raw.decode("utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("Beyoncé - Déjà Vu @ 0:05")
    assert lines[1].endswith("| no song | late 12 ms")
    assert catchlog.read_recent(1) == [lines[1]]
    assert catchlog.read_recent(5) == lines


def test_read_recent_when_no_file():
    assert catchlog.read_recent(5) == []


def test_rotates_at_2_mb(data_dir, monkeypatch):
    monkeypatch.setattr(catchlog, "MAX_BYTES", 300)
    for i in range(10):
        catchlog.log_catch(make(word=f"pineapple{i}"))
    old = data_dir / "catches.1.log"
    assert old.exists()
    assert settings.catch_log_path().stat().st_size <= 300
    recent = catchlog.read_recent(4)
    assert [line.split('"')[1] for line in recent] == [f"pineapple{i}" for i in range(6, 10)]


def test_thread_safe(data_dir):
    def work(k):
        for i in range(50):
            catchlog.log_catch(make(word=f"banana{k}x{i}"))

    threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = settings.catch_log_path().read_text(encoding="utf-8").splitlines()
    assert len(lines) == 400
    assert all(line.count("|") == 2 and line.startswith("2026-11-22") for line in lines)
