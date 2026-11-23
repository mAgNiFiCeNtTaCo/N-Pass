"""LRC parsing and the danger-window maths (pure: no network, no clock).
Stand-in blocked words only: pineapple, banana."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from npass.lyrics.lrclib import parse_lrc                                   # noqa: E402
from npass.lyrics.windows import (build_windows, censored_match, dangerous_lines,   # noqa: E402
                                  line_matches, ms_windows, to_samples)
from npass.types import DangerWindow                                        # noqa: E402

BLOCKED = ["pineapple", "banana"]


# ---------- LRC parsing ----------

def test_parse_basic_two_digit_fraction():
    assert parse_lrc("[00:12.34] hello there\n[01:02.50] second line") == [
        (12_340, "hello there"), (62_500, "second line")]


def test_parse_odd_timestamps():
    text = "\n".join([
        "[00:05] no fraction",
        "[00:06.5] one digit",
        "[00:07.123] three digits",
        "[00:08:45] colon fraction",
        "[1:09.00] one digit minute",
        "[100:00.00] long song",
        "[ 00:10.00 ]   padded   text  ",
    ])
    assert parse_lrc(text) == [
        (5_000, "no fraction"), (6_500, "one digit"), (7_123, "three digits"),
        (8_450, "colon fraction"), (10_000, "padded text"), (69_000, "one digit minute"),
        (6_000_000, "long song")]


def test_parse_keeps_empty_timed_lines_and_skips_blank_and_untimed():
    text = "[00:01.00] first\n\n   \n[00:04.00]\nno tag here\n[00:09.00] last\r\n"
    assert parse_lrc(text) == [(1_000, "first"), (4_000, ""), (9_000, "last")]


def test_parse_multiple_tags_repeat_the_line_and_sort():
    text = "[00:30.00][00:10.00][01:00.00] chorus\n[00:20.00] verse"
    assert parse_lrc(text) == [(10_000, "chorus"), (20_000, "verse"), (30_000, "chorus"),
                               (60_000, "chorus")]


def test_parse_metadata_offset_and_word_tags():
    text = ("[ar: Somebody]\n[ti: Fruit Song]\n[length: 03:10]\n[offset: +500]\n"
            "[00:10.00] <00:10.00> one <00:10.50> two\n[00:00.20] early")
    # a positive offset moves the lyrics earlier; never below zero
    assert parse_lrc(text) == [(0, "early"), (9_500, "one two")]


def test_parse_nothing():
    assert parse_lrc("") == [] and parse_lrc(None) == []
    assert parse_lrc("just words\nno timing at all") == []
    assert parse_lrc("[ar: x]\n[garbage]\n[99:99:99:99] x") == []


# ---------- which lines are dangerous ----------

def test_line_matches_plain_word_and_punctuation():
    assert line_matches("I ate a Pineapple, yeah!", BLOCKED) == ("pineapple",)
    assert line_matches("(banana) ... \"pineapple\"", BLOCKED) == ("pineapple", "banana")
    assert line_matches("nothing to see here", BLOCKED) == ()
    assert line_matches("", BLOCKED) == () and line_matches("pineapple", []) == ()


def test_line_matches_stretched_plural_hyphen_and_split_word():
    assert line_matches("piiineappllle", BLOCKED) == ("pineapple",)
    assert line_matches("two bananas please", BLOCKED) == ("banana",)
    assert line_matches("that sweet-pineapple taste", BLOCKED) == ("pineapple",)
    assert line_matches("pine apple on the table", BLOCKED) == ("pineapple",)


def test_censored_token_matches_first_letter_and_length_within_one():
    assert censored_match("p*******e", BLOCKED) == ["pineapple"]          # the brief's example
    assert censored_match("p********", BLOCKED) == ["pineapple"]          # same length, tail hidden
    assert censored_match("p*******", BLOCKED) == ["pineapple"]           # one shorter
    assert censored_match("p*********", BLOCKED) == ["pineapple"]         # one longer
    assert censored_match("P*******E!", BLOCKED) == ["pineapple"]         # case and punctuation
    assert censored_match("b####a", BLOCKED) == ["banana"]
    assert censored_match("b--a", ["bana"]) == ["bana"]
    assert censored_match("p*****", BLOCKED) == []                        # 3 short of pineapple
    assert censored_match("x*******e", BLOCKED) == []                     # wrong first letter
    assert censored_match("pineapple", BLOCKED) == []                     # not censored at all
    assert censored_match("--", BLOCKED) == []                            # a bare dash is no word


def test_fully_hidden_token_matches_on_length_alone():
    assert censored_match("******", BLOCKED) == ["banana"]
    assert censored_match("***", BLOCKED) == []


def test_line_with_censored_token_is_dangerous():
    lines = [(1_000, "a clean line"), (5_000, "pass the p*******e now"), (9_000, "clean again")]
    got = dangerous_lines(lines, BLOCKED)
    assert [(d.start_ms, d.end_ms, d.words) for d in got] == [(5_000, 9_000, ("pineapple",))]


# ---------- window maths ----------

LINES = [(10_000, "clean"), (20_000, "one pineapple"), (25_000, "clean"), (40_000, ""),
         (50_000, "big banana"), (52_000, "more pineapple"), (58_000, "clean"),
         (90_000, "last banana")]


def test_window_edges_and_sample_conversion():
    # song is at 15.000 s when the mic is at sample 1_000_000
    w = build_windows([(10_000, "clean"), (20_000, "one pineapple"), (25_000, "clean")],
                      BLOCKED, position_ms=15_000, now_sample=1_000_000, pre_ms=500, post_ms=700)
    assert w == [DangerWindow(start=1_000_000 + (19_500 - 15_000) * 48,
                              end=1_000_000 + (25_700 - 15_000) * 48,
                              line="one pineapple", words=("pineapple",))]
    assert w[0].start == 1_216_000 and w[0].end == 1_513_600


def test_window_in_the_past_has_earlier_samples():
    w = build_windows([(20_000, "one pineapple"), (25_000, "clean")], BLOCKED,
                      position_ms=60_000, now_sample=5_000_000, pre_ms=500, post_ms=700)
    assert w[0].start == 5_000_000 - 40_500 * 48 and w[0].end == 5_000_000 - 34_300 * 48


def test_last_line_gets_six_seconds_plus_the_usual_margin():
    w = build_windows([(90_000, "last banana")], BLOCKED, 0, 0, 500, 700)
    assert w == [DangerWindow(89_500 * 48, 96_700 * 48, "last banana", ("banana",))]


def test_empty_timed_line_ends_the_window():
    w = build_windows([(30_000, "pineapple"), (33_000, ""), (60_000, "clean")], BLOCKED, 0, 0,
                      500, 700)
    assert (w[0].start, w[0].end) == (29_500 * 48, 33_700 * 48)


def test_overlapping_windows_merge():
    w = build_windows(LINES, BLOCKED, 0, 0, 500, 700)
    assert [(x.start // 48, x.end // 48) for x in w] == [
        (19_500, 25_700), (49_500, 58_700), (89_500, 96_700)]
    merged = w[1]
    assert merged.line == "big banana / more pineapple"
    assert set(merged.words) == {"banana", "pineapple"}


def test_no_merge_when_apart_and_merge_when_margins_touch():
    lines = [(10_000, "pineapple"), (11_000, "x"), (12_200, "banana"), (13_000, "x")]
    apart = build_windows(lines, BLOCKED, 0, 0, pre_ms=100, post_ms=100)
    assert len(apart) == 2                                  # 9.9-11.1 and 12.1-13.1
    touching = build_windows(lines, BLOCKED, 0, 0, pre_ms=500, post_ms=700)
    assert len(touching) == 1 and (touching[0].start, touching[0].end) == (9_500 * 48, 13_700 * 48)


def test_windows_sorted_even_if_lines_are_not():
    w = build_windows(list(reversed(LINES)), BLOCKED, 0, 0, 500, 700)
    assert [x.start for x in w] == sorted(x.start for x in w) and len(w) == 3


def test_no_blocked_words_or_no_lines_means_no_windows():
    assert build_windows(LINES, [], 0, 0, 500, 700) == []
    assert build_windows([], BLOCKED, 0, 0, 500, 700) == []


def test_extra_margin_widens_and_can_merge():
    ms = ms_windows(dangerous_lines(
        [(10_000, "pineapple"), (11_000, "x"), (12_200, "banana"), (13_000, "x")], BLOCKED),
        100, 100)
    assert len(to_samples(ms, 0, 0)) == 2
    wide = to_samples(ms, 0, 0, extra_ms=500)
    assert len(wide) == 1 and (wide[0].start, wide[0].end) == (9_400 * 48, 13_600 * 48)
