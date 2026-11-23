"""Sound-alike matching. Stand-in blocked entries only: `pineapple` and `banana`."""
from __future__ import annotations

import pytest

from npass.match import best_match, normalize, prefix_score, score, sound_key
from npass.settings import Settings

BLOCKED = ["pineapple", "banana"]
THRESHOLD = Settings().threshold
WINDOW_THRESHOLD = Settings().window_threshold

COMMON_WORDS = """
the be to of and a in that have i it for not on with he as you do at this but his by from they we
say her she or an will my one all would there their what so up out if about who get which go me
when make can like time no just him know take people into year your good some could them see other
than then now look only come its over think also back after use two how our work first well way
even new want because any these give day most us is are was were been has had did does said got
made went came saw knew thought took gave found told asked felt left put kept let began seemed
helped showed heard played ran moved lived believed brought happened wrote sat stood lost paid met
man woman child world life hand part place case week company system program question government
number night point home water room mother area money story fact month lot right study book eye job
word business issue side kind head house service friend father power hour game line end member law
car city community name president team minute idea kid body information parent face others level
office door health person art war history party result change morning reason research girl guy
moment air teacher force education foot boy age policy music love baby dance heart fire light sing
song singer dream feel tonight forever never always maybe yeah baby girl boy money honey sunny
funny happy sorry better bigger butter water letter number summer winter dinner paper banner
manner corner color power tower flower shower river never ever over under after sister brother
panel pencil apple people simple sample purple maple table cable double trouble little middle
pine pain pan pin pen fine line mine nine wine sign open happen happy penny pony piano
band bend bond bone bean been ban bin bun nana mama papa drama llama comma gonna wanna gotta
running coming going being doing having making taking looking feeling singing dancing
beautiful wonderful different important another around between through without against
""".split()


def test_common_word_list_is_big_enough():
    assert len(set(COMMON_WORDS)) >= 300


def test_normalize_keeps_its_meaning():
    assert normalize("Heeeyyy!") == "hey"
    assert normalize("Ba-na-na") == "banana"
    assert normalize("") == ""


def test_exact_and_empty():
    assert score("banana", "banana") == 1.0
    assert score("pineapple", "pineapple") == 1.0
    assert score("", "banana") == 0.0
    assert score("banana", "") == 0.0
    assert best_match("banana", []) == ("", 0.0)


@pytest.mark.parametrize("heard", [
    "Banana!", "bananaaaa", "baanaanaa", "ba-na-na", "bananah", "bananuh", "bananer", "bananas",
    "bananaz", "bananers", "bananahs", "bunana", "benana", "bonana", "banahna", "bananna",
    "bannana", "panana", "banan", "bananar",
])
def test_sung_and_respelled_banana_matches(heard):
    assert score(heard, "banana") >= THRESHOLD, (heard, sound_key(heard))


@pytest.mark.parametrize("heard", [
    "Pineapple", "piiineapple", "pineappleee", "pine-apple", "pineapples", "pineapplez",
    "pineappel", "pineapel", "pinapple", "pynapple", "pineappl", "pineapp", "pineable",
    "bineapple", "pineappo",
])
def test_sung_and_respelled_pineapple_matches(heard):
    assert score(heard, "pineapple") >= THRESHOLD, (heard, sound_key(heard))


def test_plural_entry_matches_singular_heard():
    assert score("banana", "bananas") == 1.0
    assert score("bananaz", "bananas") == 1.0


def test_common_words_stay_under_the_threshold():
    too_close = []
    for w in sorted(set(COMMON_WORDS)):
        entry, s = best_match(w, BLOCKED)
        if s >= THRESHOLD:
            too_close.append((w, entry, round(s, 2)))
    assert not too_close, too_close


def test_most_common_words_stay_under_the_window_threshold():
    """Inside a danger window the bar is lower on purpose; it must still not catch everything."""
    close = [w for w in set(COMMON_WORDS) if best_match(w, BLOCKED)[1] >= WINDOW_THRESHOLD]
    assert len(close) <= len(set(COMMON_WORDS)) * 0.10, sorted(close)


def test_prefix_needs_half_the_word():
    assert prefix_score("b", "banana") == 0.0
    assert prefix_score("ba", "banana") == 0.0
    assert prefix_score("pi", "pineapple") == 0.0


def test_prefix_grows_with_the_word():
    steps = [prefix_score(p, "banana") for p in ("ban", "bana", "banan", "banana")]
    assert steps == sorted(steps)
    assert steps[0] >= WINDOW_THRESHOLD          # half the word is enough inside a window
    assert steps[1] >= THRESHOLD                 # two thirds is enough anywhere
    assert steps[-1] == 1.0
    assert prefix_score("pineap", "pineapple") >= THRESHOLD
    assert prefix_score("pinea", "pineapple") >= WINDOW_THRESHOLD


def test_prefix_of_a_finished_word_is_the_plain_score():
    assert prefix_score("bananas", "banana") == score("bananas", "banana")
    assert prefix_score("table", "banana") < WINDOW_THRESHOLD


def test_prefix_does_not_fire_on_unrelated_starts():
    for w in ("tab", "pan", "pen", "dan", "man", "can", "bit", "pot", "son"):
        assert best_match(w, BLOCKED, partial=True)[1] < THRESHOLD, w


def test_best_match_picks_the_right_entry():
    assert best_match("bananas", BLOCKED)[0] == "banana"
    assert best_match("pineappel", BLOCKED)[0] == "pineapple"
    entry, s = best_match("pinea", BLOCKED, partial=True)
    assert entry == "pineapple" and s > 0
