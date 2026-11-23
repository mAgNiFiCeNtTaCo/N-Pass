"""Sound-alike matching of a heard word against the blocked list.

Baseline written by the lead on 2026-11-22; insides rewritten by the detect station on 2026-11-23.
The signatures and meaning of `normalize`, `score`, `prefix_score` and `best_match` are fixed
(lyrics and the harness import them).

Singing stretches words ("heeeeyyy"), recognisers respell them, endings vary (-a / -er / -as / -az)
and final consonants get dropped. So each word is reduced to a rough SOUND KEY (letters squeezed,
spelling habits folded, vowels put in three classes) and two keys are compared with an edit
distance whose costs follow how singing and recognisers actually go wrong:

- a different vowel is cheap (0.3): sung vowels drift and get respelled
- a near consonant (b/p, d/t, g/k, m/n ...) costs half
- a different FIRST sound costs more (1.5): that is what tells ordinary words apart
- a dropped final consonant costs half; a plural -s / -z is free
- a missing or extra syllable costs a full step, so shorter everyday words stay well below the line

All tests use the stand-in entries `pineapple` and `banana`.
"""
from __future__ import annotations

import re
from functools import lru_cache

_NON_LETTER = re.compile(r"[^a-z]")
_REPEATS = re.compile(r"(.)\1+")
_VOWELS = frozenset("aio")          # the three vowel classes used inside a sound key
_VOWEL_CLASS = {"a": "a", "e": "i", "i": "i", "o": "o", "u": "o"}
_VOWEL_GROUP = re.compile(r"[aeiou]+")
_ENDING = re.compile(r"(?:er|ar|or|ur|ah|uh|a)$")

# consonants a recogniser (or a sung line) swaps easily; upper-case = digraph symbols
# C = ch, S = sh, T = th
_CLOSE = frozenset(
    frozenset(p) for p in (
        "bp", "dt", "gk", "fv", "mn", "sS", "SC", "Cj", "Tt", "Td", "Tf", "vb", "gj", "kC", "lr",
    )
)

SUB_VOWEL = 0.3
SUB_CLOSE = 0.5
SUB_OTHER = 1.0
ONSET_FACTOR = 1.5
INDEL = 1.0                 # a missing or extra vowel (a syllable)
INDEL_CONSONANT = 1.5       # a missing or extra consonant in the middle of a word is rare
DROP_FINAL_CONSONANT = 0.5


def normalize(word: str) -> str:
    """Lower-case letters only, repeated letters squeezed: 'Heeeyyy!' -> 'hey'."""
    w = _NON_LETTER.sub("", word.lower().replace("’", "'"))
    return _REPEATS.sub(r"\1", w)


def _split_plural(w: str) -> tuple[str, bool]:
    if len(w) > 3 and w[-1] in "sz" and w[-2] not in "sz":
        return w[:-1], True
    return w, False


@lru_cache(maxsize=8192)
def sound_key(word: str) -> str:
    """Rough phonetic key: same key = sounds about the same. A final 's' is the plural."""
    w = normalize(word)
    if not w:
        return ""
    w, plural = _split_plural(w)
    # the same sound, spelled differently
    for a, b in (("tch", "C"), ("ch", "C"), ("sh", "S"), ("ph", "f"), ("th", "T"), ("ck", "k"),
                 ("dg", "j"), ("qu", "kw"), ("q", "k"), ("x", "ks"), ("z", "s")):
        w = w.replace(a, b)
    w = re.sub(r"c(?=[eiy])", "s", w).replace("c", "k")
    w = re.sub(r"(?<=[aeiou])gh", "", w).replace("gh", "g")
    # y is a consonant only at the start of a word before a vowel
    if w[0] == "y" and len(w) > 1 and w[1] in "aeiou":
        w = "y" + w[1:].replace("y", "i")
    else:
        w = w.replace("y", "i")
    # endings that sing and hear the same: -er / -ar / -or / -a / -ah / -uh
    if len(w) > 2:
        w = _ENDING.sub("a", w)
    # "-le" after a consonant is just an l; other final e's after a consonant are silent
    if len(w) > 2 and w.endswith("le") and w[-3] not in "aeiou":
        w = w[:-1]
    elif len(w) > 2 and w[-1] == "e" and w[-2] not in "aeiou" and _VOWEL_GROUP.search(w[:-1]):
        w = w[:-1]
    # h and w carry no sound of their own after the first letter (except in "kw")
    head, tail = w[0], w[1:]
    tail = tail.replace("h", "")
    tail = re.sub(r"(?<=[aeiou])w", "", tail)
    w = head + tail
    w = _VOWEL_GROUP.sub(lambda m: _VOWEL_CLASS[m.group(0)[0]], w)
    w = _REPEATS.sub(r"\1", w)
    return w + ("s" if plural and not w.endswith("s") else "")


def _sub_cost(x: str, y: str, onset: bool) -> float:
    if x == y:
        return 0.0
    xv, yv = x in _VOWELS, y in _VOWELS
    if xv and yv:
        return SUB_VOWEL
    if not xv and not yv and frozenset((x, y)) in _CLOSE:
        c = SUB_CLOSE
    else:
        c = SUB_OTHER
    return c * ONSET_FACTOR if onset else c


def _indel(x: str) -> float:
    return INDEL if x in _VOWELS else INDEL_CONSONANT


def _distance(a: str, b: str) -> float:
    """Weighted edit distance from heard key `a` to entry key `b`."""
    la, lb = len(a), len(b)
    prev = [0.0] * (lb + 1)
    for j in range(1, lb + 1):
        prev[j] = prev[j - 1] + _indel(b[j - 1])
    for i in range(1, la + 1):
        cur = [prev[0] + _indel(a[i - 1])] + [0.0] * lb
        for j in range(1, lb + 1):
            # leaving out entry sound b[j-1]; the entry's last consonant drops cheaply in singing
            skip_b = _indel(b[j - 1])
            if i == la and j == lb and b[j - 1] not in _VOWELS:
                skip_b = DROP_FINAL_CONSONANT
            cur[j] = min(
                prev[j - 1] + _sub_cost(a[i - 1], b[j - 1], i == 1 and j == 1),
                prev[j] + _indel(a[i - 1]),          # an extra heard sound
                cur[j - 1] + skip_b,
            )
        prev = cur
    return prev[lb]


def _similar(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return max(0.0, 1.0 - _distance(a, b) / max(len(a), len(b)))


def _strip_s(k: str) -> str:
    return k[:-1] if len(k) > 2 and k.endswith("s") else k


@lru_cache(maxsize=16384)
def score(heard: str, entry: str) -> float:
    """0..1: how much `heard` sounds like the blocked entry (1.0 = same sound key)."""
    a, b = sound_key(heard), sound_key(entry)
    if not a or not b:
        return 0.0
    # a plural -s / -z on either side is free
    return max(_similar(a, b), _similar(_strip_s(a), _strip_s(b)))


@lru_cache(maxsize=16384)
def prefix_score(heard_so_far: str, entry: str) -> float:
    """0..1: how much a word that is STILL BEING SUNG looks like the start of the entry.
    Needs at least half of the entry's sound key before it can score above 0."""
    a, b = sound_key(heard_so_far), _strip_s(sound_key(entry))
    if not a or not b or len(a) * 2 < len(b):
        return 0.0
    full = score(heard_so_far, entry)
    if len(a) >= len(b):
        return full
    best = 0.0
    # the recogniser may have dropped or added a sound, so try the neighbouring prefix lengths too
    for k in (len(a) - 1, len(a), len(a) + 1):
        if k < 1 or k > len(b) or k * 2 < len(b):
            continue
        s = _similar(a, b[:k]) * (k / len(b)) ** 0.5
        if s > best:
            best = s
    return max(best, full)


def best_match(heard: str, blocked: list[str], partial: bool = False) -> tuple[str, float]:
    """(entry, score) of the best blocked entry for `heard`; ('', 0.0) when the list is empty."""
    best, best_s = "", 0.0
    fn = prefix_score if partial else score
    for entry in blocked:
        s = fn(heard, entry)
        if s > best_s:
            best, best_s = entry, s
    return best, best_s
