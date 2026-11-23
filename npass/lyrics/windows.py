"""Danger windows: which stretches of mic time carry a lyric line with a blocked word.

Pure maths: no network, no clock. The service keeps the song-time ("ms") form and only shifts it
onto the mic sample clock when asked, so a new position from Spotify costs nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

from npass.match import best_match
from npass.types import DangerWindow

MATCH_SCORE = 0.85
LAST_LINE_MS = 6000
SAMPLES_PER_MS = 48

_MASK = set("*#_•＊×")           # what lyric sites put over letters
_EDGE = " \t.,!?;:\"'()[]{}<>‘’“”…¿¡/\\|~-–—"


@dataclass(frozen=True)
class DangerLine:
    start_ms: int
    end_ms: int                  # when the next line starts (last line: start + 6 s)
    text: str
    words: tuple[str, ...]       # blocked-list entries found in the line


@dataclass(frozen=True)
class MsWindow:
    start_ms: int
    end_ms: int
    line: str
    words: tuple[str, ...]


def _letters(s: str) -> int:
    return sum(1 for c in s if c.isalpha())


def is_censored(token: str) -> bool:
    if any(c in _MASK for c in token):
        return True
    return "--" in token and any(c.isalpha() for c in token)     # n--a style, not a bare dash


def censored_match(token: str, blocked: list[str]) -> list[str]:
    """Entries a starred-out token such as `p*******e` could stand for: same first letter and a
    length within 1. A token with every letter hidden (`****`) is judged on length alone."""
    tok = token.strip(_EDGE.replace("-", "")).lower()
    if not tok or not is_censored(tok):
        return []
    first = tok[0] if tok[0].isalpha() else ""
    out = []
    for entry in blocked:
        size = _letters(entry)
        if not size or abs(len(tok) - size) > 1:
            continue
        head = next((c for c in entry.lower() if c.isalpha()), "")
        if first and first != head:
            continue
        out.append(entry)
    return out


def line_matches(text: str, blocked: list[str]) -> tuple[str, ...]:
    """Blocked-list entries present in one lyric line (in list order, each once)."""
    if not text or not blocked:
        return ()
    hits: set[str] = set()
    plain: list[str] = []
    for raw in text.split():
        if is_censored(raw):
            hits.update(censored_match(raw, blocked))
            continue
        tok = raw.strip(_EDGE)
        if not tok:
            continue
        parts = [tok] + ([p for p in tok.replace("–", "-").split("-") if p] if "-" in tok else [])
        for p in parts:
            entry, s = best_match(p, blocked)
            if entry and s >= MATCH_SCORE:
                hits.add(entry)
        plain.append(tok)
    for a, b in zip(plain, plain[1:]):            # an entry sung or written as two words
        entry, s = best_match(a + b, blocked)
        if entry and s >= MATCH_SCORE and (" " in entry or len(a) > 1 and len(b) > 1):
            hits.add(entry)
    return tuple(e for e in blocked if e in hits)


def dangerous_lines(lines: list[tuple[int, str]], blocked: list[str]) -> list[DangerLine]:
    ordered = sorted(lines, key=lambda p: p[0])
    out: list[DangerLine] = []
    for i, (start, text) in enumerate(ordered):
        words = line_matches(text, blocked)
        if not words:
            continue
        end = next((t for t, _ in ordered[i + 1:] if t > start), start + LAST_LINE_MS)
        out.append(DangerLine(int(start), int(end), text, words))
    return out


def _merge(items: list[tuple[int, int, str, tuple[str, ...]]]):
    merged: list[list] = []
    for start, end, line, words in sorted(items, key=lambda w: (w[0], w[1])):
        if merged and start <= merged[-1][1]:
            last = merged[-1]
            last[1] = max(last[1], end)
            if line and line not in last[2]:
                last[2].append(line)
            last[3] = last[3] + tuple(w for w in words if w not in last[3])
        else:
            merged.append([start, end, [line] if line else [], tuple(words)])
    return [(s, e, " / ".join(ls), ws) for s, e, ls, ws in merged]


def ms_windows(danger: list[DangerLine], pre_ms: int, post_ms: int) -> list[MsWindow]:
    """Song-time windows [line start - pre, next line start + post], overlapping ones merged."""
    return [MsWindow(*w) for w in _merge(
        [(d.start_ms - int(pre_ms), d.end_ms + int(post_ms), d.text, d.words) for d in danger])]


def to_samples(windows: list[MsWindow], position_ms: float, now_sample: float,
               extra_ms: int = 0) -> list[DangerWindow]:
    """Song time -> mic samples: sample = now_sample + (t_ms - position_ms) * 48."""
    def conv(t_ms: float) -> int:
        return int(round(now_sample + (t_ms - position_ms) * SAMPLES_PER_MS))
    items = [(conv(w.start_ms - extra_ms), conv(w.end_ms + extra_ms), w.line, w.words)
             for w in windows]
    return [DangerWindow(s, e, line, ws) for s, e, line, ws in _merge(items)]


def build_windows(lines: list[tuple[int, str]], blocked: list[str], position_ms: int,
                  now_sample: int, pre_ms: int, post_ms: int) -> list[DangerWindow]:
    """Every danger window of the song on the mic clock, given that the song is at `position_ms`
    at mic sample `now_sample`. Windows already in the past come back too (the caller filters)."""
    return to_samples(ms_windows(dangerous_lines(lines, blocked), pre_ms, post_ms),
                      position_ms, now_sample)
