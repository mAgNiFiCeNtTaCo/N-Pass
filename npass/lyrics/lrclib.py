"""Timed lyrics from LRCLIB (https://lrclib.net), with a disk cache.

Checked against https://lrclib.net/docs.md on 2026-11-23:
- GET /api/get    track_name, artist_name (required), album_name, duration in seconds (1..3600,
                  matched within about 2 s). 404 = not found.
- GET /api/search track_name / artist_name / album_name (or q). At most 20 records.
- records carry `instrumental` and `syncedLyrics` (LRC text, or null).
- 429 comes with Retry-After, which must be obeyed.

The only things sent: the playing song's artist, title, album and length. Nothing about the user.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
from typing import Callable
from urllib.parse import urlsplit

import requests

from npass import settings as _settings

log = logging.getLogger("npass.lyrics")

GET_URL = "https://lrclib.net/api/get"
SEARCH_URL = "https://lrclib.net/api/search"
ALLOWED_HOST = "lrclib.net"
USER_AGENT = "N-Pass (local lyric lookup)"
HTTP_TIMEOUT_S = (3.05, 6.0)
SEARCH_DURATION_SLACK_S = 3.0

Lines = list[tuple[int, str]]

_TAG = re.compile(r"\[([^\[\]]*)\]")
_TIME = re.compile(r"^\s*(\d{1,3})\s*:\s*(\d{1,2})(?:[.:,](\d{1,3}))?\s*$")
_OFFSET = re.compile(r"^\s*offset\s*:\s*([+-]?\d{1,7})\s*$", re.IGNORECASE)
_WORD_TIME = re.compile(r"<\s*\d{1,3}:\d{1,2}(?:[.:,]\d{1,3})?\s*>")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class LyricsUnavailable(Exception):
    """The lyrics site could not be asked right now. `retry_after_s` says when to try again."""

    def __init__(self, retry_after_s: float = 30.0):
        super().__init__("the lyrics site could not be reached")
        self.retry_after_s = retry_after_s


def parse_lrc(text: str | None) -> Lines:
    """LRC text -> [(start_ms, text), ...] sorted by time.

    Understands [mm:ss], [mm:ss.x], [mm:ss.xx], [mm:ss.xxx], [mm:ss:xx], several time tags on one
    line (the line repeats), [offset:+/-ms] and skips other tags such as [ar:...]. A timed line
    with no words is kept (text ''): it marks where the line before it ends.
    """
    if not text:
        return []
    out: Lines = []
    offset_ms = 0
    for raw in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw.strip().lstrip("﻿")
        if not line:
            continue
        times: list[int] = []
        pos = 0
        while True:
            m = _TAG.match(line, pos)
            if not m:
                break
            inner = m.group(1)
            t = _TIME.match(inner)
            if t:
                frac = t.group(3) or ""
                ms = int((frac + "000")[:3]) if frac else 0
                times.append(int(t.group(1)) * 60_000 + int(t.group(2)) * 1000 + ms)
            else:
                o = _OFFSET.match(inner)
                if o:
                    offset_ms = int(o.group(1))
            pos = m.end()
            while pos < len(line) and line[pos] in " \t":
                pos += 1
        if not times:
            continue
        words = _WORD_TIME.sub(" ", line[pos:])
        words = " ".join(words.split())
        for ms in times:
            out.append((ms, words))
    if offset_ms:                               # a positive offset makes the lyrics come earlier
        out = [(max(0, ms - offset_ms), w) for ms, w in out]
    out.sort(key=lambda p: p[0])
    return out


def _plain(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


class LrcLibClient:
    def __init__(self, session=None):
        self._http = session if session is not None else requests.Session()

    def _get(self, url: str, params: dict):
        if urlsplit(url).hostname != ALLOWED_HOST or not url.startswith("https://"):
            raise RuntimeError("blocked a request to a host that is not the lyrics site")
        try:
            r = self._http.request("GET", url, params=params, timeout=HTTP_TIMEOUT_S,
                                   allow_redirects=False,
                                   headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        except requests.RequestException as e:
            log.debug("Lyrics request failed (%s)", type(e).__name__)
            raise LyricsUnavailable() from None
        if r.status_code == 429:
            try:
                wait = float(r.headers.get("Retry-After", "30"))
            except (TypeError, ValueError):
                wait = 30.0
            raise LyricsUnavailable(max(1.0, min(wait, 3600.0)))
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            log.debug("Lyrics site answered HTTP %d", r.status_code)
            raise LyricsUnavailable()
        try:
            return r.json()
        except ValueError:
            raise LyricsUnavailable() from None

    @staticmethod
    def _lines_of(record) -> Lines | None:
        if not isinstance(record, dict) or record.get("instrumental"):
            return None
        raw = record.get("syncedLyrics")
        # the lyrics site is written by anyone: a broken record means "no timed lyrics", not a crash
        if not isinstance(raw, str) or len(raw) > 1_000_000:
            return None
        try:
            lines = parse_lrc(raw)
        except (ValueError, OverflowError, AttributeError, TypeError):
            return None
        return lines if any(text for _, text in lines) else None

    def fetch(self, artist: str, title: str, album: str = "", duration_ms: int = 0) -> Lines | None:
        """Timed lines for the song, or None (not found, instrumental, or only untimed lyrics).
        Raises LyricsUnavailable when the site could not be asked."""
        if not title or not artist:
            return None
        seconds = int(round(duration_ms / 1000.0)) if duration_ms else 0
        params = {"track_name": title, "artist_name": artist}
        if album:
            params["album_name"] = album
        if 1 <= seconds <= 3600:
            params["duration"] = seconds
        record = self._get(GET_URL, params)
        if isinstance(record, dict):
            if record.get("instrumental"):
                return None
            lines = self._lines_of(record)
            if lines:
                return lines
        found = self._get(SEARCH_URL, {"track_name": title, "artist_name": artist})
        if not isinstance(found, list):
            return None
        best: tuple[float, Lines] | None = None
        for rec in found:
            if not isinstance(rec, dict):
                continue
            if _plain(str(rec.get("trackName") or "")) != _plain(title):
                continue
            their, ours = _plain(str(rec.get("artistName") or "")), _plain(artist)
            if not their or not ours or (their not in ours and ours not in their):
                continue
            try:
                gap = abs(float(rec.get("duration") or 0) - duration_ms / 1000.0) if seconds else 0.0
            except (TypeError, ValueError):
                continue
            if gap > SEARCH_DURATION_SLACK_S:      # a different cut of the song: timings would be off
                continue
            lines = self._lines_of(rec)
            if lines and (best is None or gap < best[0]):
                best = (gap, lines)
        return best[1] if best else None


class LyricStore:
    """Lyrics by track id: memory, then disk, then LRCLIB. A song with no timed lyrics is
    remembered (for this run) so the site is asked once."""

    def __init__(self, client: LrcLibClient | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self._client = client if client is not None else LrcLibClient()
        self._clock = clock
        self._lock = threading.Lock()
        self._found: dict[str, Lines] = {}
        self._missing: set[str] = set()
        self._not_before = 0.0

    @staticmethod
    def cache_path(track_id: str):
        name = track_id if _SAFE_ID.match(track_id) else \
            "x" + hashlib.sha1(track_id.encode("utf-8")).hexdigest()
        folder = _settings.data_dir() / "lyrics_cache"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / (name + ".json")

    def _read_disk(self, track_id: str) -> Lines | None:
        try:
            data = json.loads(self.cache_path(track_id).read_text(encoding="utf-8"))
            lines = [(int(ms), str(text)) for ms, text in data["lines"]]
            return lines or None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write_disk(self, track_id: str, artist: str, title: str, lines: Lines) -> None:
        path = self.cache_path(track_id)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps({"track_id": track_id, "artist": artist, "title": title,
                                       "source": "lrclib.net",
                                       "lines": [[ms, text] for ms, text in lines]},
                                      ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            log.debug("Could not save lyrics to the cache folder")

    def get(self, track_id: str, artist: str, title: str, album: str = "",
            duration_ms: int = 0) -> Lines | None:
        with self._lock:
            if track_id in self._found:
                return self._found[track_id]
            if track_id in self._missing:
                return None
            lines = self._read_disk(track_id)
            if lines is None:
                wait = self._not_before - self._clock()
                if wait > 0:
                    raise LyricsUnavailable(wait)
                try:
                    lines = self._client.fetch(artist, title, album, duration_ms)
                except LyricsUnavailable as e:
                    self._not_before = self._clock() + e.retry_after_s
                    raise
                if lines:
                    self._write_disk(track_id, artist, title, lines)
            if lines:
                self._found[track_id] = lines
                return lines
            self._missing.add(track_id)
            return None
