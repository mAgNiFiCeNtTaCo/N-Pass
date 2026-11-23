"""LyricService: follows Spotify, fetches timed lyrics, and answers "which stretches of mic time
are risky?" for the detector and the safety guard. See docs/CONTRACT.md.

How the position is known: Spotify is asked about once a second. Each answer gives a song
position that was true at the middle of that request; between answers the position moves with the
mic sample clock. The window margins from settings (window_pre_ms / window_post_ms) cover the
leftover error.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable

from npass.types import SAMPLE_RATE, DangerWindow, TrackInfo

from .lrclib import LyricStore, LyricsUnavailable
from .spotify import (MSG_NO_CLIENT_ID, MSG_NOT_SIGNED_IN, MSG_UNREACHABLE, Playback,
                      SpotifyClient)
from .windows import DangerLine, MsWindow, dangerous_lines, ms_windows, to_samples

log = logging.getLogger("npass.lyrics")

POLL_S = 1.0
JUMP_MS = 700            # position off by more than this = the user seeked or skipped
JUMP_WIDEN_MS = 500      # extra margin each side of every window...
JUMP_WIDEN_S = 2.0       # ...for this long after a jump
STALE_S = 8.0            # no fresh position for this long: stop trusting the windows

MSG_OFF = "Lyric mode is off"
MSG_NOT_PLAYING = "Spotify is not playing"
MSG_PAUSED = "Spotify is paused"
MSG_NO_LYRICS = "Lyric mode: this song has no timed lyrics, listening only"
MSG_LYRICS_DOWN = "Lyric mode: the lyrics site is not reachable, listening only"


@dataclass(frozen=True)
class _Snap:
    """What windows() needs, swapped in whole so readers never see half an update."""
    position_ms: float
    sample: float            # mic capture index at which the song was at position_ms
    at: float                # clock time of that moment
    windows: tuple[MsWindow, ...]
    widen_until: float


class LyricService:
    def __init__(self, settings, words: Callable[[], list[str]], now_sample: Callable[[], int],
                 *, spotify: SpotifyClient | None = None, store: LyricStore | None = None,
                 clock: Callable[[], float] = time.monotonic, fetch_async: bool = True):
        self._settings = settings
        self._words = words
        self._now_sample = now_sample
        self._clock = clock
        self._spotify = spotify if spotify is not None else SpotifyClient(settings, clock=clock)
        self._store = store if store is not None else LyricStore(clock=clock)
        self._fetch_async = fetch_async

        self._lock = threading.RLock()
        self._snap: _Snap | None = None
        self._status = MSG_NOT_PLAYING
        self._pb: Playback | None = None          # last answer that had a track
        self._track_id = ""
        self._playing = False
        self._pos_ms = 0.0
        self._pos_at = 0.0
        self._pos_sample = 0.0
        self._widen_until = 0.0
        self._coast_until = 0.0                   # >now: Spotify is silent, windows kept running
        self._lyrics = "none"                   # none | loading | ready | missing | error
        self._lines: list[tuple[int, str]] | None = None
        self._retry_lyrics_at = 0.0
        self._danger: list[DangerLine] = []
        self._danger_key: tuple | None = None

        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    # ---- life cycle ----

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="npass-lyrics", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        t = self._thread
        if t and t is not threading.current_thread():
            t.join(timeout=3.0)
        self._thread = None
        with self._lock:
            self._snap = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception as e:                 # never let the thread die; never echo details
                log.warning("Lyric mode hit a problem (%s); trying again", type(e).__name__)
                with self._lock:
                    self._snap = None
            self._wake.wait(POLL_S)
            self._wake.clear()

    # ---- the contract ----

    def windows(self, from_sample: int, to_sample: int) -> list[DangerWindow]:
        snap = self._snap
        if snap is None or not snap.windows or not getattr(self._settings, "lyrics_enabled", True):
            return []
        now = self._clock()
        coasting = now < self._coast_until
        if now - snap.at > STALE_S and not coasting:
            return []
        # coasting = Spotify stopped answering mid-song: keep following the last known position on
        # the mic clock (the song is most likely still playing), with wider margins
        extra = JUMP_WIDEN_MS if (coasting or now < snap.widen_until) else 0
        lo_ms = snap.position_ms + (from_sample - snap.sample) * 1000.0 / SAMPLE_RATE - extra
        hi_ms = snap.position_ms + (to_sample - snap.sample) * 1000.0 / SAMPLE_RATE + extra
        near = [w for w in snap.windows if w.end_ms >= lo_ms - 1 and w.start_ms <= hi_ms + 1]
        if not near:
            return []
        return [w for w in to_samples(near, snap.position_ms, snap.sample, extra)
                if w.end > from_sample and w.start < to_sample]

    def track(self) -> TrackInfo | None:
        with self._lock:
            pb = self._pb
            if pb is None or pb.track is None:
                return None
            pos = self._pos_ms
            if self._playing:
                pos += max(0.0, self._clock() - self._pos_at) * 1000.0
                if pb.duration_ms:
                    pos = min(pos, pb.duration_ms)
            return replace(pb.track, position_ms=int(pos), is_playing=self._playing)

    def status(self) -> str:
        if not getattr(self._settings, "lyrics_enabled", True):
            return MSG_OFF
        blocked = self._spotify.status()
        if blocked:
            return blocked
        return self._status

    def degraded(self) -> bool:
        """True while Spotify is not answering and the windows run on the last known position."""
        return self._clock() < self._coast_until

    def logged_in(self) -> bool:
        return self._spotify.logged_in()

    def login(self) -> None:
        self._spotify.login()
        self._wake.set()

    def logout(self) -> None:
        self._spotify.logout()
        with self._lock:
            self._clear(MSG_NOT_SIGNED_IN)
        self._wake.set()

    # ---- one round (the thread calls this about once a second; tests call it directly) ----

    def poll_once(self) -> None:
        s = self._settings
        if not getattr(s, "lyrics_enabled", True):
            with self._lock:
                self._clear(MSG_OFF)
            return
        why_not = self._spotify.status()
        if why_not:
            with self._lock:
                self._clear(why_not)
            return
        pb = self._spotify.poll()
        t_now, s_now = self._clock(), float(self._now_sample())
        fetch: Playback | None = None
        with self._lock:
            if pb.problem:
                if self._snap is not None and self._playing and self._pb is not None:
                    # Spotify trouble while a song with risky lines was playing: do NOT drop the
                    # windows. Keep them running until the song's end (10 minutes at most).
                    if t_now >= self._coast_until:
                        dur = float(self._pb.duration_ms or 0)
                        pos = self._pos_ms + max(0.0, t_now - self._pos_at) * 1000.0
                        left_s = (dur - pos) / 1000.0 if dur else 600.0
                        self._coast_until = t_now + max(0.0, min(600.0, left_s))
                    if t_now < self._coast_until:
                        self._status = ("Spotify is not answering. Still following the last "
                                        "known place in the song.")
                        return
                self._coast_until = 0.0
                self._clear(pb.problem)
                return
            self._coast_until = 0.0
            if pb.track is None:
                self._clear(MSG_NOT_PLAYING)
                return
            if pb.fresh:
                fetch = self._take(pb, t_now, s_now)
            elif self._lyrics == "error" and t_now >= self._retry_lyrics_at:
                fetch = self._pb
            self._rebuild()
        if fetch is not None:
            self._start_fetch(fetch)

    def _clear(self, status: str) -> None:
        """Nothing to follow right now: no windows. The lyrics we already hold are kept."""
        self._snap = None
        self._playing = False
        self._status = status
        if status in (MSG_NOT_PLAYING, MSG_NOT_SIGNED_IN, MSG_NO_CLIENT_ID, MSG_OFF):
            self._pb = None
            self._track_id = ""
            self._lines, self._lyrics, self._danger, self._danger_key = None, "none", [], None
        elif status != MSG_UNREACHABLE:
            self._pb = None

    def _take(self, pb: Playback, t_now: float, s_now: float) -> Playback | None:
        """Fold a fresh answer in. Returns the playback to fetch lyrics for, if any."""
        track = pb.track
        assert track is not None
        fetch = None
        jumped = False
        if track.track_id != self._track_id:
            self._track_id = track.track_id
            self._lines, self._lyrics, self._danger, self._danger_key = None, "loading", [], None
            jumped = True
            fetch = pb
            log.info("Now following a new song")
        elif self._lyrics == "error" and t_now >= self._retry_lyrics_at:
            self._lyrics = "loading"
            fetch = pb
        if track.is_playing:
            if not self._playing:
                jumped = True                       # started or resumed: exact moment unknown
            else:
                expected = self._pos_ms + (pb.anchor - self._pos_at) * 1000.0
                if abs(track.position_ms - expected) > JUMP_MS:
                    jumped = True
        self._pb = pb
        self._playing = track.is_playing
        self._pos_ms = float(track.position_ms)
        self._pos_at = pb.anchor
        # the mic index at the middle of the request = the index now, minus what passed since
        self._pos_sample = s_now - max(0.0, t_now - pb.anchor) * SAMPLE_RATE
        if jumped and track.is_playing:
            self._widen_until = t_now + JUMP_WIDEN_S
        return fetch

    def _rebuild(self) -> None:
        """Work out status and the windows snapshot from what is known. Call with the lock held."""
        pb = self._pb
        if pb is None or pb.track is None:
            self._snap = None
            return
        name = f"{pb.track.artist} - {pb.track.title}" if pb.track.artist else pb.track.title
        if self._lyrics == "loading":
            self._status = f"Lyric mode: {name} (looking up the lyrics)"
        elif self._lyrics == "missing":
            self._status = MSG_NO_LYRICS
        elif self._lyrics == "error":
            self._status = MSG_LYRICS_DOWN
        if self._lyrics != "ready" or not self._lines:
            self._snap = None
            if not self._playing and self._lyrics != "loading":
                self._status = MSG_PAUSED
            return
        blocked = list(self._words() or [])
        key = (self._track_id, tuple(blocked))
        if key != self._danger_key:                  # new song, or the user edited the word list
            self._danger = dangerous_lines(self._lines, blocked)
            self._danger_key = key
        n = len(self._danger)
        risky = "no risky lines" if n == 0 else "1 risky line" if n == 1 else f"{n} risky lines"
        if not self._playing:
            self._status = MSG_PAUSED
            self._snap = None                        # paused: nothing advances, so no windows
            return
        self._status = f"Lyric mode: {name} ({risky})"
        pre = int(getattr(self._settings, "window_pre_ms", 500))
        post = int(getattr(self._settings, "window_post_ms", 700))
        self._snap = _Snap(self._pos_ms, self._pos_sample, self._pos_at,
                           tuple(ms_windows(self._danger, pre, post)), self._widen_until)

    # ---- lyrics lookup (off the polling thread, so the position keeps updating) ----

    def _start_fetch(self, pb: Playback) -> None:
        if self._fetch_async:
            threading.Thread(target=self._fetch, args=(pb,), name="npass-lyrics-fetch",
                             daemon=True).start()
        else:
            self._fetch(pb)

    def _fetch(self, pb: Playback) -> None:
        track = pb.track
        if track is None:
            return
        state, lines, retry_s = "missing", None, 0.0
        try:
            lines = self._store.get(track.track_id, pb.primary_artist or track.artist, track.title,
                                    pb.album, pb.duration_ms)
            if lines:
                state = "ready"
        except LyricsUnavailable as e:
            state, retry_s = "error", e.retry_after_s
        except Exception as e:
            log.warning("Lyrics lookup failed (%s)", type(e).__name__)
            state, retry_s = "error", 30.0
        with self._lock:
            if self._track_id != track.track_id:     # the song changed while we were asking
                return
            self._lines, self._lyrics = lines, state
            self._danger_key = None
            if state == "error":
                self._retry_lyrics_at = self._clock() + retry_s
            self._rebuild()
