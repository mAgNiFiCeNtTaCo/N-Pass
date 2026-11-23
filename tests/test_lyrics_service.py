"""LyricService with a scripted Spotify and a scripted lyrics store: pause/resume, seek, track
change, no lyrics, word-list edits, status lines. No network, no real clock."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from npass import settings as st                                   # noqa: E402
from npass.lyrics import LyricService                              # noqa: E402
from npass.lyrics.lrclib import LyricsUnavailable                  # noqa: E402
from npass.lyrics.spotify import Playback, SpotifyLoginError       # noqa: E402
from npass.types import DangerWindow, TrackInfo                    # noqa: E402

LINES = [(10_000, "clean start"), (20_000, "one pineapple"), (25_000, "clean"),
         (60_000, "a banana here"), (64_000, "clean end")]
# with pre 500 / post 700: [19.5 s, 25.7 s] and [59.5 s, 64.7 s]
FAR = 10 ** 12


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NPASS_DATA_DIR", str(tmp_path))


class World:
    """One fake clock that drives the mic sample counter too (48 samples per ms)."""

    def __init__(self):
        self.t = 500.0
        self.sample0 = 1_000_000

    def clock(self):
        return self.t

    def now_sample(self):
        return self.sample0 + int(round((self.t - 500.0) * 48_000))

    def advance(self, seconds):
        self.t += seconds


class FakeSpotify:
    def __init__(self, world):
        self.world = world
        self.signed_in = True
        self.client_id = "cid"
        self.next: Playback | None = None
        self.polls = 0
        self.login_error = None

    def status(self):
        if not self.client_id:
            return "Add your Spotify client ID in Settings to turn on lyric mode"
        return "" if self.signed_in else "Not signed in to Spotify"

    def logged_in(self):
        return self.signed_in

    def login(self):
        if self.login_error:
            raise self.login_error
        self.signed_in = True

    def logout(self):
        self.signed_in = False

    def poll(self):
        self.polls += 1
        return self.next

    # helpers for the tests
    def playing(self, pos, track_id="trk1", playing=True, artist="The Stand Ins",
                title="Fruit Song", lag=0.0):
        self.next = Playback(track=TrackInfo(track_id, artist, title, pos, playing),
                             album="Fruit Album", primary_artist=artist, duration_ms=200_000,
                             anchor=self.world.t - lag)


class FakeStore:
    def __init__(self):
        self.by_id = {"trk1": LINES}
        self.asked = []
        self.fail = None

    def get(self, track_id, artist, title, album="", duration_ms=0):
        self.asked.append((track_id, artist, title, album, duration_ms))
        if self.fail:
            raise self.fail
        return self.by_id.get(track_id)


def make(words=None, **settings):
    world = World()
    spotify, store = FakeSpotify(world), FakeStore()
    words = words if words is not None else ["pineapple", "banana"]
    s = st.Settings(spotify_client_id="cid", **settings)
    svc = LyricService(s, lambda: list(words), world.now_sample, spotify=spotify, store=store,
                       clock=world.clock, fetch_async=False)
    return svc, world, spotify, store, s, words


def step(svc, world, spotify, pos, seconds=1.0, **kw):
    """Let `seconds` pass, then Spotify reports position `pos`."""
    world.advance(seconds)
    spotify.playing(pos, **kw)
    svc.poll_once()


def settle(svc, world, spotify, pos):
    """Start at `pos` and run until the 2 s widening after the start is over."""
    step(svc, world, spotify, pos, 0.0)
    for i in range(1, 4):
        step(svc, world, spotify, pos + i * 1000)
    return pos + 3000


def spans(svc, world):
    """All windows, as (start_ms, end_ms) of SONG time worked back from the mic clock."""
    out = []
    t = svc.track()
    for w in svc.windows(-FAR, FAR):
        out.append((round(t.position_ms + (w.start - world.now_sample()) / 48),
                    round(t.position_ms + (w.end - world.now_sample()) / 48)))
    return out


# ---------- the normal case ----------

def test_windows_land_on_the_mic_clock():
    svc, world, spotify, store, s, _ = make()
    pos = settle(svc, world, spotify, 5_000)                # now at 8.000 s
    assert pos == 8_000
    now = world.now_sample()
    ws = svc.windows(-FAR, FAR)
    assert ws == [
        DangerWindow(now + (19_500 - 8_000) * 48, now + (25_700 - 8_000) * 48,
                     "one pineapple", ("pineapple",)),
        DangerWindow(now + (59_500 - 8_000) * 48, now + (64_700 - 8_000) * 48,
                     "a banana here", ("banana",)),
    ]
    assert svc.status() == "Lyric mode: The Stand Ins - Fruit Song (2 risky lines)"
    assert store.asked == [("trk1", "The Stand Ins", "Fruit Song", "Fruit Album", 200_000)]


def test_windows_filter_by_span_and_stay_put_as_time_passes():
    svc, world, spotify, *_ = make()
    settle(svc, world, spotify, 5_000)
    now = world.now_sample()
    first_start = now + 11_500 * 48
    assert svc.windows(now, now + 48_000) == []                           # next second: clear
    assert len(svc.windows(now, first_start + 1)) == 1                    # just touches
    assert svc.windows(now, first_start) == []                            # `to` is exclusive
    assert len(svc.windows(first_start + 10, first_start + 20)) == 1      # inside a window
    world.advance(0.5)                                                    # no new poll: same samples
    assert svc.windows(-FAR, FAR)[0].start == first_start
    step(svc, world, spotify, 9_000, 0.5)                                 # on-time poll: same again
    assert svc.windows(-FAR, FAR)[0].start == first_start


def test_request_mid_time_is_the_anchor():
    svc, world, spotify, *_ = make()
    settle(svc, world, spotify, 5_000)
    # the answer says 9.000 s, but it was true 300 ms ago (middle of a slow request)
    step(svc, world, spotify, 9_000 - 300, lag=0.3)
    assert spans(svc, world)[0] == (19_500, 25_700)
    assert svc.track().position_ms == pytest.approx(9_000, abs=2)


def test_track_position_moves_between_polls():
    svc, world, spotify, *_ = make()
    step(svc, world, spotify, 30_000, 0.0)
    world.advance(0.4)
    t = svc.track()
    assert (t.track_id, t.is_playing) == ("trk1", True) and t.position_ms == pytest.approx(30_400, abs=2)
    assert t.label() == "The Stand Ins - Fruit Song @ 0:30"


# ---------- pause / resume ----------

def test_pause_gives_no_windows_and_resume_rebuilds():
    svc, world, spotify, *_ = make()
    settle(svc, world, spotify, 5_000)
    assert len(svc.windows(-FAR, FAR)) == 2
    step(svc, world, spotify, 9_000, playing=False)
    assert svc.windows(-FAR, FAR) == [] and svc.status() == "Spotify is paused"
    paused_at = svc.track()
    world.advance(30.0)                                     # a long pause: the position stands still
    spotify.playing(9_000, playing=False)
    svc.poll_once()
    assert svc.windows(-FAR, FAR) == []
    assert svc.track().position_ms == paused_at.position_ms == 9_000
    assert svc.track().is_playing is False
    step(svc, world, spotify, 9_050)                        # resumed
    assert svc.status().startswith("Lyric mode: The Stand Ins - Fruit Song")
    # rebuilt against the new moment, and widened by 500 ms each side for the first 2 s
    assert spans(svc, world) == [(19_000, 26_200), (59_000, 65_200)]


def test_after_resume_widening_ends_after_two_seconds():
    svc, world, spotify, *_ = make()
    settle(svc, world, spotify, 5_000)
    step(svc, world, spotify, 9_000, playing=False)
    step(svc, world, spotify, 9_000)                        # resume at t0
    assert spans(svc, world)[0] == (19_000, 26_200)
    step(svc, world, spotify, 10_000)                       # t0 + 1 s
    assert spans(svc, world)[0] == (19_000, 26_200)
    step(svc, world, spotify, 11_100, 1.1)                  # t0 + 2.1 s
    assert spans(svc, world)[0] == (19_500, 25_700)


# ---------- seek / skip ----------

def test_seek_rebuilds_at_once_and_widens_for_two_seconds():
    svc, world, spotify, *_ = make()
    settle(svc, world, spotify, 5_000)                      # at 8 s
    before = svc.windows(-FAR, FAR)
    step(svc, world, spotify, 50_000)                       # expected 9 s, got 50 s: a seek
    now = world.now_sample()
    after = svc.windows(-FAR, FAR)
    assert after != before
    # the banana line (59.5-64.7 s) is now 9 s away, plus the extra 500 ms each side
    assert (after[1].start, after[1].end) == (now + (59_000 - 50_000) * 48,
                                              now + (65_200 - 50_000) * 48)
    assert after[0].end < now                               # the first window is already behind us
    step(svc, world, spotify, 51_000)
    assert spans(svc, world)[1] == (59_000, 65_200)         # 1 s after the seek: still widened
    step(svc, world, spotify, 52_100, 1.1)
    assert spans(svc, world)[1] == (59_500, 64_700)         # 2.1 s after: back to normal


def test_small_drift_is_not_a_seek():
    svc, world, spotify, *_ = make()
    settle(svc, world, spotify, 5_000)
    step(svc, world, spotify, 9_000 + 600)                  # 600 ms off: under the 700 ms line
    assert spans(svc, world) == [(19_500, 25_700), (59_500, 64_700)]
    step(svc, world, spotify, 10_600 + 701)                 # 701 ms off: a jump
    assert spans(svc, world) == [(19_000, 26_200), (59_000, 65_200)]


def test_seek_backwards():
    svc, world, spotify, *_ = make()
    settle(svc, world, spotify, 40_000)
    step(svc, world, spotify, 15_000)
    now = world.now_sample()
    w = svc.windows(now, now + 10 * 48_000)
    assert len(w) == 1 and w[0].start == now + (19_000 - 15_000) * 48


# ---------- track change ----------

def test_track_change_swaps_lyrics_and_widens():
    svc, world, spotify, store, *_ = make()
    store.by_id["trk2"] = [(3_000, "banana banana"), (6_000, "clean")]
    settle(svc, world, spotify, 5_000)
    step(svc, world, spotify, 500, track_id="trk2", title="Second Song")
    assert [a[0] for a in store.asked] == ["trk1", "trk2"]
    assert svc.status() == "Lyric mode: The Stand Ins - Second Song (1 risky line)"
    assert svc.track().track_id == "trk2"
    assert spans(svc, world) == [(2_000, 7_200)]            # 2.5-6.7 s, widened right after the change
    ws = svc.windows(-FAR, FAR)
    assert ws[0].line == "banana banana" and ws[0].words == ("banana",)
    step(svc, world, spotify, 1_500, track_id="trk2", title="Second Song")
    step(svc, world, spotify, 2_600, 1.1, track_id="trk2", title="Second Song")
    assert spans(svc, world) == [(2_500, 6_700)]
    assert len(store.asked) == 2                            # not asked again on every poll


def test_old_windows_vanish_the_moment_the_track_changes_even_before_lyrics_arrive():
    svc, world, spotify, store, *_ = make()
    settle(svc, world, spotify, 5_000)
    release = threading.Event()
    real_get = store.get

    def slow_get(*a, **k):
        release.wait(5)
        return real_get(*a, **k)

    store.get = slow_get
    store.by_id["trk2"] = [(3_000, "pineapple"), (6_000, "x")]
    svc._fetch_async = True
    step(svc, world, spotify, 500, track_id="trk2", title="Second Song")
    assert svc.windows(-FAR, FAR) == []
    assert svc.status() == "Lyric mode: The Stand Ins - Second Song (looking up the lyrics)"
    release.set()
    deadline = time.monotonic() + 5
    while not svc.windows(-FAR, FAR) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(svc.windows(-FAR, FAR)) == 1                 # rebuilt as soon as the lyrics came in


def test_lyrics_that_arrive_for_an_old_track_are_dropped():
    svc, world, spotify, store, *_ = make()
    settle(svc, world, spotify, 5_000)
    stale = Playback(track=TrackInfo("old", "A", "Old Song", 0, True), anchor=world.t)
    store.by_id["old"] = [(1_000, "pineapple pineapple pineapple")]
    svc._fetch(stale)
    assert svc.status() == "Lyric mode: The Stand Ins - Fruit Song (2 risky lines)"
    assert len(svc.windows(-FAR, FAR)) == 2


# ---------- no lyrics, no risky lines, lookup trouble ----------

def test_song_without_timed_lyrics():
    svc, world, spotify, store, *_ = make()
    step(svc, world, spotify, 5_000, 0.0, track_id="nolyrics")
    assert svc.windows(-FAR, FAR) == []
    assert svc.status() == "Lyric mode: this song has no timed lyrics, listening only"
    for _ in range(3):
        step(svc, world, spotify, 6_000, track_id="nolyrics")
    assert len(store.asked) == 1 and svc.track().track_id == "nolyrics"


def test_song_with_lyrics_but_no_blocked_word():
    svc, world, spotify, store, *_ = make()
    store.by_id["clean"] = [(1_000, "all clean"), (5_000, "still clean")]
    step(svc, world, spotify, 500, 0.0, track_id="clean")
    assert svc.windows(-FAR, FAR) == []
    assert svc.status() == "Lyric mode: The Stand Ins - Fruit Song (no risky lines)"


def test_lyrics_site_down_is_retried_later():
    svc, world, spotify, store, *_ = make()
    store.fail = LyricsUnavailable(20.0)
    step(svc, world, spotify, 5_000, 0.0)
    assert svc.windows(-FAR, FAR) == []
    assert svc.status() == "Lyric mode: the lyrics site is not reachable, listening only"
    step(svc, world, spotify, 6_000)
    assert len(store.asked) == 1                            # not hammering the site
    store.fail = None
    step(svc, world, spotify, 26_500, 20.5)
    assert len(store.asked) == 2 and len(svc.windows(-FAR, FAR)) >= 1
    assert svc.status() == "Lyric mode: The Stand Ins - Fruit Song (2 risky lines)"


# ---------- word list and settings changes ----------

def test_word_list_change_rebuilds_windows():
    svc, world, spotify, store, s, words = make(words=["banana"])
    settle(svc, world, spotify, 5_000)
    assert spans(svc, world) == [(59_500, 64_700)]
    assert svc.status().endswith("(1 risky line)")
    words.append("pineapple")
    step(svc, world, spotify, 9_000)
    assert spans(svc, world) == [(19_500, 25_700), (59_500, 64_700)]
    words.clear()
    step(svc, world, spotify, 10_000)
    assert svc.windows(-FAR, FAR) == [] and svc.status().endswith("(no risky lines)")
    assert len(store.asked) == 1


def test_margins_come_from_settings_live():
    svc, world, spotify, store, s, _ = make(window_pre_ms=1000, window_post_ms=2000)
    settle(svc, world, spotify, 5_000)
    assert spans(svc, world)[0] == (19_000, 27_000)
    s.window_pre_ms, s.window_post_ms = 200, 300
    step(svc, world, spotify, 9_000)
    assert spans(svc, world)[0] == (19_800, 25_300)


def test_lyric_mode_off():
    svc, world, spotify, store, s, _ = make()
    settle(svc, world, spotify, 5_000)
    s.lyrics_enabled = False
    assert svc.windows(-FAR, FAR) == [] and svc.status() == "Lyric mode is off"   # at once
    polls = spotify.polls
    svc.poll_once()
    assert spotify.polls == polls and svc.track() is None    # Spotify is not asked while off
    s.lyrics_enabled = True
    step(svc, world, spotify, 12_000)
    assert len(svc.windows(-FAR, FAR)) == 2


# ---------- signed out, not playing, unreachable ----------

def test_not_signed_in_and_no_client_id():
    svc, world, spotify, *_ = make()
    spotify.signed_in = False
    svc.poll_once()
    assert svc.status() == "Not signed in to Spotify" and svc.logged_in() is False
    assert svc.windows(-FAR, FAR) == [] and svc.track() is None and spotify.polls == 0
    spotify.client_id = ""
    assert svc.status() == "Add your Spotify client ID in Settings to turn on lyric mode"
    spotify.client_id, spotify.signed_in = "cid", True
    step(svc, world, spotify, 5_000)
    assert svc.logged_in() and len(svc.windows(-FAR, FAR)) == 2


def test_login_and_logout_pass_through():
    svc, world, spotify, *_ = make()
    settle(svc, world, spotify, 5_000)
    svc.logout()
    assert not svc.logged_in() and svc.windows(-FAR, FAR) == [] and svc.track() is None
    assert svc.status() == "Not signed in to Spotify"
    svc.login()
    assert svc.logged_in()
    spotify.login_error = SpotifyLoginError("Spotify sign-in timed out. Try again")
    with pytest.raises(SpotifyLoginError):
        svc.login()


def test_nothing_playing():
    svc, world, spotify, *_ = make()
    settle(svc, world, spotify, 5_000)
    world.advance(1.0)
    spotify.next = Playback(anchor=world.t)                  # 204, podcast, ad, private session
    svc.poll_once()
    assert svc.windows(-FAR, FAR) == [] and svc.track() is None
    assert svc.status() == "Spotify is not playing"


def test_network_trouble_holds_then_drops_then_recovers():
    svc, world, spotify, store, *_ = make()
    settle(svc, world, spotify, 5_000)
    held = spotify.next
    start = svc.windows(-FAR, FAR)[0].start
    world.advance(1.0)
    spotify.next = Playback(track=held.track, album=held.album, primary_artist=held.primary_artist,
                            duration_ms=held.duration_ms, anchor=held.anchor, fresh=False)
    svc.poll_once()
    assert svc.windows(-FAR, FAR)[0].start == start          # carried on from the last good answer
    world.advance(5.0)
    spotify.next = Playback(problem="Spotify not reachable")
    svc.poll_once()
    # changed 2026-11-23 after the fail-safe review: the song is most likely still playing, so the
    # risky lines stay watched (from the last known position) instead of being dropped
    assert len(svc.windows(-FAR, FAR)) == 2 and svc.degraded()
    assert svc.status().startswith("Spotify is not answering")
    world.advance(30.0)                                      # still silent, still watching
    svc.poll_once()
    assert svc.degraded() and svc.windows(-FAR, FAR) != []
    step(svc, world, spotify, 46_000)
    assert not svc.degraded()                                # Spotify is back: normal again
    assert svc.status() != "Spotify not reachable"
    assert len(store.asked) == 1


def test_coasting_ends_at_the_end_of_the_song():
    svc, world, spotify, *_ = make()
    settle(svc, world, spotify, 5_000)
    held = spotify.next
    spotify.next = Playback(problem="Spotify not reachable")
    svc.poll_once()
    assert svc.degraded()
    world.advance(max(700.0, (held.duration_ms or 0) / 1000.0 + 1.0))
    svc.poll_once()
    assert not svc.degraded() and svc.windows(-FAR, FAR) == []
    assert svc.status() == "Spotify not reachable"


def test_windows_expire_if_polling_stops():
    svc, world, spotify, *_ = make()
    settle(svc, world, spotify, 5_000)
    world.advance(7.0)
    assert len(svc.windows(-FAR, FAR)) == 2
    world.advance(2.0)                                       # 9 s without any answer
    assert svc.windows(-FAR, FAR) == []


def test_a_crashing_words_callable_or_poll_does_not_kill_the_thread():
    world = World()
    spotify, store = FakeSpotify(world), FakeStore()
    calls = []

    def poll():
        calls.append(1)
        raise RuntimeError("boom")

    spotify.poll = poll
    svc = LyricService(st.Settings(spotify_client_id="cid"), lambda: ["pineapple"],
                       world.now_sample, spotify=spotify, store=store, clock=world.clock,
                       fetch_async=False)
    svc.start()
    svc.start()                                              # twice is harmless
    try:
        deadline = time.monotonic() + 5
        while len(calls) < 2 and time.monotonic() < deadline:
            svc._wake.set()
            time.sleep(0.01)
        assert len(calls) >= 2 and svc.windows(-FAR, FAR) == []
    finally:
        svc.stop()
    assert svc._thread is None


def test_real_thread_follows_a_song():
    svc, world, spotify, *_ = make()
    spotify.playing(5_000)
    svc.start()
    try:
        deadline = time.monotonic() + 5
        while not svc.windows(-FAR, FAR) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(svc.windows(-FAR, FAR)) == 2
    finally:
        svc.stop()
    assert svc.windows(-FAR, FAR) == []


def test_windows_is_safe_to_call_from_many_threads_while_polling():
    svc, world, spotify, *_ = make()
    settle(svc, world, spotify, 5_000)
    errors, stop = [], threading.Event()

    def reader():
        try:
            while not stop.is_set():
                for w in svc.windows(-FAR, FAR):
                    assert w.end > w.start
                svc.track()
                svc.status()
        except Exception as e:                               # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=reader) for _ in range(4)]
    for t in threads:
        t.start()
    pos = 9_000
    for i in range(200):
        step(svc, world, spotify, pos if i % 7 else pos + 30_000, 0.01)
        pos += 10
    stop.set()
    for t in threads:
        t.join(5)
    assert errors == []


def test_default_construction_needs_nothing_but_the_contract_arguments():
    svc = LyricService(st.Settings(), lambda: [], lambda: 0)
    assert svc.status() == "Add your Spotify client ID in Settings to turn on lyric mode"
    assert svc.logged_in() is False and svc.track() is None and svc.windows(0, 48_000) == []
    svc.poll_once()                                          # no client id: no request is made
    svc.stop()
