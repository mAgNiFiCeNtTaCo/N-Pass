"""Token file, Spotify client and LRCLIB client against fake HTTP.
No real network (the sign-in test only talks to its own 127.0.0.1 listener), no real token,
NPASS_DATA_DIR is a temp folder."""
from __future__ import annotations

import logging
import socket
import sys
import threading
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from npass import settings as st                                            # noqa: E402
from npass.lyrics import lrclib, spotify, tokens                            # noqa: E402
from npass.lyrics.lrclib import LrcLibClient, LyricStore, LyricsUnavailable  # noqa: E402
from npass.lyrics.spotify import Playback, SpotifyClient, SpotifyLoginError  # noqa: E402
from npass.lyrics.tokens import TokenSet                                    # noqa: E402

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows data protection")

ACCESS_1, REFRESH_1 = "ACCESS-SECRET-ONE-aaaa", "REFRESH-SECRET-ONE-bbbb"
ACCESS_2, REFRESH_2 = "ACCESS-SECRET-TWO-cccc", "REFRESH-SECRET-TWO-dddd"
AUTH_CODE = "AUTHCODE-SECRET-eeee"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NPASS_DATA_DIR", str(tmp_path))
    return tmp_path


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class FakeHttp:
    """Stands in for requests.Session: hands out queued answers and records every call."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        if not self.answers:
            raise AssertionError("unexpected request to " + url)
        a = self.answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a(method, url, kw) if callable(a) else a


def player(track_id="trk1", pos=30_000, playing=True, kind="track", private=False, **item):
    it = {"id": track_id, "type": "track", "name": "Fruit Song", "duration_ms": 200_000,
          "artists": [{"name": "The Stand Ins"}, {"name": "Guest"}],
          "album": {"name": "Fruit Album"}}
    it.update(item)
    return Resp(200, {"device": {"is_private_session": private}, "progress_ms": pos,
                      "is_playing": playing, "currently_playing_type": kind, "item": it})


def make_client(http, clock=None, wall=None, signed_in=True, expires_in=3600.0, client_id="cid123"):
    s = st.Settings(spotify_client_id=client_id)
    wall = wall or Clock(5_000_000.0)
    if signed_in:
        tokens.save_tokens(TokenSet(ACCESS_1, REFRESH_1, wall() + expires_in, spotify.SCOPE))
    return SpotifyClient(s, session=http, clock=clock or Clock(), wall=wall)


# ---------- token file ----------

def test_token_round_trip_and_file_is_not_plain_text():
    ts = TokenSet(ACCESS_1, REFRESH_1, 1234.5, "a b")
    tokens.save_tokens(ts)
    raw = st.token_path().read_bytes()
    assert ACCESS_1.encode() not in raw and REFRESH_1.encode() not in raw
    assert ACCESS_1.encode("utf-16-le") not in raw
    assert tokens.load_tokens() == ts
    assert not st.token_path().with_suffix(".tmp").exists()


def test_token_overwrite_and_delete():
    tokens.save_tokens(TokenSet(ACCESS_1, REFRESH_1, 1.0))
    tokens.save_tokens(TokenSet(ACCESS_2, REFRESH_2, 2.0))
    assert tokens.load_tokens().refresh_token == REFRESH_2
    tokens.delete_tokens()
    assert not st.token_path().exists() and tokens.load_tokens() is None
    tokens.delete_tokens()                                   # twice is fine


@pytest.mark.parametrize("junk", [b"", b"not a real session file", b"\x00" * 64,
                                  b"\x01\x00\x00\x00" + b"\xd0\x8c\x9d\xdf" * 20])
def test_corrupt_or_missing_token_file_means_logged_out(junk):
    assert tokens.load_tokens() is None                      # missing
    st.token_path().write_bytes(junk)
    assert tokens.load_tokens() is None                      # corrupt: no crash
    assert make_client(FakeHttp(), signed_in=False).logged_in() is False


def test_truncated_token_file_means_logged_out():
    tokens.save_tokens(TokenSet(ACCESS_1, REFRESH_1, 1.0))
    raw = st.token_path().read_bytes()
    st.token_path().write_bytes(raw[: len(raw) // 2])
    assert tokens.load_tokens() is None


def test_valid_blob_with_wrong_contents_means_logged_out():
    st.token_path().write_bytes(tokens._dpapi(b'{"hello": 1}', protect=True))
    assert tokens.load_tokens() is None
    st.token_path().write_bytes(tokens._dpapi(b"not json", protect=True))
    assert tokens.load_tokens() is None


def test_token_repr_shows_nothing():
    ts = TokenSet(ACCESS_1, REFRESH_1, 1.0)
    for text in (repr(ts), str(ts), f"{ts}", repr([ts]), repr({"t": ts})):
        assert ACCESS_1 not in text and REFRESH_1 not in text
    assert repr(ts) == "TokenSet(<hidden>)"


# ---------- currently playing ----------

def test_playing_track_is_parsed_and_request_is_shaped_right():
    clock = Clock()
    http = FakeHttp(player())
    c = make_client(http, clock)
    pb = c.poll()
    t = pb.track
    assert (t.track_id, t.artist, t.title, t.position_ms, t.is_playing) == (
        "trk1", "The Stand Ins, Guest", "Fruit Song", 30_000, True)
    assert (pb.album, pb.primary_artist, pb.duration_ms, pb.problem, pb.fresh) == (
        "Fruit Album", "The Stand Ins", 200_000, "", True)
    method, url, kw = http.calls[0]
    assert method == "GET" and url == "https://api.spotify.com/v1/me/player"
    assert kw["headers"]["Authorization"] == "Bearer " + ACCESS_1
    assert kw["timeout"] and kw["allow_redirects"] is False


def test_anchor_is_the_middle_of_the_request():
    clock = Clock(100.0)

    def slow(method, url, kw):
        clock.t += 0.4                                       # the request took 400 ms
        return player()

    pb = make_client(FakeHttp(slow), clock).poll()
    assert pb.anchor == pytest.approx(100.2)


@pytest.mark.parametrize("answer", [
    Resp(204),
    player(kind="episode"),
    player(kind="ad"),
    player(kind="unknown"),
    player(private=True),
    Resp(200, {"is_playing": True, "progress_ms": 5, "currently_playing_type": "track",
               "item": None}),
    Resp(200, {"is_playing": True, "progress_ms": None, "currently_playing_type": "track",
               "item": {"id": "x", "name": "y", "artists": []}}),
])
def test_nothing_to_follow_gives_none(answer):
    c = make_client(FakeHttp(answer))
    assert c.currently_playing() is None


def test_paused_track_is_returned_as_not_playing():
    t = make_client(FakeHttp(player(playing=False, pos=77_000))).currently_playing()
    assert t is not None and t.is_playing is False and t.position_ms == 77_000


def test_local_file_gets_a_stable_made_up_id():
    a = make_client(FakeHttp(player(track_id=None, uri="spotify:local:x"))).currently_playing()
    b = make_client(FakeHttp(player(track_id=None, uri="spotify:local:x"))).currently_playing()
    assert a.track_id == b.track_id and a.track_id.startswith("local-")


def test_no_client_id_and_not_signed_in_make_no_requests():
    http = FakeHttp()
    c = make_client(http, client_id="")
    assert c.status() == "Add your Spotify client ID in Settings to turn on lyric mode"
    assert c.currently_playing() is None and c.poll().problem == c.status()
    c2 = make_client(http, signed_in=False)
    tokens.delete_tokens()
    assert c2.status() == "Not signed in to Spotify"
    assert c2.poll().problem == "Not signed in to Spotify"
    assert http.calls == []


def test_401_refreshes_saves_the_rotated_token_and_retries():
    http = FakeHttp(Resp(401, {"error": {"status": 401}}),
                    Resp(200, {"access_token": ACCESS_2, "refresh_token": REFRESH_2,
                               "expires_in": 3600, "token_type": "Bearer"}),
                    player(pos=41_000))
    c = make_client(http)
    t = c.currently_playing()
    assert t is not None and t.position_ms == 41_000
    assert [m for m, _, _ in http.calls] == ["GET", "POST", "GET"]
    _, url, kw = http.calls[1]
    assert url == "https://accounts.spotify.com/api/token"
    assert kw["data"] == {"grant_type": "refresh_token", "refresh_token": REFRESH_1,
                          "client_id": "cid123"}
    assert kw["headers"]["Content-Type"] == "application/x-www-form-urlencoded"
    assert "Authorization" not in kw["headers"]              # PKCE: no client secret
    assert http.calls[2][2]["headers"]["Authorization"] == "Bearer " + ACCESS_2
    saved = tokens.load_tokens()
    assert (saved.access_token, saved.refresh_token) == (ACCESS_2, REFRESH_2)


def test_refresh_without_a_new_refresh_token_keeps_the_old_one():
    http = FakeHttp(Resp(401), Resp(200, {"access_token": ACCESS_2, "expires_in": 3600}), player())
    c = make_client(http)
    assert c.currently_playing() is not None
    saved = tokens.load_tokens()
    assert (saved.access_token, saved.refresh_token) == (ACCESS_2, REFRESH_1)


def test_token_is_refreshed_before_it_expires():
    wall = Clock(5_000_000.0)
    http = FakeHttp(player(),
                    Resp(200, {"access_token": ACCESS_2, "refresh_token": REFRESH_2,
                               "expires_in": 3600}),
                    player())
    c = make_client(http, wall=wall, expires_in=600.0)
    c.poll()
    assert [m for m, _, _ in http.calls] == ["GET"]          # 10 minutes left: no refresh yet
    wall.t += 560.0                                          # 40 s left: inside the safety margin
    c.poll()
    assert [m for m, _, _ in http.calls] == ["GET", "POST", "GET"]
    assert tokens.load_tokens().expires_at == pytest.approx(wall.t + 3600)


def test_refresh_refused_means_signed_out_and_file_removed():
    http = FakeHttp(Resp(401), Resp(400, {"error": "invalid_grant"}))
    c = make_client(http)
    pb = c.poll()
    assert pb.track is None and pb.problem == "Not signed in to Spotify"
    assert c.logged_in() is False and not st.token_path().exists()


def test_still_401_after_refresh_does_not_loop():
    http = FakeHttp(Resp(401), Resp(200, {"access_token": ACCESS_2, "expires_in": 3600}),
                    Resp(401))
    c = make_client(http)
    assert c.poll().problem == "Spotify not reachable"
    assert len(http.calls) == 3


def test_429_waits_for_retry_after():
    clock = Clock()
    http = FakeHttp(player(pos=10_000), Resp(429, headers={"Retry-After": "7"}),
                    player(pos=25_000))
    c = make_client(http, clock)
    assert c.poll().fresh is True
    clock.t += 1.0
    held = c.poll()                                          # 429: the last answer, marked stale
    assert held.fresh is False and held.track.position_ms == 10_000 and len(http.calls) == 2
    for _ in range(6):                                       # 6 more seconds: still told to wait
        clock.t += 1.0
        c.poll()
    assert len(http.calls) == 2                              # no request went out while waiting
    assert c.poll().problem == "Spotify not reachable"       # and the old answer is too old now
    clock.t += 1.1                                           # 8.1 s after the 429
    again = c.poll()
    assert len(http.calls) == 3 and again.fresh and again.track.position_ms == 25_000


def test_429_without_a_usable_header_still_backs_off():
    clock = Clock()
    http = FakeHttp(Resp(429, headers={"Retry-After": "soon"}), player())
    c = make_client(http, clock)
    assert c.poll().problem == "Spotify not reachable"
    clock.t += 2.0
    c.poll()
    assert len(http.calls) == 1
    clock.t += 4.0
    assert c.poll().track is not None


def test_network_down_keeps_last_state_briefly_then_says_not_reachable():
    clock = Clock()
    down = requests.ConnectionError("offline")
    http = FakeHttp(player(pos=10_000), down, down, requests.Timeout("slow"), player(pos=50_000))
    c = make_client(http, clock)
    first = c.poll()
    clock.t += 1.0
    held = c.poll()
    assert held.fresh is False and held.track == first.track and held.anchor == first.anchor
    clock.t += 3.0
    assert c.poll().fresh is False                           # 4 s: still holding
    clock.t += 2.0
    gone = c.poll()                                          # 6 s: give up
    assert gone.track is None and gone.problem == "Spotify not reachable"
    clock.t += 1.0
    back = c.poll()
    assert back.fresh and back.track.position_ms == 50_000


def test_server_error_and_bad_json_are_treated_as_unreachable():
    assert make_client(FakeHttp(Resp(503))).poll().problem == "Spotify not reachable"
    assert make_client(FakeHttp(Resp(200, None))).poll().problem == "Spotify not reachable"


def test_403_tells_the_user_what_to_check():
    pb = make_client(FakeHttp(Resp(403))).poll()
    assert pb.track is None and "developer app" in pb.problem


def test_only_spotify_hosts_with_timeouts():
    http = FakeHttp(Resp(401), Resp(200, {"access_token": ACCESS_2, "expires_in": 3600}), player())
    c = make_client(http)
    c.poll()
    for _, url, kw in http.calls:
        assert urlsplit(url).scheme == "https"
        assert urlsplit(url).hostname in ("accounts.spotify.com", "api.spotify.com")
        assert kw.get("timeout")
    with pytest.raises(RuntimeError):
        c._send("GET", "https://example.com/v1/me/player")
    with pytest.raises(RuntimeError):
        c._send("GET", "http://api.spotify.com/v1/me/player")
    assert len(http.calls) == 3


def test_logout_deletes_the_file():
    c = make_client(FakeHttp())
    assert c.logged_in() and st.token_path().exists()
    c.logout()
    assert not c.logged_in() and not st.token_path().exists()


# ---------- sign-in (PKCE) against our own loopback listener ----------

def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def local_get(url):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=5) as r:
            return r.status, r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


class Browser:
    """Plays the user's browser: reads the sign-in address, then calls back like Spotify would."""

    def __init__(self, port, visits):
        self.port, self.visits = port, visits
        self.url, self.results, self.thread = "", [], None

    def __call__(self, url):
        self.url = url
        self.thread = threading.Thread(target=self._go, daemon=True)
        self.thread.start()

    def _go(self):
        state = parse_qs(urlsplit(self.url).query)["state"][0]
        for path in self.visits:
            self.results.append(local_get(f"http://127.0.0.1:{self.port}" + path.format(state=state)))


def login_client(http, port):
    s = st.Settings(spotify_client_id="cid123", spotify_redirect_port=port)
    return SpotifyClient(s, session=http)


def test_login_full_flow():
    port = free_port()
    http = FakeHttp(Resp(200, {"access_token": ACCESS_1, "refresh_token": REFRESH_1,
                               "expires_in": 3600, "scope": spotify.SCOPE}))
    c = login_client(http, port)
    browser = Browser(port, ["/favicon.ico",
                             "/callback?code=WRONG&state=not-the-state",
                             "/callback?code=" + AUTH_CODE + "&state={state}"])
    c.login(open_browser=browser, timeout_s=10)
    browser.thread.join(5)

    u = urlsplit(browser.url)
    q = {k: v[0] for k, v in parse_qs(u.query).items()}
    assert (u.scheme, u.netloc, u.path) == ("https", "accounts.spotify.com", "/authorize")
    assert q["client_id"] == "cid123" and q["response_type"] == "code"
    assert q["redirect_uri"] == f"http://127.0.0.1:{port}/callback"
    assert q["scope"] == "user-read-currently-playing user-read-playback-state"
    assert q["code_challenge_method"] == "S256" and len(q["code_challenge"]) == 43
    assert len(q["state"]) >= 16 and "client_secret" not in q

    assert [r[0] for r in browser.results] == [404, 400, 200]
    assert "You can close this tab" in browser.results[2][1]
    assert AUTH_CODE not in browser.results[2][1]

    assert len(http.calls) == 1                              # the wrong-state visit exchanged nothing
    method, url, kw = http.calls[0]
    form = kw["data"]
    assert method == "POST" and url == "https://accounts.spotify.com/api/token"
    assert form["grant_type"] == "authorization_code" and form["code"] == AUTH_CODE
    assert form["redirect_uri"] == q["redirect_uri"] and form["client_id"] == "cid123"
    assert "client_secret" not in form and kw["timeout"]
    verifier = form["code_verifier"]
    assert 43 <= len(verifier) <= 128
    import base64
    import hashlib
    want = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    assert q["code_challenge"] == want

    assert c.logged_in() and tokens.load_tokens().refresh_token == REFRESH_1
    with pytest.raises(OSError):                             # one-shot: the listener is gone
        local_get(f"http://127.0.0.1:{port}/callback")


def test_login_listener_is_loopback_only():
    assert spotify._CallbackServer.allow_reuse_address is False
    port = free_port()
    srv = spotify._CallbackServer(("127.0.0.1", port), spotify._CallbackHandler)
    try:
        assert srv.server_address[0] == "127.0.0.1"
    finally:
        srv.server_close()


def test_login_cancelled_in_the_browser():
    port = free_port()
    http = FakeHttp()
    c = login_client(http, port)
    browser = Browser(port, ["/callback?error=access_denied&state={state}"])
    with pytest.raises(SpotifyLoginError, match="cancelled"):
        c.login(open_browser=browser, timeout_s=10)
    assert http.calls == [] and not c.logged_in()


def test_login_times_out():
    port = free_port()
    c = login_client(FakeHttp(), port)
    with pytest.raises(SpotifyLoginError, match="timed out"):
        c.login(open_browser=lambda url: None, timeout_s=0.6)
    assert not c.logged_in()


def test_login_needs_a_client_id_and_a_free_port():
    port = free_port()
    c = SpotifyClient(st.Settings(spotify_client_id="", spotify_redirect_port=port),
                      session=FakeHttp())
    with pytest.raises(SpotifyLoginError, match="client ID"):
        c.login(open_browser=lambda url: None, timeout_s=1)
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", port))
        busy.listen(1)
        c2 = login_client(FakeHttp(), port)
        with pytest.raises(SpotifyLoginError, match="in use"):
            c2.login(open_browser=lambda url: None, timeout_s=1)


def test_login_rejected_by_spotify():
    port = free_port()
    http = FakeHttp(Resp(400, {"error": "invalid_grant", "error_description": "bad code"}))
    c = login_client(http, port)
    with pytest.raises(SpotifyLoginError, match="HTTP 400") as e:
        c.login(open_browser=Browser(port, ["/callback?code=" + AUTH_CODE + "&state={state}"]),
                timeout_s=10)
    assert AUTH_CODE not in str(e.value) and not c.logged_in()


# ---------- nothing secret in logs, reprs or errors ----------

def test_no_secret_reaches_logs_reprs_or_errors(caplog, capsys):
    caplog.set_level(logging.DEBUG)
    port = free_port()
    seen = []
    http = FakeHttp(
        Resp(200, {"access_token": ACCESS_1, "refresh_token": REFRESH_1, "expires_in": 3600}),
        player(),                                            # normal poll
        Resp(401),                                           # expired early ...
        Resp(200, {"access_token": ACCESS_2, "refresh_token": REFRESH_2, "expires_in": 3600}),
        player(),                                            # ... refreshed and retried
        requests.ConnectionError("down " + ACCESS_2),        # a library error that quotes a token
        Resp(429, headers={"Retry-After": "3"}),
        Resp(503),
        Resp(401), Resp(400, {"error": "invalid_grant"}),    # session ended
    )
    clock = Clock()
    c = SpotifyClient(st.Settings(spotify_client_id="cid123", spotify_redirect_port=port),
                      session=http, clock=clock)
    # sign-in uses the real clock for its wait loop
    c._clock = __import__("time").monotonic
    c.login(open_browser=Browser(port, ["/callback?code=" + AUTH_CODE + "&state={state}"]),
            timeout_s=10)
    c._clock = clock
    verifier = http.calls[0][2]["data"]["code_verifier"]
    for _ in range(7):
        seen.append(c.poll())
        clock.t += 10.0
    assert not c.logged_in()

    # a failed sign-in too
    http.answers.append(Resp(400, {"error": "invalid_grant"}))
    c._clock = __import__("time").monotonic
    try:
        c.login(open_browser=Browser(port, ["/callback?code=" + AUTH_CODE + "&state={state}"]),
                timeout_s=10)
    except SpotifyLoginError as e:
        seen.append(e)
        seen.append(repr(e))
        seen.append(e.__cause__)
        seen.append(e.__context__ if not e.__suppress_context__ else None)

    out = capsys.readouterr()
    haystack = "\n".join([caplog.text, out.out, out.err, repr(c), str(c), repr(c.__dict__.keys()),
                          repr(TokenSet(ACCESS_1, REFRESH_1, 0.0)),
                          "\n".join(repr(x) + str(x) for x in seen)]
                         + [r.getMessage() + repr(r.args) for r in caplog.records])
    assert "Signed in to Spotify" in haystack                # the log was really captured
    for secret in (ACCESS_1, REFRESH_1, ACCESS_2, REFRESH_2, AUTH_CODE, verifier):
        assert secret not in haystack


# ---------- LRCLIB ----------

SYNCED = "[00:10.00] one pineapple\n[00:15.00] clean line\n"


def test_lrclib_get_sends_only_song_details_and_a_generic_user_agent():
    http = FakeHttp(Resp(200, {"syncedLyrics": SYNCED, "instrumental": False}))
    lines = LrcLibClient(http).fetch("The Stand Ins", "Fruit Song", "Fruit Album", 200_400)
    assert lines == [(10_000, "one pineapple"), (15_000, "clean line")]
    method, url, kw = http.calls[0]
    assert method == "GET" and url == "https://lrclib.net/api/get"
    assert kw["params"] == {"track_name": "Fruit Song", "artist_name": "The Stand Ins",
                            "album_name": "Fruit Album", "duration": 200}
    assert kw["headers"]["User-Agent"] == "N-Pass (local lyric lookup)"
    assert set(kw["headers"]) == {"User-Agent", "Accept"}
    assert kw["timeout"] and kw["allow_redirects"] is False


def test_lrclib_falls_back_to_search_and_picks_the_matching_length():
    http = FakeHttp(
        Resp(404, {"name": "TrackNotFound", "statusCode": 404}),
        Resp(200, [
            {"trackName": "Fruit Song", "artistName": "The Stand Ins", "duration": 260,
             "syncedLyrics": "[00:01.00] the long live cut"},
            {"trackName": "Other Song", "artistName": "The Stand Ins", "duration": 200,
             "syncedLyrics": "[00:01.00] some other song"},
            {"trackName": "Fruit Song", "artistName": "Somebody Else", "duration": 200,
             "syncedLyrics": "[00:01.00] a cover"},
            {"trackName": "Fruit Song", "artistName": "The Stand Ins", "duration": 201,
             "syncedLyrics": None, "plainLyrics": "words only"},
            {"trackName": "fruit song", "artistName": "The Stand Ins", "duration": 202,
             "syncedLyrics": "[00:02.00] the right one"},
        ]))
    assert LrcLibClient(http).fetch("The Stand Ins", "Fruit Song", "", 200_000) == [
        (2_000, "the right one")]
    assert http.calls[1][1] == "https://lrclib.net/api/search"
    assert http.calls[1][2]["params"] == {"track_name": "Fruit Song",
                                          "artist_name": "The Stand Ins"}
    assert "duration" in http.calls[0][2]["params"] and "album_name" not in http.calls[0][2]["params"]


@pytest.mark.parametrize("answers", [
    [Resp(404), Resp(200, [])],                                              # not found anywhere
    [Resp(200, {"instrumental": True, "syncedLyrics": None})],               # instrumental
    [Resp(200, {"instrumental": False, "syncedLyrics": None, "plainLyrics": "la la"}),
     Resp(200, [])],                                                         # only untimed words
    [Resp(200, {"instrumental": False, "syncedLyrics": "[00:05.00]\n[00:09.00]"}),
     Resp(404)],                                                             # timing but no words
])
def test_lrclib_no_timed_lyrics_gives_none(answers):
    assert LrcLibClient(FakeHttp(*answers)).fetch("A", "T", "Al", 100_000) is None


def test_lrclib_trouble_raises_unavailable_and_429_carries_the_wait():
    with pytest.raises(LyricsUnavailable):
        LrcLibClient(FakeHttp(requests.ConnectionError("x"))).fetch("A", "T")
    with pytest.raises(LyricsUnavailable):
        LrcLibClient(FakeHttp(Resp(500))).fetch("A", "T")
    with pytest.raises(LyricsUnavailable) as e:
        LrcLibClient(FakeHttp(Resp(429, headers={"Retry-After": "12"}))).fetch("A", "T")
    assert e.value.retry_after_s == 12
    with pytest.raises(RuntimeError):
        LrcLibClient(FakeHttp())._get("https://example.com/api/get", {})


def test_lrclib_skips_duration_outside_the_allowed_range():
    http = FakeHttp(Resp(200, {"syncedLyrics": SYNCED}))
    LrcLibClient(http).fetch("A", "T", "", 2 * 3600 * 1000)
    assert "duration" not in http.calls[0][2]["params"]


def test_store_caches_found_lyrics_on_disk(data_dir):
    http = FakeHttp(Resp(200, {"syncedLyrics": SYNCED}))
    store = LyricStore(LrcLibClient(http))
    lines = store.get("trk1", "A", "T", "Al", 100_000)
    assert lines and store.get("trk1", "A", "T", "Al", 100_000) == lines and len(http.calls) == 1
    path = data_dir / "lyrics_cache" / "trk1.json"
    assert path.exists() and LyricStore.cache_path("trk1") == path
    fresh = LyricStore(LrcLibClient(FakeHttp()))             # a new run: read from disk, no request
    assert fresh.get("trk1", "A", "T", "Al", 100_000) == lines


def test_store_asks_once_for_a_song_without_timed_lyrics(data_dir):
    http = FakeHttp(Resp(404), Resp(200, []))
    store = LyricStore(LrcLibClient(http))
    assert store.get("trk2", "A", "T") is None
    assert store.get("trk2", "A", "T") is None
    assert len(http.calls) == 2                              # one get + one search, never again
    assert not (data_dir / "lyrics_cache" / "trk2.json").exists()


def test_store_survives_a_corrupt_cache_file_and_odd_ids(data_dir):
    LyricStore.cache_path("trk3").write_text("{ not json", encoding="utf-8")
    store = LyricStore(LrcLibClient(FakeHttp(Resp(200, {"syncedLyrics": SYNCED}))))
    assert store.get("trk3", "A", "T")
    odd = LyricStore.cache_path("../../evil:name")
    assert odd.parent == data_dir / "lyrics_cache" and ".." not in odd.name and ":" not in odd.name


def test_store_does_not_remember_a_network_failure_and_waits_before_retrying():
    clock = Clock()
    http = FakeHttp(Resp(429, headers={"Retry-After": "20"}), Resp(200, {"syncedLyrics": SYNCED}))
    store = LyricStore(LrcLibClient(http), clock=clock)
    with pytest.raises(LyricsUnavailable):
        store.get("trk4", "A", "T")
    clock.t += 5.0
    with pytest.raises(LyricsUnavailable):
        store.get("trk4", "A", "T")
    assert len(http.calls) == 1                              # obeyed Retry-After: no second request
    clock.t += 16.0
    assert store.get("trk4", "A", "T")
