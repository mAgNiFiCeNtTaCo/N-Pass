"""Spotify: sign-in (Authorization Code with PKCE, no client secret) and "what is playing now".

Checked against Spotify's developer docs on 2026-11-23:
- authorize: GET https://accounts.spotify.com/authorize (client_id, response_type=code, redirect_uri,
  state, scope, code_challenge_method=S256, code_challenge)
- token:     POST https://accounts.spotify.com/api/token, form-encoded; PKCE sends client_id and
  code_verifier (exchange) or client_id and refresh_token (refresh). A refresh may or may not return
  a new refresh token: keep the old one when it does not.
- redirect:  http is allowed only for the loopback IP literal (127.0.0.1), "localhost" is refused.
- playback:  GET https://api.spotify.com/v1/me/player (200, 204, 401, 403, 429; Retry-After in seconds)

Never printed, logged or put in an exception: access token, refresh token, auth code, PKCE verifier.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import http.server
import logging
import secrets
import threading
import time
import webbrowser
from dataclasses import dataclass, replace
from typing import Callable
from urllib.parse import parse_qs, urlencode, urlsplit

import requests

from npass.types import TrackInfo

from . import tokens as _tokens
from .tokens import TokenSet

log = logging.getLogger("npass.lyrics")

AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
PLAYER_URL = "https://api.spotify.com/v1/me/player"
ALLOWED_HOSTS = ("accounts.spotify.com", "api.spotify.com")
SCOPE = "user-read-currently-playing user-read-playback-state"

HTTP_TIMEOUT_S = (3.05, 5.0)     # connect, read
LOGIN_TIMEOUT_S = 120.0
HOLD_S = 5.0                     # network trouble: keep the last answer this long
REFRESH_MARGIN_S = 60.0          # refresh this long before the access token runs out

MSG_NO_CLIENT_ID = "Add your Spotify client ID in Settings to turn on lyric mode"
MSG_NOT_SIGNED_IN = "Not signed in to Spotify"
MSG_UNREACHABLE = "Spotify not reachable"
MSG_REFUSED = ("Spotify refused the request. Check that your Spotify account is added "
               "to your developer app")

_PAGE = ("<!doctype html><html><head><meta charset='utf-8'><title>N-Pass</title></head>"
         "<body style='font-family:sans-serif;margin:3em'><p>{}</p></body></html>")


class SpotifyLoginError(Exception):
    """Sign-in did not finish. The message is plain English and safe to show."""


class _Unavailable(Exception):
    """Spotify could not be asked right now (network, server error, rate limit)."""


@dataclass(frozen=True)
class Playback:
    """One answer from Spotify. `anchor` is the time.monotonic() middle of the request: the
    moment `track.position_ms` was true."""
    track: TrackInfo | None = None
    album: str = ""
    primary_artist: str = ""
    duration_ms: int = 0
    anchor: float = 0.0
    problem: str = ""        # plain English when something is wrong, else ""
    fresh: bool = True       # False: the last good answer, repeated during network trouble


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)                      # 64 characters, allowed set
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return verifier, base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


class _CallbackServer(http.server.HTTPServer):
    allow_reuse_address = False      # on Windows "reuse" would let another program share the port
    outcome: tuple[str, str] | None = None     # ("code", value) or ("error", plain message)
    expected_state = ""
    expected_host = ""


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    server_version = "NPass"
    sys_version = ""
    timeout = 5      # a connection that sends nothing must not block the real callback

    def log_message(self, *args) -> None:     # the request line holds the auth code: log nothing
        return

    def _page(self, status: int, text: str) -> None:
        body = _PAGE.format(text).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        srv: _CallbackServer = self.server            # type: ignore[assignment]
        parts = urlsplit(self.path)
        if parts.path != "/callback" or self.headers.get("Host", "") != srv.expected_host:
            self._page(404, "Nothing here.")
            return
        q = parse_qs(parts.query)
        state = (q.get("state") or [""])[0]
        if not state or not hmac.compare_digest(state.encode(), srv.expected_state.encode()):
            self._page(400, "This sign-in link does not belong to the sign-in N-Pass started. "
                            "Go back to N-Pass and try again.")
            return
        code = (q.get("code") or [""])[0]
        if code:
            srv.outcome = ("code", code)
            self._page(200, "N-Pass is signed in to Spotify. You can close this tab.")
        else:
            srv.outcome = ("error", "Spotify sign-in was cancelled")
            self._page(200, "Sign-in was cancelled. You can close this tab.")


class SpotifyClient:
    """Thread-safe. `settings` is read live, so a client id typed in later is picked up."""

    def __init__(self, settings, session=None, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time):
        self._settings = settings
        self._http = session if session is not None else requests.Session()
        self._clock = clock
        self._wall = wall
        self._lock = threading.RLock()        # token/state fields: held only for moments
        self._net_lock = threading.Lock()     # one conversation with Spotify at a time
        self._gen = 0                         # bumped by logout/login so a late answer is dropped
        self._login_lock = threading.Lock()
        self._tokens: TokenSet | None = None
        self._tokens_loaded = False
        self._last_ok: Playback | None = None
        self._retry_at = 0.0

    def __repr__(self) -> str:
        return "SpotifyClient()"

    # ---- session ----

    def client_id(self) -> str:
        return str(getattr(self._settings, "spotify_client_id", "") or "").strip()

    def redirect_uri(self) -> str:
        return f"http://127.0.0.1:{int(self._settings.spotify_redirect_port)}/callback"

    def _current_tokens(self) -> TokenSet | None:
        with self._lock:
            if not self._tokens_loaded:
                self._tokens = _tokens.load_tokens()
                self._tokens_loaded = True
            return self._tokens

    def logged_in(self) -> bool:
        return self._current_tokens() is not None

    def logout(self) -> None:
        with self._lock:
            self._tokens = None
            self._tokens_loaded = True
            self._last_ok = None
            self._retry_at = 0.0
            self._gen += 1
            _tokens.delete_tokens()
        log.info("Signed out of Spotify")

    def status(self) -> str:
        """'' when ready, else the plain-English reason lyric mode cannot ask Spotify."""
        if not self.client_id():
            return MSG_NO_CLIENT_ID
        if not self.logged_in():
            return MSG_NOT_SIGNED_IN
        return ""

    # ---- HTTP (two hosts only, always with a timeout, never following redirects) ----

    def _send(self, method: str, url: str, **kw):
        if urlsplit(url).hostname not in ALLOWED_HOSTS or not url.startswith("https://"):
            raise RuntimeError("blocked a request to a host that is not Spotify")
        try:
            return self._http.request(method, url, timeout=HTTP_TIMEOUT_S,
                                      allow_redirects=False, **kw)
        except requests.RequestException as e:
            log.debug("Spotify request failed (%s)", type(e).__name__)
            raise _Unavailable() from None

    def _token_request(self, form: dict):
        r = self._send("POST", TOKEN_URL, data=form,
                       headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            body = r.json() if r.status_code in (200, 400, 401) else {}
        except ValueError:
            body = {}
        return r.status_code, (body if isinstance(body, dict) else {}), r

    def _store(self, body: dict, old_refresh: str = "", gen: int | None = None) -> TokenSet | None:
        access = body.get("access_token")
        refresh = body.get("refresh_token") or old_refresh      # not always sent again: keep ours
        if not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh:
            raise _Unavailable()
        try:
            life = float(body.get("expires_in", 3600))
        except (TypeError, ValueError):
            life = 3600.0
        ts = TokenSet(access, refresh, self._wall() + life, str(body.get("scope", "")))
        with self._lock:
            if gen is not None and gen != self._gen:      # signed out while we were asking
                return None
            self._tokens, self._tokens_loaded = ts, True
            try:
                _tokens.save_tokens(ts)
            except OSError:
                log.warning("Could not save the Spotify session file; signed in until N-Pass closes")
        return ts

    def _refresh(self, old: TokenSet) -> TokenSet | None:
        """A fresh access token; None when Spotify says the session is over (signed out)."""
        gen = self._gen
        status, body, r = self._token_request({"grant_type": "refresh_token",
                                               "refresh_token": old.refresh_token,
                                               "client_id": self.client_id()})
        if status == 200:
            log.debug("Spotify access renewed")
            return self._store(body, old.refresh_token, gen)
        if status in (400, 401) and body.get("error") in ("invalid_grant", "invalid_client"):
            log.info("Spotify ended the session; sign in again")
            with self._lock:
                if gen == self._gen:
                    self.logout()
            return None
        self._note_rate_limit(status, r)
        raise _Unavailable()

    def _note_rate_limit(self, status: int, response) -> None:
        if status != 429:
            return
        wait = 5.0
        if response is not None:
            try:
                wait = float(response.headers.get("Retry-After", "5"))
            except (TypeError, ValueError):
                wait = 5.0
        wait = max(1.0, min(wait, 3600.0))
        self._retry_at = self._clock() + wait
        log.info("Spotify asked N-Pass to slow down for %d s", int(wait))

    # ---- what is playing ----

    def currently_playing(self) -> TrackInfo | None:
        return self.poll().track

    def poll(self) -> Playback:
        with self._net_lock:
            if not self.client_id():
                return Playback(problem=MSG_NO_CLIENT_ID)
            try:
                pb = self._poll_once()
            except _Unavailable:
                last = self._last_ok
                if last is not None and self._clock() - last.anchor <= HOLD_S:
                    return replace(last, fresh=False)
                self._last_ok = None
                return Playback(problem=MSG_UNREACHABLE)
            self._last_ok = pb if not pb.problem else None
            return pb

    def _poll_once(self) -> Playback:
        if self._clock() < self._retry_at:           # still inside a 429 Retry-After
            raise _Unavailable()
        tok = self._current_tokens()
        if tok is None:
            return Playback(problem=MSG_NOT_SIGNED_IN)
        if tok.expired(self._wall(), REFRESH_MARGIN_S):
            tok = self._refresh(tok)
            if tok is None:
                return Playback(problem=MSG_NOT_SIGNED_IN)
        r, anchor = self._get_player(tok)
        if r.status_code == 401:                     # expired early or revoked: renew, ask once more
            tok = self._refresh(tok)
            if tok is None:
                return Playback(problem=MSG_NOT_SIGNED_IN)
            r, anchor = self._get_player(tok)
            if r.status_code == 401:
                raise _Unavailable()
        if r.status_code == 204:
            return Playback(anchor=anchor)
        if r.status_code == 403:
            return Playback(anchor=anchor, problem=MSG_REFUSED)
        if r.status_code == 429:
            self._note_rate_limit(429, r)
            raise _Unavailable()
        if r.status_code != 200:
            log.debug("Spotify answered HTTP %d", r.status_code)
            raise _Unavailable()
        try:
            data = r.json()
        except ValueError:
            raise _Unavailable() from None
        return self._parse(data, anchor)

    def _get_player(self, tok: TokenSet):
        t0 = self._clock()
        r = self._send("GET", PLAYER_URL, headers={"Authorization": "Bearer " + tok.access_token})
        t1 = self._clock()
        return r, (t0 + t1) / 2.0

    @staticmethod
    def _parse(data, anchor: float) -> Playback:
        if not isinstance(data, dict):
            return Playback(anchor=anchor)
        device = data.get("device") or {}
        item = data.get("item")
        if (device.get("is_private_session") or data.get("currently_playing_type") != "track"
                or not isinstance(item, dict) or item.get("type", "track") != "track"):
            return Playback(anchor=anchor)             # private session, podcast, ad, nothing
        title = str(item.get("name") or "").strip()
        names = [str(a.get("name") or "").strip() for a in (item.get("artists") or [])
                 if isinstance(a, dict)]
        names = [n for n in names if n]
        progress = data.get("progress_ms")
        if not title or not isinstance(progress, (int, float)):
            return Playback(anchor=anchor)
        track_id = item.get("id")
        if not isinstance(track_id, str) or not track_id:      # a local file has no Spotify id
            seed = str(item.get("uri") or "") + "|" + title + "|" + ",".join(names)
            track_id = "local-" + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:20]
        try:
            duration = int(item.get("duration_ms") or 0)
        except (TypeError, ValueError):
            duration = 0
        album = item.get("album") if isinstance(item.get("album"), dict) else {}
        track = TrackInfo(track_id=track_id, artist=", ".join(names), title=title,
                          position_ms=int(progress), is_playing=bool(data.get("is_playing")))
        return Playback(track=track, album=str(album.get("name") or "").strip(),
                        primary_artist=names[0] if names else "", duration_ms=duration,
                        anchor=anchor)

    # ---- sign-in ----

    def login(self, open_browser: Callable[[str], object] = webbrowser.open,
              timeout_s: float = LOGIN_TIMEOUT_S) -> None:
        """Open the browser, wait for Spotify to send the user back, save the session.
        Blocks until done or `timeout_s`. Raises SpotifyLoginError with a plain message."""
        client_id = self.client_id()
        if not client_id:
            raise SpotifyLoginError(MSG_NO_CLIENT_ID)
        if not self._login_lock.acquire(blocking=False):
            raise SpotifyLoginError("A Spotify sign-in is already open in your browser")
        try:
            verifier, challenge = _pkce_pair()
            state = secrets.token_urlsafe(24)
            port = int(self._settings.spotify_redirect_port)
            try:
                server = _CallbackServer(("127.0.0.1", port), _CallbackHandler)
            except OSError:
                raise SpotifyLoginError(
                    f"Port {port} is in use by another program. Close it, or pick another "
                    "port in Settings and in your Spotify app") from None
            try:
                server.expected_state = state
                server.expected_host = f"127.0.0.1:{port}"
                server.timeout = 0.25
                url = AUTH_URL + "?" + urlencode({
                    "client_id": client_id, "response_type": "code",
                    "redirect_uri": self.redirect_uri(), "state": state, "scope": SCOPE,
                    "code_challenge_method": "S256", "code_challenge": challenge})
                log.info("Opening the Spotify sign-in page")
                try:
                    open_browser(url)
                except Exception:
                    raise SpotifyLoginError("Could not open your web browser") from None
                deadline = self._clock() + timeout_s
                while server.outcome is None and self._clock() < deadline:
                    server.handle_request()
                outcome = server.outcome
            finally:
                server.server_close()
            if outcome is None:
                raise SpotifyLoginError("Spotify sign-in timed out. Try again")
            kind, value = outcome
            if kind != "code":
                raise SpotifyLoginError(value)
            try:
                status, body, _r = self._token_request({
                    "grant_type": "authorization_code", "code": value,
                    "redirect_uri": self.redirect_uri(), "client_id": client_id,
                    "code_verifier": verifier})
                if status != 200:
                    raise SpotifyLoginError(f"Spotify did not accept the sign-in (HTTP {status})")
                with self._lock:
                    self._gen += 1
                self._store(body)
            except _Unavailable:
                raise SpotifyLoginError("Spotify could not be reached to finish the sign-in") from None
            with self._lock:
                self._last_ok = None
                self._retry_at = 0.0
            log.info("Signed in to Spotify")
        finally:
            self._login_lock.release()
