"""A pretend Core, so the window can be run and tested without audio, a GPU or Spotify.

    uv run python -m npass.ui.fake_core                      the demo window (moves on its own)
    uv run python -m npass.ui.fake_core --scene warn         start in another state: main | warn | settings
    uv run python -m npass.ui.fake_core --screenshot out/ui-demo.png    save a picture, then close

Nothing here listens to a microphone or goes online. The only "blocked word" is the stand-in
`pineapple`.
"""
from __future__ import annotations

import math
import sys
import threading
import time

from npass.types import Catch, CutOrder, CutReason, EngineStatus, Health, TrackInfo

DEMO_TRACK = TrackInfo("demo", "Demo Artist", "Fruit Salad", 0, True)


class FakeCore:
    """Implements the Core protocol from docs/CONTRACT.md with made-up numbers.

    animate=False (tests): nothing changes unless the test changes it.
    animate=True (demo): the meter moves, a catch shows up every few seconds and the warnings cycle.
    """

    def __init__(self, animate: bool = False, login_seconds: float = 2.0):
        self.animate = animate
        self.login_seconds = login_seconds
        self.started = time.time()
        self.lock = threading.Lock()
        self.calls: list[tuple] = []          # every setter call, for tests
        # state the window reads
        self.enabled = True
        self.health = Health.OK
        self.engine_name = "GPU (faster-whisper small.en)"
        self.pass_ms = 42.0
        self.lyrics_status = "Lyrics found. Watching 3 risky lines in this song."
        self.track: TrackInfo | None = DEMO_TRACK
        self.in_window = False
        self.cable_found = True
        self.warnings: list[str] = []
        self.engine_warnings: list[str] = []
        self.lookahead_ms = 450
        self.extra_delay_ms = 37.0            # what the sound drivers add on top of the look-ahead
        self.level_db = -22.0
        self.cutting = False
        self.removal_mode = "silence"
        self.engine_choice = "auto"
        self.lyrics_enabled = True
        self.spotify_logged_in = False
        self.spotify_client_id = ""
        self.hotkey = "ctrl+alt+n"
        self.input_names = ["Microphone (NVIDIA Broadcast)", "Microphone (USB Audio Device)",
                            "Headset Microphone (Wireless)"]
        self.input_name = self.input_names[0]
        self.word_list = ["pineapple", "banana"]
        self.catches: list[Catch] = []
        self.quit_called = False
        self._next_catch = self.started + 6.0
        self._cut_until = 0.0
        self._cut_id = 0

    # ------------------------------------------------------------- helpers for tests and the demo

    def add_catch(self, word: str = "pineapple", confidence: float = 0.71,
                  reason: CutReason = CutReason.ASR, track: TrackInfo | None = None,
                  late_ms: float = 0.0, wall_time: float | None = None) -> Catch:
        with self.lock:
            self._cut_id += 1
            order = CutOrder(self._cut_id, 0, 24_000, word=word, heard=word, confidence=confidence,
                             reason=reason)
            catch = Catch(time.time() if wall_time is None else wall_time, order, track, late_ms)
            self.catches.append(catch)
            return catch

    def _animate(self, now: float) -> None:
        t = now - self.started
        # a voice-like wobble between about -45 and -4 dB
        wobble = 0.5 + 0.5 * math.sin(t * 5.3) * math.sin(t * 1.7 + 1.0)
        self.level_db = -45.0 + 41.0 * max(0.0, wobble) ** 0.6
        self.pass_ms = 40.0 + 8.0 * math.sin(t * 0.9)
        if self.track is not None:
            self.track = TrackInfo(self.track.track_id, self.track.artist, self.track.title,
                                   int((t * 1000) % 200_000), True)
        if self.enabled and now >= self._next_catch:
            in_song = int(t) % 2 == 0
            self.add_catch("pineapple", 0.62 + 0.3 * abs(math.sin(t)),
                           CutReason.LYRIC_WINDOW if in_song else CutReason.ASR,
                           self.track if in_song else None,
                           late_ms=35.0 if int(t) % 5 == 0 else 0.0)
            self._next_catch = now + 4.0
            self._cut_until = now + 0.6
        self.cutting = now < self._cut_until
        self.in_window = self.cutting and self.track is not None
        # every 40 s: 20 s healthy, then 6 s falling behind, 6 s of a mic warning, 8 s cable missing
        phase = t % 40.0
        self.health = Health.LATE if 20.0 <= phase < 26.0 else Health.OK
        self.engine_warnings = (["The microphone was unplugged. Trying again every few seconds."]
                                if 26.0 <= phase < 32.0 else [])
        self.cable_found = not (32.0 <= phase < 40.0)

    # ------------------------------------------------------------- the Core protocol

    def snapshot(self) -> dict:
        if self.animate:
            self._animate(time.time())
        engine = EngineStatus(
            running=True, input_ok=True, output_ok=self.cable_found,
            input_name=self.input_name, output_name="CABLE Input (VB-Audio Virtual Cable)",
            lookahead_ms=float(self.lookahead_ms),
            measured_delay_ms=self.lookahead_ms + self.extra_delay_ms,
            level_db=self.level_db, cutting=self.cutting and self.enabled,
            warnings=list(self.engine_warnings))
        return {
            "enabled": self.enabled, "engine": engine, "health": self.health,
            "engine_name": self.engine_name, "pass_ms": self.pass_ms,
            "caught": len(self.catches), "lyrics_status": self.lyrics_status,
            "track": self.track, "in_window": self.in_window, "cable_found": self.cable_found,
            "warnings": list(self.warnings), "lookahead_ms": self.lookahead_ms,
            "removal_mode": self.removal_mode, "lyrics_enabled": self.lyrics_enabled,
            "spotify_logged_in": self.spotify_logged_in,
            "spotify_client_id": self.spotify_client_id, "hotkey": self.hotkey,
        }

    def set_enabled(self, on: bool) -> None:
        self.calls.append(("set_enabled", on))
        self.enabled = bool(on)

    def set_lookahead(self, ms: int) -> None:
        self.calls.append(("set_lookahead", ms))
        self.lookahead_ms = int(ms)

    def set_removal(self, mode: str) -> None:
        self.calls.append(("set_removal", mode))
        self.removal_mode = mode

    def set_engine(self, engine: str) -> None:
        self.calls.append(("set_engine", engine))
        self.engine_choice = engine
        self.engine_name = "CPU (Vosk)" if engine == "cpu" else "GPU (faster-whisper small.en)"

    def inputs(self) -> list[str]:
        return list(self.input_names)

    def set_input(self, name: str) -> None:
        self.calls.append(("set_input", name))
        self.input_name = name

    def words(self) -> list[str]:
        return list(self.word_list)

    def set_words(self, words: list[str]) -> None:
        self.calls.append(("set_words", list(words)))
        self.word_list = list(words)

    def set_lyrics_enabled(self, on: bool) -> None:
        self.calls.append(("set_lyrics_enabled", on))
        self.lyrics_enabled = bool(on)
        self.lyrics_status = ("Lyrics found. Watching 3 risky lines in this song." if on
                              else "Lyric help is off.")

    def spotify_login(self) -> None:
        """Blocks like the real one does while the browser is open."""
        self.calls.append(("spotify_login",))
        time.sleep(self.login_seconds)
        self.spotify_logged_in = True

    def spotify_logout(self) -> None:
        self.calls.append(("spotify_logout",))
        self.spotify_logged_in = False

    def set_spotify_client_id(self, cid: str) -> None:
        self.calls.append(("set_spotify_client_id", cid))
        self.spotify_client_id = cid

    def recent_catches(self) -> list[Catch]:
        with self.lock:
            return list(self.catches)

    def quit(self) -> None:
        self.calls.append(("quit",))
        self.quit_called = True


def demo_core(scene: str = "main", animate: bool = True) -> FakeCore:
    """A FakeCore that already has something to show."""
    core = FakeCore(animate=animate)
    now = time.time()
    core.add_catch("pineapple", 0.71, CutReason.LYRIC_WINDOW,
                   TrackInfo("demo", "Demo Artist", "Fruit Salad", 102_000, True), wall_time=now - 95)
    core.add_catch("banana", 0.83, CutReason.ASR, None, wall_time=now - 61)
    core.add_catch("pineapple", 0.58, CutReason.LYRIC_WINDOW,
                   TrackInfo("demo", "Demo Artist", "Fruit Salad", 151_000, True), late_ms=35.0,
                   wall_time=now - 40)
    core.add_catch("", 0.0, CutReason.FAILSAFE,
                   TrackInfo("demo", "A Band With A Rather Long Name", "A Song Title That Goes On", 12_000, True),
                   wall_time=now - 12)
    if scene == "warn":
        core.animate = False
        core.health = Health.DOWN
        core.cable_found = False
        core.cutting = True
        core.extra_delay_ms = 290.0
        core.warnings = ["Spotify sign-in ran out. Sign in again in Settings."]
    return core


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    scene = "main"
    shot = ""
    if "--scene" in args:
        scene = args[args.index("--scene") + 1]
    if "--screenshot" in args:
        shot = args[args.index("--screenshot") + 1]
    from npass.ui.window import run_ui
    core = demo_core(scene, animate=True)
    run_ui(core, start_tab="settings" if scene == "settings" else "main",
           screenshot_to=shot or None, use_hotkey=not shot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
