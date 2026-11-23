"""N-Pass: wires the pieces together and is the `Core` the window talks to (docs/CONTRACT.md).

Run:  uv run python -m npass.app        (or N-Pass.cmd, which starts it without a console)
"""
from __future__ import annotations

import ctypes
import logging
import logging.handlers
import os
import queue
import sys
import threading
import time
from collections import deque

from . import settings as cfg
from .types import Catch, CutOrder, CutReason, DangerWindow, EngineStatus, Health

log = logging.getLogger("npass")

GUARD_TICK_S = 0.010


class AppCore:
    """Owns the audio engine, the recogniser, the lyric helper and the safety guard."""

    def __init__(self) -> None:
        self.settings = cfg.load_settings()
        self._words = cfg.load_words()
        self._lock = threading.RLock()
        self._catches: deque[Catch] = deque(maxlen=200)
        self._caught = 0
        self._late: dict[int, float] = {}
        self._logged_failsafe: set[int] = set()
        self._health = Health.STARTING
        self._in_window = False
        self._login_problem = ""
        self._core_problem = ""
        self._delay_note = ""
        self._over_since = 0.0
        self._windows_failed_at = 0.0
        self._held: dict[int, CutOrder] = {}       # cuts ordered while the filter was off
        self._records: queue.Queue = queue.Queue()
        threading.Thread(target=self._record_loop, name="npass-catchlog", daemon=True).start()
        self._stop = threading.Event()
        self._guard_thread: threading.Thread | None = None
        self._quit_cb = None
        self.engine = None
        self.detector = None
        self.lyrics = None
        self.guard = None
        self._build()

    # ------------------------------------------------------------------ build / teardown
    def _build(self) -> None:
        from .audio import AudioEngine
        from .detect import Detector, SafetyGuard
        from .lyrics import LyricService

        s = self.settings
        self.lyrics = LyricService(s, self.words, self._captured)
        self.detector = Detector(s, self.words, self._windows, self._submit, self._on_catch)
        self.engine = AudioEngine(s, tap=self._tap)
        self.guard = SafetyGuard(self._submit, ahead_ms=250, bridge_ms=300)
        # optional "mute the whole risky line" mode: a second guard that treats every risky line
        # as unchecked. Its cut ids start elsewhere so they never collide with the first guard's.
        self.line_guard = SafetyGuard(self._submit, ahead_ms=250, bridge_ms=300)
        self.line_guard._next_id = 2_000_000_000
        self._health = Health.STARTING
        # cut ids start again with every new detector/guard: forget what the old ones logged
        self._late.clear()
        self._logged_failsafe.clear()
        self._held.clear()

    def start(self) -> None:
        self._stop.clear()
        self.engine.start()
        self.detector.start()
        if self.settings.lyrics_enabled:
            self.lyrics.start()
        self._guard_thread = threading.Thread(target=self._guard_loop, name="npass-guard", daemon=True)
        self._guard_thread.start()

    def stop(self) -> None:
        """Audio goes quiet FIRST, while the guard is still watching; only then the rest.
        (The other way round, live mic audio would reach the cable with nobody checking it.)"""
        with self._lock:
            try:
                if self.engine is not None:
                    self.engine.stop()
            except Exception:
                log.exception("stopping the audio engine failed")
            self._stop.set()
            t = self._guard_thread
            if t and t.is_alive() and t is not threading.current_thread():
                t.join(timeout=2)
            for part in (self.detector, self.lyrics):
                try:
                    if part is not None:
                        part.stop()
                except Exception:
                    log.exception("stopping %s failed", type(part).__name__)

    def _restart(self) -> None:
        """Rebuild the whole chain (new engine choice). The sample clock restarts at 0."""
        with self._lock:
            try:
                self.stop()
                self._build()
                self.start()
                self._core_problem = ""
            except Exception:
                log.exception("restart failed")
                self._health = Health.DOWN
                self._core_problem = ("N-Pass could not restart its audio. Close N-Pass and "
                                      "start it again.")

    # ------------------------------------------------------------------ wiring
    def _captured(self) -> int:
        eng = self.engine
        return eng.processor.captured if eng is not None else 0

    def _windows(self, a: int, b: int) -> list[DangerWindow]:
        if not self.settings.lyrics_enabled or self.lyrics is None:
            return []
        try:
            return self.lyrics.windows(a, b)
        except Exception:
            if time.monotonic() - self._windows_failed_at > 30:
                log.exception("lyric windows failed")
            self._windows_failed_at = time.monotonic()
            return []

    def _tap(self, block, start_sample: int) -> None:
        det = self.detector
        if det is not None:
            det.feed(block, start_sample)

    def _submit(self, order: CutOrder) -> None:
        """Detector / guard -> cutter. While the filter is off the order is kept, not applied:
        switching on re-sends the ones that have not played yet (a word said just before the
        hotkey press is still in the buffer)."""
        if not self.settings.enabled:
            self._held[order.cut_id] = order
            if len(self._held) > 200:
                playhead = self.engine.processor.playhead
                self._held = {i: o for i, o in self._held.items() if o.end > playhead}
            return
        late = float(self.engine.submit_cut(order) or 0.0)
        if order.reason == CutReason.FAILSAFE:
            # the guard has no on_catch of its own: log each failsafe mute once
            if order.cut_id not in self._logged_failsafe:
                self._logged_failsafe.add(order.cut_id)
                if len(self._logged_failsafe) > 5000:
                    self._logged_failsafe.clear()
                self._record(order, late)
            return
        if late > 0:
            self._late[order.cut_id] = max(late, self._late.get(order.cut_id, 0.0))
            if len(self._late) > 2000:
                self._late.clear()

    def _on_catch(self, order: CutOrder) -> None:
        if not self.settings.enabled:
            return
        self._record(order, self._late.get(order.cut_id, 0.0))

    def _record(self, order: CutOrder, late_ms: float) -> None:
        """Called on the guard / detector threads: only queues. The file write happens elsewhere,
        so a slow disk can never delay a mute."""
        self._records.put((order, late_ms, time.time()))

    def _record_loop(self) -> None:
        while True:
            order, late_ms, when = self._records.get()
            try:
                self._write_record(order, late_ms, when)
            except Exception:
                log.exception("recording a catch failed")

    def _write_record(self, order: CutOrder, late_ms: float, when: float) -> None:
        track = None
        try:
            track = self.lyrics.track() if self.settings.lyrics_enabled else None
        except Exception:
            pass
        catch = Catch(wall_time=when, order=order, track=track, late_ms=late_ms)
        with self._lock:
            self._catches.append(catch)
            self._caught += 1
        try:
            from .catchlog import log_catch
            log_catch(catch)
        except Exception:
            log.exception("writing the catch log failed")

    def _guard_loop(self) -> None:
        ticks = 0
        last_logged = 0.0
        while not self._stop.wait(GUARD_TICK_S):
            try:
                proc = self.engine.processor
                playhead, captured = proc.playhead, proc.captured
                wins = self._windows(playhead, captured)
                self._in_window = any(w.start <= captured < w.end for w in wins)
                ticks += 1
                if ticks % 100 == 0:
                    self._keep_under_cap()
                if not self.settings.enabled:
                    self._health = self.detector.health()
                    continue
                self._health = self.guard.tick(playhead, self.detector.verified_upto(),
                                               self.detector.health(), wins, captured=captured)
                if self.settings.mute_risky_lines and wins:
                    self.line_guard.tick(playhead, playhead, Health.OK, wins, captured=captured)
            except Exception:
                # the guard itself must never die quietly: say so and keep ticking at full pace
                self._health = Health.DOWN
                if time.monotonic() - last_logged > 5:
                    last_logged = time.monotonic()
                    log.exception("safety guard tick failed")

    def _keep_under_cap(self) -> None:
        """The 700 ms limit is a real cap: if the measured delay stays above it, the look-ahead is
        lowered by the excess (the drivers' own delay cannot be changed from here)."""
        delay = self.engine.status().measured_delay_ms
        if delay <= cfg.MAX_TOTAL_DELAY_MS:
            self._over_since = 0.0
            return
        now = time.monotonic()
        if not self._over_since:
            self._over_since = now
            return
        if now - self._over_since < 3.0:
            return
        s = self.settings
        new = max(cfg.LOOKAHEAD_MIN_MS, int(s.lookahead_ms - (delay - cfg.MAX_TOTAL_DELAY_MS) - 5))
        if new < s.lookahead_ms:
            s.lookahead_ms = new
            self.engine.set_lookahead(new)
            cfg.save_settings(s)
            self._delay_note = (f"The look-ahead was lowered to {new} ms to keep the total "
                                f"delay under {cfg.MAX_TOTAL_DELAY_MS} ms.")
        self._over_since = 0.0

    # ------------------------------------------------------------------ Core (the window's view)
    def snapshot(self) -> dict:
        s = self.settings
        try:
            est: EngineStatus = self.engine.status()
        except Exception:
            est = EngineStatus(warnings=["The audio part is not answering."])
        try:
            from .audio import find_cable
            cable = find_cable() is not None if not est.output_ok else True
        except Exception:
            cable = est.output_ok
        warnings: list[str] = []
        if not self._words:
            warnings.append("Your blocked word list is empty. Add words in Settings.")
        if est.measured_delay_ms > cfg.MAX_TOTAL_DELAY_MS:
            warnings.append(f"The delay is over {cfg.MAX_TOTAL_DELAY_MS} ms. Lower the look-ahead.")
        for note in (self._core_problem, self._delay_note, self._login_problem):
            if note:
                warnings.append(note)
        try:
            if s.lyrics_enabled and self.lyrics.degraded():
                warnings.append("Spotify is not answering. Risky lines are timed from the last "
                                "known place in the song and may be off.")
        except Exception:
            pass
        if s.lyrics_enabled and time.monotonic() - self._windows_failed_at < 10:
            warnings.append("Lyric mode had a problem. Risky lines are not being watched.")
        lyr_status, track, logged_in = "Lyric mode is off", None, False
        try:
            logged_in = self.lyrics.logged_in()
            if s.lyrics_enabled:
                lyr_status, track = self.lyrics.status(), self.lyrics.track()
        except Exception:
            lyr_status = "Lyric mode had a problem. Listening only."
        try:
            engine_name, pass_ms = self.detector.engine_name(), self.detector.last_pass_ms()
        except Exception:
            engine_name, pass_ms = "not running", 0.0
        if s.enabled and "cpu" in engine_name.lower():
            warnings.append("CPU mode is too slow to cut a word in time. It only mutes risky "
                            "lyric lines. Use GPU mode for a stream.")
        return {
            "enabled": s.enabled, "engine": est, "health": self._health,
            "engine_name": engine_name, "pass_ms": pass_ms, "caught": self._caught,
            "lyrics_status": lyr_status, "track": track, "in_window": self._in_window,
            "cable_found": cable, "warnings": warnings, "lookahead_ms": s.lookahead_ms,
            "removal_mode": s.removal_mode, "lyrics_enabled": s.lyrics_enabled,
            "spotify_logged_in": logged_in, "spotify_client_id": s.spotify_client_id,
            "hotkey": s.hotkey, "engine_choice": s.engine,
            "mute_risky_lines": s.mute_risky_lines,
        }

    def set_enabled(self, on: bool) -> None:
        was = self.settings.enabled
        self.settings.enabled = bool(on)
        if on and not was:
            # words caught while it was off that are still waiting in the buffer get cut now
            held, self._held = self._held, {}
            playhead = self.engine.processor.playhead
            for order in held.values():
                if order.end > playhead:
                    self._submit(order)
        cfg.save_settings(self.settings)

    def set_lookahead(self, ms: int) -> None:
        self.settings.lookahead_ms = int(ms)
        self.settings.clamp()
        self.engine.set_lookahead(self.settings.lookahead_ms)
        cfg.save_settings(self.settings)

    def set_removal(self, mode: str) -> None:
        self.settings.removal_mode = mode
        self.settings.clamp()
        self.engine.set_mode(self.settings.removal_mode, self.settings.dip_db, self.settings.fade_ms)
        cfg.save_settings(self.settings)

    def set_engine(self, engine: str) -> None:
        self.settings.engine = engine
        self.settings.clamp()
        cfg.save_settings(self.settings)
        threading.Thread(target=self._restart, name="npass-restart", daemon=True).start()

    def inputs(self) -> list[str]:
        from .audio import list_inputs
        return list_inputs()

    def set_input(self, name: str) -> None:
        self.settings.input_device = name
        cfg.save_settings(self.settings)
        # the engine swaps the mic live and keeps its sample clock, so nothing else restarts
        self.engine.set_input(name)

    def words(self) -> list[str]:
        return list(self._words)

    def set_words(self, words: list[str]) -> None:
        cfg.save_words(words)
        self._words = cfg.load_words()

    def set_lyrics_enabled(self, on: bool) -> None:
        self.settings.lyrics_enabled = bool(on)
        cfg.save_settings(self.settings)
        try:
            self.lyrics.start() if on else self.lyrics.stop()
        except Exception:
            log.exception("switching lyric mode failed")

    def set_mute_risky_lines(self, on: bool) -> None:
        self.settings.mute_risky_lines = bool(on)
        cfg.save_settings(self.settings)

    def spotify_login(self) -> None:
        try:
            self.lyrics.login()
        except Exception as exc:
            # SpotifyLoginError carries a plain-English message that is safe to show
            from .lyrics import SpotifyLoginError
            self._login_problem = (str(exc) if isinstance(exc, SpotifyLoginError)
                                   else "Signing in to Spotify did not work. Try again.")
            if not isinstance(exc, SpotifyLoginError):
                log.error("spotify login failed: %s", type(exc).__name__)
            return
        self._login_problem = ""
        if self.settings.lyrics_enabled:
            self.lyrics.start()

    def spotify_logout(self) -> None:
        self.lyrics.logout()

    def set_spotify_client_id(self, cid: str) -> None:
        self.settings.spotify_client_id = cid.strip()
        cfg.save_settings(self.settings)

    def recent_catches(self) -> list[Catch]:
        with self._lock:
            return list(self._catches)

    def quit(self) -> None:
        self.stop()
        cfg.save_settings(self.settings)
        if self._quit_cb:
            self._quit_cb()


# ---------------------------------------------------------------------- start-up
def _setup_logging() -> None:
    if sys.stdout is None:            # started by pythonw: there is no console
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    handler = logging.handlers.RotatingFileHandler(
        cfg.data_dir() / "npass.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    for noisy in ("urllib3", "httpx", "httpcore", "huggingface_hub", "faster_whisper"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _already_running() -> bool:
    """One copy only: two would fight over the mic and the cable."""
    if os.name != "nt":
        return False
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW(None, False, "NPass.Filter.SingleInstance")
    return ctypes.get_last_error() == 183   # ERROR_ALREADY_EXISTS


def main() -> int:
    _setup_logging()
    if _already_running():
        try:
            ctypes.windll.user32.MessageBoxW(
                None, "N-Pass is already running. Look for its icon near the clock.", "N-Pass", 0x40)
        except Exception:
            pass
        return 1
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("NPass.Filter")
    except Exception:
        pass
    try:
        core = AppCore()
        core.start()
    except Exception:
        log.exception("N-Pass could not start")
        try:
            ctypes.windll.user32.MessageBoxW(
                None, "N-Pass could not start. The reason is in npass.log in "
                      f"{cfg.data_dir()}.", "N-Pass", 0x10)
        except Exception:
            pass
        return 1
    try:
        from .ui import run_ui
        run_ui(core)
    finally:
        core.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
