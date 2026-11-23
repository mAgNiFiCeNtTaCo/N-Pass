"""The live engine: microphone -> LookaheadProcessor -> small FIFO -> virtual cable.

Threads:
  - the mic callback (PortAudio): processor.process(), push to the FIFO, queue the block for the tap
  - the cable callback (PortAudio): pull from the FIFO (with drift correction)
  - the tap thread: hands every captured block to `tap(block, start_sample)`
  - the supervisor: opens devices, notices when one stops, retries, checks the total delay

The cable is the ONLY output this engine ever opens. No cable = no output stream at all.
"""
from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque

import numpy as np

from npass import settings as settings_mod
from npass.audio import devices
from npass.audio.processor import LookaheadProcessor
from npass.types import SAMPLE_RATE, AudioTap, CutOrder, EngineStatus, ms_to_samples

CABLE_MISSING = "The virtual audio cable is not installed. See the README, step 1."
INPUT_RETRY_S = 1.0
OUTPUT_RETRY_S = 2.0
STALL_S = 1.0                 # no callback for this long = the device is gone
DEFAULT_FIFO_MS = 30.0


class JitterFifo:
    """Mono sample queue between the mic clock and the cable clock.

    pull() keeps the fill near `target`: when the smoothed fill drifts above it, one block is made
    from one sample more (a tiny resample, so no click); below it, from one sample less.
    """

    def __init__(self, target: int, block: int):
        self._lock = threading.Lock()
        self._block = max(1, int(block))
        self._size = max(SAMPLE_RATE, 16 * self._block)
        self._buf = np.zeros(self._size, dtype=np.float32)
        self._r = 0
        self._fill = 0
        self._priming = True
        self._ema = 0.0
        self.min_target = 2 * self._block
        self.target = max(self.min_target, int(target))
        self.underruns = 0
        self.overflows = 0
        self.corrections = 0

    @property
    def fill(self) -> int:
        return self._fill

    @property
    def smoothed_fill(self) -> float:
        """Average number of samples waiting when the cable asks for a block (0 while refilling)."""
        return 0.0 if self._priming else self._ema

    def set_target(self, target: int) -> None:
        with self._lock:
            self.target = max(self.min_target, int(target))

    def clear(self) -> None:
        with self._lock:
            self._r = 0
            self._fill = 0
            self._priming = True
            self._ema = 0.0

    def push(self, x: np.ndarray) -> None:
        n = x.shape[0]
        with self._lock:
            if n >= self._size:
                x = x[n - self._size + 1:]
                n = x.shape[0]
            over = self._fill + n - self._size
            if over > 0:                       # the cable stopped pulling: forget the oldest
                self._r = (self._r + over) % self._size
                self._fill -= over
                self.overflows += 1
            w = (self._r + self._fill) % self._size
            first = min(n, self._size - w)
            self._buf[w:w + first] = x[:first]
            if first < n:
                self._buf[:n - first] = x[first:]
            self._fill += n

    def _take(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.float32)
        first = min(n, self._size - self._r)
        out[:first] = self._buf[self._r:self._r + first]
        if first < n:
            out[first:] = self._buf[:n - first]
        self._r = (self._r + n) % self._size
        self._fill -= n
        return out

    def pull(self, frames: int) -> np.ndarray:
        with self._lock:
            fade_in = False
            if self._priming:
                if self._fill < max(self.target, frames):
                    return np.zeros(frames, dtype=np.float32)
                self._priming = False
                self._ema = float(self._fill)
                fade_in = True
            else:
                self._ema += 0.01 * (self._fill - self._ema)
            # far too full (the cable stalled for a while): jump back to the target
            if self._fill > self.target + 6 * max(frames, self._block):
                drop = self._fill - self.target
                self._r = (self._r + drop) % self._size
                self._fill -= drop
                self._ema = float(self._fill)
                self.overflows += 1
                fade_in = True
            want = frames
            tol = max(8, frames // 8)
            if frames > 2:
                if self._ema > self.target + tol and self._fill >= frames + 1:
                    want = frames + 1
                    self._ema -= 1.0
                elif self._ema < self.target - tol and self._fill >= frames - 1:
                    want = frames - 1
                    self._ema += 1.0
            if self._fill < want:              # ran dry: fade out what is left, then refill
                have = self._fill
                out = np.zeros(frames, dtype=np.float32)
                if have > 0:
                    out[:have] = self._take(have) * np.linspace(1.0, 0.0, have, dtype=np.float32)
                self._priming = True
                self.underruns += 1
                return out
            data = self._take(want)
        if want != frames:
            self.corrections += 1
            data = np.interp(np.linspace(0.0, want - 1.0, frames), np.arange(want), data
                             ).astype(np.float32)
        if fade_in:
            data *= np.linspace(0.0, 1.0, frames, dtype=np.float32)
        return data


class AudioEngine:
    def __init__(self, settings, tap: AudioTap | None = None, *, backend=None,
                 supervisor: bool = True):
        """`backend` and `supervisor` are for tests: a fake device layer, and no background
        supervisor thread (the test calls `_tick(now)` itself)."""
        self.settings = settings
        self._tap = tap
        self._backend = backend if backend is not None else devices.SoundBackend()
        self._use_supervisor = supervisor
        self.processor = LookaheadProcessor(settings.lookahead_ms, settings.fade_ms,
                                            settings.removal_mode, settings.dip_db)
        self._block = max(16, ms_to_samples(settings.block_ms))
        fifo_ms = float(getattr(settings, "fifo_target_ms", DEFAULT_FIFO_MS))
        self._fifo = JitterFifo(ms_to_samples(fifo_ms), self._block)
        self._fifo_target0 = self._fifo.target

        self._sup_lock = threading.RLock()      # everything the supervisor changes
        self._feed_lock = threading.Lock()      # one feeder at a time (mic callback or silence)
        self._state_lock = threading.Lock()     # warnings + names, read by status()
        self._running = False
        self._stop_evt = threading.Event()
        self._sup_thread: threading.Thread | None = None
        self._tap_thread: threading.Thread | None = None
        self._tap_q: queue.Queue = queue.Queue(maxsize=3000)
        self.tap_dropped = 0
        self.tap_errors = 0

        self._in_stream = None
        self._out_stream = None
        self._in_name = ""
        self._out_name = ""
        self._sticky_in = ""                    # the mic that worked: keep asking for that one
        self._warn: dict[str, str] = {}
        self._fp = None

        self._silence = np.zeros(self._block, dtype=np.float32)
        self._peaks: deque = deque(maxlen=max(1, math.ceil(50.0 / max(1, settings.block_ms))))
        self._fade_in_next = False
        self._in_lat = 0.0
        self._out_lat = 0.0
        self._delay_ms: float | None = None
        self._last_in_t = 0.0
        self._last_out_t = 0.0
        self._next_in_retry = 0.0
        self._next_out_retry = 0.0
        self._silence_t = 0.0
        self._on_fallback = False
        self._next_fp_check = 0.0
        self.glitches = 0

    # ------------------------------------------------------------ public face

    def start(self) -> None:
        """Never raises for a device problem: see status().warnings. Safe to call twice."""
        with self._sup_lock:
            if self._running:
                return
            self._running = True
            self._stop_evt.clear()
            self._sticky_in = ""
            self._on_fallback = False
            now = time.monotonic()
            self._silence_t = now
            try:
                self._reopen_all(now)
            except Exception as e:                       # belt and braces: start() must not raise
                self._set_warn("mic", f"The sound devices could not be opened ({_short(e)}). "
                                      "N-Pass keeps trying.")
            self._next_in_retry = now + INPUT_RETRY_S
            self._next_out_retry = now + OUTPUT_RETRY_S
        if self._tap is not None:
            self._tap_thread = threading.Thread(target=self._tap_loop, name="npass-tap", daemon=True)
            self._tap_thread.start()
        if self._use_supervisor:
            self._sup_thread = threading.Thread(target=self._sup_loop, name="npass-audio-supervisor",
                                                daemon=True)
            self._sup_thread.start()

    def stop(self) -> None:
        """Idempotent. The sample clock is kept, so a later start() carries on counting."""
        with self._sup_lock:
            was_running = self._running
            self._running = False
            self._stop_evt.set()
            self._close_input()
            self._close_output()
        if not was_running:
            return
        me = threading.current_thread()
        for t in (self._sup_thread, self._tap_thread):
            if t is not None and t is not me:
                t.join(timeout=3.0)
        self._sup_thread = None
        self._tap_thread = None
        while True:                                       # nothing stale for the next start()
            try:
                self._tap_q.get_nowait()
            except queue.Empty:
                break
        with self._state_lock:
            self._warn.clear()
        self._peaks.clear()

    def submit_cut(self, order: CutOrder) -> float:
        return self.processor.submit_cut(order)

    def set_lookahead(self, ms) -> None:
        self.processor.set_lookahead(ms)

    def set_mode(self, mode, dip_db, fade_ms) -> None:
        self.processor.set_mode(mode, dip_db, fade_ms)

    def set_input(self, name: str) -> None:
        """Switch microphone while running ("" = go back to settings.input_device / input_hint).
        Not in the contract; offered for Core.set_input."""
        with self._sup_lock:
            self._sticky_in = (name or "").strip()
            self._on_fallback = False
            if not self._running:
                return
            self._close_input()
            self._clear_warn("mic", "fallback", "rate_in")
            self._next_in_retry = 0.0                     # the next supervisor tick reopens it
            self._silence_t = time.monotonic()
            if not self._use_supervisor:
                self._tick()

    def status(self) -> EngineStatus:
        if self._delay_ms is None:
            self._update_delay()
        with self._state_lock:
            warnings = list(self._warn.values())
            in_name, out_name = self._in_name, self._out_name
        peak = max(self._peaks, default=0.0) if self._in_stream is not None else 0.0
        level = 20.0 * math.log10(peak) if peak > 1e-6 else -120.0
        return EngineStatus(
            running=self._running,
            input_ok=self._in_stream is not None,
            output_ok=self._out_stream is not None,
            input_name=in_name,
            output_name=out_name,
            lookahead_ms=self.processor.lookahead_ms,
            measured_delay_ms=float(self._delay_ms or 0.0),
            level_db=max(-120.0, level),
            cutting=self.processor.cutting,
            warnings=warnings,
        )

    # ------------------------------------------------------------ audio callbacks

    def _on_input(self, mono: np.ndarray, latency: float | None, glitch: bool) -> None:
        """Mic callback (PortAudio thread)."""
        if not self._running:
            return
        if glitch:
            self.glitches += 1
        if latency is not None:
            self._in_lat += 0.05 * (latency - self._in_lat)
        self._last_in_t = time.monotonic()
        self._feed(mono, real=True)

    def _feed(self, mono: np.ndarray, real: bool) -> None:
        with self._feed_lock:
            block = np.array(mono, dtype=np.float32).reshape(-1)       # our own copy
            if real:
                if self._fade_in_next:                    # first block after a (re)connect
                    block *= np.linspace(0.0, 1.0, block.shape[0], dtype=np.float32)
                    self._fade_in_next = False
                self._peaks.append(float(np.max(np.abs(block))) if block.shape[0] else 0.0)
            start = self.processor.captured
            out = self.processor.process(block)
            if self._out_stream is not None:
                self._fifo.push(out)
            if self._tap is not None:
                try:
                    self._tap_q.put_nowait((block, start))
                except queue.Full:
                    self.tap_dropped += 1

    def _on_output(self, frames: int, latency: float | None) -> np.ndarray:
        """Cable callback (PortAudio thread)."""
        if latency is not None:
            self._out_lat += 0.05 * (latency - self._out_lat)
        self._last_out_t = time.monotonic()
        if not self._running:
            return np.zeros(frames, dtype=np.float32)
        return self._fifo.pull(frames)

    def _tap_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                block, start = self._tap_q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._tap(block, start)
            except Exception:
                self.tap_errors += 1

    # ------------------------------------------------------------ supervisor

    def _sup_loop(self) -> None:
        while not self._stop_evt.wait(0.02 if self._in_stream is None else 0.1):
            try:
                self._tick()
            except Exception:
                pass                                       # never let the supervisor die

    def _tick(self, now: float | None = None) -> None:
        with self._sup_lock:
            if not self._running:
                return
            now = time.monotonic() if now is None else now

            # 1. has a device stopped?
            if self._in_stream is not None and (
                    not _active(self._in_stream) or now - self._last_in_t > STALL_S):
                name = self._in_name
                self._close_input()
                self._set_warn("mic", _mic_gone(name))
                self._next_in_retry = now + INPUT_RETRY_S
                self._silence_t = now
            if self._out_stream is not None and (
                    not _active(self._out_stream) or now - self._last_out_t > STALL_S):
                name = self._out_name
                self._close_output()
                self._set_warn("cable", f'The virtual audio cable ("{name}") stopped responding. '
                                        "N-Pass is reconnecting to it.")
                self._next_out_retry = now + OUTPUT_RETRY_S

            # 2. retries
            if self._in_stream is None and now >= self._next_in_retry:
                self._next_in_retry = now + INPUT_RETRY_S
                fp = self._backend.fingerprint()
                if fp is None or fp != self._fp or self._out_stream is None:
                    self._reopen_all(now)                  # fresh device list (needs all closed)
                else:
                    self._open_input(now)
                if self._out_stream is None:
                    self._next_out_retry = now + OUTPUT_RETRY_S
            elif self._out_stream is None and now >= self._next_out_retry:
                self._next_out_retry = now + OUTPUT_RETRY_S
                fp = self._backend.fingerprint()
                if fp is not None and fp != self._fp:
                    self._reopen_all(now)                  # Windows' device list changed
                else:
                    self._open_output(now)
            elif self._on_fallback and self._in_stream is not None and now >= self._next_fp_check:
                # running on a stand-in mic: when Windows' device list changes, look again for
                # the wanted one (it may simply have started after N-Pass)
                self._next_fp_check = now + OUTPUT_RETRY_S
                fp = self._backend.fingerprint()
                if fp is not None and fp != self._fp:
                    self._fp = fp
                    if devices.match_device(self._rescanned_inputs(), self.settings.input_hint):
                        self._sticky_in = ""
                    self._open_input(now)
                    self._open_output(now)

            # 3. mic down: keep the clock running on silence, at real-time pace
            if self._in_stream is None:
                due = int((now - self._silence_t) * SAMPLE_RATE / self._block)
                if due > 100:                              # we were asleep: do not flood
                    self._silence_t = now - 100 * self._block / SAMPLE_RATE
                    due = 100
                for _ in range(max(0, due)):
                    self._feed(self._silence, real=False)
                self._silence_t += max(0, due) * self._block / SAMPLE_RATE

            # 4. total delay
            self._update_delay()

    def _reopen_all(self, now: float) -> None:
        self._close_input()
        self._close_output()
        self._backend.rescan()
        self._fp = self._backend.fingerprint()
        self._open_input(now)
        self._open_output(now)

    def _rescanned_inputs(self):
        """Close everything, refresh the device list, return the microphones."""
        self._close_input()
        self._close_output()
        self._backend.rescan()
        return self._backend.inputs()

    def _pick_input(self):
        """-> (device or None, the name we were looking for, fell back to another mic?)"""
        devs = self._backend.inputs()
        s = self.settings
        wanted = self._sticky_in or (s.input_device or "").strip()
        if wanted:
            return devices.match_device(devs, wanted), wanted, False
        hint = (s.input_hint or "").strip()
        dev = devices.match_device(devs, hint)
        if dev is not None:
            return dev, hint, False
        loop = devices.CABLE_RECORD_END.lower()
        usable = [d for d in devs if loop not in d.name.lower()]
        default = self._backend.default_input()
        if default is not None and loop not in default.name.lower():
            return default, hint, True
        return (usable[0] if usable else None), hint, True

    def _open_input(self, now: float) -> bool:
        via_sticky = bool(self._sticky_in)
        dev, wanted, fell_back = self._pick_input()
        if dev is None:
            self._set_warn("mic", _mic_gone(wanted) if wanted and not fell_back else
                           "No microphone was found. Plug one in and N-Pass carries on by itself.")
            return False
        self._last_in_t = now
        self._fade_in_next = True
        try:
            stream, converted = self._backend.open_input(dev, self._block, self._on_input)
        except Exception as e:
            self._set_warn("mic", f'The microphone "{dev.name}" could not be opened ({_short(e)}). '
                                  "N-Pass is sending silence and tries again every second.")
            return False
        self._in_lat = _latency(stream)
        self._last_in_t = now
        self._peaks.clear()
        with self._state_lock:
            self._in_name = dev.name
        self._in_stream = stream
        self._sticky_in = dev.name
        self._clear_warn("mic")
        if fell_back:
            self._on_fallback = True
            self._set_warn("fallback", f'No microphone with "{wanted}" in its name was found. '
                                       f'Using "{dev.name}" instead.')
        elif not via_sticky:
            self._on_fallback = False
            self._clear_warn("fallback")
        self._rate_warning("rate_in", dev, converted)
        return True

    def _open_output(self, now: float) -> bool:
        hint = (getattr(self.settings, "output_hint", "") or devices.CABLE_HINT).strip()
        dev = devices.match_device(self._backend.outputs(), hint)
        if dev is None:
            with self._state_lock:
                self._out_name = ""
            self._set_warn("cable", CABLE_MISSING)
            return False
        self._fifo.clear()
        self._last_out_t = now
        try:
            stream, converted = self._backend.open_output(dev, self._block, self._on_output)
        except Exception as e:
            self._set_warn("cable", f'The virtual audio cable ("{dev.name}") could not be opened '
                                    f"({_short(e)}). N-Pass tries again every 2 seconds.")
            return False
        self._out_lat = _latency(stream)
        self._last_out_t = now
        with self._state_lock:
            self._out_name = dev.name
        self._out_stream = stream
        self._clear_warn("cable")
        self._rate_warning("rate_out", dev, converted)
        return True

    def _rate_warning(self, key: str, dev, converted: bool) -> None:
        rate = float(getattr(dev, "default_rate", 0.0) or 0.0)
        if converted or (rate and abs(rate - SAMPLE_RATE) >= 1.0):
            now_at = f"{rate:.0f} Hz" if rate and abs(rate - SAMPLE_RATE) >= 1.0 else "another rate"
            self._set_warn(key, f'"{dev.name}" is set to {now_at} in Windows, so Windows is '
                                f"converting the sound to {SAMPLE_RATE} Hz. To fix it: Windows Sound "
                                f'settings -> "{dev.name}" -> Advanced -> {SAMPLE_RATE} Hz.')
        else:
            self._clear_warn(key)

    def _close_input(self) -> None:
        stream, self._in_stream = self._in_stream, None
        _close(stream)
        self._peaks.clear()

    def _close_output(self) -> None:
        stream, self._out_stream = self._out_stream, None
        _close(stream)
        self._fifo.clear()

    # ------------------------------------------------------------ delay

    def _raw_delay_ms(self) -> float:
        ms = self.processor.lookahead_ms
        if self._in_stream is not None:
            ms += self._in_lat * 1000.0
        if self._out_stream is not None:
            ms += self._out_lat * 1000.0 + self._fifo.smoothed_fill * 1000.0 / SAMPLE_RATE
        return ms

    def _update_delay(self) -> None:
        raw = self._raw_delay_ms()
        limit = float(settings_mod.MAX_TOTAL_DELAY_MS)
        if self._delay_ms is None:
            self._delay_ms = raw
        else:
            self._delay_ms += 0.2 * (raw - self._delay_ms)
        if self._delay_ms > limit:
            # first give back FIFO time (the drift corrector drains it without a click) ...
            excess = ms_to_samples(self._delay_ms - limit)
            self._fifo.set_target(self._fifo.target - excess)
            # ... and say so when that cannot be enough
            self._set_warn("delay", f"The total delay is {self._delay_ms:.0f} ms, more than the "
                                    f"{limit:.0f} ms limit. Lower the look-ahead in N-Pass.")
        else:
            self._clear_warn("delay")
            if self._delay_ms < limit - 40.0 and self._fifo.target < self._fifo_target0:
                self._fifo.set_target(self._fifo_target0)

    # ------------------------------------------------------------ warnings

    def _set_warn(self, key: str, text: str) -> None:
        with self._state_lock:
            self._warn[key] = text

    def _clear_warn(self, *keys: str) -> None:
        with self._state_lock:
            for k in keys:
                self._warn.pop(k, None)


def _mic_gone(name: str) -> str:
    return (f'The microphone "{name}" is not connected. N-Pass is sending silence and carries on '
            "by itself when it is back.")


def _short(e: Exception) -> str:
    text = " ".join(str(e).split())
    return (text[:117] + "...") if len(text) > 120 else (text or type(e).__name__)


def _active(stream) -> bool:
    try:
        return bool(stream.active)
    except Exception:
        return False


def _latency(stream) -> float:
    try:
        v = stream.latency
        if isinstance(v, (tuple, list)):
            v = max(v)
        v = float(v)
        return v if 0.0 <= v < 1.0 else 0.0
    except Exception:
        return 0.0


def _close(stream) -> None:
    if stream is None:
        return
    for step in ("abort", "close"):
        try:
            getattr(stream, step)()
        except Exception:
            pass
