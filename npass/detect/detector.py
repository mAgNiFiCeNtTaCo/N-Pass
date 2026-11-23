"""The detector: listens to the buffered mic audio and orders cuts over blocked words.

`feed()` (the AudioTap) only queues. One worker thread resamples 48 kHz -> 16 kHz, keeps the last
`settings.context_s` seconds, and every `settings.hop_ms` runs the recogniser over that buffer.
Passes never queue up: every pass takes ALL the audio that has arrived, so after a slow pass the
next one starts from the newest audio (the skipped hops are counted in `stats`).

What a pass does with the words it heard:
- each word (and each pair of neighbouring words, because recognisers split long words) is scored
  against the blocked list; the bar is `threshold`, or `window_threshold` inside a danger window
- the LAST word may still be being sung, so it is also scored as a beginning (`prefix_score`); a
  hit there is cut on suspicion, with an end a little past what was heard, and the SAME cut_id is
  sent again with a later end on every later pass while the word keeps going
- cut edges come from the loudness track (envelope.py), not from the recogniser's word times
- a word seen again in the next overlapping pass is the same catch: `on_catch` fires once per word

`verified_upto()` is the end of the audio the last COMPLETED pass looked at (less the backend's
own `latency_ms` when it has one: Vosk names a word well after it started).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import soxr

from .. import match
from ..types import (ASR_RATE, SAMPLE_RATE, CutOrder, CutReason, HeardWord, Health, Recogniser,
                     SubmitCut, WindowsProvider, ms_to_samples)
from .envelope import rms_track, snap_edges

log = logging.getLogger("npass.detect")

_RATIO = SAMPLE_RATE // ASR_RATE
_WORD = re.compile(r"[a-z']+")

TAIL_MS = 150            # a last word ending this close to the end of the audio may still be going
JOIN_GAP_MS = 200        # neighbouring words closer than this are also tried as one word
SAME_WORD_MS = 300       # a word starting this close to an earlier catch is that word seen again
MIN_WORD_MS = 60         # a finished "word" shorter than this has no audio behind it (made up)
HOLD_EXTRA_MS = 40       # safety on top of hop + pass time when a cut is held open
STALE_MS = 300           # a NEW hit that finished playing this long ago is dropped
MIN_STEP_MS = 5          # do not re-send a cut for a change smaller than this
MAX_QUEUE_BLOCKS = 6000  # about a minute of 10 ms blocks; older audio is dropped if nobody drains
MAX_HINT_WORDS = 60
RESAMPLE_QUALITY = "LQ"  # about 6 ms of delay; plenty for 16 kHz speech


@dataclass
class _Hit:
    entry: str
    heard: str
    score: float
    start: int           # word time from the recogniser, mic clock
    end: int
    growing: bool
    reason: CutReason


@dataclass
class _CatchState:
    cut_id: int
    entry: str
    heard: str
    start: int           # the cut as last submitted
    end: int
    confidence: float
    reason: CutReason
    open: bool           # True while the word was still going in the last pass that saw it
    seen_in: int = 0     # number of the last pass that matched a word to this catch
    w_start: int = 0     # the word's own span as the recogniser last gave it (not the cut)
    w_end: int = 0


class Detector:
    def __init__(self, settings, words: Callable[[], list[str]], windows: WindowsProvider,
                 submit_cut: SubmitCut, on_catch: Callable[[CutOrder], None],
                 recogniser: Recogniser | None = None):
        self._settings = settings
        self._words = words
        self._windows = windows
        self._submit = submit_cut
        self._on_catch = on_catch
        self._given = recogniser            # None = pick by settings.engine
        self._rec: Recogniser | None = None
        self._engine_name = "starting"
        self.restart_s = 5.0                # wait between attempts to bring a crashed backend back
        self.last_error = ""

        self._q: deque[tuple[np.ndarray, int]] = deque()
        self._q_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        # worker-owned audio state
        self._resampler = None
        self._buf = np.zeros(0, dtype=np.float32)   # 16 kHz rolling buffer
        self._buf_start = 0                          # mic-clock index of _buf[0]
        self._next_in: int | None = None             # mic-clock index expected from the next block
        self._new16 = 0                              # 16 kHz samples added since the last pass
        self._captured = 0                           # end of the newest block handed to feed()

        self._verified = 0
        self._health = Health.STARTING
        self._last_pass_ms = 0.0
        self._next_load_try = 0.0
        self._catches: list[_CatchState] = []
        self._next_cut_id = 1
        self.stats = {"passes": 0, "silent_passes": 0, "hops_skipped": 0, "restarts": 0,
                      "crashes": 0, "gaps": 0, "catches": 0, "stale_hits": 0, "callback_errors": 0}

    # ------------------------------------------------------------------ public face
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if hasattr(self, "_live_callbacks"):      # started again after a stop()
            self._submit, self._on_catch = self._live_callbacks
        self._stop.clear()
        self._health = Health.STARTING
        self._thread = threading.Thread(target=self._run, name="npass-detector", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        # a pass that returns after this must not reach the (possibly rebuilt) audio engine
        if not hasattr(self, "_live_callbacks"):
            self._live_callbacks = (self._submit, self._on_catch)
        self._submit = lambda order: None
        self._on_catch = lambda order: None
        t = self._thread
        if t is not None:
            t.join(timeout=10.0)
        self._thread = None
        self._close_backend()

    def feed(self, block48k: np.ndarray, start_sample: int) -> None:
        """The AudioTap: copies the block into a queue and returns."""
        block = np.array(block48k, dtype=np.float32, copy=True).reshape(-1)
        with self._q_lock:
            self._q.append((block, int(start_sample)))
            if len(self._q) > MAX_QUEUE_BLOCKS:
                self._q.popleft()
        self._captured = int(start_sample) + len(block)
        self._wake.set()

    def verified_upto(self) -> int:
        return self._verified

    def health(self) -> Health:
        return self._health

    def engine_name(self) -> str:
        return self._engine_name

    def last_pass_ms(self) -> float:
        return self._last_pass_ms

    # ------------------------------------------------------------------ worker
    def _run(self) -> None:
        hop_wait = max(0.005, self._settings.hop_ms / 4000.0)
        while not self._stop.is_set():
            try:
                if self._rec is None:
                    self._drain()
                    if time.monotonic() < self._next_load_try:
                        self._stop.wait(0.05)
                        continue
                    self._load_backend()
                self._wake.wait(hop_wait)
                self._wake.clear()
                if self._stop.is_set():
                    break
                self.process_pending()
            except Exception as exc:       # never let the thread die silently
                self._crashed(exc)

    def process_pending(self, force: bool = False) -> bool:
        """Take in queued audio and run one pass if a hop's worth has arrived (or `force`).
        The worker calls this; tests and the offline harness may call it directly instead of
        starting the thread. Returns True when a pass ran."""
        if self._rec is None:
            self._load_backend()
        self._drain()
        hop16 = max(1, int(self._settings.hop_ms * ASR_RATE / 1000))
        if self._new16 < hop16 and not (force and self._new16 > 0):
            return False
        self.stats["hops_skipped"] += max(0, self._new16 // hop16 - 1)
        self._new16 = 0
        self._one_pass()
        self._health = Health.OK
        return True

    def _crashed(self, exc: Exception) -> None:
        self.stats["crashes"] += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        log.error("detector worker error: %s", self.last_error)
        self._health = Health.DOWN
        self._close_backend()
        self._engine_name = "not running (the speech model stopped, trying again)"
        self._next_load_try = time.monotonic() + self.restart_s

    def _close_backend(self) -> None:
        rec, self._rec = self._rec, None
        if rec is not None:
            try:
                rec.close()
            except Exception:
                pass

    def _load_backend(self) -> None:
        self.stats["restarts"] += 1
        if self._given is not None:
            self._given.load()
            self._rec = self._given
            self._engine_name = getattr(self._given, "name", "custom")
            return
        self._rec, self._engine_name = pick_backend(self._settings)

    # ------------------------------------------------------------------ audio in
    def _drain(self) -> None:
        with self._q_lock:
            if not self._q:
                return
            blocks = list(self._q)
            self._q.clear()
        for block, start in blocks:
            if self._next_in is None or start != self._next_in:
                # first block, or a hole in the audio: start a fresh buffer at this block
                if self._next_in is not None:
                    self.stats["gaps"] += 1
                self._resampler = soxr.ResampleStream(SAMPLE_RATE, ASR_RATE, 1, dtype="float32",
                                                      quality=RESAMPLE_QUALITY)
                self._buf = np.zeros(0, dtype=np.float32)
                self._buf_start = start
            self._next_in = start + len(block)
            out = self._resampler.resample_chunk(block)
            if len(out):
                self._buf = np.concatenate((self._buf, out))
                self._new16 += len(out)
        keep = int(self._settings.context_s * ASR_RATE)
        extra = len(self._buf) - keep
        if extra > 0:
            self._buf = self._buf[extra:]
            self._buf_start += extra * _RATIO

    # ------------------------------------------------------------------ one pass
    def _one_pass(self) -> None:
        t0 = time.perf_counter()
        audio = self._buf
        buf_start = self._buf_start
        buf_end = buf_start + len(audio) * _RATIO
        blocked = [w for w in self._words() if match.normalize(w)]
        self.stats["passes"] += 1

        heard: list[HeardWord] = []
        if blocked and len(audio):
            track, frame = rms_track(audio, ASR_RATE)
            if len(track) and float(track.max()) < self._gate():
                self.stats["silent_passes"] += 1      # nothing but silence: no need to ask the model
            else:
                heard = self._rec.transcribe(audio, buf_start, self._hints(blocked, buf_start, buf_end))
                hits = self._find_hits(heard, blocked, buf_end)
                for hit in hits:
                    self._apply(hit, track, frame, len(audio), buf_start, buf_end, t0)
            self._sustain(track, frame, buf_start, buf_end, t0)

        for c in self._catches:
            if c.open and c.end < buf_end:
                c.open = False
        self._catches = [c for c in self._catches if c.end > buf_start - SAMPLE_RATE]
        # a backend that answers late (Vosk) says so; its last stretch is not verified yet
        behind = ms_to_samples(float(getattr(self._rec, "latency_ms", 0.0) or 0.0))
        self._verified = max(self._verified, buf_end - behind)
        self._last_pass_ms = (time.perf_counter() - t0) * 1000.0

    def _gate(self) -> float:
        return 10.0 ** (float(getattr(self._settings, "silence_gate_db", -50.0)) / 20.0)

    def _hints(self, blocked: list[str], buf_start: int, buf_end: int) -> list[str]:
        """Words the recogniser is told to expect (Whisper "hotwords"). OFF unless switched on.

        Measured 2026-11-23 with small.en on one spoken test clip (three blocked stand-in words in
        two sentences), 3 s rolling buffer:
        - blocked words always given as hotwords: 8 of 43 passes came back with a blocked word
          that was never said, often a whole row of them, placed over ordinary words
        - hotwords only inside a danger window (blocked words + the lyric line): the made-up
          words also pushed a REAL blocked word out of the transcript, and part of it played
        - no hotwords: all three words heard in every pass that covered them, nothing made up
        The prompt that carries hotwords is meant for long recordings; on a short buffer that
        starts mid-sentence the model tends to repeat it. So the default is no hotwords at all.
        Optional settings, read with getattr: `hotwords_in_windows` (hints while a danger window
        is near) and `hotwords_always`; both default to False.
        """
        always = bool(getattr(self._settings, "hotwords_always", False))
        in_windows = bool(getattr(self._settings, "hotwords_in_windows", False))
        if not always and not in_windows:
            return []
        ahead = ms_to_samples(self._settings.lookahead_ms)
        wins = self._windows(buf_start, buf_end + ahead)
        if not wins and not always:
            return []
        hints = list(blocked)
        for w in wins:
            for tok in _WORD.findall(w.line.lower()):
                if tok not in hints:
                    hints.append(tok)
                    if len(hints) >= MAX_HINT_WORDS:
                        return hints
        return hints

    def _find_hits(self, heard: list[HeardWord], blocked: list[str], buf_end: int) -> list[_Hit]:
        tail = ms_to_samples(TAIL_MS)
        join_gap = ms_to_samples(JOIN_GAP_MS)
        min_word = ms_to_samples(MIN_WORD_MS)
        heard = sorted((w for w in heard if match.normalize(w.text)), key=lambda w: w.start)
        n = len(heard)
        # (text, start, end, may still be growing, is the tail of the audio)
        cands: list[tuple[str, int, int, bool, bool]] = []
        for i, w in enumerate(heard):
            last = i == n - 1
            cands.append((w.text, w.start, w.end, w.partial or last,
                          last and (w.partial or buf_end - w.end <= tail)))
            if not last:
                nx = heard[i + 1]
                if nx.start - w.end <= join_gap:
                    nx_last = i + 1 == n - 1
                    cands.append((w.text + nx.text, w.start, nx.end, nx.partial or nx_last,
                                  nx_last and (nx.partial or buf_end - nx.end <= tail)))

        hits: list[_Hit] = []
        for text, start, end, may_grow, growing in cands:
            in_window = bool(self._windows(start, max(end, start + 1)))
            bar = self._settings.window_threshold if in_window else self._settings.threshold
            entry, s = match.best_match(text, blocked)
            if may_grow:
                e2, s2 = match.best_match(text, blocked, partial=True)
                if s2 > s:
                    entry, s = e2, s2
            if entry and s >= bar and (growing or end - start >= min_word):
                hits.append(_Hit(entry, text.strip(), s, start, end, growing,
                                 CutReason.LYRIC_WINDOW if in_window else CutReason.ASR))

        # a word and the pair it belongs to can both hit: keep one hit per stretch of audio
        hits.sort(key=lambda h: h.start)
        merged: list[_Hit] = []
        for h in hits:
            m = merged[-1] if merged else None
            if m is not None and h.start < m.end:
                # the better match decides where the word is: a weak pair ("some" + the word)
                # must not drag the cut over the innocent neighbour. Equal scores: cover both.
                if h.score == m.score:
                    merged[-1] = _Hit(m.entry, m.heard, m.score, min(m.start, h.start),
                                      max(m.end, h.end), m.growing or h.growing, m.reason)
                else:
                    best, other = (h, m) if h.score > m.score else (m, h)
                    merged[-1] = _Hit(best.entry, best.heard, best.score, best.start, best.end,
                                      best.growing or (other.growing and other.end >= best.end),
                                      best.reason)
            else:
                merged.append(h)
        return merged

    def _apply(self, hit: _Hit, track: np.ndarray, frame: int, total16: int,
               buf_start: int, buf_end: int, t0: float) -> None:
        s = self._settings
        pad = ms_to_samples(s.cut_pad_ms)
        s16 = (hit.start - buf_start) // _RATIO
        e16 = (hit.end - buf_start) // _RATIO
        if not hit.growing and len(track):
            # a finished word with no sound at all behind it was not said
            f0 = max(0, s16 // frame - 5)
            f1 = min(len(track), e16 // frame + 6)
            if f1 > f0 and float(track[f0:f1].max()) < self._gate():
                return
        edges = snap_edges(track, frame, s16, e16, total16)
        cut_start = buf_start + edges.start * _RATIO - pad
        cut_end = buf_start + edges.end * _RATIO + pad
        if hit.growing:
            # the word may go on: hold the cut open until the next pass can extend it
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            hold_ms = elapsed_ms + s.hop_ms + min(self._last_pass_ms, 500.0) + HOLD_EXTRA_MS
            cut_end = max(cut_end, buf_end + pad + ms_to_samples(hold_ms))

        pass_no = self.stats["passes"]
        playhead = self._captured - ms_to_samples(s.lookahead_ms)   # an estimate; only used to
        step = ms_to_samples(MIN_STEP_MS)                           # avoid re-sending played audio
        same = ms_to_samples(SAME_WORD_MS)

        # is this a word we already caught in an earlier pass? Judged by the word's own span
        # (cuts are padded and held open, so they would swallow the next word)
        best: _CatchState | None = None
        best_gap = 0
        for c in self._catches:
            if c.seen_in == pass_no:
                continue                      # already matched to another word of this pass
            gap = abs(hit.start - c.w_start)
            overlap = min(c.w_end, hit.end) - max(c.w_start, hit.start)
            shorter = max(1, min(c.w_end - c.w_start, hit.end - hit.start))
            if (gap <= same or overlap * 2 >= shorter) and (best is None or gap < best_gap):
                best, best_gap = c, gap

        if best is None:
            if cut_end < playhead - ms_to_samples(STALE_MS):
                # First heard long after it played, although dozens of earlier passes over the
                # same audio did not hear it: nothing is left to cut, and it is far more likely
                # made up by the model than missed. Counted, not logged as a catch.
                self.stats["stale_hits"] += 1
                return
            c = _CatchState(self._next_cut_id, hit.entry, hit.heard, cut_start, cut_end,
                            hit.score, hit.reason, hit.growing, pass_no, hit.start, hit.end)
            self._next_cut_id += 1
            self._catches.append(c)
            self.stats["catches"] += 1
            order = self._order(c)
            self._submit(order)
            try:
                self._on_catch(order)
            except Exception:
                self.stats["callback_errors"] += 1
                log.exception("on_catch failed")
            return

        c = best
        c.seen_in = pass_no
        c.w_start, c.w_end = hit.start, hit.end
        new_start, new_end = c.start, c.end
        if cut_start <= c.start - step and c.start > playhead:
            new_start = cut_start
        if cut_end >= c.end + step and cut_end > playhead:
            new_end = cut_end
        # A cut never shrinks. Once the word has ended it simply stops being extended: the end
        # that was held open stays (at most a hop and a pass past the word), because pulling it
        # back would trust the recogniser's end time, and a stop consonant looks like a pause.
        c.open = hit.growing
        if hit.score > c.confidence:
            c.confidence, c.heard, c.entry = hit.score, hit.heard, hit.entry
        if new_start != c.start or new_end != c.end:
            c.start, c.end = new_start, new_end
            self._submit(self._order(c))

    def _sustain(self, track: np.ndarray, frame: int, buf_start: int, buf_end: int,
                 t0: float) -> None:
        """Keep a cut open while the voice that sang the word has not paused.

        Measured 2026-11-23 on held (sung-like) words: the recogniser gives a held word a short
        span (0.36 s for a word that lasted 1.27 s) and never extends it, so most of the word
        played. The loudness of the voice is the better judge of where a held word ends: the cut
        now runs to the first real dip after the recognised span, or `max_hold_ms` at most.
        A miss is worse than a wrong cut (the user's choice), so with no dip in sight it holds.
        """
        s = self._settings
        max_hold = ms_to_samples(float(getattr(s, "max_hold_ms", 1500.0)))
        if max_hold <= 0 or not len(track) or not self._catches:
            return
        pad = ms_to_samples(s.cut_pad_ms)
        step = ms_to_samples(MIN_STEP_MS)
        playhead = self._captured - ms_to_samples(s.lookahead_ms)
        fr48 = frame * _RATIO
        need = max(1, int(round(40.0 / (frame * 1000.0 / ASR_RATE))))      # a 40 ms dip
        gate = self._gate()
        limits = self.__dict__.setdefault("_hold_limits", {})
        for c in self._catches:
            if c.end <= playhead:
                continue                                   # fully played: nothing left to hold
            limit = limits.setdefault(c.cut_id, c.w_end + max_hold)
            if c.end >= limit:
                continue
            f0 = max(0, (c.w_start - buf_start) // fr48)
            f1 = min(len(track), (c.w_end - buf_start) // fr48 + 1)
            if f1 <= f0:
                continue
            peak = float(track[f0:f1].max())
            if peak < gate:
                continue
            floor = max(gate, peak * 0.3)
            start_f = max(f0, f1 - 1 - need)
            run, pause_at = 0, None
            for i in range(start_f, len(track)):
                if track[i] < floor:
                    run += 1
                    if run >= need:
                        pause_at = i - run + 1
                        break
                else:
                    run = 0
            if pause_at is None:
                # still sounding at the end of what we have: hold until the next pass can look
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                hold_ms = elapsed_ms + s.hop_ms + min(self._last_pass_ms, 500.0) + HOLD_EXTRA_MS
                new_end = buf_end + pad + ms_to_samples(hold_ms)
            else:
                new_end = buf_start + pause_at * fr48 + pad
            new_end = min(new_end, limit)
            if new_end >= c.end + step and new_end > playhead:
                c.end = new_end
                self.stats["sustained"] = self.stats.get("sustained", 0) + 1
                self._submit(self._order(c))
        if len(limits) > 500:
            live = {c.cut_id for c in self._catches}
            for k in [k for k in limits if k not in live]:
                del limits[k]

    @staticmethod
    def _order(c: _CatchState) -> CutOrder:
        return CutOrder(cut_id=c.cut_id, start=c.start, end=c.end, word=c.entry, heard=c.heard,
                        confidence=round(float(c.confidence), 3), reason=c.reason)


def pick_backend(settings) -> tuple[Recogniser, str]:
    """Load the backend `settings.engine` asks for. Returns (loaded recogniser, engine name).
    "auto": the GPU model when it loads and fits `gpu_mem_limit_mb`, else Vosk on the CPU.
    Raises when nothing could be loaded."""
    engine = getattr(settings, "engine", "auto")
    gpu_note = ""
    if engine in ("auto", "gpu"):
        from .whisper_backend import WhisperRecogniser
        rec = WhisperRecogniser(model=settings.whisper_model, compute=settings.whisper_compute)
        try:
            rec.load()
            limit = float(getattr(settings, "gpu_mem_limit_mb", 0) or 0)
            used = rec.gpu_mem_mb
            if engine == "auto" and limit and used is not None and used > limit:
                rec.close()
                raise RuntimeError(f"the GPU model uses {used:.0f} MB, over the {limit:.0f} MB limit")
            return rec, rec.name
        except Exception as exc:
            log.warning("GPU speech model not used: %s", exc)
            if engine == "gpu":
                raise
            gpu_note = ", the GPU model did not load"
    from .vosk_backend import VoskRecogniser
    rec = VoskRecogniser()
    rec.load()
    return rec, rec.name + gpu_note
