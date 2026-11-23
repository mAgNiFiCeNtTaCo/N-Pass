"""Run recorded clips through the same filter the live app uses, and say exactly what was caught.

    uv run python tools/run_clips.py test_clips
    uv run python tools/run_clips.py test_clips/standin --words pineapple,banana --lookahead 550

Every .wav / .mp3 / .flac / .m4a in the folder is fed, 10 ms at a time and at REAL-TIME pace, into
the classes the live app is built from: `LookaheadProcessor` (the delay and the cutter), `Detector`
(the recogniser and the matcher) and `SafetyGuard`. No sound device is opened and nothing is played.
Because the pace is real, a recogniser that is too slow really is late here, as it would be live.

Ground truth: a text file with the same name next to each clip (`my clip.wav` -> `my clip.txt`),
one line for every time a blocked word is said:

    start_seconds end_seconds word

No file, or an empty one, means "this clip holds no blocked word". Optional `my clip.lrc` (synced
lyrics for that clip, time 0 = start of the clip) switches lyric windows on for that clip.

What is written to --out: `<clip>.filtered.wav` (what OBS would have got, lined up with the
original so the two can be compared by ear), `report.txt` and `report.json`.

The blocked words of a run are --words, or else your own list. Your words are never printed: the
report calls them `word 1`, `word 2` (their line number in the list) unless you pass --show-words.

Exit code: 0 when every expected word was fully silenced, 1 when any was missed or partly played,
2 when the run itself could not be done.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from npass import settings as cfg                                   # noqa: E402
from npass.audio.processor import LookaheadProcessor               # noqa: E402
from npass.detect import Detector, SafetyGuard                     # noqa: E402
from npass.types import (SAMPLE_RATE, CutOrder, CutReason, DangerWindow, Health,   # noqa: E402
                         ms_to_samples)

AUDIO_TYPES = (".wav", ".mp3", ".flac", ".m4a", ".ogg")
BLOCK = 480                    # 10 ms at 48 kHz, the live block size
TAIL_MS = 600                  # silence fed after the clip so the last cut can still arrive
CAUGHT_SHARE = 0.90            # this much of a word's sound energy gone = caught
MISSED_SHARE = 0.10            # less than this gone = missed outright
GUARD_AHEAD_MS = 250           # the safety guard's settings in npass/app.py: keep these the same
GUARD_BRIDGE_MS = 300
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


# ----------------------------------------------------------------------------- reading clips

def load_audio(path: Path) -> np.ndarray:
    """Any clip -> mono float32 at 48 kHz. soundfile first, ffmpeg when soundfile cannot."""
    import soxr
    try:
        import soundfile as sf
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
    except Exception:
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-f", "f32le", "-ac", "1",
               "-ar", str(SAMPLE_RATE), "-"]
        try:
            done = subprocess.run(cmd, capture_output=True, timeout=300, creationflags=_NO_WINDOW)
        except OSError as exc:
            raise RuntimeError(f"could not read {path.name}: ffmpeg is not installed") from exc
        if done.returncode != 0 or not done.stdout:
            raise RuntimeError(f"could not read {path.name}: not a sound file ffmpeg understands")
        mono, rate = np.frombuffer(done.stdout, dtype=np.float32), SAMPLE_RATE
    if rate != SAMPLE_RATE:
        mono = soxr.resample(mono, rate, SAMPLE_RATE)
    return np.ascontiguousarray(mono, dtype=np.float32)


@dataclass(frozen=True)
class Expected:
    start: int            # sample index, 48 kHz, 0 = start of the clip
    end: int
    word: str


def read_truth(path: Path) -> list[Expected]:
    """The sidecar: `start_seconds end_seconds word` per line. Missing or empty = no blocked word."""
    out: list[Expected] = []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return out
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.replace(",", " ").split(None, 2)
        try:
            a, b = float(parts[0]), float(parts[1])
        except (IndexError, ValueError):
            raise ValueError(f"{path.name}, line {n}: expected `start_seconds end_seconds word`")
        if b <= a:
            raise ValueError(f"{path.name}, line {n}: the end time must be after the start time")
        word = parts[2].strip().lower() if len(parts) > 2 else ""
        out.append(Expected(int(round(a * SAMPLE_RATE)), int(round(b * SAMPLE_RATE)), word))
    return sorted(out, key=lambda e: e.start)


def read_windows(path: Path, words: list[str], settings) -> list[DangerWindow] | None:
    """`<clip>.lrc` -> danger windows on the clip's clock. None when there is no lyric file."""
    if not path.is_file():
        return None
    from npass.lyrics.lrclib import parse_lrc
    from npass.lyrics.windows import build_windows
    lines = parse_lrc(path.read_text(encoding="utf-8-sig"))
    return build_windows(lines, words, 0, 0, settings.window_pre_ms, settings.window_post_ms)


# ----------------------------------------------------------------------------- the recogniser

class KeepLoaded:
    """Wraps a real recogniser so that one loaded model serves every clip of a run.

    The Detector closes its recogniser when it stops; loading the GPU model again for each clip
    would only cost time. Everything else goes straight to the real backend.
    """

    def __init__(self, inner):
        self.inner = inner
        self.name = getattr(inner, "name", "custom")
        latency = getattr(inner, "latency_ms", None)
        if latency is not None:
            self.latency_ms = latency
        self._loaded = False

    def load(self) -> None:
        if not self._loaded:
            self.inner.load()
            self._loaded = True

    def transcribe(self, audio16k, start_sample48k, hints):
        return self.inner.transcribe(audio16k, start_sample48k, hints)

    def close(self) -> None:          # the Detector's stop(): keep the model
        pass

    def really_close(self) -> None:
        self._loaded = False
        self.inner.close()


def load_recogniser(settings) -> KeepLoaded:
    """The backend `settings.engine` asks for, loaded once (same choice the live app makes)."""
    from npass.detect import pick_backend
    rec, name = pick_backend(settings)
    keep = KeepLoaded(rec)
    keep.name = name
    keep._loaded = True
    return keep


# ----------------------------------------------------------------------------- one clip

@dataclass
class WordResult:
    word: str
    start_s: float
    end_s: float
    status: str               # "caught" | "part played" | "MISSED"
    removed_share: float      # 0..1 of the word's sound energy that was removed
    played_ms: float          # how long the word was audible (gain above half)
    start_error_ms: float | None = None   # full silence began this long after the word began (<0: early)
    end_error_ms: float | None = None     # full silence ended this long after the word ended (>0: wider)
    heard_after_ms: float | None = None   # first cut order arrived this long after the word began
    late_ms: float = 0.0      # the first cut order arrived this long after its own start had played
    reason: str = ""
    heard: str = ""
    confidence: float = 0.0


@dataclass
class WrongCut:
    start_s: float
    end_s: float
    ms: float
    reason: str
    word: str
    heard: str
    confidence: float


@dataclass
class ClipResult:
    name: str
    seconds: float
    expected: int = 0
    caught: int = 0
    part_played: int = 0
    missed: int = 0
    words: list[WordResult] = field(default_factory=list)
    wrong_cuts: list[WrongCut] = field(default_factory=list)
    safety_mutes: int = 0             # failsafe mutes over a lyric line (no blocked word under them)
    safety_mute_ms: float = 0.0
    late_cuts: int = 0
    worst_late_ms: float = 0.0
    worst_lag_ms: float = 0.0         # newest audio taken in minus what the recogniser had checked
    typical_lag_ms: float = 0.0
    worst_pass_ms: float = 0.0
    unverified_ms: float = 0.0        # audio that played before the recogniser had checked it
    silenced_ms: float = 0.0
    lyric_windows: int = 0
    passes: int = 0
    hops_skipped: int = 0
    crashes: int = 0
    health_end: str = ""
    error: str = ""


@dataclass
class _Sent:
    order: CutOrder
    late_ms: float
    captured: int             # mic clock when the order arrived


def measured_gain(audio: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Gain the filter applied to every sample, read back from the result (out / in).
    Where the input is digital silence the last known gain is carried on."""
    loud = np.abs(audio) > 1e-5
    gain = np.ones(len(audio), dtype=np.float32)
    np.divide(out, audio, out=gain, where=loud)
    idx = np.where(loud, np.arange(len(audio)), -1)
    np.maximum.accumulate(idx, out=idx)
    filled = gain[np.maximum(idx, 0)]
    filled[idx < 0] = 1.0
    return np.clip(filled, 0.0, 1.0)


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(start, end)) of every stretch where mask is True."""
    if not len(mask):
        return []
    d = np.diff(mask.astype(np.int8), prepend=0, append=0)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def run_clip(audio: np.ndarray, truth: list[Expected], settings, words: list[str],
             recogniser=None, windows: list[DangerWindow] | None = None,
             realtime: bool = True, name: str = "clip") -> tuple[ClipResult, np.ndarray]:
    """Feed one clip through processor + detector + guard. Returns (result, filtered audio).

    `realtime=True` is the honest mode: blocks arrive every 10 ms and the detector runs on its own
    thread, as live. `realtime=False` is for tests with a scripted recogniser: no thread, no
    waiting, one recogniser pass allowed after every block.
    The filtered audio is lined up with the input (the look-ahead delay is taken out again).
    """
    audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
    wins = list(windows or [])
    look = ms_to_samples(settings.lookahead_ms)
    proc = LookaheadProcessor(settings.lookahead_ms, settings.fade_ms, settings.removal_mode,
                              settings.dip_db)
    floor = 0.0 if settings.removal_mode != "dip" else float(10.0 ** (settings.dip_db / 20.0))
    sent: list[_Sent] = []
    lock = threading.Lock()

    def submit(order: CutOrder) -> None:
        late = float(proc.submit_cut(order) or 0.0)
        with lock:
            sent.append(_Sent(order, late, proc.captured))

    def provider(a: int, b: int) -> list[DangerWindow]:
        return [w for w in wins if w.start < b and w.end > a]

    det = Detector(settings, lambda: list(words), provider, submit, lambda order: None,
                   recogniser=recogniser)
    guard = SafetyGuard(submit, ahead_ms=GUARD_AHEAD_MS, bridge_ms=GUARD_BRIDGE_MS)   # as npass/app.py

    total = len(audio) + look + ms_to_samples(TAIL_MS)
    total += (-total) % BLOCK
    padded = np.zeros(total, dtype=np.float32)
    padded[:len(audio)] = audio
    out = np.empty(total, dtype=np.float32)
    lags: list[float] = []
    unverified = 0
    worst_pass = 0.0
    health = Health.STARTING

    if realtime:
        det.start()
        deadline = time.monotonic() + 120.0
        while det._rec is None and time.monotonic() < deadline:      # model loading is not the test
            if det.stats["crashes"]:
                break
            time.sleep(0.01)
    t0 = time.perf_counter()
    try:
        for i in range(0, total, BLOCK):
            if realtime:
                wait = t0 + i / SAMPLE_RATE - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
            block = padded[i:i + BLOCK]
            start = proc.captured
            out[i:i + BLOCK] = proc.process(block)
            det.feed(block, start)
            if not realtime:
                try:
                    det.process_pending()
                except Exception as exc:
                    det._crashed(exc)
            playhead, verified = proc.playhead, det.verified_upto()
            health = guard.tick(playhead, verified, det.health(),
                                provider(playhead, proc.captured), captured=proc.captured)
            worst_pass = max(worst_pass, det.last_pass_ms())
            if det.stats["passes"]:
                lags.append((proc.captured - verified) * 1000.0 / SAMPLE_RATE)
            if 0 <= playhead - BLOCK < len(audio) and verified < playhead:
                unverified += min(BLOCK, playhead - verified)
    finally:
        if realtime:
            det.stop()

    filtered = out[look:look + len(audio)].copy()
    res = ClipResult(name=name, seconds=len(audio) / SAMPLE_RATE, expected=len(truth),
                     lyric_windows=len(wins), passes=det.stats["passes"],
                     hops_skipped=det.stats["hops_skipped"], crashes=det.stats["crashes"],
                     health_end=str(health.value), worst_pass_ms=round(worst_pass, 1),
                     unverified_ms=round(unverified * 1000.0 / SAMPLE_RATE, 1))
    if lags:
        res.worst_lag_ms = round(max(lags), 1)
        res.typical_lag_ms = round(float(np.median(lags)), 1)
    _score(res, audio, filtered, truth, sent, floor)
    return res, filtered


def _score(res: ClipResult, audio: np.ndarray, filtered: np.ndarray, truth: list[Expected],
           sent: list[_Sent], floor: float) -> None:
    ms = 1000.0 / SAMPLE_RATE
    gain = measured_gain(audio, filtered)
    full = gain <= floor + 1e-3                       # fully cut (or fully dipped)
    half = gain <= floor + (1.0 - floor) * 0.5        # the fade's half-way point
    silences = runs_of(full)
    res.silenced_ms = round(float(np.count_nonzero(full)) * ms, 1)

    first: dict[int, _Sent] = {}
    last: dict[int, CutOrder] = {}
    for s in sent:
        first.setdefault(s.order.cut_id, s)
        last[s.order.cut_id] = s.order
    late = [s.late_ms for s in first.values() if s.late_ms > 0]
    res.late_cuts = len(late)
    res.worst_late_ms = round(max(late), 1) if late else 0.0

    claimed: set[int] = set()
    for exp in truth:
        a, b = max(0, exp.start), min(len(audio), exp.end)
        energy_in = float(np.sum(audio[a:b].astype(np.float64) ** 2))
        energy_out = float(np.sum(filtered[a:b].astype(np.float64) ** 2))
        removed = 1.0 - energy_out / energy_in if energy_in > 0 else 1.0
        played = float(np.count_nonzero(~half[a:b])) * ms
        overlapping = [k for k, (s, e) in enumerate(silences) if s < b and e > a]
        claimed.update(overlapping)
        orders = [s for s in first.values()
                  if last[s.order.cut_id].start < b and last[s.order.cut_id].end > a]
        orders.sort(key=lambda s: s.captured)
        status = ("caught" if removed >= CAUGHT_SHARE else
                  "MISSED" if removed < MISSED_SHARE else "part played")
        wr = WordResult(word=exp.word, start_s=round(a / SAMPLE_RATE, 3),
                        end_s=round(b / SAMPLE_RATE, 3), status=status,
                        removed_share=round(max(0.0, removed), 4), played_ms=round(played, 1))
        if overlapping:
            wr.start_error_ms = round((silences[overlapping[0]][0] - a) * ms, 1)
            wr.end_error_ms = round((silences[overlapping[-1]][1] - b) * ms, 1)
        if orders:
            o = orders[0]
            best = max((last[s.order.cut_id] for s in orders), key=lambda c: c.confidence)
            wr.heard_after_ms = round((o.captured - a) * ms, 1)
            wr.late_ms = round(o.late_ms, 1)
            wr.reason = str(best.reason.value)
            wr.heard, wr.confidence = best.heard, float(best.confidence)
        res.words.append(wr)
        res.caught += status == "caught"
        res.part_played += status == "part played"
        res.missed += status == "MISSED"

    for k, (s, e) in enumerate(silences):
        if k in claimed:
            continue
        orders = [c for c in last.values() if c.start < e and c.end > s]
        named = [c for c in orders if c.reason != CutReason.FAILSAFE]
        if orders and not named:
            res.safety_mutes += 1
            res.safety_mute_ms = round(res.safety_mute_ms + (e - s) * ms, 1)
            continue
        best = max(named, key=lambda c: c.confidence) if named else None
        res.wrong_cuts.append(WrongCut(
            start_s=round(s / SAMPLE_RATE, 3), end_s=round(e / SAMPLE_RATE, 3),
            ms=round((e - s) * ms, 1),
            reason=str(best.reason.value) if best else "unknown",
            word=best.word if best else "", heard=best.heard if best else "",
            confidence=float(best.confidence) if best else 0.0))


# ----------------------------------------------------------------------------- the report

class Namer:
    """Turns a blocked word into `word 3` unless the user asked to see the words."""

    def __init__(self, words: list[str], show: bool):
        self.words, self.show = list(words), show

    def word(self, w: str) -> str:
        if self.show:
            return f'"{w}"' if w else "(no word given)"
        if w in self.words:
            return f"word {self.words.index(w) + 1}"
        return "a word that is not in the list" if w else "(no word given)"

    def heard(self, text: str) -> str:
        if not text:
            return "nothing"
        return f'"{text}"' if self.show else "hidden (run with --show-words to see it)"


def _fmt_ms(v: float | None) -> str:
    return "n/a" if v is None else f"{v:+.0f} ms"


def clip_lines(r: ClipResult, namer: Namer, lookahead_ms: int) -> list[str]:
    out = [f"{r.name}  ({r.seconds:.1f} s)"]
    if r.error:
        return out + [f"  COULD NOT RUN: {r.error}", ""]
    out.append(f"  expected {r.expected}, caught {r.caught}, part played {r.part_played}, "
               f"MISSED {r.missed}, wrong cuts {len(r.wrong_cuts)}, late cuts {r.late_cuts}")
    for w in r.words:
        line = (f"  {w.status:<11} {namer.word(w.word)} at {w.start_s:.2f}-{w.end_s:.2f} s: "
                f"{w.removed_share * 100:.1f}% of its sound removed, audible for {w.played_ms:.0f} ms")
        out.append(line)
        if w.start_error_ms is not None:
            out.append(f"              silence began {_fmt_ms(w.start_error_ms)} from the word's start "
                       f"(minus = early), ended {_fmt_ms(w.end_error_ms)} from its end (plus = after)")
        if w.heard_after_ms is not None:
            late = f", {w.late_ms:.0f} ms too late" if w.late_ms > 0 else ", in time"
            out.append(f"              cut ordered {w.heard_after_ms:.0f} ms after the word began{late}; "
                       f"reason: {w.reason}; heard {namer.heard(w.heard)}; score {w.confidence:.2f}")
        else:
            out.append("              no cut was ordered for this word")
    for c in r.wrong_cuts:
        out.append(f"  WRONG CUT   {c.start_s:.2f}-{c.end_s:.2f} s ({c.ms:.0f} ms of innocent audio): "
                   f"reason {c.reason}; matched {namer.word(c.word)}; heard {namer.heard(c.heard)}; "
                   f"score {c.confidence:.2f}")
    if r.safety_mutes:
        out.append(f"  safety mutes over lyric lines: {r.safety_mutes} ({r.safety_mute_ms:.0f} ms)")
    out.append(f"  recogniser: worst lag {r.worst_lag_ms:.0f} ms, typical {r.typical_lag_ms:.0f} ms "
               f"(look-ahead {lookahead_ms} ms); slowest check {r.worst_pass_ms:.0f} ms; "
               f"{r.passes} checks, {r.hops_skipped} skipped")
    out.append(f"  played before it was checked: {r.unverified_ms:.0f} ms; "
               f"total silenced: {r.silenced_ms:.0f} ms"
               + (f"; lyric windows: {r.lyric_windows}" if r.lyric_windows else "")
               + (f"; recogniser crashes: {r.crashes}" if r.crashes else ""))
    out.append("")
    return out


def totals_of(results: list[ClipResult]) -> dict:
    ok = [r for r in results if not r.error]
    heard = [w.heard_after_ms for r in ok for w in r.words if w.heard_after_ms is not None]
    return {
        "clips": len(results),
        "clips_failed": len(results) - len(ok),
        "seconds": round(sum(r.seconds for r in ok), 1),
        "expected": sum(r.expected for r in ok),
        "caught": sum(r.caught for r in ok),
        "part_played": sum(r.part_played for r in ok),
        "missed": sum(r.missed for r in ok),
        "wrong_cuts": sum(len(r.wrong_cuts) for r in ok),
        "wrong_cut_ms": round(sum(c.ms for r in ok for c in r.wrong_cuts), 1),
        "safety_mutes": sum(r.safety_mutes for r in ok),
        "late_cuts": sum(r.late_cuts for r in ok),
        "worst_late_ms": max((r.worst_late_ms for r in ok), default=0.0),
        "worst_lag_ms": max((r.worst_lag_ms for r in ok), default=0.0),
        "worst_pass_ms": max((r.worst_pass_ms for r in ok), default=0.0),
        "unverified_ms": round(sum(r.unverified_ms for r in ok), 1),
        "silenced_ms": round(sum(r.silenced_ms for r in ok), 1),
        "heard_after_ms_min": min(heard) if heard else None,
        "heard_after_ms_max": max(heard) if heard else None,
    }


def summary_lines(results: list[ClipResult], totals: dict) -> list[str]:
    head = (f"{'clip':<34} {'exp':>3} {'caught':>6} {'part':>4} {'MISS':>4} {'wrong':>5} "
            f"{'late':>4} {'lag ms':>6}")
    out = ["Summary", head, "-" * len(head)]
    for r in results:
        name = r.name if len(r.name) <= 34 else r.name[:31] + "..."
        if r.error:
            out.append(f"{name:<34} could not run")
            continue
        out.append(f"{name:<34} {r.expected:>3} {r.caught:>6} {r.part_played:>4} {r.missed:>4} "
                   f"{len(r.wrong_cuts):>5} {r.late_cuts:>4} {r.worst_lag_ms:>6.0f}")
    out.append("-" * len(head))
    out.append(f"{'TOTAL':<34} {totals['expected']:>3} {totals['caught']:>6} "
               f"{totals['part_played']:>4} {totals['missed']:>4} {totals['wrong_cuts']:>5} "
               f"{totals['late_cuts']:>4} {totals['worst_lag_ms']:>6.0f}")
    return out


def find_clips(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in AUDIO_TYPES
                  and not p.name.lower().endswith(".filtered.wav"))


def run_folder(folder: Path, settings, words: list[str], out_dir: Path | None,
               recogniser=None, realtime: bool = True, show_words: bool = False,
               log=print) -> dict:
    """Run every clip in `folder`. Returns the report as a dict (also written to `out_dir`)."""
    import soundfile as sf
    namer = Namer(words, show_words)
    clips = find_clips(folder)
    results: list[ClipResult] = []
    lines: list[str] = []
    for path in clips:
        try:
            audio = load_audio(path)
            truth = read_truth(path.with_suffix(".txt"))
            wins = read_windows(path.with_suffix(".lrc"), words, settings)
            res, filtered = run_clip(audio, truth, settings, words, recogniser=recogniser,
                                     windows=wins, realtime=realtime, name=path.name)
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                sf.write(str(out_dir / (path.stem + ".filtered.wav")), filtered, SAMPLE_RATE,
                         subtype="PCM_16")
        except Exception as exc:
            res = ClipResult(name=path.name, seconds=0.0, error=f"{type(exc).__name__}: {exc}")
        results.append(res)
        block = clip_lines(res, namer, settings.lookahead_ms)
        lines += block
        for ln in block:
            log(ln)
    totals = totals_of(results)
    return {"results": results, "totals": totals, "lines": lines,
            "summary": summary_lines(results, totals)}


def gpu_memory() -> dict:
    """GPU memory right now: this process, and the whole card (MB). None where Windows won't say."""
    try:
        from npass.detect.whisper_backend import gpu_total_used_mb, process_gpu_mem_mb
        return {"process_mb": process_gpu_mem_mb(), "card_used_mb": gpu_total_used_mb()}
    except Exception:
        return {"process_mb": None, "card_used_mb": None}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="run_clips", description="Run recorded clips through the N-Pass filter and report.")
    ap.add_argument("folder", help="folder with the clips (and a .txt next to each)")
    ap.add_argument("--lookahead", type=int, default=None, help="look-ahead in ms (350 to 600)")
    ap.add_argument("--engine", choices=["auto", "gpu", "cpu"], default=None)
    ap.add_argument("--words", default=None,
                    help="blocked words for this run, comma separated (default: your own list)")
    ap.add_argument("--out", default=str(ROOT / "out" / "clips"))
    ap.add_argument("--show-words", action="store_true",
                    help="print the blocked words and what was heard (off by default)")
    ap.add_argument("--defaults", action="store_true",
                    help="ignore your saved settings and use the built-in ones")
    args = ap.parse_args(argv)

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"There is no folder called {folder}.")
        return 2
    settings = cfg.Settings() if args.defaults else cfg.load_settings()
    if args.lookahead is not None:
        settings.lookahead_ms = args.lookahead
    if args.engine is not None:
        settings.engine = args.engine
    settings.clamp()
    if args.words is not None:
        words = [w.strip().lower() for w in args.words.split(",") if w.strip()]
    else:
        words = cfg.load_words()
    if not words:
        print("The blocked word list is empty. Add your words in N-Pass (Settings), "
              "or pass --words.")
        return 2
    if not find_clips(folder):
        print(f"No sound files (.wav, .mp3, .flac, .m4a) were found in {folder}.")
        return 2

    try:
        rec = load_recogniser(settings)
    except Exception as exc:
        print(f"The speech model could not be loaded ({type(exc).__name__}: {exc}).")
        print("If this is a new install, run once:  uv run python -m npass.detect download")
        return 2
    gpu_before = gpu_memory()
    head = [
        f"N-Pass clip run, {time.strftime('%Y-%m-%d %H:%M')}",
        f"folder: {folder}",
        f"engine: {rec.name}; look-ahead {settings.lookahead_ms} ms; removal {settings.removal_mode}; "
        f"{len(words)} blocked word{'s' if len(words) != 1 else ''}",
        "pace: real time (a slow recogniser is late here exactly as it would be live)",
        "",
    ]
    for ln in head:
        print(ln)
    out_dir = Path(args.out)
    t0 = time.time()
    try:
        report = run_folder(folder, settings, words, out_dir, recogniser=rec, realtime=True,
                            show_words=args.show_words)
    finally:
        gpu_after = gpu_memory()
        rec.really_close()
    totals = report["totals"]
    foot = list(report["summary"]) + [""]
    if totals["heard_after_ms_min"] is not None:
        foot.append(f"Cuts were ordered {totals['heard_after_ms_min']:.0f} to "
                    f"{totals['heard_after_ms_max']:.0f} ms after the word began "
                    f"(the look-ahead is {settings.lookahead_ms} ms).")
    mem = gpu_after.get("process_mb")
    if "gpu" in rec.name.lower():
        foot.append("GPU memory used by this run: "
                    + (f"{mem:.0f} MB" if mem is not None else "Windows would not say")
                    + (f" (whole card: {gpu_after['card_used_mb']:.0f} MB in use)"
                       if gpu_after.get("card_used_mb") is not None else ""))
    foot.append(f"Filtered clips and this report are in {out_dir}")
    if totals["missed"] or totals["part_played"]:
        foot.append(f"NOT ALL CAUGHT: {totals['missed']} missed, {totals['part_played']} part played.")
    for ln in foot:
        print(ln)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text("\n".join(head + report["lines"] + foot) + "\n",
                                        encoding="utf-8")
    namer = Namer(words, args.show_words)

    def clip_json(r: ClipResult) -> dict:
        d = asdict(r)
        for w in d["words"]:
            w["word"] = namer.word(w["word"])
            w["heard"] = namer.heard(w["heard"])
        for c in d["wrong_cuts"]:
            c["word"] = namer.word(c["word"])
            c["heard"] = namer.heard(c["heard"])
        return d

    body = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"), "folder": str(folder), "engine": rec.name,
        "lookahead_ms": settings.lookahead_ms, "removal_mode": settings.removal_mode,
        "whisper_model": settings.whisper_model, "whisper_compute": settings.whisper_compute,
        "word_count": len(words), "run_seconds": round(time.time() - t0, 1),
        "gpu_before": gpu_before, "gpu_after": gpu_after,
        "totals": totals, "clips": [clip_json(r) for r in report["results"]],
    }
    (out_dir / "report.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
    if totals["clips_failed"]:
        return 2
    return 1 if (totals["missed"] or totals["part_played"]) else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.exit(main())
