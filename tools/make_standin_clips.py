"""Build the stand-in test clips in test_clips/standin/ from real, licensed human speech.

    uv run python tools/make_standin_clips.py            (needs the internet once, and the models)

Why stand-ins: the tests and `tools/run_clips.py` need clips before the user has recorded any, and
nothing in this project may ever generate speech. So these clips are cut from LibriSpeech
(read audiobooks from LibriVox volunteers, licence CC BY 4.0, https://www.openslr.org/12), picked
because the reader says one of the harmless stand-in words `pineapple` / `banana`.

What this script does, per utterance in SOURCES:
1. Downloads the one utterance (16 kHz FLAC) through the Hugging Face dataset viewer and checks
   that its id and transcript are the ones listed here. Kept in out/standin-cache/ (not in git).
2. Finds when the stand-in word is said, with two careful offline passes over the whole
   utterance: faster-whisper small.en with beam 5 and word times, and Vosk's word times. The
   dataset's own transcript says how many times the word must be there; anything else stops the
   build. The word's span is the earliest start and the latest end of the two passes, and the
   loudness of that span is checked against the clip's quiet level (a span with no voice in it
   stops the build too).
3. Cuts a short clip around the word (on quiet spots, with 10 ms fades) and writes it with its
   `.txt` sidecar: `start_seconds end_seconds word`.
4. Makes "sung-like" copies in which only the word is stretched to 1.5x, 2x or 3x its length with
   a phase vocoder (the same recorded voice, slowed down; nothing is synthesised).
5. Adds clips WITHOUT any stand-in word (empty sidecar), some with sound-alike words in them.
6. Writes SOURCES.md: every clip, its source utterance, licence, and what was changed.

Clips are stored as 16 kHz mono FLAC, which keeps the folder small. `run_clips.py` brings every
clip to 48 kHz itself.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RATE = 16_000
OUT_DIR = ROOT / "test_clips" / "standin"
CACHE_DIR = ROOT / "out" / "standin-cache"
MODELS_DIR = ROOT / "models"
DATASET = "openslr/librispeech_asr"
ROWS_URL = "https://datasets-server.huggingface.co/rows"
WORDS = ("pineapple", "banana")
LEAD_S = 3.5              # speech kept before the word (the recogniser works on about 3 s)
TRAIL_S = 1.8             # and after it
CLEAN_S = 9.0             # length of a clip without a stand-in word
STRETCHES = (1.5, 2.0, 3.0)
DISAGREE_LIMIT_S = 0.20   # the two offline passes must agree this closely on each edge


@dataclass(frozen=True)
class Source:
    key: str              # clip name stem
    utt: str              # LibriSpeech utterance id: speaker-chapter-number
    config: str
    split: str
    row: int              # row number in that split on the dataset viewer
    word: str = ""        # stand-in word said in it ("" = a clean clip)
    pick: int = 0         # which occurrence of the word (0 = first)
    stretch: tuple[float, ...] = ()
    note: str = ""


# Found with a transcript search of the dataset viewer on 2026-11-23. The viewer only searches the
# first part of each split, which is why most of the "banana" clips are one reader (speaker 460).
_T100, _T360, _VAL = ("clean", "train.100"), ("clean", "train.360"), ("clean", "validation")
SOURCES: list[Source] = [
    Source("spoken_banana_01", "460-172359-0062", *_T100, 12390, "banana", 0, (2.0,)),
    Source("spoken_banana_02", "460-172359-0062", *_T100, 12390, "banana", 1),
    Source("spoken_banana_03", "460-172359-0070", *_T100, 12398, "banana", 0, (),
           "the word is the very first thing said"),
    Source("spoken_banana_04", "460-172359-0088", *_T100, 12416, "banana", 0, (1.5, 3.0)),
    Source("spoken_banana_05", "460-172359-0078", *_T100, 12406, "banana", 0),
    Source("spoken_banana_06", "460-172357-0012", *_T100, 12314, "banana", 0, (1.5,)),
    Source("spoken_banana_07", "460-172359-0067", *_T100, 12395, "banana", 0),
    Source("spoken_banana_08", "1556-139350-0006", *_T360, 19866, "banana", 0, (2.0, 3.0)),
    Source("spoken_pineapple_01", "7226-86964-0047", *_T100, 11964, "pineapple", 0, (1.5, 2.0, 3.0)),
    Source("clean_apple_01", "445-123857-0008", *_T100, 15970, note="says \"apple\""),
    Source("clean_apple_02", "4214-7146-0027", *_T100, 1627, note="says \"apples\""),
    Source("clean_apple_03", "2790-142824-0040", *_T360, 20522, note="says \"apple tart\""),
    Source("clean_apple_04", "7976-110124-0013", *_VAL, 240, note="says \"apples\" and \"apple\""),
    Source("clean_apple_05", "1472-142848-0003", *_T360, 9874, note="says \"apple and pear juice\""),
    Source("clean_plain_01", "460-172359-0063", *_T100, 12391,
           note="same reader as most banana clips, no fruit"),
    Source("clean_plain_02", "7226-86964-0048", *_T100, 11965,
           note="same reader as the pineapple clip, no fruit"),
]


@dataclass
class Made:
    name: str
    source: Source
    seconds: float
    words: list[tuple[float, float, str]] = field(default_factory=list)
    change: str = ""
    detail: str = ""


# ----------------------------------------------------------------------------- download

def fetch(src: Source) -> tuple[np.ndarray, str]:
    """(audio at 16 kHz, transcript) of one utterance; downloaded once, then read from the cache."""
    import soundfile as sf
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    flac, meta = CACHE_DIR / f"{src.utt}.flac", CACHE_DIR / f"{src.utt}.json"
    if not (flac.is_file() and meta.is_file()):
        url = ROWS_URL + "?" + urllib.parse.urlencode(
            {"dataset": DATASET, "config": src.config, "split": src.split,
             "offset": src.row, "length": 1})
        row = None
        for attempt in range(8):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    row = json.load(r)["rows"][0]["row"]
                break
            except Exception:
                time.sleep(5 + 5 * attempt)
        if row is None:
            raise RuntimeError(f"could not reach the dataset viewer for {src.utt}")
        if row["id"] != src.utt:
            raise RuntimeError(f"row {src.row} of {src.config}/{src.split} is {row['id']}, "
                               f"not {src.utt}: the dataset viewer's order changed")
        req = urllib.request.Request(row["audio"][0]["src"], headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=120) as r:
            flac.write_bytes(r.read())
        meta.write_text(json.dumps({"id": row["id"], "text": row["text"],
                                    "speaker_id": row["speaker_id"],
                                    "chapter_id": row["chapter_id"]}), encoding="utf-8")
    audio, rate = sf.read(str(flac), dtype="float32", always_2d=True)
    if rate != RATE:
        raise RuntimeError(f"{src.utt}: expected {RATE} Hz, got {rate}")
    return np.ascontiguousarray(audio.mean(axis=1)), json.loads(meta.read_text("utf-8"))["text"]


# ----------------------------------------------------------------------------- word times

def _norm(w: str) -> str:
    return "".join(c for c in w.lower() if c.isalpha())


class Aligners:
    """The two offline passes. Loaded once."""

    def __init__(self, device: str = "cuda"):
        from faster_whisper import WhisperModel

        from npass.detect import vosk_backend
        from npass.detect.whisper_backend import add_cuda_dll_dirs
        if device == "cuda":
            add_cuda_dll_dirs()
        self.whisper = WhisperModel("small.en", device=device,
                                    compute_type="float16" if device == "cuda" else "int8",
                                    download_root=str(MODELS_DIR), local_files_only=True)
        import vosk
        vosk.SetLogLevel(-1)
        self._vosk = vosk
        self.vosk_model = vosk.Model(str(MODELS_DIR / vosk_backend.MODEL_NAME))

    def whisper_words(self, audio: np.ndarray) -> list[tuple[str, float, float]]:
        segments, _ = self.whisper.transcribe(audio, language="en", beam_size=5,
                                              word_timestamps=True, vad_filter=False,
                                              condition_on_previous_text=False)
        return [(_norm(w.word), float(w.start), float(w.end)) for s in segments for w in s.words]

    def vosk_words(self, audio: np.ndarray) -> list[tuple[str, float, float]]:
        rec = self._vosk.KaldiRecognizer(self.vosk_model, RATE)
        rec.SetWords(True)
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
        out: list[tuple[str, float, float]] = []

        def take(raw: str) -> None:
            for it in json.loads(raw).get("result", []):
                out.append((_norm(it["word"]), float(it["start"]), float(it["end"])))
        for i in range(0, len(pcm), 8000):
            if rec.AcceptWaveform(pcm[i:i + 8000]):
                take(rec.Result())
        take(rec.FinalResult())
        return out


def _occurrences(words: list[tuple[str, float, float]], stem: str) -> list[tuple[float, float]]:
    out = []
    for k, (w, a, b) in enumerate(words):
        if w.startswith(stem):
            out.append((a, b))
        elif stem == "pineapple" and w == "pine" and k + 1 < len(words) \
                and words[k + 1][0].startswith("apple"):
            out.append((a, words[k + 1][2]))          # heard as two words
    return out


def rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-9)


def find_word(al: Aligners, audio: np.ndarray, transcript: str, src: Source) -> tuple[float, float, str]:
    """(start_s, end_s, how it was found) of the chosen occurrence of the stand-in word."""
    expected = sum(1 for w in transcript.lower().split() if _norm(w).startswith(src.word))
    if expected == 0:
        raise RuntimeError(f"{src.utt}: the dataset transcript has no '{src.word}'")
    a = _occurrences(al.whisper_words(audio), src.word)
    b = _occurrences(al.vosk_words(audio), src.word)
    if len(a) != expected:
        raise RuntimeError(f"{src.utt}: the transcript has {expected} x '{src.word}', "
                           f"the faster-whisper pass found {len(a)}")
    wa = a[src.pick]
    if len(b) == expected:
        wb = b[src.pick]
        if abs(wa[0] - wb[0]) > DISAGREE_LIMIT_S or abs(wa[1] - wb[1]) > DISAGREE_LIMIT_S:
            raise RuntimeError(f"{src.utt}: the two passes disagree on '{src.word}': "
                               f"{wa[0]:.2f}-{wa[1]:.2f} s against {wb[0]:.2f}-{wb[1]:.2f} s")
        start, end = min(wa[0], wb[0]), max(wa[1], wb[1])
        how = (f"faster-whisper {wa[0]:.2f}-{wa[1]:.2f} s, Vosk {wb[0]:.2f}-{wb[1]:.2f} s "
               f"(in the source utterance)")
    else:
        start, end = wa
        how = (f"faster-whisper {wa[0]:.2f}-{wa[1]:.2f} s; Vosk did not return the word "
               f"(found {len(b)} of {expected}), so its times could not be compared")
    # ear-free check: there must be a voice in that span
    span = audio[int(start * RATE):int(end * RATE)]
    frames = audio[:len(audio) // 320 * 320].reshape(-1, 320)
    quiet = float(np.percentile(20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-9), 5))
    loud = rms_db(span)
    if loud < quiet + 10.0:
        raise RuntimeError(f"{src.utt}: no voice in the span found for '{src.word}' "
                           f"({loud:.0f} dB, quiet level {quiet:.0f} dB)")
    how += f"; span loudness {loud:.0f} dB against a quiet level of {quiet:.0f} dB"
    return start, end, how


# ----------------------------------------------------------------------------- cutting

def quiet_spot(audio: np.ndarray, at_s: float, reach_s: float = 0.35) -> int:
    """Sample index of the quietest 20 ms within `reach_s` of `at_s`."""
    centre = int(at_s * RATE)
    lo, hi = max(0, centre - int(reach_s * RATE)), min(len(audio), centre + int(reach_s * RATE))
    frame = 320
    if hi - lo < 2 * frame:
        return min(max(centre, 0), len(audio))
    best, best_v = centre, None
    for i in range(lo, hi - frame, frame // 2):
        v = float(np.mean(audio[i:i + frame] ** 2))
        if best_v is None or v < best_v:
            best, best_v = i + frame // 2, v
    return best


def excerpt(audio: np.ndarray, start_s: float, end_s: float) -> tuple[np.ndarray, int]:
    """Cut [start, end] on quiet spots, 10 ms fades at both ends. Returns (clip, first sample)."""
    a = 0 if start_s <= 0.2 else quiet_spot(audio, start_s)
    b = len(audio) if end_s >= len(audio) / RATE - 0.2 else quiet_spot(audio, end_s)
    clip = audio[a:b].copy()
    n = int(0.010 * RATE)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    clip[:n] *= ramp
    clip[-n:] *= ramp[::-1]
    return clip, a


def phase_vocoder(x: np.ndarray, factor: float, n_fft: int = 512, hop: int = 128) -> np.ndarray:
    """Make `x` `factor` times longer without changing its pitch (classic phase vocoder)."""
    win = np.hanning(n_fft + 1)[:-1]
    pad = np.concatenate([np.zeros(n_fft), x.astype(np.float64), np.zeros(n_fft)])
    count = 1 + (len(pad) - n_fft) // hop
    spec = np.stack([np.fft.rfft(win * pad[i * hop:i * hop + n_fft]) for i in range(count)])
    steps = np.arange(0, count - 1, 1.0 / factor)
    omega = 2 * np.pi * hop * np.arange(n_fft // 2 + 1) / n_fft
    phase = np.angle(spec[0])
    out = np.zeros(len(steps) * hop + n_fft)
    norm = np.zeros_like(out)
    for k, t in enumerate(steps):
        i = int(t)
        frac = t - i
        mag = (1 - frac) * np.abs(spec[i]) + frac * np.abs(spec[i + 1])
        frame = np.fft.irfft(mag * np.exp(1j * phase), n_fft)
        out[k * hop:k * hop + n_fft] += win * frame
        norm[k * hop:k * hop + n_fft] += win ** 2
        delta = np.angle(spec[i + 1]) - np.angle(spec[i]) - omega
        delta -= 2 * np.pi * np.round(delta / (2 * np.pi))
        phase = phase + omega + delta
    out /= np.maximum(norm, 1e-3)
    start = int(n_fft * factor)
    return out[start:start + int(len(x) * factor)].astype(np.float32)


def stretch_word(clip: np.ndarray, ws: int, we: int, factor: float) -> tuple[np.ndarray, int, int]:
    """Stretch only samples [ws, we) of the clip. Returns (new clip, new word start, new word end)."""
    margin = int(0.030 * RATE)                       # a little of each neighbour goes with it
    a, b = max(0, ws - margin), min(len(clip), we + margin)
    ctx = 512                                        # real audio around it, so the edges are clean
    a2, b2 = max(0, a - ctx), min(len(clip), b + ctx)
    long = phase_vocoder(clip[a2:b2], factor)
    lead = int((a - a2) * factor)
    mid = long[lead:lead + int((b - a) * factor)]
    peak_in = float(np.max(np.abs(clip[a:b]))) or 1.0
    peak_out = float(np.max(np.abs(mid))) or 1.0
    mid = mid * min(1.0, peak_in / peak_out)
    fade = int(0.008 * RATE)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    head, tail = clip[:a].copy(), clip[b:].copy()
    mid = mid.copy()
    mid[:fade] = mid[:fade] * ramp + clip[a:a + fade] * (1 - ramp)
    mid[-fade:] = mid[-fade:] * (1 - ramp) + clip[b - fade:b] * ramp
    out = np.concatenate([head, mid, tail]).astype(np.float32)
    new_ws = a + int((ws - a) * factor)
    new_we = a + len(mid) - int((b - we) * factor)
    return out, new_ws, new_we


# ----------------------------------------------------------------------------- build

def write_clip(name: str, clip: np.ndarray, words: list[tuple[float, float, str]]) -> None:
    import soundfile as sf
    peak = float(np.max(np.abs(clip)))
    if peak > 0.98:
        clip = clip * (0.98 / peak)
    sf.write(str(OUT_DIR / f"{name}.flac"), clip, RATE, format="FLAC", subtype="PCM_16")
    lines = [f"{a:.3f} {b:.3f} {w}" for a, b, w in words]
    (OUT_DIR / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def build(device: str) -> list[Made]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in list(OUT_DIR.glob("*.flac")) + list(OUT_DIR.glob("*.txt")):
        old.unlink()
    al = Aligners(device)
    made: list[Made] = []
    for src in SOURCES:
        audio, transcript = fetch(src)
        if not src.word:
            if any(_norm(w).startswith(WORDS) for w in transcript.lower().split()):
                raise RuntimeError(f"{src.utt}: listed as clean but its transcript has a stand-in word")
            heard = [w for w, _, _ in al.whisper_words(audio)]
            if any(w.startswith(WORDS) for w in heard):
                raise RuntimeError(f"{src.utt}: listed as clean but the offline pass heard a stand-in word")
            clip, _ = excerpt(audio, 0.0, min(CLEAN_S, len(audio) / RATE))
            write_clip(src.key, clip, [])
            made.append(Made(src.key, src, len(clip) / RATE, [], "trimmed" if len(clip) < len(audio)
                             else "unchanged", src.note))
            print(f"{src.key}: clean, {len(clip) / RATE:.1f} s")
            continue
        start, end, how = find_word(al, audio, transcript, src)
        clip, first = excerpt(audio, start - LEAD_S, end + TRAIL_S)
        ws, we = int(start * RATE) - first, int(end * RATE) - first
        others = [(a, b) for a, b in _occurrences(al.whisper_words(audio), src.word)
                  if (a, b) != (start, end) and b * RATE > first and a * RATE < first + len(clip)
                  and abs(a - start) > 0.3]
        if others:
            raise RuntimeError(f"{src.utt}: a second '{src.word}' falls inside the excerpt")
        write_clip(src.key, clip, [(ws / RATE, we / RATE, src.word)])
        made.append(Made(src.key, src, len(clip) / RATE, [(ws / RATE, we / RATE, src.word)],
                         "trimmed to the stretch around the word", how))
        print(f"{src.key}: '{src.word}' at {ws / RATE:.2f}-{we / RATE:.2f} s of {len(clip) / RATE:.1f} s")
        for factor in src.stretch:
            long, s2, e2 = stretch_word(clip, ws, we, factor)
            name = f"{src.key.replace('spoken', 'stretched')}_x{factor:g}".replace(".", "_")
            write_clip(name, long, [(s2 / RATE, e2 / RATE, src.word)])
            made.append(Made(name, src, len(long) / RATE, [(s2 / RATE, e2 / RATE, src.word)],
                             f"trimmed; the word alone stretched to {factor:g}x its length "
                             f"(phase vocoder), {(we - ws) / RATE * 1000:.0f} ms -> "
                             f"{(e2 - s2) / RATE * 1000:.0f} ms", how))
            print(f"  {name}: word now {(e2 - s2) / RATE * 1000:.0f} ms long")
    return made


def write_sources(made: list[Made]) -> None:
    size = sum(p.stat().st_size for p in OUT_DIR.glob("*.flac"))
    out = [
        "# Where the stand-in clips come from",
        "",
        "Every clip in this folder is real human speech from **LibriSpeech**",
        "(Vassil Panayotov, Guoguo Chen, Daniel Povey, Sanjeev Khudanpur: \"LibriSpeech: an ASR corpus",
        "based on public domain audio books\", ICASSP 2015), https://www.openslr.org/12 ,",
        "licence **CC BY 4.0** (https://creativecommons.org/licenses/by/4.0/). LibriSpeech is made",
        "from audiobooks read by LibriVox volunteers. No clip here was made with text-to-speech or any",
        "other synthetic voice.",
        "",
        "The clips were fetched on 2026-11-23 through the Hugging Face copy of the corpus",
        "(`openslr/librispeech_asr`). An utterance id reads speaker-chapter-number.",
        "",
        "Changes made to the originals (CC BY asks that changes are stated): each clip is a cut-out",
        "of one utterance with 10 ms fades at both ends; clips named `stretched_...` also have the one",
        "stand-in word slowed down with a phase vocoder, to stand in for a word that is held while",
        "singing. They are not singing, and a stretched word sounds processed.",
        "",
        "The stand-in \"blocked\" words are `pineapple` and `banana`. The times in the `.txt` files come",
        "from two offline passes over the whole source utterance (faster-whisper small.en with beam 5,",
        "and Vosk), cross-checked against the dataset's transcript and a loudness check. Nobody",
        "listened to them. Expect the edges to be right to about 50 ms, not better.",
        "",
        f"Total size of the clips: {size / 1e6:.1f} MB. Rebuild with `uv run python tools/make_standin_clips.py`.",
        "",
        "| clip | source utterance | split | length | stand-in word at | what was changed |",
        "|---|---|---|---|---|---|",
    ]
    for m in made:
        at = ", ".join(f"{a:.2f}-{b:.2f} s ({w})" for a, b, w in m.words) or "none (clean clip)"
        out.append(f"| `{m.name}.flac` | {m.source.utt} | {m.source.config}/{m.source.split} | "
                   f"{m.seconds:.1f} s | {at} | {m.change} |")
    out += ["", "## How each word was located", ""]
    seen = set()
    for m in made:
        if m.words and m.source.key not in seen:
            seen.add(m.source.key)
            out.append(f"- `{m.source.key}` ({m.source.utt}): {m.detail}")
    notes = [m for m in made if not m.words and m.detail]
    if notes:
        out += ["", "## Clean clips", ""]
        out += [f"- `{m.name}` ({m.source.utt}): {m.detail}" for m in notes]
    (OUT_DIR / "SOURCES.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build test_clips/standin/ from licensed human speech.")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                    help="where the offline faster-whisper pass runs")
    args = ap.parse_args(argv)
    if not SOURCES:
        print("No source utterances are listed.")
        return 2
    made = build(args.device)
    write_sources(made)
    size = sum(p.stat().st_size for p in OUT_DIR.glob("*.flac"))
    print(f"{len(made)} clips, {size / 1e6:.1f} MB, in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
