"""Draws the pictures the README shows (docs/img/*.svg) from real data. No drawing libraries needed.

    uv run python tools/make_readme_art.py

- banner.svg     the title card: a voice wave with one word cut out
- waveform.svg   a real stand-in clip before and after the filter (needs a run of tools/run_clips.py)
- results.svg    the measured stand-in results (numbers typed in below: keep them in step with the README)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "img"

BG, PANEL, LINE = "#14161c", "#1c1f27", "#2b303b"
TEXT, DIM = "#e8eaf0", "#8b93a5"
GREEN, AMBER, RED, GREY = "#2ecc71", "#f0b429", "#ef5350", "#566074"
FONT = "font-family=\"Segoe UI, system-ui, -apple-system, Helvetica, Arial, sans-serif\""

# measured 2026-11-23, GPU, look-ahead 550 ms, 9 words each (see README "What has been measured")
RESULTS = [
    ("Spoken at normal speed", 9, 0, 0),
    ("Held long (slowed down)", 5, 4, 0),
]
CLIP = "spoken_banana_01"
RUN = ROOT / "out" / "clips" / "hold-gpu-550"


def envelope(x: np.ndarray, cols: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x) // cols
    x = x[: n * cols].reshape(cols, n)
    return x.min(axis=1), x.max(axis=1)


def wave_path(lo: np.ndarray, hi: np.ndarray, x0: float, y_mid: float, w: float, h: float) -> str:
    cols = len(lo)
    xs = [x0 + w * i / (cols - 1) for i in range(cols)]
    top = " ".join(f"{x:.1f},{y_mid - h * v:.1f}" for x, v in zip(xs, hi))
    bot = " ".join(f"{x:.1f},{y_mid - h * v:.1f}" for x, v in zip(reversed(xs), reversed(lo)))
    return f"M {top} L {bot} Z"


def fake_voice(cols: int, seed: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """A speech-shaped envelope for the banner (decoration, not data)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, cols)
    syll = np.abs(np.sin(t * 34)) ** 1.5 * (0.55 + 0.45 * np.sin(t * 7 + 1) ** 2)
    hi = np.clip(syll * (0.6 + 0.4 * rng.random(cols)), 0.02, 1)
    return -hi * (0.8 + 0.2 * rng.random(cols)), hi


def banner() -> str:
    W, H = 1200, 300
    cols = 420
    lo, hi = fake_voice(cols)
    a, b = int(cols * 0.56), int(cols * 0.70)          # the cut word
    fade = 10
    gain = np.ones(cols)
    gain[a:b] = 0
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))
    gain[a - fade:a] = 1 - ramp
    gain[b:b + fade] = ramp
    x0, w, y, h = 60, W - 120, 205, 52
    xa, xb = x0 + w * a / cols, x0 + w * b / cols
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="N-Pass: a voice wave with one word cut out">
  <rect width="{W}" height="{H}" rx="18" fill="{BG}"/>
  <text x="60" y="92" {FONT} font-size="64" font-weight="700" fill="{TEXT}">N-Pass</text>
  <text x="62" y="130" {FONT} font-size="22" fill="{DIM}">Cuts your blocked words out of your stream mic before OBS hears them.</text>
  <path d="{wave_path(lo, hi, x0, y, w, h)}" fill="{GREY}" opacity="0.35"/>
  <path d="{wave_path(lo * gain, hi * gain, x0, y, w, h)}" fill="{GREEN}"/>
  <line x1="{xa:.1f}" y1="{y - h - 8}" x2="{xa:.1f}" y2="{y + h + 8}" stroke="{LINE}" stroke-width="2" stroke-dasharray="4 5"/>
  <line x1="{xb:.1f}" y1="{y - h - 8}" x2="{xb:.1f}" y2="{y + h + 8}" stroke="{LINE}" stroke-width="2" stroke-dasharray="4 5"/>
  <text x="{(xa + xb) / 2:.1f}" y="{y + h + 30}" {FONT} font-size="15" fill="{DIM}" text-anchor="middle">one word, silenced with 15 ms fades</text>
</svg>
"""


def social() -> str:
    """1280x640 card for GitHub's "Social preview" (rendered to PNG by hand, see the README art notes)."""
    W, H = 1280, 640
    cols = 420
    lo, hi = fake_voice(cols)
    a, b = int(cols * 0.56), int(cols * 0.70)
    fade = 10
    gain = np.ones(cols)
    gain[a:b] = 0
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))
    gain[a - fade:a] = 1 - ramp
    gain[b:b + fade] = ramp
    x0, w, y, h = 90, W - 180, 430, 95
    xa, xb = x0 + w * a / cols, x0 + w * b / cols
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
  <rect width="{W}" height="{H}" fill="{BG}"/>
  <text x="90" y="170" {FONT} font-size="112" font-weight="700" fill="{TEXT}">N-Pass</text>
  <text x="94" y="232" {FONT} font-size="34" fill="{DIM}">Cuts your blocked words out of your stream mic</text>
  <text x="94" y="276" {FONT} font-size="34" fill="{DIM}">before OBS hears them. No beep. Runs on your PC.</text>
  <path d="{wave_path(lo, hi, x0, y, w, h)}" fill="{GREY}" opacity="0.35"/>
  <path d="{wave_path(lo * gain, hi * gain, x0, y, w, h)}" fill="{GREEN}"/>
  <line x1="{xa:.1f}" y1="{y - h - 12}" x2="{xa:.1f}" y2="{y + h + 12}" stroke="{LINE}" stroke-width="3" stroke-dasharray="5 6"/>
  <line x1="{xb:.1f}" y1="{y - h - 12}" x2="{xb:.1f}" y2="{y + h + 12}" stroke="{LINE}" stroke-width="3" stroke-dasharray="5 6"/>
  <text x="{(xa + xb) / 2:.1f}" y="{y + h + 48}" {FONT} font-size="24" fill="{DIM}" text-anchor="middle">one word, silenced</text>
</svg>
"""


def waveform() -> str | None:
    src = ROOT / "test_clips" / "standin" / f"{CLIP}.flac"
    flt = RUN / f"{CLIP}.filtered.wav"
    side = src.with_suffix(".txt")
    if not (src.exists() and flt.exists() and side.exists()):
        return None
    a, sr_a = sf.read(src, dtype="float32")
    b, sr_b = sf.read(flt, dtype="float32")
    w0, w1 = (float(v) for v in side.read_text(encoding="utf-8").split()[:2])
    t0, t1 = max(0.0, w0 - 1.3), w1 + 1.3
    a = a[int(t0 * sr_a): int(t1 * sr_a)]
    b = b[int(t0 * sr_b): int(t1 * sr_b)]
    peak = float(max(np.abs(a).max(), 1e-6))
    cols = 520
    W, H = 1200, 430
    x0, w, h = 210, W - 270, 62
    rows = [("Your mic", a / peak, GREY, 150), ("What OBS gets", b / peak, GREEN, 310)]
    xa = x0 + w * (w0 - t0) / (t1 - t0)
    xb = x0 + w * (w1 - t0) / (t1 - t0)
    parts = [f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="A real test clip before and after N-Pass: the blocked stand-in word is silent in the output">
  <rect width="{W}" height="{H}" rx="18" fill="{BG}"/>
  <text x="60" y="52" {FONT} font-size="22" font-weight="600" fill="{TEXT}">A real test clip, before and after</text>
  <text x="60" y="78" {FONT} font-size="15" fill="{DIM}">Audiobook reader saying the stand-in word "banana" ({(w1 - w0) * 1000:.0f} ms). GPU, look-ahead 550 ms. Drawn from the actual files.</text>
  <rect x="{xa:.1f}" y="96" width="{xb - xa:.1f}" height="290" fill="{AMBER}" opacity="0.10"/>
  <text x="{(xa + xb) / 2:.1f}" y="408" {FONT} font-size="14" fill="{AMBER}" text-anchor="middle">the blocked word</text>"""]
    for label, x, colour, y in rows:
        lo, hi = envelope(x, cols)
        parts.append(f'  <text x="60" y="{y + 5}" {FONT} font-size="16" fill="{DIM}">{label}</text>')
        parts.append(f'  <line x1="{x0}" y1="{y}" x2="{x0 + w}" y2="{y}" stroke="{LINE}" stroke-width="1"/>')
        parts.append(f'  <path d="{wave_path(lo, hi, x0, y, w, h)}" fill="{colour}"/>')
    parts.append("</svg>\n")
    return "\n".join(parts)


def results() -> str:
    W, H = 1200, 300
    x0, w, bar_h = 330, 760, 34
    parts = [f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Measured results on stand-in clips: spoken words 9 of 9 caught; held words 5 caught, 4 partly heard">
  <rect width="{W}" height="{H}" rx="18" fill="{BG}"/>
  <text x="60" y="52" {FONT} font-size="22" font-weight="600" fill="{TEXT}">Measured on stand-in clips (2026-11-23)</text>
  <text x="60" y="78" {FONT} font-size="15" fill="{DIM}">9 words per row, GPU, look-ahead 550 ms. Real speech with harmless stand-in words. Not real singing.</text>"""]
    y = 120
    for label, caught, part, missed in RESULTS:
        total = caught + part + missed
        parts.append(f'  <text x="60" y="{y + 23}" {FONT} font-size="17" fill="{TEXT}">{label}</text>')
        x = x0
        for n, colour, name in ((caught, GREEN, "caught"), (part, AMBER, "partly heard"), (missed, RED, "missed")):
            if not n:
                continue
            seg = w * n / total
            parts.append(f'  <rect x="{x:.1f}" y="{y}" width="{seg - 3:.1f}" height="{bar_h}" rx="5" fill="{colour}"/>')
            parts.append(f'  <text x="{x + seg / 2:.1f}" y="{y + 23}" {FONT} font-size="15" font-weight="600" fill="{BG}" text-anchor="middle">{n} {name}</text>')
            x += seg
        y += 62
    parts.append(f'  <text x="60" y="{y + 16}" {FONT} font-size="15" fill="{DIM}">"Caught" = at least 90% of the word removed. Held words are the weak spot: the start of a long held word can still be heard.</text>')
    parts.append("</svg>\n")
    return "\n".join(parts)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "banner.svg").write_bytes(banner().encode("utf-8"))
    (OUT / "results.svg").write_bytes(results().encode("utf-8"))
    (OUT / "social-preview.svg").write_bytes(social().encode("utf-8"))
    wf = waveform()
    if wf is None:
        print("waveform.svg skipped: run tools/run_clips.py on test_clips/standin first "
              f"(looked in {RUN})")
    else:
        (OUT / "waveform.svg").write_bytes(wf.encode("utf-8"))
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.svg"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
