"""Settings, data folder and the blocked word list. Owned by the lead: do not edit in a station
(ask for a new field in your report instead; read unknown needs with getattr defaults meanwhile).

Everything the user changes lives in %LOCALAPPDATA%\\NPass, outside the code folder.
Tests MUST set NPASS_DATA_DIR to a temp folder so they never touch the real one.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

LOOKAHEAD_MIN_MS = 350
LOOKAHEAD_MAX_MS = 600
MAX_TOTAL_DELAY_MS = 700     # hard cap on the measured end-to-end delay


def data_dir() -> Path:
    override = os.environ.get("NPASS_DATA_DIR")
    if override:
        p = Path(override)
    else:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        p = Path(base) / "NPass"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class Settings:
    # audio
    input_device: str = ""              # "" = pick by `input_hint`
    input_hint: str = "NVIDIA Broadcast"
    output_hint: str = "CABLE Input"    # VB-CABLE's playback end; OBS reads "CABLE Output"
    lookahead_ms: int = 550             # measured 2026-11-23: 450 left some cuts up to 98 ms late
    block_ms: int = 10
    fifo_target_ms: float = 30      # small cushion between the mic's clock and the cable's clock
    # removal
    fade_ms: int = 15                   # 10..20
    removal_mode: str = "silence"       # "silence" | "dip"
    dip_db: float = -30.0               # level kept in "dip" mode
    # detection
    enabled: bool = True
    engine: str = "auto"                # "auto" | "gpu" | "cpu"
    whisper_model: str = "small.en"
    whisper_compute: str = "float16"    # steadier pass time than int8_float16 (about 95 ms, 814 MB)
    hotwords_in_windows: bool = False   # off: hints made the recogniser invent or lose words in tests
    hotwords_always: bool = False
    silence_gate_db: float = -50.0      # quieter than this = nothing to recognise
    gpu_mem_limit_mb: int = 1500
    hop_ms: int = 80                    # how often the recogniser runs
    context_s: float = 3.0              # audio the recogniser sees per pass
    threshold: float = 0.72             # match score needed outside a danger window
    window_threshold: float = 0.50      # inside a danger window
    cut_pad_ms: int = 40                # extra silence each side of the word
    max_hold_ms: float = 1500.0         # a held (sung) word: keep cutting this long at most, until the voice dips
    mute_risky_lines: bool = False      # True: silence every whole lyric line that contains a blocked word
    # lyrics (all optional)
    lyrics_enabled: bool = True
    spotify_client_id: str = ""         # public id of the user's Spotify developer app (not a secret)
    spotify_redirect_port: int = 8898
    window_pre_ms: int = 500            # danger window opens this early
    window_post_ms: int = 700           # and closes this late
    # app
    hotkey: str = "ctrl+alt+n"
    obs_ws_host: str = "127.0.0.1"
    obs_ws_port: int = 4455

    def clamp(self) -> "Settings":
        self.lookahead_ms = max(LOOKAHEAD_MIN_MS, min(LOOKAHEAD_MAX_MS, int(self.lookahead_ms)))
        self.fade_ms = max(10, min(20, int(self.fade_ms)))
        if self.removal_mode not in ("silence", "dip"):
            self.removal_mode = "silence"
        if self.engine not in ("auto", "gpu", "cpu"):
            self.engine = "auto"
        return self


_lock = threading.Lock()


def settings_path() -> Path:
    return data_dir() / "settings.json"


def load_settings() -> Settings:
    s = Settings()
    try:
        raw = json.loads(settings_path().read_text(encoding="utf-8"))
        known = {f.name for f in fields(Settings)}
        defaults = Settings()
        for k, v in raw.items():
            if k not in known:
                continue
            want = type(getattr(defaults, k))
            if want is float and isinstance(v, int) and not isinstance(v, bool):
                v = float(v)
            if type(v) is want:           # a wrong kind of value keeps the default
                setattr(s, k, v)
        s.clamp()
    except Exception:
        s = Settings()                    # a damaged file must never stop N-Pass from starting
    return s.clamp()


def save_settings(s: Settings) -> None:
    s.clamp()
    data = json.dumps(asdict(s), indent=2).encode("utf-8")
    with _lock:
        tmp = settings_path().with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, settings_path())


# --- blocked word list: one entry per line, the user's own words; ships empty ---

def words_path() -> Path:
    return data_dir() / "blocked_words.txt"


def load_words() -> list[str]:
    try:
        lines = words_path().read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[str] = []
    for line in lines:
        w = line.strip().lower()
        if w and not w.startswith("#") and w not in out:
            out.append(w)
    return out


def save_words(words: list[str]) -> None:
    clean: list[str] = []
    for w in words:
        w = w.strip().lower()
        if w and w not in clean:
            clean.append(w)
    data = ("\n".join(clean) + ("\n" if clean else "")).encode("utf-8")
    with _lock:
        tmp = words_path().with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, words_path())


def catch_log_path() -> Path:
    return data_dir() / "catches.log"


def token_path() -> Path:
    """Spotify tokens (DPAPI-encrypted for this Windows account). Never print or log the contents."""
    return data_dir() / "spotify-session.bin"
