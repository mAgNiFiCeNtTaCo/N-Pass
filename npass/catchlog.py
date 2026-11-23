"""The catch log: one line per caught word, in the user's data folder.

Line format (docs/CONTRACT.md):
    2026-11-22 21:14:03 | caught "<word>" (0.71, lyric window) | Artist - Song @ 1:42
No song known: the last part is `no song`. A late cut adds ` | late 35 ms`.

Local time, UTF-8, append-only, safe to call from any thread. When the file reaches 2 MB it is
renamed to catches.1.log (replacing the older one) and a new file is started.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from npass import settings
from npass.types import Catch, CutReason

MAX_BYTES = 2 * 1024 * 1024
FAILSAFE_WORD = "(unchecked lyric line)"   # a failsafe cut has no word: this stands in for it

_lock = threading.Lock()


def display_word(catch: Catch) -> str:
    """The word to show for a catch. A failsafe cut has none, so it gets a fixed label."""
    word = (catch.order.word or "").strip()
    if word:
        return word
    if catch.order.reason == CutReason.FAILSAFE:
        return FAILSAFE_WORD
    return (catch.order.heard or "").strip() or "?"


def reason_text(catch: Catch) -> str:
    reason = catch.order.reason
    return reason.value if isinstance(reason, CutReason) else str(reason)


def format_catch(catch: Catch) -> str:
    """One log line, without the line break."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(catch.wall_time))
    word = display_word(catch).replace('"', "'").replace("\n", " ").replace("\r", " ")
    song = catch.track.label() if catch.track is not None else "no song"
    song = song.replace("\n", " ").replace("\r", " ")
    line = f'{stamp} | caught "{word}" ({catch.order.confidence:.2f}, {reason_text(catch)}) | {song}'
    if catch.late_ms > 0:
        line += f" | late {catch.late_ms:.0f} ms"
    return line


def rotated_path() -> Path:
    p = settings.catch_log_path()
    return p.with_name(p.stem + ".1" + p.suffix)   # catches.1.log


def log_catch(catch: Catch) -> None:
    """Append one line for this catch. Never raises: a full disk must not stop the filter."""
    data = (format_catch(catch) + "\n").encode("utf-8")
    with _lock:
        try:
            path = settings.catch_log_path()
            try:
                if path.stat().st_size + len(data) > MAX_BYTES:
                    os.replace(path, rotated_path())
            except OSError:
                pass   # no file yet, or it is open elsewhere: just keep appending
            with open(path, "ab") as f:
                f.write(data)
        except OSError:
            pass


def read_recent(n: int = 5) -> list[str]:
    """The last `n` log lines, oldest first. Looks into the rotated file when the new one is short."""
    if n <= 0:
        return []
    lines: list[str] = []
    with _lock:
        for path in (settings.catch_log_path(), rotated_path()):
            if len(lines) >= n:
                break
            lines = _tail(path, n - len(lines)) + lines
    return lines[-n:]


def _tail(path: Path, n: int) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 64 * 1024))
            chunk = f.read()
    except OSError:
        return []
    rows = chunk.decode("utf-8", errors="replace").splitlines()
    if size > 64 * 1024 and rows:
        rows = rows[1:]   # the first row of a partial read may be cut in half
    return [r for r in rows if r.strip()][-n:]
