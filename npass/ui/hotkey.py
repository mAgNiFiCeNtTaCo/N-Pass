"""A global on/off hotkey with Win32 RegisterHotKey (ctypes, no keyboard hook).

RegisterHotKey belongs to the thread that calls it, so the listener registers the key and runs its
own small message loop in one thread. `on_press` is called from that thread: the window only puts
it on its own queue.
"""
from __future__ import annotations

import ctypes
import threading
from typing import Any, Callable

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012
HOTKEY_ID = 0x4E50   # any number that is ours within this thread

_MODS = {"ctrl": MOD_CONTROL, "control": MOD_CONTROL, "alt": MOD_ALT, "shift": MOD_SHIFT,
         "win": MOD_WIN, "windows": MOD_WIN}
_KEYS = {
    "space": 0x20, "enter": 0x0D, "return": 0x0D, "tab": 0x09, "esc": 0x1B, "escape": 0x1B,
    "backspace": 0x08, "insert": 0x2D, "ins": 0x2D, "delete": 0x2E, "del": 0x2E,
    "home": 0x24, "end": 0x23, "pageup": 0x21, "pgup": 0x21, "pagedown": 0x22, "pgdn": 0x22,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "pause": 0x13, "scrolllock": 0x91, "printscreen": 0x2C,
    "plus": 0xBB, "minus": 0xBD, "comma": 0xBC, "period": 0xBE,
    ";": 0xBA, "=": 0xBB, ",": 0xBC, "-": 0xBD, ".": 0xBE, "/": 0xBF, "`": 0xC0,
    "[": 0xDB, "\\": 0xDC, "]": 0xDD, "'": 0xDE,
}
_KEYS.update({f"f{i}": 0x6F + i for i in range(1, 25)})          # F1..F24
_KEYS.update({f"num{i}": 0x60 + i for i in range(10)})            # number pad
_PRETTY = {"ctrl": "Ctrl", "control": "Ctrl", "alt": "Alt", "shift": "Shift", "win": "Win",
           "windows": "Win"}


def parse_hotkey(text: str) -> tuple[int, int]:
    """"ctrl+alt+n" -> (modifier flags, virtual key code). Raises ValueError when it makes no sense."""
    parts = [p.strip().lower() for p in (text or "").split("+")]
    if not parts or any(not p for p in parts):
        raise ValueError(f"Not a key combination: {text!r}")
    mods = 0
    vk: int | None = None
    for p in parts:
        if p in _MODS:
            mods |= _MODS[p]
            continue
        if vk is not None:
            raise ValueError(f"More than one main key in: {text!r}")
        if len(p) == 1 and (p.isascii() and p.isalnum()):
            vk = ord(p.upper())
        elif p in _KEYS:
            vk = _KEYS[p]
        else:
            raise ValueError(f"Unknown key {p!r} in: {text!r}")
    if vk is None:
        raise ValueError(f"No main key in: {text!r}")
    if mods == 0 and not (0x70 <= vk <= 0x87):
        raise ValueError(f"Add Ctrl, Alt, Shift or Win to: {text!r}")
    return mods, vk


def pretty(text: str) -> str:
    """"ctrl+alt+n" -> "Ctrl+Alt+N" for showing on screen."""
    out = []
    for p in (text or "").split("+"):
        p = p.strip()
        if p:
            fkey = p[:1].lower() == "f" and p[1:].isdigit()
            out.append(_PRETTY.get(p.lower(), p.upper() if len(p) == 1 or fkey else p.title()))
    return "+".join(out)


class _Win32Api:
    """The four calls the listener needs. Tests pass a fake with the same methods."""

    def __init__(self) -> None:
        from ctypes import wintypes
        self._wintypes = wintypes
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._user32.RegisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT]
        self._user32.RegisterHotKey.restype = wintypes.BOOL
        self._user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
        self._user32.UnregisterHotKey.restype = wintypes.BOOL
        self._user32.GetMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                             wintypes.UINT, wintypes.UINT]
        self._user32.GetMessageW.restype = ctypes.c_int
        self._user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND,
                                              wintypes.UINT, wintypes.UINT, wintypes.UINT]
        self._user32.PostThreadMessageW.argtypes = [wintypes.DWORD, wintypes.UINT,
                                                    wintypes.WPARAM, wintypes.LPARAM]
        self._user32.PostThreadMessageW.restype = wintypes.BOOL
        self._kernel32.GetCurrentThreadId.restype = wintypes.DWORD

    def thread_id(self) -> int:
        return int(self._kernel32.GetCurrentThreadId())

    def register(self, hotkey_id: int, mods: int, vk: int) -> bool:
        msg = self._wintypes.MSG()
        self._user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)   # makes this thread's queue
        return bool(self._user32.RegisterHotKey(None, hotkey_id, mods | MOD_NOREPEAT, vk))

    def unregister(self, hotkey_id: int) -> None:
        self._user32.UnregisterHotKey(None, hotkey_id)

    def next_message(self) -> tuple[int, int] | None:
        """Wait for the next message of this thread: (message, wParam). None = time to stop."""
        msg = self._wintypes.MSG()
        got = self._user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
        if got <= 0:
            return None
        return int(msg.message), int(msg.wParam)

    def post_quit(self, thread_id: int) -> None:
        self._user32.PostThreadMessageW(thread_id, WM_QUIT, 0, 0)


class HotkeyListener:
    """Registers one global hotkey and calls `on_press` each time it is pressed.

    `start()` returns True when the key is registered. When it returns False, `error` holds one
    plain-English line for the warning area and nothing else is wrong: the app keeps running.
    """

    def __init__(self, combo: str, on_press: Callable[[], None], api: Any = None):
        self.combo = combo
        self.on_press = on_press
        self.error = ""
        self.active = False
        self._api = api
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ready = threading.Event()

    def start(self, wait_s: float = 2.0) -> bool:
        try:
            mods, vk = parse_hotkey(self.combo)
        except ValueError:
            self.error = (f'The hotkey "{self.combo}" is not a key combination N-Pass understands. '
                          "Use the ON / OFF button instead.")
            return False
        try:
            if self._api is None:
                self._api = _Win32Api()
        except Exception:
            self.error = "The hotkey could not be set up on this PC. Use the ON / OFF button instead."
            return False
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, args=(mods, vk), name="npass-hotkey",
                                        daemon=True)
        self._thread.start()
        self._ready.wait(wait_s)
        return self.active

    def _run(self, mods: int, vk: int) -> None:
        api = self._api
        try:
            self._thread_id = api.thread_id()
            ok = api.register(HOTKEY_ID, mods, vk)
        except Exception:
            ok = False
        if not ok:
            self.error = (f"The hotkey {pretty(self.combo)} is already used by another program. "
                          "Use the ON / OFF button instead.")
            self._ready.set()
            return
        self.active = True
        self._ready.set()
        try:
            while True:
                got = api.next_message()
                if got is None:
                    break
                message, wparam = got
                if message == WM_HOTKEY and wparam == HOTKEY_ID:
                    try:
                        self.on_press()
                    except Exception:
                        pass
        finally:
            try:
                api.unregister(HOTKEY_ID)
            except Exception:
                pass
            self.active = False

    def stop(self, wait_s: float = 2.0) -> None:
        """Unregister the key and end the thread. Safe to call twice."""
        thread = self._thread
        if thread is None or not thread.is_alive():
            return
        try:
            self._api.post_quit(self._thread_id)
        except Exception:
            pass
        thread.join(wait_s)
        self._thread = None
