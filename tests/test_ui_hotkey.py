import sys
from pathlib import Path

# plain `uv run pytest` does not put the project root on the path (no conftest, package = false)
for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import queue
import threading

import pytest

from npass.ui import hotkey
from npass.ui.hotkey import (HOTKEY_ID, MOD_ALT, MOD_CONTROL, MOD_SHIFT, MOD_WIN, WM_HOTKEY,
                             HotkeyListener, parse_hotkey, pretty)


def test_parse():
    assert parse_hotkey("ctrl+alt+n") == (MOD_CONTROL | MOD_ALT, ord("N"))
    assert parse_hotkey(" Ctrl + Shift + F9 ") == (MOD_CONTROL | MOD_SHIFT, 0x78)
    assert parse_hotkey("win+shift+5") == (MOD_WIN | MOD_SHIFT, ord("5"))
    assert parse_hotkey("control+space") == (MOD_CONTROL, 0x20)
    assert parse_hotkey("f8") == (0, 0x77)          # a lone F key is allowed
    assert parse_hotkey("alt+pause") == (MOD_ALT, 0x13)


@pytest.mark.parametrize("bad", ["", "ctrl+", "ctrl+alt", "n", "ctrl+a+b", "ctrl+nope", "+", None])
def test_parse_rejects(bad):
    with pytest.raises(ValueError):
        parse_hotkey(bad)


def test_pretty():
    assert pretty("ctrl+alt+n") == "Ctrl+Alt+N"
    assert pretty("control+shift+f9") == "Ctrl+Shift+F9"
    assert pretty("win+space") == "Win+Space"


class FakeApi:
    def __init__(self, register_ok=True):
        self.register_ok = register_ok
        self.messages: queue.Queue = queue.Queue()
        self.registered = []
        self.unregistered = []
        self.register_thread = None
        self.loop_thread = None

    def thread_id(self):
        return 1234

    def register(self, hotkey_id, mods, vk):
        self.register_thread = threading.get_ident()
        self.registered.append((hotkey_id, mods, vk))
        return self.register_ok

    def unregister(self, hotkey_id):
        self.unregistered.append(hotkey_id)

    def next_message(self):
        self.loop_thread = threading.get_ident()
        return self.messages.get(timeout=5)

    def post_quit(self, thread_id):
        assert thread_id == 1234
        self.messages.put(None)


def test_listener_calls_back_and_unregisters():
    api = FakeApi()
    pressed = threading.Semaphore(0)
    listener = HotkeyListener("ctrl+alt+n", pressed.release, api=api)
    assert listener.start() is True and listener.active and listener.error == ""
    assert api.registered == [(HOTKEY_ID, MOD_CONTROL | MOD_ALT, ord("N"))]
    api.messages.put((0x0113, 0))                 # some other message: ignored
    api.messages.put((WM_HOTKEY, 999))            # someone else's hotkey id: ignored
    api.messages.put((WM_HOTKEY, HOTKEY_ID))
    assert pressed.acquire(timeout=5)
    assert not pressed.acquire(timeout=0.1)
    assert api.register_thread == api.loop_thread != threading.get_ident()   # one thread owns both
    listener.stop()
    listener.stop()                               # twice is fine
    assert api.unregistered == [HOTKEY_ID] and not listener.active


def test_a_failing_callback_does_not_kill_the_listener():
    api = FakeApi()
    hits = []
    done = threading.Event()

    def cb():
        hits.append(1)
        if len(hits) == 1:
            raise RuntimeError("boom")
        done.set()

    listener = HotkeyListener("ctrl+alt+n", cb, api=api)
    listener.start()
    api.messages.put((WM_HOTKEY, HOTKEY_ID))
    api.messages.put((WM_HOTKEY, HOTKEY_ID))
    assert done.wait(5)
    listener.stop()


def test_taken_combination_gives_a_warning_line_not_an_error():
    api = FakeApi(register_ok=False)
    listener = HotkeyListener("ctrl+alt+n", lambda: None, api=api)
    assert listener.start() is False
    assert listener.error == ("The hotkey Ctrl+Alt+N is already used by another program. "
                              "Use the ON / OFF button instead.")
    assert api.unregistered == []
    listener.stop()


def test_nonsense_combination_gives_a_warning_line():
    listener = HotkeyListener("ctrl+nope", lambda: None, api=FakeApi())
    assert listener.start() is False
    assert "not a key combination" in listener.error


def test_real_win32_api_object_can_be_made():
    api = hotkey._Win32Api()                      # only loads user32; registers nothing
    assert api.thread_id() > 0
