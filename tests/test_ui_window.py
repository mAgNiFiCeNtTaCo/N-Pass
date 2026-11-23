"""The window, built for real but never shown and never left in a main loop."""
import sys
from pathlib import Path

# plain `uv run pytest` does not put the project root on the path (no conftest, package = false)
for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import time
import tkinter as tk

import pytest

from npass import settings
from npass.types import CutReason, Health, TrackInfo
from npass.ui import window as window_mod
from npass.ui.fake_core import FakeCore, demo_core
from npass.ui.window import MainWindow, catch_columns, collect_warnings, mask_word
from test_ui_support import drop, new_top

EM_DASH = "—"


@pytest.fixture
def made(tmp_path, monkeypatch):
    """(window, core, opened paths). The window is destroyed afterwards, whatever the test did."""
    monkeypatch.setenv("NPASS_DATA_DIR", str(tmp_path))
    root = new_top()
    opened: list[str] = []
    core = FakeCore(animate=False, login_seconds=0.0)
    win = None
    try:
        win = MainWindow(core, root, readme_path=tmp_path / "README.md", opener=opened.append,
                         inline_threads=True)
        pump(win)
        yield win, core, opened
    finally:
        if win is not None and win.obs_dialog is not None:
            win.obs_dialog.close()
        drop(root)


def pump(win, times=3):
    for _ in range(times):
        win.refresh()
        win.root.update()


def all_texts(widget):
    out = []
    for child in widget.winfo_children():
        try:
            out.append(str(child.cget("text")))
        except tk.TclError:
            pass
        out += all_texts(child)
    return out


def test_main_view_shows_the_snapshot(made):
    win, core, _ = made
    assert win.toggle_btn.cget("text") == "ON"
    assert win.hotkey_lbl.cget("text") == "Click, or press Ctrl+Alt+N, to turn off"
    assert win.delay_lbl.cget("text") == "Delay: 487 ms (look-ahead 450 ms)"
    assert win.look_value_lbl.cget("text") == "450 ms"
    assert win.caught_lbl.cget("text") == "Caught: 0"
    assert win.engine_lbl.cget("text") == "Listening with: GPU (faster-whisper small.en), 42 ms per check"
    assert win.lyrics_lbl.cget("text") == "Lyrics: Lyrics found. Watching 3 risky lines in this song."
    assert win.song_lbl.cget("text") == "Song: Demo Artist - Fruit Salad @ 0:00"
    assert win.warning_texts() == [] and win.cable_box is None


def test_on_off_button_and_hotkey_path(made):
    win, core, _ = made
    win.toggle_btn.invoke()
    pump(win)
    assert core.enabled is False
    assert win.toggle_btn.cget("text") == "OFF"
    assert win.hotkey_lbl.cget("text") == "Click, or press Ctrl+Alt+N, to turn on"
    assert win.engine_lbl.cget("text").startswith("Off.")
    win.post(win.toggle)          # what the hotkey thread and the tray menu do
    pump(win)
    assert core.enabled is True and win.toggle_btn.cget("text") == "ON"


def test_meter_colours_and_cutting(made):
    win, core, _ = made
    for level, colour in ((-30.0, window_mod.theme.GREEN), (-12.0, window_mod.theme.YELLOW),
                          (-2.0, window_mod.theme.RED)):
        core.level_db = level
        pump(win, 1)
        assert win.meter_fill_value == colour and win.meter_text_value == ""
    core.cutting = True
    pump(win, 1)
    assert win.meter_fill_value == window_mod.theme.GREY and win.meter_text_value == "Cutting"


def test_delay_turns_red_over_700(made):
    win, core, _ = made
    assert str(win.delay_lbl.cget("foreground")) == window_mod.theme.TEXT
    core.extra_delay_ms = 260.0
    pump(win)
    assert win.delay_lbl.cget("text").startswith("Delay: 710 ms (look-ahead 450 ms)")
    assert str(win.delay_lbl.cget("foreground")) == window_mod.theme.RED


def test_slider_tells_the_core_only_when_released(made):
    win, core, _ = made
    win._dragging = True
    win.look_var.set(523.0)
    win._look_moved()
    pump(win)
    assert win.look_value_lbl.cget("text") == "520 ms"
    assert not [c for c in core.calls if c[0] == "set_lookahead"]
    win.commit_lookahead()
    pump(win)
    assert [c for c in core.calls if c[0] == "set_lookahead"] == [("set_lookahead", 520)]
    assert win.delay_lbl.cget("text") == "Delay: 557 ms (look-ahead 520 ms)"


def test_catch_counter_and_last_five(made):
    win, core, _ = made
    assert win.no_catch_lbl.winfo_manager() == "grid"
    track = TrackInfo("t", "Artist", "Song", 102_000, True)
    now = time.time()
    for i in range(6):
        core.add_catch("pineapple" if i % 2 == 0 else "banana", 0.5 + i / 20, CutReason.LYRIC_WINDOW,
                       track if i % 2 == 0 else None, wall_time=now + i)
    core.add_catch("pineapple", 0.71, CutReason.ASR, track, late_ms=35.0, wall_time=now + 6)
    pump(win)
    assert win.caught_lbl.cget("text") == "Caught: 7"
    assert win.no_catch_lbl.winfo_manager() == ""
    top = [lbl.cget("text") for lbl in win.catch_rows[0]]
    assert top[0] == time.strftime("%H:%M:%S", time.localtime(now + 6))
    assert top[1] == "p••••••••"      # masked until "Show words"
    assert top[2] == "0.71"
    assert top[3].startswith("late 35 ms, Artist")
    assert win.catch_rows[1][3].cget("text") == "no song"
    assert all(cells[0].cget("text") for cells in win.catch_rows)
    shown = " ".join(all_texts(win.main_tab))
    assert "pineapple" not in shown and "banana" not in shown
    win.show_words_var.set(True)
    win._show_words_changed()
    pump(win)
    assert win.catch_rows[0][1].cget("text") == "pineapple"


def test_failsafe_catch_row():
    core = FakeCore()
    catch = core.add_catch("", 0.0, CutReason.FAILSAFE, None)
    assert catch_columns(catch, show_words=False)[1:] == ("line muted", "-", "no song")
    assert mask_word("pineapple") == "p" + "•" * 8


def test_health_and_warning_lines(made):
    win, core, _ = made
    core.health = Health.LATE
    pump(win)
    assert win.warning_texts() == ["Recognition is falling behind"]
    assert str(win.warning_labels[0].cget("fg")) == window_mod.theme.YELLOW
    core.health = Health.DOWN
    core.warnings = ["Spotify sign-in ran out."]
    core.engine_warnings = ["The microphone was unplugged.", "Spotify sign-in ran out."]
    win.local_warnings.append("The hotkey Ctrl+Alt+N is already used by another program.")
    pump(win)
    assert win.warning_texts() == [
        "Recognition is not running. Risky lyric lines are being muted.",
        "Spotify sign-in ran out.",
        "The microphone was unplugged.",
        "The hotkey Ctrl+Alt+N is already used by another program.",
    ]
    assert str(win.warning_labels[0].cget("fg")) == window_mod.theme.RED


def test_no_health_warning_while_off():
    snap = FakeCore().snapshot()
    snap["enabled"], snap["health"] = False, Health.DOWN
    assert collect_warnings(snap, []) == []


def test_cable_missing_box_and_how_to_fix(made, tmp_path, monkeypatch):
    win, core, opened = made
    core.cable_found = False
    pump(win)
    assert win.cable_box is not None and win.fix_btn.cget("text") == "How to fix"
    assert "virtual cable is missing" in " ".join(all_texts(win.cable_box))
    shown = []
    monkeypatch.setattr(window_mod.messagebox, "showinfo", lambda *a, **k: shown.append(a))
    win.fix_btn.invoke()                       # no README yet: the short steps are shown instead
    assert opened == [] and len(shown) == 1 and "VB-CABLE" in shown[0][1]
    (tmp_path / "README.md").write_text("readme", encoding="utf-8")
    win.fix_btn.invoke()
    assert opened == [str(tmp_path / "README.md")]
    core.cable_found = True
    pump(win)
    assert win.cable_box is None


def test_settings_controls_call_the_core(made):
    win, core, _ = made
    assert win.mic_var.get() == "Microphone (NVIDIA Broadcast)"
    assert list(win.mic_box.cget("values")) == core.input_names
    win.mic_var.set(core.input_names[1])
    win._mic_picked()
    win.removal_var.set("dip")
    for child in all_radios(win.settings_tab):
        if child.cget("text") == "Soft dip":
            child.invoke()
        if child.cget("text") == "CPU only":
            child.invoke()
    win.lyrics_var.set(False)
    win.core.set_lyrics_enabled(False)
    pump(win)
    assert ("set_input", core.input_names[1]) in core.calls
    assert ("set_removal", "dip") in core.calls
    assert ("set_engine", "cpu") in core.calls
    assert win.removal_var.get() == "dip" and win.engine_var.get() == "cpu"
    assert win.engine_lbl.cget("text").startswith("Listening with: CPU (Vosk)")


def all_radios(widget):
    out = []
    for child in widget.winfo_children():
        if child.winfo_class() == "TRadiobutton":
            out.append(child)
        out += all_radios(child)
    return out


def test_engine_choice_falls_back_to_the_saved_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("NPASS_DATA_DIR", str(tmp_path))
    saved = settings.Settings(engine="cpu")
    settings.save_settings(saved)
    root = new_top()
    try:
        win = MainWindow(FakeCore(), root, inline_threads=True)
        assert win.engine_var.get() == "cpu"
    finally:
        drop(root)


def test_spotify_sign_in_and_out(made):
    win, core, _ = made
    assert win.spotify_btn.cget("text") == "Sign in to Spotify"
    assert win.spotify_lbl.cget("text") == "Not signed in."
    win.spotify_btn.invoke()                     # no client ID yet
    pump(win)
    assert ("spotify_login",) not in core.calls
    assert win.spotify_lbl.cget("text") == "Paste your Spotify client ID first."
    win.cid_var.set("  abc123  ")
    win.spotify_btn.invoke()
    pump(win)
    assert ("set_spotify_client_id", "abc123") in core.calls
    assert ("spotify_login",) in core.calls
    assert win.spotify_btn.cget("text") == "Sign out"
    assert win.spotify_lbl.cget("text") == "Signed in."
    win.spotify_btn.invoke()
    pump(win)
    assert ("spotify_logout",) in core.calls and win.spotify_btn.cget("text") == "Sign in to Spotify"


def test_spotify_sign_in_does_not_block_the_window(made):
    win, core, _ = made
    win.inline_threads = False
    core.login_seconds = 0.4
    win.cid_var.set("abc123")
    started = time.time()
    win.spotify_btn.invoke()
    assert time.time() - started < 0.2           # the click came straight back
    pump(win, 1)
    assert win.spotify_btn.cget("text") == "Waiting for the browser"
    assert str(win.spotify_btn.cget("state")) == "disabled"
    deadline = time.time() + 5
    while win._login_busy and time.time() < deadline:
        time.sleep(0.02)
        pump(win, 1)
    pump(win)
    assert win.spotify_btn.cget("text") == "Sign out"


def test_failed_sign_in_shows_a_plain_line_without_the_error_text(made):
    win, core, _ = made

    def boom():
        raise RuntimeError("secret-auth-code-banana")

    core.spotify_login = boom
    win.cid_var.set("abc123")
    win.spotify_btn.invoke()
    pump(win)
    assert win.spotify_lbl.cget("text") == "Sign-in did not finish. Try again."
    assert "secret-auth-code-banana" not in " ".join(all_texts(win.root))


def test_blocked_words_are_dots_until_show_words(made):
    win, core, _ = made
    dots = win.words_text.get("1.0", "end-1c")
    assert dots == "•" * 9 + "\n" + "•" * 6
    assert str(win.words_text.cget("state")) == "disabled"
    assert str(win.words_save_btn.cget("state")) == "disabled"
    assert win.words_lbl.cget("text") == '2 words hidden. Tick "Show words" to edit.'
    win.show_words_var.set(True)
    win._show_words_changed()
    assert win.words_text.get("1.0", "end-1c") == "pineapple\nbanana"
    win.words_text.insert("end", "\n  Mango \n\nbanana")
    win.show_words_var.set(False)                 # hiding keeps what was typed
    win._show_words_changed()
    assert "Mango" not in win.words_text.get("1.0", "end-1c")
    assert not [c for c in core.calls if c[0] == "set_words"]
    win.show_words_var.set(True)
    win._show_words_changed()
    win.words_save_btn.invoke()
    assert core.calls[-1] == ("set_words", ["pineapple", "banana", "mango"])
    assert win.words_lbl.cget("text") == "Saved 3 words."
    assert win.words_text.get("1.0", "end-1c") == "pineapple\nbanana\nmango"


def test_open_catch_log(made):
    win, core, opened = made
    win.open_catch_log()
    assert opened == [] and "empty" in win.tools_lbl.cget("text")
    settings.catch_log_path().write_text("line\n", encoding="utf-8")
    win.open_catch_log()        # first click only warns: the file shows the words in full
    assert opened == [] and "Click again" in win.tools_lbl.cget("text")
    win.open_catch_log()
    assert opened == [str(settings.catch_log_path())]


def test_close_hides_and_quit_calls_the_core(made):
    win, core, _ = made

    class StubTray:
        def __init__(self):
            self.notes, self.updates, self.stopped = [], [], False

        def update(self, state, enabled):
            self.updates.append((state, enabled))

        def notify(self, message):
            self.notes.append(message)

        def stop(self):
            self.stopped = True

    win.root.deiconify()
    win.root.update()
    win.hide()                                   # no tray: minimise, never vanish
    win.root.update()
    assert win.root.state() == "iconic" and not core.quit_called
    tray = StubTray()
    win.tray = tray
    win.quit_hooks.append(tray.stop)
    win.show()
    win.hide()
    win.hide()
    win.root.update()
    assert win.root.state() == "withdrawn" and len(tray.notes) == 1
    pump(win, 1)
    assert tray.updates[-1] == ("green", True)
    win.post(win.quit_app)                       # the tray's Quit
    win.refresh()
    assert core.quit_called and tray.stopped and win.closed
    try:
        gone = not int(win.root.winfo_exists())
    except tk.TclError:
        gone = True
    assert gone


def test_demo_scenes_have_no_em_dash_and_no_real_words(made):
    win, _core, _ = made
    for scene in ("main", "warn"):
        win.core = demo_core(scene, animate=False)
        win._catch_key = None
        pump(win)
        texts = " ".join(all_texts(win.root))
        assert EM_DASH not in texts
    assert win.cable_box is not None
    assert "Recognition is not running. Risky lyric lines are being muted." in win.warning_texts()
