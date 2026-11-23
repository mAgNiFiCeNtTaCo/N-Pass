"""The OBS sync dialog against a fake obs-websocket client. No network, no real OBS."""
import sys
from pathlib import Path

# plain `uv run pytest` does not put the project root on the path (no conftest, package = false)
for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import logging

import pytest

from npass import obs_sync
from npass.ui.fake_core import FakeCore
from npass.ui.window import MainWindow
from test_obs_sync import SECRET, FakeClient
from test_ui_support import drop, new_top


@pytest.fixture
def made(tmp_path, monkeypatch):
    monkeypatch.setenv("NPASS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(obs_sync, "READBACK_WAIT_S", 0.0)
    root = new_top()
    seen = {"passwords": [], "clients": []}

    def fake_connect(host, port, password):
        seen["passwords"].append(password)
        if password != SECRET:
            raise obs_sync.ObsError("OBS did not let N-Pass in. Check the password.")
        session = obs_sync.connect(host, port, password, factory=FakeClient)
        seen["clients"].append(session._client)
        return session

    dialog = None
    try:
        win = MainWindow(FakeCore(animate=False), root, inline_threads=True)
        from npass.ui.obs_dialog import ObsSyncDialog
        dialog = ObsSyncDialog(win, delay_ms=487.2, connect=fake_connect, host="127.0.0.1", port=4455)
        dialog.top.withdraw()
        root.update()
        yield dialog, seen, win
    finally:
        if dialog is not None:
            dialog.close()
        drop(root)


def test_steps_are_shown_with_the_delay(made):
    dialog, _, _ = made
    text = dialog.steps.get("1.0", "end")
    assert "487 ms" in text and "Video Delay (Async)" in text and "Sync Offset" in text
    assert str(dialog.apply_btn.cget("state")) == "disabled"
    assert str(dialog.undo_btn.cget("state")) == "disabled"


def test_wrong_password_shows_a_plain_message(made):
    dialog, seen, _ = made
    dialog.password_var.set("nope")
    dialog.connect_btn.invoke()
    assert dialog.password_var.get() == ""
    assert dialog.status().startswith("OBS did not let N-Pass in")
    assert dialog.session is None and str(dialog.apply_btn.cget("state")) == "disabled"


def test_connect_apply_undo(made, caplog):
    caplog.set_level(logging.DEBUG)
    dialog, seen, win = made
    dialog.password_var.set(SECRET)
    dialog.connect_btn.invoke()
    assert dialog.password_var.get() == ""            # the box is emptied straight away
    assert dialog.row_labels() == [
        "Desktop Audio: 0 ms -> 487 ms",
        "Music: 120 ms -> 607 ms",
        "Mic/Aux: 0 ms (the N-Pass mic, stays as it is)",
    ]
    assert [v.get() for v in dialog.row_vars] == [True, True, False]
    assert str(dialog.row_checks[2].cget("state")) == "disabled"   # the mic cannot be ticked
    client = seen["clients"][0]
    assert client.sets == []                          # nothing changes before Apply

    dialog.row_vars[1].set(False)                     # leave Music alone
    dialog._buttons()
    dialog.apply_btn.invoke()
    assert client.sets == [("Desktop Audio", 487)]
    assert "Desktop Audio: 0 ms -> 487 ms" in dialog.status()
    assert dialog.row_labels()[0] == "Desktop Audio: 487 ms (already right)"
    assert str(dialog.undo_btn.cget("state")) == "normal"

    dialog.undo_btn.invoke()
    assert client.offsets["Desktop Audio"] == 0
    assert "Desktop Audio: 487 ms -> 0 ms" in dialog.status()
    assert str(dialog.undo_btn.cget("state")) == "disabled"

    # the password is nowhere: not on the dialog, not on the session, not in the logs
    assert SECRET not in repr(vars(dialog)) and SECRET not in repr(dialog.session)
    assert SECRET not in repr(client) and SECRET not in caplog.text
    dialog.close()
    assert client.closed and not dialog.alive()
    win.root.update()


def test_read_back_mismatch_is_shown(made):
    dialog, seen, _ = made
    dialog.password_var.set(SECRET)
    dialog.connect_btn.invoke()
    seen["clients"][0].ignore_set.add("Music")
    dialog.apply_btn.invoke()
    assert '"Music" should be 607 ms but OBS says 120 ms. Set it by hand in OBS.' in dialog.status()
    assert "Desktop Audio: 0 ms -> 487 ms" in dialog.status()


def test_window_opens_one_dialog(made, monkeypatch):
    dialog, _, win = made
    dialog.close()
    win.open_obs_sync()
    first = win.obs_dialog
    first.top.withdraw()
    assert first.delay_ms == 487
    win.open_obs_sync()
    assert win.obs_dialog is first
