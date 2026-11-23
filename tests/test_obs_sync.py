import sys
from pathlib import Path

# plain `uv run pytest` does not put the project root on the path (no conftest, package = false)
for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import json
import logging

import pytest

from npass import obs_sync

SECRET = "hunter2-banana-password"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("NPASS_DATA_DIR", str(tmp_path))
    return tmp_path


class Reply:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeClient:
    """Stands in for obsws_python.ReqClient. Same method names, same reply shapes."""

    def __init__(self, **kwargs):
        # The real library logs the password and prints it in its repr: copy both bad habits.
        logging.getLogger("obsws_python.baseclient.ObsClient").info(
            "Connecting with parameters: host='%s' password='%s'", kwargs.get("host"), kwargs.get("password"))
        self.base_client = Reply(password=kwargs.get("password"))
        self.offsets = {"Desktop Audio": 0, "Music": 120, "Mic/Aux": 0}
        self.no_audio = {"Webcam Image"}
        self.devices = {"Mic/Aux": ("{id-cable}", "CABLE Output (VB-Audio Virtual Cable)"),
                        "Desktop Audio": ("default", "Default")}
        self.stuck: dict[str, int] = {}     # input -> number of reads that still show the old value
        self.ignore_set: set[str] = set()   # inputs where a set silently does nothing
        self.sets: list[tuple[str, int]] = []
        self.closed = False
        self._pending: dict[str, int] = {}

    def __repr__(self):
        return f"FakeClient(password='{self.base_client.password}')"

    def get_input_list(self, kind=None):
        names = list(self.offsets) + sorted(self.no_audio)
        return Reply(inputs=[{"inputName": n, "inputKind": "k_v2", "unversionedInputKind": "k"} for n in names])

    def get_input_audio_sync_offset(self, name):
        if name not in self.offsets:
            raise RuntimeError("Request GetInputAudioSyncOffset returned code 604.")
        if name in self._pending:
            if self.stuck.get(name, 0) > 0:
                self.stuck[name] -= 1
            else:
                self.offsets[name] = self._pending.pop(name)
        return Reply(input_audio_sync_offset=self.offsets[name])

    def set_input_audio_sync_offset(self, name, offset):
        if name not in self.offsets:
            raise RuntimeError("Request SetInputAudioSyncOffset returned code 600.")
        self.sets.append((name, offset))
        if name in self.ignore_set:
            return
        self._pending[name] = offset

    def get_input_settings(self, name):
        dev = self.devices.get(name)
        return Reply(input_settings={"device_id": dev[0]} if dev else {})

    def get_input_properties_list_property_items(self, input_name, prop_name):
        dev = self.devices.get(input_name)
        items = [{"itemName": "Something else", "itemValue": "{other}", "itemEnabled": True}]
        if dev:
            items.append({"itemName": dev[1], "itemValue": dev[0], "itemEnabled": True})
        return Reply(property_items=items)

    def disconnect(self):
        self.closed = True


def session_and_client():
    client = FakeClient(host="127.0.0.1", password="")
    return obs_sync.ObsSession(client), client


def no_sleep(_s):
    pass


def test_steps_text_has_the_number_and_the_three_places():
    text = obs_sync.steps_text(486.6)
    assert "487 ms" in text
    assert "Advanced Audio Properties" in text and "Sync Offset" in text
    assert '"Video Delay (Async)"' in text
    assert '"CABLE Output (VB-Audio Virtual Cable)"' in text
    assert "—" not in text


def test_plan_marks_the_npass_mic_and_skips_inputs_without_sound():
    session, client = session_and_client()
    rows = {r.name: r for r in obs_sync.build_plan(session, 487)}
    assert set(rows) == {"Desktop Audio", "Music", "Mic/Aux"}
    assert (rows["Desktop Audio"].before, rows["Desktop Audio"].after, rows["Desktop Audio"].ticked) == (0, 487, True)
    assert (rows["Music"].before, rows["Music"].after) == (120, 607)
    mic = rows["Mic/Aux"]
    assert mic.is_npass_mic and not mic.ticked and mic.after == mic.before
    assert "N-Pass mic" in mic.label()
    assert rows["Desktop Audio"].label() == "Desktop Audio: 0 ms -> 487 ms"
    assert client.sets == []   # building a plan changes nothing


def test_plan_clamps_to_what_obs_accepts():
    session, client = session_and_client()
    client.offsets["Music"] = 19900
    rows = {r.name: r for r in obs_sync.build_plan(session, 487)}
    assert rows["Music"].after == 20000


def test_apply_changes_only_ticked_rows_and_saves_undo():
    session, client = session_and_client()
    rows = obs_sync.build_plan(session, 487)
    for r in rows:
        if r.name == "Music":
            r.ticked = False
        if r.is_npass_mic:
            r.ticked = True      # even if someone ticks it, the mic is never changed
    result = obs_sync.apply_plan(session, rows, sleep=no_sleep)
    assert result.ok
    assert result.changed == ["Desktop Audio: 0 ms -> 487 ms"]
    assert client.sets == [("Desktop Audio", 487)]
    assert client.offsets == {"Desktop Audio": 487, "Music": 120, "Mic/Aux": 0}
    saved = json.loads(obs_sync.undo_path().read_text(encoding="utf-8"))
    assert saved["inputs"] == {"Desktop Audio": {"before": 0, "after": 487}}


def test_apply_waits_for_a_change_that_lands_late():
    session, client = session_and_client()
    client.stuck["Desktop Audio"] = 2
    naps = []
    rows = [r for r in obs_sync.build_plan(session, 400) if r.name == "Desktop Audio"]
    result = obs_sync.apply_plan(session, rows, sleep=naps.append)
    assert result.ok and result.changed == ["Desktop Audio: 0 ms -> 400 ms"]
    assert len(naps) == 2


def test_apply_reports_a_read_back_mismatch():
    session, client = session_and_client()
    client.ignore_set.add("Desktop Audio")
    rows = obs_sync.build_plan(session, 487)
    result = obs_sync.apply_plan(session, rows, sleep=no_sleep)
    assert not result.ok
    assert result.changed == ["Music: 120 ms -> 607 ms"]
    assert result.problems == ['"Desktop Audio" should be 487 ms but OBS says 0 ms. Set it by hand in OBS.']
    assert "Set it by hand" in result.summary()


def test_apply_twice_does_not_add_the_delay_twice():
    session, client = session_and_client()
    obs_sync.apply_plan(session, obs_sync.build_plan(session, 487), sleep=no_sleep)
    rows = {r.name: r for r in obs_sync.build_plan(session, 500)}
    assert (rows["Music"].before, rows["Music"].after) == (607, 620)
    obs_sync.apply_plan(session, list(rows.values()), sleep=no_sleep)
    assert client.offsets["Music"] == 620 and client.offsets["Desktop Audio"] == 500
    assert obs_sync.load_undo()["Music"] == {"before": 120, "after": 620}


def test_plan_respects_a_value_the_user_changed_by_hand():
    session, client = session_and_client()
    obs_sync.apply_plan(session, obs_sync.build_plan(session, 487), sleep=no_sleep)
    client.offsets["Music"] = 50    # the user typed this in OBS afterwards
    rows = {r.name: r for r in obs_sync.build_plan(session, 487)}
    assert (rows["Music"].before, rows["Music"].after) == (50, 537)


def test_undo_puts_the_old_values_back():
    session, client = session_and_client()
    obs_sync.apply_plan(session, obs_sync.build_plan(session, 487), sleep=no_sleep)
    assert obs_sync.has_undo()
    result = obs_sync.undo_last(session, sleep=no_sleep)
    assert result.ok
    assert sorted(result.changed) == ["Desktop Audio: 487 ms -> 0 ms", "Music: 607 ms -> 120 ms"]
    assert client.offsets == {"Desktop Audio": 0, "Music": 120, "Mic/Aux": 0}
    assert not obs_sync.undo_path().exists()
    assert not obs_sync.has_undo()


def test_undo_keeps_what_it_could_not_restore():
    session, client = session_and_client()
    obs_sync.apply_plan(session, obs_sync.build_plan(session, 487), sleep=no_sleep)
    del client.offsets["Music"]     # the source was removed from OBS
    result = obs_sync.undo_last(session, sleep=no_sleep)
    assert not result.ok and "Music" in result.problems[0]
    assert list(obs_sync.load_undo()) == ["Music"]


def test_undo_with_nothing_saved():
    session, _ = session_and_client()
    assert obs_sync.undo_last(session, sleep=no_sleep).problems == ["There is nothing to undo."]


def test_password_never_shows_up(caplog, data_dir):
    caplog.set_level(logging.DEBUG)
    session = obs_sync.connect("127.0.0.1", 4455, SECRET, factory=FakeClient)
    client = session._client
    assert SECRET not in repr(session) and SECRET not in str(session)
    assert SECRET not in repr(client)          # the wrapper wiped the library's copy
    assert SECRET not in repr(vars(session))
    rows = obs_sync.build_plan(session, 487)
    result = obs_sync.apply_plan(session, rows, sleep=no_sleep)
    obs_sync.undo_last(session, sleep=no_sleep)
    session.close()
    assert client.closed
    assert SECRET not in caplog.text
    assert SECRET not in result.summary() and SECRET not in repr(rows)
    for f in data_dir.iterdir():
        assert SECRET not in f.read_text(encoding="utf-8", errors="replace")


def test_connect_errors_are_plain_and_hide_the_password(caplog):
    caplog.set_level(logging.DEBUG)

    def refused(**kwargs):
        raise ConnectionRefusedError(f"no one home for {kwargs['password']}")

    def bad_password(**kwargs):
        raise RuntimeError(f"auth failed with {kwargs['password']}")

    with pytest.raises(obs_sync.ObsError) as e1:
        obs_sync.connect("127.0.0.1", 4455, SECRET, factory=refused)
    with pytest.raises(obs_sync.ObsError) as e2:
        obs_sync.connect("127.0.0.1", 4455, SECRET, factory=bad_password)
    assert "Could not reach OBS" in str(e1.value)
    assert "password" in str(e2.value)
    for e in (e1.value, e2.value):
        assert SECRET not in str(e) and SECRET not in repr(e)
        assert e.__cause__ is None and e.__suppress_context__
    assert SECRET not in caplog.text
