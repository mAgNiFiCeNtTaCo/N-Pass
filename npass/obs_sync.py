"""Lining OBS up with the N-Pass delay.

N-Pass holds the voice back by a fixed delay. Everything else in OBS (other sound, the camera) has to
be held back by the same amount. `steps_text` gives the manual steps. The optional one-click path
talks to obs-websocket 5.x and changes ONLY the audio "Sync Offset" of the inputs the user ticked:

    GetInputList, GetInputAudioSyncOffset, SetInputAudioSyncOffset   (change + read back)
    GetInputSettings, GetInputPropertiesListPropertyItems            (read only: find the N-Pass mic)

It never creates filters and never reads or writes OBS's own files on disk. The password is used for
the connection and then dropped: it is not kept on any object here, not saved and not logged.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from npass import settings

OBS_MIN_OFFSET_MS = -950      # limits of SetInputAudioSyncOffset in obs-websocket 5.x
OBS_MAX_OFFSET_MS = 20000
CABLE_OBS_NAME = "CABLE Output (VB-Audio Virtual Cable)"
READBACK_TRIES = 6
READBACK_WAIT_S = 0.1

log = logging.getLogger(__name__)


class _NoPasswordFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return "password" not in record.getMessage().lower()
        except Exception:
            return False


def _quiet_library_logs() -> None:
    """obsws-python writes the password into an INFO log line when it connects. Stop that."""
    logging.getLogger("obsws_python").setLevel(logging.WARNING)
    for name in ("obsws_python", "obsws_python.baseclient", "obsws_python.baseclient.ObsClient",
                 "obsws_python.reqs", "obsws_python.reqs.ReqClient"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, _NoPasswordFilter) for f in lg.filters):
            lg.addFilter(_NoPasswordFilter())


_quiet_library_logs()


class ObsError(Exception):
    """Something the user should read. The message is plain English and never holds the password."""


def steps_text(delay_ms: float) -> str:
    """The manual steps, with the delay filled in."""
    d = int(round(delay_ms))
    return (
        f"N-Pass holds your voice back by about {d} ms so it can cut a word before it plays.\n"
        f"Hold everything else in OBS back by the same {d} ms, so sound and picture stay together.\n"
        "\n"
        "1. Point OBS at the filtered mic\n"
        f'   Settings > Audio > Mic/Auxiliary Audio: choose "{CABLE_OBS_NAME}".\n'
        "   If your mic is a source in a scene instead: right-click it > Properties > Device,\n"
        "   and choose the same name there.\n"
        "   Leave the Sync Offset of this mic at 0.\n"
        "\n"
        "2. Delay your other sound (desktop audio, game, music, alerts)\n"
        "   In the Audio Mixer, open the menu on any source > Advanced Audio Properties.\n"
        f"   In the Sync Offset column, type {d} for every source except the N-Pass mic.\n"
        "\n"
        "3. Delay your camera\n"
        '   Right-click the camera source > Filters > + > "Video Delay (Async)".\n'
        f"   Set Delay to {d} ms.\n"
        '   For a game or screen capture, use the "Render Delay" filter instead (it stops at 500 ms).\n'
        "\n"
        "If you move the look-ahead slider later, do these steps again with the new number."
    )


def undo_path() -> Path:
    return settings.data_dir() / "obs_sync_undo.json"


# ---------------------------------------------------------------- connection

def _field(obj: Any, snake: str, camel: str, default: Any = None) -> Any:
    """obsws-python answers with objects (snake_case); nested items stay dicts (camelCase)."""
    if isinstance(obj, dict):
        return obj.get(camel, obj.get(snake, default))
    return getattr(obj, snake, getattr(obj, camel, default))


class ObsSession:
    """A connected obs-websocket client, wrapped so the password can never leak through a repr."""

    def __init__(self, client: Any, where: str = "OBS"):
        self._client = client
        self._where = where

    def __repr__(self) -> str:
        return f"ObsSession({self._where})"

    __str__ = __repr__

    def list_inputs(self) -> list[tuple[str, str]]:
        """(name, kind) of every input OBS knows."""
        reply = self._client.get_input_list()
        out = []
        for item in _field(reply, "inputs", "inputs", []) or []:
            name = _field(item, "input_name", "inputName", "")
            kind = _field(item, "unversioned_input_kind", "unversionedInputKind", "") or \
                _field(item, "input_kind", "inputKind", "")
            if name:
                out.append((str(name), str(kind or "")))
        return out

    def get_offset(self, name: str) -> int:
        reply = self._client.get_input_audio_sync_offset(name)
        return int(round(float(_field(reply, "input_audio_sync_offset", "inputAudioSyncOffset", 0))))

    def set_offset(self, name: str, ms: int) -> None:
        self._client.set_input_audio_sync_offset(name, int(ms))

    def device_label(self, name: str) -> str:
        """The device an audio capture input listens to, as OBS shows it. "" when unknown."""
        try:
            reply = self._client.get_input_settings(name)
            device_id = (_field(reply, "input_settings", "inputSettings", {}) or {}).get("device_id")
            if not device_id:
                return ""
            items = self._client.get_input_properties_list_property_items(name, "device_id")
            for item in _field(items, "property_items", "propertyItems", []) or []:
                if _field(item, "item_value", "itemValue") == device_id:
                    return str(_field(item, "item_name", "itemName", "") or "")
        except Exception:
            pass
        return ""

    def close(self) -> None:
        try:
            self._client.disconnect()
        except Exception:
            pass


def _default_factory(**kwargs: Any) -> Any:
    import obsws_python
    return obsws_python.ReqClient(**kwargs)


def connect(host: str, port: int, password: str, timeout: float = 3.0,
            factory: Callable[..., Any] | None = None) -> ObsSession:
    """Connect to obs-websocket. Raises ObsError with a message the user can act on."""
    _quiet_library_logs()
    # OBS on this PC only: the login answer must never be sent to another machine
    if str(host).strip().lower() not in ("127.0.0.1", "localhost", "::1"):
        raise ObsError("N-Pass only talks to OBS on this PC. The OBS address in the settings "
                       "file was changed to another computer, so nothing was sent.")
    make = factory or _default_factory
    try:
        client = make(host=host, port=int(port), password=password, timeout=timeout)
    except Exception as e:
        kind = type(e).__name__
        refused = isinstance(e, (ConnectionError, TimeoutError, OSError)) or "Timeout" in kind
        if refused:
            msg = ("Could not reach OBS. Open OBS, then turn on Tools > WebSocket Server Settings > "
                   "Enable WebSocket server, and try again.")
        else:
            msg = ("OBS did not let N-Pass in. Check the password in OBS under Tools > WebSocket "
                   "Server Settings > Show Connect Info.")
        raise ObsError(msg) from None   # `from None`: the library's own error is not carried along
    # The library keeps the password on the client (and prints it in its repr). Drop it now.
    try:
        client.base_client.password = None
    except Exception:
        pass
    return ObsSession(client, where=f"{host}:{int(port)}")


# ---------------------------------------------------------------- plan

@dataclass
class PlanRow:
    name: str
    before: int                 # the Sync Offset OBS has now, ms
    after: int                  # what it would be set to, ms
    ticked: bool = True
    is_npass_mic: bool = False  # the filtered mic itself: never changed
    note: str = ""

    def label(self) -> str:
        if self.is_npass_mic:
            return f"{self.name}: {self.before} ms (the N-Pass mic, stays as it is)"
        if self.before == self.after:
            return f"{self.name}: {self.before} ms (already right)"
        return f"{self.name}: {self.before} ms -> {self.after} ms"


@dataclass
class SyncResult:
    changed: list[str] = field(default_factory=list)      # "Name: 0 ms -> 487 ms"
    problems: list[str] = field(default_factory=list)     # plain English, shown as-is

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        lines = []
        if self.changed:
            lines.append("Changed in OBS:")
            lines += [f"  {c}" for c in self.changed]
        elif not self.problems:
            lines.append("Nothing needed changing.")
        lines += self.problems
        return "\n".join(lines)


def clamp_offset(ms: float) -> int:
    return max(OBS_MIN_OFFSET_MS, min(OBS_MAX_OFFSET_MS, int(round(ms))))


def load_undo() -> dict[str, dict[str, int]]:
    """{input name: {"before": ms, "after": ms}} from the last Apply. {} when there is none."""
    try:
        raw = json.loads(undo_path().read_text(encoding="utf-8"))
        inputs = raw.get("inputs", {})
        return {str(k): {"before": int(v["before"]), "after": int(v["after"])}
                for k, v in inputs.items()}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def _save_undo(entries: dict[str, dict[str, int]]) -> None:
    path = undo_path()
    if not entries:
        try:
            path.unlink()
        except OSError:
            pass
        return
    body = {"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "inputs": entries}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(body, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def has_undo() -> bool:
    return bool(load_undo())


def _looks_like_npass_mic(session: ObsSession, name: str) -> bool:
    if "cable output" in name.lower():
        return True
    return "cable output" in session.device_label(name).lower()


def build_plan(session: ObsSession, delay_ms: float) -> list[PlanRow]:
    """One row per audio input: its Sync Offset now and what N-Pass would set. Changes nothing.

    The new value is the input's own offset plus the delay. If N-Pass set this input before (the
    undo file says so and OBS still has that value), the plan starts again from the value it had
    before N-Pass touched it, so pressing Apply twice never adds the delay twice.
    """
    delay = int(round(delay_ms))
    undo = load_undo()
    rows: list[PlanRow] = []
    for name, _kind in session.list_inputs():
        try:
            before = session.get_offset(name)
        except Exception:
            continue   # OBS refuses this request for inputs without sound: not an audio input
        if _looks_like_npass_mic(session, name):
            rows.append(PlanRow(name, before, before, ticked=False, is_npass_mic=True))
            continue
        base = before
        mine = undo.get(name)
        if mine and mine["after"] == before:
            base = mine["before"]
        after = clamp_offset(base + delay)
        rows.append(PlanRow(name, before, after, ticked=(after != before)))
    return rows


def _read_back(session: ObsSession, name: str, want: int,
               sleep: Callable[[float], None]) -> int | None:
    """Read the offset until it is `want`. Returns the last value seen (None: could not read)."""
    seen: int | None = None
    for attempt in range(READBACK_TRIES):
        try:
            seen = session.get_offset(name)
        except Exception:
            seen = None
        if seen == want:
            return seen
        if attempt < READBACK_TRIES - 1:
            sleep(READBACK_WAIT_S)   # OBS can answer before the change has landed
    return seen


def apply_plan(session: ObsSession, rows: list[PlanRow],
               sleep: Callable[[float], None] = time.sleep) -> SyncResult:
    """Set the Sync Offset of every ticked row, then read each one back to make sure it took.

    The "before" values are written to the undo file first, so a half-finished Apply can be undone.
    """
    result = SyncResult()
    todo = [r for r in rows if r.ticked and not r.is_npass_mic and r.after != r.before]
    if not todo:
        return result

    undo = load_undo()
    for r in todo:
        old = undo.get(r.name)
        # Keep the oldest "before" while OBS still holds the value N-Pass set last time.
        before = old["before"] if old and old["after"] == r.before else r.before
        undo[r.name] = {"before": before, "after": r.after}
    try:
        _save_undo(undo)
    except OSError:
        result.problems.append("Could not save the undo file, so nothing was changed in OBS.")
        return result

    for r in todo:
        try:
            session.set_offset(r.name, r.after)
        except Exception:
            result.problems.append(f'OBS refused the change for "{r.name}". It was left as it is.')
            continue
        seen = _read_back(session, r.name, r.after, sleep)
        if seen == r.after:
            result.changed.append(f"{r.name}: {r.before} ms -> {r.after} ms")
        elif seen is None:
            result.problems.append(f'Could not check "{r.name}" after the change. Look at it in OBS.')
        else:
            result.problems.append(
                f'"{r.name}" should be {r.after} ms but OBS says {seen} ms. Set it by hand in OBS.')
    return result


def undo_last(session: ObsSession, sleep: Callable[[float], None] = time.sleep) -> SyncResult:
    """Put back the values saved by the last Apply. Entries that could not be restored are kept."""
    result = SyncResult()
    undo = load_undo()
    if not undo:
        result.problems.append("There is nothing to undo.")
        return result
    left: dict[str, dict[str, int]] = {}
    for name, entry in undo.items():
        want = entry["before"]
        try:
            now = session.get_offset(name)
            session.set_offset(name, want)
        except Exception:
            result.problems.append(f'Could not put "{name}" back. Is it still in OBS?')
            left[name] = entry
            continue
        seen = _read_back(session, name, want, sleep)
        if seen == want:
            if now != want:
                result.changed.append(f"{name}: {now} ms -> {want} ms")
        else:
            shown = "nothing" if seen is None else f"{seen} ms"
            result.problems.append(
                f'"{name}" should be back at {want} ms but OBS says {shown}. Set it by hand in OBS.')
            left[name] = entry
    try:
        _save_undo(left)
    except OSError:
        pass
    return result
