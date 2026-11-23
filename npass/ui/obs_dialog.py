"""The "OBS sync" dialog: the manual steps, plus an optional one-click change of the audio Sync
Offsets through obs-websocket (npass/obs_sync.py does the work).

The password is read from the box when "Connect" is pressed, handed to the connection, and the box
is cleared. It is not kept on this object, not saved and not logged.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from npass import obs_sync, settings
from npass.ui import theme


class ObsSyncDialog:
    def __init__(self, window: Any, delay_ms: float, connect: Callable[..., Any] | None = None,
                 host: str | None = None, port: int | None = None):
        self.window = window                 # the MainWindow: gives px(), post() and background runs
        self.delay_ms = int(round(delay_ms))
        self._connect = connect or obs_sync.connect
        if host is None or port is None:
            saved = settings.load_settings()
            host = host or saved.obs_ws_host
            port = port or saved.obs_ws_port
        self.host, self.port = host, int(port)
        self.session: Any = None
        self.rows: list[obs_sync.PlanRow] = []
        self.row_vars: list[tk.BooleanVar] = []
        self.busy = False
        px = window.px

        self.top = tk.Toplevel(window.root)
        self.top.title("N-Pass: OBS sync")
        self.top.configure(bg=theme.BG)
        self.top.resizable(False, False)
        self.top.protocol("WM_DELETE_WINDOW", self.close)
        body = ttk.Frame(self.top, padding=px(14))
        body.pack(fill="both", expand=True)
        wrap = px(500)

        ttk.Label(body, text="Do it by hand", style="Head.TLabel").pack(fill="x")
        steps = obs_sync.steps_text(self.delay_ms)
        self.steps = tk.Text(body, width=86, height=steps.count("\n") + 1, bg=theme.PANEL, fg=theme.TEXT,
                             relief="flat", font=(theme.FONT, 9), padx=px(8), pady=px(6), wrap="word",
                             highlightthickness=0)
        self.steps.insert("1.0", steps)
        self.steps.configure(state="disabled")
        self.steps.pack(fill="x", pady=(px(4), px(12)))

        ttk.Label(body, text="Or let N-Pass set the Sync Offsets", style="Head.TLabel").pack(fill="x")
        ttk.Label(body, style="Dim.TLabel", wraplength=wrap, justify="left",
                  text="This only changes the Sync Offset of the sound sources you tick. The camera "
                       "delay stays a job for step 3 above. Find the password in OBS under Tools > "
                       "WebSocket Server Settings > Show Connect Info. It is used once and not saved."
                  ).pack(fill="x", pady=(px(2), px(6)))
        row = ttk.Frame(body)
        row.pack(fill="x")
        ttk.Label(row, text="OBS WebSocket password").pack(side="left")
        self.password_var = tk.StringVar()
        self.password_entry = ttk.Entry(row, textvariable=self.password_var, show="•", width=24)
        self.password_entry.pack(side="left", padx=px(8))
        self.password_entry.bind("<Return>", lambda e: self.connect_clicked())
        self.connect_btn = ttk.Button(row, text="Connect", command=self.connect_clicked)
        self.connect_btn.pack(side="left")

        self.rows_frame = tk.Frame(body, bg=theme.PANEL, padx=px(8), pady=px(6))
        self.rows_frame.pack(fill="x", pady=(px(8), 0))
        self.rows_hint = tk.Label(self.rows_frame, bg=theme.PANEL, fg=theme.DIM, font=(theme.FONT, 9),
                                  text="Connect to see your OBS sound sources here.", anchor="w")
        self.rows_hint.pack(fill="x")

        row = ttk.Frame(body)
        row.pack(fill="x", pady=(px(8), 0))
        self.apply_btn = ttk.Button(row, text="Apply", command=self.apply_clicked, state="disabled")
        self.apply_btn.pack(side="left")
        self.undo_btn = ttk.Button(row, text="Undo last change", command=self.undo_clicked,
                                   state="disabled")
        self.undo_btn.pack(side="left", padx=(px(8), 0))
        ttk.Button(row, text="Close", command=self.close).pack(side="right")
        self.status_lbl = ttk.Label(body, wraplength=wrap, justify="left")
        self.status_lbl.pack(fill="x", pady=(px(8), 0))

    # ------------------------------------------------------------------ helpers

    def alive(self) -> bool:
        try:
            return bool(self.top.winfo_exists())
        except tk.TclError:
            return False

    def lift(self) -> None:
        self.top.deiconify()
        self.top.lift()

    def status(self) -> str:
        return str(self.status_lbl.cget("text"))

    def _say(self, text: str, colour: str = theme.TEXT) -> None:
        if self.alive():
            self.status_lbl.configure(text=text, foreground=colour)

    def _run(self, work: Callable[[], Any], done: Callable[[Any], None], busy_text: str) -> None:
        """Do `work` off the window thread. `done` gets its result, or an ObsError to show."""
        if self.busy:
            return
        self.busy = True
        self._say(busy_text, theme.DIM)
        self._buttons()

        def safe() -> Any:
            try:
                return work()
            except obs_sync.ObsError as e:
                return e
            except Exception:
                return obs_sync.ObsError("Something went wrong while talking to OBS. Is it still open?")

        def finish(result: Any) -> None:
            self.busy = False
            if not self.alive():
                if isinstance(result, obs_sync.ObsSession):
                    result.close()
                return
            if isinstance(result, obs_sync.ObsError):
                self._say(str(result), theme.RED)
            else:
                done(result)
            self._buttons()

        self.window._in_background(safe, finish)

    def _buttons(self) -> None:
        if not self.alive():
            return
        connected = self.session is not None
        idle = not self.busy
        self.connect_btn.configure(state="normal" if idle else "disabled",
                                   text="Connect again" if connected else "Connect")
        can_apply = connected and idle and any(
            r.ticked and not r.is_npass_mic and r.after != r.before for r in self._ticked_rows())
        self.apply_btn.configure(state="normal" if can_apply else "disabled")
        self.undo_btn.configure(state="normal" if connected and idle and obs_sync.has_undo() else "disabled")

    def _ticked_rows(self) -> list[obs_sync.PlanRow]:
        for r, var in zip(self.rows, self.row_vars):
            r.ticked = bool(var.get()) and not r.is_npass_mic
        return self.rows

    def _show_rows(self, rows: list[obs_sync.PlanRow]) -> None:
        # a source the user unticked stays unticked when the list is read again
        unticked = {r.name for r, var in zip(self.rows, self.row_vars)
                    if not var.get() and not r.is_npass_mic and r.after != r.before}
        for r in rows:
            if r.name in unticked:
                r.ticked = False
        for child in self.rows_frame.winfo_children():
            child.destroy()
        self.rows = rows
        self.row_vars = []
        self.row_checks: list[ttk.Checkbutton] = []
        if not rows:
            tk.Label(self.rows_frame, bg=theme.PANEL, fg=theme.DIM, font=(theme.FONT, 9), anchor="w",
                     text="OBS has no sound sources to change.").pack(fill="x")
            return
        style = ttk.Style(self.top)
        style.configure("Row.TCheckbutton", background=theme.PANEL)
        style.map("Row.TCheckbutton", background=[("active", theme.PANEL)],
                  foreground=[("disabled", theme.DIM)])
        for r in rows:
            var = tk.BooleanVar(value=r.ticked and not r.is_npass_mic)
            locked = r.is_npass_mic or r.after == r.before
            check = ttk.Checkbutton(self.rows_frame, text=r.label(), variable=var, style="Row.TCheckbutton",
                                    state="disabled" if locked else "normal", command=self._buttons)
            check.pack(fill="x", anchor="w")
            self.row_vars.append(var)
            self.row_checks.append(check)

    def row_labels(self) -> list[str]:
        return [r.label() for r in self.rows]

    # ------------------------------------------------------------------ actions

    def connect_clicked(self) -> None:
        if self.busy:
            return
        password = self.password_var.get()
        self.password_var.set("")            # the box is emptied at once; nothing keeps the password
        old, self.session = self.session, None
        host, port, delay, connect = self.host, self.port, self.delay_ms, self._connect

        def work() -> Any:
            if old is not None:
                old.close()
            session = connect(host, port, password)
            try:
                return session, obs_sync.build_plan(session, delay)
            except Exception:
                session.close()
                raise

        def done(result: Any) -> None:
            self.session, rows = result
            self._show_rows(rows)
            self._say("Connected. Tick the sources to change, then press Apply.")

        self._run(work, done, "Connecting to OBS")

    def apply_clicked(self) -> None:
        session = self.session
        if session is None:
            return
        rows = [obs_sync.PlanRow(r.name, r.before, r.after, r.ticked, r.is_npass_mic, r.note)
                for r in self._ticked_rows()]
        delay = self.delay_ms

        def work() -> Any:
            result = obs_sync.apply_plan(session, rows)
            return result, obs_sync.build_plan(session, delay)

        def done(pair: Any) -> None:
            result, fresh = pair
            self._show_rows(fresh)
            self._say(result.summary(), theme.GREEN if result.ok else theme.YELLOW)

        self._run(work, done, "Changing the Sync Offsets in OBS")

    def undo_clicked(self) -> None:
        session = self.session
        if session is None:
            return
        delay = self.delay_ms

        def work() -> Any:
            result = obs_sync.undo_last(session)
            return result, obs_sync.build_plan(session, delay)

        def done(pair: Any) -> None:
            result, fresh = pair
            self._show_rows(fresh)
            text = result.summary()
            if result.ok and result.changed:
                text = "Put back:\n" + "\n".join(f"  {c}" for c in result.changed)
            self._say(text, theme.GREEN if result.ok else theme.YELLOW)

        self._run(work, done, "Putting the old values back")

    def close(self) -> None:
        session, self.session = self.session, None
        if session is not None:
            session.close()
        try:
            self.password_var.set("")
            self.top.destroy()
        except tk.TclError:
            pass
