"""The N-Pass window. It is driven only by a `Core` object (docs/CONTRACT.md).

`run_ui(core)` builds the window, the tray icon and the hotkey and runs until Quit.
`MainWindow` on its own (no tray, no hotkey, no main loop) is what the tests use.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable

from npass import catchlog, settings
from npass.types import Catch, CutReason, Health
from npass.ui import theme
from npass.ui.hotkey import HotkeyListener, pretty
from npass.ui.tray import Tray, ensure_ico, set_app_id, tray_state

log = logging.getLogger(__name__)

WIDTH = 420
REFRESH_MS = 66            # about 15 times a second
CATCH_ROWS = 5
DOT = "•"
README = Path(__file__).resolve().parents[2] / "README.md"

TEXT_OFF = "Off. Your voice passes through with the same delay. Nothing is cut."
TEXT_LATE = "Recognition is falling behind"
TEXT_DOWN = "Recognition is not running. Risky lyric lines are being muted."
TEXT_NO_CABLE = "The virtual cable is missing, so OBS cannot hear the filtered mic."
TEXT_FIX_FALLBACK = (
    "N-Pass sends the filtered mic to OBS through a free virtual cable called VB-CABLE.\n\n"
    "1. Install VB-CABLE (from vb-audio.com/Cable).\n"
    "2. Restart the PC.\n"
    "3. Start N-Pass again.\n"
    '4. In OBS, pick "CABLE Output (VB-Audio Virtual Cable)" as the mic.'
)


def mask_word(word: str) -> str:
    """First letter, then dots: enough to recognise your own word, not readable on stream."""
    word = word.strip()
    if len(word) <= 1:
        return DOT * len(word)
    return word[0] + DOT * (len(word) - 1)


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: max(1, limit - 3)].rstrip() + "..."


def catch_columns(catch: Catch, show_words: bool) -> tuple[str, str, str, str]:
    """(time, word, score, song) for one row of the last-catches list."""
    when = time.strftime("%H:%M:%S", time.localtime(catch.wall_time))
    if catch.order.reason == CutReason.FAILSAFE and not catch.order.word:
        word, score = "line muted", "-"
    else:
        raw = catchlog.display_word(catch)
        word = raw if show_words else mask_word(raw)
        score = f"{catch.order.confidence:.2f}"
    song = catch.track.label() if catch.track is not None else "no song"
    if catch.late_ms > 0:
        song = f"late {catch.late_ms:.0f} ms, {song}"
    return when, shorten(word, 11), score, shorten(song, 28)


def collect_warnings(snap: dict, local: list[str]) -> list[tuple[str, str]]:
    """Everything for the warning area as (level, text). Levels: "red" | "yellow"."""
    out: list[tuple[str, str]] = []
    if snap.get("enabled", False):
        health = snap.get("health", Health.OK)
        if health == Health.DOWN:
            out.append(("red", TEXT_DOWN))
        elif health == Health.LATE:
            out.append(("yellow", TEXT_LATE))
    seen = {text for _lvl, text in out}
    engine = snap.get("engine")
    for text in list(snap.get("warnings") or []) + list(getattr(engine, "warnings", None) or []) + list(local):
        text = str(text).strip()
        if text and text not in seen:
            seen.add(text)
            out.append(("yellow", text))
    return out


class MainWindow:
    def __init__(self, core: Any, root: tk.Tk | None = None, readme_path: Path | None = None,
                 opener: Callable[[str], None] | None = None,
                 inline_threads: bool = False):
        self.core = core
        self.root = root if root is not None else tk.Tk()
        self.readme_path = Path(readme_path) if readme_path is not None else README
        self.opener = opener or (lambda path: os.startfile(path))   # noqa: S606 (opens a local file)
        self.inline_threads = inline_threads     # tests: run "background" work right away
        self.tray: Tray | None = None
        self.local_warnings: list[str] = []      # lines from the UI itself (hotkey taken, ...)
        self.quit_hooks: list[Callable[[], None]] = []
        self.closed = False
        self.obs_dialog: Any = None
        self._calls: "queue.SimpleQueue[Callable[[], None]]" = queue.SimpleQueue()
        self._after_id: str | None = None
        self._cache: dict[tuple[int, str], Any] = {}
        self._seen: dict[str, Any] = {}          # last value of each setting that came from the core
        self._dragging = False
        self._login_busy = False
        self._catch_key: Any = None
        self._warn_key: Any = None
        self._told_tray = False
        self._last_snap: dict = {}
        self._words_buffer: str | None = None    # the editor's text while it is hidden behind dots
        self.s = theme.scale_of(self.root)

        self.root.title("N-Pass")
        self.root.configure(bg=theme.BG)
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.hide)
        self.root.report_callback_exception = self._callback_error
        theme.apply_style(self.root)

        self._build()
        self.refresh()

    # ------------------------------------------------------------------ small helpers

    def px(self, n: float) -> int:
        return int(round(n * self.s))

    def post(self, fn: Callable[[], None]) -> None:
        """Ask the window to run `fn` on its own thread soon. Safe from any thread."""
        self._calls.put(fn)

    def _set(self, widget: tk.Misc, **opts: Any) -> None:
        """configure(), skipped when nothing changed (this runs 15 times a second)."""
        for key, value in opts.items():
            slot = (id(widget), key)
            if self._cache.get(slot) != value:
                self._cache[slot] = value
                widget.configure(**{key: value})

    def _callback_error(self, exc: type, val: BaseException, tb: Any) -> None:
        log.error("window error: %s", exc.__name__, exc_info=(exc, val, tb))

    def _in_background(self, work: Callable[[], Any], done: Callable[[Any], None]) -> None:
        """Run `work` off the window thread, then `done(result)` back on it."""
        if self.inline_threads:
            done(work())
            return
        threading.Thread(target=lambda: (lambda r: self.post(lambda: done(r)))(work()),
                         name="npass-ui-work", daemon=True).start()

    # ------------------------------------------------------------------ building

    def _build(self) -> None:
        tk.Frame(self.root, width=self.px(WIDTH), height=0, bg=theme.BG).pack()   # fixes the width
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True)
        self.main_tab = ttk.Frame(self.tabs, padding=self.px(14))
        self.settings_tab = ttk.Frame(self.tabs, padding=self.px(14))
        self.tabs.add(self.main_tab, text="Live")
        self.tabs.add(self.settings_tab, text="Settings")
        self._build_main(self.main_tab)
        self._build_settings(self.settings_tab)

    def _build_main(self, f: ttk.Frame) -> None:
        wrap = self.px(WIDTH - 44)
        self.toggle_btn = tk.Button(
            f, text="ON", font=(theme.FONT, 24, "bold"), fg="#0c1f14", bg=theme.GREEN,
            activebackground=theme.GREEN, activeforeground="#0c1f14", relief="flat", bd=0,
            cursor="hand2", command=self.toggle, pady=self.px(6))
        self.toggle_btn.pack(fill="x")
        self.hotkey_lbl = ttk.Label(f, style="Dim.TLabel", anchor="center")
        self.hotkey_lbl.pack(fill="x", pady=(self.px(4), self.px(12)))

        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Label(row, text="Mic", style="Head.TLabel", width=5).pack(side="left")
        self.meter = tk.Canvas(row, height=self.px(20), bg=theme.FIELD, highlightthickness=0)
        self.meter.pack(side="left", fill="x", expand=True)
        self._meter_bar = self.meter.create_rectangle(0, 0, 0, 0, width=0, fill=theme.GREEN)
        self._meter_text = self.meter.create_text(0, 0, text="", fill="#ffffff",
                                                  font=(theme.FONT, 9, "bold"))
        self.meter_text_value = ""

        self.delay_lbl = ttk.Label(f, wraplength=wrap, justify="left")
        self.delay_lbl.pack(fill="x", pady=(self.px(10), 0))

        row = ttk.Frame(f)
        row.pack(fill="x", pady=(self.px(4), 0))
        ttk.Label(row, text="Look-ahead", style="Dim.TLabel").pack(side="left")
        self.look_var = tk.DoubleVar(value=450)
        self.look_value_lbl = ttk.Label(row, width=7, anchor="e")
        self.look_value_lbl.pack(side="right")
        self.look_scale = ttk.Scale(row, from_=settings.LOOKAHEAD_MIN_MS, to=settings.LOOKAHEAD_MAX_MS,
                                    orient="horizontal", variable=self.look_var,
                                    command=self._look_moved)
        self.look_scale.pack(side="left", fill="x", expand=True, padx=self.px(8))
        self.look_scale.bind("<ButtonPress-1>", lambda e: setattr(self, "_dragging", True))
        self.look_scale.bind("<ButtonRelease-1>", lambda e: self.commit_lookahead())
        self.look_scale.bind("<KeyRelease>", lambda e: self.commit_lookahead())
        ttk.Label(f, text="Lower is less delay. Higher gives the filter more time to catch a word.",
                  style="Dim.TLabel", wraplength=wrap, justify="left").pack(fill="x")

        ttk.Separator(f).pack(fill="x", pady=self.px(10))
        self.caught_lbl = ttk.Label(f, style="Count.TLabel")
        self.caught_lbl.pack(fill="x")
        self.catch_frame = tk.Frame(f, bg=theme.PANEL, padx=self.px(8), pady=self.px(6))
        self.catch_frame.pack(fill="x", pady=(self.px(4), 0))
        self.catch_frame.columnconfigure(3, weight=1)
        mono = ("Consolas", 9)
        self.catch_rows: list[list[tk.Label]] = []
        for r in range(CATCH_ROWS):
            cells = []
            for c, (width, anchor) in enumerate(((8, "w"), (11, "w"), (4, "e"), (0, "w"))):
                lbl = tk.Label(self.catch_frame, bg=theme.PANEL, fg=theme.TEXT, font=mono, anchor=anchor,
                               width=width if width else 1)
                lbl.grid(row=r, column=c, sticky="we", padx=(0, self.px(8) if c < 3 else 0))
                cells.append(lbl)
            self.catch_rows.append(cells)
        self.no_catch_lbl = tk.Label(self.catch_frame, bg=theme.PANEL, fg=theme.DIM,
                                     font=(theme.FONT, 9), text="Nothing caught yet.", anchor="w")

        ttk.Separator(f).pack(fill="x", pady=self.px(10))
        self.engine_lbl = ttk.Label(f, wraplength=wrap, justify="left")
        self.engine_lbl.pack(fill="x")
        self.lyrics_lbl = ttk.Label(f, style="Dim.TLabel", wraplength=wrap, justify="left")
        self.lyrics_lbl.pack(fill="x", pady=(self.px(2), 0))
        self.song_lbl = ttk.Label(f, style="Dim.TLabel", wraplength=wrap, justify="left")
        self.song_lbl.pack(fill="x", pady=(self.px(2), 0))

        self.warn_frame = tk.Frame(f, bg=theme.BG)
        self.warn_frame.pack(fill="x", pady=(self.px(8), 0))
        self.warning_labels: list[tk.Label] = []
        self.cable_box: tk.Frame | None = None
        self.fix_btn: ttk.Button | None = None

    def _build_settings(self, f: ttk.Frame) -> None:
        wrap = self.px(WIDTH - 44)
        gap = self.px(10)

        ttk.Label(f, text="Microphone", style="Head.TLabel").pack(fill="x")
        self.mic_var = tk.StringVar()
        self.mic_box = ttk.Combobox(f, textvariable=self.mic_var, state="readonly",
                                    postcommand=self._fill_mics)
        self.mic_box.pack(fill="x", pady=(self.px(2), gap))
        self.mic_box.bind("<<ComboboxSelected>>", lambda e: self._mic_picked())

        ttk.Label(f, text="How a caught word is removed", style="Head.TLabel").pack(fill="x")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(0, gap))
        self.removal_var = tk.StringVar(value="silence")
        for text, value in (("Silence", "silence"), ("Soft dip", "dip")):
            ttk.Radiobutton(row, text=text, value=value, variable=self.removal_var,
                            command=lambda: self.core.set_removal(self.removal_var.get())
                            ).pack(side="left", padx=(0, self.px(16)))

        ttk.Label(f, text="Recognition runs on", style="Head.TLabel").pack(fill="x")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(0, gap))
        self.engine_var = tk.StringVar(value="auto")
        for text, value in (("Auto", "auto"), ("GPU", "gpu"), ("CPU only", "cpu")):
            ttk.Radiobutton(row, text=text, value=value, variable=self.engine_var,
                            command=lambda: self.core.set_engine(self.engine_var.get())
                            ).pack(side="left", padx=(0, self.px(16)))

        ttk.Label(f, text="Song lyrics", style="Head.TLabel").pack(fill="x")
        self.lyrics_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Use Spotify lyrics to watch risky lines", variable=self.lyrics_var,
                        command=lambda: self.core.set_lyrics_enabled(bool(self.lyrics_var.get()))
                        ).pack(fill="x")
        self.risky_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Mute the whole risky line (safest, cuts more)",
                        variable=self.risky_var, command=self._set_risky).pack(fill="x")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(self.px(4), 0))
        ttk.Label(row, text="Spotify client ID", style="Dim.TLabel").pack(side="left")
        self.cid_var = tk.StringVar()
        self.cid_entry = ttk.Entry(row, textvariable=self.cid_var)
        self.cid_entry.pack(side="left", fill="x", expand=True, padx=(self.px(8), 0))
        self.cid_entry.bind("<FocusOut>", lambda e: self.commit_client_id())
        self.cid_entry.bind("<Return>", lambda e: self.commit_client_id())
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(self.px(6), gap))
        self.spotify_btn = ttk.Button(row, text="Sign in to Spotify", command=self.spotify_clicked)
        self.spotify_btn.pack(side="left")
        self.spotify_lbl = ttk.Label(row, style="Dim.TLabel", wraplength=self.px(220), justify="left")
        self.spotify_lbl.pack(side="left", padx=(self.px(10), 0))

        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Label(row, text="Blocked words, one per line", style="Head.TLabel").pack(side="left")
        self.show_words_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row, text="Show words", variable=self.show_words_var,
                        command=self._show_words_changed).pack(side="right")
        self.words_text = tk.Text(f, height=5, width=1, bg=theme.FIELD, fg=theme.TEXT, insertbackground=theme.TEXT,
                                  relief="flat", font=(theme.FONT, 10), padx=self.px(6), pady=self.px(4),
                                  highlightthickness=1, highlightbackground=theme.LINE,
                                  highlightcolor=theme.BLUE, undo=True, wrap="none")
        self.words_text.pack(fill="x", pady=(self.px(2), 0))
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(self.px(6), gap))
        self.words_save_btn = ttk.Button(row, text="Save", command=self.save_words)
        self.words_save_btn.pack(side="left")
        self.words_lbl = ttk.Label(row, style="Dim.TLabel", wraplength=self.px(280), justify="left")
        self.words_lbl.pack(side="left", padx=(self.px(10), 0))

        ttk.Separator(f).pack(fill="x", pady=(0, gap))
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Button(row, text="Open catch log", command=self.open_catch_log).pack(side="left")
        ttk.Button(row, text="OBS sync", command=self.open_obs_sync).pack(side="left", padx=(self.px(8), 0))
        self.tools_lbl = ttk.Label(f, style="Dim.TLabel", wraplength=wrap, justify="left")
        self.tools_lbl.pack(fill="x", pady=(self.px(6), 0))

        self._fill_mics()
        self._render_words()

    # ------------------------------------------------------------------ refresh (15x/s)

    def start(self) -> None:
        """Begin the regular refresh. Only run_ui calls this; tests call refresh() themselves."""
        self._tick()

    def _tick(self) -> None:
        if self.closed:
            return
        try:
            self.refresh()
        except Exception:
            log.exception("refresh failed")
        if not self.closed:
            self._after_id = self.root.after(REFRESH_MS, self._tick)

    def refresh(self) -> None:
        """Run queued calls from other threads, then redraw from one snapshot."""
        while True:
            try:
                fn = self._calls.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception:
                log.exception("queued call failed")
            if self.closed:
                return
        snap = self.core.snapshot()
        self._last_snap = snap
        self._draw_main(snap)
        self._sync_settings(snap)
        if self.tray is not None:
            self.tray.update(tray_state(snap, bool(self.local_warnings)), bool(snap.get("enabled")))

    def _draw_main(self, snap: dict) -> None:
        enabled = bool(snap.get("enabled", False))
        engine = snap.get("engine")
        key = pretty(str(snap.get("hotkey", "")))
        if enabled:
            self._set(self.toggle_btn, text="ON", bg=theme.GREEN, activebackground=theme.GREEN,
                      fg="#0c1f14", activeforeground="#0c1f14")
        else:
            self._set(self.toggle_btn, text="OFF", bg=theme.GREY, activebackground=theme.GREY,
                      fg="#ffffff", activeforeground="#ffffff")
        action = "turn off" if enabled else "turn on"
        self._set(self.hotkey_lbl, text=(f"Click, or press {key}, to {action}" if key
                                         else f"Click to {action}"))

        self._draw_meter(float(getattr(engine, "level_db", -120.0)), bool(getattr(engine, "cutting", False)))

        look = int(snap.get("lookahead_ms", 0))
        delay = float(getattr(engine, "measured_delay_ms", 0.0))
        if delay > 0:
            too_long = delay > settings.MAX_TOTAL_DELAY_MS
            text = f"Delay: {delay:.0f} ms (look-ahead {look} ms)"
            if too_long:
                text += ". Too long, lower the look-ahead."
            self._set(self.delay_lbl, text=text, foreground=theme.RED if too_long else theme.TEXT)
        else:
            self._set(self.delay_lbl, text=f"Delay: measuring (look-ahead {look} ms)",
                      foreground=theme.DIM)
        if not self._dragging and self._seen.get("look") != look:
            self._seen["look"] = look
            self.look_var.set(look)
            self._set(self.look_value_lbl, text=f"{look} ms")

        caught = int(snap.get("caught", 0))
        self._set(self.caught_lbl, text=f"Caught: {caught}")
        show = bool(self.show_words_var.get())
        if self._catch_key != (caught, show):
            self._catch_key = (caught, show)
            self._draw_catches(show)

        if not enabled:
            self._set(self.engine_lbl, text=TEXT_OFF, foreground=theme.DIM)
        elif snap.get("health") == Health.STARTING:
            self._set(self.engine_lbl, text="Recognition is starting. This can take a few seconds.",
                      foreground=theme.YELLOW)
        else:
            name = str(snap.get("engine_name", "") or "nothing yet")
            ms = float(snap.get("pass_ms", 0.0))
            self._set(self.engine_lbl, text=f"Listening with: {name}, {ms:.0f} ms per check",
                      foreground=theme.TEXT)
        self._set(self.lyrics_lbl, text="Lyrics: " + (str(snap.get("lyrics_status") or "").strip()
                                                      or "nothing to report"))
        track = snap.get("track")
        if track is None:
            self._set(self.song_lbl, text="Song: none playing", foreground=theme.DIM)
        elif snap.get("in_window"):
            self._set(self.song_lbl, text=f"Song: {shorten(track.label(), 44)}. Risky line now.",
                      foreground=theme.YELLOW)
        else:
            self._set(self.song_lbl, text=f"Song: {shorten(track.label(), 60)}", foreground=theme.DIM)

        self._draw_warnings(collect_warnings(snap, self.local_warnings),
                            cable_missing=not snap.get("cable_found", True))

    def _draw_meter(self, level_db: float, cutting: bool) -> None:
        w = max(1, self.meter.winfo_width())
        h = max(1, self.meter.winfo_height())
        if cutting:
            fill, frac, text = theme.GREY, 1.0, "Cutting"
        else:
            frac = max(0.0, min(1.0, (level_db + 60.0) / 60.0))
            fill = theme.RED if level_db > -6.0 else theme.YELLOW if level_db > -18.0 else theme.GREEN
            text = ""
        self.meter.coords(self._meter_bar, 0, 0, int(w * frac), h)
        self.meter.coords(self._meter_text, w // 2, h // 2)
        slot = (id(self.meter), "look")
        if self._cache.get(slot) != (fill, text):
            self._cache[slot] = (fill, text)
            self.meter.itemconfigure(self._meter_bar, fill=fill)
            self.meter.itemconfigure(self._meter_text, text=text)
        self.meter_text_value = text
        self.meter_fill_value = fill

    def _draw_catches(self, show_words: bool) -> None:
        try:
            catches = list(self.core.recent_catches())[-CATCH_ROWS:]
        except Exception:
            catches = []
        catches.reverse()   # newest on top
        if catches:
            self.no_catch_lbl.grid_forget()
        else:
            self.no_catch_lbl.grid(row=0, column=0, columnspan=4, sticky="w")
        for r, cells in enumerate(self.catch_rows):
            if r < len(catches):
                cols = catch_columns(catches[r], show_words)
                late = catches[r].late_ms > 0
                for c, (lbl, text) in enumerate(zip(cells, cols)):
                    lbl.configure(text=text, fg=theme.YELLOW if late and c == 3 else
                                  theme.DIM if c in (0, 3) else theme.TEXT)
            else:
                for lbl in cells:
                    lbl.configure(text="")

    def _draw_warnings(self, items: list[tuple[str, str]], cable_missing: bool) -> None:
        key = (tuple(items), cable_missing)
        if key == self._warn_key:
            return
        self._warn_key = key
        for child in self.warn_frame.winfo_children():
            child.destroy()
        self.warning_labels = []
        self.cable_box = None
        self.fix_btn = None
        wrap = self.px(WIDTH - 44)
        if cable_missing:
            box = tk.Frame(self.warn_frame, bg=theme.RED_BOX, highlightthickness=1,
                           highlightbackground=theme.RED, padx=self.px(10), pady=self.px(8))
            box.pack(fill="x", pady=(0, self.px(6)))
            tk.Label(box, text=TEXT_NO_CABLE, bg=theme.RED_BOX, fg="#ffd9d4", font=(theme.FONT, 10, "bold"),
                     wraplength=wrap - self.px(110), justify="left", anchor="w").pack(side="left", fill="x",
                                                                                     expand=True)
            self.fix_btn = ttk.Button(box, text="How to fix", command=self.how_to_fix)
            self.fix_btn.pack(side="right", padx=(self.px(8), 0))
            self.cable_box = box
        for level, text in items:
            colour = theme.RED if level == "red" else theme.YELLOW
            lbl = tk.Label(self.warn_frame, text=text, bg=theme.BG, fg=colour, font=(theme.FONT, 10, "bold"),
                           wraplength=wrap, justify="left", anchor="w")
            lbl.pack(fill="x", pady=(0, self.px(3)))
            self.warning_labels.append(lbl)

    def warning_texts(self) -> list[str]:
        return [str(lbl.cget("text")) for lbl in self.warning_labels]

    def _sync_settings(self, snap: dict) -> None:
        """Copy a setting into its control only when the core's value changed, so we never fight
        the user's hand on the control."""
        engine = snap.get("engine")
        pairs = (
            ("mic", str(getattr(engine, "input_name", "") or ""), self.mic_var),
            ("removal", str(snap.get("removal_mode", "silence")), self.removal_var),
            ("lyrics", bool(snap.get("lyrics_enabled", True)), self.lyrics_var),
            ("risky", bool(snap.get("mute_risky_lines", False)), self.risky_var),
        )
        for name, value, var in pairs:
            if self._seen.get(name) != value:
                self._seen[name] = value
                var.set(value)
        # Which engine the user chose is not in the snapshot (the contract has no key for it), so
        # read `engine_choice` when a core offers it and fall back to the saved setting once.
        choice = snap.get("engine_choice")
        if choice is None and "engine" not in self._seen:
            choice = settings.load_settings().engine
        if choice is not None and self._seen.get("engine") != choice:
            self._seen["engine"] = choice
            self.engine_var.set(choice)
        cid = str(snap.get("spotify_client_id", "") or "")
        if self._seen.get("cid") != cid and self.root.focus_get() is not self.cid_entry:
            self._seen["cid"] = cid
            self.cid_var.set(cid)
        logged_in = bool(snap.get("spotify_logged_in", False))
        if self._login_busy:
            self._set(self.spotify_btn, text="Waiting for the browser", state="disabled")
            self._set(self.spotify_lbl, text="Finish signing in, in your browser.")
        else:
            self._set(self.spotify_btn, text="Sign out" if logged_in else "Sign in to Spotify",
                      state="normal")
            if self._seen.get("logged_in") != logged_in:
                self._seen["logged_in"] = logged_in
                self._seen.pop("spotify_note", None)
            note = self._seen.get("spotify_note")
            self._set(self.spotify_lbl, text=note or ("Signed in." if logged_in else "Not signed in."))

    # ------------------------------------------------------------------ actions: main view

    def toggle(self) -> None:
        self.core.set_enabled(not bool(self.core.snapshot().get("enabled", False)))

    def _look_moved(self, _value: str = "") -> None:
        self._set(self.look_value_lbl, text=f"{self._look_rounded()} ms")

    def _look_rounded(self) -> int:
        ms = int(round(float(self.look_var.get()) / 10.0)) * 10
        return max(settings.LOOKAHEAD_MIN_MS, min(settings.LOOKAHEAD_MAX_MS, ms))

    def commit_lookahead(self) -> None:
        """The slider was let go: this is the only moment the core hears about it."""
        self._dragging = False
        ms = self._look_rounded()
        self.look_var.set(ms)
        self._set(self.look_value_lbl, text=f"{ms} ms")
        if ms != self._seen.get("look"):
            self._seen["look"] = ms
            self.core.set_lookahead(ms)

    def how_to_fix(self) -> None:
        if self.readme_path.exists():
            try:
                self.opener(str(self.readme_path))
                return
            except Exception:
                pass
        messagebox.showinfo("N-Pass: virtual cable", TEXT_FIX_FALLBACK, parent=self.root)

    # ------------------------------------------------------------------ actions: settings

    def _fill_mics(self) -> None:
        try:
            names = list(self.core.inputs())
        except Exception:
            names = []
        self.mic_box.configure(values=names)

    def _mic_picked(self) -> None:
        name = self.mic_var.get()
        self._seen["mic"] = name
        self.mic_box.selection_clear()
        self.core.set_input(name)

    def commit_client_id(self) -> None:
        cid = self.cid_var.get().strip()
        if cid != self._seen.get("cid"):
            self._seen["cid"] = cid
            self.core.set_spotify_client_id(cid)

    def spotify_clicked(self) -> None:
        if self._login_busy:
            return
        if bool(self.core.snapshot().get("spotify_logged_in", False)):
            self.core.spotify_logout()
            return
        self.commit_client_id()
        if not self.cid_var.get().strip():
            self._seen["spotify_note"] = "Paste your Spotify client ID first."
            return
        self._login_busy = True
        self._seen.pop("spotify_note", None)

        def work() -> str:
            try:
                self.core.spotify_login()
                return ""
            except Exception:
                # The error text is not shown or logged: it could hold sign-in details.
                return "Sign-in did not finish. Try again."

        def done(problem: str) -> None:
            self._login_busy = False
            if problem:
                self._seen["spotify_note"] = problem

        self._in_background(work, done)

    def _editor_text(self) -> str:
        return self.words_text.get("1.0", "end-1c")

    def _put_editor(self, text: str, editable: bool) -> None:
        self.words_text.configure(state="normal")
        self.words_text.delete("1.0", "end")
        self.words_text.insert("1.0", text)
        self.words_text.edit_reset()
        self.words_text.configure(state="normal" if editable else "disabled",
                                  fg=theme.TEXT if editable else theme.DIM)

    def _render_words(self) -> None:
        """Fill the editor: the real words when "Show words" is ticked, dots otherwise."""
        if self._words_buffer is None:
            try:
                self._words_buffer = "\n".join(self.core.words())
            except Exception:
                self._words_buffer = ""
        show = bool(self.show_words_var.get())
        if show:
            self._put_editor(self._words_buffer, editable=True)
            self._set(self.words_save_btn, state="normal")
            self._set(self.words_lbl, text="")
        else:
            lines = [ln.strip() for ln in self._words_buffer.splitlines() if ln.strip()]
            self._put_editor("\n".join(DOT * len(ln) for ln in lines), editable=False)
            self._set(self.words_save_btn, state="disabled")
            count = len(lines)
            hidden = "No words yet." if count == 0 else f"{count} word{'s' if count != 1 else ''} hidden."
            self._set(self.words_lbl, text=f'{hidden} Tick "Show words" to edit.')

    def _show_words_changed(self) -> None:
        if not self.show_words_var.get():
            self._words_buffer = self._editor_text()   # keep unsaved typing while it is hidden
        self._render_words()

    def save_words(self) -> None:
        if not self.show_words_var.get():
            return
        words: list[str] = []
        for line in self._editor_text().splitlines():
            w = line.strip().lower()
            if w and w not in words:
                words.append(w)
        self.core.set_words(words)
        self._words_buffer = "\n".join(words)
        self._put_editor(self._words_buffer, editable=True)
        count = len(words)
        self._set(self.words_lbl, text=f"Saved {count} word{'s' if count != 1 else ''}.")

    def _set_risky(self) -> None:
        setter = getattr(self.core, "set_mute_risky_lines", None)
        if setter is not None:
            setter(bool(self.risky_var.get()))

    def open_catch_log(self) -> None:
        path = settings.catch_log_path()
        if not path.exists():
            self._set(self.tools_lbl, text="The catch log is empty. Nothing has been caught yet.")
            return
        # the log shows the blocked words in full: ask for a second click so it is not opened
        # by accident while the screen is being streamed
        now = time.monotonic()
        if now - getattr(self, "_log_asked_at", -100.0) > 15:
            self._log_asked_at = now
            self._set(self.tools_lbl, text="This file shows your blocked words in full. Do not "
                                           "open it while you are live. Click again to open it.")
            return
        self._log_asked_at = -100.0
        try:
            self.opener(str(path))
            self._set(self.tools_lbl, text="")
        except Exception:
            self._set(self.tools_lbl, text=f"Could not open the catch log. It is here: {path}")

    def open_obs_sync(self) -> None:
        from npass.ui.obs_dialog import ObsSyncDialog
        if self.obs_dialog is not None and self.obs_dialog.alive():
            self.obs_dialog.lift()
            return
        engine = self._last_snap.get("engine")
        delay = float(getattr(engine, "measured_delay_ms", 0.0)) or float(self._last_snap.get("lookahead_ms", 0))
        self.obs_dialog = ObsSyncDialog(self, delay_ms=delay)

    # ------------------------------------------------------------------ show / hide / quit

    def show(self) -> None:
        self.root.deiconify()
        self.root.lift()
        try:
            self.root.focus_force()
        except tk.TclError:
            pass

    def hide(self) -> None:
        """The X button: keep running in the tray. Without a tray icon, minimise instead, so the
        user can always get the window back."""
        if self.tray is None:
            self.root.iconify()
            return
        self.root.withdraw()
        if not self._told_tray:
            self._told_tray = True
            self.tray.notify("N-Pass is still running. Right-click this icon and pick Quit to close it.")

    def quit_app(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except tk.TclError:
                pass
        try:
            self.core.quit()
        except Exception:
            log.exception("core.quit failed")
        for hook in self.quit_hooks:
            try:
                hook()
            except Exception:
                pass
        if self.obs_dialog is not None:
            self.obs_dialog.close()
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def window_box(root: tk.Tk) -> tuple[int, int, int, int]:
    """Screen box of the whole window, title bar included."""
    import ctypes
    from ctypes import wintypes
    try:
        hwnd = int(root.wm_frame(), 16)
        rect = wintypes.RECT()
        # 9 = DWMWA_EXTENDED_FRAME_BOUNDS: the visible frame, without the invisible resize border
        if ctypes.windll.dwmapi.DwmGetWindowAttribute(wintypes.HWND(hwnd), 9, ctypes.byref(rect),
                                                      ctypes.sizeof(rect)) == 0:
            return rect.left, rect.top, rect.right, rect.bottom
    except Exception:
        pass
    x, y = root.winfo_rootx(), root.winfo_rooty()
    return x, y, x + root.winfo_width(), y + root.winfo_height()


def run_ui(core: Any, start_tab: str = "main", screenshot_to: str | None = None,
           use_tray: bool = True, use_hotkey: bool = True) -> None:
    """Build window + tray + hotkey and run the Tk loop until Quit.

    `screenshot_to`: demo use only. Saves a picture of the window after a moment, then quits.
    """
    theme.set_dpi_aware()
    set_app_id()                       # before the first window, or the taskbar shows Python's icon
    for attempt in range(3):           # Tk on Windows now and then fails its first start; try again
        try:
            root = tk.Tk()
            break
        except tk.TclError:
            if attempt == 2:
                raise
            time.sleep(0.2)
    try:
        root.iconbitmap(default=str(ensure_ico()))   # .ico only: adding iconphoto blanks the icon here
    except Exception:
        pass
    win = MainWindow(core, root)
    if start_tab == "settings":
        win.tabs.select(win.settings_tab)

    hotkey: HotkeyListener | None = None
    if use_hotkey:
        combo = str(core.snapshot().get("hotkey", "") or "")
        if combo:
            hotkey = HotkeyListener(combo, lambda: win.post(win.toggle))
            if not hotkey.start():
                win.local_warnings.append(hotkey.error)
            win.quit_hooks.append(hotkey.stop)

    tray: Tray | None = None
    if use_tray:
        tray = Tray(on_show=lambda: win.post(win.show), on_toggle=lambda: win.post(win.toggle),
                    on_quit=lambda: win.post(win.quit_app))
        if tray.start():
            win.tray = tray
        win.quit_hooks.append(tray.stop)

    if screenshot_to:
        def shoot() -> None:
            from PIL import ImageGrab
            root.attributes("-topmost", True)
            root.lift()
            root.update()
            time.sleep(0.3)
            root.update()
            out = Path(screenshot_to)
            out.parent.mkdir(parents=True, exist_ok=True)
            ImageGrab.grab(bbox=window_box(root)).save(out)
            win.quit_app()
        root.after(2500, shoot)

    win.start()
    try:
        root.mainloop()
    finally:
        if hotkey is not None:
            hotkey.stop()
        if tray is not None:
            tray.stop()
