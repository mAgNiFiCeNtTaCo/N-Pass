"""Colours, fonts and the dark ttk style shared by the window and its dialogs."""
from __future__ import annotations

import ctypes
import tkinter as tk
from tkinter import ttk

BG = "#15171c"
PANEL = "#1e222a"
FIELD = "#262b35"
LINE = "#343b48"
TEXT = "#e7e9ee"
DIM = "#929aa8"
GREEN = "#2ecc71"
YELLOW = "#f1c40f"
RED = "#ff6b5b"
RED_BOX = "#3a1d1d"
GREY = "#5d6572"
BLUE = "#4c8dff"

FONT = "Segoe UI"


def set_dpi_aware() -> None:
    """Ask Windows for sharp text on scaled screens. Must run before the first window exists."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def scale_of(root: tk.Misc) -> float:
    """1.0 on a normal screen, 1.5 at 150 % scaling, and so on."""
    try:
        return max(1.0, float(root.winfo_fpixels("1i")) / 96.0)
    except Exception:
        return 1.0


def apply_style(root: tk.Misc) -> ttk.Style:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    base = (FONT, 10)
    style.configure(".", background=BG, foreground=TEXT, font=base, bordercolor=LINE,
                    lightcolor=LINE, darkcolor=LINE, troughcolor=FIELD, focuscolor=BLUE)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=TEXT)
    style.configure("Dim.TLabel", foreground=DIM)
    style.configure("Head.TLabel", font=(FONT, 10, "bold"))
    style.configure("Count.TLabel", font=(FONT, 15, "bold"))
    style.configure("TButton", background=FIELD, foreground=TEXT, padding=(10, 4), relief="flat")
    style.map("TButton", background=[("disabled", PANEL), ("active", LINE)],
              foreground=[("disabled", GREY)])
    for kind in ("TCheckbutton", "TRadiobutton"):
        style.configure(kind, background=BG, foreground=TEXT, indicatorbackground=FIELD,
                        indicatorforeground="#ffffff", upperbordercolor=GREY, lowerbordercolor=GREY,
                        indicatormargin=(1, 1, 6, 1))
        style.map(kind, background=[("active", BG)], foreground=[("disabled", GREY)],
                  indicatorbackground=[("selected", BLUE), ("disabled", PANEL)],
                  upperbordercolor=[("selected", BLUE)], lowerbordercolor=[("selected", BLUE)])
    style.configure("TEntry", fieldbackground=FIELD, foreground=TEXT, insertcolor=TEXT)
    style.configure("TCombobox", fieldbackground=FIELD, background=FIELD, foreground=TEXT,
                    arrowcolor=TEXT, selectbackground=FIELD, selectforeground=TEXT)
    style.map("TCombobox", fieldbackground=[("readonly", FIELD)], foreground=[("readonly", TEXT)],
              selectbackground=[("readonly", FIELD)], selectforeground=[("readonly", TEXT)])
    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(8, 6, 8, 0))
    style.configure("TNotebook.Tab", background=BG, foreground=DIM, padding=(16, 5),
                    font=(FONT, 10, "bold"))
    style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", TEXT)])
    style.configure("Horizontal.TScale", background="#b9c1cf", troughcolor=FIELD,
                    bordercolor=FIELD, lightcolor="#b9c1cf", darkcolor="#b9c1cf")
    style.map("Horizontal.TScale", background=[("active", "#ffffff")])
    style.configure("TSeparator", background=LINE)
    # the drop-down list of a combobox is a plain Tk listbox
    root.option_add("*TCombobox*Listbox.background", FIELD)
    root.option_add("*TCombobox*Listbox.foreground", TEXT)
    root.option_add("*TCombobox*Listbox.selectBackground", BLUE)
    root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    root.option_add("*TCombobox*Listbox.font", base)
    return style
