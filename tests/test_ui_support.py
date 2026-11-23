"""Shared helper for the window tests (no tests in here).

Tk on Windows now and then fails to start a second or third interpreter in one process ("Can't find
a usable init.tcl"). So the tests share ONE hidden root for the whole run and build each window in
its own Toplevel, which is destroyed after the test.
"""
import sys
from pathlib import Path

# plain `uv run pytest` does not put the project root on the path (no conftest, package = false)
for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gc
import tkinter as tk

import pytest

_root = None


def shared_root() -> tk.Tk:
    global _root
    if _root is None:
        last = None
        for _ in range(5):
            try:
                _root = tk.Tk()
                break
            except tk.TclError as e:
                last = e
        else:
            pytest.skip(f"no display for tkinter: {last}")
        _root.withdraw()
    return _root


def new_top() -> tk.Toplevel:
    top = tk.Toplevel(shared_root())
    top.withdraw()
    return top


def drop(top) -> None:
    try:
        top.destroy()
    except tk.TclError:
        pass
    gc.collect()   # let Tk variables go on this thread, not later on some worker thread
