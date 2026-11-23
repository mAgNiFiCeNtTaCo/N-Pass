"""The N-Pass window, tray icon and hotkey. The window talks only to a `Core` (docs/CONTRACT.md)."""
from __future__ import annotations

from typing import Any


def run_ui(core: Any, **kwargs: Any) -> None:
    """Build window + tray + hotkey and run until the user picks Quit. See npass.ui.window.run_ui."""
    from npass.ui.window import run_ui as _run_ui
    _run_ui(core, **kwargs)


__all__ = ["run_ui"]
