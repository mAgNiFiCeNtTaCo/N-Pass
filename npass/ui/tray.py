"""The tray icon: a coloured dot that shows the filter's state, with Show / Turn on-off / Quit.

green = on and healthy, grey = off, yellow = a warning, red = recognition down or cable missing.
The icon runs in pystray's own thread. Menu clicks arrive on that thread, so every callback given
to `Tray` must be safe to call from another thread (the window hands them its queue).
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageDraw

from npass.types import Health

APP_ID = "NPass.Filter"
ASSETS = Path(__file__).resolve().parent / "assets"
ICO_PATH = ASSETS / "npass.ico"

COLOURS = {
    "green": (46, 204, 113),
    "grey": (125, 133, 144),
    "yellow": (241, 196, 15),
    "red": (231, 76, 60),
}
TIPS = {
    "green": "N-Pass: on",
    "grey": "N-Pass: off",
    "yellow": "N-Pass: on, with a warning",
    "red": "N-Pass: needs attention",
}


def set_app_id() -> None:
    """Give the process its own taskbar identity. Call before the first window is made."""
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass


def tray_state(snap: dict, extra_warnings: bool = False) -> str:
    """Which colour the dot should be for this snapshot."""
    if not snap.get("enabled", False):
        return "grey"
    health = snap.get("health", Health.OK)
    if health == Health.DOWN or not snap.get("cable_found", True):
        return "red"
    engine = snap.get("engine")
    warned = bool(snap.get("warnings")) or bool(getattr(engine, "warnings", None)) or extra_warnings
    if health in (Health.LATE, Health.STARTING) or warned:
        return "yellow"
    return "green"


def make_image(state: str, size: int = 64) -> Image.Image:
    """A round dot in the state's colour on a dark disc, drawn large and scaled down so it is smooth."""
    colour = COLOURS.get(state, COLOURS["grey"])
    big = size * 4
    img = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((big * 0.03, big * 0.03, big * 0.97, big * 0.97), fill=(24, 27, 33, 255))
    d.ellipse((big * 0.20, big * 0.20, big * 0.80, big * 0.80), fill=colour + (255,))
    return img.resize((size, size), Image.LANCZOS)


def ensure_ico(path: Path = ICO_PATH) -> Path:
    """Write the window icon (.ico, several sizes) if it is not there yet. Returns its path."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        make_image("green", 256).save(
            path, format="ICO", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (256, 256)])
    return path


class Tray:
    """Wraps a pystray icon. `start()` returns False when the tray cannot be used on this PC."""

    def __init__(self, on_show: Callable[[], None], on_toggle: Callable[[], None],
                 on_quit: Callable[[], None]):
        self._on_show = on_show
        self._on_toggle = on_toggle
        self._on_quit = on_quit
        self._icon: Any = None
        self._state = "grey"
        self._enabled = False
        self._images = {name: make_image(name) for name in COLOURS}

    def start(self) -> bool:
        try:
            import pystray
            menu = pystray.Menu(
                pystray.MenuItem("Show N-Pass", lambda icon, item: self._on_show(), default=True),
                pystray.MenuItem(lambda item: "Turn off" if self._enabled else "Turn on",
                                 lambda icon, item: self._on_toggle()),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", lambda icon, item: self._on_quit()),
            )
            self._icon = pystray.Icon("npass", self._images[self._state], TIPS[self._state], menu)
            self._icon.run_detached()
            return True
        except Exception:
            self._icon = None
            return False

    def update(self, state: str, enabled: bool) -> None:
        """Called from the window's refresh. Touches the real icon only when something changed."""
        if state not in COLOURS:
            state = "grey"
        changed = state != self._state or enabled != self._enabled
        self._state, self._enabled = state, enabled
        if self._icon is None or not changed:
            return
        try:
            self._icon.icon = self._images[state]
            self._icon.title = TIPS[state]
            self._icon.update_menu()
        except Exception:
            pass

    def notify(self, message: str) -> None:
        """A small Windows notification from the tray icon. Silently skipped if it cannot be shown."""
        if self._icon is None:
            return
        try:
            self._icon.notify(message, "N-Pass")
        except Exception:
            pass

    def stop(self) -> None:
        icon, self._icon = self._icon, None
        if icon is not None:
            try:
                icon.stop()
            except Exception:
                pass
