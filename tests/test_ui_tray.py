import sys
from pathlib import Path

# plain `uv run pytest` does not put the project root on the path (no conftest, package = false)
for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from PIL import Image

from npass.types import Health
from npass.ui import tray
from npass.ui.fake_core import FakeCore


def snap(**changes):
    core = FakeCore()
    for k, v in changes.items():
        setattr(core, k, v)
    return core.snapshot()


def test_state_colours():
    assert tray.tray_state(snap()) == "green"
    assert tray.tray_state(snap(enabled=False)) == "grey"
    assert tray.tray_state(snap(enabled=False, health=Health.DOWN, cable_found=False)) == "grey"
    assert tray.tray_state(snap(health=Health.LATE)) == "yellow"
    assert tray.tray_state(snap(warnings=["something"])) == "yellow"
    assert tray.tray_state(snap(engine_warnings=["mic unplugged"])) == "yellow"
    assert tray.tray_state(snap(), extra_warnings=True) == "yellow"
    assert tray.tray_state(snap(health=Health.DOWN)) == "red"
    assert tray.tray_state(snap(cable_found=False)) == "red"


def test_images_are_dots_in_the_right_colour():
    for name, rgb in tray.COLOURS.items():
        img = tray.make_image(name, 64)
        assert img.size == (64, 64) and img.mode == "RGBA"
        assert img.getpixel((32, 32))[:3] == rgb
        assert img.getpixel((0, 0))[3] == 0          # clear corners


def test_ico_is_written_once(tmp_path):
    path = tmp_path / "assets" / "npass.ico"
    assert tray.ensure_ico(path) == path and path.exists()
    stamp = path.stat().st_mtime_ns
    tray.ensure_ico(path)
    assert path.stat().st_mtime_ns == stamp
    with Image.open(path) as ico:
        assert ico.format == "ICO"
        assert (16, 16) in ico.info["sizes"] and (256, 256) in ico.info["sizes"]


def test_update_without_a_running_icon_is_safe():
    t = tray.Tray(lambda: None, lambda: None, lambda: None)
    t.update("red", True)
    t.update("nonsense", False)
    t.notify("hello")
    t.stop()


def test_update_touches_the_icon_only_on_change():
    class StubIcon:
        def __init__(self):
            self.sets = 0
            self.menus = 0
            self.stopped = False

        def __setattr__(self, key, value):
            if key == "icon":
                object.__setattr__(self, "sets", self.sets + 1)
            object.__setattr__(self, key, value)

        def update_menu(self):
            self.menus += 1

        def stop(self):
            self.stopped = True

    t = tray.Tray(lambda: None, lambda: None, lambda: None)
    icon = StubIcon()
    t._icon = icon
    t.update("green", True)
    t.update("green", True)
    t.update("green", True)
    assert icon.sets == 1 and icon.title == "N-Pass: on"
    t.update("red", True)
    assert icon.sets == 2 and icon.title == "N-Pass: needs attention"
    t.stop()
    assert icon.stopped
