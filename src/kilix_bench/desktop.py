"""Measure one Kilix 95 checkout without opening the live desktop.

The desk is constructed with no terminal, so compose never talks to Kitty.
Damage math is the desktop's own frame_transmit, and the wire comparison is
the Kilix frame presenter.
"""

import argparse
import base64
import json
import os
import platform
import re
import subprocess
import sys
import time


def _parse():
    parser = argparse.ArgumentParser(prog="kilix-bench-desktop")
    parser.add_argument("--desktop", required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.width < 320 or args.height < 240:
        raise SystemExit("the desk must be at least 320x240")
    if args.repeats < 1 or args.warmup < 0:
        raise SystemExit("repeats and warmup must be non-negative")
    return args


def _time(operation, repeats, warmup):
    for _ in range(warmup):
        operation()
    samples = []
    extra = None
    for _ in range(repeats):
        started = time.perf_counter()
        extra = operation()
        samples.append(time.perf_counter() - started)
    return samples, extra or {}


def _revision(desktop):
    try:
        result = subprocess.run(
            ["git", "-C", desktop, "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _version(desktop):
    path = os.path.join(desktop, "VERSION")
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _seed(directory, count):
    os.makedirs(directory, mode=0o700, exist_ok=True)
    for number in range(count):
        path = os.path.join(directory, f"note-{number:02d}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("note\n")


def _rect_bytes(rects):
    total = 0
    for rect in rects:
        total += int(rect[2]) * int(rect[3]) * 3
    return total


def _transmit_extra(kind, found, width, height, rects):
    """Bytes the desktop would send, plus the presenter's own rectangles."""
    full = width * height * 3
    rows = 0
    if kind == "unchanged":
        pixel = 0
    elif kind == "full":
        pixel = full
        rows = height
    elif kind == "rects":
        pixel = _rect_bytes(found or ())
    elif kind == "move":
        pixel = _rect_bytes(found[2] if found else ())
    else:
        rows = 0 if not found else int(found[1])
        pixel = width * rows * 3
    return {
        "kind": kind,
        "rows": rows,
        "pixel_bytes": pixel,
        "full_bytes": full,
        "rect_count": len(rects),
        "rect_bytes": _rect_bytes(rects),
    }


def measure(args):
    desktop = os.path.abspath(args.desktop)
    sys.path.insert(0, desktop)
    started = time.perf_counter()
    import host as kilix_host
    kilix_host.add_kilix_config_path()
    import apps
    import main as desk_main
    import widgets as W
    from kilix_sdk import graphics as kilix_graphics
    import_seconds = time.perf_counter() - started

    from kilix_bench.report import scenario

    width, height = args.width, args.height
    _seed(os.environ["KILIX_DESKTOP_DIR"], 48)

    def construct():
        return desk_main.Desk(term=None, size=(width, height), draw_cursor=False)

    construct_samples, desk = _time(construct, min(args.repeats, 6), 1)
    icon_count = len(desk.shell.grid.items)

    def compose():
        desk.dirty = True
        desk.render()

    compose_samples, _ = _time(compose, args.repeats, args.warmup)
    idle_bytes = desk.fb.tobytes()

    apps.open(desk, "notepad")
    apps.open(desk, "calc")
    apps.open(desk, "settings")
    window_count = len(desk.wm.windows)
    app_samples, _ = _time(compose, args.repeats, args.warmup)

    desk.draw_cursor = True
    cursor_samples, _ = _time(compose, args.repeats, args.warmup)
    desk.draw_cursor = False
    compose()

    notepad = next(
        win for win in desk.wm.windows if type(win).__name__ == "Notepad")
    desk.wm.activate(notepad)

    def hover():
        desk.dispatch_mouse(W.Ev(
            kind="mouse", x=notepad.x + 48, y=notepad.y + 8, move=True))

    hover_samples, _ = _time(hover, max(args.repeats, 40), args.warmup)

    def tobytes():
        return desk.fb.tobytes()

    byte_samples, frame = _time(tobytes, args.repeats, args.warmup)

    def unchanged():
        kind, found = desk_main.frame_transmit(frame, frame, width, height)
        return _transmit_extra(kind, found, width, height, ())

    same_samples, same_extra = _time(unchanged, args.repeats, args.warmup)

    clock = bytearray(frame)
    stride = width * 3
    # The taskbar is the bottom strip. Touch one pixel on its top row.
    row = height - 20
    clock[row * stride] ^= 0xFF
    clock_bytes = bytes(clock)
    clock_rects = kilix_graphics.diff_rects(frame, clock_bytes, width, height)

    def transmit_clock():
        kind, found = desk_main.frame_transmit(
            frame, clock_bytes, width, height)
        return _transmit_extra(kind, found, width, height, clock_rects)

    clock_samples, clock_extra = _time(transmit_clock, args.repeats, args.warmup)

    desk.render()
    before_rect = (notepad.x, notepad.y, notepad.w, notepad.h)
    before_move = desk.fb.tobytes()
    notepad.x += 12
    desk.add_damage(*before_rect)
    desk.add_damage(notepad.x, notepad.y, notepad.w, notepad.h)
    desk.render()
    after_rect = (notepad.x, notepad.y, notepad.w, notepad.h)
    after_move = desk.fb.tobytes()
    move = (before_rect, after_rect)
    move_rects = kilix_graphics.diff_rects(
        before_move, after_move, width, height)

    def transmit_move():
        kind, found = desk_main.frame_plan(
            before_move, after_move, width, height, move=move)
        return _transmit_extra(kind, found, width, height, move_rects)

    move_samples, move_extra = _time(transmit_move, args.repeats, args.warmup)

    def drag():
        rect = (notepad.x, notepad.y, notepad.w, notepad.h)
        notepad.x += 4
        desk.add_damage(*rect)
        desk.add_damage(notepad.x, notepad.y, notepad.w, notepad.h)
        desk.render()

    drag_samples, _ = _time(drag, args.repeats, args.warmup)
    desk.draw_cursor = True
    compose()

    def pointer():
        old = desk.mouse_pos
        desk.mouse_pos = (old[0] + 5, old[1] + 3)
        desk.add_damage(*desk._cursor_box(*old))
        desk.add_damage(*desk._cursor_box(*desk.mouse_pos))
        desk.render()

    pointer_samples, _ = _time(pointer, args.repeats, args.warmup)

    class _Term:
        """Read and unlink shared-memory frames so the presenter can reuse slots."""

        _shm = re.compile(r"\x1b_G([^;]*\bt=s\b[^;]*);([A-Za-z0-9+/=]+)\x1b\\")

        def __init__(self):
            self.written = 0

        def write(self, data):
            self.written += len(data)
            for match in self._shm.finditer(data):
                name = base64.b64decode(match.group(2)).decode("ascii")
                path = "/dev/shm/" + name.lstrip("/")
                os.unlink(path)

    term = _Term()
    presenter = kilix_graphics.FramePresenter(term, image_id=7, max_fps=0)
    columns = max(1, width // 8)
    rows = max(1, height // 16)
    present_samples = []
    present_extra = {}
    try:
        passes = 1 + min(args.repeats, 8)
        for index in range(passes):
            presenter.present(frame, width, height, columns, rows)
            presenter.flush()
            started = time.perf_counter()
            result = presenter.present(
                clock_bytes, width, height, columns, rows)
            presenter.flush()
            elapsed = time.perf_counter() - started
            if index == 0:
                continue
            present_samples.append(elapsed)
            present_extra = {
                "kind": result.kind,
                "pixel_bytes": result.pixel_bytes,
                "wire_bytes": result.wire_bytes,
                "full_bytes": width * height * 3,
            }
    finally:
        presenter.close(discard=True)

    def with_fps(samples, what, extra=None):
        row_extra = dict(extra or {})
        median = sorted(samples)[len(samples) // 2]
        if median > 0:
            row_extra["compose_fps"] = round(1 / median, 1)
        return scenario(what[0], what[1], samples, row_extra)

    scenarios = [
        scenario("import_desktop",
                 "Import the desktop and the Kilix graphics SDK",
                 [import_seconds], {"icons": icon_count}),
        scenario("construct_desk",
                 "Build an offscreen desk, its icons, and its taskbar",
                 construct_samples, {"icons": icon_count}),
        with_fps(compose_samples, (
            "compose_idle",
            "Redraw the idle desk after the wallpaper is cached")),
        with_fps(app_samples, (
            "compose_apps",
            "Redraw the desk with Notepad, Calculator, and Settings open"),
            {"windows": window_count}),
        with_fps(cursor_samples, (
            "compose_cursor",
            "Redraw the desk and paint the software cursor")),
        scenario("hover_title",
                 "Hit-test one move across a window title",
                 hover_samples),
        with_fps(drag_samples, (
            "compose_drag",
            "Repaint the strips a window move exposes")),
        with_fps(pointer_samples, (
            "compose_pointer",
            "Repaint the software cursor after a short move")),
        scenario("frame_tobytes",
                 "Copy the framebuffer to RGB bytes",
                 byte_samples, {"bytes": width * height * 3}),
        scenario("transmit_unchanged",
                 "Ask the desktop send decision about an identical frame",
                 same_samples, same_extra),
        scenario("transmit_clock",
                 "Ask the desktop and the presenter what a taskbar change sends",
                 clock_samples, clock_extra),
        scenario("transmit_move",
                 "Ask the desktop and the presenter what a 12px window move sends",
                 move_samples, move_extra),
        scenario("presenter_clock",
                 "FramePresenter encodes the same taskbar change",
                 present_samples, present_extra),
    ]
    return {
        "tool": "kilix-bench",
        "tool_version": "0.2.3",
        "subject": "kilix-95",
        "subject_version": _version(desktop),
        "subject_revision": _revision(desktop),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "size": [width, height],
        "repeats": args.repeats,
        "warmup": args.warmup,
        "scenarios": scenarios,
    }


def main():
    payload = measure(_parse())
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
