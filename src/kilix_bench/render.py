"""Measure how Kilix sends a frame.

The samples are the graphics SDK and FramePresenter. They do not start a
desktop, and they do not include the GPU. A vertical shift can be copied
inside the existing image. A window that slides sideways stays a rectangle
unless the caller names the strips that changed.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

from kilix_bench.report import scenario
from kilix_bench.terminal import ConsumerTerm


def _parse():
    parser = argparse.ArgumentParser(prog="kilix-bench-render")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.width < 64 or args.height < 32:
        raise SystemExit("the frame must be at least 64x32")
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


def window_move(width, height, dx=12):
    """A sideways block and the two strips the move exposes.

    The block is large enough that its bounding rectangle is at least a
    fifth of the frame, which is the presenter's threshold for trying a
    scroll. The scroll detector only accepts vertical motion, so this move
    stays a rectangle.
    """
    if dx < 1:
        raise ValueError("window move needs a positive step")
    block_w = max(dx + 1, min(width - dx - 2, int(width * 0.7)))
    block_h = max(8, min(height - 4, int(height * 0.45)))
    x, y = 1, 1
    if x + block_w + dx > width or y + block_h > height:
        raise ValueError("frame is too small for the window-move probe")
    strips = ((x, y, dx, block_h), (x + block_w, y, dx, block_h))
    return (x, y, block_w, block_h), strips


def _paint(buf, width, x, y, w, h, color):
    pixel = bytes((color[0] & 255, color[1] & 255, color[2] & 255))
    stride = width * 3
    row = pixel * w
    view = memoryview(buf)
    for yy in range(y, y + h):
        start = yy * stride + x * 3
        view[start:start + w * 3] = row


def _blank(width, height, color):
    pixel = bytes((color[0] & 255, color[1] & 255, color[2] & 255))
    return bytearray(pixel * (width * height))


def one_pixel(width, height):
    base = bytes(width * height * 3)
    changed = bytearray(base)
    changed[0] = 1
    return base, bytes(changed)


def scrolled_frame(width, height, dy):
    """Shift every row down by ``dy`` and fill the exposed band."""
    if dy < 1 or dy >= height:
        raise ValueError("scroll does not fit in the frame")
    source = bytearray(width * height * 3)
    for y in range(height):
        _paint(source, width, 0, y, width, 1,
               (y & 255, (y >> 8) & 255, 40))
    out = bytearray(len(source))
    stride = width * 3
    out[dy * stride:] = source[: (height - dy) * stride]
    _paint(out, width, 0, 0, width, dy, (255, 0, 0))
    return bytes(source), bytes(out)


def moved_window(width, height, dx=12):
    rect, strips = window_move(width, height, dx)
    x, y, block_w, block_h = rect
    base = _blank(width, height, (8, 16, 24))
    moved = _blank(width, height, (8, 16, 24))
    _paint(base, width, x, y, block_w, block_h, (20, 90, 160))
    _paint(moved, width, x + dx, y, block_w, block_h, (20, 90, 160))
    return bytes(base), bytes(moved), strips


def _rect_bytes(rects):
    total = 0
    for rect in rects:
        total += int(rect[2]) * int(rect[3]) * 3
    return total


def _load_graphics():
    home = os.environ.get("KILIX_HOME", "")
    config = os.path.join(home, "config")
    if not home or not os.path.isdir(config):
        raise SystemExit("KILIX_HOME must contain config/kilix_sdk")
    sys.path.insert(0, config)
    started = time.perf_counter()
    from kilix_sdk import graphics
    return graphics, time.perf_counter() - started


def _revision(home):
    try:
        result = subprocess.run(
            ["git", "-C", home, "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout.strip()


def _version(home):
    path = os.path.join(home, "VERSION")
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _present(graphics, base, changed, width, height, repeats, warmup,
             stream=False, enable_scroll=False, force_full=False, damage=None):
    """Time the present that turns ``base`` into ``changed``.

    Each sample first restores ``base``, untimed, so the timed call is the
    same transition every repeat. ``max_fps`` is zero, so a frame is not
    queued. The stand-in terminal releases shared-memory names immediately.
    """
    term = ConsumerTerm()
    # A streamed presenter otherwise spends its first seconds on keyframes.
    # The suite turns that off so a sample is the edit itself. force_full
    # still asks for a placement.
    presenter = graphics.FramePresenter(
        term, image_id=11, stream=stream, max_fps=0,
        enable_scroll=enable_scroll,
        stream_warmup_seconds=0, stream_keyframe_seconds=0)
    columns = max(1, width // 8)
    rows = max(1, height // 16)
    samples = []
    extra = {}
    kwargs = {}
    if force_full:
        kwargs["force_full"] = True
    if damage is not None:
        kwargs["damage"] = damage
    try:
        passes = warmup + repeats
        for index in range(passes):
            presenter.present(base, width, height, columns, rows)
            presenter.flush()
            started = time.perf_counter()
            result = presenter.present(
                changed, width, height, columns, rows, **kwargs)
            presenter.flush()
            elapsed = time.perf_counter() - started
            if index < warmup:
                continue
            samples.append(elapsed)
            extra = {
                "kind": result.kind,
                "pixel_bytes": result.pixel_bytes,
                "wire_bytes": result.wire_bytes,
                "full_bytes": width * height * 3,
                "rect_count": len(result.rects),
            }
    finally:
        presenter.close(discard=True)
    return samples, extra


def measure(args):
    graphics, load_seconds = _load_graphics()
    width, height = args.width, args.height
    full = width * height * 3
    pixel_before, pixel_after = one_pixel(width, height)
    window_before, window_after, strips = moved_window(width, height)
    dy = max(2, height // 45)
    scroll_before, scroll_after = scrolled_frame(width, height, dy)

    def diff_pixel():
        rects = graphics.diff_rects(pixel_before, pixel_after, width, height)
        return {
            "diff_rects": len(rects),
            "diff_bytes": _rect_bytes(rects),
            "full_bytes": full,
        }

    def diff_window():
        rects = graphics.diff_rects(
            window_before, window_after, width, height)
        return {
            "diff_rects": len(rects),
            "diff_bytes": _rect_bytes(rects),
            "full_bytes": full,
        }

    def diff_hint():
        rects = graphics.diff_damage_rects(
            window_before, window_after, width, height, strips)
        return {
            "diff_rects": len(rects),
            "diff_bytes": _rect_bytes(rects),
            "full_bytes": full,
        }

    pixel_diff, pixel_diff_extra = _time(diff_pixel, args.repeats, args.warmup)
    window_diff, window_diff_extra = _time(
        diff_window, args.repeats, args.warmup)
    hint_diff, hint_diff_extra = _time(diff_hint, args.repeats, args.warmup)

    place, place_extra = _present(
        graphics, pixel_before, pixel_after, width, height,
        args.repeats, args.warmup, force_full=True)
    same, same_extra = _present(
        graphics, pixel_before, pixel_before, width, height,
        args.repeats, args.warmup)
    edited, edited_extra = _present(
        graphics, pixel_before, pixel_after, width, height,
        args.repeats, args.warmup)
    boxed, boxed_extra = _present(
        graphics, window_before, window_after, width, height,
        args.repeats, args.warmup, enable_scroll=True)
    hinted, hinted_extra = _present(
        graphics, window_before, window_after, width, height,
        args.repeats, args.warmup, enable_scroll=True, damage=strips)
    scroll, scroll_extra = _present(
        graphics, scroll_before, scroll_after, width, height,
        args.repeats, args.warmup, enable_scroll=True)
    stream_pixel, stream_pixel_extra = _present(
        graphics, pixel_before, pixel_after, width, height,
        args.repeats, args.warmup, stream=True)
    stream_hint, stream_hint_extra = _present(
        graphics, window_before, window_after, width, height,
        args.repeats, args.warmup, stream=True, damage=strips)
    stream_place, stream_place_extra = _present(
        graphics, pixel_before, pixel_after, width, height,
        args.repeats, args.warmup, stream=True, force_full=True)

    source = (1, 1, 8, 8)
    dest = (5, 1, 8, 8)

    def compose():
        command = graphics.build_compose(11, source, dest)
        return {
            "kind": "compose",
            "pixel_bytes": 0,
            "wire_bytes": len(command),
            "full_bytes": full,
        }

    compose_samples, compose_extra = _time(compose, args.repeats, args.warmup)
    home = os.environ.get("KILIX_HOME", "")
    return {
        "tool": "kilix-bench",
        "tool_version": "0.2.3",
        "subject": "kilix-render",
        "subject_version": _version(home),
        "subject_revision": _revision(home),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "size": [width, height],
        "repeats": args.repeats,
        "warmup": args.warmup,
        "note": (
            "Times are the graphics library and FramePresenter. "
            "They do not include the GPU. "
            "An inline wire size is the zlib-compressed escape."),
        "scenarios": [
            scenario("load_graphics",
                     "Import the Kilix graphics SDK",
                     [load_seconds]),
            scenario("diff_pixel",
                     "diff_rects on a one-pixel change",
                     pixel_diff, pixel_diff_extra),
            scenario("diff_window",
                     "diff_rects on a sideways window move",
                     window_diff, window_diff_extra),
            scenario("diff_hint",
                     "diff_damage_rects limited to the exposed strips",
                     hint_diff, hint_diff_extra),
            scenario("present_full",
                     "Shared-memory full placement of a frame",
                     place, place_extra),
            scenario("present_same",
                     "FramePresenter sees an identical frame",
                     same, same_extra),
            scenario("present_pixel",
                     "Shared-memory edit of one pixel",
                     edited, edited_extra),
            scenario("present_window",
                     "Sideways window move with scroll detection on",
                     boxed, boxed_extra),
            scenario("present_hint",
                     "The same move limited to the exposed strips",
                     hinted, hinted_extra),
            scenario("present_scroll",
                     "Vertical shift copied inside the frame",
                     scroll, scroll_extra),
            scenario("stream_pixel",
                     "Inline edit of one pixel",
                     stream_pixel, stream_pixel_extra),
            scenario("stream_hint",
                     "Inline upload of the exposed strips",
                     stream_hint, stream_hint_extra),
            scenario("stream_full",
                     "Inline full placement of a frame",
                     stream_place, stream_place_extra),
            scenario("compose_overlap",
                     "Build an overlapping compose command",
                     compose_samples, compose_extra),
        ],
    }


def main():
    payload = measure(_parse())
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
