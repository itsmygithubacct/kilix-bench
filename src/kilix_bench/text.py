"""Compare three ways to put one screen of text on a terminal.

The workload is one full-grid rewrite. ``kitty_text`` is the stream from
vtebench's light_cells benchmark: cursor home, then the same character in
every cell. The terminal's glyph rasterization and GPU drawing are outside
the measurement. ``bitmap_*`` is Kilix's bitmap font
drawn by soft-raster into a pixel buffer. ``neuronte_*`` is the same grid
drawn by Neuronte into the same kind of buffer. The Kitty graphics
samples are FramePresenter placements of those buffers. ``sixel`` is the
other pixel protocol, a two-colour encoding of the bitmap frame.

The frame is snapped down to whole 8 by 16 cells, the bitmap font's cell.
The grid is the bench frame, not the live pane.
"""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import struct
import subprocess
import sys
import time

from kilix_bench.render import _load_graphics, _present
from kilix_bench.report import scenario, summarize


CELL_W = 8
CELL_H = 16
# Zell's dark paper and light ink, opaque. soft-raster takes the ink
# without the alpha byte and stores the alpha itself.
BG = 0xFF11151C
FG = 0x00E9E6DF
INK = 0xFF000000 | FG
FG_RGB = (0xE9, 0xE6, 0xDF)
BG_RGB = (0x11, 0x15, 0x1C)
ASCII_SPAN = 95  # U+0020..U+007E


def _parse():
    parser = argparse.ArgumentParser(prog="kilix-bench-text")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        raise SystemExit("repeats and warmup must be non-negative")
    screen_grid(args.width, args.height)
    return args


def screen_grid(width, height):
    """Pixel size snapped down to whole terminal cells."""
    if width < CELL_W or height < CELL_H:
        raise SystemExit("the text frame must cover at least one 8x16 cell")
    columns = width // CELL_W
    rows = height // CELL_H
    return columns, rows, columns * CELL_W, rows * CELL_H


def light_cells_rewrite(columns, rows, char):
    """One iteration of vtebench light_cells.

    The benchmark runs ``printf '\\e[H%*s' $((columns * lines)) | tr ' ' C``
    for each letter C. The width argument prints that many spaces, and
    ``tr`` turns each space into the letter. There is no newline.
    """
    if len(char) != 1 or ord(char) < 32 or ord(char) > 126:
        raise ValueError("light_cells wants one printable ASCII character")
    return b"\x1b[H" + char.encode("ascii") * (columns * rows)


def light_cells_alphabet(columns, rows):
    """The whole light_cells payload: A through Z, each a full rewrite."""
    return b"".join(
        light_cells_rewrite(columns, rows, chr(code))
        for code in range(ord("A"), ord("Z") + 1))


def repeat_rows(columns, rows, char):
    line = char.encode("ascii") * columns
    return tuple(line for _ in range(rows))


def ascii_rows(columns, rows):
    """Printable ASCII, row-major, cycling U+0020..U+007E."""
    rows_out = []
    for row in range(rows):
        base = row * columns
        rows_out.append(bytes(
            32 + (base + column) % ASCII_SPAN for column in range(columns)))
    return tuple(rows_out)


def sixel_encode(rgb, width, height, fg=FG_RGB, bg=BG_RGB):
    """Two-colour sixel of an RGB frame.

    A pixel is ink when it equals ``fg``. Every other pixel is paper.
    Bits past the last row of a short band stay clear. Run length uses
    the sixel ``!`` repeat when it is shorter than spelling the byte.
    """
    if len(rgb) != width * height * 3:
        raise ValueError("RGB frame length does not match the size")
    fg_bytes = bytes(fg)
    parts = [
        "\x1bPq",
        _sixel_color(0, bg),
        _sixel_color(1, fg),
        f'"1;1;{width};{height}',
    ]
    for top in range(0, height, 6):
        ink = []
        paper = []
        for x in range(width):
            ink_bits = 0
            paper_bits = 0
            for dy in range(6):
                y = top + dy
                if y >= height:
                    continue
                at = (y * width + x) * 3
                if rgb[at:at + 3] == fg_bytes:
                    ink_bits |= 1 << dy
                else:
                    paper_bits |= 1 << dy
            ink.append(ink_bits)
            paper.append(paper_bits)
        parts.append("#0" + _sixel_runs(paper) + "$")
        parts.append("#1" + _sixel_runs(ink) + "-")
    parts.append("\x1b\\")
    return "".join(parts).encode("ascii")


def _sixel_color(index, rgb):
    red, green, blue = rgb
    return "#{index};2;{red};{green};{blue}".format(
        index=index,
        red=red * 100 // 255,
        green=green * 100 // 255,
        blue=blue * 100 // 255)


def _sixel_runs(masks):
    out = []
    index = 0
    count = len(masks)
    while index < count:
        end = index + 1
        while end < count and masks[end] == masks[index]:
            end += 1
        run = end - index
        char = chr(63 + masks[index])
        if run >= 3:
            out.append(f"!{run}{char}")
        else:
            out.append(char * run)
        index = end
    return "".join(out)


def libraries():
    """Absolute paths of the two libraries and the weight file.

    The caller passes these into the measurement process. They are not
    written into the result.
    """
    home = os.environ.get("NEURONTE_HOME") or os.environ.get("NEURONTE", "~/projects/neuronte")
    return {
        "NEURONTE_LIBRARY": _pick("NEURONTE_LIBRARY", (
            os.path.join(
                home,
                "build", "libneuronte.so"),
        )),
        "NEURONTE_WEIGHTS": _pick("NEURONTE_WEIGHTS", (
            os.path.join(
                home,
                "weights", "neuronte-split-v5.nrt"),
        )),
        "SOFT_RASTER_LIBRARY": _pick("SOFT_RASTER_LIBRARY", (
            "~/gpu_terminal/kilix-modules/soft-raster/build/libsoft-raster.so",
        )),
    }


def _pick(name, defaults):
    raw = os.environ.get(name, "")
    candidates = (raw,) if raw else defaults
    for item in candidates:
        path = os.path.abspath(os.path.expanduser(item))
        if os.path.isfile(path):
            return path
    raise SystemExit(f"could not find {name}")


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


def _per_cell(samples, cells, **extra):
    median_ms = summarize(samples)["median_ms"]
    extra["cells"] = cells
    extra["us_per_cell"] = round(median_ms * 1000.0 / cells, 4)
    return extra


class _Canvas(ctypes.Structure):
    _fields_ = [
        ("px", ctypes.POINTER(ctypes.c_uint32)),
        ("w", ctypes.c_int),
        ("h", ctypes.c_int),
        ("clip_x0", ctypes.c_int),
        ("clip_y0", ctypes.c_int),
        ("clip_x1", ctypes.c_int),
        ("clip_y1", ctypes.c_int),
        ("owns_px", ctypes.c_bool),
    ]


def _soft_raster():
    path = os.environ.get("SOFT_RASTER_LIBRARY", "")
    if not path:
        path = libraries()["SOFT_RASTER_LIBRARY"]
    lib = ctypes.CDLL(path)
    lib.sr_canvas_wrap.argtypes = [
        ctypes.POINTER(_Canvas), ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int, ctypes.c_int]
    lib.sr_text.argtypes = [
        ctypes.POINTER(_Canvas), ctypes.c_float, ctypes.c_float,
        ctypes.c_char_p, ctypes.c_uint32, ctypes.c_float, ctypes.c_int]
    lib.sr_pack_rgb.argtypes = [
        ctypes.POINTER(_Canvas), ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t]
    lib.sr_pack_rgb.restype = ctypes.c_bool
    return lib


def _neuronte():
    path = os.environ.get("NEURONTE_LIBRARY", "")
    weights = os.environ.get("NEURONTE_WEIGHTS", "")
    if not path or not weights:
        found = libraries()
        path = path or found["NEURONTE_LIBRARY"]
        weights = weights or found["NEURONTE_WEIGHTS"]
    lib = ctypes.CDLL(path)
    lib.neuronte_open_file.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.neuronte_open_file.restype = ctypes.c_int
    lib.neuronte_close.argtypes = [ctypes.c_void_p]
    lib.neuronte_parameters.argtypes = [ctypes.c_void_p]
    lib.neuronte_parameters.restype = ctypes.c_size_t
    lib.neuronte_nets.argtypes = [ctypes.c_void_p]
    lib.neuronte_nets.restype = ctypes.c_int
    lib.neuronte_draw_utf8.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_char_p, ctypes.c_size_t, ctypes.c_uint32]
    lib.neuronte_draw_utf8.restype = ctypes.c_int
    lib.neuronte_coverage.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    lib.neuronte_coverage.restype = ctypes.c_int
    handle = ctypes.c_void_p()
    rc = lib.neuronte_open_file(weights.encode(), ctypes.byref(handle))
    if rc != 0 or not handle.value:
        raise SystemExit(f"neuronte weights failed to open ({rc})")
    return lib, handle, weights


def _solid(width, height, color):
    return struct.pack("<I", color) * (width * height)


def _blank(pixels, solid):
    ctypes.memmove(pixels, solid, len(solid))


def _pack(lib, canvas, width, height):
    out = (ctypes.c_uint8 * (width * height * 3))()
    if not lib.sr_pack_rgb(ctypes.byref(canvas), out, len(out)):
        raise SystemExit("sr_pack_rgb failed")
    return bytes(out)


def _draw_bitmap(lib, canvas, rows, width):
    y = 0.0
    step = float(CELL_H)
    for line in rows:
        lib.sr_text(ctypes.byref(canvas), 0.0, y, line, FG, 1.0, 1)
        y += step
    return width


def _draw_neuronte(lib, handle, pixels, rows, width, height):
    view = ctypes.cast(pixels, ctypes.POINTER(ctypes.c_uint32))
    y = 0
    for line in rows:
        rc = lib.neuronte_draw_utf8(
            handle, view, width, width, height, 0, y, CELL_W, CELL_H,
            line, len(line), INK)
        if rc < 0:
            raise SystemExit(f"neuronte draw failed ({rc})")
        y += CELL_H
    return rc


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


def measure(args):
    columns, rows, width, height = screen_grid(args.width, args.height)
    cells = columns * rows
    full = width * height * 3
    graphics, _load = _load_graphics()
    raster = _soft_raster()
    neuronte, handle, weights = _neuronte()
    try:
        return _measure(
            args, graphics, raster, neuronte, handle, weights,
            columns, rows, width, height, cells, full)
    finally:
        neuronte.neuronte_close(handle)


def _measure(args, graphics, raster, neuronte, handle, weights,
             columns, rows, width, height, cells, full):
    pixels = (ctypes.c_uint32 * (width * height))()
    canvas = _Canvas()
    raster.sr_canvas_wrap(
        ctypes.byref(canvas),
        ctypes.cast(pixels, ctypes.POINTER(ctypes.c_uint32)),
        width, height)
    paper = _solid(width, height, BG)
    letter = repeat_rows(columns, rows, "A")
    varied = ascii_rows(columns, rows)
    home = os.environ.get("KILIX_HOME", "")

    def text_rewrite():
        payload = light_cells_rewrite(columns, rows, "A")
        return {"payload": payload}

    def text_alphabet():
        payload = light_cells_alphabet(columns, rows)
        return {"payload": payload}

    def bitmap(lines):
        def draw():
            _blank(pixels, paper)
            _draw_bitmap(raster, canvas, lines, width)
        return draw

    def net(lines):
        def draw():
            _blank(pixels, paper)
            _draw_neuronte(neuronte, handle, pixels, lines, width, height)
        return draw

    text_samples, text_extra = _time(text_rewrite, args.repeats, args.warmup)
    alphabet_samples, alphabet_extra = _time(
        text_alphabet, args.repeats, args.warmup)
    bitmap_samples, _ = _time(bitmap(letter), args.repeats, args.warmup)
    net(letter)()
    neuronte_samples, _ = _time(net(letter), args.repeats, args.warmup)
    bitmap_ascii_samples, _ = _time(bitmap(varied), args.repeats, args.warmup)
    # One untimed pass loads the ASCII cells into the draw cache, so the
    # timed samples are the steady screen even when warmup is zero.
    net(varied)()
    neuronte_ascii_samples, _ = _time(net(varied), args.repeats, args.warmup)

    coverage = (ctypes.c_float * (CELL_W * CELL_H))()

    def cold():
        for scalar in range(32, 127):
            rc = neuronte.neuronte_coverage(
                handle, scalar, CELL_W, CELL_H, 0, coverage)
            if rc != 0:
                raise SystemExit(f"neuronte coverage failed ({rc})")
        return {}

    cold_samples, _ = _time(cold, args.repeats, args.warmup)

    _blank(pixels, paper)
    _draw_bitmap(raster, canvas, varied, width)
    bitmap_rgb = _pack(raster, canvas, width, height)
    _blank(pixels, paper)
    _draw_neuronte(neuronte, handle, pixels, varied, width, height)
    neuronte_rgb = _pack(raster, canvas, width, height)
    _blank(pixels, paper)
    _draw_bitmap(raster, canvas, letter, width)
    letter_rgb = _pack(raster, canvas, width, height)
    empty = bytes(full)

    shm_samples, shm_extra = _present(
        graphics, empty, bitmap_rgb, width, height,
        args.repeats, args.warmup, force_full=True)
    inline_samples, inline_extra = _present(
        graphics, empty, bitmap_rgb, width, height,
        args.repeats, args.warmup, stream=True, force_full=True)
    neuronte_inline_samples, neuronte_inline_extra = _present(
        graphics, empty, neuronte_rgb, width, height,
        args.repeats, args.warmup, stream=True, force_full=True)

    def sixel():
        encoded = sixel_encode(letter_rgb, width, height)
        return {"encoded": encoded}

    sixel_samples, sixel_extra = _time(sixel, args.repeats, args.warmup)

    text_wire = len(text_extra["payload"])
    alphabet_wire = len(alphabet_extra["payload"])
    sixel_wire = len(sixel_extra["encoded"])
    with open(weights, "rb") as handle_file:
        digest = hashlib.sha256(handle_file.read()).hexdigest()
    return {
        "tool": "kilix-bench",
        "tool_version": "0.2.4",
        "subject": "kilix-text",
        "subject_version": _version(home),
        "subject_revision": _revision(home),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "size": [width, height],
        "repeats": args.repeats,
        "warmup": args.warmup,
        "font": os.path.basename(weights),
        "weights_sha256": digest,
        "parameters": neuronte.neuronte_parameters(handle),
        "nets": neuronte.neuronte_nets(handle),
        "note": (
            "One sample is one full-grid rewrite at 8x16. "
            "kitty_text is vtebench light_cells, one letter; "
            "kitty_text_alphabet is that benchmark's A to Z payload. "
            "Kitty's own font raster is not in the time. "
            "bitmap is soft-raster's 8x16 font. "
            "neuronte_repeat and neuronte_ascii are steady-state draws, "
            "after the cell cache is warm. neuronte_cold is coverage of "
            "the 95 ASCII characters with no cache, after weights are loaded; "
            "it is not process or filesystem cold start. "
            "Graphics times are FramePresenter and do not include the GPU. "
            "sixel is a two-colour encoding of the bitmap letter frame. "
            "Its time is this encoder; its wire size is the comparison."),
        "scenarios": [
            scenario(
                "kitty_text",
                "vtebench light_cells, one full-grid rewrite",
                text_samples,
                _per_cell(
                    text_samples, cells, kind="cells",
                    wire_bytes=text_wire, full_bytes=full)),
            scenario(
                "kitty_text_alphabet",
                "vtebench light_cells, the A to Z payload",
                alphabet_samples,
                _per_cell(
                    alphabet_samples, cells * 26, kind="cells",
                    wire_bytes=alphabet_wire, full_bytes=full)),
            scenario(
                "bitmap_repeat",
                "soft-raster 8x16, one character in every cell",
                bitmap_samples,
                _per_cell(bitmap_samples, cells, pixel_bytes=full)),
            scenario(
                "neuronte_repeat",
                "Neuronte, one character in every cell, cache warm",
                neuronte_samples,
                _per_cell(neuronte_samples, cells, pixel_bytes=full)),
            scenario(
                "bitmap_ascii",
                "soft-raster 8x16, printable ASCII cycling across the grid",
                bitmap_ascii_samples,
                _per_cell(bitmap_ascii_samples, cells, pixel_bytes=full)),
            scenario(
                "neuronte_ascii",
                "Neuronte, the same ASCII grid, cache warm",
                neuronte_ascii_samples,
                _per_cell(neuronte_ascii_samples, cells, pixel_bytes=full)),
            scenario(
                "neuronte_cold",
                "Neuronte coverage of the 95 ASCII characters, no cache",
                cold_samples,
                _per_cell(cold_samples, ASCII_SPAN, pixel_bytes=CELL_W * CELL_H * 4)),
            scenario(
                "kitty_shm",
                "Shared-memory placement of the bitmap ASCII frame",
                shm_samples, dict(shm_extra, cells=cells)),
            scenario(
                "kitty_inline",
                "Inline placement of the bitmap ASCII frame",
                inline_samples, dict(inline_extra, cells=cells)),
            scenario(
                "neuronte_inline",
                "Inline placement of the Neuronte ASCII frame",
                neuronte_inline_samples,
                dict(neuronte_inline_extra, cells=cells)),
            scenario(
                "sixel",
                "Two-colour sixel of the bitmap letter frame",
                sixel_samples,
                _per_cell(
                    sixel_samples, cells, kind="sixel",
                    pixel_bytes=full, full_bytes=full, wire_bytes=sixel_wire)),
        ],
    }


def main():
    payload = measure(_parse())
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
