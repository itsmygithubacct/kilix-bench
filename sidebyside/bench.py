#!/usr/bin/env python3
"""Time the three faces drawing terminal cells: Neuronte 2, Neuronte 0.1.4, DejaVu.

usage: bench.py [--repeats N] [--cells 8x16,12x24,16x32] [--json FILE]

For each face and cell size, over the characters all three have:

  open     opening the face (the library and bundle, or the font files)
  first    one character's cell the first time it is asked for, a fresh
           face each repeat: the net runs, or FreeType rasterises
  again    the same cell asked for again from the same face: Neuronte
           copies it from the handle's cache; FreeType draws it again
           (Pillow keeps no cell)
  screen   an 80 x 24 screen composed from cells this program has kept,
           which is what a pane does after the first time

Times include the Python call into each library, the same for all three.
The median of the repeats is reported; run it on an idle machine.
"""
import json
import os
import statistics
import sys
import time

import charsets
import faces
import pane


def _median_us(samples, per):
    return statistics.median(samples) * 1e6 / per


def measure(which, chars, sizes, repeats):
    out = {}
    opens = []
    for _ in range(repeats):
        t = time.perf_counter()
        f = faces.face(which)
        if which == "dejavu":
            f.maps()
        opens.append(time.perf_counter() - t)
    out["name"] = f.name
    out["open_ms"] = statistics.median(opens) * 1e3
    scalars = [ord(c) for c in chars]
    for w, h in sizes:
        first, again = [], []
        for _ in range(repeats):
            f = faces.face(which)
            if which == "dejavu":
                f.maps()
                f._fonts(w, h)
            t = time.perf_counter()
            for s in scalars:
                f.cell(s, w, h)
            first.append(time.perf_counter() - t)
            t = time.perf_counter()
            for s in scalars:
                f.cell(s, w, h)
            again.append(time.perf_counter() - t)
        cache, screen = {}, []
        rows = [[chars[(r * 80 + c) % len(chars)] for c in range(80)] for r in range(24)]
        pane.draw(f, cache, rows, None, 80, 24, w, h)
        for _ in range(repeats * 4):
            t = time.perf_counter()
            pane.draw(f, cache, rows, None, 80, 24, w, h)
            screen.append(time.perf_counter() - t)
        out[f"{w}x{h}"] = {"first_us_per_cell": _median_us(first, len(scalars)),
                           "again_us_per_cell": _median_us(again, len(scalars)),
                           "screen_ms": statistics.median(screen) * 1e3}
    return out


def report(result):
    lines = [f"{result['characters']} characters all three have; median of {result['repeats']} repeats", ""]
    for which in faces.ORDER:
        r = result["faces"][which]
        lines.append(f"{r['name']}    open {r['open_ms']:.2f} ms")
    for size in result["cells"]:
        lines += ["", f"{size} cells {'':14s}{'first, us/cell':>16s} {'again, us/cell':>16s} {'80x24 screen, ms':>18s}"]
        for which in faces.ORDER:
            r = result["faces"][which]
            m = r[size]
            lines.append(f"  {r['name'].split('  ')[0][:24]:24s}{m['first_us_per_cell']:16.1f} "
                         f"{m['again_us_per_cell']:16.2f} {m['screen_ms']:18.2f}")
    return "\n".join(lines) + "\n"


def main(argv):
    def opt(name, default):
        return argv[argv.index(name) + 1] if name in argv else default
    repeats = int(opt("--repeats", "7"))
    sizes = [tuple(int(v) for v in s.split("x")) for s in opt("--cells", "8x16,12x24,16x32").split(",")]
    try:
        common = sorted(s for s in charsets.sets()["common"] if charsets.shown(s))
        chars = [chr(s) for s in common]
        result = {"characters": len(chars), "repeats": repeats, "cells": [f"{w}x{h}" for w, h in sizes],
                  "threads": os.environ.get("OPENBLAS_NUM_THREADS", ""),
                  "faces": {w: measure(w, chars, sizes, repeats) for w in faces.ORDER}}
    except faces.Missing as e:
        sys.stderr.write(f"bench: {e}\n")
        return 1
    sys.stdout.write(report(result))
    if "--json" in argv:
        with open(opt("--json", ""), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
