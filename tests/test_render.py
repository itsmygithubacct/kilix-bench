"""The render suite names a window move without recording a local path."""

import os
import stat
import tempfile
import unittest
from pathlib import Path

from kilix_bench.render import (
    _present,
    moved_window,
    one_pixel,
    scrolled_frame,
    window_move,
)
from kilix_bench.report import render_text, write_run


def _graphics():
    home = os.environ.get("KILIX_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "gpu_terminal", "sources", "kilix")
    config = os.path.join(home, "config")
    if not os.path.isdir(os.path.join(config, "kilix_sdk")):
        return None
    previous = os.environ.get("KILIX_HOME")
    os.environ["KILIX_HOME"] = home
    try:
        from kilix_bench.render import _load_graphics
        graphics, _elapsed = _load_graphics()
    finally:
        if previous is None:
            os.environ.pop("KILIX_HOME", None)
        else:
            os.environ["KILIX_HOME"] = previous
    return graphics


class RenderSuiteTest(unittest.TestCase):
    def test_a_sideways_move_exposes_two_strips(self):
        rect, strips = window_move(40, 20, dx=4)
        x, y, width, height = rect
        self.assertEqual(strips, (
            (x, y, 4, height),
            (x + width, y, 4, height)))
        union = (width + 4) * height
        self.assertGreaterEqual(union, 40 * 20 // 5)

    def test_render_results_do_not_replace_the_desktop_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            os.environ["KILIX_BENCH_HOME"] = temporary
            try:
                payload = {
                    "tool_version": "0.2.3",
                    "subject": "kilix-render",
                    "subject_version": "0.2.3",
                    "subject_revision": "abc",
                    "python": "3.13.0",
                    "machine": "x86_64",
                    "size": [64, 32],
                    "note": "Times are the graphics library.",
                    "scenarios": [{
                        "name": "present_scroll",
                        "what": "vertical shift",
                        "unit": "ms",
                        "samples": 1,
                        "min_ms": 1.0,
                        "median_ms": 1.0,
                        "p95_ms": 1.0,
                        "max_ms": 1.0,
                        "extra": {
                            "kind": "scroll",
                            "pixel_bytes": 24,
                            "full_bytes": 192,
                            "rect_count": 1,
                            "wire_bytes": 40,
                        },
                    }],
                }
                stamped, latest, text = write_run(payload)
                self.assertTrue(str(stamped).endswith("-kilix-render.json"))
                self.assertEqual(latest.name, "kilix-render-latest.json")
                self.assertFalse((Path(temporary) / "results"
                                  / "kilix-95-latest.json").exists())
                mode = stat.S_IMODE(latest.stat().st_mode)
                self.assertEqual(mode & 0o077, 0)
                rendered = text.read_text(encoding="utf-8")
                self.assertIn("graphics library", rendered)
                self.assertEqual(rendered, render_text(payload))
                self.assertNotIn("PIL redraw", rendered)
                self.assertIn("presenter rects 1", rendered)
                self.assertNotIn("presenter rects 1 (0 bytes)", rendered)
                counted = dict(payload)
                counted["scenarios"] = [dict(payload["scenarios"][0])]
                counted["scenarios"][0]["extra"] = dict(
                    payload["scenarios"][0]["extra"], rect_bytes=24)
                self.assertIn(
                    "presenter rects 1 (24 bytes)", render_text(counted))
            finally:
                os.environ.pop("KILIX_BENCH_HOME", None)

    def test_presenter_kinds_match_the_render_probes(self):
        graphics = _graphics()
        if graphics is None:
            self.skipTest("graphics SDK is not available")
        width, height = 64, 32
        before, after = one_pixel(width, height)
        _samples, pixel = _present(
            graphics, before, after, width, height, repeats=1, warmup=0)
        self.assertEqual(pixel["kind"], "rect")
        self.assertEqual(pixel["pixel_bytes"], 3)

        before, after, strips = moved_window(width, height, dx=4)
        _samples, boxed = _present(
            graphics, before, after, width, height, repeats=1, warmup=0,
            enable_scroll=True)
        self.assertEqual(boxed["kind"], "rect")
        self.assertGreater(boxed["pixel_bytes"], 4 * 2 * 3)
        strip_bytes = sum(rect[2] * rect[3] * 3 for rect in strips)

        _samples, hinted = _present(
            graphics, before, after, width, height, repeats=1, warmup=0,
            enable_scroll=True, damage=strips)
        self.assertEqual(hinted["kind"], "rect")
        self.assertLessEqual(hinted["pixel_bytes"], boxed["pixel_bytes"])
        self.assertLessEqual(boxed["pixel_bytes"], strip_bytes * 8)
        self.assertEqual(hinted["pixel_bytes"], strip_bytes)

        dy = 2
        before, after = scrolled_frame(width, height, dy)
        _samples, scroll = _present(
            graphics, before, after, width, height, repeats=1, warmup=0,
            enable_scroll=True)
        self.assertEqual(scroll["kind"], "scroll")
        self.assertEqual(scroll["pixel_bytes"], width * dy * 3)

        _samples, streamed = _present(
            graphics, before, after, width, height, repeats=1, warmup=0,
            stream=True, force_full=True)
        self.assertEqual(streamed["kind"], "full")
        self.assertEqual(streamed["pixel_bytes"], width * height * 3)
        self.assertGreater(streamed["wire_bytes"], 16)
        self.assertLess(streamed["wire_bytes"], streamed["pixel_bytes"])


if __name__ == "__main__":
    unittest.main()
