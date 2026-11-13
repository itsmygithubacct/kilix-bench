"""The text suite matches vtebench's cell stream and stays free of local paths."""

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from kilix_bench.report import render_text, write_run
from kilix_bench.text import (
    ascii_rows,
    light_cells_alphabet,
    light_cells_rewrite,
    screen_grid,
    sixel_encode,
)


class TextSuiteTest(unittest.TestCase):
    def test_grid_snaps_down_to_whole_cells(self):
        columns, rows, width, height = screen_grid(100, 40)
        self.assertEqual((columns, rows, width, height), (12, 2, 96, 32))

    def test_a_short_frame_is_refused(self):
        with self.assertRaises(SystemExit):
            screen_grid(7, 16)

    def test_light_cells_matches_the_vtebench_pipeline(self):
        got = light_cells_rewrite(5, 2, "Q")
        raw = subprocess.check_output(
            ["bash", "-c", "printf '\\e[H%*s' 10 | tr ' ' Q"])
        self.assertEqual(got, raw)
        self.assertEqual(got, b"\x1b[H" + b"Q" * 10)

    def test_the_alphabet_is_twenty_six_rewrites(self):
        body = light_cells_alphabet(3, 2)
        one = light_cells_rewrite(3, 2, "A")
        self.assertEqual(len(body), 26 * len(one))
        self.assertTrue(body.startswith(one))
        self.assertIn(light_cells_rewrite(3, 2, "Z"), body)

    def test_ascii_rows_cycle_the_printable_set(self):
        rows = ascii_rows(3, 2)
        self.assertEqual(rows[0], bytes((32, 33, 34)))
        self.assertEqual(rows[1], bytes((35, 36, 37)))

    def test_sixel_of_one_ink_pixel_sets_the_top_bit(self):
        encoded = sixel_encode(bytes((0xE9, 0xE6, 0xDF)), 1, 1)
        text = encoded.decode("ascii")
        self.assertTrue(text.startswith("\x1bPq"))
        self.assertTrue(text.endswith("\x1b\\"))
        self.assertIn("#1@", text)
        self.assertIn('"1;1;1;1', text)

    def test_sixel_run_length_of_a_paper_span(self):
        paper = bytes((0x11, 0x15, 0x1C)) * 8
        encoded = sixel_encode(paper, 8, 1)
        self.assertIn(b"#0!8@", encoded)
        self.assertIn(b"#1!8?", encoded)

    def test_results_do_not_record_a_library_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            os.environ["KILIX_BENCH_HOME"] = temporary
            try:
                payload = {
                    "tool_version": "0.2.3",
                    "subject": "kilix-text",
                    "subject_version": "0.2.3",
                    "subject_revision": "abc",
                    "python": "3.13.0",
                    "machine": "x86_64",
                    "size": [80, 16],
                    "note": "One sample is one full-grid rewrite.",
                    "scenarios": [{
                        "name": "kitty_text",
                        "what": "one rewrite",
                        "unit": "ms",
                        "samples": 1,
                        "min_ms": 0.1,
                        "median_ms": 0.1,
                        "p95_ms": 0.1,
                        "max_ms": 0.1,
                        "extra": {
                            "kind": "cells",
                            "cells": 10,
                            "wire_bytes": 13,
                            "full_bytes": 3840,
                            "us_per_cell": 10.0,
                        },
                    }],
                }
                stamped, latest, text = write_run(
                    payload, forbidden=("/projects/neuronte",))
                self.assertIn("-kilix-text-", stamped.name)
                self.assertEqual(stamped.suffix, ".json")
                self.assertEqual(latest.name, "kilix-text-latest.json")
                mode = stat.S_IMODE(latest.stat().st_mode)
                self.assertEqual(mode & 0o077, 0)
                rendered = text.read_text(encoding="utf-8")
                self.assertEqual(rendered, render_text(payload))
                self.assertIn("10 cells  wire 13 bytes", rendered)
                self.assertIn("10.0000 us/cell", rendered)
                self.assertNotIn("projects", rendered)
            finally:
                os.environ.pop("KILIX_BENCH_HOME", None)


if __name__ == "__main__":
    unittest.main()
