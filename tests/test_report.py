"""Result records stay private and do not carry local paths."""

import json
import os
import stat
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from kilix_bench.report import (
    assert_no_local_paths,
    render_text,
    summarize,
    scenario,
    write_run,
)


class ReportTest(unittest.TestCase):
    def test_raw_samples_are_retained_and_bad_measurements_refused(self):
        self.assertEqual(scenario("a", "b", [0.000123456])["raw_seconds"], [0.000123456])
        for value in [float("nan"), float("inf"), -1]:
            with self.assertRaises(ValueError):
                summarize([value])

    def test_two_runs_in_the_same_second_keep_both_records(self):
        payload = {"tool_version": "0.2.3", "subject": "collision", "subject_version": "1",
                   "size": [1, 1], "python": "test", "machine": "test", "scenarios": []}
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"KILIX_BENCH_HOME": temporary}):
            with patch("kilix_bench.report.datetime") as clock:
                clock.now.return_value.strftime.return_value = "20260101T000000Z"
                first, _, _ = write_run(payload)
                second, _, _ = write_run(payload)
            self.assertNotEqual(first, second)
            self.assertEqual(json.loads(first.read_text()), json.loads(second.read_text()))

    def test_median_of_three(self):
        stats = summarize([0.001, 0.003, 0.002])
        self.assertEqual(stats["samples"], 3)
        self.assertEqual(stats["median_ms"], 2.0)
        self.assertEqual(stats["min_ms"], 1.0)
        self.assertEqual(stats["max_ms"], 3.0)

    def test_local_paths_are_refused(self):
        root = "/" + "home" + "/someone"
        with self.assertRaises(SystemExit):
            assert_no_local_paths({"note": root + "/src"}, (root,))

    def test_write_is_private_and_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            os.environ["KILIX_BENCH_HOME"] = temporary
            try:
                payload = {
                    "tool_version": "0.2.3",
                    "subject": "kilix-95",
                    "subject_version": "0.2.3",
                    "subject_revision": "abc",
                    "python": "3.13.0",
                    "machine": "x86_64",
                    "size": [1920, 1080],
                    "scenarios": [{
                        "name": "compose_idle",
                        "what": "redraw",
                        "unit": "ms",
                        "samples": 1,
                        "min_ms": 10.0,
                        "median_ms": 10.0,
                        "p95_ms": 10.0,
                        "max_ms": 10.0,
                        "extra": {"compose_fps": 100.0},
                    }],
                }
                stamped, latest, text = write_run(payload)
                self.assertTrue(stamped.is_file())
                mode = stat.S_IMODE(Path(temporary).stat().st_mode)
                self.assertEqual(mode & 0o077, 0)
                saved = json.loads(latest.read_text(encoding="utf-8"))
                self.assertEqual(saved["subject"], "kilix-95")
                rendered = text.read_text(encoding="utf-8")
                self.assertIn("compose_idle", rendered)
                self.assertIn("compose only", rendered)
                self.assertEqual(rendered, render_text(payload))
            finally:
                os.environ.pop("KILIX_BENCH_HOME", None)


if __name__ == "__main__":
    unittest.main()
