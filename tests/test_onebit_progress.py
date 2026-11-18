"""Needle's tool-call answers, the progress file and the progress chart."""

import json
import os
import tempfile
import unittest

from kilix_bench.onebit import contract, progress
from kilix_bench.onebit.backends import (
    NeedleBackend, needle_input, parse_needle_calls)


class NeedleParseTest(unittest.TestCase):
    def test_one_answer_call(self):
        self.assertEqual(parse_needle_calls(
            [{"name": "answer", "arguments": {"value": 0}}]), 0)
        self.assertEqual(parse_needle_calls(
            [{"name": "answer", "arguments": {"value": 1}}]), 1)
        self.assertEqual(parse_needle_calls(
            [{"name": "answer", "arguments": '{"value": 1}'}]), 1)
        self.assertEqual(parse_needle_calls(
            [{"name": "answer", "arguments": {"value": "0"}}]), 0)

    def test_everything_else_is_a_format_failure(self):
        for calls in (
                [], None, "0",
                [{"name": "answer", "arguments": {"value": 2}}],
                [{"name": "answer", "arguments": {"value": True}}],
                [{"name": "answer", "arguments": {"value": 0.0}}],
                [{"name": "answer", "arguments": {}}],
                [{"name": "answer", "arguments": {"value": 0, "why": "x"}}],
                [{"name": "reply", "arguments": {"value": 0}}],
                [{"name": "answer", "arguments": {"value": 0}}] * 2):
            with self.subTest(calls=calls):
                self.assertIsNone(parse_needle_calls(calls))

    def test_input_is_one_line(self):
        text, reason = needle_input("<information>\nexit 1\n</information>"
                                    "\n\nDid it fail?")
        self.assertIsNone(reason)
        self.assertEqual(
            text, "<information> / exit 1 / </information> /  / Did it fail?")
        self.assertEqual(needle_input("a \\ b"),
                         (None, "skipped_unrepresentable"))

    def test_backend_with_a_fake_engine(self):
        sent = []

        def transport(path, body):
            sent.append((path, body))
            if path == "/reset":
                return {}
            return {"function_calls": [
                {"name": "answer", "arguments": {"value": 1}}]}

        backend = NeedleBackend("", max_bytes=80, transport=transport)
        item = {"context": "ERROR disk full", "prompt": "Verdict on this job.",
                "answer": 1}
        answer = backend.ask(contract.messages(item), "raw")
        self.assertEqual(answer["parsed"], 1)
        self.assertEqual(sent[0], ("/reset", b"{}"))
        self.assertEqual(sent[1][1].decode(), json.dumps(
            {"input": "<information> / ERROR disk full / </information> /  "
                      "/ Verdict on this job."}, separators=(",", ":")))
        long_item = {"prompt": "x" * 81, "answer": 0}
        skipped = backend.ask(contract.messages(long_item), "raw")
        self.assertEqual(skipped["skipped"], "skipped_too_long")
        self.assertEqual(skipped["input_bytes"], 81)
        self.assertFalse(backend.supports("constrained")[0])


def result(accuracy_raw, logs_raw, constrained=None):
    """A result file reduced to what a progress line reads."""
    def arm(accuracy, logs):
        return {"supported": True, "summary": {
            "overall": {"n": 400, "skipped": 0, "accuracy": accuracy,
                        "accuracy_all_items": accuracy,
                        "balanced_accuracy": accuracy, "mcc": 0.1,
                        "format_failure_rate": 0.02, "zero_share": 0.55},
            "grade": {"grade_level": 3},
            "by_band": {"computer-use": {"accuracy": logs}},
            "by_band_domain": {"computer-use/logs": {"accuracy": logs}}}}
    arms = {"raw": arm(accuracy_raw, logs_raw)}
    arms["constrained"] = arm(*constrained) if constrained else {
        "supported": False, "reason": "none"}
    return {"model": {"label": "m"}, "items": {"split": "dev",
                                               "set_sha256": "abc"},
            "contract": {"version": "onebit-contract/v1"}, "arms": arms}


class ProgressTest(unittest.TestCase):
    def test_line_carries_the_scores(self):
        line = progress.progress_line(result(0.625, 0.75, (0.7, 0.8)),
                                      "qwen", "r1", 400, when="T")
        self.assertEqual(line["arm"], "qwen")
        self.assertEqual(line["step"], 400)
        self.assertEqual(line["split"], "dev")
        self.assertEqual(line["decodes"]["raw"]["accuracy"], 0.625)
        self.assertEqual(line["decodes"]["raw"]["grade_level"], 3)
        self.assertEqual(line["decodes"]["constrained"]["by_band"],
                         {"computer-use": 0.8})
        needle = progress.progress_line(result(0.5, 0.5), "needle", "r1", 0)
        self.assertEqual(needle["decodes"]["constrained"],
                         {"supported": False})

    def test_append_and_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = os.path.join(temporary, "progress.jsonl")
            progress.append(path, progress.progress_line(
                result(0.5, 0.5), "qwen", "r1", 0, when="T"))
            progress.append(path, progress.progress_line(
                result(0.75, 0.9), "qwen", "r1", 100, when="T"))
            lines = progress.read(path)
        self.assertEqual([line["step"] for line in lines], [0, 100])
        self.assertEqual(progress.series(lines, "raw", "overall"),
                         {("qwen", "r1"): [(0, 0.5), (100, 0.75)]})
        self.assertEqual(progress.series(lines, "raw", "computer-use/logs"),
                         {("qwen", "r1"): [(0, 0.5), (100, 0.9)]})

    def test_chart_coordinates(self):
        lines = [
            progress.progress_line(result(0.5, 0.25), "qwen", "r1", 0, "T"),
            progress.progress_line(result(0.75, 1.0), "qwen", "r1", 100, "T"),
            progress.progress_line(result(0.5, 0.5), "needle", "r1", 100, "T"),
        ]
        # x: step 0 -> 50, step 100 -> 330; y: 1.0 -> 30, 0.5 -> 130, 0 -> 230
        self.assertEqual(progress.scale(0, 0.5, 0, 100), (50.0, 130.0))
        self.assertEqual(progress.scale(100, 0.75, 0, 100), (330.0, 80.0))
        self.assertEqual(progress.scale(100, 1.0, 0, 100, panel=1),
                         (710.0, 30.0))
        svg = progress.chart_svg(lines, decode="raw", group="computer-use")
        self.assertIn('points="50.0,130.0 330.0,80.0"', svg)      # qwen overall
        self.assertIn('points="430.0,180.0 710.0,30.0"', svg)     # qwen logs
        self.assertIn('points="330.0,130.0"', svg)                # needle
        self.assertIn("chance 50%", svg)
        self.assertIn("prefers-color-scheme: dark", svg)
        # Needle takes the first colour slot, Qwen the second, whatever order.
        self.assertIn('class="s1" fill="none" stroke-width="2" '
                      'points="50.0,130.0', svg)
        table = progress.chart_table(lines, decode="raw",
                                     group="computer-use")
        self.assertIn("qwen       r1               100    75.0%"
                      "           100.0%", table)


if __name__ == "__main__":
    unittest.main()
