"""Presentation probes classify consumers without inventing display latency."""

import json
import os
import shutil
import tempfile
import unittest

from kilix_bench.presentation import build_probe, framebuffer_source, run_fixture, run_live, kitty_binary
from kilix_bench.stages import apply_probe, catalog


def assert_hardware_unavailable(test, stages):
    for stage_id in ("gpu-upload", "gpu-execution", "sdf-atlas", "compositor-present", "scanout"):
        row = next(item for item in stages if item["id"] == stage_id)
        test.assertEqual(row["status"], "unavailable", stage_id)


class StageTests(unittest.TestCase):
    def test_catalog_names_measured_and_unavailable_work(self):
        stages = catalog(True)
        self.assertEqual(next(item for item in stages if item["id"] == "cpu-glyph-raster")["status"], "measured")
        self.assertEqual(next(item for item in stages if item["id"] == "mask-cache-compose")["status"], "not-run")
        drawn = catalog(True, True)
        self.assertEqual(next(item for item in drawn if item["id"] == "mask-cache-compose")["status"], "measured")
        self.assertEqual(next(item for item in stages if item["id"] == "application-protocol")["status"], "separate-suite")
        assert_hardware_unavailable(self, stages)
        idle = catalog(False)
        self.assertEqual(next(item for item in idle if item["id"] == "cpu-glyph-raster")["status"], "not-run")

    def test_fixture_peer_cannot_report_consumption_or_rank_quality(self):
        probe = {
            "peer": "protocol-fixture", "graphics_query": "answered",
            "shm_consumed": "not-a-consumer", "shm_consumed_ns": None,
            "display_latency": "unavailable", "scanout": "unavailable",
            "gpu_latency": "unavailable",
        }
        stages = apply_probe(catalog(False), probe)
        query = next(item for item in stages if item["id"] == "consumer-graphics-query")
        shared = next(item for item in stages if item["id"] == "consumer-shm-read")
        self.assertEqual(query["status"], "fixture-only")
        self.assertEqual(shared["status"], "fixture-only")
        self.assertNotIn("session_start_ns", query)
        assert_hardware_unavailable(self, stages)
        probe["shm_consumed_ns"] = 10
        with self.assertRaisesRegex(ValueError, "shared-memory"):
            apply_probe(catalog(False), probe)
        probe["shm_consumed_ns"] = None
        probe["display_latency"] = "measured"
        with self.assertRaisesRegex(ValueError, "display_latency"):
            apply_probe(catalog(False), probe)

    def test_live_ack_is_separate_from_scanout(self):
        probe = {
            "peer": "kilix-kitty", "graphics_query": "answered",
            "shm_consumed": "observed", "shm_consumed_ns": 50,
            "session_start_ns": 20, "frame_enqueue_ns": 5, "frame_written_ns": 30,
            "display_latency": "unavailable", "scanout": "unavailable",
            "gpu_latency": "unavailable",
        }
        stages = apply_probe(catalog(True), probe)
        query = next(item for item in stages if item["id"] == "consumer-graphics-query")
        shared = next(item for item in stages if item["id"] == "consumer-shm-read")
        self.assertEqual(query["status"], "measured")
        self.assertEqual(shared["status"], "measured")
        self.assertIn("Not scanout", query["reason"])
        self.assertIn("not display latency", shared["reason"].lower())
        assert_hardware_unavailable(self, stages)
        missed = dict(probe, shm_consumed="timeout", shm_consumed_ns=None)
        missed_stages = apply_probe(catalog(True), missed)
        self.assertEqual(
            next(item for item in missed_stages if item["id"] == "consumer-shm-read")["status"],
            "unavailable")


class ProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("cc") or framebuffer_source() is None:
            raise unittest.SkipTest("compiler and kitty-framebuffer sources required")
        cls.directory = tempfile.TemporaryDirectory(prefix="presentation-tests-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.binary = build_probe(cls.directory.name)

    def test_compiled_fixture_is_not_a_display_measurement(self):
        record = run_fixture(self.binary)
        self.assertEqual(record["peer"], "protocol-fixture")
        self.assertEqual(record["graphics_query"], "answered")
        self.assertEqual(record["shm_consumed"], "not-a-consumer")
        self.assertIsNone(record["shm_consumed_ns"])
        self.assertIsNone(record["session_start_ns"])
        for key in ("display_latency", "scanout", "gpu_latency"):
            self.assertEqual(record[key], "unavailable")
        encoded = json.dumps(record)
        self.assertNotIn("ssim", encoded.lower())
        self.assertNotIn(str(os.path.expanduser("~")), encoded)
        stages = apply_probe(catalog(False), record)
        self.assertEqual(
            next(item for item in stages if item["id"] == "consumer-graphics-query")["status"],
            "fixture-only")
        assert_hardware_unavailable(self, stages)

    def test_private_kitty_ack_does_not_claim_gpu_or_scanout(self):
        if kitty_binary() is None or shutil.which("Xvfb") is None:
            self.skipTest("private Kitty and Xvfb required")
        record = run_live(self.binary, kitty_binary(), timeout=25)
        encoded = json.dumps(record)
        self.assertNotIn(str(os.path.expanduser("~")), encoded)
        for key in ("display_latency", "scanout", "gpu_latency"):
            self.assertEqual(record[key], "unavailable")
        self.assertNotEqual(record.get("peer"), "protocol-fixture")
        stages = apply_probe(catalog(False), record)
        assert_hardware_unavailable(self, stages)
        if record.get("peer") != "kilix-kitty" or record.get("graphics_query") != "answered":
            self.fail("private Kitty did not answer the graphics query: " + record.get("reason", ""))
        query = next(item for item in stages if item["id"] == "consumer-graphics-query")
        self.assertEqual(query["status"], "measured")
        self.assertIn("Not scanout", query["reason"])
        shared = next(item for item in stages if item["id"] == "consumer-shm-read")
        self.assertEqual(record.get("shm_consumed"), "observed", encoded)
        self.assertIsInstance(record["shm_consumed_ns"], int)
        self.assertGreater(record["shm_consumed_ns"], 0)
        self.assertEqual(shared["status"], "measured")
        self.assertIn("not display latency", shared["reason"].lower())
        self.assertNotIn("KITTY_PID", encoded)


if __name__ == "__main__":
    unittest.main()
