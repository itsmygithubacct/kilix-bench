"""Presentation probes classify consumers without inventing display latency."""

import json
import os
import shutil
import tempfile
import unittest

from kilix_bench.presentation import build_probe, framebuffer_source, run, run_fixture, run_live, kitty_binary
from kilix_bench.report import render_text
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
        shared = next(item for item in stages if item["id"] == "consumer-shm-read")
        self.assertIn("Enqueue-to-observed-unlink", shared["measures"])
        self.assertIn("frame_written_ns", shared["measures"])
        self.assertIn("not exact", shared["does_not_measure"].lower())
        self.assertIn("one sample", shared["does_not_measure"].lower())
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
            "clock": "enqueue-to-observed-unlink",
            "milestone_start": "immediately-before-kittyfb-present",
            "poll": "first 200 busy, then 200000 ns until 4000 spins, then 1000000 ns",
        }
        stages = apply_probe(catalog(True), probe)
        query = next(item for item in stages if item["id"] == "consumer-graphics-query")
        shared = next(item for item in stages if item["id"] == "consumer-shm-read")
        self.assertEqual(query["status"], "measured")
        self.assertEqual(shared["status"], "measured")
        self.assertIn("Not scanout", query["reason"])
        self.assertIn("not display latency", shared["reason"].lower())
        self.assertEqual(shared["clock"], "enqueue-to-observed-unlink")
        self.assertEqual(shared["milestone_start"], "immediately-before-kittyfb-present")
        self.assertIn("4000", shared["poll"])
        self.assertIn("enqueue-to-observed-unlink", shared["reason"])
        self.assertIn("enqueue-to-observed-frames-encoded", shared["reason"])
        self.assertNotIn("exact consumer latency", shared["reason"].replace("not exact consumer latency", ""))
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
        self.assertIsNone(record["frame_enqueue_ns"])
        self.assertIsNone(record["frame_written_ns"])
        self.assertEqual(record["clock"], "not-measured")
        self.assertIn("discarded", record["poll"])
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
        self.assertEqual(record["clock"], "enqueue-to-observed-unlink")
        self.assertEqual(record["milestone_start"], "immediately-before-kittyfb-present")
        self.assertIn("200000", record["poll"])
        self.assertIn("1000000", record["poll"])
        self.assertIn("4000", record["poll"])
        self.assertIn("not exact consumer latency", record["poll"])
        self.assertIsInstance(record["frame_written_ns"], int)
        self.assertGreater(record["shm_consumed_ns"], 0)
        self.assertGreaterEqual(record["shm_consumed_ns"], record["frame_written_ns"])
        self.assertNotIn("write_to_unlink_ns", record)

    def test_normal_result_embeds_probe_provenance_without_paths(self):
        fixture = run("fixture", self.directory.name)
        provenance = fixture["provenance"]
        self.assertEqual(fixture["clock"], "not-measured")
        self.assertIsNone(provenance["kitty"])
        self.assertEqual(len(provenance["probe_source_sha256"]), 64)
        self.assertEqual(len(provenance["probe_binary_sha256"]), 64)
        self.assertTrue(provenance["kitty_framebuffer_revision"])
        names = [item["name"] for item in provenance["kitty_framebuffer_sources"]]
        self.assertIn("kitty_framebuffer.c", names)
        for item in provenance["kitty_framebuffer_sources"]:
            self.assertNotIn("/", item["name"])
            self.assertEqual(len(item["sha256"]), 64)
        blob = json.dumps(fixture)
        self.assertNotIn(str(os.path.expanduser("~")), blob)
        self.assertNotIn("/home/", blob)
        text = render_text({
            "tool_version": "0.2.4", "subject": "kilix-fonts", "subject_version": "1",
            "size": [80, 25], "python": "3", "machine": "x86_64", "scenarios": [],
            "presentation": fixture,
        })
        self.assertIn("probe source " + provenance["probe_source_sha256"], text)
        self.assertIn("probe binary " + provenance["probe_binary_sha256"], text)
        self.assertIn(provenance["kitty_framebuffer_revision"], text)
        if kitty_binary() is None or shutil.which("Xvfb") is None:
            return
        live = run("live", self.directory.name)
        kitty = live["provenance"]["kitty"]
        self.assertEqual(kitty["name"], "kitty")
        self.assertEqual(len(kitty["sha256"]), 64)
        self.assertTrue(kitty["version"])
        self.assertNotIn("/", kitty["version"])
        live_blob = json.dumps(live)
        self.assertNotIn(str(os.path.expanduser("~")), live_blob)
        self.assertNotIn("/home/", live_blob)
        rendered = render_text({
            "tool_version": "0.2.4", "subject": "kilix-fonts", "subject_version": "1",
            "size": [80, 25], "python": "3", "machine": "x86_64", "scenarios": [],
            "presentation": live,
        })
        self.assertIn("kitty kitty " + kitty["sha256"], rendered)


if __name__ == "__main__":
    unittest.main()
