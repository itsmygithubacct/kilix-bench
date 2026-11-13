"""Expert coverage, native pixels, cache boundaries and CLI isolation."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from kilix_bench import fonts, neuronte
from kilix_bench.font_profiles import expert_profile
from kilix_bench.report import assert_no_local_paths


def arguments():
    parser = argparse.ArgumentParser()
    fonts.add_arguments(parser)
    neuronte.add_arguments(parser)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    return parser.parse_args([])


class ExpertGuardTests(unittest.TestCase):
    def test_unknown_model_cannot_claim_training_coverage(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "neuronte-split-v5.nrt"
            path.write_bytes(b"an unreviewed file with the expected filename")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                expert_profile(path)

    def test_successful_but_wrong_route_is_rejected(self):
        expert = {"index": 3, "scalars": [0x391, 0x410]}
        neuronte.check_routes({"routes": [3, 3]}, expert)
        with self.assertRaisesRegex(ValueError, "routes disagree"):
            neuronte.check_routes({"routes": [0, 0]}, expert)

    def test_short_or_cached_coverage_batches_are_rejected(self):
        row = {"calls": 6, "coverage_calls": 6, "draw_calls": 0, "priming_calls": 3}
        neuronte.check_trial(row, b"pixels", b"pixels", "coverage", 3, 6)
        for key, value in [("calls", 2), ("coverage_calls", 0), ("draw_calls", 6), ("priming_calls", 0)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                neuronte.check_trial({**row, key: value}, b"pixels", b"pixels", "coverage", 3, 6)
        with self.assertRaisesRegex(ValueError, "output differs"):
            neuronte.check_trial(row, b"wrong", b"pixels", "coverage", 3, 6)


class NativeExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        home = os.environ.get("NEURONTE_HOME")
        if not home or not shutil.which("cc") or not shutil.which("pkg-config"):
            raise unittest.SkipTest("NEURONTE_HOME, compiler and pkg-config required")
        if subprocess.run(["pkg-config", "--exists", "freetype2"]).returncode:
            raise unittest.SkipTest("FreeType development files required by the shared worker")
        cls.paths = {"library": Path(home) / "build/libneuronte.so",
                     "weights": Path(home) / "weights/neuronte-split-v5.nrt"}
        cls.directory = tempfile.TemporaryDirectory(prefix="expert-tests-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.binary, _ = fonts.build_worker(cls.directory.name)
        cls.experts = expert_profile(cls.paths["weights"])

    def run_native(self, mode, codes, calls=None):
        args = arguments()
        args.raster_calls = calls or len(codes)
        return fonts.run_worker(self.binary, self.paths, "neuronte", "nrt-" + mode,
                                (2, 4), codes, args, self.directory.name)

    def test_all_six_complete_repertoires_route_and_render_through_each_api(self):
        self.assertEqual([len(e["scalars"]) for e in self.experts.values()], [95, 131, 49, 318, 327, 256])
        self.assertEqual(len({c for e in self.experts.values() for c in e["scalars"]}), 1176)
        self.assertIn(0x391, self.experts["accent"]["scalars"])
        self.assertIn(0x451, self.experts["accent"]["scalars"])
        for name, expert in self.experts.items():
            with self.subTest(expert=name):
                codes = expert["scalars"]
                routes, _ = self.run_native("routes", codes)
                neuronte.check_routes(routes, expert)
                validation, masks = self.run_native("validate", codes)
                self.assertEqual(len(masks), len(codes) * 8 * 4)
                self.assertEqual(validation["coverage_calls"], len(codes))
                row, actual = self.run_native("coverage", codes, len(codes) * 2)
                neuronte.check_trial(row, actual, masks[-32:], "coverage", len(codes), len(codes) * 2)
                self.assertGreater(row["initialize_ns"], 0)
                self.assertGreater(row["first_coverage_ns"], 0)
                expected = neuronte.draw_reference(masks[-32:])
                for case in ("draw-cold", "draw-warm"):
                    row, actual = self.run_native(case, codes)
                    neuronte.check_trial(row, actual, expected, case, len(codes), len(codes))
                    self.assertIsNone(row["cache_hits"])
                    self.assertIsNone(row["cache_misses"])

    def test_open_is_separate_and_incomplete_corpus_pass_is_refused(self):
        row, raw = self.run_native("open", [32])
        self.assertFalse(raw)
        self.assertGreater(row["initialize_ns"], 0)
        self.assertNotIn("coverage_calls", row)
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_native("coverage", [65, 66], 3)

    def test_cli_writes_normalized_private_results_without_a_font_or_host(self):
        with tempfile.TemporaryDirectory(prefix="expert-cli-") as root:
            env = dict(os.environ, KILIX_BENCH_HOME=root, KILIX_HOME="", KILIX_STATE_LIBRARY="",
                       NEURONTE_LIBRARY=str(self.paths["library"]), NEURONTE_WEIGHTS=str(self.paths["weights"]))
            subprocess.run([sys.executable, "-m", "kilix_bench", "--suite", "neuronte",
                            "--experts", "braille", "--cells", "2x4", "--repeats", "1", "--warmup", "0"],
                           env=env, check=True, capture_output=True, text=True, timeout=60)
            path = Path(root) / "results/kilix-neuronte-latest.json"
            data = json.loads(path.read_text())
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(len(data["scenarios"]), 4)
            self.assertEqual(len(data["validation"]), 1)
            self.assertEqual(len(data["trials"]), 3)
            for row, case in zip(data["scenarios"][1:], neuronte.CASES):
                trial = next(t for t in data["trials"] if t["case"] == case)
                self.assertAlmostEqual(row["raw_seconds"][0], trial["elapsed_ns"] / 256 / 1e9)
                self.assertEqual(row["extra"]["samples_per_axis"], 2)
            assert_no_local_paths(data, (str(Path.home()), root))


if __name__ == "__main__":
    unittest.main()
