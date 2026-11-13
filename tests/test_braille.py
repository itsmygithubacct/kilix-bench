"""Decomposition routing, sampling, composition and measured quality."""

import ctypes as C
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from kilix_bench import braille, braille_prepare, fonts, neuronte
from kilix_bench.report import assert_no_local_paths
from test_neuronte import arguments


class QualityTests(unittest.TestCase):
    def test_missing_ink_is_retained_as_failure_and_blank_is_exact(self):
        target = np.zeros((2, 4, 2), dtype=np.float32)
        target[1, 1, 0] = 1
        result = braille.quality(np.zeros_like(target), target, [0x2800, 0x2801])
        self.assertEqual(result["exact_glyphs"], 1)
        self.assertEqual(result["wrong_pixels"], 1)
        self.assertEqual(result["iou_mean"], .5)
        self.assertFalse(result["all_binary_exact"])
        self.assertEqual(result["per_glyph"][0]["iou"], 1)
        self.assertEqual(result["per_glyph"][1]["scalar"], 0x2801)


class NativeBrailleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        home = os.environ.get("NEURONTE_HOME")
        if not home or not shutil.which("cc") or not shutil.which("pkg-config"):
            raise unittest.SkipTest("NEURONTE_HOME, compiler and pkg-config required")
        if subprocess.run(["pkg-config", "--exists", "freetype2"]).returncode:
            raise unittest.SkipTest("FreeType development files required by the shared worker")
        cls.home = Path(home)
        cls.paths = {"library": cls.home / "build/libneuronte.so",
                     "weights": cls.home / "weights/neuronte-split-v5.nrt"}
        cls.directory = tempfile.TemporaryDirectory(prefix="braille-tests-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.binary, _ = fonts.build_worker(cls.directory.name)

    def run_native(self, mode, codes, cell=(2, 4), weights=None, parts=False, passes=1):
        args = arguments()
        args.raster_calls = len(codes) * passes
        return fonts.run_worker(self.binary, {**self.paths, "weights": weights or self.paths["weights"]},
                                "neuronte", "nrt-" + mode, cell, codes, args, self.directory.name,
                                second_weights=self.paths["weights"] if parts else None)

    def test_parts_match_independent_calls_and_full_width_sampling_at_boundary(self):
        lib = C.CDLL(str(self.paths["library"]))
        lib.neuronte_open_file.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
        lib.neuronte_close.argtypes = [C.c_void_p]
        lib.neuronte_coverage.argtypes = [C.c_void_p, C.c_uint32, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_float)]
        ptr = C.c_void_p()
        self.assertEqual(lib.neuronte_open_file(os.fsencode(self.paths["weights"]), C.byref(ptr)), 0)
        self.addCleanup(lib.neuronte_close, ptr)
        codes = [0x2800, 0x2847, 0x28b8, 0x28ff]
        for w, h in ((8, 16), (32, 64)):
            with self.subTest(width=w):
                samples = 2 if w < 24 else 1
                expected = []
                for code in codes:
                    halves = []
                    for mask in (0x47, 0xb8):
                        output = (C.c_float * (w // 2 * h))()
                        self.assertEqual(lib.neuronte_coverage(ptr, 0x2800 | ((code - 0x2800) & mask),
                                                              w // 2, h, samples, output), 0)
                        halves.append(np.frombuffer(output, dtype=np.float32).reshape(h, w // 2))
                    expected.append(np.concatenate(halves, axis=1))
                row, actual = self.run_native("validate", codes, (w, h), parts=True)
                self.assertEqual(actual, np.asarray(expected).tobytes())
                self.assertEqual(row["coverage_calls"], 2 * len(codes))
                for mode in ("draw-cold", "draw-warm"):
                    row, drawn = self.run_native(mode, codes, (w, h), parts=True)
                    self.assertEqual(drawn, neuronte.draw_reference(expected[-1].tobytes()))
                    self.assertEqual(row["draw_calls"], 2 * len(codes))
                    self.assertEqual(row["priming_calls"], 0 if mode == "draw-cold" else len(codes))
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_native("validate", codes, (3, 4), parts=True)

    def test_constructed_gate_and_table_select_identical_experts_for_all_patterns(self):
        # Use existing weights as fixtures: these tests establish routing and
        # equality, not the quality of a newly trained width-24 experiment.
        sys.path.insert(0, str(self.home / "train"))
        import pack
        directory = Path(self.directory.name)
        paths = []
        for name, labelled in (("test-lower", "U+2800..U+287F"), ("test-upper", "U+2880..U+28FF")):
            doc = json.loads((self.home / "weights/neuronte-braille-v0.json").read_text())
            doc["training"].update(set=name, labelled=labelled)
            path = directory / (name + ".json")
            path.write_text(json.dumps(doc))
            paths.append(str(path))
        table, _ = pack.bundle(paths)
        gate = braille_prepare.make_gate(["test-lower", "test-upper"], doc["shape"]["parity_masks"])
        gate_path = directory / "gate.json"
        gate_path.write_text(json.dumps(gate))
        gated, _ = pack.bundle(paths, str(gate_path))
        table_path, gated_path = directory / "table.nrt", directory / "gate.nrt"
        table_path.write_bytes(table)
        gated_path.write_bytes(braille_prepare.remove_ranges(gated))
        codes = list(range(0x2800, 0x2900))
        outputs = []
        for path in (table_path, gated_path):
            routes, _ = self.run_native("routes", codes, weights=path)
            self.assertEqual(routes["routes"], [0] * 128 + [1] * 128)
            row, _ = self.run_native("route-bench", codes, weights=path, passes=3)
            self.assertEqual(row["calls"], 768)
            self.assertEqual(row["route_checksum"], 384)
            _, pixels = self.run_native("validate", codes, weights=path)
            outputs.append(pixels)
        self.assertEqual(*outputs)

    def test_cli_records_quality_even_when_not_exact_and_checks_asset_hashes(self):
        with tempfile.TemporaryDirectory(prefix="braille-cli-") as root:
            directory = Path(root)
            weights = directory / "control.nrt"
            shutil.copyfile(self.paths["weights"], weights)
            # Deliberately wrong teacher to prove quality failures cannot turn
            # into success just because native buffers were produced correctly.
            truth = directory / "truth.npy"
            np.save(truth, np.ones((256, 4, 2), dtype=np.float32))
            case = {"name": "control", "kind": "bundle", "members": ["control"], "routes": [5] * 256}
            manifest = {"schema": "kilix-bench.braille-experiment/v1", "corpus": list(range(0x2800, 0x2900)),
                        "cases": [case], "models": {"control": {"file": weights.name, "sha256": fonts.digest(weights)}},
                        "targets": {"2x4": {"file": truth.name, "sha256": fonts.digest(truth)}},
                        "runtime_revision": "test-fixture", "teacher": {"kind": "deliberately wrong fixture"}}
            path = directory / "manifest.json"
            path.write_text(json.dumps(manifest))
            env = dict(os.environ, KILIX_BENCH_HOME=str(directory / "bench"), KILIX_HOME="", KILIX_STATE_LIBRARY="",
                       NEURONTE_LIBRARY=str(self.paths["library"]), NEURONTE_WEIGHTS=str(weights))
            subprocess.run([sys.executable, "-m", "kilix_bench", "--suite", "neuronte",
                            "--braille-experiment", str(path), "--cells", "2x4", "--repeats", "1", "--warmup", "0"],
                           env=env, check=True, capture_output=True, text=True, timeout=60)
            result = directory / "bench/results/kilix-braille-latest.json"
            data = json.loads(result.read_text())
            self.assertEqual(result.stat().st_mode & 0o777, 0o600)
            self.assertEqual(len(data["scenarios"]), 5)
            self.assertEqual(len(data["trials"]), 5)
            self.assertEqual(data["quality"][0]["exact_glyphs"], 0)
            for row in data["scenarios"]:
                trial = next(t for t in data["trials"] if t["case"] == row["extra"]["case"])
                self.assertAlmostEqual(row["raw_seconds"][0], trial["ns_per_operation"] / 1e9)
            assert_no_local_paths(data, (str(Path.home()), root))
            weights.write_bytes(b"changed asset")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                braille.read_experiment(path)


if __name__ == "__main__":
    unittest.main()
