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
from kilix_bench.braille_layouts import LAYOUTS, REGIONS, crop_field, tiles
from kilix_bench.report import assert_no_local_paths
from test_neuronte import arguments


class QualityTests(unittest.TestCase):
    def test_regions_partition_every_dot_and_preserve_original_coordinates(self):
        dots = ((.25, .25), (.25, .75), (.25, 1.25), (.75, .25),
                (.75, .75), (.75, 1.25), (.25, 1.75), (.75, 1.75))
        for layout, parts in LAYOUTS.items():
            with self.subTest(layout=layout):
                rectangles = tiles(layout, 8, 16)
                covered = np.zeros((16, 8), dtype=np.int32)
                for _, x, y, w, h in rectangles:
                    covered[y:y+h, x:x+w] += 1
                np.testing.assert_array_equal(covered, 1)
                # Every dot's ink belongs only to tiles retaining its bit.
                for bit, (x, y) in enumerate(dots):
                    for part in parts:
                        mask, x0, x1, y0, y1 = REGIONS[part]
                        nearest_x, nearest_y = np.clip(x, x0, x1), np.clip(y, y0, y1)
                        if (nearest_x-x)**2 + (nearest_y-y)**2 < .12**2:
                            self.assertTrue(mask & (1 << bit))
                for part in parts:
                    _, x0, x1, y0, y1 = REGIONS[part]
                    x, y = np.linspace(0, 1, 9), np.linspace(0, 2, 17)
                    full = x[None, :] + 3 * y[:, None]
                    expected = (x0 + x[None, :] * (x1-x0)) + 3 * (y0 + y[:, None] / 2 * (y1-y0))
                    np.testing.assert_array_equal(crop_field(full, REGIONS[part]), expected)

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

    def run_native(self, mode, codes, cell=(2, 4), weights=None, parts=False, passes=1, layout="columns"):
        args = arguments()
        args.raster_calls = len(codes) * passes
        return fonts.run_worker(self.binary, {**self.paths, "weights": weights or self.paths["weights"]},
                                "neuronte", "nrt-" + mode, cell, codes, args, self.directory.name,
                                part_weights=[self.paths["weights"]] * (len(LAYOUTS[layout])-1) if parts else None,
                                part_layout=layout)

    def test_parts_match_independent_calls_and_full_width_sampling_at_boundary(self):
        lib = C.CDLL(str(self.paths["library"]))
        lib.neuronte_open_file.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
        lib.neuronte_close.argtypes = [C.c_void_p]
        lib.neuronte_coverage.argtypes = [C.c_void_p, C.c_uint32, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_float)]
        ptr = C.c_void_p()
        self.assertEqual(lib.neuronte_open_file(os.fsencode(self.paths["weights"]), C.byref(ptr)), 0)
        self.addCleanup(lib.neuronte_close, ptr)
        codes = [0x2800, 0x2847, 0x28b8, 0x281b, 0x28e4, 0x2836, 0x28ff]
        for layout in LAYOUTS:
            for w, h in ((8, 16), (32, 64)):
                with self.subTest(layout=layout, width=w):
                    samples = 2 if w < 24 else 1
                    rectangles = tiles(layout, w, h)
                    expected = []
                    for code in codes:
                        canvas = np.empty((h, w), dtype=np.float32)
                        for mask, x, y, tw, th in rectangles:
                            output = (C.c_float * (tw * th))()
                            self.assertEqual(lib.neuronte_coverage(ptr, 0x2800 | ((code - 0x2800) & mask),
                                                                  tw, th, samples, output), 0)
                            canvas[y:y+th, x:x+tw] = np.frombuffer(output, dtype=np.float32).reshape(th, tw)
                        expected.append(canvas)
                    row, actual = self.run_native("validate", codes, (w, h), parts=True, layout=layout)
                    self.assertEqual(actual, np.asarray(expected).tobytes())
                    self.assertEqual(row["coverage_calls"], len(rectangles) * len(codes))
                    for mode in ("draw-cold", "draw-warm"):
                        row, drawn = self.run_native(mode, codes, (w, h), parts=True, layout=layout)
                        self.assertEqual(drawn, neuronte.draw_reference(expected[-1].tobytes()))
                        self.assertEqual(row["draw_calls"], len(rectangles) * len(codes))
                        self.assertEqual(row["priming_calls"], 0 if mode == "draw-cold" else len(codes))
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_native("validate", codes, (3, 3), parts=True, layout=layout)

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
            np.save(truth, np.ones((256, 16, 8), dtype=np.float32))
            cases = [{"name": "control", "kind": "bundle", "members": ["control"], "routes": [5] * 256},
                     {"name": "rows", "kind": "parts", "layout": "rows", "members": ["control"] * 2},
                     *[{"name": name, "kind": "parts", "layout": name, "members": ["control"] * len(LAYOUTS[name])}
                       for name in ("quadrants", "dots", "grid12", "grid24", "grid48")]]
            manifest = {"schema": "kilix-bench.braille-experiment/v1", "corpus": list(range(0x2800, 0x2900)),
                        "cases": cases, "models": {"control": {"file": weights.name, "sha256": fonts.digest(weights)}},
                        "targets": {"8x16": {"file": truth.name, "sha256": fonts.digest(truth)}},
                        "runtime_revision": "test-fixture", "teacher": {"kind": "deliberately wrong fixture"}}
            path = directory / "manifest.json"
            path.write_text(json.dumps(manifest))
            env = dict(os.environ, KILIX_BENCH_HOME=str(directory / "bench"), KILIX_HOME="", KILIX_STATE_LIBRARY="",
                       NEURONTE_LIBRARY=str(self.paths["library"]), NEURONTE_WEIGHTS=str(weights))
            subprocess.run([sys.executable, "-m", "kilix_bench", "--suite", "neuronte",
                            "--braille-experiment", str(path), "--cells", "8x16", "--repeats", "1", "--warmup", "0"],
                           env=env, check=True, capture_output=True, text=True, timeout=60)
            result = directory / "bench/results/kilix-braille-latest.json"
            data = json.loads(result.read_text())
            self.assertEqual(result.stat().st_mode & 0o777, 0o600)
            self.assertEqual(len(data["scenarios"]), 29)
            self.assertEqual(len(data["trials"]), 29)
            self.assertEqual([q["exact_glyphs"] for q in data["quality"]], [0] * 7)
            for row in data["scenarios"]:
                trial = next(t for t in data["trials"] if t["case"] == row["extra"]["case"] and
                             t["model"] == row["extra"]["model"])
                self.assertAlmostEqual(row["raw_seconds"][0], trial["ns_per_operation"] / 1e9)
                if trial["case"] == "coverage":
                    self.assertEqual(trial["coverage_calls"], 256 * (1 if trial["model"] == "control" else len(LAYOUTS[trial["model"]])))
            assert_no_local_paths(data, (str(Path.home()), root))
            for bad in ({**cases[2], "members": ["control"] * 2}, {**cases[1], "layout": "unknown"}):
                path.write_text(json.dumps({**manifest, "cases": [bad]}))
                with self.assertRaisesRegex(ValueError, "invalid experiment"):
                    braille.read_experiment(path)
            path.write_text(json.dumps(manifest))
            weights.write_bytes(b"changed asset")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                braille.read_experiment(path)


if __name__ == "__main__":
    unittest.main()
