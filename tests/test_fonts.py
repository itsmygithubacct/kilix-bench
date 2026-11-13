"""The font comparison verifies support, native pixels, and cache accounting."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from kilix_bench.font_geometry import analyze
from kilix_bench.font_profiles import CORPORA, PROFILES, profile
from kilix_bench.fonts import (
    add_arguments, build_worker, digest, inputs, paired_ratio, parse_cells,
    reference_frame, run_worker,
)
from kilix_bench.isolate import sandbox


def arguments():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    return parser.parse_args([])


class ProfileTests(unittest.TestCase):
    def test_coverage_comes_from_exact_weights_not_the_filename_or_route(self):
        with self.assertRaisesRegex(ValueError, "reviewed coverage"):
            profile("f" * 64, ["ascii"])
        name, corpora, missing = profile(next(iter(PROFILES)), ["ascii", "greek", "russian"])
        self.assertEqual(name, "split-v5")
        self.assertEqual([len(corpora[x]) for x in corpora], [95, 49, 66])
        self.assertFalse(missing)
        name, corpora, missing = profile(list(PROFILES)[1], ["greek", "ascii"])
        self.assertEqual(corpora["greek"], [])
        self.assertEqual(missing, {"greek"})

    def test_invalid_cell_dimensions_and_duplicate_sizes_are_rejected(self):
        for value in ["", "0x16", "8x257", "8x16x2", "8x16,8x16", "8,16"]:
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_cells(value)
        self.assertEqual(parse_cells("6x12,8x16"), [(6, 12), (8, 16)])

    def test_bootstrap_resamples_paired_trials(self):
        data = [float(n) for n in range(1, 31)]
        result = paired_ratio([n * 3 for n in data], data, 19)
        self.assertEqual(result["ratio"], 3)
        self.assertEqual(result["bootstrap_95_ci"], [3, 3])
        self.assertIsNone(paired_ratio([2], [1], 1)["bootstrap_95_ci"])

    def test_font_isolation_does_not_require_a_host_or_state_library(self):
        with patch("kilix_bench.isolate.state_library", side_effect=AssertionError("unused")):
            env, root = sandbox("", require_state=False)
        try:
            self.assertNotEqual(env["HOME"], str(Path.home()))
            self.assertEqual(env["KILIX_STATE_LIBRARY"], "")
        finally:
            shutil.rmtree(root)


class NativeFontTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("cc") or not shutil.which("pkg-config"):
            raise unittest.SkipTest("C compiler and pkg-config required")
        if subprocess.run(["pkg-config", "--exists", "freetype2"]).returncode:
            raise unittest.SkipTest("FreeType development files required")
        cls.font = Path(os.environ.get("KILIX_BENCH_TEST_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"))
        if not cls.font.is_file():
            raise unittest.SkipTest("DejaVu Sans Mono or KILIX_BENCH_TEST_FONT required")
        cls.directory = tempfile.TemporaryDirectory(prefix="font-tests-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.binary, cls.build = build_worker(cls.directory.name)

    def setUp(self):
        self.args = arguments()
        self.args.raster_calls = 23
        self.paths = {"font": self.font, "library": Path("/unused"), "weights": Path("/unused")}
        self.scratch = tempfile.TemporaryDirectory(prefix="font-case-")
        self.addCleanup(self.scratch.cleanup)

    def run_font(self, mode, codes, engine="freetype", capacity=1024):
        return run_worker(self.binary, self.paths, engine, mode, (8, 16), codes,
                          self.args, self.scratch.name, capacity)

    def test_provenance_fingerprints_the_linked_library_without_its_path(self):
        library = self.build["freetype_library"]
        self.assertTrue(library["name"].startswith("libfreetype"))
        self.assertNotIn("/", library["name"])
        self.assertRegex(library["sha256"], r"^[0-9a-f]{64}$")
        self.assertGreater(library["bytes"], 0)

    def test_missing_glyph_is_excluded_by_probe_and_refused_by_renderer(self):
        data, _ = self.run_font("probe", [ord("A"), 0x10ffff])
        self.assertEqual(data["supported"], [True, False])
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_font("raster", [0x10ffff])

    def test_cached_frames_equal_independent_composition_and_count_misses(self):
        codes = list(range(32, 64))
        reference, masks = self.run_font("validate", codes)
        self.assertEqual(masks[:128], bytes(128))  # space, not a fallback box
        self.assertTrue(any(masks[128:256]))
        self.assertGreaterEqual(reference["clipped_ink_samples"], 0)
        expected = reference_frame(masks, len(codes), (8, 16), 2000)
        cold, frame = self.run_font("draw-cold", codes)
        warm, warmed = self.run_font("draw-warm", codes)
        self.assertEqual(frame, expected)
        self.assertEqual(warmed, expected)
        self.assertEqual(cold["cache_misses"], len(codes))
        self.assertEqual(warm["cache_misses"], 0)
        self.assertEqual(warm["raster_calls"], 0)
        self.assertEqual(warm["cache_hits"], 2000)
        churn, churned = self.run_font("draw-warm", codes, capacity=4)
        self.assertEqual(churned, expected)
        self.assertEqual(churn["cache_hits"] + churn["cache_misses"], 2000)
        self.assertGreater(churn["cache_misses"], cold["cache_misses"])
        self.assertEqual(churn["raster_calls"], churn["cache_misses"])

    def test_uncached_native_batch_calls_the_rasterizer_every_time(self):
        row, _ = self.run_font("raster", [ord("A"), ord("g")])
        self.assertEqual(row["raster_calls"], self.args.raster_calls)
        self.assertEqual(row["cache_hits"], 0)
        self.assertGreater(row["elapsed_ns"], 0)
        self.assertGreater(row["initialize_ns"], 0)

    def test_missing_braille_sextant_and_octant_are_not_timed(self):
        codes = [0x2800, 0x28FF, 0x1FB00, 0x1CD00]
        data, _ = self.run_font("probe", codes)
        self.assertEqual(len(data["supported"]), len(codes))
        self.assertTrue(data["fixed_width"])
        if any(data["supported"]):
            self.skipTest("this font has some of the extended corpora")
        for code in codes:
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_font("raster", [code])

    def test_box_and_block_validation_records_geometry_without_ranking(self):
        codes = [0x2500, 0x2502, 0x2588, 0x2580]
        row, masks = self.run_font("validate", codes)
        glyphs = row["glyphs"]
        summary, _rows, _observed = analyze("box", (8, 16), codes, {
            "freetype": {"masks": masks, "glyphs": glyphs},
            "neuronte": {"masks": masks, "glyphs": glyphs},
        })
        self.assertEqual(summary["engines"]["freetype"]["clipped_ink_samples"], row["clipped_ink_samples"])
        self.assertIsNone(summary["comparison"]["quality_rank"])
        self.assertIsNone(summary["comparison"]["similarity_winner"])
        self.assertTrue(summary["comparison"]["pixel_equivalent"])
        self.assertIn("not a quality ranking", summary["comparison"]["note"].lower())
        self.assertTrue(all("outline" in glyph and "edges" in glyph for glyph in glyphs))

    def test_corrupt_font_never_produces_a_timing_record(self):
        broken = Path(self.scratch.name) / "broken.ttf"
        broken.write_bytes(b"not a font")
        self.paths["font"] = broken
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_font("raster", [ord("A")])

    def test_neuronte_masks_match_public_coverage_api_and_cached_frames(self):
        import ctypes as C
        import numpy as np
        home = os.environ.get("NEURONTE_HOME")
        if not home:
            self.skipTest("NEURONTE_HOME required for packed-model integration")
        self.paths.update(library=Path(home) / "build/libneuronte.so",
                          weights=Path(home) / "weights/neuronte-split-v5.nrt")
        self.assertIn(digest(self.paths["weights"]), PROFILES)
        codes = [32, ord("A"), ord("g"), 0x253c, 0x2588, 0x28ff]
        _, masks = self.run_font("validate", codes, "neuronte")
        lib = C.CDLL(str(self.paths["library"]))
        lib.neuronte_open_file.argtypes = [C.c_char_p, C.POINTER(C.c_void_p)]
        lib.neuronte_close.argtypes = [C.c_void_p]
        lib.neuronte_coverage.argtypes = [C.c_void_p, C.c_uint32, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_float)]
        ptr = C.c_void_p()
        self.assertEqual(lib.neuronte_open_file(os.fsencode(self.paths["weights"]), C.byref(ptr)), 0)
        try:
            expected = bytearray()
            for scalar in codes:
                buffer = (C.c_float * 128)()
                self.assertEqual(lib.neuronte_coverage(ptr, scalar, 8, 16, 0, buffer), 0)
                values = np.frombuffer(buffer, dtype=np.float32)
                expected.extend(np.floor(np.clip(values, 0, 1) * 255 + .5).astype(np.uint8).tobytes())
            self.assertEqual(masks, expected)
        finally:
            lib.neuronte_close(ptr)
        row, frame = self.run_font("draw-warm", codes, "neuronte")
        self.assertEqual(row["raster_calls"], 0)
        self.assertEqual(frame, reference_frame(masks, len(codes), (8, 16), 2000))


if __name__ == "__main__":
    unittest.main()
