"""Coverage planning refuses to time a missing glyph or mix faces."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from kilix_bench.__main__ import _font_command
from kilix_bench.font_coverage import coverage_plan, resolve_faces
from kilix_bench.fonts import add_arguments


def parser():
    result = argparse.ArgumentParser()
    add_arguments(result)
    result.add_argument("--repeats", type=int, default=1)
    result.add_argument("--warmup", type=int, default=0)
    return result


class Stub:
    def __init__(self, face_id, role, pitch="fixed"):
        self.id = face_id
        self.role = role
        self.pitch = pitch


class CoveragePlanTests(unittest.TestCase):
    def test_primary_intersection_is_timed_without_mixing_faces(self):
        faces = [Stub("primary", "primary"), Stub("extra", "coverage")]
        corpora = {"box": [1, 2, 3]}
        probes = {
            "primary": {"box": [True, False, True]},
            "extra": {"box": [True, True, True]},
        }
        report, jobs = coverage_plan(corpora, set(), faces, probes)
        self.assertEqual(report["box"]["action"], "timed-against-primary")
        self.assertEqual(report["box"]["comparison_face"], "primary")
        self.assertEqual(report["box"]["shared"], [1, 3])
        self.assertEqual(report["box"]["freetype_missing"], [2])
        self.assertEqual(jobs, [("box", "primary", [1, 3])])
        self.assertNotIn("present_scalars", report["box"]["faces"][0])

    def test_empty_primary_uses_one_coverage_face_and_does_not_time_notdef(self):
        faces = [Stub("primary", "primary"), Stub("later", "coverage"), Stub("earlier", "coverage")]
        corpora = {"braille": [0x2800, 0x2801]}
        probes = {
            "primary": {"braille": [False, False]},
            "later": {"braille": [True, True]},
            "earlier": {"braille": [True, False]},
        }
        report, jobs = coverage_plan(corpora, set(), faces, probes)
        self.assertEqual(report["braille"]["action"], "timed-against-coverage-face")
        self.assertEqual(report["braille"]["comparison_face"], "later")
        self.assertEqual(jobs, [("braille", "later", [0x2800, 0x2801])])

    def test_equal_coverage_keeps_manifest_order(self):
        faces = [Stub("primary", "primary"), Stub("first", "coverage"), Stub("second", "coverage")]
        probes = {
            "primary": {"braille": [False]},
            "first": {"braille": [True]},
            "second": {"braille": [True]},
        }
        report, jobs = coverage_plan({"braille": [1]}, set(), faces, probes)
        self.assertEqual(report["braille"]["comparison_face"], "first")
        self.assertEqual(jobs[0][1], "first")

    def test_no_face_writes_no_job(self):
        faces = [Stub("primary", "primary"), Stub("extra", "coverage")]
        probes = {"primary": {"octants": [False]}, "extra": {"octants": [False]}}
        report, jobs = coverage_plan({"octants": [0x1CD00]}, set(), faces, probes)
        self.assertEqual(report["octants"]["status"], "unavailable")
        self.assertEqual(report["octants"]["action"], "not-timed")
        self.assertEqual(jobs, [])
        self.assertIn("not drawn", report["octants"]["note"])

    def test_weights_outside_the_profile_are_not_probed_into_a_timing(self):
        faces = [Stub("primary", "primary")]
        report, jobs = coverage_plan({"greek": [0x391]}, {"greek"}, faces, {})
        self.assertEqual(report["greek"]["status"], "unavailable")
        self.assertEqual(jobs, [])
        self.assertIn("Weights", report["greek"]["note"])

    def test_manifest_and_flags_resolve_one_primary_and_record_no_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            font = root / "PrimaryMono.ttf"
            licence = root / "primary-licence.txt"
            other = root / "CoverageMono.ttf"
            other_licence = root / "coverage-licence.txt"
            for path in (font, licence, other, other_licence):
                path.write_text("asset", encoding="utf-8")
            manifest = root / "faces.json"
            manifest.write_text(json.dumps({"faces": [
                {"id": "primary", "role": "primary", "file": str(font),
                 "license": str(licence), "license_name": "primary terms"},
                {"id": "cover", "role": "coverage", "file": str(other),
                 "license": str(other_licence), "license_name": "coverage terms",
                 "pitch": "cell"},
            ]}), encoding="utf-8")
            args = parser().parse_args(["--font-manifest", str(manifest)])
            faces = resolve_faces(args)
            self.assertEqual([face.id for face in faces], ["primary", "cover"])
            self.assertEqual(faces[1].pitch, "cell")
            recorded = faces[0].record(lambda path: "ab" * 32)
            self.assertEqual(recorded["name"], "PrimaryMono.ttf")
            self.assertNotIn(str(root), json.dumps(recorded))
            with self.assertRaisesRegex(ValueError, "replaces"):
                resolve_faces(parser().parse_args([
                    "--font-manifest", str(manifest), "--font", str(font),
                    "--font-license", str(licence)]))
            args = parser().parse_args([
                "--font", str(font), "--font-license", str(licence),
                "--font-license-name", "primary terms",
                "--coverage-font", f"cover={other}",
                "--coverage-license", f"cover={other_licence}",
                "--coverage-license-name", "cover=coverage terms",
                "--coverage-pitch", "cover=cell",
                "--cases", "raster,empty-cache",
                "--presentation", "fixture",
                "--geometry-cells", "none",
            ])
            command = _font_command(args, resolve_faces(args))
            self.assertIn("--coverage-pitch", command)
            self.assertIn("cover=cell", command)
            self.assertIn("--presentation", command)
            self.assertIn("fixture", command)
            self.assertIn("--geometry-cells", command)
            self.assertIn("none", command)
            self.assertNotIn("--font-manifest", command)


if __name__ == "__main__":
    unittest.main()
