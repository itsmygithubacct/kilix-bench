"""Desktop-face experiments: packed-face reference, oracles, manifests and native runs."""

import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from kilix_bench import face, face_prepare

MASKS = [1 << (i % 21) | 1 << ((i * 7 + 3) % 21) for i in range(43)]


def net_v1(h, seed, bias=0.0):
    rng = np.random.default_rng(seed)
    out = bytearray(b"NEURONTE") + struct.pack("<5I", 1, 6, 64, h, 3) + struct.pack("<f", .15)
    out += struct.pack("<43I", *MASKS)
    for rows, cols in ((h, 90), (h, h), (h, h), (1, h)):
        out += rng.normal(0, .3, rows * cols).astype("<f4").tobytes()
        b = rng.normal(0, .1, rows).astype("<f4")
        if rows == 1:
            b += bias
        out += b.tobytes()
    return bytes(out)


def advances(records, fallback=0x3F):
    out = bytearray(b"NEURONTE") + struct.pack("<6I", 4, 10, 10, 2, len(records), fallback)
    for s, (r, b) in sorted(records.items()):
        out += struct.pack("<I2H", s, r, b)
    return bytes(out)


def bundle(nets, ranges):
    out = bytearray(b"NEURONTE") + struct.pack("<5I", 2, len(nets), len(ranges), 0, 0)
    out += struct.pack(f"<{len(nets)}I", *map(len, nets))
    for r in ranges:
        out += struct.pack("<3I", *r)
    return bytes(out) + b"".join(nets)


def drawing(members):
    out = bytearray(b"NEURONTE") + struct.pack("<2I", 5, len(members))
    off = 16 + 8 * len(members)
    for m in members:
        out += struct.pack("<2I", off, len(m))
        off += len(m)
    return bytes(out) + b"".join(members)


class FaceFileTests(unittest.TestCase):
    def test_single_net_and_advances(self):
        table = advances({0x3F: (50, 60), 0x41: (64, 71)})
        packed = face.FaceFile(drawing([net_v1(8, 1), table]))
        self.assertEqual(packed.version, 5)
        self.assertEqual(packed.route(0x41), 0)
        self.assertEqual(packed.advance(0x41, 0, 10), 6)
        self.assertEqual(packed.advance(0x41, 1, 10), 7)
        self.assertEqual(packed.advance(0x4E2D, 0, 10), 5, "an unknown scalar is measured as the fallback")
        cov = packed.coverage(0x41, 6, 12)
        self.assertEqual(cov.shape, (12, 6))
        self.assertTrue(((cov >= 0) & (cov <= 1)).all())

    def test_bundle_routes_by_its_table(self):
        b = bundle([net_v1(8, 1), net_v1(8, 2)], [(0x20, 0x4F, 0), (0x50, 0x7E, 1)])
        packed = face.FaceFile(drawing([b, advances({0x3F: (50, 60)})]))
        self.assertEqual([packed.route(s) for s in (0x20, 0x4F, 0x50, 0x7E)], [0, 0, 1, 1])
        with self.assertRaisesRegex(ValueError, "outside"):
            packed.route(0x2026)

    def test_nets_beside_a_bundle_and_two_tables_are_refused(self):
        b = bundle([net_v1(4, 1)], [(0x20, 0x7E, 0)])
        table = advances({0x3F: (50, 60)})
        with self.assertRaises(ValueError):
            face.FaceFile(drawing([b, net_v1(4, 2), table]))
        with self.assertRaisesRegex(ValueError, "two advance"):
            face.FaceFile(drawing([net_v1(4, 2), table, table]))

    def test_canvas_reference_cuts_or_composites(self):
        cov = np.zeros((4, 2), np.float32)
        cov[0, 0], cov[1, 1], cov[2, 0] = .7, .5, .2
        cut = np.frombuffer(face.canvas_reference(cov, 2, True), np.uint32).reshape(4, 4)
        self.assertEqual(cut[0, 0], 0xFFFFFFFF)
        self.assertEqual(cut[1, 1], 0, "exactly one half is not ink")
        self.assertEqual(int(cut[:, 2:].sum()), 0, "the two columns past the cell stay clear")
        soft = np.frombuffer(face.canvas_reference(cov, 2, False), np.uint32).reshape(4, 4)
        self.assertEqual(soft[2, 0] >> 24, int(np.float32(np.float32(.2 * 255) + .5)))

    def test_quality_counts_wrong_pixels_against_the_drawing(self):
        target = np.zeros((2, 4, 4))
        target[0, 1, 1] = 1
        canvases = np.zeros((2, 4, 4), np.uint32)
        canvases[0, 1, 1] = 0xFFFFFFFF
        canvases[1, 0, 0] = 0xFFFFFFFF
        q = face.quality(canvases, target, [0x41, 0x42], True)
        self.assertEqual((q["exact_glyphs"], q["wrong_pixels"], q["differ"]), (1, 1, ["U+0042"]))


class ManifestTests(unittest.TestCase):
    def test_tampered_model_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "m.nrt").write_bytes(b"x")
            np.save(root / "t.npy", np.zeros((1, 2, 3)))
            manifest = {"schema": face.SCHEMA, "face": "ui1", "ems": [1], "binary": True, "corpus": [0x41],
                        "models": {"m": {"file": "m.nrt", "sha256": "0" * 64}},
                        "targets": {"1": {"file": "t.npy", "sha256": face.fonts.digest(root / "t.npy")}},
                        "teacher": {}, "runtime_revision": "", "cases": [{"name": "m", "model": "m"}]}
            (root / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                face.read_experiment(root / "manifest.json")

    def test_pixel_face_reader(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "ui-2.txt"
            path.write_text("# a comment\nA 2\n#.\n.#\nU+0042 1\n#\n#\n")
            got = face_prepare.pixel_face(path, 2)
            self.assertEqual(got, {0x41: (2, ["#.", ".#"]), 0x42: (1, ["#", "#"])})


class NativeFaceTests(unittest.TestCase):
    """A real face from a Neuronte checkout: prepare, validate and time one round."""

    @classmethod
    def setUpClass(cls):
        home = os.environ.get("NEURONTE_HOME")
        if not home or not shutil.which("cc"):
            raise unittest.SkipTest("NEURONTE_HOME and a compiler required")
        cls.home = Path(home)
        cls.model = cls.home / "weights/neuronte-ui8-v0.nrt"
        if not (cls.home / "build/libneuronte.so").is_file() or not cls.model.is_file():
            raise unittest.SkipTest("built library and neuronte-ui8-v0 required")

    def arguments(self, manifest, **extra):
        values = dict(face_experiment=str(manifest), neuronte_home=str(self.home), expert_cases="coverage,draw-cold,draw-warm",
                      expert_passes=2, experiment_quality_only=False, repeats=1, warmup=0, seed=7, cpu=None,
                      job_timeout=120)
        values.update(extra)
        return argparse.Namespace(**values)

    def test_prepare_measure_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manifest = face_prepare.prepare(root / "exp", self.home, "ui8", False,
                                            [("ui8-v0", self.model)], {"ui8-v0": {"seed": 0}}, [])
            with self.assertRaises(FileExistsError):
                face_prepare.prepare(root / "exp", self.home, "ui8", False, [("ui8-v0", self.model)], {}, [])
            artifacts = root / "artifacts"
            artifacts.mkdir()
            os.environ["KILIX_BENCH_ARTIFACT_DIR"] = str(artifacts)
            result = face.measure(self.arguments(manifest))
            q = result["quality"][0]
            self.assertEqual((q["glyphs"], q["exact_glyphs"], q["wrong_pixels"]), (96, 96, 0))
            self.assertLessEqual(result["reference"][0]["max_abs_difference"], 1e-3)
            self.assertEqual(result["advances"][0]["checked"], 192)
            cases = {s["extra"]["case"] for s in result["scenarios"]}
            self.assertEqual(cases, {"open", "coverage", "draw-cold", "draw-warm"})
            cold = [t for t in result["trials"] if t["case"] == "draw-cold"][0]
            self.assertEqual((cold["calls"], cold["draw_calls"], cold["width_calls"], cold["priming_calls"]),
                             (96, 96, 192, 0))
            warm = [t for t in result["trials"] if t["case"] == "draw-warm"][0]
            self.assertEqual((warm["calls"], warm["priming_calls"]), (192, 96))
            self.assertEqual(result["model_cases"][0]["seed"], 0)
            json.dumps(result, allow_nan=False)

    def test_quality_only_runs_no_timing(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manifest = face_prepare.prepare(root / "exp", self.home, "ui8", False, [("ui8-v0", self.model)], {}, [])
            artifacts = root / "artifacts"
            artifacts.mkdir()
            os.environ["KILIX_BENCH_ARTIFACT_DIR"] = str(artifacts)
            result = face.measure(self.arguments(manifest, experiment_quality_only=True))
            self.assertEqual(result["trials"], [])
            self.assertEqual(result["subject"], "kilix-face-quality")

    def test_other_advances_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            # ui9's file against ui8's drawing: the advances differ, so preparation refuses it.
            with self.assertRaisesRegex(ValueError, "advance"):
                face_prepare.prepare(root / "exp", self.home, "ui8", False,
                                     [("wrong", self.home / "weights/neuronte-ui9-v0.nrt")], {}, [])

    def test_a_net_that_misses_its_drawing_is_recorded_not_dropped(self):
        def members(path):
            data = path.read_bytes()
            count = struct.unpack_from("<I", data, 12)[0]
            return [data[a:a + n] for a, n in (struct.unpack_from("<2I", data, 16 + 8 * i) for i in range(count))]
        ui8 = members(self.model)
        ui9 = members(self.home / "weights/neuronte-ui9-v0.nrt")
        net9 = [m for m in ui9 if struct.unpack_from("<I", m, 8)[0] == 1][0]
        table8 = [m for m in ui8 if struct.unpack_from("<I", m, 8)[0] == 4][0]
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            mixed = root / "mixed.nrt"
            mixed.write_bytes(drawing([net9, table8]))
            manifest = face_prepare.prepare(root / "exp", self.home, "ui8", False,
                                            [("ui8-v0", self.model), ("mixed", mixed)], {}, [])
            artifacts = root / "artifacts"
            artifacts.mkdir()
            os.environ["KILIX_BENCH_ARTIFACT_DIR"] = str(artifacts)
            result = face.measure(self.arguments(manifest, experiment_quality_only=True))
            exact = {q["model"]: q["exact_glyphs"] for q in result["quality"]}
            self.assertEqual(exact["ui8-v0"], 96)
            self.assertLess(exact["mixed"], 96)

    def test_cli_runs_in_an_isolated_home(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manifest = face_prepare.prepare(root / "exp", self.home, "ui8", False, [("ui8-v0", self.model)], {}, [])
            env = dict(os.environ, KILIX_BENCH_HOME=str(root / "bench"),
                       PYTHONPATH=str(Path(face.__file__).resolve().parents[1]))
            done = subprocess.run([sys.executable, "-m", "kilix_bench", "--suite", "neuronte", "--neuronte-home",
                                   str(self.home), "--face-experiment", str(manifest), "--repeats", "1",
                                   "--warmup", "0", "--expert-passes", "1"],
                                  env=env, capture_output=True, text=True, timeout=600)
            self.assertEqual(done.returncode, 0, done.stderr + done.stdout)
            latest = list((root / "bench" / "results").glob("*kilix-face*.json"))
            self.assertTrue(latest, list((root / "bench" / "results").iterdir()))


if __name__ == "__main__":
    unittest.main()
