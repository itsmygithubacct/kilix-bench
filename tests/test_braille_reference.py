"""The independent packed-file reference used to verify production Braille."""

import ctypes as C
import os
from pathlib import Path
import struct
import unittest

import numpy as np

from kilix_bench import braille
from kilix_bench.braille_reference import Packed, Tiled


def distance_image(h, seed, tau=.3):
    rng = np.random.default_rng(seed)
    floats = rng.normal(0, .5, 2 * h * h + 94 * h + 1).astype("<f4")
    masks = [(0x9E3779B1 * (i + 1)) & 0x1FFFFF or 1 for i in range(43)]
    head = b"NEURONTE" + struct.pack("<5If43I", 1, 6, 64, h, 3, tau, *masks)
    return head + floats.tobytes()


def tiled_image(columns, rows, tiles, nets, first=0x2800, last=0x28FF):
    out = b"NEURONTE" + struct.pack("<6I", 6, len(columns) - 1, len(rows) - 1, len(nets), first, last)
    out += struct.pack(f"<{len(columns)}f", *columns) + struct.pack(f"<{len(rows)}f", *rows)
    for t in tiles:
        out += struct.pack("<4I", *t)
    out += struct.pack(f"<{len(nets)}I", *map(len, nets))
    return out + b"".join(nets)


def bundle_image(members, ranges):
    out = b"NEURONTE" + struct.pack("<5I", 2, len(members), len(ranges), 0, 0)
    out += struct.pack(f"<{len(members)}I", *map(len, members))
    for r in ranges:
        out += struct.pack("<3I", *r)
    return out + b"".join(members)


class ReferenceTests(unittest.TestCase):
    def test_whole_wrapper_is_the_net_and_codes_follow_the_tile_records(self):
        net = distance_image(3, 1)
        whole = Packed(tiled_image([0, 1], [0, 2], [(0, 0xFF, 0x2800, 0)], [net]))
        alone = Packed(net)
        for code in (0x2800, 0x2855, 0x28FF):
            for cell in ((1, 2), (7, 13), (24, 48)):
                np.testing.assert_array_equal(whole.coverage(code, *cell), alone.coverage(code, *cell))
        tiled = Tiled(tiled_image([0, .5, 1], [0, 1, 2],
                                  [(0, 0x03, 0x2800, 1), (0, 0x18, 0x2800, 1), (0, 0x44, 0x41, 0), (0, 0xA0, 0x2800, 1)],
                                  [net]))
        self.assertEqual(tiled.code(0x2800 | 0x10, tiled.tiles[1]), 0x2802)
        self.assertEqual(tiled.code(0x2800 | 0x80, tiled.tiles[3]), 0x2802)
        self.assertEqual(tiled.code(0x2800 | 0x44, tiled.tiles[2]), 0x41 | 0x44)
        self.assertEqual(tiled.code(0x27FF, tiled.tiles[0]), 0x2803)  # b wraps modulo 2^32

    def test_samples_on_an_edge_belong_to_the_tile_that_starts_there(self):
        nets = [distance_image(2, 2), distance_image(2, 3)]
        image = Packed(tiled_image([0, .25, 1], [0, 2], [(0, 0, 0x2800, 0), (1, 0, 0x2800, 0)], nets))
        right = image.members[0].nets[1]
        # A 2 x 1 cell, one sample per pixel: x = .25 and .75, y = 1. The
        # first sample lies on the edge, so the right tile draws it at its
        # own x = 0; the second is at (.75 - .25) / .75 of that tile.
        got = image.coverage(0x2800, 2, 1, 1)
        for px, local in ((0, 0.), (1, 2 / 3)):
            f = right.field(0x2800, np.array([local]), np.array([1.]))[0]
            self.assertAlmostEqual(got[0, px], float(np.clip(.5 + f * right.tau * .75 * 2, 0, 1)), places=12)

    def test_bundle_routes_by_table_and_refuses_gate_scalars(self):
        a, b = distance_image(2, 4), tiled_image([0, 1], [0, 2], [(0, 0xFF, 0x2800, 0)], [distance_image(2, 5)])
        packed = Packed(bundle_image([a, b], [(0x20, 0x7E, 0), (0x2800, 0x28FF, 1)]))
        self.assertEqual(packed.route(0x41), 0)
        self.assertEqual(packed.route(0x28C3), 1)
        with self.assertRaisesRegex(ValueError, "range table"):
            packed.route(0x3000)
        with self.assertRaisesRegex(ValueError, "length"):
            Packed(bundle_image([a, b], [(0x20, 0x7E, 0)]) + b"x")

    def test_production_check_fails_closed(self):
        import tempfile
        net = distance_image(2, 6)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "net.nrt"
            path.write_bytes(net)
            codes = list(range(0x2800, 0x2900))
            ref = Packed(net)
            coverage = np.stack([ref.coverage(c, 8, 16) for c in codes]).astype(np.float32)
            case = {"name": "fixture", "routes": [0] * 256}
            row = braille.check_reference(case, path, coverage, codes, (8, 16))
            self.assertLessEqual(row["max_abs_difference"], 1e-6)
            coverage[codes.index(0x2855)] += 2e-3
            with self.assertRaisesRegex(ValueError, "from the reference"):
                braille.check_reference(case, path, coverage, codes, (8, 16))
            with self.assertRaisesRegex(ValueError, "declared routes"):
                braille.check_reference({"name": "fixture", "routes": [1] * 256}, path, coverage, codes, (8, 16))


class NativeReferenceTests(unittest.TestCase):
    """The packed-file reference agrees with the library it checks."""

    def setUp(self):
        home = os.environ.get("NEURONTE_HOME")
        if not home or not (Path(home) / "build/libneuronte.so").exists():
            raise unittest.SkipTest("NEURONTE_HOME with a built library required")
        self.lib = C.CDLL(str(Path(home) / "build/libneuronte.so"))
        self.lib.neuronte_open.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(C.c_void_p)]
        self.lib.neuronte_coverage.argtypes = [C.c_void_p, C.c_uint32, C.c_int, C.c_int, C.c_int, C.POINTER(C.c_float)]
        self.lib.neuronte_close.argtypes = [C.c_void_p]
        self.split = (Path(home) / "weights/neuronte-split-v5.nrt").read_bytes()

    def coverage(self, image, code, w, h):
        handle, buf = C.c_void_p(), (C.c_float * (w * h))()
        data = C.create_string_buffer(image, len(image))
        if self.lib.neuronte_open(data, len(image), C.byref(handle)) != 0:
            return None
        try:
            self.assertEqual(self.lib.neuronte_coverage(handle, code, w, h, 0, buf), 0)
        finally:
            self.lib.neuronte_close(handle)
        return np.frombuffer(buf, np.float32).reshape(h, w)

    def test_split_v5_braille_and_a_tiled_wrapper_of_it(self):
        packed = Packed(self.split)
        for code in (0x2800, 0x2847, 0x28FF):
            for cell in ((8, 16), (25, 49)):
                got = self.coverage(self.split, code, *cell)
                self.assertLess(np.abs(got - packed.coverage(code, *cell)).max(), 1e-3)
        # The Braille member, its bytes unchanged, inside a version 6 file.
        nets, ranges, gate = struct.unpack_from("<3I", self.split, 12)[0], struct.unpack_from("<I", self.split, 16)[0], struct.unpack_from("<I", self.split, 24)[0]
        lengths = struct.unpack_from(f"<{nets}I", self.split, 28)
        off = 28 + 4 * nets + 12 * ranges + 4 * (43 + gate * 65 + nets * gate + nets) + sum(lengths[:5])
        member = self.split[off:off + lengths[5]]
        wrapped = tiled_image([0, .5, 1], [0, 1, 2], [(0, 0xFF, 0x2800, 0)] * 4, [member])
        if self.coverage(wrapped, 0x2800, 2, 4) is None:
            raise unittest.SkipTest("this runtime predates version 6 tiled nets")
        for code in (0x2800, 0x2847, 0x28FF):
            for cell in ((8, 16), (23, 47)):
                got = self.coverage(wrapped, code, *cell)
                self.assertLess(np.abs(got - Packed(wrapped).coverage(code, *cell)).max(), 1e-3)


if __name__ == "__main__":
    unittest.main()
