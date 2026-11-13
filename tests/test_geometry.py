"""Geometry fixtures describe seams and design deltas without ranking them."""

import tempfile
import unittest
from pathlib import Path

from kilix_bench.font_geometry import (
    analyze, design_comparison, edges_from_mask, insets, seam_report, write_pair, SEAM_GRID, BLOCK_GRID,
)


def mask(width, height, paint):
    data = bytearray(width * height)
    for x, y in paint:
        data[y * width + x] = 255
    return bytes(data)


def glyph(scalar, edges, ink, clipped=0, source=None, placed=None):
    return {
        "scalar": scalar, "ink": ink, "clipped": clipped, "edges": edges,
        "source": source, "placed": placed, "advance": 8, "outline": True,
    }


class GeometryTests(unittest.TestCase):
    def test_block_fixture_joins_overlap_in_the_ideal_shapes(self):
        for width, height in ((2, 2), (4, 6), (8, 16)):
            with self.subTest(cell=(width, height)):
                regions = {
                    0x2588: lambda x, y: True,
                    0x2580: lambda x, y: y < height // 2,
                    0x2584: lambda x, y: y >= height // 2,
                    0x258C: lambda x, y: x < width // 2,
                    0x2590: lambda x, y: x >= width // 2,
                }
                masks = {scalar: mask(width, height, [(x, y) for y in range(height)
                         for x in range(width) if inside(x, y)])
                         for scalar, inside in regions.items()}
                edges = {scalar: edges_from_mask(data, width, height)
                         for scalar, data in masks.items()}
                result = seam_report(BLOCK_GRID, edges, masks, (width, height))
                self.assertGreater(result["checked_joins"], 10)
                self.assertEqual(result["connected_joins"], result["checked_joins"])
                self.assertEqual(result["unchecked_joins"], 0)
                self.assertEqual(result["gap_count"], 0)

    def test_edges_insets_and_expected_joins(self):
        width, height = 4, 4
        full = mask(width, height, [(x, y) for x in range(width) for y in range(height)])
        self.assertEqual(edges_from_mask(full, width, height), "LRTB")
        self.assertEqual(insets(full, width, height), {"L": 0, "R": 0, "T": 0, "B": 0})
        gap = mask(width, height, [(1, 0)])
        self.assertNotIn("L", edges_from_mask(gap, width, height))
        # Edge letters alone are not a connectivity measurement.
        edges = {
            0x2500: edges_from_mask(bytes([255] * 4 + [0] * 12), 4, 4),
            0x2501: edges_from_mask(bytes([0] * 12 + [255] * 4), 4, 4),
        }
        self.assertIn("L", edges[0x2500])
        self.assertIn("R", edges[0x2500])
        self.assertIn("L", edges[0x2501])
        self.assertIn("R", edges[0x2501])
        blind = seam_report(((0x2500, 0x2501),), edges)
        self.assertEqual(blind["gap_count"], 0)
        self.assertEqual(blind["connected_joins"], 0)
        self.assertEqual(blind["checked_joins"], 0)
        self.assertGreater(blind["unchecked_joins"], 0)
        self.assertIsNone(blind["quality_rank"])
        self.assertIsNone(blind["similarity_winner"])

    def test_misaligned_edge_contact_is_a_gap_and_a_missing_glyph_is_not(self):
        left = bytes([255] * 4 + [0] * 12)
        right = bytes([0] * 12 + [255] * 4)
        edges = {
            0x2500: edges_from_mask(left, 4, 4),
            0x2501: edges_from_mask(right, 4, 4),
        }
        grid = ((0x2500, 0x2501),)
        report = seam_report(grid, edges, masks={0x2500: left, 0x2501: right}, cell=(4, 4))
        self.assertEqual(report["fixture"], "custom")
        self.assertEqual(report["checked_joins"], 1)
        self.assertEqual(report["edge_contact_joins"], 1)
        self.assertEqual(report["connected_joins"], 0)
        self.assertEqual(report["gap_count"], 1)
        self.assertEqual(report["gaps"], [{"at": [0, 0], "side": "R", "scalars": [0x2500, 0x2501]}])
        self.assertEqual(report["unchecked_joins"], 0)
        self.assertEqual(report["missing_scalars"], [])
        self.assertIsNone(report["quality_rank"])
        self.assertIsNone(report["similarity_winner"])
        self.assertIn("not quality parity", report["note"])
        missing = seam_report(grid, edges, masks={0x2500: left}, cell=(4, 4))
        self.assertEqual(missing["gap_count"], 0)
        self.assertEqual(missing["checked_joins"], 0)
        self.assertEqual(missing["connected_joins"], 0)
        self.assertEqual(missing["unchecked_joins"], 1)
        self.assertIn(0x2501, missing["missing_scalars"])
        self.assertNotIn(0x2500, missing["missing_scalars"])
        aligned = bytes([255] * 4 + [0] * 12)
        connected = seam_report(
            grid, edges, masks={0x2500: aligned, 0x2501: aligned}, cell=(4, 4))
        self.assertEqual(connected["connected_joins"], 1)
        self.assertEqual(connected["gap_count"], 0)
        self.assertIsNone(connected["quality_rank"])
        self.assertIsNone(connected["similarity_winner"])
        column = bytearray(16)
        other = bytearray(16)
        for y in range(4):
            column[y * 4] = 255
            other[y * 4 + 3] = 255
        vertical = seam_report(
            ((0x2502,), (0x2503,)),
            {0x2502: "TB", 0x2503: "TB"},
            masks={0x2502: bytes(column), 0x2503: bytes(other)},
            cell=(4, 4))
        self.assertEqual(vertical["edge_contact_joins"], 1)
        self.assertEqual(vertical["connected_joins"], 0)
        self.assertEqual(vertical["gap_count"], 1)
        self.assertEqual(vertical["gaps"][0]["side"], "B")
        with self.assertRaisesRegex(ValueError, "cell size"):
            seam_report(grid, edges, masks={0x2500: left, 0x2501: right})
        with self.assertRaisesRegex(ValueError, "mask size"):
            seam_report(grid, edges, masks={0x2500: left, 0x2501: right[:8]}, cell=(4, 4))

    def test_equal_and_unequal_masks_have_no_winner(self):
        same = design_comparison(b"\x00\xff", b"\x00\xff")
        self.assertTrue(same["pixel_equivalent"])
        self.assertIsNone(same["quality_rank"])
        self.assertIsNone(same["similarity_winner"])
        self.assertNotIn("ssim", same["note"].lower())
        self.assertIn("quality parity", same["note"])
        self.assertIn("not a quality ranking", same["note"])
        delta = design_comparison(b"\x00\xff", b"\xff\x00")
        self.assertFalse(delta["pixel_equivalent"])
        self.assertEqual(delta["design_delta_samples"], 2)
        self.assertIsNone(delta["quality_rank"])
        self.assertIsNone(delta["similarity_winner"])
        mismatch = design_comparison(b"\x00", b"\x00\x00")
        self.assertIsNone(mismatch["quality_rank"])
        self.assertIsNone(mismatch["similarity_winner"])

    def test_analyze_checks_worker_edges_and_exports_an_unranked_pair(self):
        cell = (4, 3)
        scalars = [0x2500, 0x2502]
        left = mask(*cell, [(0, 1), (1, 1), (2, 1), (3, 1)])
        right = mask(*cell, [(0, 1), (3, 1)])
        vertical = mask(*cell, [(1, 0), (1, 1), (1, 2)])
        blank = bytes(12)
        by_engine = {
            "neuronte": {
                "masks": left + vertical,
                "glyphs": [
                    glyph(0x2500, "LR", 4, source=[0, 1, 3, 1], placed=[0, 1, 3, 1]),
                    glyph(0x2502, "TB", 3, source=[1, 0, 1, 2], placed=[1, 0, 1, 2]),
                ],
            },
            "freetype": {
                "masks": right + blank,
                "glyphs": [
                    glyph(0x2500, "LR", 2, clipped=4, source=[-1, 1, 3, 1], placed=[0, 1, 3, 1]),
                    glyph(0x2502, "", 0, clipped=2, source=[1, -1, 1, 3], placed=None),
                ],
            },
        }
        summary, rows, _observed = analyze("box", cell, scalars, by_engine)
        self.assertEqual(summary["engines"]["freetype"]["clipped_ink_samples"], 6)
        self.assertEqual(summary["engines"]["freetype"]["glyphs_with_source_overhang"], 2)
        self.assertIsNone(summary["comparison"]["quality_rank"])
        self.assertFalse(summary["comparison"]["pixel_equivalent"])
        self.assertIn("quality parity", summary["comparison"]["note"])
        self.assertIn("not an error", summary["comparison"]["note"])
        seam = summary["engines"]["neuronte"]["seam"]
        self.assertEqual(seam["gap_count"], 0)
        self.assertEqual(seam["connected_joins"], 0)
        self.assertGreater(seam["unchecked_joins"], 0)
        self.assertIsNone(seam["quality_rank"])
        self.assertTrue(seam["missing_scalars"])
        self.assertEqual(len(rows), 4)
        by_engine["freetype"]["glyphs"][0]["edges"] = "L"
        with self.assertRaisesRegex(ValueError, "edge flags"):
            analyze("box", cell, scalars, by_engine)

    def test_fixture_images_are_private_and_do_not_rank_pixels(self):
        cell = (2, 2)
        scalars = [0x2500, 0x2502, 0x250C, 0x2510, 0x2514, 0x2518, 0x251C, 0x2524,
                   0x252C, 0x2534, 0x253C, 0x20]
        one = bytes([255, 255, 0, 0])
        other = bytes([255, 0, 0, 255])
        masks = {scalar: one for scalar in scalars}
        other_masks = {scalar: other for scalar in scalars}
        # write_pair looks up masks in corpus order, not grid order.
        flat = b"".join(masks[scalar] for scalar in scalars)
        flat_other = b"".join(other_masks[scalar] for scalar in scalars)
        with tempfile.TemporaryDirectory() as temporary:
            names = write_pair(Path(temporary), "box-2x2", cell, scalars, {
                "neuronte": {"masks": flat},
                "freetype": {"masks": flat_other},
            })
            self.assertIn("box-2x2-design-delta.png", names)
            for name in names:
                path = Path(temporary) / name
                self.assertEqual(path.stat().st_mode & 0o077, 0)
            self.assertEqual(write_pair(Path(temporary), "braille-2x2", cell, scalars, {
                "neuronte": {"masks": flat},
            }), [])


if __name__ == "__main__":
    unittest.main()
