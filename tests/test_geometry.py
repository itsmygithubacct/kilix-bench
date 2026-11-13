"""Geometry fixtures describe seams and design deltas without ranking them."""

import tempfile
import unittest
from pathlib import Path

from kilix_bench.font_geometry import (
    analyze, design_comparison, edges_from_mask, insets, seam_report, write_pair, SEAM_GRID,
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
    def test_edges_insets_and_expected_joins(self):
        width, height = 4, 4
        full = mask(width, height, [(x, y) for x in range(width) for y in range(height)])
        self.assertEqual(edges_from_mask(full, width, height), "LRTB")
        self.assertEqual(insets(full, width, height), {"L": 0, "R": 0, "T": 0, "B": 0})
        gap = mask(width, height, [(1, 0)])
        self.assertNotIn("L", edges_from_mask(gap, width, height))
        observed = {scalar: "LRTB" for row in SEAM_GRID for scalar in row}
        observed[0x20] = ""
        observed[0x2500] = "L"
        report = seam_report(SEAM_GRID, observed)
        self.assertGreater(report["checked_joins"], 0)
        self.assertGreater(report["gap_count"], 0)
        self.assertTrue(any(item["side"] == "R" for item in report["gaps"]))

    def test_equal_and_unequal_masks_have_no_winner(self):
        same = design_comparison(b"\x00\xff", b"\x00\xff")
        self.assertTrue(same["pixel_equivalent"])
        self.assertIsNone(same["quality_rank"])
        self.assertIsNone(same["similarity_winner"])
        self.assertNotIn("ssim", same["note"].lower())
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
