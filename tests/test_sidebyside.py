"""The side-by-side test's shared text, key handling and character listings.

Nothing here needs a Neuronte tree, a font or a terminal."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sidebyside"))

import charsets  # noqa: E402
import faces  # noqa: E402
import pane  # noqa: E402


class Flat:
    """A face that fills a cell with its character's number."""
    name, marked = "flat", False

    def cell(self, scalar, w, h):
        return np.full((h, w), scalar % 256, np.uint8)


class ScreenTest(unittest.TestCase):
    def test_text_wraps_and_the_cursor_follows(self):
        s = pane.Screen()
        s.feed("abcdef\ngh")
        rows, cursor = s.rows(4, 10)
        self.assertEqual(["".join(r) for r in rows], ["abcd", "ef", "gh"])
        self.assertEqual(cursor, (2, 2))

    def test_backspace_clear_size_and_mark(self):
        s = pane.Screen()
        s.feed("ab\n" + pane.ERASE + pane.ERASE)
        self.assertEqual(s.lines, [["a"]])
        s.feed(pane.SIZE + pane.MARK)
        self.assertEqual((s.size, s.marked), (2, True))
        s.feed(pane.CLEAR)
        self.assertEqual((s.lines, s.size, s.marked), ([[]], 2, True))

    def test_scrolling_stops_at_the_top_and_typing_returns(self):
        s = pane.Screen()
        s.feed("\n".join(str(i) for i in range(30)))
        s.feed(pane.UP * 50)
        rows, cursor = s.rows(10, 10)
        self.assertEqual("".join(rows[0]), "0")
        self.assertIsNone(cursor)
        s.feed("x")
        rows, cursor = s.rows(10, 10)
        self.assertEqual("".join(rows[-1]), "29x")

    def test_two_screens_fed_the_same_bytes_agree(self):
        a, b = pane.Screen(), pane.Screen()
        text = "one\ttwo\n✓ é" + pane.ERASE + pane.SIZE
        a.feed(text)
        for ch in text:
            b.feed(ch)
        self.assertEqual((a.lines, a.size), (b.lines, b.size))


class KeysTest(unittest.TestCase):
    def test_keys_become_stream_bytes(self):
        self.assertEqual(pane.keys(b"a\r\x08"), b"a\n\x7f")
        self.assertEqual(pane.keys(b"\x1b[5~\x1b[6~\x1b[A"), (pane.UP + pane.DOWN).encode())
        self.assertEqual(pane.keys("é✓".encode()), "é✓".encode())
        self.assertIsNone(pane.keys(b"ab\x03"))


class DrawTest(unittest.TestCase):
    def test_each_cell_lands_in_its_place(self):
        img = pane.draw(Flat(), {}, [list("A B")], None, 4, 2, 3, 5)
        self.assertEqual(img.shape, (10, 12, 3))
        self.assertTrue((img[:5, 3:6] == pane.BG.astype(np.uint8)).all())
        self.assertFalse((img[:5, 0:3] == img[:5, 6:9]).all())

    def test_the_picture_is_sent_whole(self):
        import base64
        import zlib
        img = np.random.default_rng(0).integers(0, 255, (40, 60, 3), dtype=np.uint8)
        data = pane.kitty_image(img)
        parts = [p.split(b";", 1) for p in data.split(b"\x1b\\") if p]
        self.assertIn(b"s=60,v=40", parts[0][0])
        self.assertTrue(parts[-1][0].endswith(b"m=0"))
        self.assertEqual(zlib.decompress(base64.standard_b64decode(b"".join(p[1] for p in parts))), img.tobytes())


class ListingTest(unittest.TestCase):
    def test_blocks_and_what_is_shown(self):
        self.assertEqual(charsets.block(0x41)[1], "Basic Latin")
        self.assertEqual(charsets.block(0x2713)[1], "Dingbats")
        self.assertFalse(charsets.shown(0x301))
        self.assertFalse(charsets.shown(0xA0))
        text = charsets.listing({0x41, 0x42, 0x301, 0x2713})
        self.assertEqual(text, "Basic Latin (2)\nA B\n\nDingbats (1)\n✓\n")

    def test_a_missing_tree_is_said_plainly(self):
        saved = os.environ.pop("NEURONTE_V2_HOME", None)
        try:
            with self.assertRaises(faces.Missing):
                faces.face("v2")
        finally:
            if saved is not None:
                os.environ["NEURONTE_V2_HOME"] = saved

    @unittest.skipUnless(os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"), "no DejaVu here")
    def test_the_font_map_reads_a_real_file(self):
        have = faces.cmap("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
        self.assertTrue({0x41, 0xE9, 0x2500, 0x2713} <= have)
        self.assertNotIn(0x4E2D, have)


if __name__ == "__main__":
    unittest.main()
