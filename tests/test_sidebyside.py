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
        s.feed("abcdef\r\ngh")
        rows, cursor = s.rows(4, 10)
        self.assertEqual(["".join(r) for r in rows], ["abcd", "ef", "gh"])
        self.assertEqual(cursor, (2, 2))

    def test_what_a_plain_shell_sends(self):
        s = pane.Screen()
        s.feed("$ abc\b\b\x1b[K")                      # two characters rubbed out
        self.assertEqual("".join(s.lines[-1]), "$ a")
        s.feed("\rxy\x1b]0;title\x07\x1b[1mz\x1b[0m")   # overwrite, a title, bold
        self.assertEqual("".join(s.lines[-1]), "xyz")
        s.feed("\x1b[")                                  # a sequence cut in two
        s.feed("2Jnew")
        self.assertEqual(["".join(l) for l in s.lines], ["new"])

    def test_commands_change_the_view_and_not_the_text(self):
        s = pane.Screen()
        s.feed("one\r\ntwo")
        s.command(pane.SIZE)
        s.command(pane.MARK)
        self.assertEqual((s.size, s.marked, len(s.lines)), (2, True, 2))
        s.command(pane.CLEAR)
        self.assertEqual(["".join(l) for l in s.lines], ["two"])

    def test_scrolling_stops_at_the_top_and_output_returns(self):
        s = pane.Screen()
        s.feed("\r\n".join(str(i) for i in range(30)))
        for _ in range(50):
            s.command(pane.UP)
        rows, cursor = s.rows(10, 10)
        self.assertEqual("".join(rows[0]), "0")
        self.assertIsNone(cursor)
        s.feed("x")
        rows, cursor = s.rows(10, 10)
        self.assertEqual("".join(rows[-1]), "29x")


class KeysTest(unittest.TestCase):
    def test_pane_keys_become_commands_and_the_rest_passes(self):
        self.assertEqual(pane.keys(b"ls\r\x03\x1b[A"), b"ls\r\x03\x1b[A")
        self.assertEqual(pane.keys(b"\x1b[5~\x1b[6~\x1bOQ"), (pane.UP + pane.DOWN + pane.SIZE).encode())
        self.assertEqual(pane.split(b"ab" + pane.MARK.encode() + "\u2713".encode()),
                         [b"ab", pane.MARK, "\u2713".encode()])


class ShellTest(unittest.TestCase):
    def test_a_shell_runs_and_its_output_is_read(self):
        import select
        import tempfile
        with tempfile.TemporaryDirectory() as here:
            pid, fd = pane.shell(["sh", "-c", "printf 'in %s' \"$TERM\"; pwd"], here)
            s, data = pane.Screen(), b""
            while select.select([fd], [], [], 5)[0]:
                try:
                    piece = os.read(fd, 4096)
                except OSError:
                    break
                if not piece:
                    break
                data += piece
            os.waitpid(pid, 0)
            s.feed(data.decode())
            self.assertEqual("".join(s.lines[0]), "in dumb" + os.path.realpath(here))


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
