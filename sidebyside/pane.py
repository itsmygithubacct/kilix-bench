#!/usr/bin/env python3
"""One pane of the side-by-side test: draws the shared text with one face.

usage: pane.py v2|n014|dejavu [--stream FILE] [--cell WxH]

Every pane follows the same stream file. A key typed in any pane is appended
to the stream, never drawn directly, so each character sent reaches every
pane. The pane draws the text as pixels with its own face and shows them
with the Kitty graphics protocol, at the pixel size they were drawn.

Keys, which act on every pane:
  Enter, Backspace   as in a terminal        Ctrl-L   clear the text
  PgUp, PgDn         scroll                  Ctrl-T   next cell size
  Ctrl-U             mark characters a face does not have, on and off
  Ctrl-C             close this pane
"""
import base64
import codecs
import fcntl
import os
import select
import struct
import sys
import termios
import tty
import zlib

import numpy as np

import faces

SIZES = [(8, 16), (12, 24), (16, 32), (24, 48)]
FG = np.array([232, 232, 232], np.float32)
BG = np.array([16, 18, 22], np.float32)
CLEAR, SIZE, MARK, UP, DOWN, ERASE = "\x0c", "\x14", "\x15", "\x02", "\x06", "\x7f"


def stream_path(arg=None):
    if arg:
        return arg
    base = os.environ.get("XDG_RUNTIME_DIR") or os.path.join(os.path.expanduser("~"), ".cache")
    d = os.path.join(base, "kilix-bench-sidebyside")
    os.makedirs(d, mode=0o700, exist_ok=True)
    return os.path.join(d, os.environ.get("SIDEBYSIDE_SESSION", "default") + ".stream")


class Screen:
    """The shared text, as every pane understands it."""

    def __init__(self, size=1):
        self.lines, self.size, self.marked, self.scroll = [[]], size, False, 0

    def feed(self, text):
        for ch in text:
            if ch == CLEAR:
                self.lines, self.scroll = [[]], 0
            elif ch == SIZE:
                self.size = (self.size + 1) % len(SIZES)
            elif ch == MARK:
                self.marked = not self.marked
            elif ch == UP:
                self.scroll += 1
            elif ch == DOWN:
                self.scroll = max(0, self.scroll - 1)
            elif ch == ERASE:
                if self.lines[-1]:
                    self.lines[-1].pop()
                elif len(self.lines) > 1:
                    self.lines.pop()
            elif ch in "\r\n":
                self.lines.append([])
            elif ch == "\t":
                self.lines[-1].extend(" " * (8 - len(self.lines[-1]) % 8))
            elif ord(ch) >= 0x20:
                self.lines[-1].append(ch)
            if ch not in (UP, DOWN, SIZE, MARK):
                self.scroll = 0

    def rows(self, columns, count):
        """The wrapped rows to show, the cursor's place among them, and the scroll used."""
        wrapped = []
        for line in self.lines:
            for i in range(0, max(len(line), 1), columns):
                wrapped.append(line[i:i + columns])
        if len(self.lines[-1]) and len(self.lines[-1]) % columns == 0:
            wrapped.append([])
        half = max(1, count // 2)
        self.scroll = min(self.scroll, max(0, (len(wrapped) - count + half - 1) // half))
        end = len(wrapped) - self.scroll * half
        start = max(0, end - count)
        cursor = (len(wrapped) - 1 - start, len(wrapped[-1])) if self.scroll == 0 else None
        return wrapped[start:max(end, start)], cursor


def draw(face, cache, rows, cursor, columns, count, w, h):
    """The picture of the rows: height, width, 3 bytes of RGB."""
    cover = np.zeros((count * h, columns * w), np.uint8)
    for r, row in enumerate(rows[:count]):
        for c, ch in enumerate(row[:columns]):
            if ch == " ":
                continue
            key = (ord(ch), w, h, face.marked)
            if key not in cache:
                cache[key] = face.cell(ord(ch), w, h)
            cover[r * h:(r + 1) * h, c * w:(c + 1) * w] = cache[key]
    if cursor and cursor[0] < count and cursor[1] < columns:
        r, c = cursor
        cover[(r + 1) * h - max(2, h // 8):(r + 1) * h, c * w:(c + 1) * w] = 255
    a = cover.astype(np.float32)[..., None] / 255
    return (BG + (FG - BG) * a + 0.5).astype(np.uint8)


def kitty_image(rgb):
    """The escape codes that show the picture at the cursor, replacing the last one."""
    h, w = rgb.shape[:2]
    data = base64.standard_b64encode(zlib.compress(rgb.tobytes(), 1))
    out, first = [], True
    for i in range(0, len(data), 4096):
        more = 1 if i + 4096 < len(data) else 0
        head = f"a=T,f=24,o=z,s={w},v={h},i=1,p=1,q=2,C=1,m={more}" if first else f"m={more}"
        out.append(b"\x1b_G" + head.encode() + b";" + data[i:i + 4096] + b"\x1b\\")
        first = False
    return b"".join(out)


def window():
    rows, cols, xp, yp = struct.unpack("HHHH", fcntl.ioctl(1, termios.TIOCGWINSZ, b"\0" * 8))
    if not xp or not yp:
        xp, yp = cols * 10, rows * 20
    return rows, cols, xp, yp


def keys(data):
    """Turn what the keyboard sent into what goes on the stream; None closes the pane."""
    out, i = bytearray(), 0
    while i < len(data):
        b = data[i]
        if b in (3, 4):
            return None
        if b == 0x1B:
            seq = data[i:i + 8]
            if seq.startswith(b"\x1b[5~"):
                out += UP.encode()
            elif seq.startswith(b"\x1b[6~"):
                out += DOWN.encode()
            j = i + 1
            if j < len(data) and data[j] in b"[O":
                j += 1
                while j < len(data) and not 0x40 <= data[j] <= 0x7E:
                    j += 1
            i = j + 1
            continue
        if b == 8:
            b = 0x7F
        if b == 13:
            b = 10
        if b >= 0x20 or b in (9, 10, 12, 0x14, 0x15):
            out.append(b)
        i += 1
    return bytes(out)


def main(argv):
    if not argv or argv[0] not in faces.ORDER:
        sys.stderr.write(__doc__)
        return 2
    path = stream_path(argv[argv.index("--stream") + 1] if "--stream" in argv else None)
    screen = Screen()
    if "--cell" in argv:
        want = tuple(int(v) for v in argv[argv.index("--cell") + 1].lower().split("x"))
        screen.size = SIZES.index(want) if want in SIZES else 1
    try:
        face = faces.face(argv[0])
    except faces.Missing as e:
        sys.stderr.write(f"pane: {e}\n")
        try:
            input("Enter closes this pane. ")
        except (EOFError, KeyboardInterrupt):
            pass
        return 1
    append = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    follow = open(path, "rb")
    decode = codecs.getincrementaldecoder("utf-8")("replace")
    cache, out, shown = {}, sys.stdout.buffer, None
    interactive = sys.stdin.isatty()
    saved = termios.tcgetattr(0) if interactive else None
    try:
        if interactive:
            tty.setraw(0)
        out.write(b"\x1b[?25l\x1b[2J")
        while True:
            data = follow.read()
            if data:
                screen.feed(decode.decode(data))
            rows_t, cols_t, xp, yp = window()
            w, h = SIZES[screen.size]
            top = -(-yp // rows_t)                      # one terminal row for the title
            columns, count = max(1, xp // w), max(1, (yp - top) // h)
            if face.marked != screen.marked:
                face.mark(screen.marked)
            rows, cursor = screen.rows(columns, count)
            state = (tuple(map(tuple, rows)), cursor, columns, count, w, h, face.marked)
            if state != shown:
                shown = state
                title = f" {face.name}  {w}x{h}  {columns}x{count}" + ("  marks on" if face.marked else "")
                out.write(b"\x1b[H\x1b[7m" + title[:cols_t].ljust(cols_t).encode() + b"\x1b[0m\x1b[2;1H")
                out.write(kitty_image(draw(face, cache, rows, cursor, columns, count, w, h)))
                out.flush()
            ready, _, _ = select.select([0] if interactive else [], [], [], 0.04)
            if ready:
                typed = keys(os.read(0, 4096))
                if typed is None:
                    return 0
                if typed:
                    os.write(append, typed)
    except KeyboardInterrupt:
        return 0
    finally:
        if saved:
            termios.tcsetattr(0, termios.TCSADRAIN, saved)
        out.write(b"\x1b_Ga=d,d=I,i=1,q=2\x1b\\\x1b[?25h\x1b[2J\x1b[H")
        out.flush()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
