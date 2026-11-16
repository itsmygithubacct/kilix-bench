#!/usr/bin/env python3
"""One pane of the side-by-side test: a shell whose text is drawn by one face.

usage: pane.py v2|n014|dejavu [--stream FILE] [--cell WxH] [-- COMMAND...]

The pane runs a shell (bash, in this folder, unless a command is given) and
draws what the shell prints as pixels with its own face, shown with the
Kitty graphics protocol at the pixel size they were drawn.

Every pane follows the same stream file. A key typed in any pane is appended
to the stream and each pane passes the stream to its own shell, so each
character sent reaches every shell and all of them run the same commands.

Keys, which act on every pane:
  PgUp, PgDn   scroll             F2   next cell size
  F3           mark characters a face does not have, on and off
  F4           clear what is shown
  exit         typed at the prompt, closes all the panes

The shell sees TERM=dumb, so it prints plain text and full-screen programs
do not work in it.
"""
import base64
import codecs
import fcntl
import os
import pty
import select
import signal
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
# Bytes on the stream that are for the panes and never reach a shell.
UP, SIZE, MARK, CLEAR, DOWN = "\x1c", "\x1d", "\x1e", "\x1f", "\x00"
COMMANDS = (UP + SIZE + MARK + CLEAR + DOWN).encode()


def stream_path(arg=None):
    if arg:
        return arg
    base = os.environ.get("XDG_RUNTIME_DIR") or os.path.join(os.path.expanduser("~"), ".cache")
    d = os.path.join(base, "kilix-bench-sidebyside")
    os.makedirs(d, mode=0o700, exist_ok=True)
    return os.path.join(d, os.environ.get("SIDEBYSIDE_SESSION", "default") + ".stream")


class Screen:
    """What a shell has printed, kept as whole lines and wrapped when drawn."""

    KEEP = 5000

    def __init__(self, size=1):
        self.lines, self.col, self.size, self.marked, self.scroll = [[]], 0, size, False, 0
        self.pending = ""

    def command(self, ch):
        if ch == SIZE:
            self.size = (self.size + 1) % len(SIZES)
        elif ch == MARK:
            self.marked = not self.marked
        elif ch == UP:
            self.scroll += 1
        elif ch == DOWN:
            self.scroll = max(0, self.scroll - 1)
        elif ch == CLEAR:
            self.lines, self.scroll = [self.lines[-1]], 0

    def _escape(self, seq):
        """Act on one escape sequence; the few a plain shell sends."""
        if not seq.startswith("\x1b["):
            return
        final, body = seq[-1], seq[2:-1]
        n = int(body) if body.isdigit() else 1
        line = self.lines[-1]
        if final == "K":
            del line[self.col:]
        elif final == "C":
            self.col += n
        elif final == "D":
            self.col = max(0, self.col - n)
        elif final == "G":
            self.col = max(0, n - 1)
        elif final == "P":
            del line[self.col:self.col + n]
        elif final == "@":
            line[self.col:self.col] = [" "] * n
        elif final == "J" and body in ("2", "3"):
            self.lines, self.col = [[]], 0

    def feed(self, text):
        """Take what the shell printed."""
        text, self.pending = self.pending + text, ""
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "\x1b":
                j = i + 1
                if j >= len(text):
                    self.pending = text[i:]
                    return
                if text[j] == "[":
                    j += 1
                    while j < len(text) and not "@" <= text[j] <= "~":
                        j += 1
                elif text[j] == "]":
                    while j < len(text) and text[j] != "\x07" and text[j - 1:j + 1] != "\x1b\\":
                        j += 1
                if j >= len(text):
                    self.pending = text[i:]
                    return
                self._escape(text[i:j + 1])
                i = j + 1
                continue
            i += 1
            line = self.lines[-1]
            if ch == "\n":
                self.lines.append([])
                self.col = 0
                del self.lines[:-self.KEEP]
            elif ch == "\r":
                self.col = 0
            elif ch == "\b":
                self.col = max(0, self.col - 1)
            elif ch == "\t":
                self._put(" " * (8 - self.col % 8))
            elif ord(ch) >= 0x20 and ch != "\x7f":
                self._put(ch)
            self.scroll = 0

    def _put(self, chars):
        line = self.lines[-1]
        if self.col > len(line):
            line.extend(" " * (self.col - len(line)))
        line[self.col:self.col + len(chars)] = list(chars)
        self.col += len(chars)

    def rows(self, columns, count):
        """The wrapped rows to show and the cursor's place among them."""
        wrapped = []
        for line in self.lines:
            for i in range(0, max(len(line), 1), columns):
                wrapped.append(line[i:i + columns])
        last = len(wrapped) - 1 - (max(len(self.lines[-1]), 1) - 1) // columns + self.col // columns
        if last >= len(wrapped):
            wrapped.append([])
        half = max(1, count // 2)
        self.scroll = min(self.scroll, max(0, (len(wrapped) - count + half - 1) // half))
        end = len(wrapped) - self.scroll * half
        start = max(0, end - count)
        cursor = (last - start, self.col % columns) if self.scroll == 0 else None
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


KEYS = {b"\x1b[5~": UP, b"\x1b[6~": DOWN, b"\x1b[12~": SIZE, b"\x1bOQ": SIZE,
        b"\x1b[13~": MARK, b"\x1bOR": MARK, b"\x1b[14~": CLEAR, b"\x1bOS": CLEAR}


def keys(data):
    """Turn what the keyboard sent into what goes on the stream: the pane's
    own keys become its command bytes and everything else passes to the shells."""
    for seq, command in KEYS.items():
        data = data.replace(seq, command.encode())
    return data


def split(data):
    """Part a piece of the stream into runs for the shell and command bytes."""
    out, run = [], bytearray()
    for b in data:
        if b in COMMANDS:
            if run:
                out.append(bytes(run))
                run = bytearray()
            out.append(chr(b))
        else:
            run.append(b)
    if run:
        out.append(bytes(run))
    return out


def shell(argv, here):
    """Start the pane's shell on a pseudo-terminal; its process id and the terminal."""
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(here)
        env = dict(os.environ, TERM="dumb", PS1="\\$ ", SIDEBYSIDE_PANE="1")
        for name in ("PROMPT_COMMAND", "KITTY_WINDOW_ID"):
            env.pop(name, None)
        command = argv or ["bash", "--norc", "--noprofile", "-i"]
        try:
            os.execvpe(command[0], command, env)
        finally:
            os._exit(127)
    return pid, fd


def main(argv):
    if not argv or argv[0] not in faces.ORDER:
        sys.stderr.write(__doc__)
        return 2
    command = []
    if "--" in argv:
        command, argv = argv[argv.index("--") + 1:], argv[:argv.index("--")]
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
    pid, child = shell(command, os.path.dirname(os.path.abspath(__file__)))
    decode = codecs.getincrementaldecoder("utf-8")("replace")
    cache, out, shown, grid = {}, sys.stdout.buffer, None, None
    interactive = sys.stdin.isatty()
    saved = termios.tcgetattr(0) if interactive else None
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        if interactive:
            tty.setraw(0)
        out.write(b"\x1b[?25l\x1b[2J")
        while True:
            for piece in split(follow.read()):
                if isinstance(piece, str):
                    screen.command(piece)
                else:
                    os.write(child, piece)
            rows_t, cols_t, xp, yp = window()
            w, h = SIZES[screen.size]
            top = -(-yp // rows_t)                      # one terminal row for the title
            columns, count = max(1, xp // w), max(1, (yp - top) // h)
            if grid != (columns, count):
                grid = (columns, count)
                fcntl.ioctl(child, termios.TIOCSWINSZ, struct.pack("HHHH", count, columns, columns * w, count * h))
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
            ready, _, _ = select.select(([0] if interactive else []) + [child], [], [], 0.04)
            if child in ready:
                try:
                    data = os.read(child, 65536)
                except OSError:
                    data = b""
                if not data:
                    return 0
                screen.feed(decode.decode(data))
            if 0 in ready:
                typed = keys(os.read(0, 4096))
                if typed:
                    os.write(append, typed)
    finally:
        if saved:
            termios.tcsetattr(0, termios.TCSADRAIN, saved)
        out.write(b"\x1b_Ga=d,d=I,i=1,q=2\x1b\\\x1b[?25h\x1b[2J\x1b[H")
        out.flush()
        try:
            os.kill(pid, signal.SIGHUP)
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
