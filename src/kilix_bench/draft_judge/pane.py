"""Present the monitor in the current Kilix pane and read its keys.

Kitty graphics come from the Kilix frame presenter. Keyboard and mouse
come from the terminal: SGR mouse clicks are mapped back onto the canvas.
"""

import fcntl
import importlib.util
import os
from pathlib import Path
import select
import struct
import sys
import termios
import tty

from kilix_bench.draft_judge.display import HEIGHT, WIDTH


def _kilix_roots():
    roots = []
    forced = os.environ.get("KILIX_HOME", "").strip()
    if forced:
        roots.append(Path(forced))
    roots.append(Path.home() / "gpu_terminal" / "kilix")
    roots.append(Path.home() / "gpu_terminal" / "kitty-frame-presenter-rc5")
    return roots


def presenter_class():
    """The Kilix FramePresenter, which places RGB frames in this pane."""
    for root in _kilix_roots():
        source = root / "third_party" / "kitty-frame-presenter" / "src"
        if not source.is_dir() and (root / "src" / "kitty_frame_presenter").is_dir():
            source = root / "src"
        gfx = root / "config" / "gfx.py"
        if source.is_dir() and str(source) not in sys.path:
            sys.path.insert(0, str(source))
        if gfx.is_file():
            spec = importlib.util.spec_from_file_location("kilix_gfx_presenter", gfx)
            module = importlib.util.module_from_spec(spec)
            module.__file__ = str(gfx)
            spec.loader.exec_module(module)
            return module.FramePresenter
    from kitty_frame_presenter import FramePresenter
    return FramePresenter


class StdoutWriter:
    def write(self, data):
        if isinstance(data, str):
            data = data.encode("latin-1", "replace")
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()


def cell_to_pixel(col, row, cols, rows, width=WIDTH, height=HEIGHT):
    """Map a 1-based terminal cell to a pixel in the stretched canvas."""
    cols = max(1, cols)
    rows = max(1, rows)
    x = int((col - 0.5) * width / cols)
    y = int((row - 0.5) * height / rows)
    return max(0, min(width - 1, x)), max(0, min(height - 1, y))


def _kitty_key(code):
    named = {13: "enter", 27: "esc", 127: "backspace", 8: "backspace"}
    if code in named:
        return named[code]
    if 32 <= code < 127:
        return chr(code)
    return ""


def decode(data, pending=b""):
    """Turn raw terminal bytes into key and click events.

    A trailing incomplete escape stays in the returned remainder.
    """
    buf = pending + data
    events = []
    index = 0
    limit = len(buf)
    while index < limit:
        byte = buf[index]
        if byte == 0x1B:
            if index + 1 >= limit:
                break
            if buf[index + 1] != ord("["):
                events.append(("key", "esc"))
                index += 1
                continue
            cursor = index + 2
            if cursor < limit and buf[cursor] == ord("<"):
                end = cursor + 1
                while end < limit and buf[end] not in (ord("M"), ord("m")):
                    if buf[end] < 0x20 or buf[end] > 0x7E:
                        break
                    end += 1
                if end >= limit or buf[end] not in (ord("M"), ord("m")):
                    break
                if buf[end] == ord("M"):
                    parts = buf[cursor + 1:end].split(b";")
                    if len(parts) == 3:
                        try:
                            _button, col, row = (int(part) for part in parts)
                        except ValueError:
                            col = row = 0
                        if col and row:
                            events.append(("click", col, row))
                index = end + 1
                continue
            end = cursor
            while end < limit and not 0x40 <= buf[end] <= 0x7E:
                end += 1
            if end >= limit:
                break
            final = buf[end]
            body = buf[cursor:end]
            index = end + 1
            if final == ord("A"):
                events.append(("key", "up"))
            elif final == ord("B"):
                events.append(("key", "down"))
            elif final == ord("C"):
                events.append(("key", "right"))
            elif final == ord("D"):
                events.append(("key", "left"))
            elif final == ord("~") and body == b"3":
                events.append(("key", "backspace"))
            elif final == ord("u"):
                head = body.split(b";", 1)[0]
                try:
                    name = _kitty_key(int(head))
                except ValueError:
                    name = ""
                if name:
                    events.append(("key", name))
            continue
        if byte >= 128:
            taken = False
            for length in (2, 3, 4):
                chunk = buf[index:index + length]
                if len(chunk) < length:
                    return events, buf[index:]
                try:
                    text = chunk.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                events.append(("key", text))
                index += length
                taken = True
                break
            if not taken:
                index += 1
            continue
        if byte in (13, 10):
            events.append(("key", "enter"))
        elif byte in (8, 127):
            events.append(("key", "backspace"))
        elif 32 <= byte < 127:
            events.append(("key", chr(byte)))
        index += 1
    return events, buf[index:]


class Session:
    """Raw keyboard and SGR mouse on the alternate screen."""

    def __init__(self):
        self.fd = None
        self.old = None
        self.pending = b""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setraw(self.fd, termios.TCSANOW)
        attrs = termios.tcgetattr(self.fd)
        attrs[6][termios.VMIN] = 0
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, attrs)
        sys.stdout.buffer.write(b"\x1b[?1049h\x1b[?25l\x1b[?1000h\x1b[?1006h")
        sys.stdout.buffer.flush()
        return self

    def __exit__(self, *_exc):
        try:
            sys.stdout.buffer.write(b"\x1b[?1006l\x1b[?1000l\x1b[?25h\x1b[?1049l\x1b_Ga=d,d=A\x1b\\")
            sys.stdout.buffer.flush()
        except OSError:
            pass
        if self.old is not None and self.fd is not None:
            termios.tcsetattr(self.fd, termios.TCSANOW, self.old)
        return False

    def size(self):
        packed = fcntl.ioctl(sys.stdout.fileno(), termios.TIOCGWINSZ, b"\0" * 8)
        rows, cols, _xp, _yp = struct.unpack("HHHH", packed)
        return cols or 80, rows or 24

    def read_events(self, timeout):
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            if self.pending == b"\x1b":
                self.pending = b""
                return [("key", "esc")]
            return []
        data = os.read(self.fd, 256)
        events, self.pending = decode(data, self.pending)
        return events


def present_loop(monitor, face):
    try:
        cls = presenter_class()
    except Exception:
        sys.stderr.write("draft-judge: the Kilix frame presenter is not available; use --cli\n")
        return 2
    writer = StdoutWriter()
    code = 2
    try:
        with Session() as session:
            presenter = cls(writer, image_id=41, max_fps=12)
            try:
                while not monitor.quit:
                    monitor.poll()
                    canvas = monitor.render(face)
                    cols, rows = session.size()
                    presenter.present(
                        canvas.rgb(), canvas.w, canvas.h, max(1, cols), max(1, rows))
                    for event in session.read_events(0.05):
                        if event[0] == "click":
                            x, y = cell_to_pixel(
                                event[1], event[2], cols, rows, canvas.w, canvas.h)
                            monitor.handle(("click", x, y), canvas)
                        else:
                            monitor.handle(event, canvas)
                        if monitor.quit:
                            break
            finally:
                presenter.close(discard=True)
        code = monitor.exit_code
    except Exception as exc:
        sys.stderr.write(f"draft-judge: {exc}\n")
        code = 2
    finally:
        if monitor.thread is not None:
            monitor.thread.join()
    if monitor.directory:
        report = monitor.report or ""
        sys.stdout.write(report)
        if report and not report.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.write(f"{monitor.directory}\n")
    return code
