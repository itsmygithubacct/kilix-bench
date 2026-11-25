"""A full-canvas monitor for one draft-judge run.

The layout follows the puzzllm bench: a dark panel, a status lamp, and a
dock of buttons along the bottom. Neuronte draws the glyphs when its
library and a weight file open. Otherwise a built-in 5x7 face is used.
The frame itself is presented by pane.py.
"""

import argparse
import array
import base64
import ctypes
import os
from pathlib import Path
import queue
import sys
import threading

from kilix_bench.draft_judge.tui import (
    existing_file, looks_explicit_path, match_models, split_device,
    strip_wrapping_quotes,
)
from kilix_bench.draft_judge.workflow import ModelSpec, discover_models

WIDTH = 1280
HEIGHT = 720
PAD = 26
DOCK_TOP = HEIGHT - 95

BG = 0x101821
PANEL = 0x192530
LINE = 0x344753
INK = 0xEBF1EF
MUTED = 0xA4B8BE
AQUA = 0x70DFD3
GOLD = 0xF6D388
RED = 0xFFA692
BUTTON = 0x263743
SELECT = 0x30434F

# Original 5x7, one byte per row, five low bits, left bit is 0x10.
_FONT = base64.b64decode(
    "AAAAAAAAAAQEBAQABAAKCgAAAAAACh8KCh8KAAQPFA4FHgQYGQIECBMDDBIUCBUSDQQE"
    "AAAAAAACBAgICAQCCAQCAgIECAAKBB8ECgAABAQfBAQAAAAABAQIAAAAAB8AAAAAAAAA"
    "BAQAAQICBAgIEA4RExUZEQ4EDAQEBAQODhEBBggQHw4RAQYBEQ4CBgoSHwICHxAeAQER"
    "Dg4QEB4REQ4fAQIECAgIDhERDhERDg4REQ8BAQ4ABAQABAQAAAQEAAQIAAIECBAIBAIA"
    "AB8AHwAACAQCAQIECA4RAQYEAAQOERcVFxAODhERHxERER4RER4RER4OERAQEBEOHhER"
    "ERERHh8QEB4QEB8fEBAeEBAQDhEQFxERDhERER8REREOBAQEBAQOBwICAgISDBESFBgU"
    "EhEQEBAQEBAfERsVFREREREZFRMREREOEREREREOHhERHhAQEA4REREVEg0eEREeFBIR"
    "DxAQDgEBHh8EBAQEBAQREREREREOEREREREKBBERERUVGxEREQoEChEREREKBAQEBB8B"
    "AgQIEB8OCAgICAgOEAgIBAICAQ4CAgICAg4EChEAAAAAAAAAAAAAHwgEAAAAAAAAAA4B"
    "DxEPEBAeERERHgAADhAQEA4BAQ8REREPAAAOER8QDgYICBwICAgAAA8RDwEOEBAeERER"
    "EQQABAQEBA4CAAICAhIMEBASFBgUEgQEBAQEBA4AABoVFRERAAAeEREREQAADhEREQ4A"
    "AB4RHhAQAAAPEQ8BAQAAFhgQEBAAAA8QDgEeCAgcCAgIBgAAEREREQ8AABEREQoEAAAR"
    "ERUVCgAAEQoEChEAABERDwEOAAAfAgQIHwYICBAICAYEBAQEBAQEDAICAQICDAAAChQA"
    "AAA="
)
_BOX = (0x1F, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1F)


def _opaque(rgb):
    return 0xFF000000 | (rgb & 0xFFFFFF)


def _glyph(ch):
    code = ord(ch)
    if code < 32 or code > 126:
        return _BOX
    start = (code - 32) * 7
    return _FONT[start:start + 7]


class BitmapFace:
    """Monospace fallback. Each source pixel is drawn as a 2x2 block."""

    cell_w = 12
    cell_h = 18

    def width(self, text):
        return len(text) * self.cell_w

    def draw(self, canvas, x, y, text, rgb):
        color = _opaque(rgb)
        pen = int(x)
        top = int(y)
        for ch in text:
            for row, value in enumerate(_glyph(ch)):
                for col in range(5):
                    if value & (1 << (4 - col)):
                        px = pen + 1 + col * 2
                        py = top + 2 + row * 2
                        canvas.put(px, py, color)
                        canvas.put(px + 1, py, color)
                        canvas.put(px, py + 1, color)
                        canvas.put(px + 1, py + 1, color)
            pen += self.cell_w

    def close(self):
        return None


def _gpu_terminal():
    return Path.home() / "gpu_terminal"


def sdk_path():
    forced = os.environ.get("DRAFT_JUDGE_NEURONTE_SDK", "").strip()
    if forced:
        return forced
    path = _gpu_terminal() / "kilix-modules" / "neuronte-sdk" / "build" / "libneuronte-sdk.so"
    if path.is_file():
        return str(path)
    return ""


def weight_path():
    for key in ("DRAFT_JUDGE_NEURONTE_WEIGHTS", "NEURONTE_WEIGHTS", "PUZZLLM_NEURONTE_WEIGHTS"):
        value = os.environ.get(key, "").strip()
        if value and Path(value).is_file():
            return value
    root = _gpu_terminal() / "worktrees"
    preferred = (
        root / "neuronte-v2-release-2.4.0" / "weights" / "neuronte-split-v9.nrt",
        root / "neuronte-term-blocks" / "weights" / "neuronte-split-v4.nrt",
    )
    for path in preferred:
        if path.is_file():
            return str(path)
    if root.is_dir():
        for name in ("neuronte-split-v9.nrt", "neuronte-split-v4.nrt"):
            found = sorted(root.glob(f"*/weights/{name}"))
            if found:
                return str(found[0])
    return ""


def ensure_neuronte_lib():
    if os.environ.get("NEURONTE_LIB", "").strip():
        return
    root = _gpu_terminal() / "worktrees"
    candidates = [
        root / "neuronte-v2-release-2.4.0" / "build" / "libneuronte.so",
        root / "neuronte-rgba" / "build" / "libneuronte.so",
    ]
    if root.is_dir():
        candidates.extend(sorted(root.glob("*/build/libneuronte.so")))
    for path in candidates:
        if path.is_file():
            os.environ["NEURONTE_LIB"] = str(path)
            return


class NeuronteFace:
    """Glyphs from neuronte-sdk. One cell per Unicode scalar."""

    cell_w = 8
    cell_h = 16

    def __init__(self, sdk, handle):
        self.sdk = sdk
        self.handle = handle

    @classmethod
    def open(cls):
        library = sdk_path()
        if not library:
            raise FileNotFoundError("neuronte sdk")
        sdk = ctypes.CDLL(library)
        sdk.nsdk_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        sdk.nsdk_open.restype = ctypes.c_int
        sdk.nsdk_close.argtypes = [ctypes.c_void_p]
        sdk.nsdk_set_binary.argtypes = [ctypes.c_void_p, ctypes.c_int]
        sdk.nsdk_set_binary.restype = ctypes.c_int
        sdk.nsdk_grid_text.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t,
            ctypes.c_uint32,
        ]
        sdk.nsdk_grid_text.restype = ctypes.c_int
        ensure_neuronte_lib()
        weight = weight_path()
        handle = ctypes.c_void_p()
        rc = sdk.nsdk_open(weight.encode() if weight else None, ctypes.byref(handle))
        if rc != 0 or not handle.value:
            raise OSError(f"neuronte weights did not open ({rc})")
        sdk.nsdk_set_binary(handle, 1)
        return cls(sdk, handle)

    def close(self):
        if self.handle is not None:
            self.sdk.nsdk_close(self.handle)
            self.handle = None

    def width(self, text):
        return len(text) * self.cell_w

    def draw(self, canvas, x, y, text, rgb):
        if not text or self.handle is None:
            return
        encoded = text.encode("utf-8")
        color = _opaque(rgb)
        buf = (ctypes.c_uint32 * len(canvas.px)).from_buffer(canvas.px)
        self.sdk.nsdk_grid_text(
            self.handle, buf, canvas.w, canvas.w, canvas.h,
            int(x), int(y), self.cell_w, self.cell_h, 0, 0,
            encoded, len(encoded), color,
        )


def open_face():
    if os.environ.get("DRAFT_JUDGE_RASTER") == "1":
        return BitmapFace()
    try:
        return NeuronteFace.open()
    except Exception:
        return BitmapFace()


class Canvas:
    def __init__(self, width, height, face):
        self.w = width
        self.h = height
        self.face = face
        self.px = array.array("I", [0]) * (width * height)
        self.hits = []

    def put(self, x, y, color):
        if 0 <= x < self.w and 0 <= y < self.h:
            self.px[y * self.w + x] = color

    def pixel(self, x, y):
        return self.px[y * self.w + x]

    def count_rgb(self, rgb):
        want = _opaque(rgb)
        return sum(1 for value in self.px if value == want)

    def ink(self):
        blank = _opaque(BG)
        return sum(1 for value in self.px if value not in (0, blank))

    def fill(self, x, y, w, h, rgb):
        color = _opaque(rgb)
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(self.w, int(x + w))
        y1 = min(self.h, int(y + h))
        if x0 >= x1 or y0 >= y1:
            return
        row = array.array("I", [color]) * (x1 - x0)
        span = x1 - x0
        for yy in range(y0, y1):
            start = yy * self.w + x0
            self.px[start:start + span] = row

    def stroke(self, x, y, w, h, rgb):
        self.fill(x, y, w, 1, rgb)
        self.fill(x, y + h - 1, w, 1, rgb)
        self.fill(x, y, 1, h, rgb)
        self.fill(x + w - 1, y, 1, h, rgb)

    def disc(self, cx, cy, radius, rgb):
        color = _opaque(rgb)
        limit = radius * radius
        for y in range(int(cy - radius), int(cy + radius + 1)):
            for x in range(int(cx - radius), int(cx + radius + 1)):
                if 0 <= x < self.w and 0 <= y < self.h and (x - cx) ** 2 + (y - cy) ** 2 <= limit:
                    self.px[y * self.w + x] = color

    def text(self, x, y, text, rgb):
        self.face.draw(self, x, y, text, rgb)

    def clip_text(self, x, y, width, text, rgb):
        cols = max(0, int(width) // self.face.cell_w)
        if cols <= 0:
            return
        if len(text) > cols:
            text = (text[:max(0, cols - 3)] + "...")[:cols]
        self.text(x, y, text, rgb)

    def hit(self, x, y, w, h, name):
        self.hits.append((int(x), int(y), int(w), int(h), name))

    def hit_at(self, x, y):
        for left, top, w, h, name in reversed(self.hits):
            if left <= x < left + w and top <= y < top + h:
                return name
        return None

    def rgb(self):
        out = bytearray(self.w * self.h * 3)
        view = memoryview(out)
        index = 0
        for value in self.px:
            view[index] = (value >> 16) & 255
            view[index + 1] = (value >> 8) & 255
            view[index + 2] = value & 255
            index += 3
        return bytes(out)


def _paragraph(canvas, x, y, width, text, color, bottom):
    cols = max(1, int(width) // canvas.face.cell_w)
    line = ""
    for ch in text.replace("\n", " "):
        if ch == " " and not line:
            continue
        trial = line + ch
        if len(trial) > cols:
            canvas.text(x, y, line.rstrip(), color)
            y += canvas.face.cell_h
            line = "" if ch == " " else ch
            if y + canvas.face.cell_h > bottom:
                return y
        else:
            line = trial
    if line and y + canvas.face.cell_h <= bottom:
        canvas.text(x, y, line, color)
    return y


class Monitor:
    """Setup, model list, prompt, and the live scoreboard."""

    def __init__(self, models, settings=None):
        settings = settings or argparse.Namespace(
            output=None, llama_server=None, gpu_device=None)
        self.models = list(models)
        self.output = getattr(settings, "output", None)
        self.llama_server = getattr(settings, "llama_server", None)
        self.gpu_device = getattr(settings, "gpu_device", None)
        self.page = "setup"
        self.drafts = []
        self.judge = None
        self.count = 4
        self.schedule = "rotate"
        self.code = False
        self.prompt = ""
        self.prompt_edit = ""
        self.query = ""
        self.pick_for = "draft"
        self.pick_device = "gpu"
        self.pick_index = 0
        self.draft_cursor = 0
        self.dock = 0
        self.phase = "Ready"
        self.notice = "Choose draft models, a judge, and a prompt."
        self.notice_bad = False
        self.rows = []
        self.vanilla = ""
        self.selected = 0
        self.stage = "drafts"
        self.running = False
        self.done = False
        self.quit = False
        self.exit_code = 1
        self.directory = None
        self.report = ""
        self.queue = queue.Queue()
        self.thread = None

    def render(self, face):
        canvas = Canvas(WIDTH, HEIGHT, face)
        canvas.fill(0, 0, WIDTH, HEIGHT, BG)
        self._chrome(canvas)
        if self.page == "setup":
            self._setup(canvas)
        elif self.page == "pick":
            self._pick(canvas)
        elif self.page == "prompt":
            self._prompt(canvas)
        else:
            self._watch(canvas)
        self._dock(canvas)
        return canvas

    def _chrome(self, canvas):
        canvas.text(PAD, 18, "draft-judge", GOLD)
        phase = self.phase
        width = canvas.face.width(phase)
        lamp = RED if self.phase == "Failed" else AQUA
        canvas.disc(WIDTH - PAD - width - 16, 26, 4, lamp)
        canvas.text(WIDTH - PAD - width, 18, phase, INK)
        canvas.fill(0, 48, WIDTH, 1, LINE)

    def _setup(self, canvas):
        meta = f"{self.count} drafts    {self.schedule}    code {'on' if self.code else 'off'}"
        canvas.clip_text(PAD, 60, WIDTH // 2, meta, INK)
        color = RED if self.notice_bad else MUTED
        canvas.clip_text(PAD, 86, WIDTH - 2 * PAD, self.notice, color)
        top = 118
        list_h = DOCK_TOP - top - 100
        gap = 16
        left_w = (WIDTH - 2 * PAD - gap) // 2
        self._model_panel(
            canvas, PAD, top, left_w, list_h, "Drafts",
            self.drafts, self.draft_cursor, "draft")
        self._model_panel(
            canvas, PAD + left_w + gap, top, left_w, list_h, "Judge",
            [self.judge] if self.judge is not None else [], 0, "judge")
        y = top + list_h + 12
        canvas.fill(PAD, y, WIDTH - 2 * PAD, 64, PANEL)
        canvas.stroke(PAD, y, WIDTH - 2 * PAD, 64, LINE)
        canvas.text(PAD + 12, y + 8, "Prompt", MUTED)
        shown = self.prompt or "Type a prompt."
        canvas.clip_text(PAD + 12, y + 30, WIDTH - 2 * PAD - 24, shown,
                         INK if self.prompt else MUTED)
        canvas.hit(PAD, y, WIDTH - 2 * PAD, 64, "prompt")

    def _model_panel(self, canvas, x, y, w, h, title, specs, selected, kind):
        canvas.fill(x, y, w, h, PANEL)
        canvas.stroke(x, y, w, h, LINE)
        canvas.text(x + 12, y + 10, title, AQUA)
        row_h = max(28, canvas.face.cell_h + 8)
        body = y + canvas.face.cell_h + 20
        room = max(1, (y + h - 10 - body) // row_h)
        first = 0
        if specs and selected >= room:
            first = min(selected, len(specs) - room)
            first = max(0, first)
        if not specs:
            empty = "No drafts yet." if kind == "draft" else "Choose a judge."
            canvas.text(x + 12, body, empty, MUTED)
            canvas.hit(x, y, w, h, "drafts" if kind == "draft" else "judge")
            return
        for slot, index in enumerate(range(first, min(len(specs), first + room))):
            spec = specs[index]
            ry = body + slot * row_h
            if index == selected:
                canvas.fill(x + 1, ry, w - 2, row_h - 2, SELECT)
                canvas.fill(x + 1, ry, 3, row_h - 2, AQUA)
            label = f"{index + 1}  {spec.name}   {spec.device}"
            canvas.clip_text(x + 14, ry + 6, w - 28, label, INK)
            canvas.hit(x, ry, w, row_h, f"{kind}-{index}")

    def _pick(self, canvas):
        title = "Choose a draft" if self.pick_for == "draft" else "Choose a judge"
        canvas.text(PAD, 60, title, AQUA)
        canvas.clip_text(PAD + 280, 60, WIDTH - PAD - 300, self.notice,
                         RED if self.notice_bad else MUTED)
        canvas.fill(PAD, 92, WIDTH - 2 * PAD, 40, PANEL)
        canvas.stroke(PAD, 92, WIDTH - 2 * PAD, 40, AQUA)
        shown = self.query or "name, or name cpu"
        canvas.clip_text(PAD + 12, 102, WIDTH - 2 * PAD - 24, shown,
                         INK if self.query else MUTED)
        visible = self.visible_models()
        top = 148
        height = DOCK_TOP - top - 12
        canvas.fill(PAD, top, WIDTH - 2 * PAD, height, PANEL)
        canvas.stroke(PAD, top, WIDTH - 2 * PAD, height, LINE)
        if not visible:
            message = "No model matches that name." if self.query else "No GGUF files were found."
            canvas.text(PAD + 12, top + 16, message, MUTED)
            return
        index = min(self.pick_index, len(visible) - 1)
        row_h = max(28, canvas.face.cell_h + 8)
        room = max(1, (height - 16) // row_h)
        first = 0 if index < room else index - room + 1
        for slot, item in enumerate(range(first, min(len(visible), first + room))):
            _number, path = visible[item]
            ry = top + 8 + slot * row_h
            if item == index:
                canvas.fill(PAD + 1, ry, WIDTH - 2 * PAD - 2, row_h - 2, SELECT)
                canvas.fill(PAD + 1, ry, 3, row_h - 2, AQUA)
            canvas.clip_text(PAD + 14, ry + 6, WIDTH - 2 * PAD - 28, path.name, INK)
            canvas.hit(PAD, ry, WIDTH - 2 * PAD, row_h, f"pick-{item}")

    def _prompt(self, canvas):
        canvas.text(PAD, 64, "Prompt", AQUA)
        canvas.text(PAD, 96, "Enter keeps this text. Esc goes back.", MUTED)
        canvas.fill(PAD, 132, WIDTH - 2 * PAD, 88, PANEL)
        canvas.stroke(PAD, 132, WIDTH - 2 * PAD, 88, AQUA)
        cols = max(1, (WIDTH - 2 * PAD - 36) // canvas.face.cell_w)
        shown = self.prompt_edit[-cols:]
        canvas.text(PAD + 12, 160, shown, INK)
        cursor = PAD + 12 + canvas.face.width(shown)
        canvas.fill(cursor, 158, 2, canvas.face.cell_h, AQUA)

    def _watch(self, canvas):
        self._bar(canvas, 64)
        canvas.clip_text(PAD, 84, WIDTH - 2 * PAD, self.notice,
                         RED if self.notice_bad else MUTED)
        top = 116
        height = DOCK_TOP - top - 12
        left_w = 520
        canvas.fill(PAD, top, left_w, height, PANEL)
        canvas.stroke(PAD, top, left_w, height, LINE)
        canvas.text(PAD + 12, top + 10, "Drafts", MUTED)
        self._score_rows(canvas, PAD, top + 36, left_w, height - 44)
        right_x = PAD + left_w + 16
        right_w = WIDTH - right_x - PAD
        gap = 12
        half = (height - gap) // 2
        self._reply(canvas, right_x, top, right_w, half, "Vanilla", self.vanilla, False)
        header, body, gold = self._chosen_panel()
        self._reply(canvas, right_x, top + half + gap, right_w, half, header, body, gold)

    def _bar(self, canvas, y):
        width = WIDTH - 2 * PAD
        canvas.fill(PAD, y, width, 6, LINE)
        fraction = self.progress_fraction()
        if fraction > 0:
            canvas.fill(PAD, y, max(1, int(width * fraction)), 6, AQUA)

    def progress_fraction(self):
        if self.done and not self.running:
            return 1 if self.rows else 0
        total = max(1, self.count)
        written = len(self.rows)
        if self.stage == "judge":
            judged = sum(1 for row in self.rows if row.get("judged") or not row.get("ok"))
            return 0.5 + 0.5 * (judged / total)
        return 0.5 * (written / total)

    def _score_rows(self, canvas, x, y, w, h):
        if not self.rows:
            canvas.text(x + 12, y, "Replies show up here as they are written.", MUTED)
            return
        row_h = max(28, canvas.face.cell_h + 8)
        room = max(1, h // row_h)
        selected = min(self.selected, len(self.rows) - 1)
        first = 0 if selected < room else selected - room + 1
        winner = self.winner_index()
        for slot, index in enumerate(range(first, min(len(self.rows), first + room))):
            row = self.rows[index]
            ry = y + slot * row_h
            if index == selected:
                canvas.fill(x + 1, ry, w - 2, row_h - 2, SELECT)
                canvas.fill(x + 1, ry, 3, row_h - 2, AQUA)
            if not row["ok"]:
                detail, color = "skipped", RED
            elif row["score"] is None:
                detail, color = "...", MUTED
            else:
                detail = f"{row['score']:.3f}"
                color = AQUA if row["index"] == winner else INK
            label = f"{row['index']}  {row['model']}   {detail}"
            canvas.clip_text(x + 14, ry + 6, w - 28, label, color)
            canvas.hit(x, ry, w, row_h, f"score-{index}")

    def _chosen_panel(self):
        winner = self.winner_index()
        if not self.rows:
            return "Chosen", "", False
        index = min(self.selected, len(self.rows) - 1)
        row = self.rows[index]
        if winner is not None and row["index"] == winner:
            return "Chosen", row.get("text") or "", True
        return f"Draft {row['index']}", row.get("text") or "", False

    def _reply(self, canvas, x, y, w, h, title, text, gold):
        canvas.fill(x, y, w, h, PANEL)
        canvas.stroke(x, y, w, h, AQUA if gold else LINE)
        canvas.text(x + 12, y + 10, title, GOLD if gold else MUTED)
        body = text or "(empty)"
        color = INK if text else MUTED
        _paragraph(canvas, x + 12, y + 14 + canvas.face.cell_h, w - 24, body, color, y + h - 8)

    def _dock(self, canvas):
        canvas.fill(0, DOCK_TOP, WIDTH, HEIGHT - DOCK_TOP, PANEL)
        canvas.fill(0, DOCK_TOP, WIDTH, 1, LINE)
        canvas.text(PAD, HEIGHT - 48, "kilix", MUTED)
        buttons = self.dock_buttons()
        count = max(1, len(buttons))
        slot = min(186, (WIDTH - 128) // count)
        for index, (action, label, primary) in enumerate(buttons):
            x = 100 + index * slot
            y = HEIGHT - 60
            w = slot - 8
            h = 37
            canvas.fill(x, y, w, h, AQUA if primary else BUTTON)
            if index == self.dock and not primary:
                canvas.stroke(x, y, w, h, AQUA)
            canvas.clip_text(x + 11, y + 10, w - 18, label, BG if primary else INK)
            canvas.hit(x, y, w, h, action)

    def dock_buttons(self):
        if self.page == "setup":
            code = "on" if self.code else "off"
            return [
                ("drafts", "D  Drafts", False),
                ("judge", "J  Judge", False),
                ("prompt", "P  Prompt", False),
                ("count", f"N  {self.count}", False),
                ("schedule", f"S  {self.schedule}", False),
                ("code", f"C  {code}", False),
                ("start", "R  Start", True),
                ("quit", "Q  Quit", False),
            ]
        if self.page == "pick":
            return [
                ("select", "Enter  Select", True),
                ("device", f"G  {self.pick_device}", False),
                ("back", "Esc  Back", False),
            ]
        if self.page == "prompt":
            return [
                ("keep", "Enter  Keep", True),
                ("back", "Esc  Back", False),
            ]
        if self.running:
            return [("busy", "Running", False)]
        return [("back", "Esc  Back", True)]

    def visible_models(self):
        query = self.query.strip()
        name, _device = split_device(query) if query else ("", None)
        if not name:
            return list(enumerate(self.models, 1))
        return match_models(self.models, name)

    def winner_index(self):
        best = None
        best_score = None
        for row in self.rows:
            score = row.get("score")
            if score is None:
                continue
            if best_score is None or score > best_score:
                best = row["index"]
                best_score = score
        return best

    def chosen_text(self):
        winner = self.winner_index()
        for row in self.rows:
            if row["index"] == winner:
                return row.get("text") or ""
        return ""

    def poll(self):
        changed = False
        while True:
            try:
                event = self.queue.get_nowait()
            except queue.Empty:
                return changed
            self.apply(event)
            changed = True

    def apply(self, event):
        kind = event.get("kind")
        if kind == "note":
            self.notice = event.get("text") or ""
            self.notice_bad = False
            if self.notice.startswith("judging"):
                self.stage = "judge"
        elif kind == "warn":
            self.notice = event.get("text") or ""
            self.notice_bad = True
        elif kind == "run":
            if event.get("directory"):
                self.directory = event["directory"]
        elif kind == "vanilla":
            self.vanilla = event.get("text") or ""
        elif kind == "draft":
            self.rows.append({
                "index": event.get("index"),
                "model": event.get("model") or "",
                "text": event.get("text") or "",
                "ok": bool(event.get("ok")),
                "error": event.get("error") or "",
                "score": None,
                "judged": False,
            })
            self.selected = len(self.rows) - 1
        elif kind == "judge":
            for row in self.rows:
                if row["index"] == event.get("index"):
                    row["judged"] = True
                    if event.get("ok"):
                        row["score"] = event.get("score")
                    else:
                        row["error"] = event.get("error") or row["error"]
            self.stage = "judge"
        elif kind == "load_failure":
            self.phase = "Failed"
            self.notice = event.get("text") or "The judge failed to load."
            self.notice_bad = True
        elif kind == "result" and event.get("ok"):
            for index, row in enumerate(self.rows):
                if row["index"] == event.get("index"):
                    self.selected = index
        elif kind == "finished":
            self.running = False
            self.done = True
            self.exit_code = int(event.get("code") or 0)
            self.report = event.get("report") or ""
            if event.get("directory"):
                self.directory = event["directory"]
            self.phase = "Complete" if self.exit_code == 0 else "Failed"

    def handle(self, event, canvas):
        if event[0] == "key":
            self.key(event[1])
        elif event[0] == "click":
            name = canvas.hit_at(event[1], event[2])
            if name:
                self.action(name)

    def key(self, name):
        if self.page == "prompt":
            if name == "backspace":
                self.prompt_edit = self.prompt_edit[:-1]
            elif name == "enter":
                self.action("keep")
            elif name == "esc":
                self.action("back")
            elif name == "left":
                self.action("dock-left")
            elif name == "right":
                self.action("dock-right")
            elif len(name) == 1 and name >= " ":
                self.prompt_edit += name
            return
        if self.page == "pick":
            if name == "backspace":
                self.query = self.query[:-1]
                self.pick_index = 0
            elif name == "enter":
                self.action("select")
            elif name == "esc":
                self.action("back")
            elif name in ("up", "down"):
                self.move(-1 if name == "up" else 1)
            elif name == "left":
                self.action("dock-left")
            elif name == "right":
                self.action("dock-right")
            elif name == "g" and not self.query:
                self.action("device")
            elif len(name) == 1 and name >= " ":
                self.query += name
                self.pick_index = 0
            return
        if name == "esc":
            self.action("quit" if self.page == "setup" else "back")
        elif name == "enter":
            self.action("activate")
        elif name == "left":
            self.action("dock-left")
        elif name == "right":
            self.action("dock-right")
        elif name in ("up", "down"):
            self.move(-1 if name == "up" else 1)
        elif name == "backspace":
            self.action("backspace")
        elif self.page == "setup" and name == "N":
            self.bump_count(-1)
        elif self.page == "setup" and name in "djpnscrqx":
            self.action({
                "d": "drafts", "j": "judge", "p": "prompt", "n": "count",
                "s": "schedule", "c": "code", "r": "start", "q": "quit", "x": "drop",
            }[name])
        elif self.page == "watch" and name == "b":
            self.action("back")

    def move(self, delta):
        if self.page == "pick":
            visible = self.visible_models()
            if visible:
                self.pick_index = max(0, min(len(visible) - 1, self.pick_index + delta))
        elif self.page == "watch" and self.rows:
            self.selected = max(0, min(len(self.rows) - 1, self.selected + delta))
        elif self.page == "setup" and self.drafts:
            self.draft_cursor = max(0, min(len(self.drafts) - 1, self.draft_cursor + delta))

    def action(self, name):
        if name == "activate":
            buttons = self.dock_buttons()
            if buttons:
                self.action(buttons[min(self.dock, len(buttons) - 1)][0])
            return
        if name == "dock-left":
            self.dock = max(0, self.dock - 1)
            return
        if name == "dock-right":
            self.dock = min(len(self.dock_buttons()) - 1, self.dock + 1)
            return
        if name == "backspace":
            if self.page == "setup":
                self.drop()
            return
        if name.startswith("pick-"):
            self.pick_index = int(name.split("-", 1)[1])
            self.select_model()
            return
        if name.startswith("draft-"):
            self.draft_cursor = int(name.split("-", 1)[1])
            return
        if name.startswith("judge-"):
            self.action("judge")
            return
        if name.startswith("score-"):
            self.selected = int(name.split("-", 1)[1])
            return
        if name == "drafts":
            self.open_pick("draft")
        elif name == "judge":
            self.open_pick("judge")
        elif name == "prompt":
            self.open_prompt()
        elif name == "count":
            self.count = 1 if self.count >= 16 else self.count + 1
        elif name == "schedule":
            self.schedule = "together" if self.schedule == "rotate" else "rotate"
        elif name == "code":
            self.code = not self.code
        elif name == "start":
            self.start()
        elif name == "quit":
            self.quit_monitor()
        elif name == "back":
            self.back()
        elif name == "busy":
            self.fail("The run is still going.")
        elif name == "device":
            self.pick_device = "cpu" if self.pick_device == "gpu" else "gpu"
        elif name == "select":
            self.select_model()
        elif name == "keep":
            self.keep_prompt()
        elif name == "drop":
            self.drop()

    def bump_count(self, delta):
        self.count = min(16, max(1, self.count + delta))

    def fail(self, text):
        self.notice = text
        self.notice_bad = True

    def open_pick(self, role):
        self.pick_for = role
        self.pick_index = 0
        self.query = ""
        self.page = "pick"
        self.dock = 0
        self.notice = "Type a name, or click a row. Add cpu or gpu to set the device."
        self.notice_bad = False

    def open_prompt(self):
        self.prompt_edit = self.prompt
        self.page = "prompt"
        self.dock = 0

    def keep_prompt(self):
        self.prompt = strip_wrapping_quotes(self.prompt_edit.strip())
        self.page = "setup"
        self.dock = 0
        if self.prompt:
            self.notice = "Prompt kept."
            self.notice_bad = False
        else:
            self.fail("The prompt is empty.")

    def select_model(self):
        query = self.query.strip()
        name, device = split_device(query) if query else ("", None)
        device = device or self.pick_device
        if name and looks_explicit_path(name):
            found = existing_file(name)
            if found is None:
                self.fail("That file was not found.")
                return
            spec = ModelSpec(str(found), device)
        else:
            visible = self.visible_models()
            if not visible:
                self.fail("No model matches that name.")
                return
            index = min(self.pick_index, len(visible) - 1)
            spec = ModelSpec(str(visible[index][1]), device)
        if self.pick_for == "judge":
            self.judge = spec
            self.notice = f"Judge {spec.name} on {spec.device}."
        else:
            if any(item.path == spec.path and item.device == spec.device for item in self.drafts):
                self.fail(f"{spec.name} is already a draft.")
                return
            self.drafts.append(spec)
            self.draft_cursor = len(self.drafts) - 1
            self.notice = f"Added {spec.name} on {spec.device}."
        self.notice_bad = False
        self.page = "setup"
        self.query = ""
        self.dock = 0

    def drop(self):
        if not self.drafts:
            return
        index = min(self.draft_cursor, len(self.drafts) - 1)
        removed = self.drafts.pop(index)
        self.draft_cursor = max(0, min(self.draft_cursor, len(self.drafts) - 1))
        self.notice = f"Removed {removed.name}."
        self.notice_bad = False

    def back(self):
        if self.page == "watch" and self.running:
            self.fail("The run is still going.")
            return
        if self.page == "setup":
            self.quit_monitor()
            return
        self.page = "setup"
        self.dock = 0

    def quit_monitor(self):
        if self.running:
            self.fail("The run is still going.")
            return
        self.quit = True
        if not self.done:
            self.exit_code = 1

    def start(self):
        if self.running:
            return
        if not self.drafts:
            self.fail("Add at least one draft model.")
            return
        if self.judge is None:
            self.fail("Choose a judge.")
            return
        if not self.prompt.strip():
            self.fail("Enter a prompt.")
            return
        self.page = "watch"
        self.running = True
        self.done = False
        self.rows = []
        self.vanilla = ""
        self.selected = 0
        self.stage = "drafts"
        self.phase = "Running"
        self.notice = "Loading models."
        self.notice_bad = False
        self.exit_code = 1
        self.dock = 0
        self.thread = threading.Thread(target=self._run, name="draft-judge")
        self.thread.start()

    def _run(self):
        from kilix_bench.draft_judge.workflow import execute
        def on_event(event):
            self.queue.put(event)
        code, directory, report = 1, None, ""
        try:
            code, directory, report = execute({
                "draft_models": list(self.drafts),
                "judge": self.judge,
                "draft_count": self.count,
                "schedule": self.schedule,
                "prompt": self.prompt,
                "code": self.code,
                "output": self.output,
                "llama_server": self.llama_server,
                "gpu_device": self.gpu_device,
                "on_event": on_event,
            })
        except SystemExit as exc:
            message = exc.args[0] if exc.args else "stopped"
            self.queue.put({"kind": "warn", "text": str(message)})
            code = exc.code if isinstance(exc.code, int) else 1
        except Exception as exc:
            self.queue.put({"kind": "warn", "text": str(exc)})
            code = 1
        self.queue.put({
            "kind": "finished",
            "code": code if isinstance(code, int) else 1,
            "directory": "" if directory is None else str(directory),
            "report": report or "",
        })


def monitor_parser():
    tool = argparse.ArgumentParser(prog="draft-judge")
    tool.add_argument("--model-dir", action="append", default=[],
                      help="extra directory of GGUF files")
    tool.add_argument("--output", help="run directory")
    tool.add_argument("--llama-server", help="llama-server binary")
    tool.add_argument("--gpu-device", help="device name for every GPU model")
    return tool


def run_monitor(argv):
    args = monitor_parser().parse_args(argv)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        sys.stderr.write("draft-judge: the monitor needs a terminal; use --cli run\n")
        return 2
    from kilix_bench.draft_judge.pane import present_loop
    monitor = Monitor(discover_models(args.model_dir), args)
    face = open_face()
    try:
        return present_loop(monitor, face)
    finally:
        closer = getattr(face, "close", None)
        if closer is not None:
            closer()
