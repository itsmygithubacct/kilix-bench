"""The three ways this folder draws a terminal cell.

    v2       Neuronte 2's library and terminal bundle   (NEURONTE_V2_HOME)
    n014     Neuronte 0.1.4's library and a bundle of it (NEURONTE_014_HOME)
    dejavu   DejaVu Sans Mono, then DejaVu Sans, by FreeType (DEJAVU_FONT)

Each face answers the same three questions: which characters it has, what one
character looks like in a cell of a given size, and what it is called.
A cell is a NumPy array of coverage, height by width, 0 to 255.
"""
import ctypes
import glob
import os

import numpy as np

SCALARS = [s for s in range(0x20, 0x110000) if not 0xD800 <= s < 0xE000]
DEJAVU_PLACES = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu",
                 "/usr/share/fonts/TTF", "/usr/local/share/fonts")
BUNDLES = {"v2": ("NEURONTE_V2_HOME", "NEURONTE_V2_BUNDLE", "neuronte-split-v*.nrt"),
           "n014": ("NEURONTE_014_HOME", "NEURONTE_014_BUNDLE", "neuronte-split-v5.nrt")}


class Missing(RuntimeError):
    pass


def _version(home):
    try:
        with open(os.path.join(home, "VERSION"), encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return "?"


class Neuronte:
    def __init__(self, which):
        home_var, bundle_var, pattern = BUNDLES[which]
        home = os.environ.get(home_var, "")
        if not home:
            raise Missing(f"set {home_var} to a built Neuronte tree")
        library = os.path.join(home, "build", "libneuronte.so")
        bundle = os.environ.get(bundle_var, "")
        if not bundle:
            found = sorted(glob.glob(os.path.join(home, "weights", pattern)),
                           key=lambda p: int("".join(c for c in os.path.basename(p) if c.isdigit()) or 0))
            if not found:
                raise Missing(f"no weights/{pattern} under {home_var}")
            bundle = found[-1]
        elif os.sep not in bundle:
            bundle = os.path.join(home, "weights", bundle)
        for path in (library, bundle):
            if not os.path.exists(path):
                raise Missing(f"{os.path.basename(path)} is not under {home_var}; run make there")
        lib = self.lib = ctypes.CDLL(library)
        lib.neuronte_open_file.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.neuronte_draw_scalar.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)] + [ctypes.c_int] * 7 \
            + [ctypes.c_uint32, ctypes.c_uint32]
        self.can_mark = hasattr(lib, "neuronte_untrained")
        if self.can_mark:
            lib.neuronte_untrained.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            lib.neuronte_set_untrained.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.handle = ctypes.c_void_p()
        if lib.neuronte_open_file(os.fsencode(bundle), ctypes.byref(self.handle)) != 0:
            raise Missing(f"the library under {home_var} refuses {os.path.basename(bundle)}")
        self.path = bundle
        self.kind = "Neuronte 2" if which == "v2" else "Neuronte"
        self.name = f"{self.kind} {_version(home)}  {os.path.basename(bundle)[:-4]}"
        self.marked = False

    def has(self):
        """The characters the bundle was trained on."""
        if not self.can_mark:
            raise Missing("this Neuronte library cannot say what it was trained on (needs 0.1.4 or 2.2.0)")
        u, h = self.lib.neuronte_untrained, self.handle
        return {s for s in SCALARS if u(h, s) == 0}

    def mark(self, on):
        """Draw a plain mark for characters the bundle was not trained on."""
        if self.can_mark:
            self.lib.neuronte_set_untrained(self.handle, 1 if on else 0)
            self.marked = bool(on)

    def cell(self, scalar, w, h):
        """White on black through the draw call a terminal uses, which keeps
        each cell in the handle after the net has drawn it once."""
        out = (ctypes.c_uint32 * (w * h))()
        self.lib.neuronte_draw_scalar(self.handle, out, w, w, h, 0, 0, w, h, scalar, 0xFFFFFFFF)
        return (np.frombuffer(out, np.uint32).reshape(h, w) & 0xFF).astype(np.uint8)


def cmap(path):
    """The characters a TrueType file maps to a glyph: its cmap table,
    subtable format 12 if it has one and format 4 otherwise."""
    import struct
    with open(path, "rb") as f:
        data = f.read()
    count = struct.unpack(">H", data[4:6])[0]
    table = None
    for i in range(count):
        tag, _, offset, _ = struct.unpack(">4sIII", data[12 + 16 * i:28 + 16 * i])
        if tag == b"cmap":
            table = offset
    if table is None:
        raise Missing(f"{os.path.basename(path)} has no cmap table")
    subtables = {}
    for i in range(struct.unpack(">H", data[table + 2:table + 4])[0]):
        _, _, offset = struct.unpack(">HHI", data[table + 4 + 8 * i:table + 12 + 8 * i])
        subtables.setdefault(struct.unpack(">H", data[table + offset:table + offset + 2])[0], table + offset)
    found = set()
    if 12 in subtables:
        at = subtables[12]
        for i in range(struct.unpack(">I", data[at + 12:at + 16])[0]):
            start, end, glyph = struct.unpack(">III", data[at + 16 + 12 * i:at + 28 + 12 * i])
            found.update(s for s in range(start, end + 1) if glyph + s - start)
    elif 4 in subtables:
        at = subtables[4]
        n = struct.unpack(">H", data[at + 6:at + 8])[0] // 2
        ends = struct.unpack(f">{n}H", data[at + 14:at + 14 + 2 * n])
        starts = struct.unpack(f">{n}H", data[at + 16 + 2 * n:at + 16 + 4 * n])
        deltas = struct.unpack(f">{n}H", data[at + 16 + 4 * n:at + 16 + 6 * n])
        ranges_at = at + 16 + 6 * n
        ranges = struct.unpack(f">{n}H", data[ranges_at:ranges_at + 2 * n])
        for k in range(n):
            for s in range(starts[k], ends[k] + 1):
                if s == 0xFFFF:
                    continue
                if ranges[k] == 0:
                    glyph = (s + deltas[k]) & 0xFFFF
                else:
                    p = ranges_at + 2 * k + ranges[k] + 2 * (s - starts[k])
                    glyph = struct.unpack(">H", data[p:p + 2])[0]
                    glyph = (glyph + deltas[k]) & 0xFFFF if glyph else 0
                if glyph:
                    found.add(s)
    else:
        raise Missing(f"{os.path.basename(path)} has no cmap subtable this program reads")
    return found


class DejaVu:
    """DejaVu Sans Mono, and DejaVu Sans for the characters Mono lacks, as a
    terminal falls back from its font to the next one."""

    def __init__(self):
        from PIL import ImageFont  # noqa: F401  (fail early if absent)
        self.paths = []
        given = [p for p in os.environ.get("DEJAVU_FONT", "").split(os.pathsep) if p]
        for name in given or ("DejaVuSansMono.ttf", "DejaVuSans.ttf"):
            places = [name] if given else [os.path.join(d, name) for d in DEJAVU_PLACES]
            found = [p for p in places if os.path.exists(p)]
            if not found:
                raise Missing(f"{os.path.basename(name)} was not found; set DEJAVU_FONT to the .ttf files, "
                              f"separated by '{os.pathsep}'")
            self.paths.append(found[0])
        self.path = self.paths[0]
        self.kind = "DejaVu"
        self.name = " + ".join(os.path.basename(p)[:-4] for p in self.paths) + "  FreeType"
        self.fonts, self.marked, self._maps = {}, False, None

    def maps(self):
        if self._maps is None:
            self._maps = [{s for s in cmap(p) if s >= 0x20} for p in self.paths]
        return self._maps

    def has(self):
        return set().union(*self.maps())

    def mark(self, on):
        self.marked = bool(on)

    def _fonts(self, w, h):
        """Every file at the largest size whose line fits the cell's height and
        whose 'M', in the first file, fits its width."""
        if (w, h) not in self.fonts:
            from PIL import ImageFont
            best = 4
            for size in range(5, 4 * h):
                f = ImageFont.truetype(self.path, size)
                ascent, descent = f.getmetrics()
                if ascent + descent > h or f.getlength("M") > w + 0.01:
                    break
                best = size
            self.fonts[(w, h)] = [ImageFont.truetype(p, best) for p in self.paths]
        return self.fonts[(w, h)]

    def cell(self, scalar, w, h):
        from PIL import Image, ImageDraw
        fonts = self._fonts(w, h)
        which = next((i for i, m in enumerate(self.maps()) if scalar in m), None)
        im = Image.new("L", (w, h), 0)
        if which is None and self.marked:
            ImageDraw.Draw(im).rectangle((1, 2, w - 2, h - 3), outline=255)
            return np.asarray(im)
        f = fonts[which or 0]
        ascent, descent = fonts[0].getmetrics()
        ch = chr(scalar)
        x = max(0, (w - f.getlength(ch)) / 2)
        y = (h - ascent - descent) // 2 + ascent - f.getmetrics()[0]
        ImageDraw.Draw(im).text((x, y), ch, fill=255, font=f)
        return np.asarray(im)


def face(which):
    if which in BUNDLES:
        return Neuronte(which)
    if which == "dejavu":
        return DejaVu()
    raise Missing(f"no face called {which}: v2, n014 or dejavu")


ORDER = ("v2", "n014", "dejavu")
