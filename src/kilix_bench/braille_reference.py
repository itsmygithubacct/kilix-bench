"""Independent float64 coverage of packed Neuronte distance and tiled files.

Reads the packed bytes only (no Neuronte Python or C code) and implements,
from the published file layout, version 1 distance nets, version 2 bundles
whose range table covers the scalar, and version 6 tiled nets. It exists to
check the production library's output outside every timed batch.

Version 1: "NEURONTE", u32 version, position octaves (6), code bits (64),
hidden width H, hidden layers (3), f32 tau, 43 u32 parity masks, then f32
layers l1 (H x 90, H), l2 (H x H, H), l3 (H x H, H), out (1 x H, 1).
Version 2: u32 nets, ranges, fallback, gate width G; net lengths; ranges as
(first, last, net); the gate (43 masks, G x 64 + G, nets x G + nets floats)
if G; the nets. Version 6: u32 columns C, rows R, nets M, first, last;
C + 1 and R + 1 f32 edges; C x R (net, mask, base, extract); M lengths;
the M version 1 nets.
"""

import math
import struct

import numpy as np


def _normal(s):
    return 0xFFFD if s > 0x10FFFF or 0xD800 <= s <= 0xDFFF else s


def auto_samples(width):
    return 2 if width < 24 else 1


class Distance:
    def __init__(self, data):
        if data[:8] != b"NEURONTE":
            raise ValueError("not a Neuronte image")
        version, octaves, bits, h, layers = struct.unpack_from("<5I", data, 8)
        if (version, octaves, bits, layers) != (1, 6, 64, 3) or not 1 <= h <= 1024:
            raise ValueError("not a version 1 distance net")
        self.tau = struct.unpack_from("<f", data, 28)[0]
        self.masks = struct.unpack_from("<43I", data, 32)
        floats = np.frombuffer(data, "<f4", offset=204).astype(np.float64)
        if floats.size != 2 * h * h + 94 * h + 1:
            raise ValueError("version 1 net length")
        self.layers, off = [], 0
        for rows, cols in ((h, 90), (h, h), (h, h), (1, h)):
            w = floats[off:off + rows * cols].reshape(rows, cols)
            off += rows * cols
            self.layers.append((w, floats[off:off + rows]))
            off += rows
        self.hidden = h

    def code(self, c):
        bits = [(c >> i) & 1 for i in range(21)] + [bin(c & m).count("1") & 1 for m in self.masks]
        return np.where(np.asarray(bits) == 1, 1.0, -1.0)

    def field(self, c, x, y):
        f = [2 * x - 1, y - 1]
        for k in range(6):
            w = 2 * math.pi * 2 ** k
            f += [np.sin(w * x), np.sin(w * y), np.cos(w * x), np.cos(w * y)]
        h = np.concatenate([np.stack(f, -1), np.broadcast_to(self.code(c), (len(x), 64))], 1)
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < 3:
                h = np.maximum(h, 0)
        return h[:, 0]

    def coverage(self, s, cw, ch, ss):
        x = (np.arange(cw * ss) + .5) / (cw * ss)
        y = (np.arange(ch * ss) + .5) / (ch * ss) * 2
        yy, xx = np.meshgrid(y, x, indexing="ij")
        f = self.field(_normal(s), xx.ravel(), yy.ravel())
        return np.clip(.5 + f * self.tau * cw * ss, 0, 1).reshape(ch, ss, cw, ss).mean((1, 3))


class Tiled:
    def __init__(self, data):
        version, cols, rows, m, self.first, self.last = struct.unpack_from("<6I", data, 8)
        if version != 6 or not (1 <= cols <= 16 and 1 <= rows <= 16 and 1 <= m <= min(64, cols * rows)):
            raise ValueError("not a version 6 tiled net")
        off = 32
        self.xe = list(struct.unpack_from(f"<{cols + 1}f", data, off)); off += 4 * (cols + 1)
        self.ye = list(struct.unpack_from(f"<{rows + 1}f", data, off)); off += 4 * (rows + 1)
        self.tiles = [struct.unpack_from("<4I", data, off + 16 * k) for k in range(cols * rows)]
        off += 16 * cols * rows
        lengths = struct.unpack_from(f"<{m}I", data, off); off += 4 * m
        self.nets = []
        for n in lengths:
            self.nets.append(Distance(data[off:off + n]))
            off += n
        if off != len(data):
            raise ValueError("version 6 length")
        self.cols, self.rows = cols, rows

    def code(self, s, tile):
        _, mask, base, extract = tile
        b = (s - self.first) % (1 << 32)
        if not extract:
            return base | (b & mask)
        v, bit = 0, 0
        for i in range(32):
            if mask >> i & 1:
                v |= (b >> i & 1) << bit
                bit += 1
        return base + v

    def coverage(self, s, cw, ch, ss):
        s = _normal(s)
        sw, sh = cw * ss, ch * ss
        x = (np.arange(sw) + .5) / sw
        y = (np.arange(sh) + .5) / sh * 2
        xe, ye = np.asarray(self.xe, float), np.asarray(self.ye, float)
        col = np.minimum(np.searchsorted(xe, x, side="right") - 1, self.cols - 1)
        row = np.minimum(np.searchsorted(ye, y, side="right") - 1, self.rows - 1)
        lx = (x - xe[col]) / (xe[col + 1] - xe[col])
        ly = (y - ye[row]) / (ye[row + 1] - ye[row]) * 2
        out = np.full((sh, sw), np.nan)
        for k, tile in enumerate(self.tiles):
            r, c = divmod(k, self.cols)
            jx, jy = np.nonzero(col == c)[0], np.nonzero(row == r)[0]
            if len(jx) and len(jy):
                net = self.nets[tile[0]]
                yy, xx = np.meshgrid(ly[jy], lx[jx], indexing="ij")
                f = net.field(self.code(s, tile), xx.ravel(), yy.ravel()).reshape(len(jy), len(jx))
                out[np.ix_(jy, jx)] = np.clip(.5 + f * net.tau * (xe[c + 1] - xe[c]) * cw * ss, 0, 1)
        if np.isnan(out).any():
            raise ValueError("a sample belongs to no tile")
        return out.reshape(ch, ss, cw, ss).mean((1, 3))


def member(data):
    version = struct.unpack_from("<I", data, 8)[0]
    if version == 1:
        return Distance(data)
    if version == 6:
        return Tiled(data)
    raise ValueError(f"version {version} member is not evaluated by this reference")


class Packed:
    """A packed file, routed by its range table when it is a bundle."""

    def __init__(self, data):
        data = bytes(data)
        if data[:8] != b"NEURONTE":
            raise ValueError("not a Neuronte file")
        self.version = struct.unpack_from("<I", data, 8)[0]
        if self.version != 2:
            self.members, self.ranges = [member(data)], []
            return
        _, nets, ranges, _, gate = struct.unpack_from("<5I", data, 8)
        lengths = struct.unpack_from(f"<{nets}I", data, 28)
        off = 28 + 4 * nets
        self.ranges = [struct.unpack_from("<3I", data, off + 12 * i) for i in range(ranges)]
        off += 12 * ranges
        if gate:
            off += 4 * (43 + gate * 64 + gate + nets * gate + nets)
        self.members = []
        for n in lengths:
            v = struct.unpack_from("<I", data, off + 8)[0]
            self.members.append(member(data[off:off + n]) if v in (1, 6) else None)
            off += n
        if off != len(data):
            raise ValueError("version 2 length")

    def route(self, s):
        if not self.ranges:
            return 0
        s = _normal(s)
        for first, last, net in self.ranges:
            if first <= s <= last:
                return net
        raise ValueError(f"U+{s:04X} is outside the range table; the gate is not evaluated here")

    def coverage(self, s, cw, ch, ss=0):
        net = self.members[self.route(s)]
        if net is None:
            raise ValueError("the routed member is not a distance or tiled net")
        return net.coverage(s, cw, ch, ss or auto_samples(cw))
