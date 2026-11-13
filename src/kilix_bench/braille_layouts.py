"""Canonical Braille bits and regions in the original [0,1] x [0,2] cell."""

# Each tuple is (bit mask, x0, x1, y0, y1), ordered as the output tiles.
REGIONS = {
    "left": (0x47, 0., .5, 0., 2.),
    "right": (0xb8, .5, 1., 0., 2.),
    "top": (0x1b, 0., 1., 0., 1.),
    "bottom": (0xe4, 0., 1., 1., 2.),
    "top-left": (0x03, 0., .5, 0., 1.),
    "top-right": (0x18, .5, 1., 0., 1.),
    "bottom-left": (0x44, 0., .5, 1., 2.),
    "bottom-right": (0xa0, .5, 1., 1., 2.),
}
LAYOUTS = {"columns": ("left", "right"), "rows": ("top", "bottom"),
           "quadrants": ("top-left", "top-right", "bottom-left", "bottom-right")}
WIDTHS = {"columns": 24, "rows": 24, "quadrants": 12,
          "dots": 6, "grid12": 4, "grid24": 2, "grid48": 1}
# Unequal columns keep every boundary on a pixel at widths 8, 16 and 48.
# Middle columns of the three-column grids contain halves of BOTH dots.
GRIDS = {"dots": ((0., .5, 1.), 4),
         "grid12": ((0., .25, .75, 1.), 4),
         "grid24": ((0., .25, .75, 1.), 8),
         "grid48": ((0., .125, .25, .5, .75, .875, 1.), 8)}
DOT_BITS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
for _layout, (_edges, _rows) in GRIDS.items():
    _parts = []
    for _row in range(_rows):
        for _col, (_x0, _x1) in enumerate(zip(_edges, _edges[1:])):
            _left, _right = DOT_BITS[_row * 4 // _rows]
            _mask = (_left if _x0 < .5 else 0) | (_right if _x1 > .5 else 0)
            _name = f"{_layout}-r{_row}-c{_col}"
            REGIONS[_name] = (_mask, _x0, _x1, 2*_row/_rows, 2*(_row+1)/_rows)
            _parts.append(_name)
    LAYOUTS[_layout] = tuple(_parts)


def tiles(layout, width, height):
    """Return integer pixel rectangles without rounding sample positions."""
    out = []
    for part in LAYOUTS[layout]:
        mask, x0, x1, y0, y1 = REGIONS[part]
        edges = (x0*width, x1*width, y0*height/2, y1*height/2)
        if any(int(v) != v for v in edges):
            raise ValueError("cell dimensions must divide evenly into the requested layout")
        x0, x1, y0, y1 = map(int, edges)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("empty tile")
        out.append((mask, x0, y0, x1-x0, y1-y0))
    return out


def crop_field(full, region):
    """Bilinearly sample the original grid, preserving its distance units."""
    import numpy as np
    _, x0, x1, y0, y1 = region
    n = full.shape[1] - 1
    if full.shape != (2 * n + 1, n + 1):
        raise ValueError("expected the original 1:2 field grid")
    x = np.arange(n + 1) * (x1 - x0) + x0 * n
    y = np.arange(2 * n + 1) * (y1 - y0) / 2 + y0 * n
    ix, iy = x.astype(np.int64), y.astype(np.int64)
    fx, fy = x - ix, y - iy
    horizontal = full[:, ix] * (1 - fx) + full[:, np.minimum(ix + 1, n)] * fx
    return horizontal[iy] * (1 - fy[:, None]) + horizontal[np.minimum(iy + 1, 2*n)] * fy[:, None]
