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
