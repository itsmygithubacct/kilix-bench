"""Cell geometry for box seams, clipping, and small cells.

Expectations are Unicode joining classes. Edge contact and a connected join
are different facts. Mask differences are a design delta: they never select
a quality winner, and a connected subset of a fixture is not quality parity.
"""

# Joining classes for the fixtures and the common box/block edges.
# Unlisted scalars are observed and not judged.
EXPECT = {
    0x2500: "LR", 0x2501: "LR", 0x2502: "TB", 0x2503: "TB",
    0x250C: "RB", 0x250F: "RB", 0x2510: "LB", 0x2513: "LB",
    0x2514: "RT", 0x2517: "RT", 0x2518: "LT", 0x251B: "LT",
    0x251C: "TRB", 0x2523: "TRB", 0x2524: "LTB", 0x252B: "LTB",
    0x252C: "LRB", 0x2533: "LRB", 0x2534: "LRT", 0x253B: "LRT",
    0x253C: "LRTB", 0x254B: "LRTB",
    0x2550: "LR", 0x2551: "TB", 0x2554: "RB", 0x2557: "LB",
    0x255A: "RT", 0x255D: "LT", 0x2560: "TRB", 0x2563: "LTB",
    0x2566: "LRB", 0x2569: "LRT", 0x256C: "LRTB",
    0x2580: "LRT", 0x2584: "LRB", 0x2588: "LRTB", 0x258C: "LTB", 0x2590: "RTB",
}

SEAM_GRID = (
    (0x250C, 0x2500, 0x252C, 0x2500, 0x2510),
    (0x2502, 0x0020, 0x2502, 0x0020, 0x2502),
    (0x251C, 0x2500, 0x253C, 0x2500, 0x2524),
    (0x2502, 0x0020, 0x2502, 0x0020, 0x2502),
    (0x2514, 0x2500, 0x2534, 0x2500, 0x2518),
)
# Every scored boundary overlaps in the ideal block shapes. In particular,
# an upper half beside a lower half is intentionally disconnected, so it
# must not be used as an expected horizontal join.
BLOCK_GRID = (
    (0x2588, 0x2580, 0x2580, 0x2588, 0x2584, 0x2584, 0x2588),
    (0x258C, 0x2588, 0x2588, 0x2590, 0x2588, 0x258C, 0x2588),
    (0x258C, 0x2588, 0x2588, 0x2590, 0x2588, 0x258C, 0x2588),
)
EMPTY_OK = frozenset((0x20, 0x2800))
COMPARISON_NOTE = (
    "Unequal samples are a design delta between two rasters. "
    "They are not an error, a similarity score, a winner, or quality parity."
)
MATCH_NOTE = "Identical cell masks. This is not a quality ranking or quality parity."
SEAM_NOTE = (
    "Edge contact means ink anywhere on a named side. A connected join needs "
    "ink on both sides of one shared boundary row or column. A fixture glyph "
    "with no mask is unchecked, not a gap. A connected or byte-equal subset "
    "is not quality parity."
)


def edges_from_mask(mask, width, height):
    edges = ""
    if any(mask[y * width] for y in range(height)):
        edges += "L"
    if any(mask[y * width + width - 1] for y in range(height)):
        edges += "R"
    if any(mask[x] for x in range(width)):
        edges += "T"
    if any(mask[(height - 1) * width + x] for x in range(width)):
        edges += "B"
    return edges


def insets(mask, width, height):
    if not any(mask):
        return None
    left = next(x for x in range(width) if any(mask[y * width + x] for y in range(height)))
    right = next(x for x in range(width - 1, -1, -1) if any(mask[y * width + x] for y in range(height)))
    top = next(y for y in range(height) if any(mask[y * width + x] for x in range(width)))
    bottom = next(y for y in range(height - 1, -1, -1) if any(mask[y * width + x] for x in range(width)))
    return {"L": left, "R": width - 1 - right, "T": top, "B": height - 1 - bottom}


def source_overhangs(geom, width, height):
    box = geom.get("source")
    if not box:
        return False
    x0, y0, x1, y1 = box
    return x0 < 0 or y0 < 0 or x1 >= width or y1 >= height


def design_comparison(left, right):
    """Describe mask bytes without ranking the designs."""
    result = {
        "pixel_equivalent": False,
        "design_delta_samples": None,
        "design_delta_fraction": None,
        "quality_rank": None,
        "similarity_winner": None,
        "note": "Mask sizes differ. This is not a quality ranking.",
    }
    if len(left) != len(right) or not left:
        return result
    delta = sum(a != b for a, b in zip(left, right))
    result["pixel_equivalent"] = delta == 0
    result["design_delta_samples"] = delta
    result["design_delta_fraction"] = delta / len(left)
    result["note"] = MATCH_NOTE if delta == 0 else COMPARISON_NOTE
    return result


def _touches(mask, width, height, side):
    if side == "R":
        return any(mask[y * width + width - 1] for y in range(height))
    if side == "L":
        return any(mask[y * width] for y in range(height))
    if side == "B":
        return any(mask[(height - 1) * width + x] for x in range(width))
    if side == "T":
        return any(mask[x] for x in range(width))
    raise ValueError("unknown seam side")


def _connected(left, right, width, height, side):
    """True when one shared boundary row or column has ink on both faces."""
    if side == "R":
        return any(left[y * width + width - 1] and right[y * width] for y in range(height))
    if side == "B":
        return any(left[(height - 1) * width + x] and right[x] for x in range(width))
    raise ValueError("unknown seam side")


def seam_report(grid, observed, masks=None, cell=None):
    """Score expected joins from boundary samples.

    ``observed`` is the any-edge-contact map. It does not decide connectivity.
    Without ``masks``, every expected join stays unchecked. A missing mask is
    an unchecked join, not a gap.
    """
    del observed  # edge letters are not a connectivity measurement
    gaps = []
    checked = connected = edge_contact = unchecked = 0
    missing = []
    width = height = None
    if masks is not None:
        if cell is None or len(cell) != 2:
            raise ValueError("seam connectivity needs the cell size")
        width, height = cell
    for y, row in enumerate(grid):
        for x, scalar in enumerate(row):
            here = EXPECT.get(scalar, "")
            pairs = []
            if x + 1 < len(row):
                other = row[x + 1]
                if "R" in here and "L" in EXPECT.get(other, ""):
                    pairs.append(("R", "L", other))
            if y + 1 < len(grid):
                other = grid[y + 1][x]
                if "B" in here and "T" in EXPECT.get(other, ""):
                    pairs.append(("B", "T", other))
            for side, facing, other in pairs:
                left = None if masks is None else masks.get(scalar)
                right = None if masks is None else masks.get(other)
                if left is None or right is None:
                    unchecked += 1
                    for code, sample in ((scalar, left), (other, right)):
                        if sample is None and code not in missing:
                            missing.append(code)
                    continue
                if len(left) != width * height or len(right) != width * height:
                    raise ValueError("seam mask size differs from the cell")
                checked += 1
                if _touches(left, width, height, side) and _touches(right, width, height, facing):
                    edge_contact += 1
                if _connected(left, right, width, height, side):
                    connected += 1
                else:
                    gaps.append({"at": [x, y], "side": side, "scalars": [scalar, other]})
    if grid == SEAM_GRID:
        fixture = "box-light"
    elif grid == BLOCK_GRID:
        fixture = "blocks"
    else:
        fixture = "custom"
    return {
        "fixture": fixture,
        "checked_joins": checked,
        "connected_joins": connected,
        "edge_contact_joins": edge_contact,
        "unchecked_joins": unchecked,
        "missing_scalars": missing,
        "gap_count": len(gaps),
        "gaps": gaps[:16],
        "quality_rank": None,
        "similarity_winner": None,
        "note": SEAM_NOTE,
    }


def _grid_for(corpus):
    if corpus == "box":
        return SEAM_GRID
    if corpus == "blocks":
        return BLOCK_GRID
    return None


def analyze(corpus, cell, scalars, by_engine):
    width, height = cell
    pixels = width * height
    engines = {}
    rows = []
    for name, data in by_engine.items():
        masks, glyphs = data["masks"], data["glyphs"]
        if len(glyphs) != len(scalars) or len(masks) != len(scalars) * pixels:
            raise ValueError("geometry mask count differs from the corpus")
        observed = {}
        seam_masks = {}
        clipped = overhang = 0
        blanks = []
        for index, scalar in enumerate(scalars):
            geom = glyphs[index]
            if geom["scalar"] != scalar:
                raise ValueError("geometry scalar order differs from the corpus")
            mask = masks[index * pixels:(index + 1) * pixels]
            edges = edges_from_mask(mask, width, height)
            if edges != geom["edges"]:
                raise ValueError("worker edge flags differ from the mask")
            observed[scalar] = edges
            seam_masks[scalar] = mask
            clipped += geom["clipped"]
            outside = source_overhangs(geom, width, height)
            overhang += int(outside)
            if geom["ink"] == 0 and scalar not in EMPTY_OK:
                blanks.append(scalar)
            rows.append({
                "engine": name,
                "corpus": corpus,
                "cell": f"{width}x{height}",
                "scalar": scalar,
                "ink": geom["ink"],
                "clipped": geom["clipped"],
                "edges": edges,
                "expected_edges": EXPECT.get(scalar, ""),
                "source_overhang": outside,
                "insets": insets(mask, width, height),
                "advance": geom["advance"],
                "outline": geom["outline"],
            })
        grid = _grid_for(corpus)
        engines[name] = {
            "clipped_ink_samples": clipped,
            "glyphs_with_source_overhang": overhang,
            "blank_glyphs": blanks,
            "seam": seam_report(grid, observed, seam_masks, cell) if grid else None,
            "observed": observed,
        }
    names = list(by_engine)
    comparison = None
    if len(names) == 2:
        comparison = design_comparison(by_engine[names[0]]["masks"], by_engine[names[1]]["masks"])
    summary = {
        "corpus": corpus,
        "cell": [width, height],
        "engines": {
            name: {key: value for key, value in row.items() if key != "observed"}
            for name, row in engines.items()
        },
        "comparison": comparison,
    }
    return summary, rows, {name: engines[name]["observed"] for name in engines}


def paint_grid(masks_by_scalar, grid, cell):
    from PIL import Image
    width, height = cell
    image = Image.new("L", (len(grid[0]) * width, len(grid) * height), 17)
    for y, row in enumerate(grid):
        for x, scalar in enumerate(row):
            mask = masks_by_scalar.get(scalar)
            if mask is None:
                continue
            image.paste(Image.frombytes("L", cell, mask), (x * width, y * height))
    return image


def masks_by_scalar(scalars, masks, cell):
    width, height = cell
    pixels = width * height
    return {scalar: masks[index * pixels:(index + 1) * pixels] for index, scalar in enumerate(scalars)}


def write_pair(directory, label, cell, scalars, by_engine):
    """Save seam or block panels and a design-delta image. Return artifact names."""
    from PIL import Image, ImageChops
    grid_name = None
    names = []
    corpus = label.split("-")[0]
    grid = _grid_for(corpus)
    if grid is None:
        return names
    painted = {}
    for engine, data in by_engine.items():
        lookup = masks_by_scalar(scalars, data["masks"], cell)
        image = paint_grid(lookup, grid, cell)
        painted[engine] = image
        name = f"{label}-{engine}-fixture.png"
        image.save(directory / name)
        (directory / name).chmod(0o600)
        names.append(name)
        grid_name = name
    engines = list(painted)
    if len(engines) == 2:
        left, right = painted[engines[0]], painted[engines[1]]
        delta = ImageChops.difference(left, right).convert("L")
        pair = Image.new("L", (left.width, left.height * 3 + 4), 70)
        pair.paste(left, (0, 0))
        pair.paste(right, (0, left.height + 2))
        pair.paste(delta, (0, left.height * 2 + 4))
        name = f"{label}-design-delta.png"
        pair.save(directory / name)
        (directory / name).chmod(0o600)
        names.append(name)
    return names if grid_name else names
