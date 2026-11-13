"""Declared training coverage, bound to exact packed model bytes.

Routing an arbitrary scalar to a net does not establish glyph support. These
small benchmark corpora are subsets of the published model coverage, not a
claim to enumerate every supported scalar. Colour and proportional UI faces
need their own comparisons and are deliberately not accepted here.
"""

import hashlib
from pathlib import Path
import struct

ASCII = list(range(0x20, 0x7f))
CORPORA = {
    "ascii": ASCII,
    "box": list(range(0x2500, 0x2580)),
    "blocks": list(range(0x2580, 0x25a0)),
    "braille": list(range(0x2800, 0x2900)),
    "latin": [ord(c) for c in "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ"],
    "greek": [c for c in range(0x391, 0x3aa) if c != 0x3a2] + list(range(0x3b1, 0x3ca)),
    "russian": [0x401, 0x451] + list(range(0x410, 0x450)),
    "sextants": list(range(0x1fb00, 0x1fb3c)),
    "octants": list(range(0x1cd00, 0x1cde6)),
}

PROFILES = {
    "8a072c27f1241c536b04ea77d82efdc005399762421f86bd18d950206528ec83":
        ("split-v5", tuple(CORPORA)),
    "1e2d89c4d8cc96e7c23eeec2ef688d31222aeb5e4303ecb2dce0c7c783aa8cf5":
        ("split-v4", tuple(k for k in CORPORA if k not in ("greek", "russian"))),
    "e1d3b97811dba4616dd0c90f12a66ba446c22228854e64e610ebf96c89d92f89":
        ("ascii-v0", ("ascii",)),
}


def profile(digest, requested):
    if digest not in PROFILES:
        raise ValueError("weights have no reviewed coverage profile; add an exact hash and declared corpora")
    name, allowed = PROFILES[digest]
    unknown = set(requested) - set(CORPORA)
    unsupported = set(requested) - set(allowed)
    if unknown:
        raise ValueError("unknown corpora: " + ", ".join(sorted(unknown)))
    return name, {key: CORPORA[key] if key in allowed else [] for key in requested}, unsupported


SPLIT_V5_SHA256 = "8a072c27f1241c536b04ea77d82efdc005399762421f86bd18d950206528ec83"
EXPERTS = (
    ("ascii", "ascii-v0", 128, 95), ("draw", "draw-v2", 192, 131),
    ("symbol", "symbol-v0", 128, 49), ("accent", "accent-v2", 256, 318),
    ("pixel", "pixel-v1", 128, 327), ("braille", "braille-v0", 48, 256),
)


def expert_profile(path, requested="all"):
    """All training labels from the reviewed bundle, never arbitrary gate routes.

    The exact hash binds the table to the six published training repertoires.
    Do not generalize this to an unknown model solely because it has a router.
    """
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SPLIT_V5_SHA256:
        raise ValueError("expert suite requires the reviewed split-v5 SHA256")
    names = [item[0] for item in EXPERTS] if requested == "all" else requested.split(",")
    if len(set(names)) != len(names) or set(names) - {item[0] for item in EXPERTS}:
        raise ValueError("unknown or duplicate expert")
    if raw[:8] != b"NEURONTE" or struct.unpack_from("<5I", raw, 8) != (2, 6, 68, 0, 32):
        raise ValueError("unexpected split-v5 container")
    sizes = struct.unpack_from("<6I", raw, 28)
    codes = [[] for _ in EXPERTS]
    previous = -1
    for offset in range(52, 52 + 68 * 12, 12):
        low, high, index = struct.unpack_from("<3I", raw, offset)
        if not previous < low <= high <= 0x10ffff or index >= 6:
            raise ValueError("invalid expert ranges")
        codes[index].extend(range(low, high + 1))
        previous = high
    offset = 52 + 68 * 12 + 43 * 4 + (32 * 65 + 6 * 33) * 4
    result = {}
    for index, (name, model, width, count) in enumerate(EXPERTS):
        params = 2 * width * width + 94 * width + 1
        if (len(codes[index]) != count or sizes[index] != 204 + params * 4 or
                raw[offset:offset + 8] != b"NEURONTE" or
                struct.unpack_from("<5I", raw, offset + 8) != (1, 6, 64, width, 3)):
            raise ValueError("expert shape or repertoire differs from its profile")
        result[name] = {"index": index, "model": model, "hidden_width": width,
                        "parameters": params, "scalars": codes[index]}
        offset += sizes[index]
    if offset != len(raw):
        raise ValueError("unexpected split-v5 length")
    return {name: result[name] for name in names}
