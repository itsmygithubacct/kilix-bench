"""Declared training coverage, bound to exact packed model bytes.

Routing an arbitrary scalar to a net does not establish glyph support. These
small benchmark corpora are subsets of the published model coverage, not a
claim to enumerate every supported scalar. Colour and proportional UI faces
need their own comparisons and are deliberately not accepted here.
"""

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
