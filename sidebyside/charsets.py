#!/usr/bin/env python3
"""Which characters the three faces have, and which they do not share.

usage: charsets.py [common | dejavu-only | neuronte-only | summary] [--against v2|n014|both] [--plain]

  common          characters Neuronte 2, Neuronte 0.1.4 and DejaVu all have
  dejavu-only     characters DejaVu has and Neuronte does not
  neuronte-only   characters Neuronte has and DejaVu does not
  summary         the counts, block by block (the default)

"Neuronte" in the last two is Neuronte 2's bundle unless --against says
otherwise. "DejaVu" is DejaVu Sans Mono and DejaVu Sans together, the two
files the DejaVu pane draws with. The listing goes to standard output as text, a heading for each
Unicode block; --plain leaves the headings out. Combining marks and
characters with no picture (spaces, format controls) are counted but not
listed, since a terminal cell shows nothing for them.
"""
import sys
import unicodedata

import faces

BLOCKS = [
    (0x0020, 0x007F, "Basic Latin"), (0x00A0, 0x00FF, "Latin-1 Supplement"),
    (0x0100, 0x017F, "Latin Extended-A"), (0x0180, 0x024F, "Latin Extended-B"),
    (0x0250, 0x02AF, "IPA Extensions"), (0x02B0, 0x02FF, "Spacing Modifier Letters"),
    (0x0300, 0x036F, "Combining Diacritical Marks"), (0x0370, 0x03FF, "Greek and Coptic"),
    (0x0400, 0x04FF, "Cyrillic"), (0x0500, 0x052F, "Cyrillic Supplement"),
    (0x0530, 0x058F, "Armenian"), (0x0590, 0x05FF, "Hebrew"), (0x0600, 0x06FF, "Arabic"),
    (0x07C0, 0x07FF, "NKo"), (0x0E00, 0x0E7F, "Thai"), (0x0E80, 0x0EFF, "Lao"),
    (0x10A0, 0x10FF, "Georgian"), (0x1400, 0x167F, "Canadian Aboriginal Syllabics"),
    (0x1680, 0x169F, "Ogham"), (0x1D00, 0x1DBF, "Phonetic Extensions"),
    (0x1E00, 0x1EFF, "Latin Extended Additional"), (0x1F00, 0x1FFF, "Greek Extended"),
    (0x2000, 0x206F, "General Punctuation"), (0x2070, 0x209F, "Superscripts and Subscripts"),
    (0x20A0, 0x20CF, "Currency Symbols"), (0x20D0, 0x20FF, "Combining Marks for Symbols"),
    (0x2100, 0x214F, "Letterlike Symbols"), (0x2150, 0x218F, "Number Forms"),
    (0x2190, 0x21FF, "Arrows"), (0x2200, 0x22FF, "Mathematical Operators"),
    (0x2300, 0x23FF, "Miscellaneous Technical"), (0x2400, 0x243F, "Control Pictures"),
    (0x2440, 0x245F, "Optical Character Recognition"), (0x2460, 0x24FF, "Enclosed Alphanumerics"),
    (0x2500, 0x257F, "Box Drawing"), (0x2580, 0x259F, "Block Elements"),
    (0x25A0, 0x25FF, "Geometric Shapes"), (0x2600, 0x26FF, "Miscellaneous Symbols"),
    (0x2700, 0x27BF, "Dingbats"), (0x27C0, 0x27EF, "Miscellaneous Mathematical Symbols-A"),
    (0x27F0, 0x27FF, "Supplemental Arrows-A"), (0x2800, 0x28FF, "Braille Patterns"),
    (0x2900, 0x297F, "Supplemental Arrows-B"), (0x2980, 0x29FF, "Miscellaneous Mathematical Symbols-B"),
    (0x2A00, 0x2AFF, "Supplemental Mathematical Operators"),
    (0x2B00, 0x2BFF, "Miscellaneous Symbols and Arrows"), (0x2C60, 0x2C7F, "Latin Extended-C"),
    (0x2D00, 0x2D2F, "Georgian Supplement"), (0x2D30, 0x2D7F, "Tifinagh"),
    (0x2E00, 0x2E7F, "Supplemental Punctuation"), (0x4DC0, 0x4DFF, "Yijing Hexagram Symbols"),
    (0xA4D0, 0xA4FF, "Lisu"), (0xA640, 0xA69F, "Cyrillic Extended-B"),
    (0xA700, 0xA71F, "Modifier Tone Letters"), (0xA720, 0xA7FF, "Latin Extended-D"),
    (0xE000, 0xF8FF, "Private Use Area"), (0xFB00, 0xFB4F, "Alphabetic Presentation Forms"),
    (0xFB50, 0xFDFF, "Arabic Presentation Forms-A"), (0xFE00, 0xFE0F, "Variation Selectors"),
    (0xFE20, 0xFE2F, "Combining Half Marks"), (0xFE70, 0xFEFF, "Arabic Presentation Forms-B"),
    (0xFFF0, 0xFFFF, "Specials"), (0x10300, 0x1032F, "Old Italic"),
    (0x1CC00, 0x1CEBF, "Symbols for Legacy Computing Supplement"),
    (0x1D300, 0x1D35F, "Tai Xuan Jing Symbols"), (0x1D400, 0x1D7FF, "Mathematical Alphanumeric Symbols"),
    (0x1EE00, 0x1EEFF, "Arabic Mathematical Alphabetic Symbols"), (0x1F000, 0x1F02F, "Mahjong Tiles"),
    (0x1F030, 0x1F09F, "Domino Tiles"), (0x1F0A0, 0x1F0FF, "Playing Cards"),
    (0x1F300, 0x1F5FF, "Miscellaneous Symbols and Pictographs"), (0x1F600, 0x1F64F, "Emoticons"),
    (0x1FB00, 0x1FBFF, "Symbols for Legacy Computing"),
    (0x100000, 0x10FFFF, "Supplementary Private Use Area-B"),
]


def block(scalar):
    for lo, hi, name in BLOCKS:
        if lo <= scalar <= hi:
            return lo, name
    return scalar & ~0xFF, f"U+{scalar & ~0xFF:04X} to U+{scalar | 0xFF:04X}"


def shown(scalar):
    """Whether a terminal cell would show a picture for it."""
    cat = unicodedata.category(chr(scalar))
    return cat[0] != "M" and cat not in ("Cc", "Cf", "Cs", "Zs", "Zl", "Zp")


def sets(against="v2"):
    v2, n014, dejavu = (faces.face(w).has() for w in faces.ORDER)
    neuronte = {"v2": v2, "n014": n014, "both": v2 | n014}[against]
    return {"common": v2 & n014 & dejavu, "dejavu-only": dejavu - neuronte,
            "neuronte-only": neuronte - dejavu, "v2": v2, "n014": n014, "dejavu": dejavu}


def by_block(scalars):
    out = {}
    for s in sorted(scalars):
        out.setdefault(block(s), []).append(s)
    return sorted(out.items())


def listing(scalars, plain=False):
    lines = []
    for (lo, name), members in by_block(scalars):
        chars = [chr(s) for s in members if shown(s)]
        if not chars:
            continue
        if not plain:
            lines.append(f"{name} ({len(chars)})")
        lines.append("".join(chars) if plain else " ".join(chars))
        if not plain:
            lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def summary(s):
    lines = [f"Neuronte 2 bundle      {len(s['v2']):6d} characters",
             f"Neuronte 0.1.4 bundle  {len(s['n014']):6d}",
             f"DejaVu Sans Mono + Sans{len(s['dejavu']):6d}",
             f"all three have         {len(s['common']):6d}",
             f"DejaVu only            {len(s['dejavu-only']):6d}",
             f"Neuronte only          {len(s['neuronte-only']):6d}", "",
             f"{'block':44s} {'common':>7s} {'DejaVu only':>12s} {'Neuronte only':>14s}"]
    rows = {}
    for col, key in enumerate(("common", "dejavu-only", "neuronte-only")):
        for (lo, name), members in by_block(s[key]):
            rows.setdefault((lo, name), [0, 0, 0])[col] = len(members)
    for (lo, name), counts in sorted(rows.items()):
        lines.append(f"{name[:44]:44s} {counts[0]:7d} {counts[1]:12d} {counts[2]:14d}")
    return "\n".join(lines) + "\n"


def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    what = args[0] if args else "summary"
    against = "v2"
    if "--against" in argv:
        against = argv[argv.index("--against") + 1]
        args = [a for a in args if a != against]
        what = args[0] if args else "summary"
    if what not in ("common", "dejavu-only", "neuronte-only", "summary") or against not in ("v2", "n014", "both"):
        sys.stderr.write(__doc__)
        return 2
    try:
        s = sets(against)
    except faces.Missing as e:
        sys.stderr.write(f"charsets: {e}\n")
        return 1
    sys.stdout.write(summary(s) if what == "summary" else listing(s[what], "--plain" in argv))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
