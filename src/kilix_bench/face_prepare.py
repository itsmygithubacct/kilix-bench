"""Prepare a face experiment: the drawing's own pixels as teacher, and SHA256-bound faces.

usage: python3 -m kilix_bench.face_prepare NEW_DIR --neuronte-home CHECKOUT --face ui24
           [--italic] --case NAME=FILE.nrt [--case ...] [--meta NAME=JSON] [--ems 33,48,64]

A pixel face (font/ui-EM.txt or font/ui-EM-italic.txt in the checkout) is read
here, from its text: each character's rows of . and #, placed from the cell row
that font/ui-tops.json gives for that em, at the pen. Nothing of Neuronte's own
code is used for it. The teacher at the face's em is those pixels, binary.

The scalable face (--face uilarge) has no pixels. Its teacher at each em is the
drawing's true coverage, the fraction of each pixel inside the ink from 8 x 8
samples of the checkout's own target field (train/glyphs.py, run in a separate
process), the same truth the face was scored against when it was trained.

Every case is copied into NEW_DIR and bound by SHA256. The directory must not
exist: an earlier experiment, its teacher and its results are never replaced.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from kilix_bench.face import SCHEMA, FaceFile
from kilix_bench.fonts import digest, revision

LARGE_TEACHER = r"""
import json, math, sys
import numpy as np
sys.path.insert(0, sys.argv[1] + "/train")
import glyphs
ems = [int(e) for e in sys.argv[3].split(",")]
rows = glyphs.charset("uilarge")
pen = glyphs.PENS["uilarge"]
fields = [glyphs.field(s, 256, pen, .15) for _, _, s in rows]
n = 256
out = {}
for em in ems:
    t = 8
    xs = (np.arange(em * t) + .5) / (em * t) * n
    ys = (np.arange(2 * em * t) + .5) / (2 * em * t) * 2 * n
    ix, iy = np.minimum(xs.astype(int), n - 1), np.minimum(ys.astype(int), 2 * n - 1)
    fx, fy = xs - ix, ys - iy
    cells = []
    for f in fields:
        a = f[iy[:, None], ix] * (1 - fx) + f[iy[:, None], ix + 1] * fx
        b = f[iy[:, None] + 1, ix] * (1 - fx) + f[iy[:, None] + 1, ix + 1] * fx
        inside = (a * (1 - fy[:, None]) + b * fy[:, None]) > 0
        cov = inside.reshape(2 * em, t, em, t).mean((1, 3))
        cell = np.zeros((2 * em, em + 2))
        cell[:, :em] = cov
        cells.append(cell)
    np.save(sys.argv[2] + f"/teacher-{em}px.npy", np.asarray(cells, dtype=np.float64), allow_pickle=False)
print(json.dumps([ord(c) for c, _, _ in rows]))
"""


def pixel_face(path, em):
    """{scalar: (advance, rows)} read from a pixel face's text."""
    out, current = {}, None
    for number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip("\n")
        if line.startswith("# ") or not line or (line == "#" and current is None):
            continue
        if " " in line:
            name, _, adv = line.partition(" ")
            scalar = int(name[2:], 16) if name.startswith("U+") else ord(name) if len(name) == 1 else None
            if scalar is None or scalar in out or not adv.isdigit():
                raise ValueError(f"{path}:{number}: bad character line")
            current = scalar
            out[scalar] = (int(adv), [])
        elif current is None or set(line) - set(".#") or len(line) > em:
            raise ValueError(f"{path}:{number}: bad pixel row")
        else:
            out[current][1].append(line)
    if any(len(rows) != em for _, rows in out.values()):
        raise ValueError(f"{path}: every character needs {em} rows")
    return out


def prepare(directory, runtime, face, italic, cases, meta, large_ems):
    import numpy as np
    directory, runtime = Path(directory).resolve(), Path(runtime).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    sources = {}
    if face == "uilarge":
        if italic:
            raise ValueError("the scalable face has no italic")
        ems, binary = large_ems, False
        for rel in ("font/ui-large.txt", "font/ui-large-advances.txt", "train/glyphs.py", "train/strokes.py"):
            sources[rel] = digest(runtime / rel)
        done = subprocess.run([sys.executable, "-I", "-c", LARGE_TEACHER, str(runtime), str(directory),
                               ",".join(map(str, ems))], check=True, capture_output=True, text=True)
        corpus = sorted(json.loads(done.stdout))
        teacher = {"kind": "true coverage, 8 x 8 samples of the checkout's target field",
                   "script_sha256": __import__("hashlib").sha256(LARGE_TEACHER.encode()).hexdigest()}
    else:
        em = int(face[2:])
        rel = f"font/ui-{em}{'-italic' if italic else ''}.txt"
        sources[rel] = digest(runtime / rel)
        sources["font/ui-tops.json"] = digest(runtime / "font/ui-tops.json")
        top = json.loads((runtime / "font/ui-tops.json").read_text())["tops"][str(em)]
        drawing = pixel_face(runtime / rel, em)
        corpus = sorted(drawing)
        cells = []
        for scalar in corpus:
            _, rows = drawing[scalar]
            cell = np.zeros((2 * em, em + 2))
            for r, row in enumerate(rows):
                for c, v in enumerate(row):
                    if v == "#":
                        cell[top + r, c] = 1.0
            cells.append(cell)
        np.save(directory / f"teacher-{em}px.npy", np.asarray(cells, dtype=np.float64), allow_pickle=False)
        ems, binary = [em], True
        teacher = {"kind": "the drawing's pixels, read from its text", "ui_top": top,
                   "advances": {f"U+{s:04X}": drawing[s][0] for s in corpus}}
    teacher.update(sources_sha256=sources)
    targets = {str(e): {"file": f"teacher-{e}px.npy", "sha256": digest(directory / f"teacher-{e}px.npy")}
               for e in ems}
    models, entries = {}, []
    for name, source in cases:
        source = Path(source).resolve()
        destination = directory / source.name
        if destination.exists():
            raise ValueError("two cases share a file name: " + source.name)
        shutil.copyfile(source, destination)
        packed = FaceFile(destination.read_bytes())
        if packed.advances is None or set(packed.advances) < set(corpus):
            raise ValueError(f"{name}: the drawing file lacks an advance for some character")
        if binary:
            for s in corpus:
                pixels = (packed.advances[s][0] * ems[0] + 50) // 100
                if pixels != teacher["advances"][f"U+{s:04X}"]:
                    raise ValueError(f"{name} U+{s:04X}: packed advance {pixels}, the drawing says "
                                     f"{teacher['advances'][f'U+{s:04X}']}")
        models[name] = {"file": destination.name, "sha256": digest(destination), "bytes": destination.stat().st_size}
        entries.append({"name": name, "model": name, "routes": [packed.route(s) for s in corpus],
                        **meta.get(name, {})})
    manifest = {"schema": SCHEMA, "face": face + ("-italic" if italic else ""), "ems": ems, "binary": binary,
                "corpus": corpus, "models": models, "targets": targets, "teacher": teacher,
                "runtime_revision": revision(runtime), "cases": entries, "preparer_sha256": digest(Path(__file__))}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="new directory on active storage")
    parser.add_argument("--neuronte-home", required=True, type=Path)
    parser.add_argument("--face", required=True, help="uiEM or uilarge")
    parser.add_argument("--italic", action="store_true")
    parser.add_argument("--case", action="append", required=True, help="NAME=FILE.nrt")
    parser.add_argument("--meta", action="append", default=[], help="NAME=JSON object retained in the case")
    parser.add_argument("--ems", default="33,48,64", help="ems for the scalable face")
    args = parser.parse_args()
    cases = [tuple(item.split("=", 1)) for item in args.case]
    meta = {}
    for item in args.meta:
        name, _, value = item.partition("=")
        meta[name] = json.loads(value)
    if args.face != "uilarge" and not (args.face.startswith("ui") and args.face[2:].isdigit()):
        parser.error("face is uiEM or uilarge")
    print(prepare(args.directory, args.neuronte_home, args.face, args.italic, cases, meta,
                  [int(e) for e in args.ems.split(",")]))


if __name__ == "__main__":
    main()
