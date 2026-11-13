"""Pack trained Braille experiments and bind their teacher masks to a manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

from kilix_bench.fonts import digest, parse_cells, revision


def make_gate(names, masks, width=32):
    """An exact bit-7 decision expressed through the existing neural gate ABI."""
    w1 = [0.] * (width * 64)
    w1[7] = 1.
    w2 = [0.] * (2 * width)
    w2[0], w2[width] = -1., 1.
    return {"format": "neuronte-gate", "version": 1, "nets": names,
            "hidden_width": width, "parity_masks": masks, "fallback": 0,
            "training": {"method": "constructed exact bit-7 router, not learned"},
            "layers": {"l1": {"rows": width, "cols": 64, "weight": w1, "bias": [0.] * width},
                       "out": {"rows": 2, "cols": width, "weight": w2, "bias": [.5, -.5]}}}


def remove_ranges(bundle):
    """Force gate evaluation while keeping exactly the same expert bytes."""
    if bundle[:8] != b"NEURONTE":
        raise ValueError("invalid bundle magic")
    version, nets, ranges, fallback, gate = struct.unpack_from("<5I", bundle, 8)
    if version != 2 or nets != 2 or ranges != 2 or not gate:
        raise ValueError("expected two experts, two ranges and a gate")
    end = 28 + nets * 4
    return bundle[:16] + struct.pack("<I", 0) + bundle[20:end] + bundle[end + ranges * 12:]


def target_masks(glyphs, cell, samples=8):
    """The original trainer's FP16 grid and eight-sample binary-area teacher."""
    import numpy as np
    w, h = cell
    n = 256
    xs = (np.arange(w * samples, dtype=np.float32) + .5) / (w * samples)
    ys = (np.arange(h * samples, dtype=np.float32) + .5) / (h * samples) * 2
    xx, yy = np.meshgrid(xs, ys)
    u, v = xx * n, yy * n
    x0 = np.minimum(np.floor(u).astype(int), n - 1)
    y0 = np.minimum(np.floor(v).astype(int), 2 * n - 1)
    fx, fy = u - x0.astype(np.float32), v - y0.astype(np.float32)
    masks = []
    for _, _, shapes in glyphs.charset("braille"):
        field = glyphs.field(shapes, n, glyphs.PEN, .15).astype(np.float16).astype(np.float32)
        top = field[y0, x0] * (1 - fx) + field[y0, x0 + 1] * fx
        bottom = field[y0 + 1, x0] * (1 - fx) + field[y0 + 1, x0 + 1] * fx
        inside = top * (1 - fy) + bottom * fy > 0
        masks.append(inside.reshape(h, samples, w, samples).mean((1, 3)))
    return np.asarray(masks, dtype=np.float32)


def prepare(directory, runtime, cells, seeds):
    import numpy as np
    directory, runtime = Path(directory).resolve(), Path(runtime).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(runtime / "train"))
    import glyphs
    import pack
    models, cases = {}, []
    def save(name, raw):
        path = directory / (name + ".nrt")
        path.write_bytes(raw)
        models[name] = {"file": path.name, "sha256": digest(path), "bytes": len(raw)}
    def case(name, members, widths, routes=None, seed=None, gate_width=0):
        entry = {"name": name, "members": members, "hidden_widths": widths,
                 "seed": seed, "gate_width": gate_width, "routes": routes,
                 "kind": "parts" if len(members) == 2 else "bundle",
                 "parameters": sum(2*h*h+94*h+1 for h in widths) + (gate_width*67+2 if gate_width else 0)}
        cases.append(entry)
    baseline_path = runtime / "weights/neuronte-braille-v0.json"
    baseline = json.loads(baseline_path.read_text())
    if baseline["shape"]["hidden_width"] != 48:
        raise ValueError("expected the existing width-48 Braille baseline")
    save("braille48", pack.pack(str(baseline_path)))
    save("split-v5", (runtime / "weights/neuronte-split-v5.nrt").read_bytes())
    case("braille48", ["braille48"], [48], [0] * 256, seed=0)
    case("split-v5", ["split-v5"], [48], [5] * 256, seed=0)
    cases[-1]["parameters"] = 392716
    cases[-1]["active_expert_parameters"] = 9121
    source_models = {"braille48": {"file": baseline_path.name, "sha256": digest(baseline_path),
                                   "training": baseline["training"], "shape": baseline["shape"]}}
    for seed in seeds:
        files = {}
        for part in ("lower", "upper", "left", "right"):
            name = f"{part}-s{seed}"
            path = directory / (name + ".json")
            doc = json.loads(path.read_text())
            experiment = doc["training"]["experiment"]
            if (doc["shape"]["hidden_width"] != 24 or experiment["part"] != part or
                    experiment["seed"] != seed or doc["shape"]["tau"] != (.30 if part in ("left", "right") else .15)):
                raise ValueError("unexpected training model shape or identity")
            files[part] = path
            source_models[name] = {"file": path.name, "sha256": digest(path),
                                  "training": doc["training"], "shape": doc["shape"]}
            save(name, pack.pack(str(path)))
        pair = [str(files[p]) for p in ("lower", "upper")]
        bundle, ranges = pack.bundle(pair)
        if ranges != [[0x2800, 0x287f, 0], [0x2880, 0x28ff, 1]]:
            raise ValueError("unexpected repertoire partition")
        table_name, gate_name = f"table24-s{seed}", f"gate24-s{seed}"
        save(table_name, bundle)
        doc = json.loads(files["lower"].read_text())
        gate = make_gate([pack.scalars(p)[0] for p in pair], doc["shape"]["parity_masks"])
        gate_path = directory / f"router-s{seed}.json"
        gate_path.write_text(json.dumps(gate) + "\n")
        image, _ = pack.bundle(pair, str(gate_path))
        save(gate_name, remove_ranges(image))
        routes = [0] * 128 + [1] * 128
        case(table_name, [table_name], [24, 24], routes, seed)
        case(gate_name, [gate_name], [24, 24], routes, seed, gate_width=32)
        case(f"parts24-s{seed}", [f"left-s{seed}", f"right-s{seed}"], [24, 24], seed=seed)
    targets = {}
    for cell in cells:
        name = f"truth-{cell[0]}x{cell[1]}.npy"
        path = directory / name
        np.save(path, target_masks(glyphs, cell), allow_pickle=False)
        targets[f"{cell[0]}x{cell[1]}"] = {"file": name, "sha256": digest(path)}
    manifest = {"schema": "kilix-bench.braille-experiment/v1", "runtime_revision": revision(runtime),
                "corpus": list(range(0x2800, 0x2900)), "models": models, "cases": cases,
                "training_models": source_models, "targets": targets,
                "teacher": {"grid": 256, "grid_dtype": "float16", "supersampling": 8,
                            "tau": .15, "dot_radius": glyphs.BRAILLE_R,
                            "glyphs_sha256": digest(runtime / "train/glyphs.py")},
                "preparer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--neuronte-home", required=True, type=Path)
    parser.add_argument("--cells", type=parse_cells, default=parse_cells("8x16,16x32,48x96"))
    parser.add_argument("--seeds", default="0,1")
    args = parser.parse_args()
    print(prepare(args.directory, args.neuronte_home, args.cells, [int(s) for s in args.seeds.split(",")]))


if __name__ == "__main__":
    main()
