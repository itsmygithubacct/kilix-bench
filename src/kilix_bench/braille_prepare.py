"""Pack trained Braille experiments and bind their teacher masks to a manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

from kilix_bench.fonts import digest, parse_cells, revision
from kilix_bench.braille_layouts import LAYOUTS, REGIONS, WIDTHS


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


def prepare(directory, runtime, cells, seeds, spatial=False, fine=False):
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
    def case(name, members, widths, routes=None, seed=None, gate_width=0, layout=None):
        entry = {"name": name, "members": members, "hidden_widths": widths,
                 "seed": seed, "gate_width": gate_width, "routes": routes,
                 "kind": "parts" if len(members) > 1 else "bundle",
                 "parameters": sum(2*h*h+94*h+1 for h in widths) + (gate_width*67+2 if gate_width else 0)}
        cases.append(entry)
        if layout:
            entry["layout"] = layout
            entry["regions"] = [{"part": part, "mask": REGIONS[part][0],
                                  "x_domain": list(REGIONS[part][1:3]),
                                  "y_domain": list(REGIONS[part][3:5])} for part in LAYOUTS[layout]]
            entry["patterns_per_member"] = [1 << REGIONS[part][0].bit_count() for part in LAYOUTS[layout]]
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
    layouts = tuple(LAYOUTS) if fine else ("columns", "rows", "quadrants") if spatial else ("columns",)
    widths = {part: WIDTHS[layout] for layout in layouts for part in LAYOUTS[layout]}
    for seed in seeds:
        files = {}
        parts = list(widths) if spatial or fine else ["lower", "upper", "left", "right"]
        for part in parts:
            width = widths.get(part, 24)
            region = REGIONS.get(part, (0xff, 0., 1., 0., 2.))
            tau = .15 / (region[2] - region[1])
            name = f"{part}-s{seed}"
            path = directory / (name + ".json")
            doc = json.loads(path.read_text())
            experiment = doc["training"]["experiment"]
            if (doc["shape"]["hidden_width"] != width or experiment["part"] != part or
                    experiment["seed"] != seed or doc["shape"]["tau"] != tau or
                    experiment["x_domain"] != list(region[1:3]) or experiment["y_domain"] != list(region[3:5])):
                raise ValueError("unexpected training model shape or identity")
            if part in REGIONS:
                expected = {0x2800 | bits for bits in range(256) if not bits & ~region[0]}
                if pack.scalars(str(path))[1] != expected:
                    raise ValueError("unexpected canonical training repertoire")
            files[part] = path
            source_models[name] = {"file": path.name, "sha256": digest(path),
                                  "training": doc["training"], "shape": doc["shape"]}
            save(name, pack.pack(str(path)))
        if not spatial and not fine:
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
        for layout in layouts:
            width = WIDTHS[layout]
            name = "parts24" if layout == "columns" else (f"{layout}{width}" if layout in ("rows", "quadrants") else f"{layout}-h{width}")
            members = [f"{part}-s{seed}" for part in LAYOUTS[layout]]
            case(f"{name}-s{seed}", members, [width] * len(members), seed=seed, layout=layout)
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


def prepare_production(directory, runtime, cells, entries):
    """An experiment of already packed files, measured through the public API.

    entries are (name, path, route): each file is copied unchanged and must
    send every Braille scalar to member `route` by its own range table (0
    for a standalone net). Members are version 1 or version 6 nets."""
    import numpy as np
    from kilix_bench.braille_reference import Packed, Tiled
    directory, runtime = Path(directory).resolve(), Path(runtime).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(runtime / "train"))
    import glyphs
    models, cases = {}, []
    for name, source, route in entries:
        raw = Path(source).read_bytes()
        packed = Packed(raw)
        if any(packed.route(c) != route for c in range(0x2800, 0x2900)):
            raise ValueError(f"{name}: Braille does not route to member {route}")
        net = packed.members[route]
        nets = net.nets if isinstance(net, Tiled) else [net]
        widths = [n.hidden for n in nets]
        path = directory / (name + ".nrt")
        path.write_bytes(raw)
        models[name] = {"file": path.name, "sha256": digest(path), "bytes": len(raw),
                        "source_name": Path(source).name}
        entry = {"name": name, "members": [name], "kind": "bundle", "production": True,
                 "format_version": packed.version, "routes": [route] * 256, "seed": None,
                 "gate_width": 0, "hidden_widths": widths,
                 "parameters": sum(2*h*h+94*h+1 for h in widths)}
        if isinstance(net, Tiled):
            entry["tiled"] = {"columns": net.xe, "rows": net.ye, "nets": len(net.nets),
                              "tiles": [list(t) for t in net.tiles]}
        cases.append(entry)
    targets = {}
    for cell in cells:
        name = f"truth-{cell[0]}x{cell[1]}.npy"
        path = directory / name
        np.save(path, target_masks(glyphs, cell), allow_pickle=False)
        targets[f"{cell[0]}x{cell[1]}"] = {"file": name, "sha256": digest(path)}
    manifest = {"schema": "kilix-bench.braille-experiment/v1", "runtime_revision": revision(runtime),
                "corpus": list(range(0x2800, 0x2900)), "models": models, "cases": cases,
                "training_models": {}, "targets": targets,
                "teacher": {"grid": 256, "grid_dtype": "float16", "supersampling": 8,
                            "tau": .15, "dot_radius": glyphs.BRAILLE_R,
                            "glyphs_sha256": digest(runtime / "train/glyphs.py")},
                "preparer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def _production_entry(text):
    name, _, rest = text.partition("=")
    source, _, route = rest.rpartition(":")
    if not name or not source or not route.isdigit():
        raise argparse.ArgumentTypeError("expected NAME=FILE.nrt:ROUTE")
    return name, Path(source), int(route)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--neuronte-home", required=True, type=Path)
    parser.add_argument("--cells", type=parse_cells, default=parse_cells("8x16,16x32,48x96"))
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--spatial", action="store_true",
                        help="compare columns/rows at width 24 and four width-12 quadrants")
    parser.add_argument("--fine", action="store_true", help="also compare 8x6, 12x4, 24x2 and 48x1 spatial experts")
    parser.add_argument("--production", type=_production_entry, action="append", default=[],
                        metavar="NAME=FILE.nrt:ROUTE",
                        help="measure these packed files unchanged through the public API instead of training outputs")
    args = parser.parse_args()
    if args.production:
        print(prepare_production(args.directory, args.neuronte_home, args.cells, args.production))
        return
    print(prepare(args.directory, args.neuronte_home, args.cells, [int(s) for s in args.seeds.split(",")], args.spatial, args.fine))


if __name__ == "__main__":
    main()
