"""Paired CPU font measurements with a shared mask cache and compositor.

The native worker times batches, not individual Python calls. This suite does
not measure shaping, GPU rendering, a live terminal or display latency.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shlex
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone

from kilix_bench import __version__
from kilix_bench.font_coverage import coverage_plan, resolve_faces
from kilix_bench.font_geometry import analyze, write_pair
from kilix_bench.font_profiles import CORPORA, profile
from kilix_bench.presentation import public_text, run as run_presentation
from kilix_bench.report import scenario
from kilix_bench.stages import apply_probe, catalog


CASES = ("load", "raster", "empty-cache", "prefilled-cache", "cache-churn")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_cases(value):
    items = tuple(value.split(","))
    if not items or len(items) != len(set(items)) or any(item not in CASES for item in items):
        raise argparse.ArgumentTypeError("unknown or duplicate font case")
    return items


def parse_optional_cells(value):
    if value == "none":
        return []
    return parse_cells(value)


def parse_cells(value):
    try:
        cells = [tuple(int(part) for part in item.split("x")) for item in value.split(",")]
        if not cells or len(cells) != len(set(cells)):
            raise ValueError
        if any(len(pair) != 2 or not 1 <= pair[0] <= 128 or not 1 <= pair[1] <= 256 for pair in cells):
            raise ValueError
        return cells
    except ValueError:
        raise argparse.ArgumentTypeError("cells must be distinct WxH pairs, at most 128x256") from None


def add_arguments(parser):
    parser.add_argument("--font", default="", help="primary scalable monospace font for FreeType")
    parser.add_argument("--font-license", default="", help="primary font licence file; recorded by hash")
    parser.add_argument("--font-license-name", default="", help="name of the primary font licence")
    parser.add_argument("--font-manifest", default="", help="JSON face list; replaces --font")
    parser.add_argument("--coverage-font", action="append", default=[], help="id=path coverage face")
    parser.add_argument("--coverage-license", action="append", default=[], help="id=path coverage licence")
    parser.add_argument("--coverage-license-name", action="append", default=[], help="id=name coverage licence")
    parser.add_argument("--coverage-pitch", action="append", default=[], help="id=fixed or id=cell")
    parser.add_argument("--neuronte-home", default="", help="Neuronte checkout or installed payload")
    parser.add_argument("--cells", type=parse_cells, default=parse_cells("6x12,8x16,10x20"))
    parser.add_argument("--geometry-cells", type=parse_optional_cells, default=parse_optional_cells("4x6,5x8"))
    parser.add_argument("--geometry", choices=("on", "off"), default="on")
    parser.add_argument("--cases", type=parse_cases, default=parse_cases(",".join(CASES)))
    parser.add_argument("--presentation", choices=("off", "fixture", "live"), default="off")
    parser.add_argument("--corpora", default="ascii", help="comma-separated: " + ",".join(CORPORA))
    parser.add_argument("--raster-calls", type=int, default=95, help="uncached calls per native batch")
    parser.add_argument("--frames", type=int, default=1, help="80x25 frames per draw batch")
    parser.add_argument("--cache-entries", type=int, default=1024)
    parser.add_argument("--churn-entries", type=int, default=16)
    parser.add_argument("--hinting", choices=("native", "none"), default="native")
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--cpu", type=int, default=None, help="pin this benchmark child to one available CPU")
    parser.add_argument("--job-timeout", type=float, default=180, help="seconds allowed for each native batch")


def inputs(args):
    faces = resolve_faces(args)
    home = Path(args.neuronte_home or os.environ.get("NEURONTE_HOME") or
                os.environ.get("NEURONTE") or "~/projects/neuronte").expanduser().resolve()
    paths = {
        "library": Path(os.environ.get("NEURONTE_LIBRARY") or home / "build/libneuronte.so").expanduser().resolve(),
        "weights": Path(os.environ.get("NEURONTE_WEIGHTS") or home / "weights/neuronte-split-v5.nrt").expanduser().resolve(),
        "font": faces[0].path,
        "font_license": faces[0].license_path,
    }
    for key, path in paths.items():
        if path is None or not path.is_file():
            raise ValueError("font suite needs an existing " + key.replace("_", " "))
    if not 1 <= args.repeats <= 10000 or not 0 <= args.warmup <= 100:
        raise ValueError("repeats must be 1..10000; warmup must be 0..100")
    if not 1 <= args.raster_calls <= 1000000 or not 1 <= args.frames <= 500:
        raise ValueError("raster calls must be 1..1000000; frames must be 1..500")
    if not 0 < args.job_timeout <= 86400:
        raise ValueError("job timeout must be positive and at most one day")
    for capacity in (args.cache_entries, args.churn_entries):
        if not 4 <= capacity <= 4096 or capacity % 4:
            raise ValueError("cache capacities must be multiples of four, from 4 to 4096")
    requested = args.corpora.split(",")
    if len(requested) != len(set(requested)):
        raise ValueError("duplicate corpus")
    profile(digest(paths["weights"]), requested)
    return paths, faces


def build_worker(directory):
    """Build outside timed regions; never invoke a shell with compiler flags."""
    source = Path(__file__).with_name("font_worker.c")
    cc = shlex.split(os.environ.get("CC", "cc"))
    flags = ["-O3", "-std=c11", "-Wall", "-Wextra", "-Werror", "-Wpedantic"]
    extra = shlex.split(os.environ.get("CFLAGS", ""))
    pkg = subprocess.check_output(["pkg-config", "--cflags", "--libs", "freetype2"], text=True).strip()
    version = subprocess.check_output(cc + ["--version"], text=True).splitlines()[0]
    binary = Path(directory) / "font-worker"
    command = cc + flags + extra + [str(source), "-o", str(binary)] + shlex.split(pkg) + ["-ldl", "-lm"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    linked = {}
    dependencies = subprocess.run(["ldd", str(binary)], check=True, capture_output=True, text=True)
    for line in dependencies.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0].startswith("libfreetype") and fields[1] == "=>":
            library = Path(fields[2]).resolve()
            linked = {"name": library.name, "sha256": digest(library), "bytes": library.stat().st_size}
    if not linked:
        raise ValueError("could not fingerprint the linked FreeType library")
    return binary, {"source_sha256": digest(source), "binary_sha256": digest(binary),
                    "compiler": version, "flags": flags + extra,
                    "freetype_library": linked,
                    "freetype_pkg_config_version": subprocess.check_output(
                        ["pkg-config", "--modversion", "freetype2"], text=True).strip()}


def with_face(paths, face):
    updated = dict(paths)
    updated["font"] = face.path
    updated["font_license"] = face.license_path
    return updated


def run_worker(binary, paths, engine, mode, cell, scalars, args, directory, capacity=None, pitch="fixed"):
    corpus = Path(directory) / "corpus.txt"
    corpus.write_text("".join(f"{code:x}\n" for code in scalars), encoding="ascii")
    output = Path(directory) / "pixels.bin"
    calls = 80 * 25 * args.frames if mode.startswith("draw-") else args.raster_calls
    command = [str(binary), engine, str(paths["library"]),
               str(paths["weights" if engine == "neuronte" else "font"]), mode,
               *map(str, cell), "0", str(int(args.hinting == "native")),
               str(capacity or args.cache_entries), str(calls), "80", "25", str(corpus), str(output),
               "0" if pitch == "fixed" else "1"]
    start = time.perf_counter_ns()
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=args.job_timeout)
    wall = time.perf_counter_ns() - start
    data = json.loads(completed.stdout)
    if mode == "probe":
        return data, b""
    raw = output.read_bytes()
    if data["output_bytes"] != len(raw):
        raise ValueError("worker output size differs from its record")
    data.update(worker_wall_ns=wall, output_sha256=hashlib.sha256(raw).hexdigest())
    return data, raw


def reference_frame(masks, count, cell, calls, cols=80, rows=25):
    """Independent composition oracle, outside the timed worker."""
    from PIL import Image
    w, h = cell
    pixels = w * h
    lookup = [(233 * a + 17 * (255 - a) + 127) // 255 for a in range(256)]
    glyphs = [Image.frombytes("L", cell, masks[i * pixels:(i + 1) * pixels]).point(lookup)
              for i in range(count)]
    frame = Image.new("L", (cols * w, rows * h), 17)
    start = calls - cols * rows
    for i in range(cols * rows):
        frame.paste(glyphs[(start + i) % count], ((i % cols) * w, (i // cols) * h))
    return frame.convert("RGB").tobytes()


def preview(masks, count, cell, destination):
    from PIL import Image
    w, h = cell
    columns = min(count, 16)
    sheet = Image.new("L", (columns * (w + 2), ((count + columns - 1) // columns) * (h + 2)), 70)
    for i in range(count):
        glyph = Image.frombytes("L", cell, masks[i * w * h:(i + 1) * w * h])
        sheet.paste(glyph, ((i % columns) * (w + 2) + 1, (i // columns) * (h + 2) + 1))
    sheet.save(destination)
    Path(destination).chmod(0o600)


def paired_ratio(neuronte, freetype, seed):
    """Ratio of batch medians with paired-trial bootstrap resampling."""
    if len(neuronte) != len(freetype) or not neuronte or min(freetype) <= 0:
        raise ValueError("invalid paired samples")
    if len(neuronte) < 20:
        return {"ratio": statistics.median(neuronte) / statistics.median(freetype),
                "bootstrap_95_ci": None, "note": "fewer than 20 trial pairs"}
    rng = random.Random(seed)
    count = len(neuronte)
    ratios = []
    for _ in range(2000):
        indices = [rng.randrange(count) for _ in range(count)]
        ratios.append(statistics.median(neuronte[i] for i in indices) /
                      statistics.median(freetype[i] for i in indices))
    ratios.sort()
    return {"ratio": statistics.median(neuronte) / statistics.median(freetype),
            "bootstrap_95_ci": [ratios[49], ratios[1949]], "bootstrap_samples": 2000}


def revision(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else ""


def hardware():
    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.partition(":")[2].strip()
                break
    except OSError:
        pass
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    governors = set()
    for cpu in cpus:
        try:
            governors.add(Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor").read_text().strip())
        except OSError:
            pass
    return {"cpu_model": model, "affinity": cpus, "governors": sorted(governors),
            "kernel": platform.release(), "load_average": list(os.getloadavg())}


def _capture(binary, paths, engine, cell, shared, args, directory, pitch):
    validation, masks = run_worker(
        binary, paths, engine, "validate", cell, shared, args, directory, pitch=pitch)
    glyphs = validation.pop("glyphs", None)
    if glyphs is None or len(glyphs) != len(shared):
        raise ValueError("validation did not record one geometry row per scalar")
    if len(masks) != len(shared) * cell[0] * cell[1]:
        raise ValueError("validation mask size mismatch")
    return validation, masks, glyphs


def _geometry_csv(row):
    insets = row.get("insets") or {}
    return {
        "engine": row["engine"], "corpus": row["corpus"], "cell": row["cell"],
        "scalar": row["scalar"], "ink": row["ink"], "clipped": row["clipped"],
        "edges": row["edges"], "expected_edges": row["expected_edges"],
        "source_overhang": int(bool(row["source_overhang"])),
        "inset_l": insets.get("L", ""), "inset_r": insets.get("R", ""),
        "inset_t": insets.get("T", ""), "inset_b": insets.get("B", ""),
        "advance": row["advance"], "outline": int(bool(row["outline"])),
    }


def _unranked(geometry):
    for item in geometry:
        comparison = item.get("comparison") or {}
        if comparison.get("quality_rank") is not None or comparison.get("similarity_winner") is not None:
            raise ValueError("geometry comparison ranked a font")
        if "ssim" in json.dumps(comparison).lower():
            raise ValueError("geometry comparison invented a similarity score")
        for engine in (item.get("engines") or {}).values():
            seam = (engine or {}).get("seam") or {}
            if seam.get("quality_rank") is not None or seam.get("similarity_winner") is not None:
                raise ValueError("seam report ranked a font")


def measure(args):
    paths, faces = inputs(args)
    if args.cpu is not None:
        if args.cpu not in os.sched_getaffinity(0):
            raise ValueError("requested CPU is outside the allowed affinity")
        os.sched_setaffinity(0, {args.cpu})
    artifacts = Path(os.environ["KILIX_BENCH_ARTIFACT_DIR"])
    profile_name, corpora, unsupported = profile(digest(paths["weights"]), args.corpora.split(","))
    result = {"tool": "kilix-bench", "tool_version": __version__, "subject": "kilix-fonts",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "subject_version": "1", "subject_revision": revision(Path(__file__).parent),
              "python": platform.python_version(), "machine": platform.machine(), "size": [80, 25],
              "note": "CPU terminal-cell comparison. Native batch times include mask normalization. "
                      "Both engines use the same u8 mask cache and CPU compositor. Missing cmap entries "
                      "are excluded instead of timed as fallback glyphs. No shaping, GPU execution, "
                      "compositor present, or scanout. Different font designs are not ranked.",
              "weights_profile": profile_name,
              "assets": {key: {"name": p.name, "sha256": digest(p), "bytes": p.stat().st_size}
                         for key, p in paths.items()},
              "faces": [face.record(digest) for face in faces],
              "neuronte_revision": revision(paths["library"].parent),
              "neuronte_weights_revision": revision(paths["weights"].parent),
              "hardware_start": hardware(), "seed": args.seed, "repeats": args.repeats, "warmup": args.warmup,
              "cells": args.cells, "hinting": args.hinting, "job_timeout_seconds": args.job_timeout,
              "cache_policy": "shared four-way set-associative LRU",
              "process_policy": "fresh native process per trial; OS file cache is not dropped; "
                                "raster primes one corpus; draw-warm primes the shared cache; draw-cold leaves it empty",
              "freetype_size_request": "FT_SIZE_REQUEST_TYPE_CELL; uniform scale; native baseline; clipped ink counted",
              "coverage": {}, "geometry": [], "scenarios": [], "ratios": [], "trials": [], "artifacts": []}
    rng = random.Random(args.seed)
    geometry_rows = []
    modes = [("load", "load", args.cache_entries), ("raster", "raster", args.cache_entries),
             ("empty-cache", "draw-cold", args.cache_entries),
             ("prefilled-cache", "draw-warm", args.cache_entries),
             ("cache-churn", "draw-warm", args.churn_entries)]
    modes = [item for item in modes if item[0] in args.cases]
    with tempfile.TemporaryDirectory(prefix="font-worker-") as temporary:
        binary, build = build_worker(temporary)
        result["build"] = build
        probes = {face.id: {} for face in faces}
        fixed = {}
        for corpus_name, codes in corpora.items():
            if corpus_name in unsupported or not codes:
                continue
            for face in faces:
                probe, _ = run_worker(
                    binary, with_face(paths, face), "freetype", "probe", args.cells[0],
                    codes, args, temporary, pitch=face.pitch)
                probes[face.id][corpus_name] = probe["supported"]
                fixed[face.id] = probe["fixed_width"]
                result["freetype_version"] = probe["freetype_version"]
        report, jobs = coverage_plan(corpora, unsupported, faces, probes)
        for entry in report.values():
            for row in entry["faces"]:
                if row["id"] in fixed:
                    row["fixed_width"] = fixed[row["id"]]
        for face_row in result["faces"]:
            if face_row["id"] in fixed:
                face_row["fixed_width"] = fixed[face_row["id"]]
        result["coverage"] = report
        by_id = {face.id: face for face in faces}
        for corpus_name, face_id, shared in jobs:
            face = by_id[face_id]
            face_paths = with_face(paths, face)
            rng.shuffle(shared)
            result["coverage"][corpus_name]["shared"] = shared
            timed_cells = list(args.cells)
            extra_cells = [cell for cell in args.geometry_cells if cell not in timed_cells]
            for cell in timed_cells + (extra_cells if args.geometry == "on" else []):
                label = f"{corpus_name}-{cell[0]}x{cell[1]}"
                masks, frames, glyphs = {}, {}, {}
                for engine in ("neuronte", "freetype"):
                    pitch = face.pitch if engine == "freetype" else "fixed"
                    validation, masks[engine], glyphs[engine] = _capture(
                        binary, face_paths, engine, cell, shared, args, temporary, pitch)
                    if cell in timed_cells:
                        frames[engine] = reference_frame(masks[engine], len(shared), cell, 2000 * args.frames)
                    name = f"{label}-{engine}.png"
                    preview(masks[engine], len(shared), cell, artifacts / name)
                    result["artifacts"].append({"name": name, "sha256": digest(artifacts / name),
                                                "cell": list(cell), "engine": engine, "corpus": corpus_name,
                                                "face": face.id, "scalars": shared, "validation": validation})
                if args.geometry == "on":
                    summary, rows, _observed = analyze(corpus_name, cell, shared, {
                        engine: {"masks": masks[engine], "glyphs": glyphs[engine]}
                        for engine in masks})
                    summary["comparison_face"] = face.id
                    summary["action"] = result["coverage"][corpus_name]["action"]
                    result["geometry"].append(summary)
                    geometry_rows.extend(_geometry_csv(row) for row in rows)
                    for name in write_pair(artifacts, label, cell, shared, {
                            engine: {"masks": masks[engine]} for engine in masks}):
                        result["artifacts"].append({"name": name, "sha256": digest(artifacts / name),
                                                    "cell": list(cell), "corpus": corpus_name, "kind": "geometry"})
                if cell not in timed_cells:
                    continue
                for case, mode, capacity in modes:
                    samples = {"neuronte": [], "freetype": []}
                    records = {"neuronte": [], "freetype": []}
                    for trial in range(-args.warmup, args.repeats):
                        order = ["neuronte", "freetype"]
                        rng.shuffle(order)
                        for order_index, engine in enumerate(order):
                            pitch = face.pitch if engine == "freetype" else "fixed"
                            row, raw = run_worker(
                                binary, face_paths, engine, mode, cell, shared, args, temporary,
                                capacity, pitch)
                            expected = frames[engine] if mode.startswith("draw-") else masks[engine][
                                ((row["calls"] - 1) % len(shared)) * cell[0] * cell[1]:
                                (((row["calls"] - 1) % len(shared)) + 1) * cell[0] * cell[1]]
                            if raw != expected:
                                raise ValueError(f"{engine} {case}: pixels differ from the independent reference")
                            if mode.startswith("draw-") and row["cache_hits"] + row["cache_misses"] != row["calls"]:
                                raise ValueError("cache accounting mismatch")
                            if trial >= 0:
                                measured = row["elapsed_ns"]
                                if mode == "load":
                                    measured += row["initialize_ns"] + row["first_glyph_ns"]
                                row.update(engine=engine, corpus=corpus_name, cell=cell, case=case, trial=trial,
                                           order_in_pair=order_index, cache_entries=capacity,
                                           measured_ns=measured, ns_per_glyph=row["elapsed_ns"] / row["calls"])
                                result["trials"].append(row)
                                records[engine].append(row)
                                samples[engine].append(measured / 1e9)
                    for engine in samples:
                        rows = records[engine]
                        result["scenarios"].append(scenario(f"{label}/{case}/{engine}",
                            ("engine initialization + first glyph + first corpus" if mode == "load" else
                             "one native batch; quantiles describe batch times, not individual-glyph tail latency"),
                            samples[engine], {"calls": rows[0]["calls"], "cache_entries": capacity,
                                "cache_hits": [r["cache_hits"] for r in rows], "cache_misses": [r["cache_misses"] for r in rows],
                                "median_ns_per_glyph": statistics.median(r["ns_per_glyph"] for r in rows)}))
                    result["ratios"].append({
                        "case": f"{label}/{case}", "definition": "Neuronte time / FreeType time",
                        "freetype_face": face.id,
                        "coverage_action": result["coverage"][corpus_name]["action"],
                        **paired_ratio(samples["neuronte"], samples["freetype"], args.seed)})
        probe = run_presentation(args.presentation, Path(temporary) / "presentation")
    if isinstance(probe, dict) and isinstance(probe.get("reason"), str):
        probe["reason"] = public_text(probe["reason"])
    result["presentation"] = probe
    draw_cases = {"empty-cache", "prefilled-cache", "cache-churn"}
    result["stages"] = apply_probe(catalog(
        bool(result["trials"]),
        any(row["case"] in draw_cases for row in result["trials"])), probe)
    result["hardware_end"] = hardware()
    if any(item.get("action") == "timed-against-coverage-face" for item in result["coverage"].values()):
        result["note"] += " A corpus whose primary face has no cmap entries uses one named coverage face."
    _unranked(result["geometry"])
    if not result["trials"]:
        result["note"] += " No shared supported corpus was available; no timings were produced."
    csv_path = artifacts / "trials.csv"
    if result["trials"]:
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result["trials"][0]))
            writer.writeheader()
            writer.writerows(result["trials"])
        csv_path.chmod(0o600)
        result["artifacts"].append({"name": "trials.csv", "sha256": digest(csv_path)})
    if geometry_rows:
        geometry_path = artifacts / "geometry.csv"
        with geometry_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(geometry_rows[0]))
            writer.writeheader()
            writer.writerows(geometry_rows)
        geometry_path.chmod(0o600)
        result["artifacts"].append({"name": "geometry.csv", "sha256": digest(geometry_path)})
    return result


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    try:
        result = measure(args)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"font benchmark failed: {error}\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
