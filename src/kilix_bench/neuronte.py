"""Complete split-v5 expert batches, with explicit runtime cache states."""

import argparse
from array import array
import csv
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import tempfile

from kilix_bench import fonts
from kilix_bench.font_profiles import EXPERTS, expert_profile
from kilix_bench.report import scenario

CASES = ("coverage", "draw-cold", "draw-warm")


def add_arguments(parser):
    experiments = parser.add_mutually_exclusive_group()
    experiments.add_argument("--braille-experiment", default="",
                        help="SHA256-bound Braille decomposition manifest; compare all its cases")
    experiments.add_argument("--ascii-experiment", default="",
                        help="SHA256-bound ASCII spatial/repertoire manifest; compare all its cases")
    experiments.add_argument("--expert-experiment", default="",
                        help="SHA256-bound decomposition manifest for an explicit expert repertoire")
    experiments.add_argument("--face-experiment", default="",
                        help="SHA256-bound desktop-face manifest: whole faces at their em (kilix_bench.face)")
    parser.add_argument("--experiment-quality-only", action="store_true",
                        help="validate a manifest's routes and quality without running timing jobs")
    parser.add_argument("--experts", default="all", help="all, or comma-separated: " +
                        ",".join(item[0] for item in EXPERTS))
    parser.add_argument("--expert-cases", default=",".join(CASES),
                        help="comma-separated: coverage,draw-cold,draw-warm")
    parser.add_argument("--expert-passes", type=int, default=1,
                        help="complete repertoire passes per coverage/warm batch; cold always uses one")


def inputs(args):
    if getattr(args, "face_experiment", ""):
        from kilix_bench.face import inputs as face_inputs
        return face_inputs(args)
    if getattr(args, "expert_experiment", ""):
        from kilix_bench.decomposition import inputs as experiment_inputs
        return experiment_inputs(args, family="expert")
    if getattr(args, "ascii_experiment", ""):
        from kilix_bench.ascii import inputs as experiment_inputs
        return experiment_inputs(args)
    if args.braille_experiment:
        from kilix_bench.braille import inputs as experiment_inputs
        return experiment_inputs(args)
    if getattr(args, "experiment_quality_only", False):
        raise ValueError("quality-only requires an experiment manifest")
    paths = fonts.inputs(args, experts=True)
    experts = expert_profile(paths["weights"], args.experts)
    cases = args.expert_cases.split(",")
    if len(set(cases)) != len(cases) or set(cases) - set(CASES):
        raise ValueError("unknown or duplicate expert case")
    if not 1 <= args.expert_passes <= 1000:
        raise ValueError("expert passes must be 1..1000")
    if max(len(e["scalars"]) for e in experts.values()) * args.expert_passes > 1000000:
        raise ValueError("expert batch exceeds one million calls")
    return paths


def draw_reference(coverage):
    """White over transparent, from float coverage; no draw API or cache used."""
    values = array("f")
    values.frombytes(coverage)
    pixels = array("I")
    for a in values:
        # Match the float32 public compositor's alpha rounding, including tiny
        # positive coverage with zero rounded alpha and nonzero RGB channels.
        alpha = int(ctypes.c_float(ctypes.c_float(a * 255).value + .5).value)
        pixels.append((alpha << 24) | 0xffffff if a > 0 else 0)
    return pixels.tobytes()


def check_routes(row, expert):
    if row.get("routes") != [expert["index"]] * len(expert["scalars"]):
        raise ValueError("runtime routes disagree with the reviewed expert repertoire")


def check_trial(row, raw, expected, case, count, calls):
    if raw != expected:
        raise ValueError("expert output differs from the validated coverage")
    if row["calls"] != calls:
        raise ValueError("expert batch did not cover all requested corpus passes")
    direct = case == "coverage"
    if row["coverage_calls"] != (calls if direct else 0) or row["draw_calls"] != (0 if direct else calls):
        raise ValueError("expert API call accounting mismatch")
    if row["priming_calls"] != (0 if case == "draw-cold" else count):
        raise ValueError("expert cache state mismatch")


def measure(args):
    if getattr(args, "face_experiment", ""):
        from kilix_bench.face import measure as face_measure
        return face_measure(args)
    if getattr(args, "expert_experiment", ""):
        from kilix_bench.decomposition import measure as experiment_measure
        return experiment_measure(args, family="expert")
    if getattr(args, "ascii_experiment", ""):
        from kilix_bench.ascii import measure as experiment_measure
        return experiment_measure(args)
    if args.braille_experiment:
        from kilix_bench.braille import measure as experiment_measure
        return experiment_measure(args)
    paths = inputs(args)
    if args.cpu is not None:
        if args.cpu not in os.sched_getaffinity(0):
            raise ValueError("requested CPU is outside the allowed affinity")
        os.sched_setaffinity(0, {args.cpu})
    experts = expert_profile(paths["weights"], args.experts)
    rng = random.Random(args.seed)
    for expert in experts.values():
        rng.shuffle(expert["scalars"])
    artifacts = Path(os.environ["KILIX_BENCH_ARTIFACT_DIR"])
    result = {
        "tool": "kilix-bench", "tool_version": "0.2.3", "subject": "kilix-neuronte",
        "subject_version": "1", "subject_revision": fonts.revision(Path(__file__).parent),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "machine": platform.machine(), "size": args.cells[0],
        "note": "Complete split-v5 expert repertoires. Coverage/draw medians are ms per glyph, "
                "averaged over each complete batch; engine-open is ms per bundle. "
                "Draw uses the runtime cache and clears one ARGB cell per call. "
                "No desktop startup, disk atlas, shaping, GPU or presentation is measured.",
        "assets": {key: {"name": p.name, "sha256": fonts.digest(p), "bytes": p.stat().st_size}
                   for key, p in paths.items()},
        "neuronte_revision": fonts.revision(paths["weights"].parent),
        "harness_sources": {p.name: fonts.digest(p) for p in
                            (Path(__file__), Path(fonts.__file__), Path(__file__).with_name("font_profiles.py"))},
        "hardware_start": fonts.hardware(), "seed": args.seed, "repeats": args.repeats,
        "warmup": args.warmup, "cells": args.cells, "expert_passes": args.expert_passes,
        "job_timeout_seconds": args.job_timeout, "weights_profile": "split-v5",
        "cache_policy": "coverage bypasses both caches; draw uses Neuronte's 1024-entry, four-way "
                        "cache, not the font comparison cache; runtime hit/miss counters unavailable",
        "process_policy": "fresh native process and handle for every batch; OS page cache retained; "
                          "engine opening and first direct coverage excluded from batch time; "
                          "coverage primes a full corpus without a glyph cache; draw-cold leaves the "
                          "runtime cache empty; draw-warm primes a full corpus; priming excluded",
        "startup_policy": "no desktop imports, persisted letter atlas, bytecode preparation or "
                          "background preload; engine-open includes dlopen, model parsing/allocation "
                          "and one coverage buffer, excludes process launch and glyph inference",
        "experts": experts, "validation": [], "scenarios": [], "trials": [], "open_trials": [],
        "artifacts": [],
    }
    cases = args.expert_cases.split(",")
    expected = {}
    jobs = [(name, cell, case) for name in experts for cell in args.cells for case in cases]
    journal = artifacts / "trials.jsonl"
    with tempfile.TemporaryDirectory(prefix="neuronte-worker-") as temporary, journal.open("w") as stream:
        journal.chmod(0o600)
        binary, result["build"] = fonts.build_worker(temporary)
        def run(mode, cell, codes, calls):
            local = argparse.Namespace(**vars(args))
            local.raster_calls = calls
            return fonts.run_worker(binary, paths, "neuronte", "nrt-" + mode,
                                    cell, codes, local, temporary)
        for name, expert in experts.items():
            codes = expert["scalars"]
            route, _ = run("routes", (1, 1), codes, len(codes))
            check_routes(route, expert)
            for cell in args.cells:
                validation, masks = run("validate", cell, codes, len(codes))
                pixels = cell[0] * cell[1]
                if len(masks) != len(codes) * pixels * 4:
                    raise ValueError("incomplete expert coverage validation")
                last = masks[-pixels * 4:]
                expected[name, cell, "coverage"] = last
                for case in ("draw-cold", "draw-warm"):
                    expected[name, cell, case] = draw_reference(last)
                values = array("f")
                values.frombytes(masks)
                # Preview conversion is outside every timed native batch.
                preview = artifacts / f"{name}-{cell[0]}x{cell[1]}.png"
                fonts.preview(bytes(int(min(255, max(0, a * 255 + .5))) for a in values),
                              len(codes), cell, preview)
                result["artifacts"].append({"name": preview.name, "sha256": fonts.digest(preview)})
                result["validation"].append({"expert": name, "cell": cell,
                                             "scalars": len(codes), "coverage_sha256": validation["output_sha256"],
                                             "routes_checked": True})
        for repeat in range(-args.warmup, args.repeats):
            # Rotate all experts, sizes and cache states together to reduce
            # systematic bias from temperature or other work on this machine.
            order = list(jobs)
            rng.shuffle(order)
            open_row, _ = run("open", (1, 1), [32], 1)
            open_row.update(repeat=repeat, case="engine-open")
            result["open_trials"].append(open_row)
            stream.write(json.dumps(open_row) + "\n")
            stream.flush()
            for order_index, (name, cell, case) in enumerate(order):
                codes = experts[name]["scalars"]
                calls = len(codes) * (1 if case == "draw-cold" else args.expert_passes)
                row, raw = run(case, cell, codes, calls)
                check_trial(row, raw, expected[name, cell, case], case, len(codes), calls)
                row.update(expert=name, cell=list(cell), case=case, repeat=repeat, order=order_index,
                           ns_per_glyph=row["elapsed_ns"] / calls, cpu_ns_per_glyph=row["cpu_ns"] / calls)
                result["trials"].append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
        opening = [r["initialize_ns"] / 1e9 for r in result["open_trials"] if r["repeat"] >= 0]
        result["scenarios"].append(scenario("engine-open", "fresh handle; whole bundle; OS file cache retained", opening))
        for name, cell, case in jobs:
            rows = [r for r in result["trials"] if r["repeat"] >= 0 and
                    (r["expert"], tuple(r["cell"]), r["case"]) == (name, cell, case)]
            samples = [r["ns_per_glyph"] / 1e9 for r in rows]
            result["scenarios"].append(scenario(
                f"{name}/{cell[0]}x{cell[1]}/{case}", "mean per glyph in a complete repertoire batch", samples,
                {"expert": name, "cell": cell, "case": case, "unit": "ms/glyph",
                 "corpus_size": len(experts[name]["scalars"]), "calls": rows[0]["calls"],
                 "samples_per_axis": 2 if cell[0] < 24 else 1,
                 "median_cpu_ms_per_glyph": statistics.median(r["cpu_ns_per_glyph"] for r in rows) / 1e6,
                 "median_batch_ms": statistics.median(r["elapsed_ns"] for r in rows) / 1e6,
                 "median_prime_ms": statistics.median(r["prime_ns"] for r in rows) / 1e6}))
    csv_path = artifacts / "trials.csv"
    with csv_path.open("w", newline="") as stream:
        csv_path.chmod(0o600)
        writer = csv.DictWriter(stream, fieldnames=list(result["trials"][0]))
        writer.writeheader()
        writer.writerows(result["trials"])
    for path in (journal, csv_path):
        result["artifacts"].append({"name": path.name, "sha256": fonts.digest(path)})
    result["hardware_end"] = fonts.hardware()
    return result


def main():
    parser = argparse.ArgumentParser()
    fonts.add_arguments(parser)
    add_arguments(parser)
    parser.set_defaults(cells=fonts.parse_cells("8x16,16x32,48x96"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    try:
        result = measure(args)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"Neuronte benchmark failed: {error}\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
