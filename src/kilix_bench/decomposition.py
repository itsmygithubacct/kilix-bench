"""Shared native protocol for Braille and ASCII decomposition manifests."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import re
import statistics
import tempfile

from kilix_bench import fonts
from kilix_bench.braille_layouts import LAYOUTS, tiles
from kilix_bench.neuronte import CASES, draw_reference
from kilix_bench.report import scenario


def read_experiment(path, *, family="braille"):
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text())
    corpus = data.get("corpus") if family == "expert" else list(range(32, 127)) if family == "ascii" else list(range(0x2800, 0x2900))
    if (not isinstance(corpus, list) or not 1 <= len(corpus) <= 4096 or
            any(type(c) is not int or not 0 <= c <= 0x10ffff or 0xd800 <= c <= 0xdfff for c in corpus) or
            corpus != sorted(set(corpus))):
        raise ValueError("expected an ordered, distinct Unicode scalar corpus")
    if family not in ("ascii", "braille", "expert") or data.get("schema") != f"kilix-bench.{family}-experiment/v1" or data.get("corpus") != corpus:
        raise ValueError("expected a complete " + family + " experiment manifest")
    paths = {"experiment": path}
    for prefix, entries in (("model", data["models"]), ("target", data["targets"])):
        for name, entry in entries.items():
            asset = (path.parent / entry["file"]).resolve()
            if not asset.is_relative_to(path.parent) or fonts.digest(asset) != entry["sha256"]:
                raise ValueError("experiment asset path or SHA256 mismatch")
            paths[prefix + ":" + name] = asset
    names = [case["name"] for case in data["cases"]]
    if not names or len(names) != len(set(names)) or any(not re.fullmatch(r"[a-z0-9-]+", n) for n in names):
        raise ValueError("invalid or duplicate experiment case")
    for case in data["cases"]:
        layout = case.get("layout", "columns")
        if case["kind"] not in (("bundle", "parts") if family == "braille" else ("bundle",)) or layout not in LAYOUTS:
            raise ValueError("invalid experiment kind or layout")
        if len(case["members"]) != (len(LAYOUTS[layout]) if case["kind"] == "parts" else 1):
            raise ValueError("invalid experiment member count")
        if any(name not in data["models"] for name in case["members"]):
            raise ValueError("missing experiment model")
        if case["kind"] == "bundle" and (len(case["routes"]) != len(corpus) or any(type(i) is not int or i < 0 for i in case["routes"])):
            raise ValueError("invalid expected experiment routes")
    return data, paths


def inputs(args, *, family="braille"):
    paths = fonts.inputs(args, experts=True)
    data, extra = read_experiment(getattr(args, family + "_experiment"), family=family)
    paths.update(extra)
    for case in data["cases"]:
        if case["kind"] == "parts":
            layout = case.get("layout", "columns")
            for w, h in args.cells:
                tiles(layout, w, h)
    if any(f"{w}x{h}" not in data["targets"] for w, h in args.cells):
        raise ValueError("prepare teacher masks for every requested cell size")
    cases = args.expert_cases.split(",")
    if len(cases) != len(set(cases)) or set(cases) - set(CASES):
        raise ValueError("unknown or duplicate expert case")
    if not 1 <= args.expert_passes <= 1000:
        raise ValueError("expert passes must be 1..1000")
    if len(data["corpus"]) * max(args.expert_passes, 512) > 1000000:
        raise ValueError("experiment batch exceeds one million calls")
    return paths


def quality(coverage, target, codes):
    import numpy as np
    if (coverage.shape != target.shape or coverage.ndim != 3 or len(codes) != len(coverage) or
            not np.isfinite(coverage).all() or not np.isfinite(target).all() or
            np.any((coverage < 0) | (coverage > 1)) or np.any((target < 0) | (target > 1))):
        raise ValueError("invalid quality comparison shape or values")
    actual, wanted = coverage > .5, target > .5
    intersection = (actual & wanted).sum((1, 2))
    union = (actual | wanted).sum((1, 2))
    iou = np.divide(intersection, union, out=np.ones(len(codes), dtype=float), where=union != 0)
    wrong = (actual != wanted).sum((1, 2))
    firm_wrong = ((actual != wanted) & (np.abs(target - .5) >= .1)).sum((1, 2))
    errors = np.abs(coverage - target)
    mae = errors.mean((1, 2))
    nearest, nearest_ok = [], []
    for glyph in coverage:
        nearest.append(int(np.abs(target - glyph).sum((1, 2)).argmin()))
    for i, index in enumerate(nearest):
        nearest_ok.append(bool(np.abs(target[i] - target[index]).sum() < .5))
    return {"exact_glyphs": int((wrong == 0).sum()), "glyphs": len(codes),
            "wrong_pixels": int(wrong.sum()), "iou_mean": float(iou.mean()),
            "iou_min": float(iou.min()), "coverage_mae": float(mae.mean()),
            "coverage_error_p99": float(np.quantile(errors, .99)),
            "firm_wrong_pixels": int(firm_wrong.sum()),
            "firm_exact_glyphs": int((firm_wrong == 0).sum()),
            "nearest_ok": sum(nearest_ok),
            "all_binary_exact": bool((wrong == 0).all()),
            "per_glyph": [{"scalar": code, "iou": float(iou[i]), "wrong_pixels": int(wrong[i]),
                           "firm_wrong_pixels": int(firm_wrong[i]),
                           "nearest_scalar": codes[nearest[i]], "nearest_ok": nearest_ok[i],
                           "coverage_mae": float(mae[i])} for i, code in enumerate(codes)]}


def measure(args, *, family="braille"):
    import numpy as np
    paths = inputs(args, family=family)
    manifest, _ = read_experiment(paths["experiment"], family=family)
    if args.cpu is not None:
        if args.cpu not in os.sched_getaffinity(0):
            raise ValueError("requested CPU is outside the allowed affinity")
        os.sched_setaffinity(0, {args.cpu})
    artifacts = Path(os.environ["KILIX_BENCH_ARTIFACT_DIR"])
    rng = random.Random(args.seed)
    codes = list(manifest["corpus"])
    count = len(codes)
    positions = {code: i for i, code in enumerate(codes)}
    rng.shuffle(codes)
    quality_only = getattr(args, "experiment_quality_only", False)
    repeats, warmup = (0, 0) if quality_only else (args.repeats, args.warmup)
    result = {
        "tool": "kilix-bench", "tool_version": "0.2.3", "subject": "kilix-" + family + ("-quality" if quality_only else ""),
        "subject_version": "1", "subject_revision": fonts.revision(Path(__file__).parent),
        "started_utc": datetime.now(timezone.utc).isoformat(), "size": args.cells[0],
        "python": platform.python_version(), "machine": platform.machine(),
        "note": f"All {count} {family} characters. Coverage and draw: ms/glyph, full-batch means. "
                "Opening: ms/model set. Routes: ms/selection. Tiled coverage includes "
                "scalar mapping, every tile inference and joining. Quality failures are retained.",
        "assets": {key: {"name": p.name, "sha256": fonts.digest(p), "bytes": p.stat().st_size}
                   for key, p in paths.items()},
        "model_cases": manifest["cases"], "teacher": manifest["teacher"],
        "runtime_revision": manifest["runtime_revision"], "seed": args.seed,
        "repeats": repeats, "warmup": warmup, "cells": args.cells, "quality_only": quality_only,
        "corpus": codes, "expert_passes": args.expert_passes, "hardware_start": fonts.hardware(),
        "process_policy": "fresh process/handles each job; OS page cache retained; loading, first "
                          "coverage and priming excluded from glyph batches; all cases shuffled each round",
        "cache_policy": "coverage bypasses runtime caches; cold draw starts empty but may hit "
                        "within the batch; canonical pattern counts per handle are recorded in model_cases; "
                        "warm draw primes all patterns; public cache counters unavailable",
        "sampling_policy": "automatic samples from FULL glyph width, including every tile",
        "startup_policy": "no desktop imports, background preparation or persisted atlas; "
                          "opening includes every handle and coverage buffer, excludes glyph inference",
        "job_timeout_seconds": args.job_timeout,
        "quality": [], "trials": [], "scenarios": [], "artifacts": [],
        "harness_sources": {name: fonts.digest(Path(__file__).with_name(name)) for name in
                            ("decomposition.py", "braille.py", "ascii.py", "braille_layouts.py", "fonts.py")},
    }
    modes = args.expert_cases.split(",")
    jobs = []
    for case in ([] if quality_only else manifest["cases"]):
        jobs.append((case, args.cells[0], "open"))
        if case["kind"] == "bundle":
            jobs.append((case, (2, 2), "route-bench"))
        jobs.extend((case, cell, mode) for cell in args.cells for mode in modes)
    expected = {}
    journal = artifacts / "trials.jsonl"
    with tempfile.TemporaryDirectory(prefix=family + "-worker-") as temporary, journal.open("w") as stream:
        journal.chmod(0o600)
        binary, result["build"] = fonts.build_worker(temporary)
        def run(case, cell, mode, calls):
            local = argparse.Namespace(**vars(args))
            local.raster_calls = calls
            models = [paths["model:" + name] for name in case["members"]]
            return fonts.run_worker(binary, {**paths, "weights": models[0]}, "neuronte", "nrt-" + mode,
                                    cell, codes, local, temporary, part_weights=models[1:] if len(models) > 1 else None,
                                    part_layout=case.get("layout", "columns"))
        for case in manifest["cases"]:
            name = case["name"]
            if case["kind"] == "bundle":
                row, _ = run(case, (2, 2), "routes", count)
                if row["routes"] != [case["routes"][positions[c]] for c in codes]:
                    raise ValueError("experiment routes do not match the declared partition")
            for cell in args.cells:
                validation, masks = run(case, cell, "validate", count)
                w, h = cell
                coverage = np.frombuffer(masks, dtype=np.float32).reshape(count, h, w)
                target = np.load(paths[f"target:{w}x{h}"], allow_pickle=False)
                if target.shape != (count, h, w) or not np.isfinite(target).all() or np.any((target < 0) | (target > 1)):
                    raise ValueError("invalid teacher masks")
                score = quality(coverage, target[[positions[c] for c in codes]], codes)
                score.update(model=name, cell=list(cell), coverage_sha256=validation["output_sha256"])
                result["quality"].append(score)
                last = masks[-w * h * 4:]
                expected[name, cell, "coverage"] = last
                for mode in ("draw-cold", "draw-warm"):
                    expected[name, cell, mode] = draw_reference(last)
                preview = artifacts / f"{name}-{w}x{h}.png"
                fonts.preview(np.floor(np.clip(coverage, 0, 1) * 255 + .5).astype(np.uint8).tobytes(),
                              count, cell, preview)
                result["artifacts"].append({"name": preview.name, "sha256": fonts.digest(preview)})
        quality_path = artifacts / "quality.json"
        quality_path.write_text(json.dumps(result["quality"], indent=2) + "\n")
        quality_path.chmod(0o600)
        for repeat in range(-warmup, repeats):
            order = list(jobs)
            rng.shuffle(order)
            for order_index, (case, cell, mode) in enumerate(order):
                calls = count * (512 if mode == "route-bench" else 1 if mode in ("open", "draw-cold") else args.expert_passes)
                row, raw = run(case, cell, mode, calls)
                if mode == "open":
                    duration, cpu, denominator = row["initialize_ns"], row["initialize_cpu_ns"], 1
                else:
                    duration, cpu, denominator = row["elapsed_ns"], row["cpu_ns"], calls
                    if row["calls"] != calls:
                        raise ValueError("incomplete decomposition batch")
                    if mode == "route-bench":
                        if row["route_checksum"] != sum(case["routes"]) * 512:
                            raise ValueError("routing checksum mismatch")
                    else:
                        factor = len(case["members"])
                        direct = mode == "coverage"
                        if (raw != expected[case["name"], cell, mode] or
                                row["coverage_calls"] != (calls * factor if direct else 0) or
                                row["draw_calls"] != (0 if direct else calls * factor) or
                                row["priming_calls"] != (0 if mode == "draw-cold" else count)):
                            raise ValueError("Decomposition pixels, API calls or cache setup differ from protocol")
                row.update(model=case["name"], cell=list(cell), case=mode, repeat=repeat, order=order_index,
                           ns_per_operation=duration / denominator, cpu_ns_per_operation=cpu / denominator)
                result["trials"].append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
        for case, cell, mode in jobs:
            rows = [r for r in result["trials"] if r["repeat"] >= 0 and
                    (r["model"], tuple(r["cell"]), r["case"]) == (case["name"], cell, mode)]
            unit = "ms/model set" if mode == "open" else "ms/route" if mode == "route-bench" else "ms/glyph"
            result["scenarios"].append(scenario(
                f"{case['name']}/{cell[0]}x{cell[1]}/{mode}", unit,
                [r["ns_per_operation"] / 1e9 for r in rows],
                {"model": case["name"], "cell": cell, "case": mode, "unit": unit,
                 "median_ns_per_operation": statistics.median(r["ns_per_operation"] for r in rows),
                 "median_cpu_ms": statistics.median(r["cpu_ns_per_operation"] for r in rows) / 1e6}))
    csv_path = artifacts / "trials.csv"
    with csv_path.open("w", newline="") as stream:
        csv_path.chmod(0o600)
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in result["trials"]))))
        writer.writeheader()
        writer.writerows(result["trials"])
    for path in (journal, csv_path, quality_path):
        result["artifacts"].append({"name": path.name, "sha256": fonts.digest(path)})
    result["hardware_end"] = fonts.hardware()
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return result
