"""Desktop-face experiments: a whole face at its em, through a desktop's public calls.

A manifest (schema kilix-bench.face-experiment/v1) names one face: its corpus
(every character it draws), the ems it is drawn at, whether it is drawn binary
(a pixel face) or blended (a scalable face), SHA256-bound drawing files and the
drawing's own pixels at each em as the teacher. Each case is one drawing file.

Per case and em, before timing: routes; every character's public uncached
coverage and its first draw from an empty draw cache; an independent float64
evaluation of the packed bytes (no Neuronte code) on a spaced sample and both
sides of every route change, within 1e-3; every regular and bold advance
against the packed advance table; every drawn canvas against the coverage
(binary cut at one half, or the float32 compositor); and quality against the
teacher, which is recorded and never removes a case.

Timed jobs, each in a fresh process, shuffled every round: open (dlopen,
open, binary switch, buffers), coverage (public uncached call, whole corpus
passes), draw-cold (a new handle drawing the whole face once, as a desktop
building its letter cells: two text-width calls and one draw per character)
and draw-warm (the whole face primed, then whole passes drawn from the
runtime cache). Opening is reported per face, the others per glyph and per
face (corpus size times the per-glyph mean).
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import shlex
import statistics
import struct
import subprocess
import tempfile
import time

from kilix_bench import fonts
from kilix_bench.braille_reference import _normal, auto_samples, member
from kilix_bench.neuronte import draw_reference
from kilix_bench.report import scenario

SCHEMA = "kilix-bench.face-experiment/v1"
CASES = ("coverage", "draw-cold", "draw-warm")
REFERENCE_TOLERANCE = 1e-3


class FaceFile:
    """A packed face, read from its bytes: the net(s) that draw it and its advances.

    Version 5 drawing files hold version 1/6 members (the first draws every
    scalar) or one version 2 bundle, and at most one version 4 advance table.
    A bundle routes by its range table; scalars outside it are not evaluated.
    """

    def __init__(self, data):
        data = bytes(data)
        if data[:8] != b"NEURONTE":
            raise ValueError("not a Neuronte file")
        self.version = struct.unpack_from("<I", data, 8)[0]
        self.nets, self.ranges, self.advances, self.fallback_scalar = [], [], None, None
        if self.version != 5:
            self._net(data)
            return
        count = struct.unpack_from("<I", data, 12)[0]
        if not 1 <= count <= 16:
            raise ValueError("drawing file member count")
        off = 16 + 8 * count
        for i in range(count):
            at, length = struct.unpack_from("<2I", data, 16 + 8 * i)
            if at != off:
                raise ValueError("drawing file members are not contiguous")
            part = data[at:at + length]
            if struct.unpack_from("<I", part, 8)[0] == 4:
                if self.advances is not None:
                    raise ValueError("two advance tables")
                self._advances(part)
            else:
                self._net(part)
            off += length
        if off != len(data):
            raise ValueError("drawing file length")

    def _net(self, part):
        version = struct.unpack_from("<I", part, 8)[0]
        if version in (1, 6):
            if self.ranges:
                raise ValueError("nets beside a bundle")
            self.nets.append(member(part))
            return
        if version != 2 or self.nets:
            raise ValueError(f"version {version} is not a face member this reference evaluates")
        _, nets, ranges, _, gate = struct.unpack_from("<5I", part, 8)
        lengths = struct.unpack_from(f"<{nets}I", part, 28)
        off = 28 + 4 * nets
        self.ranges = [struct.unpack_from("<3I", part, off + 12 * i) for i in range(ranges)]
        off += 12 * ranges
        if gate:
            off += 4 * (43 + gate * 64 + gate + nets * gate + nets)
        for n in lengths:
            v = struct.unpack_from("<I", part, off + 8)[0]
            self.nets.append(member(part[off:off + n]) if v in (1, 6) else None)
            off += n
        if off != len(part) or not self.ranges:
            raise ValueError("bundle length or empty range table")

    def _advances(self, part):
        _, units, steps, weights, records, fallback = struct.unpack_from("<6I", part, 8)
        if (units, steps, weights) != (10, 10, 2) or len(part) != 32 + 8 * records:
            raise ValueError("advance table layout")
        self.advances = {}
        for i in range(records):
            s, regular, bold = struct.unpack_from("<I2H", part, 32 + 8 * i)
            self.advances[s] = (regular, bold)
        if fallback not in self.advances:
            raise ValueError("advance fallback has no record")
        self.fallback_scalar = fallback

    def route(self, s):
        if not self.ranges:
            return 0
        s = _normal(s)
        for first, last, net in self.ranges:
            if first <= s <= last:
                return net
        raise ValueError(f"U+{s:04X} is outside the face's range table")

    def coverage(self, s, cw, ch):
        net = self.nets[self.route(s)]
        if net is None:
            raise ValueError("the routed member is not a distance or tiled net")
        return net.coverage(s, cw, ch, auto_samples(cw))

    def advance(self, s, bold, em):
        if self.advances is None:
            raise ValueError("the face has no advance table")
        steps = self.advances.get(s, self.advances[self.fallback_scalar])[bold]
        return (steps * em + 50) // 100


def read_experiment(path):
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text())
    corpus, ems = data.get("corpus"), data.get("ems")
    if data.get("schema") != SCHEMA:
        raise ValueError("expected a " + SCHEMA + " manifest")
    if (not isinstance(corpus, list) or not 1 <= len(corpus) <= 4096 or
            any(type(c) is not int or not 0x20 <= c <= 0x10ffff or 0xd800 <= c <= 0xdfff for c in corpus) or
            corpus != sorted(set(corpus))):
        raise ValueError("expected an ordered, distinct printable Unicode scalar corpus")
    if (not isinstance(ems, list) or not ems or ems != sorted(set(ems)) or
            any(type(e) is not int or not 1 <= e <= 128 for e in ems)):
        raise ValueError("expected an ordered list of distinct ems from 1 to 128")
    if type(data.get("binary")) is not bool or not isinstance(data.get("face"), str):
        raise ValueError("expected the face's name and whether it draws binary")
    paths = {"experiment": path}
    for prefix, entries in (("model", data["models"]), ("target", data["targets"])):
        for name, entry in entries.items():
            asset = (path.parent / entry["file"]).resolve()
            if not asset.is_relative_to(path.parent) or fonts.digest(asset) != entry["sha256"]:
                raise ValueError("experiment asset path or SHA256 mismatch")
            paths[prefix + ":" + name] = asset
    if set(data["targets"]) != {str(e) for e in ems}:
        raise ValueError("one teacher target per em")
    names = [case["name"] for case in data["cases"]]
    if not names or len(names) != len(set(names)) or any(not re.fullmatch(r"[a-z0-9-]+", n) for n in names):
        raise ValueError("invalid or duplicate face case")
    for case in data["cases"]:
        if case.get("model") not in data["models"]:
            raise ValueError("missing face model")
    return data, paths


def inputs(args):
    home = Path(args.neuronte_home or os.environ.get("NEURONTE_HOME") or "~/projects/neuronte").expanduser().resolve()
    library = Path(os.environ.get("NEURONTE_LIBRARY") or home / "build/libneuronte.so").expanduser().resolve()
    if not library.is_file():
        raise ValueError("face suite needs an existing library")
    data, paths = read_experiment(args.face_experiment)
    paths["library"] = library
    paths["weights"] = paths["model:" + data["cases"][0]["model"]]
    cases = args.expert_cases.split(",")
    if len(cases) != len(set(cases)) or set(cases) - set(CASES):
        raise ValueError("unknown or duplicate face case")
    if not 1 <= args.expert_passes <= 1000:
        raise ValueError("face passes must be 1..1000")
    if len(data["corpus"]) * args.expert_passes > 1000000:
        raise ValueError("face batch exceeds one million calls")
    if not 1 <= args.repeats <= 10000 or not 0 <= args.warmup <= 100:
        raise ValueError("repeats must be 1..10000; warmup must be 0..100")
    if not 0 < args.job_timeout <= 86400:
        raise ValueError("job timeout must be positive and at most one day")
    return paths


def build_worker(directory):
    """Build outside timed regions; never invoke a shell with compiler flags."""
    source = Path(__file__).with_name("face_worker.c")
    cc = shlex.split(os.environ.get("CC", "cc"))
    flags = ["-O3", "-std=c11", "-Wall", "-Wextra", "-Werror", "-Wpedantic"]
    extra = shlex.split(os.environ.get("CFLAGS", ""))
    version = subprocess.check_output(cc + ["--version"], text=True).splitlines()[0]
    binary = Path(directory) / "face-worker"
    subprocess.run(cc + flags + extra + [str(source), "-o", str(binary), "-ldl", "-lm"],
                   check=True, capture_output=True, text=True)
    return binary, {"source_sha256": fonts.digest(source), "binary_sha256": fonts.digest(binary),
                    "compiler": version, "flags": flags + extra}


def run_worker(binary, library, model, mode, em, binary_mode, scalars, calls, directory, timeout):
    corpus = Path(directory) / "corpus.txt"
    corpus.write_text("".join(f"{code:x}\n" for code in scalars), encoding="ascii")
    output = Path(directory) / "pixels.bin"
    if output.exists():
        output.unlink()
    command = [str(binary), str(library), str(model), mode, str(em), str(int(binary_mode)), str(calls),
               str(corpus), str(output)]
    start = time.perf_counter_ns()
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
    wall = time.perf_counter_ns() - start
    data = json.loads(completed.stdout)
    data["worker_wall_ns"] = wall
    if mode in ("open", "routes"):
        return data, b""
    raw = output.read_bytes()
    if data["output_bytes"] != len(raw):
        raise ValueError("worker output size differs from its record")
    data["output_sha256"] = hashlib.sha256(raw).hexdigest()
    return data, raw


def canvas_reference(coverage, em, binary_mode):
    """The canvas a regular letter drawn at x 0, y 0 leaves, from its float coverage alone."""
    import numpy as np
    cw, ch, width = em, 2 * em, em + 2
    cov = np.asarray(coverage, dtype=np.float32).reshape(ch, cw)
    if binary_mode:
        cut = np.zeros((ch, width), dtype=np.uint32)
        cut[:, :cw] = np.where(cov > .5, np.uint32(0xFFFFFFFF), np.uint32(0))
        return cut.tobytes()
    rows = np.frombuffer(draw_reference(cov.tobytes()), dtype=np.uint32).reshape(ch, cw)
    out = np.zeros((ch, width), dtype=np.uint32)
    out[:, :cw] = rows
    return out.tobytes()


def quality(canvases, target, codes, binary_mode):
    """Drawn letters against the drawing's own pixels at this em (teacher)."""
    import numpy as np
    alpha = ((canvases >> 24) & 0xFF).astype(np.float64) / 255
    if alpha.shape != target.shape:
        raise ValueError("teacher and canvas shapes differ")
    if binary_mode:
        drawn, wanted = alpha > .5, target > .5
        wrong = (drawn != wanted).sum((1, 2))
        return {"glyphs": len(codes), "exact_glyphs": int((wrong == 0).sum()), "wrong_pixels": int(wrong.sum()),
                "all_exact": bool((wrong == 0).all()),
                "differ": [f"U+{c:04X}" for c, w in zip(codes, wrong) if w],
                "per_glyph_wrong_pixels": {f"U+{c:04X}": int(w) for c, w in zip(codes, wrong)}}
    error = np.abs(alpha - target)
    inked = (alpha > 0) | (target > 0)
    return {"glyphs": len(codes), "mean_error_inked": float(error[inked].mean()) if inked.any() else 0.0,
            "largest_error": float(error.max()), "over_quarter": int((error > .25).sum()),
            "over_half": int((error > .5).sum()), "inked_pixels": int(inked.sum())}


def check_reference(face, codes, coverage, em):
    import numpy as np
    ordered = sorted(codes)
    positions = {c: i for i, c in enumerate(codes)}
    selected = {ordered[i] for i in range(0, len(ordered), max(1, len(ordered) // 16))} | {ordered[-1]}
    routes = [face.route(c) for c in ordered]
    for i in range(1, len(ordered)):
        if routes[i] != routes[i - 1]:
            selected.update(ordered[i - 1:i + 1])
    worst = 0.0
    for c in sorted(selected):
        reference = face.coverage(c, em, 2 * em)
        if not np.isfinite(reference).all():
            raise ValueError("nonfinite reference coverage")
        worst = max(worst, float(np.abs(coverage[positions[c]].astype(np.float64) - reference).max()))
    if worst > REFERENCE_TOLERANCE:
        raise ValueError(f"production coverage is {worst:.2e} from the float64 reference at {em} px")
    return {"em": em, "glyphs": len(selected), "scalars": sorted(selected), "format_version": face.version,
            "max_abs_difference": worst}


def measure(args):
    import numpy as np
    paths = inputs(args)
    manifest, _ = read_experiment(paths["experiment"])
    if args.cpu is not None:
        if args.cpu not in os.sched_getaffinity(0):
            raise ValueError("requested CPU is outside the allowed affinity")
        os.sched_setaffinity(0, {args.cpu})
    artifacts = Path(os.environ["KILIX_BENCH_ARTIFACT_DIR"])
    rng = random.Random(args.seed)
    codes = list(manifest["corpus"])
    count = len(codes)
    positions = {c: i for i, c in enumerate(manifest["corpus"])}
    rng.shuffle(codes)
    binary_mode, ems = manifest["binary"], manifest["ems"]
    modes = args.expert_cases.split(",")
    quality_only = getattr(args, "experiment_quality_only", False)
    repeats, warmup = (0, 0) if quality_only else (args.repeats, args.warmup)
    library = paths["library"]
    result = {
        "tool": "kilix-bench", "tool_version": "0.2.3", "subject": "kilix-face" + ("-quality" if quality_only else ""),
        "subject_version": "1", "subject_revision": fonts.revision(Path(__file__).parent),
        "started_utc": datetime.now(timezone.utc).isoformat(), "size": [ems[0], 2 * ems[0]],
        "python": platform.python_version(), "machine": platform.machine(),
        "face": manifest["face"], "ems": ems, "binary": binary_mode, "corpus": codes,
        "note": f"Desktop face {manifest['face']}: all {count} characters at em {ems}. Coverage and draw: "
                "ms/glyph full-batch means and ms/face (x corpus). Opening: ms/face handle. Cold draw is "
                "a fresh handle drawing the whole face once, as a desktop builds its letter cells.",
        "assets": {key: {"name": p.name, "sha256": fonts.digest(p), "bytes": p.stat().st_size}
                   for key, p in paths.items()},
        "model_cases": manifest["cases"], "teacher": manifest["teacher"],
        "runtime_revision": manifest["runtime_revision"], "seed": args.seed, "repeats": repeats,
        "warmup": warmup, "passes": args.expert_passes, "quality_only": quality_only,
        "hardware_start": fonts.hardware(), "job_timeout_seconds": args.job_timeout,
        "call_policy": "per character: neuronte_text_width regular and bold, then neuronte_draw_text of the "
                       "regular letter at x 0, y 0, em px, white, into a cleared (em + 2) x 2 em canvas; "
                       "neuronte_set_binary(1) for a binary face; coverage cases call neuronte_coverage at "
                       "em x 2 em with automatic samples",
        "process_policy": "fresh process and handle every job; OS page cache retained; opening, first "
                          "coverage and priming excluded from glyph batches; all jobs shuffled each round",
        "cache_policy": "coverage bypasses the runtime cache; draw-cold starts with an empty cache and draws "
                        "each character once; draw-warm primes every character first",
        "reference_policy": "float64 NumPy from the packed bytes on a spaced sample and both sides of every "
                            "route change, <= 1e-3; every advance against the packed table; every first draw "
                            "against its own coverage; quality against the drawing is recorded, not gating",
        "harness_sources": {name: fonts.digest(Path(__file__).with_name(name)) for name in
                            ("face.py", "face_worker.c", "braille_reference.py", "neuronte.py", "fonts.py")},
        "quality": [], "reference": [], "advances": [], "trials": [], "scenarios": [], "artifacts": [],
    }
    jobs = []
    for case in ([] if quality_only else manifest["cases"]):
        jobs.append((case, ems[0], "open"))
        jobs.extend((case, em, mode) for em in ems for mode in modes)
    expected = {}
    journal = artifacts / "trials.jsonl"
    with tempfile.TemporaryDirectory(prefix="face-worker-") as temporary, journal.open("w") as stream:
        journal.chmod(0o600)
        binary, result["build"] = build_worker(temporary)

        def run(case, em, mode, calls):
            return run_worker(binary, library, paths["model:" + case["model"]], mode, em, binary_mode,
                              codes, calls, temporary, args.job_timeout)

        for case in manifest["cases"]:
            path = paths["model:" + case["model"]]
            face = FaceFile(path.read_bytes())
            row, _ = run(case, ems[0], "routes", count)
            want = []
            for c in codes:
                try:
                    want.append(face.route(c))
                except ValueError:
                    want.append(None)
            if any(w is not None and w != r for w, r in zip(want, row["routes"])):
                raise ValueError(f"{case['name']}: native routes disagree with the packed range table")
            for em in ems:
                _, raw = run(case, em, "validate", count)
                cw, ch, width = em, 2 * em, em + 2
                cov_bytes = count * cw * ch * 4
                coverage = np.frombuffer(raw[:cov_bytes], dtype=np.float32).reshape(count, ch, cw)
                tail = raw[cov_bytes:]
                step = width * ch * 4 + 8
                canvases = np.stack([np.frombuffer(tail[i * step:i * step + width * ch * 4], dtype=np.uint32)
                                     .reshape(ch, width) for i in range(count)])
                adv = [struct.unpack_from("<2i", tail, i * step + width * ch * 4) for i in range(count)]
                result["reference"].append({"model": case["name"], **check_reference(face, codes, coverage, em)})
                for c, (regular, bold) in zip(codes, adv):
                    if (regular, bold) != (face.advance(c, 0, em), face.advance(c, 1, em)):
                        raise ValueError(f"{case['name']} U+{c:04X} at {em} px: advances {regular}, {bold} "
                                         "differ from the packed table")
                result["advances"].append({"model": case["name"], "em": em, "checked": 2 * count})
                for i, c in enumerate(codes):
                    if canvases[i].tobytes() != canvas_reference(coverage[i], em, binary_mode):
                        raise ValueError(f"{case['name']} U+{c:04X} at {em} px: drawn canvas differs from "
                                         "its own coverage")
                target = np.load(paths[f"target:{em}"], allow_pickle=False)
                if target.shape != (count, ch, width) or not np.isfinite(target).all():
                    raise ValueError("invalid teacher pixels")
                score = quality(canvases, target[[positions[c] for c in codes]], codes, binary_mode)
                score.update(model=case["name"], em=em,
                             coverage_sha256=hashlib.sha256(coverage.tobytes()).hexdigest())
                result["quality"].append(score)
                expected[case["name"], em, "coverage"] = coverage[-1].tobytes()
                for mode in ("draw-cold", "draw-warm"):
                    expected[case["name"], em, mode] = canvases[-1].tobytes()
                preview = artifacts / f"{case['name']}-{em}px.png"
                fonts.preview(np.floor(np.clip(coverage, 0, 1) * 255 + .5).astype(np.uint8).tobytes(),
                              count, (cw, ch), preview)
                result["artifacts"].append({"name": preview.name, "sha256": fonts.digest(preview)})
        quality_path = artifacts / "quality.json"
        quality_path.write_text(json.dumps(result["quality"], indent=2) + "\n")
        quality_path.chmod(0o600)
        for repeat in range(-warmup, repeats):
            order = list(jobs)
            rng.shuffle(order)
            for order_index, (case, em, mode) in enumerate(order):
                calls = count * (1 if mode in ("open", "draw-cold") else args.expert_passes)
                row, raw = run(case, em, mode, calls)
                if mode == "open":
                    duration, cpu, denominator = row["initialize_ns"], row["initialize_cpu_ns"], 1
                else:
                    duration, cpu, denominator = row["elapsed_ns"], row["cpu_ns"], calls
                    direct = mode == "coverage"
                    if (row["calls"] != calls or raw != expected[case["name"], em, mode] or
                            row["coverage_calls"] != (calls if direct else 0) or
                            row["draw_calls"] != (0 if direct else calls) or
                            row["width_calls"] != (0 if direct else 2 * calls) or
                            row["priming_calls"] != (count if mode == "draw-warm" else 0)):
                        raise ValueError("face pixels, API calls or cache setup differ from protocol")
                row.update(model=case["name"], em=em, case=mode, repeat=repeat, order=order_index,
                           ns_per_operation=duration / denominator, cpu_ns_per_operation=cpu / denominator)
                result["trials"].append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
        for case, em, mode in jobs:
            rows = [r for r in result["trials"] if r["repeat"] >= 0 and
                    (r["model"], r["em"], r["case"]) == (case["name"], em, mode)]
            unit = "ms/face open" if mode == "open" else "ms/glyph"
            median = statistics.median(r["ns_per_operation"] for r in rows)
            extra = {"model": case["name"], "em": em, "case": mode, "unit": unit,
                     "median_ns_per_operation": median,
                     "median_cpu_ms": statistics.median(r["cpu_ns_per_operation"] for r in rows) / 1e6}
            if mode != "open":
                extra["median_ms_per_face"] = median * count / 1e6
            result["scenarios"].append(scenario(f"{case['name']}/{em}px/{mode}", unit,
                                                [r["ns_per_operation"] / 1e9 for r in rows], extra))
    csv_path = artifacts / "trials.csv"
    with csv_path.open("w", newline="") as stream:
        csv_path.chmod(0o600)
        if result["trials"]:
            writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in result["trials"]))))
            writer.writeheader()
            writer.writerows(result["trials"])
    for path in (journal, csv_path, quality_path):
        result["artifacts"].append({"name": path.name, "sha256": fonts.digest(path)})
    result["hardware_end"] = fonts.hardware()
    result["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--face-experiment", required=True)
    parser.add_argument("--neuronte-home", default="")
    parser.add_argument("--expert-cases", default=",".join(CASES))
    parser.add_argument("--expert-passes", type=int, default=1)
    parser.add_argument("--experiment-quality-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--job-timeout", type=float, default=900)
    args = parser.parse_args()
    try:
        result = measure(args)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"face benchmark failed: {error}\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
