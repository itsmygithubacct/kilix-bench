"""Comparable timing statistics and labels for the leaderboard and its TSV export."""
from collections import defaultdict
import csv
import json
import math
import re


def duration(value):
    return value if type(value) in (int, float) and 0 <= value < math.inf else None


def mean(values):
    return round(sum(values) / len(values), 6) if values else None


def invalid_totals(runs):
    known = [r["invalid"] for r in runs if r["invalid"] is not None]
    return dict(invalid_answers=sum(known) if known else None, invalid_counted_runs=len(known))


def timing_summary(latencies, wall=None, setup=None, ended=None, answered=0):
    values = sorted(latencies)
    return dict(ended_at=ended, wall_seconds=wall, setup_seconds=setup, timed_answers=len(values),
                answer_seconds=round(sum(values), 6) if values else None, mean_answer_seconds=mean(values),
                p95_answer_seconds=values[math.ceil(len(values)*0.95)-1] if values else None,
                questions_per_second=round(answered / wall, 6) if wall else None)


def normalize_measurements(run):
    # Older reports and partial hardware probes can have any subset of these
    # fields. A bad optional measurement must not invalidate a verified score.
    misses = run["answered"] - run["correct"]
    invalid, wrong = run.get("invalid"), run.get("wrong")
    if type(invalid) is int and 0 <= invalid <= misses:
        wrong = misses - invalid
    elif type(wrong) is int and 0 <= wrong <= misses:
        invalid = misses - wrong
    elif misses == 0 and run["status"] not in ("Unreadable", "Damaged"):
        wrong = invalid = 0
    else:
        wrong = invalid = None
    run.update(wrong=wrong, invalid=invalid)
    runner = run.get("runner")
    if not (isinstance(runner, dict) and isinstance(runner.get("cpu"), dict)
            and all(v is None or isinstance(v, str) for v in (runner.get("id"), runner.get("label")))
            and isinstance(runner.get("gpus"), list)
            and all(isinstance(g, dict) and isinstance(g.get("model"), str) for g in runner["gpus"])):
        run["runner"] = None
    usage = run.get("runner_usage")
    if not (isinstance(usage, dict) and isinstance(usage.get("gpus"), list)
            and all(isinstance(g, dict) and isinstance(g.get("slot"), str)
                    and (g.get("mean_busy_percent") is None or duration(g["mean_busy_percent"]) is not None)
                    for g in usage["gpus"])):
        run["runner_usage"] = None
    if not isinstance(run.get("backend_scope"), str):
        run["backend_scope"] = None
    if not isinstance(run.get("artifact_id"), str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", run["artifact_id"]):
        run["artifact_id"] = None
    timing = run.get("timing")
    if not isinstance(timing, dict):
        run["timing"] = timing_summary([])
        return
    count = timing.get("timed_answers")
    if type(count) is not int or not 0 <= count <= run["answered"] or duration(timing.get("answer_seconds")) is None:
        count = 0
    cleaned = timing_summary([], duration(timing.get("wall_seconds")), duration(timing.get("setup_seconds")),
                             timing.get("ended_at") if isinstance(timing.get("ended_at"), str) else None, run["answered"])
    if count:
        seconds = timing["answer_seconds"]
        cleaned.update(timed_answers=count, answer_seconds=seconds, mean_answer_seconds=round(seconds/count, 6),
                       p95_answer_seconds=duration(timing.get("p95_answer_seconds")))
    run["timing"] = cleaned


def machine_comparisons(attempts):
    groups = defaultdict(list)
    for run in attempts:
        runner = run.get("runner") or {}
        # A new label alone does not make a new machine. Hardware changes do.
        hardware = {key: runner.get(key) for key in ("cpu", "gpus", "ram_bytes")}
        key = runner.get("id"), json.dumps(hardware, sort_keys=True), run.get("backend_scope")
        groups[key].append(run)
    results = []
    for group in groups.values():
        done = [r for r in group if r["status"] == "Complete"]
        timings = [r.get("timing") or {} for r in done]
        walls = [t["wall_seconds"] for t in timings if duration(t.get("wall_seconds")) is not None]
        wall_seconds = sum(walls)
        setups = [t["setup_seconds"] for t in timings if duration(t.get("setup_seconds")) is not None]
        wall_answers = sum(r["answered"] for r in done if (r.get("timing") or {}).get("wall_seconds"))
        timed_answers = sum(t.get("timed_answers", 0) for t in timings)
        seconds = sum(t.get("answer_seconds") or 0 for t in timings)
        results.append(dict(runner=group[0].get("runner"), backend_scope=group[0].get("backend_scope"),
                            runs=len(group), finished=len(done), **invalid_totals(group), wall_timed_runs=len(walls),
                            mean_wall_seconds=mean(walls), mean_setup_seconds=mean(setups),
                            questions_per_second=round(wall_answers/wall_seconds, 6) if wall_seconds else None,
                            timed_answers=timed_answers,
                            mean_answer_seconds=round(seconds / timed_answers, 6) if timed_answers else None,
                            avg_percent=mean([100*r["correct"]/r["total"] for r in done])))
    return results


def size_label(value):
    return f"{value / 1024**3:.1f} GiB" if duration(value) is not None else ""


def runner_labels(run):
    runner = run.get("runner") or {}
    cpu = runner.get("cpu") or {}
    model = str(cpu.get("model") or cpu.get("architecture") or "Unknown")
    cores, threads = cpu.get("physical_cores"), cpu.get("logical_cpus")
    if cores and threads:
        model += f" ({cores} cores, {threads} threads)"
    elif threads:
        model += f" ({threads} threads)"
    devices = []
    for gpu in runner.get("gpus") or []:
        name = gpu.get("model") or "Unknown"
        vram = size_label(gpu.get("vram_bytes"))
        devices.append(f"{name} ({vram})" if vram else name)
    tag = str(runner.get("id") or "")[:8]
    name = str(cpu.get("model") or "").split(" @ ")[0].replace("(R)", "").replace("(TM)", "").removesuffix(" CPU")
    label = runner.get("label") or name
    machine = f"{label} [{tag}]" if label and tag else label or tag or "Unrecorded"
    return dict(machine_label=machine, cpu_label=model, gpu_label=" + ".join(devices) or
                ("None detected" if runner.get("gpus") == [] else "Unknown"),
                ram_label=size_label(runner.get("ram_bytes")) or "Unknown")


def usage_label(usage):
    if not usage:
        return ""
    values = []
    if duration(usage.get("cpu_busy_percent")) is not None:
        values.append(f"CPU {usage['cpu_busy_percent']:.1f}% busy")
    ram = size_label(usage.get("peak_ram_used_bytes"))
    if ram:
        values.append("Peak RAM " + ram)
    for gpu in usage.get("gpus") or []:
        busy, vram = gpu.get("mean_busy_percent"), size_label(gpu.get("peak_vram_used_bytes"))
        if busy is not None or vram:
            label = f"GPU {gpu['slot']}"
            if busy is not None:
                label += f" {busy:.1f}% busy"
            if vram:
                label += f" peak VRAM {vram}"
            values.append(label)
    return "    ".join(values)


def write_comparison(document, output):
    table = csv.writer(output, delimiter="\t", lineterminator="\n")
    table.writerow(["Provider", "Model", "Effort", "Artifact", "Machine", "Machine ID", "CPU", "GPU", "RAM",
                    "Backend", "Runs", "Finished", "Invalid", "Runs with invalid counts", "Avg correct %", "Wall-timed runs", "Mean wall s",
                    "Mean setup s", "Questions/s", "Timed answers", "Mean s/answer", "Suite", "Split", "Protocol"])
    for model in document["models"]:
        for row in model["machine_comparisons"]:
            labels = runner_labels(row)
            table.writerow([model["provider"], model["model"], model["effort"], model.get("artifact_id"), labels["machine_label"],
                            (row.get("runner") or {}).get("id"), labels["cpu_label"], labels["gpu_label"],
                            labels["ram_label"], row["backend_scope"], row["runs"], row["finished"],
                            row["invalid_answers"], row["invalid_counted_runs"], row["avg_percent"], row["wall_timed_runs"], row["mean_wall_seconds"],
                            row["mean_setup_seconds"], row["questions_per_second"],
                            row["timed_answers"], row["mean_answer_seconds"], document["suite"], document["split"],
                            document["protocol"]])
