"""Build benchmark standings from the original per-run journals."""
from collections import defaultdict
from datetime import datetime
import fcntl
from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile

from kilix_bench.yesno.comparison import (duration, invalid_totals, machine_comparisons, normalize_measurements, runner_labels,
                                      timing_summary, usage_label)


def model_id(provider, model):
    if not model:
        return ""
    if provider == "llama.cpp":
        try:
            return str(Path(model).expanduser().resolve())
        except (OSError, RuntimeError, ValueError):
            return model
    if provider == "ollama":
        model = model.removeprefix("registry.ollama.ai/").removeprefix("library/")
        if ":" not in model.rsplit("/", 1)[-1]:
            model += ":latest"
    return model


def run_date(path):
    try:
        created = datetime.strptime(path.name.split("-", 1)[0], "%Y%m%dT%H%M%S.%fZ")
        return created.isoformat(timespec="microseconds") + "Z"
    except ValueError:
        return None


def read_run(path):
    provider = model = status = ""
    effort = None
    identities = set()
    answers, latencies = {}, {}
    runner = usage = scope = artifact = None
    wall = setup = ended = None
    total = 0
    damaged = False
    with path.open("rb") as source:
        try:
            fcntl.flock(source, fcntl.LOCK_SH | fcntl.LOCK_NB)
            live = False
        except BlockingIOError:
            live = True
        for line in source:
            if not line.endswith(b"\n"):
                break
            try:
                event = json.loads(line)
                kind = event["event"]
                if kind == "begin":
                    backend, choice = event["provider"], event.get("model") or ""
                    if not isinstance(backend, str) or not isinstance(choice, str):
                        raise ValueError("invalid model")
                    provider, model = backend, choice
                    effort = event.get("effort")
                    if effort is not None and not isinstance(effort, str):
                        raise ValueError("invalid effort")
                    scope = event.get("backend_scope")
                    total = int(event.get("total", 0))
                elif kind == "machine":
                    runner = event.get("runner")
                elif kind == "artifact":
                    artifact = event.get("id")
                elif kind == "ready":
                    setup = duration(event.get("setup_seconds"))
                elif kind == "selection" and isinstance(event.get("model"), str):
                    model = event["model"]
                elif kind == "identity":
                    name = event.get("model")
                    if isinstance(name, str) and name:
                        identities.add(model_id(provider, name))
                elif kind == "size":
                    total = int(event["total"])
                elif kind == "answer":
                    n, verdict = event["n"], event["verdict"]
                    if type(n) is not int or n < 1 or verdict not in ("correct", "wrong", "invalid"):
                        raise ValueError("invalid answer")
                    if n in answers and answers[n] != verdict:
                        damaged = True
                    answers[n] = verdict
                    seconds = duration(event.get("seconds"))
                    if seconds is not None:
                        latencies[n] = seconds
                elif kind in ("end", "stopped", "error"):
                    wall, ended = duration(event.get("wall_seconds")), event.get("ended_at")
                    usage = event.get("runner_usage")
                    status = {"end": "Complete", "stopped": "Stopped", "error": "Failed"}[kind]
            except (KeyError, TypeError, ValueError, UnicodeError):
                damaged = True
    if not provider and not damaged:
        # The worker can publish the locked file just before its first write.
        return None if live else dict(provider="", model="", status="Interrupted", file=path.name,
                                      started=run_date(path), answered=0, total=0, correct=0)
    if damaged:
        status = "Damaged"
    elif status == "Complete" and (total < 1 or len(answers) != total or max(answers, default=0) > total):
        status = "Interrupted"
    elif not status:
        status = "Running" if live else "Interrupted"
    identity = next(iter(identities)) if len(identities) == 1 else model_id(provider, model)
    if len(identities) > 1:
        identity = ""
    return dict(provider=provider, model=identity, effort=effort, file=path.name, started=run_date(path),
                status=status, total=total, answered=len(answers), runner=runner, backend_scope=scope,
                runner_usage=usage, artifact_id=artifact,
                timing=timing_summary(latencies.values(), wall, setup, ended, len(answers)),
                correct=sum(verdict == "correct" for verdict in answers.values()),
                wrong=sum(verdict == "wrong" for verdict in answers.values()),
                invalid=sum(verdict == "invalid" for verdict in answers.values()))


def archived_runs(previous, *, restoring=True):
    # A restored rankings file can contain scores whose question journals were
    # archived separately. Merge by the original run filename on every refresh.
    runs = {}
    if previous.get("benchmark") != "OneBit" or previous.get("split") != "dev":
        raise ValueError("Unknown benchmark")
    for group in previous["models"]:
        provider, model = group["provider"] or "", group["model"] or ""
        if not isinstance(provider, str) or not isinstance(model, str):
            raise ValueError("Invalid model identity")
        for entry in group["history"]:
            selected = entry.get("model_path") if provider == "llama.cpp" else None
            run = dict(entry, provider=provider, model=selected if isinstance(selected, str) and selected else model)
            run["effort"] = entry.get("effort") or group.get("effort")
            if run["effort"] is not None and not isinstance(run["effort"], str):
                raise ValueError("Invalid reasoning effort")
            if restoring:
                run["journal_available"] = False
            name = run["file"]
            if Path(name).name != name or not name.endswith(".jsonl") or name in runs:
                raise ValueError("Invalid run filename")
            if run["status"] not in ("Running", "Complete", "Stopped", "Failed", "Interrupted", "Damaged", "Unreadable"):
                raise ValueError("Invalid run status")
            if any(type(run[key]) is not int for key in ("correct", "answered", "total")):
                raise ValueError("Invalid answer count")
            if not 0 <= run["correct"] <= run["answered"] or (
                    run["status"] == "Complete" and not 0 < run["answered"] == run["total"]):
                raise ValueError("Inconsistent score")
            if run["started"] is not None and not isinstance(run["started"], str):
                raise ValueError("Invalid start date")
            if restoring and run["status"] == "Running":
                run["status"] = "Interrupted"
            runs[name] = run
    if len(runs) != previous["total_runs"]:
        raise ValueError("Inconsistent run count")
    return runs


def rankings_document(folder, runs=None):
    if runs is None:
        runs = {}
    for path in folder.glob("*.jsonl"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            run = read_run(path)
        except FileNotFoundError:
            continue
        except OSError:
            run = dict(provider="", model="", file=path.name, started=run_date(path),
                       status="Unreadable", total=0, answered=0, correct=0)
        if run is None:
            continue
        saved = runs.get(path.name)
        if saved and saved["status"] == "Complete":
            if run["status"] != "Complete":
                continue
            if all(run[key] == saved[key] for key in ("provider", "model", "total", "correct")):
                for key in ("runner", "runner_usage", "backend_scope", "artifact_id", "effort"):
                    if run.get(key) is None:
                        run[key] = saved.get(key)
                old_timing = saved.get("timing") or {}
                if isinstance(old_timing, dict):
                    for key in ("ended_at", "wall_seconds", "setup_seconds"):
                        if run["timing"].get(key) is None:
                            run["timing"][key] = old_timing.get(key)
                    if type(old_timing.get("timed_answers")) is int and old_timing["timed_answers"] > run["timing"]["timed_answers"]:
                        for key in ("timed_answers", "answer_seconds", "mean_answer_seconds", "p95_answer_seconds"):
                            run["timing"][key] = old_timing.get(key)
        runs[path.name] = dict(run, journal_available=run["status"] not in ("Damaged", "Unreadable"))
    return summarize_runs(runs)


def summarize_runs(runs):
    # Keep the original path in each journal. A measured content digest also
    # recognizes an identical GGUF installed under different paths.
    artifacts = defaultdict(set)
    for run in runs.values():
        normalize_measurements(run)
        if run["provider"] == "llama.cpp" and run.get("artifact_id"):
            artifacts[run["model"]].add(run["artifact_id"])
    by_model = defaultdict(list)
    for name in sorted(runs, reverse=True):
        run = dict(runs[name])
        provider, model = run.pop("provider"), run.pop("model")
        group_id = model
        if provider == "llama.cpp":
            run["model_path"] = model
            if run.get("artifact_id"):
                group_id = run["artifact_id"]
            elif len(artifacts[model]) == 1:
                group_id = next(iter(artifacts[model]))
        by_model[provider, group_id, run.get("effort")].append(run)
    groups = []
    for (provider, group_id, effort), attempts in by_model.items():
        model = min(r["model_path"] for r in attempts) if provider == "llama.cpp" else group_id
        complete = [r for r in attempts if r["status"] == "Complete"]
        count = len(complete)
        mean = Fraction(sum(r["correct"] for r in complete), count) if count else None
        accuracy = sum((Fraction(r["correct"], r["total"]) for r in complete), Fraction()) / count if count else None
        groups.append((provider, model, attempts, count, mean, accuracy, group_id, effort))
    groups.sort(key=lambda g: (not (g[1] and g[3]), -(g[5] or 0), g[0].casefold(), g[1].casefold(), g[1], g[6], g[7] or ""))
    previous_accuracy = None
    rank = 0
    models = []
    for position, (provider, model, attempts, count, mean, accuracy, group_id, effort) in enumerate(groups, 1):
        ranked = bool(model and count)
        if ranked and accuracy != previous_accuracy:
            rank, previous_accuracy = position, accuracy
        models.append(dict(provider=provider or None, model=model or None, rank=rank if ranked else None,
                           runs=len(attempts), finished=count, effort=effort, **invalid_totals(attempts),
                           avg_correct=float(mean) if count else None,
                           avg_percent=float(accuracy * 100) if count else None, history=attempts,
                           artifact_id=group_id if group_id != model else None,
                           machine_comparisons=machine_comparisons(attempts)))
    return dict(schema_version=3, benchmark="OneBit", split="dev", total_runs=len(runs), models=models)


def ranking_events(document):
    for group in document["models"]:
        fields = dict(group)
        history = fields.pop("history")
        machines = fields.pop("machine_comparisons")
        fields.update(provider=fields["provider"] or "", model=fields["model"] or "", rank=fields["rank"] or 0)
        yield dict(event="standing", **fields)
        for run in history:
            started = run["started"]
            date = started[:19].replace("T", " ") if started else Path(run["file"]).stem
            status = run["status"]
            if status == "Complete" and not run["journal_available"]:
                status = "Complete (summary only)"
            timing = run["timing"]
            yield dict(event="saved_run", provider=fields["provider"], model=fields["model"], file=run["file"],
                       started=date, status=status, correct=run["correct"], invalid=run["invalid"], effort=group["effort"],
                       answered=run["answered"], total=run["total"],
                       wall_seconds=timing.get("wall_seconds"), mean_answer_seconds=timing.get("mean_answer_seconds"),
                       backend_scope=run.get("backend_scope"), usage_label=usage_label(run.get("runner_usage")),
                       **runner_labels(run))
        for row in machines:
            yield dict(event="machine_result", runs=row["runs"], finished=row["finished"], avg_percent=row["avg_percent"],
                       invalid_answers=row["invalid_answers"], invalid_counted_runs=row["invalid_counted_runs"],
                       mean_wall_seconds=row["mean_wall_seconds"], mean_answer_seconds=row["mean_answer_seconds"],
                       wall_timed_runs=row["wall_timed_runs"], timed_answers=row["timed_answers"],
                       backend_scope=row["backend_scope"], **runner_labels(row))


def compare_files(paths):
    def evidence(row):
        return (row["status"] == "Complete", row["answered"], row["status"] != "Running",
                row.get("effort") is not None, row["invalid"] is not None, bool(row.get("runner")), row["timing"]["timed_answers"])

    runs = {}
    for path in paths:
        document = json.loads(path.read_bytes())
        if not isinstance(document, dict) or document.get("schema_version") not in (1, 2, 3):
            raise ValueError(f"Unrecognized rankings format in {path.name}.")
        for name, run in archived_runs(document, restoring=False).items():
            normalize_measurements(run)
            old = runs.get(name)
            if old and old["status"] == run["status"] == "Complete":
                for key in ("provider", "model", "correct", "total"):
                    if old[key] != run[key]:
                        raise ValueError(f"Conflicting completed scores for {name}.")
                old_id, new_id = (old.get("runner") or {}).get("id"), (run.get("runner") or {}).get("id")
                if old_id and new_id and old_id != new_id:
                    raise ValueError(f"Conflicting machines for {name}.")
                for key in ("invalid", "effort"):
                    if old.get(key) is not None and run.get(key) is not None and old[key] != run[key]:
                        raise ValueError(f"Conflicting {key} for {name}.")
            if old is None:
                runs[name] = run
                continue
            best, other = (run, old) if evidence(run) > evidence(old) else (old, run)
            if old["status"] == run["status"] == "Complete":
                for key in ("runner", "runner_usage", "backend_scope", "artifact_id", "effort", "wrong", "invalid"):
                    if best.get(key) is None:
                        best[key] = other.get(key)
                if other["timing"]["timed_answers"] > best["timing"]["timed_answers"]:
                    for key in ("timed_answers", "answer_seconds", "mean_answer_seconds", "p95_answer_seconds"):
                        best["timing"][key] = other["timing"][key]
                for key in ("wall_seconds", "setup_seconds", "ended_at"):
                    if best["timing"].get(key) is None:
                        best["timing"][key] = other["timing"].get(key)
            runs[name] = best
    return summarize_runs(runs)


def save_rankings(folder):
    """Publish one complete UTF-8 snapshot, serialized with other writers."""
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = folder / "rankings.json"
    with (folder / ".rankings.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            old_bytes = target.read_bytes()
        except FileNotFoundError:
            old_bytes = None
        runs = None
        damaged = False
        if old_bytes is not None:
            try:
                previous = json.loads(old_bytes)
            except (UnicodeError, ValueError):
                damaged = True
            else:
                if not isinstance(previous, dict) or previous.get("schema_version") not in (1, 2, 3):
                    raise ValueError("Unrecognized rankings.json format. Its original contents have been preserved.")
                try:
                    runs = archived_runs(previous)
                except (ValueError, KeyError, TypeError):
                    damaged = True
        document = rankings_document(folder, runs)
        if damaged:
            if not document["total_runs"]:
                raise ValueError("Cannot rebuild rankings.json without run journals. Its contents have been preserved.")
            # Retain the exact bytes for recovery before repairing from journals.
            with tempfile.NamedTemporaryFile(dir=folder, prefix="rankings-damaged-", suffix=".json", delete=False) as out:
                out.write(old_bytes)
                out.flush()
                os.fsync(out.fileno())
        encoded = (json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
        if encoded != old_bytes:
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=folder, prefix=".rankings-", delete=False) as out:
                    temporary = Path(out.name)
                    out.write(encoded)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(temporary, target)
                directory = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return document
