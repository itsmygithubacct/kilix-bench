"""Command-line entry points for the literal YES/NO benchmark."""
import argparse
import json
import os
from pathlib import Path
import sys
from . import worker
from .comparison import write_comparison
from .models import effort_options, hosted_catalog
from .records import compare_files, read_run, save_rankings


def default_state():
    data = Path(os.environ.get("XDG_DATA_HOME") or Path.home()/".local/share")
    return Path(os.environ.get("PUZZLLM_STATE") or data/"puzzllm")


def model_names(provider):
    return hosted_catalog(provider)[0] if provider in worker.HOSTED else worker.local_models(provider)


def journal(folder, name=None):
    if name is None:
        return max(folder.glob("*.jsonl"), default=None)
    if name != Path(name).name or not name.endswith(".jsonl"):
        raise ValueError("Use a journal filename from the runs list.")
    path = folder/name
    if path.is_symlink() or not path.is_file():
        raise ValueError("The question journal is unavailable.")
    return path


def answers(path):
    if path is None:
        return []
    rows = {}
    with path.open("rb") as source:
        for line in source:
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            if row.get("event") == "answer":
                rows[row["n"]] = row
    return list(rows.values())


def invalid_count(group):
    value = group["invalid_answers"]
    if value is None:
        return "-"
    return str(value) + ("+" if group["invalid_counted_runs"] < group["runs"] else "")


def score(value):
    return "-" if value is None else f"{value:.1f}"


def effort_name(group):
    return group.get("effort") or ("Unrecorded" if group["provider"] in worker.HOSTED else "-")


def ranking_fields(group):
    return [str(group["rank"] or "-"), group["provider"] or "-", group["model"] or "Unrecorded default",
            effort_name(group), str(group["runs"]), str(group["finished"]), invalid_count(group),
            score(group["avg_correct"]), score(group["avg_percent"])]


RANK_COLUMNS = ["Rank", "Provider", "Model", "Effort", "Runs", "Finished", "Invalid", "Avg correct", "Avg %"]


def show_event(row):
    kind = row["event"]
    if kind == "answer":
        print(f'{row["n"]:>4}  {row["verdict"]:<7}  {row.get("seconds", 0):.2f}s  '
              f'{json.dumps(row.get("reply", ""), ensure_ascii=False)}', flush=True)
    elif kind in ("begin", "end", "stopped", "error", "status", "storage_warning"):
        print(row.get("text") or row.get("file") or kind, flush=True)


def parser():
    p = argparse.ArgumentParser(prog="kilix-yesno", description="Run OneBit questions with literal YES/NO replies.")
    p.add_argument("--state", type=Path, default=default_state(), help="game data directory (shared with PuzzLLM)")
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("providers", help="list the supported providers")
    sub.add_parser("models", help="list model IDs").add_argument("provider", choices=worker.PROVIDERS)
    e = sub.add_parser("efforts", help="list a hosted model's effort levels")
    e.add_argument("provider", choices=worker.HOSTED)
    e.add_argument("model", nargs="?", default="")
    r = sub.add_parser("run", help="start a run or rejoin the active one")
    r.add_argument("provider", choices=worker.PROVIDERS)
    r.add_argument("model", nargs="?", default="")
    r.add_argument("--effort", default="")
    r.add_argument("--detach", action="store_true", help="print the journal path and return")
    for name, help_text in (("status","show the latest run"), ("rankings","rank each model and effort"),
                            ("runs","list saved runs"), ("answers","inspect saved replies")):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("--json", action="store_true")
        if name == "answers":
            cmd.add_argument("journal", nargs="?")
            cmd.add_argument("--invalid", action="store_true")
    sub.add_parser("watch", help="follow the latest run")
    sub.add_parser("stop", help="request a stop and keep partial answers")
    sub.add_parser("compare", help="compare machine timings as TSV").add_argument("files", type=Path, nargs="*")
    sub.add_parser("tui", help="open the interactive terminal monitor")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    folder = args.state.expanduser()/"bench/yesno"
    os.umask(0o077)
    try:
        if args.action == "tui":
            from .tui import launch
            return launch(folder)
        if args.action in ("providers", "models", "efforts"):
            if args.action == "providers":
                choices = worker.PROVIDERS
            elif args.action == "models":
                choices = model_names(args.provider)
            else:
                choices = effort_options(args.provider, args.model, discover=True)[0]
            for name in choices:
                print(name)
        elif args.action in ("run", "watch"):
            path = (worker.start_run(args.provider, args.model, folder, args.effort)
                    if args.action == "run" else journal(folder))
            if path is None:
                print("No runs recorded.")
            elif getattr(args, "detach", False):
                print(path)
            else:
                return worker.follow(path, show_event)
        elif args.action == "stop":
            print("Stop requested." if worker.stop_run(folder) else "No running benchmark.")
        elif args.action == "status":
            path = journal(folder)
            run = read_run(path) if path else None
            if args.json:
                print(json.dumps(run, ensure_ascii=False))
            elif run:
                print(run["status"], run["provider"], run["model"], effort_name(run),
                      f'{run["answered"]}/{run["total"]} answered', f'{run["correct"]} correct',
                      f'{run.get("invalid", 0)} invalid', sep="  ")
            else:
                print("No runs recorded.")
        elif args.action == "answers":
            for row in answers(journal(folder, args.journal)):
                if not args.invalid or row["verdict"] == "invalid":
                    if args.json:
                        print(json.dumps(row, ensure_ascii=False))
                    else:
                        print(f'{row["n"]}  {row["question"]}  [expected {row["expected"]}]')
                        show_event(row)
        elif args.action == "compare":
            write_comparison(compare_files(args.files) if args.files else save_rankings(folder), sys.stdout)
        else:
            board = save_rankings(folder)
            if args.action == "rankings" and args.json:
                print(json.dumps(board, indent=2, ensure_ascii=False))
            elif args.action == "rankings":
                print("\t".join(RANK_COLUMNS))
                for group in board["models"]:
                    row = "\t".join(ranking_fields(group))
                    if (group["invalid_answers"] or 0) > 0:
                        row = "! " + row
                        if sys.stdout.isatty() and "NO_COLOR" not in os.environ:
                            row = "\033[91m" + row + "\033[0m"
                    print(row)
            else:
                for group in board["models"]:
                    for run in group["history"]:
                        if args.json:
                            print(json.dumps(dict(run, provider=group["provider"], model=group["model"]), ensure_ascii=False))
                        else:
                            count = "-" if run["invalid"] is None else run["invalid"]
                            print(run["file"], group["provider"], group["model"], effort_name(group), run["status"],
                                  f'{run["correct"]}/{run["answered"]}', f'{count} invalid', sep="\t")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


def tui_main(argv=None):
    return main([*(sys.argv[1:] if argv is None else argv), "tui"])
