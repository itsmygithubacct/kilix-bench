"""Command line of the OneBit suite.

    python3 -m kilix_bench --suite onebit --endpoint URL --model NAME [...]
    python3 -m kilix_bench --suite onebit --command "kilix-llm judge --stdin"
    python3 -m kilix_bench onebit compare A.json B.json [--arm constrained]
    python3 -m kilix_bench onebit score RESULT.json [--arm raw]
    python3 -m kilix_bench onebit check [--draft] [FILE...]
    python3 -m kilix_bench onebit progress --arm A --run R --step N RESULT.json
    python3 -m kilix_bench onebit chart progress.jsonl [--group BAND[/DOMAIN]]
    python3 -m kilix_bench --suite onebit --backend needle [--needle-engine PATH]
    python3 -m kilix_bench --suite onebit --backend codex \\
        --codex-profile onebit-luna6-low [--jobs 4]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys

from kilix_bench.isolate import plain_sandbox
from kilix_bench.onebit import backends, items as itemset, progress, scoring
from kilix_bench.onebit.runner import render
from kilix_bench.report import bench_home, write_run


def run_parser():
    parser = argparse.ArgumentParser(
        prog="kilix-bench --suite onebit",
        description="Ask a model every OneBit item and score the one-digit "
                    "answers.")
    parser.add_argument(
        "--backend", choices=tuple(backends.REGISTRY), default="",
        help="chat (with --endpoint, the default), command (with "
             "--command), needle (the needle2 engine) or codex (codex exec "
             "with --codex-profile)")
    parser.add_argument("--split", choices=("dev", "test", "all"),
                        default="dev")
    parser.add_argument("--bands", default="",
                        help="Comma-separated bands (default every band)")
    parser.add_argument("--arms", default="",
                        help="raw,constrained (default: the backend's arms)")
    parser.add_argument("--timeout", type=float, default=0,
                        help="Seconds per item (default 120; codex 180)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Only the first N selected items (a smoke run)")
    parser.add_argument("--jobs", type=int, default=0,
                        help="Items asked at once (default 1; codex 4)")
    parser.add_argument(
        "--cache-dir", default="",
        help="Answer cache for resuming (default: onebit-cache in the bench "
             "home; used by backends that support it, such as codex)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ask every item again; keep no answers")
    parser.add_argument("--model-file", default="",
                        help="Weights file to hash into the record")
    parser.add_argument("--model-sha256", default="")
    parser.add_argument("--label", default="",
                        help="Name of this model in the result file name")
    parser.add_argument("--max-input-tokens", type=int, default=0,
                        help="Skip (and count) items longer than this")
    parser.add_argument(
        "--sizer-command", default="",
        help="Command that prints the sizer JSON; 'off' to skip the sizer")
    for module in backends.REGISTRY.values():
        module.add_arguments(parser)
    return parser


def _slug(text):
    slug = re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-")
    return slug[:60] or "model"


def default_needle_engine():
    return backends.needle.default_engine()


def spec_from_args(args):
    module = backends.module_for(args.backend)
    spec = {
        "backend": args.backend,
        "split": args.split,
        "bands": [band for band in args.bands.split(",") if band],
        "arms": [arm for arm in args.arms.split(",") if arm],
        "limit": args.limit,
        "jobs": args.jobs,
        "model_file": os.path.abspath(args.model_file) if args.model_file
        else "",
        "model_sha256": args.model_sha256,
        "model_label": args.label,
        "max_input_tokens": args.max_input_tokens,
        "sizer_command": args.sizer_command,
        "cache_dir": "" if args.no_cache else os.path.abspath(
            args.cache_dir or str(bench_home() / "onebit-cache")),
    }
    if args.timeout:
        spec["timeout"] = args.timeout
    elif args.backend != "codex":
        spec["timeout"] = 120.0
    spec.update(module.spec(args))
    return spec


def run_command(argv):
    parser = run_parser()
    args = parser.parse_args(argv)
    if not args.backend:
        args.backend = "command" if args.command else "chat"
    if args.backend != "chat" and args.endpoint:
        parser.error(f"--endpoint is for the chat backend, not "
                     f"{args.backend}")
    if args.backend != "command" and args.command:
        parser.error(f"--command is for the command backend, not "
                     f"{args.backend}")
    module = backends.module_for(args.backend)
    module.check(args, parser)
    for band in filter(None, args.bands.split(",")):
        if band not in itemset.BANDS:
            raise SystemExit(f"unknown band {band}")
    spec = spec_from_args(args)
    keep = {name: os.environ[name] for name in args.pass_env
            if name in os.environ}
    env, root = plain_sandbox()
    env.update(keep)
    if args.api_key_env and args.api_key_env in os.environ:
        env[args.api_key_env] = os.environ[args.api_key_env]
    src = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "kilix_bench.onebit.runner",
             json.dumps(spec)],
            env=env, stdout=subprocess.PIPE, text=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if result.returncode != 0:
        # A stopped run (exit 3) printed why on stderr; nothing is written.
        raise SystemExit(result.returncode or 1)
    payload = json.loads(result.stdout)
    forbidden = [os.path.expanduser("~"), src, os.getcwd(),
                 *module.forbidden(spec)]
    if spec["cache_dir"]:
        forbidden.append(spec["cache_dir"])
    if args.model_file:
        forbidden.append(os.path.dirname(os.path.abspath(args.model_file)))
    name = args.label or module.label(args) or args.backend
    slug = f"onebit-{_slug(name)}-{args.split}"
    stamped, _latest, text = write_run(
        payload, forbidden=tuple(forbidden), slug=slug, render=render)
    sys.stdout.write(render(payload))
    sys.stdout.write(f"\nresults {stamped}\ntext {text}\n")
    return 0


def _load_arm(path, arm):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    arms = payload.get("arms") or {}
    if arm:
        body = arms.get(arm)
    else:
        supported = [name for name in ("constrained", "raw", "command")
                     if (arms.get(name) or {}).get("supported")]
        arm = supported[0] if supported else ""
        body = arms.get(arm)
    if not body or not body.get("supported"):
        raise SystemExit(f"{path}: no supported arm {arm or ''}".rstrip())
    return payload, arm, body["records"]


def _split_filter(records, split):
    if split == "all":
        return records
    return [row for row in records if row["split"] == split]


def compare_command(argv):
    parser = argparse.ArgumentParser(
        prog="kilix-bench onebit compare",
        description="Paired comparison of two OneBit result files (exact "
                    "sign test on the items only one of them got right).")
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--arm", default="",
                        help="Arm of A (default constrained, else raw)")
    parser.add_argument("--arm-b", default="",
                        help="Arm of B (default: the same as --arm)")
    parser.add_argument("--split", choices=("dev", "test", "all"),
                        default="all")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload_a, arm_a, records_a = _load_arm(args.a, args.arm)
    payload_b, arm_b, records_b = _load_arm(args.b, args.arm_b or arm_a)
    result = scoring.compare(_split_filter(records_a, args.split),
                             _split_filter(records_b, args.split))
    if payload_a["items"]["set_sha256"] != payload_b["items"]["set_sha256"]:
        result["warning"] = "the two runs used different item files"
    if args.json:
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        return 0

    def name(payload, arm):
        model = payload["model"]
        return "{0} [{1}]".format(
            model.get("label") or model.get("model") or model.get("program"),
            arm)

    lines = scoring.render_compare(result, name(payload_a, arm_a),
                                   name(payload_b, arm_b))
    if "warning" in result:
        lines.append("warning: " + result["warning"])
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


def score_command(argv):
    parser = argparse.ArgumentParser(prog="kilix-bench onebit score")
    parser.add_argument("result")
    parser.add_argument("--arm", default="")
    parser.add_argument("--split", choices=("dev", "test", "all"),
                        default="all")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    _payload, arm, records = _load_arm(args.result, args.arm)
    summary = scoring.summarize(_split_filter(records, args.split))
    if args.json:
        sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    else:
        sys.stdout.write(f"arm {arm}\n")
        sys.stdout.write("\n".join(scoring.render_summary(summary)) + "\n")
    return 0


def progress_command(argv):
    parser = argparse.ArgumentParser(
        prog="kilix-bench onebit progress",
        description="Append one checkpoint's scores to a progress file.")
    parser.add_argument("--arm", required=True,
                        help="Training arm (model being tuned): needle, "
                             "qwen, lfm2, ...")
    parser.add_argument("--run", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--out", default="progress.jsonl")
    parser.add_argument("result")
    args = parser.parse_args(argv)
    with open(args.result, encoding="utf-8") as handle:
        payload = json.load(handle)
    line = progress.progress_line(payload, args.arm, args.run, args.step)
    progress.append(args.out, line)
    decodes = ", ".join(
        f"{name} {100 * body['accuracy']:.1f}%"
        for name, body in line["decodes"].items()
        if body.get("supported") and body.get("accuracy") is not None)
    sys.stdout.write(f"{args.out}: {args.arm} {args.run} step {args.step}: "
                     f"{decodes or 'no supported decode'}\n")
    return 0


def chart_command(argv):
    parser = argparse.ArgumentParser(
        prog="kilix-bench onebit chart",
        description="Accuracy against step, one line per arm and run.")
    parser.add_argument("progress_file")
    parser.add_argument("--decode", default="constrained",
                        help="raw or constrained (Needle has raw only)")
    parser.add_argument("--group", default="computer-use",
                        help="A band (computer-use) or band/domain "
                             "(computer-use/logs) drawn beside overall")
    parser.add_argument("--out", default="",
                        help="SVG file (default: beside the progress file)")
    args = parser.parse_args(argv)
    lines = progress.read(args.progress_file)
    out = args.out or os.path.splitext(args.progress_file)[0] + ".svg"
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(progress.chart_svg(lines, args.decode, args.group))
    sys.stdout.write(progress.chart_table(lines, args.decode, args.group))
    sys.stdout.write(f"chart {out}\n")
    return 0


def main(argv):
    commands = {
        "progress": progress_command,
        "chart": chart_command,
        "compare": compare_command,
        "score": score_command,
        "check": lambda rest: itemset.main(["check", *rest]),
    }
    if not argv or argv[0] not in commands:
        sys.stderr.write(__doc__)
        return 2
    return commands[argv[0]](argv[1:])
