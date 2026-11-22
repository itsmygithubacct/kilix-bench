"""kilix-bench — measure a Kilix frame path and store the result privately."""

import argparse
import json
import os
import shutil
import subprocess
import sys

from kilix_bench import local_models
from kilix_bench.isolate import kilix_home_from_env, sandbox
from kilix_bench.report import render_text, write_run


def _parser():
    parser = argparse.ArgumentParser(prog="kilix-bench")
    parser.add_argument(
        "--suite", choices=("desktop", "render", "text", "onebit", *local_models.SUITES),
        default="desktop",
        help="desktop measures a Kilix 95 desk; render measures the graphics SDK; "
             "text compares Kitty cells, Kitty graphics, and Neuronte; "
             "onebit asks a model yes/no items it answers with 0 or 1 "
             "(see --suite onebit --help). For local model benchmarks, see "
             "--suite SUITE --help.")
    parser.add_argument("--desktop", default="",
                        help="Path of a Kilix 95 checkout (desktop suite)")
    parser.add_argument("--kilix-home", default="",
                        help="Kilix host checkout that contains config/kilix_sdk")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    return parser


def _parse():
    return _parser().parse_args()


def _parse_argv(argv):
    return _parser().parse_args(argv)


def _run(command, env, root):
    try:
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=180)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        sys.stderr.write(result.stdout)
        raise SystemExit(result.returncode or 1)
    return json.loads(result.stdout)


def _suite_options(argv):
    for index, arg in enumerate(argv):
        if arg == "--suite" and index + 1 < len(argv):
            return argv[index + 1], argv[:index] + argv[index + 2:]
        if arg.startswith("--suite="):
            return arg.partition("=")[2], argv[:index] + argv[index + 1:]
    return None, argv


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "yesno":
        from kilix_bench.yesno.cli import main as yesno_main
        return yesno_main(argv[1:])
    if argv and argv[0] == "onebit":
        from kilix_bench.onebit.cli import main as onebit_main
        return onebit_main(argv[1:])
    suite, rest = _suite_options(argv)
    if suite in local_models.SUITES:
        return local_models.run(suite, rest)
    if suite == "onebit":
        from kilix_bench.onebit.cli import run_command
        return run_command(rest)
    args = _parse_argv(argv)
    kilix_home = args.kilix_home or kilix_home_from_env()
    if not kilix_home or not os.path.isdir(os.path.join(kilix_home, "config")):
        raise SystemExit("pass --kilix-home pointing at a Kilix checkout")
    env, root = sandbox(os.path.abspath(kilix_home))
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    shared = [
        "--width", str(args.width),
        "--height", str(args.height),
        "--repeats", str(args.repeats),
        "--warmup", str(args.warmup),
    ]
    forbidden = [os.path.abspath(kilix_home), os.path.expanduser("~")]
    if args.suite == "render":
        command = [sys.executable, "-m", "kilix_bench.render", *shared]
    elif args.suite == "text":
        from kilix_bench.text import libraries
        for key, path in libraries().items():
            env[key] = path
            forbidden.append(path)
        command = [sys.executable, "-m", "kilix_bench.text", *shared]
    else:
        if not args.desktop:
            raise SystemExit("the desktop suite needs --desktop")
        desktop = os.path.abspath(args.desktop)
        if not os.path.isfile(os.path.join(desktop, "main.py")):
            raise SystemExit("desktop checkout has no main.py")
        forbidden.append(desktop)
        command = [
            sys.executable, "-m", "kilix_bench.desktop",
            "--desktop", desktop, *shared]
    payload = _run(command, env, root)
    stamped, _latest, text = write_run(payload, forbidden=tuple(forbidden))
    sys.stdout.write(render_text(payload))
    sys.stdout.write(f"\nresults {stamped}\ntext {text}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
