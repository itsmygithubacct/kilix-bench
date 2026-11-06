"""kilix-bench — measure a Kilix frame path and store the result privately."""

import argparse
import json
import os
import shutil
import subprocess
import sys

from kilix_bench.isolate import kilix_home_from_env, sandbox
from kilix_bench.report import render_text, write_run


def _parser():
    parser = argparse.ArgumentParser(prog="kilix-bench")
    parser.add_argument(
        "--suite", choices=("desktop", "render"), default="desktop",
        help="desktop measures a Kilix 95 desk; render measures the graphics SDK")
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


def main(argv=None):
    args = _parse() if argv is None else _parse_argv(argv)
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
