"""kilix-bench — measure a Kilix frame path and store the result privately."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

from kilix_bench.isolate import kilix_home_from_env, sandbox
from kilix_bench.report import render_text, write_run


def _parser():
    parser = argparse.ArgumentParser(prog="kilix-bench")
    parser.add_argument(
        "--suite", choices=("desktop", "render", "text", "video"), default="desktop",
        help="desktop measures a Kilix 95 desk; render measures the graphics SDK; "
             "text compares Kitty cells, Kitty graphics, and Neuronte; "
             "video measures Neuronte RGB composition and decoder startup")
    parser.add_argument("--desktop", default="",
                        help="Path of a Kilix 95 checkout (desktop suite)")
    parser.add_argument("--kilix-home", default="",
                        help="Kilix host checkout that contains config/kilix_sdk")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--neuronte", help="Built Neuronte checkout (video suite)")
    parser.add_argument("--baseline-neuronte", help="Built baseline Neuronte checkout")
    parser.add_argument("--video", help="Optional video file for decoder thread measurements")
    parser.add_argument("--start", type=float, default=0.)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--seek-only", action="store_true", help="only video startup/seek comparisons")
    return parser


def _parse():
    return _parser().parse_args()


def _parse_argv(argv):
    return _parser().parse_args(argv)


def _run(command, env, root, timeout=180):
    try:
        result = subprocess.run(
            command, env=env, capture_output=True, text=True, timeout=timeout)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        sys.stderr.write(result.stdout)
        raise SystemExit(result.returncode or 1)
    return json.loads(result.stdout)


def main(argv=None):
    args = _parse() if argv is None else _parse_argv(argv)
    if (min(args.width, args.height, args.repeats, args.frames) < 1 or
            min(args.warmup, args.start) < 0 or not math.isfinite(args.start)):
        raise SystemExit("invalid size, repetition count or start")
    if args.seek_only and (args.suite != 'video' or not args.video):
        raise SystemExit('--seek-only requires --suite video and --video')
    kilix_home = args.kilix_home or kilix_home_from_env()
    if args.suite != 'video' and (not kilix_home or not os.path.isdir(os.path.join(kilix_home, "config"))):
        raise SystemExit("pass --kilix-home pointing at a Kilix checkout")
    shared = [
        "--width", str(args.width),
        "--height", str(args.height),
        "--repeats", str(args.repeats),
        "--warmup", str(args.warmup),
    ]
    forbidden = [os.path.expanduser("~")]
    extra_env = {}
    if kilix_home:
        kilix_home = os.path.abspath(kilix_home)
        forbidden.append(kilix_home)
    if args.suite == "video":
        if not args.neuronte:
            raise SystemExit("the video suite needs --neuronte")
        command = [sys.executable, '-m', 'kilix_bench.video', *shared,
                   '--start', str(args.start), '--frames', str(args.frames)]
        if args.seek_only:
            command.append('--seek-only')
        for name in ('neuronte', 'baseline_neuronte', 'video'):
            value = getattr(args, name)
            if value:
                path = os.path.abspath(os.path.expanduser(value))
                if not os.path.exists(path):
                    raise SystemExit(f'{name} path does not exist')
                command.extend(['--' + name.replace('_', '-'), path])
                forbidden.append(path)
    elif args.suite == "render":
        command = [sys.executable, "-m", "kilix_bench.render", *shared]
    elif args.suite == "text":
        from kilix_bench.text import libraries
        for key, path in libraries().items():
            extra_env[key] = path
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
    env, root = sandbox(kilix_home, require_state=args.suite != 'video')
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env.update(extra_env)
    # At most eight decoder arms and nine start/seek cells per repetition,
    # plus the kernel measurements. The child retains its own per-run bounds.
    timeout = max(180, (args.repeats + args.warmup) * (9 if args.seek_only else 17) * 65 + 180) if args.suite == 'video' else 180
    payload = _run(command, env, root, timeout=timeout)
    stamped, _latest, text = write_run(payload, forbidden=tuple(forbidden))
    sys.stdout.write(render_text(payload))
    sys.stdout.write(f"\nresults {stamped}\ntext {text}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
