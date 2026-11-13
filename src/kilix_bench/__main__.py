"""kilix-bench — measure a Kilix frame path and store the result privately."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from kilix_bench.isolate import kilix_home_from_env, sandbox
from kilix_bench.report import render_text, results_dir, write_run


def _parser():
    parser = argparse.ArgumentParser(prog="kilix-bench")
    parser.add_argument(
        "--suite", choices=("desktop", "render", "text", "fonts", "neuronte"), default="desktop",
        help="desktop measures a Kilix 95 desk; render measures the graphics SDK; "
             "text compares application frame paths; fonts compares Neuronte and FreeType; "
             "neuronte measures every split-v5 expert")
    parser.add_argument("--desktop", default="",
                        help="Path of a Kilix 95 checkout (desktop suite)")
    parser.add_argument("--kilix-home", default="",
                        help="Kilix host checkout that contains config/kilix_sdk")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    from kilix_bench.fonts import add_arguments
    add_arguments(parser)
    from kilix_bench.neuronte import add_arguments as expert_arguments
    expert_arguments(parser)
    parser.set_defaults(cells=None)
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
    if args.repeats is None:
        args.repeats = 20 if args.suite == "fonts" else 5 if args.suite == "neuronte" else 12
    if args.cells is None:
        from kilix_bench.fonts import parse_cells
        args.cells = parse_cells("8x16,16x32,48x96" if args.suite == "neuronte" else "6x12,8x16,10x20")
    kilix_home = args.kilix_home or kilix_home_from_env()
    if args.suite not in ("fonts", "neuronte") and (not kilix_home or not os.path.isdir(os.path.join(kilix_home, "config"))):
        raise SystemExit("pass --kilix-home pointing at a Kilix checkout")
    if args.suite in ("fonts", "neuronte"):
        return _fonts(args, kilix_home)
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


def _fonts(args, kilix_home):
    if args.suite == "neuronte":
        from kilix_bench.neuronte import inputs
    else:
        from kilix_bench.fonts import inputs
    try:
        paths = inputs(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    # The CPU font comparison needs no running host or state library.
    env, root = sandbox(os.path.abspath(kilix_home) if kilix_home else "", require_state=False)
    try:
        src = str(Path(__file__).resolve().parents[1])
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ-") + args.suite
        artifacts = results_dir() / stamp
        artifacts.mkdir(mode=0o700)
        env["KILIX_BENCH_ARTIFACT_DIR"] = str(artifacts)
        env["NEURONTE_LIBRARY"] = str(paths["library"])
        env["NEURONTE_WEIGHTS"] = str(paths["weights"])
        command = [sys.executable, "-m", "kilix_bench." + args.suite, "--cells",
                   ",".join(f"{w}x{h}" for w, h in args.cells), "--seed", str(args.seed),
                   "--job-timeout", str(args.job_timeout),
                   "--repeats", str(args.repeats), "--warmup", str(args.warmup)]
        if args.suite == "neuronte":
            command += ["--experts", args.experts, "--expert-cases", args.expert_cases,
                        "--expert-passes", str(args.expert_passes)]
            if args.braille_experiment:
                command += ["--braille-experiment", str(paths["experiment"])]
            if args.ascii_experiment:
                command += ["--ascii-experiment", str(paths["experiment"])]
            if args.expert_experiment:
                command += ["--expert-experiment", str(paths["experiment"])]
        else:
            command += ["--font", str(paths["font"]), "--font-license", str(paths["font_license"]),
                        "--corpora", args.corpora, "--raster-calls", str(args.raster_calls),
                        "--frames", str(args.frames), "--cache-entries", str(args.cache_entries),
                        "--churn-entries", str(args.churn_entries), "--hinting", args.hinting]
        if args.cpu is not None:
            command += ["--cpu", str(args.cpu)]
        # Each native job has its own 180s bound. A campaign is many jobs.
        payload = _run(command, env, root, timeout=None)
        payload["artifact_directory"] = artifacts.name
        forbidden = (str(Path.home()), *(str(path) for path in paths.values()))
        stamped, _, text = write_run(payload, forbidden=forbidden)
        sys.stdout.write(render_text(payload))
        sys.stdout.write(f"\nresults {stamped}\ntext {text}\nartifacts {artifacts}\n")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
