"""Command line for draft-judge."""

import argparse
import sys

from kilix_bench.draft_judge.workflow import execute, parse_spec

SEARCH = (
    "The llama-server binary is taken from DRAFT_JUDGE_LLAMA_SERVER, "
    "then --llama-server, then a known build, then PATH."
)


def parser():
    tool = argparse.ArgumentParser(
        prog="draft-judge",
        description=(
            "Draft replies from GGUF models and keep the highest score. "
            "With no arguments this opens a monitor in the terminal. " + SEARCH
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  draft-judge [--model-dir DIR] [--output DIR] [--gpu-device NAME]\n"
            "  draft-judge --cli\n"
            "  draft-judge --cli run --draft model.gguf[:cpu|gpu] "
            "--judge judge.gguf --prompt TEXT\n"
            "\n"
            "The monitor needs a terminal. --cli is the text menu, and\n"
            "--cli run is the same workflow without the menu.\n"
            "Defaults: four drafts, gpu, and schedule rotate."
        ),
    )
    return tool


def run_parser():
    tool = argparse.ArgumentParser(
        prog="draft-judge run",
        description="Run one draft-judge workflow. " + SEARCH,
    )
    tool.add_argument("--draft", action="append", required=True, metavar="FILE[:cpu|gpu]",
                      help="draft GGUF, optionally :cpu or :gpu (default gpu)")
    tool.add_argument("--judge", required=True, metavar="FILE[:cpu|gpu]",
                      help="judge GGUF, optionally :cpu or :gpu (default gpu)")
    tool.add_argument("--drafts", type=int, default=4, help="how many drafts to write (default 4)")
    tool.add_argument("--schedule", choices=("together", "rotate"), default="rotate",
                      help="load every model at once, or one at a time (default rotate)")
    tool.add_argument("--prompt", help="prompt text")
    tool.add_argument("--prompt-file", help="file whose contents are the prompt")
    tool.add_argument("--code", action="store_true",
                      help="use the coding prompt (a careful Python programmer)")
    tool.add_argument("--output", help="run directory (default: a new directory under the state home)")
    tool.add_argument("--llama-server", help="llama-server binary")
    tool.add_argument("--gpu-device", help="device name for every GPU model")
    return tool


def show(code, directory, report):
    if directory is None:
        return code
    sys.stdout.write(report)
    if not report.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.write(f"{directory}\n")
    return code


def read_prompt(args):
    if (args.prompt is None) == (args.prompt_file is None):
        sys.stderr.write("draft-judge: pass exactly one of --prompt and --prompt-file\n")
        return None
    if args.prompt is not None:
        return args.prompt
    try:
        with open(args.prompt_file, encoding="utf-8") as handle:
            return handle.read()
    except OSError as exc:
        sys.stderr.write(f"draft-judge: {exc}\n")
        return None


def cmd_run(argv):
    args = run_parser().parse_args(argv)
    if args.drafts < 1:
        sys.stderr.write("draft-judge: --drafts must be at least 1\n")
        return 2
    prompt = read_prompt(args)
    if prompt is None:
        return 2
    drafts = [parse_spec(item) for item in args.draft]
    judge = parse_spec(args.judge)
    code, directory, report = execute({
        "draft_models": drafts,
        "judge": judge,
        "draft_count": args.drafts,
        "schedule": args.schedule,
        "prompt": prompt,
        "code": args.code,
        "output": args.output,
        "llama_server": args.llama_server,
        "gpu_device": args.gpu_device,
    })
    return show(code, directory, report)


def text_main(argv):
    """The text menu and the run command. Reach them with --cli."""
    if argv in (["-h"], ["--help"]):
        parser().print_help()
        return 0
    if not argv or argv[0] == "tui":
        rest = argv[1:] if argv else []
        if "-h" in rest or "--help" in rest:
            from kilix_bench.draft_judge.tui import parser as tui_parser
            tui_parser().print_help()
            return 0
        if not sys.stdin.isatty():
            sys.stderr.write("draft-judge: the menu needs a terminal; use --cli run\n")
            return 2
        from kilix_bench.draft_judge.tui import run_tui
        return run_tui(rest)
    if argv[0] != "run":
        parser().print_help()
        return 2
    return cmd_run(argv[1:])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--cli"]:
        return text_main(argv[1:])
    if argv in (["-h"], ["--help"]):
        parser().print_help()
        return 0
    if argv and argv[0] in ("run", "tui"):
        sys.stderr.write("draft-judge: the text menu and run command are behind --cli\n")
        return 2
    from kilix_bench.draft_judge.display import run_monitor
    return run_monitor(argv)
