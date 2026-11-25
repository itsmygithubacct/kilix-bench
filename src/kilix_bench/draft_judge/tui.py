"""Plain prompts for the same choices as draft-judge run."""

import argparse
import os
from pathlib import Path
import sys

from kilix_bench.draft_judge.cli import show
from kilix_bench.draft_judge.workflow import ModelSpec, discover_models, execute


def parser():
    tool = argparse.ArgumentParser(
        prog="draft-judge tui",
        description=(
            "Choose draft models, a judge, and a prompt in the terminal. "
            "The llama-server binary is taken from DRAFT_JUDGE_LLAMA_SERVER, "
            "then --llama-server, then a known build, then PATH."
        ),
    )
    tool.add_argument("--model-dir", action="append", default=[],
                      help="extra directory of GGUF files")
    tool.add_argument("--output", help="run directory")
    tool.add_argument("--llama-server", help="llama-server binary")
    tool.add_argument("--gpu-device", help="device name for every GPU model")
    return tool


def ask(label):
    return input(label)


def choose_model(models, label, allow_blank):
    while True:
        raw = ask(label).strip()
        if not raw:
            if allow_blank:
                return None
            print("Enter a number or a path.")
            continue
        if raw.isdigit():
            number = int(raw)
            if 1 <= number <= len(models):
                return str(models[number - 1])
            print("That number is not in the list.")
            continue
        return os.path.expanduser(raw)


def choose_device(name):
    while True:
        raw = ask(f"Device for {name} (cpu or gpu) [gpu]: ").strip().casefold()
        if raw in ("", "gpu"):
            return "gpu"
        if raw == "cpu":
            return "cpu"
        print("Enter cpu or gpu.")


def choose_count():
    while True:
        raw = ask("How many drafts [4]: ").strip()
        if not raw:
            return 4
        try:
            number = int(raw)
        except ValueError:
            number = 0
        if number >= 1:
            return number
        print("Enter a positive integer.")


def choose_schedule():
    while True:
        raw = ask("Schedule (together or rotate) [rotate]: ").strip().casefold()
        if raw in ("", "rotate"):
            return "rotate"
        if raw == "together":
            return "together"
        print("Enter together or rotate.")


def choose_prompt():
    raw = ask("Prompt file (blank to type a prompt): ").strip()
    if raw:
        path = Path(os.path.expanduser(raw))
        if not path.is_file():
            print("That file is not available.")
            return choose_prompt()
        return path.read_text(encoding="utf-8")
    text = ask("Prompt: ")
    if not text.strip():
        print("Enter a prompt.")
        return choose_prompt()
    return text


def choose_code():
    while True:
        raw = ask("Coding prompt (y or n) [n]: ").strip().casefold()
        if raw in ("", "n", "no"):
            return False
        if raw in ("y", "yes"):
            return True
        print("Enter y or n.")


def show_models(models):
    if not models:
        print("No GGUF files found. Type a path.")
        return
    print("Models:")
    for index, path in enumerate(models, 1):
        print(f"  {index}  {path.name}  {path}")


def run_tui(argv):
    args = parser().parse_args(argv)
    try:
        return ask_and_run(args)
    except EOFError:
        sys.stderr.write("draft-judge: end of input\n")
        return 1


def ask_and_run(args):
    models = discover_models(args.model_dir)
    print("Draft models. Enter a number or a path. A blank line finishes the list.")
    show_models(models)
    drafts = []
    while True:
        picked = choose_model(
            models, "Draft model (number or path, blank to finish): ", allow_blank=True)
        if picked is None:
            if drafts:
                break
            print("Choose at least one draft model.")
            continue
        device = choose_device(Path(picked).name)
        drafts.append(ModelSpec(os.path.abspath(picked), device))
    picked = choose_model(models, "Judge (number or path): ", allow_blank=False)
    judge = ModelSpec(os.path.abspath(picked), choose_device(Path(picked).name))
    code, directory, report = execute({
        "draft_models": drafts,
        "judge": judge,
        "draft_count": choose_count(),
        "schedule": choose_schedule(),
        "prompt": choose_prompt(),
        "code": choose_code(),
        "output": args.output,
        "llama_server": args.llama_server,
        "gpu_device": args.gpu_device,
    })
    return show(code, directory, report)
