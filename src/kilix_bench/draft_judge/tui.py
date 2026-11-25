"""Plain prompts for the same choices as draft-judge run."""

import argparse
import os
from pathlib import Path
import sys

from kilix_bench.draft_judge.cli import show
from kilix_bench.draft_judge.workflow import ModelSpec, discover_models, execute


class Back(Exception):
    """The user left this question for the previous one."""


class Quit(Exception):
    """The user left the menu before a run."""


KEEP = object()

HELP = (
    "A number, part of a file name, or a path selects a model.\n"
    "Punctuation is ignored, so d3:2b can match a name like d3-2b.\n"
    "Add cpu or gpu on the same line to set the device. The default is gpu.\n"
    "b goes back, l lists the models, p shows full paths, q quits."
)

STEPS = ("drafts", "judge", "count", "schedule", "prompt", "code", "confirm")
EDIT = {
    "1": "drafts",
    "2": "judge",
    "3": "count",
    "4": "schedule",
    "5": "prompt",
    "6": "code",
}


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


def take(label, free_text=False):
    raw = ask(label)
    token = raw.strip().casefold()
    if free_text:
        if token == "..":
            raise Back()
        return raw
    if token in ("q", "quit"):
        raise Quit()
    if token in ("b", "back"):
        raise Back()
    return raw


def normalize_name(text):
    folded = "".join(ch for ch in text.casefold() if ch.isalnum())
    if folded.endswith("gguf") and len(folded) > 4:
        folded = folded[:-4]
    return folded


def match_models(models, query):
    """Models whose file name contains the query, ignoring punctuation.

    One exact stem or file name wins over several partial matches.
    """
    needle = normalize_name(query)
    if not needle:
        return []
    hits = []
    for index, path in enumerate(models, 1):
        folded = {normalize_name(path.name), normalize_name(path.stem)}
        if any(needle in item for item in folded):
            hits.append((index, path))
    exact = [
        hit for hit in hits
        if needle in {normalize_name(hit[1].name), normalize_name(hit[1].stem)}
    ]
    if len(exact) == 1:
        return exact
    return hits


def split_device(text):
    """Split a trailing cpu or gpu from a model choice."""
    raw = text.strip()
    folded = raw.casefold()
    for suffix in (":cpu", ":gpu"):
        if folded.endswith(suffix) and len(raw) > len(suffix):
            return raw[: -len(suffix)].rstrip(), suffix[1:]
    parts = raw.rsplit(None, 1)
    if len(parts) == 2 and parts[1].casefold() in ("cpu", "gpu") and parts[0].strip():
        return parts[0].strip(), parts[1].casefold()
    return raw, None


def looks_explicit_path(text):
    return (
        text.startswith("~")
        or text.startswith("/")
        or text.startswith("./")
        or text.startswith("../")
        or "/" in text
    )


def existing_file(text):
    path = Path(os.path.expanduser(text))
    if path.is_file():
        return path.resolve()
    return None


def resolve_model(models, text):
    """A Path, a list of (number, Path) when several match, or None to ask again."""
    if text.isdigit():
        number = int(text)
        if 1 <= number <= len(models):
            return models[number - 1]
        print("That number is not in the list.")
        return None
    if looks_explicit_path(text):
        found = existing_file(text)
        if found is None:
            print(f"No file at {Path(os.path.expanduser(text))}.")
            return None
        return found
    for path in models:
        if path.name.casefold() == text.casefold():
            return path
    hits = match_models(models, text)
    if len(hits) == 1:
        print(f"Matched {hits[0][0]}  {hits[0][1].name}")
        return hits[0][1]
    if len(hits) > 1:
        return hits
    found = existing_file(text)
    if found is not None:
        return found
    print("No model matches. Enter a number, part of a name, or a path.")
    return None


def size_label(path):
    try:
        nbytes = path.stat().st_size
    except OSError:
        return ""
    gib = 1024 ** 3
    mib = 1024 ** 2
    if nbytes >= gib:
        return f"{nbytes / gib:.1f}G"
    if nbytes >= mib:
        return f"{nbytes / mib:.0f}M"
    return f"{nbytes}B"


def display_name(path, models):
    names = [item.name for item in models]
    if names.count(path.name) > 1:
        return f"{path.name}  ({path.parent})"
    return path.name


def show_models(models):
    if not models:
        print("No GGUF files found. Type a path.")
        return
    print("Models:")
    width = max(len(display_name(path, models)) for path in models)
    for index, path in enumerate(models, 1):
        size = size_label(path)
        gap = f"  {size}" if size else ""
        print(f"  {index:>2}  {display_name(path, models):<{width}}{gap}")


def show_paths(models):
    if not models:
        print("No GGUF files found.")
        return
    for index, path in enumerate(models, 1):
        print(f"  {index:>2}  {path}")


def show_hits(hits, device):
    print("Several models match. Enter one of these numbers:")
    for index, path in hits:
        print(f"  {index:>2}  {path.name}")
    if device:
        print(f"Device {device} will be used for the model you pick.")


def prompt_model(models, label, allow_blank, keep=False):
    """Ask until a model is chosen.

    Blank finishes the draft list, or keeps the current judge when keep is set.
    A returned device is None when the line did not name one.
    """
    held_device = None
    pending_hits = False
    while True:
        try:
            raw = take(label)
        except Back:
            if pending_hits or held_device:
                pending_hits = False
                held_device = None
                print("Back at the model list.")
                continue
            raise
        if not raw.strip():
            if allow_blank:
                return None
            if keep:
                return KEEP
            print("Enter a number, part of a name, or a path.")
            continue
        token = raw.strip().casefold()
        if token in ("?", "help", "l", "list"):
            print(HELP)
            show_models(models)
            continue
        if token in ("p", "paths"):
            show_paths(models)
            continue
        model_text, device = split_device(raw)
        if device:
            held_device = device
        chosen = resolve_model(models, model_text)
        if chosen is None:
            continue
        if isinstance(chosen, list):
            pending_hits = True
            show_hits(chosen, held_device)
            continue
        pending_hits = False
        use_device = device if device else held_device
        held_device = None
        return chosen, use_device


def choose_device(name):
    while True:
        raw = take(f"Device for {name} (cpu or gpu) [gpu]: ").strip().casefold()
        if raw in ("", "gpu"):
            return "gpu"
        if raw == "cpu":
            return "cpu"
        print("Enter cpu or gpu.")


def attach_device(path, device):
    if device is None:
        device = choose_device(Path(path).name)
    return ModelSpec(os.path.abspath(path), device)


def show_drafts(drafts):
    if not drafts:
        return
    print("Drafts so far:")
    for index, spec in enumerate(drafts, 1):
        print(f"  {index}  {spec.name}  {spec.device}")


def collect_drafts(models, drafts, introduce):
    if introduce:
        print(HELP)
        print("A blank line finishes the draft list. b removes the last draft.")
    else:
        print("Drafts. A blank line keeps this list. b removes the last draft.")
    show_models(models)
    while True:
        show_drafts(drafts)
        try:
            picked = prompt_model(
                models, "Draft model (blank when the list is done): ", allow_blank=True)
        except Back:
            if drafts:
                removed = drafts.pop()
                print(f"Removed {removed.name}.")
                continue
            raise
        if picked is None:
            if drafts:
                return
            print("Choose at least one draft model.")
            continue
        path, device = picked
        try:
            spec = attach_device(path, device)
        except Back:
            print("That model was not added.")
            continue
        drafts.append(spec)
        print(f"Added {spec.name} on {spec.device}.")


def collect_judge(models, current):
    print("Judge. The same shortcuts select the model.")
    show_models(models)
    if current is None:
        label = "Judge (number, name, or path): "
    else:
        label = f"Judge (Enter keeps {current.name}): "
    while True:
        picked = prompt_model(models, label, allow_blank=False, keep=current is not None)
        if picked is KEEP:
            print(f"Keeping {current.name} on {current.device}.")
            return current
        path, device = picked
        try:
            spec = attach_device(path, device)
        except Back:
            print("Still choosing the judge.")
            continue
        print(f"Judge is {spec.name} on {spec.device}.")
        return spec


def choose_count(current):
    while True:
        raw = take(f"How many drafts [{current}]: ").strip()
        if not raw:
            return current
        try:
            number = int(raw)
        except ValueError:
            number = 0
        if number >= 1:
            return number
        print("Enter a positive integer.")


def choose_schedule(current):
    print("together keeps every model loaded. rotate loads one model at a time.")
    while True:
        raw = take(f"Schedule (together or rotate) [{current}]: ").strip().casefold()
        if not raw:
            return current
        if raw in ("rotate", "together"):
            return raw
        print("Enter together or rotate.")


def strip_wrapping_quotes(text):
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        inner = text[1:-1]
        if inner.strip():
            return inner
    return text


def one_line(text, limit=80):
    flat = " ".join(text.split())
    if len(flat) > limit:
        return flat[: limit - 3] + "..."
    return flat


def read_prompt_file():
    while True:
        raw = take("Prompt file: ").strip()
        if not raw:
            print("Enter a path, or b to go back.")
            continue
        path = Path(os.path.expanduser(raw))
        if not path.is_file():
            print(f"No file at {path}.")
            continue
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"Could not read {path}: {exc}")


def choose_prompt(current):
    if current is not None:
        print(f"Current prompt: {one_line(current)}")
        while True:
            raw = take(
                "Enter keeps it, f reads a file, .. goes back, or type a new prompt: ",
                free_text=True,
            )
            if not raw.strip():
                return current
            if raw.strip().casefold() == "f":
                return read_prompt_file()
            text = strip_wrapping_quotes(raw)
            if not text.strip():
                print("Enter a prompt.")
                continue
            print(f"Using prompt: {one_line(text)}")
            return text
    while True:
        raw = take("Prompt file (blank to type a prompt): ").strip()
        if raw:
            path = Path(os.path.expanduser(raw))
            if not path.is_file():
                print(f"No file at {path}.")
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                print(f"Could not read {path}: {exc}")
                continue
            print(f"Using prompt file {path.name}.")
            return text
        text = take("Prompt (.. goes back): ", free_text=True)
        text = strip_wrapping_quotes(text)
        if not text.strip():
            print("Enter a prompt.")
            continue
        print(f"Using prompt: {one_line(text)}")
        return text


def choose_code(current):
    default = "y" if current else "n"
    print("y asks for a Python function. n sends the prompt as you typed it.")
    while True:
        raw = take(f"Coding prompt (y or n) [{default}]: ").strip().casefold()
        if raw == "":
            return current
        if raw in ("n", "no"):
            return False
        if raw in ("y", "yes"):
            return True
        print("Enter y or n.")


def show_ready(state):
    print()
    print("Ready.")
    for index, spec in enumerate(state["drafts"], 1):
        title = "drafts   " if index == 1 else "         "
        print(f"  {title}{index}  {spec.name}  {spec.device}")
    judge = state["judge"]
    print(f"  judge    {judge.name}  {judge.device}")
    print(f"  count    {state['count']}")
    print(f"  schedule {state['schedule']}")
    print(f"  code     {'yes' if state['code'] else 'no'}")
    print(f"  prompt   {one_line(state['prompt'])}")
    print("Enter runs. b changes the last answer. q quits.")
    print("Change one answer: 1 drafts, 2 judge, 3 count, 4 schedule, 5 prompt, 6 code.")


def confirm(state):
    show_ready(state)
    while True:
        raw = take("Run: ").strip().casefold()
        if raw == "":
            return "run"
        if raw in EDIT:
            return raw
        print("Enter runs, b changes the last answer, or type 1-6 to change one answer.")


def run_step(step, models, state, editing):
    if step == "drafts":
        collect_drafts(models, state["drafts"], introduce=not editing)
        return None
    if step == "judge":
        state["judge"] = collect_judge(models, state["judge"])
        return None
    if step == "count":
        state["count"] = choose_count(state["count"])
        return None
    if step == "schedule":
        state["schedule"] = choose_schedule(state["schedule"])
        return None
    if step == "prompt":
        state["prompt"] = choose_prompt(state["prompt"])
        return None
    if step == "code":
        state["code"] = choose_code(state["code"])
        return None
    return confirm(state)


def ask_and_run(args):
    models = discover_models(args.model_dir)
    state = {
        "drafts": [],
        "judge": None,
        "count": 4,
        "schedule": "rotate",
        "prompt": None,
        "code": False,
    }
    index = 0
    editing = False
    try:
        while index < len(STEPS):
            step = STEPS[index]
            try:
                action = run_step(step, models, state, editing)
            except Back:
                if step == "confirm":
                    index = STEPS.index("code")
                    editing = True
                    continue
                if editing:
                    index = STEPS.index("confirm")
                    editing = False
                    continue
                if index == 0:
                    print("The draft list is the first step.")
                    continue
                index -= 1
                continue
            if step == "confirm":
                if action == "run":
                    break
                index = STEPS.index(EDIT[action])
                editing = True
                continue
            if editing:
                index = STEPS.index("confirm")
                editing = False
            else:
                index += 1
    except Quit:
        sys.stderr.write("draft-judge: cancelled\n")
        return 1
    code, directory, report = execute({
        "draft_models": state["drafts"],
        "judge": state["judge"],
        "draft_count": state["count"],
        "schedule": state["schedule"],
        "prompt": state["prompt"],
        "code": state["code"],
        "output": args.output,
        "llama_server": args.llama_server,
        "gpu_device": args.gpu_device,
    })
    return show(code, directory, report)


def run_tui(argv):
    args = parser().parse_args(argv)
    try:
        return ask_and_run(args)
    except EOFError:
        sys.stderr.write("draft-judge: end of input\n")
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("\ndraft-judge: cancelled\n")
        return 1
