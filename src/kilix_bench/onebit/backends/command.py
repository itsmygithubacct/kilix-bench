"""The command backend: a program reads the item and answers by exit status.

``--backend command --command "kilix-llm judge --stdin"``: the user message
(shown information, then the question) goes to stdin; exit 0 or 1 is the
answer, 2 and above a tool failure.
"""

import shlex
import subprocess
import time

from kilix_bench.onebit.backends.base import Backend

NAME = "command"


class CommandBackend(Backend):
    """A program that reads the item on stdin and exits 0 or 1."""

    kind = "command"
    flavor = "command"

    def __init__(self, argv, env=None, timeout=120.0):
        self.argv = list(argv)
        self.env = env
        self.timeout = timeout

    def describe(self):
        return {"flavor": "command", "program": self.argv[0].rsplit("/", 1)[-1]}

    def supports(self, arm):
        if arm == "command":
            return True, "the exit status is the answer"
        return False, "a command backend has a single arm"

    def ask(self, messages, arm):
        text = messages[-1]["content"]
        started = time.perf_counter()
        try:
            result = subprocess.run(
                self.argv, input=text, capture_output=True, text=True,
                timeout=self.timeout, env=self.env)
        except subprocess.TimeoutExpired:
            return {"reply": None, "latency_ms": self.timeout * 1000,
                    "exit_status": None, "tool_failure": "timeout"}
        latency = (time.perf_counter() - started) * 1000
        status = result.returncode
        answer = status if status in (0, 1) else None
        return {
            # The exit status is the answer; stdout is kept to check it agrees.
            "reply": None if answer is None else str(answer),
            "stdout": result.stdout[:40],
            "latency_ms": round(latency, 2),
            "exit_status": status,
            "tool_failure": None if answer is not None else f"exit {status}",
        }

    def default_arms(self):
        return ["command"]

    def settings(self):
        return {"timeout_s": self.timeout}


def add_arguments(parser):
    group = parser.add_argument_group("command backend")
    group.add_argument(
        "--command",
        help="Program that reads an item on stdin and exits 0 or 1, such as "
             "'kilix-llm judge --stdin'")
    group.add_argument(
        "--pass-env", action="append", default=[],
        help="Environment variable the command keeps (repeatable)")


def check(args, parser):
    if not args.command:
        parser.error("the command backend needs --command")


def spec(args):
    return {"command": args.command}


def build(spec, env=None):
    return CommandBackend(shlex.split(spec["command"]), env=env,
                          timeout=spec.get("timeout", 120.0))


def forbidden(spec):
    return []


def label(args):
    return args.command.split()[0].rsplit("/", 1)[-1] if args.command else ""
