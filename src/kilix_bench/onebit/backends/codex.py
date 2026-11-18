"""The Codex backend: one ``codex exec`` per item, through a named profile.

``--backend codex --codex-profile onebit-luna6-low [--jobs 4]``.

Each item runs ``codex exec -p PROFILE`` in a fresh, empty scratch directory
(no repository, no AGENTS.md of a project), with the read-only sandbox, no
session files (``--ephemeral``), no execpolicy rules (``--ignore-rules``)
and the base config's MCP servers switched off (``-c mcp_servers={}``), so
no tool server starts and no tool schemas reach the prompt. The prompt on
stdin is the contract system prompt, a blank line, and the item's user
message: the same text the other backends send as two chat messages.

The raw arm parses the last message exactly like any text reply. The
constrained arm adds ``--output-schema`` with ``{"answer": "0" | "1"}``
(required, no other keys) and parses that object; it is reported
unsupported only if the CLI rejects the schema.

Codex's JSON events do not name the model, so the result records the
profile file's sha256, the model and effort it sets, and ``codex
--version``. A failed call (usage exhausted, rate limited, signed out,
network, timeout) raises ``BackendError``: the run stops, the answers so
far stay cached, and a later run resumes from them.
"""

import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import tomllib

from kilix_bench.onebit import contract
from kilix_bench.onebit.backends.base import (
    ArmUnsupported, Backend, BackendError, usage)

NAME = "codex"
BACKEND_VERSION = "onebit-codex/v1"
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["0", "1"]}},
    "required": ["answer"],
    "additionalProperties": False,
}
FLAGS = ("--ephemeral", "--skip-git-repo-check", "-s", "read-only",
         "--ignore-rules", "--json")
CONFIG_OVERRIDES = ("mcp_servers={}",)

# What a failed call's message says about its cause. The first match wins.
_CAUSES = (
    ("limit", r"usage limit|rate.?limit|quota|too many requests|\b429\b|"
              r"insufficient|credits|purchase more"),
    ("auth", r"unauthori[sz]ed|\b401\b|\b403\b|forbidden|log ?in|sign(ed)? ?in|"
             r"auth|refresh token|token (has )?expired"),
    ("network", r"network|connect|disconnected|dns|resolve|unreachable|"
                r"error sending request|stream|\b50[234]\b|timed? ?out|reset"),
)


def classify(message):
    """limit, auth, network or error, from a failure message."""
    text = (message or "").lower()
    for kind, pattern in _CAUSES:
        if re.search(pattern, text):
            return kind
    return "error"


def parse_events(stdout):
    """The useful parts of ``codex exec --json`` output.

    Returns the agent messages in order, the turn's usage (or None) and any
    error messages from ``error`` or ``turn.failed`` events.
    """
    messages, errors, found = [], [], None
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        kind = event.get("type")
        if kind == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "agent_message":
                messages.append(item.get("text"))
        elif kind == "turn.completed":
            found = event.get("usage") or {}
        elif kind == "turn.failed":
            error = event.get("error") or {}
            errors.append(str(error.get("message") or error or "turn failed"))
        elif kind == "error":
            errors.append(str(event.get("message") or "error"))
    turn_usage = None
    if found is not None:
        turn_usage = usage(
            input_tokens=found.get("input_tokens"),
            cached_input_tokens=found.get("cached_input_tokens"),
            output_tokens=found.get("output_tokens"),
            reasoning_output_tokens=found.get("reasoning_output_tokens"),
            cache_write_input_tokens=found.get("cache_write_input_tokens"))
    return {"messages": messages, "usage": turn_usage, "errors": errors}


def parse_schema_reply(text):
    """0 or 1 from exactly ``{"answer": "0"}`` or ``{"answer": "1"}``."""
    if text is None:
        return None
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not isinstance(value, dict) or set(value) != {"answer"}:
        return None
    return {"0": 0, "1": 1}.get(value["answer"]) if isinstance(
        value["answer"], str) else None


def _sha256_file(path):
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


class CodexBackend(Backend):
    kind = "codex"
    flavor = "codex"

    def __init__(self, profile, codex_home, binary="codex", jobs=4,
                 timeout=180.0, env=None):
        self.profile = profile
        self.codex_home = codex_home
        self.binary = binary
        self.jobs = jobs
        self.timeout = timeout
        self.env = dict(env if env is not None else os.environ)
        self.env["CODEX_HOME"] = codex_home
        self.env.setdefault("NO_COLOR", "1")
        self.profile_path = os.path.join(codex_home, f"{profile}.config.toml")
        self.profile_sha256 = _sha256_file(self.profile_path)

    def _profile_settings(self):
        try:
            with open(self.profile_path, "rb") as handle:
                data = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError):
            return {}
        return {key: data[key] for key in ("model", "model_reasoning_effort")
                if isinstance(data.get(key), str)}

    def version(self):
        try:
            result = subprocess.run([self.binary, "--version"],
                                    capture_output=True, text=True,
                                    timeout=30, env=self.env)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip()[:80] or None

    def describe(self):
        settings = self._profile_settings()
        return {
            "flavor": "codex",
            "model": settings.get("model") or self.profile,
            "profile": self.profile,
            "profile_sha256": self.profile_sha256,
            "reasoning_effort": settings.get("model_reasoning_effort"),
            "codex_version": self.version(),
            "backend_version": BACKEND_VERSION,
            # The user-level AGENTS.md of CODEX_HOME still reaches the prompt.
            "codex_home_agents_md_sha256": _sha256_file(
                os.path.join(self.codex_home, "AGENTS.md")),
        }

    def cache_identity(self):
        if not self.profile_sha256:
            return None
        return f"{BACKEND_VERSION}|{self.profile}|{self.profile_sha256}"

    def supports(self, arm):
        if arm == "raw":
            return True, ("codex exec, the last message must be exactly "
                          "0 or 1")
        if arm == "constrained":
            return True, ('codex exec --output-schema {"answer": "0"|"1"}')
        return False, f"unknown arm {arm}"

    def argv(self, last_path, schema_path=None):
        argv = [self.binary, "exec", "-p", self.profile]
        for override in CONFIG_OVERRIDES:
            argv += ["-c", override]
        argv += list(FLAGS)
        if schema_path:
            argv += ["--output-schema", schema_path]
        return argv + ["-o", last_path, "-"]

    def ask(self, messages, arm):
        prompt = contract.flat_prompt(messages)
        with tempfile.TemporaryDirectory(prefix="onebit-codex-") as scratch:
            work = os.path.join(scratch, "work")
            os.mkdir(work, 0o700)
            last_path = os.path.join(scratch, "last.txt")
            schema_path = None
            if arm == "constrained":
                schema_path = os.path.join(scratch, "schema.json")
                with open(schema_path, "w", encoding="utf-8") as handle:
                    json.dump(SCHEMA, handle)
            started = time.perf_counter()
            try:
                result = subprocess.run(
                    self.argv(last_path, schema_path), input=prompt,
                    capture_output=True, text=True, timeout=self.timeout,
                    env=self.env, cwd=work)
            except subprocess.TimeoutExpired as error:
                raise BackendError(f"codex took over {self.timeout:.0f} s",
                                   kind="timeout") from error
            except OSError as error:
                raise BackendError(f"cannot run codex: {error}") from error
            latency = (time.perf_counter() - started) * 1000
            events = parse_events(result.stdout)
            reply = None
            if os.path.exists(last_path):
                with open(last_path, encoding="utf-8") as handle:
                    reply = handle.read()
        if events["errors"] or result.returncode != 0:
            message = " | ".join(dict.fromkeys(events["errors"])) or (
                result.stderr.strip().splitlines() or ["no message"])[-1]
            if schema_path and re.search(r"schema", message, re.I) and \
                    classify(message) == "error":
                raise ArmUnsupported(f"codex rejected the output schema: "
                                     f"{message[:200]}")
            raise BackendError(
                f"codex exit {result.returncode}: {message[:300]}",
                kind=classify(message))
        if reply is None and events["messages"]:
            reply = events["messages"][-1]
        return {
            "reply": reply,
            "latency_ms": round(latency, 2),
            "usage": events["usage"],
            "exit_status": result.returncode,
            "messages": len(events["messages"]),
        }

    def parse(self, answer, arm):
        if arm == "constrained":
            return parse_schema_reply(answer.get("reply"))
        return contract.parse_reply(answer.get("reply"))

    def settings(self):
        return {
            "jobs": self.jobs,
            "timeout_s": self.timeout,
            "flags": list(FLAGS),
            "config_overrides": list(CONFIG_OVERRIDES),
            "temperature": "service default",
            "max_tokens": "service default",
            "thinking_requested": "profile reasoning effort",
        }


def default_home():
    return os.path.abspath(os.path.expanduser(
        os.environ.get("CODEX_HOME") or "~/.codex"))


def add_arguments(parser):
    group = parser.add_argument_group("codex backend")
    group.add_argument("--codex-profile", default="",
                       help="Profile: $CODEX_HOME/NAME.config.toml "
                            "(for example onebit-luna6-low)")
    group.add_argument("--codex-home", default="",
                       help="Codex home (default $CODEX_HOME or ~/.codex)")
    group.add_argument("--codex-bin", default="codex",
                       help="The codex executable (default: codex on PATH)")


def check(args, parser):
    if not args.codex_profile:
        parser.error("the codex backend needs --codex-profile")
    home = os.path.abspath(args.codex_home) if args.codex_home else \
        default_home()
    if not os.path.isfile(os.path.join(
            home, f"{args.codex_profile}.config.toml")):
        parser.error(f"no profile {args.codex_profile}.config.toml in the "
                     "codex home")


def spec(args):
    return {"codex_profile": args.codex_profile,
            "codex_home": os.path.abspath(args.codex_home) if args.codex_home
            else default_home(),
            "codex_bin": args.codex_bin}


def build(spec, env=None):
    return CodexBackend(spec["codex_profile"], spec["codex_home"],
                        binary=spec.get("codex_bin") or "codex",
                        jobs=spec.get("jobs") or 4,
                        timeout=spec.get("timeout") or 180.0, env=env)


def forbidden(spec):
    return [spec["codex_home"]]


def label(args):
    return args.codex_profile
