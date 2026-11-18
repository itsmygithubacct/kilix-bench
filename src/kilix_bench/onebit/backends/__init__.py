"""Ways to ask a model one OneBit item, one module per backend.

``REGISTRY`` maps each ``--backend`` name to its module; ``base`` says what
a module provides. Every backend gets the same rendered item
(``contract.messages``) and, unless its reply is not plain text, the same
parser (``contract.parse_reply``).
"""

from kilix_bench.onebit.backends import chat, codex, command, needle
from kilix_bench.onebit.backends.base import (
    USAGE_KEYS, ArmUnsupported, Backend, BackendError, usage)
from kilix_bench.onebit.backends.chat import (
    CHAT_PATH, GRAMMAR, ChatBackend, chat_usage, first_token_logprobs)
from kilix_bench.onebit.backends.codex import (
    CodexBackend, parse_events, parse_schema_reply)
from kilix_bench.onebit.backends.command import CommandBackend
from kilix_bench.onebit.backends.needle import (
    NEEDLE_CONTRACT, NEEDLE_CONTRACT_VERSION, NEEDLE_LEAD, NEEDLE_MAX_BYTES,
    NEEDLE_TOOLS, NeedleBackend, needle_input, parse_needle_calls)

REGISTRY = {module.NAME: module for module in (chat, command, needle, codex)}


def module_for(name):
    try:
        return REGISTRY[name]
    except KeyError:
        raise SystemExit(f"unknown backend {name}") from None


def build(spec, env=None):
    """The backend object a run spec names."""
    return module_for(spec.get("backend") or "chat").build(spec, env)


__all__ = [
    "REGISTRY", "build", "module_for", "USAGE_KEYS", "ArmUnsupported",
    "Backend", "BackendError", "usage", "CHAT_PATH", "GRAMMAR",
    "ChatBackend", "chat_usage", "first_token_logprobs", "CodexBackend",
    "parse_events", "parse_schema_reply", "CommandBackend",
    "NEEDLE_CONTRACT", "NEEDLE_CONTRACT_VERSION", "NEEDLE_LEAD",
    "NEEDLE_MAX_BYTES", "NEEDLE_TOOLS", "NeedleBackend", "needle_input",
    "parse_needle_calls",
]
