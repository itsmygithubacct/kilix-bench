"""The fixed contract every model is given, and how an item becomes a chat."""

import hashlib
from pathlib import Path

CONTRACT_VERSION = "onebit-contract/v1"
_PROMPT_FILE = Path(__file__).resolve().parent / "system-prompt-v1.txt"


def system_prompt():
    return _PROMPT_FILE.read_text(encoding="utf-8").strip()


def system_prompt_sha256():
    return hashlib.sha256(system_prompt().encode("utf-8")).hexdigest()


def user_message(item):
    """The text a model sees for one item: the shown information, then the ask."""
    context = item.get("context")
    if context:
        return (f"<information>\n{context}\n</information>\n\n"
                f"{item['prompt']}")
    return item["prompt"]


def messages(item):
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user_message(item)},
    ]


def flat_prompt(chat):
    """The chat as one text, for a backend that takes a single prompt: the
    system prompt, a blank line, then the user message."""
    return "\n\n".join(message["content"] for message in chat)


def item_sha256(item):
    """Identity of what was asked and the key, so two runs can be paired."""
    blob = "\0".join((
        item.get("context") or "", item["prompt"], str(item["answer"])))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def parse_reply(text):
    """0 or 1 when the reply is exactly that digit after stripping whitespace.

    Anything else ("0.", "Yes", "", "01", a thinking block) is a format
    failure and returns None.
    """
    if text is None:
        return None
    stripped = text.strip()
    if stripped == "0":
        return 0
    if stripped == "1":
        return 1
    return None
