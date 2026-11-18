"""Run OneBit items through one backend and print the result as JSON.

The parent (``python3 -m kilix_bench --suite onebit``) starts this module in a
sandboxed environment and passes a JSON spec as the only argument. Progress
goes to stderr; the result record goes to stdout.
"""

import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from kilix_bench import __version__
from kilix_bench.onebit import contract, items as itemset, scoring
from kilix_bench.onebit.backends import (
    BackendError, ChatBackend, CommandBackend, NeedleBackend)

RESULT_SCHEMA = "kilix-bench.onebit.result/v1"
REPLY_KEEP = 200


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sizer_limit(spec):
    """The input limit for this machine: the option, else the sizer, else none."""
    if spec.get("max_input_tokens"):
        return {"source": "option", "max_input_tokens":
                int(spec["max_input_tokens"])}
    command = spec.get("sizer_command")
    if command == "off":
        return {"source": "off", "max_input_tokens": None}
    argv = shlex.split(command) if command else None
    if argv is None:
        if not shutil.which("plebian-model-sizer"):
            return {"source": "none (no plebian-model-sizer on PATH)",
                    "max_input_tokens": None}
        argv = ["plebian-model-sizer", "recommend", "binary-judge", "--json"]
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                timeout=60)
        value = json.loads(result.stdout).get("max_input_tokens")
        if result.returncode != 0 or not isinstance(value, int):
            raise ValueError(f"exit {result.returncode}")
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return {"source": f"sizer failed ({error})", "max_input_tokens": None}
    return {"source": "plebian-model-sizer", "max_input_tokens": value}


def estimate_tokens(messages):
    """A deliberately high token estimate: one token per three bytes."""
    size = sum(len(message["content"].encode("utf-8")) for message in messages)
    return size // 3 + 16


def make_backend(spec, env=None):
    if spec.get("backend") == "needle":
        return NeedleBackend(spec["needle_engine"],
                             max_bytes=spec.get("needle_max_bytes") or 400,
                             timeout=spec.get("timeout", 120.0))
    if spec.get("command"):
        return CommandBackend(shlex.split(spec["command"]), env=env,
                              timeout=spec.get("timeout", 120.0))
    api_key = os.environ.get(spec["api_key_env"]) if spec.get(
        "api_key_env") else None
    return ChatBackend(
        spec["endpoint"], spec.get("model") or "default",
        flavor=spec.get("flavor", "auto"),
        max_tokens=spec.get("max_tokens", 4),
        thinking=spec.get("thinking", "off"),
        timeout=spec.get("timeout", 120.0),
        api_key=api_key,
        logprobs=not spec.get("no_logprobs"))


def run_arm(backend, arm, selected, limit, progress=None, lengths=None):
    records = []
    for index, item in enumerate(selected, 1):
        chat = contract.messages(item)
        record = {
            "id": item["id"],
            "band": item["band"],
            "domain": item["domain"],
            "form": item["form"],
            "polarity": item["polarity"],
            "split": item["split"],
            "answer": item["answer"],
            "item_sha256": contract.item_sha256(item),
        }
        if lengths and lengths.get(item["id"]) is not None:
            record["input_tokens"] = lengths[item["id"]]
        tokens = record.get("input_tokens") or estimate_tokens(chat)
        if limit and tokens > limit:
            record.update({"skipped": "longer than max_input_tokens",
                           "parsed": None})
            records.append(record)
            continue
        try:
            answer = backend.ask(chat, arm)
        except BackendError as error:
            answer = {"reply": None, "latency_ms": None,
                      "error": str(error)[:300]}
        reply = answer.get("reply")
        record.update({
            "reply": reply[:REPLY_KEEP] if isinstance(reply, str) else reply,
            # A backend that does not answer in text (Needle's tool call)
            # parses its own reply.
            "parsed": answer["parsed"] if "parsed" in answer
            else contract.parse_reply(reply),
            **{key: value for key, value in answer.items()
               if key not in ("reply", "parsed")},
        })
        if answer.get("skipped"):
            record["parsed"] = None
        records.append(record)
        if progress and (index % 50 == 0 or index == len(selected)):
            progress(f"{arm}: {index}/{len(selected)}")
    return records


def run(spec, backend=None, progress=None, env=None):
    """The whole result record for one model."""
    backend = backend or make_backend(spec, env=env)
    selected = itemset.load(split=spec.get("split", "dev"),
                            bands=spec.get("bands") or None)
    if spec.get("limit"):
        selected = selected[: int(spec["limit"])]
    files, combined = itemset.set_digest()
    limit = sizer_limit(spec)
    model = backend.describe()
    if spec.get("model_sha256"):
        model["sha256"] = spec["model_sha256"]
    if spec.get("model_file"):
        model["sha256"] = file_sha256(spec["model_file"])
        model["model_file"] = os.path.basename(spec["model_file"])
    if spec.get("model_label"):
        model["label"] = spec["model_label"]
    arms = spec.get("arms") or (
        ["command"] if backend.kind == "command" else ["raw", "constrained"])
    started = _now()
    lengths = item_lengths(backend, selected, progress)
    result_arms = {}
    try:
        _run_arms(backend, arms, selected, limit, progress, lengths,
                  result_arms)
    finally:
        if hasattr(backend, "close"):
            backend.close()
    return {

        "schema": RESULT_SCHEMA,
        "suite": "onebit",
        "subject": "onebit",
        "tool_version": __version__,
        "started": started,
        "finished": _now(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "backend": backend.kind,
        "endpoint": spec.get("endpoint") if not spec.get("command") else None,
        "model": model,
        "settings": backend.settings(),
        # A backend with no system prompt (Needle) names its own contract.
        "contract": {
            "version": model.get("contract") or contract.CONTRACT_VERSION,
            "system_prompt_sha256": None if model.get("contract")
            else contract.system_prompt_sha256(),
        },
        "items": {
            "split": spec.get("split", "dev"),
            "bands": spec.get("bands") or "all",
            "count": len(selected),
            "limit": spec.get("limit"),
            "files_sha256": files,
            "set_sha256": combined,
            "canary": itemset.canary(),
        },
        "input_limit": limit,
        "item_lengths": length_summary(selected, lengths, result_arms),
        "arms": result_arms,
    }


def _run_arms(backend, arms, selected, limit, progress, lengths, result_arms):
    for arm in arms:
        supported, how = backend.supports(arm)
        if not supported:
            result_arms[arm] = {"supported": False, "reason": how}
            continue
        began = time.perf_counter()
        records = run_arm(backend, arm, selected,
                          limit["max_input_tokens"], progress, lengths)
        result_arms[arm] = {
            "supported": True,
            "how": how,
            "wall_s": round(time.perf_counter() - began, 2),
            "summary": scoring.summarize(records),
            "records": records,
        }


def item_lengths(backend, selected, progress=None):
    """id -> prompt tokens under the model's tokenizer, when it can count."""
    count = getattr(backend, "count_tokens", None)
    if count is None:
        return {}
    lengths = {}
    for item in selected:
        value = count(contract.messages(item))
        if value is None:
            return {}
        lengths[item["id"]] = value
    if progress:
        progress(f"counted tokens of {len(lengths)} items")
    return lengths


def _stats(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"n": len(ordered), "min": ordered[0],
            "median": ordered[len(ordered) // 2], "max": ordered[-1]}


def length_summary(selected, lengths, result_arms):
    """Item lengths per band: tokens when counted, else Needle's bytes."""
    unit = "tokens" if lengths else None
    values = dict(lengths)
    if not values:
        for body in result_arms.values():
            for row in body.get("records") or []:
                if row.get("input_bytes") is not None:
                    values[row["id"]] = row["input_bytes"]
                    unit = "bytes (single-line engine input)"
    if not values:
        return {"unit": None}
    by_band = {}
    for item in selected:
        if item["id"] in values:
            by_band.setdefault(item["band"], []).append(values[item["id"]])
    return {
        "unit": unit,
        "overall": _stats(list(values.values())),
        "by_band": {band: _stats(rows) for band, rows in sorted(
            by_band.items())},
    }


def render(payload):
    """Text form of a result record."""
    model = payload["model"]
    lines = [
        "kilix-bench {version}  OneBit  {name}  ({flavor})".format(
            version=payload["tool_version"],
            name=model.get("label") or model.get("model")
            or model.get("program"),
            flavor=model.get("flavor")),
        "split {split}  {count} items  contract {contract}  "
        "max_tokens {tokens}  thinking {thinking}".format(
            split=payload["items"]["split"], count=payload["items"]["count"],
            contract=payload["contract"]["version"],
            tokens=payload["settings"].get("max_tokens", "-"),
            thinking=payload["settings"].get("thinking_requested", "-")),
        "",
    ]
    for arm, body in payload["arms"].items():
        if not body.get("supported"):
            lines.append(f"arm {arm}: unsupported ({body['reason']})")
            lines.append("")
            continue
        lines.append(f"arm {arm}  ({body['how']}, {body['wall_s']} s)")
        lines += scoring.render_summary(body["summary"])
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv):
    spec = json.loads(argv[0])
    env = dict(os.environ)
    payload = run(spec, progress=lambda text: print(text, file=sys.stderr),
                  env=env)
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
