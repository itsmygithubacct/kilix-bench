"""Run OneBit items through one backend and print the result as JSON.

The parent (``python3 -m kilix_bench --suite onebit``) starts this module in a
sandboxed environment and passes a JSON spec as the only argument. Progress
goes to stderr; the result record goes to stdout.

A backend that names a cache identity (Codex) keeps each answer in a
per-item cache keyed by (item id, item-set sha, contract version, backend
identity, arm), so an interrupted run resumes without asking twice. When a
call fails for a reason outside the answer (``BackendError``: usage limit,
auth, network), the run stops after the items in flight, exits 3 and
prints no result: a partial run is never written as a finished one.
"""

import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from kilix_bench import __version__
from kilix_bench.onebit import contract, items as itemset, scoring
from kilix_bench.onebit import backends as registry
from kilix_bench.onebit.backends import ArmUnsupported, BackendError

RESULT_SCHEMA = "kilix-bench.onebit.result/v1"
REPLY_KEEP = 200
STOPPED_EXIT = 3


class RunStopped(Exception):
    """A backend error stopped the run; nothing is written as finished."""

    def __init__(self, arm, error, answered, total, cached, failed):
        super().__init__(str(error))
        self.arm = arm
        self.error = error
        self.answered = answered
        self.total = total
        self.cached = cached
        self.failed = failed

    def message(self):
        return (f"onebit: stopped in arm {self.arm} after {self.answered} of "
                f"{self.total} items ({self.cached} from the cache); "
                f"backend_error {self.failed} ({self.error.kind}): "
                f"{self.error}. Answers so far are cached; run the same "
                "command again to resume.")


class AnswerCache:
    """One private JSON file per answered item, written atomically."""

    def __init__(self, directory, set_sha256, contract_version, identity):
        self.directory = directory
        self.prefix = [set_sha256, contract_version, identity]
        os.makedirs(directory, mode=0o700, exist_ok=True)
        os.chmod(directory, 0o700)

    def _path(self, item_id, arm):
        key = json.dumps([item_id, *self.prefix, arm])
        name = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return os.path.join(self.directory, name + ".json"), key

    def get(self, item_id, arm):
        path, key = self._path(item_id, arm)
        try:
            with open(path, encoding="utf-8") as handle:
                saved = json.load(handle)
        except (OSError, ValueError):
            return None
        return saved.get("answer") if saved.get("key") == key else None

    def put(self, item_id, arm, answer):
        path, key = self._path(item_id, arm)
        temporary = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                             0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"key": key, "answer": answer}, handle)
        os.replace(temporary, path)


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
    name = spec.get("backend") or ("command" if spec.get("command")
                                   else "chat")
    backend = registry.build(dict(spec, backend=name), env=env)
    if spec.get("jobs"):
        backend.jobs = int(spec["jobs"])
    return backend


def run_arm(backend, arm, selected, limit, progress=None, lengths=None,
            cache=None):
    """One record per item, in item order. Items run ``backend.jobs`` at a
    time; the first backend error stops the arm (``RunStopped``)."""
    stop = threading.Event()
    failures = []
    lock = threading.Lock()
    done = [0]

    def one(item):
        chat = backend.render(item)
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
            return record
        answer = cache.get(item["id"], arm) if cache else None
        if answer is not None:
            record["from_cache"] = True
        else:
            if stop.is_set():
                return None
            try:
                answer = backend.ask(chat, arm)
            except (ArmUnsupported, BackendError) as error:
                stop.set()
                with lock:
                    failures.append(error)
                return None
            if cache:
                cache.put(item["id"], arm, answer)
        reply = answer.get("reply")
        record.update({
            "reply": reply[:REPLY_KEEP] if isinstance(reply, str) else reply,
            "parsed": backend.parse(answer, arm),
            **{key: value for key, value in answer.items()
               if key not in ("reply", "parsed")},
        })
        if answer.get("skipped"):
            record["parsed"] = None
        with lock:
            done[0] += 1
            if progress and (done[0] % 50 == 0 or done[0] == len(selected)):
                progress(f"{arm}: {done[0]}/{len(selected)}")
        return record

    jobs = max(1, int(getattr(backend, "jobs", 1) or 1))
    if jobs == 1:
        records = []
        for item in selected:
            records.append(one(item))
            if stop.is_set():
                break
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            records = list(pool.map(one, selected))
    if failures:
        unsupported = [error for error in failures
                       if isinstance(error, ArmUnsupported)]
        if unsupported:
            raise unsupported[0]
        answered = [row for row in records if row is not None]
        raise RunStopped(arm, failures[0], len(answered), len(selected),
                         sum(bool(row.get("from_cache")) for row in answered),
                         len(failures))
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
    arms = spec.get("arms") or backend.default_arms()
    contract_version = model.get("contract") or contract.CONTRACT_VERSION
    cache = None
    identity = backend.cache_identity()
    if identity and spec.get("cache_dir"):
        cache = AnswerCache(spec["cache_dir"], combined, contract_version,
                            identity)
    started = _now()
    lengths = item_lengths(backend, selected, progress)
    result_arms = {}
    try:
        _run_arms(backend, arms, selected, limit, progress, lengths,
                  result_arms, cache)
    finally:
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
        "endpoint": spec.get("endpoint") if backend.kind == "chat" else None,
        "model": model,
        "settings": backend.settings(),
        # A backend with no system prompt (Needle) names its own contract.
        "contract": {
            "version": contract_version,
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
        "run_totals": run_totals(result_arms),
    }


def _run_arms(backend, arms, selected, limit, progress, lengths, result_arms,
              cache=None):
    for arm in arms:
        supported, how = backend.supports(arm)
        if not supported:
            result_arms[arm] = {"supported": False, "reason": how}
            continue
        began = time.perf_counter()
        try:
            records = run_arm(backend, arm, selected,
                              limit["max_input_tokens"], progress, lengths,
                              cache)
        except ArmUnsupported as error:
            result_arms[arm] = {"supported": False, "reason": str(error)}
            continue
        result_arms[arm] = {
            "supported": True,
            "how": how,
            "jobs": max(1, int(getattr(backend, "jobs", 1) or 1)),
            "wall_s": round(time.perf_counter() - began, 2),
            "from_cache": sum(bool(row.get("from_cache")) for row in records),
            "summary": scoring.summarize(records),
            "records": records,
        }


def run_totals(result_arms):
    """Wall time and usage over every arm of the run."""
    totals = {"wall_s": 0.0, "items": 0, "from_cache": 0}
    tokens = {}
    for body in result_arms.values():
        if not body.get("supported"):
            continue
        totals["wall_s"] += body["wall_s"]
        totals["items"] += len(body["records"])
        totals["from_cache"] += body.get("from_cache", 0)
        cost = body["summary"].get("cost")
        if cost:
            for key, value in cost["totals"].items():
                tokens[key] = tokens.get(key, 0) + value
    totals["wall_s"] = round(totals["wall_s"], 2)
    totals["usage"] = tokens or None
    return totals


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
        cached = (f", {body['from_cache']} answers from the cache"
                  if body.get("from_cache") else "")
        lines.append(f"arm {arm}  ({body['how']}, {body['wall_s']} s"
                     f"{cached})")
        lines += scoring.render_summary(body["summary"])
        lines.append("")
    totals = payload.get("run_totals") or {}
    if totals.get("usage"):
        used = totals["usage"]
        lines.append(
            "run totals: {wall} s, {inp:,} input tokens ({cached:,} cached), "
            "{out:,} output ({reason:,} reasoning)".format(
                wall=totals["wall_s"], inp=used.get("input_tokens", 0),
                cached=used.get("cached_input_tokens", 0),
                out=used.get("output_tokens", 0),
                reason=used.get("reasoning_output_tokens", 0)))
    return "\n".join(lines) + "\n"


def main(argv):
    spec = json.loads(argv[0])
    env = dict(os.environ)
    try:
        payload = run(spec, progress=lambda text: print(text,
                                                        file=sys.stderr),
                      env=env)
    except RunStopped as stopped:
        print(stopped.message(), file=sys.stderr)
        return STOPPED_EXIT
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
