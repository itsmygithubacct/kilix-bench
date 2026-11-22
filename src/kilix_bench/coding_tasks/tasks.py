"""Pinned coding tasks for lm-eval. Only the isolated grader executes predictions."""

from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.request import urlopen

CRUX = ("https://huggingface.co/datasets/cruxeval-org/cruxeval/resolve/"
        "b96af0450242eb4da433032b90998f25588a5d0f/test.jsonl",
        "8368b81047dc5014e4caf5a2f97604eff7644e0ecd7415e3ceeb184bbc2e0c96")
HUMAN = ("https://github.com/evalplus/humanevalplus_release/releases/download/"
         "v0.1.10/HumanEvalPlus.jsonl.gz",
         "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101")
EVALPLUS = ("https://files.pythonhosted.org/packages/13/66/"
            "de5ac13e00c681cff679c981b1bae98ab3b940fa3db2f0d8d70a0213424d/"
            "evalplus-0.3.1-py3-none-any.whl",
            "cd601debb67419113d10ac5c3317689d847f27de5d8cf3837975f3cab571b75d")


@lru_cache(maxsize=3)
def artifact(spec):
    url, digest = spec
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "kilix-bench/coding"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (digest + "-" + url.rsplit("/", 1)[1])
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return path
    with urlopen(url, timeout=120) as source:
        data = source.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Checksum mismatch for {url}")
    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as tmp:
        try:
            tmp.write(data)
            tmp.flush()
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
    return path


def sandbox(mode):
    wheel = (["--ro-bind", str(artifact(EVALPLUS)), "/evalplus.whl"]
             if mode == "humaneval" else [])
    return ["bwrap", "--unshare-user", "--unshare-all", "--disable-userns", "--cap-drop", "ALL",
            "--new-session", "--die-with-parent", "--ro-bind", "/usr", "/usr",
            "--ro-bind-try", "/lib", "/lib", "--ro-bind-try", "/lib64", "/lib64",
            "--ro-bind-try", "/etc/alternatives", "/etc/alternatives",
            "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
            "--dev", "/dev", "--proc", "/proc", "--size", "67108864", "--tmpfs", "/tmp",
            "--size", "16777216", "--tmpfs", "/dev/shm", "--chdir", "/tmp", "--clearenv", "--setenv", "OPENBLAS_NUM_THREADS", "1",
            "--setenv", "EVALPLUS_MAX_MEMORY_BYTES", "1073741824", *wheel,
            "--ro-bind", str(Path(__file__).with_name("grader.py").resolve()), "/grader.py",
            "--", "/usr/bin/python3", "-I", "/grader.py"]


@lru_cache(maxsize=2)
def dataset(spec):
    from datasets import Dataset, DatasetDict

    mode = "humaneval" if spec == HUMAN else "input"
    try:
        subprocess.run(sandbox(mode), input=json.dumps([mode, None, None]), text=True,
                       capture_output=True, timeout=30, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise RuntimeError(f"Cannot start the bubblewrap Python grader: {detail.strip()}") from exc
    if spec == HUMAN:
        with gzip.open(artifact(HUMAN), "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        # Arrow columns cannot mix argument types across different functions.
        for row in rows:
            for field in ("base_input", "plus_input"):
                row[field] = json.dumps(row[field])
        count, id_field = 164, "task_id"
    else:
        with artifact(CRUX).open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        count, id_field = 800, "id"
    if len(rows) != count or len({row[id_field] for row in rows}) != count:
        raise ValueError(f"Expected {count} unique problems in {spec[0]}")
    return DatasetDict(test=Dataset.from_list(rows))


def crux_dataset(**kwargs):
    return dataset(CRUX)


def human_dataset(**kwargs):
    return dataset(HUMAN)


def input_prompt(doc):
    return (doc["code"] + "\n\nFind arguments for f that produce " + doc["output"] +
            ". Multiple inputs may work. Return only the Python argument expressions "
            "as they would appear inside f(...).\nArguments: ")


def output_prompt(doc):
    return (doc["code"] + "\n\nWhat does f(" + doc["input"] +
            ") return? Reply with just the Python literal, including quotes around strings.\nOutput: ")


def grade(mode, doc, answers):
    try:
        result = subprocess.run(sandbox(mode), input=json.dumps([mode, doc, answers[0]]),
                                text=True, capture_output=True, timeout=180 if mode == "humaneval" else 30)
    except subprocess.TimeoutExpired:
        return {"pass@1": 0}
    if result.stderr:
        raise RuntimeError("Python grader failed: " + result.stderr.strip()[-500:])
    return {"pass@1": int(result.returncode == 42)}


def score_input(doc, answers):
    return grade("input", doc, answers)


def score_output(doc, answers):
    return grade("output", doc, answers)


def score_human(doc, answers):
    return grade("humaneval", doc, answers)
