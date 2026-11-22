"""Run the Terminal-Bench 2.0 engineering tasks with Harbor and Terminus 2."""

import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
from urllib.request import Request, urlopen

from kilix_bench.local_models import positive

REVISION = "2fd12b88aafdd04a52c298e3940bcb189f9766d6"
REPOSITORY = "https://github.com/laude-institute/terminal-bench-2.git"
HARBOR = "harbor==0.27.0"
TASKS = (
    "log-summary-date-ranges", "fix-git", "cancel-async-tasks", "query-optimize",
    "adaptive-rejection-sampler", "bn-fit-modify", "build-cython-ext", "build-pmars",
    "build-pov-ray", "caffe-cifar-10", "chess-best-move", "circuit-fibsqrt",
    "cobol-modernization", "code-from-image", "compile-compcert", "configure-git-webserver",
    "constraints-scheduling", "count-dataset-tokens", "custom-memory-heap-crash",
    "distribution-search", "extract-moves-from-video", "filter-js-from-html",
    "financial-document-processor", "fix-ocaml-gc", "gcode-to-text", "git-multibranch",
    "gpt2-codegolf", "headless-terminal", "hf-model-inference", "install-windows-3.11",
    "kv-store-grpc", "large-scale-text-editing", "largest-eigenval",
    "llm-inference-batching-scheduler", "mailman", "make-doom-for-mips", "make-mips-interpreter",
    "mcmc-sampling-stan", "merge-diff-arc-agi-task", "modernize-scientific-stack",
    "mteb-leaderboard", "mteb-retrieve", "multi-source-data-merger", "nginx-request-logging",
    "openssl-selfsigned-cert", "overfull-hbox", "path-tracing", "path-tracing-reverse",
    "polyglot-c-py", "polyglot-rust-c", "portfolio-optimization", "prove-plus-comm",
    "pypi-server", "pytorch-model-cli", "pytorch-model-recovery", "qemu-alpine-ssh",
    "qemu-startup", "raman-fitting", "regex-chess", "regex-log", "reshard-c4-data",
    "rstan-to-pystan", "sam-cell-seg", "schemelike-metacircular-eval", "sparql-university",
    "sqlite-db-truncate", "sqlite-with-gcov", "torch-pipeline-parallelism",
    "torch-tensor-parallelism", "train-fasttext", "tune-mjcf", "video-processing",
    "winning-avg-corewars", "write-compressor",
)


def api_base(endpoint):
    url = endpoint.rstrip("/")
    return url if url.endswith("/v1") else url + "/v1"


def endpoint_model(endpoint):
    headers = {}
    if key := os.environ.get("KILIX_BENCH_API_KEY"):
        headers["Authorization"] = "Bearer " + key
    with urlopen(Request(api_base(endpoint) + "/models", headers=headers), timeout=15) as response:
        names = [m["id"] for m in json.load(response)["data"] if m.get("id")]
    if len(names) != 1:
        raise ValueError("Specify the model ID when the endpoint advertises multiple models or none.")
    return names[0]


def collect(job, expected):
    result = json.loads((job / "result.json").read_text())
    if not result.get("finished_at"):
        raise ValueError("Harbor job is incomplete")
    scores = {}
    for path in job.glob("*/result.json"):
        trial = json.loads(path.read_text())
        name = trial["task_name"]
        rewards = (trial.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward")
        if (name in scores or not trial.get("reward_valid") or not isinstance(reward, (int, float))
                or not math.isfinite(reward) or not 0 <= reward <= 1):
            raise ValueError(f"No unique, valid verifier reward for {name}. See {path}")
        scores[name] = reward
    if set(scores) != set(expected):
        raise ValueError(f"Harbor returned {len(scores)}/{len(expected)} expected task results. See {job}")
    return scores


def run_job(model, endpoint, output, selected, status=print, context_length=None, max_tokens=None):
    if not selected or not set(selected) <= set(TASKS) or len(set(selected)) != len(selected):
        raise ValueError("Choose distinct Terminal-Bench engineering tasks.")
    if binary := shutil.which("harbor"):
        launcher = [binary]
    elif uv := shutil.which("uv"):
        launcher = [uv, "tool", "run", "--python", "3.12", "--from", HARBOR, "harbor"]
    else:
        raise ValueError("Install Harbor or uv and Docker with the Compose plugin.")
    env = os.environ.copy()
    options = {}
    if endpoint:
        model = model or endpoint_model(endpoint)
        model = "openai/" + model.removeprefix("openai/")
        options["api_base"] = api_base(endpoint)
        env["OPENAI_API_KEY"] = env.get("KILIX_BENCH_API_KEY") or "local"
    if not model:
        raise ValueError("Specify a Harbor model or an OpenAI-compatible endpoint.")
    if context_length:
        tokens = max_tokens or min(2048, context_length // 4)
        if not 0 < tokens < context_length:
            raise ValueError("The output token budget must be smaller than the context size.")
        options["model_info"] = {"max_input_tokens": context_length - tokens, "max_output_tokens": tokens}
        options["proactive_summarization_threshold"] = min(2048, (context_length - tokens) // 4)
    elif max_tokens:
        raise ValueError("Specify a context size with the output token budget.")

    parent = Path(output).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="terminal-bench-", dir=parent))
    config = {"jobs_dir": str(run), "job_name": "trials", "n_concurrent_trials": 1,
              "environment": {"type": "docker"},
              "agents": [{"name": "terminus-2", "model_name": model, "kwargs": options}],
              "tasks": [{"path": name, "git_url": REPOSITORY, "git_commit_id": REVISION}
                        for name in selected]}
    config_path = run / "job.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    status(f"Terminal-Bench 2.0 engineering | {len(selected)} tasks | {model}")
    status(f"Harbor log: {run / 'worker.log'}")
    with (run / "worker.log").open("w") as log:
        proc = subprocess.Popen([*launcher, "run", "--config", str(config_path)], env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait()
        finally:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    if rc:
        raise RuntimeError(f"Harbor exited {rc}. See {log.name}")
    scores = collect(run / "trials", selected)
    mean = sum(scores.values()) / len(scores)
    (run / "score.json").write_text(json.dumps({"suite": "terminal-bench", "profile": "engineering",
        "revision": REVISION, "model": model, "mean_reward": mean, "rewards": scores}, indent=2) + "\n")
    status(f"Terminal-Bench engineering score {mean:.1%} ({len(scores)} tasks)")
    return run


def cli(parser, argv):
    parser.add_argument("--model", help="Harbor provider/model, or the served model ID with --endpoint")
    parser.add_argument("--endpoint", help="OpenAI-compatible local server base URL")
    parser.add_argument("--output", default="jobs", help="parent directory for unique Harbor jobs")
    parser.add_argument("--task", choices=TASKS, action="append", help="select a task (repeatable)")
    parser.add_argument("--limit", type=positive, help="run the first N selected tasks")
    parser.add_argument("--context-length", type=positive, help="context size for a custom local model")
    parser.add_argument("--max-tokens", type=positive, help="output budget within that context")
    args = parser.parse_args(argv)
    selected = list(dict.fromkeys(args.task or TASKS))[:args.limit]
    try:
        run_job(args.model, args.endpoint, args.output, selected,
                context_length=args.context_length, max_tokens=args.max_tokens)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"{parser.prog}: {exc}\n")
    return 0
