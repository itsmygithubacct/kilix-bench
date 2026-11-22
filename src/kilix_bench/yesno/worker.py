#!/usr/bin/env python3
"""Run one independent YES/NO trial per question in the selected benchmark."""
import argparse
import fcntl
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import select
import signal
import socket
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
import uuid

from kilix_bench.yesno.comparison import write_comparison
from kilix_bench.yesno.local import KILIX_MODELS, NEEDLE_CONTRACT, NativeReply, decision_reply, kilix_answerer
from kilix_bench.yesno.machine import RunnerMonitor, backend_scope
from kilix_bench.yesno.models import effort_options, executable, gguf_artifact, gguf_snapshot, hosted_catalog, stop_child
from kilix_bench.yesno.records import compare_files, model_id, ranking_events, rankings_document, save_rankings
from kilix_bench.yesno.suites import BOOLQ, K12, SUITES, boolq_items

INSTRUCTION = "Respond with only YES or NO."
HOSTED = ("codex", "grok", "claude")
HOSTED_TIMEOUTS = (180, 360)
PROVIDERS = ("ollama", "llama.cpp", *HOSTED, "kilix")


def emit(event, **fields):
    print(json.dumps(dict(event=event, **fields), ensure_ascii=True), flush=True)


def ollama_url():
    host = os.environ.get("OLLAMA_HOST") or "127.0.0.1:11434"
    if "://" not in host:
        host = "http://" + host
    url = urlsplit(host)
    if url.port is None and not url.path.strip("/"):
        return host.rstrip("/") + ":11434"
    return host.rstrip("/")


def request(url, body=None, timeout=5):
    data = None if body is None else json.dumps(body).encode()
    return urlopen(Request(url, data=data, headers={"Content-Type": "application/json"}), timeout=timeout)


def local_models(provider):
    if provider == "kilix":
        return list(KILIX_MODELS)
    if provider == "ollama":
        with request(ollama_url() + "/api/tags") as response:
            return sorted({entry["name"] for entry in json.load(response)["models"]})
    home = Path(os.environ.get("GPU_TERMINAL_HOME") or Path.home() / ".local/gpu_terminal")
    folder = Path(os.environ.get("KILIX_LLM_MODELS_DIR") or home / "kilix-llm/models")
    configured = os.environ.get("PUZZLLM_MODEL")
    paths = {str(p.absolute()) for p in folder.rglob("*") if p.suffix.lower() == ".gguf" and p.is_file()}
    if configured and Path(configured).expanduser().is_file():
        paths.add(str(Path(configured).expanduser().absolute()))
    return sorted(paths, key=lambda p: (Path(p).name.casefold(), p))


def ensure_ollama(model, status):
    url = ollama_url()
    try:
        with request(url + "/api/show", {"model": model}) as response:
            return json.load(response)
    except HTTPError as error:
        if error.code != 404:
            raise
    status("Pulling " + model)
    with request(url + "/api/pull", {"model": model, "stream": True}, timeout=180) as response:
        for line in response:
            part = json.loads(line)
            if part.get("error"):
                raise RuntimeError(part["error"])
            if part.get("status") == "success":
                break
        else:
            raise RuntimeError("Model download ended before Ollama reported success.")
    with request(url + "/api/show", {"model": model}) as response:
        return json.load(response)


@contextmanager
def llama_server(model, scratch, status):
    path = Path(model).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".gguf":
        raise ValueError("Choose a readable GGUF model file.")
    binary = os.environ.get("PUZZLLM_LLAMA_SERVER") or shutil.which("llama-server")
    if not binary and os.environ.get("PUZZLLM_ROOT"):
        helper = Path(os.environ["PUZZLLM_ROOT"])/"libexec/puzzllm-sandbox"
        settings = subprocess.check_output([helper, "paths"], text=True)
        binary = dict(line.split("=", 1) for line in settings.splitlines() if "=" in line)["llama_server"]
    if not binary:
        home = Path(os.environ.get("GPU_TERMINAL_HOME") or Path.home()/".local/gpu_terminal")
        binary = next((str(p) for p in (home/"kilix-llm/bin/llama-server",
                                        home/"kilix-llm/llama.cpp/build/bin/llama-server") if os.access(p, os.X_OK)), None)
    if not binary:
        raise RuntimeError("Install llama-server or set PUZZLLM_LLAMA_SERVER.")
    alias = "yesno-" + uuid.uuid4().hex
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env = {**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(filter(None, (
        str(Path(binary).resolve().parent), os.environ.get("LD_LIBRARY_PATH"))))}
    args = [binary, "--model", str(path), "--alias", alias, "--host", "127.0.0.1",
            "--port", str(port), "--parallel", "1", "--ctx-size", "4096", "--jinja", "--no-webui"]
    status("Loading " + path.name)
    with (scratch / "server.log").open("w+") as log:
        server = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                  env=env, start_new_session=True)
        try:
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    log.seek(0)
                    lines = log.read().strip().splitlines()
                    raise RuntimeError("llama-server stopped. " + (lines[-1] if lines else "Check the model file."))
                try:
                    with request(url + "/v1/models", timeout=2) as response:
                        if any(row.get("id") == alias for row in json.load(response)["data"]):
                            break
                except (OSError, ValueError):
                    pass
                time.sleep(0.15)
            else:
                raise RuntimeError("Timed out loading the GGUF model.")
            yield url, alias
        finally:
            stop_child(server)


def hosted_answer(provider, model, question, scratch, identify=None, effort=None, *, timeout=HOSTED_TIMEOUTS[0]):
    binary = executable(provider)
    model_args = ["--model", model] if model else []
    if effort and effort != "default":
        if provider == "codex":
            model_args.extend(["-c", "model_reasoning_effort=" + json.dumps(effort)])
        else:
            model_args.extend(["--reasoning-effort" if provider == "grok" else "--effort", effort])
    last = scratch / "last-message"
    prompt = question
    if provider == "codex":
        last.unlink(missing_ok=True)
        args = ["exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
                "-c", "mcp_servers={}", "-c", "features.shell_tool=false",
                "-c", "web_search=\"disabled\"", *model_args, "-o", str(last), "-"]
        prompt = INSTRUCTION + "\n\n" + question
    elif provider == "grok":
        # An empty Grok allowlist restores its defaults. A named allowlist keeps
        # just that tool and the two MCP tools, all of which the denylist removes.
        args = ["--single", question, "--system-prompt-override", INSTRUCTION, "--tools", "read_file",
                "--disallowed-tools", "read_file,search_tool,use_tool", "--disable-web-search", "--no-subagents",
                "--output-format", "plain", *model_args]
        prompt = ""
    else:
        args = ["--print", "--system-prompt", INSTRUCTION, "--tools", "", "--strict-mcp-config",
                "--no-session-persistence", "--output-format", "json", *model_args]
    with subprocess.Popen([binary, *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, errors="replace",
                          cwd=scratch, start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(prompt, timeout=timeout)
        finally:
            stop_child(process)
    if process.returncode:
        detail = (stderr.strip() or stdout.strip()).splitlines()
        raise RuntimeError(f"{provider} exited {process.returncode}. " + (detail[-1][:300] if detail else ""))
    if provider == "claude":
        try:
            result = json.loads(stdout)
        except ValueError:
            result = None
        if isinstance(result, dict) and "result" in result:
            if result.get("is_error"):
                raise RuntimeError(str(result["result"])[:300])
            used = result.get("modelUsage") or {}
            if identify and isinstance(used, dict) and len(used) == 1:
                identify(next(iter(used)))
            return result["result"]
    return last.read_text() if provider == "codex" else stdout


def onebit():
    checkout = os.environ.get("PUZZLLM_YESNO_BENCH")
    if checkout:
        import kilix_bench
        package = str(Path(checkout) / "src/kilix_bench")
        if package not in kilix_bench.__path__:
            kilix_bench.__path__.insert(0, package)
    try:
        from kilix_bench.onebit import contract, items
        from kilix_bench.onebit.backends.chat import ChatBackend
    except ImportError as error:
        raise RuntimeError("Install kilix-bench with OneBit or set PUZZLLM_YESNO_BENCH to its checkout.") from error
    return contract, items, ChatBackend


@contextmanager
def answerer(provider, model, scratch, chat_backend, status, identify=None, effort=None, artifact=None):
    if provider == "kilix":
        with kilix_answerer(model, INSTRUCTION, artifact, scratch) as ask:
            yield ask
    elif provider == "ollama":
        info = ensure_ollama(model, status)
        if "decision" in (info.get("capabilities") or []):
            def decide(question, *, prompt=None, context=None):
                state = INSTRUCTION + ("\n\n" + context if context else "")
                body = {"model": model, "state": state,
                        "questions": {"answer": {"type": "noul", "instructions": prompt or question}}}
                try:
                    with request(ollama_url() + "/v1/systemone", body, timeout=180) as response:
                        result = json.load(response)
                except HTTPError as error:
                    detail = error.read().decode("utf-8", "replace")
                    raise RuntimeError(f"Ollama decision request returned HTTP {error.code}. {detail[:300]}") from error
                if isinstance(result, dict) and result.get("error"):
                    raise RuntimeError(str(result["error"]))
                return decision_reply(result)
            yield decide
            return
        url = ollama_url() + "/api/chat"
        thinking = {"think": False}

        def ask_ollama(question, **_):
            body = {"model": model, "messages": messages(question), "stream": False,
                    "options": {"temperature": 0}, **thinking}
            try:
                response = request(url, body, timeout=180)
            except HTTPError as error:
                detail = error.read().decode("utf-8", "replace")
                if thinking and error.code == 400 and "think" in detail.lower():
                    thinking.clear()
                    del body["think"]
                    response = request(url, body, timeout=180)
                else:
                    raise RuntimeError(f"Ollama returned HTTP {error.code}. {detail[:250]}") from error
            with response:
                data = json.load(response)
                if data.get("error"):
                    raise RuntimeError(data["error"])
                return data["message"].get("content") or ""
        yield ask_ollama
    elif provider == "llama.cpp":
        with llama_server(model, scratch, status) as (url, alias):
            backend = chat_backend(url, alias, flavor="llama-server", max_tokens=256, logprobs=False)
            yield lambda question, **_: backend.ask(messages(question), "raw").get("reply") or ""
    else:
        def resolved(name):
            nonlocal model
            if identify:
                identify(name)
            if not model:
                model = name
        def ask_hosted(question, **_):
            for attempt, timeout in enumerate(HOSTED_TIMEOUTS):
                if attempt:
                    status(f"{provider} timed out after {HOSTED_TIMEOUTS[attempt-1]:g} seconds. "
                           f"Retrying this question with a {timeout:g}-second timeout.")
                try:
                    return hosted_answer(provider, model, question, scratch, resolved, effort, timeout=timeout)
                except subprocess.TimeoutExpired:
                    continue
            limits = ", ".join(f"{limit:g}s" for limit in HOSTED_TIMEOUTS)
            raise RuntimeError(f"{provider} returned no answer after {len(HOSTED_TIMEOUTS)} timed-out attempts ({limits}).")

        yield ask_hosted


def messages(question):
    return [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": question}]


def display(record):
    # Keep PTY messages bounded. The saved record retains the full question and reply.
    fields = dict(record)
    raw = fields.pop("raw_reply", None)
    fields.pop("decision", None)
    fields.pop("needle_reply", None)
    fields.pop("model_message", None)
    if raw and raw.isspace():
        fields["reply"] = "(whitespace only)"
    for key in ("question", "reply", "text", "context", "rationale"):
        if key in fields:
            fields[key] = " ".join(fields[key].split())[:600]
    emit(**fields)


def perform(provider, model, output, path, requested_effort="", suite=K12):
    def record(event, **fields):
        output.write(json.dumps(dict(event=event, **fields), ensure_ascii=False) + "\n")
        output.flush()
        if event == "answer":
            os.fdatasync(output.fileno())

    def checkpoint():
        try:
            os.fsync(output.fileno())
            save_rankings(path.parent, suite)
        except (OSError, ValueError, TypeError, KeyError) as error:
            record("storage_warning", text=f"Could not save rankings.json ({error})")

    stop_file = path.with_suffix(".stop")
    finished = threading.Event()

    def check_stop():
        while not finished.wait(0.2):
            if stop_file.exists():
                os.kill(os.getpid(), signal.SIGTERM)
                return

    start = time.monotonic()
    scope = backend_scope(provider, os.environ.get("OLLAMA_HOST"))
    meter = None
    model_files = None
    outcome, detail = "end", {}
    effort = requested_effort if provider in HOSTED else None
    record("begin", provider=provider, model=model, effort=effort, total=0, backend_scope=scope,
           file=path.name, instruction=INSTRUCTION, **suite.metadata(),
           **({"answer_contract": NEEDLE_CONTRACT} if provider == "kilix" and model in ("needle2", "needle3") else {}))
    monitor = threading.Thread(target=check_stop, daemon=True)
    monitor.start()
    try:
        try:
            meter = RunnerMonitor()
        except (OSError, ValueError, RuntimeError):
            pass
        if meter:
            record("machine", runner=meter.profile)
        checkpoint()
        requested_id = model_id(provider, model)
        identity = None
        def identify(name):
            nonlocal identity
            name = model_id(provider, name)
            if name and name != identity:
                record("identity", model=name)
                identity = name
                if name != requested_id:
                    checkpoint()
        if not model and provider in HOSTED:
            _, model = hosted_catalog(provider, need_default=True)
            if provider == "claude" and model:
                record("selection", model=model)
        # A hosted alias can resolve to a concrete deployment in the reply.
        if provider != "claude":
            identify(model)
        def status(text):
            record("status", text=text)
        contract, itemset, chat_backend = onebit()
        questions = boolq_items(status) if suite == BOOLQ else itemset.load(split=suite.split)
        record("size", total=len(questions))
        if not questions:
            raise RuntimeError(f"{suite.dataset} contains no {suite.split} questions.")
        if provider == "llama.cpp":
            try:
                model_files = gguf_snapshot(model)
            except (OSError, RuntimeError, ValueError):
                pass
        with tempfile.TemporaryDirectory(dir=path.parent) as temporary, \
                answerer(provider, model, Path(temporary), chat_backend, status, identify, effort,
                         lambda digest: record("artifact", id="sha256:"+digest)) as ask:
            record("ready", setup_seconds=round(time.monotonic() - start, 6))
            for n, item in enumerate(questions, 1):
                info = dict(n=n, band=item["band"], question=item["prompt"])
                record("question", **info)
                started = time.monotonic()
                response = ask(contract.user_message(item), prompt=item["prompt"], context=item.get("context"))
                evidence = {}
                if isinstance(response, NativeReply):
                    raw_reply, evidence = response
                else:
                    raw_reply = response
                reply = raw_reply.strip()
                expected = ("YES", "NO")[item["answer"]]
                if reply not in ("YES", "NO"):
                    verdict = "invalid"
                    evidence["raw_reply"] = raw_reply
                else:
                    verdict = "correct" if reply == expected else "wrong"
                record("answer", **info, id=item["id"], context=item.get("context", ""),
                       rationale=item["rationale"], reply=reply, expected=expected,
                       verdict=verdict, seconds=round(time.monotonic() - started, 6), **evidence)
    except KeyboardInterrupt:
        outcome, detail = "stopped", {"text": "Stopped. Answered questions are saved."}
    except Exception as error:
        outcome, detail = "error", {"text": str(error)}
    finally:
        elapsed = round(time.monotonic() - start, 6)
        ended_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        finished.set()
        try:
            usage = meter.finish() if meter else None
        except (OSError, ValueError, RuntimeError):
            usage = None
        record(outcome, **detail, wall_seconds=elapsed, ended_at=ended_at, runner_usage=usage)
        monitor.join()
        checkpoint()
        if model_files and outcome == "end":
            try:
                artifact = gguf_artifact(model_files, path.parent / ".gguf-digests", meter.profile["id"] if meter else None)
                if artifact:
                    record("artifact", id=artifact)
                    checkpoint()
            except (OSError, RuntimeError, ValueError):
                pass
        stop_file.unlink(missing_ok=True)


def follow(path, show=display):
    # An exclusive lock on the journal lives with its writer, even if every
    # display closes. Read complete lines and then wait for the next one.
    last = None
    input_open = True
    announced = False
    with path.open("rb") as source:
        pending = b""
        while True:
            try:
                fcntl.flock(source, fcntl.LOCK_SH | fcntl.LOCK_NB)
                live = False
            except BlockingIOError:
                live = True
            if live and not announced:
                emit("attached")
                announced = True
            chunk = source.read(65536)
            if chunk:
                lines = (pending + chunk).split(b"\n")
                pending = lines.pop()
                for line in lines:
                    row = json.loads(line)
                    if row["event"] != "artifact":
                        show(row)
                    if row["event"] not in ("storage_warning", "artifact"):
                        last = row["event"]
                continue
            if not live:
                break
            if input_open and select.select([0], [], [], 0)[0]:
                command = os.read(0, 256)
                input_open = bool(command)
                if b"stop" in command.splitlines():
                    path.with_suffix(".stop").touch()
            time.sleep(0.08)
    if last not in ("end", "error", "stopped"):
        emit("stopped", text="Interrupted. Showing saved answers.")
    return 1 if last == "error" else 0


def latest(folder, suite=K12):
    try:
        save_rankings(folder, suite)
    except (OSError, ValueError, TypeError, KeyError) as error:
        emit("storage_warning", text=f"Could not save rankings.json ({error})")
    path = max(folder.glob("*.jsonl"), default=None)
    return follow(path) if path else 0


def start_run(provider, model, folder, effort="", suite=K12):
    if provider not in PROVIDERS or (provider not in HOSTED and not model):
        raise ValueError("Choose a provider and model.")
    if provider == "kilix" and model not in KILIX_MODELS:
        raise ValueError("Choose one of " + ", ".join(KILIX_MODELS))
    if provider in HOSTED:
        try:
            levels, default = effort_options(provider, model, discover=bool(effort))
        except (OSError, RuntimeError):
            # A missing CLI is a failed run and belongs in the history.
            levels, default = effort_options(provider, model)
        if effort and effort not in levels:
            raise ValueError("Unsupported reasoning effort for the selected model.")
        effort = effort or default or "default"
    elif effort:
        raise ValueError("Only hosted models accept a reasoning effort.")
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Only publication of the journal is serialized. The child keeps the run
    # lock, and every display opens its own read handle after setup returns.
    with (folder / ".start.lock").open("a") as setup:
        fcntl.flock(setup, fcntl.LOCK_EX)
        with (folder / ".run.lock").open("a+") as gate:
            try:
                fcntl.flock(gate, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                gate.seek(0)
                name = gate.read()
                if Path(name).name != name or not name.endswith(".jsonl"):
                    raise RuntimeError("The running benchmark has no readable journal.")
                path = folder / name
            else:
                start = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                path = folder / f"{start}-{uuid.uuid4().hex[:8]}.jsonl"
                with path.with_suffix(".starting").open("x", encoding="utf-8") as output:
                    fcntl.flock(output, fcntl.LOCK_EX)
                    os.rename(output.name, path)
                    gate.seek(0)
                    gate.truncate()
                    gate.write(path.name)
                    gate.flush()
                    if os.fork() == 0:
                        try:
                            setup.close()
                            os.setsid()
                            with open(os.devnull, "r+b", buffering=0) as null:
                                for fd in (0, 1, 2):
                                    os.dup2(null.fileno(), fd)
                            sys.dont_write_bytecode = True
                            if "TMPDIR" not in os.environ:
                                tempfile.tempdir = str(folder.resolve())
                            signal.signal(signal.SIGTERM, interrupt)
                            signal.signal(signal.SIGHUP, interrupt)
                            perform(provider, model, output, path, effort, suite)
                        finally:
                            os._exit(0)
    return path


def stop_run(folder):
    if not (folder / ".start.lock").is_file():
        return False
    with (folder / ".start.lock").open("a") as setup:
        fcntl.flock(setup, fcntl.LOCK_EX)
        with (folder / ".run.lock").open("r") as gate:
            try:
                fcntl.flock(gate, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                name = gate.read()
                if Path(name).name != name or not name.endswith(".jsonl"):
                    raise ValueError("The running benchmark has no readable journal.")
                (folder/name).with_suffix(".stop").touch()
                return True
    return False


def interrupt(_signal, _frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=SUITES, default=K12.key)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("models").add_argument("provider", choices=PROVIDERS)
    efforts = sub.add_parser("efforts")
    efforts.add_argument("provider", choices=HOSTED)
    efforts.add_argument("model", nargs="?", default="")
    sub.add_parser("latest").add_argument("state", type=Path)
    sub.add_parser("compare").add_argument("files", type=Path, nargs="+")
    sub.add_parser("records").add_argument("state", type=Path)
    report = sub.add_parser("report")
    report.add_argument("state", type=Path)
    report.add_argument("file")
    start = sub.add_parser("run")
    start.add_argument("state", type=Path)
    start.add_argument("provider", choices=PROVIDERS)
    start.add_argument("model", nargs="?", default="")
    start.add_argument("effort", nargs="?", default="")
    args = parser.parse_args()
    os.umask(0o077)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    try:
        if args.action == "compare":
            write_comparison(compare_files(args.files), sys.stdout)
        elif args.action == "efforts":
            for level in effort_options(args.provider, args.model, discover=True)[0]:
                emit("model", model=level)
        elif args.action == "models":
            models = hosted_catalog(args.provider)[0] if args.provider in HOSTED else local_models(args.provider)
            if not models and args.provider in HOSTED:
                emit("status", text="The CLI reported no model IDs. Enter a model ID or use the configured default.")
            for model in models:
                emit("model", model=model)
        else:
            suite = SUITES[args.suite]
            folder = suite.folder(args.state)
            if args.action == "latest":
                return latest(folder, suite)
            elif args.action == "records":
                warning = None
                try:
                    document = save_rankings(folder, suite)
                except (OSError, ValueError, TypeError, KeyError) as error:
                    document = rankings_document(folder, suite=suite)
                    warning = f"Could not save rankings.json ({error})"
                for row in ranking_events(document):
                    emit(**row)
                if warning:
                    emit("error", text=warning)
            elif args.action == "report":
                path = folder / args.file
                if path.name != args.file or path.suffix != ".jsonl" or path.is_symlink():
                    raise ValueError("Invalid benchmark report name.")
                if not path.is_file():
                    raise FileNotFoundError("The score is saved in rankings.json. The question journal is unavailable.")
                return follow(path)
            else:
                return follow(start_run(args.provider, args.model, folder, args.effort, suite))
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        if args.action == "compare":
            print(f"Could not compare rankings ({error})", file=sys.stderr)
        else:
            emit("error", text=str(error))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
