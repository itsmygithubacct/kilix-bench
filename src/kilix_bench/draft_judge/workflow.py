"""Sample draft replies, score them, and write one run directory."""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
from urllib.request import Request, urlopen

# Saved at import so a test can replace subprocess.Popen without catching
# nvidia-smi, which goes through this reference.
_POPEN = subprocess.Popen

# The third place llama-server is looked up. The help text names the
# environment variable and the flag, not this path.
KNOWN_LLAMA_SERVER = Path(
    "/home/pleb/research/d3-qwen-hybrid-20261123/llama-v1.0.0-vulkan/bin/llama-server")

SYSTEM = "You are a careful Python programmer."
ASK = ("Complete the following Python function. Reply with the complete function "
       "(signature and body) in one ```python block and nothing else.\n\n")
QUESTION = "Is this reply fully correct?\n\n"
MAX_TOKENS = 768
SEED = 1000
SAMPLE_TEMPERATURE = 0.8
CONTEXT = 8192
SKIP_DEVICE = ("intel", "llvmpipe", "microsoft")


def find_llama_server(explicit=None):
    """DRAFT_JUDGE_LLAMA_SERVER, then the flag, then the known build, then PATH."""
    env = os.environ.get("DRAFT_JUDGE_LLAMA_SERVER", "").strip()
    if env:
        return env
    if explicit:
        return explicit
    if KNOWN_LLAMA_SERVER.is_file():
        return str(KNOWN_LLAMA_SERVER)
    found = shutil.which("llama-server")
    if found:
        return found
    raise SystemExit(
        "draft-judge: llama-server was not found; set DRAFT_JUDGE_LLAMA_SERVER or pass --llama-server")


def parse_spec(text, default_device="gpu"):
    path, device = text, default_device
    if text.endswith(":cpu") or text.endswith(":gpu"):
        path, _, device = text.rpartition(":")
    if device not in ("cpu", "gpu"):
        raise SystemExit(f"draft-judge: device for {text} must be cpu or gpu")
    if not path:
        raise SystemExit("draft-judge: a model file is required")
    return ModelSpec(os.path.abspath(os.path.expanduser(path)), device)


class ModelSpec:
    def __init__(self, path, device):
        self.path = path
        self.device = device

    @property
    def name(self):
        return Path(self.path).name

    @property
    def key(self):
        return (self.path, self.device)

    def to_json(self):
        return {"path": self.path, "file": self.name, "device": self.device}


def under_ollama_store(path):
    parts = [part.casefold() for part in Path(path).parts]
    if len(parts) >= 3 and parts[1:3] == ["srv", "ollama"]:
        return True
    return "ollama" in parts and "blobs" in parts


def discover_models(extra_dirs):
    """GGUF files in the Kilix model directory and any extra directories.

    Paths inside an Ollama blob store are left out. Nothing is written there.
    """
    roots = [Path.home() / ".local/gpu_terminal/kilix-llm/models"]
    roots.extend(Path(item).expanduser() for item in extra_dirs)
    found, seen = [], set()
    for root in roots:
        if under_ollama_store(root) or not root.is_dir():
            continue
        for path in sorted(root.glob("*.gguf")):
            if not path.is_file() or under_ollama_store(path):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)
    return found


def parse_device_lines(text):
    """Pairs of (id, description) from llama-server --list-devices."""
    devices = []
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, _, desc = line.strip().partition(":")
        name = name.strip()
        if not name or any(ch.isspace() for ch in name):
            continue
        devices.append((name, desc.strip()))
    return devices


def pick_device(text):
    """First device whose id and description skip Intel, llvmpipe, and Microsoft.

    The id before the colon is what --device accepts. The words to skip show
    up in the description ("Vulkan0: Intel(R) ..."), so the whole line is checked.
    """
    for name, desc in parse_device_lines(text):
        blob = f"{name} {desc}".casefold()
        if any(word in blob for word in SKIP_DEVICE):
            continue
        return name
    raise RuntimeError("no GPU device is available")


def messages_for(prompt, code):
    if code:
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": ASK + prompt},
        ]
    return [{"role": "user", "content": prompt}]


def is_d3(path):
    return Path(path).name.casefold().startswith("d3")


def error_text(exc):
    if isinstance(exc, urllib.error.HTTPError):
        detail = _http_detail(exc)
        if detail:
            return f"HTTP {exc.code}: {detail}"
        return f"HTTP {exc.code}"
    if isinstance(exc, urllib.error.URLError):
        return str(exc.reason)
    return str(exc)


def _http_detail(exc):
    """A short message from an HTTP error body, when the server sent one."""
    try:
        raw = exc.read()
    except Exception:
        return ""
    if isinstance(raw, bytes):
        raw = raw.decode(errors="replace")
    raw = raw.strip()
    if not raw:
        return ""
    message = raw
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        err = parsed.get("error", parsed)
        if isinstance(err, dict):
            message = err.get("message") or err.get("msg") or raw
        elif isinstance(err, str) and err:
            message = err
    message = " ".join(str(message).split())
    if len(message) > 300:
        message = message[:297] + "..."
    return message


def post_json(url, body, timeout):
    payload = json.dumps(body).encode()
    request = Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def yes_probability(payload):
    """P(Yes) / (P(Yes) + P(No)) from the first token's top logprobs."""
    entries = payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    yes = no = 0.0
    for item in entries:
        word = str(item.get("token", "")).strip().casefold()
        probability = math.exp(float(item["logprob"]))
        if word == "yes":
            yes += probability
        elif word == "no":
            no += probability
    if yes + no <= 0:
        return 0.5
    return yes / (yes + no)


def noul_score(payload, key):
    return float(payload["answers"][key]["noul"])


_progress = threading.local()


def bind_listener(fn):
    """Send notes, warnings, and run events to fn instead of the terminal."""
    _progress.fn = fn


def emit(kind, **fields):
    fn = getattr(_progress, "fn", None)
    if fn is not None:
        fn({"kind": kind, **fields})


def warn(message):
    if getattr(_progress, "fn", None) is not None:
        emit("warn", text=message)
        return
    sys.stderr.write(f"draft-judge: {message}\n")
    sys.stderr.flush()


def note(message):
    """Progress for an interactive run. Captured output stays a report only."""
    if getattr(_progress, "fn", None) is not None:
        emit("note", text=message)
        return
    if sys.stderr.isatty():
        warn(message)


def model_files_ready(specs):
    missing = [spec.path for spec in specs if not Path(spec.path).is_file()]
    for path in missing:
        warn(f"model file not found: {path}")
    return not missing


def log_excerpt(path):
    """The last server error lines, without the llama.cpp timestamp."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    errors = [line for line in lines if " E " in line or "error" in line.casefold()]
    picked = errors[-2:] or lines[-2:]
    cleaned = []
    for line in picked:
        if " E " in line:
            line = line.split(" E ", 1)[1]
        cleaned.append(line)
    excerpt = " | ".join(cleaned)
    if len(excerpt) > 300:
        excerpt = excerpt[:297] + "..."
    return excerpt


def free_port(avoid):
    """An unused loopback port, skipping any port this run already chose."""
    import socket
    port = 20000
    while port < 65000:
        if port in avoid:
            port += 1
            continue
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                port += 1
                continue
        return port
    raise RuntimeError("no free port")


def default_output():
    root = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = root / "draft-judge" / "runs" / stamp
    if path.exists():
        path = root / "draft-judge" / "runs" / f"{stamp}-{time.time_ns()}"
    return path


def gpu_memory_mib():
    proc = None
    try:
        proc = _POPEN(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        output, _ = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        if proc is not None:
            proc.kill()
            proc.communicate()
        return None
    except (OSError, ValueError):
        return None
    if proc.returncode != 0:
        return None
    numbers = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            numbers.append(int(line))
        except ValueError:
            return None
    if not numbers:
        return None
    return sum(numbers)


def rss_bytes(pids):
    page = os.sysconf("SC_PAGE_SIZE")
    total = 0
    for pid in pids:
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            fields = raw.rsplit(")", 1)[1].split()
            total += int(fields[21]) * page
        except (OSError, IndexError, ValueError):
            continue
    return total


class Sampler:
    """Write a resource row on each phase change and every few seconds."""

    def __init__(self, path, pids, started):
        self.path = path
        self.pids = pids
        self.started = started
        self.phase = "load"
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, name="draft-judge-resources", daemon=True)

    def start(self):
        self.sample()
        self.thread.start()

    def set_phase(self, phase):
        self.phase = phase
        self.sample()

    def _loop(self):
        while not self.stop.wait(3):
            self.sample()

    def sample(self):
        try:
            row = {
                "seconds": round(time.perf_counter() - self.started, 3),
                "phase": self.phase,
                "gpu_memory_mib": gpu_memory_mib(),
                "rss": rss_bytes(self.pids()),
            }
            line = json.dumps(row) + "\n"
            with self.lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line)
        except Exception:
            return

    def close(self):
        self.stop.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=5)


class Server:
    def __init__(self, spec, command, port):
        self.spec = spec
        self.command = command
        self.port = port
        self.proc = None
        self.log = None
        self.error = None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def launch(self, log_path):
        self.log_path = log_path
        self.log = open(log_path, "w", encoding="utf-8")
        try:
            self.proc = subprocess.Popen(
                self.command,
                stdin=subprocess.DEVNULL,
                stdout=self.log,
                stderr=subprocess.STDOUT)
            self._wait_ready()
        except Exception as exc:
            self.error = error_text(exc)
            excerpt = log_excerpt(log_path)
            if excerpt and excerpt not in self.error:
                self.error = f"{self.error}: {excerpt}"
            self.stop()

    def _wait_ready(self):
        deadline = time.monotonic() + 180
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited ({self.proc.returncode})")
            try:
                with urlopen(self.url + "/health", timeout=2) as response:
                    body = response.read()
                    if isinstance(body, str):
                        body = body.encode()
                    if b"ok" in body.lower():
                        return
            except Exception:
                pass
            if time.monotonic() >= deadline:
                raise RuntimeError("llama-server did not become ready")
            time.sleep(0.2)

    def stop(self):
        proc = self.proc
        self.proc = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if self.log is not None:
            self.log.close()
            self.log = None


class Fleet:
    def __init__(self, schedule, binary, gpu_device, directory):
        self.schedule = schedule
        self.binary = binary
        self.gpu_device = gpu_device
        self.directory = directory
        self.live = {}
        self.lock = threading.Lock()
        self.ports = set()
        self._picked = None
        self._logs = 0

    def pids(self):
        with self.lock:
            found = []
            for server in self.live.values():
                proc = server.proc
                if proc is not None and proc.poll() is None:
                    found.append(proc.pid)
            return found

    def _device_flags(self, spec):
        if spec.device == "cpu":
            return ["-ngl", "0", "--device", "none"]
        return ["-ngl", "99", "--device", self._gpu_name()]

    def _gpu_name(self):
        if self.gpu_device:
            return self.gpu_device
        if self._picked is None:
            proc = subprocess.Popen(
                [self.binary, "--list-devices"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                output, _ = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                output, _ = proc.communicate()
            self._picked = pick_device(output or "")
        return self._picked

    def _command(self, spec, port):
        # d3 scores a whole reply in one physical batch. The server default
        # is 512, which is smaller than a full draft, so the batch matches
        # the context.
        batch = ["-b", str(CONTEXT), "-ub", str(CONTEXT)] if is_d3(spec.path) else []
        return [
            self.binary, "-m", spec.path,
            "--jinja", "--reasoning", "off",
            "-fa", "on",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "-c", str(CONTEXT), *batch, "-np", "1",
            "--host", "127.0.0.1", "--port", str(port),
            *self._device_flags(spec),
        ]

    def _launch(self, spec):
        port = free_port(self.ports)
        self.ports.add(port)
        self._logs += 1
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in spec.name)
        log_path = self.directory / f"server-{self._logs}-{safe}.log"
        try:
            command = self._command(spec, port)
        except Exception as exc:
            server = Server(spec, [], port)
            server.error = error_text(exc)
            return server
        server = Server(spec, command, port)
        server.launch(log_path)
        return server

    def acquire(self, spec):
        with self.lock:
            if self.schedule == "rotate":
                stale = [key for key in self.live if key != spec.key]
                stopping = [self.live.pop(key) for key in stale]
            else:
                stopping = []
            existing = self.live.get(spec.key)
        for server in stopping:
            server.stop()
        if existing is not None:
            return existing
        server = self._launch(spec)
        with self.lock:
            current = self.live.get(spec.key)
            if current is not None:
                server.stop()
                return current
            self.live[spec.key] = server
        return server

    def start_all(self, specs):
        seen = set()
        for spec in specs:
            if spec.key in seen:
                continue
            seen.add(spec.key)
            self.acquire(spec)

    def stop_all(self):
        with self.lock:
            servers = list(self.live.values())
            self.live.clear()
        for server in servers:
            server.stop()


def generation_body(prompt, code, temperature, seed):
    return {
        "messages": messages_for(prompt, code),
        "temperature": temperature,
        "seed": seed,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def judge_message(prompt, draft):
    return "Task:\n" + prompt + "\n\n" + QUESTION + draft + "\n\nAnswer with only Yes or No."


def one_generation(server, prompt, code, temperature, seed):
    started = time.perf_counter()
    row = {"text": "", "tokens": None, "seconds": 0.0, "ok": False, "error": ""}
    if server.error:
        row["error"] = server.error
        return row
    try:
        payload = post_json(
            server.url + "/v1/chat/completions",
            generation_body(prompt, code, temperature, seed),
            timeout=900)
        text = payload["choices"][0]["message"]["content"] or ""
        tokens = int(payload["usage"]["completion_tokens"])
    except Exception as exc:
        row["seconds"] = round(time.perf_counter() - started, 3)
        row["error"] = error_text(exc)
        return row
    row.update(text=text, tokens=tokens, seconds=round(time.perf_counter() - started, 3), ok=True)
    return row


def one_judge(server, prompt, draft_text, index):
    started = time.perf_counter()
    row = {"index": index, "score": None, "seconds": 0.0, "ok": False, "error": ""}
    if server.error:
        row["error"] = server.error
        row["seconds"] = 0.0
        return row
    try:
        if is_d3(server.spec.path):
            key = f"d{index}"
            payload = post_json(server.url + "/v1/systemone", {
                "model": server.spec.name,
                "state": {"task": prompt},
                "questions": {key: {"type": "noul", "instructions": QUESTION + draft_text}},
            }, timeout=900)
            score = noul_score(payload, key)
        else:
            payload = post_json(server.url + "/v1/chat/completions", {
                "messages": [{"role": "user", "content": judge_message(prompt, draft_text)}],
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }, timeout=900)
            score = yes_probability(payload)
    except Exception as exc:
        row["seconds"] = round(time.perf_counter() - started, 3)
        row["error"] = error_text(exc)
        return row
    row.update(score=score, seconds=round(time.perf_counter() - started, 3), ok=True)
    return row


def choose_winner(drafts, judges):
    by_index = {row["index"]: row for row in judges}
    winner = None
    for draft in drafts:
        judged = by_index.get(draft["index"])
        if not draft["ok"] or not judged or not judged["ok"]:
            continue
        if winner is None or judged["score"] > winner[0]["score"]:
            winner = (judged, draft)
    return winner


def score_line(draft, judged):
    if draft["ok"] and judged and judged["ok"]:
        detail = f"score {judged['score']:.3f}"
    elif not draft["ok"]:
        detail = "skipped"
        if draft.get("error"):
            detail = f"skipped: {draft['error']}"
    else:
        detail = "judge error"
        if judged and judged.get("error"):
            detail = f"judge error: {judged['error']}"
    return f"draft {draft['index']}  {draft['model']}  temp {draft['temp']:g}  {detail}"


def render_report(vanilla, drafts, judges, winner):
    if winner is None:
        head = "result: no reply could be judged"
    else:
        head = (
            f"result: draft {winner[1]['index']}  {winner[1]['model']}  "
            f"score {winner[0]['score']:.3f}"
        )
    by_index = {row["index"]: row for row in judges}
    lines = [head, ""]
    for draft in drafts:
        lines.append(score_line(draft, by_index.get(draft["index"])))
    lines.append("")
    lines.append("vanilla")
    lines.append(vanilla.get("text") or "")
    lines.append("")
    lines.append("chosen")
    if winner is None:
        lines.append("(no reply could be judged)")
    else:
        lines.append(winner[1]["text"] or "")
    lines.append("")
    return "\n".join(lines)


def render_load_failure(spec, error):
    return f"result: judge failed to load\njudge: {spec.name}\n{error}\n"


def _append(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def execute(config):
    """Run the workflow. Return (exit code, run directory, report text)."""
    specs = [*config["draft_models"], config["judge"]]
    if not model_files_ready(specs):
        return 2, None, ""
    directory = Path(config["output"]) if config.get("output") else default_output()
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    drafts_path = directory / "drafts.jsonl"
    judge_path = directory / "judge.jsonl"
    for path in (drafts_path, judge_path, directory / "resources.jsonl"):
        path.write_text("", encoding="utf-8")
    binary = find_llama_server(config.get("llama_server"))
    fleet = Fleet(config["schedule"], binary, config.get("gpu_device") or None, directory)
    started = time.perf_counter()
    sampler = Sampler(directory / "resources.jsonl", fleet.pids, started)
    timing = {"vanilla": 0.0, "drafts": 0.0, "judge": 0.0}
    vanilla = {"text": "", "tokens": None, "seconds": 0.0, "ok": False, "error": ""}
    drafts, judges = [], []
    previous = os.umask(0o077)
    try:
        bind_listener(config.get("on_event"))
        emit("run", directory=str(directory))
        _write_json(directory / "config.json", {
            "draft_models": [spec.to_json() for spec in config["draft_models"]],
            "judge": config["judge"].to_json(),
            "draft_count": config["draft_count"],
            "schedule": config["schedule"],
            "prompt": config["prompt"],
            "code": bool(config["code"]),
            "gpu_device": config.get("gpu_device") or None,
        })
        sampler.start()
        if config["schedule"] == "together":
            note("loading models")
            sampler.set_phase("load")
            fleet.start_all([*config["draft_models"], config["judge"]])
            with fleet.lock:
                judge_server = fleet.live.get(config["judge"].key)
            if judge_server is not None and judge_server.error:
                warn(f"judge failed to load: {judge_server.error}")
                emit("load_failure", text=judge_server.error)
                report = render_load_failure(config["judge"], judge_server.error)
                (directory / "report.txt").write_text(report, encoding="utf-8")
                _write_json(directory / "timing.json", timing)
                return 1, directory, report

        sampler.set_phase("vanilla")
        note("writing the vanilla reply")
        mark = time.perf_counter()
        server = fleet.acquire(config["draft_models"][0])
        vanilla = one_generation(server, config["prompt"], config["code"], 0.0, SEED)
        timing["vanilla"] = round(time.perf_counter() - mark, 3)
        (directory / "vanilla.txt").write_text(vanilla.get("text") or "", encoding="utf-8")
        _write_json(directory / "vanilla.json", vanilla)
        emit("vanilla", text=vanilla.get("text") or "", ok=bool(vanilla.get("ok")),
             error=vanilla.get("error") or "", tokens=vanilla.get("tokens"),
             seconds=vanilla.get("seconds") or 0)
        if vanilla["ok"]:
            note(f"vanilla  {vanilla['tokens']} tokens  {vanilla['seconds']}s")
        else:
            warn(f"vanilla failed: {vanilla['error']}")

        sampler.set_phase("drafts")
        mark = time.perf_counter()
        models = config["draft_models"]
        count = config["draft_count"]
        for index in range(count):
            spec = models[index % len(models)]
            temperature = 0.0 if index == 0 else SAMPLE_TEMPERATURE
            note(f"writing draft {index + 1} of {count}  {spec.name}")
            server = fleet.acquire(spec)
            generated = one_generation(
                server, config["prompt"], config["code"], temperature, SEED + index)
            row = {
                "index": index,
                "model": spec.name,
                "temp": temperature,
                "text": generated["text"],
                "tokens": generated["tokens"],
                "seconds": generated["seconds"],
                "ok": generated["ok"],
                "error": generated["error"],
            }
            drafts.append(row)
            _append(drafts_path, row)
            emit("draft", index=index, model=spec.name, text=generated["text"] or "",
                 ok=bool(generated["ok"]), error=generated["error"] or "")
            if generated["ok"]:
                note(
                    f"draft {index + 1} of {count}  {generated['tokens']} tokens  "
                    f"{generated['seconds']}s")
            else:
                warn(f"draft {index + 1} of {count} failed: {generated['error']}")
        timing["drafts"] = round(time.perf_counter() - mark, 3)

        sampler.set_phase("judge")
        note("judging")
        mark = time.perf_counter()
        judge_server = fleet.acquire(config["judge"])
        if judge_server.error:
            warn(f"judge failed to load: {judge_server.error}")
        for draft in drafts:
            if not draft["ok"]:
                continue
            judged = one_judge(judge_server, config["prompt"], draft["text"], draft["index"])
            judges.append(judged)
            _append(judge_path, judged)
            emit("judge", index=judged["index"], score=judged["score"],
                 ok=bool(judged["ok"]), error=judged["error"] or "")
            if judged["ok"]:
                note(f"draft {draft['index']}  score {judged['score']:.3f}  {judged['seconds']}s")
            else:
                warn(f"draft {draft['index']}  judge error: {judged['error']}")
        timing["judge"] = round(time.perf_counter() - mark, 3)

        winner = choose_winner(drafts, judges)
        if winner is None:
            note("result: no reply could be judged")
            emit("result", ok=False, index=None, score=None)
        else:
            note(
                f"result: draft {winner[1]['index']}  "
                f"score {winner[0]['score']:.3f}")
            emit("result", ok=True, index=winner[1]["index"], score=winner[0]["score"])
        report = render_report(vanilla, drafts, judges, winner)
        (directory / "report.txt").write_text(report, encoding="utf-8")
        _write_json(directory / "timing.json", timing)
        return (0 if winner else 1), directory, report
    finally:
        bind_listener(None)
        sampler.close()
        fleet.stop_all()
        os.umask(previous)
