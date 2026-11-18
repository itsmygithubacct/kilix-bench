"""The OpenAI-compatible chat backend: llama-server, Ollama, kilix-llm serve.

``--backend chat --endpoint URL``. On llama-server the constrained arm is a
GBNF grammar that allows only "0" or "1"; thinking is switched off through
the chat template (llama-server) or the ``think`` field (Ollama).
"""

import json
import os
import time
import urllib.error
import urllib.request

from kilix_bench.onebit.backends.base import Backend, BackendError, usage

NAME = "chat"
GRAMMAR = 'root ::= "0" | "1"'
CHAT_PATH = "/v1/chat/completions"


def _base(endpoint):
    """The server root of an endpoint given either way."""
    if endpoint.endswith(CHAT_PATH):
        return endpoint[: -len(CHAT_PATH)]
    return endpoint.rstrip("/")


def _get_json(url, timeout, headers=None):
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def _post_json(url, body, timeout, headers=None):
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json",
                                 **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        raise BackendError(f"HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise BackendError(str(error), kind="network") from error


def first_token_logprobs(choice):
    """(logprob of "0", logprob of "1") from an OpenAI-style choice."""
    logprobs = (choice or {}).get("logprobs") or {}
    content = logprobs.get("content") or []
    if not content:
        return None, None
    first = content[0]
    found = {}
    candidates = list(first.get("top_logprobs") or [])
    candidates.append({"token": first.get("token"),
                       "logprob": first.get("logprob")})
    for entry in candidates:
        token = (entry.get("token") or "").strip()
        if token in ("0", "1") and entry.get("logprob") is not None:
            found.setdefault(token, entry["logprob"])
    return found.get("0"), found.get("1")


class ChatBackend(Backend):
    kind = "chat"

    def __init__(self, endpoint, model, flavor="auto", max_tokens=4,
                 thinking="off", timeout=120.0, api_key=None, logprobs=True):
        self.base = _base(endpoint)
        self.url = self.base + CHAT_PATH
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.flavor = self.detect() if flavor == "auto" else flavor
        self.logprobs = logprobs
        self.thinking_control = None   # set after the first request
        self._fallbacks = set()

    def detect(self):
        props = _get_json(self.base + "/props", 10, self.headers)
        if isinstance(props, dict) and (
                "default_generation_settings" in props
                or "model_path" in props):
            return "llama-server"
        version = _get_json(self.base + "/api/version", 10, self.headers)
        if isinstance(version, dict) and "version" in version:
            return "ollama"
        return "openai"

    def describe(self):
        """What the server says about the model, without local paths."""
        info = {"flavor": self.flavor, "model": self.model}
        if self.flavor == "llama-server":
            props = _get_json(self.base + "/props", 10, self.headers) or {}
            path = props.get("model_path") or ""
            if path:
                info["model_file"] = path.replace("\\", "/").rsplit("/", 1)[-1]
            if props.get("build_info"):
                info["engine"] = str(props["build_info"])[:120]
        elif self.flavor == "ollama":
            tags = _get_json(self.base + "/api/tags", 10, self.headers) or {}
            for entry in tags.get("models") or []:
                if entry.get("name") == self.model or entry.get(
                        "model") == self.model:
                    info["digest"] = entry.get("digest")
                    details = entry.get("details") or {}
                    info["quantization"] = details.get("quantization_level")
                    info["parameter_size"] = details.get("parameter_size")
            version = _get_json(self.base + "/api/version", 10, self.headers)
            if version:
                info["engine"] = f"ollama {version.get('version')}"
        return info

    def supports(self, arm):
        if arm == "raw":
            return True, "no grammar; the reply must be exactly 0 or 1"
        if arm == "constrained":
            if self.flavor == "llama-server":
                return True, "GBNF grammar " + GRAMMAR
            return False, (f"{self.flavor} has no grammar or token-level logit "
                           "bias through this API")
        return False, f"unknown arm {arm}"

    def _thinking_fields(self):
        if self.thinking == "default":
            return {}
        enable = self.thinking == "on"
        if self.flavor == "llama-server":
            return {"chat_template_kwargs": {"enable_thinking": enable}}
        if self.flavor == "ollama":
            return {"think": enable}
        return {}

    def body(self, messages, arm):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.flavor in ("llama-server", "ollama"):
            body["seed"] = 0
        if self.logprobs and "logprobs" not in self._fallbacks:
            body["logprobs"] = True
            body["top_logprobs"] = 5
        if "thinking" not in self._fallbacks:
            body.update(self._thinking_fields())
        if arm == "constrained":
            body["grammar"] = GRAMMAR
        return body

    def ask(self, messages, arm):
        started = time.perf_counter()
        try:
            response = _post_json(
                self.url, self.body(messages, arm), self.timeout, self.headers)
        except BackendError as error:
            # A server that rejects an optional field is asked again without
            # it, once, and the fallback is recorded for the whole run.
            if "HTTP 400" not in str(error):
                raise
            for field in ("thinking", "logprobs"):
                if field not in self._fallbacks:
                    self._fallbacks.add(field)
                    return self.ask(messages, arm)
            raise
        latency = (time.perf_counter() - started) * 1000
        if self.thinking_control is None:
            self.thinking_control = (
                "not sent" if self.thinking == "default" or
                not self._thinking_fields() else
                "rejected by server" if "thinking" in self._fallbacks else
                "sent")
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        reply = message.get("content")
        thinking_seen = bool(
            message.get("reasoning_content") or message.get("reasoning")
            or (reply and "<think>" in reply))
        logprob_0, logprob_1 = first_token_logprobs(choice)
        return {
            "usage": chat_usage(response.get("usage")),
            "reply": reply,
            "latency_ms": round(latency, 2),
            "logprob_0": logprob_0,
            "logprob_1": logprob_1,
            "thinking_seen": thinking_seen,
            "finish_reason": choice.get("finish_reason"),
        }

    def count_tokens(self, messages):
        """Prompt tokens under the server's own tokenizer and chat template.

        llama-server only; None elsewhere or when the server refuses.
        """
        if self.flavor != "llama-server":
            return None
        try:
            prompt = _post_json(self.base + "/apply-template",
                                {"messages": messages}, self.timeout,
                                self.headers).get("prompt")
            tokens = _post_json(self.base + "/tokenize", {"content": prompt},
                                self.timeout, self.headers).get("tokens")
        except BackendError:
            return None
        return len(tokens) if isinstance(tokens, list) else None

    def settings(self):
        return {
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "seed": 0 if self.flavor in ("llama-server", "ollama") else None,
            "thinking_requested": self.thinking,
            "thinking_control": self.thinking_control,
            "logprobs_requested": self.logprobs and
            "logprobs" not in self._fallbacks,
        }


def chat_usage(raw):
    """OpenAI-style usage in the shared shape, or None."""
    if not isinstance(raw, dict) or raw.get("prompt_tokens") is None:
        return None
    prompt_details = raw.get("prompt_tokens_details") or {}
    completion_details = raw.get("completion_tokens_details") or {}
    return usage(
        input_tokens=raw.get("prompt_tokens"),
        cached_input_tokens=prompt_details.get("cached_tokens"),
        output_tokens=raw.get("completion_tokens"),
        reasoning_output_tokens=completion_details.get("reasoning_tokens"))


def add_arguments(parser):
    group = parser.add_argument_group("chat backend (OpenAI-compatible)")
    group.add_argument(
        "--endpoint",
        help="OpenAI-compatible server: its root URL or its "
             "/v1/chat/completions URL (llama-server, Ollama, kilix-llm serve)")
    group.add_argument("--model", default="",
                       help="Model name sent to the server")
    group.add_argument(
        "--flavor", choices=("auto", "llama-server", "ollama", "openai"),
        default="auto", help="Server type; auto asks the server")
    group.add_argument("--max-tokens", type=int, default=4)
    group.add_argument("--thinking", choices=("off", "on", "default"),
                       default="off",
                       help="Ask the server to switch thinking off (default)")
    group.add_argument("--no-logprobs", action="store_true")
    group.add_argument("--api-key-env", default="",
                       help="Environment variable holding an API key")


def check(args, parser):
    if not args.endpoint:
        parser.error("the chat backend needs --endpoint")


def spec(args):
    return {"endpoint": args.endpoint, "model": args.model,
            "flavor": args.flavor, "max_tokens": args.max_tokens,
            "thinking": args.thinking, "no_logprobs": args.no_logprobs,
            "api_key_env": args.api_key_env}


def build(spec, env=None):
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


def forbidden(spec):
    return []


def label(args):
    return args.model
