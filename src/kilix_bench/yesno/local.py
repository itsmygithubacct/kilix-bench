"""Inference-only adapters for the three local Kilix models and Ollama decisions."""
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import sys
from typing import NamedTuple

from kilix_bench.yesno.needle2 import open_needle2


KILIX_MODELS = ("needle2", "needle3", "lfm2")


class NativeReply(NamedTuple):
    text: str
    evidence: dict


def decision_reply(result):
    answers = result.get("answers") if isinstance(result, dict) else None
    answer = answers.get("answer") if isinstance(answers, dict) else None
    probability = answer.get("noul") if isinstance(answer, dict) and answer.get("type") == "noul" else None
    evidence = answers if isinstance(answers, dict) else {"unexpected_response": result}
    if type(probability) in (int, float) and 0 <= probability <= 1:
        text = "YES" if probability >= 0.5 else "NO"
    else:
        text = json.dumps(evidence, ensure_ascii=False)
    return NativeReply(text, {"decision": evidence})


NEEDLE_CONTRACT = "yesno-needle-tools/v2"
NEEDLE_CHOICES = {"answer_yes": "YES", "answer_no": "NO"}


def needle_tools():
    return [{"name": name, "description": f"Respond {vote} to a yes/no question when the answer is {vote}.",
             "parameters": {"type": "object", "properties": {}, "required": []}}
            for name, vote in NEEDLE_CHOICES.items()]


def needle_reply(result):
    calls = result.get("function_calls")
    raw = result.get("raw_text")
    failed = (result.get("error") or result.get("type") == "error" or
              result.get("success") is False or result.get("reason") == "runtime_failure")
    if not failed and isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict):
        call = calls[0]
        if call.get("arguments") == {} and isinstance(call.get("name"), str) and call["name"] in NEEDLE_CHOICES:
            return NativeReply(NEEDLE_CHOICES[call["name"]], {"needle_reply": result})
    if not isinstance(raw, str) or ((failed or calls not in (None, [])) and raw.strip() in ("YES", "NO")):
        raw = json.dumps(result, ensure_ascii=False)
    return NativeReply(raw, {"needle_reply": result})


@contextmanager
def kilix_answerer(model, instruction, artifact, scratch):
    home = Path(os.environ.get("GPU_TERMINAL_HOME") or Path.home()/".local/gpu_terminal")
    data = Path(os.environ.get("KILIX_DATA_HOME") or home/"kilix/data")
    root = Path(os.environ.get("KILIX_CONTENT_ROOT") or data/"desktop-apps")
    app = Path(os.environ.get("PUZZLLM_KILIX_NEEDLE") or data/"desktop-apps/kilix-needle")
    if not (app/"asset.py").is_file():
        raise RuntimeError("Install the Kilix Needle runtime or set PUZZLLM_KILIX_NEEDLE to its directory.")
    sys.path.insert(0, str(app))
    try:
        import asset
        if model == "lfm2":
            import commands_runtime
            chosen = commands_runtime.selection()
            with commands_runtime.Runtime(root=str(root)) as runtime:
                artifact(chosen["files"][chosen["model_member"]]["sha256"])
                def ask_lfm(question, **_):
                    body = {"messages": [{"role": "system", "content": instruction},
                                         {"role": "user", "content": question}],
                            "temperature": 0, "max_tokens": 256, "stream": False}
                    message = runtime.post("/v1/chat/completions", body)["choices"][0]["message"]
                    if message.get("tool_calls"):
                        return NativeReply(json.dumps(message, ensure_ascii=False), {"model_message": message})
                    return message.get("content") or ""
                yield ask_lfm
            return
        if model not in ("needle2", "needle3"):
            raise ValueError("Choose needle2, needle3, or lfm2.")
        tools = needle_tools()
        with ExitStack() as stack:
            if model == "needle2":
                import tuning
                complete = stack.enter_context(open_needle2(asset, tuning, str(root), tools, scratch, artifact))
            else:
                import commands_needle3
                import compact_model
                import libengine
                compact_model.require_cached_platform()
                release = commands_needle3.selection()
                images = []
                for member in ("libneedle3-cached.so", "weights.cact"):
                    image = stack.enter_context(asset.installed_member(commands_needle3.ASSET_ID, member, str(root)))
                    if image.sha256 != release["files"][member]["sha256"]:
                        raise ValueError("The installed Needle3 file differs from the selected Kilix release.")
                    images.append(image)
                library, weights = images
                artifact(weights.sha256)
                engine = stack.enter_context(libengine.LibEngine(
                    library, tools, weights, generation=3, backend="rust3", preparation_timeout=90))

                def complete(text):
                    engine.reset()
                    return engine.complete(text)

            def ask_needle(question, **_):
                if model == "needle2":
                    request = "Answer this question with only YES or NO: " + question
                else:
                    request = f"Answer this yes/no question: {question}\n{instruction}"
                return needle_reply(complete(request))
            yield ask_needle
    finally:
        sys.path.remove(str(app))
