"""Inference-only adapters for the three local Kilix models and Ollama decisions."""
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import sys
from typing import NamedTuple


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


def needle_reply(result):
    calls = result.get("function_calls")
    text = result.get("raw_text")
    if isinstance(calls, list) and len(calls) == 1 and isinstance(calls[0], dict):
        call = calls[0]
        args = call.get("arguments")
        if (not result.get("error") and result.get("type") != "error" and result.get("success") is not False
                and call.get("name") == "answer" and isinstance(args, dict)
                and set(args) == {"value"} and args["value"] in ("YES", "NO")):
            return NativeReply(args["value"], {"needle_reply": result})
    if not isinstance(text, str) or (calls and text.strip() in ("YES", "NO")):
        text = json.dumps(result, ensure_ascii=False)
    return NativeReply(text, {"needle_reply": result})


@contextmanager
def kilix_answerer(model, instruction, artifact):
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
        import libengine
        tools = [{"name": "answer", "description": instruction,
                  "parameters": {"type": "object", "properties": {
                      "value": {"type": "string", "enum": ["YES", "NO"]}}, "required": ["value"]}}]
        with ExitStack() as stack:
            if model == "needle3":
                import commands_needle3
                import compact_model
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
                engine = libengine.LibEngine(library, tools, weights, generation=3, backend="rust3", preparation_timeout=90)
            else:
                import tuning
                selected = tuning.selected()
                if selected:
                    library = stack.enter_context(asset.installed_library(str(root)))
                    path = Path(selected["weights"])
                    weights = stack.enter_context(asset.load_verified(path, selected["sha256"], path.stat().st_size))
                    artifact(weights.sha256)
                    engine = libengine.LibEngine(library, tools, weights)
                else:
                    import engine as needle_engine
                    image = stack.enter_context(asset.from_installed(str(root)))
                    artifact(image.sha256)
                    engine = needle_engine.Engine(image, tools)
            runtime = stack.enter_context(engine)
            def ask_needle(question, **_):
                runtime.reset()
                return needle_reply(runtime.complete("Answer the question: " + " ".join(question.splitlines())))
            yield ask_needle
    finally:
        sys.path.remove(str(app))
