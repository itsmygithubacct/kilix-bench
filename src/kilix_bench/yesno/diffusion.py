"""Answer each question with a text diffusion model through llama-diffusion-cli.

llama-server cannot serve diffusion language models, so every question is one
llama-diffusion-cli invocation. The prompt goes in a file, the instruction is
the system prompt, and the denoised reply is read from the log file that
llama.cpp writes. Its stderr also carries GPU driver messages.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import struct
import subprocess

from kilix_bench.yesno.models import stop_child

PROVIDER = "llama.cpp-diffusion"
# Room for a one-word reply. Unused positions denoise to end-of-text tokens.
ANSWER_TOKENS = 32
# The chat template wraps the system and user text in a few dozen tokens.
TEMPLATE_TOKENS = 64
QUESTION_TIMEOUT = 900
# Sampling settings per GGUF architecture, from the llama.cpp diffusion
# examples. Every architecture uses the timestep schedule, which has no
# block-length constraint on the sequence length.
ARCHITECTURES = {
    "llada": dict(algorithm=4),
    "llada-moe": dict(algorithm=4),
    "dream": dict(algorithm=1),
    "rnd1": dict(algorithm=1),
}


def models_folder():
    folder = os.environ.get("PUZZLLM_DIFFUSION_MODELS")
    if folder:
        return Path(folder).expanduser()
    data = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local/share"
    return Path(data) / "puzzllm/diffusion"


def local_models():
    folder = models_folder()
    paths = {str(p.absolute()) for p in folder.rglob("*") if p.suffix.lower() == ".gguf" and p.is_file()}
    return sorted(paths, key=lambda p: (Path(p).name.casefold(), p))


def tool():
    binary = os.environ.get("PUZZLLM_DIFFUSION_CLI") or shutil.which("llama-diffusion-cli")
    if not binary:
        found = sorted(models_folder().glob("llama-*/llama-diffusion-cli"))
        binary = str(found[-1]) if found else None
    if not binary or not os.access(binary, os.X_OK):
        raise RuntimeError("Install llama-diffusion-cli or set PUZZLLM_DIFFUSION_CLI.")
    return Path(binary).resolve()


def _read(source, size):
    data = source.read(size)
    if len(data) != size:
        raise ValueError("The GGUF header ends early.")
    return data


_SCALARS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _skip(source, kind):
    if kind in _SCALARS:
        _read(source, _SCALARS[kind])
    elif kind == 8:
        _read(source, struct.unpack("<Q", _read(source, 8))[0])
    elif kind == 9:
        inner, count = struct.unpack("<IQ", _read(source, 12))
        if inner in _SCALARS:
            _read(source, _SCALARS[inner] * count)
        else:
            for _ in range(count):
                _skip(source, inner)
    else:
        raise ValueError(f"Unknown GGUF value type {kind}.")


def architecture(path):
    """Return general.architecture from a GGUF file's metadata."""
    with open(path, "rb") as source:
        if _read(source, 4) != b"GGUF":
            raise ValueError("Not a GGUF file.")
        version, _tensors, count = struct.unpack("<IQQ", _read(source, 20))
        if version < 2:
            raise ValueError("GGUF version 1 is not supported.")
        for _ in range(count):
            key = _read(source, struct.unpack("<Q", _read(source, 8))[0]).decode("utf-8", "replace")
            kind = struct.unpack("<I", _read(source, 4))[0]
            if key == "general.architecture" and kind == 8:
                return _read(source, struct.unpack("<Q", _read(source, 8))[0]).decode("utf-8", "replace")
            _skip(source, kind)
    raise ValueError("The GGUF file names no architecture.")


def reply_text(log):
    """The denoised text follows the generation timing line on the log stream."""
    marker = log.rfind("\ntotal time: ")
    if marker < 0:
        return None
    end = log.find("\n", marker + 1)
    if end < 0:
        return None
    text = log[end + 1:]
    text = text[1:] if text.startswith("\n") else text
    return text[:-1] if text.endswith("\n") else text


def token_count(binary, model, text, scratch, env):
    """Count with the bundled tokenizer, or bound the count by the UTF-8 length."""
    counter = binary.with_name("llama-tokenize")
    if counter.is_file() and os.access(counter, os.X_OK):
        source = scratch / "count.txt"
        source.write_text(text, encoding="utf-8")
        result = subprocess.run([str(counter), "-m", str(model), "-f", str(source), "--show-count",
                                 "--log-disable"], capture_output=True, text=True, env=env, timeout=120)
        for line in reversed(result.stdout.splitlines()):
            if line.startswith("Total number of tokens:"):
                return int(line.rsplit(":", 1)[1])
    return len(text.encode("utf-8"))


@contextmanager
def diffusion_answerer(model, instruction, scratch, status):
    path = Path(model).expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".gguf":
        raise ValueError("Choose a readable GGUF model file.")
    kind = architecture(path)
    if kind not in ARCHITECTURES:
        raise ValueError(f"{path.name} is a {kind} model. llama-diffusion-cli runs "
                         + ", ".join(sorted(ARCHITECTURES)) + " models.")
    binary = tool()
    env = {**os.environ, "LD_LIBRARY_PATH": os.pathsep.join(filter(None, (
        str(binary.parent), os.environ.get("LD_LIBRARY_PATH"))))}
    device = os.environ.get("PUZZLLM_DIFFUSION_DEVICE") or os.environ.get("PUZZLLM_GPU")
    settings = ARCHITECTURES[kind]
    status(f"Using {path.name} ({kind}) with llama-diffusion-cli")

    def ask(question, **_):
        prompt = scratch / "prompt.txt"
        prompt.write_text(question, encoding="utf-8")
        length = token_count(binary, path, instruction + "\n" + question, scratch, env)
        length = length + TEMPLATE_TOKENS + ANSWER_TOKENS
        args = [str(binary), "-m", str(path), "-f", str(prompt), "-sys", instruction,
                "-c", str(length), "-b", str(length), "-ub", str(length),
                "--diffusion-steps", str(ANSWER_TOKENS), "--diffusion-eps", "0.001",
                "--diffusion-algorithm", str(settings["algorithm"]),
                "--temp", "0", "--seed", "0", "-ngl", "99"]
        if device:
            args += ["--device", device]
        # llama.cpp's own log file holds only its messages, without the
        # prefixes it adds off a terminal. GPU drivers also write to stderr.
        log = scratch / "diffusion.log"
        log.unlink(missing_ok=True)
        args += ["--log-file", str(log), "--no-log-prefix", "--no-log-timestamps", "--log-colors", "off"]
        with (scratch / "diffusion.stderr").open("w+", encoding="utf-8", errors="replace") as errors:
            process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors,
                                       env=env, start_new_session=True)
            try:
                code = process.wait(timeout=QUESTION_TIMEOUT)
            except subprocess.TimeoutExpired:
                stop_child(process)
                raise RuntimeError(f"llama-diffusion-cli gave no answer within {QUESTION_TIMEOUT} seconds.")
            errors.seek(0)
            stderr = errors.read()
        try:
            with log.open(encoding="utf-8", errors="replace", newline="") as source:
                output = source.read()
        except FileNotFoundError:
            output = ""
        text = reply_text(output)
        if code != 0 or text is None:
            lines = [line for line in (output or stderr).strip().splitlines() if line.strip()]
            raise RuntimeError(f"llama-diffusion-cli exited {code}. " + (lines[-1][:300] if lines else ""))
        return text

    yield ask
