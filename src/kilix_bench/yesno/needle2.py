"""One independent, bounded Needle 2 turn using the selected Kilix asset."""
from contextlib import ExitStack, contextmanager
import ctypes
import json
import os
from pathlib import Path
import resource
import subprocess
import sys


# The upstream executable's default. The pane-routing library allows much
# longer generations for command arguments; the bench only needs one call.
MAX_TOKENS = 256
TIMEOUT = 120


def tuned_turn(library_fd, weights_fd, tools_path):
    """Child process. No native state or generated output carries into another item."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    lib = ctypes.CDLL(f"/proc/self/fd/{library_fd}")
    lib.needle_load.argtypes = [ctypes.c_char_p, ctypes.c_ulonglong]
    lib.needle_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.needle_complete.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                    ctypes.POINTER(ctypes.c_char), ctypes.c_int]
    weights = os.pread(weights_fd, os.fstat(weights_fd).st_size, 0)
    if lib.needle_load(weights, len(weights)) != 0:
        raise RuntimeError("Needle 2 rejected the selected weights")
    if lib.needle_init(None, tools_path.read_bytes(), None) <= 0:
        raise RuntimeError("Needle 2 could not initialize the answer tools")
    buf = ctypes.create_string_buffer(65536)
    if lib.needle_complete(sys.stdin.buffer.read(), MAX_TOKENS, buf, len(buf)) < 0:
        raise RuntimeError("Needle 2 could not complete the question")
    sys.stdout.buffer.write(buf.value)


@contextmanager
def open_needle2(asset, tuning, root, tools, scratch, artifact):
    """Hold verified image descriptors throughout the run, including retries."""
    selection = tuning.selected()
    tools_path = Path(scratch).resolve() / "needle-answers.json"
    tools_path.write_text(json.dumps(tools), encoding="utf-8")
    with ExitStack() as images:
        if selection:
            library = images.enter_context(asset.installed_library(root))
            path = Path(selection["weights"])
            weights = images.enter_context(asset.load_verified(path, selection["sha256"], path.stat().st_size))
            fds = (library.fd, weights.fd)
            command = [sys.executable, "-IB", str(Path(__file__).resolve()),
                       str(library.fd), str(weights.fd), str(tools_path)]
            artifact(weights.sha256)
        else:
            binary = images.enter_context(asset.from_installed(root))
            fds = (binary.fd,)
            command = [f"/proc/self/fd/{binary.fd}", "--tools", str(tools_path),
                       "--max", str(MAX_TOKENS), "--prompt"]
            artifact(binary.sha256)

        def complete(question):
            # The stock HTTP endpoint loses quotes and backslashes. A direct
            # argv (stock) or stdin (tuned) preserves the complete OneBit input.
            if "\0" in question:
                raise ValueError("Needle 2 cannot represent a NUL in the input question")
            argv = command if selection else [*command, question]
            data = question.encode("utf-8") if selection else b""
            result = subprocess.run(argv, input=data, capture_output=True, pass_fds=fds,
                                    cwd=scratch, timeout=TIMEOUT, env={
                                        "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
                                        "NEEDLE_TELEMETRY": "0", "DO_NOT_TRACK": "1",
                                        "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""})
            if result.returncode or not result.stdout.strip():
                message = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
                detail = message[-1][:200] if message else "no native response"
                raise RuntimeError(f"Needle 2 exited with status {result.returncode}: {detail}")
            try:
                reply = json.loads(result.stdout)
            except (UnicodeError, ValueError):
                # The tuned library can emit malformed JSON in its reasoning.
                # Keep the actual output as invalid without losing later turns.
                return {"type": "error", "error": "malformed native reply",
                        "raw_text": result.stdout.decode("utf-8", errors="backslashreplace")}
            if not isinstance(reply, dict):
                return {"type": "error", "error": "unexpected native reply", "response": reply}
            return reply

        yield complete


if __name__ == "__main__":
    tuned_turn(int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]))
