"""Presentation-stage probe on top of kitty-framebuffer.

The fixture proves the harness refuses to call an in-process reply a
display measurement. A live probe starts a private virtual display and a
separate Kitty process. It never writes to the caller's terminal.
"""

import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time


_PATH = re.compile(r"/[^ \n\"']+")


def framebuffer_source():
    env = os.environ.get("KITTY_FRAMEBUFFER_HOME")
    candidates = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path.home() / "gpu_terminal/kilix-modules/kitty-framebuffer")
    for candidate in candidates:
        if (candidate / "src/kitty_framebuffer.c").is_file():
            return candidate
    return None


def kitty_binary():
    env = os.environ.get("KILIX_BENCH_KITTY")
    candidates = [Path(env).expanduser()] if env else []
    candidates.append(Path.home() / ".local/gpu_terminal/kilix/prebuilt/kitty.app/bin/kitty")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def public_text(text):
    cleaned = _PATH.sub("<path>", text or "")
    return cleaned[-500:]


def build_probe(directory, source=None):
    """Compile the probe and kitty-framebuffer into ``directory``."""
    source = source or framebuffer_source()
    if source is None:
        raise ValueError("kitty-framebuffer sources are not available")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cc = os.environ.get("CC", "cc")
    include = source / "include"
    library_source = source / "src/kitty_framebuffer.c"
    flags = ["-std=c11", "-Wall", "-Wextra", "-Wpedantic", "-Werror", "-pthread",
             "-D_POSIX_C_SOURCE=200809L", "-D_DEFAULT_SOURCE", f"-I{include}"]
    library_flags = ["-std=c11", "-Wall", "-Wextra", "-Wpedantic", "-Wconversion",
                     "-Wshadow", "-Wstrict-prototypes", "-Wmissing-prototypes",
                     "-Wformat=2", "-fPIC", "-pthread", "-O2",
                     "-D_POSIX_C_SOURCE=200809L", "-D_DEFAULT_SOURCE", f"-I{include}"]
    object_file = directory / "kitty_framebuffer.o"
    binary = directory / "presentation-probe"
    probe = Path(__file__).with_name("presentation_probe.c")
    subprocess.run([cc, *library_flags, "-c", str(library_source), "-o", str(object_file)],
                   check=True, capture_output=True, text=True)
    subprocess.run([cc, *flags, str(probe), str(object_file), "-lz", "-lutil", "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    return binary


def _load(stdout):
    line = stdout.strip().splitlines()
    if not line:
        raise ValueError("presentation probe produced no record")
    return json.loads(line[-1])


def run_fixture(binary):
    completed = subprocess.run([str(binary), "fixture"], check=False, capture_output=True, text=True, timeout=10)
    if completed.returncode != 0 and not completed.stdout.strip():
        raise ValueError("fixture probe failed: " + public_text(completed.stderr))
    record = _load(completed.stdout)
    record["display"] = "none"
    record["gl_forcing"] = "not-applicable"
    # The fixture clock is an in-process reply. Drop the numbers so a
    # later reader cannot quote them as acknowledgement or display time.
    for key in ("session_start_ns", "frame_enqueue_ns", "frame_written_ns", "shm_consumed_ns"):
        record[key] = None
    return record


def _free_display():
    for number in range(40, 60):
        if not Path(f"/tmp/.X{number}-lock").exists():
            return f":{number}"
    raise ValueError("no free virtual display")


def _stop(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=2)


def run_live(binary, kitty, timeout=25):
    """Ask a private Kitty, on its own virtual display, to answer the probe."""
    if kitty is None or shutil.which("Xvfb") is None:
        return {
            "schema": 1, "peer": "unavailable", "graphics_query": "not-run",
            "shm_consumed": "not-run", "shm_consumed_ns": None,
            "display_latency": "unavailable", "scanout": "unavailable",
            "gpu_latency": "unavailable", "display": "none",
            "reason": "A private Kitty and Xvfb are required. The caller's terminal was not used.",
        }
    display = _free_display()
    root = Path(tempfile.mkdtemp(prefix="font-present-"))
    result = root / "probe.json"
    runtime = root / "runtime"
    runtime.mkdir(mode=0o700)
    log = root / "stderr.txt"
    xvfb = kitty_proc = None
    try:
        with log.open("w", encoding="utf-8") as stream:
            xvfb = subprocess.Popen(
                ["Xvfb", display, "-screen", "0", "800x600x24", "-nolisten", "tcp"],
                stdout=subprocess.DEVNULL, stderr=stream, start_new_session=True)
            time.sleep(0.4)
            if xvfb.poll() is not None:
                raise ValueError("virtual display exited")
            env = {
                "PATH": os.environ.get("PATH", "/usr/bin"),
                "HOME": str(root / "home"),
                "DISPLAY": display,
                "XDG_RUNTIME_DIR": str(runtime),
                "LIBGL_ALWAYS_SOFTWARE": "1",
                "KITTY_DISABLE_WAYLAND": "1",
            }
            (root / "home").mkdir(mode=0o700)
            # --session none ignores the program to run. A clean HOME and
            # --config NONE are what keep another session from starting.
            kitty_proc = subprocess.Popen(
                [str(kitty), "--single-instance=no",
                 "--instance-group", "kilix-bench-font-coverage",
                 "--config", "NONE",
                 "-o", "allow_remote_control=no",
                 "-o", "linux_display_server=x11",
                 "-o", "initial_window_width=640",
                 "-o", "initial_window_height=400",
                 "--class", "KilixBenchProbe",
                 str(binary), "live", str(result)],
                cwd=root, env=env, stdout=subprocess.DEVNULL, stderr=stream,
                start_new_session=True)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if result.is_file() and result.stat().st_size > 0:
                    break
                if kitty_proc.poll() is not None and not result.is_file():
                    break
                time.sleep(0.05)
        if not result.is_file():
            detail = public_text(log.read_text(encoding="utf-8", errors="replace"))
            return {
                "schema": 1, "peer": "unavailable", "graphics_query": "error",
                "shm_consumed": "not-run", "shm_consumed_ns": None,
                "display_latency": "unavailable", "scanout": "unavailable",
                "gpu_latency": "unavailable", "display": "virtual",
                "gl_forcing": "software",
                "reason": "private Kitty produced no probe record " + detail,
            }
        record = json.loads(result.read_text(encoding="utf-8"))
        record["display"] = "virtual"
        record["gl_forcing"] = "software"
        record.setdefault("display_latency", "unavailable")
        record.setdefault("scanout", "unavailable")
        record.setdefault("gpu_latency", "unavailable")
        return record
    finally:
        _stop(kitty_proc)
        _stop(xvfb)
        shutil.rmtree(root, ignore_errors=True)


def run(mode, directory):
    if mode == "off":
        return None
    binary = build_probe(directory)
    if mode == "fixture":
        return run_fixture(binary)
    if mode == "live":
        return run_live(binary, kitty_binary())
    raise ValueError("unknown presentation mode")
