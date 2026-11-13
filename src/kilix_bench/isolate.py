"""Run a desktop measurement away from the live Kilix data directories."""

import os
import tempfile


_STRIP = ("GPU_TERMINAL_", "KILIX", "KITTY_", "BASH_FUNC_")


def state_library():
    """The host's already-built libkilix-state, never a path inside the sandbox."""
    configured = os.environ.get("KILIX_STATE_LIBRARY", "")
    if configured and os.path.isfile(configured):
        return configured
    storage = os.environ.get("KILIX_STORAGE_HOME") or os.path.expanduser(
        "~/.local/gpu_terminal/kilix")
    candidate = os.path.join(
        storage, "build", "libraries", "kilix-state", "libkilix-state.so")
    if os.path.isfile(candidate):
        return candidate
    return ""


def sandbox(kilix_home, require_state=True):
    """Environment for one offscreen desk. The caller deletes the directory."""
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(_STRIP)
    }
    library = state_library() if require_state else ""
    if require_state and not library:
        raise SystemExit("the Kilix state library is not built")
    root = tempfile.mkdtemp(prefix="kilix-bench-")
    home = os.path.join(root, "home")
    data_root = os.path.join(root, "gpu-terminal-data")
    kilix_root = os.path.join(data_root, "kilix")
    kilix95_root = os.path.join(data_root, "kilix-95")
    os.makedirs(home, mode=0o700)
    env.update({
        "HOME": home,
        "GPU_TERMINAL_SOURCE_HOME": os.path.dirname(kilix_home),
        "GPU_TERMINAL_HOME": data_root,
        "GPU_TERMINAL_SETTINGS_FILE": os.path.join(data_root, "settings.conf"),
        "KILIX_HOME": kilix_home,
        "KILIX_STORAGE_HOME": kilix_root,
        "KILIX_CONFIG_HOME": os.path.join(kilix_root, "config"),
        "KILIX_STATE_DIRECTORY": os.path.join(kilix_root, "state"),
        "KILIX_CACHE_HOME": os.path.join(kilix_root, "cache"),
        "KILIX_DATA_HOME": os.path.join(kilix_root, "data"),
        "KILIX_SESSION_HOME": os.path.join(kilix_root, "session"),
        "KILIX_BUILD_DIRECTORY": os.path.join(kilix_root, "build"),
        "KILIX95_STORAGE_HOME": kilix95_root,
        "KILIX95_CONFIG_HOME": os.path.join(kilix95_root, "config"),
        "KILIX95_STATE_HOME": os.path.join(kilix95_root, "state"),
        "KILIX95_CACHE_HOME": os.path.join(kilix95_root, "cache"),
        "KILIX95_DATA_HOME": os.path.join(kilix95_root, "data"),
        "KILIX95_SESSION_HOME": os.path.join(kilix95_root, "session"),
        "KILIX_DESKTOP_DIR": os.path.join(root, "desktop"),
        "KILIX_RECYCLE_DIR": os.path.join(root, "recycle"),
        "XDG_CONFIG_HOME": os.path.join(root, "xdg", "config"),
        "XDG_STATE_HOME": os.path.join(root, "xdg", "state"),
        "XDG_CACHE_HOME": os.path.join(root, "xdg", "cache"),
        "XDG_DATA_HOME": os.path.join(root, "xdg", "data"),
        "XDG_DATA_DIRS": os.path.join(root, "xdg", "system"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "KILIX_STATE_LIBRARY": library,
    })
    os.makedirs(env["KILIX_DESKTOP_DIR"], mode=0o700)
    return env, root


def kilix_home_from_env():
    env = os.environ.get("KILIX_HOME")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    source = os.environ.get("GPU_TERMINAL_SOURCE_HOME") or os.path.expanduser(
        "~/.local/gpu_terminal/sources")
    candidate = os.path.join(os.path.abspath(os.path.expanduser(source)), "kilix")
    if os.path.isdir(os.path.join(candidate, "config")):
        return candidate
    return ""
