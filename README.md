# kilix-bench

Offscreen frame-path measurements for Kilix: how long a Kilix 95 desk takes to
redraw, and how many bytes the graphics SDK would send for a change.

Not shipped: a development tool run from a checkout. No Plebian-OS release,
Kilix or catalog tree references it.

## What it does

Offscreen measurements for Kilix rendering. The desktop suite builds a Kilix
95 desk with no terminal, times the PIL redraw and title-bar hit testing, and
asks the frame presenter how many bytes a change would send. The render suite
times the graphics SDK itself: rectangle diffs, shared-memory placement,
inline placement, a vertical compose, and a sideways window move.

A desktop compose time is the redraw inside the desk. A render time is the
graphics library. Neither includes the GPU.

- Runs each suite in a child process with a private temporary home, so the
  live Kilix and Kilix 95 data directories are never touched.
- Reports median, minimum, p95 and maximum per scenario, plus pixel bytes,
  presenter rectangles and wire bytes where they apply.
- Stores every run as JSON and text in a user-private directory, and refuses
  to write a record that contains the checkout or home paths.

## Run

```sh
PYTHONPATH=src python3 -m kilix_bench --desktop /path/to/kilix-95
PYTHONPATH=src python3 -m kilix_bench --suite render
```

`--suite desktop` is the default and needs `--desktop`. `--suite render` does
not. `--kilix-home` is the Kilix host checkout that contains `config/kilix_sdk`.
When it is omitted, `KILIX_HOME` or the host sources directory is used.

The live desktop is not started. Desktop state for the run is a temporary
directory and is removed when the run finishes.

The render suite needs a Kilix checkout whose `kitty-frame-presenter` can be
imported (an initialised submodule, or the package on `PYTHONPATH`). The
desktop suite also needs Kilix's other submodules. Both suites need a built
`libkilix-state.so`, found through `KILIX_STATE_LIBRARY` or Kilix's build
directory: the parent stops before starting either suite when it is missing.

## Usage

`python3 -m kilix_bench [options]` (also installed as `kilix-bench` by
`pip install .`)

| Option | Default | Meaning |
|---|---|---|
| `--suite desktop\|render` | `desktop` | What to measure |
| `--desktop PATH` | none | Kilix 95 checkout (desktop suite; must contain `main.py`) |
| `--kilix-home PATH` | `KILIX_HOME`, else `~/.local/gpu_terminal/sources/kilix` | Kilix checkout with `config/kilix_sdk` |
| `--width N` / `--height N` | `1920` / `1080` | Frame size |
| `--repeats N` | `12` | Timed samples per scenario |
| `--warmup N` | `2` | Untimed samples first |
| `KILIX_BENCH_HOME` | `~/.local/gpu_terminal/kilix-bench` | Where results are written |

## How it works

```
kilix_bench (parent) --sandbox env, temp HOME--> kilix_bench.desktop | .render
        |                                              |
        |                       imports Kilix 95 main.py / kilix_sdk.graphics
        |                       times scenarios; ConsumerTerm counts the bytes
        <------------------- JSON on stdout ----------+
  write_run(): refuse local paths -> results/<stamp>-<suite>.json, -latest.*
```

The parent strips `GPU_TERMINAL_*`, `KILIX*` and `KITTY_*` variables, points
every Kilix, Kilix 95 and XDG directory into a fresh temporary tree, runs the
suite module with a three-minute timeout and deletes the tree afterwards
(`isolate.py`, `__main__.py`). The desktop suite builds the desk with Kilix 95's
own modules and never starts a terminal; graphics writes go to a stand-in
terminal that counts bytes and releases shared-memory names as Kitty would
(`terminal.py`). The render suite drives `FramePresenter` directly. Every
scenario, and how to read the rectangle and byte counts, is in
[docs/scenarios.md](docs/scenarios.md).

## Results

Each run writes three files under `~/.local/gpu_terminal/kilix-bench/results/`.
The desktop suite uses the `kilix-95` name. The render suite uses `kilix-render`.

- `YYYYMMDDTHHMMSSZ-<suite>.json`, the run that just finished
- `<suite>-latest.json`, the same record
- `<suite>-latest.txt`, the same numbers as text

The directory is user-private. Set `KILIX_BENCH_HOME` to put that tree somewhere
else. The JSON record has the desktop version and revision, the Python version,
and the machine type. It does not record local paths.

## Tests

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Documentation

| Document | Contents |
|---|---|
| [docs/scenarios.md](docs/scenarios.md) | Every desktop and render scenario, and how damage, scroll and inline bytes are counted |

## Related repositories

- [kilix-95](https://github.com/itsmygithubacct/kilix-95#readme): the desktop
  the desktop suite builds.
- [kilix](https://github.com/itsmygithubacct/kilix#readme): the host whose
  `kilix_sdk` graphics both suites import.
- [kitty-frame-presenter](https://github.com/itsmygithubacct/kitty-frame-presenter#readme):
  the `FramePresenter` both suites time.
- [kilix-state](https://github.com/itsmygithubacct/kilix-state#readme): the
  state library the desktop suite needs.

## Part of Plebian-OS

kilix-bench is one of the repositories behind
[Plebian-OS](https://github.com/itsmygithubacct/plebian-os#readme), a Debian
install whose desktop is [Kilix](https://github.com/itsmygithubacct/kilix#readme)
running the [Kilix 95](https://github.com/itsmygithubacct/kilix-95#readme)
desktop through [Pleb](https://github.com/itsmygithubacct/pleb#readme).
It does not reach users: developers run it from a checkout to measure the
frame path. The map of every repository and how it ships is on the
[components page](https://github.com/itsmygithubacct/plebian-os/blob/main/docs/components.md).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
