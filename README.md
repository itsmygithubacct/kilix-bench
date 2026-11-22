# kilix-bench

Offscreen frame-path measurements for Kilix: how long a Kilix 95 desk takes to
redraw, how many bytes the graphics SDK would send for a change, and how a
full grid of text compares as Kitty cells, Kitty graphics, and Neuronte. Also
OneBit, a yes/no test of how well a model answers with a single `0` or `1`.

Not shipped: a development tool run from a checkout. No Plebian-OS release,
Kilix or catalog tree references it.

## What it does

Offscreen measurements for Kilix rendering. The desktop suite builds a Kilix
95 desk with no terminal, times the PIL redraw and title-bar hit testing, and
asks the frame presenter how many bytes a change would send. The render suite
times the graphics SDK itself: rectangle diffs, shared-memory placement,
inline placement, a vertical compose, and a sideways window move. The text
suite times one full grid of text three ways: the Kitty cell stream, the
bitmap font sent through the Kitty graphics protocol, and Neuronte drawing
into that same protocol. It also encodes the bitmap frame as sixel.

A desktop compose time is the redraw inside the desk. A render time is the
graphics library. A text time is the application side of one rewrite. None
of them includes Kitty's GPU font raster.

- Runs each suite in a child process with a private temporary home, so the
  live Kilix and Kilix 95 data directories are never touched.
- Reports median, minimum, p95 and maximum per scenario, plus pixel bytes,
  presenter rectangles and wire bytes where they apply.
- Stores every run as JSON and text in a user-private directory, and refuses
  to write a record that contains the checkout or home paths.

The OneBit suite is a different kind of test: it asks a model 1,768 yes/no
and pass/fail items it may answer only with `0` or `1`, from grade 1 to
college, computer use, reading shown information, and traps. See
[OneBit](#onebit) below.

[sidebyside/](sidebyside/README.md) is a separate, interactive test: three
shell panes in a Kilix tab, typed into together, drawn with Neuronte 2,
Neuronte 0.1.4 and DejaVu, with scripts that list the characters they share and a
three-way benchmark.

## Run

```sh
PYTHONPATH=src python3 -m kilix_bench --desktop /path/to/kilix-95
PYTHONPATH=src python3 -m kilix_bench --suite render
PYTHONPATH=src python3 -m kilix_bench --suite text
```

`--suite desktop` is the default and needs `--desktop`. `--suite render` and
`--suite text` do not. `--kilix-home` is the Kilix host checkout that contains
`config/kilix_sdk`. When it is omitted, `KILIX_HOME` or the host sources
directory is used.

The text suite also needs soft-raster's shared library, Neuronte's shared
library, and `neuronte-split-v4.nrt`. It looks for them under
`~/gpu_terminal/kilix-modules/soft-raster/build/` and
`~/projects/neuronte/` (or `$NEURONTE`). `SOFT_RASTER_LIBRARY`,
`NEURONTE_LIBRARY`, and `NEURONTE_WEIGHTS` override those paths. The result
records the weight file's name and a short hash, not the path.

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
| `--suite desktop\|render\|text\|onebit\|boolq\|ifeval\|gsm8k\|llama-bench\|cruxeval\|humaneval-plus\|terminal-bench` | `desktop` | What to measure (`onebit` takes its own options, below) |
| `--desktop PATH` | none | Kilix 95 checkout (desktop suite; must contain `main.py`) |
| `--kilix-home PATH` | `KILIX_HOME`, else `~/.local/gpu_terminal/sources/kilix` | Kilix checkout with `config/kilix_sdk` |
| `--width N` / `--height N` | `1920` / `1080` | Frame size |
| `--repeats N` | `12` | Timed samples per scenario |
| `--warmup N` | `2` | Untimed samples first |
| `KILIX_BENCH_HOME` | `~/.local/gpu_terminal/kilix-bench` | Where results are written |

## How it works

```
kilix_bench (parent) --sandbox env, temp HOME--> kilix_bench.desktop | .render | .text
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
(`terminal.py`). The render suite drives `FramePresenter` directly. The text suite sets
`NEURONTE_LIBRARY`, `NEURONTE_WEIGHTS`, and `SOFT_RASTER_LIBRARY` after the
sandbox, because the sandbox strips `KILIX*` and changes `HOME`. Every
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
and the machine type. It does not record local paths. The text suite uses the
`kilix-text` name.

## OneBit

How much intelligence fits in a one-bit answer. Every item is a yes/no
question (`0` = yes, `1` = no) or a judgement of shown information (`0` =
pass / ok / correct, `1` = fail / error). A question about a bad outcome is
answered literally: "Did the build fail?" is `0` when it failed.

```sh
PYTHONPATH=src python3 -m kilix_bench --suite onebit --endpoint http://127.0.0.1:8080 --model NAME
PYTHONPATH=src python3 -m kilix_bench --suite onebit --command "kilix-llm judge --stdin"
PYTHONPATH=src python3 -m kilix_bench --suite onebit --backend needle
PYTHONPATH=src python3 -m kilix_bench --suite onebit --backend codex --codex-profile NAME
PYTHONPATH=src python3 -m kilix_bench onebit compare A.json B.json
PYTHONPATH=src python3 -m kilix_bench onebit progress --arm qwen --run r1 --step 400 RESULT.json
PYTHONPATH=src python3 -m kilix_bench onebit chart progress.jsonl
```

- 1,768 original items in 16 bands: grades 1 to 12, college, computer use
  (command output, logs, diffs, permissions, configs, safety), reading
  information that contradicts common knowledge, and traps (negations,
  conflict wording, instructions planted in the shown text, false premises).
  Half the keys in every band and domain are `0`. Every key was answered again
  by a separate blind check.
- A quarter of the items are `dev`. The rest are a held-out `test` split that
  no tuning may train on, and every item file carries a canary string.
- Two arms per model: raw (the reply must be exactly `0` or `1`) and
  constrained (a `0|1` grammar where the server takes one, or a JSON schema
  for Codex). Accuracy, balanced
  accuracy, MCC, format failures, the share of `0` answers, and a grade level
  are reported beside the 50% chance baseline. `compare` pairs two runs with a
  sign test. A backend that reports token usage also gets cost columns
  (tokens per item, tokens per correct answer, seconds per item).
- Backends are one module each (chat, command, needle, codex). A Codex run
  asks several items at a time, caches each answer, and stops cleanly on a
  usage limit or network error. Running it again resumes it.
- Results go to the same private results directory, as
  `onebit-<model>-<split>-*`.

The contract, item format, split rules, scoring and backends are in
[docs/onebit.md](docs/onebit.md).

## Local language models

Use BoolQ for passage-based yes/no accuracy, IFEval for verifiable instructions,
and GSM8K for exact numerical answers. Kilix delegates to the named
EleutherAI task with its upstream prompt, split, scoring and few-shot defaults.
BoolQ scores candidate word likelihoods. OneBit measures the separate ability
to return a correct digit.

Install a version of [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
with the dependencies for your selected tasks and backend. For GGUF use a
revision with the teacher-forced likelihood implementation, such as
`d6de81643928d653435c431bae19945d41d32520`. In a Python environment:

```sh
pip install 'lm_eval[ifeval] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@d6de81643928d653435c431bae19945d41d32520'
```

Include the `hf` extra for Transformers models. With a compatible llama-server
serving a GGUF, or an instruction model in a local Transformers directory:

```sh
kilix-bench --suite boolq --backend gguf --model-args base_url=http://127.0.0.1:8080
kilix-bench --suite ifeval --backend hf --model-args pretrained=./instruct-model --apply-chat-template
kilix-bench --suite gsm8k --backend hf --model-args pretrained=./instruct-model --apply-chat-template
```

Use `--limit 10` for a smoke test and `--output DIRECTORY` to retain the
native quality report. The harness may download datasets on the first run.
Record the harness version, model revision, quantization and chat settings
when comparing scores.

### Coding benchmarks

`--suite cruxeval` runs both input and output prediction over all 800
[CRUXEval](https://github.com/facebookresearch/cruxeval) functions. Each
track reports its own pass@1. Input prediction accepts any arguments producing
the requested output. `--suite humaneval-plus` runs all 164 function
completions against both the base and expanded [EvalPlus](https://github.com/evalplus/evalplus)
v0.1.10 cases. Both suites generate one greedy answer per item. Use a
completion backend for HumanEval+ to continue the supplied function.

These suites use the same lm-eval installation and backend options as the
quality tasks above. On Linux, install bubblewrap with user namespaces and
system Python 3. HumanEval+ also needs NumPy and psutil for that interpreter.
The grader runs in a separate namespace with a read-only runtime and private
temporary storage and network. Pinned datasets and the EvalPlus 0.3.1 grader
wheel are downloaded on first use, checked by SHA-256 and reused from
`$XDG_CACHE_HOME/kilix-bench/coding` (default `~/.cache` for the cache root).
`--limit N` selects N items per track. `--output` saves per-item answers
alongside the aggregate report.

```sh
kilix-bench --suite cruxeval --backend gguf --model-args base_url=http://127.0.0.1:8080 --output crux-results
kilix-bench --suite humaneval-plus --backend gguf --model-args base_url=http://127.0.0.1:8080 --output human-results
```

`--suite terminal-bench` runs the **74-task engineering profile** of
[Terminal-Bench 2.0](https://github.com/laude-institute/terminal-bench-2/tree/2fd12b88aafdd04a52c298e3940bcb189f9766d6)
in the upstream Docker environments using Harbor and Terminus 2. Its score
is the mean verifier reward across the selected tasks. Keep the task set when
comparing scores. `--help` lists the individual task names. The default runs
the entire profile sequentially. Repeating `--task` selects individual tasks.
Each run saves its pinned Harbor job configuration, worker log, trials and
score in a new directory. An incomplete or invalid verifier run reports an error.

Install Docker with its Compose plugin and Harbor 0.27.0, or install `uv`
to run the pinned Harbor version on demand. The first run fetches the task
files and container images. For an OpenAI-compatible server, `--endpoint`
uses the model from its models list, or accepts an explicit served ID with
`--model`. Use `KILIX_BENCH_API_KEY` if that endpoint needs authentication.
Without `--endpoint`, supply a Harbor provider/model and that provider's
normal credentials.

```sh
kilix-bench --suite terminal-bench --endpoint http://127.0.0.1:8080 --context-length 16384 --task fix-git --output terminal-results
```

For throughput, install llama.cpp's `llama-bench` executable on PATH. Kilix
writes its native JSON to standard output. By default it measures 512 prompt
tokens and 128 generated tokens in separate tests, with five repetitions each.

```sh
kilix-bench --suite llama-bench --model-file ./model.gguf > throughput.json
```

## Tests

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Documentation

| Document | Contents |
|---|---|
| [docs/scenarios.md](docs/scenarios.md) | Every desktop, render, and text scenario, and how damage, scroll and inline bytes are counted |
| [docs/onebit.md](docs/onebit.md) | The OneBit yes/no suite: contract, items, held-out split, backends, scores |

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
