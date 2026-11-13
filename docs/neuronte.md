# Split-v5 expert protocol

```sh
PYTHONPATH=src python3 -m kilix_bench --suite neuronte \
  --neuronte-home /path/to/neuronte \
  --cells 8x16,16x32,48x96 --repeats 5 --warmup 2
```

This CPU suite uses the native font worker, a built `libneuronte.so`, Pillow,
a C compiler, `pkg-config` and FreeType development headers. No FreeType font,
Kilix host, state library or running desktop is required. `NEURONTE_LIBRARY`
and `NEURONTE_WEIGHTS` override the library and model paths.

Only the reviewed split-v5 model is accepted, SHA256
`8a072c27f1241c536b04ea77d82efdc005399762421f86bd18d950206528ec83`.
The complete training repertoires are extracted from that exact bundle's
68 routing ranges. Unknown files are refused, including renamed models.
A successful route through the learned fallback gate never establishes
support. The suite checks every selected scalar against `neuronte_route`.

| Expert | Model | Hidden width | Trained scalars |
|---|---|---:|---:|
| ascii | ascii-v0 | 128 | 95 |
| draw | draw-v2 | 192 | 131 |
| symbol | symbol-v0 | 128 | 49 |
| accent | accent-v2 | 256 | 318 |
| pixel | pixel-v1 | 128 | 327 |
| braille | braille-v0 | 48 | 256 |

The accent corpus includes its Latin, Greek and Russian labels. Every
character in each selected repertoire runs in every measured batch.

## Timing boundaries

| Case | Timed work | State before timing |
|---|---|---|
| engine-open | `dlopen`, weight parsing/allocation and one 1×1 float buffer | Fresh process and handle; whole bundle, once per round |
| coverage | `neuronte_coverage` with float output | Loaded model; complete corpus primed; no glyph cache |
| draw-cold | `neuronte_draw_scalar` and clearing one ARGB cell per call | Empty runtime draw cache; exactly one corpus pass |
| draw-warm | Same public draw call and cell clearing | Complete corpus drawn once into the runtime cache before timing |

Coverage and draw summaries are **milliseconds per glyph**: the median of
complete-batch mean costs. They are not individual glyph latency percentiles.
Raw batch wall and process CPU times are retained. Engine-open is milliseconds
per whole bundle, not a per-expert or per-glyph load cost. Process launch,
output file writes and correctness checks are outside native timers.

Each worker opens a fresh handle. The first direct coverage call is measured
separately; it cannot populate the draw cache. Priming is separately timed and
excluded from the batch. OS file caches are **not** dropped: engine-open is
not cold-disk startup. Automatic sampling is 2×2 below width 24 and 1×1
otherwise, so size comparisons must account for both pixels and sample count.

The draw cases use Neuronte's own 1,024-entry, four-way cache, including its
scalar/size/sampling key and collision behavior. They do not use the font
comparison suite's u8 cache or compositor. Runtime cache counters are not
exposed by the public API and are recorded as null. Primed does not guarantee
all hits: set collisions may evict masks even below total capacity. The draw
buffer is one cell, white on transparent, reset before each glyph; this is
not a full terminal frame or an application redraw.

## Controls and records

- `--experts all` is the default; select names such as `ascii,accent,braille`.
- `--expert-cases coverage,draw-cold,draw-warm` is the default. Use
  `--expert-cases coverage` for inference alone; engine opening is still recorded.
- `--expert-passes N` repeats each complete corpus N times in coverage and
  warm batches. Cold batches always use one pass so later passes cannot dilute
  the empty-cache cost. Default: 1; maximum: 1,000.
- Defaults are 8×16, 16×32 and 48×96, five measured rounds and two discarded
  rounds. `--cpu` pins the benchmark child; `--seed` reproduces corpus and
  scenario order. All expert/size/case jobs are shuffled in each round.
- `--job-timeout` bounds each native job, including its setup and priming.
  No system CPU governor, global cache or live application state is changed.

Results use the `kilix-neuronte` name under `KILIX_BENCH_HOME/results`.
Private artifacts include PNG coverage sheets, CSV trials and an incrementally
flushed JSONL journal, including discarded warmups. JSON records include model,
library, harness and worker hashes; revisions; corpus order; compiler flags;
CPU model, affinity, governor and load; initialization, first coverage,
priming, wall/CPU time, RSS and fault counts. Harness flags describe the worker,
not the externally built runtime. Retain that runtime's build log separately.

Before timing, every selected glyph's float coverage must be finite and in
range. Each timed batch checks its last output against those validated bytes;
draw output is independently derived from the reference float coverage.
Every API error, wrong route, incomplete pass or output mismatch aborts the
campaign. The reference checks implementation consistency, not visual quality.

## RC9 desktop cache and startup

RC9's proportional desktop faces are separate `neuronte-ui<em>` assets,
not the six split-v5 experts. The desktop caches binary letter cells in memory
and in persisted `NRTATLAS` files. Their identity includes the model and
library hashes, format, size, slant and placement. A complete valid file avoids
inference for its letters; a missing cache starts background preparation of
the two startup faces. `main.py --prepare-cache` finishes that preparation and
private bytecode compilation during installation. Generated `font/ui-tops.json`
also removes startup evaluation of training code. See the pinned
[desktop changes](https://github.com/itsmygithubacct/kilix-95/commit/ac9ae400c8e849b660b1e48ee115abf1d504b583).

This suite imports no desktop code, starts no preload thread and reads no
persisted atlas. These cases therefore measure split-v5 costs independently
of those startup changes. A desktop startup study must separately measure an
empty private cache, a completed installation preparation, and a subsequent
restart, retaining each state within its own isolated home. The existing
desktop suite's fresh private home does not represent a prepared RC9 launch.
