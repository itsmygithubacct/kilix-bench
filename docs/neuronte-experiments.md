# Comparing Neuronte decompositions

The Neuronte suite accepts an immutable experiment manifest as an alternative
to the built-in split-v5 inventory. Use `--ascii-experiment manifest.json` for
all 95 printable ASCII characters, `--braille-experiment manifest.json` for
all 256 Braille patterns, or `--expert-experiment manifest.json` for an explicit
ordered repertoire, including disjoint Unicode ranges.

```sh
python3 -m kilix_bench --suite neuronte --neuronte-home /path/to/runtime \
  --expert-experiment /path/to/experiment/manifest.json \
  --cells 8x16,16x32,48x96 --warmup 1 --repeats 10 --expert-passes 4 --cpu 3
```

Choose a CPU allowed by the host's affinity. Run competing experiments
serially and retain power, temperature, frequency and background-load
observations. The suite measures a native single-threaded renderer, not a
desktop's startup, persistent atlas, shaping or display presentation.
The top-level CLI sets `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and
`MKL_NUM_THREADS` to one in its measurement subprocess and records those limits.
This keeps the NumPy reference worker from creating a competing thread pool.
When calling the internal worker module directly, set those variables yourself.

The JSON object has these fields:

| Field | Meaning |
|---|---|
| `schema` | `kilix-bench.expert-experiment/v1`, or the corresponding `ascii`/`braille` schema |
| `corpus` | Ordered distinct integer Unicode scalars; target plane order and expected route order |
| `models` | Object keyed by asset name; each entry has relative `file` and exact `sha256` |
| `targets` | Object keyed by `WxH`; each entry has relative NumPy `file` and exact `sha256` |
| `teacher` | Reproducible teacher/source/provenance metadata, retained in the result |
| `runtime_revision` | Intended source revision; the actual loaded library is separately hashed |
| `cases` | Nonempty list of uniquely named cases, retaining passing and failing candidates |

Each ASCII/general expert case has `name`, `kind: "bundle"`, `members` with
one model asset key, and `routes` with one expected native expert index per
corpus scalar. A standalone scalar net or a tiled member is also a valid
single-handle asset; its route index is zero. Spatial composition belongs in
the runtime member and is included in the measured public call. Repertoire
experts are packed as a routed bundle. Additional metadata, such as seeds,
layout, parameter counts and training-source hashes, is retained verbatim.

Braille retains its historical `parts` cases and known layouts for comparing
composition in the benchmark caller with composition inside a model. An
ASCII/general expert manifest cannot silently request that Braille-specific
scalar transformation.

Targets are finite coverage arrays of shape `(len(corpus), height, width)`,
bounded by zero and one. Every asset must resolve inside the manifest's
directory and match its declared hash. The native route for every scalar is
checked before timing. Scalars are shuffled reproducibly; disjoint ranges
index targets and routes by manifest position, never by scalar subtraction.

Opening is timed separately and includes library loading, all model handles
and coverage buffers. Each timed job starts in a fresh process. The OS page
cache is retained. Coverage uses the public uncached coverage API; warm drawing
primes the entire corpus outside timing; first-corpus drawing starts with an
empty runtime draw cache. It can still hit within that corpus when multiple
scalars share canonical inputs. Every glyph draw clears its ARGB cell. Opening,
routing, coverage and both drawing modes have separate results. Raw wall and
CPU times, full-batch per-glyph means and medians over trials are retained.

Independent reference pixels, complete API-call counts and priming counts
must match before a timing is accepted. A rendered buffer alone is not a
quality pass. Quality records binary-exact and firm-exact glyphs, wrong pixels,
IoU, coverage MAE, pixel-error p99 and nearest teacher identity, including
per-glyph details. A firm pixel has teacher coverage at least 0.1 from one-half.
Nearest-identity scoring accepts teacher aliases with less than half a pixel
of total coverage difference. No quality failure removes a candidate from
the result. Declare selection budgets relative to the original model before
examining candidate scores, and validate additional geometry before shipping.

JSONL and CSV trials, the complete quality record and per-case coverage
previews are hashed artifacts. The CLI writes separate latest results for
ASCII, Braille and general expert experiments. Use a separate
`KILIX_BENCH_HOME` for each concurrent study so results remain attributable.

For training progress and baseline calibration, add
`--experiment-quality-only`. It checks the same complete routes, coverage,
teacher comparisons and previews, with zero timing jobs. Its subject/latest
filename ends in `-quality`, so it cannot replace the performance result.

Packed one-handle cases are also checked outside timing against an independent
float64 NumPy evaluation of the packed version 1/2/6 bytes. The sample includes
spaced corpus positions, the last scalar and both sides of every table route
change. Braille uses its fixed 37-scalar sample. A maximum absolute coverage
difference above 1e-3 aborts the run. The report records each checked scalar,
the maximum difference and the reference source hash. Legacy multi-handle
Braille parts retain their existing compositional checks.

## Packaged ASCII models

The ASCII release includes a four-H64 character split and a two-H64 top/bottom
model. Prepare their comparison from a Neuronte checkout containing
`weights/neuronte-ascii-next.validation.json`:

```sh
python3 -m kilix_bench.ascii_prepare /path/to/new-ascii-experiment \
  --neuronte-home /path/to/neuronte
python3 -m kilix_bench --suite neuronte --neuronte-home /path/to/neuronte \
  --ascii-experiment /path/to/new-ascii-experiment/manifest.json \
  --cells 8x16,16x32,48x96 --warmup 1 --repeats 10 --expert-passes 4 --cpu 3
```

The first command verifies release hashes, rebuilds and verifies the qualified
teacher, and copies six exact model files into a new directory. The cases are
the original standalone ASCII expert, original split-v5, and each replacement
in standalone and complete-font form. Their routes are checked from the packed
bytes. Preparation refuses to overwrite an existing directory.

This measures the packaged seed-0 models. The two-seed selection campaign and
additional 19-size qualification are recorded in the release model card and
validation file; three benchmark sizes alone cannot replace those checks.
The complete replacement bundles retain the original fallback expert, so their
storage and opening cost differ from the standalone files. Keep that distinction
when comparing speed, size and cache behavior.
