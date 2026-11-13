# Neuronte and FreeType CPU font protocol

This suite measures terminal glyph generation and CPU drawing with a shared
cache policy. The two systems draw different font designs. Their images are
provided for inspection; pixel differences are not an accuracy score. The
native C worker contains both adapters and one compositor. Python orchestrates
trial pairs, verifies every output and saves records. Python loop and FFI
dispatch overhead are outside the native batch timing.

## Run

Dependencies: Linux, a C compiler, `pkg-config`, FreeType development headers, Pillow,
and a built Neuronte shared library. Supply a scalable monospace font and its
licence file; both are hashed locally and are not copied into the repository.

```sh
PYTHONPATH=src python3 -m kilix_bench --suite fonts \
  --neuronte-home /path/to/neuronte \
  --font /path/to/DejaVuSansMono.ttf --font-license /path/to/font-copyright \
  --cells 6x12,8x16,10x20 --corpora ascii --repeats 20 --warmup 2
```

Other corpora: `box,blocks,braille,latin,greek,russian,sextants,octants`.
Missing FreeType glyphs are listed and excluded from both arms. An empty
intersection is unavailable, with no performance ratio. Neuronte coverage is
declared by exact packed-weight SHA256 in `font_profiles.py`; a successful
route or fallback drawing never establishes support. Profiles currently accept
split-v5, retained split-v4, and ascii-v0. These corpora are subsets of their
declared training sets. Other weights require an explicit reviewed profile.

`--cpu N` restricts only the benchmark child to an allowed CPU. Governor,
affinity, CPU model, kernel and start/end load average are recorded. The tool
does not change machine-wide scheduling or flush system caches. Results on a
busy workstation are exploratory; repeat on a controlled host and across
sessions before treating small differences as stable.

## Measured operations

| Case | Timed operation | Cache state |
| --- | --- | --- |
| `load` | Engine initialization, first glyph, then first corpus; all three durations are also stored separately | Fresh process/engine; OS file cache unspecified |
| `raster` | Repeated normalized 8-bit cell-mask generation | Initialized engine, one corpus warmup, no mask cache |
| `empty-cache` | Whole 80×25 frames | Shared mask cache starts empty |
| `prefilled-cache` | The same drawing | One corpus prefill; actual hits and misses recorded |
| `cache-churn` | The same drawing with a small cache | One corpus prefill, default 16 entries; adversarial working set |

The normal shared cache has 1,024 entries and four-way set-associative LRU
replacement. It is a benchmark control, not FreeType's FTC subsystem or
Neuronte's internal draw cache. Both adapters bypass application mask caches.
`raster_calls`, `cache_hits` and `cache_misses` show the rasterization remaining
in each batch. This does not measure persistent disk caches or establish a
particular application's redraw performance.

Neuronte uses `neuronte_coverage` with automatic sampling and quantizes float
coverage to 8 bits. FreeType uses an explicit Unicode charmap,
`FT_LOAD_RENDER | FT_LOAD_NO_BITMAP | FT_LOAD_TARGET_NORMAL`, and grayscale
outline rendering. `--hinting none` adds `FT_LOAD_NO_HINTING | FT_LOAD_NO_AUTOHINT`;
the default is native hinting. Bitmap strikes and colour fonts are outside this
arm. Both normalized masks contain every pixel in the requested cell.

FreeType uses `FT_SIZE_REQUEST_TYPE_CELL`: a uniform scale fitted using maximum
advance and ascender/descender dimensions. This compares terminal cells, not
equal nominal em sizes. Actual x/y ppem, baseline and nonzero samples clipped
outside the cell are recorded. See the [sizing API](https://freetype.org/freetype2/docs/reference/ft2-sizing_and_scaling.html)
and [glyph retrieval API](https://freetype.org/freetype2/docs/reference/ft2-glyph_retrieval.html).
Inspect previews and clipped-ink counts before interpreting speed as useful
rendering quality. No font fallback or substitute box is timed.

## Controls and evidence

| Option | Default | Meaning |
| --- | --- | --- |
| `--cells` | `6x12,8x16,10x20` | Exact output cells, at most 128×256 |
| `--raster-calls` | `95` | Uncached calls per batch; the actual count is recorded |
| `--frames` | `1` | Whole 80×25 frames per draw batch |
| `--cache-entries` | `1024` | Normal capacity, a multiple of four |
| `--churn-entries` | `16` | Churn capacity, a multiple of four |
| `--seed` | `20261113` | Corpus order and engine order within trial pairs |
| `--job-timeout` | `180` seconds | Bound on each process; increase explicitly for long runs |

`NEURONTE_LIBRARY` and `NEURONTE_WEIGHTS` override assets under
`--neuronte-home`, `NEURONTE_HOME` or `NEURONTE`. Every worker is a fresh process.
Fetch current development refs before selecting Neuronte, and use a clean,
isolated checkout of the revision being tested. Rebuild that checkout with
`make -B build/libneuronte.so` before the campaign. The report records the
library checkout revision separately from the weights checkout revision;
installed assets outside Git have no checkout revision. These identify source
locations, not proof that an existing binary was rebuilt. Preserve the build
log and verify the recorded library hash against the freshly built file.
Untimed trial pairs warm the operating environment, not a cache shared between
workers. Compilation, validation, hashes and PNG encoding are outside native
timed regions. Worker compiler flags, worker/source hashes, library/font hashes,
runtime FreeType version and package metadata are recorded. The worker reports
wall/process CPU times, loop calibration, page faults and peak process RSS.
RSS is not exclusive model memory or a measure of multi-process sharing.
Worker wall time additionally includes process launch, preparation and output;
it is not application startup or visible-frame latency. Loop overhead is
reported rather than silently subtracted.

Each engine first renders the shared corpus into reference masks. A separate
Pillow compositor constructs the expected frame. Every timed frame must match
it byte for byte, including churn runs; uncached output must match the expected
mask. The packed-model test also checks normalization directly against the
public Neuronte coverage API. Errors abort the run instead of manufacturing a
speedup from a failed operation.

Raw trials remain in JSON and `trials.csv`. Each pair randomizes engine order.
Quantiles describe batch durations, not individual-glyph tail latency. With at
least 20 pairs, the report stores a 95% interval from 2,000 paired bootstrap
resamples for **Neuronte time / FreeType time**. Fewer pairs produce a ratio
without an interval. A p99 from a small number of batches is only a sample
statistic. Previews preserve recorded scalar order. No SSIM, quality ranking
or aggregate winner is inferred from different font designs. Runs have unique
records and concurrent saves use distinct temporary files. Storage is private.

## Other layers

Run `--suite text` and `--suite render` separately for existing Kilix application
frame paths and graphics SDK measurements. Those use a stand-in consumer;
they exclude terminal processing, GPU execution, compositor presentation and
scanout. A protocol acknowledgment is not input-to-photon latency.

Proportional UI layout, colour icons/emoji, shell/Zell redraws, persistent-cache
restart, multi-face memory pressure and visible-frame timing need separate
fixtures. The RC9 runtime has upright UI sizes 1–32, italic sizes 3–32 and
icon-v1; earlier split-v4-only inventories do not describe that runtime.

For GPU experiments, compare a fixed-size alpha atlas first, then sampled SDF.
The public coverage API does not expose raw signed distance: verify raw-SDF
access and distance normalization before implementing that arm. Measure atlas
construction, upload, GPU execution and readback independently. Size-specific
UI models need fidelity checks across sizes; colour-layer models are not a
single distance channel. CUDA, Vulkan, RKNN and cross-device runs are outside
this CPU suite and have no performance results here.
