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

A second face is optional and is never mixed into the primary face's
intersection. Pass `--coverage-font id=path`, `--coverage-license id=path`,
and `--coverage-license-name id=name`, or one `--font-manifest` JSON file
with `faces` listed primary-first. Each face is recorded by basename, byte
size, SHA256, licence name, and licence SHA256. The font files stay outside
this package. When the primary cmap intersection is non-empty, only that
intersection is timed. When it is empty, the one coverage face with the
largest intersection is timed instead, and the report names that face. If
every declared face misses the corpus, the corpus stays unavailable and no
trial is written. `.notdef` is not a successful comparison. The primary face
must be scalable and fixed-pitch. A coverage face may declare `pitch=cell`
when it is scalable but not fixed-pitch; its advances are not used as the
cell size. `--cases` limits the timed cases. The default remains all five.

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

## Geometry fixtures

With `--geometry on` (the default), the existing validation raster also
records per-glyph ink, clipped samples, placed and source boxes, edge
contact, advance, and outline mode. `--geometry-cells` defaults to `4x6,5x8`
and is not timed. Box drawing and block corpora also export a seam or block
panel plus a design-delta image. A gap is recorded only where both glyphs
are expected to join. Identical cell masks set `pixel_equivalent` and still
leave `quality_rank` and `similarity_winner` null. Unequal samples are a
design delta, not a winner and not an SSIM. `--geometry off` skips the
fixture images and the extra cells.

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
| `--geometry` | `on` | Record cell geometry and, for box and block corpora, fixture images |
| `--geometry-cells` | `4x6,5x8` | Extra untimed cells for geometry only; `none` disables them |
| `--cases` | all five | Comma-separated subset of the cases above |
| `--presentation` | `off` | `fixture` or `live`; see stages below |

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
Each batch also records its process ID, realtime/monotonic start timestamps,
context switches and page-fault deltas. Counter snapshots are taken outside
the timed loop and can be correlated with separate host diagnostic logs.
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

## Stages

Each font result lists proposed workload stages and a status. `cpu-glyph-raster`
is `measured` when this suite wrote trials. `mask-cache-compose` is `measured`
only when a draw case ran (`empty-cache`, `prefilled-cache`, or `cache-churn`).
`application-protocol` stays `separate-suite`: the text and render suites time
application and graphics-SDK work against a byte-counting stand-in, which does
not acknowledge a frame. `gpu-upload`, `gpu-execution`, `sdf-atlas`,
`compositor-present`, and `scanout` stay `unavailable`. The public coverage
API returns a cell mask, not a raw distance field, and this suite has no
present fence or photon timestamp.

`--presentation fixture` compiles `presentation_probe.c` against the
kitty-framebuffer sources (`KITTY_FRAMEBUFFER_HOME`, or the checkout under
the invoking user's home) and answers its own graphics query on a
pseudoterminal. The font sandbox replaces `HOME`, so the parent passes that
checkout and `KILIX_BENCH_KITTY` into the child. Neither path is stored.
That peer is `protocol-fixture`. Its stages are `fixture-only`.
The numbers from that reply are discarded. `--presentation live` starts Xvfb
and a new Kitty with software GL, its own instance group, and a clean
environment. It does not attach to the caller's terminal. A graphics-query
reply and, when observed, disappearance of the shared-memory name are consumer
acknowledgements. `session_start_ns` includes terminal setup. `shm_consumed_ns`
is the time from `kittyfb_present` until the name is gone after the write.
Neither is GPU time, compositor time, or scanout. A missing Kitty or Xvfb
records the consumer stages unavailable.

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
