# Video suite

`kilix-bench --suite video` compares the real Neuronte modules and presenter
using deterministic RGB input at the requested width and height. Build both
candidate and optional baseline libraries first. Results use the
`neuronte-video` name under `KILIX_BENCH_HOME/results`.

| Scenario | Boundary |
|---|---|
| `palette_reference` | Edge padding, palette encoding, learned-mask decoding and crop |
| `native_rgb24` | Fused composition into a reusable RGB buffer |
| `read_allocated` | Read a complete frame from memory into newly allocated bytes |
| `read_reused` | Read the same frame into the reusable player input buffer |
| `shared_memory` | Real player presentation, consumer read/hash and unlink acknowledgement |
| `decode_D_raw_O` | FFmpeg startup, accurate input seek, decoding, scaling and pipe output |

Kernel pixel equality is checked before timing. Read measurements exclude
disk and pipe costs. The shared-memory consumer checks geometry and length,
reads the payload before unlinking it, and checks pixel hashes against the
source before timing. Its read and hash costs are included in that scenario.
These offscreen measurements do not measure physical display frame rate.

When a video is supplied, decoder/output thread pairs are `(0,0)`, `(0,1)`,
`(1,1)`, `(2,1)` and `(4,1)`; zero means automatic. Scaling uses one worker,
Lanczos and the source's declared chroma location through Neuronte's shared
decoder-command builder. The first complete frame measures seek/startup
latency. All complete RGB hashes must agree across every configuration;
partial frames, changed pixels and unexpected frame counts fail the run.

Scenario order rotates each repetition; warmups are excluded from summaries.
JSON retains individual measured samples and decoder runs, child CPU usage,
peak RSS (KiB on Linux), first-frame latency, driver CPU, commands with the
input path removed, load averages, source metadata/hash, FFmpeg identity,
NumPy version, model/library hashes and checkout revisions/dirty flags.
Wall times and CPU summaries are milliseconds; raw timings are seconds.
Decoder CPU is collected with `wait4` for that child only. The driver is
reported separately. Each decoder has a 60-second limit and is reaped on
failure. The parent stores results only after the complete suite passes.

Use alternating, repeated whole-player runs as well before drawing conclusions
about playback. Kernel speedups alone cannot establish a whole-player gain,
and load averages document contention without removing its effects.
