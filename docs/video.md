# Video suite

`kilix-bench --suite video` compares the real Neuronte modules and presenter
using deterministic RGB input at the requested width and height. Build both
candidate and optional baseline libraries first. Results use the
`neuronte-video` name under `KILIX_BENCH_HOME/results`.

| Scenario | Boundary |
|---|---|
| `palette_reference` | Edge padding, palette encoding, learned-mask decoding and crop |
| `native_rgb24` | Fused composition into a reusable RGB buffer |
| `native_planes` | The same composition from G, B, R planes, the layout the player decodes |
| `read_allocated` | Read a complete frame from memory into newly allocated bytes |
| `read_reused` | Read the same frame into the reusable player input buffer |
| `shared_memory` | Real player presentation, consumer read/hash and unlink acknowledgement |
| `decode_D_raw_O` | FFmpeg startup, accurate input seek, decoding, scaling and pipe output |
| `decode_0_raw_1_gbrp` | The same command writing planes instead of packed RGB |
| `library_0_rgb24`, `library_0_gbrp` | Neuronte's in-process decoder through its pipe harness, at the command's boundary |
| `seek_command`, `seek_library` | A start or seek to the first complete frame: the command as a new process, or the in-process decoder |

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

With a candidate that has planar decoding, the command also writes `gbrp`,
and with a built `libneuronte-media` the in-process decoder's harness
(`build/neuronte-media-decode`) decodes the same interval in both layouts.
Planar frames are hashed as packed RGB, so every configuration, layout and
decoder must give the same hashes. The harness pays a process start and the
pipe like the command; the player itself pays neither, which the seek
scenarios and whole-player runs show. The seek scenarios take the first
frame from three starts, `--start` and 4.9 and 13.2 seconds later, in
rotated order; both decoders must give the same first frame from each. The
command's CPU is its own, from `wait4`; the in-process decoder's is this
process's from open to close, so it includes its threads.

Scenario order rotates each repetition; warmups are excluded from summaries.
JSON retains individual measured samples and decoder runs, child CPU usage,
peak RSS (KiB on Linux), first-frame latency, driver CPU, commands with the
input path removed, load averages, source metadata/hash, FFmpeg identity,
NumPy version, model/library hashes and checkout revisions/dirty flags.
Wall times and CPU summaries are milliseconds; raw timings are seconds.
Decoder CPU is collected with `wait4` for that child only. The driver is
reported separately. Each command or pipe-harness child has a 60-second limit
and is reaped on failure. The outer runner also bounds the complete suite,
including in-process seeks, and stores results only after it passes.

Use alternating, repeated whole-player runs as well before drawing conclusions
about playback. Kernel speedups alone cannot establish a whole-player gain,
and load averages document contention without removing its effects.
