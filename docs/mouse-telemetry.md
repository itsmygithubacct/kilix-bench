# Optional Neuronte callback telemetry

Analyze a closed `neuronte.pointer.v1` JSONL file without hardware access or the
mouse parser dependency:

```sh
PYTHONPATH=src python3 -m kilix_bench.mouse_telemetry \
  --input telemetry.jsonl --output telemetry-analysis.json
# Optional run provenance; this does not enable a timing join:
PYTHONPATH=src python3 -m kilix_bench.mouse_telemetry \
  --input telemetry.jsonl --run run.json --output separate-analysis.json
```

The output is separate from `mouse run` and `mouse report`. Existing output
files are refused; new reports use mode 0600. Without `--output`, JSON goes to
stdout. Exit 0 means the telemetry stream is structurally valid, **not** that a
benchmark passed. Malformed data, inconsistent counters, invalid references,
unsupported events and noncausal timestamps return 2 without a partial report.
The benchmark's existing missing-event and timeout failure criteria are unchanged.

## Measurement and counting

Latency starts at the runtime's userspace evdev `read()` completion and ends at
its userspace DRM pageflip callback observation, using `CLOCK_MONOTONIC`. Kernel
input queue residence and photon latency are excluded. Full and pointer frames
have separate sample counts, median, 95th/99th percentiles and maximum in milliseconds; quantiles
use linear interpolation between sorted samples. No samples produces null
statistics. Snapshot-enabled metadata is retained so diagnostic runs can be
kept separate from production-path measurements.

Only the **first successful callback observation** referencing each dispatched
input sequence supplies a latency sample. Later scene redraws of an unchanged
pointer are counted as repeated observations, not additional latency samples.
An initial frame or frame after a correlation reset can have coordinates but
no input sequence; it is counted as uncorrelated, with no latency sample.
Intermediate dispatched coordinates that never receive their own callback are
counted as `dispatched_without_individual_callback`, not lost transport events.

`coalesced_motion` counts actual `pointer_input` records with `dispatched=false`.
These are deliberately suppressed motion reports, not inferred from missing
application receipts. Dispatched counts include button, wheel and motion events.
No coalesced input is treated as presented. A cancelled observation supplies
neither pixels nor successful timing, even if it references a known input. A
later successful retry can score that input once. An interrupted observation
does not prove cancellation of an already submitted DRM flip.

VT events reset input correlation and interval counters. `SYN_DROPPED` records
an input discontinuity and resets correlation but, matching the runtime, does
not reset interval counters. The report lists discontinuities and counts inputs
reset by VT. Neither missing kernel events during overflow nor total transport
loss can be recovered from this stream. End-of-file interval counts remain
visible. The format has no end marker: a record-aligned truncated tail cannot be
distinguished from a closed stream. Incomplete final JSONL lines are rejected.

The validator enforces contiguous input sequence numbers, valid action/button
pairs, monotonic input read timestamps, interval counters and current dispatched
references, matching coordinates, and read/render/submit/callback causality.
Input records from a retained batch can carry a read timestamp older than a
preceding frame callback; this is valid and is tested. JSON duplicate keys,
nonfinite numbers, booleans used as timestamps, and unknown events are rejected.

## Why injection joins are omitted

Current runtime metadata declares the clock name but has **no boot identity or
coordinate mapping**. A run's boot ID and declared geometry alone cannot prove
that a separate telemetry file shares that boot, clock epoch, device mapping,
or child viewport. Matching numbers or coordinates are not proof. The analyzer
never consults the analysis machine's current boot ID to authenticate old files.

`--run` therefore checks the basic run schema/clock/geometry declarations and
records its SHA256 only. Output explicitly marks the join omitted; it contains
no injected-to-callback latency or per-scenario event attribution. It does not
recompute, overwrite or relax the run's receipt verdict. Use the existing
`python3 -m kilix_bench.mouse report --input RUN_DIRECTORY` to verify its artifact
manifest and assess that separate measurement. The telemetry file hash is always
recorded; filenames and arbitrary private metadata are not copied into reports.

A future join needs independently bound boot identity and physical-to-child
geometry on both datasets, plus unique coordinate/action/button keys. Duplicate
keys, unresolved mapping, different boots and negative timing must not be joined.
This version intentionally does not offer an override that guesses that evidence.

Tests: `PYTHONPATH=src python3 -m unittest discover -s tests -p test_mouse_telemetry.py -v`.
