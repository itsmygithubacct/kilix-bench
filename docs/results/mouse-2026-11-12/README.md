# Physical terminal mouse comparison — 2026-11-12

Measured on one Intel i7-6820HQ / HD 530 machine, Debian 13.7, kernel 6.12.113, internal 3840×2160 panel. These are single runs per configuration, not repeated-run confidence intervals. The benchmark uses local CLOCK_MONOTONIC timestamps from immediately before uinput batch write to the application's completing-chunk read return. This is **not photon latency**. Visible mode redraws a marker after each receipt; its output affects subsequent events.

Both backends ran the same 3000×1500 interior target path directly, without tmux. Each complete run planned 755 events: 3 calibration, 120 serialized, 120 at each of 60/125/250/500/1000 Hz, 9 button, 8 wheel and 15 drag events. Regular Kilix used X11 with no window manager, a 240×67 grid and origin (1,9). Neuronte used KMS, a 240×64 child grid, origin (0,64), and a 3840×2144 mapped framebuffer. Both used a 3840×2160 absolute injector. Calibration verifies the exact inverse integer mapping. Reports retain actual injection rates and lateness.

| Run | Serialized median / p95 (ms) | Received / planned | Strict result |
|---|---:|---:|---|
| kilix-baseline-quiet-01 | — | 0/755 | failed |
| kilix-fixed-quiet-01 | 14.38 / 15.48 | 755/755 | failed |
| kilix-fixed-v2-quiet-01 | 14.32 / 15.83 | 755/755 | passed |
| kilix-fixed-v2-visible-01 | 14.37 / 15.57 | 755/755 | passed |
| neuronte-baseline-diagnostic-quiet-01 | 330.25 / 378.20 | 412/755 | failed |
| neuronte-baseline-quiet-01 | 115.33 / 115.99 | 582/755 | failed |
| neuronte-baseline-visible-01 | 115.31 / 116.08 | 582/755 | failed |
| neuronte-improved-quiet-02 | 14.71 / 15.76 | 390/755 | failed |
| neuronte-improved-visible-01 | 115.82 / 132.63 | 200/755 | failed |
| neuronte-final-quiet-01 | 14.73 / 15.76 | 390/755 | failed |
| neuronte-final-visible-01 | 31.98 / 32.98 | 276/755 | failed |

Regular Kilix's original calibration failed because slave XI_Motion coordinates lagged the master pointer by one event. The first fix delivered all planned events but introduced an extra movement after horizontal scrolling. The final fix routes master motion, ignores emulated core wheel events before cursor updates, and refreshes scroll baselines on focus/entry. Both final runs pass all 755 events with no unexpected, duplicate or out-of-order receipts. Fixes were tested in isolated builds; the installed regular Kilix generation was preserved.

Neuronte's final quiet serialized median is 14.73 ms, down from 115.33 ms; visible is 31.99 ms, down from 115.31 ms. Final button (9), wheel (8), and drag (15) sequences all arrived in both modes. Baseline horizontal wheels were missing. Final quiet motion sweep receipts are 120/59/31/16/9; visible receipts are 61/30/16/9/5 at 60/125/250/500/1000 Hz. The runtime intentionally coalesces pending motions, but the common metric labels absent samples as loss-or-possible-coalescing. The strict all-events result remains FAILED; low latency does not establish delivery of every intermediate position. Visible mode remains about twice as slow as quiet mode and misses intermediate motion even at 60 Hz.

Full diagnostic snapshots are excluded from production comparisons. The separate baseline diagnostic run shows their cost: 330.25 ms median versus 115.33 ms with capture off. The optimized runtime avoids full composition for pointer-only updates and caches text ink/statistics/native cell records for content updates. Its separately trained conventional arrow model preserves existing models.

Separate telemetry on the intermediate runtime recorded 365 explicitly coalesced logical inputs, 391 dispatched inputs and successful callback observations for all 391; pointer read-to-callback median/p95 were 16.46/16.60 ms. That boundary excludes kernel queue time, is not photon latency, and was not measured on the final runtime. The telemetry lacks boot identity, so matching aggregate counts do not prove an exact join with injection events.

Source cutoffs:

- Benchmark implementation: `9f9c8b5`; optional telemetry analyzer: `0c629aec5dd149578fabc55c50383bce383772f5`.
- Parser: kitty-input 0.2.1, `0f009064f3f32c7ea972a3d90de1522eda409e89`; four source hashes are enforced and included in reports.
- Regular baseline underlying source: `1bbe430140b0cec6df3a83bca33d0c80209d3a95`; initial fix `65111a03a4c05d8c6586880ac398262f1feeb5d7`; final `16ba1b1a177b9cd6a4e79c1a865286185a2bd9e5`.
- Neuronte baseline `51d8b61c35b9660370de8df95ba96556f1447fcb`; intermediate `9ba3957573cc587e472f349a55a4a1b8c7508c0b`; final `c3edbecdb6af9021f37fadfdcde2c7b21cefdbed`.

Validation: 38 benchmark tests, 96 final runtime tests and 15 X11 routing tests passed. Two final application framebuffer captures, before and after a VT switch, independently reconstruct exactly against a frozen pre-cache reference (8,232,960 pixels each). These are application snapshots, not external scanout captures. Original failed runs remain in the results.

Each adjacent JSON is the unchanged public report. `artifact-hashes.json` binds reports to the retained private raw run artifacts; raw event logs and host configuration are intentionally kept outside the repository. See [suite usage](../../mouse.md) for reproduction and [telemetry](../../mouse-telemetry.md) for the separate callback boundary.
