# Terminal mouse sessions

Run directly in the terminal under test, without a multiplexer. Python 3.11+
and Linux are required. The application uses the standard-library kitty-input
parser at revision `0f009064f3f32c7ea972a3d90de1522eda409e89`, version 0.2.1.
It checks SHA256 hashes of all parser source files before enabling input.
No parser implementation is duplicated here.

Install with `python3 -m pip install '.[mouse]'`, or use checkouts:

```sh
export PYTHONPATH="$PWD/src:$KITTY_INPUT_CHECKOUT/python/src"
python3 -m kilix_bench.mouse --help
install -d -m 700 "$HOME/mouse-private"
```

A coordinating operator creates a bounded, persistent virtual device in a
separate local shell. It has absolute mouse axes, three buttons, vertical and
horizontal wheels, and INPUT_PROP_POINTER. It has no keyboard capability.
Root and explicit opt-in are both required. The socket belongs to the client
UID, in a directory private to that UID. Existing socket paths are refused.

```sh
sudo env PYTHONPATH="$PYTHONPATH" python3 -m kilix_bench.mouse inject \
  --allow-uinput --uid "$(id -u)" --socket "$HOME/mouse-private/inject.sock" \
  --width 3840 --height 2160 --lifetime 1800
```

The injector prints a unique name and sysfs input name. Verify the associated
`event*` node with sysfs/udevadm, confirm `ID_INPUT_MOUSE=1`, and check
`libinput list-devices` and `xinput list`. Configure absolute mapping to the
target output, with no scaling or acceleration, and attach the node to the
other backend as required. Device discovery and display switching belong to
the operator. No input is sent before a command. The device is destroyed on
normal exit, signals or errors, and expires after the lifetime/event bound.

Start the app directly in the target terminal:

```sh
python3 -m kilix_bench.mouse app --socket "$HOME/mouse-private/app.sock" --mode quiet
```

Run in another shell on the **same machine**, with verified content origin and
mapped framebuffer dimensions. This is an illustrative transform, not automatic
geometry detection:

```sh
python3 -m kilix_bench.mouse run \
  --app "$HOME/mouse-private/app.sock" --injector "$HOME/mouse-private/inject.sock" \
  --output "$HOME/mouse-private/regular-baseline-quiet" \
  --backend regular --variant baseline --revision SOURCE_COMMIT \
  --width 3000 --height 1500 --origin-x 0 --origin-y 8 \
  --mapped-width 3840 --mapped-height 2160 --timeout 1 \
  --count 120 --rates 60 125 250 500 1000
python3 -m kilix_bench.mouse report --input "$HOME/mouse-private/regular-baseline-quiet"
python3 -m kilix_bench.mouse control --socket "$HOME/mouse-private/app.sock" stop
```

Use a fresh application and output directory for every run. Start the second
phase with `--mode visible`. Stop the injector via `control ... stop` when done.
`control ... info` inspects readiness, mode, terminal grid, boot clock, and
axis bounds. `control ... drain` exposes receipts during manual preflight;
calling it during a measured run would steal receipts and invalidate the run.

## Geometry and workloads

`--width/height` specify a shared interior target rectangle in child-app pixels,
not workspace rows. `--origin-x/y` are the child content origin in the mapped
framebuffer. `--mapped-width/height` describe the backend's absolute-axis mapping,
which can differ from physical scanout size. For each axis, injection uses
`ceil((child_pixel+origin)*(axis_size-1)/(mapped_size-1))` and verifies the
forward floor mapping is exact. Coordinates are zero-based after SGR decoding.
The app records the actual terminal columns/rows; the run records both mapping
and device bounds. Calibration at three widely spaced interior targets must
succeed before any scored scenario. Verify content bounds yourself; an origin
must account for terminal padding, borders and workspace chrome.

Each run includes serialized motion, rate sweeps, three-button press/release,
four wheel directions, and a held-button drag. Motion targets are unique across
the run. Transitions reuse the preceding coordinate with distinct action/button
keys to avoid generating an unintended extra motion in their SYN_REPORT.
Serialized cases wait for the matching coordinate/action/button/wheel receipt.
Sweeps use local monotonic deadlines; reports include actual injection rate and
maximum schedule lateness, since socket/syscall overhead can limit high rates.
No physical pointer or keyboard activity should occur during a run.

Quiet mode emits no terminal output during receipt, allowing the pointer-only
renderer path to operate. Visible mode writes a changing marker at the top-left
for every mouse receipt, deliberately invalidating terminal content. Reports
always label these phases separately. Neither mode queries the terminal during
the measurement. Entry/exit save and restore tty settings and DEC mouse modes;
unsupported private-mode save/restore falls back to disabling mouse protocols.

## Boundaries and artifacts

The common metric is injection batch write-start to the application's read-return
for the chunk completing the event. It includes kernel queues, compositor/input
routing, terminal processing and application scheduling. Events completed in the
same chunk share its receipt timestamp; fragments use the final chunk. Parser
completion and injector write completion have separate timestamps. These are
software receipt measurements, not photon latency. Visible marker rendering is
not included in receipt latency for that event, but affects later work.

Optional Neuronte renderer telemetry must remain a separate dataset: userspace
evdev-read to DRM pageflip callback observation excludes kernel queue residence
and is not the same boundary. Do not substitute it for common receipt latency
or infer presentation of intermediate coalesced coordinates.

Artifacts are user-private:

- `run.json`: complete planned sequence, injection attempts/acknowledgements,
  timestamps, receipts, failures, geometry, boot-clock identity, revisions and
  exact parser pin. Nonmouse text is redacted, but event kind/time is retained.
- `report.json`: per-scenario `p50_ms`, `p95_ms`, `p99_ms`, maximum,
  receipt throughput, injection rate, missing sequences, unattempted commands,
  uncertain unacknowledged injections, contamination, duplicates and order errors.
- `manifest.json`: relative artifact names and SHA256 digests. `report` checks
  these before recomputing the public summary.

Public labels cannot contain paths; reports also enforce the existing local-path
exclusion policy. Do not publish private runtime logs as public reports.
No missing sample is silently removed: missing and unattempted events are
listed, and empty distributions have null statistics. Lost motion is labelled
`loss_or_possible_coalescing`: the common boundary alone cannot distinguish
coalescing from transport loss. Timed-out samples arriving later remain recorded,
but their timeout still fails the run. Calibration failure aborts scoring and
performs a final grace drain; events arriving even later can be inspected with
`control drain`. Counts for unused planned scenarios remain visible.
Any timeout, contamination, overflow, duplicate, ordering error, negative latency,
or missing sample produces nonzero run/report exit status. Injector buttons are
released even on failed runs. A failed acknowledgement means injection outcome
is unknown, not that no event was sent. Hardware results require actual runs;
unit tests and pseudo-terminal tests are not hardware performance evidence.

## Tests

```sh
PYTHONPATH="src:$KITTY_INPUT_CHECKOUT/python/src" python3 -m unittest discover -s tests -v
```

Tests cover parser fragmentation/combined reports, malformed resynchronization,
pixel/cell labels, exact inverse mapping, correlation/error accounting, private
artifacts, checksum verification, mouse-only injection validation, cleanup, and
real pseudo-terminal quiet/visible behavior. Parser-dependent tests are explicitly
skipped when the pinned package is unavailable.

For separate Neuronte userspace-read to DRM callback analysis, see
[Optional Neuronte callback telemetry](mouse-telemetry.md). The standalone
`python3 -m kilix_bench.mouse_telemetry` command does not change this benchmark's
receipt metrics or failure criteria.
