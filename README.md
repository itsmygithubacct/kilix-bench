# kilix-bench

Offscreen measurements for Kilix rendering. The desktop suite builds a Kilix
95 desk with no terminal, times the PIL redraw and title-bar hit testing, and
asks the frame presenter how many bytes a change would send. The render suite
times the graphics SDK itself: rectangle diffs, shared-memory placement,
inline placement, a vertical compose, and a sideways window move.

A desktop compose time is the redraw inside the desk. A render time is the
graphics library. Neither includes the GPU.

## Run

```sh
PYTHONPATH=src python3 -m kilix_bench --desktop /path/to/kilix-95
PYTHONPATH=src python3 -m kilix_bench --suite render
```

`--suite desktop` is the default and needs `--desktop`. `--suite render` does
not. `--kilix-home` is the Kilix host checkout that contains `config/kilix_sdk`.
When it is omitted, `KILIX_HOME` or the host sources directory is used.

The live desktop is not started. Desktop state for the run is a temporary
directory and is removed when the run finishes.

## Results

Each run writes three files under `~/.local/gpu_terminal/kilix-bench/results/`.
The desktop suite uses the `kilix-95` name. The render suite uses `kilix-render`.

- `YYYYMMDDTHHMMSSZ-<suite>.json`, the run that just finished
- `<suite>-latest.json`, the same record
- `<suite>-latest.txt`, the same numbers as text

The directory is user-private. Set `KILIX_BENCH_HOME` to put that tree somewhere
else. The JSON record has the desktop version and revision, the Python version,
and the machine type. It does not record local paths.

## What is timed

| Scenario | Measures |
| --- | --- |
| `import_desktop` | Importing the desktop and the graphics SDK |
| `construct_desk` | Building the desk, icons, and taskbar |
| `compose_idle` | A steady redraw of the icon desk |
| `compose_apps` | The same redraw with Notepad, Calculator, and Settings open |
| `compose_cursor` | That redraw plus the software cursor |
| `hover_title` | One mouse move across a title bar |
| `compose_drag` | Repainting the strips a window move exposes |
| `compose_pointer` | Repainting the software cursor after a short move |
| `frame_tobytes` | Copying the framebuffer to RGB bytes |
| `transmit_unchanged` | The desktop send decision for an identical frame |
| `transmit_clock` | A one-pixel taskbar change, using the desktop send decision |
| `transmit_move` | A window moved 12 pixels |
| `presenter_clock` | `FramePresenter` encoding that taskbar change |

The desktop asks `diff_rects` for ordinary damage and `build_compose` when
one window is a pure translation. A change larger than 65% of the frame is
still a full placement. The presenter rectangle count is `diff_rects`,
counted once outside the timed samples, so a translated window can show a
small compose payload beside a much larger bounding rectangle.
`presenter_clock` is the presenter's own encode of the taskbar change,
including the bytes it would write.

## Render suite

| Scenario | Measures |
| --- | --- |
| `load_graphics` | Importing the graphics SDK |
| `diff_pixel` | `diff_rects` on one pixel |
| `diff_window` | `diff_rects` on a sideways window move |
| `diff_hint` | `diff_damage_rects` limited to the exposed strips |
| `present_full` | A shared-memory full placement |
| `present_same` | An identical frame |
| `present_pixel` | A shared-memory edit of one pixel |
| `present_window` | That window move with scroll detection on |
| `present_hint` | The same move limited to the exposed strips |
| `present_scroll` | A vertical shift copied inside the frame |
| `stream_pixel` | An inline edit of one pixel |
| `stream_hint` | An inline upload of the exposed strips |
| `stream_full` | An inline full placement |
| `compose_overlap` | Building an overlapping `a=c` command |

Scroll detection runs only when the changed rectangle is at least a fifth of
the frame, and it only accepts vertical motion. The window probe is larger
than that fifth and moves sideways, so it is not a scroll. `diff_rects` keeps
a clean column gap when a spare rectangle slot remains and the two pieces
cover less than 85% of the box. `diff_window` and an unhinted present are then
the two exposed strips. `diff_hint` names those strips as damage. An inline
frame is a zlib-compressed escape (`o=z`), chunked to fit the graphics
protocol, so a flat frame's wire size is far below the raw RGB. A shared-memory
frame carries a name. The stream samples turn off the presenter's warmup
keyframe, so the timed call is the edit. `stream_full` still asks for a
placement.

## Tests

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
