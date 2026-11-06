# kilix-bench

Offscreen measurements for Kilix desktops. The 0.2.3 measurement builds a
Kilix 95 desk with no terminal, times the PIL redraw and title-bar hit testing,
and asks the Kilix frame presenter how many bytes a change would send.

A compose time is the redraw inside the desktop. It does not include Kitty.

## Run

```sh
PYTHONPATH=src python3 -m kilix_bench --desktop /path/to/kilix-95
```

`--kilix-home` is the Kilix host checkout that contains `config/kilix_sdk`.
When it is omitted, `KILIX_HOME` or the host sources directory is used.

The live desktop is not started. Desktop state for the run is a temporary
directory and is removed when the run finishes.

## Results

Each run writes three files under `~/.local/gpu_terminal/kilix-bench/results/`:

- `YYYYMMDDTHHMMSSZ-kilix-95.json`, the run that just finished
- `kilix-95-latest.json`, the same record
- `kilix-95-latest.txt`, the same numbers as text

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

## Tests

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
