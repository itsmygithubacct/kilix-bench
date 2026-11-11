# Scenarios

Moved verbatim from the README: what each suite times and how to read the
byte counts.

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

## Text suite

The frame is snapped down to whole 8 by 16 cells. One sample is one
full-grid rewrite on that grid, which is the bench frame, not the live pane.
The suite needs soft-raster's shared library, Neuronte's shared library, and
`neuronte-split-v4.nrt`. It looks for them under
`~/gpu_terminal/kilix-modules/soft-raster/build/` and `~/projects/neuronte/`
(or `$NEURONTE`). `SOFT_RASTER_LIBRARY`, `NEURONTE_LIBRARY`, and
`NEURONTE_WEIGHTS` override those paths.

| Scenario | Measures |
| --- | --- |
| `kitty_text` | One vtebench `light_cells` rewrite: cursor home, then one character in every cell |
| `kitty_text_alphabet` | That benchmark's whole payload, A through Z |
| `bitmap_repeat` | soft-raster's 8×16 font, one character in every cell |
| `neuronte_repeat` | Neuronte drawing that same screen, after the cell cache is warm |
| `bitmap_ascii` | The bitmap font, printable ASCII cycling across the grid |
| `neuronte_ascii` | Neuronte drawing that ASCII grid, cache warm |
| `neuronte_cold` | `neuronte_coverage` of the 95 ASCII characters, which does not use the cache |
| `kitty_shm` | Shared-memory placement of the bitmap ASCII frame |
| `kitty_inline` | Inline placement of that frame |
| `neuronte_inline` | Inline placement of the Neuronte ASCII frame |
| `sixel` | A two-colour sixel of the bitmap letter frame. The time is this encoder; the wire size is the comparison |

`kitty_text` is the ordinary way to render a terminal: the application
sends cells, and Kitty draws the glyphs. The sample stops at the bytes
the terminal would read. vtebench's `light_cells` is that standard; its
script reads the live pane size, and this suite uses the bench grid in
its place. The graphics rows are the Kitty protocol Kilix uses when the
application owns the pixels. Neuronte and the bitmap font fill those
pixels. Sixel is the other protocol that carries a pixel frame. A Neuronte
time at 8×16 uses the library's automatic sample count, which is 2.
