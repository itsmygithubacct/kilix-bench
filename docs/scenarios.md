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
