# Side by side: Neuronte 2, Neuronte 0.1.4 and DejaVu

Three terminal panes in one Kilix tab draw the same text, each with its own
face. A character typed in any pane goes to all three, so the same text can
be read side by side.

| Pane | Draws with | Set by |
|---|---|---|
| left | Neuronte 2's library and its terminal bundle, the newest `neuronte-split-v*.nrt` in the tree | `NEURONTE_V2_HOME` |
| middle | Neuronte 0.1.4's library and `neuronte-split-v5.nrt`, the terminal font 0.1.4 uses by default | `NEURONTE_014_HOME`, `NEURONTE_014_BUNDLE` |
| right | DejaVu Sans Mono through FreeType, and DejaVu Sans for the characters Mono lacks | `DEJAVU_FONT` |

Neuronte 0.1.4 also carries `neuronte-split-v9.nrt`, the same file Neuronte
2.2.0 ships. `NEURONTE_014_BUNDLE=neuronte-split-v9.nrt` puts it in the middle
pane; the two Neuronte panes then show the same pixels.

Each pane draws its text as pixels and shows them with the Kitty graphics
protocol at the size they were drawn, so what is on screen is the face's own
output and not the terminal's font.

## Set it up

Build both Neuronte trees with `make`, then say where they are, in
`~/.config/kilix-bench/sidebyside.env` or in the environment:

```sh
NEURONTE_V2_HOME=/path/to/neuronte-v2
NEURONTE_014_HOME=/path/to/neuronte        # checked out at v0.1.4
```

It needs Python 3 with NumPy and Pillow. DejaVu is found in the usual font
directories.

## Use it

```sh
sidebyside/launch.sh              # open the tab; run it from a Kilix pane
sidebyside/send.sh 'some text'    # send text to all three from anywhere
sidebyside/show-common.sh         # every character all three have
sidebyside/show-dejavu-only.sh    # every character DejaVu has and Neuronte does not
sidebyside/charsets.py summary    # the counts, block by block
sidebyside/bench.sh               # time the three
```

Keys in a pane act on all three:

| Key | Does |
|---|---|
| Enter, Backspace | as in a terminal |
| PgUp, PgDn | scroll |
| Ctrl-L | clear the text |
| Ctrl-T | next cell size: 8x16, 12x24, 16x32, 24x48 |
| Ctrl-U | mark the characters a face does not have, on and off |
| Ctrl-C | close that pane |

`send.sh --clear`, `--size`, `--marks`, `--up` and `--down` do the same from
a script.

## What the listings show

With Neuronte 2.2.0, Neuronte 0.1.4 and DejaVu 2.37:

| | Characters |
|---|---:|
| Neuronte 2 bundle (`neuronte-split-v9`) | 1,341 |
| Neuronte 0.1.4 bundle (`neuronte-split-v5`) | 1,176 |
| DejaVu Sans Mono and DejaVu Sans together | 6,065 |
| All three have | 875 |
| DejaVu has and Neuronte 2 does not | 5,026 |
| Neuronte 2 has and DejaVu does not | 302 |

A Neuronte bundle draws a character it was not trained on anyway, with a
net that never saw it, so it comes out as some other shape.
`show-dejavu-only.sh` shows exactly those characters; Ctrl-U draws a plain
mark in their place. `--against n014` or `--against both` compares DejaVu
with the other bundle or with the two together. Combining marks and
characters with no picture are counted and not listed.

## The benchmark

`bench.sh` times each face over the characters all three have:

- **first**: a character's cell the first time it is asked for. The net
  runs, or FreeType rasterises.
- **again**: the same cell asked for again. Neuronte copies it from the
  handle's cache when it is still there; the cache is small, and with 873
  different characters some are drawn again. FreeType draws every time.
- **screen**: an 80 x 24 screen composed from cells the program has kept,
  which is what a pane does after the first time. It is the same work for
  all three.

Times include the Python call into each library. Measured on idle machines,
one at a time, median of 9 repeats, 873 characters:

| Cell | Face | Core i7-6820HQ first | again | Core i7-10700F first | again |
|---|---|---:|---:|---:|---:|
| 8x16 | Neuronte 2.2.0 | 421 µs | 68 µs | 322 µs | 51 µs |
| | Neuronte 0.1.4 | 744 µs | 122 µs | 585 µs | 95 µs |
| | DejaVu | 73 µs | 73 µs | 59 µs | 59 µs |
| 12x24 | Neuronte 2.2.0 | 992 µs | 152 µs | 753 µs | 116 µs |
| | Neuronte 0.1.4 | 1,676 µs | 266 µs | 1,295 µs | 203 µs |
| | DejaVu | 75 µs | 74 µs | 60 µs | 60 µs |
| 16x32 | Neuronte 2.2.0 | 1,825 µs | 278 µs | 1,298 µs | 199 µs |
| | Neuronte 0.1.4 | 3,002 µs | 461 µs | 2,326 µs | 366 µs |
| | DejaVu | 76 µs | 75 µs | 61 µs | 61 µs |

Neuronte 2.2.0 draws a cell for the first time about 1.7 times as fast as
Neuronte 0.1.4's default bundle. FreeType is 5 to 24 times as fast as
Neuronte 2.2.0 on a first draw, and its time barely changes with the cell
size while a net's grows with the pixels. Once a program keeps the cells,
a screen costs the same whichever face drew them: 4 to 6 ms at 8x16 and
17 to 27 ms at 16x32.

## Check it

```sh
make test
```

`tests/test_sidebyside.py` checks the shared text, the keys, the picture
sent to the terminal and the listings. It needs no Neuronte tree and no
terminal.
