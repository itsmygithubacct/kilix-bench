# Desktop-face experiments

A Neuronte desktop face is a drawing file: the net or nets that draw printable
ASCII and the ellipsis, and that face's advances. A desktop opens one face per
em and style and draws each character once into its letter cells, so the
uncached cost of a whole face is paid at start-up: when caches are first
built, after every library change, and for any size or style not yet cached.
This experiment measures that cost through the calls a desktop makes.

```sh
python3 -m kilix_bench.face_prepare /path/to/new-experiment \
  --neuronte-home /path/to/neuronte --face ui24 \
  --case v0=/path/to/neuronte/weights/neuronte-ui24-v0.nrt \
  --case candidate=/path/to/candidate.nrt --meta 'candidate={"seed": 0}'
python3 -m kilix_bench --suite neuronte --neuronte-home /path/to/neuronte \
  --face-experiment /path/to/new-experiment/manifest.json \
  --repeats 10 --warmup 1 --expert-passes 2 --cpu 3
```

Add `--italic` for `font/ui-EM-italic.txt`. `--face uilarge` prepares the
scalable face at `--ems` (default 33, 48 and 64), drawn blended. Preparation
refuses an existing directory.

## Teacher

For a pixel face the teacher is the drawing itself, read from its text with no
Neuronte code: each character's rows of `.` and `#` placed from the cell row
`font/ui-tops.json` gives for the em, at the pen. Every case's packed regular
advance must be the drawing's advance in pixels, or preparation stops. The
scalable face has no pixels; its teacher is the true coverage from 8 × 8
samples of the checkout's own target field, computed in a separate process.

## What runs before timing

For every case and em, in one process: the native routes, which must match the
packed range table; every character's public uncached coverage; every
character's first draw from an empty draw cache; then, outside the worker,

- an independent float64 NumPy evaluation of the packed bytes (version 5
  drawing files holding version 1 or 6 members, or one version 2 bundle) on a
  spaced sample and both sides of every route change, within 1e-3;
- every regular and bold advance against the packed advance table;
- every drawn canvas against its own coverage: cut at one half for a binary
  face, the float32 compositor for a blended one;
- quality against the teacher: exact glyphs and wrong pixels for a binary face,
  mean, largest, over-0.25 and over-0.5 errors for a blended one.

A reference, advance or canvas mismatch aborts the run. Quality is recorded and
never removes a case: declare acceptance gates before reading it.

## Timed jobs

Each job is a fresh process and handle; all jobs are shuffled every round, and
discarded warmup rounds come first.

| Case | Timed work | State before timing |
|---|---|---|
| open | `dlopen`, `neuronte_open_file`, `neuronte_set_binary`, buffers | fresh process |
| coverage | `neuronte_coverage` at em × 2em, automatic samples, whole corpus passes | first coverage timed apart |
| draw-cold | per character: `neuronte_text_width` regular and bold, one `neuronte_draw_text` of the regular letter into a cleared (em + 2) × 2em canvas | new handle, empty runtime cache, one pass |
| draw-warm | the same calls, whole passes | every character drawn once first |

Glyph cases report ms per glyph (median of full-batch means) and ms per face
(that times the corpus size). draw-cold per face plus open is a desktop's cost
to build one face's letter cells. Every timed batch's last output must equal
the validated bytes, and its coverage, draw, width and priming call counts must
match the protocol. Raw JSONL and CSV trials, quality, coverage previews and
the worker's source and binary hashes are kept as hashed artifacts.

The face worker (`face_worker.c`) needs only a C compiler and `libdl`; it does
not link FreeType. It does not read or write a desktop's persisted letter
cells, import desktop code or start a preload thread: a desktop start-up study
must measure those separately in an isolated home.
