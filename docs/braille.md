# Braille decomposition comparison

The Neuronte suite can compare a width-48 Braille expert with two trained
width-24 experts or four width-12 quadrant experts. It measures native CPU inference, dispatch, composition,
opening and cache reuse, and scores all 256 Braille patterns independently
against the original signed-distance teacher. Install the `neuronte` optional
dependencies for preparation and timing. Training additionally needs PyTorch;
choose its CPU or CUDA build for the machine doing the training.

## Prepare models

Use one pinned Neuronte checkout containing `train/`, `weights/` and a built
`build/libneuronte.so`. Keep generated models and results outside the source
tree. This example trains both designs with two fixed seeds:

```sh
export NEURONTE_HOME=/path/to/neuronte
export PYTHONPATH=src
experiment=/path/to/private/braille-experiment
for seed in 0 1; do
  for part in lower upper left right; do
    python3 -m kilix_bench.braille_train "$experiment/$part-s$seed.json" \
      --neuronte-home "$NEURONTE_HOME" --part "$part" --seed "$seed" \
      --steps 20000 --checkpoint "$experiment/$part-s$seed.pt"
  done
done
python3 -m kilix_bench.braille_prepare "$experiment" \
  --neuronte-home "$NEURONTE_HOME" --cells 8x16,16x32,48x96 --seeds 0,1
```

The adapter runs that checkout's original trainer and checkpoint machinery.
It records its own hash in checkpoint identity and model metadata. Resume a
part with the same arguments plus `--resume` naming its checkpoint. Training
defaults to CUDA; use `--device cpu` when appropriate. TF32 is disabled. The
trainer's small-size scores for individual parts are preliminary: the suite
scores the complete assembled glyph at each actual benchmark size.

The preparer creates a manifest binding packed models and teacher masks by
SHA256. It includes the existing standalone Braille model and split-v5 as
controls. Every requested seed remains a separate case; failed quality is
reported rather than silently discarded.

| Design | Selection and output | Expert parameters |
|---|---|---:|
| One width 48 | One full-cell prediction | 9,121 |
| Two width 24, repertoire | Lower/upper 128 scalars; one full-cell prediction | 6,818 total; 3,409 active |
| Two width 24, parts | Left/right columns; two half-cell predictions | 6,818 total and active |
| Two width 24, rows | Top/bottom halves; two half-cell predictions | 6,818 total and active |
| Four width 12 | Four quadrants; all four predict | 5,668 total and active |

Repertoire selection is tested both through two table ranges and through
the existing dense 64→32→2 neural gate API. The gate is **constructed**, not
learned: it makes an exact decision from scalar bit 7 (Braille dot 8). Its
2,146 parameters raise the gated total to 8,964. The gated bundle has no
routing ranges, so the runtime must execute the gate. Both dispatch variants
contain identical expert weights. This tests the existing dense gate's cost,
not the minimum cost of a bit test or a trained gate's generalization.

The parts design maps each scalar to its canonical left (`bits & 0x47`) and
right (`bits & 0xb8`) four-dot patterns. Each model learns 16 patterns. The
targets map x to x/2 or x/2+1/2; y remains unchanged. File tau is 0.30 so
`tau * half_width` preserves the original 0.15 full-width coverage scale.
The native worker preserves sampling chosen from the **full** width, even
when a half falls below the runtime's automatic 24-pixel threshold.

## Top/bottom halves and quadrants

Prepare the spatial-only comparison with `--spatial`. It includes the existing
width-48 controls and all three layouts. Each quadrant is an actual width-12
network, which requires a Neuronte distance-model loader and packer supporting
widths divisible by four. A reader requiring multiples of eight rejects these
files. Use the same runtime build for every control and candidate.

```sh
for seed in 0 1; do
  for part in left right top bottom; do
    python3 -m kilix_bench.braille_train "$experiment/$part-s$seed.json" \
      --neuronte-home "$NEURONTE_HOME" --part "$part" --seed "$seed" \
      --width 24 --steps 20000 --checkpoint "$experiment/$part-s$seed.pt"
  done
  for part in top-left top-right bottom-left bottom-right; do
    python3 -m kilix_bench.braille_train "$experiment/$part-s$seed.json" \
      --neuronte-home "$NEURONTE_HOME" --part "$part" --seed "$seed" \
      --width 12 --steps 20000 --checkpoint "$experiment/$part-s$seed.pt"
  done
done
python3 -m kilix_bench.braille_prepare "$experiment" --spatial \
  --neuronte-home "$NEURONTE_HOME" --cells 8x16,16x32,48x96 --seeds 0,1
```

Previously trained left/right JSON files can be reused without retraining.
The preparer checks widths, seeds, canonical repertoires, coordinate domains
and tau. The benchmark verifies every packed-model and teacher hash.

| Layout | Canonical masks, in tile order | Patterns per model | File tau |
|---|---|---:|---:|
| Left/right | 0x47, 0xb8 | 16 | 0.30 |
| Top/bottom | 0x1b, 0xe4 | 16 | 0.15 |
| Quadrants, row order | 0x03, 0x18, 0x44, 0xa0 | 4 | 0.30 |

Targets crop the original FP16 field, then map the crop to the runtime's
x=[0,1], y=[0,2] domain. Top/bottom halves map y to y/2 or y/2+1;
quadrants additionally use the left/right x mapping. Distances retain their
original units. Tau compensates for the tile's width in the coverage formula;
halving only height does not change tau. All tiles together evaluate the same
number of pixel samples as the full glyph. The benchmark includes two or four
model calls, canonical bit masking, and assembly into the full output cell.

## Measure

```sh
python3 -m kilix_bench --suite neuronte \
  --neuronte-home "$NEURONTE_HOME" \
  --braille-experiment "$experiment/manifest.json" \
  --cells 8x16,16x32,48x96 --repeats 10 --warmup 1 --expert-passes 4
```

`--cpu` optionally pins the worker. The manifest selects all experiment cases;
`--experts` does not filter them. Dimensions must divide evenly into the layout:
even width for columns, even height for rows, and both for quadrants. Teacher masks
must have been prepared for each requested size. Loading or content errors,
incorrect routes, incomplete batches and inconsistent output abort the run.

| Case | Timed work |
|---|---|
| open | Open all handles and allocate buffers for a model set, before inference |
| route-bench | 131,072 public routing calls with a checked checksum; excludes inference |
| coverage | Full corpus, bypassing the runtime cache; parts include bit masking, every inference and row joins |
| draw-cold | Full corpus from empty handle caches, with cell clearing and native composition |
| draw-warm | Same draw path after a full corpus priming pass |

All jobs are shuffled within each round and use fresh processes and handles.
Process launch and priming are outside the native batch timers; OS file caches
remain populated. The parts draw path makes two or four calls into adjacent
tiles of the same destination cell. Repertoire cases make one call per glyph.

Half models have 16 distinct cache keys per handle; quadrant models have four.
The first corpus
pass can therefore reuse parts of later glyphs. **Cold draw is not an
all-miss inference measurement.** Use coverage for that comparison. Cache
counters are unavailable through the public runtime API. Runtime caches are
separate from the RC9 persisted desktop atlas and startup preparation; neither
desktop facility participates here. Fully cached results skip expert inference
but still include routing. The reviewed runtime selects the expert before
cache lookup and again after lookup, so a gate-only bundle evaluates its gate
twice on a warm draw. Table hits avoid that gate work.

Results are named `kilix-braille` under `KILIX_BENCH_HOME/results`. They include
per-glyph quality, preview sheets, model/library/harness hashes, raw wall/CPU
times, memory/fault counts, and an incremental journal. Summaries use ms/glyph,
ms/model set or ms/route as recorded in each scenario. For very small routing
times, use `extra.median_ns_per_operation` or raw samples rather than the text
report's rounded milliseconds.

Teacher masks use the original FP16 256-grid field, bilinear interpolation
and 8×8 area supersampling. Quality includes binary IoU at a 0.5 threshold,
exact glyphs, wrong pixels and mean absolute coverage error. Small contour
differences can prevent exactness at large sizes; read IoU and coverage error
alongside exact counts. Compare quality before drawing conclusions from speed.
This benchmark does not measure GPU rendering, screen presentation or complete
application startup.
