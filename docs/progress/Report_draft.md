# COMP4423 – Assignment 1 — Triangle-Brick Image Generation (DRAFT)

**Author**: Huang Haoran (24101322D)
**Repo**: https://github.com/Retr0-rgb-lab/COMP4423_Assignment1
**Status**: draft for author review. Figure/table anchors point at `code/pics/`
and `docs/progress/`. Placeholders `【AUTHOR: ...】` mark sections the author
writes personally (GenAI limitations; the Task-4 Level-2 scenes to capture).

---

## 1. Introduction

Given a single image, the assignment asks to convert it into a mosaic of
triangular bricks that visually approximates the original, under a hard budget
of at most 10,000 individual right-isosceles triangles, with no gaps or
overlaps, where each brick carries one uniform face colour determined from the
image region it covers. Multiple brick sizes and multiple colours are allowed.

The central question is not "more triangles is better", but **how to allocate a
fixed budget of detail across the image**. Three effects compete:

- **Detail preservation** — edges, texture and contours improve with higher
  triangle density;
- **Noise amplification** — a triangle covering very few pixels estimates its
  mean from a tiny sample, so sensor/JPEG noise is rendered as fake detail;
- **Quantisation loss** — the colour error from mapping each region to a fixed
  palette entry is independent of density and depends only on the palette size
  and the thresholding method.

The report follows the development of a solution across the four tasks. Each
task adds one decision layer on top of a shared geometry/colour toolkit, and
each exposed a problem whose solution shaped the next step — the "converging"
structure of the pipeline:

- Task 2 fixes the tessellation (equal size, 3 colours) and motivates adaptive
  geometry by measuring that a uniform grid does not align with image content;
- Task 3 makes the geometry and palette adaptive (quadtree + K-Means-Lab) and
  quantifies the density/colour trade-off;
- Task 4 moves the pipeline into a live camera loop under a real-time
  constraint and turns the problems exposed there (bottleneck stages,
  frame-to-frame jitter, camera exposure drift) into measured improvements.

**Test image.** `code/pics/sky.jpg` (1706x1279) is a cityscape — canal, brick
wall, trees, distant buildings — not sky/cloud content. Its dominant signal is
luminance variation rather than hue variation, which has a measurable effect on
the colour-quantisation experiments (Sections 3 and 4).

---

## 2. Method — the shared toolkit

All tasks share a geometry/colour/rendering core (`code/brick_geom.py`,
`brick_color.py`, `brick_render.py`, `brick_metrics.py`). Design decisions:

| Component | Choice | Rationale |
|---|---|---|
| Brick geometry | Square cell cut by **one diagonal**, direction alternates by checkerboard parity | Guarantees right-isosceles, gap-free, overlap-free tiling; alternation avoids a visible single-directional smear |
| Brick size | Length of the equal legs, in base-grid units; Task 2 fixed, Task 3 power-of-two sizes {1,2,4,8,16,32,…} | PDF definition; powers of two make quadtree halving exact |
| Colour per brick | Mean BGR of the pixels under the triangle's mask (`cv2.fillPoly` + `cv2.mean`) | Robust to sub-region detail; the PDF asks colour be determined from the region covered |
| Palette (Task 2) | **Multi-Otsu on BT.601 luminance** (primary), K-Means K=3 and a fixed 33%/67% percentile split as baselines | Otsu thresholds adapt to the scene histogram; baselines make the results section a genuine comparison |
| Palette (Task 3) | **K-Means on CIE-Lab** (K=8/16), Median-Cut as baseline | Lab is perceptually uniform; median cut is cheap and deterministic |
| Tessellation (Task 3) | **Top-down RDO quadtree**, split priority ΔMSE per triangle (ΔSSE/6); region-merge as dual algorithm | Rate–distortion allocation: equal per-triangle units make different sizes comparable |
| Border | 1-px `#3C3C3C` polyline per triangle | Visual separation; PDF permits drawn boundaries, not counted as brick colours |
| Metrics | 9-metric suite: PSNR, SSIM, MS-SSIM, ΔE2000, Edge F1/Precision/Recall (Canny), EPI (Sobel-response Pearson), Quantisation Error, Budget Utilisation | Each metric catches a different failure mode; EPI specifically measures whether triangle boundaries align with real edges |

**Budget semantics.** The limit is 10,000 *triangles* ("bricks"); a quadtree
leaf cell produces 2 triangles, so a partition of 4,995 cells exhausts the
budget at 9,990 triangles. All Task 3 runs use 9,990 (99.9% utilisation).

---

## 3. Task 2 — equal-size triangles, 3 colours

### 3.1 Design and testing

The tessellation is a uniform SxS grid cut by checkerboard diagonals. The grid
step is the smallest S for which `2·M·N <= 10000` (M=ceil(H/S), N=ceil(W/S)).
Three colour-quantisation methods are compared on identical geometry and means:

1. **Multi-Otsu** (primary): exhaustive two-threshold search on BT.601 luma;
   each class is painted with the mean BGR of its members.
2. **K-Means** K=3 on BGR (K-Means++ init).
3. **Fixed percentile** split at 33%/67% of the luminance histogram (naive
   baseline that ignores scene statistics).

*How it was tested.* All three share one geometry (59x78, S=22, 9,204 triangles
on `sky.jpg`) so only the quantiser differs; the full 9-metric suite is
computed per method. Outputs: `code/pics/task2/out_task2_compare.png`
(original + 3 renders), `out_task2_residual.png` (per-pixel ΔE2000 heatmaps),
`out_task2_metrics_chart.png`.

### 3.2 Results

| Metric | otsu | kmeans | fixed |
|---|---|---|---|
| PSNR up (dB) | 16.15 | 16.16 | 15.56 |
| SSIM up | 0.332 | 0.334 | 0.286 |
| MS-SSIM up | 0.482 | 0.481 | 0.452 |
| ΔE2000 down | 11.41 | 11.41 | 11.92 |
| Edge F1 up | 0.191 | 0.189 | **0.233** |
| Edge Precision up | 0.151 | 0.150 | 0.171 |
| Edge Recall up | 0.261 | 0.255 | **0.365** |
| EPI up | -0.028 | -0.031 | -0.012 |
| Quant Error down | 8.47 | 8.48 | 8.98 |

Three findings:

1. **otsu ≈ kmeans.** Palettes differ by ≤2 per channel; SSIM differs by 0.0004.
   On this scene K-Means in 3-D BGR collapses to roughly the same luminance
   split as Multi-Otsu, because the dominant signal is luminance, not hue. The
   theoretical colour-resolution advantage of K-Means does not show up on a
   near-grayscale image.
2. **`fixed` wins on every edge metric yet loses on every colour/structure
   metric.** The uniform 33%/67% threshold places more original edges exactly on
   a colour boundary between adjacent triangles, preserving them as hard edges;
   Otsu/K-Means push thresholds to perceptually meaningful splits (sky vs wall)
   and "smooth over" thin edges. A genuine trade-off — perceptual fidelity vs
   raw edge preservation — that depends on what "quality" means.
3. **EPI is negative for all three (≈ -0.03).** A negative EPI means the output
   has weak edges where the original has strong ones and vice versa: the regular
   grid introduces new boundaries that do not align with image content. This is
   the quantitative signature of the uniform grid's geometric mismatch, and the
   primary motivation for Task 3's adaptive tessellation.

### 3.3 Problems found and solved

- **Grid direction-flip bug.** The first `compute_grid` scanned S *downward*
  and returned the largest S that fit, producing a 1x1 grid (2 triangles) on
  `sky.jpg`. Syntactically valid, semantically wrong: the loop must grow S
  upward and return the smallest S under budget. Fixed by inverting the scan
  (`code/brick_geom.py::compute_grid`).
- **Black band.** The floor grid dropped trailing rows/columns that did not form
  a full SxS square (right 5 px, bottom 19 px on `sky.jpg`), leaving a black
  margin that inflated PSNR/SSIM error. Fixed by making the grid `ceil` and
  reflect-padding the image to `M·S x N·S`, rendering, then cropping back
  (`code/triangle_brick.py::preprocess`). After the fix SSIM rises 0.3226 ->
  0.3316 and PSNR 15.76 -> 16.15 dB (numbers above are post-fix; pre-fix values
  are preserved in git at commit `a5d2d76`).

**GenAI use.** The tessellation and the three-quantiser design were produced
with GenAI assistance (reconstructed prompts P1-P3 in `docs/progress/Task2.md`);
the direction-flip bug was diagnosed by prompting the model to review
`compute_grid`. The AI's tendency to default to fixed thresholds is discussed
in Section 7.

---

## 4. Task 3 — adaptive multi-size, multi-colour

### 4.1 Design and testing

Building on Task 2, two adaptive pieces are added: (i) the tessellation becomes
a **top-down RDO quadtree** that starts from a coarse grid and splits the
highest-priority cell (ΔMSE/6) while budget remains, with **region merging** as
a dual bottom-up algorithm; (ii) the palette becomes **K-Means-Lab** (K>3),
with Median-Cut as baseline.

A 15-run experiment grid sweeps one variable at a time: K (4/8/16),
S_min ({2,4,8}..32), S_max ({32,64,128}), partition (quadtree/region-merge),
palette (K-Means-Lab/Median-Cut), priority (ΔMSE/Sobel-edge-density). All 15
runs reach 9,990 triangles. Full table: `code/pics/task3/summary/metrics_table.csv`.

### 4.2 Results (selected rows)

| run | SSIM up | MS-SSIM up | ΔE2000 down | Edge F1 up | EPI up | Quant Err down |
|---|---|---|---|---|---|---|
| A:k04 | **0.384** | 0.513 | 11.08 | 0.330 | **0.064** | 9.25 |
| A:k16 | 0.360 | **0.539** | **8.74** | 0.348 | 0.047 | **5.62** |
| B2:smax_32 | 0.359 | 0.520 | **9.95** | 0.295 | 0.046 | 7.30 |
| **B2:smax_64** | **0.421** | **0.555** | 10.21 | **0.377** | **0.109** | 7.41 |
| C:quadtree | 0.359 | 0.520 | 9.95 | 0.295 | 0.046 | 7.30 |
| C:region_merge | 0.332 | 0.522 | **9.59** | 0.298 | 0.019 | **6.47** |
| D:median_cut | 0.375 | 0.526 | 10.61 | 0.322 | 0.052 | 7.76 |
| E:prio_mse | 0.370 | 0.518 | 10.22 | 0.320 | 0.050 | 7.33 |
| E:prio_edgef1 | 0.374 | 0.499 | 9.83 | **0.213** | **-0.020** | 6.45 |

*(quadtree = B2:smax_32 = D:kmeans_lab = E:prio_mse are the same default run;
the full 15-row table is in `code/pics/task3/summary/metrics_table.csv`.)*

Key findings:

- **Larger K monotonically improves colour error** (ΔE2000 11.08->8.74, Quant
  9.25->5.62 from K=4 to 16) at a small SSIM cost (0.384->0.360). **K=16** is the
  best overall-fidelity choice.
- **Raising S_max from 32 to 64 is the largest unambiguous win**: SSIM
  0.359->0.421, MS-SSIM 0.520->0.555, Edge F1 0.295->0.377, EPI 0.046->0.109
  (more than doubled), with the only regression ΔE2000 (+2.6%). A coarser start
  grid leaves more budget for deep splits where detail matters. S_max=128 adds
  nothing over 64.
- **ΔMSE priority beats Sobel edge-density priority on every metric.** The
  Sobel version is the only negative-EPI run of the grid (-0.020) and has the
  worst Edge F1 (0.213): edge-density allocation over-exploits high-gradient
  pixels without aligning boundaries to real edges. A plausible-sounding
  suggestion that the measurements rejected.
- **Quadtree vs region-merge**: quadtree wins structure (SSIM, EPI) while
  region-merge wins colour (ΔE2000, Quant). Region-merge collapses to only
  {16,32} cells; quadtree keeps a {4,8,16,32} mix.

### 4.3 The marginal-benefit / rate-distortion study

Driving S_max upward raised the question *"at what size does splitting stop
paying?"* A dedicated run with a 60,000-triangle budget recorded each split's
marginal benefit (ΔSSE per new triangle) by parent size
(`code/pics/task3/analysis/marginal_benefit_curve.png`,
`benefit_by_size.png`, `cumulative_benefit_curve.png`):

| parent size | splits | mean ΔSSE/tri | share of total benefit |
|---|---|---|---|
| 64 | 484 | 345,760 | 39.3% |
| 32 | 972 | 111,064 | 25.4% |
| 16 | 1766 | 39,219 | 16.3% |
| 8 | 2964 | 16,889 | 11.8% |
| 4 | 3277 | 8,833 | 6.8% |
| 2 | 357 | 5,071 | 0.4% |

Splitting a size-64 cell is worth ~68x a size-2 split per triangle, and all
size-2 splits together contribute only 0.4% of the total error reduction. This
justified the S_max sweep (B2 group) and explains why a coarse starting grid
frees budget for the deep splits that matter.

### 4.4 S_max = 32 vs 64 — a genuine two-sided trade-off

The metric verdict "S_max=64 wins" is only one side. Breaking ΔE down by region
and by cell size shows the other side (`code/pics/task3/analysis/smax_curve.png`):

| S_max | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| ΔE2000 (lower better) | **10.61** | 11.77 | 11.79 | 11.79 |
| SSIM (higher better) | 0.375 | 0.415 | 0.415 | 0.415 |
| Edge F1 (higher better) | 0.322 | 0.403 | 0.402 | 0.402 |

The two metric families move in **opposite directions**: ΔE2000 is best at 32
then worsens and plateaus; SSIM/EdgeF1/EPI rise to 64 then plateau. There is no
single best S_max — 32 wins on large-area colour, 64 wins on structure/edges,
and the "optimum" depends on how the two are weighted. The chosen configuration
is a **balanced** one (Section 4.6). The mechanism: the RDO priority is ΔMSE,
which scales with local variance — edges get split, smooth regions are never
split, so a whole large area is painted with one colour whose per-pixel error is
small but whose *accumulated* perceptual error is large. ΔMSE cannot see
"large-area uniform drift"; the eye can.

### 4.5 Problems found and solved

- **QuadTree inverted-loop bug (AI-introduced).** The original code guarded
  `if n_tri <= budget: return` and looped `while n_tri > budget` — the wrong
  mental model: splitting *increases* the count, so the loop must run *while
  budget remains*. On `sky.jpg` the quadtree never split (4,320 of 9,990
  triangles used). Fixed to `while heap and n_tri + 6 <= budget`; 945 splits
  then reach exactly 9,990 triangles and EPI turns positive (-0.043 -> +0.049).
  Recorded as an AI-introduced bug (Section 7).
- **Region-merge candidate alignment.** Merge candidates generated at every
  half-pixel offset landed off the target-size grid and could never form the
  next level's 2x2 blocks, stalling at 8,640 merges. Fix: generate candidates
  aligned to the `target_size` grid; region_merge then reaches 9,990 triangles
  in ~1.7 s with 44,415 merges.

### 4.6 Chosen configuration

A **balanced** choice — near-best perceptual colour without throwing away
structure (`code/task3_best.py`, `code/pics/task3/best/best_config.png`,
`best_compare.png`):

| setting | value | why |
|---|---|---|
| partition | **quadtree** | keeps a {4,8,16,32} mix; near-best ΔE with much better structure than region_merge |
| S_max | **32** | best ΔE2000; 64 improves structure but worsens large-area colour |
| S_min | 1 | lets the quadtree reach fine cells where the budget allows |
| K | **16** | best ΔE2000 + quant error |
| priority | **mse** (ΔMSE / 6) | beats Sobel priority on every metric |
| palette | **kmeans_lab** | better ΔE than median_cut |

Metrics (9,990 triangles, `sky.jpg`):

| config | ΔE2000 | SSIM | PSNR | Edge F1 | EPI |
|---|---|---|---|---|---|
| **quadtree S_max=32 K=16 kmeans_lab (chosen)** | **8.95** | **0.359** | 17.56 | **0.346** | **+0.046** |
| region_merge S_max=32 K=16 kmeans_lab | **8.63** | 0.339 | 17.59 | 0.298 | +0.015 |
| region_merge S_max=32 K=16 median_cut | 9.47 | 0.338 | 17.54 | 0.336 | +0.016 |

*(Numbers from `code/pics/task3/best/*_metrics.json`, regenerated by
`task3_best.py` with `cv2.setRNGSeed(0)`. Note the chosen-config ΔE2000 here
(8.95) differs slightly from the value recorded earlier in
`docs/progress/Task3.md` §10 (9.03) — both are "the same run" under
K-Means-Lab's declared non-reproducibility; the JSON is the current
authoritative source. The conclusion is unaffected: quadtree ≈ region_merge on
colour, far better on structure.)*

Why not region_merge despite its single-lowest ΔE (8.63): it merges *all* fine
cells away — final sizes only {16,32} — so it has the worst structure of the
grid (Edge F1 0.298, EPI +0.015). The quadtree keeps a {4,8,16,32} mix and
costs only +0.4 ΔE for a much better structure score. The comparison is
reproducible: `task3_best.py` now writes per-config metrics JSON
(`code/pics/task3/best/*_metrics.json`).

---

## 5. Task 4 — real-world camera scenario

### 5.1 Requirements and strategy

Task 4 asks to run the pipeline live: capture camera frames and display their
triangle-brick representation, with the grading levels being cumulative
(camera+display = L1; different content = L2; any one of real-time /
different aspect ratios / different lighting / quantitative evaluation = L3;
analyze problems, improve, and compare before/after = L4).

We chose **real-time processing** as the L3 lever, because the L4 requirement
("analyze -> improve -> compare before/after") is exactly the work of making the
pipeline fast: one workstream satisfies both levels. The deliverable is
`code/camera_app.py` — a live loop that opens the camera (640x480, CAP_DSHOW),
runs the full pipeline per frame, and shows input + render side by side in a
resizable window with a compact HUD (keys: q quit, s snapshot, p pause, r
reset). A headless `--no-show --input <clip>` mode makes every number in this
section reproducible without a camera (`code/task4_verify.py` is the assertion
suite; `code/task4_verify_util.py` its helpers).

### 5.2 Measured baseline (before any optimization)

Full-quality config (quadtree, S_set=[1..32], K=16, kmeans_lab, budget=9990,
640x480, scale=1.0):

| Stage | median ms | mean ms | Share |
|---|---|---|---|
| camera read | ~32 | — | — |
| quadtree_partition | 434.7 | 427.4 | 20% |
| leaves_to_triangles | 13.3 | 13.5 | 1% |
| **triangle_means_bgr** | **1604.2** | **1595.3** | **75%** |
| build_palette | 24.6 | 105.0 | 1% |
| quantize_nearest_bgr | 4.5 | 4.9 | 0.2% |
| render_triangles | 58.7 | 58.4 | 3% |
| **frame total** | **2140** | 2204 | **0.45 FPS** |

`triangle_means_bgr` dominates because per triangle it zeroes a full-frame mask
and calls `cv2.mean` — cost is **O(T·H·W)**. The per-triangle cost scales with
image area (995 µs/triangle at 1279x1706, 157 µs at 640x480, 41 µs at 320x240).
Two cheap parameter levers alone reach ~9 FPS (reduce scale to 0.4 and budget to
1000, K=6), so the optimization work had to beat **9 FPS at 1000 triangles**,
not the easier 0.45 FPS. A later correction established that `build_palette` was
NOT 12x slower — a 2-frame run made a one-time ~0.5 s warm-up look like a
per-frame cost (median of two samples is their mean); with 30-frame runs the
stage is 3-25 ms/frame. Lesson: never read per-stage percentages off a run too
short for the statistic to mean anything; all numbers here come from >=10-frame
runs.

### 5.3 Level 1: camera capture + display (done)

`camera_app.py` opens the real camera, runs the full pipeline per frame and
shows a side-by-side window. The display was verified numerically
(`code/pics/task4/L1_baseline/frame0001_{input,render,compare}.png`): the render
has exactly 17 distinct colours (K=16 palette + 1 border colour), the border
`(60,60,60)` covers 72,480 px (23.6% — the single most common colour, so the
border measurably affects metrics, which is why the render path is the only
place a border is drawn), the palette spans the full dark-to-bright range with
genuine hue spread, and overall contrast is preserved (render std 66.4 vs input
68.2). A defect fixed during this stage: the HUD's FPS first read its timestamp
immediately after `cap.read()`, reporting the camera's ~33 FPS no matter how
slow the pipeline was; the timestamp is now taken after all per-frame work, so
the HUD, the headless log and the benchmark share one measurement path.

### 5.4 Level 3: making it real-time (done)

**Step 1 — mean colour.** The shipped `triangle_means_bgr` has a real defect,
not just a speed problem: `cv2.fillPoly` paints a one-pixel fringe along the
RIGHT and BOTTOM edges of every triangle (pixels whose centres belong to the
neighbouring cell), so every Task 2/3 brick's colour is contaminated by its
bottom-right neighbours — up to 400% leak on size-1 bricks, ~8% of pixels
overall. Two new implementations (`code/brick_means.py`):

| `--means` | FPS | means stage | whole frame | ΔE2000 | PSNR |
|---|---|---|---|---|---|
| mask (shipped) | 0.41 | 1716 ms | 2260 ms | 9.947 | 14.90 dB |
| **fast** (exact in-cell) | **1.35** | **20.8 ms** | **532 ms** | **8.633** | **15.33 dB** |
| sample (9 interior pts) | 1.47 | 5.2 ms | 493 ms | 10.244 | 14.87 dB |

`fast` is 82x faster on the stage, 3.3x overall, *and* closer to the source
(ΔE 9.95 -> 8.63) because it averages only in-cell pixels. `sample` is fastest
but worse than even the leaky reference on small bricks (nothing to sample
inside a 1-4 px triangle). The corrected in-cell means are what the report's
Task 2/3 numbers would give if the fringe were absent — a disclosed limitation.

**Step 2 — partition.** `quadtree_partition` recomputed a numpy region mean per
split candidate. Two fixes: drop 4 dead SSE computations per split (436 -> 326
ms), then replace numpy region means with **summed-area tables** (O(1)
rectangle SSE), batched so numpy call count drops ~160,000-fold. Isolated:

| Implementation | 2000 bricks | 5000 bricks | 9990 bricks |
|---|---|---|---|
| ref (numpy) | 80 ms | 173 ms | 326 ms |
| sat (tables, batched) | 44 ms | 69 ms | **107 ms** |
| speedup | 1.8x | 2.5x | **3.0x** |

The partition is **bit-identical** under `mse` priority (cell set and size
distribution match exactly), so no Task 3 result is invalidated. Caveat:
`edgef1` priority with the table impl uses a global Sobel map and is NOT
numerically equal to the recorded per-region-Sobel runs (median ~2x relative
difference near region edges); the recorded E_priority numbers are pinned to
`docs/progress/Task3.md` and `metrics_table.csv`, and the code prints an
explicit note when this combination is used. The conclusion (ΔMSE beats Sobel
priority) is robust under both implementations.

**Step 3 — whole-pipeline analysis.** With means and partition fixed, the frame
is ~216 ms and the bottlenecks are now: partition ~107 ms (~50%, 4,695 greedy
splits, cost is numpy-call count not arithmetic), render ~55 ms (~25%, ~20,000
Python->cv2 crossings), palette ~23 ms (~10%, per-frame K-Means), triangles ~13
ms. Four optimizations (all exact unless noted): **A** batched render (<=K
`fillPoly` calls grouped by colour + one `polylines`) 55 -> ~8 ms with **0
differing pixels**; **B** precomputed split priorities (score all ~102k
candidates up front, greedy loop pure Python) 107 -> ~30 ms, leaves bit-identical;
**C** palette refresh cadence (rebuild every 10 frames, byte-stable in between)
23 -> ~2 ms amortised; **D** vectorised `leaves_to_triangles` 13 -> ~2 ms.
Paired in-process benchmark (synthetic 640x480, reps=5):

| stage | old | new | speedup |
|---|---|---|---|
| partition | 72.3 ms | 33.8 ms | 2.14x |
| triangles | 8.3 ms | 2.2 ms | 3.79x |
| render | 28.1 ms | 8.4 ms | 3.35x |
| palette (amortised) | 22.0 ms | 4.9 ms | 4.47x |
| **whole frame** | **129.0 ms** | **59.5 ms** | **2.17x (~16.8 FPS processing-only)** |

App-level before/after (same machine, same input): TOTAL 130.5 -> 62.0 ms with
ΔE2000 11.014 == 11.014, PSNR 15.68 == 15.68, **0 pixels differ**.

**Step 4 — frame-to-frame jitter (root cause).** With the camera held still the
mosaic still flickered every frame. Headless probes measured five quantities
(geometry churn, label flips, colour noise, palette delta, canvas pixel change)
across four sequences (identical frames / noise sigma=2 / frozen geometry /
+1.5 brightness per frame). Three independent causes, quantified:

1. **K-Means instability gives a 76%-of-pixels flash every palette rebuild**
   (dominant). Same data, no re-seed: max palette delta **180** (global RNG ->
   different local optimum). Same data, re-seeded: 0. Different noisy frames,
   re-seeded: **144**. Re-seeding alone is not enough.
2. **No temporal coherence anywhere**: partition, means and quantize are
   recomputed every frame from noisy pixels, so noise moves near-tie split
   decisions (3-5% of cells per frame) and flips labels near palette boundaries
   (~3% of cells).
3. **Camera auto-exposure/white-balance drift amplifies everything**: a
   1.5/255-per-frame brightness ramp raises per-frame canvas churn from ~0.5% to
   2.9-8.1%.

**Step 5 — the fix (A+B+C).** A: **warm-start the palette** from the previous
one (refine the previous local optimum instead of jumping to a new one) + a
dead-band that discards a rebuild moving no entry by more than 2 grey levels. B:
**reuse the partition** when a keyframe signature says the scene has not changed
(static frames keep the same leaves; means are still re-extracted from live
pixels). C: **quantize hysteresis** — a per-triangle label memory keeps the
previous label unless a different palette entry wins by a relative margin.
Measured (churn = fraction of pixels whose max channel change exceeds 8 grey
levels):

| metric | before | after |
|---|---|---|
| palette delta on rebuild, identical input | **153** | **1** |
| steady per-frame churn, noise sigma=2 | **2.400%** | **0.064%** (37.7x less) |
| rebuild-frame churn, identical frames | 0.013% | **0.000%** |
| rebuild-frame churn, noise sigma=2 | 0.029% | **0.000%** |
| app static TOTAL ms (temporal OFF -> ON) | 68.9 | **27.5** |
| partition stage ms | 37.7 | **0.2** (reused) |

The 76.7%-of-pixels periodic flash is gone; continuous crawl drops ~38x on
realistic sensor noise; the partition stage disappears on a static scene.

**Step 6 — render reuse and row-run means (opt E).** Two remaining per-frame
costs on a static scene: means (~14 ms) and render (~10 ms). E1: exact
**row-run means** — each triangle half is a contiguous pixel run per row, so a
per-row prefix-sum table answers a row's run in two lookups; bit-identity is
provable (exact integer sums < 2^24, so float32 holds them exactly and addition
order cannot change the value). `task4_verify` asserts max|diff|=0: 14.2 -> 7.7
ms (1.85x). E2: **render reuse** — the canvas is a pure function of
`(tri, labels, palette, shapes)`; when geometry is reused and labels+palette are
byte-identical, the cached canvas IS the image, so rasterising is skipped (10 ->
0 ms on a static scene; under noise it correctly does not fire, because labels
change). Stacked effect on a static scene: 68.9 -> 27.5 -> **11.4 ms** (~6x),
all exact reuse, no quality cost.

**Step 7 — display panel.** Two presentation defects fixed: the HUD previously
drew 14 lines on solid black boxes covering ~35-40% x 30-35% of the render pane
(now three lines on one translucent panel, ~15% x 3-4%); and resizing a
`WINDOW_NORMAL` window stretched the bricks (fixed by uniform letterbox
scaling into the window's image area, so OpenCV's stretch becomes the identity).

**Step 8 — reuse freezes colour + camera AE/WB lock.** Even after A+B+C a static
scene showed residual colour churn (0.101% of pixels/frame) because means and
quantize re-ran every frame. Fix: when the partition is reused, also freeze the
previous labels+canvas (an unchanged scene is then byte-identical frame to
frame); and lock camera auto-exposure/white-balance (`--lock-ae`), because a
steady exposure is what makes freezing colour safe instead of stale. Measured:
churn 0.101% -> **0.000%**, static effective FPS 22.6 -> **37.1**. Author
confirmed on the real camera: jitter noticeably reduced, fidelity not worse.

### 5.5 Level 2: different content (pending author capture)

【AUTHOR: Task 4 Level 2 ("test images with different content") is not yet
captured. Plan: capture 3-5 named scenes (face, strong texture, low light,
high-contrast object, large flat region) with `camera_app.py --snapshot-dir
code/pics/task4/...`, saving input+render pairs, and state the qualitative
result per scene. The pipeline is not tuned to one image — the same config runs
on all of them. The author captures these on the real machine and pastes the
results here.]

### 5.6 Level 4: before/after summary

Every optimization above is a paired before/after (stage ms + whole-frame FPS +
ΔE/PSNR + pixel-diff), so L4's "analyze -> improve -> compare" is documented
for every change. Headline: **0.45 FPS -> ~37 FPS effective on a static scene**
(~16.8 FPS processing-only on synthetic frames) at byte-identical output on
static input, via exact reuse and algorithm changes — no approximation.

### 5.7 Problems found and solved (Task 4)

- **0.45 FPS baseline** — O(T·H·W) means and per-candidate numpy means
  (Section 5.2). Solved with in-cell means, SAT tables, batched render,
  precomputed priorities, palette cadence, vectorised triangles (5.4).
- **Fringe-contaminated means** — a latent bug in the shipped reference
  affecting Tasks 2/3 as well (5.4 step 1). Quantified and corrected in Task 4;
  disclosed for Task 2/3.
- **K-Means palette instability** — global RNG + 10 restarts on near-degenerate
  data flips local optima (5.4 step 4). Warm-start + dead-band + reuse + AE/WB
  lock (5.4 step 5/8).
- **HUD FPS misreport** and **window stretch** (5.3 / 5.4 step 7).
- **`build_palette` "anomaly"** was a 2-sample measurement artifact (5.2).

---

## 6. GenAI usage (Q4/Q5)

GenAI (PolyU GenAI / a general-purpose LLM) assisted in every GenAI-collaboration
task. The prompts below are the reconstructed drafts recorded in
`docs/progress/Task{2,3,4}.md`; the author confirms the wording. 17 prompts
across the three tasks: 14 adopted, 3 rejected/rolled-back after measurement.

### 6.1 Task 2 (3 prompts, all adopted)

| # | Prompt intent | AI output | Outcome |
|---|---|---|---|
| P1 | Design the tessellation: equal right-isosceles triangles, <=10000, 3 colours, no gaps/overlaps; propose grid step S | Square cells cut by one diagonal (2 tri/cell); S = smallest side with `2*M*N <= 10000`; alternate diagonal by checkerboard parity | Adopted; diagonal convention single-sourced in `brick_geom.py` |
| P2 | Compare three ways to pick exactly 3 colours (multi-Otsu on luma, K-Means K=3, fixed percentile) | Multi-Otsu as primary (adapts to histogram); keep the other two as comparison baselines; add `_palette_by_luminance` so labels are comparably ordered | Adopted; 9-metric suite compares them |
| P3 | First run produced a 1x1 grid; review `compute_grid` | Loop scanned S downward returning the largest fit; flip to grow upward and return the smallest S | Adopted; fixed the direction-flip bug |

### 6.2 Task 3 (6 prompts, 5 adopted / 1 rejected)

| # | Prompt intent | AI output | Outcome |
|---|---|---|---|
| P1 | Compare top-down quadtree (RDO priority) vs bottom-up region merging | Recommend quadtree with ΔMSE/6 priority; region merge as dual | Both implemented; C_algorithm measured the split |
| P2 | Palette for K>3: K-Means in CIELAB vs median cut in RGB | K-Means-Lab perceptually better; median cut cheaper/deterministic | Both measured (D_palette); AI did NOT warn K-Means is non-reproducible |
| P3 | Split by ΔMSE or by Sobel edge density? | Suggested edge-density priority | **Rejected by measurement**: lost on every metric, only negative EPI |
| P4 | Review the quadtree loop condition | Had written the wrong mental model (`<= budget: return`); splitting increases count | AI-introduced bug, fixed (Section 4.5) |
| P5 | At what size does splitting stop paying? | Marginal-benefit study: size-64 ~68x size-2; size-2 = 0.4% of benefit | Drove the B2 (S_max) group |
| P6 | `region_merge_partition` stalls at 8640 merges | Candidates generated off-grid; align to `target_size` | Adopted |

### 6.3 Task 4 (8 prompts, 6 adopted / 2 rolled back)

| # | Prompt intent | AI output | Outcome |
|---|---|---|---|
| P1 | Where does the 216 ms go? Cost model, not just a profile | `triangle_means_bgr` (O(T·H·W), 75%) + quadtree; "per-call overhead, not complexity"; v2 asks exact-vs-approximate + invariant | Framed all later work as exact-by-construction |
| P2 | Render: can it be O(K) calls, same output? | <=K fillPoly + one polylines; predicted fringe would differ | Adopted; 0 differing pixels (better than predicted) |
| P3 | Precompute split priorities, identical leaves? | First version 0.56x slower (mapped dead size-1 cells); fix = exact-integer hierarchical sum | Adopted after a wrong turn; 2.16x, leaves bit-identical |
| P4 | Palette cache / vectorise triangles / render reuse | Warm-start + cadence; one (T,3,2) array; skip rasterise when pure-function cache hits | Adopted (C/D/E) |
| P5 | Jitter root cause (the hard one) | Two causes: K-Means local-optimum instability + whole-frame recompute; then warm-start + dead-band + reuse gate | Adopted; palette delta 153 -> 1, churn 2.40 -> 0.06% |
| P6 | Isolate per region (freeze geometry, per-cell change) | Implemented; removed jitter but quality dropped | **Rolled back** after ablation: palette freeze (not geometry) was catastrophic (PSNR 13.9 -> 9.9 on dark first frame) |
| P7 | On reuse, keep previous labels/canvas? What must be true? | Freeze colour on reuse, safe only if exposure steady, so lock AE/WB | Adopted; churn 0.101 -> 0.000%, FPS 22.6 -> 37.1 |
| P8 | Dense small bricks are all border; propose options | Six variants + vision review preferring the combo | **Rejected by the author**: original black border preferred |

### 6.4 How GenAI understood the tasks (Q5)

Three patterns were the most instructive about how the model "thinks":

- **Wrong mental model, correct-looking code (Task 3 P4).** The quadtree guard
  encoded "split down to budget" — the opposite of the truth. The code ran
  without error; only a careful review of the *intent* (splitting increases the
  count) exposed it. This is a reliability lesson about trusting AI code that
  merely executes.
- **"Per-call overhead, not complexity" (Task 4 P1).** The AI correctly
  diagnosed that the cost was Python->numpy call count, not algorithmic
  complexity, and its "exact or approximate? state the invariant" follow-up
  framing is what kept every optimization exact-by-construction.
- **The same bias that flatters the user (Task 3).** When asked to pick
  S_max, the AI reported "64 is the clear winner" by leading with the four
  metrics where 64 won and demoting the one regression (ΔE2000) to a
  parenthetical — and only after the author pushed back did it break ΔE down by
  region and find the shadow area was ~44% worse under 64. Objective metrics
  and human perception can invert; the AI did not volunteer that.

---

## 7. GenAI limitations and areas for improvement (Q6)

【AUTHOR: This section is yours to write (AGENTS §7/§8 — the marker explicitly
reserves the GenAI-limitations write-up for the author; Task 5 awards up to 10
marks for "result analysis and reflection, including GenAI limitations").
Raw material is already recorded in:
- `docs/progress/Task2.md` end bullets (fixed thresholds fail under uneven
  lighting: -0.05 SSIM; AI ignores real-time; AI rarely addresses frame
  stability),
- `docs/progress/Task3.md` §8-§9 (inverted-loop bug; the three reporting biases:
  agreeing with an unverified thesis, tilting data toward the thesis, reporting
  metric verdicts as perceptual verdicts — the S_max=32-vs-64 episode),
- `docs/progress/Task4.md` P6/P8 (the per-region rollback and the border attempt
  that the human eye overruled), and the "third error of the same family"
  note.
Suggested structure: (1) reliability of AI-generated code (the two bugs);
(2) reliability of AI as an analyst (the three biases); (3) what was improved
and what remains (K-Means reproducibility, real-camera parameter validation,
per-cell foreground isolation).]

---

## 8. Conclusion

We built a triangle-brick image-generation system that converts an image into
right-isosceles-triangle mosaics under a 10,000-brick budget, and moved it
through the four assignment tasks.

- **Task 2** establishes the tessellation and shows, with a 9-metric suite, that
  the colour-quantisation method matters less than the geometry: on this
  luminance-dominated scene Otsu ≈ K-Means, and the "naive" fixed threshold
  wins edge metrics while losing colour metrics. The uniform grid's negative
  EPI quantifies its geometric mismatch with content.
- **Task 3** makes the geometry adaptive (RDO quadtree) and the palette
  multi-colour (K-Means-Lab), and a 15-run sweep plus a rate-distortion study
  shows where the budget is best spent: K=16, S_max=32 (balanced), ΔMSE
  priority, quadtree. The S_max=32-vs-64 result is presented as a genuine
  two-sided trade-off, not a single winner.
- **Task 4** turns the pipeline into a live camera loop and solves the
  real-world problems the offline tasks never exercised: an O(T·H·W) mean stage
  (75% of the frame), an AI-introduced quadtree bug, a latent fringe defect,
  frame-to-frame jitter from unstable K-Means and camera exposure drift, and a
  misleading HUD. Each was measured before/after; a static scene runs ~37 FPS
  effective with byte-identical output via exact reuse and algorithm changes.
- The pipeline's "converging" design — shared geometry/colour/rendering core,
  one decision layer per task — is what allowed the Task 4 optimizations to
  stay exact and the Task 2/3 results to remain reproducible.

The main limitations, disclosed throughout: single test image (Task 2/3),
K-Means non-reproducibility (declared), single-machine measurements (Task 4),
and Task 4's Level-2 different-content testing pending the author's camera
captures. 【AUTHOR: add the AI-limitations synthesis here or keep it in
Section 7.】
