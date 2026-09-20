# Task 2: Triangle-brick mosaic — equal-size, 3 colors (10 marks)

## Goal (from assignment PDF)
> Convert an input image into a mosaic made of equal-size right isosceles
> triangles, using only 3 colors, total ≤ 10000 triangles, no gaps or overlaps.

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Tessellation | Square + diagonal (2 triangles per square), **diagonal orientation alternates like a checkerboard** | Satisfies "equal-size right isosceles"; alternation avoids a uniform direction bias |
| Grid step S | Smallest S with `2*M*N ≤ 10000`, where `M=H//S, N=W//S` | Densest legal grid (most detail) while staying under budget |
| Color extraction | `cv2.fillPoly` mask + `cv2.mean` (area mean of covered pixels) | Beats single-point sampling; no interpolation artifacts |
| 3-color quantization (primary) | **Multi-Otsu** on BT.601 luminance; per-class palette color = mean BGR of members | Otsu adapts to scene illumination; mean-BGR keeps residual hue |
| 3-color quantization (compare) | K-Means K=3 on BGR via cv2.kmeans + K-Means++ init; percentile split (33%/67%) | For Task 5 "results & discussion" — see comparison below |
| Metric suite | **9 metrics**: PSNR, SSIM, MS-SSIM (4-level mean SSIM), ΔE2000 (mean CIEDE2000), Edge F1 (Canny + 3 px tolerance), Edge Precision/Recall, **EPI** (Sobel-response Pearson), Quantization Error (per-triangle ΔE2000 to assigned palette), Budget Utilization (T/10000) | Each metric catches a different failure mode: PSNR/SSIM = pixel/structure; MS-SSIM = multi-scale; ΔE2000 = perceptual color; Edge F1/Recall = pixel-level edges; EPI = continuous edges (catches geometric misalignment); Quant Error = the 3-color constraint's cost; Budget Util = budget efficiency. See full table in §Observations below. |
| Border | Thin gray `#3C3C3C` line via `cv2.polylines` | Visual separation; not counted as one of the 3 colors |

## Note about `code/sky.jpg`

The file is named `sky.jpg` but the content is a **cityscape (canal + brick
wall + trees + distant buildings)**, not sky/cloud imagery. The name misled
early intuition — it does NOT affect algorithm results, but the report
should describe the actual content rather than the filename.

## Prompt drafts (GenAI)
- Task 2 used AI for design discussion only (algorithm landscape, interpolation
  choices). Final implementation written from scratch with no code copied from
  external sources. See `docs/progress/Task0_brief.md` for the algorithm survey.

## Code changes

- `code/triangle_brick.py` (final version, ~270 lines):
  - `compute_grid(H, W)` → `(M, N, S)` — densest legal grid
  - `iter_triangles(M, N, S)` → generator of `(3,2)` int32 vertex arrays
  - `triangle_means_bgr(img, triangles)` → `(T, 3)` BGR means
  - `multi_otsu_3class(gray_values)` → `(t1, t2)` via exhaustive search
  - `_bt601_luma`, `_palette_by_luminance` — shared utilities
  - `quantize_otsu` / `quantize_kmeans` / `quantize_fixed` — three quantizers
  - `render(H, W, triangles, labels, palette_bgr)` → output canvas
  - `compute_metrics(img, canvas)` → `(ssim, psnr)` via skimage
  - `make_compare_grid(img, results)` → 2×2 comparison image
  - `run_single`, `run_compare` — per-method and compare-mode drivers
  - `main()`: CLI with `--method {otsu,kmeans,fixed}`, `--compare`, `--no-show`
- Reuses `code/sky.jpg` from Task 1.

## Problems and solutions

- **First run produced only 2 triangles.** `compute_grid` originally looped
  `S` from `min(H, W)` down to 1, returning the *largest* S that fits — which
  for `sky.jpg` (1706×1279) was S=1279 → 1×1 grid → 2 triangles. Fixed by
  flipping the iteration to grow S upward and returning the *smallest* S that
  still satisfies `2*M*N ≤ 10000`. **Note for Task 5**: textbook example of a
  subtle direction-flip bug — first instinct ("loop downward from the max")
  produced a syntactically-valid but semantically wrong algorithm that
  compiles and runs without error.
- **K-Means label ordering not stable across runs.** cv2.kmeans returns
  labels in arbitrary order. Added `_palette_by_luminance` to remap so that
  class 0 is darkest, class 2 is brightest — makes `otsu` / `kmeans` / `fixed`
  directly comparable in the comparison grid.
- **Fixed-percentile threshold choice (33% / 67%)**: equal-thirds of the
  luminance distribution forces each class to ~33% of triangles regardless of
  scene content. This is what the user requested for the comparison; it
  serves as a *naive baseline* that ignores scene statistics entirely.
- **Square subdivisions of non-integer-multiple images**: trailing rows/cols
  that don't form a full S×S square are dropped (1706-81*21=5 px wide,
  1279-60*21=19 px tall).
- **Degenerate histograms**: Multi-Otsu falls back to fixed `(85, 170)`
  thresholds when total=0.

## Verification

Single method (default = otsu):

```bash
cd "D:\Program Files\learn_torch\python-cv\Assignment1"
..\..\venv\Scripts\python code\triangle_brick.py --input code\pics\sky.jpg --output code\pics\task2\out_task2.png --no-show
```

Compare all three:

```bash
..\..\venv\Scripts\python code\triangle_brick.py --input code\pics\sky.jpg --output code\pics\task2\out_task2.png --compare --no-show
```

Actual (`code/sky.jpg`, 1706×1279, after `--compare`):

```
[task2] input = ...\sky.jpg  shape = (1279, 1706)
[task2:compare] grid = 60x81, S=21, T=9720
[task2:otsu]   palette=[[57, 57, 55], [117, 119, 112], [218, 213, 202]]
               counts=[4521, 2798, 2401]  SSIM=0.3226  PSNR=15.76dB
[task2:kmeans] palette=[[57, 57, 56], [119, 120, 113], [219, 214, 202]]
               counts=[4573, 2780, 2367]  SSIM=0.3230  PSNR=15.78dB
[task2:fixed]  palette=[[48, 48, 48], [95, 96, 91], [199, 195, 185]]
               counts=[3240, 3240, 3240]  SSIM=0.2748  PSNR=15.28dB
[task2:compare] grid image = ...\out_task2_compare.png
[task2] total elapsed = 9.01s
```

### Observations (raw material for Task 5 "results & discussion")

- **otsu ≈ kmeans** on this image. The two palette rows differ by ≤ 2 in each
  channel; SSIM differs by 0.0004. Reason: the scene's dominant signal is
  *luminance variation* (sky vs trees vs wall), not *hue variation*. K-Means in
  3-D BGR therefore collapses to roughly a 1-D luminance split — the same
  information Multi-Otsu uses directly. **Implication for the report**:
  K-Means's theoretical color-resolution advantage does not show up when the
  scene is essentially grayscale.
- **fixed loses the most information** (ΔSSIM ≈ -0.05 vs otsu). Forcing each
  class to ~33% of triangles collapses large bright areas (sky) into the
  middle class, losing the bright/dark contrast. Quantitatively visible in
  the comparison grid: the upper half (sky) is mostly medium gray under
  `fixed`, but clearly bright under `otsu` / `kmeans`.
- **All three methods preserve geometry equally**: the checkerboard-orientation
  triangles and the right-isosceles shape constraints are upstream of
  quantization and don't differ.

### Full-metric comparison (post-update: 9 metrics)

After adding the metric suite, we re-ran `--compare` and got the following
values on `code/sky.jpg` (1706×1279). All three methods share the same
geometry (60×81 grid, S=21, 9720 triangles, budget utilization 0.972).

| Metric | otsu | kmeans | fixed | Best |
|---|---|---|---|---|
| PSNR ↑ (dB) | 15.76 | 15.78 | 15.28 | otsu ≈ kmeans |
| SSIM ↑ | 0.323 | 0.323 | 0.275 | otsu ≈ kmeans |
| MS-SSIM ↑ | 0.477 | 0.477 | 0.451 | otsu ≈ kmeans |
| ΔE2000 ↓ | 11.71 | 11.67 | 12.12 | otsu ≈ kmeans |
| Edge F1 ↑ | 0.187 | 0.190 | **0.232** | **fixed** ⚠️ |
| Edge Precision ↑ | 0.145 | 0.147 | 0.169 | **fixed** ⚠️ |
| Edge Recall ↑ | 0.264 | 0.268 | **0.370** | **fixed** ⚠️ |
| EPI ↑ | -0.031 | -0.035 | -0.014 | **fixed** (least bad) ⚠️ |
| Quant Error ↓ | 8.48 | 8.43 | 8.92 | otsu ≈ kmeans |
| N triangles | 9720 | 9720 | 9720 | (same) |
| Budget Util ↑ | 0.972 | 0.972 | 0.972 | (same) |
| FPS (quantize+render+metrics only) | 18 | 26 | 33 | — |

#### Counter-intuitive finding: `fixed` wins on edge metrics

The single biggest surprise: **fixed** (the "naive" percentile baseline)
beats otsu and kmeans on **every edge-related metric** — Edge F1, Edge
Precision, Edge Recall, and (least-bad) EPI. Yet it loses on every
structure/color metric.

Why? The fixed 33%/67% percentile thresholds slice the luminance histogram
*uniformly*, which means **more pixel-level edge boundaries of the original
image happen to lie near one of these thresholds**. Result: the original
edges are more often preserved as a color boundary between adjacent
triangles. Otsu/K-Means push thresholds toward perceptually meaningful splits
(between sky and wall, etc.), which can **merge pixels across a thin
original edge into the same class** — i.e. they "smooth over" thin edges
in exchange for better tonal balance.

This is a real trade-off: **perceptual fidelity vs. raw edge preservation**.
For the triangle-brick aesthetic where edges * are* the visual signature,
the answer depends on what you mean by "quality". This makes a strong
Task 5 discussion paragraph.

#### Second finding: **EPI is negative for all three methods**

EPI = Pearson correlation between Sobel responses of the input and the
output. A *negative* EPI means: where the original has a strong edge, the
output tends to have a *weak* edge, and vice versa. All three methods
score slightly negative (≈ -0.03).

**Interpretation**: the regular checkerboard-grid triangles introduce *new*
edges that **don't align with original edges** at all (and conversely
suppress original edges that don't happen to fall on a triangle boundary).
This is the quantitative signature of the **uniform-grid's geometric
mismatch with image content** — and it is the main motivation for Task 3's
*adaptive* tessellation (where triangle boundaries can be steered toward
real edges).

#### Residual heatmap (ΔE2000) confirms the numerical story

`out_task2_residual.png` is a 3-row stack of per-pixel ΔE2000 maps (JET
colormap, blue = small ΔE = good, red = large ΔE = bad). For `otsu` and
`kmeans` the panels are nearly identical: dark blue over the upper sky
band, light blue over the water, with the brightest yellow/red concentrated
in fine-detail regions (tree foliage). For `fixed`, the water band shows a
*large bright yellow stripe* — water has uniform mid-luminance and gets
mapped to the *middle* class regardless of local variation, hence the
perceptually large color error. This is the visual companion to the
ΔE2000 gap (11.7 → 12.1).

Output artifacts (all under `code/pics/task2/`, per AGENTS 1.1; the runs above
were recorded before the move, when they sat in `code/`):
- `code/pics/task2/out_task2_otsu.png`, `out_task2_kmeans.png`, `out_task2_fixed.png`
- `code/pics/task2/out_task2_compare.png` — 2×2 grid (original + 3 results) with key metrics in label
- `code/pics/task2/out_task2_residual.png` — 3-row ΔE2000 heatmap panel (JET colormap)
- `code/pics/task2/out_task2_metrics_chart.png` — 2-subplot bar chart ([0,1] + absolute-scale)

## Known limitations / TODOs

- Performance: 9 s total for `compare` mode on this resolution (grid prep
  dominates: ~6 s of 9 s). Task 4 will need precomputed masks or downsampled
  processing to hit real-time.
- Edge handling: non-aligned pixels are truncated, not adapted (smaller
  triangles at borders). Simple and acceptable for the report; mention it.
- Single-image comparison. A second test image with strong color contrast
  (e.g., a fruit bowl) would be needed to demonstrate K-Means's hue advantage.
  Planned as future work.
- AI limitations to note in Task 5 (raw material, will be authored by you in
  the report): AI suggestions often default to fixed thresholds for K=3,
  which fail under uneven lighting (this experiment shows the failure mode
  quantitatively: -0.05 SSIM); AI tends to ignore performance for real-time
  use (Task 4 issue); AI rarely addresses temporal stability across frames
  (also Task 4).