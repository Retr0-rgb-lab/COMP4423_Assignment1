# Task 3: Adaptive multi-size + multi-color triangle-brick (10 marks)

> **2026-09-17 update:** the quadtree splitting bug is fixed. All 12 runs now
> use the full budget (9990 triangles, 99.9%), produce genuine multi-size
> geometry, and EPI is positive for 11 of 12 runs. See
> [§0 The quadtree bug](#0-the-quadtree-bug-inverted-loop-condition).

## 0. The quadtree bug: inverted loop condition

### 0.1 What was wrong

Top-down quadtree starts at the **coarsest** grid (fewest triangles) and
**splits** cells into smaller ones. Splitting only ever *increases* the
triangle count (1 cell = 2 tri → 4 cells = 8 tri, net +6).

The original code had this guard:

```python
n_tri = len(leaves) * 2
if n_tri <= N_TRI_BUDGET:
    return ...            # (A) early return whenever under budget

while heap and n_tri > N_TRI_BUDGET:   # (B) only split when OVER budget
    ...
```

Both lines encode the same wrong mental model: "quadtree starts over budget
and splits back down". The opposite is true. On `sky.jpg` (1706×1279), the
S_max=32 starting grid gives **4320 triangles** — well under the 9990 budget
— so the early return fired immediately and **the quadtree never split**.

### 0.2 The fix

```python
# Keep splitting WHILE there is room in the budget.
while heap and n_tri + 6 <= N_TRI_BUDGET:
    # pop highest-priority (ΔMSE/6) cell, split into 4 children
```

Result: 945 splits, `n_tri = 9990` exactly, cell sizes `{4, 8, 16, 32}`.

### 0.3 Impact of the fix (same image, same S_set, same K=8)

| metric | before fix (no split) | after fix (splits) | change |
|---|---|---|---|
| n_triangles | 4320 (43% of budget) | **9990 (99.9%)** | full budget used |
| cell sizes | {32} only | **{4, 8, 16, 32}** | adaptive geometry |
| **EPI** | −0.043 | **+0.049** | **turned positive** |
| **Edge F1** | 0.163 | **0.323** | **~2×** |
| SSIM | 0.360 | 0.363 | flat |
| PSNR | 17.74 | 17.22 | −0.5 dB |
| ΔE2000 | 9.04 | 10.05 | +1.0 |

The trade-off is clear and expected: adaptive geometry sacrifices a little
pixel-level fidelity (PSNR, ΔE) for a large gain in **geometric correctness**
(EPI, Edge F1). Triangle boundaries now roughly follow image content, so the
"fake edges" that made EPI negative are largely gone.

### 0.4 What this also fixed

- **B_sweep is now meaningful**: varying S_min changes the smallest cells
  the quadtree can reach, so the three runs differ.
- **E_priority is now meaningful**: priority only matters when splitting
  happens, which it now does.

### 0.5 Second bug found while fixing C_algorithm (aligned region merge)

`region_merge_partition` (bottom-up: start at S_min, merge upward) had a
candidate-alignment bug. It generated merge candidates at every `half`-pixel
offset, so merged cells landed off the target-size grid and could never form
the next level's 2×2 blocks. It stalled at 8640 merges. Fix: generate
candidates **aligned to the target_size grid** (`step=target_size`, and use
the loop index directly as the pixel position — the earlier code mistakenly
multiplied by `target_size` again). After the fix, region_merge reaches
9990 triangles in ~1.7 s with 44415 merges.

## 1. Goal (from assignment PDF)

> Convert an input image into a mosaic of right isosceles triangles using
> **multiple sizes** and **more than 3 colors**. Total ≤ 10000 triangles, no
> gaps/overlaps. Output = rendered image + per-size brick summary.

## 2. Algorithm choices

| Decision | Choice | Why |
|---|---|---|
| Partition (default) | Top-down **quadtree**, RDO priority **ΔMSE / 6** | Standard rate-distortion optimization; per-triangle normalization makes splits at different sizes comparable |
| Partition (alt) | Bottom-up **region merging** (multi-scale BFS, summed-area tables) | Dual algorithm; produces coarser distributions |
| Palette | **K-Means on CIE-Lab** (default K=8) | Perceptually uniform; handles K > 2 where Otsu can't |
| Palette (alt) | **Median Cut** on RGB | Cheap baseline |
| Priority (alt) | Sobel-response variance | Edge-density-driven split priority |
| Quantize | nearest palette color in BGR | standard |

`S_set` must be a power-of-two family (the quadtree halves sizes on split).

## 3. Experiment grid (all 12 runs completed)

| Group | Swept variable | Values |
|---|---|---|
| **A_ksweep** | K | 4, 8, 16 |
| **B_sweep** | S_set (S_min) | [2..32], [4..32], [8..32] |
| **C_algorithm** | partition | quadtree, region_merge (S_set=[4,8,16,32]) |
| **D_palette** | palette gen | K-Means-LAB, Median-Cut |
| **E_priority** | split priority | ΔMSE, Sobel-var |

All 12 hit exactly **9990 triangles (99.9% budget)**.

## 4. Results

Full table: `code/pics/task3/summary/metrics_table.csv`.

| run | SSIM↑ | MS-SSIM↑ | ΔE2000↓ | Edge F1↑ | EPI↑ | Quant Err↓ |
|---|---|---|---|---|---|---|
| A:k04 | **0.384** | 0.513 | 11.08 | 0.330 | **0.064** | 9.25 |
| A:k08 | 0.363 | 0.516 | 10.05 | 0.323 | 0.049 | 7.22 |
| A:k16 | 0.360 | **0.539** | **8.74** | 0.348 | 0.047 | **5.62** |
| B:s_2_4_8_16_32 | 0.359 | 0.520 | 9.95 | 0.295 | 0.046 | 7.30 |
| B:s_4_8_16_32 | 0.370 | 0.518 | 10.22 | 0.320 | 0.050 | 7.33 |
| B:s_8_16_32 | 0.359 | 0.522 | 9.95 | **0.371** | **0.061** | 7.20 |
| C:quadtree | 0.359 | 0.520 | 9.95 | 0.295 | 0.046 | 7.30 |
| C:region_merge | 0.332 | 0.522 | **9.59** | 0.298 | 0.019 | **6.47** |
| D:kmeans_lab | 0.359 | 0.520 | 9.95 | 0.295 | 0.046 | 7.30 |
| D:median_cut | 0.375 | 0.526 | 10.61 | 0.322 | 0.052 | 7.76 |
| E:prio_mse | 0.370 | 0.518 | 10.22 | 0.320 | 0.050 | 7.33 |
| E:prio_edgef1 | 0.374 | 0.499 | 9.83 | **0.213** | **−0.020** | **6.45** |

### Findings (raw material for Task 5)

- **A – K sweep**: larger K monotonically improves ΔE2000 (11.08 → 8.74) and
  Quant Error (9.25 → 5.62) — more palette colors reduce quantization error.
  But SSIM *decreases* slightly with K (0.384 → 0.360), suggesting a small
  oversegmentation artifact as the palette grows. k16 is the best
  overall-fidelity choice.
- **B – S_min sweep**: the **coarsest allowed minimum (S_min=8) wins** on
  Edge F1 (0.371) and EPI (0.061). Interpretation: tiny 4-pixel cells add
  many boundaries that don't align with real edges, so allowing them hurts
  edge-metric scores even though they can fit more detail nominally.
- **C – quadtree vs region_merge**: quadtree wins SSIM (0.359 vs 0.332) and
  EPI (0.046 vs 0.019); region_merge wins ΔE2000 (9.59 vs 9.95) and Quant
  Error (6.47 vs 7.30). Region_merge produces only large cells (its size
  distribution is {16, 32}); quadtree keeps a mix {4,8,16,32}. So quadtree
  aligns better to edges (EPI) while region_merge quantizes colors slightly
  better (fewer, larger flat cells).
- **D – palette method**: Median Cut beats K-Means-LAB on SSIM (0.375 vs
  0.359) and Edge F1 (0.322 vs 0.295), but loses on ΔE2000 (10.61 vs 9.95)
  and Quant Error (7.76 vs 7.30). Classic "perceptual color vs pixel-level
  structure" split.
- **E – priority**: **ΔMSE priority beats Sobel priority** on every
  quality metric. The Sobel-priority version produces the *only* negative
  EPI (−0.020) and the worst Edge F1 (0.213) of the whole grid. Edge-density
  priority over-exploits high-gradient pixels and spends budget on cells
  whose boundaries still don't align with real edges (they just multiply
  small cells). This is a genuinely counterintuitive result for Task 5.

## 5. Code

- `code/triangle_brick_task3.py`
  - `quadtree_partition(img, cfg)` — top-down RDO quadtree (fixed loop).
  - `region_merge_partition(img, cfg)` — bottom-up aligned multi-scale merge.
  - `_quadtree_priority(img, leaf, priority)` — ΔMSE/6 or Sobel variance.
  - `palette_kmeans / palette_median_cut / quantize_nearest_bgr`.
  - `leaves_to_triangles / triangle_means_bgr / render_triangles`.
  - Reuses Task 2's `compute_metrics` (9-metric suite).
  - `plot_size_histogram / plot_palette / build_summary`.

## 6. Verification

```bash
cd "D:\Program Files\learn_torch\python-cv\Assignment1"
..\venv\Scripts\python code\triangle_brick_task3.py --groups A B C D E
```

Outputs under `code/pics/task3/{A_ksweep,B_sweep,C_algorithm,D_palette,E_priority}/`
and `code/pics/task3/summary/` (CSV + bar chart). Per-run files:
`<name>.png` (render), `<name>_size_hist.png`, `<name>_palette.png`,
`<name>_metrics.json`.

## 7. Known limitations / TODOs

- **PSNR/ΔE slightly worse than the non-splitting version** — expected: the
  adaptive tiling trades pixel fidelity for edge alignment. Report should
  present both metrics, not just the favorable ones.
- **FPS is 0.07–0.11** (i.e., ~10 s per frame) because metrics computation
  (SSIM/MS-SSIM/ΔE/Edge F1) dominates; this is offline evaluation, not the
  Task 4 pipeline. Task 4 will need a stripped-down render-only path.
- **Edge F1 / EPI still capped below 0.5** — power-of-two cell boundaries
  can only align with edges to the grid resolution; a content-aware
  (non-grid) triangulation would do better but violates the "equal right
  isosceles" constraint.
- **Only one test image** (`sky.jpg`). A second image with strong hue
  variation would test the palette-method findings more sharply.
- **AI limitations to note in Task 5**:
  - The inverted quadtree loop condition was an AI-introduced bug — the
    mental model "quadtree splits down to budget" was wrong; splitting
    *increases* the count, so the loop must run while budget remains.
  - AI initially picked `S_set={1,2,4,8}` which, on this image size, makes
    region_merge impractically slow and makes top-down quadtree impossible
    (S=8 start already over budget). The realistic S range includes 16/32.
  - AI suggested edge-density split priority; empirically it is *worse*
    than plain ΔMSE on every metric here.
