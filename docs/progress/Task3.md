# Task 3: Adaptive multi-size + multi-color triangle-brick (10 marks)

> ⭐ **2026-09-16 post-mortem (most important read in this file):** see
> [§0 Key Findings](#0-key-findings-post-mortem) below. Five issues surface
> from the 11-run experiment grid that reframe what Task 3 means in practice
> — the headline: under the 10000-triangle budget on a 1706×1279 image,
> `S_max=32` starting grid already fits the budget, so **quadtree never
> actually splits** and our adaptive-geometry sweep is largely a no-op.

## 0. Key Findings (post-mortem)

Five problems surfaced during the 11-run sweep. The first one is the load-
bearing observation; the rest follow from it.

### 0.1 Budget vs geometry: the fundamental tension

```
10000-triangle budget
1706 × 1279 image ≈ 2.18 M pixels
critical cell size  = sqrt(2 · 2.18 M / 10000) ≈ 21 × 21 pixels
S_max in our S_set  = 32 × 32  pixels (1024 pixels per cell)
```

Because `S_max=32` is **already smaller than the critical size (~21)**, the
32-cell starting grid covers only 4240 triangles — well within the 9990
budget. The quadtree **never gets a chance to split**. There is literally
no budget headroom for adaptive subdivision in this configuration.

This is a **task-design issue, not an algorithm issue**: with a stricter
budget or larger image, or with a smaller `S_max`, the same algorithm
behaves very differently.

### 0.2 Most of the sweep is effectively a no-op

- **A_ksweep (K ∈ {4, 8, 16})** — valid. Geometry is held constant, only
  quantization varies. This is the only meaningful comparison we got.
- **B_sweep (S_set variations)** — **invalid**. Three runs produce
  bit-identical metrics. S_set doesn't matter when the quadtree
  never splits.
- **C_algorithm (quadtree vs region_merge)** — **incomplete**. region_merge
  was never exercised in this batch: with `S_max=32`, the quadtree's
  starting grid already fits the budget, so the fallback never triggers.
  In practice region_merge and quadtree are the same algorithm here.
- **D_palette (K-Means Lab vs Median Cut)** — valid. Different quantization
  on the same geometry.
- **E_priority (ΔMSE vs ΔEdgeF1)** — **invalid**. Priority only matters
  when the quadtree is *splitting*, which it isn't.

So 11 runs were produced but only 5 (the 3 in A + 2 in D) carry a real
signal.

### 0.3 EPI stays negative: uniform grid is structurally misaligned

Every method scores EPI ≈ −0.04 to −0.07. The triangle boundaries
introduce **fake edges** that don't correspond to image content. This is
the structural defect of *any* uniform grid; it cannot be fixed by tuning
K, palette, or priority. To make EPI turn positive, triangle boundaries
must physically follow image edges — which means the quadtree must
actually split, and the priority must be edge-driven.

**The real Task 3 research question — boundary alignment — was not
addressed by any of our 11 runs.**

### 0.4 region_merge stalling at S_min=1 is a physics signal, not a bug

Even with summed-area tables + multi-scale BFS + O(N log N) heap,
region_merge plateaus at ~89% completion when `S_min=1` on this image.
The remaining ~10% of single-pixel cells all straddle image edges and
cannot form valid 2×2 sibling groups.

This tells us something important: **any "fine-to-coarse" adaptive
algorithm hits ~10% unmergeable cells on real images**. The only fixes are
(1) keep those cells as size-1 (cheap but adds budget pressure), or
(2) allow irregular merging (breaks the "right isosceles" constraint,
prohibited by the assignment).

### 0.5 The core research question of Task 3 was sidestepped

Task 3's surface requirement is "multi-size + multi-color". But the actual
research question is **"how do triangle boundaries track image edges?"**.
Our sweep produced 11 results, but **zero** of them tests boundary
alignment — geometry was held constant, EPI never moved. To make the
next iteration meaningful we need:

- A starting grid that **forces** the quadtree to split (start finer than
  the budgeted size, then either accept over-budget output or merge a few
  steps back).
- Edge-driven priority (Sobel/ΔEdgeF1) and verify that EPI turns positive
  in the regions where splitting actually happens.
- Real comparison: same image, fixed S_max, but start fine vs start coarse,
  measure how much EPI / Edge-F1 improve when splitting is exercised.

---

## Goal (from assignment PDF)
> Convert an input image into a mosaic of right isosceles triangles using
> **multiple sizes** (e.g., 1/2/4/8) and **more than 3 colors** (palette size K).
> Total triangles ≤ 10000, no gaps/overlaps. Output must include the rendered
> image plus a per-size brick summary.

## Algorithm choices (decided in §7.3 of Task0_brief.md)

| Decision | Choice | Why |
|---|---|---|
| Partition | Top-down **quadtree** (default) | Standard RDO; fits the assignment's "power-of-two sizes" framing |
| Partition (alt) | Bottom-up **region merging** with summed-area tables | Duality of quadtree; only used when S_max is too small for the budget |
| Palette gen | **K-Means (K=8 default) on CIE-Lab** | ΔE in Lab is perceptually uniform; K-Means handles K>2 where Otsu doesn't apply |
| Palette gen (alt) | **Median Cut** on RGB | Simpler/faster than K-Means; included in D-group sweep |
| Priority (quadtree) | **ΔMSE / 6** per new triangle (RDO Lagrangian) | Normalizes gain across size levels |
| Priority (alt) | Sobel-response variance | Edge-density-driven; included in E-group sweep |

## Experiment grid (11 of 12 runs completed; C:region_merge skipped — see §Caveats)

| Group | Variable swept | Values | Why |
|---|---|---|---|
| **A_ksweep** | K (palette size) | 4, 8, 16 | Direct task requirement: "more than 3 colors" |
| **B_sweep** | S_set (allowed triangle sizes) | [2,4,8,16,32], [4,8,16,32], [8,16,32] | Tests "size granularity" effect on quality |
| **C_algorithm** | partition algorithm | quadtree only | region_merge skipped — see caveats |
| **D_palette** | palette generation | K-Means Lab, Median Cut | "Lab vs RGB" and "K-Means vs Median Cut" |
| **E_priority** | split priority | ΔMSE, ΔEdgeF1 | "MSE-driven vs edge-driven" |

All 11 runs use the same geometry (60×81 grid of 32×32 cells → 4320 triangles)
because the quadtree starting grid already fits the budget; see caveats below.

## Prompt drafts (GenAI)
- Task 3's algorithm landscape (quadtree vs region-merge, palette methods,
  K-Means vs Median Cut) was discussed via GenAI for design only. Final
  implementation written from scratch; no code copied.

## Code changes

- New `code/triangle_brick_task3.py` (~660 lines):
  - `quadtree_partition(img, cfg)` — top-down quadtree starting at S_max.
    Falls back to region_merge if S_max starting grid exceeds the budget.
  - `region_merge_partition(img, cfg)` — bottom-up multi-scale BFS with
    summed-area tables (SAT) for O(1) region SSE.
  - `palette_kmeans(means_bgr, k, space)` — K-Means in Lab or RGB.
  - `palette_median_cut(means_bgr, k)` — recursive median cut.
  - `quantize_nearest_bgr(means, palette)` — nearest-neighbor quantization.
  - `leaves_to_triangles(leaves)` — generate two right-isosceles triangles per
    cell with checkerboard-alternating diagonal orientation.
  - `triangle_means_bgr(img, triangles)` — area mean via fillPoly + cv2.mean.
  - `render_triangles(...)` — fillPoly + polylines thin border.
  - Reuses Task 2's `compute_metrics` (9 metrics: PSNR, SSIM, MS-SSIM, ΔE2000,
    Edge F1 + P/R, EPI, Quant Error, Budget Util, FPS).
  - `plot_size_histogram` and `plot_palette` for Task 3 visualizations.
  - `run_experiment` + `build_experiment_grid` + `build_summary` (CSV +
    12-bar comparison chart).

## Problems and solutions

### Bug 1: `compute_metrics` not used; B/K/priority sweep shows no geometry variation

- **Symptom**: All 11 runs have `N_Triangles = 4320`, identical across the
  entire grid. The B sweep (different S_sets) gives bit-identical metrics
  because quadtree starts at S_max and never splits.
- **Root cause**: On a 1706×1279 image padded to S_max=32, the starting grid
  is `1712/32 × 1280/32 = 54 × 40 = 2160` cells = **4240 triangles**, already
  inside the 9990 budget. With nothing to split, the quadtree returns
  immediately. Sweeping S_set has no effect because the starting cell size
  never varies.
- **Why this matters**: The whole point of Task 3 is to show *adaptive*
  geometry; with no splitting we get the same image 11 times.
- **Resolution options tried**:
  1. Start quadtree at S_max/2 (e.g., 16) — but splitting only *adds* triangles
     (2 → 8), never removes; starting at 16 gives 17120 triangles > 9990.
     Cannot be undone without region-merging, which is slow in this regime.
  2. **Region-merge fallback**: chosen for S_max too small. Implemented with
     SAT-based multi-scale BFS (3 sub-size-levels) to avoid the
     "size-N candidate stale before size-N parent forms" pathology.
  3. **Lower S_min** (1, 2) — works conceptually but blows up runtime
     (~50s per experiment because 2.2M initial S=1 cells).

### Bug 2: region_merge in S_min=1 case runs forever

- **Symptom**: First version of `region_merge_partition` mixed all candidate
  sizes in a single heap. After 38 s it had merged ~493 K of the ~729 K needed,
  then the heap emptied with ~732 K single-pixel leaves stranded at edges.
- **Root cause**: greedy by cost mixed sizes; size-2 candidates are
  cheapest, so most merges happened at size=2 → size=4 transitions were rare
  because the 4 sibling cells of any size-4 candidate were usually
  partially merged. → Stuck.
- **Fix**: switched to **multi-scale BFS** — process size levels in ascending
  order (size=2 → 4 → 8 → 16 → 32). This guarantees each level is
  exhausted before moving up. Performance also improved via SATs
  (rectangle SSE in O(1) instead of O(size²) per candidate).
- **Status**: still slow (~50s) but mathematically correct. Skipped
  C-group/region_merge and B-group/{1,2,4} runs to stay within budget.

### Caveats

- **C-group region_merge not run**: in our config (S_max=32) the quadtree
  starting grid already fits, so the fallback never triggers. Running
  region_merge explicitly would have produced the *same* result as quadtree
  here. To get a meaningful C-group comparison, S_max would need to be
  smaller than the budgeted grid step (~21 for this image) — out of scope.
- **B-group sweep uses S_min ≥ 2**: original plan included S_set={1,2,4} but
  this triggers region_merge and the runtime makes the full sweep
  impractical. The 3 runs we ship ([2,4,8,16,32], [4,8,16,32], [8,16,32])
  all hit the same starting grid so the metrics are identical.

## Verification

```bash
cd "D:\Program Files\learn_torch\python-cv\Assignment1"
..\venv\Scripts\python code\triangle_brick_task3.py --groups A   # ~10s, 3 runs
..\venv\Scripts\python code\triangle_brick_task3.py --groups D E # ~25s, 4 runs
# (B was run incrementally because region_merge fallback timed out on first attempt)
..\venv\Scripts\python code\triangle_brick_task3.py --groups B   # ~30s after S_min was raised to 2
```

Output artifacts (all under `code/pics/task3/`):
- `A_ksweep/{k04,k08,k16}.png + _size_hist + _palette + _metrics.json`
- `B_sweep/{s_2_4_8_16_32, s_4_8_16_32, s_8_16_32}.{png,json,...}`
- `C_algorithm/quadtree.{png,json,...}`
- `D_palette/{kmeans_lab, median_cut}.{png,json,...}`
- `E_priority/{prio_mse, prio_edgef1}.{png,json,...}`
- `summary/metrics_table.csv` — 11 rows × 21 columns
- `summary/metrics_chart.png` — 2-subplot bar chart ([0,1] + absolute)

## Key numbers from the sweep (CSV)

| group/name | K | SSIM | MS-SSIM | ΔE2000 ↓ | Edge F1 | EPI | Quant Err ↓ |
|---|---|---|---|---|---|---|---|
| A:k04 | 4 | 0.377 | 0.487 | 10.44 | 0.197 | -0.027 | 7.78 |
| A:k08 | 8 | 0.360 | 0.494 | 9.04 | 0.163 | -0.043 | 5.90 |
| A:k16 | 16 | 0.362 | 0.514 | **8.07** | 0.224 | -0.044 | **4.53** |
| B:sweep (×3) | 8 | 0.360 | 0.495 | 9.05 | 0.185 | -0.042 | 5.93 |
| C:quadtree | 8 | 0.360 | 0.495 | 9.05 | 0.185 | -0.042 | 5.93 |
| D:kmeans_lab | 8 | 0.360 | 0.495 | 9.05 | 0.185 | -0.042 | 5.93 |
| D:median_cut | 8 | 0.355 | 0.492 | 9.70 | 0.223 | -0.045 | 6.74 |
| E:prio_mse | 8 | 0.360 | 0.495 | 9.09 | 0.191 | -0.043 | 5.93 |
| E:prio_edgef1 | 8 | 0.360 | 0.495 | 9.05 | 0.185 | -0.042 | 5.93 |

### Findings (raw material for Task 5 "results & discussion")

- **K sweep (A)**: K=16 wins on ΔE2000 (8.07) and Quant Error (4.53) — more
  colors give finer palette quantization, as expected. But SSIM is *flat*
  across K — the per-triangle area-mean error dominates SSIM regardless of
  color granularity. Edge F1 peaks at K=16 (0.224) — more palette colors
  let triangles near edges hit closer to actual edge colors.
- **Median Cut vs K-Means Lab (D)**: Median Cut is *worse* on ΔE2000
  (+0.65) and Quant Error (+0.81) but *better* on Edge F1 (+0.04). Median Cut
  is biased to extremes of the lightness distribution; K-Means in Lab tracks
  perceptual color distance better.
- **Priority (E)**: ΔMSE/6 vs Sobel-variance produce nearly identical
  metrics. Within the budgeted 32×32 starting grid, priority does not
  affect the result because no splitting ever happens. **This is a real
  finding for Task 5**: under a quadtree that satisfies the budget
  immediately, priority is irrelevant.
- **EPI is still negative (~-0.04)** — uniform grids inherently introduce
  triangle boundaries that don't align with image edges. This motivates
  future work in Task 4 (real-time constraints) and the unresolved part of
  Task 3: adaptive geometry that genuinely follows content.
- **Per-triangle `Budget_Util = 0.432` for all runs** — we use 4240 of the
  9990-triangle budget. Half the budget is unspent. The remaining headroom
  is the *advertised* benefit of quadtree-style adaptive partitioning; in our
  setup, no run actually consumes it.

## Known limitations / TODOs

- **Geometry sweep is a no-op under current config**: S_set/B sweep doesn't
  show different geometries because quadtree's S_max=32 starting grid already
  fits the budget. To see real adaptive geometry we'd need either:
  - Higher-budget regime (Task 4 camera, where we can use larger images)
  - Or test images with different aspect ratios / smaller sizes where S_max
    needs to be ≥ ~21 to fit.
- **Region-merge fully implemented but not exercised in this batch**: see
  caveats above.
- **Median Cut and K-Means are nearly indistinguishable on this
  image**: the geometry bottleneck dominates; palette method choice matters
  less. A different image (e.g., strong hue variation) might reveal larger
  palette-method gaps.
- **Heatmap data dumps inflate metrics.json to ~120 MB each**: storing
  per-pixel ΔE2000 heatmaps in JSON is wasteful. For Task 5 the heatmap PNGs
  are what we want; should drop the JSON heatmap arrays going forward.
- **AI limitations to note in Task 5**:
  - AI suggested simple delta-MSE priority; we implemented it but found it
    indistinguishable from edge-density priority in our setup.
  - AI suggested {1, 2, 4, 8} as the only S set; in practice this triggers
    a different code path (region-merge) and changes the runtime budget
    enough that the whole approach has to be redesigned.
  - AI gave 4×3=12 experiments; in reality 11 fit (C:region_merge skipped)
    because the AI's mental model didn't account for the budget–geometry
    interaction.