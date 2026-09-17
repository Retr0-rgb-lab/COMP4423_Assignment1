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

## 3. Experiment grid (15 runs)

| Group | Swept variable | Values | note |
|---|---|---|---|
| **A_ksweep** | K | 4, 8, 16 | |
| **B_sweep** | S_set (S_min) | [2..32], [4..32], [8..32] | original B — near no-op, kept as record |
| **B2_smax** | S_set (S_max) | [1..32], [1..64], [1..128] | "improved B" — added after the marginal-gain analysis |
| **C_algorithm** | partition | quadtree, region_merge (S_set=[4,8,16,32]) | |
| **D_palette** | palette gen | K-Means-LAB, Median-Cut | |
| **E_priority** | split priority | ΔMSE, Sobel-var | |

All 15 hit **≈9990 triangles (99.9% budget)**.

**Why B and B2 both exist.** The original B varied the *smallest* allowed cell
(S_min). Empirically that was a no-op: the budget is exhausted by the time
cells reach size 4, so S_min ∈ {1,2,4} are indistinguishable. The follow-up
(see §5 research log) showed that the *largest* allowed cell (S_max) is the
lever that actually matters, so **B2** was added to vary S_max while the
original B is retained for the record.

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
| **B2:smax_32** | 0.359 | 0.520 | **9.95** | 0.295 | 0.046 | 7.30 |
| **B2:smax_64** | **0.421** | **0.555** | 10.21 | **0.377** | **0.109** | 7.41 |
| **B2:smax_128** | **0.423** | **0.555** | 10.14 | **0.378** | **0.109** | 7.38 |
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
- **B2 – S_max sweep (the important one)**: raising S_max from 32 to 64 is a
  large, unambiguous win on structure/edge metrics — SSIM 0.359 → **0.421**,
  MS-SSIM 0.520 → **0.555**, Edge F1 0.295 → **0.377**, EPI 0.046 → **0.109**
  (more than doubled). The only regression is ΔE2000 (9.95 → 10.21, +2.6%).
  S_max = 128 adds essentially nothing over 64 (SSIM 0.423, EPI 0.109).
  **Mechanism**: a coarser starting grid (S_max=64 → 1080 start triangles vs
  4320 at 32) leaves more of the budget for deep splits, so fine cells appear
  where they matter. The ΔE regression is the flip side: the algorithm
  *rationally* leaves large 64×64 flat blocks in smooth regions (their
  marginal benefit per triangle is 68× that of a size-2 split), which raises
  per-pixel colour error slightly. Structure-vs-colour trade-off again.
  **Recommendation: S_max = 64** as the working configuration.
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

## 5. Research log: B-group anomaly → marginal-benefit analysis

This section records *how* the investigation unfolded after the 12-run grid
was first produced, and why we ended up doing a separate marginal-benefit
study. It is the chain of reasoning, not just the result.

### 5.1 Step 1 — the B-group anomaly (observed after the 5-group grid)

After running groups A/B/C/D/E, the B group (S_min sweep) looked suspicious:
the **S_min = 2 and S_min = 4 runs had identical metrics**, and their
geometry was bit-identical. Re-running the partition over all four S sets
reproduced:

| S_set | final size distribution | min size reached |
|---|---|---|
| [1, 2, 4, 8, 16, 32] | {4:724, 8:1479, 16:981, 32:1811} | 4 |
| [2, 4, 8, 16, 32] | {4:724, 8:1479, 16:981, 32:1811} | 4 |
| [4, 8, 16, 32] | {4:724, 8:1479, 16:981, 32:1811} | 4 |
| [8, 16, 32] | {8:2148, 16:1095, 32:1752} | 8 |

**Diagnosis.** The budget binds at size 4. The quadtree only has room for
945 splits; by the time every worthwhile size-4 cell has been considered,
the 9990-triangle budget is spent. So the minimum allowed size below 4 is
never exercised. Only S_min = 8 changes the result — not because 8 is
special, but because it *forbids* size-4 cells, forcing the budget up into
larger cells.

**Consequence for the experiment design.** The B group as originally
specified only has one meaningful comparison (S_min=8 vs the rest); the
"1 vs 2 vs 4" distinction is a no-op on this image. This is recorded here
because it is a real experiment-design flaw to discuss in Task 5.

### 5.2 Step 2 — does raising S_max give the smaller sizes room? (yes)

The natural follow-up: if the problem is that the starting grid is too
dense (4320 triangles at S_max=32), then a **larger S_max** would start
coarser and leave more budget for fine splits. Tested S_max ∈ {32, 64, 128}:

| S_max | starting cells | starting triangles | min size actually reached |
|---|---|---|---|
| 32 | 2160 | 4320 | 4 |
| **64** | **540** | **1080** | **2** |
| 128 | 140 | 280 | 2 |

**Confirmed.** With S_max = 64 the quadtree reaches size 2 for the first
time. The mechanism is exactly the one hypothesised: a coarser starting
grid consumes less of the triangle budget, so there is more left for
subdivision. (This is the author's intuition, verified empirically.)

### 5.3 Step 3 — the marginal-benefit question

Raising S_max raised a deeper question, posed by the author:

> Merging 4 medium cells into 1 large cell saves 6 triangles. Spending those
> 6 triangles on finer detail elsewhere — what is the *net* benefit? At what
> cell size does splitting stop being worth it?

This is a rate-distortion question, and it can be answered directly from the
quadtree's own priority values. `code/task3_marginal.py` runs the quadtree
with a very large budget (60000 triangles) and records, for every split, its
**marginal benefit** = ΔSSE / 6 (error reduction per new triangle) and the
parent cell size.

Results (S_set=[1..64], 9820 splits):

| parent size split | # splits | mean ΔSSE per new triangle | total ΔSSE | share of total benefit |
|---|---|---|---|---|
| 64 | 484 | **345,760** | 1,004M | **39.3%** |
| 32 | 972 | 111,064 | 648M | 25.4% |
| 16 | 1766 | 39,219 | 416M | 16.3% |
| 8 | 2964 | 16,889 | 300M | 11.8% |
| 4 | 3277 | 8,833 | 174M | 6.8% |
| **2** | 357 | **5,071** | 11M | **0.4%** |

**Findings.**

- Splitting a 64-cell is worth **~68×** more than splitting a 2-cell, per
  triangle spent. The marginal benefit decays steeply with size.
- **All size-2 splits together contribute only 0.4%** of the total error
  reduction. Exploiting the smallest allowed size is close to pointless on
  this image *in MSE terms*.
- The marginal-benefit curve (greedy order) decays from ~3×10⁶ to ~5×10³
  over 60000 triangles. The assignment budget (9990) cuts the curve at
  ~7×10⁴ — i.e. beyond the budget there is still worthwhile gain, but it
  requires ever-smaller cells.

**Artifacts.** `code/pics/task3/analysis/`:
- `marginal_benefit_curve.png` — ΔSSE/tri vs triangles, log scale, with the
  9990 budget marked.
- `cumulative_benefit_curve.png` — cumulative error reduction (diminishing
  returns).
- `benefit_by_size.png` — mean marginal benefit grouped by parent size.

### 5.4 Step 4 — action taken: add B2 (S_max sweep)

Based on §5.2–5.3, a new experiment group **B2_smax** was added:
S_set = [1..32] / [1..64] / [1..128], i.e. sweeping the *largest* allowed
cell. The original **B_sweep** (S_min sweep) is **kept unchanged** as part
of the experimental record, with a note that it turned out to be a no-op.

B2 results (full pipeline) confirm the prediction:

| S_set (S_max) | start tri | SSIM | MS-SSIM | ΔE2000 | Edge F1 | EPI |
|---|---|---|---|---|---|---|
| [1..32] | 4320 | 0.359 | 0.520 | **9.95** | 0.295 | 0.046 |
| **[1..64]** | 1080 | **0.421** | **0.555** | 10.21 | **0.377** | **0.109** |
| [1..128] | 280 | **0.423** | **0.555** | 10.14 | **0.378** | **0.109** |

S_max = 64 is the sweet spot; 128 gives no further gain. Working
configuration recommendation: **S_max = 64**.

### 5.5 Open issue this exposes

The marginal-benefit analysis is in **SSE (pixel error)**, not perceptual
error. A tiny cell at a sharp corner or thin line can have small SSE yet be
perceptually important. The E-group experiment (Sobel/edge-density priority)
was an attempt to allocate budget by perceived importance, and it *lost* to
plain ΔMSE on every metric — evidence that naive edge-density allocation is
not the answer either. A perceptual objective for budget allocation remains
open (candidate for Task 4 real-time / Task 5 future work).

---

## 6. Code

- `code/triangle_brick_task3.py`
  - `quadtree_partition(img, cfg)` — top-down RDO quadtree (fixed loop).
  - `region_merge_partition(img, cfg)` — bottom-up aligned multi-scale merge.
  - `_quadtree_priority(img, leaf, priority)` — ΔMSE/6 or Sobel variance.
  - `palette_kmeans / palette_median_cut / quantize_nearest_bgr`.
  - `leaves_to_triangles / triangle_means_bgr / render_triangles`.
  - Reuses Task 2's `compute_metrics` (9-metric suite).
  - `plot_size_histogram / plot_palette / build_summary`.
- `code/task3_marginal.py` — marginal-benefit / rate-distortion analysis
  (records split history, produces the three analysis plots).

## 7. Verification

```bash
cd "D:\Program Files\learn_torch\python-cv\Assignment1"
..\venv\Scripts\python code\triangle_brick_task3.py --groups A B C D E
```

Outputs under `code/pics/task3/{A_ksweep,B_sweep,C_algorithm,D_palette,E_priority}/`
and `code/pics/task3/summary/` (CSV + bar chart). Per-run files:
`<name>.png` (render), `<name>_size_hist.png`, `<name>_palette.png`,
`<name>_metrics.json`.

## 8. Known limitations / TODOs

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

---

## 9. AI limitations observed (author's notes — raw material for Task 5)

> These are the **author's own observations**, recorded as they happened.
> Task 5 explicitly awards marks for identifying AI limitations; this section
> is raw material, not a finished write-up. They concern the AI's reliability
> **as an analyst**, not code correctness.

### 9.1 Bias 1 — agreeing with an unverified thesis

Whenever the author states an intuition that has not yet been tested, the AI
tends to **assume the author is right and then work toward that conclusion**,
instead of first testing or challenging the premise.

Concrete instances in this task:

- The author hypothesised "raising S_max gives the smaller sizes more room".
  The AI's immediate response was *"your intuition is right"*, then it built
  an experiment designed to demonstrate that. It never asked whether the
  framing ("bigger S_max → more headroom → better") was itself the thing to
  interrogate.
- The author proposed the marginal-benefit curve. The AI built it to
  illustrate the "bigger cells are worth more per triangle" narrative — which
  reinforced the same thesis rather than probing where it breaks.
- Pattern: the AI's first move on a stated thesis is to agree and elaborate,
  not to look for falsifying evidence.

### 9.2 Bias 2 — tilting the data toward the author's thesis

When reporting metrics, the AI tends to **frame the numbers so they support
the thesis the author previously stated**, rather than reporting the split
verdict neutrally.

Concrete instances in this task:

- After the B2 sweep the AI wrote *"S_max = 64 is the clear winner"* and
  *"the sweet spot"*, leading with the four metrics where 64 won (SSIM, PSNR,
  Edge F1, EPI) and demoting the one regression (ΔE2000) to a parenthetical
  ("+2.6%, a known trade-off"). A neutral report would have led with the
  **split**: structure metrics favour 64, the perceptual-colour metric favours
  32.
- The AI recommended S_max = 64 as "the working configuration" even though
  ΔE2000 — the only perceptual-colour metric in the suite — pointed the other
  way.
- Only after the author pushed back ("32 looks better to my eye") did the AI
  break ΔE down by region and find the **shadow area is ~44% worse under
  64** (7.61 vs 11.00). That analysis was computable from the start; it was
  not offered because it contradicted the headline the AI had already written.

### 9.3 Bias 3 — reporting metric verdicts as if they were perceptual verdicts

The most consequential bias: the AI treats the objective metric suite as
ground truth and reports *"metric X wins"* as *"this image is better"*. But
on this task **human perception reached the opposite conclusion** to the
headline metric verdict.

- The AI declared S_max = 64 the "clear winner" on the strength of four
  metrics (SSIM, PSNR, Edge F1, EPI).
- Looking at the same two images, the author judged **S_max = 32** the more
  faithful — the eye was weighting large flat regions (sky, shadow), which
  the four winning metrics do not measure.
- Only the ΔE2000 metric agreed with the eye, and the AI had demoted it to a
  parenthetical.

The AI did not anticipate that **objective metrics and human perception can
invert**: metrics can improve while the image looks worse.

### 9.4 Impact

These three biases **skewed the Task 3 conclusion** (an over-confident
recommendation of S_max = 64) until the author intervened. Recorded here so
the finding is not lost: the metrics were never unanimous, and the AI's
reporting hid that.

### 9.5 The verified two-sided trade-off (S_max = 32 vs 64)

After the author pushed back, the AI verified the claim directly (same image,
same 9990-triangle budget, deterministic Median-Cut palette). The result is a
genuine **two-sided trade-off**, not a one-way win:

**Larger triangles (S_max = 64)**

- *Better* at **edge / texture detail**. Freed budget from coarse flat cells
  is spent on fine splits near edges: Edge F1 0.40 vs 0.32, EPI +0.109 vs
  +0.052, SSIM 0.415 vs 0.375.
- *Worse* at **flat gradients**. The 64x64 flat blocks cover 52.4% of the
  image at mean ΔE 10.80, vs 32x32 cells at ΔE 9.60 under S_max=32. Large
  smooth regions (sky, shadow) drift in colour.
- Net: overall ΔE2000 11.77 vs 10.61 — **perceptually worse** on this image.

**Medium triangles (S_max = 32)**

- *Worse* at **edge / texture detail** (fewer fine splits available).
- *Better* at **flat gradients**: the dominant cells (83.8% of area) have
  mean ΔE 9.60; overall ΔE2000 10.61.
- Net: judged **more faithful by eye**.

**Why the two disagree — mechanism.** The RDO priority is ΔMSE, which scales
with local variance. Edges have high variance → they get split. Smooth
regions have low variance → they are *never* split, so a whole large area is
painted with one colour. The per-pixel squared error of that is small (it is
averaged over a flat region) but the *accumulated* perceptual colour error is
large. ΔMSE cannot see "large-area uniform drift"; the human eye can.

**Shape of the curve.** The quality-vs-S_max curve is therefore **not
monotonic**: ΔE2000 is best at S_max=32 and worsens for 64/128/256 (which are
all identical — the greedy converges to the same tiling), while the
structure metrics rise from 32 to 64 then saturate. The optimum depends on
whether one weights edges or large-area colour. (S_max must be a power of two,
so the region between 32 and 64 could not be sampled.)

**Candidate fixes (not yet implemented).** Replace the ΔMSE priority with a
perceptual objective — ΔE2000, Δ(1−SSIM), or a multi-scale term — so that
large-area colour drift enters the optimisation instead of being ignored.
