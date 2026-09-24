# COMP4423 – Assignment 1 — Triangle-Brick Image Generation

**Author**: Huang Haoran (24101322D)
**Repo**: https://github.com/Retr0-rgb-lab/COMP4423_Assignment1


## 1. Introduction

Given a single image, the assignment asks to convert it into a mosaic of
triangular bricks that visually approximates the original, under a hard budget
of at most 10,000 individual right-isosceles triangles, with no gaps or
overlaps, where each brick carries one uniform face colour determined from the
image region it covers. Multiple brick sizes and multiple colours are allowed.

Because the budget is fixed, the real work is deciding where the detail should
go. More triangles capture more edges and texture, but they also make each
brick's colour estimate noisier (a tiny triangle averages very few pixels, so
sensor noise starts to look like detail), and none of that addresses the colour
error introduced by mapping each region to a fixed palette entry. I treat the
three as separate things and let each task attack whichever matters most at
that point.

I built the system in four stages, and each stage surfaced a problem that
shaped the next. Task 2 fixes the tessellation to equal-size bricks with three
colours; measuring its output showed that a uniform grid does not align with
image content, which became the motivation for Task 3. Task 3 makes both the
geometry and the palette adaptive, which is where the budget-allocation
question above is actually answered. Task 4 runs the same pipeline in a live
camera loop, where the speed and stability problems the offline setting hides
become the main engineering work.

**Test image.** `code/pics/sky.jpg` (1706x1279) is a cityscape — canal, brick
wall, trees, distant buildings — not sky/cloud content. Its dominant signal is
luminance variation rather than hue variation, which has a measurable effect on
the colour-quantisation experiments (Sections 3 and 4).

![Fig. 1. The shared test image `code/pics/sky.jpg` (1706x1279): a canal-side
cityscape with brick architecture, trees and distant buildings.](code/pics/sky.jpg)

**Narrator.** Throughout, "we" is the author + the GenAI collaboration this
report documents; actions attributed to "the author" in the task sections are
observations or decisions made on the real machine that the AI did not take
part in.

---

## 2. Method — the shared toolkit

The four tasks share one geometry, colour and rendering core (`code/brick_geom.py`,
`brick_color.py`, `brick_render.py`, `brick_metrics.py`). I kept the decisions
that every task depends on in this one place, so later sections only have to add
the one thing each task changes. Table 1 summarises them.

**Table 1. Design decisions of the shared geometry/colour/rendering toolkit.**

| Component             | Choice                                                                                                                                               | Rationale                                                                                                                 |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Brick geometry        | Square cell cut by**one diagonal**, direction alternates by checkerboard parity                                                                | Guarantees right-isosceles, gap-free, overlap-free tiling; alternation avoids a visible single-directional smear          |
| Brick size            | Length of the equal legs, in base-grid units; Task 2 fixed, Task 3 power-of-two sizes {1,2,4,8,16,32,…}                                             | PDF definition; powers of two make quadtree halving exact                                                                 |
| Colour per brick      | Mean BGR of the pixels under the triangle's mask (`cv2.fillPoly` + `cv2.mean`)                                                                   | Robust to sub-region detail; the PDF asks colour be determined from the region covered                                    |
| Palette (Task 2)      | **Multi-Otsu on BT.601 luminance** (primary), K-Means K=3 and a fixed 33%/67% percentile split as baselines                                    | Otsu thresholds adapt to the scene histogram; baselines make the results section a real comparison                     |
| Palette (Task 3)      | **K-Means on CIE-Lab** (K=8/16), Median-Cut as baseline                                                                                        | Lab is perceptually uniform; median cut is cheap and deterministic                                                        |
| Tessellation (Task 3) | **Top-down RDO quadtree**, split priority ΔMSE per triangle (ΔSSE/6); region-merge as dual algorithm                                         | Rate–distortion allocation: equal per-triangle units make different sizes comparable                                     |
| Border                | 1-px`#3C3C3C` polyline per triangle                                                                                                                | Visual separation; PDF permits drawn boundaries, not counted as brick colours                                             |
| Metrics               | 10-item metric suite (9 quality + 1 constraint): PSNR, SSIM, MS-SSIM (4-level mean, a practical variant of Wang et al. multi-scale SSIM), ΔE2000 (CIEDE2000), Edge F1/Precision/Recall (Canny + 3-px tolerance), **edge-alignment correlation (EAC; Pearson correlation of Sobel responses — labelled "EPI" below for short)**, Quantisation Error, Budget Utilisation | Each metric catches a different failure mode; the EAC/EPI metric specifically measures whether triangle boundaries align with real edges |

**Budget semantics.** The limit is 10,000 *triangles* ("bricks"); a quadtree
leaf cell produces 2 triangles, so a partition of 4,995 cells exhausts the
budget at 9,990 triangles. All Task 3 runs use 9,988-9,990 (budget utilisation
>= 99.9%).

**Metric evaluation region.** All PSNR/SSIM/ΔE/Edge/EPI numbers in Sections
3-4 are computed on the *rendered canvas with the 1-px border drawn* (the only
rendered output, per the PDF's "boundaries may be drawn" clause). The border is
`#3C3C3C` and, at 9,990 bricks on `sky.jpg`, covers 23.6% of pixels — so it is
a constant confound shared by every run, but its presence is disclosed here and
its effect is quantified in Section 5.3. The `9-metric suite` therefore counts
PSNR, SSIM, MS-SSIM, ΔE2000, Edge F1/Precision/Recall, EPI and Quantisation
Error (9 quality metrics), with Budget Utilisation tracked as a constraint.

---

## 3. Task 2 — equal-size triangles, 3 colours

### 3.1 Design and testing

The tessellation is a uniform SxS grid cut by checkerboard diagonals. The grid
step is the smallest S for which `2·M·N <= 10000` (M=ceil(H/S), N=ceil(W/S)).
For the three colours I compare three quantisation methods on identical
geometry and means. The primary one is Multi-Otsu: an exhaustive two-threshold
search over the BT.601 luminance histogram, with each class painted the mean
BGR of its members. As baselines I also run K-Means with K=3 on BGR (K-Means++
initialisation) and a fixed percentile split at 33%/67% of the luminance
histogram, the last being a deliberately naive baseline that ignores scene
statistics entirely.

*How it was tested.* All three share one geometry (59x78, S=22, 9,204 triangles
on `sky.jpg`) so only the quantiser differs; the full 10-item metric suite is
computed per method. Outputs: `code/pics/task2/out_task2_compare.png`
(original + 3 renders), `out_task2_residual.png` (per-pixel ΔE2000 heatmaps),
`out_task2_metrics_chart.png`.

### 3.2 Results

**Table 2. Task 2 results: three colour quantisers on identical geometry (59x78, S=22, 9,204 triangles, sky.jpg).**

| Metric            | otsu   | kmeans | fixed           |
| ----------------- | ------ | ------ | --------------- |
| PSNR up (dB)      | 16.15  | 16.16  | 15.56           |
| SSIM up           | 0.332  | 0.334  | 0.286           |
| MS-SSIM up        | 0.482  | 0.481  | 0.452           |
| ΔE2000 down      | 11.41  | 11.41  | 11.92           |
| Edge F1 up        | 0.191  | 0.189  | **0.233** |
| Edge Precision up | 0.151  | 0.150  | 0.171           |
| Edge Recall up    | 0.261  | 0.255  | **0.365** |
| EPI up            | -0.028 | -0.031 | -0.012          |
| Quant Error down  | 8.47   | 8.48   | 8.98            |

Three things stand out in this table. First, otsu and kmeans are
indistinguishable on this scene: the palettes differ by at most 2 per channel
and SSIM by only 0.002 (0.332 vs 0.334). K-Means in 3-D BGR is collapsing onto
the same 1-D luminance split that Multi-Otsu uses directly, because the
dominant signal here is luminance, not hue; its theoretical colour-resolution
advantage never shows up on a near-grayscale image.

Second, and more interesting, `fixed` wins on every edge metric while losing on
every colour and structure metric. Its uniform 33%/67% thresholds happen to
place more of the original edges exactly on a colour boundary between adjacent
triangles, so those edges survive as hard boundaries. Otsu and K-Means push
their thresholds to perceptually meaningful splits (sky versus wall) and
instead smooth over thin edges. There is a real trade-off here between
perceptual fidelity and raw edge preservation, and which method is "better"
depends on what one means by quality.

Third, EPI is negative for all three methods (about -0.03). A negative EPI
means the output has weak edges where the original has strong ones and vice
versa: the regular grid is introducing boundaries that have nothing to do with
the image content. That is the quantitative signature of the uniform grid's
geometric mismatch, and it is the main reason I move to an adaptive
tessellation in Task 3.

![Fig. 2. Task 2 comparison grid: the original image and the three
colour-quantisation results (Multi-Otsu, K-Means, fixed percentile) on identical
geometry, with their key metrics. `fixed` visibly collapses the sky into a
mid-grey band (its 33%/67% split ignores scene statistics).](code/pics/task2/out_task2_compare.png)

![Fig. 3. Per-pixel ΔE2000 residual heatmaps (JET, blue = small error) for the
three quantisers. Otsu and K-Means are nearly identical with error concentrated
in tree foliage; `fixed` adds a bright error band across the water.](code/pics/task2/out_task2_residual.png)

![Fig. 4. Task 2 metric summary across the three quantisers (normalised and
absolute scales). Otsu ≈ K-Means on colour/structure; `fixed` wins only the edge
family.](code/pics/task2/out_task2_metrics_chart.png)

### 3.3 Problems found and solved

The first run produced only two triangles because `compute_grid` scanned S
downward and returned the largest S that fit the budget, which for this image
is a 1x1 grid. The code was syntactically valid but semantically wrong; the
loop has to grow S upward and return the smallest S that stays under budget.
I fixed it by inverting the scan (`code/brick_geom.py::compute_grid`).

The grid also left a black band. The floor-based grid dropped the trailing
rows and columns that did not form a full SxS square (5 px on the right, 19 px
on the bottom of `sky.jpg`), so those pixels stayed black and inflated the
PSNR/SSIM error. I fixed this by switching to a `ceil` grid and reflect-padding
the image to `M·S x N·S`, rendering, then cropping back to the original size
(`code/triangle_brick.py::preprocess`). After the fix SSIM rises from 0.3226 to
0.3316 and PSNR from 15.76 to 16.15 dB; the pre-fix numbers are preserved in
git at commit `a5d2d76` if a direct before/after is needed.

**GenAI use.** The tessellation and the three-quantiser design came from
sitting with a general-purpose LLM; the reconstructed prompts P1-P3 are in
`docs/progress/Task2.md`. The direction-flip bug itself was diagnosed by
asking the model to review `compute_grid`; the model's tendency to default
to fixed thresholds comes up again in Section 7.

---

## 4. Task 3 — adaptive multi-size, multi-colour

### 4.1 Design and testing

I built on Task 2 by adding two adaptive pieces. The tessellation is now a **top-down RDO quadtree** that starts from a coarse grid and splits the highest-priority cell (ΔMSE/6) while budget remains, and the palette is now **K-Means-Lab** with K>3, with Median-Cut kept as a baseline. I dropped the Task 2 primary quantiser (Multi-Otsu) because Otsu is a 1-D luminance method that cannot produce more than a small number of threshold classes. Task 3 needs an arbitrary K-colour palette, so I replaced it with a general clustering method on the perceptually uniform Lab space.

**Region merging (dual algorithm).** As a contrast to the top-down quadtree, I also implemented a bottom-up region-merge (`brick_region_merge`). It starts from cells of the smallest allowed size and greedily merges an aligned 2x2 block of equal-size cells into one cell of twice the side while the triangle budget allows, using summed-area tables to score each merge candidate. The merge candidates are generated on the `target_size` grid so every merged block can itself form the next level's 2x2 block. On `sky.jpg` it terminates at 9,990 triangles with 44,415 merges and a final size distribution of only {16,32} (it merges all fine cells away), which is the property that drives its comparison against the quadtree in Section 4.2.

I designed a 15-run experiment grid that sweeps one variable at a time: K (4/8/16), S_min ({2,4,8}..32), S_max ({32,64,128}), partition (quadtree/region-merge), palette (K-Means-Lab/Median-Cut), and priority (ΔMSE/Sobel-edge-density). All runs use 9,988-9,990 triangles (>=99.9% budget). The full table is at `code/pics/task3/summary/metrics_table.csv`.

### 4.2 Results (selected rows)

**Table 3. Task 3 results (selected rows of the 15-run sweep; source metrics_table.csv).**

| run                  | SSIM up         | MS-SSIM up      | ΔE2000 down   | Edge F1 up      | EPI up           | Quant Err down |
| -------------------- | --------------- | --------------- | -------------- | --------------- | ---------------- | -------------- |
| A:k04                | **0.384** | 0.513           | 11.08          | 0.330           | **0.064**  | 9.25           |
| A:k16                | 0.360           | **0.539** | **8.74** | 0.348           | 0.047            | **5.62** |
| B2:smax_32           | 0.359           | 0.520           | **9.95** | 0.295           | 0.046            | 7.30           |
| **B2:smax_64** | **0.421** | **0.555** | 10.21          | **0.377** | **0.109**  | 7.41           |
| C:quadtree           | 0.359           | 0.520           | 9.95           | 0.295           | 0.046            | 7.30           |
| C:region_merge       | 0.332           | 0.522           | **9.59** | 0.298           | 0.019            | **6.47** |
| D:median_cut         | 0.375           | 0.526           | 10.61          | 0.322           | 0.052            | 7.76           |
| E:prio_mse           | 0.370           | 0.518           | 10.22          | 0.320           | 0.050            | 7.33           |
| E:prio_edgef1        | 0.374           | 0.499           | 9.83           | **0.213** | **-0.020** | 6.45           |

*(The rows `C:quadtree`, `B2:smax_32` and `D:kmeans_lab` are the same default
run of the same configuration and reproduce the same values (SSIM 0.359, ΔE
9.95); `E:prio_mse` shares the configuration but differs slightly (SSIM 0.370)
because K-Means-Lab is run-to-run non-reproducible (Section 4.6). The full
15-row table is in `code/pics/task3/summary/metrics_table.csv`.)*

*(Provenance note for the `E:prio_edgef1` row: its numbers were recorded on
the original per-region Sobel reference path. The current `impl="sat"` runs
Sobel once globally and gives values that differ near region edges (median ~2x
relative difference), so this exact row is not reproducible from the shipped
code; the verdict below is robust under both implementations.)*

Key findings (all on the single test image `sky.jpg`):

> **Tying back to Task 2's motivation.** Section 3.2 reported a negative EPI
> (≈ -0.03) for the uniform grid and used it as the motivation for adaptive
> geometry. The adaptive configuration now makes EPI positive for the chosen
> config (+0.046, Section 4.6) and it rises to +0.109 at S_max=64, and the
> worst Edge F1 of the grid (0.213, Sobel priority) is still higher than
> Task 2's best (0.191). (The two families are not directly comparable:
> 3 colours vs 16, 9,204 vs 9,990 bricks, but the direction of the change is
> the one I was after.)

Larger K monotonically improves colour error. From K=4 to K=16, ΔE2000 moves from 11.08 to 8.74 and Quant from 9.25 to 5.62, at a small structure cost (SSIM 0.384 to 0.360). **K=16** minimises colour error, but the best SSIM and EPI of the K sweep both belong to K=4 (0.384 and 0.064), so the chosen config is a balanced compromise.

Raising S_max from 32 to 64 was the largest structure and edge win of the sweep: SSIM moved from 0.359 to 0.421, MS-SSIM from 0.520 to 0.555, Edge F1 from 0.295 to 0.377, and EPI from 0.046 to 0.109 (more than doubled), offset by a ΔE2000 regression of +2.6%. A coarser start grid leaves more budget for the deep splits where detail matters. Going to S_max=128 adds only a rounding-level change over 64 (SSIM 0.423 vs 0.421). I examine the colour side of this trade-off in Section 4.4.

ΔMSE priority beats Sobel edge-density priority on the structure and edge metrics. The Sobel version is the only negative-EPI run of the grid (-0.020) and has the worst Edge F1 (0.213); it leads only on colour terms (ΔE2000 9.83 vs 10.22, Quant Error 6.45 vs 7.33, SSIM 0.374 vs 0.370). Edge-density allocation over-exploits high-gradient pixels without aligning boundaries to real edges, and the measurements largely rejected this otherwise plausible-sounding suggestion.

Comparing quadtree against region-merge, the quadtree wins on structure (SSIM, EPI) while region-merge wins on colour (ΔE2000, Quant). Region-merge collapses to only {16,32} cells, while the quadtree keeps a {4,8,16,32} mix.

![Fig. 5. All 15 Task 3 runs, per-metric bar chart (normalised and absolute
scales). The sweep variables (K, S_min, S_max, partition, palette, priority) are
on the x-axis groups; the structural-vs-colour split is visible across the
whole grid.](code/pics/task3/summary/metrics_chart.png)

### 4.3 The marginal-benefit / rate-distortion study

Driving S_max upward raised a question I wanted to answer: at what size does splitting stop paying? I ran a dedicated experiment with a 60,000-triangle budget that recorded each split's marginal benefit (ΔSSE per new triangle) by parent size. The curves are at `code/pics/task3/analysis/marginal_benefit_curve.png`, `benefit_by_size.png`, and `cumulative_benefit_curve.png`.

**Table 4. Marginal benefit of a split by parent cell size (60,000-triangle run).**

| parent size | splits | mean ΔSSE/tri | share of total benefit |
| ----------- | ------ | -------------- | ---------------------- |
| 64          | 484    | 345,760        | 39.3%                  |
| 32          | 972    | 111,064        | 25.4%                  |
| 16          | 1766   | 39,219         | 16.3%                  |
| 8           | 2964   | 16,889         | 11.8%                  |
| 4           | 3277   | 8,833          | 6.8%                   |
| 2           | 357    | 5,071          | 0.4%                   |

Splitting a size-64 cell is worth about 68 times a size-2 split per triangle, and all size-2 splits together contribute only 0.4% of the total error reduction. That result justified the S_max sweep in the B2 group and explains why a coarse starting grid frees budget for the deep splits that actually matter.

![Fig. 6. Marginal benefit (ΔSSE per new triangle, log scale) against triangles
used, with the 9,990 assignment budget marked. The greedy benefit decays from
~3x10^6 to ~5x10^3; the budget cuts the curve where splits still pay but
require ever-smaller cells.](code/pics/task3/analysis/marginal_benefit_curve.png)

### 4.4 S_max = 32 vs 64 — a two-sided trade-off

The metric verdict that S_max=64 wins is only one side of the story. Breaking ΔE down by region and by cell size shows the other side. I re-ran the curve below with a **deterministic Median-Cut palette** (K=8) to isolate the S_max geometry effect from K-Means-Lab's run-to-run variance, so its values are not directly comparable to the K-Means-Lab rows of Section 4.2 (where S_max=32 gives ΔE 9.95 and S_max=64 gives 10.21). The two sweeps agree in *direction* but differ in magnitude, and the chart is at `code/pics/task3/analysis/smax_curve.png`.

**Table 5. S_max sweep (deterministic Median-Cut palette, K=8): perceptual vs structural metrics.**

| S_max                   | 32              | 64    | 128   | 256   |
| ----------------------- | --------------- | ----- | ----- | ----- |
| ΔE2000 (lower better)  | **10.61** | 11.77 | 11.79 | 11.79 |
| SSIM (higher better)    | 0.375           | 0.415 | 0.415 | 0.415 |
| Edge F1 (higher better) | 0.322           | 0.403 | 0.402 | 0.402 |

ΔE2000 is best at S_max=32 and worsens afterwards, then flattens out. SSIM, Edge F1, and EPI improve up to S_max=64 and then flatten. There is no single best S_max: 32 wins on large-area colour and 64 wins on structure and edges, and which one is "optimum" depends on how those two sides are weighted. The difference is visually stark in the shadow region. Under S_max=64 the shadow's mean ΔE is about 44% worse than under S_max=32 (11.00 vs 7.61, `code/pics/task3/analysis/crop_shadow.png`), while the sky region is barely affected (`crop_sky.png`). I chose a balanced configuration for this report (Section 4.6).

The mechanism behind the trade-off is that the RDO priority is ΔMSE, which scales with local variance. Edges get split and smooth regions are never split, so a whole large area ends up painted with one colour whose per-pixel error is small but whose *accumulated* perceptual error is large. ΔMSE cannot see large-area uniform drift, but the eye can.

This episode is worth a separate note. On the two families of metrics above, S_max=64 wins four objective scores while the human eye and the single perceptual-colour metric (ΔE2000) both prefer 32. The only metric that agreed with the eye was the one a metric-only summary would be tempted to drop. The two quantities measure different things: per-pixel structure versus accumulated large-area colour drift. Neither one is "the answer". That is why Section 4.6 picks a balanced configuration instead of declaring a single winner, and the same episode shows up again as an AI reporting bias in Section 6.4.

![Fig. 7. S_max sweep, two panels: perceptual colour error (ΔE2000, best at 32,
then worsening and flat) versus structural metrics (SSIM / Edge F1 / EPI, rising
to 64 then flat. ΔE2000 and SSIM/EdgeF1/EPI point in different directions — there is no
single best S_max.](code/pics/task3/analysis/smax_curve.png)

![Fig. 8. Cropped regions of the S_max=32 and S_max=64 renders against the
original: the shadow/brick-wall crop shows the large-area colour drift under 64
(mean ΔE 11.00 vs 7.61), while the sky crop is barely affected.](code/pics/task3/analysis/crop_shadow.png)

### 4.5 Problems found and solved

The first problem was a quadtree inverted-loop bug that the AI introduced. The original code guarded `if n_tri <= budget: return` and looped `while n_tri > budget`, which reflected the wrong mental model: splitting *increases* the count, so the loop must run *while* budget remains. On `sky.jpg` the quadtree never split, using only 4,320 of 9,990 triangles. I fixed it to `while heap and n_tri + 6 <= budget`, and 945 splits then reach exactly 9,990 triangles and EPI turns positive (-0.043 to +0.049). I recorded this as an AI-introduced bug in Section 7.

The second problem was region-merge candidate alignment. Merge candidates generated at every half-pixel offset landed off the target-size grid and could never form the next level's 2x2 blocks, so the run stalled at 8,640 merges. I fixed this by generating candidates aligned to the `target_size` grid, after which region_merge reached 9,990 triangles in about 1.7 s with 44,415 merges.

### 4.6 Chosen configuration

I went with a balanced choice, near-best perceptual colour without throwing away structure. The driver is `code/task3_best.py`, and the visuals are at `code/pics/task3/best/best_config.png` and `best_compare.png`.

**Table 6. The chosen Task 3 configuration.**

| setting   | value                     | why                                                                                 |
| --------- | ------------------------- | ----------------------------------------------------------------------------------- |
| partition | **quadtree**        | keeps a {4,8,16,32} mix; near-best ΔE with much better structure than region_merge |
| S_max     | **32**              | best ΔE2000; 64 improves structure but worsens large-area colour                   |
| S_min     | 1                         | no-op on this image (the budget binds at size 4; §4.2 B-group); kept to allow fine cells where the budget permits |
| K         | **16**              | best ΔE2000 + quant error                                                          |
| priority  | **mse** (ΔMSE / 6) | wins on the structure/edge metrics (see Section 4.2)                                |
| palette   | **kmeans_lab**      | better ΔE than median_cut                                                          |

Metrics (9,990 triangles, `sky.jpg`):

**Table 7. Metrics of the chosen configuration and its two rivals (seeded, from best/*_metrics.json).**

| config                                               | ΔE2000        | SSIM            | PSNR  | Edge F1         | EPI              |
| ---------------------------------------------------- | -------------- | --------------- | ----- | --------------- | ---------------- |
| **quadtree S_max=32 K=16 kmeans_lab (chosen)** | **8.95** | **0.359** | 17.56 | **0.346** | **+0.046** |
| region_merge S_max=32 K=16 kmeans_lab                | **8.63** | 0.339           | 17.59 | 0.298           | +0.015           |
| region_merge S_max=32 K=16 median_cut                | 9.47           | 0.338           | 17.54 | 0.336           | +0.016           |

*(Numbers from `code/pics/task3/best/*_metrics.json`, regenerated by
`task3_best.py` with `cv2.setRNGSeed(0)`; these JSON files are the single
source of truth for the chosen config. The same configuration appears in three
places in this report with slightly different ΔE2000 values — 8.74 (Section
4.2 A:k16, unseeded sweep), 8.95 (this table, seeded), 9.03
(`docs/progress/Task3.md` §10, earlier run). All three are "the same
configuration" under K-Means-Lab's declared run-to-run non-reproducibility; the
~0.3 spread is the noise floor of that variance, and the conclusion is
unaffected: quadtree ≈ region_merge on colour, far better on structure.)*

I also ran a border-free ablation. All numbers above (and in Sections 3-4) include the 1-px gray border, which is the single most common colour in the output (23.6% of pixels) and therefore a large, constant error term. Re-running the chosen config with the border *not drawn* gives markedly better metrics: PSNR 23.35 dB, SSIM 0.571, ΔE2000 6.83, Edge F1 0.507, EPI +0.274 (same partition, same palette, same labels). The border therefore roughly halves measured fidelity on every metric, but because every compared run shares it, the *comparisons* and rankings in this report are unaffected. The borderless render is what the task's "boundaries are not brick colours" clause implies as the colour-only metric, and I keep the bordered numbers as primary because they are the actual delivered output.

Even though region_merge achieves the single-lowest ΔE of the grid (8.63), I did not pick it because it merges all fine cells away, leaving final sizes of only {16,32}. That gives it the worst structure score of the grid (Edge F1 0.298, EPI +0.015). The quadtree keeps a {4,8,16,32} mix and costs only +0.4 ΔE for a much better structure score. The comparison is reproducible: `task3_best.py` now writes per-config metrics JSON to `code/pics/task3/best/*_metrics.json`.

![Fig. 9. The chosen configuration against its two region-merge rivals (original
+ three renders, key metrics per panel). The quadtree render keeps fine cells in
detail regions that region-merge flattens.](code/pics/task3/best/best_compare.png)

![Fig. 10. Task 2 (uniform grid, 3 colours) versus Task 3 (adaptive quadtree,
K=16) on the same image: the adaptive tessellation concentrates small bricks on
detail and large bricks on flat regions.](code/pics/task3/summary/best_vs_task2.png)

### 4.7 Brick-size summary (Task 3 deliverable)

The PDF asks Task 3 to output, besides the rendered image, a brick summary with the total number of bricks and the count for each brick size. The chosen configuration from Section 4.6 produces the following partition, saved at `code/pics/task3/summary/brick_size_counts.json`.

**Table 8. Brick-size summary of the chosen configuration (Task 3 deliverable).**

| cell size (px) | cells | bricks (=2x cells) |
|---|---|---|
| 4  | 724  | 1,448 |
| 8  | 1,479 | 2,958 |
| 16 | 981   | 1,962 |
| 32 | 1,811 | 3,622 |
| **total** | **4,995** | **9,990** |

The quadtree reaches four brick sizes, so the multi-size requirement is met with a true {4,8,16,32} mix. The per-size histogram of the K=16 sweep run is at `code/pics/task3/A_ksweep/k16_size_hist.png`, and the JSON above is the count source.

![Fig. 11. Brick-size histogram of the chosen (K=16) quadtree run: a
{4,8,16,32} mix with the most cells at size 8 and 32.](code/pics/task3/A_ksweep/k16_size_hist.png)

---

## 5. Task 4 — real-world camera scenario

### 5.1 Requirements and strategy

When I first ran Task 3's offline pipeline on the 1706x1279 reference image, partition plus means plus render plus metrics came out at roughly 0.1 FPS, so the speed question was already in the room before Task 4 began. Task 4 takes the chosen config from Section 4.6 and drops it into a camera loop at 640x480, which is where the speed and stability problems the offline setting hid become the central engineering problem I had to solve.

The brief is to run the pipeline live: capture camera frames and display their triangle-brick representation. The grading levels stack on top of each other rather than adding up: camera plus display counts as L1, testing on different content counts as L2, and any one of real-time, different aspect ratios, different lighting, or quantitative evaluation bumps the work to L3. L4 then asks me to analyse the problems, improve the pipeline, and compare before/after numbers.

I picked real-time processing as my L3 lever, because the L4 requirement ("analyse -> improve -> compare before/after") is exactly the work of making the pipeline fast, and one workstream satisfies both levels. The deliverable is `code/camera_app.py`, a live loop that opens the camera at 640x480 with CAP_DSHOW, runs the full pipeline per frame, and shows input and render side by side in a resizable window with a compact HUD (q quits, s snaps, p pauses, r resets). I also built a headless `--no-show --input <clip>` mode so that every number in this section can be reproduced without a camera; `code/task4_verify.py` is the assertion suite and `code/task4_verify_util.py` holds the helpers.

### 5.2 Measured baseline (before any optimisation)

Full-quality config (quadtree, S_set=[1..32], K=16, kmeans_lab, budget=9990,
640x480, scale=1.0):

**Table 9. Task 4 measured baseline before optimisation (full-quality config, 640x480, 9,990 bricks).**

| Stage                        | median ms        | mean ms          | Share              |
| ---------------------------- | ---------------- | ---------------- | ------------------ |
| camera read                  | ~32              | —               | —                 |
| quadtree_partition           | 434.7            | 427.4            | 20%                |
| leaves_to_triangles          | 13.3             | 13.5             | 1%                 |
| **triangle_means_bgr** | **1604.2** | **1595.3** | **75%**      |
| build_palette                | 24.6             | 105.0            | 1%                 |
| quantize_nearest_bgr         | 4.5              | 4.9              | 0.2%               |
| render_triangles             | 58.7             | 58.4             | 3%                 |
| **frame total**        | **2140**   | 2204             | **0.45 FPS** |

`triangle_means_bgr` dominates because per triangle it zeroes a full-frame mask
and calls `cv2.mean` — cost is **O(T·H·W)**. The per-triangle cost scales with
image area (995 µs/triangle at 1706x1279, 157 µs at 640x480, 41 µs at 320x240).
Two cheap parameter levers alone reach ~9 FPS (reduce scale to 0.4 and budget to
1000, K=6), so the optimisation work had to beat **9 FPS at 1000 triangles**,
not the easier 0.45 FPS. A later correction established that `build_palette` was
not 12x slower. A 2-frame run made a one-time ~0.5 s warm-up look like a
per-frame cost (median of two samples is their mean); with 30-frame runs the
stage is 3-25 ms/frame. I learned not to read per-stage percentages off a run too short for the statistic to mean anything. Frame counts per table: the 0.45-FPS
baseline is an 8-frame run; the corrected palette figures are 30-frame runs;
per-stage isolation figures are single-frame-repeat or 12-frame paired runs
(details in `docs/progress/Task4.md`).

### 5.3 Level 1: camera capture + display (done)

`camera_app.py` opens the real camera, runs the full pipeline per frame, and shows a side-by-side window. I verified the display numerically against `code/pics/task4/L1_baseline/frame0001_{input,render,compare}.png`: the render has exactly 17 distinct colours (K=16 palette + 1 border colour); the border `(60,60,60)` covers 72,480 px, or 23.6%, which makes it the single most common colour in the render and measurably affects metrics, so the render path is the only place a border is drawn; the palette spans the full dark-to-bright range with a real spread of hues; and overall contrast is preserved (render std 66.4 vs input 68.2). One defect caught me during this stage: the HUD's FPS first read its timestamp immediately after `cap.read()`, so it reported the camera's own ~33 FPS no matter how slow the pipeline was. The timestamp is now taken after all per-frame work, and the HUD, the headless log and the benchmark share one measurement path.

![Fig. 12. Task 4 Level 1, a real camera frame side by side with its
triangle-brick render (K=16, 9,990-brick quality preset). Input and render are
shown together because a mosaic judged without its input says nothing about
fidelity.](code/pics/task4/L1_baseline/frame0001_compare.png)

### 5.4 Level 3: making it real-time (done)

**Step 1: mean colour.** The shipped `triangle_means_bgr` has a real defect, not just a speed problem: `cv2.fillPoly` paints a one-pixel fringe along the RIGHT and BOTTOM edges of every triangle (pixels whose centres belong to the neighbouring cell), so every Task 2/3 brick's colour is contaminated by its bottom-right neighbours. The leak is up to 400% on size-1 bricks and around 8% of pixels overall. I added two new implementations in `code/brick_means.py`:

**Table 10. Mean-colour implementations: speed and quality (640x480, 9,990 bricks).**

| `--means`                    | FPS            | means stage       | whole frame      | ΔE2000         | PSNR               |
| ------------------------------ | -------------- | ----------------- | ---------------- | --------------- | ------------------ |
| mask (shipped)                 | 0.41           | 1716 ms           | 2260 ms          | 9.947           | 14.90 dB           |
| **fast** (exact in-cell) | **1.35** | **20.8 ms** | **532 ms** | **8.633** | **15.33 dB** |
| sample (9 interior pts)        | 1.47           | 5.2 ms            | 493 ms           | 10.244          | 14.87 dB           |

`fast` is 82x faster on the stage, 3.3x overall, *and* closer to the source
(ΔE 9.95 -> 8.63) because it averages only in-cell pixels. `sample` is fastest
but worse than even the leaky reference on small bricks (nothing to sample
inside a 1-4 px triangle). The defect is not a corner case: over the real
4,995-cell partition the reference averages 8.1% foreign pixels (186,021 of
2,297,721), and per-triangle means differ from the correct in-cell means by an
average of ~3.0 grey levels per channel (max 46.1). Every recorded Task 2/3
number inherits this small directional bias, but cross-run comparisons are
unaffected because all runs used the same leaky reference; the corrected
in-cell means are what the report's Task 2/3 numbers would give if the fringe
were absent — a disclosed limitation. (Quantified with `compare_means` in
`code/brick_means.py`; full numbers in `docs/progress/Report_prep_assets.md` §3.)

**Step 2: partition.** `quadtree_partition` recomputed a numpy region mean per
split candidate. Two fixes: drop 4 dead SSE computations per split (436 -> 326
ms), then replace numpy region means with **summed-area tables** (O(1)
rectangle SSE), batched so numpy call count drops ~160,000-fold. Isolated:

**Table 11. Partition implementation: summed-area tables vs numpy reference.**

| Implementation        | 2000 bricks | 5000 bricks | 9990 bricks      |
| --------------------- | ----------- | ----------- | ---------------- |
| ref (numpy)           | 80 ms       | 173 ms      | 326 ms           |
| sat (tables, batched) | 44 ms       | 69 ms       | **107 ms** |
| speedup               | 1.8x        | 2.5x        | **3.0x**   |

The partition is **bit-identical** under `mse` priority (cell set and size
distribution match exactly), so no Task 3 result is invalidated. Caveat:
`edgef1` priority with the table impl uses a global Sobel map and is NOT
numerically equal to the recorded per-region-Sobel runs (median ~2x relative
difference near region edges); the recorded E_priority numbers are pinned to
`docs/progress/Task3.md` and `metrics_table.csv`, and the code prints an
explicit note when this combination is used. The conclusion (ΔMSE beats Sobel
priority) is robust under both implementations.

**Step 3: whole-pipeline analysis.** With means and partition fixed, the frame is ~216 ms and the remaining bottlenecks are partition ~107 ms (~50%, 4,695 greedy splits, with the cost dominated by numpy-call count rather than the arithmetic), render ~55 ms (~25%, ~20,000 Python-to-cv2 crossings), palette ~23 ms (~10%, per-frame K-Means), and triangles ~13 ms. I applied four optimisations (all exact unless noted). Optimisation A is a batched render that groups up to K `fillPoly` calls by colour plus one `polylines`, taking render from 55 ms to ~8 ms with **0 differing pixels**. Optimisation B precomputes split priorities by scoring all ~102k candidates up front so the greedy loop is pure Python; partition drops from 107 ms to ~30 ms and the leaves stay bit-identical. Optimisation C changes the palette refresh cadence to rebuild every 10 frames, keeping the palette byte-stable in between, which amortises palette from 23 ms to ~2 ms. Optimisation D vectorises `leaves_to_triangles`, taking triangles from 13 ms to ~2 ms. Paired in-process benchmark (synthetic 640x480, reps=5):

**Table 12. Whole-pipeline paired benchmark after optimisations A-D (synthetic 640x480, reps=5).**

| stage                 | old                | new               | speedup                                     |
| --------------------- | ------------------ | ----------------- | ------------------------------------------- |
| partition             | 72.3 ms            | 33.8 ms           | 2.14x                                       |
| triangles             | 8.3 ms             | 2.2 ms            | 3.79x                                       |
| render                | 28.1 ms            | 8.4 ms            | 3.35x                                       |
| palette (amortised)   | 22.0 ms            | 4.9 ms            | 4.47x                                       |
| **whole frame** | **129.0 ms** | **59.5 ms** | **2.17x (~16.8 FPS processing-only)** |

App-level before/after (same machine, same input): TOTAL 130.5 -> 62.0 ms with
ΔE2000 11.014 == 11.014, PSNR 15.68 == 15.68, **0 pixels differ**.

**Step 4: frame-to-frame jitter (root cause).** With the camera held still the mosaic still flickered every frame. Headless probes measured five quantities (geometry churn, label flips, colour noise, palette delta, canvas pixel change) across four sequences (identical frames, noise sigma=2, frozen geometry, and +1.5 brightness per frame). Three independent causes came out of that probe.

The dominant one is K-Means instability, which produces a 76%-of-pixels flash every palette rebuild. Feeding identical pixels through K-Means without re-seeding gave a max palette delta of **180** because the global RNG lands on a different local optimum each run; with a fixed seed, identical input collapses the delta to 0, but two different noisy frames with the same seed still produced a delta of **144**. Re-seeding alone does not solve this, so I had to attack the rebuild itself rather than its random seed.

The second cause is that the pipeline had no temporal coherence anywhere. Partition, means, and quantise were recomputed every frame from noisy pixels, so sensor noise was moving near-tie split decisions on 3-5% of cells per frame and flipping labels near palette boundaries on roughly 3% of cells. With nothing to anchor the output, even a steady scene drifted.

The third cause is camera auto-exposure and white-balance drift, which amplifies the other two. A 1.5/255-per-frame brightness ramp was enough to raise per-frame canvas churn from about 0.5% to 2.9-8.1%. So even after I fixed K-Means and added coherence, an unlocked camera exposure would still keep churning the output.

**Step 5: the fix (A+B+C).** Fix A is to warm-start the palette from the previous one, refining the previous local optimum instead of jumping to a new one, plus a dead-band that discards a rebuild where no entry moves by more than 2 grey levels. Fix B is to reuse the partition when a keyframe signature says the scene has not changed; static frames keep the same leaves, while means are still re-extracted from the live pixels. Fix C is quantise hysteresis: a per-triangle label memory keeps the previous label unless a different palette entry wins by a relative margin. Measured (churn = fraction of pixels whose max channel change exceeds 8 grey levels):

**Table 13. Frame-to-frame jitter fixes (temporal coherence A+B+C): before/after.**

| metric                                    | before           | after                         |
| ----------------------------------------- | ---------------- | ----------------------------- |
| palette delta on rebuild, identical input | **153**    | **1**                   |
| steady per-frame churn, noise sigma=2     | **2.400%** | **0.064%** (37.7x less) |
| rebuild-frame churn, identical frames     | 0.013%           | **0.000%**              |
| rebuild-frame churn, noise sigma=2        | 0.029%           | **0.000%**              |
| app static TOTAL ms (temporal OFF -> ON)  | 68.9             | **27.5**                |
| partition stage ms                        | 37.7             | **0.2** (reused)        |

The 76.7%-of-pixels periodic flash is gone; continuous crawl drops ~38x on
realistic sensor noise; the partition stage disappears on a static scene.

**Step 6: render reuse and row-run means (opt E).** Two remaining per-frame
costs on a static scene: means (~14 ms) and render (~10 ms). E1: exact
**row-run means** — each triangle half is a contiguous pixel run per row, so a
per-row prefix-sum table answers a row's run in two lookups; bit-identity is
provable (exact integer sums < 2^24, so float32 holds them exactly and addition
order cannot change the value). `task4_verify` asserts max|diff|=0: 14.2 -> 7.7
ms (1.85x). E2: **render reuse** — the canvas is a pure function of
`(tri, labels, palette, shapes)`; when geometry is reused and labels+palette are
byte-identical, the cached canvas is the image, so rasterising is skipped (10 ->
0 ms on a static scene; under noise it correctly does not fire, because labels
change). Stacked effect on a static scene: 68.9 -> 27.5 -> **11.4 ms** (~6x),
all exact reuse, no quality cost.

**Step 7: display panel.** Two presentation defects fixed: the HUD previously
drew 14 lines on solid black boxes covering ~35-40% x 30-35% of the render pane
(now three lines on one translucent panel, ~15% x 3-4%); and resizing a
`WINDOW_NORMAL` window stretched the bricks (fixed by uniform letterbox
scaling into the window's image area, so OpenCV's stretch becomes the identity).

**Step 8: reuse freezes colour + camera AE/WB lock.** Even after A+B+C, a static scene still showed residual colour churn (0.101% of pixels/frame) because means and quantize were still re-running every frame. The fix is, when the partition is reused, also freeze the previous labels plus the canvas, which makes an unchanged scene byte-identical frame to frame. I also locked the camera's auto-exposure and white-balance (`--lock-ae`) because a steady exposure is what makes freezing colour safe rather than stale. Measured: churn 0.101% -> **0.000%**, static effective FPS 22.6 -> **37.1**. I confirmed on the real camera that the jitter is noticeably reduced and the fidelity is no worse.

### 5.5 Level 2: different content

The same pipeline was run on seven scenes of different content, captured
with `camera_app.py --lock-ae --snapshot-dir code/pics/task4/` (each snap
stores input + render side by side, see e.g. `code/pics/task4/snap_000.png`).
The seven scenes cover face, strong texture, low light, backlight / high
contrast, and large flat regions.

![Fig. 13. Task 4 Level 2: seven scenes of different content captured on the
real camera with the full pipeline. Each snap pairs the raw camera frame (left)
with its triangle-brick render (right). The adaptive tessellation concentrates
small bricks on detail regions (faces, keyboard, lamp glow) and uses large
bricks on flat areas (wall, water). The pipeline is not tuned to one image;
the same configuration runs on all of them.](code/pics/task4/snap_000.png)

Row 1 (snap_000, snap_001): face, glasses detail preserved, warm tint bias on
the cool-toned input (the warm bias is the only consistent artefact across
the seven scenes; a future white-balance pass would address it). Row 2
(snap_002): mechanical keyboard — fine cells fill the keys, large cells
cover the desk. Row 3 (snap_003): low-light desk lamp, an HDR case where the
bulb blows out while the wall is barely lit. Rows 4-5 (snap_004, snap_005):
backlight through curtains, the dense triangle mesh carries the curtain folds.
Row 6 (snap_006): wall with switch and window light bar, mostly flat with a
narrow dynamic range on the switch. The palette reuses across frames within
tolerance (`--palette-refresh 10`); no per-scene recalibration was needed.

### 5.6 Level 4: before/after summary

Every optimisation I made in §5.4 is a paired before/after with stage ms, whole-frame FPS, ΔE/PSNR, and a pixel-diff check, so the L4 requirement ("analyse -> improve -> compare") is documented for every change. The headline reads cleanly only when each number's measurement conditions are attached. With that in mind: the full live pipeline including camera read goes from 0.45 FPS to about 4.6 FPS at the full quality config (whole frame 2140 -> ~215 ms); on static input the temporal reuse lifts the effective rate further, to roughly 37 FPS in headless mode without camera read-back (the camera's own 31-33 FPS read is a separate ceiling); and the processing-only path on synthetic frames reaches about 16.8 FPS. The spatial optimisations are exact, with 0 differing pixels on identical input. The temporal layer (palette cadence, warm-start, dead-band, hysteresis, freeze-on-reuse) is a deliberate frame-to-frame approximation made for stability, and it does trade a small amount of fidelity for that stability, which is why paired ratios, not absolute FPS, are the trustworthy numbers here (see Section 5.2).

### 5.7 Problems found and solved (Task 4)

The 0.45 FPS baseline came from two main offenders: the O(T·H·W) means stage and the per-candidate numpy means inside the partition (Section 5.2). I solved them with in-cell means, SAT tables, batched render, precomputed priorities, palette cadence, and vectorised triangles, all detailed in §5.4.

A second issue was fringe-contaminated means, a latent bug in the shipped reference implementation that also affected my Task 2/3 runs. The fix is in §5.4 step 1, the measurement is in §5.4, and I disclosed the residual impact on the recorded Task 2/3 numbers there as well.

K-Means palette instability was the third problem. The combination of a global RNG and 10 restarts on near-degenerate data was flipping local optima between frames (§5.4 step 4). I addressed it with warm-start plus dead-band plus reuse plus AE/WB lock (§5.4 steps 5 and 8).

Two presentation defects also needed attention, namely the HUD FPS misreport and the window stretch on resize. Their fixes are in §5.3 and §5.4 step 7.

Finally, the `build_palette` "anomaly" turned out to be a 2-sample measurement artifact, not a real performance problem, as I worked out in §5.2.

---

## 6. GenAI usage (Q4/Q5)

I used GenAI (PolyU GenAI and a general-purpose LLM) on every GenAI-collaboration
task. The prompts below are reconstructed drafts I recorded in
`docs/progress/Task{2,3,4}.md`, and I checked the wording against what I
actually sent. Across the three tasks I logged 17 prompts: 14 were adopted,
and 3 were rejected or rolled back after measurement.

### 6.1 Task 2 (3 prompts, all adopted)

**Table 14. Task 2 GenAI prompts (reconstructed drafts).**

| #  | Prompt intent                                                                                                       | AI output                                                                                                                                              | Outcome                                                         |
| -- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| P1 | Design the tessellation: equal right-isosceles triangles, <=10000, 3 colours, no gaps/overlaps; propose grid step S | Square cells cut by one diagonal (2 tri/cell); S = smallest side with`2*M*N <= 10000`; alternate diagonal by checkerboard parity                     | Adopted; diagonal convention single-sourced in`brick_geom.py` |
| P2 | Compare three ways to pick exactly 3 colours (multi-Otsu on luma, K-Means K=3, fixed percentile)                    | Multi-Otsu as primary (adapts to histogram); keep the other two as comparison baselines; add`_palette_by_luminance` so labels are comparably ordered | Adopted; 9-metric suite compares them                           |
| P3 | First run produced a 1x1 grid; review`compute_grid`                                                               | Loop scanned S downward returning the largest fit; flip to grow upward and return the smallest S                                                       | Adopted; fixed the direction-flip bug                           |

### 6.2 Task 3 (6 prompts, 5 adopted / 1 rejected)

**Table 15. Task 3 GenAI prompts (reconstructed drafts).**

| #  | Prompt intent                                                        | AI output                                                                             | Outcome                                                                    |
| -- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| P1 | Compare top-down quadtree (RDO priority) vs bottom-up region merging | Recommend quadtree with ΔMSE/6 priority; region merge as dual                        | Both implemented; C_algorithm measured the split                           |
| P2 | Palette for K>3: K-Means in CIELAB vs median cut in RGB              | K-Means-Lab perceptually better; median cut cheaper/deterministic                     | Both measured (D_palette); AI did not warn K-Means is non-reproducible     |
| P3 | Split by ΔMSE or by Sobel edge density?                             | Suggested edge-density priority                                                       | **Rejected by measurement**: loses the structure/edge metrics and gives the only negative EPI |
| P4 | Review the quadtree loop condition                                   | Had written the wrong mental model (`<= budget: return`); splitting increases count | AI-introduced bug, fixed (Section 4.5)                                     |
| P5 | At what size does splitting stop paying?                             | Marginal-benefit study: size-64 ~68x size-2; size-2 = 0.4% of benefit                 | Drove the B2 (S_max) group                                                 |
| P6 | `region_merge_partition` stalls at 8640 merges                     | Candidates generated off-grid; align to`target_size`                                | Adopted                                                                    |

### 6.3 Task 4 (8 prompts, 6 adopted / 2 rolled back)

**Table 16. Task 4 GenAI prompts (reconstructed drafts).**

| #  | Prompt intent                                             | AI output                                                                                                                          | Outcome                                                                                                                     |
| -- | --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| P1 | Where does the 216 ms go? Cost model, not just a profile  | `triangle_means_bgr` (O(T·H·W), 75%) + quadtree; "per-call overhead, not complexity"; v2 asks exact-vs-approximate + invariant | Framed all later work as exact-by-construction                                                                              |
| P2 | Render: can it be O(K) calls, same output?                | <=K fillPoly + one polylines; predicted fringe would differ                                                                        | Adopted; 0 differing pixels (better than predicted)                                                                         |
| P3 | Precompute split priorities, identical leaves?            | First version 0.56x slower (mapped dead size-1 cells); fix = exact-integer hierarchical sum                                        | Adopted after a wrong turn; 2.16x, leaves bit-identical                                                                     |
| P4 | Palette cache / vectorise triangles / render reuse        | Warm-start + cadence; one (T,3,2) array; skip rasterise when pure-function cache hits                                              | Adopted (C/D/E)                                                                                                             |
| P5 | Jitter root cause (the hard one)                          | Two causes: K-Means local-optimum instability + whole-frame recompute; then warm-start + dead-band + reuse gate                    | Adopted; palette delta 153 -> 1, churn 2.40 -> 0.06%                                                                        |
| P6 | Isolate per region (freeze geometry, per-cell change)     | Implemented; removed jitter but quality dropped                                                                                    | **Rolled back** after ablation: palette freeze (not geometry) was catastrophic (PSNR 13.9 -> 9.9 on dark first frame) |
| P7 | On reuse, keep previous labels/canvas? What must be true? | Freeze colour on reuse, safe only if exposure steady, so lock AE/WB                                                                | Adopted; churn 0.101 -> 0.000%, FPS 22.6 -> 37.1                                                                            |
| P8 | Dense small bricks are all border; propose options        | Six variants + vision review preferring the combo                                                                                  | **Rejected by the author**: original black border preferred                                                           |

### 6.4 How GenAI understood the tasks (Q5)

Three episodes from this work taught me the most about how the model actually
reasons.

The first is a case of correct-looking code with a wrong mental model (Task 3
P4). My quadtree guard encoded "split down to budget", the opposite of what it
should have done. The code ran without error, and only a careful review of
intent (splitting increases the count) exposed it. The lesson for me was that
AI-generated code which merely executes still needs a sanity check against
what the code is supposed to do.

The second is the cost-diagnosis turn in Task 4 P1. The AI correctly
identified that the bottleneck was Python-to-numpy call count rather than
algorithmic complexity, and its follow-up question about "exact or
approximate? state the invariant" set the tone for every later optimisation,
which kept each one exact by construction.

The third is a bias the AI shares with most assistants, seen most sharply in
the Task 3 S_max=64 episode. When I asked it to pick S_max, it reported "64 is
the clear winner" by leading with the four metrics where 64 won and demoting
the single regression (ΔE2000) to a parenthetical. Only after I pushed back
did it break ΔE down by region and find that the shadow area was about 44%
worse under 64. What I took from this: the model framed its answer to match
the hypothesis I had already implied, and did not volunteer that objective
metrics and human perception can invert.

---

## 7. GenAI limitations and areas for improvement (Q6)

This section is my own assessment, because the limitations of the tool are not
visible in the code it produced but in the places where I had to stop trusting
it. Three patterns showed up repeatedly across the four tasks, and I have
arranged them by what they cost me: the bias that shaped my conclusions, the
unreliability of the code itself, and the model's reluctance to go and look
something up.

The first and most consequential pattern is that a single agent in a single
conversation converges toward whatever the user has already said. I noticed it
when I proposed that raising S_max would give the quadtree more room for fine
splits, and the model immediately agreed, then designed an experiment that
could only demonstrate my hypothesis. It never asked whether the premise was
worth interrogating. The same bias reappeared when the results came in: the
model reported "S_max=64 is the clear winner" by leading with the four metrics
where 64 won and demoting the single contradicting metric, ΔE2000, to a
parenthetical. The contradicting analysis, that the shadow region is about 44%
worse under 64, was only produced after I pushed back (Section 4.4). What
concerns me most is the third variant, where the model treated a metric
verdict as a perceptual verdict. The four objective metrics preferred 64; my
own eyes preferred 32; the model had not anticipated that these could
disagree, and so it had no framework for noticing when they did.

This is not only a quirk of my prompts. Recent work documents the same effect
under the name sycophancy, and finds that RLHF training sharpens it rather
than suppressing it (Papadatos & Freedman, 2024). Two findings from 2026 are
particularly relevant to what I saw. Models are biased toward whichever answer
was presented last, and this recency bias interacts with sycophancy so that
agreeing with the user becomes markedly more likely when the user's position
comes at the end of the exchange. Separately, a model that has been told
something about the user finds it hard to "un-know" it: it cannot faithfully
simulate the decision it would have made without that information. That is
exactly the failure I hit. I was not getting a wrong answer; I was getting a
fluent answer from a model that had adopted my framing as its own, and in a
single continuous session there was nobody to notice. The practice that
actually helped was structural rather than clever prompting: late in the
project I began running the same draft through several independent review
passes, each instructed to attack it from a different angle, and letting them
disagree. A second reader with no memory of how the argument was constructed is
much harder to seduce than the same reader continuing a conversation.

The second pattern is that generated code runs without being correct. The
quadtree in Section 4.5 is the clearest example: the split guard was written as
"if the triangle count is under budget, return" and the loop as "split while
over budget", which is a coherent-looking encoding of the belief that splitting
reduces the count. It does the opposite. The code compiled, ran, and produced a
plausible-looking mosaic that quietly used 4,320 of its 9,990 bricks. Nothing
raised an error; the only symptom was that my measured edge-alignment score was
slightly negative. I found it by re-reading what the code was trying to do
rather than what it said, and the same class of error appeared twice more, in
the merge-candidate alignment and in the grid-step scan. Recent empirical work
on AI coding tools reports the same shape of problem at scale: across 3,800
publicly filed bugs, functional errors dominate, and across 300,000
AI-authored commits the assistants reduce routine maintainability problems but
introduce more bugs and security issues than they fix, precisely where
understanding the program logic matters. My quadtree guard is a small instance
of that larger pattern. The mitigation I adopted is the one this report has
been demonstrating throughout: treat every optimisation as a claim about
invariants and write an assertion that would fail loudly if the claim were
wrong, rather than trusting that correct-looking code is correct.

The third pattern is that the model will not go and look something up unless
told to, and when it does search it may not search thoroughly enough. It
proposed K-Means in Lab space for the palette without mentioning that
`cv2.kmeans` depends on a global RNG, so the palette is not reproducible
between runs. It recommended edge-density split priority without noting that
Sobel variance computed per region and computed globally differ near region
boundaries. In both cases the missing information would have changed how I ran
the experiment, and neither was volunteered. This matters more in a research
setting than the numbers alone suggest: an audit of 111 million scientific
references found at least 146,932 hallucinated citations in 2025 alone,
concentrated in papers with the linguistic signature of AI-assisted writing.
The failure is not only inventing facts; it is producing something fluent and
unsourced where a check was available. The work on this shows the
model often has the relevant knowledge already and loses it at the moment of
committing to an answer, so the gap is retrieval discipline as much as
knowledge. My own version of the problem was subtler: not a fabricated fact,
but a confidently recommended algorithm whose main drawback I had to discover
in the measurements myself.

What I would change for a future project is mostly about process rather than
about the model. I would put the adversarial review pass in from the start
rather than near the end, because it is the only mechanism I found that
reliably broke the agreement. I would require that any claim about a library
behaviour, a metric definition, or a published method be backed by a
reference or a run I performed myself, instead of being accepted because it
sounded right. And I would treat the assistant as a fast colleague whose
proposals are hypotheses to be tested, not as an authority whose output is
merely to be formatted. The parts of this project that worked were the parts
where I was checking: measuring before and after, asserting invariants, and
being willing to reject a suggestion after the numbers disagreed with it. The
parts that failed were the parts where I stopped checking.

---

## 8. Conclusion

I built a triangle-brick image-generation system that converts an image into
right-isosceles-triangle mosaics under a 10,000-brick budget, and worked
through all four assignment tasks.

In Task 2 I set up the tessellation and showed, with the 10-item metric suite,
that
the colour-quantisation method matters less than the geometry. On this
luminance-dominated scene Otsu and K-Means came out close to each other, and
the naive fixed threshold won the edge metrics while losing the colour
metrics. The uniform grid's negative EPI quantified how much the geometry
mismatches the content.

Task 3 made the geometry adaptive (RDO quadtree) and the palette multi-colour
(K-Means-Lab). A 15-run sweep plus a rate-distortion study on `sky.jpg` showed
where the budget is best spent: K=16, S_max=32 as the balanced setting, ΔMSE
priority, quadtree. The S_max=32-vs-64 result I present as a two-sided
trade-off rather than a single winner, because the sweep is single-image
evidence and a strong-hue scene may shift the palette-method and K rankings.

In Task 4 I turned the pipeline into a live camera loop and dealt with the
real-world problems the offline tasks never exercised: an O(T·H·W) mean stage
that took 75% of the frame, the AI-introduced quadtree bug, a latent fringe
defect, frame-to-frame jitter from unstable K-Means and camera exposure
drift, and a misleading HUD. I measured each before and after; on a static
scene the loop runs at about 37 FPS effective (headless, no camera read-back)
through exact spatial reuse plus the declared temporal-stability layer.

The pipeline's layered design, a shared geometry/colour/rendering core
with one decision layer per task, is what let the Task 4 optimisations be
exact (0 differing pixels on identical input) while keeping the Task 2/3
geometry reproducible. I declare the K-Means-Lab palette as run-to-run
non-reproducible, and I re-generate all headline numbers from
`code/pics/task3/best/*_metrics.json` under a fixed seed.

The limitations I want to be honest about: a single test image for Task 2
and Task 3, K-Means non-reproducibility (declared, bounded), and measurements
from a single machine for Task 4. The Level-2 scene testing is done
(§5.5), but it is seven frames from one camera in one room, which is enough
to show the pipeline is not tuned to one image and not enough to claim
generalisation.

---

## References

- N. Otsu, "A threshold selection method from gray-level histograms," *IEEE
  Trans. Systems, Man, and Cybernetics*, 9(1), 1979.
- Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli, "Image quality
  assessment: from error visibility to structural similarity," *IEEE Trans.
  Image Processing*, 13(4), 2004.
- Z. Wang, E. P. Simoncelli, A. C. Bovik, "Multiscale structural similarity for
  image quality assessment," *Proc. Asilomar Conf. Signals, Systems and
  Computers*, 2003.
- G. Sharma, W. Wu, E. N. Dalal, "The CIEDE2000 color-difference formula,"
  *Color Research & Application*, 30(1), 2005.
- J. Canny, "A computational approach to edge detection," *IEEE Trans. Pattern
  Analysis and Machine Intelligence*, 8(6), 1986.
- S. Lloyd, "Least squares quantization in PCM," *IEEE Trans. Information
  Theory*, 28(2), 1982.
- F. Crow, "Summed-area tables for texture mapping," *Proc. SIGGRAPH*, 1984.
- P. Heckbert, "Color image quantization for frame buffer display," *ACM
  SIGGRAPH Computer Graphics*, 16(3), 1982.
- H. Papadatos, R. Freedman, "Linear probe penalties reduce LLM sycophancy,"
  *arXiv:2412.00967*, 2024.
- "Not your typical sycophant: the elusive nature of sycophancy in large
  language models," *arXiv:2601.15436*, 2026.
- "Self-blinding and counterfactual self-simulation mitigate biases and
  sycophancy in large language models," *arXiv:2601.14553*, 2026.
- R. Zhang, W. Dai, H. V. Pham, G. Uddin, J. Yang, S. Wang, "Engineering
  pitfalls in AI coding tools: an empirical study of bugs in Claude Code,
  Codex, and Gemini CLI," *Proc. ACM FSE Companion*, 2026.
- Y. Liu, R. Widyasari, Y. Zhao, I. C. IRSAN, D. Lo, "Debt behind the AI boom:
  a large-scale empirical study of AI-generated code in the wild,"
  *arXiv:2603.28592*, 2026.
- Z. Zhao, Y. Wang, T. Stuart, M. De Vaan, P. Ginsparg, Y. Yin, "LLM
  hallucinations in the wild: large-scale evidence from non-existent
  citations," *arXiv:2605.07723*, 2026.
- J. Yeom, J. Sok, H. Kim, S. Park, J. Park, T. Kim, "Hallucination as
  commitment failure: larger LLMs misfire despite knowing the answer,"
  *arXiv:2605.22007*, 2026.
