# Report preparation assets (2026-09-22)

Source material gathered for the report draft. Two read-only research passes
(sub-agents) produced these; everything traces back to the cited files. This
file is an INDEX for the drafter, not report prose.

## 1. pics product inventory (green/red anchors)

Audited `code/pics/` against AGENTS §1.1 and the Task docs.

### Green (directly citable)
- Task 2: `out_task2_{otsu,kmeans,fixed,compare,residual,metrics_chart}.png` + `out_task2.png` (Task2.md:225-228)
- Task 3 four-piece sets (`<name>.png` + `_metrics.json` + `_palette.png` + `_size_hist.png`), 15 runs:
  A_ksweep{k04,k08,k16}, B_sweep{s_2_4_8_16_32,s_4_8_16_32,s_8_16_32}, B2_smax{smax_32,64,128},
  C_algorithm{quadtree,region_merge}, D_palette{kmeans_lab,median_cut}, E_priority{prio_mse,prio_edgef1}
- Task 3 analysis/: benefit_by_size, crop_shadow, crop_sky, cumulative_benefit_curve, marginal_benefit_curve, smax_curve
- Task 3 best/: best_compare, best_config, quadtree_smax32_k16_lab, region_merge_smax32_k16_lab, region_merge_smax32_k16_median
- Task 3 summary/: metrics_table.csv (15 rows, matches Task3.md §4), metrics_chart.png, brick_size_counts.json
- Task 4: L1_baseline, L3_means/{fast,mask,sample}, new_opt, old_base (each frame0001_{compare,input,render}.png)

### Red (missing / must regenerate)
- `code/pics/task3/summary/best_vs_task2.png` — promised by AGENTS §1.1, does not exist
- `code/pics/task3/best/*_metrics.json` — absent; the 8.63 headline in task3_best.py
  docstring has no product-file backing (trace only to Task3.md §10 prose)

## 2. GenAI Usage material (Q4/Q5)

Extracted from Task2.md:26-64, Task3.md:326-392, Task4.md:1070-1180.
All drafts are RECONSTRUCTED (author must confirm wording). 17 prompts total:
- Task 2: 3 prompts, 3 adopted (tessellation P1, 3-colour quantise P2, grid bug P3)
- Task 3: 6 prompts, 5 adopted / 1 rejected (P3 edge-density priority measured & rejected)
  + P4 is an AI-introduced bug (quadtree inverted loop) fixed later
- Task 4: 8 prompts, 6 adopted / 2 rolled-back (P6 per-region rolled back, P8 border
  overruled by author) + P3 had a 0.56x wrong turn corrected

Rejected/rolled-back = strong "identify AI limitations" material (Task 5):
- Task3 P3: Sobel edge-density priority lost on EVERY metric (EPI -0.020)
- Task4 P6: per-region freeze removed jitter but palette-freeze was catastrophic (PSNR 13.9 -> 9.9 dB)
- Task4 P8: six border variants measured + rendered, author still preferred the original

### Writing tips (Q4 vs Q5)
- Q4 = workflow (how used): decision chain design -> implement -> measure -> decide/rollback
- Q5 = AI's understanding/mental model: quadtree split-direction misconception (Task3 P4),
  "per-call overhead not complexity" (Task4 P1), exact-vs-approximate framing (Task4 P1v2)

## 3. Fringe-leak quantification (measured 2026-09-22)

`compare_means` on the real 4995 leaves of the Task 3 chosen config
(quadtree, S_set=[1,2,4,8,16,32], padded image). Input `code/pics/sky.jpg`.

- ref (fillPoly) vs in-cell mean: **mean abs diff 3.0 / max 46.1** (0-255 units)
- leaked pixels: **186,021 / 2,297,721 = 8.1%** of the pixels the reference averages
- sample method vs in-cell mean: mean 3.7 (the honest accuracy of the sampled path)
- Implication: the recorded Task 2/3 metrics inherit a small, directional bias
  (mean ~3 channel units on a 0-255 scale); cross-run comparisons are unaffected
  because every run used the same reference. Task 4's corrected in-cell means
  remove it. Report should disclose this as a known limitation.

## 4. best/ metrics JSON (regenerated 2026-09-22)

`task3_best.py` now writes `<name>_metrics.json` per config (Heatmap dropped,
format matches `triangle_brick_task3`). Verified region_merge_smax32_k16_lab:
**dE=8.63**, SSIM=0.3392 — the docstring headline is now product-backed.

