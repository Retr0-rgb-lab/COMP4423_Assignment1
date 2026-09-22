# code/ -- index

All source lives here FLAT, on purpose (AGENTS section 4): Task 3 builds on
Task 2 and Task 4 rewrites Task 3, so splitting the folder by task would
duplicate code. Use this table to navigate instead.

## Naming convention

- `brick_*.py` -- reusable ENGINE modules (pure: geometry, colour, render,
  metrics, viz, and the tessellators). No task-specific policy.
- `brick_pipeline.py` -- the per-frame engine for the LIVE loop (Task 4): config,
  palette state, and `render_frame`. It is an engine module too, hence the
  prefix; it is separate from `camera_app.py` so it can be tested headless.
- Task drivers -- one entry point per task: `capture.py` (T1),
  `triangle_brick.py` (T2), `triangle_brick_task3.py` (T3), `camera_app.py` (T4).
- `task<N>_*.py` -- analysis and verification drivers (produce figures/metrics
  or run the Task 4 test harness).

## Engine modules (reusable)

| file | responsibility | task(s) |
|---|---|---|
| `brick_geom.py` | grid, reflect-pad + crop, leaf -> triangle; the single source of the diagonal convention | 2/3/4 |
| `brick_color.py` | 3-colour quantizers (multilevel Otsu, K-Means, fixed) + K-colour palette builders (K-Means-Lab, median cut) + nearest-BGR assignment | 2/3 |
| `brick_color_rt.py` | live-loop colour helpers: warm-started K-Means palette, sticky (hysteresis) quantizer | 4 |
| `brick_means.py` | per-triangle mean colour (`mask`/`fast`/`sample`), corrected in-cell means, `compare_means` | 2/3/4 |
| `brick_means_rows.py` | exact row-run-table means (bit-identical to `fast`, fewer numpy calls) | 4 |
| `brick_render.py` | reference rasteriser (`triangle_means_bgr`, `render_triangles`) + batched renderer | 2/3/4 |
| `brick_metrics.py` | 9-metric evaluation suite (PSNR/SSIM/MS-SSIM/dE2000/Edge F1/EPI/Quant Error/...) | 2/3 |
| `brick_viz.py` | plots: size histogram, palette swatch, comparison grids, heatmaps | 2/3 |
| `brick_sat.py` | summed-area tables for O(1) rectangle statistics (quadtree priorities) | 3/4 |
| `brick_prio.py` | precomputed split-priority maps (Task 4 optimisation B) | 4 |
| `brick_quadtree.py` | top-down RDO quadtree tessellation (`ref`/`sat`/`precomp` impls) | 3/4 |
| `brick_region_merge.py` | bottom-up aligned region-merge tessellation (dual algorithm) | 3/4 |
| `brick_temporal.py` | live-loop scene-stability state: partition reuse, sticky labels, render cache | 4 |
| `brick_io.py` | camera open (+ optional AE/WB lock), PNG save, `StageTimer` | 1/4 |
| `brick_display.py` | side-by-side composite, compact HUD, aspect-preserving window fit | 4 |
| `brick_pipeline.py` | per-frame live engine: `FrameConfig`, `PaletteState`, `render_frame` | 4 |

## Task drivers

| file | what it runs |
|---|---|
| `capture.py` | Task 1: read an image and display it (any key closes) |
| `triangle_brick.py` | Task 2: equal-size triangles + 3 colours |
| `triangle_brick_task3.py` | Task 3: adaptive multi-size + multi-colour (groups A-E) |
| `camera_app.py` | Task 4: live camera -> triangle-brick display (the deliverable) |

## Analysis / verification drivers

| file | what it produces |
|---|---|
| `task3_best.py` | Task 3 chosen configuration + comparison renders |
| `task3_smax_curve.py` | Task 3 perceptual quality vs the largest allowed cell (S_max) |
| `task3_marginal.py` | Task 3 marginal-benefit / rate-distortion analysis |
| `task4_verify.py` | Task 4 correctness assertions + paired benchmarks (headless) |
| `task4_verify_util.py` | helpers for `task4_verify.py` (synthetic frames, metrics, churn) |

## Where to look next

- Decisions, measurements, known limitations: `docs/progress/Task0_brief.md` ..
  `Task4.md` (the report's first-hand evidence).
- Run commands: `README.md` at the repo root, and `AGENTS.md` section 9.
- Outputs and figures: `code/pics/` (inputs in `pics/`, per-task outputs in
  `pics/task2/`, `pics/task3/`, `pics/task4/`).
