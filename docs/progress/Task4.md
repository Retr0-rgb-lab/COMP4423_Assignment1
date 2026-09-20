# Task 4: Real-world camera → triangle-brick display (25 marks)

## Goal (from assignment PDF)

> Task 4*: Implement and test the program in a **real-world scenario**. Call the
> camera to capture images and successfully display their **triangle-brick
> representations**. (25 marks)

> Task 4 is a hands-on task where you should fully understand the implementation
> and code from GenAI and make necessary modifications.

## Grading (levels are cumulative; the highest achieved is awarded, NOT the sum)

| Level | Criterion | Marks |
|---|---|---|
| 1 | Capture camera images and display triangle-brick results. | 10 |
| 2 | Additionally, test images with different content. | 15 |
| 3 | Additionally, demonstrate **any one** of: real-time processing; different aspect ratios/resolutions; different environmental conditions; quantitative evaluation. | 20 |
| 4 | Analyze problems, implement improvements, and compare results **before and after** changes. | 25 |

Note what this actually requires: a camera is mandatory (Level 1), but
**real-time is not** — it is one of four Level 3 options. Level 4 is where the
remaining 5 marks live, and it demands a measured before/after comparison.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Level 3 option | **real-time processing** | The Level 4 requirement ("analyze problems → improve → compare") is *exactly* the work of making the pipeline fast. One workstream satisfies both levels, whereas "quantitative evaluation" (already available for free via `brick_metrics`) would leave Level 4 with nothing to improve. |
| Target resolution / FPS | **deferred** — get Level 1 running first, then measure and tune | Chosen by the author; avoids picking a target before knowing the achievable envelope. |
| Mean-colour extraction | **implement both** the exact integral-image version and the sampled version, then compare | This is the single biggest bottleneck (75% of frame time). Comparing the two gives Level 4 a concrete speed-vs-quality trade-off instead of a single unexamined number. |
| Triangle budget | keep the ≤10000 cap as the default, expose it as `--budget` | The PDF's ≤10000 note sits under the Tasks 1-3 criteria and is not restated for Task 4, but a grader may read it globally. Defaulting to 9990 stays defensible; lowering it is a legitimate real-time lever and must be declared when used. |

## Measured baseline (before any Task 4 optimization)

Measured on the target machine (Windows, `learn_torch/venv`, cv2 5.0.0,
numpy 2.4.3) via the **headless benchmark mode of the deliverable itself**, so
these are the same numbers the HUD shows:

```
python code/camera_app.py --no-show --max-frames 8
```
`partition=quadtree`, `S_set=[1,2,4,8,16,32]`, `K=16`, `kmeans_lab`,
`priority=mse`, `budget=9990`, camera 640×480, `scale=1.0`, 9990 triangles.

| Stage | median ms | mean ms | Share | Note |
|---|---|---|---|---|
| camera read | ~32 | — | — | `CAP_DSHOW`, 640×480; ≈31 FPS ceiling on its own |
| `quadtree_partition` | 434.7 | 427.4 | 20% | one numpy region-mean per split candidate |
| `leaves_to_triangles` | 13.3 | 13.5 | 1% | |
| **`triangle_means_bgr`** | **1604.2** | **1595.3** | **75%** | **two full-frame ops per triangle** |
| `build_palette` | 24.6 | 105.0 | 1% | mean ≫ median: a ~600 ms one-off warm-up (cv2.kmeans / skimage Lab) |
| `quantize_nearest_bgr` | 4.5 | 4.9 | 0.2% | |
| `render_triangles` | 58.7 | 58.4 | 3% | one `fillPoly` + one `polylines` per triangle |
| **frame total** | **2140** | 2204 | | **0.45 FPS end to end (17.9 s wall for 8 frames)** |

The `palette` row is why `StageTimer` headlines the median: a one-time warm-up
would have made a mean-based table look like a uniformly slower stage.

### Why `triangle_means_bgr` dominates

Per triangle it does `mask[:] = 0` (H×W) then `cv2.mean(img, mask=mask)` (H×W),
so the cost is **O(T · H · W)** — every primitive scans the whole frame. The
measured per-triangle cost scales with image area, which is the signature:

| Frame | Triangles | `triangle_means` | per triangle |
|---|---|---|---|
| 1279×1706 | 9990 | 2830 ms | 995 µs |
| 640×480 | 9990 | 1573 ms | 157 µs |
| 320×240 | 9988 | 421 ms | 41 µs |

The second bottleneck, `quadtree_partition`, has the same shape of defect: it
recomputes a numpy region mean for every split candidate. Note that
`brick_region_merge.rect_sse` already solves exactly this with O(1) summed-area
lookups — the quadtree simply never adopted it.

### Level 1 status

`camera_app.py` runs the real camera and displays the triangle-brick result, so
Level 1's requirement (camera capture + display) is met, and it is met honestly:
full pipeline, no caching, no approximation, every stage timed. Keys: `q`/ESC
quit, `s` snapshot to `pics/task4/`, `p` pause, `r` reset the timer.

**Known defect found and fixed during this stage:** the HUD's FPS first read its
timestamp immediately after `cap.read()`, so it reported the camera's read rate
(~33–41 FPS) no matter how slow the pipeline was — the single most misleading
number the HUD could show, and one that would have been quoted in the report.
The timestamp is now taken after all per-frame work, so the HUD, the headless
log and this table all measure the full loop period.

## Plan

Staged deliberately, cheapest-to-verify first. Each stage ends with a
measurement so the Level 4 table is built from real numbers as we go.

1. **Level 1 — get it running.** `camera_app.py`: camera loop, full pipeline per
   frame, HUD, per-stage timing, snapshot key. Headless bench mode so it can be
   measured without a display. Baseline is expected to be slow (~0.5 FPS); that
   is the point of recording it.
2. **Level 2 — different content.** Capture several scenes (face, strong
   texture, low light, high-contrast object, large flat region) to
   `code/pics/task4/`, showing the pipeline is not tuned to one image.
3. **Level 3 — real-time.** Remove the two O(T·H·W) / per-candidate defects:
   - `brick_means.py`: per-triangle means via (a) **diagonal summed-area
     tables** (exact, O(1) per triangle after O(HW) preprocessing) and (b)
     **sparse interior sampling** (approximate, ~O(1) per triangle).
   - quadtree split priority via **SAT rectangle SSE** instead of numpy means.
   - palette EMA across frames (removes per-frame K-Means cost *and* colour
     flicker); batched rendering.
4. **Level 4 — before/after.** For each optimization: FPS delta **and** quality
   cost (ΔE2000 / SSIM / budget used), as a table plus side-by-side renders.

## Constraints to respect

- All new code goes in **new modules/functions**; the Task 2/3 paths stay
  behaviour-identical so their recorded 9990-triangle results remain
  reproducible.
- Files ≤ 400 lines (AGENTS 4.1); split by responsibility, not by task.
- Images (input frames and output renders) go under `code/pics/task4/`
  (AGENTS 1.1).

## Verification

- `python code/camera_app.py --no-show --max-frames 30` — headless bench; prints
  median ms per stage and effective FPS. This is how every optimization is
  measured, so the HUD and the report table share one measurement path.
- `python code/camera_app.py` — the live window (must be run on the desktop;
  WSL has no display).

## Known limitations / TODOs

- **Level 1 baseline is not interactive** (~0.5 FPS). Declared, not hidden.
- No camera-index fallback yet: if index 0 is busy, the app exits with a message
  instead of trying other indices/backends.
- Frame-to-frame stability is not addressed yet. Re-running K-Means per frame
  will make the palette jump between frames even after the speed work, so
  "flicker reduction" is expected to become its own Level 4 item.
- Nothing in this file is measured on a second machine; the numbers are from one
  Windows box and one camera.
