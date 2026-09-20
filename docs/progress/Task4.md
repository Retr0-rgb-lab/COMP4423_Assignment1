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

Evidence: `code/pics/task4/L1_baseline/frame0001_input.png` and
`frame0001_render.png` (one camera frame, saved as a before/after pair via
`--save-render`).

**Render verified programmatically** (the display could not be inspected visually
in this session, so the PNG was checked numerically instead):

| Check | Result | Meaning |
|---|---|---|
| shape | (480, 640, 3) uint8, same as input | no resize/crop surprise |
| distinct colours | **17** | exactly K=16 palette + 1 border colour |
| border colour present | `(60,60,60)`, 72480 px (23.6%) | boundaries drawn as the PDF permits |
| palette luminance range | `(22,23,27)` → `(253,249,249)` | full dark-to-bright span |
| palette hue spread | e.g. `(34,41,65)`, `(72,65,119)`, `(64,37,25)` | genuinely multi-colour with hue, not a grey ramp |
| render std vs input std | 66.4 vs 68.2 | overall contrast preserved, not washed out |

The 23.6% border coverage is worth keeping in mind: the drawn boundaries are the
single most common colour in the output, so they measurably affect any metric
computed against the render (which is why `brick_render.render_triangles`
documents them).

**Known defect found and fixed during this stage:** the HUD's FPS first read its
timestamp immediately after `cap.read()`, so it reported the camera's read rate
(~33–41 FPS) no matter how slow the pipeline was — the single most misleading
number the HUD could show, and one that would have been quoted in the report.
The timestamp is now taken after all per-frame work, so the HUD, the headless
log and this table all measure the full loop period.

**Path convention gotcha (bit me once):** the built-in defaults are
`__file__`-relative so they work from any directory, but a user-supplied
`--snapshot-dir` / `--save-render` / `--output` is resolved against the *current
directory*. Running from the repo root and passing `pics/task4/...` therefore
created a `pics/` directory at the repo root, violating "all images live under
`code/pics/`". Either run from the repo root and pass full `code/pics/...` paths
(the convention the Task 2/3 docs already use), or pass an absolute path. The
help text now says so.

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

## Frame rate at three settings, and where the cheap wins are

Steady-state numbers from 10–30 frame headless runs (long enough that the
one-time warm-up below is amortized), all at 640×480 camera capture:

| Setting (`--scale` / `--budget` / `--k`) | Effective FPS | Frame total | `partition` | `means` | `palette` | `render` |
|---|---|---|---|---|---|---|
| 1.0 / 9990 / 16 (Task 3 best config, full quality) | **0.45** | 2124 ms | 436 | 1598 | 23 | 49 |
| 0.5 / 2000 / 16 | **3.89** | 177 ms | 80 | 78 | 7 | 9 |
| 0.4 / 1000 / 6 | **9.06** | 77 ms | 42 | 26 | 3 | 5 |

Two things this establishes, both of which the Level 4 write-up needs:

1. **The full-quality configuration is genuinely ~0.45 FPS**, not a warm-up
   artifact — 10 frames took 22.5 s. So "real-time" is a real engineering problem
   here, not a rounding issue.
2. **Cheap levers alone reach ~9 FPS** (reduce resolution and triangle budget),
   with no algorithmic change. That is the right first row of the before/after
   table: subsequent optimization must beat *9 FPS at 1000 triangles / 0.4 scale*,
   not the easier target of 0.45 FPS. It also makes the trade-off explicit — the
   9 FPS mode is visibly coarser, which is exactly the quality-vs-speed comparison
   the Level 4 requirement asks for.

## Correction: the `build_palette` "anomaly" was a measurement artifact

An earlier version of this file claimed `build_palette` got ~12× slower when the
colour count dropped ~6× (24.6 ms at 9990 triangles vs ~300 ms at ~1500) and
listed it as an unexplained anomaly. **That was wrong, and the error was mine.**

The 300 ms figures came from runs of only **2 frames**. With two samples the
median is the *mean of the two*, so a run of `[~590 ms warm-up, ~5 ms normal]`
reports a median of ~300 ms — which I read as a per-frame cost. Longer runs show
the real shape:

| Run length | `palette` median | `palette` mean |
|---|---|---|
| 2 frames | 304 ms | 304 ms |
| 3 frames | 3.8–7.5 ms | 171–180 ms |
| 30 frames | 2.7 ms | 21.2 ms |

So `build_palette` has a **one-time warm-up of roughly 0.5–0.6 s on the first
call** (OpenCV K-Means and/or skimage Lab initialisation) and then costs
**3–25 ms per frame** — cheap, and monotone in the triangle count, as one would
expect. There is no anomaly.

Process lesson, recorded because it is the same class of mistake as the Task 3
reporting bias: **do not read per-stage percentages off a run too short for the
statistic to mean anything.** With n=2 a "median" is just a mean, and a single
warm-up frame can masquerade as a persistent cost for any stage. All numbers in
this file now come from ≥10-frame runs.

##### Display: side-by-side, closed by the window's X

The window shows both panes at once -- `a) camera input` left, `b) triangle
bricks` right -- with the live statistics over the render pane. Showing the source
beside the result is deliberate: a mosaic judged without its input says nothing
about fidelity, and Task 4 asks for the "real-world" comparison.

Closing is by the window's close button (X), not only by a keypress. OpenCV's
HighGUI loop does not report a click on X -- the window is destroyed underneath
the loop and `imshow` keeps drawing into nothing -- so `brick_display.window_closed`
polls `WND_PROP_VISIBLE` each frame and treats a raised `cv2.error` as "gone" too.

The composite is verified numerically rather than by eye (this session cannot
display images), against `L1_baseline/frame0001_compare.png`:

| Check | Result |
|---|---|
| composite shape | (506, 1286, 3) = (480+26 caption, 2*640+6 separator) |
| left pane vs `frame0001_input.png` | byte-identical |
| right pane vs `frame0001_render.png` | byte-identical |
| separator band | uniform background |
| caption strip | non-blank |

##### Two presets, because the honest configuration is not watchable

`--preset watch` (default) = scale 0.5 / 2000 bricks / K 8 -> a smooth preview.
`--preset quality` = scale 1.0 / 9990 bricks / K 16 -> the full Task 3 config at
~0.45 FPS, and the configuration all the baseline numbers above were measured at.
An explicit flag overrides the preset. Named presets exist so that retuning a
viewing default cannot silently move the baseline the report quotes.

##### Window size and resizability

The first version used plain `cv2.imshow`, which creates a window locked to the
image's pixel size: `WND_PROP_AUTOSIZE` returns 1.0 and the window cannot be
dragged larger, so on a large monitor the panes look tiny with no way to fix it.
It now uses `cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)`, i.e. resizable by
dragging, plus `--window-scale` for the initial size. Verified by querying the
property in both modes:

| Window creation | `WND_PROP_AUTOSIZE` | Meaning |
|---|---|---|
| plain `cv2.imshow` (before) | 1.0 | locked to image size, cannot be dragged |
| `namedWindow(..., WINDOW_NORMAL)` (now) | **0.0** | resizable |

The second half of the same complaint was that the picture was too small, and the
cause was the `watch` preset choosing `--scale 0.5` -- which shrinks *both* panes
to 320×240 and so produces a 646×266 composite. Reduced resolution is the wrong
lever for a side-by-side view: the honest speed levers are the brick count and
(only if still needed) the resolution, and lowering the count keeps the picture at
full size. `watch` now keeps `scale 1.0` and lowers only the budget:

| Preset | scale | budget | K | Composite size | FPS |
|---|---|---|---|---|---|
| `quality` | 1.0 | 9990 | 16 | 1286×506 | 0.45 |
| `watch` (default) | 1.0 | 1000 | 8 | **1286×506** | **2.90** |

So the default is now a ~4× larger picture for ~25% less frame rate than the
previous `watch`, and both presets render at the camera's native 640×480.

## Level 3, step 1: the mean-colour stage (`brick_means.py`)

### A defect found in the shipped pipeline, not just a speed problem

`brick_render.triangle_means_bgr` builds each triangle's mask with
`cv2.fillPoly`. For a polygon whose edges lie exactly on integer pixel boundaries,
`fillPoly` paints an extra one-pixel fringe along the RIGHT and BOTTOM edges --
pixels whose centres are outside the polygon, i.e. pixels belonging to the
neighbouring cell. The reference then averages that inflated set, so **every
brick's colour in Tasks 2 and 3 is contaminated by its bottom-right neighbours**,
with a fixed directional bias. Measured per cell, both halves:

| cell size | own pixels | `fillPoly` claims | leaked | leak as % of own |
|---|---|---|---|---|
| 1 | 1 | 6 | 4 | 400% |
| 2 | 4 | 12 | 6 | 150% |
| 4 | 16 | 30 | 10 | 62% |
| 8 | 64 | 90 | 18 | 28% |
| 16 | 256 | 306 | 34 | 13% |
| 32 | 1024 | 1122 | 66 | 6% |

Over a real partition this comes to **+20% extra pixels at 9990 triangles** (67610
foreign pixels against 332909 own), and it is worst exactly where it matters: the
Task 3 size distribution is dominated by sizes 2 and 4 ({2: 1489, 4: 1853} of 4995
cells). The PDF asks for each brick's colour to be taken "from its own image
region", so the analytic in-cell partition is the faithful reading and the
reference is the defective one.

The reference is deliberately NOT changed: every recorded Task 2/3 number depends
on it. The corrected implementation lives in `brick_means.py` and the difference is
quantified below, so this reads as a finding with evidence rather than as a silent
rewrite of the earlier results.

### Three implementations, measured

640×480, 9990 triangles, same camera frame, 5-frame runs, through the app's own
benchmark mode:

| `--means` | frame FPS | `means` stage | whole frame | ΔE2000 vs source | PSNR vs source |
|---|---|---|---|---|---|
| `mask` (shipped) | 0.41 | 1716 ms | 2260 ms | 9.947 | 14.90 dB |
| **`fast`** (exact in-cell) | **1.35** | **20.8 ms** | **532 ms** | **8.633** | **15.33 dB** |
| `sample` (9 interior pts) | 1.47 | 5.2 ms | 493 ms | 10.244 | 14.87 dB |

Two things to take from this, and the second is the surprising one:

1. `fast` removes the contamination **and** is 82× faster on that stage: 3.3×
   overall frame rate, ΔE 9.95 → 8.63 (**13% closer to the source**). 77.5% of
   output pixels change, so the defect was not a corner case.
2. `sample` is the fastest but is **less faithful than the defective reference**
   (ΔE 10.24). With 9990 triangles many bricks are 1–4 px, where a handful of
   interior samples simply has nothing to average. It is a real trade-off, not a
   free win, and it is only worth taking when frame rate outranks fidelity.

Note the per-stage numbers now: `means` fell from 1716 ms to 21 ms, so
**`quadtree_partition` is the bottleneck again at ~436 ms of the 532 ms frame**
(82%). It has the same defect shape as the old `means` -- a numpy region mean
recomputed for every split candidate -- and `brick_region_merge.rect_sse` already
solves that with O(1) summed-area lookups. That is the next step.

Artifacts: `pics/task4/L3_means/{mask,fast,sample}/frame0001_{input,render,compare}.png`.

## Known limitations / TODOs

- **Level 1 baseline is not interactive** (~0.5 FPS). Declared, not hidden.
- No camera-index fallback yet: if index 0 is busy, the app exits with a message
  instead of trying other indices/backends.
- Frame-to-frame stability is not addressed yet. Re-running K-Means per frame
  will make the palette jump between frames even after the speed work, so
  "flicker reduction" is expected to become its own Level 4 item.
- Nothing in this file is measured on a second machine; the numbers are from one
  Windows box and one camera.
