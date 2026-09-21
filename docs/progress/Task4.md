# Task 4: Real-world camera → triangle-brick display (25 marks)

## Status snapshot

| Level | Marks | State |
|---|---|---|
| 1 — camera capture + display | 10 | **done**. `camera_app.py` opens the real camera (640×480, CAP_DSHOW), renders the full Task 3 pipeline per frame, and shows the source and the render side by side in a resizable window that closes on its X. Verified: camera opens, 8–30 frame headless runs complete, Task 2 re-run reproduces byte-identical PNGs. |
| 2 — different content | 15 | **not started**. Needs several physical scenes; a `--snapshot-dir` + `s` capture path exists, no named-scene flow yet. |
| 3 — real-time (chosen option) | 20 | **in progress**. Mean-colour stage done (`brick_means.py`); it also uncovered a real defect in the shipped pipeline. `quadtree_partition` is now the bottleneck at ~82% of the frame. |
| 4 — before/after comparison | 25 | **not started**, but every change so far is being measured in the form it needs: same frame, 5-frame runs, FPS plus ΔE2000/PSNR against the source. |

Measured so far (640×480, 9990 triangles, same camera frame, through the app's own
benchmark mode):

| Stage / change | Frame FPS | Stage ms | ΔE2000 vs source |
|---|---|---|---|
| baseline (`--means mask`) | 0.41 | means 1716 | 9.947 |
| + corrected means (`--means fast`) | 1.35 | means 21 | **8.633** |
| + sampled means (`--means sample`) | 1.47 | means 5 | 10.244 |

Refinement ladder available today, full resolution: 2000 bricks → 3.22 FPS,
5000 → 2.01 FPS, 9990 → 1.38 FPS.

Two things are deliberately left UNFIXED because they are Level 4 material —
each is a measured problem with a known fix, which is exactly what the level
asks for:
1. the per-frame K-Means palette is re-seeded only once at startup, so the palette
   changes every frame even for a static scene (measured: per-channel drift up to
   207/255 across 5 calls on identical input; re-seeding per call makes all 5
   identical);
2. the `quadtree_partition` cost, addressed next.

Also still open and NOT to be reported as verified: no second machine, one camera,
and the display has never been inspected by eye in this session (it cannot show
images), so all render checks are numeric.

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

## Level 3, step 2: the partition stage (`brick_sat.py` + `impl=` on the quadtree)

With `means` down to 21 ms, `quadtree_partition` was 82% of the frame. It had the
same defect shape as the old `means`: a numpy region mean recomputed for every
split candidate. Two independent fixes, each measured.

### Fix A -- computer code that nobody read

`quadtree_partition` kept `leaves` as a dict `{(x,y,size) -> SSE}` and computed
that SSE at seed time and again for every child of every split. The value is never
read: the only uses are `len()`, iteration, `in`, and `list(leaves.keys())`. So at
9990 triangles, **4 of the 9 region-SSE computations per split were discarded**,
on top of the 5 the priority function actually needs. `leaves` is now a set and
the waste is gone: 436 ms → 326 ms on that stage for zero change in behaviour.

### Fix B -- O(1) lookups, then batch them, because O(1) was not enough

`brick_sat.py` builds summed-area tables (value and value²) in one O(H*W) pass, so
a rectangle's SSE is four lookups instead of a numpy pass over the region.

The instructive part is that **the first version of this was not faster at all**
(measured 0.9–1.0×). The reason: O(1) in *arithmetic* is not O(1) in *numpy call
overhead*. Each rectangle costs ~8 fancy-index operations, and the quadtree
evaluates ~5 rectangles per split, so at 9990 triangles there are ~160000 tiny
numpy calls and the overhead, not the maths, is the cost. The fix was to issue the
lookups for a whole batch of squares at once -- and then to fold the parent and its
four children into a *single* query of 5N squares. That is where the speed came
from, ~2× from batching the children and another ~1.5× from merging the parent in.

### Measured

Partition stage on one frame, isolated, repeated (so it is not a single-run
fluke):

| Implementation | 2000 bricks | 5000 bricks | 9990 bricks |
|---|---|---|---|
| `ref` (original numpy) | 80 ms | 173 ms | 326 ms |
| `sat` (tables, batched) | 44 ms | 69 ms | **107 ms** |
| speedup | 1.8× | 2.5× | **3.0×** |

End to end, full resolution, `--means fast`, best of three 12-frame runs:

| `--sse` | Frame FPS | `partition` |
|---|---|---|
| `ref` | 1.93 | 325.7 ms |
| **`sat`** | **3.43** | **106.9 ms** |

**The brick partition is bit-identical between the two** (same cell set, same size
distribution, verified per cell) for the default `mse` priority, so this is a pure
speed change and no Task 4 result is invalidated by it.

### One real behaviour difference, restricted to `edgef1`

`priority="edgef1"` is the variance of the Sobel magnitude. The reference runs
Sobel per region, which *reflects* at the region's boundary; the table version runs
Sobel once globally. The two agree for `mse` (max relative difference 0.00e+00 over
5570 cells) but not for `edgef1` (median relative difference 2.05) because the
per-region form invents a mirrored neighbourhood. The code prints a warning when
`edgef1` meets the table implementation, since the recorded E_priority results came
from the per-region form. Task 4 uses `mse`, so this does not affect the camera app.

### Measurement hygiene: the stage numbers are solid, the app-level FPS is not

Publishing end-to-end numbers from single runs was wrong more than once here, so
this records what is and is not trustworthy from this machine.

**Trustworthy: paired, in-process stage measurements.** Running both implementations
alternately inside ONE process -- so they share the machine state, the memory layout
and the clock -- gives stable results that reproduced across sessions:

| Bricks | `ref` best | `sat` best | ratio | identical bricks |
|---|---|---|---|---|
| 2000 | 78.6 ms | 43.4 ms | 1.81x | yes |
| 5000 | 173.4 ms | 65.5 ms | 2.65x | yes |
| 9990 | 341.4 ms | 116.8 ms | **2.92x** | yes |

**Not trustworthy: this machine's end-to-end FPS.** The same app invocation reported
values spanning a factor of two or more between sessions: `--sse sat` at 9990 bricks
measured 1.72, 2.73, 2.95, 3.15, 3.43 and 3.44 FPS, and `--sse ref` measured 0.61,
0.92 and 1.93 FPS. One `--sse ref` run reported a `partition` stage of 1339 ms
against a direct same-implementation measurement of 341 ms -- a 4x gap, far outside
timing noise, and I have **not** explained it. A hypothesis (the per-frame
`cv2.kmeans` in `build_palette` slowing the whole process under sustained mixed load,
which would hurt the Python-loop `ref` path more than the vectorised `sat` one) is
untested and is recorded as a hypothesis only.

Consequences for the report:

- quote the **ratios from paired measurements**, not absolute FPS, for a
  before/after claim;
- if an absolute FPS is needed, measure it on a quiet machine and say the number is
  machine-state dependent;
- the FPS column of the end-to-end table above is kept as the record of what was
  measured, but must not be read as a precise figure.

This is the third error of the same family in this task -- the n=2 "median", the
load-polluted A/B, and now the unstable absolute FPS. The common root is treating one
short observation as a measurement instead of establishing repeatability first.

Artifacts: none for this step -- the partition change is geometry-identical, so the
`pics/task4/L3_means/` renders remain valid for it.

### Regression check: does this change Task 3's results?

`quadtree_partition` is what Task 3 calls, its signature changed, and its default
implementation changed. So it was re-run and compared against the recorded
artifacts, and the comparison is more interesting than a yes/no:

| Field (Task 3 A-group re-run vs recorded) | k04 | k08 | k16 |
|---|---|---|---|
| `N_Cells` | 4995 = 4995 | 4995 = 4995 | 4995 = 4995 |
| `N_Triangles` | 9990 = 9990 | 9990 = 9990 | 9990 = 9990 |
| `counts_per_K` | **differs** | **differs** | **differs** |
| `Quant_Error` | 9.2489 → 9.2626 | 7.2185 → 7.2708 | 5.6189 → 5.7680 |

The geometry is untouched -- same cell count, and the direct per-cell comparison
above shows the same cell SET -- while the palette assignment counts differ. So the
metric drift is entirely the **K-Means palette, which Task 3 cannot reproduce**:
`triangle_brick_task3` never calls `cv2.setRNGSeed`, and `cv2.kmeans` draws from
OpenCV's global RNG, so a re-run gets a different local optimum.

Two conclusions, and the second one matters for Task 5:

1. This step does not invalidate any recorded Task 2/3 number. The artifacts were
   restored with `git checkout` so the committed evidence still matches Task3.md.
2. **The non-reproducibility recorded earlier as a live-loop flicker problem is not
   a live-loop problem.** It is a general defect of the offline pipeline too: two
   runs of the same Task 3 experiment on the same image do not agree. That is worth
   stating plainly in the report's limitations, with the numbers above as evidence,
   because it means the Task 3 sweep's *comparisons* are still meaningful (every run
   is affected the same way) while its exact decimal values are not reproducible.

## Terminology: the budget counts BRICKS, not cells

Worth stating explicitly because it is easy to conflate and the report must not:
the PDF caps "no more than 10,000 individual **triangular bricks**, counting **each
triangle as one brick**". So the budget is in triangles. A CELL is a square that
gets cut into two bricks, so

    bricks = 2 x cells        cells = bricks / 2

`--budget 9990` therefore produces **4995 cells**, not 9990. Everything in the
quadtree code that the report quotes (`N_Triangles` and `N_Cells`) is labelled to
match, and the Task 3 size summary must report the CELL counts per size doubled
into brick counts -- `plot_size_histogram` already does this.

## Brick count vs frame cost, and why the default is now the full budget

Measured with `render_frame` directly, alternating budgets in one process so they
share the machine state, best of three runs each (640x480, `mse`, `fast` means,
`sat` sse):

| `--budget` (bricks) | cells | brick sizes present | frame ms | implied FPS |
|---|---|---|---|---|
| 996 | **498** | {8, 16, 32} | 72.6 | 13.8 |
| 1998 | 999 | {4, 8, 16, 32} | 84.2 | 11.9 |
| 4998 | 2499 | {2, 4, 8, 16, 32} | 136.8 | 7.3 |
| **9990** | **4995** | {1, 2, 4, 8, 16, 32} | **215.7** | **4.6** |

Two things this settles:

1. **Cost scales gently.** Five hundred cells to five thousand cells is 3x the
   frame time, not 10x, because the per-cell work is now O(1) (tables) rather than
   proportional to the cell's area. The whole reason a low budget looked necessary
   was the old O(T*H*W) mean extraction.
2. **A low budget does not just look coarser, it loses the small sizes entirely.**
   At 498 cells the smallest brick that can appear is 8 px; at 4995 cells the
   distribution is {1: 108, 2: 1489, 4: 1853, 8: 1110, 16: 319, 32: 116}, i.e. the
   fine bricks that give the mosaic its texture are 2 and 4. This is the measured
   explanation of the earlier "the picture looks very poor" observation, and it was
   a configuration problem, not an algorithm one.

The default preset is therefore `quality` = the full 9990 bricks / K 16, with
`balanced` (5000) and `fast` (2000) available for a machine that cannot keep up.
The earlier `watch` preset at 1000 bricks was a pre-optimization crutch and is gone.

## Level 3, step 3: whole-pipeline analysis and the A/B/C/D optimization plan

(2026-09-21) Full read-through of every module on the live path, with the stage
costs re-derived from the tables above. Conclusion: **none of the remaining
bottlenecks is algorithmic complexity -- all of them are Python-loop x
small-call overhead**, so the speedups can be exact (geometry untouched)
rather than approximate. This is the plan the implementation work follows.

### Pipeline as measured today (quality preset, 9990 bricks, 640x480)

```mermaid
flowchart TD
    A["cap.read()  CAP_DSHOW 640x480  ~32 ms serial"] --> B["resize (scale=1.0: skipped)"]
    B --> C["partition: quadtree_partition  pad + build_sats + greedy heap  ~107 ms (sat)"]
    C --> D["triangles: leaves_to_triangles  9990 ndarrays in a Python loop  + a DUPLICATE pad_to_max  ~13 ms"]
    D --> E["means: extract_means=fast  per-size offset gathers  ~21 ms"]
    E --> F["palette: build_palette=kmeans_lab  cv2.kmeans per frame  ~23 ms  + palette-flicker defect"]
    F --> G["quantize: quantize_nearest_bgr  ~4.5 ms"]
    G --> H["render: render_triangles  9990 x (fillPoly+polylines)  ~55 ms"]
    H --> I["compose + HUD + imshow + waitKey  NOT timed (hidden cost)"]
```

### Bottleneck table (share of the ~216 ms frame)

| stage | ms | share | root cause |
|---|---|---|---|
| partition | ~107 | ~50% | ~4695 greedy splits, each = one 5-square SAT batch + 4 heap pushes; the cost is the numpy-call count, not arithmetic (the same lesson as the earlier "O(1) but not batched" finding) |
| render | ~55 | ~25% | 2 cv2 calls per triangle = ~20000 Python-to-cv2 crossings |
| means | ~21 | ~10% | sum(s^2/2) ~ 2730 small gathers (the size-32 half alone is 512 offsets) |
| palette | ~23 | ~10% | per-frame cv2.kmeans x10 restarts; also the recorded flicker defect |
| triangles | ~13 | 6% | Python loop + duplicate pad_to_max (partition already padded the frame) |
| quantize | ~4.5 | 2% | already vectorised, leave alone |

Hidden costs outside `StageTimer`: the ~32 ms serial camera read, and
compose/HUD/imshow (not timed at all -- a stage must be added before any of it
can be quoted).

### The four optimizations (A/B/C/D), exact unless stated

- **A -- batched render** (~55 -> ~8 ms). `labels` partitions the 9990
  triangles into <=K colour groups; `cv2.fillPoly` paints many same-colour
  polygons per call (<=K calls) and ONE `cv2.polylines` draws every border.
  Cross-colour fill order differs from today, so the 1-px `fillPoly` fringe
  may differ on shared edges; the final border pass repaints the exact
  boundaries grey. Verify: pixel-diff count vs the current render + Delta-E /
  PSNR unchanged.
- **B -- precomputed split priorities** (~107 -> ~30 ms). A cell's split
  priority depends only on image content, never on partition state -- so the
  whole priority universe (all grid cells of sizes >= 2, ~102k candidates at
  640x480) can be computed in ~6 batched calls, and the greedy loop becomes
  pure Python (heapq + list lookups, zero numpy). Priority values are
  bit-identical to the current sat path (same SAT formula, elementwise), and
  the heap keeps the same (-p, leaf) keys, so the leaves set is bit-identical.
  Verify: sorted-leaves equality against impl=sat.
- **C -- palette refresh cadence** (~23 -> ~2 ms amortised, flicker -> 0).
  Rebuild kmeans_lab every N frames (default 10), reuse it otherwise; an
  optional scene-change trigger can force a rebuild. Between refreshes the
  palette is byte-stable, so the recorded 207/255 drift disappears; the
  N-frame adaptation lag is declared in the report.
- **D -- small wastes**: vectorised `leaves_to_triangles` (~13 -> ~2 ms) with
  the ordering contract kept; `quadtree_partition` gains an optional
  `return_padded` kwarg so the pipeline does not pad the frame twice
  (backwards compatible -- Task 3 callers pass nothing new).

Also added for measurement honesty: `--input PATH` feeds the app from an image
or video file, so the in-app benchmark runs headless (WSL has no camera) and
Level 2/4 get repeatable scene inputs.

Expected end state at 9990 bricks: ~70 ms/frame ~= 14 FPS, versus the recorded
216 ms ~= 4.6 FPS, with geometry and the palette-assignment rule unchanged.
Scope chosen by the author: A+B+C+D as one workstream in a git worktree, each
step measured and recorded separately. Camera threading and the row-run means
rewrite are deferred (need the real Windows machine / a second phase).

## Level 3, step 3a: the analysis itself -- method, process, conclusions, actions

This section records HOW the optimization work was derived, separately from the
numbers it produced, because the method is reusable and the process includes
wrong turns worth keeping.

### Method

1. **Enumerate the live path**: follow `camera_app.py`'s imports to the stage
   sequence (partition -> triangles -> means -> palette -> quantize -> render),
   and read every module on it end to end.
2. **Build a cost model per stage, not just a wall time**: for each stage ask
   "how many Python-to-C calls, and how many do the input size and triangle
   count make?" A stage can be O(H*W) in arithmetic and still be dominated by
   call overhead -- that was the recurring finding in this task.
3. **Cross-check against recorded medians** from >=10-frame runs only (the
   measurement-hygiene rule established earlier in this file; the n=2 "median"
   error is why).
4. **Classify each bottleneck by root cause**: algorithmic complexity vs
   per-call overhead. Only the second class is fixable without changing output.
5. **Decide exact vs approximate before coding**, and for an "exact" claim
   state the invariant that makes it true (here: exact-integer sums < 2^53 make
   summation order irrelevant, so hierarchical sums == SAT lookups bit-for-bit).
6. **Implement behind an existing seam** (`impl=` on the partitioner, new
   functions in the engine modules), never editing the Task 2/3 reference path.
7. **Gate each optimization behind an assertion** (`task4_verify.py`) before
   quoting its speed; a speedup that changes the mosaic is not a speedup.

### Process (including the wrong turns)

- **Precompute v1 was slower (0.56x)**: the first `_priority_maps` scored every
  size including 1, i.e. Hp*Wp single-pixel candidates whose SSE is exactly 0.
  Removing that dead 75% of the universe and replacing the per-size SAT batch
  queries with an exact-integer hierarchical sum took the stage to **2.2x** and
  kept the leaves bit-identical.
- **`cv2.integral` evaluated and rejected**: measured 3x faster than the numpy
  cumsum for the tables, but the cv2 5.0 Python binding returns unusable shapes
  (a per-row list) and its squared-sum is not bit-identical. Rejected rather
  than worked around, because the precompute no longer needs SATs at all.
- **Palette cache off-by-one**: the first `PaletteState` rebuilt every
  `refresh + 1` frames, because `age` only advanced on cached frames. Caught by
  the cadence assertion (rebuilds at 0, 6, 12 instead of 0, 5, 10), fixed by
  counting the rebuild frame and exposing an explicit `refreshed` flag.
- **End-to-end 69% pixel diff, explained not patched**: the first old-vs-new
  whole-frame comparison differed on 69% of pixels. Root cause was the
  ALREADY-RECORDED K-Means non-reproducibility (a global-RNG draw advances
  between the two calls), not the optimizations. Seeding the RNG before each
  path in the check dropped it to **0 pixels**.
- **The render batching was byte-identical, better than predicted**: the plan
  expected the 1-px `fillPoly` fringe on shared edges to differ. Measured 0
  differing pixels on both a synthetic frame and a real photo, because triangle
  interiors are disjoint and the single border pass repaints every boundary.

### Conclusions

- All four remaining bottlenecks were **per-call overhead, not algorithmic
  complexity** (see the bottleneck table above): ~4700 tiny numpy calls in the
  partition loop, ~20000 cv2 calls in the render, ~2730 gathers in the means,
  and a per-frame K-Means.
- **The speedups could therefore be exact**, and were proven so: precomp
  partition leaves == sat leaves (bit-identical), vectorised triangles ==
  per-triangle loop (vertex-exact), batched render == reference render
  (0 pixels), and the whole optimized frame == the old frame (0 pixels) on
  identical inputs.
- The palette cost was removed **and** the recorded frame-to-frame flicker
  disappeared as a side effect of the same change (a byte-stable palette
  between rebuilds).
- Measured end state at the quality preset (9990 bricks, 640x480, paired
  in-process): **129 -> 59.5 ms per frame, 2.17x, ~16.8 FPS processing-only**;
  app-level benchmark 130.5 -> 62.0 ms. The original full-resolution baseline
  was 0.45 FPS (216 ms + camera read), so the road is 0.45 -> ~4.6 (earlier
  work, recorded above) -> ~16.8 FPS processing-only.

### Actions (code)

| Optimization | File / function |
|---|---|
| B -- precomputed split priorities | `code/brick_prio.py` (`priority_maps`, `prio_from_maps`); wired as `impl="precomp"` in `code/brick_quadtree.py` |
| D1 -- vectorised triangles | `code/brick_geom.py` (`leaves_to_triangles_array`) |
| D1 -- no double padding | `code/brick_quadtree.py` (`return_padded=True`) |
| A -- batched render | `code/brick_render.py` (`render_triangles_batched`) |
| C -- palette refresh cache | `code/frame_pipeline.py` (`PaletteState`) |
| driver split + headless input | `code/frame_pipeline.py` (new), `code/camera_app.py` (slimmed, `--input`, `--palette-refresh`, `--sse precomp`) |
| correctness + benchmark gate | `code/task4_verify.py` |

## Level 3, step 4: A/B/C/D implemented and measured

All four optimizations landed on branch `task4/realtime-opt` (worktree
`../Assignment1-task4-opt`), each behind the existing engine-module seams so
the Task 2/3 paths stay behaviour-identical:

| Opt | Where | What |
|---|---|---|
| B | `code/brick_prio.py` (new), `brick_quadtree.py` (`impl="precomp"`) | split-priority universe scored up front via exact-integer hierarchical sums; greedy loop runs on Python lists, zero numpy per split |
| A | `code/brick_render.py` `render_triangles_batched` | <=K `fillPoly` calls (one per palette label) + ONE `polylines` for all borders |
| C | `code/frame_pipeline.py` `PaletteState` | K-Means palette rebuilt every N frames (default 10), byte-stable between rebuilds |
| D | `code/brick_geom.py` `leaves_to_triangles_array`; `quadtree_partition(return_padded=True)` | vectorised (T,3,2) triangle build; the padded image is carried from partition to means instead of being recomputed |
| — | `code/frame_pipeline.py` (new), `camera_app.py` (slimmed) | per-frame engine moved out of the driver; `--input PATH` feeds the app from a file so the in-app benchmark runs headless |
| — | `code/task4_verify.py` (new) | assertion + paired-benchmark gate for all of the above |

### Correctness (task4_verify.py, synthetic 640x480, all asserted)

1. precomp partition leaves == sat leaves, sorted (bit-identical, 4995 cells).
2. `leaves_to_triangles_array` == `leaves_to_triangles`, vertex-exact.
3. batched render vs reference render: **0 differing pixels** on the tested
   frames (even the 1-px fringe is unchanged because interiors are disjoint
   and the single border pass repaints every boundary); Delta-E/PSNR vs
   source identical to the reference render.
4. palette cache: rebuilds exactly on frames 0, N, 2N, ...; byte-identical
   palette between rebuilds (zero drift).
5. whole-frame old path (sat + refresh=1) vs new path (precomp + refresh=10)
   on the SAME frame with the RNG re-seeded per path: **0 differing pixels**.

The end-to-end output of the optimized pipeline is byte-identical to the old
pipeline on identical inputs (given the same K-Means draw). Two design facts
make B exact: a cell's priority never depends on partition state, and all
sums are exact integers < 2^53, so ANY summation order (hierarchical here,
SAT lookups there) produces bit-identical SSE operands.

### Paired benchmarks (reps=5, in-process, best-of; synthetic 640x480)

| stage | old | new | speedup |
|---|---|---|---|
| partition | 72.3 ms | 33.8 ms | 2.14x |
| triangles | 8.3 ms | 2.2 ms | 3.79x |
| render | 28.1 ms | 8.4 ms | 3.35x |
| palette (amortised /5) | 22.0 ms | 4.9 ms | 4.47x |
| **whole frame** | **129.0 ms** | **59.5 ms** | **2.17x (~16.8 FPS processing-only)** |

### App-level before/after (camera_app --no-show --max-frames 30 --input, 9990 bricks)

Same machine, same input, alternating configs; input = sky.jpg downscaled to
640x480 (`code/pics/task4/_bench640.jpg`):

| stage | old (sat, refresh=1) | new (precomp, refresh=10) |
|---|---|---|
| partition | 75.1 | 33.0 |
| triangles | 2.6 | 2.7 |
| means | 14.2 | 13.8 |
| palette | 25.0 | 0.0 (median; 3 rebuilds in 30 frames) |
| quantize | 3.7 | 3.2 |
| render | 10.0 | 9.4 |
| **TOTAL** | **130.5** | **62.0** |

Quality on the saved first frame (old vs new renders, both vs source):
Delta-E2000 11.014 == 11.014, PSNR 15.68 == 15.68, **0 pixels differ**.
Evidence: `code/pics/task4/old_base/frame0001_{input,render}.png` and
`code/pics/task4/new_opt/...`.

Full-resolution sky.jpg (1706x1279) in the same harness: TOTAL 411 ms -> 291
ms (partition 320 -> 231). The speedup shrinks with image area because the
fixed per-split saving stays constant while the map precompute grows.

### Caveats / deferred

- Absolute numbers here came from the Windows venv running under WSL interop;
  quote RATIOS for the report (paired, in-process) and re-measure absolute FPS
  on the real machine before quoting it.
- Camera read (~32 ms) is still serial; the threaded-capture change is
  deferred to the real-machine phase as agreed.
- The means stage (~14-21 ms) is untouched; the row-run table rewrite remains
  a documented phase-2 option.
- `--palette-refresh N` introduces up to N frames of palette adaptation lag;
  N=10 is the default and must be declared when quoted.
- `cv2.integral` was evaluated and rejected for `build_sats`: the cv2 5.0
  binding returns unusable shapes (per-row list) here, and the numpy cumsum
  variants (axis swap, stacked) are bit-identical but no faster.

## Level 3, step 5: frame-to-frame jitter -- root-cause analysis

**Symptom (reported by the author):** with the camera held still, the mosaic
still changes every frame -- a continuous crawl, plus a stronger periodic
"flash". This is analysed and then fixed by the A+B+C temporal-stability
workstream below; this section is the analysis, written before the fix.

### Method

A headless probe drives the exact live pipeline (`quadtree_partition` ->
`leaves_to_triangles_array` -> `extract_means` -> `PaletteState` +
`quantize_nearest_bgr` -> `render_triangles_batched`) over synthetic frame
sequences, and measures FIVE frame-to-frame quantities so the blame lands on a
stage, not on "the pipeline":

| metric | meaning |
|---|---|
| `geo` | cells in the symmetric difference of consecutive leaf sets (geometry churn) |
| `flips` | cells present in both frames whose palette label changed (colour churn) |
| `cold1` | mean per-channel |delta| of a cell's colour (noise level reaching the quantizer) |
| `pal_d1` | max |delta| between consecutive palettes |
| `canvas_d1` | fraction of rendered pixels that changed |

Sequences: **A** identical frames (pipeline-internal instability only),
**B** frames + Gaussian noise sigma=2 (realistic webcam sensor noise),
**B3** same but geometry frozen at frame 0 (isolates colour churn from
geometry churn), **C** frames with +1.5/frame global brightness (models camera
auto-exposure drift).

### Evidence

| sequence | per-frame churn | at palette rebuild (every 10th) |
|---|---|---|
| A. identical frames | **all zero** | `pal_d1` 147, **canvas 76.6%** |
| B. noise sigma=2 | geo 118-234 cells (3-5%), flips ~130-160/4900, canvas 0.55-0.96% | `pal_d1` 166, **canvas 76.7%** |
| B3. noise, geometry frozen | geo 0, flips 132-162/4995, canvas 0.31-0.59% | `pal_d1` 180, **canvas 76.7%** |
| C. brightness drift | geo 0-34, flips 181-461, canvas **2.85-8.07%** | **canvas 76.7%** |

Identical frames produce **zero** change except at the rebuild frame, so the
jitter is not code non-determinism -- it is "recompute everything every frame
from noisy pixels" plus the palette rebuild. Three causes, quantified:

**Cause 1 -- K-Means is unstable two different ways, giving a 76% flash every
10 frames (dominant).** Micro-test on the palette alone:

| condition | max palette delta |
|---|---|
| same data, no re-seed | **180** (global RNG -> different local optimum) |
| same data, re-seeded | 0 |
| different noisy frames, re-seeded | **144** (sigma=2 data -> a different local optimum wins) |

`camera_app` calls `cv2.setRNGSeed(0)` once at startup, so each rebuild draws a
different random init; and `kmeans_lab` runs 10 restarts picking the best, on
near-degenerate data, so even a re-seeded run flips local optimum when the
input moves by sigma=2. **Re-seeding alone is therefore not enough** -- that is
the key finding.

**Cause 2 -- no temporal coherence anywhere.** Partition, means and quantize
are recomputed every frame from noisy pixels, so noise moves near-tie split
decisions (3-5% of cells change layout per frame) and flips labels near palette
boundaries (~3% of cells per frame). Frozen geometry still leaves ~0.3-0.6% of
pixels changing per frame, i.e. **most of the per-frame crawl is colour, not
geometry**, but both contribute.

**Cause 3 -- camera auto-exposure/white-balance drift amplifies everything.**
A 1.5/255-per-frame brightness ramp raises per-frame canvas churn from ~0.5% to
2.9-8.1%, because many cell colours cross quantization boundaries. This is the
strongest "breathing" component on a real camera. Note this is a MODEL run
headless; the actual camera's drift was not measured (no camera in this
session).

Also worth stating: the `PaletteState` refresh cadence added earlier reduced the
rebuild FREQUENCY to 1-in-10 but not the MAGNITUDE, converting a per-frame
flash into a periodic one. Periodic large jumps read as more objectionable
than continuous small drift.

### Fix (A+B+C), implemented on this branch

- **A -- warm-start the palette** from the previous one
  (`cv2.KMEANS_USE_INITIAL_LABELS`, initial partition = current means assigned
  to the previous palette), so a rebuild refines the previous local optimum
  instead of jumping to a new one. Entry correspondence is preserved by
  construction (labels tie to previous centroids), which also prevents
  permutation. Code: `brick_color_rt.palette_kmeans_warm`, called by
  `PaletteState.get` on rebuild frames.
- **A' -- palette dead-band**: a warm-started rebuild that moves no entry by
  more than 2 grey levels is discarded (the previous palette is kept
  byte-for-byte). Without it a 1/255 refinement still flipped a few cells
  sitting exactly on a palette boundary; the dead-band makes a static scene
  byte-frozen.
- **B -- reuse the partition when the scene has not changed** (keyframe +
  mean-subtracted 64x48 grey signature, `reuse_thresh`): static frames keep the
  SAME leaves and triangle array, means are still re-extracted from the live
  pixels, and the partition cost is skipped. Code: `brick_temporal.TemporalState`,
  wired in `frame_pipeline.render_frame`.
- **C -- quantize hysteresis**: with geometry reused, triangle rows are stable,
  so a per-triangle label memory keeps the previous label unless a different
  palette entry is better by a relative margin (`--hysteresis`, default 0.1).
  Code: `brick_color_rt.quantize_nearest_bgr_sticky`.
- **D (deferred, real machine)** -- disable camera auto-exposure/auto-WB, or
  brightness-normalise, to attack cause 3 at the source. Not applicable
  headless.

### Measured result (A+B+C)

Assertions run by `task4_verify.py` (all pass); churn is the fraction of
pixels whose max channel change exceeds 8 grey levels, which ignores the
perceptually invisible 1-2/255 palette refinements:

| metric | before | after |
|---|---|---|
| palette delta on rebuild, identical input (cold vs warm) | **153** (RNG jump) | **1** |
| **steady per-frame churn, noise sigma=2** | **2.400%** | **0.064%** (37.7x less) |
| rebuild-frame churn, identical frames | 0.013% | **0.000%** |
| rebuild-frame churn, noise sigma=2 | 0.029% | **0.000%** |
| identical frames, temporal ON | - | 0 churn, 13/14 frames reused |
| app-level, static input, TOTAL ms (temporal OFF -> ON) | 68.9 | **27.5** |
| app-level, partition stage ms (temporal OFF -> ON) | 37.7 | **0.2** (reused) |

The 76.7%-of-pixels periodic flash is gone (rebuild is now a refinement, then a
no-op); the continuous crawl drops by ~38x on realistic sensor noise. The
app-level static benchmark also shows the reuse side benefit: the partition
stage disappears (37.7 -> 0.2 ms) and the whole frame is ~2.5x faster.

Correctness is preserved where it must be: with temporal coherence OFF the
step-4 gates still pass unchanged (partition bit-identical, triangles
vertex-exact, batched render and whole frame 0 differing pixels). Temporal
coherence is an explicit, declared behaviour change -- the mosaic intentionally
stops updating cell boundaries on a still scene -- and its defaults
(`reuse_thresh=1.0`, `hysteresis=0.1`, `palette_deadband=2.0`) still need
validation against a real camera's noise and exposure behaviour (no camera in
this session).

## Known limitations / TODOs

- **Level 1 baseline is not interactive** (~0.5 FPS). Declared, not hidden.
- No camera-index fallback yet: if index 0 is busy, the app exits with a message
  instead of trying other indices/backends.
- Frame-to-frame stability is not addressed yet. Re-running K-Means per frame
  will make the palette jump between frames even after the speed work, so
  "flicker reduction" is expected to become its own Level 4 item.
- Nothing in this file is measured on a second machine; the numbers are from one
  Windows box and one camera.
