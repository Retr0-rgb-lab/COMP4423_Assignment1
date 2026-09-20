"""
Task 4: call the camera and display its triangle-brick representation (25 marks).

PDF requirement: "Implement and test the program in a real-world scenario. Call
the camera to capture images and successfully display their triangle-brick
representations."

Level 1 of Task 4 is: camera capture + display of the triangle-brick result. This
driver does exactly that and nothing more, on purpose -- it is the honest
baseline that the Level 4 before/after comparison is measured against, so it
runs the FULL Task 3 pipeline on every frame with no caching and no
approximation. Expect it to be slow (~0.45 FPS at 640x480, 9990 triangles); that
slow number is the deliverable, not a bug.

Per frame: read -> (resize) -> partition -> triangles -> means -> palette ->
quantize -> render -> HUD. Each stage is timed with `brick_io.StageTimer` and the
per-stage median is printed at exit, so the HUD and the report's before/after
table come from ONE measurement path (see docs/progress/Task4.md).

Defaults mirror the Task 3 "best config" (quadtree, S_max=32, K=16, kmeans_lab,
Delta-MSE priority), so the report can say Task 4 drives the same pipeline from a
camera instead of a file.

Usage (Windows venv; WSL has no display, so use --no-show there):
    python code\\camera_app.py                            # live window
    python code\\camera_app.py --scale 0.5 --budget 3000
    python code\\camera_app.py --no-show --max-frames 30   # benchmark
Note: the built-in defaults are `__file__`-relative, but any path you PASS is
resolved against the current directory. Run from the repo root and pass
`code/pics/task4/...` to keep images where AGENTS 1.1 requires them.

Keys: q or ESC quit, s snapshot, p pause, r reset the timer.
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brick_geom import pad_to_max, leaves_to_triangles, MAX_TRIANGLES  # noqa: E402
from brick_quadtree import quadtree_partition  # noqa: E402
from brick_region_merge import region_merge_partition  # noqa: E402
from brick_color import build_palette, quantize_nearest_bgr  # noqa: E402
from brick_render import triangle_means_bgr, render_triangles  # noqa: E402
from brick_io import save_png, StageTimer  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SNAPSHOT_DIR = os.path.join(HERE, "pics", "task4")
N_TRI_BUDGET = 9990  # 10-triangle margin under the 10000 cap, same as Task 3
WINDOW = "Task 4: triangle-brick camera"

# BGR colours for the HUD; both are dark so white text stays readable over any
# frame content (the mosaic can be any colour, including white).
HUD_BG = (32, 32, 32)
HUD_FG = (255, 255, 255)


@dataclass
class FrameConfig:
    """Settings for the live pipeline; one instance for the whole session.

    Shape/type contract (this is pure config, no arrays):
      S_set    : list[int] -- allowed cell sizes, powers of two, ascending.
                 `max(S_set)` is the largest brick side (the Task 3 S_max) and
                 `min(S_set)` the smallest; both come from `--smax` / `--smin`.
      K        : int -- palette size; > 3 is the Task 3 requirement.
      budget   : int -- triangle cap passed to the partitioner. Lowering it is a
                 legitimate real-time lever, but the value used must be declared
                 in the report.
      scale    : float -- resize factor applied to the camera frame before
                 processing. 1.0 = process at native resolution.
      partition/priority/palette_method : names dispatched inside the engine
                 modules; an unknown value raises there rather than silently
                 falling back (see brick_quadtree / brick_color).
    """

    S_set: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    K: int = 16
    budget: int = N_TRI_BUDGET
    scale: float = 1.0
    partition: str = "quadtree"
    priority: str = "mse"
    palette_method: str = "kmeans_lab"


def powers_of_two_upto(smax, smin=1):
    """[smin, 2*smin, ..., smax] as the sorted power-of-two family.

    Function: builds the S_set the partitioners require. The quadtree splits by
    halving and region_merge validates `S_max/S_min` is a power of two, so a
    non-power-of-two input would otherwise fail deep inside the engine.

    Shape: ints in, `list[int]` out, ascending, always containing `smin` and the
    largest power of two <= `smax`. Semantics: element i is the cell side in
    pixels; `out[-1]` is the largest brick size this session can produce.
    """
    if smin < 1 or smax < smin:
        raise ValueError(f"need 1 <= smin <= smax, got smin={smin} smax={smax}")
    out, s = [], smin
    while s <= smax:
        out.append(s)
        s *= 2
    return out


def open_camera(index, width=None, height=None):
    """Open camera `index`, optionally forcing the capture resolution.

    Function: `cv2.VideoCapture` with the DirectShow backend on Windows (MSMF
    often refuses to apply a requested size on laptops). Raises rather than
    returning a closed handle, so a busy or absent device fails loudly.

    Shape: returns a `cv2.VideoCapture` whose frames are (H, W, 3) uint8 BGR.
    Semantics: `width`/`height`, when given, are REQUESTS -- the driver may
    ignore them, so the caller must read back the actual size from the frames
    instead of assuming.
    """
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {index}. Try another --camera N, or close "
            f"the app that is holding the device.")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def render_frame(frame, cfg, timer):
    """Run the full Task 3 pipeline on one frame; returns (canvas, info).

    Function
    --------
    Same seven steps as `triangle_brick_task3.run_experiment`, minus the metric
    suite (metrics are far too slow for a live loop and are not needed to
    display). No step is cached or approximated in this version.

    Shapes
    ------
    Input: `frame` is (H, W, 3) uint8 BGR -- a single camera frame.
    Intermediate: `padded` is (Hp, Wp, 3) reflect-padded to a multiple of
    `max(S_set)`; `means` is (T, 3) float32 with `[..., 0]=B`; `palette` is
    (K, 3) uint8 BGR; `labels` is (T,) uint8 indexing `palette`.
    Output: `(canvas (H, W, 3) uint8 BGR, info dict)`. `info` carries the
    per-frame statistics the HUD and the exit summary read: `n_tri`,
    `n_cells`, `sizes` (cell side -> CELL count), `palette`, and `t_resize_ms`.
    """
    t_resize = time.perf_counter()
    if cfg.scale != 1.0:
        frame = cv2.resize(frame, None, fx=cfg.scale, fy=cfg.scale,
                           interpolation=cv2.INTER_AREA)
    t_resize = (time.perf_counter() - t_resize) * 1000.0

    with timer.stage("partition"):
        if cfg.partition == "quadtree":
            leaves, padded_shape, orig_shape = quadtree_partition(
                frame, cfg.S_set, cfg.budget, cfg.priority)
        else:
            leaves, padded_shape, orig_shape = region_merge_partition(
                frame, cfg.S_set, cfg.budget)

    with timer.stage("triangles"):
        triangles = leaves_to_triangles(leaves)
        padded, _, _ = pad_to_max(frame, max(cfg.S_set))

    with timer.stage("means"):
        means = triangle_means_bgr(padded, triangles)

    with timer.stage("palette"):
        palette = build_palette(means, cfg.K, cfg.palette_method)

    with timer.stage("quantize"):
        labels = quantize_nearest_bgr(means, palette)

    with timer.stage("render"):
        canvas = render_triangles(padded_shape, triangles, labels, palette,
                                  orig_shape)

    sizes = {}
    for (_, _, s) in leaves:
        sizes[s] = sizes.get(s, 0) + 1  # cells, not triangles
    info = {
        "n_tri": len(leaves) * 2,
        "n_cells": len(leaves),
        "sizes": sizes,
        "palette": palette,
        "t_resize_ms": t_resize,
    }
    return canvas, info


def draw_hud(canvas, fps, info, cfg, paused, timer):
    """Draw the FPS / config / per-stage HUD onto `canvas` in place.

    Function: the HUD is how a real-time claim is verified on screen rather than
    asserted. It shows the measured FPS and the median cost of each pipeline
    stage so a viewer can see WHERE the time goes, not just that it is slow.

    Shape: `canvas` is (H, W, 3) uint8 BGR, modified in place (no copy). The
    panel height grows with the number of stage lines; it is drawn over the
    top-left corner and short text is left-padded so nothing is clipped.
    Semantics: `fps` is the caller's rolling estimate; `timer.report()` supplies
    the stage medians, so the numbers shown match the exit summary exactly.
    """
    rep = timer.report()
    lines = [
        f"FPS {fps:5.1f}   {'PAUSED' if paused else 'live'}",
        f"tris {info['n_tri']:5d}/{cfg.budget}   cells {info['n_cells']:5d}",
        f"{cfg.partition} S_max={max(cfg.S_set)} K={cfg.K} "
        f"{cfg.palette_method}",
        "sizes " + " ".join(f"{s}:{n}" for s, n in sorted(info["sizes"].items())),
        "--- median ms per stage ---",
    ]
    for name in ["partition", "triangles", "means", "palette", "quantize",
                 "render"]:
        if name in rep:
            lines.append(f"  {name:<10s} {rep[name]['median_ms']:8.1f}")
    lines.append(f"  {'frame':<10s} {rep['__total__']['median_ms']:8.1f}")

    y = 22
    for text in lines:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (6, y - th - 4), (14 + tw, y + 4), HUD_BG, -1)
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    HUD_FG, 1, cv2.LINE_AA)
        y += th + 9
    return canvas


def print_summary(timer, cfg, n_frames, wall_s):
    """Print the exit summary: effective FPS plus median ms per stage.

    Function: this is the measurement that Task 4's Level 4 table quotes. It
    prints the same `StageTimer.report()` the HUD reads, so screen and log agree.

    Shape: no arrays. Semantics: `frames/s` is the end-to-end rate including the
    camera read, which the per-stage rows do NOT include -- that is why the
    reported FPS is always a little below `1000/median_frame_ms`.
    """
    rep = timer.report()
    wall_exec = max(wall_s, 1e-6)
    print(f"\n=== Task 4 run summary ===")
    print(f"  frames={n_frames}  wall={wall_s:.1f}s  "
          f"effective={n_frames / wall_exec:.2f} FPS")
    print(f"  resolution scale={cfg.scale}  budget={cfg.budget}  "
          f"S_set={cfg.S_set}  K={cfg.K}")
    print(f"  {'stage':<12s} {'n':>5s} {'median_ms':>10s} {'mean_ms':>10s}")
    for name in ["partition", "triangles", "means", "palette", "quantize",
                 "render"]:
        if name in rep:
            print(f"  {name:<12s} {rep[name]['n']:5d} "
                  f"{rep[name]['median_ms']:10.1f} {rep[name]['mean_ms']:10.1f}")
    print(f"  {'TOTAL':<12s} {'':>5s} "
          f"{rep['__total__']['median_ms']:10.1f} "
          f"{rep['__total__']['mean_ms']:10.1f}   (excludes camera read)")


def main():
    """CLI entry: parse args, open the camera, run the loop, print the summary.

    Shape: frames are (H, W, 3) uint8 BGR at whatever the camera yields, resized
    by `cfg.scale`; no array escapes the loop except saved snapshots.
    Semantics: `--no-show` skips `imshow` and `waitKey`, which is what makes the
    loop runnable headless (WSL/CI) -- combined with `--max-frames` it becomes a
    deterministic benchmark with no window and no keyboard. Without
    `--max-frames` the loop runs until the user quits.
    """
    p = argparse.ArgumentParser(
        description="Task 4: live camera -> triangle-brick display.")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--scale", type=float, default=1.0,
                   help="resize factor applied before processing (speed lever)")
    p.add_argument("--budget", type=int, default=N_TRI_BUDGET,
                   help=f"triangle cap (default {N_TRI_BUDGET}; hard max "
                        f"{MAX_TRIANGLES})")
    p.add_argument("--smax", type=int, default=32, help="largest brick side")
    p.add_argument("--smin", type=int, default=1, help="smallest brick side")
    p.add_argument("--k", type=int, default=16, help="palette size (>3)")
    p.add_argument("--partition", choices=["quadtree", "region_merge"],
                   default="quadtree")
    p.add_argument("--priority", choices=["mse", "edgef1"], default="mse")
    p.add_argument("--palette", dest="palette_method",
                   choices=["kmeans_lab", "kmeans_rgb", "median_cut"],
                   default="kmeans_lab")
    p.add_argument("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR,
                   help="where 's' writes snapshots")
    p.add_argument("--save-render", metavar="DIR", default=None,
                   help="save frame 1 as <DIR>/frame0001_{input,render}.png and "
                        "exit; works headless (how Task 4's evidence is made)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after N frames (0 = run until quit); for benchmarks")
    p.add_argument("--no-show", action="store_true",
                   help="headless: no window, no keyboard")
    args = p.parse_args()

    if args.budget > MAX_TRIANGLES:
        raise SystemExit(f"--budget {args.budget} exceeds the {MAX_TRIANGLES} cap")

    cfg = FrameConfig(
        S_set=powers_of_two_upto(args.smax, args.smin),
        K=args.k, budget=args.budget, scale=args.scale,
        partition=args.partition, priority=args.priority,
        palette_method=args.palette_method,
    )

    print(f"[task4] S_set={cfg.S_set} K={cfg.K} partition={cfg.partition} "
          f"priority={cfg.priority} palette={cfg.palette_method}")
    cap = open_camera(args.camera, args.width, args.height)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise SystemExit("[task4] camera opened but returned no frame")
    print(f"[task4] camera {args.camera}: {frame.shape[1]}x{frame.shape[0]}  "
          f"scale={cfg.scale}")

    # Seed OpenCV's RNG once so the K-Means palette is reproducible for the same
    # frame content; without this the palette can drift between sessions and the
    # before/after FPS-quality comparison would not be repeatable.
    cv2.setRNGSeed(0)

    timer = StageTimer()
    snapshots, paused, fps = 0, False, 0.0
    n_frames, t_start = 0, time.perf_counter()
    t_prev = t_start

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[task4] frame read failed; stopping")
                    break
            canvas, info = render_frame(frame, cfg, timer)

            # FPS is the FULL loop period (read + process + display), so the
            # timestamp must be taken AFTER the work, not right after `read`.
            # Taking it after `read` would report the camera's read rate
            # (~30 FPS) regardless of how slow the pipeline is -- which is the
            # single most misleading number this HUD could show.
            t_now = time.perf_counter()
            dt = max(t_now - t_prev, 1e-6)
            t_prev = t_now
            # IIR smoothing so the HUD does not flicker while still tracking a
            # real change within a few frames.
            fps = 0.8 * fps + 0.2 * (1.0 / dt) if fps else 1.0 / dt

            n_frames += 1
            if args.save_render:
                # Save the ORIGINAL frame next to the render on purpose: the
                # report's "real-world scenario" figures need the before/after
                # pair, and a render alone cannot show what the bricks replaced.
                # Input is the resized frame actually processed, so the two
                # images are pixel-aligned and directly comparable.
                processed = frame
                save_png(os.path.join(args.save_render, "frame0001_input.png"),
                         processed)
                save_png(os.path.join(args.save_render, "frame0001_render.png"),
                         canvas)
                print(f"[task4] saved input+render -> {args.save_render}")
                break
            if not args.no_show:
                draw_hud(canvas, fps, info, cfg, paused, timer)
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    paused = not paused
                if key == ord("r"):
                    timer.reset()
                if key == ord("s"):
                    path = os.path.join(args.snapshot_dir,
                                        f"snap_{snapshots:03d}.png")
                    save_png(path, canvas)
                    print(f"[task4] snapshot -> {path}")
                    snapshots += 1

            if n_frames % 10 == 1 or args.max_frames:
                print(f"  frame {n_frames:4d}  tris={info['n_tri']:5d}  "
                      f"fps={fps:5.2f}  "
                      f"median_frame={timer.report()['__total__']['median_ms']:.0f} ms")
            if args.max_frames and n_frames >= args.max_frames:
                break
    finally:
        cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        print_summary(timer, cfg, n_frames, time.perf_counter() - t_start)


if __name__ == "__main__":
    main()
