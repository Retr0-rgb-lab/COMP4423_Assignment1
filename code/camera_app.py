"""
Task 4: call the camera and display its triangle-brick representation (25 marks).

PDF requirement: "Implement and test the program in a real-world scenario. Call
the camera to capture images and successfully display their triangle-brick
representations."

Level 1 is camera capture + display, and this driver does exactly that: the FULL
Task 3 pipeline runs on every frame with no caching and no approximation, because
it is the honest baseline the Level 4 before/after comparison is measured against.

The window shows BOTH panes: `a) camera input` left, `b) triangle bricks` right,
live statistics drawn over the render pane. Showing the source beside the result
is the point -- a mosaic judged without its input says nothing about fidelity.
It is created with `WINDOW_NORMAL`, so it is resizable by dragging (the default
`imshow` window is locked to the image's pixel size -- `WND_PROP_AUTOSIZE` is 1.0
-- and cannot be enlarged, which looks tiny on a large monitor). `--window-scale`
sets the initial size. Close it with the window's X (or q/ESC); X is detected via
`brick_display.window_closed`.

Per frame: read -> (resize) -> partition -> triangles -> means -> palette ->
quantize -> render -> compose -> HUD. Stages are timed by `brick_io.StageTimer`
and the medians are printed at exit, so HUD and report share one measurement path
(see docs/progress/Task4.md).

Usage (Windows venv; WSL has no display, so add --no-show there):
    python code\\camera_app.py --preset watch      # default, smooth preview
    python code\\camera_app.py --preset quality    # full Task 3 config, ~0.45 FPS
    python code\\camera_app.py --no-show --max-frames 30   # benchmark
Any path you PASS resolves against the current directory (the defaults do not), so
run from the repo root and pass `code/pics/task4/...`. Keys: q/ESC quit,
s snapshot, p pause, r reset.
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
from brick_render import render_triangles  # noqa: E402
from brick_means import extract_means, METHODS as MEANS_METHODS  # noqa: E402
from brick_io import (save_png, StageTimer, print_stage_summary,
                     open_camera)  # noqa: E402
from brick_display import (make_side_by_side, composite_size, draw_hud,
                           window_closed)  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SNAPSHOT_DIR = os.path.join(HERE, "pics", "task4")
N_TRI_BUDGET = 9990  # 10-triangle margin under the 10000 cap, same as Task 3
WINDOW = "Task 4: triangle-brick camera"

# `quality` is the config whose per-stage cost is recorded in
# docs/progress/Task4.md. `watch` is the watchable one, and it lowers ONLY the
# brick count -- deliberately not the resolution, because `--scale` shrinks both
# panes of the side-by-side view and a small picture defeats the point of showing
# the source next to the render. Named presets stop the report's baseline numbers
# from drifting when a viewing default gets retuned.
PRESETS = {
    "watch": {"scale": 1.0, "budget": 1000, "k": 8},
    "quality": {"scale": 1.0, "budget": N_TRI_BUDGET, "k": 16},
}


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
      means    : "mask" | "fast" | "sample" -- colour extraction method. "mask"
                 is the slow shipped reference (it also averages a fringe of
                 neighbouring pixels); "fast" is exact over the triangle's own
                 pixels; "sample" approximates. See brick_means for measurements.
      partition/priority/palette_method : names dispatched inside the engine
                 modules; an unknown value raises there rather than silently
                 falling back (see brick_quadtree / brick_color).
    """

    S_set: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    K: int = 16
    budget: int = N_TRI_BUDGET
    scale: float = 1.0
    means: str = "fast"
    n_samples: int = 9
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
    Output: `(canvas, processed, info)`.
      canvas    : (H, W, 3) uint8 BGR -- the rendered mosaic, same size as the
                  processed input (cropped back from the padded canvas).
      processed : (H, W, 3) uint8 BGR -- the frame AFTER `--scale` resizing, i.e.
                  exactly what the pipeline saw. The side-by-side view needs this
                  rather than the raw camera frame: at `--scale 0.5` the raw frame
                  is twice the size of the render, and pairing them would misalign
                  the panes.
      info      : per-frame statistics for the HUD and the exit summary --
                  `n_tri`, `n_cells`, `sizes` (cell side -> CELL count),
                  `palette`, and `t_resize_ms`.
    """
    t_resize = time.perf_counter()
    if cfg.scale != 1.0:
        frame = cv2.resize(frame, None, fx=cfg.scale, fy=cfg.scale,
                           interpolation=cv2.INTER_AREA)
    t_resize = (time.perf_counter() - t_resize) * 1000.0
    processed = frame

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
        means = extract_means(padded, leaves, cfg.means, cfg.n_samples)

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
    return canvas, processed, info


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
    # `--preset` supplies defaults for the flags below, so it must be read first.
    # The throwaway parser peeks at argv; `parse_known_args` tolerates everything
    # else, and an explicitly passed flag still overrides the preset default.
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--preset", choices=list(PRESETS), default="watch")
    preset_name = peek.parse_known_args()[0].preset
    pre = PRESETS[preset_name]

    p.add_argument("--preset", choices=list(PRESETS), default=preset_name,
                   help="'watch' = smooth preview (reduced resolution and brick "
                        "budget); 'quality' = the full Task 3 configuration, "
                        "~0.45 FPS. Any explicit flag below overrides it.")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--scale", type=float, default=pre["scale"],
                   help="resize factor applied before processing (speed lever)")
    p.add_argument("--budget", type=int, default=pre["budget"],
                   help=f"triangle cap (hard max {MAX_TRIANGLES})")
    p.add_argument("--smax", type=int, default=32, help="largest brick side")
    p.add_argument("--smin", type=int, default=1, help="smallest brick side")
    p.add_argument("--k", type=int, default=pre["k"], help="palette size (>3)")
    p.add_argument("--means", choices=list(MEANS_METHODS), default="fast",
                   help="per-triangle colour method; 'mask' is the slow shipped")
    p.add_argument("--n-samples", type=int, default=9,
                   help="interior points per triangle when --means sample")
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
    p.add_argument("--window-scale", type=float, default=1.0,
                   help="initial window size as a multiple of the composite's "
                        "natural pixel size; the window is resizable by dragging")
    p.add_argument("--no-show", action="store_true",
                   help="headless: no window, no keyboard")
    args = p.parse_args()

    if args.budget > MAX_TRIANGLES:
        raise SystemExit(f"--budget {args.budget} exceeds the {MAX_TRIANGLES} cap")

    cfg = FrameConfig(
        S_set=powers_of_two_upto(args.smax, args.smin),
        K=args.k, budget=args.budget, scale=args.scale, means=args.means,
        n_samples=args.n_samples, partition=args.partition,
        priority=args.priority, palette_method=args.palette_method,
    )

    print(f"[task4] S_set={cfg.S_set} K={cfg.K} partition={cfg.partition} "
          f"priority={cfg.priority} palette={cfg.palette_method} "
          f"means={cfg.means}")
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

    if not args.no_show:
        # `WINDOW_NORMAL` is what makes the window draggable. The default imshow
        # window is locked to the image's pixel size, so on a large monitor the
        # two panes look tiny and there is no way to enlarge them.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        pane = frame.shape[:2]
        if cfg.scale != 1.0:
            pane = (int(pane[0] * cfg.scale), int(pane[1] * cfg.scale))
        vh, vw = composite_size(pane)
        cv2.resizeWindow(WINDOW, max(1, int(vw * args.window_scale)),
                         max(1, int(vh * args.window_scale)))
        print(f"[task4] window {vw}x{vh} px "
              f"(x{args.window_scale}); drag its edges to resize")

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
            canvas, processed, info = render_frame(frame, cfg, timer)

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

            # The window shows the camera frame and this project's render side by
            # side: the pair is the deliverable, because a mosaic judged without
            # its source tells the viewer nothing about fidelity.
            view, render_x0 = make_side_by_side(processed, canvas)

            if args.save_render:
                # Save all three on purpose: the two panes are pixel-aligned with
                # each other and with the composite, so the report can use either
                # the pair or the single figure without re-capturing.
                save_png(os.path.join(args.save_render, "frame0001_input.png"),
                         processed)
                save_png(os.path.join(args.save_render, "frame0001_render.png"),
                         canvas)
                save_png(os.path.join(args.save_render, "frame0001_compare.png"),
                         view)
                print(f"[task4] saved input+render+compare -> {args.save_render}")
                break

            if not args.no_show:
                # HUD goes over the RENDER pane (`render_x0`), not the middle of
                # the composite, so it never straddles the separator.
                draw_hud(view, fps, info, cfg, paused, timer, x0=render_x0)
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    paused = not paused
                if key == ord("r"):
                    timer.reset()
                if key == ord("s"):
                    # Snapshot what is ON SCREEN (the composite), so the saved
                    # image is exactly the evidence the viewer just looked at.
                    path = os.path.join(args.snapshot_dir,
                                        f"snap_{snapshots:03d}.png")
                    save_png(path, view)
                    print(f"[task4] snapshot -> {path}")
                    snapshots += 1
                # Clicking the window's close button destroys it behind our back;
                # this is how we notice and exit cleanly instead of drawing into
                # a window that no longer exists.
                if window_closed(WINDOW):
                    print("[task4] window closed by user")
                    break

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
        print_stage_summary(timer, n_frames, time.perf_counter() - t_start,
                            context={"scale": cfg.scale, "budget": cfg.budget,
                                     "S_set": cfg.S_set, "K": cfg.K})


if __name__ == "__main__":
    main()
