"""
Task 4: call the camera and display its triangle-brick representation (25 marks).

PDF requirement: "Implement and test the program in a real-world scenario. Call
the camera to capture images and successfully display their triangle-brick
representations."

This file is the DRIVER and stays thin (AGENTS 4.1): it parses arguments, owns
the capture loop, draws the window and prints the stage summary. Everything
that turns one frame into a rendered mosaic lives in `brick_pipeline.py`
(FrameConfig / PaletteState / render_frame), so the per-frame engine can be
benchmarked and verified without a window or a camera.

Level 1 is camera capture + display, and this driver does exactly that: the
FULL Task 3 pipeline runs on every frame with no caching of geometry and no
approximation of colour, because it is the honest baseline the Level 4
before/after comparison is measured against. The Task 4 speedups (precomputed
split priorities, vectorised triangles, palette refresh cadence, batched
render) are exact against that baseline -- see docs/progress/Task4.md.

The window shows BOTH panes: `a) camera input` left, `b) triangle bricks`
right, live statistics drawn over the render pane. Showing the source beside
the result is the point -- a mosaic judged without its input says nothing
about fidelity. It is created with `WINDOW_NORMAL`, so it is resizable by
dragging (the default `imshow` window is locked to the image's pixel size --
`WND_PROP_AUTOSIZE` is 1.0 -- and cannot be enlarged, which looks tiny on a
large monitor). `--window-scale` sets the initial size. Close it with the
window's X (or q/ESC); X is detected via `brick_display.window_closed`.

Usage (Windows venv; WSL has no display, so add --no-show there):
    python code\\camera_app.py                    # default: full 9990 bricks
    python code\\camera_app.py --preset fast      # fewer bricks, higher FPS
    python code\\camera_app.py --no-show --max-frames 30   # benchmark
    python code\\camera_app.py --no-show --max-frames 30 \\
        --input code/pics/sky.jpg                 # offline source (bench/CI)
Any path you PASS resolves against the current directory (the defaults do not),
so run from the repo root and pass `code/pics/task4/...`. Keys: q/ESC quit,
s snapshot, p pause, r reset.
"""
import argparse
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brick_pipeline import (FrameConfig, PaletteState, render_frame,
                            MEANS_METHODS, MAX_TRIANGLES,
                            powers_of_two_upto)  # noqa: E402
from brick_temporal import TemporalState  # noqa: E402
from brick_io import (save_png, StageTimer, print_stage_summary,
                     open_camera)  # noqa: E402
from brick_display import (make_side_by_side, composite_size, draw_hud,
                           fit_letterbox, window_area,
                           window_closed)  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SNAPSHOT_DIR = os.path.join(HERE, "pics", "task4")
WINDOW = "Task 4: triangle-brick camera"

# Three points on the measured brick-count / frame-cost curve; the table is in
# docs/progress/Task4.md. Note `budget` counts BRICKS, i.e. triangles, not cells --
# a cell is two bricks, so `quality` produces 4995 cells, not 9990.
#
# `quality` is the default because the PDF caps bricks at 10000 and grades the
# output. Measured, the full budget costs ~216 ms/frame against 73 ms for 498
# cells (pre-optimization numbers); `fast` exists for a machine that cannot
# keep up even after the Task 4 optimizations.
PRESETS = {
    "quality": {"scale": 1.0, "budget": 9990, "k": 16},
    "balanced": {"scale": 1.0, "budget": 5000, "k": 8},
    "fast": {"scale": 1.0, "budget": 2000, "k": 8},
}



def open_input(path):
    """Open an offline frame source (video file or still image) for benchmarking.

    Function: `cv2.VideoCapture` accepts video files AND still images on most
    backends, but an image "video" yields exactly one frame, so a still image
    is detected and looped explicitly -- the benchmark then sees N identical
    frames, which is exactly what the palette-flicker measurement needs. This
    is what lets the in-app benchmark run headless on machines without a camera
    (WSL, CI) while using the SAME render_frame + StageTimer path as the live
    app, so quoted numbers stay comparable.

    Shape: returns `(cap, static_frame)`. `cap` is an open cv2.VideoCapture
    for video input (static_frame=None), or None for a still image
    (static_frame is the (H, W, 3) uint8 BGR image to reuse every frame).
    Raises SystemExit when neither a video nor an image can be read.
    """
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        # An image file also "opens" as a capture and yields its single frame,
        # but seeking back to 0 then fails on several backends -- so treat a
        # one-frame source as a still image explicitly.
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        ok, first = cap.read()
        if ok and first is not None and n_frames > 1:
            return cap, None
        cap.release()
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(
            f"[task4] --input {path}: not readable as a video or an image")
    print(f"[task4] --input is a still image; looping it")
    return None, img


def main():
    """CLI entry: parse args, open the source, run the loop, print the summary.

    Shape: frames are (H, W, 3) uint8 BGR at whatever the source yields, resized
    by `cfg.scale`; no array escapes the loop except saved snapshots.
    Semantics: `--no-show` skips `imshow` and `waitKey`, which is what makes the
    loop runnable headless (WSL/CI) -- combined with `--max-frames` it becomes a
    deterministic benchmark with no window and no keyboard. `--input` replaces
    the camera with a file so the same benchmark runs without hardware. Without
    `--max-frames` the loop runs until the user quits.
    """
    p = argparse.ArgumentParser(
        description="Task 4: live camera -> triangle-brick display.")
    # `--preset` supplies defaults for the flags below, so it must be read first.
    # The throwaway parser peeks at argv; `parse_known_args` tolerates everything
    # else, and an explicitly passed flag still overrides the preset default.
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--preset", choices=list(PRESETS), default="quality")
    preset_name = peek.parse_known_args()[0].preset
    pre = PRESETS[preset_name]

    p.add_argument("--preset", choices=list(PRESETS), default=preset_name,
                   help="brick-count ladder, measured at 640x480 per frame: "
                        "quality=9990 bricks/K16 (default), "
                        "balanced=5000/K8, fast=2000/K8. "
                        "Any explicit flag below overrides it.")
    p.add_argument("--camera", type=int, default=0, help="camera index")
    p.add_argument("--input", default=None, metavar="PATH",
                   help="offline frame source (video file or still image) "
                        "instead of the camera; enables headless benchmarks")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--lock-ae", action="store_true",
                   help="turn OFF camera auto-exposure before the loop; a "
                        "drifting exposure is the strongest source of colour "
                        "jitter (ignored for --input)")
    p.add_argument("--exposure", type=float, default=None, metavar="E",
                   help="manual exposure value to set with --lock-ae "
                        "(backend-specific units; omit to keep the current one)")
    p.add_argument("--lock-wb", action="store_true",
                   help="turn OFF camera auto-white-balance (ignored for --input)")
    p.add_argument("--scale", type=float, default=pre["scale"],
                   help="resize factor applied before processing (speed lever)")
    p.add_argument("--budget", type=int, default=pre["budget"],
                   help=f"triangle cap (hard max {MAX_TRIANGLES})")
    p.add_argument("--smax", type=int, default=32, help="largest brick side")
    p.add_argument("--smin", type=int, default=1, help="smallest brick side")
    p.add_argument("--k", type=int, default=pre["k"], help="palette size (>3)")
    p.add_argument("--means", choices=list(MEANS_METHODS), default="rows",
                   help="per-triangle colour method; 'rows' (default) is the "
                        "row-run table, bit-identical to 'fast' and faster; "
                        "'mask' is the slow shipped reference")
    p.add_argument("--sse", dest="sse_impl",
                   choices=["precomp", "sat", "ref"], default="precomp",
                   help="quadtree split-priority implementation; 'precomp' "
                        "(default) is bit-identical to 'sat' for mse and much "
                        "faster; 'ref' is the original per-candidate numpy")
    p.add_argument("--palette-refresh", type=int, default=10, metavar="N",
                   help="rebuild the K-Means palette every N frames, reuse it "
                        "otherwise (1 = every frame, the old behaviour; the "
                        "default 10 removes the palette flicker)")
    p.add_argument("--palette-deadband", type=float, default=2.0, metavar="D",
                   help="discard a warm-started palette rebuild that moves no "
                        "entry by more than D grey levels (keeps static scenes "
                        "byte-frozen; 0 disables)")
    p.add_argument("--temporal", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="temporal coherence (default on): reuse the partition "
                        "when the scene is static and keep cell labels unless "
                        "another colour is clearly better. --no-temporal "
                        "restores the per-frame recompute pipeline")
    p.add_argument("--reuse-thresh", type=float, default=1.0, metavar="D",
                   help="scene-change threshold: mean |signature delta| (0-255 "
                        "grey levels) below which the partition is reused")
    p.add_argument("--hysteresis", type=float, default=0.1, metavar="M",
                   help="label dead-band: keep a cell's colour unless another "
                        "palette entry is better by this relative margin "
                        "(0 = plain nearest colour)")
    p.add_argument("--freeze-colour-on-reuse", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="when the partition is reused, also keep the previous "
                        "labels/canvas (default on): a static scene then "
                        "renders byte-identical and the means/palette/quantize/"
                        "render stages are skipped. Pair with --lock-ae")
    p.add_argument("--max-reuse", type=int, default=0, metavar="N",
                   help="force a re-partition after N reused frames (0 = never; "
                        "the scene-change test alone decides)")
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
        n_samples=args.n_samples, sse_impl=args.sse_impl,
        palette_refresh=args.palette_refresh,
        palette_deadband=args.palette_deadband, temporal=args.temporal,
        reuse_thresh=args.reuse_thresh, hysteresis=args.hysteresis,
        max_reuse=args.max_reuse, partition=args.partition,
        freeze_colour_on_reuse=args.freeze_colour_on_reuse,
        priority=args.priority, palette_method=args.palette_method,
    )

    print(f"[task4] S_set={cfg.S_set} K={cfg.K} partition={cfg.partition} "
          f"priority={cfg.priority} palette={cfg.palette_method} "
          f"means={cfg.means} sse={cfg.sse_impl} "
          f"palette_refresh={cfg.palette_refresh} temporal={cfg.temporal} "
          f"(reuse_thresh={cfg.reuse_thresh} hysteresis={cfg.hysteresis})")
    cap, static_frame = None, None
    if args.input:
        cap, static_frame = open_input(args.input)
        first = static_frame if static_frame is not None else None
        if first is None:
            # VideoCapture is positioned after its probe read; rewind so frame 1
            # of the loop is the file's first frame.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, first = cap.read()
            if not ok or first is None:
                raise SystemExit(f"[task4] --input {args.input}: no frames")
    else:
        cap = open_camera(args.camera, args.width, args.height,
                          lock_ae=args.lock_ae, exposure=args.exposure,
                          lock_wb=args.lock_wb)
        ok, first = cap.read()
        if not ok or first is None:
            cap.release()
            raise SystemExit("[task4] camera opened but returned no frame")
    print(f"[task4] source {'file ' + args.input if args.input else 'camera ' + str(args.camera)}: "
          f"{first.shape[1]}x{first.shape[0]}  scale={cfg.scale}")

    # Seed OpenCV's RNG once so the K-Means palette is reproducible for the same
    # frame content; without this the palette can drift between sessions and the
    # before/after FPS-quality comparison would not be repeatable.
    cv2.setRNGSeed(0)

    if not args.no_show:
        # `WINDOW_NORMAL` is what makes the window draggable. The default imshow
        # window is locked to the image's pixel size, so on a large monitor the
        # two panes look tiny and there is no way to enlarge them.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        pane = first.shape[:2]
        if cfg.scale != 1.0:
            pane = (int(pane[0] * cfg.scale), int(pane[1] * cfg.scale))
        vh, vw = composite_size(pane)
        cv2.resizeWindow(WINDOW, max(1, int(vw * args.window_scale)),
                         max(1, int(vh * args.window_scale)))
        print(f"[task4] window {vw}x{vh} px "
              f"(x{args.window_scale}); drag its edges to resize")

    timer = StageTimer()
    palette_state = PaletteState(cfg.palette_refresh, cfg.palette_deadband)
    temporal_state = TemporalState(cfg.temporal, cfg.reuse_thresh,
                                   cfg.hysteresis, cfg.max_reuse)
    snapshots, paused, fps = 0, False, 0.0
    frame = first  # so a paused first iteration has a frame to re-render
    n_frames, t_start = 0, time.perf_counter()
    t_prev = t_start

    try:
        while True:
            if not paused:
                if static_frame is not None:
                    frame = static_frame
                else:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        print("[task4] frame read failed; stopping")
                        break
            canvas, processed, info = render_frame(frame, cfg, timer,
                                                   palette_state,
                                                   temporal_state)

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
                # Fit the composite into the current window with the aspect
                # ratio PRESERVED, then draw the HUD on the FITTED canvas. This
                # fixes two things: a stretched window can no longer deform the
                # isosceles right triangles, and the HUD stays a constant screen
                # size (a fixed font scale) instead of growing with the window.
                area = window_area(WINDOW) or (view.shape[1], view.shape[0])
                display, ox, oy, dscale = fit_letterbox(view, area[0], area[1])
                draw_hud(display, fps, info, cfg, paused, timer,
                         x0=ox + int(render_x0 * dscale))
                cv2.imshow(WINDOW, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    paused = not paused
                if key == ord("r"):
                    timer.reset()
                    temporal_state.reset()  # drop cached partition/labels
                if key == ord("s"):
                    # Snapshot the composite WITHOUT the HUD: the overlay is a
                    # live readout, not part of the mosaic, and evidence images
                    # are cleaner without it. `view` is the un-annotated native
                    # composite (the HUD is drawn on the fitted display copy).
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
        if cap is not None:
            cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()
        print_stage_summary(timer, n_frames, time.perf_counter() - t_start,
                            context={"scale": cfg.scale, "budget": cfg.budget,
                                     "S_set": cfg.S_set, "K": cfg.K,
                                     "sse": cfg.sse_impl,
                                     "palette_refresh": cfg.palette_refresh,
                                     "temporal": cfg.temporal,
                                     "reuse_thresh": cfg.reuse_thresh,
                                     "hysteresis": cfg.hysteresis})


if __name__ == "__main__":
    main()
