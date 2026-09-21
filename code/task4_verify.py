"""
task4_verify — correctness assertions + paired benchmarks for optimizations A/B/C/D.

Runs headless (no camera, no window): it drives `frame_pipeline.render_frame`
and the brick_* primitives directly on synthetic 640x480 frames. This is the
verification gate the Task 4 record quotes: every optimization must pass its
exactness assertion BEFORE its speed number is allowed into
docs/progress/Task4.md.

Checks (assertion -> what would falsify it):
  1. precomp partition leaves == sat partition leaves, sorted (bit-identical
     geometry -- if this fails, optimization B changed the mosaic).
  2. leaves_to_triangles_array == leaves_to_triangles, vertex by vertex
     (D1 must not permute the ordering contract with the means rows).
  3. batched render vs reference render on identical inputs: differing-pixel
     count reported (expected: only 1-px fillPoly fringe near shared edges,
     border pass repaints exact boundaries) + Delta-E2000/PSNR of BOTH against
     the source frame (A must not move the quality metrics).
  4. palette cache: 30 identical frames -> palette byte-identical between
     rebuilds, rebuild happens exactly every `refresh` frames (C kills the
     recorded flicker).
  5. end-to-end: old path (sse=sat, list triangles, per-frame palette,
     reference render) vs new path on the SAME frame -> pixel diff + quality
     metrics of both vs source.

Benchmarks (paired, in-process, best of --reps, per the measurement-hygiene
rules in Task4.md: ratios from paired runs, not single absolute FPS):
  B: partition sat vs precomp (stage-isolated)
  D: leaves_to_triangles vs leaves_to_triangles_array
  A: render_triangles vs render_triangles_batched
  C: build_palette every frame vs cached (amortised per frame)
  E2E: render_frame baseline cfg vs optimized cfg over moving frames

Usage (Windows venv, from the repo root):
    ..\\..\\venv\\Scripts\\python code\\task4_verify.py
    ..\\..\\venv\\Scripts\\python code\\task4_verify.py --quick
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402

from brick_geom import (leaves_to_triangles, leaves_to_triangles_array,
                        pad_to_max)  # noqa: E402
from brick_quadtree import quadtree_partition  # noqa: E402
from brick_color import build_palette, quantize_nearest_bgr  # noqa: E402
from brick_render import (render_triangles, render_triangles_batched)  # noqa: E402
from brick_means import extract_means  # noqa: E402
from brick_io import StageTimer  # noqa: E402
from frame_pipeline import FrameConfig, PaletteState, render_frame  # noqa: E402
from brick_metrics import delta_e_2000_full  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio  # noqa: E402


def synthetic_frame(t, h=480, w=640, seed=7):
    """One synthetic 640x480 camera-like frame at time step t.

    Function: deterministic moving content (diagonal gradient + a drifting
    disc + a fixed high-contrast block + light noise) so the partition, the
    K-Means palette and the quantizer all see real variation frame to frame,
    while identical t always yields the identical image (flicker test needs
    that). Replaces the camera for headless verification.

    Shapes: returns (h, w, 3) uint8 BGR. Semantics: `t` shifts the disc centre
    along x; `seed` fixes the noise so all runs see the same "scene".
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.empty((h, w, 3), dtype=np.float32)
    base[..., 0] = 200 - 0.3 * xx + 0.1 * t          # B: falls left->right
    base[..., 1] = 60 + 0.4 * yy                     # G: rises top->bottom
    base[..., 2] = 40 + 0.25 * xx                    # R
    cx, cy, r = 100 + (t * 9) % (w - 200), h // 2, 90
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
    base[disc] = (40, 200, 220)                      # warm disc, BGR
    base[h // 5:h // 5 + 80, w // 5:w // 5 + 120] = (20, 20, 230)  # red block
    noise = rng.normal(0, 4, (h, w, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def quality_pair(src, canvas):
    """(mean Delta-E2000, PSNR dB) of a render against its source frame.

    Shape: both inputs (H, W, 3) uint8 BGR; two floats out. This is the same
    metric pair Task 4 uses for its before/after quality column, so the
    numbers quoted here drop straight into that table.
    """
    de, _ = delta_e_2000_full(src, canvas)
    psnr = float(peak_signal_noise_ratio(src, canvas, data_range=255))
    return de, psnr


def best_of(fn, reps):
    """Median-of-reps wall time in ms for a zero-arg callable (paired timing).

    Semantics: returns (best_ms, median_ms). Best is the headline for stage
    benchmarks (matches the Task 4 convention of quoting best-of-N for paired
    in-process runs); median is printed alongside so a bimodal cost cannot
    hide.
    """
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def check_partition_equivalence(frame, budget, reps):
    """Assert precomp and sat partitions return the same leaves (sorted)."""
    args = (frame, [1, 2, 4, 8, 16, 32], budget, "mse")
    sat = quadtree_partition(*args, impl="sat")
    pre = quadtree_partition(*args, impl="precomp")
    a, b = sorted(sat[0]), sorted(pre[0])
    assert a == b, (f"precomp partition differs from sat: "
                    f"{len(a)} vs {len(b)} cells, first diff "
                    f"{next((x for x in a if x not in b), None)}")
    assert sat[1:] == pre[1:], "padded/orig shapes differ"
    print(f"  [1] partition equivalence: OK ({len(a)} cells identical)")

    ts_sat = best_of(lambda: quadtree_partition(*args, impl="sat"), reps)
    ts_pre = best_of(lambda: quadtree_partition(*args, impl="precomp"), reps)
    print(f"  [B] partition stage  sat {ts_sat[0]:7.1f} ms | "
          f"precomp {ts_pre[0]:7.1f} ms | speedup {ts_sat[0]/ts_pre[0]:.2f}x")
    return ts_sat, ts_pre


def check_triangles(leaves, reps):
    """Assert the vectorised builder matches the per-triangle loop exactly."""
    ref = leaves_to_triangles(leaves)
    got = leaves_to_triangles_array(leaves)
    assert got.shape == (len(ref), 3, 2), f"shape {got.shape} != {(len(ref), 3, 2)}"
    assert got.dtype == np.int32
    for k, tri in enumerate(ref):
        if not np.array_equal(got[k], tri):
            raise AssertionError(f"triangle {k} differs: {got[k]} vs {tri}")
    print(f"  [2] triangles equivalence: OK ({len(ref)} triangles vertex-exact)")

    t_old = best_of(lambda: leaves_to_triangles(leaves), reps)
    t_new = best_of(lambda: leaves_to_triangles_array(leaves), reps)
    print(f"  [D] triangles stage  loop {t_old[0]:7.1f} ms | "
          f"array {t_new[0]:7.1f} ms | speedup {t_old[0]/t_new[0]:.2f}x")
    return t_old, t_new


def check_render(frame, leaves, reps):
    """Compare batched vs reference render: pixel diff + quality of both."""
    padded, _, _ = pad_to_max(frame, 32)
    tri_list = leaves_to_triangles(leaves)
    tri_arr = leaves_to_triangles_array(leaves)
    means = extract_means(padded, leaves, "fast")
    palette = build_palette(means, 16, "kmeans_lab")
    # The module-under-test's own assignment rule, so the render comparison
    # uses the same labels a real frame would carry.
    labels = quantize_nearest_bgr(means, palette)

    ref = render_triangles(padded.shape[:2], tri_list, labels, palette,
                           frame.shape[:2])
    got = render_triangles_batched(padded.shape[:2], tri_arr, labels, palette,
                                   frame.shape[:2])
    assert ref.shape == got.shape == frame.shape
    diff = int(np.count_nonzero(np.any(ref != got, axis=2)))
    frac = diff / float(ref.shape[0] * ref.shape[1])
    de_ref, psnr_ref = quality_pair(frame, ref)
    de_got, psnr_got = quality_pair(frame, got)
    print(f"  [3] render pixel diff: {diff} px ({frac*100:.3f}% of frame) "
          f"-- expected: 1-px fringe only")
    print(f"      quality vs source: ref dE {de_ref:.3f} / PSNR {psnr_ref:.2f} "
          f"| batched dE {de_got:.3f} / PSNR {psnr_got:.2f} "
          f"| dDelta-E {abs(de_ref-de_got):.4f}")
    assert frac < 0.01, f"batched render changed {frac*100:.2f}% of pixels"
    assert abs(de_ref - de_got) < 0.2, "batched render moved Delta-E visibly"

    t_ref = best_of(lambda: render_triangles(padded.shape[:2], tri_list,
                                             labels, palette,
                                             frame.shape[:2]), reps)
    t_bat = best_of(lambda: render_triangles_batched(padded.shape[:2], tri_arr,
                                                     labels, palette,
                                                     frame.shape[:2]), reps)
    print(f"  [A] render stage  loop {t_ref[0]:7.1f} ms | "
          f"batched {t_bat[0]:7.1f} ms | speedup {t_ref[0]/t_bat[0]:.2f}x")
    return t_ref, t_bat


def check_palette_cache(frame, leaves, reps):
    """Palette stays byte-identical between rebuilds; rebuilds are periodic."""
    padded, _, _ = pad_to_max(frame, 32)
    means = extract_means(padded, leaves, "fast")
    refresh, frames = 5, 22   # refresh < default so the test exercises it fast
    state = PaletteState(refresh)
    palettes, refreshed_flags = [], []
    for _ in range(frames):
        palettes.append(state.get(means, 16, "kmeans_lab").copy())
        refreshed_flags.append(state.refreshed)
    # Rebuilds exactly on frames 0, 5, 10, 15, 20.
    expect = [i % refresh == 0 for i in range(frames)]
    assert refreshed_flags == expect, f"rebuild cadence {refreshed_flags} != {expect}"
    for i in range(1, frames):
        same = np.array_equal(palettes[i], palettes[i - 1])
        if expect[i]:
            continue  # rebuild frame may legitimately differ
        assert same, f"palette drifted on a cached frame (i={i})"
    print(f"  [4] palette cache: OK ({frames} frames, rebuild every {refresh}, "
          f"zero drift between rebuilds)")

    t_full = best_of(lambda: build_palette(means, 16, "kmeans_lab"), reps)
    cache = PaletteState(refresh)
    t_amort = best_of(
        lambda: [cache.get(means, 16, "kmeans_lab") for _ in range(refresh)],
        reps)
    per_frame = t_amort[0] / refresh
    print(f"  [C] palette stage  every-frame {t_full[0]:7.1f} ms | "
          f"cached {per_frame:7.1f} ms/frame (amortised over {refresh}) | "
          f"speedup {t_full[0]/per_frame:.2f}x")
    return t_full, per_frame


def check_end_to_end(frame, cfg, reps):
    """Whole-frame old path vs new path: pixel diff + quality + stage timings."""
    base = FrameConfig(**{**cfg.__dict__, "sse_impl": "sat",
                          "palette_refresh": 1})
    opt = FrameConfig(**cfg.__dict__)
    timer_b, timer_o = StageTimer(), StageTimer()
    st_b, st_o = PaletteState(1), PaletteState(cfg.palette_refresh)
    # Seed before EACH path so both K-Means calls draw from the same RNG
    # state; otherwise the palettes legitimately differ (the recorded
    # non-reproducibility) and the pixel diff would measure that, not the
    # optimizations. camera_app seeds once at startup -- same idea.
    cv2.setRNGSeed(0)
    canvas_b, _, _ = render_frame(frame, base, timer_b, st_b)
    cv2.setRNGSeed(0)
    canvas_o, _, _ = render_frame(frame, opt, timer_o, st_o)
    diff = int(np.count_nonzero(np.any(canvas_b != canvas_o, axis=2)))
    frac = diff / float(frame.shape[0] * frame.shape[1])
    de_b, psnr_b = quality_pair(frame, canvas_b)
    de_o, psnr_o = quality_pair(frame, canvas_o)
    print(f"  [5] end-to-end pixel diff vs old path: {diff} px "
          f"({frac*100:.3f}%)")
    print(f"      quality vs source: old dE {de_b:.3f} / PSNR {psnr_b:.2f} "
          f"| new dE {de_o:.3f} / PSNR {psnr_o:.2f}")

    for name, tmr in (("old", timer_b), ("new", timer_o)):
        rep = tmr.report()
        parts = "  ".join(f"{k}={rep[k]['median_ms']:.0f}"
                          for k in ("partition", "triangles", "means",
                                    "palette", "quantize", "render"))
        print(f"      frame [{name}] total {rep['__total__']['median_ms']:6.1f} ms"
              f"   ({parts})")

    t_b = best_of(lambda: render_frame(frame, base, timer_b, st_b), reps)
    t_o = best_of(lambda: render_frame(frame, opt, timer_o, st_o), reps)
    print(f"  [E2E] whole frame  old {t_b[0]:7.1f} ms | new {t_o[0]:7.1f} ms "
          f"| speedup {t_b[0]/t_o[0]:.2f}x  "
          f"(~{1000.0/t_o[0]:.1f} FPS processing-only)")
    return t_b, t_o


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--reps", type=int, default=5, help="benchmark repetitions")
    p.add_argument("--quick", action="store_true",
                   help="single rep, for a fast smoke run")
    args = p.parse_args()
    reps = 1 if args.quick else args.reps

    print("== Task 4 optimization verification (synthetic 640x480) ==")
    frame = synthetic_frame(t=3)
    cfg = FrameConfig(S_set=[1, 2, 4, 8, 16, 32], K=16, budget=9990)

    print("\n-- partition (B) --")
    check_partition_equivalence(frame, cfg.budget, reps)

    print("\n-- geometry + stages on one partition --")
    leaves, _, _ = quadtree_partition(frame, cfg.S_set, cfg.budget, "mse",
                                      "sat")
    check_triangles(leaves, reps)
    check_render(frame, leaves, reps)
    check_palette_cache(frame, leaves, reps)

    print("\n-- end-to-end frame --")
    check_end_to_end(frame, cfg, reps)
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
