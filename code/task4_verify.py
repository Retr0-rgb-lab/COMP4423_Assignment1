"""
task4_verify — correctness assertions + paired benchmarks for optimizations A/B/C/D.

Runs headless (no camera, no window): it drives `brick_pipeline.render_frame`
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

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402

from brick_geom import (leaves_to_triangles, leaves_to_triangles_array,
                        pad_to_max)  # noqa: E402
from brick_quadtree import quadtree_partition  # noqa: E402
from brick_color import build_palette, quantize_nearest_bgr  # noqa: E402
from brick_color_rt import palette_kmeans_warm  # noqa: E402
from brick_temporal import TemporalState  # noqa: E402
from brick_render import (render_triangles, render_triangles_batched)  # noqa: E402
from brick_means import extract_means, means_by_masks  # noqa: E402
from brick_means_rows import means_by_rows, compare_with_masks  # noqa: E402
from brick_io import StageTimer  # noqa: E402
from brick_pipeline import FrameConfig, PaletteState, render_frame  # noqa: E402
from task4_verify_util import (synthetic_frame, quality_pair, best_of,
                                 churn)  # noqa: E402



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
    # deadband=0 isolates the cadence logic from the palette dead-band (which
    # would otherwise turn a no-op rebuild into a non-refresh frame).
    state = PaletteState(refresh, deadband=0.0)
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


def check_means_rows(frame, leaves, reps):
    """Row-run means (opt E) must be bit-identical to the offset-gather method.

    Function: asserts `means_by_rows == means_by_masks` exactly (the claim is
    bit-identity, proved by exact-integer sums), then times both.

    Shape/semantics: `frame` (H,W,3) uint8 BGR, `leaves` a partition; prints and
    asserts; returns the two (best_ms, median_ms) timing pairs.
    """
    padded, _, _ = pad_to_max(frame, 32)
    res = compare_with_masks(padded, leaves)
    print(f"  [E] means rows vs masks: identical={res['identical']} "
          f"max|diff|={res['max_absdiff']:.3g} n_diff={res['n_diff']}")
    assert res["identical"], "row-run means are NOT bit-identical to masks"
    t_mask = best_of(lambda: means_by_masks(padded, leaves), reps)
    t_rows = best_of(lambda: means_by_rows(padded, leaves), reps)
    print(f"  [E] means stage  masks {t_mask[0]:7.1f} ms | rows {t_rows[0]:7.1f} ms "
          f"| speedup {t_mask[0]/t_rows[0]:.2f}x")


def check_temporal(identical, noisy, cfg, reps):
    """Temporal coherence (A+B+C): palette rebuild must stop jumping and a
    static scene must stop re-partitioning.

    Function: drives render_frame over two synthetic sequences -- identical
    frames and noisy frames -- with temporal coherence OFF and ON, separating
    STEADY frames (palette cached) from REBUILD frames, and measures palette
    deltas and thresholded canvas churn. Also checks the warm-start fixed
    point: re-refining an already-optimal palette must not jump.

    Shape/semantics: sequences are lists of (H,W,3) uint8 BGR frames; both
    lists must be the same length. Prints the table and asserts the temporal
    improvements before any of it may be quoted.
    """
    # Warm-start fixed point: refining p0 should stay near p0, unlike two cold
    # builds which differed by ~153/255 (the measured flash cause).
    padded, _, _ = pad_to_max(identical[0], 32)
    leaves, _, _ = quadtree_partition(identical[0], cfg.S_set, cfg.budget, "mse",
                                      "sat")
    means = extract_means(padded, leaves, "fast")
    p0 = build_palette(means, cfg.K, "kmeans_lab")
    p1 = build_palette(means, cfg.K, "kmeans_lab")
    pw = palette_kmeans_warm(means, cfg.K, p0, "lab")
    cold = int(np.abs(p0.astype(int) - p1.astype(int)).max())
    warm = int(np.abs(p0.astype(int) - pw.astype(int)).max())
    print(f"  [A] palette stability: cold rebuild delta {cold} (RNG jump) | "
          f"warm-start delta {warm}")
    assert warm < cold // 2, "warm start did not stabilise the palette"

    def run_sequence(frames, temporal_on):
        cv2.setRNGSeed(0)
        pal_state = PaletteState(cfg.palette_refresh)
        tstate = TemporalState(cfg.temporal and temporal_on, cfg.reuse_thresh,
                               cfg.hysteresis, cfg.max_reuse)
        prev, prev_pal = None, None
        steady, flash, reused, frozen = [], [], 0, 0
        for f in frames:
            canvas, _, info = render_frame(f, cfg, StageTimer(), pal_state,
                                           tstate)
            if info["reused"]:
                reused += 1
            if info.get("colour_frozen"):
                frozen += 1
            if prev is not None:
                c = churn(canvas, prev)
                (flash if info["palette_refreshed"] else steady).append(c)
            prev, prev_pal = canvas, info["palette"]
        return steady, flash, reused, frozen, tstate.render_hits

    rows = {}
    for name, seq in (("identical", identical), ("noisy", noisy)):
        for temporal_on in (False, True):
            steady, flash, re, fr, rh = run_sequence(seq, temporal_on)
            rows[(name, temporal_on)] = (steady, flash, re, fr, rh)
            print(f"  [B/C] {name:9s} temporal={'ON ' if temporal_on else 'OFF'} "
                  f"| steady churn {np.mean(steady)*100:6.3f}% "
                  f"| rebuild-frame churn "
                  f"{(max(flash) if flash else 0.0)*100:6.3f}% "
                  f"| reused {re}/{len(seq)} "
                  f"| colour-frozen {fr}/{len(seq)}")

    # Assertions: temporal ON must freeze a static scene and cut the churn.
    # The rebuild-frame bound is 0.1%, not 0: a warm-started entry can still
    # move 1/255, which flips a handful of pixels sitting exactly on a palette
    # boundary (~0.003% measured, i.e. single-digit pixels -- imperceptible).
    id_on = rows[("identical", True)]
    assert np.max(id_on[0]) == 0.0, "identical frames still churn with temporal ON"
    assert (max(id_on[1]) if id_on[1] else 0.0) < 0.001, \
        "warm-started rebuild still visibly churns identical frames"
    assert id_on[2] == len(identical) - 1, "static frames were not all reused"
    # The canvas is reused either via the frozen-colour path or via the
    # byte-identical render cache; at least one must cover the static frames.
    assert id_on[3] + id_on[4] >= len(identical) - 1, \
        "no canvas reuse (frozen or cached) on a static scene"
    noisy_off, noisy_on = rows[("noisy", False)], rows[("noisy", True)]
    assert np.mean(noisy_on[0]) < np.mean(noisy_off[0]), \
        "temporal coherence did not reduce per-frame churn"
    print(f"  [B/C] noisy steady churn: OFF {np.mean(noisy_off[0])*100:.3f}% -> "
          f"ON {np.mean(noisy_on[0])*100:.3f}% "
          f"({np.mean(noisy_off[0])/max(np.mean(noisy_on[0]),1e-12):.1f}x less)")


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

    print("\n-- means row-run table (opt E) --")
    check_means_rows(frame, leaves, reps)

    print("\n-- end-to-end frame --")
    check_end_to_end(frame, cfg, reps)

    print("\n-- temporal coherence (A+B+C) --")
    rng = np.random.default_rng(0)
    ident = [frame.copy() for _ in range(14)]
    noisy = [np.clip(frame.astype(np.float32)
                     + rng.normal(0, 2.0, frame.shape), 0, 255).astype(np.uint8)
             for _ in range(14)]
    check_temporal(ident, noisy, cfg, reps)
    print("\nAll assertions passed.")


if __name__ == "__main__":
    main()
