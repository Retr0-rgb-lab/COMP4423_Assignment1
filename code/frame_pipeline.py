"""
frame_pipeline — the per-frame engine for the Task 4 live loop.

Split out of `camera_app.py` (AGENTS 4.1: the driver stays thin; it should
only parse args, own the capture loop and draw the window). Everything that
transforms one camera frame into a rendered mosaic lives here:

  * `FrameConfig`  — per-session settings (pure config, no arrays).
  * `PaletteState` — cross-frame palette cache (optimization C: the palette
                     is rebuilt only every N frames, which both amortises the
                     K-Means cost and removes the recorded frame-to-frame
                     colour flicker).
  * `render_frame` — the seven pipeline steps, timed by the caller's
                     StageTimer, geometry-exact against the Task 3 reference
                     path (precomp partition == sat partition bit-for-bit;
                     batched render only differs on sub-boundary fringe
                     pixels; means/quantize are the unchanged engine calls).

Per frame: resize -> partition (precomp) -> triangles (array builder) ->
means (fast) -> palette (cached) -> quantize -> render (batched). The padded
image travels from the partition stage to the means stage through
`quadtree_partition(return_padded=True)`, so the frame is padded ONCE
(optimization D1; the old driver padded it twice).
"""
import time
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

from brick_geom import (pad_to_max, leaves_to_triangles_array,
                        MAX_TRIANGLES)  # noqa: F401  (re-export: drivers cap --budget)
from brick_quadtree import quadtree_partition
from brick_region_merge import region_merge_partition
from brick_color import build_palette, quantize_nearest_bgr
from brick_color_rt import palette_kmeans_warm, quantize_nearest_bgr_sticky
from brick_render import render_triangles_batched
from brick_means import extract_means, METHODS as MEANS_METHODS  # noqa: F401


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
      sse_impl : "precomp" | "sat" | "ref" -- quadtree split-priority
                 implementation (see brick_quadtree). "precomp" is the Task 4
                 default and is bit-identical to "sat" for the "mse" priority.
      palette_refresh : int -- rebuild the K-Means palette every N frames and
                 reuse it otherwise (optimization C). 1 = rebuild every frame,
                 i.e. the old behaviour. Rebuilds are WARM-STARTED from the
                 previous palette (brick_color_rt.palette_kmeans_warm), so a
                 rebuild refines rather than jumps.
      palette_deadband : float -- if a warm-started rebuild moves no palette
                 entry by more than this many grey levels, discard it and keep
                 the previous palette (static scenes stay byte-frozen).
      temporal : bool -- enable the temporal-coherence layer (opt B+C): reuse
                 the partition when the scene is static, and apply label
                 hysteresis. False = the exact pre-temporal pipeline.
      reuse_thresh : float -- mean absolute signature difference (0-255 grey
                 levels) under which the cached partition is reused.
      hysteresis : float -- relative squared-distance margin for the sticky
                 quantizer (0 = plain nearest, 0.1 = 10% dead-band).
      max_reuse : int -- force a re-partition after this many reused frames
                 (0 = never; the signature test alone decides).
      partition/priority/palette_method : names dispatched inside the engine
                 modules; an unknown value raises there rather than silently
                 falling back (see brick_quadtree / brick_color).
    """

    S_set: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    K: int = 16
    budget: int = 9990
    scale: float = 1.0
    means: str = "fast"
    sse_impl: str = "precomp"
    n_samples: int = 9
    palette_refresh: int = 10
    palette_deadband: float = 2.0
    temporal: bool = True
    reuse_thresh: float = 1.0
    hysteresis: float = 0.1
    max_reuse: int = 0
    partition: str = "quadtree"
    priority: str = "mse"
    palette_method: str = "kmeans_lab"


class PaletteState:
    """Cross-frame palette cache: rebuild every `refresh` frames, reuse else.

    Function
    --------
    Optimization C (Task 4). The shipped loop rebuilt the K-Means palette on
    EVERY frame, which cost ~23 ms per frame AND made the palette drift on a
    static scene (measured: up to 207/255 per channel across 5 calls on
    identical input -- see docs/progress/Task4.md). Rebuilding only every N
    frames fixes both: the K-Means cost is paid once per N frames, and between
    rebuilds the palette is BYTE-IDENTICAL, so a static scene renders with a
    constant palette -- zero flicker. The adaptation lag to a genuinely new
    scene is at most N frames and must be declared in the report.

    Shapes / semantics
    ------------------
    `palette` is (K, 3) uint8 BGR (`palette[k, 0]=B`) or None before the first
    frame; `age` counts frames served since the last rebuild INCLUDING the
    rebuild frame itself (so `age` runs 1..refresh and the next rebuild
    happens when `age` would exceed `refresh`). `refreshed` is True on the
    exact frames the palette was rebuilt, False on cached frames -- the HUD
    reads it to show FRESH/CACHED. `refresh <= 1` disables caching entirely
    (rebuild every call = the old behaviour), which is how the before/after
    table measures the win honestly.
    """

    def __init__(self, refresh=10, deadband=2.0):
        """Create an empty cache.

        `refresh` is the rebuild period in frames. `deadband` is the palette
        equivalent of the label hysteresis: if a warm-started rebuild moves no
        entry by more than this many grey levels, the previous palette is kept
        byte-for-byte. Without it, a 1/255 refinement still flips a few cells
        that sit exactly on a palette boundary (~0.3% of pixels measured on a
        static frame), which is an invisible palette change causing a visible
        label change.
        """
        self.refresh = max(1, int(refresh))
        self.deadband = float(deadband)
        self.palette = None
        self.age = 0
        self.refreshed = False

    def get(self, means, k, method):
        """Return the cached palette, rebuilding it when the period expires.

        Shape: `means` is (T, 3) float32 BGR; returns (k, 3) uint8 BGR.
        Semantics: on rebuild frames the palette is REFINED from the previous
        one by `palette_kmeans_warm` (opt A) -- a warm start cannot jump to a
        different local optimum, which is what removed the measured 76% flash
        -- and `age` resets to 1; on cached frames the identical array object is
        returned and `age` grows by 1, so downstream quantize results are
        byte-stable between rebuilds. The very first call has no previous
        palette and does a normal cold `build_palette`. The K guard covers a
        config change mid-session (defensive; the app fixes K for the run).
        `median_cut` is deterministic and needs no warm start.

        Dead-band: a warm-started rebuild whose largest per-entry move is within
        `deadband` grey levels is discarded (the previous palette is kept, no
        `refreshed` flag), so a static scene is byte-frozen instead of receiving
        periodic imperceptible palette edits that flip boundary labels.
        """
        if (self.palette is None or self.age >= self.refresh
                or self.palette.shape[0] != k):
            if self.palette is not None and method in ("kmeans_lab",
                                                       "kmeans_rgb"):
                space = "lab" if method == "kmeans_lab" else "rgb"
                new = palette_kmeans_warm(means, k, self.palette, space)
                moved = int(np.abs(new.astype(int)
                                   - self.palette.astype(int)).max())
                if moved <= self.deadband:
                    # Imperceptible refinement: keep the previous palette so no
                    # boundary cell flips. Age resets so this is not retried
                    # every frame.
                    self.age = 1
                    self.refreshed = False
                    return self.palette
                self.palette = new
            else:
                self.palette = build_palette(means, k, method)
            self.age = 1
            self.refreshed = True
        else:
            self.age += 1
            self.refreshed = False
        return self.palette


def render_frame(frame, cfg, timer, palette_state, temporal_state=None):
    """Run the full Task 3 pipeline on one frame; returns (canvas, info).

    Function
    --------
    Same seven steps as the Task 3 experiment driver, minus the metric suite
    (metrics are far too slow for a live loop and are not needed to display),
    plus the Task 4 optimizations:
      B  partition split-priorities precomputed (impl="precomp");
      D1 the padded image carried from partition to means (no double pad) and
         triangles built vectorised into one (T, 3, 2) array;
      C  palette served from the cross-frame cache, warm-started on rebuild;
      A  render batched by palette label (<=K fillPoly calls + 1 polylines);
      B+C temporal coherence -- when `temporal_state` is given and the scene is
         static, the cached partition and triangle array are reused and the
         sticky quantizer applies a label dead-band. With temporal_state=None
         the function is exactly the step-4 pipeline (used by the gates).

    Shapes
    ------
    Input: `frame` is (H, W, 3) uint8 BGR -- a single camera frame.
    Intermediate: `padded` is (Hp, Wp, 3) uint8 reflect-padded to a multiple
    of `max(S_set)`; `tri` is (T, 3, 2) int32; `means` is (T, 3) float32 with
    `[..., 0]=B`; `palette` is (K, 3) uint8 BGR; `labels` is (T,) uint8.
    Output: `(canvas, processed, info)`.
      canvas    : (H, W, 3) uint8 BGR -- the rendered mosaic, same size as the
                  processed input (cropped back from the padded canvas).
      processed : (H, W, 3) uint8 BGR -- the frame AFTER `--scale` resizing,
                  i.e. exactly what the pipeline saw.
      info      : per-frame statistics for the HUD and the exit summary --
                  `n_tri`, `n_cells`, `sizes` (cell side -> CELL count),
                  `palette`, `palette_refreshed` (True on rebuild frames),
                  `reused` (True when the cached partition was reused) and
                  `t_resize_ms`.
    """
    t_resize = time.perf_counter()
    if cfg.scale != 1.0:
        frame = cv2.resize(frame, None, fx=cfg.scale, fy=cfg.scale,
                           interpolation=cv2.INTER_AREA)
    t_resize = (time.perf_counter() - t_resize) * 1000.0
    processed = frame

    temporal_on = temporal_state is not None and temporal_state.enabled
    reuse = temporal_on and temporal_state.can_reuse(frame)

    with timer.stage("partition"):
        if reuse:
            # Scene static: keep the cached cells; only re-pad THIS frame so
            # the means stage still sees live pixels.
            leaves = temporal_state.leaves
            padded_shape = temporal_state.padded_shape
            orig_shape = temporal_state.orig_shape
            padded, _, _ = pad_to_max(frame, max(cfg.S_set))
        elif cfg.partition == "quadtree":
            leaves, padded_shape, orig_shape, padded = quadtree_partition(
                frame, cfg.S_set, cfg.budget, cfg.priority, cfg.sse_impl,
                return_padded=True)
        else:
            leaves, padded_shape, orig_shape = region_merge_partition(
                frame, cfg.S_set, cfg.budget)
            padded, _, _ = pad_to_max(frame, max(cfg.S_set))

    with timer.stage("triangles"):
        tri = temporal_state.tri if reuse else leaves_to_triangles_array(leaves)

    with timer.stage("means"):
        means = extract_means(padded, leaves, cfg.means, cfg.n_samples)

    with timer.stage("palette"):
        palette = palette_state.get(means, cfg.K, cfg.palette_method)
        refreshed = palette_state.refreshed

    with timer.stage("quantize"):
        if temporal_on:
            # Sticky assignment: valid only because `tri` row order is stable
            # while reused; commit() clears the memory on a re-partition.
            labels = quantize_nearest_bgr_sticky(
                means, palette, temporal_state.prev_labels,
                temporal_state.hysteresis)
            temporal_state.prev_labels = labels
        else:
            labels = quantize_nearest_bgr(means, palette)

    with timer.stage("render"):
        canvas = render_triangles_batched(padded_shape, tri, labels, palette,
                                          orig_shape)

    if temporal_on:
        if reuse:
            temporal_state.note_reuse()
        else:
            temporal_state.commit(leaves, padded_shape, orig_shape, tri, frame)

    sizes = {}
    for (_, _, s) in leaves:
        sizes[s] = sizes.get(s, 0) + 1  # cells, not triangles
    info = {
        "n_tri": len(leaves) * 2,
        "n_cells": len(leaves),
        "sizes": sizes,
        "palette": palette,
        "palette_refreshed": refreshed,
        "reused": reuse,
        "t_resize_ms": t_resize,
    }
    return canvas, processed, info
