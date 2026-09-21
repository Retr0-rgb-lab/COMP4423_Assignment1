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

from brick_geom import (pad_to_max, leaves_to_triangles_array,
                        MAX_TRIANGLES)  # noqa: F401  (re-export: drivers cap --budget)
from brick_quadtree import quadtree_partition
from brick_region_merge import region_merge_partition
from brick_color import build_palette, quantize_nearest_bgr
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
                 i.e. the old behaviour.
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

    def __init__(self, refresh=10):
        """Create an empty cache. `refresh` is the rebuild period in frames."""
        self.refresh = max(1, int(refresh))
        self.palette = None
        self.age = 0
        self.refreshed = False

    def get(self, means, k, method):
        """Return the cached palette, rebuilding it when the period expires.

        Shape: `means` is (T, 3) float32 BGR; returns (k, 3) uint8 BGR.
        Semantics: on rebuild frames the palette comes fresh from
        `build_palette` and `age` resets to 1; on cached frames the identical
        array object is returned and `age` grows by 1, so downstream quantize
        results are byte-stable between rebuilds. The K guard covers a config
        change mid-session (defensive; the app fixes K for the whole run).
        """
        if (self.palette is None or self.age >= self.refresh
                or self.palette.shape[0] != k):
            self.palette = build_palette(means, k, method)
            self.age = 1
            self.refreshed = True
        else:
            self.age += 1
            self.refreshed = False
        return self.palette


def render_frame(frame, cfg, timer, palette_state):
    """Run the full Task 3 pipeline on one frame; returns (canvas, info).

    Function
    --------
    Same seven steps as the Task 3 experiment driver, minus the metric suite
    (metrics are far too slow for a live loop and are not needed to display),
    plus the four Task 4 optimizations:
      B  partition split-priorities precomputed (impl="precomp");
      D1 the padded image carried from partition to means (no double pad) and
         triangles built vectorised into one (T, 3, 2) array;
      C  palette served from the cross-frame cache;
      A  render batched by palette label (<=K fillPoly calls + 1 polylines).

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
                  `palette`, `palette_refreshed` (True on rebuild frames) and
                  `t_resize_ms`.
    """
    t_resize = time.perf_counter()
    if cfg.scale != 1.0:
        frame = cv2.resize(frame, None, fx=cfg.scale, fy=cfg.scale,
                           interpolation=cv2.INTER_AREA)
    t_resize = (time.perf_counter() - t_resize) * 1000.0
    processed = frame

    with timer.stage("partition"):
        if cfg.partition == "quadtree":
            leaves, padded_shape, orig_shape, padded = quadtree_partition(
                frame, cfg.S_set, cfg.budget, cfg.priority, cfg.sse_impl,
                return_padded=True)
        else:
            leaves, padded_shape, orig_shape = region_merge_partition(
                frame, cfg.S_set, cfg.budget)
            padded, _, _ = pad_to_max(frame, max(cfg.S_set))

    with timer.stage("triangles"):
        tri = leaves_to_triangles_array(leaves)

    with timer.stage("means"):
        means = extract_means(padded, leaves, cfg.means, cfg.n_samples)

    with timer.stage("palette"):
        palette = palette_state.get(means, cfg.K, cfg.palette_method)
        refreshed = palette_state.refreshed

    with timer.stage("quantize"):
        labels = quantize_nearest_bgr(means, palette)

    with timer.stage("render"):
        canvas = render_triangles_batched(padded_shape, tri, labels, palette,
                                          orig_shape)

    sizes = {}
    for (_, _, s) in leaves:
        sizes[s] = sizes.get(s, 0) + 1  # cells, not triangles
    info = {
        "n_tri": len(leaves) * 2,
        "n_cells": len(leaves),
        "sizes": sizes,
        "palette": palette,
        "palette_refreshed": refreshed,
        "t_resize_ms": t_resize,
    }
    return canvas, processed, info
