"""
brick_quadtree — top-down quadtree tessellation.

Public entry point: `quadtree_partition`.
Internal helpers:   `_quadtree_priority`, `_region_mse`, `_region_sobel_var`.
"""
import heapq
import time

import cv2
import numpy as np
from skimage import filters

from brick_geom import MAX_TRIANGLES, pad_to_max


def _region_mse(region):
    """Sum of squared deviations from the per-channel mean.

    Function
    --------
    Used as the cost signal for quadtree splits: a leaf with high MSE is
    assumed to contain more colour variation, so splitting it will save
    more distortion than splitting a flat leaf.

    Shapes
    ------
    Input:
      region : (h, w, C) or (h, w) ndarray — sub-image, dtype uint8 or float.
    Output:
      float — sum over all pixels and channels of (pixel - mean)^2.
              Larger value = the region is "more complex".
    """
    mean = region.reshape(-1, region.shape[-1]).mean(axis=0)
    diff = region.astype(np.float32) - mean
    return float((diff * diff).sum())


def _region_sobel_var(region):
    """Variance of the Sobel edge response (higher = more edges / texture).

    Shapes
    ------
    Input:
      region : (h, w, C) or (h, w) ndarray.
    Output:
      float — variance of the Sobel magnitude on the grayscale version of
              `region`. Used as the priority signal for `priority="edgef1"`
              splits, where high-edge regions deserve to be subdivided first.
    """
    if region.ndim == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region
    s = filters.sobel(gray.astype(np.float64))
    return float(s.var())


def _quadtree_priority(img, leaf, priority):
    """Priority for splitting a leaf: ΔMSE/6 (or Sobel variance).

    Shapes
    ------
    Input:
      img      : (Hp, Wp, C) — padded image.
      leaf     : (x, y, size) tuple in padded image coordinates.
      priority : "mse" -> return `(mse_parent - mse_children) / 6`,
                 "edgef1" -> return Sobel variance of the region.
    Output:
      float — priority value; higher = better candidate to split.
              The /6 normalises "SSE saved per NEW triangle" because one
              split adds 6 triangles (2 -> 8) to the budget.
              A leaf with `size <= 1` always scores 0.0 and can never be split
              further, which is how the quadtree terminates.

    Gotcha: `priority` is matched with `if priority == "edgef1" ... else mse`,
    so ANY unrecognised string (e.g. the plausible typo "sobel") silently selects
    the mse branch instead of raising. That is a real trap for the E_priority
    experiment: a mislabelled run would produce mse numbers reported under an
    edgef1 label, and the metrics alone would not reveal it. Nothing validates
    the value today -- `Task3Config.priority` has no check and the only thing
    keeping it correct is the hardcoded ["mse", "edgef1"] list in
    `triangle_brick_task3.build_experiment_grid`. Add a guard at
    `quadtree_partition` before introducing a third priority mode.
    """
    x, y, size = leaf
    if size <= 1:
        return 0.0
    region = img[y:y + size, x:x + size]
    if priority == "edgef1":
        return _region_sobel_var(region)
    half = size // 2
    mse_parent = _region_mse(region)
    mse_children = 0.0
    for dy, dx in [(0, 0), (0, half), (half, 0), (half, half)]:
        mse_children += _region_mse(region[dy:dy + half, dx:dx + half])
    return (mse_parent - mse_children) / 6.0  # per new triangle


def quadtree_partition(img, S_set, max_triangles=MAX_TRIANGLES, priority="mse"):
    """Top-down quadtree + greedy priority queue (rate-distortion optimisation).

    Function
    --------
    Start at the COARSEST grid (S_max, fewest triangles), then repeatedly
    split the highest-priority leaf while the budget still has room.
    Splitting is how cell sizes become adaptive: cells with the biggest
    MSE-reduction (edges / texture) get subdivided first, flat cells stay
    coarse. If the starting grid already EXCEEDS the budget, top-down
    cannot help (splitting only increases the count) and we fall back to
    `brick_region_merge.region_merge_partition`.

    Shapes
    ------
    Input:
      img   : (H, W, C) uint8 — source image.
      S_set : iterable of ints — allowed cell sizes; must all be powers
              of 2 so a leaf can keep halving.
      max_triangles : int — hard budget (default 10000).
      priority     : "mse" (default, uses ΔMSE/6) or "edgef1"
                     (uses Sobel variance).
    Intermediate:
      leaves : dict {(x, y, size) -> mse_float}. `leaves[k]` is the MSE
               of the square at (x, y) with side `size` in PADDED coords.
               `x, y` are integer pixel offsets; `size` is from S_set.
      heap   : list of (-priority, leaf) tuples. Negation because Python's
               heapq is a min-heap; we want to pop the LARGEST priority
               first so we negate on push and re-negate on pop.
    Output:
      (leaves_keys, padded_shape, orig_shape):
        leaves_keys    : list of (x, y, size) — final adaptive cells.
        padded_shape   : (Hp, Wp) — padded image extent.
        orig_shape     : (H, W) — original input size for later crop.
    """
    padded, orig_shape, _ = pad_to_max(img, max(S_set))
    S_max, S_min = max(S_set), min(S_set)
    Hp, Wp = padded.shape[:2]
    t0 = time.time()

    leaves = {}
    # Seed the coarsest level with one (x, y, S_max) entry per cell position.
    for i in range(Hp // S_max):
        for j in range(Wp // S_max):
            x, y = j * S_max, i * S_max
            # MSE of the cell = the budget we'll save by splitting it later.
            leaves[(x, y, S_max)] = _region_mse(padded[y:y + S_max, x:x + S_max])
    n_tri = len(leaves) * 2
    print(f"    [qt] start S_max={S_max}: {len(leaves)} cells, {n_tri} triangles "
          f"(budget {max_triangles}) ({time.time()-t0:.2f}s)")

    if n_tri > max_triangles:
        print(f"    [qt] starting grid over budget; falling back to region_merge")
        from brick_region_merge import region_merge_partition
        return region_merge_partition(img, S_set, max_triangles)

    heap = []
    for leaf in leaves:
        p = _quadtree_priority(padded, leaf, priority)
        heapq.heappush(heap, (-p, leaf))

    iters = 0
    # Keep splitting WHILE there is room in the budget (splits ADD triangles).
    while heap and n_tri + 6 <= max_triangles:
        neg_p, leaf = heapq.heappop(heap)
        if -neg_p <= 0:
            break
        x, y, size = leaf
        # Stale or already-split entry; skip.
        if size <= S_min or leaf not in leaves:
            continue
        half = size // 2
        children = [(x, y, half), (x + half, y, half),
                    (x, y + half, half), (x + half, y + half, half)]
        del leaves[leaf]
        n_tri += 6  # 1 cell (2 tri) -> 4 cells (8 tri)
        for c in children:
            cx, cy, cs = c
            leaves[c] = _region_mse(padded[cy:cy + cs, cx:cx + cs])
            heapq.heappush(heap, (-_quadtree_priority(padded, c, priority), c))
        iters += 1

    print(f"    [qt] done: splits={iters}, n_tri={n_tri}, cells={len(leaves)} "
          f"({time.time()-t0:.2f}s)")
    return list(leaves.keys()), (Hp, Wp), orig_shape