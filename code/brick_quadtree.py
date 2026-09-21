"""
brick_quadtree — top-down quadtree tessellation.

Public entry point: `quadtree_partition`.
Internal helpers:   `_quadtree_priority`, `_region_mse`, `_region_sobel_var`.
Precomputed maps:   `brick_prio.priority_maps` / `prio_from_maps` (opt B).

Three split-priority implementations (`impl=`):
  "sat"     — batched summed-area-table queries, one small batch per split
              (default; bit-identical to "ref" for the "mse" priority).
  "ref"     — original per-candidate numpy region means (slow, for A/B only).
  "precomp" — Task 4 optimization B: every candidate priority is a pure
              function of the image, so the whole universe is scored up front
              in a few batched calls and the greedy loop runs with zero numpy
              calls. Bit-identical leaves to "sat" for "mse".
"""
import heapq
import time

import cv2
import numpy as np
from skimage import filters

from brick_geom import MAX_TRIANGLES, pad_to_max
from brick_sat import (build_sats, rect_sse_total_many,
                       rect_sobel_var_many)
from brick_prio import priority_maps, prio_from_maps


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

    Gotcha: `priority` is matched with `if priority == "edgef1" ... else mse`, so
    an unrecognised string would select the mse branch silently -- a mislabelled
    run would produce mse numbers reported under an edgef1 label, and the metrics
    alone would not reveal it. `quadtree_partition` now rejects anything outside
    {"mse", "edgef1"} up front, so this branch is only reachable with a valid
    value; keep the entry-point check if a third mode is added here.
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


def _quadtree_priorities_sat(bundle, leaves, priority):
    """Priorities for a BATCH of leaves, from the integral tables.

    Function
    --------
    Same decision rule as `_quadtree_priority`, but the table lookups are issued
    for the whole batch at once. That batching is the entire point: the scalar
    version is O(1) in arithmetic yet measured no faster than the original numpy
    means, because the bottleneck was ~160000 tiny numpy calls, not the maths.
    Doing one batch per split (4 children) and one for the seed (every S_max cell)
    turns that into a few thousand numpy calls over arrays.

    Shapes/semantics: `leaves` is an iterable of `(x, y, size)` in padded
    coordinates; `priority` is "mse" or "edgef1". Returns a (N,) float64 array
    aligned with `leaves`. For "mse", entry i is the SSE reduction of splitting
    leaf i divided by the 6 triangles that adds; for "edgef1" it is the variance of
    the Sobel magnitude over that leaf. Leaves with `size <= 1` get exactly 0.0 and
    are excluded from the table queries, because a zero-area query would divide by
    zero -- and 0.0 is also what the reference returns for them, so the two agree.
    """
    arr = np.asarray(list(leaves), dtype=np.int64).reshape(-1, 3)
    out = np.zeros(len(arr), dtype=np.float64)
    if len(arr) == 0:
        return out
    xs, ys, ss = arr[:, 0], arr[:, 1], arr[:, 2]
    ok = ss > 1
    if not ok.any():
        return out
    xs, ys, ss = xs[ok], ys[ok], ss[ok]
    if priority == "edgef1":
        out[ok] = rect_sobel_var_many(bundle, xs, ys, ss)
        return out
    half = ss // 2
    # ONE query covers the parent square AND its four children: index i is the
    # parent, the next 4N entries are the children of the leaves in order. Issuing
    # it as a single call rather than two matters because the cost here is numpy
    # call overhead, not arithmetic -- see the note in brick_sat.
    qx = np.concatenate([xs, xs, xs + half, xs, xs + half])
    qy = np.concatenate([ys, ys, ys, ys + half, ys + half])
    qs = np.concatenate([ss, half, half, half, half])
    res = rect_sse_total_many(bundle, qx, qy, qs)
    n = len(xs)
    parent = res[:n]
    child = res[n:].reshape(4, -1).sum(axis=0)
    out[ok] = (parent - child) / 6.0
    return out


def quadtree_partition(img, S_set, max_triangles=MAX_TRIANGLES, priority="mse",
                       impl="sat", return_padded=False):
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
      priority     : "mse" (default, uses ΔMSE/6) or "edgef1" (uses Sobel
                     variance). Validated at the top; anything else raises.
      impl         : "sat" (default) answers the split priorities from the
                     summed-area tables in `brick_sat`, batched one query per
                     split, so the cost stops scaling with region area rather than
                     with numpy call count. "ref" keeps the original
                     per-candidate numpy means and exists so the two can be
                     measured against each other. "precomp" (Task 4) scores the
                     whole candidate universe up front in a few large batched
                     queries and runs the greedy heap loop with zero numpy calls;
                     bit-identical to "sat" for "mse", and "mse" only.
                     For the default "mse" priority the decision rule is
                     numerically IDENTICAL across all three, so impl is a pure
                     speed knob. For "edgef1" only "sat"/"ref" are valid: the
                     precomputed maps carry no Sobel information.
      return_padded: when True, also return the padded image as a 4th element,
                     so a caller that needs padded pixels for a later stage
                     (e.g. the live loop's mean extraction) does not pad twice.
                     Default False keeps the original 3-tuple for Task 2/3.
    Intermediate:
      leaves : set of (x, y, size) -- the cells that currently exist, in PADDED
               coordinates. Membership only; the SSE of a cell is recomputed on
               demand from the tables rather than stored, because the stored copy
               was never read (see the note in the body).
      heap   : list of (-priority, leaf) tuples. Negation because Python's
               heapq is a min-heap; we want to pop the LARGEST priority
               first so we negate on push and re-negate on pop.
    Output:
      (leaves_keys, padded_shape, orig_shape) — or with `return_padded=True`,
      (leaves_keys, padded_shape, orig_shape, padded):
        leaves_keys    : list of (x, y, size) — final adaptive cells.
        padded_shape   : (Hp, Wp) — padded image extent.
        orig_shape     : (H, W) — original input size for later crop.
        padded         : (Hp, Wp, 3) uint8 — the padded image (only when
                         return_padded=True).
    """
    # Validate up front: this is the only guard on `priority`, and a silent
    # fallback to "mse" would relabel an experiment without any metric showing
    # it (see the Gotcha note in _quadtree_priority's docstring).
    if priority not in ("mse", "edgef1"):
        raise ValueError(
            f"Unknown priority {priority!r}; expected 'mse' or 'edgef1'.")
    if impl not in ("sat", "ref", "precomp"):
        raise ValueError(
            f"Unknown impl {impl!r}; expected 'sat', 'ref' or 'precomp'.")
    if impl == "precomp" and priority != "mse":
        raise ValueError("impl='precomp' supports priority='mse' only "
                         "(it precomputes delta-SSE maps; edgef1 needs the "
                         "Sobel tables -- use impl='sat' for edgef1).")

    padded, orig_shape, _ = pad_to_max(img, max(S_set))
    S_max, S_min = max(S_set), min(S_set)
    Hp, Wp = padded.shape[:2]
    t0 = time.time()

    # `leaves` holds only MEMBERSHIP -- the set of cells that currently exist.
    # It used to be a dict mapping cell -> its SSE, and that SSE was computed at
    # seed time and again for every child on every split, then never read: the
    # only uses are `len()`, iteration, `in`, and `list(leaves.keys())`. At 9990
    # triangles that is 4 discarded region-SSE computations per split, on top of
    # the 5 the priority function actually needs. Storing a value nobody reads was
    # pure waste, so the container is now a set and the waste is gone.
    leaves = set()
    for i in range(Hp // S_max):
        for j in range(Wp // S_max):
            leaves.add((j * S_max, i * S_max, S_max))
    n_tri = len(leaves) * 2
    print(f"    [qt] start S_max={S_max}: {len(leaves)} cells, {n_tri} triangles "
          f"(budget {max_triangles}) ({time.time()-t0:.2f}s)")

    if n_tri > max_triangles:
        print(f"    [qt] starting grid over budget; falling back to region_merge")
        from brick_region_merge import region_merge_partition
        res = region_merge_partition(img, S_set, max_triangles)
        # `padded` is the same array region_merge recomputes internally (same
        # input, same S_max), so handing it back costs nothing and lets the
        # caller skip a duplicate pad_to_max.
        if return_padded:
            return (*res, padded)
        return res

    # --- priority source ---------------------------------------------------
    # One of three implementations, all honouring the same batch contract
    # `prio_get(leaves) -> sequence of floats aligned with leaves`:
    if impl == "precomp":
        # Task 4 optimization B: score the whole candidate universe up front
        # (~log2(S_max) large batched reductions), then run the greedy loop
        # with zero numpy calls. Values are bit-identical to the sat path (see
        # brick_prio: exact-integer sums make any order identical), so the
        # heap sees the same keys in the same order and the resulting leaves
        # set is identical -- this is a pure speed change.
        _, prio_map = priority_maps(padded, S_set)
        pmap = {s: m.tolist() for s, m in prio_map.items()}
        print(f"    [qt] impl=precomp: "
              f"{sum(m.size for m in prio_map.values())} candidate priorities "
              f"precomputed ({time.time()-t0:.2f}s)")

        def prio_get(lfs):
            """Precomputed priorities for a batch of leaves (list in/out).

            Shape/semantics: `lfs` is an iterable of `(x, y, size)` in padded
            coordinates; returns a list of floats aligned with it, read from
            the precomputed per-size lists. Zero numpy calls per leaf.
            """
            return [prio_from_maps(pmap, lf) for lf in lfs]
    elif impl == "sat":
        if priority == "edgef1":
            # The reference runs Sobel per region, which reflects at the region
            # boundary; this runs it once globally. The values differ near region
            # edges (measured: identical for "mse", median 2x relative difference
            # for "edgef1"), so say so rather than let a number quietly change
            # meaning. Recorded E_priority results came from the per-region form.
            print("    [qt] note: edgef1 + impl=sat uses a GLOBAL Sobel map; "
                  "not numerically equal to the per-region reference")
        bundle = build_sats(padded, with_sobel=(priority == "edgef1"))

        def prio_get(lfs):
            """Table-backed priorities for a batch of leaves, in one query.

            Shape/semantics: `lfs` is an iterable of `(x, y, size)` in padded
            coordinates; returns a (N,) float64 array aligned with it, as
            documented on `_quadtree_priorities_sat`. Bound to the tables and
            the priority string so the main loop is written once for all
            implementations instead of duplicating the split logic.
            """
            return _quadtree_priorities_sat(bundle, lfs, priority)
    else:
        def prio_get(lfs):
            """Reference priorities for a batch of leaves, one call per leaf.

            Shape/semantics: same contract as the table-backed version, but N
            separate numpy passes over the region -- kept so the three
            implementations can be timed against each other from the same loop.
            """
            return [_quadtree_priority(padded, lf, priority) for lf in lfs]

    heap = []
    seed = list(leaves)
    for leaf, p in zip(seed, prio_get(seed)):
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
        leaves.discard(leaf)
        n_tri += 6  # 1 cell (2 tri) -> 4 cells (8 tri)
        for c, p in zip(children, prio_get(children)):
            leaves.add(c)
            heapq.heappush(heap, (-p, c))
        iters += 1

    print(f"    [qt] done: splits={iters}, n_tri={n_tri}, cells={len(leaves)} "
          f"({time.time()-t0:.2f}s)")
    if return_padded:
        # Task 4 optimization D1: the caller (frame_pipeline.render_frame)
        # needs the padded image for the means stage; handing it back removes
        # the duplicate pad_to_max the live loop used to pay. Optional so the
        # Task 2/3 callers keep their 3-tuple contract untouched.
        return list(leaves), (Hp, Wp), orig_shape, padded
    return list(leaves), (Hp, Wp), orig_shape