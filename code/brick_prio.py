"""
brick_prio — precomputed split-priority maps for the quadtree (Task 4 opt B).

Split out of `brick_quadtree.py` to keep that file under the 400-line limit
(AGENTS 4.1). The functions here implement ONE responsibility: scoring every
possible quadtree split candidate up front, so the greedy partition loop reads
priorities out of plain Python lists instead of issuing a small numpy query per
split. See `_priority_maps` for the bit-identity argument (exact-integer sums
make any summation order identical to the summed-area-table path).
"""
import numpy as np


def priority_maps(padded, S_set):
    """Per-size sum maps and split-priority maps for EVERY grid cell.

    Function
    --------
    A cell's split priority (delta-SSE per new triangle) depends only on the
    image region it covers, never on the current partition state -- the greedy
    loop recomputes the same value every time. So instead of one small batched
    SAT query per split (~4700 numpy calls per frame), score the entire
    candidate universe here in a few LARGE batched calls (~log2(S_max) at
    640x480), then let the loop read scalars out of plain Python lists. This is
    what actually removes the per-split cost: the loop becomes heapq + list
    indexing only.

    The maps are built HIERARCHICALLY, not from summed-area tables: the size-2
    cell sums come straight from the image, and each larger size sums a 2x2
    group of the previous size's map. Bit-identity with the SAT path still
    holds, and for a stronger reason than batching: every pixel value and
    partial sum here is an exact integer far below 2^53, so float64 addition
    is EXACT and any summation order yields the identical value. The SSE
    formula `s2 - s1*s1/n` then sees bit-identical `s1`/`s2` operands, and
    IEEE arithmetic is deterministic -- same formula, same inputs, same bits.

    Shapes / semantics
    ------------------
    Input: `padded` is the (Hp, Wp, 3) uint8 padded image (Hp/Wp are multiples
    of S_max, hence of every allowed size). `S_set` gives the allowed
    power-of-two sizes; only S_min/S_max are used (the quadtree halves down
    from S_max to S_min regardless of the interior of S_set).
    Size 1 is deliberately materialised only as the image itself: a 1x1
    cell's SSE is exactly 0.0 (n=1: p^2 - p^2/1), which is what both the sat
    path (`ss > 1` filter on the child query) and this path count for it --
    so `prio_map[2]` adds a literal 0.0 for its children instead of looking a
    size-1 SSE map up. That matters for speed: a size-1 SSE map would be
    Hp*Wp entries, ~75% of the whole candidate count, all of them zeros.
    The priority map is built for sizes >= 2 (a 1x1 cell can never split; the
    caller maps it to priority 0.0, matching `_quadtree_priorities_sat`,
    which refuses the zero-area query).
    Output: `(sums, prio_map)`, two dicts keyed by size s:
      `sums[s]`    : (Hp//s, Wp//s, 3) float64 -- per-channel pixel sums; the
                     sum over the cell with origin (x=j*s, y=i*s, s) is
                     `sums[s][i, j, c]`.
      `prio_map[s]`: (Hp//s, Wp//s) float64 -- entry [i, j] is the "mse" split
                     priority of that cell: (SSE(cell) - sum of the four
                     children's SSE) / 6. Children of a size-s cell on the
                     size-(s/2) grid sit at rows (2i, 2i+1) x cols (2j, 2j+1).
                     For s == 2 the children term is the scalar 0.0.
    """
    S_max, S_min = max(S_set), min(S_set)
    img_d = padded.astype(np.float64)
    # Exact-integer hierarchy: sums[1] is the image itself; each size sums a
    # 2x2 group of the previous size. Integer exactness (all values < 2^53)
    # makes this bit-identical to any other summation order, incl. SATs.
    sums = {1: img_d}
    sqs = {1: img_d * img_d}
    s = 2
    while s <= S_max:
        prev, prevq = sums[s // 2], sqs[s // 2]
        h2, w2 = prev.shape[0] // 2, prev.shape[1] // 2
        # (h2, 2, w2, 2, C) -> sum over the two group axes -> (h2, w2, C)
        sums[s] = prev.reshape(h2, 2, w2, 2, 3).sum(axis=(1, 3))
        sqs[s] = prevq.reshape(h2, 2, w2, 2, 3).sum(axis=(1, 3))
        s *= 2

    prio_map = {}
    s = max(2, 2 * S_min)
    while s <= S_max:
        half = s // 2
        n = s * s
        # Per-channel SSE of every size-s cell, clamped at 0 exactly like
        # brick_sat.rect_sse, then summed over channels (rect_sse_total).
        sse = np.maximum(sqs[s] - sums[s] * sums[s] / n, 0.0).sum(axis=2)
        if half == 1:
            # Children are 1x1 cells: SSE exactly 0.0 (see docstring).
            children = 0.0
        else:
            csse = np.maximum(sqs[half] - sums[half] * sums[half]
                              / (half * half), 0.0).sum(axis=2)
            children = (csse[0::2, 0::2] + csse[0::2, 1::2]
                        + csse[1::2, 0::2] + csse[1::2, 1::2])
        prio_map[s] = (sse - children) / 6.0
        s *= 2
    return sums, prio_map


def prio_from_maps(pmap, leaf):
    """Priority of one leaf from the precomputed per-size lists (O(1) scalar).

    Function: list-based companion of `priority_maps` for the greedy loop --
    `pmap[s]` is `prio_map[s].tolist()`, so this is two integer divides and two
    list indexing steps, with no numpy involvement.

    Shape/semantics: `leaf` is (x, y, size) in padded coordinates. Returns the
    float priority, and exactly 0.0 for size <= 1 (unsplittable; mirrors the
    sat path's `ss > 1` filter) or for a size missing from the maps.
    """
    x, y, s = leaf
    if s <= 1:
        return 0.0
    rows = pmap.get(s)
    if rows is None:
        return 0.0
    return rows[y // s][x // s]
