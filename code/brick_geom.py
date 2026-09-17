"""
brick_geom — geometry engine for the triangle-brick assignment.

Two tessellation families, both producing right-isosceles triangles:

  * Uniform grid (Task 2): `compute_grid` + `iter_triangles`.
  * Adaptive (Task 3):
      - `quadtree_partition`      top-down RDO split.
      - `region_merge_partition`  bottom-up aligned multi-scale merge.
      - `leaves_to_triangles`     turn leaf cells into triangles.

Everything here is pure geometry; no colour, rendering, metrics, or I/O.
"""
import heapq
import time

import cv2
import numpy as np
from skimage import filters


MAX_TRIANGLES = 10000


# ---------------------------------------------------------------------------
# Uniform grid (Task 2)
# ---------------------------------------------------------------------------

def compute_grid(H, W, max_triangles=MAX_TRIANGLES):
    """Smallest S such that 2 * M * N <= max_triangles, where M = H // S, N = W // S."""
    for S in range(1, min(H, W) + 1):
        M, N = H // S, W // S
        if M >= 1 and N >= 1 and 2 * M * N <= max_triangles:
            return M, N, S
    raise ValueError(f"Image {H}x{W} cannot be tiled under {max_triangles} triangles.")


def iter_triangles(M, N, S):
    """Yield (3,2) int32 vertex arrays for an M x N grid of S x S cells.

    Diagonal orientation alternates like a checkerboard.
    """
    for i in range(M):
        for j in range(N):
            tl = (j * S, i * S)
            tr = (j * S + S, i * S)
            bl = (j * S, i * S + S)
            br = (j * S + S, i * S + S)
            if (i + j) % 2 == 0:
                yield np.array([tl, tr, br], dtype=np.int32)
                yield np.array([tl, br, bl], dtype=np.int32)
            else:
                yield np.array([tl, tr, bl], dtype=np.int32)
                yield np.array([tr, br, bl], dtype=np.int32)


# ---------------------------------------------------------------------------
# Adaptive tessellation helpers
# ---------------------------------------------------------------------------

def pad_to_max(img, S_max):
    """Pad image (reflected border) to a multiple of S_max."""
    H, W = img.shape[:2]
    pad_h = (S_max - H % S_max) % S_max
    pad_w = (S_max - W % S_max) % S_max
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return padded, (H, W), (pad_h, pad_w)


def _region_mse(region):
    """Sum of squared deviations from the per-channel mean."""
    mean = region.reshape(-1, region.shape[-1]).mean(axis=0)
    diff = region.astype(np.float32) - mean
    return float((diff * diff).sum())


def _region_sobel_var(region):
    """Variance of Sobel response (higher = more edges)."""
    if region.ndim == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region
    s = filters.sobel(gray.astype(np.float64))
    return float(s.var())


def quadtree_partition(img, S_set, max_triangles=MAX_TRIANGLES, priority="mse"):
    """Top-down quadtree + greedy priority queue (rate-distortion optimisation).

    Start at the COARSEST grid (S_max, fewest triangles), then repeatedly split
    the highest-priority leaf while the budget still has room. Splitting is how
    cell sizes become adaptive: the biggest MSE-reduction cells (edges/texture)
    get subdivided first, flat cells stay coarse.

    If the starting grid already EXCEEDS the budget, top-down cannot help
    (splitting only increases the count) and we fall back to region_merge.

    Returns (leaves, padded_shape, orig_shape):
        leaves        list of (x, y, size) with x,y top-left in padded coords
        padded_shape  (Hp, Wp) of the padded image
        orig_shape    (H, W) of the input
    """
    padded, orig_shape, _ = pad_to_max(img, max(S_set))
    S_max, S_min = max(S_set), min(S_set)
    Hp, Wp = padded.shape[:2]
    t0 = time.time()

    leaves = {}
    for i in range(Hp // S_max):
        for j in range(Wp // S_max):
            x, y = j * S_max, i * S_max
            leaves[(x, y, S_max)] = _region_mse(padded[y:y + S_max, x:x + S_max])
    n_tri = len(leaves) * 2
    print(f"    [qt] start S_max={S_max}: {len(leaves)} cells, {n_tri} triangles "
          f"(budget {max_triangles}) ({time.time()-t0:.2f}s)")

    if n_tri > max_triangles:
        print(f"    [qt] starting grid over budget; falling back to region_merge")
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


def _quadtree_priority(img, leaf, priority):
    """Priority for splitting a leaf: ΔMSE/6 (or Sobel variance)."""
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


def region_merge_partition(img, S_set, max_triangles=MAX_TRIANGLES):
    """Bottom-up region merging, MULTI-SCALE BFS with summed-area tables.

    Start with all cells at S_min and greedily merge aligned 2x2 blocks of
    equal size (smallest SSE increase first) until the budget is met.

    Candidates MUST be aligned to the target_size grid, otherwise merged cells
    land off-grid and can never form the next level's 2x2 blocks.

    Returns (leaves, padded_shape, orig_shape), same as quadtree_partition.
    """
    padded, orig_shape, _ = pad_to_max(img, max(S_set))
    S_max, S_min = max(S_set), min(S_set)
    Hp, Wp = padded.shape[:2]
    t0 = time.time()

    if S_max % S_min != 0:
        raise ValueError(f"S_min={S_min} must divide S_max={S_max} for region-merge")
    if (S_max // S_min) & ((S_max // S_min) - 1) != 0:
        raise ValueError(f"S_max/S_min must be a power of 2; got {S_max}/{S_min}")

    # Summed-area tables (1st & 2nd moments, per channel)
    img_d = padded.astype(np.float64)
    S1 = np.zeros((Hp + 1, Wp + 1, 3), dtype=np.float64)
    S2 = np.zeros((Hp + 1, Wp + 1, 3), dtype=np.float64)
    S1[1:, 1:] = img_d.cumsum(axis=0).cumsum(axis=1)
    S2[1:, 1:] = (img_d * img_d).cumsum(axis=0).cumsum(axis=1)

    def rect_sse(x, y, w, h):
        x1, y1 = x + w, y + h
        s1 = S1[y1, x1] - S1[y, x1] - S1[y1, x] + S1[y, x]
        s2 = S2[y1, x1] - S2[y, x1] - S2[y1, x] + S2[y, x]
        n = w * h
        return s1, s2 - s1 * s1 / n

    # Initial leaves: full S_min grid (single pixels have SSE = 0).
    leaves = {(j * S_min, i * S_min, S_min): np.zeros(3, dtype=np.float64)
              for i in range(Hp // S_min)
              for j in range(Wp // S_min)}
    n_tri = len(leaves) * 2
    print(f"    [rm] init {len(leaves)} cells ({time.time()-t0:.2f}s)")

    if n_tri <= max_triangles:
        print(f"    [rm] already fits budget: {n_tri} <= {max_triangles}")
        return list(leaves.keys()), (Hp, Wp), orig_shape

    merges_done = 0
    merges_by_size = {}

    target_size = 2 * S_min
    while target_size <= S_max and n_tri > max_triangles:
        half = target_size // 2
        heap = []
        for i in range(0, Hp - target_size + 1, target_size):
            for j in range(0, Wp - target_size + 1, target_size):
                X, Y = j, i   # loop indices are already pixel positions
                children = [(X, Y, half), (X + half, Y, half),
                            (X, Y + half, half), (X + half, Y + half, half)]
                if not all(c in leaves for c in children):
                    continue
                if (X, Y, target_size) in leaves:
                    continue
                _, sse_p = rect_sse(X, Y, target_size, target_size)
                sse_c = (leaves[children[0]] + leaves[children[1]]
                         + leaves[children[2]] + leaves[children[3]])
                cost = -float((sse_c - sse_p).sum() / 6.0)
                heapq.heappush(heap, (cost, X, Y, target_size))
        print(f"    [rm] size={target_size}: {len(heap)} candidates "
              f"({time.time()-t0:.2f}s)")

        level_merges = 0
        while heap and n_tri > max_triangles:
            cost, x, y, size = heapq.heappop(heap)
            half = size // 2
            children = ((x, y, half), (x + half, y, half),
                        (x, y + half, half), (x + half, y + half, half))
            if not all(c in leaves for c in children) or (x, y, size) in leaves:
                continue
            for c in children:
                del leaves[c]
            _, sse_p = rect_sse(x, y, size, size)
            leaves[(x, y, size)] = sse_p
            n_tri -= 6
            merges_done += 1
            level_merges += 1
            merges_by_size[size] = merges_by_size.get(size, 0) + 1
        print(f"    [rm] size={target_size}: merged {level_merges} "
              f"n_tri={n_tri} ({time.time()-t0:.2f}s)")
        target_size *= 2

    print(f"    [rm] done: merges={merges_done} n_tri={n_tri} "
          f"({time.time()-t0:.2f}s)")
    print(f"    [rm] merges by size: {dict(sorted(merges_by_size.items()))}")
    return list(leaves.keys()), (Hp, Wp), orig_shape


def leaves_to_triangles(leaves):
    """Turn leaf cells (x, y, size) into right-isosceles triangles.

    Each cell splits along a diagonal; orientation alternates like a
    checkerboard based on the cell-grid parity at that size.
    """
    triangles = []
    for (x, y, size) in leaves:
        i, j = y // size, x // size
        tl, tr = (x, y), (x + size, y)
        bl, br = (x, y + size), (x + size, y + size)
        if (i + j) % 2 == 0:
            triangles.append(np.array([tl, tr, br], dtype=np.int32))
            triangles.append(np.array([tl, br, bl], dtype=np.int32))
        else:
            triangles.append(np.array([tl, tr, bl], dtype=np.int32))
            triangles.append(np.array([tr, br, bl], dtype=np.int32))
    return triangles
