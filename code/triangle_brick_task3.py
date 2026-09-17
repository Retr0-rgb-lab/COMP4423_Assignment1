"""
Task 3: Adaptive multi-size + multi-color triangle-brick mosaic.

Pipeline:
    1. Pad image to a multiple of max(S_set).
    2. Partition: quadtree (greedy priority queue by delta-MSE per new triangle)
       OR region-merging (greedy merge of 2x2 siblings with smallest MSE cost).
       Both respect S_set and the triangle budget (2 leaves = 1 cell = 2 triangles;
       split adds 6 triangles net).
    3. For each leaf cell, generate two right-isosceles triangles (checkerboard
       diagonal orientation).
    4. Per-triangle BGR mean (cv2.fillPoly + cv2.mean).
    5. Build palette of size K via {kmeans_lab, kmeans_rgb, median_cut}.
    6. Quantize each triangle to its nearest palette color.
    7. Render: fillPoly + thin gray border.
    8. Metrics (reuse Task 2's 9 metrics) + per-size histogram + palette swatch.
    9. JSON dump + PNG outputs.

Five experiment groups (12 runs total):
    A_ksweep      : K in {4, 8, 16}                      (3)
    B_sweep       : S_set in {{1,2,4}, {1,2,4,8}, {1,2,4,8,16}} (3)
    C_algorithm   : partition in {quadtree, region_merge}(2)
    D_palette     : palette in {kmeans_lab, median_cut} (2)
    E_priority    : priority in {mse, edgef1}            (2)
"""
import argparse
import csv
import heapq
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage import color as skcolor
from skimage import filters

# Reuse Task 2's metric suite
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from triangle_brick import compute_metrics  # noqa: E402


DEFAULT_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics", "sky.jpg")
DEFAULT_OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics", "task3")
MAX_TRIANGLES = 10000
BORDER_COLOR_BGR = (60, 60, 60)
N_TRI_BUDGET = 9990  # leave 10-tri safety margin
PALETTE_METHODS = {
    "kmeans_lab": "K-Means on Lab",
    "kmeans_rgb": "K-Means on RGB",
    "median_cut": "Median Cut on RGB",
}
PARTITION_METHODS = {
    "quadtree": "Quadtree (top-down)",
    "region_merge": "Region Merging (bottom-up)",
}
PRIORITY_METHODS = {
    "mse": "ΔMSE per new triangle",
    "edgef1": "Sobel-response variance",
}


@dataclass
class Task3Config:
    name: str
    group: str
    # S_set must be a set of powers of two; the default range is chosen so that
    # S_max=32 starting cells already fit the 10000-triangle budget on a
    # 1706x1279 image (~4240 triangles). Without S_max=32, quadtree top-down
    # can't even start — the S=8 grid alone gives 68480 triangles.
    S_set: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    partition: str = "quadtree"   # top-down; falls back to region_merge if S_max too small
    priority: str = "mse"
    K: int = 8
    palette_method: str = "kmeans_lab"
    seed: int = 0

    def label(self):
        bits = [
            f"K={self.K}",
            f"S={self.S_set}",
            f"partition={self.partition}",
            f"priority={self.priority}",
            f"palette={self.palette_method}",
        ]
        return " | ".join(bits)


# ---------------------------------------------------------------------------
# Partition algorithms
# ---------------------------------------------------------------------------

def _pad_to_max(img, S_max):
    """Pad image (with reflected border) to a multiple of S_max."""
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


def quadtree_partition(img, cfg: Task3Config):
    """Top-down quadtree + greedy priority queue (RDO).

    Start at the COARSEST grid (S_max, fewest triangles), then repeatedly
    split the highest-priority leaf while the budget still has room
    (n_tri + 6 <= budget). Splitting is how cell sizes become adaptive: the
    biggest MSE-reduction cells (edges / texture) get subdivided first, flat
    cells stay coarse.

    If the starting grid already EXCEEDS the budget (S_max too small for the
    image size), top-down cannot help — splitting only increases the count.
    In that regime we fall back to region_merge (bottom-up).
    """
    padded, orig_shape, _pad = _pad_to_max(img, max(cfg.S_set))
    S_max = max(cfg.S_set)
    S_min = min(cfg.S_set)
    Hp, Wp = padded.shape[:2]
    t0 = time.time()

    # All leaves at S_max (coarsest grid)
    leaves = {}
    for i in range(Hp // S_max):
        for j in range(Wp // S_max):
            x, y = j * S_max, i * S_max
            region = padded[y:y + S_max, x:x + S_max]
            leaves[(x, y, S_max)] = _region_mse(region)
    n_tri = len(leaves) * 2
    print(f"    [qt] start S_max={S_max}: {len(leaves)} cells, {n_tri} triangles "
          f"(budget {N_TRI_BUDGET}) ({time.time()-t0:.2f}s)")

    # Starting grid over budget -> top-down cannot reduce; use region merge.
    if n_tri > N_TRI_BUDGET:
        print(f"    [qt] starting grid over budget; falling back to region_merge")
        return region_merge_partition(img, cfg)

    # Priority queue of split candidates (higher priority first via negation)
    heap = []
    for leaf in leaves:
        p = _quadtree_priority(padded, leaf, cfg.priority)
        heapq.heappush(heap, (-p, leaf))

    iters = 0
    # KEY FIX: keep splitting WHILE there is room in the budget.
    # (Previous bug: `n_tri > budget` -> never split when under budget.)
    while heap and n_tri + 6 <= N_TRI_BUDGET:
        neg_p, leaf = heapq.heappop(heap)
        if -neg_p <= 0:
            break  # no more useful splits
        x, y, size = leaf
        if size <= S_min:
            continue  # already at smallest allowed size
        if leaf not in leaves:
            continue  # stale entry (parent was merged/split earlier)
        half = size // 2
        children = [
            (x, y, half),
            (x + half, y, half),
            (x, y + half, half),
            (x + half, y + half, half),
        ]
        del leaves[leaf]
        n_tri += 6  # 1 cell (2 tri) -> 4 cells (8 tri)
        for c in children:
            cx, cy, cs = c
            region = padded[cy:cy + cs, cx:cx + cs]
            leaves[c] = _region_mse(region)
            p = _quadtree_priority(padded, c, cfg.priority)
            heapq.heappush(heap, (-p, c))
        iters += 1
        if iters % 500 == 0:
            print(f"    [qt] splits={iters} n_tri={n_tri} cells={len(leaves)} "
                  f"heap={len(heap)} ({time.time()-t0:.2f}s)")

    print(f"    [qt] done: splits={iters}, n_tri={n_tri}, cells={len(leaves)} "
          f"({time.time()-t0:.2f}s)")
    return list(leaves.keys()), padded.shape[:2], orig_shape


def _quadtree_priority(img, leaf, priority):
    """Priority for splitting a leaf. ΔMSE/6 (or sobel-var)."""
    x, y, size = leaf
    if size <= 1:
        return 0.0
    region = img[y:y + size, x:x + size]
    if priority == "edgef1":
        return _region_sobel_var(region)
    # MSE priority: split gain per new triangle
    half = size // 2
    mse_parent = _region_mse(region)
    mse_children = 0
    for dy, dx in [(0, 0), (0, half), (half, 0), (half, half)]:
        sub = region[dy:dy + half, dx:dx + half]
        mse_children += _region_mse(sub)
    delta = mse_parent - mse_children
    return delta / 6.0  # per new triangle


def region_merge_partition(img, cfg: Task3Config):
    """Bottom-up region merging, MULTI-SCALE BFS.

    Process cell sizes in ascending order: first exhaust all size-2*S_min
    candidates (cheapest, in flat regions), then size-4*S_min, ..., up to S_max.
    Within each size level, greedy by per-triangle SSE reduction. This
    guarantees the size-2 -> 4 -> 8 -> ... chain actually forms (a naive
    mixed-size heap can leave size-N candidates stale because their 4 children
    span cells that get merged at higher sizes first).
    """
    padded, orig_shape, _pad = _pad_to_max(img, max(cfg.S_set))
    S_max = max(cfg.S_set)
    S_min = min(cfg.S_set)
    Hp, Wp = padded.shape[:2]
    t0 = time.time()

    if S_max % S_min != 0:
        raise ValueError(f"S_min={S_min} must divide S_max={S_max} for region-merge")
    if (S_max // S_min) & ((S_max // S_min) - 1) != 0:
        raise ValueError(f"S_max/S_min must be a power of 2; got {S_max}/{S_min}")

    # SATs (1st & 2nd moments, per channel)
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

    # Initial leaves: full S_min grid (each is one pixel so SSE=0).
    leaves = {(j * S_min, i * S_min, S_min): np.zeros(3, dtype=np.float64)
              for i in range(Hp // S_min)
              for j in range(Wp // S_min)}
    n_tri = len(leaves) * 2
    print(f"    [rm] init {len(leaves)} cells ({time.time()-t0:.2f}s)")

    if n_tri <= N_TRI_BUDGET:
        print(f"    [rm] already fits budget: {n_tri} <= {N_TRI_BUDGET}")
        return list(leaves.keys()), padded.shape[:2], orig_shape

    merges_done = 0
    merges_by_size = {}

    # Multi-scale: process each target size in ascending order.
    target_size = 2 * S_min
    while target_size <= S_max and n_tri > N_TRI_BUDGET:
        # Candidates MUST be aligned to the target_size grid (top-left at a
        # multiple of target_size), otherwise the merged cells are misaligned
        # and can never form the next level's 2x2 blocks. With alignment, the
        # merge tiling is complete: (Hp/target_size)*(Wp/target_size) blocks of
        # 4 children exactly cover all (target_size/2)-cells.
        heap = []
        half = target_size // 2
        for i in range(0, Hp - target_size + 1, target_size):
            for j in range(0, Wp - target_size + 1, target_size):
                X, Y = j, i   # j/i are already pixel positions (step=target_size)
                children = [(X, Y, half), (X + half, Y, half),
                            (X, Y + half, half), (X + half, Y + half, half)]
                if not all(c in leaves for c in children):
                    continue
                if (X, Y, target_size) in leaves:
                    continue
                _, sse_p = rect_sse(X, Y, target_size, target_size)
                sse_c = leaves[children[0]] + leaves[children[1]] \
                      + leaves[children[2]] + leaves[children[3]]
                # Per-triangle SSE reduction; negate so heapq.min pops largest first.
                cost = -float((sse_c - sse_p).sum() / 6.0)
                heapq.heappush(heap, (cost, X, Y, target_size))
        print(f"    [rm] size={target_size}: {len(heap)} candidates "
              f"({time.time()-t0:.2f}s)")

        # Greedy merge at this level
        level_merges = 0
        while heap and n_tri > N_TRI_BUDGET:
            cost, x, y, size = heapq.heappop(heap)
            half = size // 2
            children = ((x, y, half), (x + half, y, half),
                        (x, y + half, half), (x + half, y + half, half))
            if not all(c in leaves for c in children):
                continue
            if (x, y, size) in leaves:
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

    print(f"    [rm] done: merges={merges_done} n_tri={n_tri} ({time.time()-t0:.2f}s)")
    print(f"    [rm] merges by size: {dict(sorted(merges_by_size.items()))}")
    return list(leaves.keys()), padded.shape[:2], orig_shape


# ---------------------------------------------------------------------------
# Color: palette generation + nearest-neighbor quantization
# ---------------------------------------------------------------------------

def palette_kmeans(means_bgr, k, space="lab", seed=0):
    if space == "lab":
        means_rgb = means_bgr[:, ::-1].astype(np.float64) / 255.0
        means_in = skcolor.rgb2lab(np.clip(means_rgb, 0, 1)).astype(np.float32)
    else:
        means_in = means_bgr[:, ::-1].astype(np.float32)  # BGR->RGB, keep as features
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, _, centers = cv2.kmeans(means_in, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    if space == "lab":
        centers_rgb = skcolor.lab2rgb(centers.reshape(1, -1, 3)).reshape(-1, 3)
        centers_bgr = np.clip(centers_rgb * 255, 0, 255).astype(np.uint8)[:, ::-1]
    else:
        centers_bgr = centers[:, ::-1].astype(np.uint8)
    return np.clip(centers_bgr, 0, 255).astype(np.uint8)


def palette_median_cut(means_bgr, k):
    """Recursive median-cut in RGB. O(n log k)."""
    arr = means_bgr[:, ::-1].astype(np.float32).reshape(-1, 3)  # RGB
    boxes = [arr]

    while len(boxes) < k and len(boxes) < arr.shape[0]:
        # pick box with largest range
        ranges = np.array([b.max(axis=0) - b.min(axis=0) for b in boxes])
        idx = int(np.argmax(ranges.max(axis=1)))
        box = boxes[idx]
        if len(box) < 2:
            break
        ch = int(np.argmax(ranges[idx]))
        sorted_box = box[box[:, ch].argsort()]
        mid = len(sorted_box) // 2
        boxes[idx] = sorted_box[:mid]
        boxes.insert(idx + 1, sorted_box[mid:])

    palette_rgb = np.array([b.mean(axis=0) for b in boxes])
    while len(palette_rgb) < k:
        palette_rgb = np.vstack([palette_rgb, palette_rgb[-1:]])
    palette_bgr = np.clip(palette_rgb[:, ::-1], 0, 255).astype(np.uint8)
    return palette_bgr[:k]


def build_palette(means_bgr, k, method, seed=0):
    if method == "kmeans_lab":
        return palette_kmeans(means_bgr, k, "lab", seed)
    if method == "kmeans_rgb":
        return palette_kmeans(means_bgr, k, "rgb", seed)
    if method == "median_cut":
        return palette_median_cut(means_bgr, k)
    raise ValueError(f"Unknown palette method: {method}")


def quantize_nearest_bgr(means_bgr, palette_bgr):
    """Nearest palette color in BGR Euclidean space."""
    means_f = means_bgr.astype(np.float32)
    pal_f = palette_bgr.astype(np.float32)
    diff = means_f[:, None, :] - pal_f[None, :, :]
    dist = (diff * diff).sum(axis=2)
    return np.argmin(dist, axis=1).astype(np.uint8)


# ---------------------------------------------------------------------------
# Triangle generation + rendering
# ---------------------------------------------------------------------------

def leaves_to_triangles(leaves):
    """For each leaf (x, y, size), generate two right-isosceles triangles with
    checkerboard-alternating diagonal orientation. Uses the *cell* index
    i=S_max/size and j=cell-grid-position to alternate."""
    # Determine i, j from leaf (x, y, size). We don't know the grid layout, so we
    # just hash by (x, y) -> use (x // size, y // size) parity.
    triangles = []
    for (x, y, size) in leaves:
        i = y // size
        j = x // size
        tl = (x, y)
        tr = (x + size, y)
        bl = (x, y + size)
        br = (x + size, y + size)
        if (i + j) % 2 == 0:
            triangles.append(np.array([tl, tr, br], dtype=np.int32))
            triangles.append(np.array([tl, br, bl], dtype=np.int32))
        else:
            triangles.append(np.array([tl, tr, bl], dtype=np.int32))
            triangles.append(np.array([tr, br, bl], dtype=np.int32))
    return triangles


def triangle_means_bgr(img, triangles):
    H, W = img.shape[:2]
    means = np.zeros((len(triangles), 3), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.uint8)
    for idx, tri in enumerate(triangles):
        mask[:] = 0
        cv2.fillPoly(mask, [tri], 1)
        means[idx] = cv2.mean(img, mask=mask)[:3]
    return means


def render_triangles(canvas_shape, triangles, labels, palette_bgr, orig_shape):
    H, W = canvas_shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for tri, lab in zip(triangles, labels):
        color = tuple(int(v) for v in palette_bgr[lab])
        cv2.fillPoly(canvas, [tri], color)
        cv2.polylines(canvas, [tri], isClosed=True,
                      color=BORDER_COLOR_BGR, thickness=1, lineType=cv2.LINE_8)
    # Crop back to original shape
    oH, oW = orig_shape
    return canvas[:oH, :oW]


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def plot_size_histogram(leaves, save_path):
    """Bar chart of triangle count per size."""
    sizes = [s for (_, _, s) in leaves]
    counts = {}
    for s in sizes:
        counts[s] = counts.get(s, 0) + 2  # 2 triangles per cell
    items = sorted(counts.items())
    sizes_x = [str(s) for s, _ in items]
    tri_counts = [c for _, c in items]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(sizes_x, tri_counts, color="#4C72B0")
    for bar, c in zip(bars, tri_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c + max(tri_counts) * 0.01,
                str(c), ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Triangle size S (right-angle leg, pixels)")
    ax.set_ylabel("Triangle count")
    ax.set_title(f"Per-size distribution  (total = {sum(tri_counts)})")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_palette(palette_bgr, save_path, K):
    """Swatch row showing palette colors."""
    fig, ax = plt.subplots(figsize=(max(6, K * 0.6), 1.2))
    for i, color_bgr in enumerate(palette_bgr):
        rgb = (color_bgr[2] / 255, color_bgr[1] / 255, color_bgr[0] / 255)
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=rgb))
        ax.text(i + 0.5, 0.5, f"{color_bgr[0]}\n{color_bgr[1]}\n{color_bgr[2]}",
                ha="center", va="center", color="white" if sum(color_bgr) < 200 else "black",
                fontsize=6, fontfamily="monospace")
    ax.set_xlim(0, K)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Palette (K={K})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def run_experiment(img_orig, cfg, out_dir):
    t0 = time.time()
    print(f"\n=== {cfg.group}/{cfg.name} ===")
    print(f"  {cfg.label()}")

    # 1. Partition
    if cfg.partition == "quadtree":
        leaves, padded_shape, orig_shape = quadtree_partition(img_orig, cfg)
    elif cfg.partition == "region_merge":
        leaves, padded_shape, orig_shape = region_merge_partition(img_orig, cfg)
    else:
        raise ValueError(f"Unknown partition: {cfg.partition}")
    n_tri = len(leaves) * 2
    print(f"  partition done: {len(leaves)} cells, {n_tri} triangles, "
          f"{time.time()-t0:.2f}s")

    # 2. Triangles + means (use padded canvas for region-mean, then crop)
    triangles = leaves_to_triangles(leaves)
    padded_img, _, _ = _pad_to_max(img_orig, max(cfg.S_set))
    means = triangle_means_bgr(padded_img, triangles)

    # 3. Palette
    t1 = time.time()
    palette = build_palette(means, cfg.K, cfg.palette_method, seed=cfg.seed)
    print(f"  palette ({cfg.palette_method}, K={cfg.K}) done: {time.time()-t1:.2f}s")

    # 4. Quantize
    labels = quantize_nearest_bgr(means, palette)
    counts = np.bincount(labels, minlength=cfg.K).tolist()

    # 5. Render
    canvas = render_triangles(padded_shape, triangles, labels, palette, orig_shape)
    elapsed = time.time() - t0

    # 6. Metrics (reuse)
    metrics = compute_metrics(img_orig, canvas, means[:n_tri], labels[:n_tri],
                              palette, n_tri, elapsed)
    metrics["N_Triangles"] = float(n_tri)
    metrics["N_Cells"] = float(len(leaves))
    metrics["Budget_Util"] = n_tri / MAX_TRIANGLES
    metrics["Palette_Util"] = float(sum(1 for c in counts if c > 0)) / cfg.K
    metrics["K"] = float(cfg.K)
    metrics["S_set"] = cfg.S_set
    metrics["partition"] = cfg.partition
    metrics["priority"] = cfg.priority
    metrics["palette_method"] = cfg.palette_method
    metrics["counts_per_K"] = counts
    metrics["group"] = cfg.group
    metrics["name"] = cfg.name

    # 7. Outputs
    os.makedirs(out_dir, exist_ok=True)
    canvas_path = os.path.join(out_dir, f"{cfg.name}.png")
    cv2.imwrite(canvas_path, canvas)
    hist_path = os.path.join(out_dir, f"{cfg.name}_size_hist.png")
    plot_size_histogram(leaves, hist_path)
    pal_path = os.path.join(out_dir, f"{cfg.name}_palette.png")
    plot_palette(palette, pal_path, cfg.K)
    metrics_path = os.path.join(out_dir, f"{cfg.name}_metrics.json")
    metrics_to_dump = {k: v for k, v in metrics.items() if k != "Heatmap"}
    with open(metrics_path, "w") as f:
        json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist())
                   for k, v in metrics_to_dump.items()}, f, indent=2)

    print(f"  outputs: {canvas_path}, {hist_path}, {pal_path}, {metrics_path}")
    print(f"  metrics: SSIM={metrics['SSIM']:.4f} PSNR={metrics['PSNR']:.2f} "
          f"ΔE={metrics['Delta_E_2000']:.2f} EdgeF1={metrics['Edge_F1']:.4f} "
          f"EPI={metrics['EPI']:.4f}  ({elapsed:.2f}s)")
    return canvas, metrics


# ---------------------------------------------------------------------------
# 12 experiment configs
# ---------------------------------------------------------------------------

def build_experiment_grid(out_root):
    # Per-group configs: each one explicitly passes every field. Use defaults
    # from Task3Config dataclass for the omitted ones.

    configs = []

    # A. K sweep
    for K in [4, 8, 16]:
        c = Task3Config(name=f"k{K:02d}", group="A_ksweep", K=K)
        configs.append(c)

    # B. S-set sweep (subsets of powers-of-two that include S_max=32 so quadtree
# finishes in microseconds; S_min=1 is impractical under the 10000 budget.)
    s_sets = [[2, 4, 8, 16, 32], [4, 8, 16, 32], [8, 16, 32]]
    for s_set in s_sets:
        tag = "_".join(str(s) for s in s_set)
        c = Task3Config(name=f"s_{tag}", group="B_sweep", S_set=s_set)
        configs.append(c)

    # C. Partition algorithm — use S_set=[4,8,16,32] so BOTH algorithms are
    # feasible: quadtree starts at 32 and splits down to 4; region_merge starts
    # at 4 and merges up to 32. (With S_min=1, region_merge is too slow.)
    for p in ["quadtree", "region_merge"]:
        c = Task3Config(name=p, group="C_algorithm", partition=p,
                        S_set=[4, 8, 16, 32])
        configs.append(c)

    # D. Palette method
    for pm in ["kmeans_lab", "median_cut"]:
        c = Task3Config(name=pm, group="D_palette", palette_method=pm)
        configs.append(c)

    # E. Priority
    for pr in ["mse", "edgef1"]:
        c = Task3Config(name=f"prio_{pr}", group="E_priority", priority=pr)
        configs.append(c)

    # Tag out dirs
    for c in configs:
        c.out_dir = os.path.join(out_root, c.group)
    return configs


def build_summary(out_root, all_metrics):
    """Aggregate metrics from all 12 runs into CSV + comparison chart."""
    summary_dir = os.path.join(out_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    csv_path = os.path.join(summary_dir, "metrics_table.csv")
    keys = ["group", "name", "K", "S_set", "partition", "priority", "palette_method",
            "N_Triangles", "N_Cells", "Budget_Util", "Palette_Util",
            "PSNR", "SSIM", "MS-SSIM", "Delta_E_2000",
            "Edge_F1", "Edge_Precision", "Edge_Recall",
            "EPI", "Quant_Error", "FPS"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for m in all_metrics:
            row = []
            for k in keys:
                v = m.get(k, "")
                if isinstance(v, list):
                    v = str(v)
                row.append(v)
            w.writerow(row)
    print(f"\nSummary CSV: {csv_path}")

    # Comparison chart
    higher_keys = ["SSIM", "MS-SSIM", "Edge_F1", "EPI"]
    lower_keys = [("PSNR", "PSNR (dB)"), ("Delta_E_2000", "ΔE2000"),
                  ("Quant_Error", "Quant Error (ΔE)")]
    labels = [f"{m['group'][0]}:{m['name']}" for m in all_metrics]
    n = len(labels)
    fig, (ax_h, ax_l) = plt.subplots(1, 2, figsize=(max(12, n * 0.6), 5))
    x = np.arange(len(higher_keys))
    width = 0.8 / n
    colors = plt.cm.tab20(np.linspace(0, 1, n))
    for i, m in enumerate(all_metrics):
        vals = [m[k] for k in higher_keys]
        ax_h.bar(x + i * width - 0.4 + width / 2, vals, width, color=colors[i],
                 label=labels[i])
    ax_h.set_xticks(x)
    ax_h.set_xticklabels(higher_keys, rotation=0)
    ax_h.set_ylim(0, 1.05)
    ax_h.set_ylabel("Higher is better")
    ax_h.set_title("[0,1]-normalized metrics")
    ax_h.legend(fontsize=7, ncol=2, loc="lower right")
    ax_h.grid(axis="y", linestyle="--", alpha=0.3)

    x = np.arange(len(lower_keys))
    for i, m in enumerate(all_metrics):
        vals = [m[k] for k, _ in lower_keys]
        ax_l.bar(x + i * width - 0.4 + width / 2, vals, width, color=colors[i],
                 label=labels[i])
    ax_l.set_xticks(x)
    ax_l.set_xticklabels([lab for _, lab in lower_keys], rotation=15, ha="right")
    ax_l.set_ylabel("Lower is better")
    ax_l.set_title("Absolute-scale metrics")
    ax_l.legend(fontsize=7, ncol=2, loc="upper right")
    ax_l.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Task 3 experiment grid — all 12 runs", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    chart_path = os.path.join(summary_dir, "metrics_chart.png")
    fig.savefig(chart_path, dpi=120)
    plt.close(fig)
    print(f"Summary chart: {chart_path}")

    return csv_path, chart_path


def main_func():
    parser = argparse.ArgumentParser(description="Task 3: adaptive triangle brick mosaic.")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT)
    parser.add_argument("--out-root", "-o", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--groups", nargs="+",
                        choices=["A", "B", "C", "D", "E", "all"], default=["all"],
                        help="Which experiment groups to run (default: all).")
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(args.input)
    print(f"[task3] input = {args.input}  shape = {img.shape[:2]}")

    configs = build_experiment_grid(args.out_root)
    if "all" not in args.groups:
        configs = [c for c in configs if c.group[0] in args.groups]

    all_metrics = []
    for cfg in configs:
        _, m = run_experiment(img, cfg, cfg.out_dir)
        all_metrics.append(m)

    csv_path, chart_path = build_summary(args.out_root, all_metrics)
    print(f"\n[task3] All {len(configs)} experiments done.")


if __name__ == "__main__":
    main_func()