"""
Task 3 marginal-benefit analysis (rate-distortion).

Answers: "as we allow more triangles, when does splitting a cell into 4
children stop paying off, and which cell sizes actually contribute?"

Method:
    1. Run the quadtree greedily, recording each split's marginal benefit
       ΔSSE/6 (SSE reduction per new triangle) and the parent cell size.
    2. Plot:
        (a) marginal benefit vs split index  (the sorted priority curve)
        (b) cumulative SSE reduction vs triangles used
        (c) mean marginal benefit grouped by parent cell size
    3. Also evaluate the full pipeline (render + metrics) at several budgets
       to get a ground-truth quality-vs-rate curve.

Outputs to code/pics/task3/analysis/.
"""
import heapq
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brick_geom import pad_to_max, _region_mse  # noqa

DEFAULT_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics", "sky.jpg")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics", "task3", "analysis")


def quadtree_with_history(img, S_set, max_triangles):
    """Run top-down quadtree recording (parent_size, delta_sse) per split.

    Greedy priority = ΔSSE / 6 (higher first). Stops when the next split
    would exceed max_triangles (or heap empty).
    """
    padded, orig_shape, _ = pad_to_max(img, max(S_set))
    S_max, S_min = max(S_set), min(S_set)
    Hp, Wp = padded.shape[:2]

    leaves = {}
    for i in range(Hp // S_max):
        for j in range(Wp // S_max):
            x, y = j * S_max, i * S_max
            region = padded[y:y + S_max, x:x + S_max]
            leaves[(x, y, S_max)] = _region_mse(region)
    n_tri = len(leaves) * 2
    start_tri = n_tri

    def priority(x, y, size):
        if size <= S_min:
            return None
        half = size // 2
        region = padded[y:y + size, x:x + size]
        sse_p = _region_mse(region)
        sse_c = 0.0
        for dy, dx in [(0, 0), (0, half), (half, 0), (half, half)]:
            sse_c += _region_mse(region[dy:dy + half, dx:dx + half])
        return (sse_p - sse_c) / 6.0, sse_p - sse_c

    heap = []
    for leaf in leaves:
        r = priority(*leaf)
        if r is not None:
            heapq.heappush(heap, (-r[0], leaf, r[1]))

    history = []  # (parent_size, marginal_benefit, delta_sse)
    while heap and n_tri + 6 <= max_triangles:
        neg_p, leaf, delta_sse = heapq.heappop(heap)
        x, y, size = leaf
        if leaf not in leaves or size <= S_min:
            continue
        half = size // 2
        children = [(x, y, half), (x + half, y, half),
                    (x, y + half, half), (x + half, y + half, half)]
        del leaves[leaf]
        n_tri += 6
        history.append((size, -neg_p, delta_sse))
        for c in children:
            leaves[c] = _region_mse(padded[c[1]:c[1] + c[2], c[0]:c[0] + c[2]])
            r = priority(*c)
            if r is not None:
                heapq.heappush(heap, (-r[0], c, r[1]))

    return history, start_tri, list(leaves.keys())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img = cv2.imread(DEFAULT_INPUT)
    print(f"[marginal] input {DEFAULT_INPUT} shape={img.shape[:2]}")

    # Use S_max=64 so the quadtree can reach fine sizes if worth it.
    S_set = [1, 2, 4, 8, 16, 32, 64]
    BIG = 60000  # large budget so we see the full benefit curve
    history, start_tri, leaves = quadtree_with_history(img, S_set, BIG)
    sizes = np.array([h[0] for h in history])
    marg = np.array([h[1] for h in history])          # ΔSSE / 6
    dsse = np.array([h[2] for h in history])          # ΔSSE
    tri_used = start_tri + 6 * np.arange(1, len(history) + 1)
    cum_dsse = np.cumsum(dsse)

    print(f"[marginal] S_set={S_set}, start_tri={start_tri}, splits={len(history)}, "
          f"final_tri={tri_used[-1] if len(tri_used) else start_tri}")

    # ---------- Figure 1: marginal benefit vs triangles ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tri_used, marg, lw=1.5, color="#4C72B0")
    ax.axvline(9990, color="red", ls="--", lw=1.5, label="assignment budget 9990")
    ax.set_yscale("log")
    ax.set_xlabel("triangles used")
    ax.set_ylabel("marginal benefit  ΔSSE / new-triangle  (log scale)")
    ax.set_title("Marginal benefit of each split, in greedy (RDO) order")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, "marginal_benefit_curve.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)

    # ---------- Figure 2: cumulative SSE reduction vs triangles ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(tri_used, cum_dsse / cum_dsse[-1], lw=2, color="#55A467")
    ax.axvline(9990, color="red", ls="--", lw=1.5, label="assignment budget 9990")
    ax.set_xlabel("triangles used")
    ax.set_ylabel("cumulative SSE reduction (normalised to 1.0)")
    ax.set_title("Cumulative error reduction vs triangles (diminishing returns)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "cumulative_benefit_curve.png")
    fig.savefig(p2, dpi=120)
    plt.close(fig)

    # ---------- Figure 3: mean marginal benefit by parent size ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    uniq = sorted(set(sizes.tolist()))
    means = [marg[sizes == s].mean() for s in uniq]
    counts = [int((sizes == s).sum()) for s in uniq]
    bars = ax.bar([str(s) for s in uniq], means, color="#DD8452")
    for b, c, m in zip(bars, counts, means):
        ax.text(b.get_x() + b.get_width() / 2, m * 1.02,
                f"n={c}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("parent cell size S (pixels) that got split")
    ax.set_ylabel("mean marginal benefit  ΔSSE / new-triangle")
    ax.set_title("Where does the benefit come from? (mean per split size)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p3 = os.path.join(OUT_DIR, "benefit_by_size.png")
    fig.savefig(p3, dpi=120)
    plt.close(fig)

    # ---------- Console report ----------
    print("\n[marginal] splits by parent size:")
    for s in uniq:
        mask = sizes == s
        print(f"  size {s:3d}: {int(mask.sum()):5d} splits, "
              f"mean ΔSSE/tri = {marg[mask].mean():10.2f}, "
              f"total ΔSSE = {dsse[mask].sum():12.0f}")
    print(f"\n[marginal] outputs: {p1}\n                   {p2}\n                   {p3}")


if __name__ == "__main__":
    main()