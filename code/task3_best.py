"""
Task 3: THE chosen best configuration + comparison renders.

Chosen configuration (balanced, not pure-ΔE):

    partition = quadtree          balanced: near-best ΔE with much better
                                  structure than region_merge (which merges
                                  away all fine cells)
    S_max     = 32                best perceptual colour (B2_smax sweep)
    S_min     = 1                 quadtree can reach fine cells where needed
    K         = 16                best ΔE2000 + quant error (A_ksweep)
    priority  = mse (ΔMSE / 6)    beats Sobel priority (E_priority)
    palette   = kmeans_lab        better ΔE than median_cut (D_palette)

cv2.setRNGSeed(0) makes the K-Means palette reproducible.

Why quadtree over region_merge (which had the single lowest ΔE, 8.63):
    region_merge collapses all cells to size {16, 32} — it drops every fine
    cell, so Edge F1 / EPI are the worst of the grid (0.298 / +0.015).
    quadtree keeps a {4, 8, 16, 32} mix: ΔE 9.03 (only 0.4 worse) but
    Edge F1 0.339 and EPI +0.047 (much better structure).

Output: code/pics/task3/best/best_config.png + best_compare.png
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brick_geom import quadtree_partition, region_merge_partition, pad_to_max, leaves_to_triangles  # noqa
from brick_render import triangle_means_bgr, render_triangles  # noqa
from brick_color import build_palette, quantize_nearest_bgr  # noqa
from brick_metrics import compute_metrics  # noqa

DEFAULT_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics", "sky.jpg")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics", "task3", "best")
BUDGET = 9990

# The chosen configuration.
BEST = ("quadtree_smax32_k16_lab", "quadtree", [1, 2, 4, 8, 16, 32], 16, "kmeans_lab")

# Kept for the comparison figure (region_merge had lower ΔE but worse structure).
OTHERS = [
    ("region_merge_smax32_k16_lab", "region_merge", [4, 8, 16, 32], 16, "kmeans_lab"),
    ("region_merge_smax32_k16_median", "region_merge", [4, 8, 16, 32], 16, "median_cut"),
]


def run(img, partition, s_set, k, palette_method):
    t0 = time.time()
    if partition == "quadtree":
        leaves, ps, osz = quadtree_partition(img, s_set, BUDGET, "mse")
    else:
        leaves, ps, osz = region_merge_partition(img, s_set, BUDGET)
    tris = leaves_to_triangles(leaves)
    padded, _, _ = pad_to_max(img, max(s_set))
    means = triangle_means_bgr(padded, tris)
    pal = build_palette(means, k, palette_method)
    labels = quantize_nearest_bgr(means, pal)
    canvas = render_triangles(ps, tris, labels, pal, osz)
    m = compute_metrics(img, canvas, means, labels, pal, len(leaves) * 2,
                        time.time() - t0)
    sizes = {}
    for (_, _, s) in leaves:
        sizes[s] = sizes.get(s, 0) + 1
    return canvas, m, sizes


def report(name, m, sizes):
    print(f"{name:32s} n_tri={m['N_Triangles']:.0f} "
          f"SSIM={m['SSIM']:.4f} PSNR={m['PSNR']:.2f} "
          f"dE={m['Delta_E_2000']:.3f} EdgeF1={m['Edge_F1']:.4f} "
          f"EPI={m['EPI']:+.4f}  sizes={dict(sorted(sizes.items()))}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img = cv2.imread(DEFAULT_INPUT)

    results = []
    best_canvas = None
    for name, partition, s_set, k, pm in [BEST] + OTHERS:
        cv2.setRNGSeed(0)  # reproducible K-Means
        canvas, m, sizes = run(img, partition, s_set, k, pm)
        report(name, m, sizes)
        results.append((name, canvas, m, sizes))
        cv2.imwrite(os.path.join(OUT_DIR, f"{name}.png"), canvas)
        if name == BEST[0]:
            best_canvas = canvas

    cv2.imwrite(os.path.join(OUT_DIR, "best_config.png"), best_canvas)
    print(f"\n[best] chosen config = {BEST[0]}  -> {OUT_DIR}/best_config.png")

    # Compare grid: original + the three renders
    H, W = img.shape[:2]
    pad, label_h = 16, 40
    cell_h = H + label_h
    grid = np.full((cell_h * 2 + pad * 3, W * 2 + pad * 3, 3), 30, dtype=np.uint8)
    items = [("original", img, None)] + [(n, c, m) for n, c, m, _ in results]
    for i, (name, canvas, m) in enumerate(items[:4]):
        r, c = divmod(i, 2)
        x0, y0 = pad + c * (W + pad), pad + r * (cell_h + pad)
        grid[y0:y0 + H, x0:x0 + W] = canvas
        txt = name if m is None else (
            f"{name}  ΔE={m['Delta_E_2000']:.2f}  SSIM={m['SSIM']:.3f}  "
            f"EdgeF1={m['Edge_F1']:.3f}")
        cv2.putText(grid, txt, (x0 + 10, y0 + H + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cmp_path = os.path.join(OUT_DIR, "best_compare.png")
    cv2.imwrite(cmp_path, grid)
    print(f"[best] compare grid: {cmp_path}")


if __name__ == "__main__":
    main()
