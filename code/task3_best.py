"""
Task 3: produce the "best" configuration image, given the analysis conclusions.

Chosen from the experiment grid + the perceptual/structure analysis:
    K        = 16            (best ΔE2000 + quant error; A_ksweep)
    S_max    = 32            (best perceptual colour; B2_smax vs 64)
    priority = mse           (ΔMSE beats Sobel; E_priority)
    palette  = kmeans_lab    (better ΔE than median_cut; D_palette)
    partition= quadtree vs region_merge  (compared here; region_merge wins ΔE
                                          but loses structure)

cv2.setRNGSeed is called so the K-Means palette is reproducible.

Output: code/pics/task3/best/best_config.png and best_compare.png
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

CANDIDATES = [
    # name, partition, S_set, K, palette_method
    ("quadtree_smax32_k16_lab", "quadtree", [1, 2, 4, 8, 16, 32], 16, "kmeans_lab"),
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.setRNGSeed(0)  # reproducible K-Means
    img = cv2.imread(DEFAULT_INPUT)

    results = []
    for name, partition, s_set, k, pm in CANDIDATES:
        cv2.setRNGSeed(0)
        canvas, m, sizes = run(img, partition, s_set, k, pm)
        results.append((name, canvas, m, sizes))
        print(f"{name:32s} n_tri={m['N_Triangles']:.0f} "
              f"SSIM={m['SSIM']:.4f} PSNR={m['PSNR']:.2f} "
              f"dE={m['Delta_E_2000']:.3f} EdgeF1={m['Edge_F1']:.4f} "
              f"EPI={m['EPI']:+.4f}  sizes={dict(sorted(sizes.items()))}")
        cv2.imwrite(os.path.join(OUT_DIR, f"{name}.png"), canvas)

    # Pick the best by ΔE2000 (the perceptual-colour objective)
    best = min(results, key=lambda r: r[2]["Delta_E_2000"])
    print(f"\n[best] by ΔE2000: {best[0]}")

    # Compare grid: original + the three candidates
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
    print(f"[best] per-candidate renders in {OUT_DIR}")


if __name__ == "__main__":
    main()
