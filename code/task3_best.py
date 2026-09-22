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
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brick_geom import pad_to_max, leaves_to_triangles  # noqa
from brick_quadtree import quadtree_partition  # noqa
from brick_region_merge import region_merge_partition  # noqa
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
    """Run one full config end to end and return its render, metrics and sizes.

    Function: the whole Task 3 pipeline for a single configuration, inlined here
    rather than imported from `triangle_brick_task3` so this script stays a
    standalone reproducer for the figures the report cites. Steps: partition ->
    2 triangles per cell -> per-triangle means (on the padded image) -> palette ->
    nearest-colour assignment -> render -> metrics.

    Shapes
    ------
    Input:
      img            : (H, W, 3) uint8 BGR.
      partition      : "quadtree" | "region_merge".
      s_set          : list[int] -- allowed cell sizes, powers of two.
      k              : int -- palette size.
      palette_method : "kmeans_lab" | "kmeans_rgb" | "median_cut".
    Intermediate:
      leaves : list of (x, y, size) in padded coords; `n_tri = 2*len(leaves)`.
      means  : (n_tri, 3) float32, `means[i, 0]=B`.
      pal    : (k, 3) uint8 BGR.
      labels : (n_tri,) uint8 -- `labels[i]` = palette index of triangle i.
    Output:
      (canvas (H, W, 3) uint8 BGR, metrics dict, sizes dict). `sizes` maps cell
      side -> number of CELLS of that side (not triangles; multiply by 2, or
      prefer metrics["counts_per_K"] for triangle counts). `metrics` is the
      `compute_metrics` dict plus N_Triangles.

    Side effect: this function does NOT seed the RNG -- `main` calls
    `cv2.setRNGSeed(0)` before each invocation, deliberately, so the seeding
    sits next to the loop that needs it.
    """
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
    """Print one aligned metrics line for a run (console evidence trail).

    Function: a fixed-width one-liner so several runs can be compared by eye in
    the terminal, and so the numbers quoted in docs/progress/Task3.md can be
    traced back to a run. `dE` here is Delta_E_2000 (ASCII spelling because the
    console is not guaranteed UTF-8).

    Shape: no arrays. Semantics: `m` is a `compute_metrics` dict and is read for
    N_Triangles / SSIM / PSNR / Delta_E_2000 / Edge_F1 / EPI only; `sizes` maps
    cell side -> cell count and is printed sorted ascending, so the printed
    `sizes=` shows the final size MIX, which is the evidence for the
    quadtree-vs-region_merge argument in the module docstring.
    """
    print(f"{name:32s} n_tri={m['N_Triangles']:.0f} "
          f"SSIM={m['SSIM']:.4f} PSNR={m['PSNR']:.2f} "
          f"dE={m['Delta_E_2000']:.3f} EdgeF1={m['Edge_F1']:.4f} "
          f"EPI={m['EPI']:+.4f}  sizes={dict(sorted(sizes.items()))}")


def main():
    """Render the chosen config plus its two rivals, and write the comparison art.

    Function: runs `BEST` followed by `OTHERS` (3 configs, same image, same
    budget), seeds cv2 before each so the K-Means palettes are reproducible, then
    builds a 4-tile compare grid (original + 3 renders) and writes everything
    into `pics/task3/best/`.

    Shapes: input image is (H, W, 3) uint8 BGR; the compare grid is
    `(2*(H+40) + 3*16, 2*W + 3*16, 3)` uint8 BGR. Semantics: the grid's caption
    per tile prints the run name plus Delta_E_2000 / SSIM / Edge_F1, which are
    the three numbers the report's "why quadtree" argument rests on.

    Timing note: the FPS recorded for each run comes from `run`'s own `t0`
    (partition + means + palette + render; metrics excluded), same caveat as the
    other drivers -- see the FPS caveat in brick_metrics.

    Outputs: `best_config.png` (the chosen render alone, the canonical artifact),
    one `<name>.png` per config, and `best_compare.png`.
    """
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
        # Persist the metrics so the report's quoted numbers (e.g. the
        # region_merge dE=8.63 headline in the module docstring) trace back to a
        # product file, not just prose. "Heatmap" is a ndarray and is dropped,
        # matching triangle_brick_task3's on-disk JSON format.
        metrics_to_dump = {k: v for k, v in m.items() if k != "Heatmap"}
        with open(os.path.join(OUT_DIR, f"{name}_metrics.json"), "w") as f:
            json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist())
                       for k, v in metrics_to_dump.items()}, f, indent=2)
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
