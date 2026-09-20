"""
Task 3 analysis: perceptual quality vs the largest allowed cell size (S_max).

This is the figure that tests whether the quality-vs-S_max curve is monotonic
or has an interior optimum. Headline result on sky.jpg:

    ΔE2000   : best at S_max=32 (10.61), then WORSE at 64/128/256 (~11.8, flat)
    SSIM etc.: rises 32 -> 64, then saturates (128 == 256 == 64)

So the curve is NOT monotonic in perceptual colour (it is monotone-worsening
then flat), and the perceptual optimum sits at the smallest S_max the
quadtree can start from on this image (~32; smaller starts exceed the budget).

Palette is Median Cut (deterministic) so runs are reproducible.

Output: code/pics/task3/analysis/smax_curve.png
"""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import sobel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brick_geom import pad_to_max, leaves_to_triangles  # noqa
from brick_quadtree import quadtree_partition  # noqa
from brick_render import triangle_means_bgr, render_triangles  # noqa
from brick_color import build_palette, quantize_nearest_bgr  # noqa
from brick_metrics import compute_metrics, delta_e_2000_full  # noqa

DEFAULT_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics", "sky.jpg")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics", "task3", "analysis")
BUDGET = 9990
S_MAXES = [32, 64, 128, 256]
ALL_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def measure(img, smax, smooth_mask, edge_mask):
    """Render one S_max configuration and return its metrics plus region split.

    Function
    --------
    Runs the standard pipeline for a single S_max (quadtree, K=8, median-cut
    palette) and additionally splits the per-pixel Delta_E_2000 map into flat
    and edge regions. That split is the point of this script: the whole-image
    Delta_E hides the fact that large flat areas and edges move in OPPOSITE
    directions as S_max grows, which is the finding the figure exists to show.

    A note on why the palette is median-cut here and K-Means elsewhere: median
    cut is deterministic, so the four S_max points differ ONLY in S_max. With
    K-Means the palette would change between points and the trend could be
    partly palette noise.

    Shapes
    ------
    Input:
      img         : (H, W, 3) uint8 BGR.
      smax        : int -- largest allowed cell size; also the padding stride.
      smooth_mask : (H, W) bool -- True on flat pixels of `img`.
      edge_mask   : (H, W) bool -- True on the strongest-edge pixels of `img`.
    Intermediate:
      de : (H, W) float64 -- per-pixel Delta_E_2000; masks index it directly
           because it shares the source image's shape.
    Output:
      dict with keys smax, n_tri, SSIM, PSNR, dE, dE_smooth, dE_edge, EdgeF1, EPI.
      Semantics: `dE` is the whole-image mean Delta_E_2000 (lower better);
      `dE_smooth` / `dE_edge` are the means over the two masks -- the
      disagreement between `dE_smooth` and `dE_edge` across S_max values is the
      result. `n_tri` is reported because it varies with S_max (the coarser the
      start, the more splits fit in the budget), so the points are not all at
      the same triangle count.

    Edge cases: `smooth_mask` and `edge_mask` are derived from the SOURCE image
    only (never from the render), so they are identical for every S_max and the
    comparison between points is like-for-like. An empty mask would not raise --
    numpy `.mean()` on an empty slice returns NaN with a warning -- so a NaN in
    `dE_smooth` / `dE_edge` means "no pixels qualified", not a crashed pipeline.
    Not reachable with the percentile thresholds used in `main`.
    """
    s_set = [s for s in ALL_SIZES if s <= smax]
    leaves, ps, osz = quadtree_partition(img, s_set, BUDGET, "mse")
    tris = leaves_to_triangles(leaves)
    padded, _, _ = pad_to_max(img, smax)
    means = triangle_means_bgr(padded, tris)
    pal = build_palette(means, 8, "median_cut")
    labels = quantize_nearest_bgr(means, pal)
    canvas = render_triangles(ps, tris, labels, pal, osz)
    m = compute_metrics(img, canvas, means, labels, pal, len(leaves) * 2, 1.0)
    _, de = delta_e_2000_full(img, canvas)
    return {
        "smax": smax,
        "n_tri": len(leaves) * 2,
        "SSIM": m["SSIM"],
        "PSNR": m["PSNR"],
        "dE": m["Delta_E_2000"],
        "dE_smooth": float(de[smooth_mask].mean()),
        "dE_edge": float(de[edge_mask].mean()),
        "EdgeF1": m["Edge_F1"],
        "EPI": m["EPI"],
    }


def main():
    """Sweep S_max, print the table, and write the two-panel curve figure.

    Function
    --------
    Builds two Sobel-based masks ONCE from the source image (bottom 60% of edge
    response = "flat", top 5% = "edge"), then measures each S_max in S_MAXES and
    plots: panel 1 = the three Delta_E_2000 variants (whole / flat / edge, lower
    is better), panel 2 = SSIM / Edge F1 / EPI (higher is better). Both x-axes
    are log2 with explicit ticks, so the S_max points are evenly spaced.

    The figure's title states the finding rather than the procedure -- "metrics
    disagree" -- which is intentional: a reader skimming figures should not be
    able to mistake this for a monotone improvement curve.

    Shapes: `rows` is a list of `measure` dicts, one per S_max, length
    len(S_MAXES) = 4. The masks are (H, W) bool; `gray` / `sob` are (H, W)
    float64. Output `pics/task3/analysis/smax_curve.png`; returns None.

    Threshold semantics (worth keeping if the masks are ever re-tuned): the
    percentiles are computed on the SOURCE only, and the 60th/95th choice is
    what makes the flat set the large majority of the image -- which is the
    regime that dominates human perception in Task 3's conclusion.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    img = cv2.imread(DEFAULT_INPUT)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    sob = sobel(gray)
    smooth_mask = sob < np.percentile(sob, 60)
    edge_mask = sob >= np.percentile(sob, 95)

    rows = []
    for smax in S_MAXES:
        r = measure(img, smax, smooth_mask, edge_mask)
        rows.append(r)
        print(f"S_max={r['smax']:>3} n_tri={r['n_tri']:>5} "
              f"SSIM={r['SSIM']:.4f} PSNR={r['PSNR']:.2f} "
              f"dE={r['dE']:.3f} dE_smooth={r['dE_smooth']:.3f} "
              f"dE_edge={r['dE_edge']:.3f} EdgeF1={r['EdgeF1']:.4f} "
              f"EPI={r['EPI']:+.4f}")

    x = [r["smax"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: perceptual colour (lower is better)
    ax1.plot(x, [r["dE"] for r in rows], "o-", lw=2, label="ΔE2000 (whole image)")
    ax1.plot(x, [r["dE_smooth"] for r in rows], "s--", lw=1.5,
             label="ΔE2000 (flat regions)")
    ax1.plot(x, [r["dE_edge"] for r in rows], "^:", lw=1.5,
             label="ΔE2000 (edge regions)")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(v) for v in x])
    ax1.set_xlabel("S_max  (largest allowed cell, pixels)")
    ax1.set_ylabel("ΔE2000  (lower = perceptually closer)")
    ax1.set_title("Perceptual colour error vs S_max")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Panel 2: structure / edge metrics (higher is better)
    ax2.plot(x, [r["SSIM"] for r in rows], "o-", lw=2, label="SSIM")
    ax2.plot(x, [r["EdgeF1"] for r in rows], "s--", lw=1.5, label="Edge F1")
    ax2.plot(x, [r["EPI"] for r in rows], "^:", lw=1.5, label="EPI")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(v) for v in x])
    ax2.set_xlabel("S_max  (largest allowed cell, pixels)")
    ax2.set_ylabel("Score  (higher = better)")
    ax2.set_title("Structure / edge metrics vs S_max")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.suptitle("S_max sweep on sky.jpg — perceptual vs structural metrics disagree",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, "smax_curve.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
