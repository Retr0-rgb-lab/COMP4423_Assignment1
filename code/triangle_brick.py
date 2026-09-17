"""
Task 2: Triangle-brick mosaic (equal-size right isosceles triangles, 3 colors).

Thin driver over the shared brick_* engine modules:
    brick_geom     compute_grid / iter_triangles  (uniform grid)
    brick_color    quantize_{otsu,kmeans,fixed}
    brick_render   triangle_means_bgr / render_triangles
    brick_metrics  compute_metrics (9-metric suite)
    brick_viz      montages + bar chart

This file only parses args, runs the pipeline, and writes outputs.
"""
import argparse
import os
import time

import cv2
import numpy as np

from brick_geom import compute_grid, iter_triangles
from brick_color import quantize_otsu, quantize_kmeans, quantize_fixed
from brick_render import triangle_means_bgr, render_triangles
from brick_metrics import compute_metrics
from brick_viz import make_compare_grid, make_residual_panel, make_metrics_bar_chart


DEFAULT_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics", "sky.jpg")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_task2.png")


def preprocess(img):
    H, W = img.shape[:2]
    M, N, S = compute_grid(H, W)
    triangles = list(iter_triangles(M, N, S))
    means = triangle_means_bgr(img, triangles)
    return H, W, M, N, S, triangles, means


def quantize(method, means_bgr):
    if method == "otsu":
        return quantize_otsu(means_bgr)
    if method == "kmeans":
        return quantize_kmeans(means_bgr)
    if method == "fixed":
        return quantize_fixed(means_bgr)
    raise ValueError(f"Unknown method: {method}")


def print_metrics_table(metrics_by_method):
    """Pretty-print a markdown-ish table of metrics."""
    methods = list(metrics_by_method.keys())
    keys = ["PSNR", "SSIM", "MS-SSIM", "Delta_E_2000",
            "Edge_F1", "Edge_Precision", "Edge_Recall",
            "EPI", "Quant_Error", "N_Triangles", "Budget_Util", "FPS"]
    print("\n[task2] Metrics summary (PSNR/SSIM/MS-SSIM/ΔE/Edge/EPI/QuantErr/Budget/FPS):")
    header = "metric".ljust(20) + "  ".join(m.ljust(10) for m in methods)
    print(header)
    print("-" * len(header))
    for k in keys:
        v0 = metrics_by_method[methods[0]][k]
        if isinstance(v0, float):
            row = k.ljust(20) + "  ".join(f"{metrics_by_method[m][k]:<10.4f}"
                                          for m in methods)
        else:
            row = k.ljust(20) + "  ".join(str(metrics_by_method[m][k]).ljust(10)
                                          for m in methods)
        print(row)


def run_single(img, method, args):
    H, W, M, N, S, triangles, means = preprocess(img)
    t0 = time.time()
    labels, palette = quantize(method, means)
    canvas = render_triangles((H, W), triangles, labels, palette)
    elapsed = time.time() - t0
    n_tri = 2 * M * N
    metrics = compute_metrics(img, canvas, means, labels, palette, n_tri, elapsed)
    counts = np.bincount(labels, minlength=3).tolist()

    print(f"[task2:{method}] grid={M}x{N}, S={S}, T={n_tri}")
    print(f"[task2:{method}] palette BGR = {palette.tolist()}")
    print(f"[task2:{method}] counts      = {counts}")
    for k in ["PSNR", "SSIM", "MS-SSIM", "Delta_E_2000", "Edge_F1",
              "Edge_Precision", "Edge_Recall", "EPI", "Quant_Error",
              "Budget_Util", "FPS"]:
        print(f"[task2:{method}] {k:<16}= {metrics[k]:.4f}")
    cv2.imwrite(args.output, canvas)
    print(f"[task2:{method}] output = {args.output}")
    return canvas, metrics


def run_compare(img, args):
    H, W, M, N, S, triangles, means = preprocess(img)
    n_tri = 2 * M * N
    print(f"[task2:compare] grid = {M}x{N}, S={S}, T={n_tri}")

    metrics_by_method = {}
    heatmaps = []
    out_dir = os.path.dirname(os.path.abspath(args.output))
    for method in ["otsu", "kmeans", "fixed"]:
        t0 = time.time()
        labels, palette = quantize(method, means)
        canvas = render_triangles((H, W), triangles, labels, palette)
        elapsed = time.time() - t0
        metrics = compute_metrics(img, canvas, means, labels, palette, n_tri, elapsed)
        counts = np.bincount(labels, minlength=3).tolist()
        print(f"[task2:{method}] palette={palette.tolist()} counts={counts} "
              f"SSIM={metrics['SSIM']:.4f} PSNR={metrics['PSNR']:.2f} "
              f"ΔE={metrics['Delta_E_2000']:.2f} EdgeF1={metrics['Edge_F1']:.4f} "
              f"({elapsed:.2f}s)")
        cv2.imwrite(os.path.join(out_dir, f"out_task2_{method}.png"), canvas)
        metrics["_canvas"] = canvas
        metrics_by_method[method] = metrics
        heatmaps.append((method, metrics["Heatmap"]))

    grid = make_compare_grid(img, [(m, metrics_by_method[m])
                                   for m in ["otsu", "kmeans", "fixed"]])
    grid_path = os.path.join(out_dir, "out_task2_compare.png")
    cv2.imwrite(grid_path, grid)

    residual = make_residual_panel([h for _, h in heatmaps],
                                   [m for m, _ in heatmaps])
    res_path = os.path.join(out_dir, "out_task2_residual.png")
    cv2.imwrite(res_path, residual)

    fig = make_metrics_bar_chart(metrics_by_method)
    chart_path = os.path.join(out_dir, "out_task2_metrics_chart.png")
    fig.savefig(chart_path, dpi=120)
    import matplotlib.pyplot as plt
    plt.close(fig)

    print_metrics_table(metrics_by_method)
    print(f"[task2:compare] compare grid   = {grid_path}")
    print(f"[task2:compare] residual panel = {res_path}")
    print(f"[task2:compare] metrics chart  = {chart_path}")
    return grid, metrics_by_method


def main():
    parser = argparse.ArgumentParser(description="Task 2: triangle-brick 3-color mosaic.")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT)
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT)
    parser.add_argument("--method", "-m",
                        choices=["otsu", "kmeans", "fixed"], default="otsu",
                        help="Quantization method (default: otsu). Ignored if --compare.")
    parser.add_argument("--compare", action="store_true",
                        help="Run all 3 methods, save per-method PNGs, residual heatmaps, bar chart.")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip cv2.imshow (e.g., on WSL).")
    args = parser.parse_args()

    t0 = time.time()
    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {args.input}")
    H, W = img.shape[:2]
    print(f"[task2] input = {args.input}  shape = ({H}, {W})")

    if args.compare:
        grid, metrics = run_compare(img, args)
        if not args.no_show:
            cv2.imshow("Task 2: compare", grid)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        canvas, metrics = run_single(img, args.method, args)
        if not args.no_show:
            cv2.imshow(f"Task 2 ({args.method})", canvas)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print(f"[task2] total elapsed = {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
