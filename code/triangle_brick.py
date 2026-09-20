"""
Task 2: Triangle-brick mosaic (equal-size right-isosceles triangles, 3 colors).

Thin driver over the shared brick_* engine modules:

  brick_geom     compute_grid / iter_triangles  (uniform grid)
  brick_color    quantize_{otsu,kmeans,fixed}
  brick_render   triangle_means_bgr / render_triangles
  brick_metrics  compute_metrics (metric dict + residual heatmap)
  brick_viz      montages + bar chart

This file only parses args, runs the pipeline, and writes outputs.

Pipeline (single method, `--method otsu|kmeans|fixed`):
  img (H, W, 3)
    -> brick_geom.compute_grid      -> (M, N, S)
    -> brick_geom.iter_triangles    -> list of (3, 2) int32 triangles
    -> brick_render.triangle_means_bgr -> (T, 3) float32 means per triangle
    -> brick_color.quantize_*       -> labels (T,), palette (K, 3) BGR
    -> brick_render.render_triangles -> canvas (H, W, 3) uint8 BGR
    -> brick_metrics.compute_metrics -> metric dict (keys listed in brick_metrics)

`--compare` runs all 3 quantization methods on the SAME geometry/means
(fair comparison) and writes per-method PNGs, a 4-tile compare grid, a
residual panel, and a metrics bar chart.

NOTE on the metric count: do not restate a number here. `brick_metrics`'s module
docstring is the single authoritative list of keys and the code has drifted from
prose before ("9 metrics" vs "12 metrics" for the same dict). Count changes go in
`brick_metrics`, not in this header.
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
    """Build the (geometry, colour-means) basis shared by every method.

    Function
    --------
    Runs the front half of the pipeline that does NOT depend on the
    quantization method: figure out the grid, enumerate triangles, sample
    mean BGR colour per triangle. Means are reused across `--method`
    choices so the comparison is fair (only the quantizer differs).

    Shapes
    ------
    Input:
      img : (H, W, 3) uint8 BGR — source image.
    Output:
      H, W       : ints — original image dimensions.
      M, N       : ints — cell-grid rows / columns. NOTE: per
                   `brick_geom.compute_grid`, M corresponds to the height
                   direction (rows) and N to the width direction (cols).
      S          : int  — cell side length in pixels.
      triangles  : list of (3, 2) int32 — 2*M*N triangles total. Each
                   `tri[k, 0]=x (col)`, `tri[k, 1]=y (row)`.
      means      : (T, 3) float32 — `means[i]` is the mean BGR colour of
                   pixels inside `triangles[i]`. `means[i, 0]=B`,
                   `means[i, 1]=G`, `means[i, 2]=R`.
    """
    H, W = img.shape[:2]
    M, N, S = compute_grid(H, W)
    triangles = list(iter_triangles(M, N, S))
    means = triangle_means_bgr(img, triangles)
    return H, W, M, N, S, triangles, means


def quantize(method, means_bgr):
    """Dispatch to one of the three quantizers in `brick_color`.

    Shapes
    ------
    Input:
      method    : "otsu" | "kmeans" | "fixed".
      means_bgr : (T, 3) float32 — per-triangle mean BGR (from
                  `preprocess`).
    Output:
      labels    : (T,) int ndarray — `labels[i]` is the palette index
                  assigned to triangle i, values in [0, K-1].
      palette   : (K, 3) uint8 BGR — `palette[k]` is the colour for
                  palette index k. K is method-dependent (3 for Task 2).
    """
    if method == "otsu":
        return quantize_otsu(means_bgr)
    if method == "kmeans":
        return quantize_kmeans(means_bgr)
    if method == "fixed":
        return quantize_fixed(means_bgr)
    raise ValueError(f"Unknown method: {method}")


def print_metrics_table(metrics_by_method):
    """Pretty-print a markdown-ish table of metrics across methods.

    Shapes
    ------
    Input:
      metrics_by_method : dict {method_name -> metrics_dict}, where each
                          metrics_dict maps a metric name (e.g. "PSNR",
                          "SSIM") to either a float or a non-float
                          (string for FPS, int for N_Triangles, etc.).
    Output:
      None — prints to stdout.
    """
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
    """Run the full T2 pipeline for ONE quantization method.

    Function
    --------
    preprocess -> quantize -> render -> metrics -> save PNG.

    Timing caveat (this is the Task 2 driver, so its FPS is NOT the Task 3 one):
    `t0` is started AFTER `preprocess`, so the reported FPS covers ONLY
    quantize + render. It excludes the per-triangle mean extraction (the most
    expensive step here) and excludes the metric computation. Do not compare
    this FPS against a Task 3 run, whose interval also includes partitioning --
    see the FPS caveat in the brick_metrics module docstring.

    Shapes
    ------
    Input:
      img    : (H, W, 3) uint8 BGR.
      method : "otsu" | "kmeans" | "fixed".
      args   : argparse Namespace; uses `args.output` for the save path.
    Output:
      canvas  : (H, W, 3) uint8 BGR — rendered mosaic.
      metrics : dict — see `brick_metrics.compute_metrics` for keys.
                Also contains the FPS for this method.
    """
    H, W, M, N, S, triangles, means = preprocess(img)
    t0 = time.time()
    labels, palette = quantize(method, means)
    canvas = render_triangles((H, W), triangles, labels, palette)
    elapsed = time.time() - t0
    n_tri = 2 * M * N
    metrics = compute_metrics(img, canvas, means, labels, palette, n_tri, elapsed)
    # `np.bincount(labels, minlength=3)` -> array of length K=3; `.tolist()`
    # converts to a Python list for printing. Index k = how many triangles
    # were assigned palette index k.
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
    """Run T2 for ALL three quantizers on the same grid and dump comparison art.

    Function
    --------
    Same geometry, same means, three different quantizers. Saves:
      out_task2_<method>.png         — per-method rendered canvas
      out_task2_compare.png           — 4-tile grid (original + 3 methods)
      out_task2_residual.png          — per-method JET heatmap panel
      out_task2_metrics_chart.png     — bar chart of all metrics

    Fairness note: `preprocess` runs ONCE and its means are reused for all three
    methods, so the only variable is the quantizer. Timing is therefore also
    measured per method (quantize+render only, same caveat as `run_single`) and
    the geometry cost is deliberately not attributed to any method.

    Shapes
    ------
    Input:
      img  : (H, W, 3) uint8 BGR.
      args : argparse Namespace; uses `args.output` to find the output dir.
    Output:
      grid              : (Gh, Gw, 3) uint8 BGR — the 4-tile compare grid.
      metrics_by_method : dict {method_name -> metrics_dict}; each
                          metrics_dict also has "_canvas" (H, W, 3) — the
                          per-method canvas — and "Heatmap" (H, W, 3) stashed
                          for `make_compare_grid` / `make_residual_panel`.
                          The keys are exactly ["otsu", "kmeans", "fixed"], in
                          that order, which `make_metrics_bar_chart` relies on
                          for its legend and colours.
    """
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
        # Stash canvas + heatmap inside metrics so make_compare_grid /
        # make_residual_panel can find them without separate plumbing.
        metrics["_canvas"] = canvas
        metrics_by_method[method] = metrics
        heatmaps.append((method, metrics["Heatmap"]))

    # 4-tile: original + 3 method canvases side by side.
    grid = make_compare_grid(img, [(m, metrics_by_method[m])
                                   for m in ["otsu", "kmeans", "fixed"]])
    grid_path = os.path.join(out_dir, "out_task2_compare.png")
    cv2.imwrite(grid_path, grid)

    # Per-method JET heatmap panel for the report.
    residual = make_residual_panel([h for _, h in heatmaps],
                                   [m for m, _ in heatmaps])
    res_path = os.path.join(out_dir, "out_task2_residual.png")
    cv2.imwrite(res_path, residual)

    # Bar chart across all metrics / methods (matplotlib figure).
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
    """CLI entry: parse args, load image, dispatch to single or compare run.

    Function
    --------
    Argparse -> cv2.imread -> branch on --compare flag -> show window.
    `--no-show` skips cv2.imshow for headless environments (WSL / CI).

    Shapes
    ------
    No ndarray I/O at the top level; the helpers above own all shapes.
    """
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