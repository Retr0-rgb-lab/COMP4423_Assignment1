"""
Task 3: Adaptive multi-size + multi-color triangle-brick mosaic.

Thin driver over the shared brick_* engine modules:
    brick_geom     quadtree_partition / region_merge_partition / leaves_to_triangles
    brick_color    palette_kmeans / palette_median_cut / build_palette
    brick_render   triangle_means_bgr / render_triangles
    brick_metrics  compute_metrics (9-metric suite)
    brick_viz      plot_size_histogram / plot_palette / make_sweep_bar_chart

This file only defines the experiment grid, runs each config, and writes
outputs. See docs/progress/Task3.md for the full research narrative.

Six experiment groups (15 runs total):
    A_ksweep      : K in {4, 8, 16}                         (3)
    B_sweep       : S_min in {2, 4, 8} (kept as record)     (3)
    B2_smax       : S_max in {32, 64, 128} (improved B)     (3)
    C_algorithm   : partition in {quadtree, region_merge}   (2)
    D_palette     : palette in {kmeans_lab, median_cut}     (2)
    E_priority    : priority in {mse, edgef1}               (2)
"""
import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brick_geom import (  # noqa: E402
    MAX_TRIANGLES, pad_to_max, quadtree_partition, region_merge_partition,
    leaves_to_triangles,
)
from brick_color import build_palette, quantize_nearest_bgr  # noqa: E402
from brick_render import triangle_means_bgr, render_triangles  # noqa: E402
from brick_metrics import compute_metrics  # noqa: E402
from brick_viz import (  # noqa: E402
    plot_size_histogram, plot_palette, make_sweep_bar_chart,
)


DEFAULT_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics", "sky.jpg")
DEFAULT_OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics", "task3")
N_TRI_BUDGET = 9990  # leave a 10-triangle safety margin


@dataclass
class Task3Config:
    name: str
    group: str
    # S_set must be a powers-of-two family; the quadtree halves cell sizes.
    S_set: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    partition: str = "quadtree"
    priority: str = "mse"
    K: int = 8
    palette_method: str = "kmeans_lab"
    seed: int = 0
    out_dir: str = ""

    def label(self):
        return " | ".join([
            f"K={self.K}", f"S={self.S_set}", f"partition={self.partition}",
            f"priority={self.priority}", f"palette={self.palette_method}",
        ])


def run_experiment(img_orig, cfg, out_dir):
    t0 = time.time()
    print(f"\n=== {cfg.group}/{cfg.name} ===")
    print(f"  {cfg.label()}")

    # 1. Partition
    if cfg.partition == "quadtree":
        leaves, padded_shape, orig_shape = quadtree_partition(
            img_orig, cfg.S_set, N_TRI_BUDGET, cfg.priority)
    elif cfg.partition == "region_merge":
        leaves, padded_shape, orig_shape = region_merge_partition(
            img_orig, cfg.S_set, N_TRI_BUDGET)
    else:
        raise ValueError(f"Unknown partition: {cfg.partition}")
    n_tri = len(leaves) * 2
    print(f"  partition done: {len(leaves)} cells, {n_tri} triangles, "
          f"{time.time()-t0:.2f}s")

    # 2. Triangles + per-triangle means (on the padded image)
    triangles = leaves_to_triangles(leaves)
    padded_img, _, _ = pad_to_max(img_orig, max(cfg.S_set))
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

    # 6. Metrics
    metrics = compute_metrics(img_orig, canvas, means[:n_tri], labels[:n_tri],
                              palette, n_tri, elapsed)
    metrics.update({
        "N_Triangles": float(n_tri),
        "N_Cells": float(len(leaves)),
        "Budget_Util": n_tri / MAX_TRIANGLES,
        "Palette_Util": float(sum(1 for c in counts if c > 0)) / cfg.K,
        "K": float(cfg.K),
        "S_set": cfg.S_set,
        "partition": cfg.partition,
        "priority": cfg.priority,
        "palette_method": cfg.palette_method,
        "counts_per_K": counts,
        "group": cfg.group,
        "name": cfg.name,
    })

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

    print(f"  metrics: SSIM={metrics['SSIM']:.4f} PSNR={metrics['PSNR']:.2f} "
          f"ΔE={metrics['Delta_E_2000']:.2f} EdgeF1={metrics['Edge_F1']:.4f} "
          f"EPI={metrics['EPI']:.4f}  ({elapsed:.2f}s)")
    return canvas, metrics


def build_experiment_grid(out_root):
    """Return the 15 Task 3 experiment configs."""
    configs = []

    # A. K sweep
    for K in [4, 8, 16]:
        configs.append(Task3Config(name=f"k{K:02d}", group="A_ksweep", K=K))

    # B. S-set sweep (S_min). Kept as a record: it turned out to be a no-op
    # because the budget binds at size 4, so S_min in {1,2,4} are equivalent.
    for s_set in [[2, 4, 8, 16, 32], [4, 8, 16, 32], [8, 16, 32]]:
        tag = "_".join(str(s) for s in s_set)
        configs.append(Task3Config(name=f"s_{tag}", group="B_sweep", S_set=s_set))

    # B2. S_MAX sweep — the "improved B". Starting coarser leaves more budget
    # headroom for deep splits, so finer cells actually appear.
    for s_set in [[1, 2, 4, 8, 16, 32],
                  [1, 2, 4, 8, 16, 32, 64],
                  [1, 2, 4, 8, 16, 32, 64, 128]]:
        configs.append(Task3Config(name=f"smax_{s_set[-1]}", group="B2_smax",
                                   S_set=s_set))

    # C. Partition algorithm (S_set=[4,8,16,32] keeps both algorithms feasible).
    for p in ["quadtree", "region_merge"]:
        configs.append(Task3Config(name=p, group="C_algorithm", partition=p,
                                   S_set=[4, 8, 16, 32]))

    # D. Palette method
    for pm in ["kmeans_lab", "median_cut"]:
        configs.append(Task3Config(name=pm, group="D_palette", palette_method=pm))

    # E. Priority
    for pr in ["mse", "edgef1"]:
        configs.append(Task3Config(name=f"prio_{pr}", group="E_priority", priority=pr))

    for c in configs:
        c.out_dir = os.path.join(out_root, c.group)
    return configs


SUMMARY_KEYS = ["group", "name", "K", "S_set", "partition", "priority",
                "palette_method", "N_Triangles", "N_Cells", "Budget_Util",
                "Palette_Util", "PSNR", "SSIM", "MS-SSIM", "Delta_E_2000",
                "Edge_F1", "Edge_Precision", "Edge_Recall",
                "EPI", "Quant_Error", "FPS"]


def build_summary(out_root, all_metrics):
    """Aggregate metrics from all runs into a CSV + comparison chart."""
    summary_dir = os.path.join(out_root, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    csv_path = os.path.join(summary_dir, "metrics_table.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_KEYS)
        for m in all_metrics:
            w.writerow([str(m.get(k, "")) if isinstance(m.get(k), list)
                        else m.get(k, "") for k in SUMMARY_KEYS])
    print(f"\nSummary CSV: {csv_path}")

    fig = make_sweep_bar_chart(all_metrics, title="Task 3 experiment grid")
    chart_path = os.path.join(summary_dir, "metrics_chart.png")
    fig.savefig(chart_path, dpi=120)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"Summary chart: {chart_path}")
    return csv_path, chart_path


def main_func():
    parser = argparse.ArgumentParser(description="Task 3: adaptive triangle brick mosaic.")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT)
    parser.add_argument("--out-root", "-o", default=DEFAULT_OUT_ROOT)
    parser.add_argument("--groups", nargs="+",
                        choices=["A", "B", "B2", "C", "D", "E", "all"],
                        default=["all"],
                        help="Which experiment groups to run (default: all).")
    args = parser.parse_args()

    group_keys = {"A": "A_ksweep", "B": "B_sweep", "B2": "B2_smax",
                  "C": "C_algorithm", "D": "D_palette", "E": "E_priority"}

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(args.input)
    print(f"[task3] input = {args.input}  shape = {img.shape[:2]}")

    configs = build_experiment_grid(args.out_root)
    if "all" not in args.groups:
        wanted = {group_keys[g] for g in args.groups}
        configs = [c for c in configs if c.group in wanted]

    all_metrics = []
    for cfg in configs:
        _, m = run_experiment(img, cfg, cfg.out_dir)
        all_metrics.append(m)

    build_summary(args.out_root, all_metrics)
    print(f"\n[task3] All {len(configs)} experiments done.")


if __name__ == "__main__":
    main_func()
