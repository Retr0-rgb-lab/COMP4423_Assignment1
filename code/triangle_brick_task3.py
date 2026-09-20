"""
Task 3: Adaptive multi-size + multi-color triangle-brick mosaic.

Thin driver over the shared brick_* engine modules:
    brick_geom     quadtree_partition / region_merge_partition / leaves_to_triangles
    brick_color    palette_kmeans / palette_median_cut / build_palette
    brick_render   triangle_means_bgr / render_triangles
    brick_metrics  compute_metrics (metric dict; keys listed in brick_metrics)
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

Metric-count changes belong in `brick_metrics`, which owns the authoritative
key list; do not restate a count in this header (prose drifted from code here
before).
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
from brick_geom import MAX_TRIANGLES, pad_to_max, leaves_to_triangles  # noqa: E402
from brick_quadtree import quadtree_partition  # noqa: E402
from brick_region_merge import region_merge_partition  # noqa: E402
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
    """One row of the Task 3 experiment grid.

    Function: a plain settings record so `build_experiment_grid` can declare the
    whole sweep as data instead of branching on the algorithm. `run_experiment`
    reads every field; nothing here is validated, so an invalid combination
    (e.g. `S_set` without `max(S_set)` a power of two, or `partition` not in
    {quadtree, region_merge}) fails later, inside the engine, not here.

    Shape/type contract (no ndarrays -- this is a pure config object):
      name, group     : str  -- `name` is the per-run file stem, `group` the
                        directory and the figure legend prefix.
      S_set           : list[int] -- allowed cell side lengths, powers of two.
                        MUST be a halving family: the quadtree splits by half, so
                        a gap (e.g. [1, 4]) would leave sizes unreachable.
      partition       : "quadtree" | "region_merge".
      priority        : "mse" | "edgef1" -- only read by the quadtree path;
                        region_merge ignores it, so a priority sweep MUST use
                        the quadtree partition or the E_group is a no-op.
      K               : int  -- palette size, >= 1.
      palette_method  : "kmeans_lab" | "kmeans_rgb" | "median_cut".
      seed            : int  -- forwarded to the palette builder. Note it does
                        NOT make K-Means reproducible on its own: cv2.kmeans uses
                        OpenCV's global RNG, so a caller must also call
                        `cv2.setRNGSeed` (task3_best does; this sweep does not,
                        which is why the sweep's K-Means runs are not bit-exact).
      out_dir         : str  -- filled in by `build_experiment_grid`.
    """
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
        """One-line human summary of the config, used in the run log header.

        Shape: no arrays. Semantics: a `" | "`-joined string of the six
        settings that vary across the sweep, in the order
        K / S_set / partition / priority / palette -- so two runs can be told
        apart by reading one log line.
        """
        return " | ".join([
            f"K={self.K}", f"S={self.S_set}", f"partition={self.partition}",
            f"priority={self.priority}", f"palette={self.palette_method}",
        ])


def run_experiment(img_orig, cfg, out_dir):
    """Run the full Task 3 pipeline for ONE configuration and write its outputs.

    Function
    --------
    Numbered steps, matching the inline `# n.` comments in the body:
      1. partition  -> adaptive leaf cells (quadtree or region_merge)
      2. triangles  -> 2 per leaf, then per-triangle mean BGR
      3. palette    -> K colours from those means
      4. quantize   -> assign every triangle to a palette index
      5. render     -> paint the canvas and crop back to the original size
      6. metrics    -> the full suite, plus Task 3 specific extras
      7. outputs    -> <name>.png, <name>_size_hist.png, <name>_palette.png,
                       <name>_metrics.json

    Timing caveat (this is the Task 3 driver, so its FPS differs from Task 2's):
    `t0` starts at the top, so the reported FPS covers partition + mean
    extraction + palette + quantize + render. It EXCLUDES the metric computation
    below, even though that step is the slowest part of the function. Compare
    FPS only against other Task 3 runs -- see the FPS caveat in the brick_metrics
    module docstring.

    Shapes
    ------
    Input:
      img_orig : (H, W, 3) uint8 BGR -- source image.
      cfg      : Task3Config -- this run's settings.
      out_dir  : str -- directory for the four output files (created if absent).
    Intermediate:
      leaves    : list of (x, y, size) in padded coords.
      triangles : list of (3, 2) int32, `tri[k, 0]=x`, `tri[k, 1]=y`.
      means     : (2*len(leaves), 3) float32, `means[i, 0]=B`. Sampled on the
                  PADDED image (`padded_img`), not the original, because triangle
                  coordinates are in padded space -- sampling the original would
                  silently shift every mean near the right/bottom edges.
      palette   : (cfg.K, 3) uint8 BGR, `palette[k, 0]=B`.
      labels    : (T,) uint8 -- `labels[i]` = palette index of triangle i.
      canvas    : (H, W, 3) uint8 BGR -- rendered, then cropped to orig_shape.
    Output:
      (canvas, metrics). `metrics` is the `compute_metrics` dict EXTENDED with
      Task 3 fields: N_Cells, Palette_Util, K, S_set, partition, priority,
      palette_method, counts_per_K (`counts_per_K[k]` = triangles painted with
      index k), group, name. "Heatmap" stays in the returned dict but is dropped
      from the JSON dump.

    Known wart: the metric call passes `means[:n_tri]` / `labels[:n_tri]` to trim
    to the triangle count. For the quadtree path `len(triangles) == 2*n_cells`
    already, so the slice is a no-op; it exists to protect the region_merge path
    from a stale padding cell. Keep it if either partition's contract changes.
    """
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
    """Return the 15 Task 3 experiment configs, with `out_dir` filled in.

    Function
    --------
    Declares the whole sweep as data. Each group varies ONE factor and inherits
    the `Task3Config` defaults for the rest, so a group's rows are comparable to
    each other; the groups are NOT comparable to each other in general (they
    differ in baseline settings). Commented notes in the body record which
    groups turned out to be informative and which did not.

    Shape: `out_root` is a directory path; the returned list is length 15
    (3+3+3+2+2+2). Semantics of the ordering: groups are emitted in the fixed
    order A, B, B2, C, D, E, and within a group in ascending swept value, which
    is the row order the summary CSV inherits. Each `cfg.out_dir` is
    `<out_root>/<group>`, and the returned configs deliberately do NOT validate
    their S_set / partition / priority combinations.

    Note on B_sweep: it is kept even though it was later shown to be a no-op
    (the budget binds at size 4, so S_min in {1, 2, 4} are equivalent). It stays
    as part of the experimental record -- see docs/progress/Task3.md -- rather
    than being deleted, so the reasoning trail is reproducible.
    """
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
    """Aggregate every run's metrics into one CSV plus one comparison chart.

    Function
    --------
    This is the cross-group summary figure/table for the report: one CSV row per
    run (all 15 groups together, in the order the runs were executed) and one
    bar chart covering all of them.

    Shape
    ------
    Input: `out_root` is the run root (the summary goes in `<out_root>/summary/`);
    `all_metrics` is a list of the dicts returned by `run_experiment`, one per
    run, N rows.
    Output: `(csv_path, chart_path)`, both strings. Files written:
    `summary/metrics_table.csv` and `summary/metrics_chart.png`.

    Semantics / gotchas:
      * CSV columns are exactly `SUMMARY_KEYS` (module-level constant), NOT all
        keys of the metric dicts. List-valued fields (e.g. `S_set`) are
        `str()`-ified so the row stays flat; a key missing from a dict becomes
        "" rather than raising, so a short row means a missing field, not a
        formatting bug.
      * `SUMMARY_KEYS` must contain "group" and "name": `make_sweep_bar_chart`
        reads both for its legend and will KeyError without them.
      * The chart is produced by matplotlib here (unlike the OpenCV montages
        elsewhere), so this is the one summary path that needs a working
        matplotlib backend; `brick_viz` forces "Agg" at import, which is what
        makes it safe to run headless.
    """
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
    """CLI entry: parse args, load the image, run the selected groups, summarise.

    Function
    --------
    `--groups` selects which of A/B/B2/C/D/E to execute (default `all`). A partial
    selection filters the grid but still calls `build_summary`, so the CSV/chart
    written by a partial run describes ONLY the runs that were executed -- it
    overwrites the full-run summary at the same path. Re-run with `all` before
    quoting the summary, or the report will describe a subset without saying so.

    Shapes: no ndarray I/O at this level; the loaded image is (H, W, 3) uint8 BGR
    and is passed straight to `run_experiment`. Outputs are the files written by
    `run_experiment` per config plus the two summary files.
    """
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
