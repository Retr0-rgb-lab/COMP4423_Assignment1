"""
brick_viz — visualisation helpers for the triangle-brick assignment.

OpenCV-based montages (compare grid, residual panel) and matplotlib charts
(metric bar charts, per-size histograms, palette swatches).
"""
import cv2
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {"otsu": "#4C72B0", "kmeans": "#DD8452", "fixed": "#55A467"}


# ---------------------------------------------------------------------------
# OpenCV montages
# ---------------------------------------------------------------------------

def make_compare_grid(img, results):
    """2x2 grid: original + up to 3 results, with key-metric labels."""
    H, W = img.shape[:2]
    pad = 20
    label_h = 44
    cell_w, cell_h = W, H + label_h
    grid = np.full((cell_h * 2 + pad * 3, cell_w * 2 + pad * 3, 3), 32,
                   dtype=np.uint8)

    items = [("original", img, None)] + [(name, m["_canvas"], m)
                                         for name, m in results[:3]]
    for i, (name, canvas, m) in enumerate(items):
        row, col = divmod(i, 2)
        x0 = pad + col * (cell_w + pad)
        y0 = pad + row * (cell_h + pad)
        grid[y0:y0 + H, x0:x0 + W] = canvas
        if m is None:
            text = name
        else:
            text = (f"{name}  SSIM={m['SSIM']:.3f}  PSNR={m['PSNR']:.1f}  "
                    f"ΔE={m['Delta_E_2000']:.1f}  EdgeF1={m['Edge_F1']:.3f}")
        cv2.putText(grid, text, (x0 + 12, y0 + H + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return grid


def make_residual_panel(heatmaps, names):
    """Single column of stacked residual heatmaps with labels."""
    H, W = heatmaps[0].shape[:2]
    pad = 16
    label_h = 36
    cell_h = H + label_h
    panel = np.full((cell_h * len(heatmaps) + pad * (len(heatmaps) + 1),
                     W + pad * 2, 3), 24, dtype=np.uint8)
    for i, (heat, name) in enumerate(zip(heatmaps, names)):
        y0 = pad + i * (cell_h + pad)
        panel[y0:y0 + H, pad:pad + W] = heat
        cv2.putText(panel, f"{name}  per-pixel ΔE2000",
                    (pad + 10, y0 + H + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return panel


# ---------------------------------------------------------------------------
# matplotlib charts
# ---------------------------------------------------------------------------

def make_metrics_bar_chart(metrics_by_method):
    """Two-subplot bar chart for the Task 2 three-method comparison."""
    higher = ["SSIM", "MS-SSIM", "Edge_F1", "EPI"]
    lower = [("PSNR", "PSNR (dB)"), ("Delta_E_2000", "ΔE2000"),
             ("Quant_Error", "Quant Error (ΔE)")]
    methods = list(metrics_by_method.keys())

    fig, (ax_h, ax_l) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(higher))
    width = 0.8 / len(methods)
    for i, method in enumerate(methods):
        vals = [metrics_by_method[method][m] for m in higher]
        bars = ax_h.bar(x + i * width - 0.4 + width / 2, vals, width,
                        label=method, color=METHOD_COLORS[method])
        for bar, v in zip(bars, vals):
            ax_h.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                      ha="center", va="bottom", fontsize=8)
    ax_h.set_xticks(x)
    ax_h.set_xticklabels(higher, rotation=0)
    ax_h.set_ylim(0, 1.05)
    ax_h.set_ylabel("Score (higher is better)")
    ax_h.set_title("[0,1]-normalized metrics")
    ax_h.legend(loc="lower right")
    ax_h.grid(axis="y", linestyle="--", alpha=0.3)

    x = np.arange(len(lower))
    for i, method in enumerate(methods):
        vals = [metrics_by_method[method][key] for key, _ in lower]
        bars = ax_l.bar(x + i * width - 0.4 + width / 2, vals, width,
                        label=method, color=METHOD_COLORS[method])
        for bar, v in zip(bars, vals):
            ax_l.text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.2f}",
                      ha="center", va="bottom", fontsize=8)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels([label for _, label in lower], rotation=15, ha="right")
    ax_l.set_ylabel("Value (lower is better)")
    ax_l.set_title("Absolute-scale metrics")
    ax_l.legend(loc="upper right")
    ax_l.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Task 2 quantization-method comparison (sky.jpg)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def make_sweep_bar_chart(metrics_list, title="Task 3 experiment grid"):
    """Two-subplot bar chart across N experiment runs (Task 3 summary)."""
    higher_keys = ["SSIM", "MS-SSIM", "Edge_F1", "EPI"]
    lower_keys = [("PSNR", "PSNR (dB)"), ("Delta_E_2000", "ΔE2000"),
                  ("Quant_Error", "Quant Error (ΔE)")]
    labels = [f"{m['group'].split('_')[0]}:{m['name']}" for m in metrics_list]
    n = len(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, max(n, 1)))

    fig, (ax_h, ax_l) = plt.subplots(1, 2, figsize=(max(12, n * 0.6), 5))
    x = np.arange(len(higher_keys))
    width = 0.8 / max(n, 1)
    for i, m in enumerate(metrics_list):
        vals = [m[k] for k in higher_keys]
        ax_h.bar(x + i * width - 0.4 + width / 2, vals, width,
                 color=colors[i], label=labels[i])
    ax_h.set_xticks(x)
    ax_h.set_xticklabels(higher_keys)
    ax_h.set_ylim(0, 1.05)
    ax_h.set_ylabel("Higher is better")
    ax_h.set_title("[0,1]-normalized metrics")
    ax_h.legend(fontsize=7, ncol=2, loc="lower right")
    ax_h.grid(axis="y", linestyle="--", alpha=0.3)

    x = np.arange(len(lower_keys))
    for i, m in enumerate(metrics_list):
        vals = [m[k] for k, _ in lower_keys]
        ax_l.bar(x + i * width - 0.4 + width / 2, vals, width,
                 color=colors[i], label=labels[i])
    ax_l.set_xticks(x)
    ax_l.set_xticklabels([lab for _, lab in lower_keys], rotation=15, ha="right")
    ax_l.set_ylabel("Lower is better")
    ax_l.set_title("Absolute-scale metrics")
    ax_l.legend(fontsize=7, ncol=2, loc="upper right")
    ax_l.grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(f"{title} — {n} runs", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_size_histogram(leaves, save_path):
    """Bar chart of triangle count per cell size."""
    counts = {}
    for (_, _, s) in leaves:
        counts[s] = counts.get(s, 0) + 2  # 2 triangles per cell
    items = sorted(counts.items())
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar([str(s) for s, _ in items], [c for _, c in items],
                  color="#4C72B0")
    top = max([c for _, c in items] + [1])
    for bar, (_, c) in zip(bars, items):
        ax.text(bar.get_x() + bar.get_width() / 2, c + top * 0.01, str(c),
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Triangle size S (right-angle leg, pixels)")
    ax.set_ylabel("Triangle count")
    ax.set_title(f"Per-size distribution  (total = {sum(c for _, c in items)})")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_palette(palette_bgr, save_path, K):
    """Swatch row showing palette colours with their BGR values."""
    fig, ax = plt.subplots(figsize=(max(6, K * 0.6), 1.2))
    for i, color_bgr in enumerate(palette_bgr):
        rgb = (int(color_bgr[2]) / 255, int(color_bgr[1]) / 255,
               int(color_bgr[0]) / 255)
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=rgb))
        ax.text(i + 0.5, 0.5,
                f"{int(color_bgr[0])}\n{int(color_bgr[1])}\n{int(color_bgr[2])}",
                ha="center", va="center", fontsize=6, fontfamily="monospace",
                color="white" if int(color_bgr[0]) + int(color_bgr[1])
                + int(color_bgr[2]) < 200 else "black")
    ax.set_xlim(0, K)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Palette (K={K})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
