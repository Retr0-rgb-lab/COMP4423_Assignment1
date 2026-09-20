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
    """2x2 montage: the original plus up to three renders, each captioned.

    Function: builds one image a report reader can compare at a glance, by
    pasting each canvas into a gray grid cell and printing the key metrics
    underneath it. Captions are drawn with `cv2.putText` on the grid buffer (not
    matplotlib) so the output is a plain PNG with no dependency on a font file
    being present.

    Shape: `img` (H, W, 3) uint8; output is
    `(2*(H+label_h) + 3*pad, 2*W + 3*pad, 3)` uint8, with `label_h=44`,
    `pad=20`. Semantics: `results` is a list of `(name, metrics_dict)` pairs
    where `metrics_dict["_canvas"]` (H, W, 3) is the render to paste and the
    metric keys are only read for the caption text.

    Preconditions / gotchas:
      * Only the FIRST three entries are drawn (`results[:3]`); a fourth is
        silently dropped, so this is a 4-up helper, not an N-up one.
      * Every `_canvas` must be exactly (H, W, 3). The slice assignment will
        raise on a size mismatch, which is the intended behaviour -- a silently
        rescaled comparison would be misleading.
      * `_canvas` is stashed inside the metrics dict by the driver
        (`triangle_brick.run_compare`) rather than passed separately.
    """
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
    """Single column of stacked residual heatmaps, one per run, captioned.

    Function: puts the per-run error maps in one image so the report can show
    "where each method is wrong" side by side vertically. Assumes all maps come
    from the same source image, which is what makes them worth stacking.

    Shape: `heatmaps` is a list of N arrays, each (H, W, 3) uint8 BGR (from
    `brick_metrics.make_residual_heatmap`). Output is
    `(N*(H+label_h) + (N+1)*pad, W + 2*pad, 3)` uint8 BGR, with `label_h=36`,
    `pad=16`.

    Preconditions / gotchas:
      * H and W are taken from `heatmaps[0]` alone -- every other map must match
        exactly or the slice assignment raises. Mixed-size input is a caller bug,
        not something this function rescales.
      * `names` must be the same length as `heatmaps`; `zip` silently truncates
        to the shorter of the two, so a wrong-length name list loses captions
        instead of erroring.
      * The colour scale of each tile is INDEPENDENT -- each heatmap was
        normalised to its own maximum upstream, so a red patch in tile 1 and one
        in tile 2 do not represent equally large errors. Read this figure as
        "where", not "how much"; use the numeric Delta_E_2000 for magnitude.
    """
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
    """Two-panel bar chart for the Task 2 three-method comparison.

    Function: the metrics have incompatible scales and directions, so they are
    split into two panels rather than one -- left: the bounded [0, 1] metrics
    where HIGHER is better (comparable on a shared 0..1.05 axis); right: the
    unbounded metrics where LOWER is better (each bar labelled with its value).
    Putting them on one axis would make PSNR's ~17 dB dwarf SSIM's 0.4 and hide
    the comparison this chart exists to make.

    Shape: returns a matplotlib `Figure` with axes (1, 2), figsize (13, 5). The
    function does NOT save or close -- the caller owns that, so the same figure
    can be written at different DPIs.

    Semantics of `metrics_by_method`: dict `{method_name -> metrics_dict}`,
    where each inner dict is a `brick_metrics.compute_metrics` result. Bar
    height for panel-left = `metrics[metric_key]` for the keys in `higher`;
    panel-right likewise for `lower`.

    Preconditions / gotchas:
      * Panel-left assumes all four `higher` keys are already in [0, 1]; EPI can
        be NEGATIVE in principle (it is a correlation), which would render below
        the axis floor. On the images tested so far it is positive.
      * `METHOD_COLORS` must contain every method name in `metrics_by_method` or
        this raises KeyError -- intentional, so a new method cannot be plotted
        in a colour that clashes with an existing series.
      * Both panels read the metric keys by name; a missing key is a KeyError
        rather than a blank bar.
    """
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
    """Two-panel bar chart across N Task 3 experiment runs.

    Function: same two-panel split as `make_metrics_bar_chart` (bounded
    higher-is-better on the left, unbounded lower-is-better on the right), but
    across RUNS instead of across methods -- used for the Task 3 summary so all
    15 configurations can be eyeballed on one figure. Bar width shrinks as
    `0.8/n` so every run gets a slot, and the figure widens with `n`.

    Shape: returns a matplotlib `Figure` with axes (1, 2),
    figsize `(max(12, n*0.6), 5)`. Not saved or closed here -- the caller does.

    Semantics of `metrics_list`: a list of `brick_metrics.compute_metrics`
    result dicts, one per run. Each MUST carry the string keys `"group"` and
    `"name"` (added by the Task 3 driver) because the legend label is derived as
    `"A:k04"` from `group.split("_")[0]` plus `name`.

    Preconditions / gotchas:
      * A run dict missing `"group"` or `"name"` raises KeyError; these are the
        only non-metric keys this function depends on.
      * The legend is drawn inside the axes with `ncol=2`, so past roughly 20
        runs it starts overlapping the bars -- split the sweep into two figures
        beyond that.
      * With more than 20 runs `plt.cm.tab20` repeats colours, making two runs
        look identical in the legend.
    """
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
    """Bar chart of triangle count per cell size; saves a PNG and closes the fig.

    Function: this is the Task 3 deliverable's "count for each brick size", so
    the title carries the total and every bar is labelled with its own count --
    the reader should not have to read values off the axis. Because the bars are
    counts of CELLS multiplied by 2, the chart is already in the unit the
    assignment grades (triangles), not cells.

    Shape: `leaves` is an iterable of (x, y, size) tuples -- the `(x, y)` part is
    ignored. Semantics: `counts[s]` accumulates `+2` per cell of side `s`
    because each square cell is cut into exactly two triangles; the x-axis is
    `size` (the right-angle leg in pixels) and the y-axis is the triangle count
    at that size. Writes `save_path` as PNG and returns None (side-effecting, so
    it must not be used in a notebook without expecting a file on disk).

    Edge case: an empty `leaves` leaves `items` empty and `top` falls back to 1,
    producing an empty chart rather than raising -- a blank histogram therefore
    means "no cells", not "plot failed".
    """
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
    """Swatch strip of the palette, each entry labelled with its BGR numbers.

    Function: shows WHICH colours a run actually used and in what order, which
    is the evidence behind "more than three brick colours" in Task 3 and the only
    way to see whether a K-Means palette landed where you expected. Saves a PNG
    and closes the figure (side-effecting).

    Shape: `palette_bgr` is (K, 3) uint8 BGR; the figure is 1 row of K unit-width
    rectangles with xlim (0, K) and ylim (0, 1) -- i.e. one rectangle per palette
    index, drawn left to right in palette order. `K` is used for the axis limits
    and the title, and must match `len(palette_bgr)` or the strip will not fill
    the axis.

    Semantics: `palette_bgr[i, 0]=B`, `[i, 1]=G`, `[i, 2]=R`; the label under
    each swatch prints B, G, R top to bottom (the reverse of the tuple order, as
    a vertically stacked text box). Palette order is significant per method --
    `brick_color` sorts the Task 2 palettes dark -> bright but leaves the Task 3
    ones unsorted, so this figure is the reference for what index the labels in
    other figures refer to.

    Gotcha: label colour is chosen by a crude heuristic -- white text when
    B+G+R < 200, else black. A mid-brightness swatch (sum just above 200) can
    therefore get black text on a dark colour and be hard to read. Cosmetic only.
    """
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
