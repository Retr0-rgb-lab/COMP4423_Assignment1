"""
Task 2: Triangle-brick mosaic (equal-size right isosceles triangles, 3 colors).

Pipeline:
    1. Compute smallest grid step S such that 2 * M * N <= MAX_TRIANGLES.
    2. Subdivide image into S x S squares; split each along a diagonal,
       checkerboard-alternating, into two right isosceles triangles.
    3. For each triangle, take the BGR mean of pixels it covers (mask-based).
    4. Quantize to 3 colors via one of {otsu, kmeans, fixed}.
    5. Render: fillPoly + thin gray border (border color is not counted).
    6. Measure a battery of metrics (PSNR, SSIM, MS-SSIM, ΔE2000, Edge F1,
       EPI, quantization error) and visualize via heatmap + bar chart.

Three quantization variants:
    - otsu  : Multi-Otsu (3 classes, 2 thresholds) on BT.601 luminance; per-class
              palette color = mean BGR of members.
    - kmeans: K-Means (K=3) on BGR feature vectors via cv2.kmeans
              (K-Means++ init); palette = cluster centers sorted by luminance.
    - fixed : Dual-threshold at 33%/67% luminance percentiles; per-class palette
              color = mean BGR of members.
"""
import argparse
import os
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, feature, filters
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


DEFAULT_INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics", "sky.jpg")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out_task2.png")
MAX_TRIANGLES = 10000
BORDER_COLOR_BGR = (60, 60, 60)        # dark gray; not counted as one of 3 colors
METHOD_COLORS = {"otsu": "#4C72B0", "kmeans": "#DD8452", "fixed": "#55A467"}


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------

def compute_grid(H, W, max_triangles=MAX_TRIANGLES):
    """Smallest S such that 2 * M * N <= max_triangles, where M = H // S, N = W // S."""
    for S in range(1, min(H, W) + 1):
        M, N = H // S, W // S
        if M >= 1 and N >= 1 and 2 * M * N <= max_triangles:
            return M, N, S
    raise ValueError(f"Image {H}x{W} cannot be tiled under {max_triangles} triangles.")


def iter_triangles(M, N, S):
    """Yield (3,2) int32 vertex arrays. Diagonal orientation alternates like a checkerboard."""
    for i in range(M):
        for j in range(N):
            tl = (j * S, i * S)
            tr = (j * S + S, i * S)
            bl = (j * S, i * S + S)
            br = (j * S + S, i * S + S)
            if (i + j) % 2 == 0:
                yield np.array([tl, tr, br], dtype=np.int32)
                yield np.array([tl, br, bl], dtype=np.int32)
            else:
                yield np.array([tl, tr, bl], dtype=np.int32)
                yield np.array([tr, br, bl], dtype=np.int32)


def triangle_means_bgr(img, triangles):
    """Mean BGR per triangle. Shape (T, 3), float32."""
    H, W = img.shape[:2]
    means = np.zeros((len(triangles), 3), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.uint8)
    for idx, tri in enumerate(triangles):
        mask[:] = 0
        cv2.fillPoly(mask, [tri], 1)
        means[idx] = cv2.mean(img, mask=mask)[:3]
    return means


# ----------------------------------------------------------------------------
# Color utilities
# ----------------------------------------------------------------------------

def _bt601_luma(means_bgr):
    """BT.601 luminance from BGR float means."""
    means = means_bgr.astype(np.float32)
    return 0.114 * means[:, 0] + 0.587 * means[:, 1] + 0.299 * means[:, 2]


def _palette_by_luminance(labels, palette):
    """Reorder (labels, palette) so class 0 is darkest, class 2 is brightest."""
    lum = 0.114 * palette[:, 0].astype(np.float32) \
        + 0.587 * palette[:, 1].astype(np.float32) \
        + 0.299 * palette[:, 2].astype(np.float32)
    order = np.argsort(lum)
    remap = np.zeros(len(palette), dtype=np.uint8)
    remap[order] = np.arange(len(palette), dtype=np.uint8)
    return remap[labels], palette[order]


# ----------------------------------------------------------------------------
# Quantizers
# ----------------------------------------------------------------------------

def multi_otsu_3class(gray_values):
    """3-class Multi-Otsu. Returns (t1, t2) inclusive indices."""
    hist, _ = np.histogram(gray_values, bins=256, range=(0, 256))
    total = hist.sum()
    if total == 0:
        return 85, 170
    prob = hist.astype(np.float64) / total
    bins = np.arange(256, dtype=np.float64)
    cum_p = np.cumsum(prob)
    cum_i = np.cumsum(prob * bins)
    grand_mean = cum_i[-1]

    best_var = -np.inf
    best_t1, best_t2 = 85, 170
    for t1 in range(1, 255):
        w1 = cum_p[t1 - 1]
        if w1 <= 0:
            continue
        m1 = cum_i[t1 - 1] / w1
        for t2 in range(t1 + 1, 255):
            w2 = cum_p[t2 - 1] - cum_p[t1 - 1]
            w3 = 1.0 - cum_p[t2 - 1]
            if w2 <= 0 or w3 <= 0:
                continue
            m2 = (cum_i[t2 - 1] - cum_i[t1 - 1]) / w2
            m3 = (cum_i[-1] - cum_i[t2 - 1]) / w3
            var = (
                w1 * (m1 - grand_mean) ** 2
                + w2 * (m2 - grand_mean) ** 2
                + w3 * (m3 - grand_mean) ** 2
            )
            if var > best_var:
                best_var = var
                best_t1, best_t2 = t1, t2
    return best_t1, best_t2


def quantize_otsu(means_bgr):
    gray = _bt601_luma(means_bgr)
    t1, t2 = multi_otsu_3class(gray)
    labels = np.zeros(len(gray), dtype=np.uint8)
    labels[gray >= t1] = 1
    labels[gray >= t2] = 2
    palette = np.zeros((3, 3), dtype=np.uint8)
    means = means_bgr.astype(np.float32)
    for c in range(3):
        members = means[labels == c]
        if len(members) > 0:
            palette[c] = np.clip(members.mean(axis=0), 0, 255).astype(np.uint8)
    return _palette_by_luminance(labels, palette)


def quantize_kmeans(means_bgr, k=3, seed=0):
    samples = means_bgr.astype(np.float32).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, labels_col, centers = cv2.kmeans(
        samples, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    labels = labels_col.flatten().astype(np.uint8)
    palette = np.clip(centers, 0, 255).astype(np.uint8)
    return _palette_by_luminance(labels, palette)


def quantize_fixed(means_bgr):
    gray = _bt601_luma(means_bgr)
    t1, t2 = np.percentile(gray, [100.0 / 3, 200.0 / 3])
    labels = np.zeros(len(gray), dtype=np.uint8)
    labels[gray >= t1] = 1
    labels[gray >= t2] = 2
    palette = np.zeros((3, 3), dtype=np.uint8)
    means = means_bgr.astype(np.float32)
    for c in range(3):
        members = means[labels == c]
        if len(members) > 0:
            palette[c] = np.clip(members.mean(axis=0), 0, 255).astype(np.uint8)
    return _palette_by_luminance(labels, palette)


def render(H, W, triangles, labels, palette_bgr):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for tri, lab in zip(triangles, labels):
        color = tuple(int(v) for v in palette_bgr[lab])
        cv2.fillPoly(canvas, [tri], color)
        cv2.polylines(canvas, [tri], isClosed=True,
                      color=BORDER_COLOR_BGR, thickness=1, lineType=cv2.LINE_8)
    return canvas


# ----------------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------------

def multi_scale_ssim(img, canvas, levels=4):
    """Mean SSIM at progressively downsampled resolutions (Laplacian pyramid-ish)."""
    g_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g_out = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float64)
    scores = []
    cur_o, cur_c = g_orig, g_out
    for _ in range(levels):
        win = min(7, min(cur_o.shape) - 1)
        if win < 3:
            break
        scores.append(structural_similarity(cur_o, cur_c, data_range=255, win_size=win))
        if len(scores) < levels:
            cur_o = cv2.pyrDown(cur_o)
            cur_c = cv2.pyrDown(cur_c)
    return float(np.mean(scores))


def delta_e_2000_full(img, canvas):
    """Per-pixel CIEDE2000. Returns (mean, full HxW map)."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    img_lab = color.rgb2lab(np.clip(img_rgb, 0, 1))
    canvas_lab = color.rgb2lab(np.clip(canvas_rgb, 0, 1))
    de = color.deltaE_ciede2000(img_lab, canvas_lab)
    return float(de.mean()), de


def edge_f1(img, canvas, sigma=2.0, tolerance=3):
    """F1 score of Canny edges with tolerance-pixel dilation for matching."""
    g_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g_out = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    e_orig = feature.canny(g_orig, sigma=sigma).astype(np.uint8)
    e_out = feature.canny(g_out, sigma=sigma).astype(np.uint8)

    k = 2 * tolerance + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    e_orig_d = cv2.dilate(e_orig, kernel)
    e_out_d = cv2.dilate(e_out, kernel)

    tp_out = int(np.sum(e_out & (e_orig_d > 0)))
    fp_out = int(np.sum(e_out & (e_orig_d == 0)))
    tp_orig = int(np.sum(e_orig & (e_out_d > 0)))
    fn_orig = int(np.sum(e_orig & (e_out_d == 0)))

    precision = tp_out / max(tp_out + fp_out, 1)
    recall = tp_orig / max(tp_orig + fn_orig, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return float(f1), float(precision), float(recall)


def edge_preservation_index(img, canvas):
    """Pearson correlation of Sobel edge responses (EPI)."""
    g_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g_out = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float64)
    s_orig = filters.sobel(g_orig)
    s_out = filters.sobel(g_out)
    so = s_orig - s_orig.mean()
    sc = s_out - s_out.mean()
    num = (so * sc).sum()
    den = np.sqrt((so ** 2).sum() * (sc ** 2).sum())
    return float(num / max(den, 1e-9))


def quantization_error(means_bgr, labels, palette_bgr):
    """Mean ΔE2000 between per-triangle BGR mean and its assigned palette color."""
    means_rgb = means_bgr[:, ::-1].astype(np.float64) / 255.0  # BGR -> RGB
    palette_rgb = palette_bgr[:, ::-1].astype(np.float64) / 255.0
    means_lab = color.rgb2lab(np.clip(means_rgb, 0, 1))
    palette_lab = color.rgb2lab(np.clip(palette_rgb, 0, 1))
    assigned = palette_lab[labels]
    de = color.deltaE_ciede2000(means_lab, assigned)
    return float(de.mean())


def make_residual_heatmap(canvas, de_map):
    """JET colormap over per-pixel ΔE2000 (capped at map's own max)."""
    d_max = max(float(de_map.max()), 1e-9)
    d_norm = np.clip(de_map / d_max, 0, 1)
    return cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)


def compute_metrics(img, canvas, means_bgr, labels, palette_bgr, n_triangles,
                    elapsed_s):
    """Full metric dict for one (img, canvas) pair + quantization context."""
    H, W = img.shape[:2]
    # PSNR
    psnr = float(peak_signal_noise_ratio(img, canvas, data_range=255))
    # SSIM (color)
    win = min(7, min(H, W) - 1 if min(H, W) % 2 == 0 else min(H, W))
    win = max(win, 3)
    ssim = float(structural_similarity(
        img, canvas, data_range=255, channel_axis=2, win_size=win
    ))
    # MS-SSIM
    msssim = multi_scale_ssim(img, canvas)
    # Delta E 2000
    de_mean, de_map = delta_e_2000_full(img, canvas)
    # Edge F1 + P/R
    f1, prec, rec = edge_f1(img, canvas)
    # EPI
    epi_val = edge_preservation_index(img, canvas)
    # Quantization error
    qerr = quantization_error(means_bgr, labels, palette_bgr)
    # Heatmap
    heatmap = make_residual_heatmap(canvas, de_map)
    # Budget utilization
    budget = n_triangles / MAX_TRIANGLES

    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "MS-SSIM": msssim,
        "Delta_E_2000": de_mean,
        "Edge_F1": f1,
        "Edge_Precision": prec,
        "Edge_Recall": rec,
        "EPI": epi_val,
        "Quant_Error": qerr,
        "Budget_Util": budget,
        "N_Triangles": float(n_triangles),
        "FPS": 1.0 / elapsed_s if elapsed_s > 0 else 0.0,
        "Heatmap": heatmap,
    }


# ----------------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------------

def make_compare_grid(img, results):
    """2x2 grid: original + 3 results, with SSIM/PSNR/ΔE labels."""
    H, W = img.shape[:2]
    pad = 20
    label_h = 44
    cell_w, cell_h = W, H + label_h
    grid_h = cell_h * 2 + pad * 3
    grid_w = cell_w * 2 + pad * 3
    grid = np.full((grid_h, grid_w, 3), 32, dtype=np.uint8)

    items = [("original", img, None)] + [
        (name, m["_canvas"], m) for name, m in results[:3]
    ]
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


def make_metrics_bar_chart(metrics_by_method):
    """Two-subplot bar chart: higher-better (left) and lower-better (right)."""
    higher = ["SSIM", "MS-SSIM", "Edge_F1", "EPI"]
    lower = [("PSNR", "PSNR (dB)"), ("Delta_E_2000", "ΔE2000"),
             ("Quant_Error", "Quant Error (ΔE)")]
    methods = list(metrics_by_method.keys())
    n_h = len(higher)
    n_l = len(lower)

    fig, (ax_h, ax_l) = plt.subplots(1, 2, figsize=(13, 5))

    # Higher-is-better
    x = np.arange(n_h)
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

    # Lower-is-better
    x = np.arange(n_l)
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


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

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
            row = k.ljust(20) + "  ".join(f"{metrics_by_method[m][k]:<10.4f}" for m in methods)
        else:
            row = k.ljust(20) + "  ".join(str(metrics_by_method[m][k]).ljust(10) for m in methods)
        print(row)


def run_single(img, method, args):
    H, W, M, N, S, triangles, means = preprocess(img)
    t0 = time.time()
    labels, palette = quantize(method, means)
    canvas = render(H, W, triangles, labels, palette)
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
    out_path = args.output
    cv2.imwrite(out_path, canvas)
    print(f"[task2:{method}] output = {out_path}")
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
        canvas = render(H, W, triangles, labels, palette)
        elapsed = time.time() - t0
        metrics = compute_metrics(img, canvas, means, labels, palette, n_tri, elapsed)
        counts = np.bincount(labels, minlength=3).tolist()
        print(
            f"[task2:{method}] palette={palette.tolist()} counts={counts} "
            f"SSIM={metrics['SSIM']:.4f} PSNR={metrics['PSNR']:.2f} "
            f"ΔE={metrics['Delta_E_2000']:.2f} EdgeF1={metrics['Edge_F1']:.4f} "
            f"({elapsed:.2f}s)"
        )
        per_path = os.path.join(out_dir, f"out_task2_{method}.png")
        cv2.imwrite(per_path, canvas)
        metrics["_canvas"] = canvas  # for compare grid
        metrics_by_method[method] = metrics
        heatmaps.append((method, metrics["Heatmap"]))

    # Save outputs
    grid = make_compare_grid(img, [(m, metrics_by_method[m]) for m in ["otsu", "kmeans", "fixed"]])
    grid_path = os.path.join(out_dir, "out_task2_compare.png")
    cv2.imwrite(grid_path, grid)

    residual = make_residual_panel(
        [h for _, h in heatmaps], [m for m, _ in heatmaps]
    )
    res_path = os.path.join(out_dir, "out_task2_residual.png")
    cv2.imwrite(res_path, residual)

    fig = make_metrics_bar_chart(metrics_by_method)
    chart_path = os.path.join(out_dir, "out_task2_metrics_chart.png")
    fig.savefig(chart_path, dpi=120)
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