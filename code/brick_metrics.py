"""
brick_metrics — evaluation metrics for triangle-brick mosaics.

Nine metrics, each catching a different failure mode:
    PSNR          pixel-level fidelity
    SSIM          local structure (luminance / contrast / correlation)
    MS-SSIM       4-level mean SSIM (multi-scale structure)
    Delta_E_2000  per-pixel perceptual colour difference (CIEDE2000)
    Edge_F1       Canny edge F1 with tolerance matching (+ precision/recall)
    EPI           Sobel-response Pearson correlation (edge alignment)
    Quant_Error   per-triangle mean vs assigned palette colour (in ΔE2000)
    Budget_Util   N triangles / MAX_TRIANGLES
    FPS           1 / elapsed seconds
"""
import cv2
import numpy as np
from skimage import color, feature, filters
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from brick_geom import MAX_TRIANGLES


def multi_scale_ssim(img, canvas, levels=4):
    """Mean SSIM at progressively downsampled resolutions."""
    g_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g_out = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float64)
    scores = []
    cur_o, cur_c = g_orig, g_out
    for _ in range(levels):
        win = min(7, min(cur_o.shape) - 1)
        if win < 3:
            break
        scores.append(structural_similarity(cur_o, cur_c, data_range=255,
                                            win_size=win))
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
    """F1 / precision / recall of Canny edges with tolerance dilation."""
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
    """Mean ΔE2000 between per-triangle mean and its assigned palette colour."""
    means_rgb = means_bgr[:, ::-1].astype(np.float64) / 255.0
    palette_rgb = palette_bgr[:, ::-1].astype(np.float64) / 255.0
    means_lab = color.rgb2lab(np.clip(means_rgb, 0, 1))
    palette_lab = color.rgb2lab(np.clip(palette_rgb, 0, 1))
    assigned = palette_lab[labels]
    de = color.deltaE_ciede2000(means_lab, assigned)
    return float(de.mean())


def make_residual_heatmap(canvas, de_map):
    """JET colormap over per-pixel ΔE2000 (capped at the map's own max)."""
    d_max = max(float(de_map.max()), 1e-9)
    d_norm = np.clip(de_map / d_max, 0, 1)
    return cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)


def compute_metrics(img, canvas, means_bgr, labels, palette_bgr, n_triangles,
                    elapsed_s):
    """Full metric dict for one (img, canvas) pair + quantization context."""
    H, W = img.shape[:2]
    psnr = float(peak_signal_noise_ratio(img, canvas, data_range=255))
    win = min(7, min(H, W) - 1 if min(H, W) % 2 == 0 else min(H, W))
    win = max(win, 3)
    ssim = float(structural_similarity(
        img, canvas, data_range=255, channel_axis=2, win_size=win
    ))
    msssim = multi_scale_ssim(img, canvas)
    de_mean, de_map = delta_e_2000_full(img, canvas)
    f1, prec, rec = edge_f1(img, canvas)
    epi_val = edge_preservation_index(img, canvas)
    qerr = quantization_error(means_bgr, labels, palette_bgr)
    heatmap = make_residual_heatmap(canvas, de_map)
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
