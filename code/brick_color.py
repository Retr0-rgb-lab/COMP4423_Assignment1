"""
brick_color — colour engine for the triangle-brick assignment.

Two families:
  * 3-class quantizers (Task 2): `quantize_otsu`, `quantize_kmeans`,
    `quantize_fixed`. Each returns (labels, palette_bgr) sorted dark -> bright.
  * K-colour palette builders (Task 3): `palette_kmeans`,
    `palette_median_cut`, dispatched by `build_palette`. Quantisation is
    `quantize_nearest_bgr`.
"""
import cv2
import numpy as np
from skimage import color as skcolor


# ---------------------------------------------------------------------------
# Shared colour utilities
# ---------------------------------------------------------------------------

def _bt601_luma(means_bgr):
    """BT.601 luminance from BGR float means."""
    means = means_bgr.astype(np.float32)
    return 0.114 * means[:, 0] + 0.587 * means[:, 1] + 0.299 * means[:, 2]


def _palette_by_luminance(labels, palette):
    """Reorder (labels, palette) so class 0 is darkest, last is brightest."""
    lum = 0.114 * palette[:, 0].astype(np.float32) \
        + 0.587 * palette[:, 1].astype(np.float32) \
        + 0.299 * palette[:, 2].astype(np.float32)
    order = np.argsort(lum)
    remap = np.zeros(len(palette), dtype=np.uint8)
    remap[order] = np.arange(len(palette), dtype=np.uint8)
    return remap[labels], palette[order]


# ---------------------------------------------------------------------------
# 3-class quantizers (Task 2)
# ---------------------------------------------------------------------------

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
            var = (w1 * (m1 - grand_mean) ** 2
                   + w2 * (m2 - grand_mean) ** 2
                   + w3 * (m3 - grand_mean) ** 2)
            if var > best_var:
                best_var = var
                best_t1, best_t2 = t1, t2
    return best_t1, best_t2


def _class_mean_palette(means, labels, k):
    """Palette = per-class mean BGR of the member triangles."""
    palette = np.zeros((k, 3), dtype=np.uint8)
    means = means.astype(np.float32)
    for c in range(k):
        members = means[labels == c]
        if len(members) > 0:
            palette[c] = np.clip(members.mean(axis=0), 0, 255).astype(np.uint8)
    return palette


def quantize_otsu(means_bgr):
    """Multi-Otsu (3 classes, 2 thresholds) on BT.601 luminance."""
    gray = _bt601_luma(means_bgr)
    t1, t2 = multi_otsu_3class(gray)
    labels = np.zeros(len(gray), dtype=np.uint8)
    labels[gray >= t1] = 1
    labels[gray >= t2] = 2
    return _palette_by_luminance(labels, _class_mean_palette(means_bgr, labels, 3))


def quantize_kmeans(means_bgr, k=3, seed=0):
    """K-Means (K=3) on BGR via cv2.kmeans + K-Means++ init."""
    samples = means_bgr.astype(np.float32).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, labels_col, centers = cv2.kmeans(
        samples, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    labels = labels_col.flatten().astype(np.uint8)
    palette = np.clip(centers, 0, 255).astype(np.uint8)
    return _palette_by_luminance(labels, palette)


def quantize_fixed(means_bgr):
    """Dual-threshold at 33.3% / 66.7% percentiles of BT.601 luminance."""
    gray = _bt601_luma(means_bgr)
    t1, t2 = np.percentile(gray, [100.0 / 3, 200.0 / 3])
    labels = np.zeros(len(gray), dtype=np.uint8)
    labels[gray >= t1] = 1
    labels[gray >= t2] = 2
    return _palette_by_luminance(labels, _class_mean_palette(means_bgr, labels, 3))


# ---------------------------------------------------------------------------
# K-colour palette builders (Task 3)
# ---------------------------------------------------------------------------

def palette_kmeans(means_bgr, k, space="lab", seed=0):
    """K-Means palette in Lab or RGB. Returns (k, 3) uint8 BGR."""
    if space == "lab":
        means_rgb = means_bgr[:, ::-1].astype(np.float64) / 255.0
        means_in = skcolor.rgb2lab(np.clip(means_rgb, 0, 1)).astype(np.float32)
    else:
        means_in = means_bgr[:, ::-1].astype(np.float32)  # BGR -> RGB features
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, _, centers = cv2.kmeans(means_in, k, None, criteria, 10,
                               cv2.KMEANS_PP_CENTERS)
    if space == "lab":
        centers_rgb = skcolor.lab2rgb(centers.reshape(1, -1, 3)).reshape(-1, 3)
        centers_bgr = np.clip(centers_rgb * 255, 0, 255).astype(np.uint8)[:, ::-1]
    else:
        centers_bgr = centers[:, ::-1].astype(np.uint8)
    return np.clip(centers_bgr, 0, 255).astype(np.uint8)


def palette_median_cut(means_bgr, k):
    """Recursive median-cut in RGB. Returns (k, 3) uint8 BGR."""
    arr = means_bgr[:, ::-1].astype(np.float32).reshape(-1, 3)  # RGB
    boxes = [arr]

    while len(boxes) < k and len(boxes) < arr.shape[0]:
        ranges = np.array([b.max(axis=0) - b.min(axis=0) for b in boxes])
        idx = int(np.argmax(ranges.max(axis=1)))
        box = boxes[idx]
        if len(box) < 2:
            break
        ch = int(np.argmax(ranges[idx]))
        sorted_box = box[box[:, ch].argsort()]
        mid = len(sorted_box) // 2
        boxes[idx] = sorted_box[:mid]
        boxes.insert(idx + 1, sorted_box[mid:])

    palette_rgb = np.array([b.mean(axis=0) for b in boxes])
    while len(palette_rgb) < k:
        palette_rgb = np.vstack([palette_rgb, palette_rgb[-1:]])
    palette_bgr = np.clip(palette_rgb[:, ::-1], 0, 255).astype(np.uint8)
    return palette_bgr[:k]


def build_palette(means_bgr, k, method, seed=0):
    if method == "kmeans_lab":
        return palette_kmeans(means_bgr, k, "lab", seed)
    if method == "kmeans_rgb":
        return palette_kmeans(means_bgr, k, "rgb", seed)
    if method == "median_cut":
        return palette_median_cut(means_bgr, k)
    raise ValueError(f"Unknown palette method: {method}")


def quantize_nearest_bgr(means_bgr, palette_bgr):
    """Nearest palette colour in BGR Euclidean space."""
    means_f = means_bgr.astype(np.float32)
    pal_f = palette_bgr.astype(np.float32)
    diff = means_f[:, None, :] - pal_f[None, :, :]
    dist = (diff * diff).sum(axis=2)
    return np.argmin(dist, axis=1).astype(np.uint8)
