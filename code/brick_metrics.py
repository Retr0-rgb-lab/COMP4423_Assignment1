"""
brick_metrics — evaluation metrics for triangle-brick mosaics.

`compute_metrics` is the ONLY authoritative statement of what is returned:
a dict of **12 numeric fields + 1 image**, 13 keys total.

Quality metrics (compare `img` against `canvas`):

    PSNR            pixel-level fidelity (dB, higher better)
    SSIM            local structure: luminance / contrast / correlation
    MS-SSIM         4-level mean SSIM (multi-scale structure)
    Delta_E_2000    per-pixel perceptual colour difference (CIEDE2000)
    Edge_F1         Canny-edge F1 with tolerance matching; higher better
    Edge_Precision  of the OUTPUT's edge pixels, the share near a source edge
    Edge_Recall     of the SOURCE's edge pixels, the share covered by output
    EPI             Sobel-response Pearson correlation (edge alignment)
                    NOTE: EPI is a locally-defined name, not a standard
                    metric -- define it in any write-up that cites it.

Quantisation metric (compares triangle means against their assigned colour):

    Quant_Error     mean Delta_E_2000 of (triangle mean vs palette colour)

Bookkeeping (no quality meaning, but fed into the CSV / charts):

    Budget_Util     n_triangles / MAX_TRIANGLES
    N_Triangles     triangle count for this run
    FPS             see the CAVEAT below

Image artifact (not a metric):

    Heatmap         (H, W, 3) uint8 JET-encoded per-pixel Delta_E_2000 map

CAVEAT -- FPS is NOT an end-to-end frame rate. `compute_metrics` only does
`1.0 / elapsed_s`; what `elapsed_s` covers is decided by each caller and
differs between them (in the Task 2 driver it is quantize+render only, in the
Task 3 drivers it also includes partitioning and mean extraction, and in no
driver does it include the metric computation itself). Compare FPS only
between runs that used the same driver, and state the interval when quoting
it in the report.
"""
import cv2
import numpy as np
from skimage import color, feature, filters
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from brick_geom import MAX_TRIANGLES


def _ssim_window(shape):
    """Largest odd SSIM window (<= 7) that still fits inside `shape`.

    Function: skimage rejects an even window and one larger than the image, and
    both callers below need exactly that clamp -- so the rule lives here once.
    They differ only in the no-valid-window case: `compute_metrics` raises,
    `multi_scale_ssim` stops descending.
    Shape: `shape` is any shape tuple; only the last two entries are read, so
    `img.shape` on an (H, W, 3) array is accepted directly. Returns an int, or
    None when the short side is < 3 (no odd window >= 3 can fit).
    Semantics: `min(7, m)` for odd `m`, `min(7, m - 1)` for even `m`, where
    `m = min(shape[-2:])`. Any image at least 9 px on its short side gives 7 --
    bit-identical to what both callers computed before this helper existed, so no
    recorded Task 2/3 number changes.
    """
    m = min(shape[-2], shape[-1])
    if m < 3:
        return None
    return min(7, m if m % 2 else m - 1)


def multi_scale_ssim(img, canvas, levels=4):
    """Mean SSIM at progressively downsampled resolutions (MS-SSIM).

    Function
    --------
    Convert both images to grayscale, then loop up to `levels` times: score the
    current pair, halve both with `cv2.pyrDown`, score again. Averaging across
    scales rewards structure that survives downsampling, so this is less fooled
    by fine misalignment than single-scale SSIM.

    Shapes
    ------
    Input:
      img, canvas : (H, W, 3) uint8 BGR -- original and rendered image.
      levels      : int -- maximum number of scales to average.
    Intermediate:
      g_orig, g_out : (H, W) float64 grayscale, then each further iteration
                      works on (H // 2**k, W // 2**k).
      win           : int -- SSIM window for the current level, from the shared
                      `_ssim_window` helper (odd, <= 7, fits the level).
    Output:
      float -- mean of the per-level SSIM scores.

    Edge case / known wart
    ----------------------
    If a level is too small for any valid window (`_ssim_window` returns None),
    the loop `break`s EARLY. The mean is then taken over fewer than `levels`
    scales, SILENTLY -- the return value does not reveal how many scales
    contributed. Do not compare this number across images of very different
    sizes without checking that both got the full 4 levels.

    Window convention: shared with `compute_metrics` via `_ssim_window`, so the
    file has one small-image rule rather than two. Only behaviour change from
    unifying them: a level exactly 3 px across now scores with win=3 instead of
    being skipped, because the old `min(7, m - 1)` rule discarded 3 as even.
    Unreachable from the Task 2-3 images (the pyramid starts at m >= 37).
    """
    g_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g_out = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float64)
    scores = []
    cur_o, cur_c = g_orig, g_out
    for _ in range(levels):
        win = _ssim_window(cur_o.shape)
        if win is None:
            break
        scores.append(structural_similarity(cur_o, cur_c, data_range=255,
                                            win_size=win))
        if len(scores) < levels:
            cur_o = cv2.pyrDown(cur_o)
            cur_c = cv2.pyrDown(cur_c)
    return float(np.mean(scores))


def delta_e_2000_full(img, canvas):
    """Per-pixel CIEDE2000 colour difference, whole image.

    Function
    --------
    CIEDE2000 is the perceptual colour distance the metric suite treats as
    ground truth for colour. It has to be computed in Lab, so both images go
    through sRGB -> Lab first. Returns BOTH the scalar mean and the full map,
    because `make_residual_heatmap` needs the per-pixel version.

    Shapes
    ------
    Input:
      img, canvas : (H, W, 3) uint8 BGR.
    Intermediate:
      img_rgb, canvas_rgb   : (H, W, 3) float64 in [0, 1] -- BGR converted to
                              RGB (skimage expects RGB) and rescaled.
      img_lab, canvas_lab   : (H, W, 3) float64 -- same size, L in [0, 100],
                              a/b roughly in [-128, 127].
    Output:
      (float, (H, W) float64 ndarray):
        mean -- scalar mean over all pixels, in Delta_E_2000 units
                (rule of thumb: > 2 is a visible difference).
        de   -- `de[y, x]` is the perceptual colour error at pixel (x, y).
                Indexing matches the input image, so masks built on the
                source (see task3_smax_curve.smooth_mask) can index it
                directly.
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    img_lab = color.rgb2lab(np.clip(img_rgb, 0, 1))
    canvas_lab = color.rgb2lab(np.clip(canvas_rgb, 0, 1))
    de = color.deltaE_ciede2000(img_lab, canvas_lab)
    return float(de.mean()), de


def edge_f1(img, canvas, sigma=2.0, tolerance=3):
    """Canny-edge F1 / precision / recall, computed with tolerance matching.

    Function
    --------
    Plain pixel-exact edge comparison is useless here: a triangle boundary can
    sit 1 pixel off a true edge and still look aligned. So each edge map is
    DILATED by `tolerance` pixels and an edge pixel counts as matched if it falls
    inside the other map's dilated version.

    READ BEFORE QUOTING THE NUMBERS -- precision and recall use DIFFERENT
    reference sets, so this is not textbook P/R:

        precision = |out edges near a source edge| / |all out edges|
        recall    = |source edges near an out edge|  / |all source edges|

    i.e. precision asks "how much of what we drew is real?", recall asks "how
    much of the real structure did we draw?". Each uses the OTHER map's dilation
    as its reference. The dilation is one-sided, so a match just outside the band
    is missed by both directions.

    Edge case: a blank canvas empties `e_out`, giving precision = 0/1 = 0 and
    recall = 0. The `max(..., 1)` guards prevent division by zero, so `f1` is 0
    rather than undefined -- read that as "nothing was drawn", not "maximally
    wrong".

    Shapes
    ------
    Input:
      img, canvas : (H, W, 3) uint8 BGR.
      sigma       : float -- Canny gaussian width (2.0 = smooth out JPEG noise).
      tolerance   : int -- dilation radius in pixels.
    Intermediate:
      g_orig, g_out : (H, W) uint8 grayscale.
      e_orig, e_out : (H, W) uint8 {0, 1} boolean edge maps from skimage.canny.
      e_*_d         : (H, W) uint8 dilated edge maps, kernel (2*tol+1, 2*tol+1).
      tp_/fp_/fn_   : int counts of pixels in each mask intersection.
    Output:
      (f1, precision, recall) : three floats in [0, 1], higher = better.
    """
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
    """EPI -- Pearson correlation of the two Sobel-response fields.

    Function
    --------
    Takes the Sobel gradient magnitude of both images, mean-centres each field,
    then computes the Pearson correlation of the pair. It measures whether strong
    edges in the mosaic land WHERE the source has strong edges, ignoring overall
    intensity -- which is why it is a signed score that can be ~0 or negative
    even on a decent mosaic.

    NAME IS LOCAL: "EPI" here is a plain Pearson correlation of Sobel magnitudes,
    NOT the classic Edge-Preservation-Index from the image-fusion literature.
    Define it explicitly wherever the report cites it.

    Shapes
    ------
    Input:
      img, canvas : (H, W, 3) uint8 BGR.
    Intermediate:
      g_orig, g_out : (H, W) float64 grayscale.
      s_orig, s_out : (H, W) float64 Sobel magnitude, same shape.
      so, sc        : (H, W) float64 mean-centred responses.
    Output:
      float in [-1, 1] -- Pearson r between the two fields.
                          > 0 means edge energy is co-located with the source.
    """
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
    """Mean quantisation error: per-triangle mean vs its assigned colour.

    Function
    --------
    Isolates the COLOUR-QUANTISATION half of the pipeline from the geometry
    half: it never looks at the rendered image, only at whether each triangle got
    a palette colour close to the one it wanted. A high Quant_Error with a good
    PSNR means the geometry is fine but the palette cannot represent the image
    (too few colours / badly placed centroids).

    Measured in CIEDE2000, matching `delta_e_2000_full`, so the two are directly
    comparable -- unlike PSNR (BGR pixel space) which is not.

    Shapes
    ------
    Input:
      means_bgr   : (T, 3) float32 -- per-triangle mean colour, `[:, 0]=B`,
                    `[:, 1]=G`, `[:, 2]=R` (from brick_render.triangle_means_bgr).
      labels      : (T,) int -- `labels[i]` is the palette index chosen for
                    triangle i, values in [0, K-1].
      palette_bgr : (K, 3) uint8 -- `palette_bgr[k]` is the colour for index k.
    Intermediate:
      means_lab      : (T, 3) float64 Lab.
      palette_lab    : (K, 3) float64 Lab.
      assigned       : (T, 3) float64 Lab -- `assigned[i] = palette_lab[labels[i]]`,
                       i.e. the colour the renderer actually painted on
                       triangle i. This is the gather that turns the
                       quantisation decision into a per-triangle comparison.
    Output:
      float -- mean Delta_E_2000 over the T triangles, lower = better.
    """
    means_rgb = means_bgr[:, ::-1].astype(np.float64) / 255.0
    palette_rgb = palette_bgr[:, ::-1].astype(np.float64) / 255.0
    means_lab = color.rgb2lab(np.clip(means_rgb, 0, 1))
    palette_lab = color.rgb2lab(np.clip(palette_rgb, 0, 1))
    assigned = palette_lab[labels]
    de = color.deltaE_ciede2000(means_lab, assigned)
    return float(de.mean())


def make_residual_heatmap(canvas, de_map):
    """JET colour-map of the per-pixel Delta_E_2000 map.

    Function
    --------
    Turns the scalar error map into something a report reader can look at:
    blue = close to the source, red = far. Used by both the Task 2 residual
    panel and the Task 3 outputs.

    Shapes
    ------
    Input:
      canvas : (H, W, 3) uint8 BGR -- the render the map was computed against.
               Not read for values; accepted so the caller cannot accidentally
               pair a heatmap with a different image's shape.
      de_map : (H, W) float64 -- per-pixel Delta_E_2000 from
               `delta_e_2000_full`.
    Output:
      (H, W, 3) uint8 BGR heatmap.

    WARNING -- normalisation is relative, not absolute: the map is scaled by
    ITS OWN maximum (`de_map.max()`), so a red pixel here means "the worst
    pixel in THIS image", not "a large error". Heatmaps from two runs are
    therefore not comparable unless both are rescaled to a shared cap. For
    absolute numbers use the scalar `Delta_E_2000` instead.
    """
    d_max = max(float(de_map.max()), 1e-9)
    d_norm = np.clip(de_map / d_max, 0, 1)
    return cv2.applyColorMap((d_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)


def compute_metrics(img, canvas, means_bgr, labels, palette_bgr, n_triangles,
                    elapsed_s):
    """Run the whole metric suite for one rendered mosaic.

    Function
    --------
    Thin orchestration: each metric is computed by its own function, including
    the grayscale / Lab conversions it needs. This function only decides WHICH
    metrics run and packages them into the 13-key dict documented in the module
    docstring.

    Shapes
    ------
    Input:
      img, canvas : (H, W, 3) uint8 BGR -- source and render, same size.
      means_bgr   : (T, 3) float32 -- per-triangle mean colour; `[:, 0]=B`.
      labels      : (T,) int -- `labels[i]` = palette index of triangle i.
      palette_bgr : (K, 3) uint8 -- `palette_bgr[k]` = colour of index k.
      n_triangles : int -- for Budget_Util / N_Triangles. Passed in rather than
                    derived, so a caller that slices `means` / `labels` can still
                    report the true triangle count (the Task 3 driver does).
      elapsed_s   : float -- seconds for whatever interval the CALLER chose; only
                    used for FPS (see the module-level FPS caveat).
    Intermediate:
      win : int -- SSIM window for the whole image, from `_ssim_window`. Any image
            at least 9 px on its short side resolves to 7, so every recorded
            SSIM is unchanged from before the helper existed.
    Output:
      dict -- 12 numeric fields + "Heatmap" (H, W, 3). Keys: PSNR, SSIM,
              MS-SSIM, Delta_E_2000, Edge_F1, Edge_Precision, Edge_Recall, EPI,
              Quant_Error, Budget_Util, N_Triangles, FPS, Heatmap.

    Raises:
      ValueError -- when `min(H, W) < 3`, i.e. no valid SSIM window exists. This
      used to crash inside skimage with an opaque window-size error, because the
      old code forced a window of 3 onto an image that could not hold one. SSIM
      is simply undefined below 3 px on a side, so failing explicitly is the fix.
    """
    H, W = img.shape[:2]
    psnr = float(peak_signal_noise_ratio(img, canvas, data_range=255))
    win = _ssim_window((H, W))
    if win is None:
        raise ValueError(
            f"Image {H}x{W} is too small for SSIM (needs min side >= 3).")
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
