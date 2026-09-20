"""
brick_render — per-triangle colour extraction and rasterisation.

- `triangle_means_bgr`: mean BGR colour under each triangle's mask.
- `render_triangles`: paints triangles with their palette colour and draws
  a thin gray border (border colour is NOT a palette entry).
"""
import cv2
import numpy as np


BORDER_COLOR_BGR = (60, 60, 60)   # dark gray; not counted as a palette colour


def triangle_means_bgr(img, triangles):
    """Compute the mean BGR colour under each triangle's mask on `img`.

    Function
    --------
    For every triangle `tri` in `triangles`, build a single-triangle binary
    mask on a canvas of size `img.shape[:2]` and call `cv2.mean(img, mask)`
    to get the per-channel mean of pixels INSIDE that triangle.

    Shapes
    ------
    Input:
      img        : (H, W, 3) uint8, BGR — source image.
      triangles  : length-T list of ndarrays, each (3, 2) int32, given as
                   `[(x1,y1), (x2,y2), (x3,y3)]` in OpenCV point order
                   (column, row).
    Intermediate:
      mask       : (H, W) uint8 — reused buffer, zeroed then filled with 1
                   inside the current triangle.
      result     : 4-tuple (B, G, R, mask_mean) float64 from cv2.mean.
    Output:
      means      : (T, 3) float32 — `means[i]` is the mean BGR of pixels
                   inside `triangles[i]`. `means[i, 0]=B`, `means[i, 1]=G`,
                   `means[i, 2]=R`. Float32 is kept for downstream palette
                   quantisation precision.
    """
    H, W = img.shape[:2]
    means = np.zeros((len(triangles), 3), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.uint8)
    for idx, tri in enumerate(triangles):
        mask[:] = 0
        # value 1 (not 255) — it is a flag for cv2.mean, not a brightness
        cv2.fillPoly(mask, [tri], 1)
        # cv2.mean returns 4-tuple (B, G, R, mask_mean); take first 3
        means[idx] = cv2.mean(img, mask=mask)[:3]
    return means


def render_triangles(canvas_shape, triangles, labels, palette_bgr, orig_shape=None):
    """Fill each triangle with its palette colour and draw a thin border.

    Function
    --------
    Allocates a black canvas, iterates over (triangle, label) pairs, fills
    each triangle with `palette_bgr[label]` and draws a 1-pixel gray
    border on top. Optionally crops back to `orig_shape`.

    Shapes
    ------
    Input:
      canvas_shape : (H, W) — size of working canvas to allocate.
      triangles    : length-T list of ndarrays, each (3, 2) int32,
                     OpenCV point order (x, y).
      labels       : (T,) int ndarray — `labels[i]` is the palette index
                     for `triangles[i]`, values in [0, K-1].
      palette_bgr  : (K, 3) ndarray — `palette_bgr[k]` is the BGR colour
                     for palette index k.
      orig_shape   : optional (oH, oW) — if given, final crop target.
    Intermediate:
      color       : 3-int Python tuple (B, G, R) cast from palette_bgr[lab],
                    needed because cv2.fillPoly/polylines reject numpy
                    scalar dtypes on some builds.
    Output:
      canvas      : (H, W, 3) uint8 BGR if `orig_shape` is None;
                    (oH, oW, 3) uint8 BGR if `orig_shape` is given.
                    Border is drawn ON TOP of the fill so it is visible
                    but does not consume a palette slot.

    Consequence of drawing the border last (deliberate, but easy to forget):
    the border OVERWRITES fill pixels along every triangle edge, so the rendered
    canvas is not a pure palette-colour image -- each cell carries a 1 px rim of
    dark gray (60, 60, 60). Those rim pixels ARE counted by every metric in
    `brick_metrics`, because PSNR / Delta_E_2000 compare the source against this
    canvas, and the rim colour is not the colour the triangle was assigned. The
    PDF explicitly permits drawn boundaries and forbids counting them as brick
    colours, so this is compliant; it is documented because the rim adds an
    identical error term to EVERY run, so the border policy has to stay constant
    for cross-run comparisons to mean anything (it does -- this is the only place
    a border is drawn).
    """
    H, W = canvas_shape
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for tri, lab in zip(triangles, labels):
        color = tuple(int(v) for v in palette_bgr[lab])
        cv2.fillPoly(canvas, [tri], color)
        cv2.polylines(canvas, [tri], isClosed=True,
                      color=BORDER_COLOR_BGR, thickness=1, lineType=cv2.LINE_8)
    if orig_shape is not None:
        oH, oW = orig_shape
        canvas = canvas[:oH, :oW]
    return canvas