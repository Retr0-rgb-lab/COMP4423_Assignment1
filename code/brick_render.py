"""
brick_render — per-triangle colour extraction and rasterisation.

`triangle_means_bgr` samples the mean BGR under each triangle's mask.
`render_triangles` fills triangles with their quantised colour and draws a
thin border (the border colour is not counted as one of the palette colours).
"""
import cv2
import numpy as np


BORDER_COLOR_BGR = (60, 60, 60)   # dark gray; not counted as a palette colour


def triangle_means_bgr(img, triangles):
    """Mean BGR under each triangle's mask. Shape (T, 3), float32."""
    H, W = img.shape[:2]
    means = np.zeros((len(triangles), 3), dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.uint8)
    for idx, tri in enumerate(triangles):
        mask[:] = 0
        cv2.fillPoly(mask, [tri], 1)
        means[idx] = cv2.mean(img, mask=mask)[:3]
    return means


def render_triangles(canvas_shape, triangles, labels, palette_bgr, orig_shape=None):
    """Fill each triangle with its class colour; draw a thin gray border.

    canvas_shape  (H, W) of the working canvas (padded size for Task 3).
    orig_shape    optional (H, W) to crop back to after rendering.
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
