"""
task3_shadow_crop — per-region Delta-E for the S_max=32 vs 64 trade-off.

Section 5.4 claims that raising S_max to 64 leaves the *sky* alone but makes
large shadowed areas visibly worse, and that is the observation the report uses
to argue the two settings are a genuine trade-off rather than one winner. That
claim needs a number, and this driver produces it deterministically: it splits
the source image into a shadow set and a bright set by a luminance percentile
and reports the mean CIEDE2000 of each set under both S_max values.

The region split is defined on the SOURCE image, not on either render, so the
two S_max values are compared over identical pixel sets. That matters: a split
derived from a render would move when the palette moves and the comparison
would no longer be like-for-like.

Writes `crop_shadow.png` and `crop_sky.png` into `code/pics/task3/analysis/`
as before/after triptychs (original crop, then the two renders).

Shapes
------
Input : `code/pics/sky.jpg` as (H, W, 3) uint8 BGR.
Output: two (crop_h*(1+n) + n*GAP, crop_w, 3) uint8 BGR strips per region, plus
        the printed per-region mean Delta-E table. Returns None (side-effecting).
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brick_geom import pad_to_max, leaves_to_triangles  # noqa: E402
from brick_quadtree import quadtree_partition  # noqa: E402
from brick_color import build_palette, quantize_nearest_bgr  # noqa: E402
from brick_render import triangle_means_bgr, render_triangles  # noqa: E402
from brick_metrics import delta_e_2000_full  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "pics", "sky.jpg")
OUT_DIR = os.path.join(HERE, "pics", "task3", "analysis")
BUDGET = 9990
K, PALETTE = 8, "median_cut"   # deterministic, so the two runs differ only in S_max
S_MAXES = (32, 64)
SHADOW_PCT = 40.0              # source pixels below this luminance percentile
GAP, CAP = 8, 26


def render_smax(img, smax):
    """Render one S_max configuration with a deterministic palette.

    Function
    --------
    The same seven-step pipeline as `task3_smax_curve.measure`, minus the metric
    suite: quadtree partition, per-triangle means, Median-Cut palette (K=8),
    nearest-palette assignment, render. Median Cut is deterministic, so the only
    difference between the two calls is S_max.

    Shapes
    ------
    Input:
      img  : (H, W, 3) uint8 BGR source.
      smax : int, largest allowed cell side; also the quadtree's start size.
    Output:
      canvas : (H, W, 3) uint8 BGR, cropped back to the source shape.
    """
    s_set = [s for s in (1, 2, 4, 8, 16, 32, 64, 128) if s <= smax]
    leaves, padded_hw, orig_shape = quadtree_partition(img, s_set, BUDGET, "mse")
    tris = leaves_to_triangles(leaves)
    padded, _, _ = pad_to_max(img, max(s_set))
    means = triangle_means_bgr(padded, tris)
    palette = build_palette(means, K, PALETTE)
    labels = quantize_nearest_bgr(means, palette)
    return render_triangles(padded_hw, tris, labels, palette, orig_shape)


def region_masks(img):
    """Split the source into a shadow set and a bright set by luminance.

    Function
    --------
    One threshold, `SHADOW_PCT`, is taken from the source's own luminance
    distribution, so the two sets are complementary and together cover every
    pixel exactly once. Defining the split on the source rather than on a render
    is what makes the S_max=32 and S_max=64 numbers comparable.

    Shapes
    ------
    Input:
      img : (H, W, 3) uint8 BGR.
    Intermediate:
      gray : (H, W) float32 luminance; `thr` is its SHADOW_PCT percentile.
    Output:
      (shadow, bright) : two (H, W) bool arrays, disjoint and jointly total.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    thr = float(np.percentile(gray, SHADOW_PCT))
    shadow = gray <= thr
    return shadow, ~shadow


def bounding_box(mask, img_w, img_h, min_frac=0.12):
    """Tight crop rectangle around a boolean mask, with a floor on its size.

    Function
    --------
    A percentile split often selects scattered pixels rather than one solid
    region, which would make an unreadable crop. This takes the mask's bounding
    box, then widens it to at least `min_frac` of each dimension so the crop is
    always a usable panel, and clamps to the image.

    Shapes
    ------
    Input:
      mask    : (H, W) bool selection.
      img_w, img_h : source dimensions, used for the clamp.
    Output:
      (x, y, w, h) : four ints, `w*h` >= min_frac*img_w*min_frac*img_h, fully
                     inside the image.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, 0, img_w, img_h
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    w, h = max(1, x1 - x0 + 1), max(1, y1 - y0 + 1)
    floor_w, floor_h = int(img_w * min_frac), int(img_h * min_frac)
    if w < floor_w:
        cx, half = (x0 + x1) // 2, (floor_w - w) // 2
        x0, w = max(0, min(cx - half, img_w - floor_w)), floor_w
    if h < floor_h:
        cy, half = (y0 + y1) // 2, (floor_h - h) // 2
        y0, h = max(0, min(cy - half, img_h - floor_h)), floor_h
    return x0, y0, min(w, img_w - x0), min(h, img_h - y0)


def save_triptych(path, original, renders, labels, mask=None, dim=0.25):
    """Write a horizontal strip: the original crop, then one crop per render.

    Function
    --------
    Used for both region figures. The panels are pixel-aligned (same crop box,
    same scale) so a viewer can compare them directly, and each carries a short
    caption bar so the figure is readable without the report text.

    When `mask` is given, pixels outside it are multiplied by `dim` in every
    panel. The region split here is a luminance percentile over the whole frame,
    not a spatial crop, so without this the two figures would be the same three
    pictures. Dimming the complement makes each figure show its own pixel set.

    Shapes
    ------
    Input:
      path     : destination .png path; parent directory must exist.
      original : (h, w, 3) uint8 BGR, the source crop.
      renders  : list of (h, w, 3) uint8 BGR crops, same shape as `original`.
      labels   : list of str, one per panel, in order.
      mask     : optional (h, w) bool; True = keep at full brightness.
      dim      : float in [0,1], brightness multiplier for masked-out pixels.
    Output: None. Writes `path`.
    """
    h, w = original.shape[:2]
    n = len(renders) + 1
    panels = [original] + list(renders)
    strip = np.full((h + CAP, n * w + (n - 1) * GAP, 3), 24, dtype=np.uint8)
    for i, (panel, label) in enumerate(zip(panels, labels)):
        x0 = i * (w + GAP)
        if mask is not None:
            panel = np.where(mask[:, :, None], panel,
                             (panel.astype(np.float32) * dim).astype(np.uint8))
        strip[:h, x0:x0 + w] = panel
        cv2.putText(strip, label, (x0 + 6, h + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.imwrite(path, strip)


def main():
    """Render both S_max values, split the source, and report per-region ΔE.

    Function
    --------
    For S_max in (32, 64): render, compute the per-pixel CIEDE2000 map against
    the source, and average it over the shadow and bright sets. Then writes one
    triptych per region and prints the table the report quotes.

    Shapes
    ------
    Input: none (reads DEFAULT_INPUT). Output: `crop_shadow.png` and
    `crop_sky.png` under `pics/task3/analysis/`, plus printed means. The ΔE map
    is (H, W) float64 and the masks are (H, W) bool.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    img = cv2.imread(DEFAULT_INPUT)
    if img is None:
        raise FileNotFoundError(f"Cannot read {DEFAULT_INPUT}")
    H, W = img.shape[:2]
    shadow, bright = region_masks(img)

    print(f"[shadow_crop] input {W}x{H}  shadow pixels={int(shadow.sum())} "
          f"({shadow.mean()*100:.1f}%)  bright={int(bright.sum())}")

    canvases, means = {}, {}
    for smax in S_MAXES:
        canvas = render_smax(img, smax)
        canvases[smax] = canvas
        _, de_map = delta_e_2000_full(img, canvas)
        means[smax] = (float(de_map[shadow].mean()),
                       float(de_map[bright].mean()))
        print(f"[shadow_crop] S_max={smax:3d}  shadow dE={means[smax][0]:6.3f}  "
              f"bright dE={means[smax][1]:6.3f}")

    d_shadow = (means[64][0] - means[32][0]) / means[32][0] * 100.0
    d_bright = (means[64][1] - means[32][1]) / means[32][1] * 100.0
    print(f"[shadow_crop] change 32 -> 64:  shadow {d_shadow:+.1f}%   "
          f"bright {d_bright:+.1f}%")

    for name, mask, idx in (("crop_shadow", shadow, 0), ("crop_sky", bright, 1)):
        x, y, w, h = bounding_box(mask, W, H)
        save_triptych(
            os.path.join(OUT_DIR, f"{name}.png"),
            img[y:y + h, x:x + w],
            [canvases[s][y:y + h, x:x + w] for s in S_MAXES],
            [f"{name}: original ({w}x{h})",
             f"S_max={S_MAXES[0]}  {['shadow', 'bright'][idx]} dE="
             f"{means[S_MAXES[0]][idx]:.2f}",
             f"S_max={S_MAXES[1]}  {['shadow', 'bright'][idx]} dE="
             f"{means[S_MAXES[1]][idx]:.2f}"],
            mask=mask[y:y + h, x:x + w])
        print(f"[shadow_crop] wrote {name}.png  crop={w}x{h} at ({x},{y})  "
              f"set={int(mask.sum())} px")


if __name__ == "__main__":
    main()
