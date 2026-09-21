"""
brick_color_rt -- temporal colour helpers for the live loop (Task 4 opt A+C).

New module rather than additions to `brick_color.py` for two reasons: that file
is already near the 400-line limit (AGENTS 4.1), and every recorded Task 2/3
result depends on it, so its code path stays byte-identical. The functions here
mirror the colour-space conventions documented in `brick_color`:

  CLUSTERING  Lab for the warm-start k-means (perceptually uniform), RGB when
              asked; ASSIGNMENT is always plain BGR Euclidean, exactly as in
              `brick_color.quantize_nearest_bgr`.
  BGR order   every array named `*_bgr` stores OpenCV order along the last
              axis -- `[..., 0]=B`, `[..., 1]=G`, `[..., 2]=R`. The `[:, ::-1]`
              flips below are required by skimage and are not cosmetic.

Two temporal-stability primitives:
  * `palette_kmeans_warm` -- one k-means pass SEEDED with the previous
    palette's assignment (`cv2.KMEANS_USE_INITIAL_LABELS`). A rebuild then
    refines the previous local optimum instead of jumping to a different one,
    which is what removed the measured 76%-of-pixels palette flash
    (docs/progress/Task4.md, step 5).
  * `quantize_nearest_bgr_sticky` -- nearest-palette assignment with a
    dead-band: a triangle keeps its previous label unless another entry is
    better by a relative margin, which stops boundary flapping on static
    scenes.
"""
import cv2
import numpy as np
from skimage import color as skcolor


def _mean_space(means_bgr, space):
    """Means (T,3) uint8/float BGR -> clustering features (T,3) float32.

    Shape/semantics: `out[i]` is sample i's feature vector; Lab when
    `space == "lab"` (L in [0,100], a,b signed), else RGB in [0,255] because
    the BGR channels are flipped to RGB. Matches `brick_color.palette_kmeans`'s
    feature construction exactly so the two are comparable.
    """
    if space == "lab":
        rgb = means_bgr[:, ::-1].astype(np.float64) / 255.0
        return skcolor.rgb2lab(np.clip(rgb, 0, 1)).astype(np.float32)
    return means_bgr[:, ::-1].astype(np.float32)


def _palette_space(palette_bgr, space):
    """Palette (K,3) uint8 BGR -> the SAME feature space as `_mean_space`.

    Shape/semantics: `out[k]` is palette entry k in clustering space, so the
    distance to `_mean_space` samples is meaningful. Same flips as `_mean_space`
    (and the same warning: a wrong flip swaps red and blue but still renders
    plausibly, so it is documented rather than "simplified").
    """
    if space == "lab":
        rgb = palette_bgr[:, ::-1].astype(np.float64) / 255.0
        return skcolor.rgb2lab(np.clip(rgb, 0, 1)).astype(np.float32)
    return palette_bgr[:, ::-1].astype(np.float32)


def _centers_to_bgr(centers, space):
    """Centroids (K,3) float in clustering space -> (K,3) uint8 BGR.

    Shape/semantics: `out[k, 0]=B`, `[k,1]=G`, `[k,2]=R`, clipped to [0,255].
    For Lab the float array is a single (1,K,3) image passed through
    `lab2rgb`; for RGB it is just a channel flip. Same mapping as
    `brick_color.palette_kmeans`.
    """
    if space == "lab":
        rgb = skcolor.lab2rgb(centers.reshape(1, -1, 3)).reshape(-1, 3)
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)[:, ::-1]
    return np.clip(centers, 0, 255).astype(np.uint8)[:, ::-1]


def palette_kmeans_warm(means_bgr, k, prev_palette_bgr, space="lab"):
    """Warm-started k-means palette: refine `prev_palette_bgr`, do not re-draw.

    Function
    --------
    A normal `cv2.kmeans` run re-draws its initial centroids from OpenCV's
    global RNG and takes the best of several restarts, so two runs on nearly
    identical data can land in completely different local optima -- the measured
    cause of the 76%-of-pixels palette jump (step 5 analysis). This version
    seeds the one allowed pass with the assignment of the previous palette
    (`KMEANS_USE_INITIAL_LABELS`), so the result is a refinement of the previous
    solution. Because the initial labels tie every sample to a previous
    centroid, palette-entry correspondence is preserved too: entry k stays
    "the same colour" across rebuilds, which also prevents label permutation.

    Shapes
    ------
    Input: `means_bgr` (T,3) BGR floats (triangle means); `k` palette size;
    `prev_palette_bgr` (K,3) uint8 BGR, the previous palette.
    Intermediate: `means_in`/`prev_in` are (T,3)/(K,3) float32 in clustering
    space; `init` is (T,1) int32 -- sample i's nearest previous entry.
    Output: (K,3) uint8 BGR; entry order follows the previous palette.
    """
    means_in = _mean_space(means_bgr, space)
    prev_in = _palette_space(prev_palette_bgr, space)
    # Initial partition: each sample to its nearest previous centroid. This is
    # exactly what quantize_nearest_bgr does, only in the clustering space.
    d = ((means_in[:, None, :] - prev_in[None, :, :]) ** 2).sum(axis=2)
    init = np.argmin(d, axis=1).astype(np.int32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    # attempts=1 is required with KMEANS_USE_INITIAL_LABELS: the whole point is
    # NOT to try random alternatives.
    _, _, centers = cv2.kmeans(means_in, int(k), init, criteria, 1,
                               cv2.KMEANS_USE_INITIAL_LABELS)
    return _centers_to_bgr(centers, space)


def quantize_nearest_bgr_sticky(means_bgr, palette_bgr, prev_labels, margin=0.1):
    """Nearest-palette assignment with a dead-band around the previous label.

    Function
    --------
    Plain `quantize_nearest_bgr` re-argmins every frame, so a triangle whose
    colour sits near a palette boundary flips labels on tiny sensor noise --
    the measured ~3%-of-cells-per-frame flap. Here a triangle KEEPS its
    previous label unless the newly-best entry is better by a relative margin
    on the squared-distance ratio:

        switch  <=>  d_new_best < d_prev * (1 - margin)

    With `margin=0` this reduces exactly to `quantize_nearest_bgr`; with
    `margin=0.1` an entry must be ~10% closer in squared distance (~5% in
    distance) to take the triangle over. This is a hysteresis, not a
    smoother: it never blurs colours, it only refuses marginal switches.

    Shapes
    ------
    Input: `means_bgr` (T,3) float BGR; `palette_bgr` (K,3) uint8 BGR;
    `prev_labels` (T,) int in [0,K) from the previous frame, or None.
    Intermediate: `dist` (T,K) float32 squared BGR distances; `d_prev`,
    `d_best` (T,).
    Output: (T,) uint8 -- triangle i's palette index. Semantics of the output
    is identical to `quantize_nearest_bgr`; only the marginal-switch decision
    differs.
    Falls back to the plain rule when `prev_labels` is missing or has a
    different length (i.e. the partition was rebuilt), because a stale label
    array would index the wrong triangles.
    """
    means = means_bgr.astype(np.float32)
    pal = palette_bgr.astype(np.float32)
    dist = ((means[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
    best = np.argmin(dist, axis=1).astype(np.uint8)
    if prev_labels is None or margin <= 0 or np.shape(prev_labels) != np.shape(best):
        return best
    rows = np.arange(len(best))
    d_prev = dist[rows, prev_labels]
    d_best = dist[rows, best]
    switch = d_best < d_prev * (1.0 - margin)
    return np.where(switch, best, prev_labels).astype(np.uint8)
