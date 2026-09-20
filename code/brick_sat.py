"""
brick_sat — summed-area tables for O(1) rectangle statistics.

A summed-area table (integral image) turns "sum of pixels over any axis-aligned
rectangle" into four array lookups, after one O(H*W) pass. Two tables give the
sum of the values and the sum of their squares, which is enough for the mean and
for the sum of squared errors:

    SSE(rect) = sum(p^2) - sum(p)^2 / n          n = w*h

That identity is what makes this useful here: the adaptive partitioners score
every candidate cell by its SSE, so a per-candidate numpy pass over the region
costs O(region) while a table lookup costs O(1). `brick_quadtree` is the reason
this module exists -- it recomputed a region mean for every split candidate, which
was 82% of a live camera frame.

Why a separate module instead of reusing `brick_region_merge.rect_sse`
--------------------------------------------------------------
That function is the same idea written inline before this module existed. It is
deliberately left alone: every recorded Task 3 number comes from that code path,
and unifying the two is a separate change that would have to be re-verified
against the whole Task 3 sweep. So the duplication is documented rather than
silently resolved, and this module is the version new code should use.

Numerical caveat (real, and worth knowing before trusting the last digits)
------------------------------------------------------------------------
`sum(p^2) - sum(p)^2/n` subtracts two large nearly-equal numbers when the variance
is small, so it loses precision: for a uint8 channel the terms reach ~1e11 (4
megapixels) while the SSE of a flat region can be a few hundred. float64 gives
roughly 15 significant digits, so a flat cell's SSE is still resolved, but do not
expect the last digits of a very flat region's SSE to be meaningful. Results are
clamped at 0 to stop the noise from producing a negative SSE, which would be
nonsense and would make a split look like it *gains* error.
"""
from typing import NamedTuple, Optional

import cv2
import numpy as np
from skimage import filters


class SatBundle(NamedTuple):
    """The tables plus the image size they describe.

    Shape/semantics: `s1` and `s2` are (H+1, W+1, C) float64 -- the inclusive
    prefix sums of the pixel values and of their squares, with a zero row and
    column so a rectangle touching the image edge needs no special case (the
    standard SAT trick). `s1[y, x] = sum of img[0:y, 0:x]`. `m1` / `m2` are the
    same two tables for a SINGLE-channel Sobel magnitude map, or None when they
    were not requested; they exist because the quadtree's `edgef1` priority is the
    variance of that map, which is another O(1) rectangle statistic.
    """

    s1: np.ndarray
    s2: np.ndarray
    m1: Optional[np.ndarray]
    m2: Optional[np.ndarray]


def build_sats(img, with_sobel=False):
    """Build the prefix-sum tables for `img`, optionally including Sobel ones.

    Function
    --------
    One O(H*W) pass over the image (`cumsum` twice) produces tables that answer
    any rectangle query in O(1). When `with_sobel` is set, the grayscale Sobel
    gradient magnitude is computed ONCE for the whole image and two more tables
    are built from it, so an `edgef1`-style variance over a region is also O(1).

    Shapes
    ------
    Input: `img` is (H, W, C) uint8 BGR, normally already padded so every cell
    fits. `with_sobel` toggles the extra tables.
    Intermediate: `img_d` is (H, W, C) float64; the Sobel step converts to a
    (H, W) float64 grayscale magnitude first.
    Output: a `SatBundle` with `s1`, `s2` (H+1, W+1, C) float64 and `m1`, `m2`
    (H+1, W+1) float64 or None.

    Semantics warning: the global Sobel map is NOT identical to running Sobel on
    each region separately. `skimage.filters.sobel` reflects at its input's
    boundary, so a per-region call sees a mirrored neighbourhood where a global
    call sees the real neighbouring pixels. The two therefore differ within about
    three pixels of a region's edge. The global version is the more defensible of
    the two (no invented pixels), but it is a behaviour change, not just a
    speed-up, so it is measured rather than assumed.
    """
    img_d = img.astype(np.float64)
    if img_d.ndim == 2:
        img_d = img_d[:, :, None]
    h, w, c = img_d.shape
    s1 = np.zeros((h + 1, w + 1, c), dtype=np.float64)
    s2 = np.zeros((h + 1, w + 1, c), dtype=np.float64)
    s1[1:, 1:] = img_d.cumsum(axis=0).cumsum(axis=1)
    s2[1:, 1:] = (img_d * img_d).cumsum(axis=0).cumsum(axis=1)

    m1 = m2 = None
    if with_sobel:
        if img_d.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            gray = img_d[:, :, 0]
        mag = filters.sobel(gray)
        m1 = np.zeros((h + 1, w + 1), dtype=np.float64)
        m2 = np.zeros((h + 1, w + 1), dtype=np.float64)
        m1[1:, 1:] = mag.cumsum(axis=0).cumsum(axis=1)
        m2[1:, 1:] = (mag * mag).cumsum(axis=0).cumsum(axis=1)
    return SatBundle(s1, s2, m1, m2)


def rect_sum(b, x, y, w, h):
    """Pixel sum over the rectangle [x, x+w) x [y, y+h), per channel.

    Function: the four-term inclusive-exclusion lookup that is the whole point of
    a summed-area table. The `+1` offsets come from the zero row/column in the
    tables, so a rectangle touching the far edge needs no clamp.

    Shape/semantics: returns (C,) float64 where `out[c]` is the sum of channel `c`
    over the rectangle; `out[0]=B` for BGR input. Raises nothing for an empty
    rectangle (w or h of 0 gives 0.0), but callers computing an SSE must not pass
    one, because that would divide by n=0.
    """
    x1, y1 = x + w, y + h
    return b.s1[y1, x1] - b.s1[y, x1] - b.s1[y1, x] + b.s1[y, x]


def rect_sse(b, x, y, w, h):
    """Sum of squared deviations from the mean, per channel, over a rectangle.

    Function: evaluates `sum(p^2) - sum(p)^2/n` from the two tables. This is the
    quantity a partitioner minimises: the error left after painting the rectangle
    with one flat colour, i.e. what splitting it can save.

    Shape/semantics: returns (C,) float64, `out[c]` the SSE of channel `c` over
    the rectangle, clamped at 0 (see the module's numerical caveat -- the identity
    can go slightly negative for a near-flat region). `n = w*h` must be positive.
    """
    n = w * h
    s1 = rect_sum(b, x, y, w, h)
    x1, y1 = x + w, y + h
    s2 = b.s2[y1, x1] - b.s2[y, x1] - b.s2[y1, x] + b.s2[y, x]
    return np.maximum(s2 - s1 * s1 / n, 0.0)


def rect_sse_total(b, x, y, w, h):
    """`rect_sse` summed over channels: the scalar cost a split decision uses.

    Function: the partitioners rank candidate splits by a single number, not by a
    per-channel vector, so this collapses the vector. Summing rather than
    averaging keeps the scale comparable to the reference implementation
    (`brick_quadtree._region_mse`), which also summed over channels -- that
    matters because the recorded Task 3 results were produced with that scale.

    Shape/semantics: rectangle in, one float out. Semantics: the total squared
    error over all pixels and all channels of the rectangle.
    """
    return float(rect_sse(b, x, y, w, h).sum())


def rect_sobel_var(b, x, y, w, h):
    """Variance of the Sobel magnitude over a rectangle, via the same tables.

    Function: Var = E[m^2] - E[m]^2, evaluated from the magnitude tables, so the
    quadtree's `edgef1` priority costs O(1) per candidate instead of running
    `filters.sobel` plus `.var()` over the region every time.

    Shape/semantics: requires `build_sats(..., with_sobel=True)`, otherwise raises
    `ValueError` rather than silently returning 0 (a priority of 0 everywhere
    would stop the quadtree from splitting anything, which would look like a
    quality regression rather than a missing table). Returns a scalar float; the
    magnitude is unsigned so the variance is always >= 0, clamped for the same
    near-flat cancellation reason as `rect_sse`.
    """
    if b.m1 is None:
        raise ValueError("Sobel tables missing; call build_sats(..., with_sobel=True)")
    n = w * h
    m_sum = b.m1[y + h, x + w] - b.m1[y, x + w] - b.m1[y + h, x] + b.m1[y, x]
    m_sq = b.m2[y + h, x + w] - b.m2[y, x + w] - b.m2[y + h, x] + b.m2[y, x]
    return float(max(m_sq / n - (m_sum / n) ** 2, 0.0))


# ---------------------------------------------------------------------------
# Batch forms -- these are what actually make the partitioners fast
# ---------------------------------------------------------------------------
# Why the batch forms exist, since it is not obvious: the per-rectangle functions
# above are O(1) in ARITHMETIC but not in NUMPY CALL OVERHEAD. Each one costs
# about eight fancy-index operations, and a scalar result forces Python to
# round-trip through numpy every time. The quadtree evaluates ~5 rectangles per
# split candidate, so at 9990 triangles that is ~160000 tiny numpy calls, and
# profiling showed the whole stage was dominated by that overhead rather than by
# arithmetic -- which is why an O(1)-per-candidate rewrite was measured to be no
# faster at all. The batch forms below do the same eight operations for a WHOLE
# ARRAY of rectangles at once, which is where the speedup actually comes from.

def rect_sse_total_many(b, xs, ys, ss):
    """`rect_sse_total` for many squares at once.

    Function: evaluates the SSE identity for an array of axis-aligned squares in
    a fixed number of numpy operations regardless of how many squares are asked
    for. This is the primitive the quadtree batches its split candidates through.

    Shape/semantics: `xs`, `ys`, `ss` are 1-D integer arrays of equal length N --
    the top-left corner and the side of each square, all in padded coordinates.
    Returns (N,) float64 where `out[i]` is the total squared error over all pixels
    and channels of square i. All squares must be non-empty (`ss > 0`) and inside
    the tables, since a zero side would divide by n=0; callers filter those out.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    ss = np.asarray(ss)
    x1 = xs + ss
    y1 = ys + ss
    s1 = b.s1[y1, x1] - b.s1[ys, x1] - b.s1[y1, xs] + b.s1[ys, xs]
    s2 = b.s2[y1, x1] - b.s2[ys, x1] - b.s2[y1, xs] + b.s2[ys, xs]
    n = (ss * ss)[:, None]
    return np.maximum(s2 - s1 * s1 / n, 0.0).sum(axis=1)


def rect_sobel_var_many(b, xs, ys, ss):
    """`rect_sobel_var` for many squares at once.

    Function: the batch form of the Sobel-magnitude variance, for the `edgef1`
    priority. Same rationale as `rect_sse_total_many`.

    Shape/semantics: `xs`, `ys`, `ss` are 1-D arrays of length N. Returns (N,)
    float64, `out[i]` the variance of the Sobel magnitude over square i, clamped
    at 0. Raises `ValueError` when the Sobel tables were not built, for the same
    loud-failure reason as the scalar form.
    """
    if b.m1 is None:
        raise ValueError("Sobel tables missing; call build_sats(..., with_sobel=True)")
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    ss = np.asarray(ss)
    x1 = xs + ss
    y1 = ys + ss
    m_sum = b.m1[y1, x1] - b.m1[ys, x1] - b.m1[y1, xs] + b.m1[ys, xs]
    m_sq = b.m2[y1, x1] - b.m2[ys, x1] - b.m2[y1, xs] + b.m2[ys, xs]
    n = (ss * ss).astype(np.float64)
    return np.maximum(m_sq / n - (m_sum / n) ** 2, 0.0)
