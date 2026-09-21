"""
brick_means — per-triangle mean colour, three implementations at different
points on the speed/accuracy curve.

The existing `brick_render.triangle_means_bgr` is the reference: for every
triangle it builds a full-frame mask and calls `cv2.mean(img, mask)`. That is
correct but its cost is **O(T · H · W)** -- every primitive scans the whole
frame -- and it is the single dominant cost of the live camera loop (75% of the
frame budget, 1604 ms of 2140 ms at 640x480 with 9990 triangles).

This module adds two alternatives that exploit the fact that a brick is a square
cell split by a diagonal, so the pixels of each triangle are known analytically:

  * `means_by_masks`    -- EXACT (same definition of "mean" as the reference),
                           O(H*W) total work, vectorised per cell size.
  * `means_by_sampling` -- APPROXIMATE: averages a few strictly-interior sample
                           points per triangle. Fully vectorised, ~O(T).

`extract_means` dispatches on a method name so the driver can switch
implementations from the command line, which is what makes the Task 4
before/after table possible.

Pixel-partition convention, and a defect in the shipped reference
----------------------------------------------------------------
`render_triangles` and `triangle_means_bgr` paint a triangle with
`cv2.fillPoly`. For a polygon whose edges lie exactly on integer pixel
boundaries, `fillPoly` paints an extra one-pixel fringe along the RIGHT and
BOTTOM edges -- pixels whose centres are outside the polygon, i.e. pixels that
belong to the NEIGHBOURING cell. `triangle_means_bgr` then averages that
inflated pixel set, so every brick's colour is contaminated by its bottom-right
neighbours, worst for the small bricks. Measured on one cell, both halves, for
each size (in-cell pixels vs the pixels `fillPoly` actually claims):

    size   own px   fillPoly px   leaked px   leak as % of own
       1        1             6           4               400%
       2        4            12           6               150%
       4       16            30          10                62%
       8       64            90          18                28%
      16      256           306          34                13%
      32     1024          1122          66                 6%

That matters because the Task 3 size distribution is dominated by the small
sizes ({2: 1489, 4: 1853} of 4995 cells), i.e. exactly the bricks where the
contamination is largest. The PDF asks for each brick's colour to come "from its
own image region", so the in-cell partition below is the faithful reading and the
reference is the defective one. The reference is NOT changed here, because every
recorded Task 2/3 number depends on it; this module is where the corrected
version lives, and `compare_means` below quantifies the difference.

The functions here partition each cell's pixels analytically:

    parity 0 (main diagonal TL->BR):  tri a = {c >= r},  tri b = {c < r}
    parity 1 (anti-diagonal TR->BL):  tri a = {c + r < s}, tri b = {c + r >= s}

with `r`, `c` the local row/column inside the cell and `s` the cell side. Those
halves are disjoint and cover the cell exactly, so no pixel is counted twice, no
pixel is lost, and no pixel from a neighbour is included.

Degenerate case: for `s == 1` a single pixel cannot be split, so one of the two
half-masks is empty. Both halves then report that pixel's colour, which is what
`fillPoly` effectively does as well (it paints both halves over the same pixel).

Ordering contract
-----------------
`means_by_masks` and `means_by_sampling` return rows in exactly the order
`leaves_to_triangles` emits triangles: for each leaf, its two halves in that
leaf's order. The caller therefore pairs `means[2k]` / `means[2k+1]` with
`triangles[2k]` / `triangles[2k+1]`. Getting this wrong paints each brick with its
neighbour's colour and raises nothing, so `compare_means` below asserts it.
"""
import numpy as np

# Methods accepted by `extract_means`. Kept in one place so the CLI choices and
# the dispatcher cannot drift apart. "rows" is the row-run table implementation
# in `brick_means_rows` (bit-identical to "fast", fewer numpy calls).
METHODS = ("mask", "fast", "sample", "rows")


def cell_groups(leaves):
    """Group leaf indices by cell size, with origins and diagonal parity.

    Function
    --------
    The fast paths precompute a mask once per SIZE rather than once per cell, so
    every implementation in this module starts by bucketing the leaves. This is
    the one piece of shared scaffolding; it exists so the three methods cannot
    disagree about what "parity" means or about which leaf is which.

    Shape/semantics: `leaves` is an iterable of `(x, y, size)` in PADDED image
    coordinates. Returns `dict[size]` -> `(idx, xs, ys, par)` where `idx` are the
    positions of those leaves in the input order (so results can be written back
    without disturbing triangle ordering), `xs`/`ys` are the cell origins, and
    `par` is `(y // size + x // size) % 2` per leaf -- the same expression
    `brick_geom.leaves_to_triangles` uses to choose the diagonal, which is what
    makes the two agree about which half is `a` and which is `b`.
    """
    leaves = list(leaves)
    groups = {}
    for k, (x, y, s) in enumerate(leaves):
        idx, xs, ys, par = groups.setdefault(s, ([], [], [], []))
        idx.append(k)
        xs.append(x)
        ys.append(y)
        par.append((y // s + x // s) % 2)
    return {s: (np.array(idx), np.array(xs), np.array(ys), np.array(par))
            for s, (idx, xs, ys, par) in groups.items()}


def _half_masks(s):
    """Boolean (s, s) masks for the two halves of a cell, one pair per parity.

    Function: `np.indices((s, s))` gives local row `r` in axis 0 and local column
    `c` in axis 1. The two parities use different diagonals, so the pair of
    masks differs; see the module docstring for the convention.

    Shape/semantics: returns `(m_main, m_anti)`, each a tuple `(a, b)` of
    `(s, s)` boolean arrays with `a & b == False` and `a | b == True` -- i.e. an
    exact partition of the cell. `m_main` is for parity 0, `m_anti` for parity 1.
    """
    r, c = np.indices((s, s))
    m_main = (c >= r, c < r)
    m_anti = (c + r < s, c + r >= s)
    return m_main, m_anti


def means_by_masks(img, leaves):
    """Exact per-triangle mean over the cell's own pixels, O(H*W) total.

    Function
    --------
    For each cell size, walk the pixels of each half-mask as an OFFSET list and
    gather all cells of that size in one vectorised indexing step per offset. The
    total number of gathered pixels across all offsets is exactly H*W, so the work
    is O(H*W) for the whole image instead of O(T*H*W).

    This is the CORRECTED mean, not a faster version of the reference: it sums
    only the triangle's own in-cell pixels, while `brick_render.triangle_means_bgr`
    additionally averages a one-pixel fringe of neighbouring pixels along the
    right and bottom edges (see the module docstring for the measured size). The
    two therefore disagree on purpose, and on small bricks the reference's extra
    pixels outnumber the real ones. Use `compare_means` to see the difference.

    Why offsets rather than a mask per cell: a cell's pixel set is determined by
    its origin plus a size-only offset pattern, so one offset list per (size,
    parity) serves every cell of that size. Building a mask per cell is what made
    the reference slow.

    Shapes
    ------
    Input: `img` is (Hp, Wp, 3) uint8 BGR -- the PADDED image, since the leaves
    are in padded coordinates. `leaves` is an iterable of `(x, y, size)`.
    Intermediate: `m_main` / `m_anti` are (s, s) booleans per size; `dys`, `dxs`
    are the flat offset vectors of the half being summed, so a (s, s) mask is
    walked as `len(dys)` gathers of shape `(n_cells, 3)`.
    Output: `(2*len(leaves), 3)` float32 -- row `2k`/`2k+1` are the means of the
    two halves of leaf `k`, channel order `[..., 0]=B`. See the ordering contract
    in the module docstring.

    Raises: `ValueError` if a cell's offsets would read outside `img`, which can
    only happen if the caller forgot to pad.
    """
    leaves = list(leaves)
    out = np.zeros((2 * len(leaves), 3), dtype=np.float32)
    Hp, Wp = img.shape[:2]
    for s, (idx, xs, ys, par) in cell_groups(leaves).items():
        m_main, m_anti = _half_masks(s)
        for parity_value, masks in ((0, m_main), (1, m_anti)):
            sel = par == parity_value
            if not sel.any():
                continue
            sx, sy = xs[sel], ys[sel]
            if sx.max() + s > Wp or sy.max() + s > Hp:
                raise ValueError(
                    f"cell size {s} at ({sx.max()},{sy.max()}) exceeds the "
                    f"{Wp}x{Hp} image; pad the image first")
            for half, mask in enumerate(masks):
                dys, dxs = np.nonzero(mask)
                if len(dys) == 0:
                    # Degenerate half: a 1-pixel cell has no room for two
                    # triangles, and for `s == 1` one of the two half-masks is
                    # always empty (mask a keeps the pixel, mask b gets nothing).
                    # `cv2.fillPoly` paints that single pixel for BOTH halves, so
                    # both must report the cell's colour -- falling back to the
                    # pixel at local (0, 0) reproduces that. Without this the
                    # affected triangles would average zero pixels.
                    dys = dxs = np.array([0])
                acc = np.zeros((sx.size, 3), dtype=np.float32)
                for dy, dx in zip(dys, dxs):
                    # One gather per offset pixel, over ALL cells of this size at
                    # once. `n_tri` is the row index for leaf `idx[sel]`, half
                    # `half`: 2k + 0 for half a, 2k + 1 for half b.
                    acc += img[sy + dy, sx + dx]
                # Divide by the number of pixels summed: this is a MEAN, and
                # omitting the division writes a sum whose values exceed 255.
                out[2 * idx[sel] + half] = acc / len(dys)
    return out


def means_by_sampling(img, leaves, n_samples=9):
    """Approximate per-triangle mean colour from a few interior sample points.

    Function
    --------
    Uses the same half-masks as `means_by_masks`, but instead of summing every
    pixel of the half it takes an evenly spaced subset of at most `n_samples`
    interior pixels and averages those. Fully vectorised: one gather per sample
    point, each covering every triangle of that (size, half, parity) class.

    Trade-off (this is the point of having it): cost is O(T * n_samples) instead
    of O(H*W), which stops depending on the image size at all, at the price of a
    colour error that grows when the triangle is not internally uniform. Because
    a triangle's colour is reported as a single flat value, the error is bounded
    by the triangle's own internal variation -- for the large cells over a smooth
    gradient that is exactly where the approximation is weakest.

    Shapes
    ------
    Input: `img` (Hp, Wp, 3) uint8 BGR padded; `leaves` iterable of
    `(x, y, size)`; `n_samples` the cap on sample points per triangle.
    Intermediate: `pts` is the (m, 2) `nonzero` of a half-mask, subsampled to
    `min(n_samples, m)` rows by `np.linspace` so the SAME points are chosen for
    every triangle of a class -- deterministic, no RNG, so two runs agree.
    Output: `(2*len(leaves), 3)` float32, same ordering contract as
    `means_by_masks`.

    Edge cases: for a 1-pixel triangle the half holds 0 or 1 pixels; a half with
    no interior pixel falls back to the single nearest in-cell pixel so a colour
    is always produced rather than a NaN.
    """
    leaves = list(leaves)
    out = np.zeros((2 * len(leaves), 3), dtype=np.float32)
    for s, (idx, xs, ys, par) in cell_groups(leaves).items():
        m_main, m_anti = _half_masks(s)
        for parity_value, masks in ((0, m_main), (1, m_anti)):
            sel = par == parity_value
            if not sel.any():
                continue
            sx, sy = xs[sel], ys[sel]
            for half, mask in enumerate(masks):
                pts = np.argwhere(mask)
                if len(pts) == 0:
                    pts = np.array([[0, 0]])
                if len(pts) > n_samples:
                    keep = np.linspace(0, len(pts) - 1, n_samples).astype(int)
                    pts = pts[keep]
                acc = np.zeros((sx.size, 3), dtype=np.float32)
                for dy, dx in pts:
                    acc += img[sy + dy, sx + dx]
                out[2 * idx[sel] + half] = acc / len(pts)
    return out


def extract_means(img, leaves, method="mask", n_samples=9):
    """Dispatch to one of the three mean-colour implementations by name.

    Function: the single switching point for the whole speed/accuracy choice, so
    the camera driver carries a string and nothing else. `mask` is the reference
    implemented in `brick_render` and is imported lazily to keep this module from
    depending on the rasteriser at import time.

    Shapes: `img` (Hp, Wp, 3) uint8 BGR padded; `leaves` iterable of
    `(x, y, size)`. Returns `(2*len(leaves), 3)` float32 with the ordering
    contract from the module docstring, whichever method is chosen.

    Semantics: an unknown `method` RAISES rather than defaulting, for the same
    reason `brick_color.build_palette` does -- a typo must not silently run a
    different algorithm under the label of the requested one, because the
    resulting before/after table would then be wrong with no visible symptom.
    """
    if method == "fast":
        return means_by_masks(img, leaves)
    if method == "sample":
        return means_by_sampling(img, leaves, n_samples)
    if method == "rows":
        # Row-run prefix sums; bit-identical to "fast" by exact-integer sums
        # (see brick_means_rows). Imported lazily to avoid a circular import.
        from brick_means_rows import means_by_rows
        return means_by_rows(img, leaves)
    if method == "mask":
        from brick_geom import leaves_to_triangles
        from brick_render import triangle_means_bgr
        return triangle_means_bgr(img, leaves_to_triangles(leaves))
    raise ValueError(f"Unknown means method {method!r}; expected one of {METHODS}.")


def compare_means(img, leaves, n_samples=9):
    """Diff the fast paths against the reference, and measure the reference's leak.

    Function
    --------
    Three silent invariants can break without an exception anywhere: the ordering
    contract, the diagonal convention, and (already broken in the shipped code)
    the reference's pixel set. This checks the first two are intact and reports
    the third, which is what the Task 4 verification runs instead of eyeballing
    the picture.

    Shape/semantics: same inputs as `extract_means`. Returns a flat dict:
      `fast_mean_absdiff` / `fast_max_absdiff` and the `sample_*` pair -- the
      difference between each fast path and the reference, in 0-255 channel units.
      These are EXPECTED to be non-zero, because the reference includes the
      neighbour fringe; a max of a few tens of units on small bricks is the
      documented defect, not a failure of this module.
      `ref_mean_absdiff_vs_own` -- the same comparison for the reference itself,
      i.e. how far the shipped mean is from the correct in-cell mean. This is the
      headline number for the defect.
      `ref_extra_px` / `own_px` -- pixels the reference averages beyond the
      triangle's own, and the own-pixel total, so the leak can be quoted as a
      ratio rather than as an unexplained colour difference.
      `sample_own_mean_absdiff` -- the sampling method against the CORRECT
      in-cell mean, which is the honest accuracy figure for it (comparing it to
      the leaky reference would flatter it).

    Reading the output: a LARGE max together with a small mean is the signature of
    a permuted row (broken ordering); a moderate max concentrated on small cells
    is the neighbour leak. Both are visible here without a picture.
    """
    from brick_geom import leaves_to_triangles
    from brick_render import triangle_means_bgr

    leaves = list(leaves)
    own = means_by_masks(img, leaves)
    ref = triangle_means_bgr(img, leaves_to_triangles(leaves))

    out = {}
    for name, got in (("fast", own), ("sample", means_by_sampling(img, leaves,
                                                                 n_samples))):
        d = np.abs(got.astype(np.float64) - ref.astype(np.float64))
        out[f"{name}_mean_absdiff"] = float(d.mean())
        out[f"{name}_max_absdiff"] = float(d.max())

    d_ref = np.abs(ref.astype(np.float64) - own.astype(np.float64))
    out["ref_mean_absdiff_vs_own"] = float(d_ref.mean())
    out["ref_max_absdiff_vs_own"] = float(d_ref.max())
    out["sample_own_mean_absdiff"] = float(
        np.abs(means_by_sampling(img, leaves, n_samples).astype(np.float64)
               - own.astype(np.float64)).mean())

    # Count the reference's own pixels vs its leaked pixels, over the real leaves,
    # so the ratio in the report comes from the same data as the colour diff.
    import cv2
    own_px = extra_px = 0
    tris = leaves_to_triangles(leaves)
    for k, (x, y, s) in enumerate(leaves):
        for half in (0, 1):
            mask = np.zeros(img.shape[:2], np.uint8)
            cv2.fillPoly(mask, [tris[2 * k + half]], 1)
            ys, xs = np.nonzero(mask)
            inside = ((ys >= y) & (ys < y + s) & (xs >= x) & (xs < x + s))
            own_px += int(inside.sum())
            extra_px += int((~inside).sum())
    out["own_px"] = own_px
    out["ref_extra_px"] = extra_px
    out["ref_extra_ratio"] = extra_px / max(own_px, 1)
    return out
