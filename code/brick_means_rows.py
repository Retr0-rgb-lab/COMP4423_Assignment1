"""
brick_means_rows -- exact per-triangle means via row-run prefix sums (Task 4).

Why this exists
---------------
`brick_means.means_by_masks` ("fast") walks every pixel of each half-mask as an
OFFSET and issues one vectorised gather per offset, so a 640x480 frame costs
~sum(s^2/2) ~ 2730 tiny numpy calls. Inside a cell, though, each triangle half
is a CONTIGUOUS run of pixels on every row, so a per-row prefix-sum table
answers one row's run in two lookups. This module does that: the call count
drops to ~sum(2s) (a few hundred), with the same pixel set and the same result.

Bit-identity with means_by_masks (why reordering is safe here)
-------------------------------------------------------------
Every triangle's pixel sum is an exact integer: at most 255 * (32*33/2) =
134640, far below 2^24, so float32 represents every partial sum exactly. Exact
integer addition is associative, so the accumulation ORDER cannot change the
value; both methods end up dividing the same exact integer sum by the same
pixel count, in float32. `compare_with_masks` asserts this equality rather than
assuming it.

Ordering contract: identical to `brick_means` -- for leaf k, output rows 2k and
2k+1 are its two halves in `brick_geom.leaves_to_triangles` order. Pixel runs
per (parity, half, row r) inside a size-s cell:

    parity 0 (diagonal TL->BR):  a = {c >= r}      -> cols [r, s)
                                 b = {c <  r}      -> cols [0, r)
    parity 1 (diagonal TR->BL):  a = {c + r <  s}  -> cols [0, s-r)
                                 b = {c + r >= s}  -> cols [s-r, s)
"""
import numpy as np

from brick_means import cell_groups


def _row_prefix(img):
    """Per-row prefix sums of `img`, with a zero column at x=0.

    Function: one `cumsum` along the width axis, so the sum of any row segment
    [x0, x1) is `P[y, x1] - P[y, x0]`.

    Shape/semantics: `img` is (Hp, Wp, C) uint8; returns (Hp, Wp+1, C) float32
    with `P[y, 0, c] = 0` and `P[y, x, c]` = sum of channel c over columns
    [0, x) of row y. float32 is exact here because a full row sums to at most
    Wp*255 < 2^24 for uint8 input.
    """
    Hp, Wp = img.shape[:2]
    P = np.zeros((Hp, Wp + 1, img.shape[2]), dtype=np.float32)
    np.cumsum(img, axis=1, dtype=np.float32, out=P[:, 1:, :])
    return P


def means_by_rows(img, leaves):
    """Exact per-triangle mean colour via per-row run sums.

    Function
    --------
    For every (size, parity) class, walk the s rows of the cell and add each
    half's contiguous column run, read from the row-prefix table. This is the
    same partition of pixels as `brick_means.means_by_masks`; only the
    summation route differs, and exact-integer arithmetic makes it bit-identical
    (see the module docstring).

    Shapes
    ------
    Input: `img` (Hp, Wp, 3) uint8 BGR -- the PADDED image (leaves are in padded
    coordinates); `leaves` an iterable of (x, y, size).
    Intermediate: per (size, parity, row) the run is two fancy-index lookups on
    `P` of shape (n_cells, C); `acc_a`/`acc_b` are (n_cells, C) float32 sums.
    Output: (2*len(leaves), 3) float32, `[..., 0]=B`, same ordering contract as
    `brick_means.extract_means`.
    Degenerate case: for `size == 1` the b half has no pixel (count 0), so both
    halves report the single cell pixel, matching `means_by_masks`' fallback.
    Raises `ValueError` when a cell would read outside `img` (unpadded input).
    """
    leaves = list(leaves)
    out = np.zeros((2 * len(leaves), 3), dtype=np.float32)
    Hp, Wp = img.shape[:2]
    P = _row_prefix(img)
    for s, (idx, xs, ys, par) in cell_groups(leaves).items():
        if xs.max() + s > Wp or ys.max() + s > Hp:
            raise ValueError(
                f"cell size {s} at ({xs.max()},{ys.max()}) exceeds the "
                f"{Wp}x{Hp} image; pad the image first")
        # Pixel counts depend only on the row index, not on the parity: half a
        # has sum_{r=0}^{s-1}(s-r) = s(s+1)/2 pixels, half b has s(s-1)/2.
        count_a = s * (s + 1) // 2
        count_b = s * (s - 1) // 2
        for parity_value in (0, 1):
            sel = par == parity_value
            if not sel.any():
                continue
            sx, sy = xs[sel], ys[sel]
            acc_a = np.zeros((sx.size, 3), dtype=np.float32)
            acc_b = np.zeros((sx.size, 3), dtype=np.float32)
            for r in range(s):
                row = sy + r
                if parity_value == 0:
                    a0, a1 = sx + r, sx + s        # a = {c >= r}
                    b0, b1 = sx, sx + r            # b = {c <  r}
                else:
                    a0, a1 = sx, sx + s - r        # a = {c + r <  s}
                    b0, b1 = sx + s - r, sx + s    # b = {c + r >= s}
                acc_a += P[row, a1] - P[row, a0]
                acc_b += P[row, b1] - P[row, b0]
            out[2 * idx[sel]] = acc_a / count_a
            if count_b == 0:
                # size 1: the b half is empty; report the cell pixel (same as
                # the offset-gather method's empty-mask fallback).
                out[2 * idx[sel] + 1] = img[sy, sx]
            else:
                out[2 * idx[sel] + 1] = acc_b / count_b
    return out


def compare_with_masks(img, leaves):
    """Diff `means_by_rows` against `brick_means.means_by_masks`.

    Function: the equality this module claims is bit-identity, which is exactly
    the kind of claim that silently rots, so it is stated as a runnable check.

    Shape/semantics: both inputs as for `means_by_rows`; returns a flat dict --
    `identical` (bool, `np.array_equal` of the two (T,3) float32 arrays),
    `max_absdiff` (float, 0.0 when identical) and `n_diff` (int, elements that
    differ at all). A non-zero result means the row-run geometry no longer
    matches the mask geometry, not a rounding tolerance issue.
    """
    from brick_means import means_by_masks
    a = means_by_masks(img, leaves)
    b = means_by_rows(img, leaves)
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"identical": bool(np.array_equal(a, b)),
            "max_absdiff": float(d.max()),
            "n_diff": int(np.count_nonzero(d))}
