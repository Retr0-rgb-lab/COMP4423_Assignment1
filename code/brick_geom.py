"""
brick_geom — geometry engine for the triangle-brick assignment.

External API (Task 2 + Task 3):

  * Uniform grid (Task 2): `compute_grid` + `iter_triangles`.
  * Padding helper (shared): `pad_to_max`.
  * Adaptive leaf -> triangle (Task 3): `leaves_to_triangles`.

Adaptive partitioning algorithms live in:
  - `brick_quadtree`    : top-down RDO split.
  - `brick_region_merge`: bottom-up aligned multi-scale merge.

Everything here is pure geometry; no colour, rendering, metrics, or I/O.

A "leaf cell" throughout this module is the tuple `(x, y, size)` where:
  x, y   : integer top-left of a square in the PADDED image coordinates.
  size   : integer side length in pixels (always a power of 2 in
           {S_min, 2*S_min, ..., S_max} when used with quadtree/region-merge).

A "triangle" returned to the renderer is a `(3, 2)` int32 ndarray of three
OpenCV-style points: `[..., 0]=x (column)`, `[..., 1]=y (row)`.

Diagonal convention (single source of truth for the whole repo)
---------------------------------------------------------------
Every square cell is cut by exactly ONE diagonal into two triangles, so the
two triangles of a cell always share an edge. The direction of that diagonal
alternates on the checkerboard parity of the cell index:

    parity 0  ->  diagonal TL -> BR   (screen slope +1)
    parity 1  ->  diagonal TR -> BL   (screen slope -1)

Two consequences that are easy to get wrong when reading the 2-triangle
list, so they are stated once here instead of in each function:

  * Neighbouring cells -- horizontally or vertically adjacent -- get
    OPPOSITE directions, because their parity flips. The shared vertices
    make the diagonals chain up into a chevron / herringbone texture. They
    do NOT form uniform stripes.
  * The alternation is deliberate: one global diagonal direction would make
    every boundary line parallel and bias the mosaic toward a single
    diagonal, which is visible as a directional smear on diagonal edges.

Rationale for alternating is a design choice, not a requirement of the PDF;
the PDF only fixes "right-angled isosceles" and "no gaps or overlaps".
"""
import cv2
import numpy as np


MAX_TRIANGLES = 10000


# ---------------------------------------------------------------------------
# Uniform grid (Task 2)
# ---------------------------------------------------------------------------

def compute_grid(H, W, max_triangles=MAX_TRIANGLES):
    """Find the smallest S that tiles an H x W image under the triangle budget.

    Function
    --------
    Scan S from 1 upward; for each S compute M = H // S, N = W // S cells per
    row/column and 2*M*N triangles (each cell splits into 2). Return the
    smallest S that keeps the count <= max_triangles.

    Shapes
    ------
    Input:
      H, W : ints — image height and width in pixels.
    Output:
      (M, N, S) — M = ceil(H/S)-like cell count, N = ceil(W/S)-like cell
                  count, S = the cell side length that fits the budget.
                  `M, N >= 1` so a partial trailing row/column is dropped.
    """
    for S in range(1, min(H, W) + 1):
        M, N = H // S, W // S
        if M >= 1 and N >= 1 and 2 * M * N <= max_triangles:
            return M, N, S
    raise ValueError(f"Image {H}x{W} cannot be tiled under {max_triangles} triangles.")


def iter_triangles(M, N, S):
    """Yield (3,2) int32 triangle vertex arrays for an M x N grid of S x S cells.

    Function
    --------
    Walk the M*N cell grid in row-major order; for each cell emit TWO
    right-isosceles triangles that share that cell's single diagonal. The
    diagonal direction alternates on (row + col) parity -- see the
    "Diagonal convention" note in the module docstring.

    Shapes
    ------
    Input:
      M, N : ints — cell-grid dimensions (from `compute_grid`).
      S    : int  — cell side length in pixels.
    Yield:
      tri  : (3, 2) int32 ndarray. `tri[k, 0]=x (col)`, `tri[k, 1]=y (row)`.
             Each cell yields its two triangles in a fixed order
             (`tl, tr, br` then `tl, br, bl`, or `tl, tr, bl` then
             `tr, br, bl`). The vertex order carries NO meaning downstream:
             `cv2.fillPoly` receives one contour per call and fills it for
             either winding. Keeping it fixed only makes the two lists
             trivially comparable when debugging. Per cell: two triangles,
             total 2*M*N yields.
    """
    for i in range(M):
        for j in range(N):
            tl = (j * S, i * S)
            tr = (j * S + S, i * S)
            bl = (j * S, i * S + S)
            br = (j * S + S, i * S + S)
            if (i + j) % 2 == 0:
                # diagonal goes TL -> BR
                yield np.array([tl, tr, br], dtype=np.int32)
                yield np.array([tl, br, bl], dtype=np.int32)
            else:
                # diagonal goes TR -> BL
                yield np.array([tl, tr, bl], dtype=np.int32)
                yield np.array([tr, br, bl], dtype=np.int32)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def pad_to_max(img, S_max):
    """Reflect-pad an image so H and W become multiples of S_max.

    Function
    --------
    Computes the minimum reflected padding needed on the bottom and right
    edges so the resulting image is exactly (kH * S_max, kW * S_max) for some
    integers kH, kW. Returns the padded image and the metadata needed to
    crop back later.

    Shapes
    ------
    Input:
      img   : (H, W, C) or (H, W) ndarray — original image.
      S_max : int — target stride; (H + pad_h) and (W + pad_w) become
              multiples of S_max.
    Output:
      padded : (Hp, Wp, C) or (Hp, Wp) ndarray — reflected-padded image.
      orig_shape : (H, W) — original size for later crop.
      (pad_h, pad_w) : ints — padding added on bottom and right (top/left
                       padding is 0). Needed only for debugging.
    """
    H, W = img.shape[:2]
    pad_h = (S_max - H % S_max) % S_max
    pad_w = (S_max - W % S_max) % S_max
    # BORDER_REFLECT mirrors pixels across the edge without repeating the
    # edge pixel — preferred for image data so we don't introduce a hard line.
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
    return padded, (H, W), (pad_h, pad_w)


# ---------------------------------------------------------------------------
# Adaptive: leaf cell -> triangle
# ---------------------------------------------------------------------------

def leaves_to_triangles(leaves):
    """Turn leaf cells (x, y, size) into right-isosceles triangles.

    Function
    --------
    For each leaf, split the square along ONE diagonal into TWO triangles.
    The direction alternates on the checkerboard parity of the cell index
    computed AT THAT CELL'S OWN SIZE:

        parity = (y // size + x // size) % 2
        parity 0 -> diagonal TL -> BR,   parity 1 -> diagonal TR -> BL

    Because the index is taken per size, two cells at the SAME location but
    with DIFFERENT sizes do not necessarily get the same direction (e.g.
    cell (x=4, y=0) is parity 1 at size 4 but parity 0 at size 2). Mixed-size
    neighbours therefore meet at arbitrary angles. See the "Diagonal
    convention" note in the module docstring for why we alternate at all.

    Shapes
    ------
    Input:
      leaves : list of (x, y, size) tuples — cell top-left + side length
               in PADDED image coordinates.
    Output:
      triangles : list of (3, 2) int32 ndarrays — 2 per leaf. `tri[k, 0]=x`
                  (column), `tri[k, 1]=y` (row), in OpenCV point order.
                  Total length = 2 * len(leaves).
    """
    triangles = []
    for (x, y, size) in leaves:
        # Grid indices at this size, used to decide diagonal direction.
        i, j = y // size, x // size
        tl, tr = (x, y), (x + size, y)
        bl, br = (x, y + size), (x + size, y + size)
        if (i + j) % 2 == 0:
            # diagonal goes TL -> BR
            triangles.append(np.array([tl, tr, br], dtype=np.int32))
            triangles.append(np.array([tl, br, bl], dtype=np.int32))
        else:
            # diagonal goes TR -> BL
            triangles.append(np.array([tl, tr, bl], dtype=np.int32))
            triangles.append(np.array([tr, br, bl], dtype=np.int32))
    return triangles


# Vertex offsets relative to a cell's top-left corner, per parity and half,
# mirroring the two branches of `leaves_to_triangles` exactly. Entry [v, 0] is
# the x offset of vertex v, [v, 1] the y offset. Kept module-level so the two
# halves of `leaves_to_triangles_array` read as the same geometry, unrolled.
_OFF = {
    0: (np.array([[0, 0], [1, 0], [1, 1]], dtype=np.int64),   # half a, diag TL->BR
        np.array([[0, 0], [1, 1], [0, 1]], dtype=np.int64)),  # half b
    1: (np.array([[0, 0], [1, 0], [0, 1]], dtype=np.int64),   # half a, diag TR->BL
        np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int64)),  # half b
}


def leaves_to_triangles_array(leaves):
    """Vectorised `leaves_to_triangles`: same geometry/order, one (T, 3, 2) array.

    Function
    --------
    Task 4 optimization D1. The per-triangle loop builds 9990 small ndarrays
    per frame (~13 ms); bucketing the leaves by (size, parity) and adding
    broadcast offset blocks builds ALL triangles in a handful of numpy ops per
    bucket instead. Diagonal convention, per-cell parity, half ordering and
    vertex order are exactly those of `leaves_to_triangles` — row 2k / 2k+1 is
    still leaf k's half a / half b, so the ordering contract with
    `brick_means.extract_means` and `brick_render.render_triangles_batched`
    (which consumes this array row-wise) is unchanged.

    Shapes
    ------
    Input:  `leaves` — list of (x, y, size), padded coordinates, any order.
    Intermediate: per (size, parity) bucket with n cells, `orig` is (n, 1, 2)
    origins and `off` a (3, 2) offset block; the sum broadcasts to (n, 3, 2).
    Output: (2*len(leaves), 3, 2) int32 — `out[t, v, 0]=x (column)`,
            `out[t, v, 1]=y (row)` for vertex v of triangle t, same content as
            `np.stack(leaves_to_triangles(leaves))`.
    """
    leaves = list(leaves)
    out = np.empty((2 * len(leaves), 3, 2), dtype=np.int32)
    buckets = {}
    for k, (x, y, s) in enumerate(leaves):
        par = (y // s + x // s) % 2
        buckets.setdefault((s, par), []).append(k)
    for (s, par), idx in buckets.items():
        idx = np.asarray(idx, dtype=np.int64)
        # origins[i] = (x, y) of leaf idx[i], shape (n, 1, 2) for broadcasting
        origins = np.array([[leaves[k][0], leaves[k][1]] for k in idx],
                           dtype=np.int64)[:, None, :]
        off_a, off_b = _OFF[par]
        out[2 * idx] = origins + off_a * s
        out[2 * idx + 1] = origins + off_b * s
    return out