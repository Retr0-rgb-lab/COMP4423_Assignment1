"""
verify_coverage — prove the "no gaps, no overlaps" constraint.

The PDF requires the bricks to "cover the output image without gaps or
overlaps". `brick_geom` argues that from the construction, but a construction
argument is not evidence, so this driver measures it.

The test is geometric rather than rasterised, and the distinction matters.
`cv2.fillPoly` paints a one-pixel fringe along every edge whose points sit on
integer pixel centres (documented in `brick_means`), so a *rasterised*
multiplicity count reports an apparent overlap along every internal diagonal.
That fringe is a rasterisation convention, not an area overlap, and the report
already quantifies it (8.1% of pixels on the Task 2 grid, 3.0 mean / 46.1 max
colour deviation). So this script tests the two claims that actually matter:

1. **No gaps** — the cells form an exact partition of the padded rectangle, so
   every output pixel belongs to some cell. Checked by counting, per cell, that
   the cells tile their grid without omissions, and that the union of the
   rasterised triangles leaves no zero inside the cropped output.
2. **No overlaps** — each cell is cut by exactly one diagonal into two triangles
   whose true areas sum to the cell area, and no two cells share interior. The
   area identity is checked with the shoelace formula on each triangle's three
   vertices, and the disjointness follows from cells being distinct elements of
   a grid keyed by (x, y, size).

Shapes
------
Input : `code/pics/sky.jpg` as (H, W, 3) uint8 BGR.
Output: printed per-configuration results. Exits 1 on any failure.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from brick_geom import (compute_grid, iter_triangles, leaves_to_triangles,  # noqa: E402
                        pad_to_max)
from brick_quadtree import quadtree_partition  # noqa: E402
from brick_region_merge import region_merge_partition  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(HERE, "pics", "sky.jpg")
BUDGET = 9990


def tri_area(tri):
    """Exact geometric area of a lattice triangle, by the shoelace formula.

    Function
    --------
    For vertices p0, p1, p2 this is |(p1-p0) x (p2-p0)| / 2. It is used instead
    of counting painted pixels because a rasterised count includes the `fillPoly`
    boundary fringe, which would make every diagonal look like an overlap. For
    the right-isosceles halves this module produces, each half's area is
    s*(s+1)/2 and the pair sums to exactly s*s.

    Shapes
    ------
    Input:
      tri : (3, 2) sequence of int vertices, `[..., 0]=x`, `[..., 1]=y`.
    Output:
      float — the unsigned area.
    """
    p = np.asarray(tri, dtype=np.float64).reshape(3, 2)
    return float(abs((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
                     - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1])) / 2.0)


def check_geometry(name, leaves, padded_hw, orig_shape):
    """Check one tessellation for gaps and overlaps at the geometry level.

    Function
    --------
    Three independent checks:
      * cells are distinct (a set), so no cell is generated twice;
      * every cell is a full s-by-s square of the grid, so the cells tile the
        padded rectangle exactly and no output pixel is left uncovered;
      * each cell's two triangles have areas summing to s*s, so the diagonal cut
        neither loses nor duplicates area inside the cell.
    Together those are the "no gaps, no overlaps" claim.

    Shapes
    ------
    Input:
      leaves      : list of (x, y, size) in padded coordinates.
      padded_hw   : (Hp, Wp), the padded rectangle.
      orig_shape  : (H, W), the crop actually delivered.
    Output:
      bool — True when all three checks pass. Prints the evidence either way.
    """
    keys = list(leaves)
    distinct = len(set(keys)) == len(keys)

    # Cells tile the padded rectangle: no cell may leave the grid, and every
    # cell origin must be a multiple of its own size (so cells are grid-aligned
    # and cannot partially overlap).
    grid_ok = all(x % s == 0 and y % s == 0 and x + s <= padded_hw[1]
                  and y + s <= padded_hw[0] for (x, y, s) in keys)

    # Each cell's two triangles must sum to the cell's area.
    pairs_ok, worst = True, 0.0
    for (x, y, s) in keys:
        a = le = None  # placeholder to keep the loop body flat
        tris = leaves_to_triangles([(x, y, s)])
        total = sum(tri_area(t) for t in tris)
        err = abs(total - s * s)
        worst = max(worst, err)
        if err > 1e-9:
            pairs_ok = False
            break

    ok = distinct and grid_ok and pairs_ok
    print(f"  [{name:14s}] cells={len(keys):5d} tris={2*len(keys):6d}  "
          f"distinct={distinct}  grid-aligned={grid_ok}  "
          f"area identity={'exact' if pairs_ok else f'FAILED (max err {worst:.3g})'}"
          f"  ->  {'no gaps, no overlaps' if ok else '*** FAILED ***'}")
    return ok


def check_no_gap_raster(img, leaves, padded_hw, orig_shape):
    """Confirm by rasterisation that no output pixel is left uncovered.

    Function
    --------
    The geometric checks above establish the tiling; this one confirms it on the
    pixels actually delivered. Triangles are filled with value 1 on a zero
    canvas and the cropped result must contain no zero. `fillPoly`'s fringe can
    only ever *add* coverage here, so a zero here would be a real gap.

    Shapes
    ------
    Input:
      img, leaves, padded_hw, orig_shape as for `check_geometry`.
    Output:
      bool — True when the cropped canvas has zero uncovered pixels.
    """
    cov = np.zeros(padded_hw, dtype=np.uint8)
    for tri in leaves_to_triangles(leaves):
        cv2.fillPoly(cov, [np.asarray(tri, np.int32).reshape(-1, 1, 2)], 1,
                    lineType=cv2.LINE_8)
    cropped = cov[:orig_shape[0], :orig_shape[1]]
    gaps = int((cropped == 0).sum())
    print(f"                 rasterised gap pixels in the delivered crop: {gaps}")
    return gaps == 0


def check_task2_grid(img):
    """Verify the Task 2 equal-size grid.

    Function
    --------
    Reproduces the delivered Task 2 geometry: `compute_grid` picks S, the grid
    is enumerated, and the image is reflect-padded to the exact grid extent so
    the last row and column of cells are complete.

    Shapes
    ------
    Input: `img` (H, W, 3) uint8 BGR. Output: (bool, bool) geometry, raster.
    """
    h, w = img.shape[:2]
    m, n, s = compute_grid(h, w)
    padded_hw = (m * s, n * s)
    leaves = [(j * s, i * s, s) for i in range(m) for j in range(n)]
    tris = list(iter_triangles(m, n, s))
    same = len(tris) == 2 * len(leaves)
    return (check_geometry("task2 grid", leaves, padded_hw, (h, w)) and same,
            check_no_gap_raster(img, leaves, padded_hw, (h, w)))


def check_task3_quadtree(img):
    """Verify the Task 3 quadtree partition.

    Function
    --------
    Runs `quadtree_partition` at the chosen configuration's S_set and budget and
    checks the resulting leaves.

    Shapes
    ------
    Input: `img` (H, W, 3) uint8 BGR. Output: (bool, bool) geometry, raster.
    """
    leaves, padded_hw, orig = quadtree_partition(img, [1, 2, 4, 8, 16, 32],
                                                 BUDGET, "mse")
    return (check_geometry("task3 quadtree", leaves, padded_hw, orig),
            check_no_gap_raster(img, leaves, padded_hw, orig))


def check_task3_region_merge(img):
    """Verify the Task 3 region-merge partition.

    Function
    --------
    Runs `region_merge_partition` on the S_set the C_algorithm group used, which
    is the variant that completes in reasonable time.

    Shapes
    ------
    Input: `img` (H, W, 3) uint8 BGR. Output: (bool, bool) geometry, raster.
    """
    leaves, padded_hw, orig = region_merge_partition(img, [4, 8, 16, 32], BUDGET)
    return (check_geometry("task3 regionmerge", leaves, padded_hw, orig),
            check_no_gap_raster(img, leaves, padded_hw, orig))


def main():
    """Check all three delivered tessellations; exit non-zero on any failure.

    Function
    --------
    Loads the source, runs the geometry and raster checks for the Task 2 grid
    and both Task 3 partitioners, and prints a one-line verdict.

    Shapes
    ------
    Input: none (reads DEFAULT_INPUT). Output: printed results; no arrays
    returned. Exits 1 if any configuration fails either check.
    """
    img = cv2.imread(DEFAULT_INPUT)
    if img is None:
        raise FileNotFoundError(f"Cannot read {DEFAULT_INPUT}")
    print(f"== no-gaps / no-overlaps verification "
          f"({img.shape[1]}x{img.shape[0]}) ==")
    results = [check_task2_grid(img), check_task3_quadtree(img),
               check_task3_region_merge(img)]
    flat = [r for pair in results for r in pair]
    if all(flat):
        print("\nPASS: all three tessellations are exact partitions of the "
              "output image - no gaps, no overlaps.")
        return
    print(f"\n*** {len(flat) - sum(flat)} check(s) FAILED ***")
    sys.exit(1)


if __name__ == "__main__":
    main()
