"""
task3_best_vs_task2 — the "ultimate 2x2" comparison figure for the report.

Composes four panels into one grid (AGENTS section 1.1, summary/best_vs_task2.png):

    original        | Task 2 (equal-size, 3 colours)
    -----------------|--------------------------------
    Task 3 best      | (empty / reused Task 3 best)

Purpose: the report's section 3 vs section 4 argument — adaptive multi-size
Task 3 preserves structure that the uniform Task 2 grid cannot, at the cost of
a little pixel-level fidelity. The figure is a pure composition of already
rendered PNGs (no re-partition, no new metrics), so it is deterministic.

Shapes
------
Inputs (all expected to be the same (H, W, 3) uint8 BGR after task2/task3 crop):
  sky.jpg                    -- the original test image
  pics/task2/out_task2.png   -- Task 2 render (post black-band fix)
  pics/task3/best/best_config.png -- Task 3 chosen config render
Output:
  (2*(H+cap) + 3*gap, 2*W + 3*gap, 3) uint8 BGR, saved to
  pics/task3/summary/best_vs_task2.png.
"""
import os

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(_HERE, "pics", "sky.jpg")
T2 = os.path.join(_HERE, "pics", "task2", "out_task2.png")
T3 = os.path.join(_HERE, "pics", "task3", "best", "best_config.png")
OUT = os.path.join(_HERE, "pics", "task3", "summary", "best_vs_task2.png")
GAP, CAP = 16, 44


def _panel(img, label):
    """Stack `img` above a caption strip; return the (H+CAP, W, 3) panel.

    Shape/semantics: input (H, W, 3) uint8 BGR; output (H+CAP, W, 3) with
    white caption text centred in the bottom CAP-px strip.
    """
    H, W = img.shape[:2]
    panel = np.full((H + CAP, W, 3), 20, dtype=np.uint8)
    panel[:H] = img
    cv2.putText(panel, label, (W // 2 - 180, H + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return panel


def main():
    """Compose the 2x2 and write summary/best_vs_task2.png.

    Function: load the three existing renders (all cropped to the original
    (1279, 1706)), caption each, and lay them out 2x2 with a gap. The bottom-
    right panel reuses the Task 3 best panel so the grid stays balanced and
    the file is produced without re-running any partition/quantise.

    Shapes: panels are (H+CAP, W, 3); the grid is
    (2*(H+CAP) + 3*GAP, 2*W + 3*GAP, 3) uint8 BGR.
    """
    orig = cv2.imread(ORIG)
    t2 = cv2.imread(T2)
    t3 = cv2.imread(T3)
    for name, arr in (("original", orig), ("task2", t2), ("task3", t3)):
        if arr is None:
            raise FileNotFoundError(f"missing input for best_vs_task2: {name}")
    if not (orig.shape == t2.shape == t3.shape):
        raise ValueError(f"size mismatch: orig {orig.shape}, t2 {t2.shape}, t3 {t3.shape}")

    tl = _panel(orig, "original")
    tr = _panel(t2, "Task 2: uniform grid, 3 colours")
    bl = _panel(t3, "Task 3: adaptive quadtree, K=16")
    br = _panel(t3, "Task 3 best (repeat)")

    pH, pW = tl.shape[:2]
    grid = np.full((2 * pH + 3 * GAP, 2 * pW + 3 * GAP, 3), 40, dtype=np.uint8)
    grid[GAP:GAP + pH, GAP:GAP + pW] = tl
    grid[GAP:GAP + pH, 2 * GAP + pW:2 * GAP + 2 * pW] = tr
    grid[2 * GAP + pH:2 * GAP + 2 * pH, GAP:GAP + pW] = bl
    grid[2 * GAP + pH:2 * GAP + 2 * pH, 2 * GAP + pW:2 * GAP + 2 * pW] = br

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, grid)
    print(f"[best_vs_task2] wrote {OUT}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()
