"""
task4_l2_contact_sheet — build the Task 4 Level-2 scene contact sheet.

Task 4 requires testing on images of different content. Seven camera snapshots
live in `code/pics/task4/snap_000.png` .. `snap_006.png` (face, keyboard,
low-light lamp, backlit lamp, backlit curtain, wall). The report needs them
side by side so the "different content" claim is visible in one figure.

This driver tiles the seven existing PNGs into a single labelled sheet and
writes it to `code/pics/task4/L2_contact_sheet.png`. It composes files that
already exist; it does not re-run the pipeline or re-capture anything.

Shapes
------
Input: seven (H, W, 3) uint8 BGR snapshots, all expected to be the same size.
Output: a (rows*(H+CAP) + (rows+1)*GAP, cols*W + (cols+1)*GAP, 3) uint8 BGR
sheet, where rows/cols follow a 3-column layout for seven images.
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SNAPS = [os.path.join(HERE, "pics", "task4", f"snap_{i:03d}.png") for i in range(7)]
OUT = os.path.join(HERE, "pics", "task4", "L2_contact_sheet.png")
LABELS = ["face, cool indoor light", "face, second exposure",
          "keyboard, strong texture", "low-light desk lamp",
          "backlit lamp, high dynamic range", "curtain against window light",
          "wall + switch, near-overexposed"]
GAP, CAP, COLS = 12, 34, 3


def main():
    """Tile the seven snapshots into a labelled contact sheet and write it out.

    Function
    --------
    Reads each snapshot, scales any that differ in size to the first one's
    shape, places them in a 3-column grid with a caption strip under each, and
    writes the composite. Every tile is one `snap_XXX.png` captured on the real
    camera, so the sheet is the visual record of the Level-2 test.

    Shapes
    ------
    Input: seven image files (H, W, 3) uint8 BGR. Output: one composite image
    written to `OUT`; nothing is returned.
    """
    imgs = []
    for path in SNAPS:
        im = cv2.imread(path)
        if im is None:
            raise FileNotFoundError(f"missing snapshot: {path}")
        imgs.append(im)
    H, W = imgs[0].shape[:2]
    imgs = [im if im.shape[:2] == (H, W) else
            cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            for im in imgs]

    rows = (len(imgs) + COLS - 1) // COLS
    cell_h = H + CAP
    sheet = np.full((rows * cell_h + (rows + 1) * GAP,
                     COLS * W + (COLS + 1) * GAP, 3), 32, dtype=np.uint8)

    for idx, im in enumerate(imgs):
        r, c = divmod(idx, COLS)
        y0 = GAP + r * (cell_h + GAP)
        x0 = GAP + c * (W + GAP)
        sheet[y0:y0 + H, x0:x0 + W] = im
        label = LABELS[idx] if idx < len(LABELS) else f"snap_{idx:03d}"
        cv2.putText(sheet, label, (x0 + 6, y0 + H + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                    cv2.LINE_AA)

    cv2.imwrite(OUT, sheet)
    print(f"[l2_sheet] {OUT}  ({sheet.shape[1]}x{sheet.shape[0]}, "
          f"{len(imgs)} scenes)")


if __name__ == "__main__":
    main()
