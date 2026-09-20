"""
Task 1: Capture / read an image and display it (5 marks).

PDF requirement: "Call the camera to capture images and display them."
Current implementation: file input via sky.jpg (Task 4 will swap this for cv2.VideoCapture).
"""
import cv2
import sys
import os

# Default input: sky.jpg in the same directory as this script (cwd-independent).
INPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pics", "sky.jpg")


def main():
    """Read one image and display it in a window; any key closes it.

    Shape: the loaded image is (H, W, 3) uint8 BGR and is passed straight to
    `cv2.imshow`; no array is returned. Semantics: the optional first CLI
    argument is the image path, defaulting to `pics/sky.jpg` next to this file.
    Raises `FileNotFoundError` when `cv2.imread` returns None (a missing file and
    an unreadable file are indistinguishable at this API, hence the single
    error). `cv2.waitKey(0)` blocks forever until a key is pressed WHILE THE
    WINDOW HAS FOCUS -- on a headless host (WSL, CI) or if the window opens
    behind another, this appears to hang rather than fail.
    """
    path = sys.argv[1] if len(sys.argv) > 1 else INPUT
    img = cv2.imread(path)                       # BGR ndarray
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    print(f"input = {path}  shape = {img.shape}  dtype = {img.dtype}")
    cv2.imshow("Task 1", img)                    # pop up window
    cv2.waitKey(0)                               # 0 = wait for any key
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
