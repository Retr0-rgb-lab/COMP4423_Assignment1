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
