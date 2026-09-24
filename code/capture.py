"""
capture — Task 1: capture an image from the camera and display it (5 marks).

PDF requirement: "Call the camera to capture images and display them."

The script opens the default camera with `cv2.VideoCapture`, grabs one frame,
shows it, and waits for a key before closing. If no camera can be opened it
falls back to reading a file (`pics/SKY.png` by default) so the display path
can still be demonstrated on a machine without a webcam.

Run on the real desktop, not headless:

    ..\\..\\venv\\Scripts\\python code\\capture.py
    ..\\..\\venv\\Scripts\\python code\\capture.py --from-file code\\pics\\sky.jpg
"""
import argparse
import os

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FILE = os.path.join(HERE, "pics", "sky.jpg")


def grab_from_camera(camera_index=0):
    """Open the camera, read one frame, and return it.

    Function
    --------
    Opens `cv2.VideoCapture(camera_index)`, checks that it actually opened,
    then reads a single frame. Releasing the capture in a `finally` block
    matters on Windows: leaving the handle open makes the next `VideoCapture`
    in `camera_app.py` fail until the process exits.

    Shapes
    ------
    Input:
      camera_index : int — camera device index, 0 is the default webcam.
    Output:
      frame : (H, W, 3) uint8 BGR, or None when no camera is available or the
              first read fails. Semantics: `[y, x]` indexing with BGR channel
              order, the same layout every other module in this repo expects.
    """
    cap = cv2.VideoCapture(camera_index)
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame
    finally:
        cap.release()


def show_and_wait(frame, title="Task 1"):
    """Display `frame` in a window and block until any key is pressed.

    Function
    --------
    Thin wrapper around `cv2.imshow` / `cv2.waitKey(0)` /
    `cv2.destroyAllWindows` so both the camera and file paths exit the same way.

    Shapes
    ------
    Input:
      frame : (H, W, 3) uint8 BGR, the image to show.
      title : str — window title.
    Output: None. Blocks on `waitKey(0)`, which requires the window to have
    focus; on a headless host it appears to hang rather than fail.
    """
    cv2.imshow(title, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """Capture (or read) one image, print its shape, and display it.

    Function
    --------
    Default path is the camera. `--from-file PATH` forces the file path, and
    `--file` lets the file act as a fallback when no camera opens. Either way
    the chosen source is printed before the window opens so the log records
    which path was exercised.

    Shapes
    ------
    Input: CLI arguments only, via argparse (`--camera-index`, `--from-file`,
    `--file`). Output: one (H, W, 3) uint8 BGR frame handed to `show_and_wait`;
    no array is returned. Semantics: `frame[y, x, c]` is a BGR channel value.

    Raises
    ------
    FileNotFoundError — only when no camera is available AND the fallback file
    cannot be read. A missing file and an unreadable file are indistinguishable
    at the `cv2.imread` API, hence the single error.
    """
    ap = argparse.ArgumentParser(description="Task 1: capture and display an image")
    ap.add_argument("--camera-index", type=int, default=0,
                    help="camera device index (default: 0)")
    ap.add_argument("--from-file", default=None,
                    help="read this file instead of opening a camera")
    ap.add_argument("--file", default=DEFAULT_FILE,
                    help="fallback file when no camera is available "
                         f"(default: {DEFAULT_FILE})")
    args = ap.parse_args()

    frame = None
    source = None

    if args.from_file is None:
        frame = grab_from_camera(args.camera_index)
        if frame is not None:
            source = f"camera[{args.camera_index}]"
        else:
            print(f"[task1] camera[{args.camera_index}] unavailable, "
                  f"falling back to file")

    if frame is None:
        path = args.from_file or args.file
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        source = path

    print(f"[task1] source = {source}  shape = {frame.shape}  dtype = {frame.dtype}")
    show_and_wait(frame)


if __name__ == "__main__":
    main()
