# Task 1: Capture / read an image and display it (5 marks)

## Goal (from assignment PDF)
> Task 1: Call the camera to capture images and display them. (5 marks)

## Key decisions

- **File input as camera placeholder**: default input is `code/sky.jpg`. Task 4 will swap `cv2.imread` for `cv2.VideoCapture(0).read()`.
- **OpenCV** for IO and display: `cv2.imread` (BGR), `cv2.imshow`, `cv2.waitKey(0)` (close on any key).
- **Default path resolved via `__file__`** so the script is cwd-independent. First CLI argument overrides.
- **Error handling**: raise `FileNotFoundError` when `imread` returns `None`, instead of failing silently inside `imshow`.

## Prompt drafts

(GenAI not used for this task; prompt drafts will be recorded starting from Task 2.)

## Code changes

- New `code/capture.py` (~22 lines): read -> validate -> print shape -> show -> wait key -> exit.
- `code/sky.jpg`: Task 1 input image.

## Problems and solutions

- **WSL has no GUI**: `cv2.imshow` cannot pop a window in the current shell. Verification must run in a Windows terminal.
- **OpenCV availability**: the venv at `../../venv` (`learn_torch\venv`) already ships with `opencv-python 5.0.0.93`, no extra install needed. Note it is two levels up, not one: there is no `python-cv\venv`.

## Verification (run in Windows terminal)

```bash
cd "D:\Program Files\learn_torch\python-cv\Assignment1"
..\..\venv\Scripts\python code/capture.py                    # default (code/pics/sky.jpg)
..\..\venv\Scripts\python code/capture.py code/pics/sky.jpg   # explicit arg
```

Expected:
- Window pops up showing sky.jpg.
- Terminal prints `input = ... shape = (H, W, 3) dtype = uint8`.
- Any key closes the window and exits the program.

## Known limitations / TODOs

- No real camera yet (deferred to Task 4).
- No resize / grayscale / color conversion (not required by Task 1).
- Single image, no loop (Task 4 will add a `while` loop).
