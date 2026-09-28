# COMP4423 Assignment 1 -- triangle-brick mosaic (Tasks 1-4)

Convert an image (and a live camera) into a mosaic of right-isosceles
triangles, under a 10000-triangle budget, with per-task constraints (3 colours
for Task 2, multiple sizes + >3 colours for Task 3, real-time for Task 4).

The submitted report is `Assignment1_Report_24101322D_HUANG_Haoran.pdf` at the
repository root; it documents the design decisions, measurements and known
limitations.

## Environment

- Python 3.12 on Windows (the camera and GUI need the real desktop; WSL has no
  display, so headless commands below use `--no-show` / `--input`).
- The virtual environment is at `../../venv` relative to this directory (i.e.
  `learn_torch/venv`), NOT inside the repo.
- Install / refresh dependencies:

```powershell
..\..\venv\Scripts\python -m pip install -r requirements.txt
```

## Layout

```
code/            all source (engine modules brick_*.py + per-task drivers)
code/pics/       input image (sky.jpg); output renders are written here
```

## Run

```powershell
# Task 1 -- capture / read an image and display it
..\..\venv\Scripts\python code\capture.py

# Task 2 -- equal-size triangles, 3 colours
..\..\venv\Scripts\python code\triangle_brick.py --input code\pics\sky.jpg --output code\pics\task2\out_task2.png

# Task 3 -- adaptive multi-size + multi-colour (writes code/pics/task3/)
..\..\venv\Scripts\python code\triangle_brick_task3.py --groups A B C D E

# Task 4 -- live camera -> triangle-brick display (needs the desktop + camera)
..\..\venv\Scripts\python code\camera_app.py
..\..\venv\Scripts\python code\camera_app.py --lock-ae        # lock exposure (recommended)

# Task 4 -- headless benchmark / verification (works without a camera)
..\..\venv\Scripts\python code\camera_app.py --no-show --max-frames 30 --input code\pics\sky.jpg
..\..\venv\Scripts\python code\task4_verify.py
```

Task 4 keys: `q`/ESC quit, `s` snapshot, `p` pause, `r` reset.
Task 4 presets: `--preset quality|balanced|fast`; brick budget `--budget N`.

## Notes

- Paths you PASS to a driver resolve against the current directory; the built-in
  defaults are `__file__`-relative.
- Outputs under `code/pics/` (for example `task2/out_task2.png`) are produced by
  the commands above at run time; they are not stored in the repository.
