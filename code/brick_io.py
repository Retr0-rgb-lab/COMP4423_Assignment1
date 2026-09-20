"""
brick_io -- file output and stage timing shared by the drivers.

Two small concerns that more than one driver needs and that no engine module
owns:

  * `save_png`   -- `cv2.imwrite` that creates the parent directory and RAISES
                    on failure, because OpenCV reports failure by return value
                    only (a missing directory silently writes nothing).
  * `StageTimer` -- per-stage wall-clock accumulation inside a per-frame loop,
                    so the live HUD, the headless benchmark and the Task 4
                    before/after table are all the SAME measurement.

Why a module: `save_png` was written for the Task 2 driver, and Task 4 needs it
too. Copying it would leave two versions of the "cv2 hides its failures" fix,
which is exactly the kind of duplication that lets one copy rot.
"""
import os
import time
from collections import defaultdict
from contextlib import contextmanager

import cv2


def save_png(path, img):
    """Write `img` to `path` as PNG, creating the parent directory first.

    Function
    --------
    Wraps `cv2.imwrite`: that call reports failure by RETURN VALUE, not by
    exception, so writing into a directory that does not exist returns False and
    prints nothing -- a run can look successful while producing no output.

    Shape: `img` is (H, W, 3) uint8 BGR (any size); `path` is a str. Semantics:
    nothing is transformed -- the array is encoded as-is, so a BGR array is
    written as a BGR PNG (correct for OpenCV, and why nothing here converts
    colour order).

    Raises:
      IOError -- when `cv2.imwrite` returns falsy: an unwritable directory, an
      unknown extension, or a dtype OpenCV cannot encode (e.g. float32).
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    if not cv2.imwrite(path, img):
        raise IOError(f"cv2.imwrite failed for {path}")


class StageTimer:
    """Accumulate per-stage durations for a repeated pipeline.

    Function
    --------
    A set of per-stage stopwatches for a per-frame loop. Usage:

        timer = StageTimer()
        with timer.stage("partition"):
            leaves = partition(...)
        with timer.stage("render"):
            canvas = render(...)
        print(timer.report()["partition"]["median_ms"])

    Shape / semantics
    -----------------
    `self.samples` is `dict[str, list[float]]` -- stage name -> list of elapsed
    SECONDS, one entry appended per completed `stage()` block, in call order. A
    stage entered but never exited (an exception mid-block) appends nothing, so
    a failing frame cannot poison the statistics with a bogus duration.
    `report()` returns `dict[str, dict]` with keys `n`, `mean_ms`, `median_ms`,
    `total_ms`, plus `"__total__"` for the summed median of every stage.

    Median is the headline statistic on purpose: over a few hundred frames a
    single GC pause, a camera hiccup or an OS scheduling spike dominates a mean,
    and a mean would then hide the real per-stage cost we are trying to compare.

    Not thread-safe: one timer per loop, which is how the camera app uses it.
    """

    def __init__(self):
        """Create an empty timer.

        Shape/semantics: `self.samples` starts as an empty
        `defaultdict(list)` -- semantically a map from stage NAME to the list of
        elapsed seconds recorded for it, so it is empty until a `stage()` block
        completes. The defaultdict is what lets callers register a new stage just
        by naming it in `stage()`, with no declaration step.
        """
        self.samples = defaultdict(list)

    @contextmanager
    def stage(self, name):
        """Time the enclosed block and record it under `name`.

        Shape/semantics: `name` is the str key; on block exit ONE float is
        appended to `self.samples[name]`, its value being the wall-clock seconds
        spent inside the block. Semantics caveat: the recording happens in a
        `finally`, so the duration is recorded even if the block raises -- the
        elapsed time of a FAILED frame is still counted. Entering the same
        `name` twice in one iteration therefore appends two samples for what the
        caller may consider one frame.
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.samples[name].append(time.perf_counter() - t0)

    def reset(self):
        """Drop all samples, keeping the stage names that were registered.

        Shape/semantics: `self.samples` becomes empty; it stays a
        `defaultdict(list)`, so previously seen stage names need no
        re-registration and `report()` on a freshly reset timer returns only the
        `__total__` row with zero counts.
        """
        self.samples.clear()

    @staticmethod
    def _median(values):
        """Median of a 1-D sequence of floats, without a numpy dependency.

        Shape: a 1-D sequence in, one float out. Semantics: for an even-length
        input the result is the MEAN of the two central values, so it need not be
        an element of the input; an empty input returns 0.0 instead of raising,
        which is what lets `report()` be called before any frame completes.
        """
        s = sorted(values)
        n = len(s)
        if n == 0:
            return 0.0
        mid = n // 2
        return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])

    def report(self):
        """Median/mean/total milliseconds per stage, plus a `__total__` row.

        Semantics: `out[name]["median_ms"]` is the median over the completed
        frames for that stage; `out["__total__"]["median_ms"]` sums the per-stage
        medians, i.e. a typical frame's cost broken down by stage. It is computed
        from medians rather than from per-frame totals, so it can differ slightly
        from a true median frame time when the slow frames are not the same frame
        for every stage.
        """
        out = {}
        for name, vals in self.samples.items():
            ms = [v * 1000.0 for v in vals]
            out[name] = {
                "n": len(ms),
                "mean_ms": sum(ms) / len(ms) if ms else 0.0,
                "median_ms": self._median(ms),
                "total_ms": sum(ms),
            }
        out["__total__"] = {
            "n": max((v["n"] for v in out.values()), default=0),
            "mean_ms": sum(v["mean_ms"] for v in out.values()),
            "median_ms": sum(v["median_ms"] for v in out.values()),
            "total_ms": sum(v["total_ms"] for v in out.values()),
        }
        return out
