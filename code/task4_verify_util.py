"""
task4_verify_util -- shared helpers for the Task 4 verification harness.

Split out of `task4_verify.py` to keep that file under the 400-line limit
(AGENTS 4.1): these four are generic test utilities (a synthetic frame source,
a quality metric pair, a paired timer, a perceptual churn measure) with no
knowledge of any particular optimization, so they belong to no single check.

All functions are deterministic and headless -- no camera, no window.
"""
import time

import numpy as np
from skimage.metrics import peak_signal_noise_ratio

from brick_metrics import delta_e_2000_full


def synthetic_frame(t, h=480, w=640, seed=7):
    """One synthetic 640x480 camera-like frame at time step t.

    Function: deterministic moving content (diagonal gradient + a drifting
    disc + a fixed high-contrast block + light noise) so the partition, the
    K-Means palette and the quantizer all see real variation frame to frame,
    while identical t always yields the identical image (flicker test needs
    that). Replaces the camera for headless verification.

    Shapes: returns (h, w, 3) uint8 BGR. Semantics: `t` shifts the disc centre
    along x; `seed` fixes the noise so all runs see the same "scene".
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.empty((h, w, 3), dtype=np.float32)
    base[..., 0] = 200 - 0.3 * xx + 0.1 * t          # B: falls left->right
    base[..., 1] = 60 + 0.4 * yy                     # G: rises top->bottom
    base[..., 2] = 40 + 0.25 * xx                    # R
    cx, cy, r = 100 + (t * 9) % (w - 200), h // 2, 90
    disc = (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
    base[disc] = (40, 200, 220)                      # warm disc, BGR
    base[h // 5:h // 5 + 80, w // 5:w // 5 + 120] = (20, 20, 230)  # red block
    noise = rng.normal(0, 4, (h, w, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def quality_pair(src, canvas):
    """(mean Delta-E2000, PSNR dB) of a render against its source frame.

    Shape: both inputs (H, W, 3) uint8 BGR; two floats out. This is the same
    metric pair Task 4 uses for its before/after quality column, so the
    numbers quoted here drop straight into that table.
    """
    de, _ = delta_e_2000_full(src, canvas)
    psnr = float(peak_signal_noise_ratio(src, canvas, data_range=255))
    return de, psnr


def best_of(fn, reps):
    """Median-of-reps wall time in ms for a zero-arg callable (paired timing).

    Semantics: returns (best_ms, median_ms). Best is the headline for stage
    benchmarks (matches the Task 4 convention of quoting best-of-N for paired
    in-process runs); median is printed alongside so a bimodal cost cannot
    hide.
    """
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def churn(a, b, thresh=8):
    """Fraction of pixels whose largest channel change exceeds `thresh`.

    Function: frame-to-frame difference that ignores sub-perceptual changes.
    Needed because a warm-started palette still moves 1-2/255 at a rebuild
    frame, which flips "any pixel differs" on most of the frame while being
    invisible; a threshold of 8 grey levels measures what a viewer sees.

    Shape/semantics: two (H,W,3) uint8 BGR frames in; one float in [0,1] out,
    the share of pixels with max-channel |delta| > `thresh`.
    """
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    return float((d > thresh).mean())
