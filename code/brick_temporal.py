"""
brick_temporal -- scene-stability state for the live loop (Task 4 opt B+C).

The measured jitter analysis (docs/progress/Task4.md, step 5) showed that even
on a still scene the mosaic changes because partition, means and quantize are
recomputed every frame from noisy camera pixels: near-tie split decisions flip
(~3-5% of cells/frame) and labels near palette boundaries flap (~3%/frame).

`TemporalState` provides the geometry half of the fix (opt B): detect that the
scene has not changed and reuse the previous partition, so no split decision is
re-made. It also carries the per-triangle label memory that the sticky
quantizer (opt C, `brick_color_rt.quantize_nearest_bgr_sticky`) needs, because
that memory is only valid while the triangle rows are stable -- i.e. while the
geometry is reused. A real scene change invalidates both.

This module holds ONLY the state and the decisions; `frame_pipeline.render_frame`
does the actual partition/triangle work so there is no import cycle.
"""
import cv2
import numpy as np

# Signature resolution for the scene-change test. 64x48 is a 10x10 box average
# of the 640x480 frame, which cuts sigma=2 sensor noise to ~0.2 grey levels
# while a real content change moves the signature by whole grey levels.
SIG_W, SIG_H = 64, 48


class TemporalState:
    """Cross-frame geometry/label state for the live loop.

    Function
    --------
    `can_reuse(frame)` decides whether this frame is close enough to the last
    partitioned frame to keep its cells. `commit(...)` records a new partition;
    `note_reuse()` ages a reused one. The label memory is exposed as
    `prev_labels` for the sticky quantizer.

    Shape/semantics
    ---------------
    `leaves` is the cached ordered list of (x, y, size) padded-coordinate cells;
    keeping a LIST (not a set) preserves triangle-row order, which is what makes
    `prev_labels` a valid per-triangle memory. `padded_shape`/`orig_shape` are
    the cached (Hp,Wp)/(H,W); `tri` is the cached (T,3,2) int32 triangle array.
    `sig` is the (SIG_H, SIG_W) float32 brightness-mean-subtracted signature of
    the partitioned frame. `prev_labels` is (T,) uint8 or None (None after a
    re-partition). `age` counts reused frames; `reused`/`refreshed`-style flags
    are read by the driver for the HUD.
    """

    def __init__(self, enabled=True, reuse_thresh=1.0, hysteresis=0.1,
                 max_reuse=0):
        """Create the state.

        `enabled=False` makes every query report "recompute", i.e. exactly the
        pre-temporal behaviour (used by the correctness gates). `reuse_thresh`
        is the mean absolute signature difference (0-255 grey levels) under
        which the partition is reused. `hysteresis` is forwarded to the sticky
        quantizer. `max_reuse > 0` forces a re-partition after that many reused
        frames even if the scene looks static (a guard against very slow drift
        that never crosses the threshold); 0 disables the guard.
        """
        self.enabled = enabled
        self.reuse_thresh = reuse_thresh
        self.hysteresis = hysteresis
        self.max_reuse = max_reuse
        self.reset()

    def reset(self):
        """Drop all cached state (camera change, window resize, user reset)."""
        self.leaves = None
        self.padded_shape = None
        self.orig_shape = None
        self.tri = None
        self.sig = None
        self.prev_labels = None
        self.age = 0
        self.reused = False
        # Render cache: the last canvas plus the EXACT inputs that drew it.
        # Reused only when labels AND palette are byte-identical (see
        # reuse_render), so it can never show a stale mosaic.
        self.render_labels = None
        self.render_palette = None
        self.canvas = None
        self.render_hits = 0

    @staticmethod
    def signature(frame):
        """Brightness-invariant scene signature of a frame.

        Function: box-downsample to (SIG_H, SIG_W), convert to grey, and
        subtract the spatial mean. Subtracting the mean is deliberate: the
        quadtree's split priorities depend on relative contrast, so a uniform
        auto-exposure shift should NOT count as a scene change. Uniform
        brightness drift therefore leaves the geometry alone while the palette
        (warm-started) tracks it.

        Shape/semantics: (H,W,3) uint8 BGR in; (SIG_H,SIG_W) float32 out, each
        entry a mean-subtracted average grey level of a ~10x10 block.
        """
        small = cv2.resize(frame, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return gray - gray.mean()

    def can_reuse(self, frame):
        """Is this frame close enough to the cached partition to reuse it?

        Shape/semantics: takes one (H,W,3) uint8 BGR frame; returns bool. False
        when disabled, when nothing is cached, when the reuse-age guard trips,
        or when the mean absolute signature difference exceeds
        `reuse_thresh`. Side-effect free.
        """
        if not self.enabled or self.leaves is None:
            return False
        if self.orig_shape != frame.shape[:2]:
            # Frame geometry changed (camera resolution switch, resize): the
            # cached cells are in the old coordinate system, so they must not
            # be reused even if the signature happens to look similar.
            return False
        if self.max_reuse and self.age >= self.max_reuse:
            return False
        diff = float(np.abs(self.signature(frame) - self.sig).mean())
        return diff <= self.reuse_thresh

    def commit(self, leaves, padded_shape, orig_shape, tri, frame):
        """Record a freshly computed partition and invalidate the label memory.

        `prev_labels` is cleared because triangle rows now describe new cells;
        keeping stale labels would apply hysteresis to the wrong triangles. The
        render cache is cleared for the same reason: the cached canvas belongs
        to the old triangle set.
        """
        self.leaves = leaves
        self.padded_shape = padded_shape
        self.orig_shape = orig_shape
        self.tri = tri
        self.sig = self.signature(frame)
        self.prev_labels = None
        self.render_labels = None
        self.render_palette = None
        self.canvas = None
        self.age = 0
        self.reused = False

    def note_reuse(self):
        """Mark that this frame reused the cached partition (advances `age`)."""
        self.age += 1
        self.reused = True

    def reuse_render(self, labels, palette, geometry_reused):
        """Return the cached canvas iff this frame's drawing inputs are identical.

        Function
        --------
        The canvas is a pure function of (tri, labels, palette, shapes). With
        geometry reused, `tri` and the shapes are already fixed, so if `labels`
        and `palette` are byte-identical to the frame that drew the cached
        canvas, the canvas IS that same image and re-rasterising it would waste
        ~10 ms. Requiring BOTH arrays to match is what makes the reuse exact
        rather than approximate.

        Shape/semantics: `labels` is (T,) uint8, `palette` (K,3) uint8 BGR;
        returns the cached (oH,oW,3) uint8 canvas or None meaning "redraw".
        Returns None unless temporal coherence is on, the geometry was reused
        this frame, a canvas is cached, and both arrays match exactly. Side
        effect: bumps `render_hits` on a hit, for the HUD/measurement.
        """
        if not (self.enabled and geometry_reused and self.canvas is not None):
            return None
        if self.render_labels is None or self.render_labels.shape != labels.shape:
            return None
        if not np.array_equal(labels, self.render_labels):
            return None
        if not np.array_equal(palette, self.render_palette):
            return None
        self.render_hits += 1
        return self.canvas

    def remember_render(self, labels, palette, canvas):
        """Cache the canvas together with the exact inputs that produced it.

        Shape/semantics: `labels` (T,) uint8, `palette` (K,3) uint8 BGR,
        `canvas` (oH,oW,3) uint8. No-op when disabled. References are stored,
        not copies, because the caller never mutates `labels`/`palette`/`canvas`
        after this point (quantize allocates fresh arrays each frame).
        """
        if self.enabled:
            self.render_labels = labels
            self.render_palette = palette
            self.canvas = canvas
