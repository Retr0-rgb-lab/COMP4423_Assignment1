"""
brick_display — composite canvas layout and HUD for the live camera view.

Presentation only: nothing here computes geometry or colour, it arranges the two
images a viewer needs (the camera frame and the triangle-brick render) into one
window and writes the on-screen statistics over them.

Split out of `camera_app.py` on purpose. That driver was already at the 400-line
limit (AGENTS 4.1), and the rule there is to split by responsibility rather than
grow a file -- "driver stays thin" is the same rule this module serves.

Three functions:
  * `make_side_by_side` -- the two-pane composite the window shows.
  * `draw_hud`          -- FPS / config / per-stage medians, drawn over a pane.
  * `window_closed`     -- did the user press the window's close button?
"""
import cv2
import numpy as np

# BGR colours. Both HUD colours are dark so white text stays readable whatever
# the frame contains -- a mosaic can end up mostly white, and light-on-light
# text would vanish exactly in the bright scenes we most want to inspect.
HUD_BG = (32, 32, 32)
HUD_FG = (255, 255, 255)
PANE_BG = (24, 24, 24)
CAPTION_FG = (230, 230, 230)

# Layout constants, exported so a caller can compute the window size it needs
# before the first frame exists -- duplicating the 26/6 literals elsewhere would
# silently break the initial window size the day the layout changes.
CAPTION_H = 26
GAP = 6


def composite_size(pane_shape):
    """Rows and columns the side-by-side composite will have for a pane.

    Shape/semantics: takes a pane shape `(h, w)` or `(h, w, 3)` and returns
    `(rows, cols)` = `(h + CAPTION_H, 2*w + GAP)`. Needed because the window is
    created before the first frame is rendered: `camera_app` uses this to size the
    window for the layout it is about to produce, using the same constants
    `make_side_by_side` uses, rather than restating them.
    """
    h, w = pane_shape[0], pane_shape[1]
    return h + CAPTION_H, 2 * w + GAP


def make_side_by_side(original, render, gap=GAP, caption_h=CAPTION_H):
    """Compose the camera frame and the render into one left/right image.

    Function
    --------
    This is what makes the program self-explanatory: the left pane is what the
    camera saw, the right pane is what this project made of it. Judging a
    triangle-brick render without its source is guesswork, so the pair is the
    unit of display rather than the render alone.

    Shape
    -----
    Both inputs must be the same (H, W, 3) uint8 BGR -- `render` is cropped to
    the original size by `render_triangles`, so the caller passes the frame it
    actually processed, not the raw camera frame. Output is
    `(H + caption_h, 2*W + gap, 3)` uint8 BGR, i.e. the panes plus a caption strip
    on top and a `gap`-wide vertical separator; total width is `2*W + gap`.

    Semantics
    ---------
    `out[caption_h:, :W]` is the ORIGINAL frame (left pane) and
    `out[caption_h:, W+gap:]` is the RENDER (right pane); the column band
    `[W, W+gap)` and the row band `[0, caption_h)` are background. Returns
    `(out, render_x0)` where `render_x0 = W + gap` is the left edge of the render
    pane, which is where the HUD must be drawn to sit over the render rather than
    straddling the separator.

    Raises: nothing; a size mismatch would misalign the two panes silently, so
    the caller is required to pass a matching pair (camera_app guarantees it by
    passing the same resized frame it fed the pipeline).
    """
    if original.shape != render.shape:
        raise ValueError(
            f"panes must match: original {original.shape} vs render {render.shape}")
    h, w = render.shape[:2]
    out = np.full((h + caption_h, 2 * w + gap, 3), PANE_BG, dtype=np.uint8)
    out[caption_h:, :w] = original
    out[caption_h:, w + gap:] = render
    # Captions are the only thing that distinguishes "before" from "after" once
    # the image is on screen, so they are not decoration.
    cv2.putText(out, "a) camera input", (8, caption_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CAPTION_FG, 1, cv2.LINE_AA)
    cv2.putText(out, "b) triangle bricks", (w + gap + 8, caption_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CAPTION_FG, 1, cv2.LINE_AA)
    return out, w + gap


def draw_hud(canvas, fps, info, cfg, paused, timer, x0=0):
    """Draw the FPS / config / per-stage HUD onto `canvas` in place.

    Function
    --------
    The HUD is how a real-time claim is verified on screen instead of asserted:
    it shows the measured FPS *and* the median cost of each pipeline stage, so a
    viewer can see WHERE the time goes rather than only that the result is slow.

    Shape
    -----
    `canvas` is (H, W, 3) uint8 BGR, modified in place. Each line is drawn on its
    own filled rectangle whose height tracks the measured text height, stacked
    downward from `y=22` at horizontal offset `x0+6`; the panel is drawn over
    whatever is already there, so it is normally the LAST thing drawn on a pane.

    Semantics
    ---------
    `x0` is the left edge of the pane the HUD belongs to -- for a two-pane
    composite the caller passes the render pane's offset, otherwise the text
    would straddle the separator. `fps` is the caller's rolling estimate, and
    `timer.report()` supplies the per-stage medians, so what is on screen and
    what the exit summary prints come from one identical measurement.
    """
    rep = timer.report()
    lines = [
        f"FPS {fps:5.1f}   {'PAUSED' if paused else 'live'}",
        f"tris {info['n_tri']:5d}/{cfg.budget}   cells {info['n_cells']:5d}",
        f"{cfg.partition} S_max={max(cfg.S_set)} K={cfg.K} "
        f"{cfg.palette_method}",
        f"{'REUSE (static scene)' if info.get('reused') else 'PARTITION'} "
        f"{'canvas CACHED' if info.get('render_reused') else 'canvas REDRAWN'} "
        f"{'palette FRESH' if info.get('palette_refreshed') else ''}",
        "sizes " + " ".join(f"{s}:{n}" for s, n in sorted(info["sizes"].items())),
        "--- median ms per stage ---",
    ]
    for name in ["partition", "triangles", "means", "palette", "quantize",
                 "render"]:
        if name in rep:
            lines.append(f"  {name:<10s} {rep[name]['median_ms']:8.1f}")
    lines.append(f"  {'frame':<10s} {rep['__total__']['median_ms']:8.1f}")
    lines.append("q/ESC quit   s snapshot   p pause   r reset")

    y = 22
    for text in lines:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x0 + 6, y - th - 4), (x0 + 14 + tw, y + 4),
                      HUD_BG, -1)
        cv2.putText(canvas, text, (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    HUD_FG, 1, cv2.LINE_AA)
        y += th + 9
    return canvas


def window_closed(name):
    """Has the user closed the window with its close button?

    Function
    --------
    OpenCV's HighGUI loop does not tell you when the user clicks the window's X:
    the window is simply destroyed underneath you and `imshow` keeps drawing into
    nowhere. Polling `WND_PROP_VISIBLE` is the way to notice, and it is what lets
    the program exit on the close button instead of only on a keypress.

    Shape/semantics: `name` is the window title as passed to `imshow`. Returns
    True when the window is gone. `cv2` raises `cv2.error` rather than returning 0
    once the window no longer exists on some backends, so that is treated as
    "closed" too -- without this the probe itself would crash the loop it exists
    to terminate.
    """
    try:
        return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True
