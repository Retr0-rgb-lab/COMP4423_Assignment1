"""
brick_display — composite canvas layout and HUD for the live camera view.

Presentation only: nothing here computes geometry or colour, it arranges the two
images a viewer needs (the camera frame and the triangle-brick render) into one
window and writes the on-screen statistics over them.

Split out of `camera_app.py` on purpose. That driver was already at the 400-line
limit (AGENTS 4.1), and the rule there is to split by responsibility rather than
grow a file -- "driver stays thin" is the same rule this module serves.

Three functions, plus the aspect-preserving window fit:
  * `make_side_by_side` -- the two-pane composite the window shows.
  * `draw_hud`          -- a compact 3-line FPS/stage readout over a pane.
  * `fit_letterbox`     -- scale the composite to a window WITHOUT stretching.
  * `window_closed`     -- did the user press the window's close button?
"""
import cv2
import numpy as np

# BGR colours. The HUD text is light and sits on a translucent dark panel, so it
# stays readable over any mosaic without the heavy solid boxes the first version
# used (which covered a large part of the frame).
HUD_BG = (24, 24, 24)
HUD_FG = (235, 235, 235)
HUD_ALPHA = 0.45          # panel opacity; 1.0 would be a solid box again
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE = 0.4           # small: the HUD must not dominate the picture
HUD_THICK = 1
HUD_LINE_H = 14           # px between HUD baselines
HUD_PAD = 6
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


def hud_lines(fps, info, cfg, paused, timer):
    """The three compact HUD lines (kept separate so they can be measured).

    Function: condenses what the first HUD version spread over 14 lines into
    three -- one status line, one timing line, one key-help line -- because the
    HUD should not cover the picture. The per-size histogram and the full config
    are dropped here on purpose: they are still in the exit summary, which is
    where anyone reading numbers should look, not the live overlay.

    Shape/semantics: returns `list[str]`. `fps` is the caller's rolling FPS;
    `info` supplies `n_tri`, `reused`, `render_reused` (see frame_pipeline);
    `timer.report()` supplies the per-stage medians, so the overlay and the exit
    summary come from one identical measurement.
    """
    rep = timer.report()
    stage = " ".join(f"{name[:4]} {rep[name]['median_ms']:.0f}"
                     for name in ("partition", "triangles", "means", "palette",
                                  "quantize", "render") if name in rep)
    mode = "REUSE" if info.get("reused") else "PART"
    mode += "/cached" if info.get("render_reused") else "/redraw"
    return [
        f"FPS {fps:4.1f} {'PAUSED' if paused else 'live'}  "
        f"tris {info['n_tri']}/{cfg.budget}  K{cfg.K}  {mode}",
        f"{stage}  tot {rep['__total__']['median_ms']:.0f}",
        "q quit  s snapshot  p pause  r reset",
    ]


def draw_hud(canvas, fps, info, cfg, paused, timer, x0=0):
    """Draw the compact HUD onto `canvas` in place.

    Function
    --------
    The HUD is how a real-time claim is verified on screen instead of asserted:
    it shows the measured FPS *and* the median cost of each stage. It is drawn
    as three small lines on ONE translucent panel (a local `addWeighted` over
    the panel ROI only), not the original stack of solid black boxes, so it no
    longer covers a large part of the frame.

    Shape
    -----
    `canvas` is (H, W, 3) uint8 BGR, modified in place. The panel is
    `(panel_w, panel_h)` starting at `(x0+4, 4)`, where panel_w/panel_h are
    measured from the text; the ROI is clamped to the canvas so a tiny window
    cannot index out of bounds. `x0` is the left edge of the pane the HUD
    belongs to (the render pane's offset in the composite), so the text sits
    over the render, not across the separator.

    Semantics
    ---------
    Text is drawn AFTER the translucent panel, so the glyphs stay fully opaque
    over the darkened background. Because the driver fits the composite to the
    window BEFORE calling this, a fixed `HUD_SCALE` keeps the HUD a constant
    screen size no matter how the window is dragged.
    """
    lines = hud_lines(fps, info, cfg, paused, timer)
    sizes = [cv2.getTextSize(t, HUD_FONT, HUD_SCALE, HUD_THICK)[0]
             for t in lines]
    tw = max(s[0] for s in sizes)
    th = max(s[1] for s in sizes)
    H, W = canvas.shape[:2]
    px0, py0 = x0 + 4, 4
    px1 = min(W, px0 + tw + 2 * HUD_PAD)
    py1 = min(H, py0 + HUD_LINE_H * len(lines) + 2 * HUD_PAD)
    if px1 > px0 and py1 > py0:
        roi = canvas[py0:py1, px0:px1]
        dark = np.full_like(roi, HUD_BG)
        cv2.addWeighted(dark, HUD_ALPHA, roi, 1.0 - HUD_ALPHA, 0.0, roi)
    y = py0 + HUD_PAD + th
    for text in lines:
        if y < H:
            cv2.putText(canvas, text, (px0 + HUD_PAD, y), HUD_FONT, HUD_SCALE,
                        HUD_FG, HUD_THICK, cv2.LINE_AA)
        y += HUD_LINE_H
    return canvas


def window_area(name):
    """Size (w, h) of a window's drawable image area, or None if unavailable.

    Function: `cv2.getWindowImageRect` reports the client area that `imshow`
    scales into, which is what the aspect-preserving fit needs to target. Some
    backends raise, or report a zero/negative rect before the window is mapped,
    so that is reported as None and the caller falls back to native size.

    Shape/semantics: window title str in; `(w, h)` ints or None out. No side
    effects on the window.
    """
    try:
        _, _, w, h = cv2.getWindowImageRect(name)
        if w > 0 and h > 0:
            return w, h
    except cv2.error:
        pass
    return None


def fit_letterbox(img, win_w, win_h, bg=PANE_BG):
    """Scale `img` to fit (win_w, win_h) with the aspect ratio PRESERVED.

    Function
    --------
    A `WINDOW_NORMAL` window stretches whatever `imshow` receives to fill it, so
    displaying the native composite in a differently-shaped window makes every
    right-isosceles triangle non-isosceles. This fits the image into the window
    with a uniform scale and centres it on a background, so the caller can
    `imshow` a window-sized canvas and the stretch becomes a no-op.

    Shape
    -----
    Input: `img` (h, w, 3) uint8 BGR; target `(win_w, win_h)`. Output:
    `(canvas, off_x, off_y, scale)` where `canvas` is exactly
    `(win_h, win_w, 3)` uint8, `scale = min(win_w/w, win_h/h)` (<=1 shrinks),
    and `(off_x, off_y)` is where the scaled image was pasted, so the caller can
    map pane coordinates into the canvas. When the sizes already match, the
    input is returned unscaled (`scale=1.0`, offsets 0) to skip a needless copy.

    Semantics: `canvas[off_y:off_y+round(h*scale), off_x:off_x+round(w*scale)]`
    is the aspect-preserved image; the rest is `bg`. Uniform `scale` guarantees
    the isosceles right triangles stay isosceles right triangles.
    """
    h, w = img.shape[:2]
    if win_w == w and win_h == h:
        return img, 0, 0, 1.0
    scale = min(win_w / w, win_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    canvas = np.full((win_h, win_w, 3), bg, dtype=np.uint8)
    ox, oy = (win_w - nw) // 2, (win_h - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas, ox, oy, scale


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
