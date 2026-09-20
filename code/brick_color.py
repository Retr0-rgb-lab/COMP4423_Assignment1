"""
brick_color — colour engine for the triangle-brick assignment.

Two families:
  * 3-class quantizers (Task 2): `quantize_otsu`, `quantize_kmeans`,
    `quantize_fixed`. Each returns (labels, palette_bgr) sorted dark -> bright.
  * K-colour palette builders (Task 3): `palette_kmeans`,
    `palette_median_cut`, dispatched by `build_palette`. Assignment is always
    `quantize_nearest_bgr`.

Channel order: everywhere in this module an array named or documented BGR
stores OpenCV order along the last axis -- `arr[..., 0]=B`, `[..., 1]=G`,
`[..., 2]=R`. The `[:, ::-1]` flips below are BGR<->RGB conversions required by
skimage and by the clustering feature space. A wrong flip still renders
plausibly, it just swaps red and blue, so do not "simplify" them away.

Colour-space contract -- three spaces, deliberately:
  CLUSTERING  Lab for `palette_kmeans(space="lab")` (perceptually uniform, so
              centroids land where the eye wants them), RGB for
              `palette_median_cut` (splitting raw channels is the classic form).
  ASSIGNMENT  always plain Euclidean BGR (`quantize_nearest_bgr`).
  EVALUATION  CIEDE2000 in Lab (brick_metrics).
The assignment step is the odd one out: a Lab-optimal centroid set can be
assigned by a BGR-nearest rule that disagrees with it, so `Quant_Error` is not
the minimum attainable for a given palette. This is a KNOWN, ACCEPTED
approximation -- BGR distance is the plain reading of the spec's "nearest
colour" and keeps Task 2 and Task 3 on one identical rule. Report it as a
limitation; do not "fix" it silently, because that invalidates the earlier
comparisons. See the note on `quantize_nearest_bgr`.
"""
import cv2
import numpy as np
from skimage import color as skcolor


# ---------------------------------------------------------------------------
# Shared colour utilities
# ---------------------------------------------------------------------------

def _bt601_luma(means_bgr):
    """BT.601 luma of BGR colours, as a 1-D vector.

    Function: collapses each colour to its perceived brightness. This scalar is
    the signal both Task 2 threshold quantizers actually classify on -- three
    colours from one number means "dark / mid / bright", not "cluster in RGB".
    Shape: (N, 3) -> (N,). Semantics: `means_bgr[i, 0]=B` is weighted 0.114,
    `[i, 1]=G` 0.587, `[i, 2]=R` 0.299. The B-first order of those weights is
    what makes this correct for OpenCV data, and is the easy thing to get
    backwards.
    """
    means = means_bgr.astype(np.float32)
    return 0.114 * means[:, 0] + 0.587 * means[:, 1] + 0.299 * means[:, 2]


def _palette_by_luminance(labels, palette):
    """Relabel so palette index 0 is darkest and the last index brightest.

    Function: K-Means and Otsu both return class ids in an implementation-
    defined order, so two runs of "K=3" can call the same colour 0 or 2.
    Sorting by luma makes runs comparable, makes the palette swatch strip read
    dark -> bright, and lets callers assume "index order == brightness order".
    Returns labels and palette together because permuting one is only valid
    together with the matching permutation of the other.
    Shape: ((T,) int, (K, 3)) -> ((T,) uint8, (K, 3) same dtype as input).
    Semantics: `lum[k]` is the luma of palette entry k; `order` is the
    dark->bright permutation of indices; `remap[old_index] = new_index` is its
    inverse, used to translate the labels. The RETURNED labels index into the
    RETURNED palette, not the input one.
    Note: the BT.601 coefficients are duplicated from `_bt601_luma` rather than
    reused (refactor candidate) -- if one definition changes, change both.
    """
    lum = 0.114 * palette[:, 0].astype(np.float32) \
        + 0.587 * palette[:, 1].astype(np.float32) \
        + 0.299 * palette[:, 2].astype(np.float32)
    order = np.argsort(lum)
    remap = np.zeros(len(palette), dtype=np.uint8)
    remap[order] = np.arange(len(palette), dtype=np.uint8)
    return remap[labels], palette[order]


# ---------------------------------------------------------------------------
# 3-class quantizers (Task 2)
# ---------------------------------------------------------------------------

def multi_otsu_3class(gray_values):
    """The two luma thresholds that maximise 3-class between-class variance.

    Function: multilevel Otsu -- the standard K-class generalisation of Otsu
    (1979), multilevel form in Liao et al. (2001). For every candidate threshold
    pair it scores how much of the total variance is explained by splitting the
    samples into three groups, and keeps the best pair. Because total variance =
    between-class + within-class, maximising the between term minimises the
    within-class spread that three flat colours cannot represent. It ADAPTS to
    the histogram, which is why it beats the fixed 33%/67% cut in
    `quantize_fixed`.
    Shape: (N,) -> (int, int). Semantics: `gray_values` are luma in [0, 256);
    the returned `(t1, t2)` are INCLUSIVE lower bounds, so callers classify as
    `gray < t1` / `t1 <= gray < t2` / `gray >= t2` (see `quantize_otsu`).
    A degenerate histogram (no samples) falls back to (85, 170); never raises.
    Why hand-rolled: `skimage.filters.threshold_multiotsu` would do this in one
    line, but Task 2 is graded on the 3-colour derivation, so the mechanism
    stays explicit and inspectable. Do not substitute a third-party snippet --
    web copying is prohibited and this file must stay self-derived.
    """
    hist, _ = np.histogram(gray_values, bins=256, range=(0, 256))
    total = hist.sum()
    if total == 0:
        return 85, 170
    prob = hist.astype(np.float64) / total
    bins = np.arange(256, dtype=np.float64)
    cum_p = np.cumsum(prob)
    cum_i = np.cumsum(prob * bins)
    grand_mean = cum_i[-1]

    best_var = -np.inf
    best_t1, best_t2 = 85, 170
    for t1 in range(1, 255):
        w1 = cum_p[t1 - 1]
        if w1 <= 0:
            continue
        m1 = cum_i[t1 - 1] / w1
        for t2 in range(t1 + 1, 255):
            w2 = cum_p[t2 - 1] - cum_p[t1 - 1]
            w3 = 1.0 - cum_p[t2 - 1]
            if w2 <= 0 or w3 <= 0:
                continue
            m2 = (cum_i[t2 - 1] - cum_i[t1 - 1]) / w2
            m3 = (cum_i[-1] - cum_i[t2 - 1]) / w3
            var = (w1 * (m1 - grand_mean) ** 2
                   + w2 * (m2 - grand_mean) ** 2
                   + w3 * (m3 - grand_mean) ** 2)
            if var > best_var:
                best_var = var
                best_t1, best_t2 = t1, t2
    return best_t1, best_t2


def _class_mean_palette(means, labels, k):
    """Palette entry = mean BGR of the items in that class.

    Function: a luma threshold decides WHICH items belong to each class; the
    representative colour should then be the average of that class's actual
    colours, not the arbitrary threshold value (e.g. flat grey 85). Taking the
    real mean preserves residual hue -- a class holding bluish-grey pixels
    yields a bluish-grey brick -- which is what makes a 3-colour result read as
    a mosaic of the photo instead of a halftone.
    Shape: ((T, 3) float, (T,) int, int) -> (k, 3) uint8 BGR. Semantics:
    `means[i, 0]=B`; `labels[i]` is in [0, k); `palette[c]` is the clipped
    per-channel mean of all i with `labels[i] == c`. A class with no members
    keeps its initialised (0, 0, 0) black instead of raising, which is possible
    when a threshold is degenerate.
    """
    palette = np.zeros((k, 3), dtype=np.uint8)
    means = means.astype(np.float32)
    for c in range(k):
        members = means[labels == c]
        if len(members) > 0:
            palette[c] = np.clip(members.mean(axis=0), 0, 255).astype(np.uint8)
    return palette


def quantize_otsu(means_bgr):
    """Task 2 primary quantizer: multilevel Otsu on luma, palette = class means.

    Function: threshold BT.601 luma with adaptive cut points (see
    `multi_otsu_3class`), then take each class's mean BGR as its colour. Primary
    choice because the thresholds follow the image histogram, so it survives an
    exposure that does not match an assumed fixed split.
    Shape: (T, 3) -> ((T,) uint8, (3, 3) uint8). Semantics: `labels[i]` is
    triangle i's palette index; `palette[0]` is the darkest of the three
    (guaranteed by `_palette_by_luminance`), so 0/1/2 read as dark/mid/bright.
    The palette colours are class MEANS, so the palette alone does not reveal
    where the thresholds fell.
    """
    gray = _bt601_luma(means_bgr)
    t1, t2 = multi_otsu_3class(gray)
    labels = np.zeros(len(gray), dtype=np.uint8)
    labels[gray >= t1] = 1
    labels[gray >= t2] = 2
    return _palette_by_luminance(labels, _class_mean_palette(means_bgr, labels, 3))


def quantize_kmeans(means_bgr, k=3, seed=0):
    """Task 2 comparison quantizer: K-Means in BGR, K=3.

    Function: clusters triangle means directly in BGR (no luma collapse) with
    OpenCV's K-Means++ init, so the report can contrast "threshold on
    brightness" (Otsu) against "cluster in colour". K-Means can split on hue --
    a red roof versus a grey road of the same brightness -- where a luma
    threshold cannot; the cost is 10 restarts and a local optimum that can
    differ between runs.
    Shape: (T, 3) -> ((T,) uint8, (3, 3) uint8). Semantics: same contract as
    `quantize_otsu` -- `labels[i]` is the palette index (0 darkest),
    `palette[k]` the BGR centroid. `seed` is accepted but NOT used: cv2.kmeans
    draws from OpenCV's global RNG, so reproducibility requires calling
    `cv2.setRNGSeed()` at the call site, which task3_best does.
    """
    samples = means_bgr.astype(np.float32).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, labels_col, centers = cv2.kmeans(
        samples, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    labels = labels_col.flatten().astype(np.uint8)
    palette = np.clip(centers, 0, 255).astype(np.uint8)
    return _palette_by_luminance(labels, palette)


def quantize_fixed(means_bgr):
    """Task 2 comparison quantizer: fixed 33.3% / 66.7% luma percentiles.

    Function: the naive baseline -- always split the luma distribution into
    equal-sized thirds regardless of content, then take class means. Kept in
    the suite because it isolates the value of adaptive thresholds: on evenly
    exposed images the two agree, on backlit or low-key ones the fixed cut
    lumps nearly everything into a single class. Percentiles equalise COUNTS,
    not brightness ranges, which is why this version tends to look wrong on
    skies.
    Shape: (T, 3) -> ((T,) uint8, (3, 3) uint8). Semantics: same contract as
    `quantize_otsu`; `labels` are 0/1/2 dark -> bright and the palette entries
    are class means.
    """
    gray = _bt601_luma(means_bgr)
    t1, t2 = np.percentile(gray, [100.0 / 3, 200.0 / 3])
    labels = np.zeros(len(gray), dtype=np.uint8)
    labels[gray >= t1] = 1
    labels[gray >= t2] = 2
    return _palette_by_luminance(labels, _class_mean_palette(means_bgr, labels, 3))


# ---------------------------------------------------------------------------
# K-colour palette builders (Task 3)
# ---------------------------------------------------------------------------

def palette_kmeans(means_bgr, k, space="lab", seed=0):
    """K-Means palette builder, clustering in Lab (default) or RGB.

    Function: Task 3's main palette. In Lab, Euclidean distance between
    centroids approximates perceptual distance, which is precisely why the
    D_palette experiment prefers it over median cut. Centroids are mapped back
    to BGR before returning. Note the split of responsibility: centroids are
    optimal in Lab, but the later assignment runs in BGR, so the two stages can
    disagree -- see the colour-space contract in the module docstring.
    Shape: (T, 3) float -> (k, 3) uint8 BGR. Semantics: `means_bgr[..., 0]=B`;
    the clustered features are Lab (L in [0, 100]) or RGB in [0, 255] depending
    on `space`, and the intermediate `means_rgb` is (T, 3) float64 in [0, 1]
    because skimage's rgb2lab expects that range. Output `palette[i, 0]=B`, NOT
    sorted (the caller assigns by distance, which is order-agnostic). A `k`
    larger than the number of distinct means yields fewer distinct centroids.
    `seed` is accepted but unused -- seed cv2 instead.
    """
    if space == "lab":
        means_rgb = means_bgr[:, ::-1].astype(np.float64) / 255.0
        means_in = skcolor.rgb2lab(np.clip(means_rgb, 0, 1)).astype(np.float32)
    else:
        means_in = means_bgr[:, ::-1].astype(np.float32)  # BGR -> RGB features
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    _, _, centers = cv2.kmeans(means_in, k, None, criteria, 10,
                               cv2.KMEANS_PP_CENTERS)
    if space == "lab":
        centers_rgb = skcolor.lab2rgb(centers.reshape(1, -1, 3)).reshape(-1, 3)
        centers_bgr = np.clip(centers_rgb * 255, 0, 255).astype(np.uint8)[:, ::-1]
    else:
        centers_bgr = centers[:, ::-1].astype(np.uint8)
    return np.clip(centers_bgr, 0, 255).astype(np.uint8)


def palette_median_cut(means_bgr, k):
    """Recursive median-cut palette builder, splitting in RGB.

    Function: classic median cut. Start with one box holding every colour; take
    the box with the largest channel range and split it at the MEDIAN of its
    widest channel; repeat until there are `k` boxes; each box's mean becomes a
    palette entry. Splitting at the median rather than the midpoint is what
    makes it robust to skewed distributions, and always choosing the widest box
    is what keeps coverage even. Deterministic, unlike K-Means -- which is why
    runs that must be reproducible use it (task3_smax_curve).
    Shape: (T, 3) float -> (k', 3) uint8 BGR where `k' = min(k, distinct)`.
    Semantics: `arr` is (T, 3) float32 RGB; `boxes` is a list of (m, 3) arrays
    whose lengths always sum to T, so no colour is dropped or duplicated by the
    split loop. The output is NOT sorted by brightness.
    Edge cases, both silent: (1) if splitting stalls -- fewer distinct samples
    than `k` -- the palette is PADDED by repeating the last colour so the shape
    is always (k, 3), which means "K=16" can be nominal only; that is what
    `Palette_Util` in the Task 3 metrics exists to expose. (2) duplicated
    entries make tie-breaking in `quantize_nearest_bgr` always resolve to the
    first copy.
    """
    arr = means_bgr[:, ::-1].astype(np.float32).reshape(-1, 3)  # RGB
    boxes = [arr]

    while len(boxes) < k and len(boxes) < arr.shape[0]:
        ranges = np.array([b.max(axis=0) - b.min(axis=0) for b in boxes])
        idx = int(np.argmax(ranges.max(axis=1)))
        box = boxes[idx]
        if len(box) < 2:
            break
        ch = int(np.argmax(ranges[idx]))
        sorted_box = box[box[:, ch].argsort()]
        mid = len(sorted_box) // 2
        boxes[idx] = sorted_box[:mid]
        boxes.insert(idx + 1, sorted_box[mid:])

    palette_rgb = np.array([b.mean(axis=0) for b in boxes])
    while len(palette_rgb) < k:
        palette_rgb = np.vstack([palette_rgb, palette_rgb[-1:]])
    palette_bgr = np.clip(palette_rgb[:, ::-1], 0, 255).astype(np.uint8)
    return palette_bgr[:k]


def build_palette(means_bgr, k, method, seed=0):
    """Dispatch to a palette builder by name -- single entry point for Task 3.

    Function: keeps the experiment grid data-driven, so `Task3Config` carries
    `palette_method` as a string and the sweep never branches on the algorithm
    itself. An unknown name raises instead of falling back to a default,
    deliberately: a silent default would re-run a different algorithm under the
    old label, and the comparison would be wrong without anyone noticing.
    Shape: (T, 3) float -> (k, 3) uint8 BGR. Semantics: `means_bgr[..., 0]=B`;
    `method` is one of "kmeans_lab" / "kmeans_rgb" / "median_cut"; the output
    `palette[i, 0]=B` is unsorted with METHOD-DEPENDENT ordering -- never index
    a palette positionally across methods.
    """
    if method == "kmeans_lab":
        return palette_kmeans(means_bgr, k, "lab", seed)
    if method == "kmeans_rgb":
        return palette_kmeans(means_bgr, k, "rgb", seed)
    if method == "median_cut":
        return palette_median_cut(means_bgr, k)
    raise ValueError(f"Unknown palette method: {method}")


def quantize_nearest_bgr(means_bgr, palette_bgr):
    """Assign each colour to its nearest palette entry, in BGR Euclidean space.

    Function: the bridge from "a palette exists" to "every triangle has a
    colour" -- computes all T x K squared distances and takes the per-row
    argmin.

    WHY BGR AND NOT Lab/CIEDE2000 -- read before changing:
    raw BGR distance is known to correlate poorly with perceived colour
    difference, which is exactly why brick_metrics evaluates with CIEDE2000. It
    is used anyway because (a) it is the plain reading of the spec's colour
    "determined from its own image region" as a nearest-colour rule, and (b) it
    keeps Task 2 and Task 3 on one identical rule, so the geometry experiments
    are not confounded by a colour change. The cost is that `Quant_Error` is not
    the minimum attainable for a given palette -- a Lab-based assignment would
    score lower on that metric. This is a documented approximation, not an
    oversight: switching it silently would invalidate the earlier comparisons,
    so report it as a limitation instead.

    Shape: input `means_bgr` (T, 3) float32 with `[..., 0]=B`; `palette_bgr`
    (K, 3) uint8 with `[k, 0]=B`. Semantics: `diff[i, k]` is the B,G,R offset
    from triangle i to palette entry k -- shape (T, K, 3) float32, the memory
    peak of the whole pipeline (T=9990, K=16 -> ~1.9 MB) and it scales with K,
    a real constraint for Task 4's real-time loop. `dist[i, k]` is (T, K)
    squared distance, deliberately unsqrt'ed: the argmin is unchanged and it
    avoids K*T square roots. Output `labels` (T,) uint8 gives triangle i's
    palette index in [0, K-1]; ties break to the LOWEST index (np.argmin), so a
    duplicated palette entry always resolves to its first copy.
    """
    means_f = means_bgr.astype(np.float32)
    pal_f = palette_bgr.astype(np.float32)
    diff = means_f[:, None, :] - pal_f[None, :, :]
    dist = (diff * diff).sum(axis=2)
    return np.argmin(dist, axis=1).astype(np.uint8)
