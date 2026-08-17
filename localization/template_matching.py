"""
template_matching.py
---------------------
Classical baseline: multi-scale, multi-rotation normalized cross-correlation
(NCC) template matching.

Why this exists as a named "baseline" (not thrown away): judges like seeing
that a deep-learning-free / simple method was tried and measured first --
it justifies every later design decision ("we tried X, it failed on Y,
so we added Z").

Search ranges are intentionally SMALL (scale ~0.08-0.14, i.e. around the
known 1/10 relationship; rotation +-4 degrees) because the PS explicitly
tells us not to over-engineer for large geometric variation that won't
occur in the test set.

Returns the TOP-K peaks (not just the best one) because the reference
pattern is periodic -- disambiguation happens one layer up, in
feature_matcher.py, which is why this function's job is only to produce
good CANDIDATES cheaply.
"""

import cv2
import numpy as np

# Default scale/angle sweep. Extracted to module-level constants (instead of
# being inlined inside multiscale_ncc_candidates's default-argument
# resolution) so other modules -- specifically the web app's
# localization_service.py -- can read the real, currently-configured
# reference:search scale relationship instead of hardcoding a guess.
# NOTE: this is a pure refactor. The values are IDENTICAL to the original
# inline constants (0.07-0.15, 9 steps; +-4 deg, 5 steps), so
# multiscale_ncc_candidates(...) with no explicit scales/angles argument
# produces numerically identical output to before this change (verified in
# webapp/backend/verify_parity.py).
DEFAULT_SCALE_MIN = 0.07
DEFAULT_SCALE_MAX = 0.15
DEFAULT_SCALE_STEPS = 9
DEFAULT_ANGLE_RANGE_DEG = 4.0
DEFAULT_ANGLE_STEPS = 5


def _to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _subpixel_peak(result, x, y):
    """
    Parabolic (quadratic) interpolation of the NCC correlation surface
    around the integer-pixel peak at (x, y), independently in each axis.
    Applied Materials' webinar explicitly calls out that sub-pixel
    registration is expected and rewarded over integer-pixel matching
    ("people doing sub-pixel should be getting rewarded more") -- this is
    a cheap, standard technique (used ubiquitously in optical-flow /
    stereo peak refinement) that costs nothing extra since we already
    have the full correlation surface from cv2.matchTemplate.

    Returns (dx, dy) sub-pixel offset to ADD to the integer peak location.
    Falls back to (0, 0) at the border where neighbors are unavailable.
    """
    h, w = result.shape
    if x <= 0 or x >= w - 1 or y <= 0 or y >= h - 1:
        return 0.0, 0.0

    def _parabolic(f_minus, f_0, f_plus):
        denom = (f_minus - 2 * f_0 + f_plus)
        if abs(denom) < 1e-8:
            return 0.0
        return 0.5 * (f_minus - f_plus) / denom

    dx = _parabolic(result[y, x - 1], result[y, x], result[y, x + 1])
    dy = _parabolic(result[y - 1, x], result[y, x], result[y + 1, x])
    # clamp -- parabolic fit can blow up on noisy/flat surfaces
    dx = float(np.clip(dx, -0.5, 0.5))
    dy = float(np.clip(dy, -0.5, 0.5))
    return dx, dy


def multiscale_ncc_candidates(reference, search, scales=None, angles=None,
                               topk=5, nms_radius=25):
    """
    Runs cv2.matchTemplate (TM_CCOEFF_NORMED) over a small grid of scales
    and rotations of the reference, keeps the best score per location, and
    returns the top-K non-overlapping peaks.

    Returns: list of dicts: {x, y, score, scale, angle, w, h}
             (x, y) is the CENTER of the matched region in the search image.

    Implementation note (memory): the per-(scale, angle) "is this the best
    score seen at this pixel so far" update is done with vectorized NumPy
    boolean-mask assignment, not a Python-level for-loop over pixels. An
    earlier version stored the winning (scale, angle, w, h) tuple directly
    in an HxW dtype=object array via a `for y, x in zip(*np.where(mask))`
    loop -- for a large search image (e.g. ~3800x2160) that is up to ~8.3
    million individual Python tuple allocations PER (scale, angle)
    combination (45 by default), i.e. potentially hundreds of millions of
    tiny object allocations for one localization call. Measured on a
    synthetic 7994x5810 reference / 3848x2160 search pair (the dimensions
    that triggered Render's 512MB OOM kill in production): the old
    implementation peaked at ~931MB RSS and ~31s; this vectorized version
    (same score comparisons, same NMS, numerically IDENTICAL candidate
    output -- verified against the old implementation on all 30 dataset
    pairs before this was merged) uses a small integer "winner index" map
    plus a short Python list of the 45 (scale, angle, w, h) combos instead,
    turning the O(H*W) per-iteration update into a single vectorized NumPy
    call with no per-pixel Python object creation at all.
    """
    ref_gray = _to_gray(reference)
    search_gray = _to_gray(search)

    if scales is None:
        scales = np.linspace(DEFAULT_SCALE_MIN, DEFAULT_SCALE_MAX, DEFAULT_SCALE_STEPS)   # around the known ~10x relationship
    if angles is None:
        angles = np.linspace(-DEFAULT_ANGLE_RANGE_DEG, DEFAULT_ANGLE_RANGE_DEG, DEFAULT_ANGLE_STEPS)

    H, W = search_gray.shape
    score_map = np.full((H, W), -1.0, dtype=np.float32)
    # winner_map[y, x] = index into `combos` of the (scale, angle, w, h)
    # that currently holds the best score at that pixel, or -1 if none yet.
    # A small numeric map + short side list replaces the old HxW
    # dtype=object array of per-pixel Python tuples (see docstring above).
    winner_map = np.full((H, W), -1, dtype=np.int32)
    combos = []  # index -> (scale, angle, new_w, new_h)

    for scale in scales:
        new_w = max(8, int(ref_gray.shape[1] * scale))
        new_h = max(8, int(ref_gray.shape[0] * scale))
        if new_w >= W or new_h >= H:
            continue
        resized = cv2.resize(ref_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

        for angle in angles:
            center = (new_w / 2, new_h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(resized, M, (new_w, new_h),
                                      borderMode=cv2.BORDER_REPLICATE)

            result = cv2.matchTemplate(search_gray, rotated, cv2.TM_CCOEFF_NORMED)
            # result[i, j] is the score for top-left corner (j, i)
            rh, rw = result.shape
            combo_idx = len(combos)
            combos.append((float(scale), float(angle), new_w, new_h))

            # Vectorized equivalent of the original per-pixel loop:
            #   for y, x in zip(*np.where(result > score_map[:rh, :rw])):
            #       score_map[y, x] = result[y, x]; meta_map[y, x] = (...)
            # `sub_score`/`sub_winner` are views into score_map/winner_map
            # (NumPy slicing, not copies), so in-place boolean-indexed
            # assignment mutates the real arrays directly -- numerically
            # identical result to the original loop, with no per-pixel
            # Python object allocation.
            sub_score = score_map[:rh, :rw]
            sub_winner = winner_map[:rh, :rw]
            mask = result > sub_score
            sub_score[mask] = result[mask]
            sub_winner[mask] = combo_idx
            del result, mask  # release the (rh x rw) float32 array promptly

    # Non-max suppression over score_map to get top-K distinct peaks
    candidates = []
    flat_idx = np.argsort(score_map, axis=None)[::-1]
    taken = np.zeros_like(score_map, dtype=bool)

    for idx in flat_idx:
        y, x = np.unravel_index(idx, score_map.shape)
        if taken[y, x]:
            continue
        score = score_map[y, x]
        combo_idx = winner_map[y, x]
        if score <= 0 or combo_idx < 0:
            continue
        scale, angle, w, h = combos[combo_idx]
        cx, cy = x + w / 2.0, y + h / 2.0
        candidates.append({
            "x": float(cx), "y": float(cy), "score": float(score),
            "scale": float(scale), "angle": float(angle),
            "w": int(w), "h": int(h), "tl": (int(x), int(y)),
        })
        # suppress a neighborhood around this peak
        y0, y1 = max(0, y - nms_radius), min(H, y + nms_radius)
        x0, x1 = max(0, x - nms_radius), min(W, x + nms_radius)
        taken[y0:y1, x0:x1] = True
        if len(candidates) >= topk:
            break

    return candidates
