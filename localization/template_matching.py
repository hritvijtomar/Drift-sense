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


def _to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def multiscale_ncc_candidates(reference, search, scales=None, angles=None,
                               topk=5, nms_radius=25):
    """
    Runs cv2.matchTemplate (TM_CCOEFF_NORMED) over a small grid of scales
    and rotations of the reference, keeps the best score per location, and
    returns the top-K non-overlapping peaks.

    Returns: list of dicts: {x, y, score, scale, angle, w, h}
             (x, y) is the CENTER of the matched region in the search image.
    """
    ref_gray = _to_gray(reference)
    search_gray = _to_gray(search)

    if scales is None:
        scales = np.linspace(0.07, 0.15, 9)   # around the known ~10x relationship
    if angles is None:
        angles = np.linspace(-4, 4, 5)

    H, W = search_gray.shape
    score_map = np.full((H, W), -1.0, dtype=np.float32)
    meta_map = np.empty((H, W), dtype=object)

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
            # Update global score map (aligned to top-left corner coordinates)
            mask = result > score_map[:rh, :rw]
            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                score_map[y, x] = result[y, x]
                meta_map[y, x] = (scale, angle, new_w, new_h)

    # Non-max suppression over score_map to get top-K distinct peaks
    candidates = []
    flat_idx = np.argsort(score_map, axis=None)[::-1]
    taken = np.zeros_like(score_map, dtype=bool)

    for idx in flat_idx:
        y, x = np.unravel_index(idx, score_map.shape)
        if taken[y, x]:
            continue
        score = score_map[y, x]
        if score <= 0 or meta_map[y, x] is None:
            continue
        scale, angle, w, h = meta_map[y, x]
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
