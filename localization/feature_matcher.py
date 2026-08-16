"""
feature_matcher.py
-------------------
This is the actual differentiator for this problem, not the choice of
ORB vs SIFT vs template matching.

DRAM/FinFET layouts are periodic, so multi-scale NCC alone (template_matching.py)
will produce SEVERAL near-identical high-score peaks -- one true match and
several aliases at the pattern's repeat pitch. Picking "the single highest
NCC score" is not reliable once noise is added, because noise can easily
flip the ranking between two peaks that are genuinely ~equally good matches
in intensity-correlation terms.

The fix implemented here: for each NCC candidate peak, crop a local window
around it in the search image and run ORB keypoint matching + RANSAC
affine estimation against the reference. A TRUE match has to be
geometrically consistent (many RANSAC inliers under a plausible small
rotation/scale), while a periodic ALIAS only correlates well in bulk
intensity but produces far fewer consistent keypoint correspondences,
because local noise realizations differ between repeats (the noise is
independently sampled, per PS requirement) and any local defects/jitter in
the pattern break the exact repeat.

Final confidence = weighted combination of NCC score and ORB inlier
fraction. If the top-2 candidates' final scores are within `ambiguous_gap`
of each other, we flag the result as "ambiguous" for the failure-analysis
slide, rather than silently guessing.
"""

import cv2
import numpy as np


def _to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def orb_verify(reference, search, candidate, orb_features=500, ratio_thresh=0.75):
    """
    Crops a window around `candidate` in the search image (sized to the
    candidate's matched scale, with padding), matches ORB features against
    the reference, and returns (inlier_count, total_good_matches,
    inlier_ratio).
    """
    ref_gray = _to_gray(reference)
    search_gray = _to_gray(search)

    w, h = candidate["w"], candidate["h"]
    x0, y0 = candidate["tl"]
    pad = int(0.25 * max(w, h))
    H, W = search_gray.shape
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(W, x0 + w + pad), min(H, y0 + h + pad)
    crop = search_gray[cy0:cy1, cx0:cx1]

    if crop.size == 0:
        return 0, 0, 0.0

    orb = cv2.ORB_create(nfeatures=orb_features)
    ref_resized = cv2.resize(ref_gray, (w, h), interpolation=cv2.INTER_AREA)

    kp1, des1 = orb.detectAndCompute(ref_resized, None)
    kp2, des2 = orb.detectAndCompute(crop, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return 0, 0, 0.0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    if len(good) < 4:
        return 0, len(good), 0.0

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inlier_mask = cv2.estimateAffinePartial2D(src_pts, dst_pts,
                                                   method=cv2.RANSAC,
                                                   ransacReprojThreshold=4.0)
    if inlier_mask is None:
        return 0, len(good), 0.0

    inliers = int(inlier_mask.sum())
    return inliers, len(good), inliers / max(1, len(good))


def rank_candidates(reference, search, candidates, ncc_weight=0.5, orb_weight=0.5):
    """
    Combines NCC score and ORB verification into one confidence per
    candidate, sorted best-first. Adds 'orb_inliers', 'orb_ratio',
    'final_score' fields to each candidate dict (mutates + returns).
    """
    for c in candidates:
        inliers, total, ratio = orb_verify(reference, search, c)
        c["orb_inliers"] = inliers
        c["orb_total"] = total
        c["orb_ratio"] = ratio
        # NCC score is already roughly in [0,1] for good matches; ORB inlier
        # count is unbounded, so normalize by a soft cap.
        orb_component = min(1.0, inliers / 15.0)
        c["final_score"] = ncc_weight * max(0.0, c["score"]) + orb_weight * orb_component

    candidates.sort(key=lambda c: c["final_score"], reverse=True)
    return candidates


def is_ambiguous(ranked_candidates, gap_threshold=0.08):
    """True if the top-2 final scores are too close to trust the ranking --
    this is exactly the 'periodic ambiguity' failure mode the PS asks us to
    detect and explain, not hide."""
    if len(ranked_candidates) < 2:
        return False
    return (ranked_candidates[0]["final_score"] - ranked_candidates[1]["final_score"]) < gap_threshold


def select_final_candidate(ranked_candidates, search_shape, valid_gap=0.005):
    """
    Official PS rule (Section 4.A, 'Multiple matches'): "If several valid
    matches exist, select the one whose centre is closest to the
    search-image centre." Our ranking so far only orders by match quality;
    this applies the explicit tie-break.

    'Valid' here = any candidate within `valid_gap` of the top score (i.e.
    candidates that are effectively tied and therefore genuinely ambiguous,
    not just the single best score). Among those, pick the one nearest the
    image center. If there's a clear single winner (no other candidate
    within the gap), that winner is returned unchanged.
    """
    if not ranked_candidates:
        return None
    top_score = ranked_candidates[0]["final_score"]
    valid = [c for c in ranked_candidates if (top_score - c["final_score"]) <= valid_gap]

    if len(valid) == 1:
        return ranked_candidates[0]

    H, W = search_shape[:2]
    cx, cy = W / 2.0, H / 2.0
    valid.sort(key=lambda c: (c["x"] - cx) ** 2 + (c["y"] - cy) ** 2)
    return valid[0]
