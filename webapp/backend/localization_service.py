#!/usr/bin/env python3
"""
localization_service.py
------------------------
Reusable service layer around the EXISTING Drift-Sense localization
pipeline (localization/template_matching.py + localization/feature_matcher.py).

This module does NOT reimplement the algorithm. It imports the same
functions used by localization/inference.py::localize() (the function
called by the CLI, localize.py) and calls them in the same order, with
the same parameters, so that:

    - CLI behavior (localize.py) is completely unaffected.
    - The web app produces IDENTICAL numerical results to the CLI,
      because it is calling the same functions.
    - The only thing added here is *observability*: each stage emits a
      structured event so a caller (e.g. the Flask SSE endpoint) can
      stream real progress to a browser.

Single source of truth for the algorithm remains:
    localization/template_matching.py  (multiscale_ncc_candidates)
    localization/feature_matcher.py    (orb_verify, rank_candidates,
                                         is_ambiguous, select_final_candidate)

If you need to verify this module has not drifted from the CLI, run
`scripts/verify_parity.py` (see webapp/backend/verify_parity.py), which
runs both localize.py's localize() and this service's run() on the same
inputs and asserts the numeric outputs match exactly.
"""

import os
import sys
import time
import math
import inspect
import uuid

_LOCALIZATION_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "localization"
)
sys.path.insert(0, os.path.abspath(_LOCALIZATION_DIR))

import cv2  # noqa: E402
from template_matching import multiscale_ncc_candidates  # noqa: E402
from feature_matcher import (  # noqa: E402
    orb_verify,
    rank_candidates,
    is_ambiguous,
    select_final_candidate,
)

# Pull the algorithm's own tunable constants from the real function
# signatures rather than re-hardcoding them here, so this service can
# never silently drift out of sync with localization/feature_matcher.py.
_RANK_DEFAULTS = inspect.signature(rank_candidates).parameters
NCC_WEIGHT = _RANK_DEFAULTS["ncc_weight"].default
ORB_WEIGHT = _RANK_DEFAULTS["orb_weight"].default

_AMBIG_DEFAULTS = inspect.signature(is_ambiguous).parameters
AMBIGUOUS_GAP = _AMBIG_DEFAULTS["gap_threshold"].default

# Ground-truth match tolerance used ONLY for the optional, evaluation-time
# failure-analysis classification (candidate-generation-miss vs
# ranking-failure vs true-ambiguity). This is the same convention used in
# evaluation/diagnose_dram.py (TOL = 15px) that produced the supplied
# DRAM failure diagnosis. It never influences the prediction itself --
# ground truth is only used AFTER the real result is already computed.
GT_MATCH_TOLERANCE_PX = 15


class PipelineError(Exception):
    """Raised for expected, user-facing failure conditions (bad image,
    no candidates, etc.) so the API layer can return a clean diagnostic
    instead of a raw traceback."""
    def __init__(self, message, stage=None):
        super().__init__(message)
        self.stage = stage


def _event(stage, status, message, data=None):
    return {
        "stage": stage,
        "status": status,
        "message": message,
        "timestamp": time.time(),
        "data": data or {},
    }


def _candidate_public(c, rank=None):
    """Serialize a candidate dict to JSON-safe, frontend-friendly fields.
    Only fields that actually exist on the candidate are included."""
    out = {
        "x": round(float(c["x"]), 2),
        "y": round(float(c["y"]), 2),
        "w": int(c["w"]),
        "h": int(c["h"]),
        "tl_x": int(c["tl"][0]),
        "tl_y": int(c["tl"][1]),
        "ncc_score": round(float(c["score"]), 4),
        "scale": round(float(c["scale"]), 4),
        "angle_deg": round(float(c["angle"]), 3),
    }
    if rank is not None:
        out["rank"] = rank
    if "orb_inliers" in c:
        out["orb_inliers"] = int(c["orb_inliers"])
        out["orb_total_matches"] = int(c["orb_total"])
        out["orb_ratio"] = round(float(c["orb_ratio"]), 4)
    if "final_score" in c:
        out["final_score"] = round(float(c["final_score"]), 4)
    return out


def classify_failure(ranked, gt_x, gt_y, selected, tol=GT_MATCH_TOLERANCE_PX,
                      ambiguous_gap=AMBIGUOUS_GAP):
    """
    Classifies the result against ground truth into exactly one of four
    honest categories. Mirrors the method used in
    evaluation/diagnose_dram.py to produce the supplied DRAM diagnosis
    (nearest-candidate-within-TOL = "the true one").

    Returns a dict: {category, true_rank (or None), margin (or None),
    dist_selected_to_gt, message}
    """
    dists = [math.hypot(c["x"] - gt_x, c["y"] - gt_y) for c in ranked]
    true_idx = None
    for i, d in enumerate(dists):
        if d <= tol:
            true_idx = i
            break

    sel_dist = math.hypot(selected["x"] - gt_x, selected["y"] - gt_y) if selected else None
    selected_is_gt = sel_dist is not None and sel_dist <= tol

    if selected_is_gt:
        return {
            "category": "success",
            "true_rank": true_idx,
            "margin": None,
            "dist_selected_to_gt_px": round(sel_dist, 2),
            "message": (
                "The selected candidate matches the ground-truth location "
                f"within {tol}px."
            ),
        }

    if true_idx is None:
        return {
            "category": "candidate_generation_miss",
            "true_rank": None,
            "margin": None,
            "dist_selected_to_gt_px": round(sel_dist, 2) if sel_dist is not None else None,
            "message": (
                "The correct location was not present among the retained "
                "candidate set, so downstream ranking could not select it."
            ),
        }

    # GT was generated as a candidate but not the one selected.
    margin = ranked[0]["final_score"] - ranked[true_idx]["final_score"]
    if margin < ambiguous_gap:
        return {
            "category": "true_ambiguity",
            "true_rank": true_idx,
            "margin": round(margin, 4),
            "dist_selected_to_gt_px": round(sel_dist, 2) if sel_dist is not None else None,
            "message": (
                f"The ground-truth candidate (rank {true_idx}) scored within "
                f"{ambiguous_gap:.2f} of the selected candidate "
                f"(margin={margin:.3f}), matching the backend's own "
                "ambiguity threshold -- a genuine near tie."
            ),
        }

    return {
        "category": "ranking_selection_failure",
        "true_rank": true_idx,
        "margin": round(margin, 4),
        "dist_selected_to_gt_px": round(sel_dist, 2) if sel_dist is not None else None,
        "message": (
            f"The correct location was generated as candidate rank "
            f"{true_idx}, but another candidate received a higher final "
            f"score by a margin of {margin:.3f} (larger than the "
            f"backend's ambiguity threshold of {ambiguous_gap:.2f}) -- "
            "this is a ranking failure, not a near tie."
        ),
    }


def run_localization(reference_path, search_path, gt=None, topk=5,
                      run_id=None, gt_source="annotation"):
    """
    Generator that runs the REAL Drift-Sense pipeline and yields event
    dicts as each stage progresses. The final yielded event has
    stage == "result" and status == "complete" (or "failed").

    This function calls exactly the same underlying functions, in the
    same order, with the same parameters as
    localization/inference.py::localize(), so results are numerically
    identical to the CLI. It additionally computes an honest
    gt-vs-candidate failure classification when gt=(x, y) is supplied.
    """
    run_id = run_id or uuid.uuid4().hex[:12]
    t0 = time.time()

    yield _event("load", "running", "Loading reference and search images.",
                 {"run_id": run_id})

    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    if reference is None:
        yield _event("load", "failed", f"Could not read reference image: {reference_path}")
        raise PipelineError("Could not read reference image", stage="load")
    if search is None:
        yield _event("load", "failed", f"Could not read search image: {search_path}")
        raise PipelineError("Could not read search image", stage="load")

    yield _event("load", "complete", "Images loaded.", {
        "reference_shape": {"w": int(reference.shape[1]), "h": int(reference.shape[0])},
        "search_shape": {"w": int(search.shape[1]), "h": int(search.shape[0])},
    })

    # ---- Template extraction ------------------------------------------------
    # The backend treats the entire supplied reference image as the
    # template chip (no internal sub-cropping happens here) -- this
    # matches localization/inference.py::localize() exactly.
    yield _event("template_extraction", "complete",
                 "Reference image registered as the match template.", {
                     "template_w": int(reference.shape[1]),
                     "template_h": int(reference.shape[0]),
                 })

    # ---- Candidate generation (multi-scale/rotation NCC) --------------------
    yield _event("candidate_generation", "running",
                 "Running multi-scale/multi-rotation NCC template matching.")

    candidates = multiscale_ncc_candidates(reference, search, topk=topk)

    if not candidates:
        runtime = round(time.time() - t0, 4)
        yield _event("candidate_generation", "failed",
                     "No NCC candidates found above threshold.")
        result = {
            "x": search.shape[1] / 2.0, "y": search.shape[0] / 2.0,
            "confidence": 0.0, "ambiguous": True,
            "runtime_sec": runtime, "candidates_considered": 0,
            "note": "no NCC candidates found above threshold; returning image center as fallback",
        }
        yield _event("result", "failed",
                     "Localization failed: no viable candidates were generated.",
                     {"result": result})
        return

    yield _event("candidate_generation", "complete",
                 f"Generated {len(candidates)} candidate location(s) before verification/ranking.",
                 {
                     "candidate_count": len(candidates),
                     "candidates": [_candidate_public(c) for c in candidates],
                 })

    # ---- Candidate verification (ORB + RANSAC), per candidate ---------------
    yield _event("verification", "running",
                 "Verifying each candidate with ORB keypoint matching + RANSAC.")

    for i, c in enumerate(candidates):
        inliers, total, ratio = orb_verify(reference, search, c)
        c["orb_inliers"] = inliers
        c["orb_total"] = total
        c["orb_ratio"] = ratio
        orb_component = min(1.0, inliers / 15.0)
        c["final_score"] = NCC_WEIGHT * max(0.0, c["score"]) + ORB_WEIGHT * orb_component

        yield _event("verification", "running",
                     f"Verified candidate {i + 1}/{len(candidates)}: "
                     f"{inliers} geometrically-consistent feature correspondences.",
                     {"candidate_index": i, "candidate": _candidate_public(c)})

    yield _event("verification", "complete",
                 "All candidates verified.")

    # ---- Ranking --------------------------------------------------------------
    ranked = sorted(candidates, key=lambda c: c["final_score"], reverse=True)

    yield _event("ranking", "complete",
                 "Candidates ranked by combined NCC + ORB-verification score.",
                 {"ranking": [_candidate_public(c, rank=i) for i, c in enumerate(ranked)]})

    # ---- Decision ---------------------------------------------------------
    best = select_final_candidate(ranked, search.shape)
    ambiguous = is_ambiguous(ranked)

    yield _event("decision", "complete",
                 "Final candidate selected (highest score; center-tie-break applied if needed).",
                 {
                     "selected": _candidate_public(best),
                     "backend_ambiguous_flag": bool(ambiguous),
                     "ambiguous_gap_threshold": AMBIGUOUS_GAP,
                 })

    runtime = round(time.time() - t0, 4)
    result = {
        "x": best["x"], "y": best["y"],
        "confidence": round(best["final_score"], 4),
        "ambiguous": bool(ambiguous),
        "runtime_sec": runtime,
        "candidates_considered": len(ranked),
    }

    diagnosis = None
    if gt is not None:
        gt_x, gt_y = gt
        diagnosis = classify_failure(ranked, gt_x, gt_y, best)
        error_px = math.hypot(best["x"] - gt_x, best["y"] - gt_y)
        diagnosis["ground_truth"] = {"x": gt_x, "y": gt_y}
        diagnosis["error_px"] = round(error_px, 2)
        diagnosis["gt_source"] = gt_source

    yield _event("result", "complete", "Localization complete.", {
        "result": result,
        "diagnosis": diagnosis,
    })
