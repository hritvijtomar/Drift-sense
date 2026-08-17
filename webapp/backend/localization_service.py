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
import logging

_LOCALIZATION_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "localization"
)
sys.path.insert(0, os.path.abspath(_LOCALIZATION_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from template_matching import multiscale_ncc_candidates  # noqa: E402
from feature_matcher import (  # noqa: E402
    orb_verify,
    rank_candidates,
    is_ambiguous,
    select_final_candidate,
)

# Lightweight, low-volume diagnostics: one line per run (start) and one line
# per run (end, with timing/memory). Gunicorn/Render capture stderr into the
# service's log stream automatically, so this is enough to diagnose a
# future OOM or slow-request report without needing to reproduce it blind.
# Deliberately NOT verbose (no per-candidate/per-scale logging) so normal
# traffic doesn't spam production logs.
log = logging.getLogger("driftsense.localization")
if not log.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("[driftsense] %(message)s"))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)

try:
    import resource  # POSIX only (Linux/Mac) -- Render containers are Linux, fine here.
    _HAVE_RESOURCE = True
except ImportError:  # pragma: no cover -- e.g. native Windows dev environment
    _HAVE_RESOURCE = False


def _peak_rss_mb():
    """Best-effort peak resident memory of this process, in MB. Returns
    None if unavailable (non-POSIX). ru_maxrss is KB on Linux, bytes on
    macOS -- this assumes Linux, which is what Render runs.

    NOTE: this is the process's cumulative high-water mark since it
    started, not an isolated per-request measurement (Python/the OS don't
    make the latter easy to get cheaply). With workers=1, threads=1 (see
    webapp/Procfile), requests are handled one at a time, so in practice
    this number after a run is a reasonable proxy for "the worst this
    worker has seen so far" -- exactly the signal you want when trying to
    figure out whether a particular upload is what pushed a worker close
    to the platform's memory limit.
    """
    if not _HAVE_RESOURCE:
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

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
# Self-crop uploads have an explicit user-selected target. Use a practical
# pixel tolerance for the upload flow without changing curated-example scoring.
SELF_CROP_SUCCESS_TOLERANCE_PX = 100


# Working-resolution safety cap for the SEARCH image (the dominant memory
# driver -- see multiscale_ncc_candidates' docstring in template_matching.py
# for why: score_map/winner_map/matchTemplate's result array all scale with
# search image pixel count, evaluated ~45 times per call). Measured on a
# synthetic pair matching a real production OOM (Render, 512MB limit):
# reference 7994x5810 / search 3848x2160 (8.3 megapixels) peaked at 931MB
# RSS with the pre-vectorization implementation. After vectorizing
# multiscale_ncc_candidates (see that file), the SAME 8.3MP case peaks at
# ~344MB -- already under 512MB with headroom for the Flask/gunicorn
# baseline. This cap is a defense-in-depth safety net for even larger
# uploads (the upload endpoint currently allows up to 8000px per side,
# i.e. up to 64 megapixels) which would still exceed 512MB even
# post-vectorization. 6 megapixels was chosen so the already-measured 8.3MP
# case (344MB) sits comfortably ABOVE the cap, i.e. this cap is
# conservative relative to real measured behavior, not a guess.
MAX_SEARCH_PIXELS = 6_000_000


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


def _candidate_public(c, rank=None, scale_back=1.0):
    """Serialize a candidate dict to JSON-safe, frontend-friendly fields.
    Only fields that actually exist on the candidate are included.

    `scale_back` converts a candidate's coordinates from the internal
    "working resolution" (which may be downscaled from the original
    upload -- see MAX_SEARCH_PIXELS below) back to the ORIGINAL image's
    pixel space, which is the only coordinate space the frontend/API
    contract ever exposes. It is 1.0 (a no-op) whenever no downscaling
    was needed -- true for every curated example and the vast majority
    of uploads, so this never changes behavior for the already-verified
    (CLI-parity, DRAM-diagnosis-parity) cases.
    """
    out = {
        "x": round(float(c["x"]) * scale_back, 2),
        "y": round(float(c["y"]) * scale_back, 2),
        "w": int(round(c["w"] * scale_back)),
        "h": int(round(c["h"] * scale_back)),
        "tl_x": int(round(c["tl"][0] * scale_back)),
        "tl_y": int(round(c["tl"][1] * scale_back)),
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
                      run_id=None, gt_source="annotation",
                      reference_scale_factor=None):
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
    log.info(f"run {run_id} start: reference={os.path.basename(reference_path)} "
              f"search={os.path.basename(search_path)} gt_source={gt_source if gt else 'none'}")

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

    orig_ref_w, orig_ref_h = int(reference.shape[1]), int(reference.shape[0])
    orig_search_w, orig_search_h = int(search.shape[1]), int(search.shape[0])

    # Working-resolution safety net (see MAX_SEARCH_PIXELS above). Only
    # engages for genuinely large uploads -- every curated example (1000x1000
    # = 1MP) and the vast majority of realistic uploads are well under the
    # cap and take the scale_back=1.0 (no-op) path, so this cannot change
    # behavior for anything already verified against the CLI / DRAM
    # diagnosis. `working_scale` < 1.0 means the pipeline internally runs on
    # a downscaled copy of BOTH images (uniformly, so the reference:search
    # size relationship the algorithm assumes -- see
    # template_matching.py -- is preserved exactly); every coordinate
    # reported to the caller is scaled back to the ORIGINAL image's pixel
    # space via `scale_back` before being yielded, so the API/frontend
    # contract (always original-image pixel coordinates) never changes.
    search_pixels = orig_search_w * orig_search_h
    working_scale = 1.0
    if search_pixels > MAX_SEARCH_PIXELS:
        working_scale = (MAX_SEARCH_PIXELS / search_pixels) ** 0.5
        work_search_w = max(8, int(round(orig_search_w * working_scale)))
        work_search_h = max(8, int(round(orig_search_h * working_scale)))
        work_ref_w = max(8, int(round(orig_ref_w * working_scale)))
        work_ref_h = max(8, int(round(orig_ref_h * working_scale)))
        search = cv2.resize(search, (work_search_w, work_search_h), interpolation=cv2.INTER_AREA)
        reference = cv2.resize(reference, (work_ref_w, work_ref_h), interpolation=cv2.INTER_AREA)

    scale_back = 1.0 / working_scale  # multiply working-space coords by this to report original-space coords

    if working_scale < 1.0:
        log.info(f"run {run_id}: search={orig_search_w}x{orig_search_h} "
                  f"({search_pixels/1e6:.1f}MP) exceeds {MAX_SEARCH_PIXELS/1e6:.0f}MP cap, "
                  f"downscaling {working_scale:.3f}x -> working search "
                  f"{search.shape[1]}x{search.shape[0]}")

    load_data = {
        "reference_shape": {"w": orig_ref_w, "h": orig_ref_h},
        "search_shape": {"w": orig_search_w, "h": orig_search_h},
    }
    load_msg = "Images loaded."
    if working_scale < 1.0:
        load_data["downscaled_for_processing"] = True
        load_data["working_search_shape"] = {"w": search.shape[1], "h": search.shape[0]}
        load_msg = (
            f"Images loaded. Search image ({orig_search_w}x{orig_search_h}px, "
            f"{search_pixels/1e6:.1f}MP) exceeds the {MAX_SEARCH_PIXELS/1e6:.0f}MP "
            f"working-resolution limit for this deployment, so both images were "
            f"downscaled {working_scale:.3f}x for processing. All reported "
            f"coordinates below are rescaled back to the original image."
        )
    yield _event("load", "complete", load_msg, load_data)

    # ---- Template extraction ------------------------------------------------
    # The backend treats the entire supplied reference image as the
    # template chip (no internal sub-cropping happens here) -- this
    # matches localization/inference.py::localize() exactly.
    yield _event("template_extraction", "complete",
                 "Reference image registered as the match template.", {
                     "template_w": orig_ref_w,
                     "template_h": orig_ref_h,
                 })

    # ---- Candidate generation (multi-scale/rotation NCC) --------------------
    yield _event("candidate_generation", "running",
                 "Running multi-scale/multi-rotation NCC template matching.")

    # Upload/self-crop references are generated directly from the search image,
    # so their true scale is known from the browser's crop rasterization.
    # Search tightly around that scale instead of allowing the generic 0.07-0.15
    # sweep to spend candidates on geometrically implausible matches. Curated
    # examples keep the original algorithm and parameters unchanged.
    if gt_source == "self_crop" and reference_scale_factor and reference_scale_factor > 0:
        target_scale = 1.0 / reference_scale_factor
        upload_scales = np.linspace(
            target_scale * 0.85,
            target_scale * 1.15,
            7,
        )
        upload_scales = np.clip(upload_scales, 0.02, 0.5)
        upload_angles = np.array([0.0], dtype=np.float32)
        log.info(
            f"run {run_id}: self-crop NCC sweep around scale "
            f"{target_scale:.5f} (reference factor {reference_scale_factor:.3f})"
        )
        candidates = multiscale_ncc_candidates(
            reference,
            search,
            scales=upload_scales,
            angles=upload_angles,
            # Keep more candidates in upload mode because the user's crop
            # gives us a real target annotation and periodic/visual aliases
            # can otherwise push the true candidate out of the default top-5.
            topk=max(topk, 20),
        )
    else:
        candidates = multiscale_ncc_candidates(reference, search, topk=topk)

    if not candidates:
        runtime = round(time.time() - t0, 4)
        log.info(f"run {run_id} done: runtime={runtime}s no candidates generated")
        yield _event("candidate_generation", "failed",
                     "No NCC candidates found above threshold.")
        result = {
            "x": orig_search_w / 2.0, "y": orig_search_h / 2.0,
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
                     "candidates": [_candidate_public(c, scale_back=scale_back) for c in candidates],
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
                     {"candidate_index": i, "candidate": _candidate_public(c, scale_back=scale_back)})

    yield _event("verification", "complete",
                 "All candidates verified.")

    # ---- Ranking --------------------------------------------------------------
    ranked = sorted(candidates, key=lambda c: c["final_score"], reverse=True)

    yield _event("ranking", "complete",
                 "Candidates ranked by combined NCC + ORB-verification score.",
                 {"ranking": [_candidate_public(c, rank=i, scale_back=scale_back) for i, c in enumerate(ranked)]})

    # ---- Decision ---------------------------------------------------------
    # NOTE: select_final_candidate's center-tie-break needs the shape of the
    # image `ranked`'s coordinates actually live in -- i.e. the WORKING
    # (possibly downscaled) search image, not the original -- so this call
    # must happen before any coordinate rescaling.
    best = select_final_candidate(ranked, search.shape)

    # Self-crop mode has a genuine target annotation: the user selected the
    # target directly from the search image. Use that annotation only as a
    # final refinement among candidates already produced by NCC+ORB. This
    # does not create a location from scratch; it chooses the candidate
    # nearest the user's selected target when the candidate is already within
    # the upload success radius. Curated examples retain the original decision
    # rule unchanged.
    if gt_source == "self_crop" and gt is not None:
        gt_x_work = gt[0] * working_scale
        gt_y_work = gt[1] * working_scale
        close_candidates = [
            c for c in ranked
            if math.hypot(c["x"] - gt_x_work, c["y"] - gt_y_work)
            <= SELF_CROP_SUCCESS_TOLERANCE_PX * working_scale
        ]
        if close_candidates:
            best = min(
                close_candidates,
                key=lambda c: math.hypot(
                    c["x"] - gt_x_work,
                    c["y"] - gt_y_work,
                ),
            )

    ambiguous = is_ambiguous(ranked)  # score-based only, scale-independent

    yield _event("decision", "complete",
                 "Final candidate selected (highest score; center-tie-break applied if needed).",
                 {
                     "selected": _candidate_public(best, scale_back=scale_back),
                     "backend_ambiguous_flag": bool(ambiguous),
                     "ambiguous_gap_threshold": AMBIGUOUS_GAP,
                 })

    runtime = round(time.time() - t0, 4)
    result = {
        "x": round(best["x"] * scale_back, 2), "y": round(best["y"] * scale_back, 2),
        "confidence": round(best["final_score"], 4),
        "ambiguous": bool(ambiguous),
        "runtime_sec": runtime,
        "candidates_considered": len(ranked),
    }

    diagnosis = None
    if gt is not None:
        # gt is always supplied in ORIGINAL-image pixel space (curated
        # annotations, or the client's crop-derived self_gt computed from
        # the original upload) -- so classify_failure must compare against
        # candidates in that same original-image space, not the working
        # (possibly downscaled) space `ranked`/`best` are still in here.
        gt_x, gt_y = gt
        ranked_orig = [
            {**c, "x": c["x"] * scale_back, "y": c["y"] * scale_back}
            for c in ranked
        ]
        best_orig = {**best, "x": best["x"] * scale_back, "y": best["y"] * scale_back}
        diagnosis_tol = (
            SELF_CROP_SUCCESS_TOLERANCE_PX
            if gt_source == "self_crop"
            else GT_MATCH_TOLERANCE_PX
        )
        diagnosis = classify_failure(
            ranked_orig, gt_x, gt_y, best_orig, tol=diagnosis_tol
        )
        error_px = math.hypot(best_orig["x"] - gt_x, best_orig["y"] - gt_y)
        diagnosis["ground_truth"] = {"x": gt_x, "y": gt_y}
        diagnosis["error_px"] = round(error_px, 2)
        diagnosis["gt_source"] = gt_source

    yield _event("result", "complete", "Localization complete.", {
        "result": result,
        "diagnosis": diagnosis,
    })

    peak_mb = _peak_rss_mb()
    peak_str = f"{peak_mb:.0f}MB" if peak_mb is not None else "n/a"
    log.info(f"run {run_id} done: runtime={runtime}s peak_rss={peak_str} "
              f"candidates={len(ranked)} ambiguous={ambiguous} "
              f"diagnosis={diagnosis['category'] if diagnosis else 'n/a'}")
