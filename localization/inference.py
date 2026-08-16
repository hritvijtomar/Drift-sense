#!/usr/bin/env python3
"""
inference.py
------------
STANDALONE localization script. This is the file Applied Materials will
run directly on their test pairs -- it must work with no manual edits on a
fresh machine, per the submission requirements.

Usage:
    python inference.py --reference path/to/ref.png --search path/to/search.png

Output (stdout, and optionally --out_json):
    {
      "x": <float>, "y": <float>,
      "confidence": <float 0-1>,
      "ambiguous": <bool>,
      "runtime_sec": <float>,
      "candidates_considered": <int>
    }

Pipeline: multi-scale/rotation NCC template matching (cheap, gets
candidates) -> ORB + RANSAC geometric verification per candidate
(disambiguates periodic repeats) -> best candidate + confidence + ambiguity
flag.
"""

import argparse
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import cv2
from template_matching import multiscale_ncc_candidates
from feature_matcher import rank_candidates, is_ambiguous, select_final_candidate


def visualize_match(reference_path, search_gray, best, gt=None, save_path="match_visualization.png"):
    """Side-by-side reference | search-with-overlay image for slide decks
    and manual QA. Draws the predicted box in red; if ground truth (x, y)
    is supplied, draws it in green for comparison."""
    ref = cv2.imread(reference_path)
    search_draw = cv2.cvtColor(search_gray, cv2.COLOR_GRAY2BGR) if search_gray.ndim == 2 else search_gray.copy()

    x0, y0 = best["tl"]
    w, h = best["w"], best["h"]
    cv2.rectangle(search_draw, (x0, y0), (x0 + w, y0 + h), (0, 0, 255), 2)
    cv2.putText(search_draw, "predicted", (x0, max(0, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    if gt is not None:
        gx, gy = int(gt[0]), int(gt[1])
        cv2.drawMarker(search_draw, (gx, gy), (0, 200, 0), cv2.MARKER_CROSS, 16, 2)

    ref_resized = cv2.resize(ref, (w, h))
    if ref_resized.ndim == 2:
        ref_resized = cv2.cvtColor(ref_resized, cv2.COLOR_GRAY2BGR)

    h_stack = max(ref_resized.shape[0], search_draw.shape[0])
    gap = 20
    import numpy as np
    label_h = 30
    total_w = ref_resized.shape[1] + search_draw.shape[1] + gap
    canvas = np.full((h_stack + label_h, total_w, 3), 255, dtype=np.uint8)
    canvas[label_h:label_h + ref_resized.shape[0], 0:ref_resized.shape[1]] = ref_resized
    canvas[label_h:label_h + search_draw.shape[0], ref_resized.shape[1] + gap:] = search_draw
    title = "Reference (100x)  |  Search (10x) -- red=predicted box, green=ground truth"
    cv2.putText(canvas, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(save_path, canvas)
    return save_path


def localize(reference_path, search_path, topk=5, verbose=False, visualize=False,
             visualize_path="match_visualization.png", gt=None):
    t0 = time.time()

    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if reference is None:
        raise FileNotFoundError(f"Could not read reference image: {reference_path}")
    if search is None:
        raise FileNotFoundError(f"Could not read search image: {search_path}")

    candidates = multiscale_ncc_candidates(reference, search, topk=topk)
    if not candidates:
        return {
            "x": search.shape[1] / 2.0, "y": search.shape[0] / 2.0,
            "confidence": 0.0, "ambiguous": True,
            "runtime_sec": time.time() - t0, "candidates_considered": 0,
            "note": "no NCC candidates found above threshold; returning image center as fallback",
        }

    ranked = rank_candidates(reference, search, candidates)
    best = select_final_candidate(ranked, search.shape)  # applies official closest-to-center tie-break
    ambiguous = is_ambiguous(ranked)

    if verbose:
        for i, c in enumerate(ranked):
            print(f"  candidate {i}: (x={c['x']:.1f}, y={c['y']:.1f}) "
                  f"ncc={c['score']:.3f} orb_inliers={c['orb_inliers']} "
                  f"final={c['final_score']:.3f}", file=sys.stderr)

    if visualize:
        visualize_match(reference_path, search, best, gt=gt, save_path=visualize_path)

    return {
        "x": best["x"], "y": best["y"],
        "confidence": round(best["final_score"], 4),
        "ambiguous": bool(ambiguous),
        "runtime_sec": round(time.time() - t0, 4),
        "candidates_considered": len(ranked),
    }


def main():
    parser = argparse.ArgumentParser(description="Localize a reference SEM patch inside a search image.")
    parser.add_argument("--reference", required=True, help="Path to reference image")
    parser.add_argument("--search", required=True, help="Path to search image")
    parser.add_argument("--out_json", default=None, help="Optional path to write JSON result")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--visualize", action="store_true", help="Save a side-by-side match visualization")
    parser.add_argument("--visualize_out", default="match_visualization.png")
    args = parser.parse_args()

    result = localize(args.reference, args.search, verbose=args.verbose,
                       visualize=args.visualize, visualize_path=args.visualize_out)
    print(json.dumps(result, indent=2))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
