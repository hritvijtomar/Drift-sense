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
from template_matching import multiscale_ncc_candidates
from feature_matcher import rank_candidates, is_ambiguous


def localize(reference_path, search_path, topk=5, verbose=False):
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
    best = ranked[0]
    ambiguous = is_ambiguous(ranked)

    if verbose:
        for i, c in enumerate(ranked):
            print(f"  candidate {i}: (x={c['x']:.1f}, y={c['y']:.1f}) "
                  f"ncc={c['score']:.3f} orb_inliers={c['orb_inliers']} "
                  f"final={c['final_score']:.3f}", file=sys.stderr)

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
    args = parser.parse_args()

    result = localize(args.reference, args.search, verbose=args.verbose)
    print(json.dumps(result, indent=2))

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
