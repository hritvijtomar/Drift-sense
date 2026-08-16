#!/usr/bin/env python3
"""
diagnose_dram.py
-----------------
For every failed DRAM pair, checks:
  1. Is the ground-truth location even among the top-5 NCC candidates?
     (If not, the fix belongs in candidate generation, not ranking.)
  2. If yes, what rank did it get, and by how much did the wrong
     candidate win (score margin)?
  3. What do NCC score and ORB inliers look like for the true location
     vs. the wrongly-picked one?
"""
import sys, os, json, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "localization"))
import cv2
from template_matching import multiscale_ncc_candidates
from feature_matcher import rank_candidates

results = json.load(open("../evaluation/results.json"))
dram_fails = [r for r in results if r["style"] == "dram" and not r["correct"]]

TOL = 15  # candidate counts as "the true one" if within this many px of GT

summary = {"gt_not_in_candidates": 0, "gt_in_candidates_but_not_top": 0}

for r in dram_fails:
    ref = cv2.imread(r["reference_path"], cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(r["search_path"], cv2.IMREAD_GRAYSCALE)
    candidates = multiscale_ncc_candidates(ref, search, topk=5)
    ranked = rank_candidates(ref, search, candidates)

    gt_x, gt_y = r["gt_x"], r["gt_y"]
    dists = [math.hypot(c["x"] - gt_x, c["y"] - gt_y) for c in ranked]
    true_idx = None
    for i, d in enumerate(dists):
        if d <= TOL:
            true_idx = i
            break

    print(f"\npair {r['pair_id']:03d} (error={r['error_px']:.0f}px)")
    for i, c in enumerate(ranked):
        marker = " <-- TRUE" if (true_idx == i) else (" <-- PICKED" if i == 0 else "")
        print(f"  rank{i}: ncc={c['score']:.3f} orb_inliers={c['orb_inliers']:3d} "
              f"final={c['final_score']:.3f} dist_to_gt={dists[i]:6.1f}px{marker}")

    if true_idx is None:
        summary["gt_not_in_candidates"] += 1
        print("  >>> GT NOT in top-5 candidates at all (candidate-generation miss)")
    elif true_idx != 0:
        summary["gt_in_candidates_but_not_top"] += 1
        print(f"  >>> GT was candidate rank {true_idx}, lost to rank 0 by "
              f"{ranked[0]['final_score']-ranked[true_idx]['final_score']:.3f} score margin")

print("\n=== SUMMARY ===")
print(json.dumps(summary, indent=2))
print(f"Total DRAM failures analyzed: {len(dram_fails)}")
