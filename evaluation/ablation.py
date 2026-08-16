#!/usr/bin/env python3
"""
ablation.py
-----------
Official requirement (PS Section D): "Results across multiple noise levels,
target positions, scales and rotations."

Generates small, controlled batches of pairs varying ONE factor at a time
(rotation, scale, noise) and reports pass rate on each, so we can show
robustness trends rather than a single aggregate number.
"""

import os
import sys
import csv

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "localization"))

import numpy as np
import cv2
from layouts import generate_canvas
from degradation import degrade_reference, degrade_search
from inference import localize

REF_SIZE = 1000
SEARCH_SIZE = 1000
INSET_SIZE = 100


def make_pair(style, seed, rotation_deg, scale, noise_level, out_dir, tag, position="random"):
    rng = np.random.default_rng(seed)
    ref_clean = generate_canvas(style, canvas_size=REF_SIZE, seed=seed)
    if style == "dram":
        from layouts import generate_dram_canvas
        bg = generate_dram_canvas(canvas_size=SEARCH_SIZE, pitch=8, line_width=1,
                                   via_radius=1, seed=seed + 777)
    else:
        from layouts import generate_finfet_canvas
        bg = generate_finfet_canvas(canvas_size=SEARCH_SIZE, fin_pitch=3, fin_width=1,
                                     gate_pitch=26, gate_width=3, seed=seed + 777)

    margin = 20
    if position == "center":
        x0 = SEARCH_SIZE // 2 - INSET_SIZE // 2
        y0 = SEARCH_SIZE // 2 - INSET_SIZE // 2
    elif position == "edge":
        x0 = margin
        y0 = SEARCH_SIZE // 2 - INSET_SIZE // 2
    elif position == "corner":
        x0 = SEARCH_SIZE - INSET_SIZE - margin
        y0 = SEARCH_SIZE - INSET_SIZE - margin
    else:  # random
        x0 = int(rng.integers(margin, SEARCH_SIZE - INSET_SIZE - margin))
        y0 = int(rng.integers(margin, SEARCH_SIZE - INSET_SIZE - margin))

    inset = cv2.resize(ref_clean, (INSET_SIZE, INSET_SIZE), interpolation=cv2.INTER_AREA)
    M = cv2.getRotationMatrix2D((INSET_SIZE / 2, INSET_SIZE / 2), rotation_deg, scale)
    inset_t = cv2.warpAffine(inset, M, (INSET_SIZE, INSET_SIZE), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    search_clean = bg.copy()
    search_clean[y0:y0 + INSET_SIZE, x0:x0 + INSET_SIZE] = inset_t

    ref_final = degrade_reference(ref_clean, seed=seed * 7 + 1)
    search_final = degrade_search(search_clean, seed=seed * 13 + 2, heavier=(noise_level == "high"))

    ref_path = os.path.join(out_dir, f"{tag}_ref.png")
    search_path = os.path.join(out_dir, f"{tag}_search.png")
    cv2.imwrite(ref_path, ref_final)
    cv2.imwrite(search_path, search_final)

    gt_center = (x0 + INSET_SIZE / 2.0, y0 + INSET_SIZE / 2.0)
    return ref_path, search_path, gt_center


def run_ablation(out_dir="../outputs/ablation", n_seeds=5):
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    # PS Section D: "Results across multiple noise levels, target positions,
    # scales and rotations." Each factor is varied independently against a
    # fixed baseline (rotation=0, scale=1.0, noise=low, position=center) so
    # the effect of each factor is isolated and interpretable.
    factors = {
        "rotation_deg": [-3, -2, -1, 0, 1, 2, 3],   # symmetric; our search covers +-4deg
        "scale": [0.90, 0.95, 1.0, 1.05, 1.10],     # PS: "approximately 9:1 to 11:1" -> +-10%
        "noise_level": ["low", "high"],
        "position": ["center", "edge", "corner", "random"],
    }

    base_rotation, base_scale, base_noise, base_position = 0, 1.0, "low", "center"

    for factor_name, values in factors.items():
        for val in values:
            for seed_i in range(n_seeds):
                seed = 5000 + seed_i * 31
                rotation = val if factor_name == "rotation_deg" else base_rotation
                scale = val if factor_name == "scale" else base_scale
                noise = val if factor_name == "noise_level" else base_noise
                position = val if factor_name == "position" else base_position
                style = "dram" if seed_i % 2 == 0 else "finfet"

                tag = f"{factor_name}_{val}_{style}_{seed_i}"
                ref_path, search_path, gt = make_pair(style, seed, rotation, scale, noise,
                                                        out_dir, tag, position=position)
                pred = localize(ref_path, search_path)
                err = ((pred["x"] - gt[0]) ** 2 + (pred["y"] - gt[1]) ** 2) ** 0.5

                rows.append({
                    "factor": factor_name, "value": val, "style": style, "seed": seed,
                    "position": position, "error_px": round(err, 2), "correct_5px": err <= 5,
                    "confidence": pred["confidence"], "ambiguous": pred["ambiguous"],
                })
                print(f"{factor_name}={val} style={style} seed={seed} err={err:.1f}px "
                      f"pass5px={err<=5}")

    return rows


def summarize(rows, out_csv):
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== ABLATION SUMMARY (pass rate @5px) ===")
    factors = sorted(set(r["factor"] for r in rows))
    for factor in factors:
        values = sorted(set(r["value"] for r in rows if r["factor"] == factor), key=str)
        for val in values:
            subset = [r for r in rows if r["factor"] == factor and r["value"] == val]
            n_pass = sum(1 for r in subset if r["correct_5px"])
            print(f"  {factor:15s} = {str(val):6s}: {n_pass}/{len(subset)} pass "
                  f"({n_pass/len(subset)*100:.0f}%)")


if __name__ == "__main__":
    rows = run_ablation(n_seeds=2)
    summarize(rows, "ablation_results.csv")
