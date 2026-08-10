"""
generator.py
------------
Produces (reference_image, search_image, ground_truth) triplets that match
the PS spec exactly:

  - Reference image: 1000x1000, high-magnification view of a DRAM-style or
    FinFET-style layout.
  - Search image: 1000x1000, lower-magnification (10x) view. The reference
    pattern is embedded, downsampled ~10x, at a KNOWN random location
    (giving a ~100x100 px inset), matching the PS description for both
    architecture styles.
  - Ground truth: exact center (x, y) of the embedded inset in the search
    image, saved as JSON, plus the bounding box and generation parameters
    (for reproducibility / debugging).

Independence & realism requirements satisfied here:
  - Reference and search images get INDEPENDENTLY sampled sensor noise
    (different RNG seeds per image, see dataset/degradation.py).
  - Edge brightening is applied to both.
  - Search image has MORE noise than reference (matches PS: "search image
    will have more noise... test data will be more noisy than training").
  - A small rotation (1-3 deg) and slight scale jitter is applied to the
    embedded inset only, before pasting, so the match is not pixel-perfect
    -- this is what makes classical matching non-trivial without requiring
    large geometric search ranges (PS explicitly says don't over-engineer
    for large rotation/scale).
  - At least one run can force a "hard" periodic region (see generate_pairs
    `hard_fraction`) to satisfy the "genuinely difficult" localization test
    case requirement.
"""

import os
import json
import numpy as np
import cv2

from layouts import generate_canvas
from degradation import degrade_reference, degrade_search

REF_SIZE = 1000
SEARCH_SIZE = 1000
SCALE_FACTOR = 10          # reference appears shrunk ~10x inside search image
INSET_SIZE = REF_SIZE // SCALE_FACTOR   # ~100x100, matches PS spec


def _make_clean_pair(style, seed):
    """Generate the clean (pre-degradation) reference canvas and a clean
    low-res search background using independently-phased grid parameters
    of the SAME style (different seed => different phase/jitter, which is
    realistic: a different die region has the same design rules but is not
    pixel-identical elsewhere)."""
    ref_clean = generate_canvas(style, canvas_size=REF_SIZE, seed=seed)

    # Low-res background: draw the same style at reduced pitch so visual
    # statistics match the "10x lower magnification" look, with a
    # different seed for a plausible independent die region.
    if style == "dram":
        from layouts import generate_dram_canvas
        bg = generate_dram_canvas(canvas_size=SEARCH_SIZE, pitch=8,
                                   line_width=1, via_radius=1, seed=seed + 777)
    else:
        from layouts import generate_finfet_canvas
        bg = generate_finfet_canvas(canvas_size=SEARCH_SIZE, fin_pitch=3,
                                     fin_width=1, gate_pitch=26,
                                     gate_width=3, seed=seed + 777)
    return ref_clean, bg


def _embed_reference(bg, ref_clean, x0, y0, rng, hard=False):
    """Downsample the clean reference to inset size, apply a small random
    rotation/scale (simulating the imaging misalignment the PS calls out),
    and paste it into the background at (x0, y0). Returns the composited
    clean search image and the exact ground-truth center."""
    inset = cv2.resize(ref_clean, (INSET_SIZE, INSET_SIZE), interpolation=cv2.INTER_AREA)

    angle = rng.uniform(-3, 3)      # PS: rotation limited to 1-3 degrees
    scale = 1.0 + rng.uniform(-0.03, 0.03)   # small scale jitter only
    M = cv2.getRotationMatrix2D((INSET_SIZE / 2, INSET_SIZE / 2), angle, scale)
    inset_transformed = cv2.warpAffine(inset, M, (INSET_SIZE, INSET_SIZE),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)

    out = bg.copy()
    out[y0:y0 + INSET_SIZE, x0:x0 + INSET_SIZE] = inset_transformed

    if hard:
        # Deliberately duplicate the same inset pattern (untransformed) at
        # a second nearby location to create a genuinely ambiguous /
        # periodic case, as required by the PS ("at least one highly
        # periodic array region where correct localization is genuinely
        # difficult").
        dx = rng.choice([-1, 1]) * rng.integers(120, 220)
        dy = rng.choice([-1, 1]) * rng.integers(120, 220)
        x1 = int(np.clip(x0 + dx, 0, SEARCH_SIZE - INSET_SIZE))
        y1 = int(np.clip(y0 + dy, 0, SEARCH_SIZE - INSET_SIZE))
        out[y1:y1 + INSET_SIZE, x1:x1 + INSET_SIZE] = inset  # untransformed decoy

    center = (x0 + INSET_SIZE / 2.0, y0 + INSET_SIZE / 2.0)
    return out, center, angle, scale


def generate_pair(style, seed, out_dir, pair_id, heavier_noise=False, hard=False):
    rng = np.random.default_rng(seed)
    ref_clean, bg_clean = _make_clean_pair(style, seed)

    margin = 20
    x0 = int(rng.integers(margin, SEARCH_SIZE - INSET_SIZE - margin))
    y0 = int(rng.integers(margin, SEARCH_SIZE - INSET_SIZE - margin))

    search_clean, center, angle, scale = _embed_reference(bg_clean, ref_clean, x0, y0, rng, hard=hard)

    ref_final = degrade_reference(ref_clean, seed=seed * 7 + 1)
    search_final = degrade_search(search_clean, seed=seed * 13 + 2, heavier=heavier_noise)

    ref_path = os.path.join(out_dir, "reference", f"{style}_{pair_id:03d}_ref.png")
    search_path = os.path.join(out_dir, "search", f"{style}_{pair_id:03d}_search.png")
    cv2.imwrite(ref_path, ref_final)
    cv2.imwrite(search_path, search_final)

    gt = {
        "pair_id": pair_id,
        "style": style,
        "reference_path": os.path.relpath(ref_path, out_dir),
        "search_path": os.path.relpath(search_path, out_dir),
        "center_x": center[0],
        "center_y": center[1],
        "bbox": [x0, y0, x0 + INSET_SIZE, y0 + INSET_SIZE],
        "inset_size": INSET_SIZE,
        "scale_factor": SCALE_FACTOR,
        "applied_rotation_deg": angle,
        "applied_scale": scale,
        "heavier_noise": heavier_noise,
        "hard_case": hard,
        "seed": seed,
    }

    gt_path = os.path.join(out_dir, "annotations", f"{style}_{pair_id:03d}.json")
    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)

    return gt


def generate_dataset(out_dir, n_pairs=30, hard_fraction=0.15, heavier_fraction=0.3, base_seed=42):
    os.makedirs(os.path.join(out_dir, "reference"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "search"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "annotations"), exist_ok=True)

    records = []
    for i in range(n_pairs):
        style = "dram" if i % 2 == 0 else "finfet"
        seed = base_seed + i
        hard = (i / n_pairs) >= (1 - hard_fraction)
        heavier = ((i * 7) % 10) < (heavier_fraction * 10)
        gt = generate_pair(style, seed, out_dir, i, heavier_noise=heavier, hard=hard)
        records.append(gt)
        print(f"[{i+1}/{n_pairs}] {style:6s} center=({gt['center_x']:.1f},{gt['center_y']:.1f}) "
              f"hard={hard} heavier_noise={heavier}")

    with open(os.path.join(out_dir, "annotations", "all_ground_truth.json"), "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nDone. {n_pairs} pairs written to {out_dir}")
    return records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="../outputs")
    parser.add_argument("--n_pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(args.out_dir, n_pairs=args.n_pairs, base_seed=args.seed)
