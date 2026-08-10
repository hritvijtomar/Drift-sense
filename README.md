# Drift-Sense: Navigation-Error Recovery
### Applied Materials Problem Statement — i4C Semicon Hackathon

Localizes a small SEM reference patch (high magnification) inside a larger,
noisier, lower-magnification search image of a DRAM-style or FinFET-style
die layout, given only ~1-3° rotation and ~10x known scale between the two.

## What's in this repo

```
Applied-Materials-Hackathon/
├── dataset/
│   ├── layouts.py          # DRAM-style / FinFET-style clean layout generation
│   ├── degradation.py       # SEM-realistic noise/blur/edge-brightening pipeline
│   └── generator.py         # Builds (reference, search, ground_truth) triplets
├── localization/
│   ├── template_matching.py # Multi-scale/rotation NCC baseline (candidate generator)
│   ├── feature_matcher.py   # ORB + RANSAC verification -> disambiguates periodic matches
│   └── inference.py         # STANDALONE script: reference path, search path -> (x, y)
├── evaluation/
│   ├── evaluate.py          # Runs inference on all generated pairs, scores accuracy
│   ├── results.json         # Per-pair predictions (generated)
│   └── summary.json         # Aggregate accuracy/runtime (generated)
├── docs/
│   ├── citations.md         # Every augmentation/noise choice, justified with references
│   ├── failure_analysis.md  # Auto-generated success + honest failure case
│   └── project_plan.md      # Milestone tracker
├── outputs/                 # Generated dataset (reference/, search/, annotations/)
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Generate the synthetic dataset (min. 30 pairs, per PS)

```bash
cd dataset
python3 generator.py --out_dir ../outputs --n_pairs 30 --seed 42
```

Produces `outputs/reference/*.png`, `outputs/search/*.png`, and
`outputs/annotations/*.json` (ground truth center + bbox for every pair).

## 2. Run localization on a single pair (the script AM will benchmark)

```bash
cd localization
python3 inference.py --reference ../outputs/reference/dram_000_ref.png \
                      --search ../outputs/search/dram_000_search.png
```

Outputs JSON: `{x, y, confidence, ambiguous, runtime_sec, candidates_considered}`.
This script takes exactly two path arguments and prints/returns a single
(x, y) — no manual edits needed, works standalone on a fresh checkout.

## 3. Run full evaluation (accuracy + failure analysis)

```bash
cd evaluation
python3 evaluate.py
```

Writes `results.json`, `summary.json`, and regenerates
`../docs/failure_analysis.md`.

## Approach summary

1. **Dataset generator** builds a shared "clean layout" (same pitch/style
   parameters) at two scales: a high-res reference and a low-res search
   background, then embeds a slightly rotated/rescaled copy of the
   reference into the background at a known location — giving exact
   ground truth by construction. Reference and search images each get
   independently-seeded sensor noise, edge brightening, and blur, applied
   in a physically-motivated order (see `docs/citations.md`).

2. **Localization** is a two-stage classical pipeline:
   - *Candidate generation*: multi-scale (0.07–0.15), multi-rotation
     (±4°) normalized cross-correlation (`cv2.matchTemplate`), returning
     the top-5 non-overlapping peaks (not just the single best — the
     pattern is periodic, so there are usually several strong peaks).
   - *Geometric verification*: for each NCC candidate, ORB keypoints +
     RANSAC affine estimation checks whether the local structure is
     actually geometrically consistent with the reference, not just
     intensity-correlated. This is what disambiguates a true match from a
     periodic-pitch alias.
   - Final confidence = weighted NCC score + ORB inlier fraction. If the
     top-2 candidates are within a small score gap, the result is flagged
     `ambiguous: true` instead of guessing silently.

3. **Evaluation**: accuracy = % of predictions within 50px (half the
   inset size) of ground truth, on our own 30 generated pairs.

## Results (own 30-pair test set, seed=42)

| Metric | Value |
|---|---|
| Overall accuracy (≤50px) | **46.7%** (14/30) |
| FinFET-style accuracy | **73.3%** (11/15) |
| DRAM-style accuracy | **20.0%** (3/15) |
| Mean localization error | 231 px |
| Mean inference time | ~3.3 s/pair (CPU) |

**Why the DRAM/FinFET gap exists (see `docs/failure_analysis.md` for full
detail):** FinFET layouts have sparse macro-scale landmarks (gate
crossings, spaced ~260px apart at reference scale) that break periodicity
locally, giving both NCC and ORB something distinctive to lock onto. DRAM
layouts are a uniform via grid with no macro landmark, so many locations
are genuinely, near-exactly self-similar — a fundamental limitation of any
intensity/keypoint-based method on that class of pattern, not a bug in our
pipeline. We surface this explicitly via the `ambiguous` confidence flag
rather than hiding it, per the PS's explainability requirement.

## Honest scope / what we deliberately did not do

- No deep learning: the PS explicitly allows classical methods, and given
  time constraints we prioritized a fully classical, explainable pipeline
  with a documented, evidence-backed limitation over an undertrained deep
  model. Slide 5 discusses a Siamese-network extension as future work.
- Rotation/scale search ranges are deliberately narrow (±4°, ~0.07–0.15
  scale), matching the PS's stated constraints rather than a general
  wide-range solution.
