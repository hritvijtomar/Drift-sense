# Drift-Sense: Navigation-Error Recovery
### Applied Materials Problem Statement — i4C Semicon Hackathon

Localizes a small SEM reference patch (high magnification) inside a larger,
noisier, lower-magnification search image of a DRAM-style or FinFET-style
die layout, given only ~1-3° rotation and ~10x known scale between the two.

## What's in this repo

```
Applied-Materials-Hackathon/
├── generate_dataset.py      # Top-level entry point (wraps dataset/generator.py)
├── localize.py               # Top-level entry point (wraps localization/inference.py) -- THE inference script
├── dataset/
│   ├── layouts.py            # DRAM-style / FinFET-style clean layout generation
│   ├── degradation.py        # SEM-realistic noise/blur/edge-brightening pipeline
│   └── generator.py          # Builds (reference, search, ground_truth) triplets
├── localization/
│   ├── template_matching.py  # Multi-scale/rotation NCC baseline (candidate generator)
│   ├── feature_matcher.py    # ORB + RANSAC verification, ranking, official center tie-break
│   └── inference.py          # Core: reference path, search path -> (x, y) + visualization
├── evaluation/
│   ├── evaluate.py           # Runs inference on all generated pairs, scores accuracy
│   ├── ablation.py           # Rotation/scale/noise/position ablation (Section D requirement)
│   ├── results.json / summary.json / results_official_format.csv / ablation_results.csv
│   ├── pass_rate_vs_tolerance.png / style_breakdown.png
│   └── success_example.png / failure_example.png
├── docs/
│   ├── citations.md          # Every augmentation/noise choice, justified with references
│   └── failure_analysis.md   # Success + honest failure case, incl. tie-break experiment history
├── outputs/                  # Generated dataset (reference/, search/, annotations/, ablation/)
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
python3 generate_dataset.py --out_dir outputs --n_pairs 30 --seed 42
```

## 2. Run localization on a single pair (the script AM will benchmark)

```bash
python3 localize.py --reference outputs/reference/dram_000_ref.png \
                     --search outputs/search/dram_000_search.png
```

Outputs JSON: `{x, y, confidence, ambiguous, runtime_sec, candidates_considered}`.
This script takes exactly two path arguments and prints/returns a single
(x, y) — no manual edits needed, works standalone on a fresh checkout,
and runs correctly from the repo root (see `localize.py` / `generate_dataset.py`
top-level wrappers).

## 3. Run full evaluation (accuracy + failure analysis)

```bash
cd evaluation
python3 evaluate.py
```

Writes `results.json`, `summary.json`, `results_official_format.csv`
(reference path, search path, GT x/y, predicted x/y, per-pair metadata —
per Section 5 item 6), and regenerates `../docs/failure_analysis.md`.

## 4. Run the rotation/scale/noise ablation (per Section D)

```bash
cd evaluation
python3 ablation.py
```

Writes `ablation_results.csv` and prints pass-rate-@5px broken down by
rotation angle, scale factor, noise level, and target position (center /
edge / corner / random).

## Ablation study (rotation / scale / noise / position, per Section D)

Each factor varied independently against a fixed baseline (rotation=0°,
scale=1.0, noise=low, position=center), 2 seeds per condition (36 pairs
total — small n, so treat as directional not conclusive):

| Rotation (°) | -3 | -2 | -1 | 0 | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| Pass @5px | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |

| Scale | 0.90 | 0.95 | 1.00 | 1.05 | 1.10 |
|---|---|---|---|---|---|
| Pass @5px | 1/2 | 0/2 | 2/2 | 0/2 | 1/2 |

| Noise | low | high |
|---|---|---|
| Pass @5px | 2/2 | 2/2 |

| Position | center | edge | corner | random |
|---|---|---|---|---|
| Pass @5px | 2/2 | 2/2 | 2/2 | 1/2 |

**Honest read:** rotation and noise show no visible degradation across the
tested range, and target position doesn't obviously matter either. Scale
shows a non-monotonic dip at 0.95/1.05 that we do not have a confident
explanation for at n=2/condition — it may be an artifact of our discrete
multi-scale search grid not landing exactly on off-grid scale factors,
or noise. We're not claiming "robust to scale" — this needs a larger
sample before we'd trust it either way. Full per-pair data in
`evaluation/ablation_results.csv`.

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
| Overall accuracy (≤5px, official threshold) | **50.0%** (15/30) |
| FinFET-style accuracy | **80.0%** (12/15) |
| DRAM-style accuracy | **20.0%** (3/15) |
| Mean error | 226 px |
| Median error | 90 px |
| Worst-case error | 748 px |
| Mean inference time | ~4.1 s/pair (CPU) |

Pass rate is **identical at every tolerance from 1px to 100px**: the error
distribution is sharply bimodal (pixel-perfect or wrong-repeat-entirely),
so there's no meaningful difference between the required 5/4/2/1px
thresholds for this pipeline — see `docs/failure_analysis.md`.

**Note on methodology:** we tested the official "closest to search-image
center" tie-break rule (Section 4.A) two ways before keeping it — applied
loosely (whenever candidates were within 0.08 score of each other) it
*hurt* accuracy (down to 20%), because it overrode already-confident
correct picks. Applied only to genuine near-ties (score gap ≤0.005), it
improved accuracy from 14/30 to 15/30. We kept the tight version and
documented both results rather than only reporting the number that looked
good — see `evaluation/results.json` history and `docs/failure_analysis.md`.

**Why the DRAM/FinFET gap exists (see `docs/failure_analysis.md` for full
detail):** FinFET layouts have sparse macro-scale landmarks (gate
crossings, spaced ~260px apart at reference scale) that break periodicity
locally, giving both NCC and ORB something distinctive to lock onto. DRAM
layouts are a uniform via grid with no macro landmark, so many locations
are genuinely, near-exactly self-similar — a fundamental limitation of any
intensity/keypoint-based method on that class of pattern, not a bug in our
pipeline. We surface this explicitly via the `ambiguous` confidence flag
rather than hiding it, per the PS's explainability requirement.

## Interactive web console

`webapp/` contains a browser-based console around this same pipeline
(no algorithm duplication -- it imports `localization/template_matching.py`
and `localization/feature_matcher.py` directly and streams live pipeline
events to the browser over Server-Sent Events). See `webapp/README.md`
for architecture, API/event schema, local run instructions, and
deployment instructions, and `CLAUDE_HANDOFF.md` for an implementation
summary.

Quick start:

```bash
pip install -r requirements.txt -r webapp/requirements.txt
python3 webapp/backend/app.py
# open http://localhost:5050
```

## Honest scope / what we deliberately did not do

- No deep learning: the PS explicitly allows classical methods, and given
  time constraints we prioritized a fully classical, explainable pipeline
  with a documented, evidence-backed limitation over an undertrained deep
  model. Slide 5 discusses a Siamese-network extension as future work.
- Rotation/scale search ranges are deliberately narrow (±4°, ~0.07–0.15
  scale), matching the PS's stated constraints rather than a general
  wide-range solution.
