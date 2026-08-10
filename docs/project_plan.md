# Project Plan — Drift-Sense (Applied Materials PS)

## ✅ Milestone 0 — Setup
- [x] Read PS + submission requirements
- [x] Repository created and structured

## ✅ Milestone 1 — Dataset generator
- [x] DRAM-style layout generator
- [x] FinFET-style layout generator
- [x] Reference/search pair generation with exact ground truth
- [x] Independent sensor noise per image
- [x] Edge brightening
- [x] Blur, rotation (1-3°), scale (~10x) degradation
- [x] 30-pair dataset generated
- [x] At least one "hard" periodic-ambiguity case included (pairs 27-29)

## ✅ Milestone 2 — Localization
- [x] Multi-scale/rotation NCC baseline (candidate generator)
- [x] ORB + RANSAC geometric verification (disambiguation)
- [x] Confidence score + ambiguity flag
- [x] Standalone inference.py (reference path, search path -> x,y)

## ✅ Milestone 3 — Evaluation
- [x] Accuracy-within-tolerance metric
- [x] Runtime measurement
- [x] Auto-generated success + honest failure case doc

## ☐ Milestone 4 — Submission polish (DO THIS NEXT)
- [ ] Fill Idea Submission Template PPT (slides 1-9)
- [ ] Add pipeline diagram to Slide 4
- [ ] Add success/failure visual comparison to Slide 6
- [ ] Push repo to GitHub (public), verify fresh-clone run works
- [ ] Record short demo video (optional but recommended)
- [ ] Fill in team details, references slide from docs/citations.md

## Known limitations (be upfront about these to judges)
- DRAM accuracy (20%) is much lower than FinFET (73%) due to inherent
  periodicity with no macro landmark — documented in docs/failure_analysis.md.
- Classical pipeline only; no deep learning (explicit, justified scope decision).
