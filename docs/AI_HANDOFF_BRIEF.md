# Handoff Brief — paste this whole file into ChatGPT/Gemini to continue

You are continuing an existing, working hackathon submission. Do NOT
restart or redesign — extend what exists. Read this whole brief before
suggesting anything.

## Context
Team of 3, only I (one member) am handling this problem statement:
**Applied Materials PS — "Drift-Sense: Navigation-Error Recovery."**
Localize a small reference SEM patch inside a larger, noisier search
image (DRAM-style or FinFET-style layout), given ~1-3° rotation and
~10x known scale. Full PS text and submission requirements are attached
separately (paste the original PS + submission requirement images/text
alongside this brief).

I do not want to write code by hand. I want runnable code (from you) that
I paste and execute. Explanations should be short unless I ask for detail.

## What already exists (working, tested end-to-end)
A full repo at `Applied-Materials-Hackathon/` with:
- `dataset/layouts.py`, `dataset/degradation.py`, `dataset/generator.py`
  — generates 30 (reference, search, ground_truth) pairs, DRAM + FinFET,
  independent noise per image, edge brightening, blur/rotation/scale,
  citations tracked in `docs/citations.md`.
- `localization/template_matching.py` — multi-scale/rotation NCC baseline,
  returns top-5 candidate peaks (periodic pattern -> multiple peaks).
- `localization/feature_matcher.py` — ORB + RANSAC verification per
  candidate, combines with NCC into a final confidence score, flags
  `ambiguous: true` when top-2 candidates are too close to trust.
- `localization/inference.py` — STANDALONE script:
  `python inference.py --reference REF.png --search SEARCH.png` ->
  JSON `{x, y, confidence, ambiguous, runtime_sec}`. This is the file
  that gets benchmarked directly — never break its CLI contract.
- `evaluation/evaluate.py` — runs inference on all 30 pairs, computes
  accuracy within 50px tolerance, writes `docs/failure_analysis.md`
  automatically.

## Current measured results (own 30-pair set, seed=42)
- Overall accuracy: 46.7% (14/30) within 50px tolerance
- FinFET-style: 73.3% (11/15)
- DRAM-style: 20.0% (3/15)
- Mean runtime: ~3.3s/pair on CPU

## WHY the DRAM/FinFET gap exists (important — don't "fix" this by hiding it)
FinFET layouts have macro-scale landmarks (gate crossings every ~260px)
that break periodicity locally. DRAM layouts are a uniform via grid with
NO macro landmark, so many locations are genuinely near-identical to the
true match. This is a real, citable limitation of intensity/keypoint-based
methods on periodic textures (see `docs/citations.md` item 10), not a bug.
The `ambiguous` flag exists specifically to surface this honestly instead
of guessing — this IS our explainability story for the presentation.

## What to do next, in priority order

### 1. PPT (highest priority — do this first)
Fill the 9-slide Idea Submission Template (image attached separately)
using:
- Slide 2: background = why degraded/misregistered SEM navigation matters
  in real fab inspection (cite the PS background text).
- Slide 3: classical pipeline (NCC + ORB/RANSAC), chosen because PS allows
  classical methods and our timeline doesn't support training a reliable
  DL model from scratch.
- Slide 4: pipeline diagram — Reference+Search -> Multi-scale NCC (top-5
  candidates) -> ORB+RANSAC verification per candidate -> confidence
  score + ambiguity flag -> (x,y). Ask me to describe the diagram and
  generate it as an image, or write it as a simple flow diagram directly.
- Slide 5: the DRAM-vs-FinFET accuracy gap IS the "innovation" story —
  we don't just report accuracy, we detect and explain when the method
  can't be trusted (ambiguous flag), which most teams won't do.
- Slide 6: use `evaluation/results.json` — pick pair 2 (dram, success,
  0px error) as the success visual, pair 0 (dram, 748px error, honest
  failure) as the failure visual. Images are at
  `outputs/reference/dram_00{0,2}_ref.png` and
  `outputs/search/dram_00{0,2}_search.png`.
- Slide 7: Python 3, OpenCV 4.13, numpy 2.4, CPU only, ~3.3s/pair, no GPU
  needed, no DL model / no model weights.
- Slide 8: GitHub link (fill in once repo is pushed).
- Slide 9: pull directly from `docs/citations.md`.

### 2. Push to GitHub
```bash
cd Applied-Materials-Hackathon
git init
git add .
git commit -m "Initial working pipeline: generator + classical localization + eval"
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```
Then clone it fresh somewhere else and re-run the README's 3 steps to
confirm nothing breaks on a clean checkout — the PS explicitly says
unrunnable = unscored.

### 3. (Optional, only if time remains) Improve DRAM accuracy
Do NOT attempt this before #1 and #2 are done. If there's spare time:
- Try adding a local "distinctiveness" pre-filter: compute local variance
  of the reference pattern's jitter (already generated randomly per
  line/via in `layouts.py`) and increase NCC candidate resolution
  specifically around near-tied peaks.
- Consider phase-correlation-based sub-pixel refinement on the winning
  NCC candidate only (cheap, doesn't change the overall architecture).
- Do NOT jump to training a CNN/Siamese network unless there is at least
  1-2 full days left — no labeled external data exists, and PyTorch was
  explicitly deprioritized earlier in this project for time reasons.

### 4. Video demo (optional per PS, "recommended")
Screen-record running `inference.py` on 2-3 pairs, one success one
(honestly labeled) failure.

## Ground rules for you (the AI continuing this)
- Give me copy-pasteable commands and full file contents, not fragments.
- Don't suggest re-architecting the pipeline unless something is actually
  broken — it works and is already measured.
- Don't invent new "hidden hints" or third-party comparisons — just
  execute the plan above.
- Keep explanations short unless I explicitly ask for depth.
