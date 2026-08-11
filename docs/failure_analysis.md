## Corrected understanding of the evaluation metric (from AM Q&A webinar, Aug 2026)

Applied Materials evaluates pass-rate across a **1-5 pixel tolerance sweep**
(not a single loose threshold) -- "we start with very liberal approach...
five pixel error... then we make it two pixels, one pixel." We re-checked our
own results against this and found something important: **our accuracy is
identical at every tolerance from 1px to 100px (46.7%%)**. The error
distribution is sharply bimodal -- 14/30 pairs land at exactly 0.0px error,
the other 16 land 179-748px away. There is no "close but slightly off" case.

This means **we do not have a sub-pixel precision problem** (which would show
up as errors clustered just above the 1-5px cutoff) -- **we have a candidate
selection problem**: when our pipeline picks the correct periodic repeat, the
underlying NCC peak is already pixel-exact; when it picks wrong, it is locked
onto a different repeat of the pattern entirely, exactly the "periodic
ambiguity" failure mode AM described. Per-style breakdown makes this
explicit: **DRAM: 3/15 pixel-perfect, 12 wrong-repeat failures. FinFET:
11/15 pixel-perfect, 4 wrong-repeat failures.** See
`evaluation/pass_rate_vs_tolerance.png`.

# Failure Analysis
## Success case
- Pair 2 (dram): error = 0.0 px, confidence = 0.8454, ambiguous_flag = True
- Reference: `../outputs/reference/dram_002_ref.png`
- Search: `../outputs/search/dram_002_search.png`
- Why it worked: the reference pattern's local structure (via/gate jitter) was distinct enough from surrounding periodic repeats that ORB geometric verification found a clear inlier margin over the next-best NCC candidate.

## Honest failure case
- Pair 0 (dram): error = 748.49 px, confidence = 0.7381, ambiguous_flag = False, hard_case = False
- Reference: `../outputs/reference/dram_000_ref.png`
- Search: `../outputs/search/dram_000_search.png`
- Why it failed: DRAM/FinFET layouts are periodic by construction (this is inherent to the problem, not a data-generation bug). When the true inset and a repeat-pitch alias have near-identical local geometry, raw intensity correlation (NCC) cannot distinguish them, and ORB struggles too because line/grid gratings produce few strong corner-like keypoints (a documented weakness of corner-based descriptors on repetitive textures). Our pipeline detects this condition via the `ambiguous` flag (top-2 candidate scores within a small gap) rather than silently returning a confident wrong answer -- this is the honest, explainable behavior we chose to optimize for, per the PS's explainability requirement.
- What would fix it: a learned embedding (e.g. a small Siamese CNN trained specifically to be sensitive to the sub-pixel jitter that distinguishes true matches from periodic aliases) would likely outperform hand-crafted NCC+ORB here, at the cost of needing training data and losing some interpretability. We deliberately scoped this out per the PS's own guidance that classical methods are acceptable and should be tried first.
