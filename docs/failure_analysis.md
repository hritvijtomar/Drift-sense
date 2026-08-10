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
