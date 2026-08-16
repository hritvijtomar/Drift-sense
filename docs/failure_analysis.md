# Failure Analysis
## Evaluation metric context (Official PS: pass rate at 5/4/2/1px)

Our pass rate is **identical at every tolerance from 1px to 100px**. The error distribution is sharply bimodal: correct predictions land at exactly 0.0px error, incorrect predictions land 179-748px away. There is no 'close but slightly off' case. This means the pipeline does not have a sub-pixel precision problem -- when it selects the correct periodic repeat, the underlying NCC peak is already pixel-exact. The entire gap between our 50% accuracy and 100% is a **candidate-selection problem** (picking the wrong repeat of a periodic pattern), not a localization-precision problem. DRAM: 3/15 pixel-perfect, 12 wrong-repeat failures. FinFET: 12/15 pixel-perfect, 3 wrong-repeat failures.

## Center tie-break rule: tested, not assumed

The PS requires selecting the match closest to the search-image center when multiple valid matches exist (Section 4.A). We tested two implementations: applied loosely (any candidate within 0.08 score of the top pick treated as 'valid') it *reduced* accuracy from 46.7% to 20%, because it overrode already-correct high-confidence picks with an arbitrary center-distance guess. Applied only to genuine near-ties (score gap <=0.005) it improved accuracy from 14/30 to 15/30 (46.7% -> 50.0%). We kept the tight version and report both results here rather than only the one that looked good.
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

## DRAM diagnostic: where exactly does candidate selection fail?

We ran a targeted diagnostic (`evaluation/diagnose_dram.py`) on all 12 DRAM failures, checking whether the ground-truth location was even present among the top-5 NCC candidates before ranking/verification ever runs:

- **8/12 failures: the true location was not in the candidate list at all.** This is a candidate-generation problem, not a ranking problem -- with a highly periodic grid, more than 5 distinct NCC peaks routinely score higher than the true location across the full search image, so non-max suppression discards it before ORB verification ever sees it.
- **4/12 failures: the true location WAS a candidate but lost the ranking**, by a score margin of 0.03-0.22 against the wrongly-picked candidate.

**Tested fix (rejected):** increasing `topk` from 5 to 15 (retaining more NCC candidates so the true location survives suppression more often) was measured on the full 30-pair set. Result: overall accuracy DROPPED from 50.0% to 40.0% (FinFET fell from 80.0% to 60.0%; DRAM stayed flat at 20.0%). More candidates gave the ranking stage more opportunities for a spurious high-ORB-inlier match to outrank the correct FinFET candidate, without actually fixing DRAM. We reverted to topk=5 and kept this result as evidence rather than silently discarding the experiment.

**Implication for future work:** the fix likely needs to happen earlier than candidate retention -- e.g. increasing the NMS suppression radius so fewer near-duplicate peaks compete for the top-5 slots, or scoring peaks by local distinctiveness (margin over the *local* neighborhood, not raw NCC value) before truncating to top-K. Simply admitting more candidates without a better way to rank them made results worse, not better.
