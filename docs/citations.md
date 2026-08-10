# Citations

Every augmentation, noise model, and structural parameter used in
`dataset/degradation.py` and `dataset/layouts.py` is justified below,
per the PS's mandatory citation requirement.

## 1. Edge brightening (`degradation.edge_brighten`)
SEM images exhibit brighter contrast along feature edges/steps because the
secondary-electron escape probability increases near edges and topographic
transitions (the "edge effect" / edge contrast).
- Reimer, L. (1998). *Scanning Electron Microscopy: Physics of Image
  Formation and Microanalysis*, 2nd ed., Springer.
- Goldstein, J. et al. (2018). *Scanning Electron Microscopy and X-Ray
  Microanalysis*, 4th ed., Springer. (Ch. on secondary electron imaging
  and edge/topographic contrast.)

## 2. Blur (`degradation.apply_blur`)
Finite electron-beam probe (spot) size and small focus variation across
the field of view limit spatial resolution and are commonly modeled as a
Gaussian point-spread function.
- Goldstein, J. et al. (2018), *Scanning Electron Microscopy and X-Ray
  Microanalysis*, 4th ed. (beam diameter and resolution limits).
- Reimer, L. (1998), as above (probe diameter and image blur).

## 3. Shot noise (`degradation.apply_shot_noise`)
Electron arrival at the detector is a counting process, so noise follows
Poisson statistics with variance proportional to signal (unlike additive
noise) -- standard in SEM/low-dose imaging literature.
- Goldstein, J. et al. (2018), as above (statistics of electron detection).
- Reimer, L. (1998), as above (shot noise from finite electron dose).

## 4. Additive Gaussian (read) noise (`degradation.apply_gaussian_noise`)
Detector/amplifier electronic readout noise is commonly modeled as
signal-independent additive Gaussian noise, standard in CCD/SEM detector
noise modeling.
- Janesick, J. R. (2001). *Scientific Charge-Coupled Devices*, SPIE Press.
  (CCD/detector read-noise modeling, applicable to SEM detector
  electronics.)

## 5. Independent noise per image (mandatory PS requirement)
Reference and search images are treated as separate physical captures, so
each is degraded with an independently seeded RNG stream
(`dataset/generator.py`, seeds derived per-image). This directly satisfies
the PS's "do NOT reuse the same noise on both images" requirement and
reflects that repeated SEM captures of the same or similar regions do not
share detector/shot-noise realizations.

## 6. DRAM periodic grid structure (`layouts.generate_dram_canvas`)
DRAM arrays consist of periodic word-lines and bit-lines crossing at right
angles with a contact/via at each intersection -- this is standard DRAM
cell-array topology.
- Itoh, K. (2001). *VLSI Memory Chip Design*, Springer. (DRAM array
  word-line/bit-line topology.)
- Kang, S.-M. & Leblebici, Y. (2003). *CMOS Digital Integrated Circuits*,
  3rd ed., McGraw-Hill. (memory array layout conventions.)

## 7. FinFET fin/gate structure (`layouts.generate_finfet_canvas`)
FinFET layouts consist of dense parallel vertical fins crossed by
horizontal gate lines at the channel region.
- Colinge, J.-P. (ed.) (2008). *FinFETs and Other Multi-Gate Transistors*,
  Springer. (fin/gate layout geometry.)
- Auth, C. et al. (2012). "A 22nm high performance and low-power CMOS
  technology," IEDM Technical Digest, Intel. (fin pitch / gate structure
  in production FinFET technology.)

## 8. Rotation/scale range (1-3 degrees, ~10x)
Values taken directly from the PS's stated constraints (Image 2 / dataset
description) rather than derived independently; kept intentionally small
per the PS's explicit guidance against over-engineering for large
geometric variation.

## 9. Classical feature verification (ORB + RANSAC)
- Rublee, E. et al. (2011). "ORB: An efficient alternative to SIFT or
  SURF," ICCV. (ORB descriptor used in `localization/feature_matcher.py`.)
- Fischler, M. A. & Bolles, R. C. (1981). "Random sample consensus,"
  Communications of the ACM. (RANSAC used for affine verification.)

## 10. Known limitation: keypoint-based methods on periodic/repetitive
textures
Corner-based descriptors (ORB, SIFT, Harris) are documented to perform
poorly on repetitive/periodic gratings because such textures lack strong,
spatially unique corner-like structure -- directly relevant to why our
pipeline's DRAM accuracy is lower than FinFET's (see
`docs/failure_analysis.md`).
- Schmid, C., Mohr, R., & Bauckhage, C. (2000). "Evaluation of interest
  point detectors," International Journal of Computer Vision.
