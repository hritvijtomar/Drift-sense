"""
degradation.py
---------------
Turns a clean CAD-style layout crop into something that looks like a real
SEM (Scanning Electron Microscope) capture.

Every function here corresponds to ONE physically-motivated imaging effect.
This is deliberate: the PS requires 2-3 citations per augmentation/noise
choice, and judges are semiconductor engineers who will ask "why did you
do this." Splitting each effect into its own function with its own
docstring+citation makes that defense trivial and keeps citations.md in
sync with the code.

Pipeline order used by generator.py:
    clean layout crop
      -> edge brightening      (SEM edge-effect from secondary electron yield)
      -> blur                  (beam spot size / focus variation)
      -> shot noise (Poisson)  (electron counting statistics)
      -> gaussian read noise   (detector/amplifier electronics)
      -> brightness/contrast drift (charging / detector gain drift)
      -> geometric transform   (rotation + scale, applied to SEARCH image only,
                                per PS: rotation 1-3 deg, scale ~10x known)
      -> downsample to final size

IMPORTANT: reference and search images MUST get independently sampled
noise (separate calls to the noise functions with different RNG state) --
this is an explicit mandatory requirement in the PS. Never reuse the same
noise array on both images.
"""

import numpy as np
import cv2


def edge_brighten(img, ksize=5, strength=0.6):
    """
    SEM images show brighter contrast along feature edges because the
    secondary-electron escape probability increases near edges/steps
    (the well-known SEM "edge brightening" / edge-contrast effect).

    Implementation: detect edges via gradient magnitude, add a scaled
    version of that back onto the image (a cheap, controllable stand-in
    for the electron-yield edge effect).

    Citations to include in citations.md:
      - Reimer, L. "Scanning Electron Microscopy: Physics of Image
        Formation and Microanalysis" (edge/topographic contrast).
      - Goldstein et al., "Scanning Electron Microscopy and X-Ray
        Microanalysis" (secondary electron edge effect).
    """
    img_f = img.astype(np.float32)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=ksize)
    grad_mag = cv2.magnitude(gx, gy)
    grad_mag = grad_mag / (grad_mag.max() + 1e-6) * 255.0
    out = img_f + strength * grad_mag
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_blur(img, sigma=0.8, seed=None):
    """
    Gaussian blur models finite electron-beam spot size and small focus
    variation across the field of view.

    Citation: Goldstein et al. (beam diameter / probe size limits
    resolution -> blur kernel).
    """
    rng = np.random.default_rng(seed)
    # tiny random jitter in sigma so reference/search aren't identically blurred
    s = max(0.15, sigma + rng.normal(0, 0.1))
    k = max(3, int(2 * round(3 * s) + 1))
    return cv2.GaussianBlur(img, (k, k), s)


def apply_shot_noise(img, scale=8.0, seed=None):
    """
    Shot noise: electron arrival is a Poisson counting process, so noise
    variance scales with signal intensity (unlike additive Gaussian noise).
    Modeled as Poisson sampling with a scaling factor controlling SNR.

    Citation: Goldstein et al.; Reimer -- SEM shot noise from finite
    electron dose per pixel.

    Independence requirement: pass a DIFFERENT seed for reference vs.
    search image calls.
    """
    rng = np.random.default_rng(seed)
    img_f = img.astype(np.float32)
    noisy = rng.poisson(img_f / 255.0 * scale) / scale * 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_gaussian_noise(img, sigma=6.0, seed=None):
    """
    Additive Gaussian noise models detector/amplifier electronic readout
    noise, independent of signal level.

    Citation: standard SEM/CCD noise modeling literature (read noise term).

    NOTE: values are allowed to exceed [0,255] before clipping -- this is
    expected/realistic per the PS ("degraded intensity range may exceed
    ground truth range").
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img.shape)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_brightness_drift(img, seed=None):
    """
    Slow brightness/contrast drift models detector gain drift and local
    charging effects common in real SEM capture.
    """
    rng = np.random.default_rng(seed)
    alpha = 1.0 + rng.normal(0, 0.05)   # contrast
    beta = rng.normal(0, 8.0)           # brightness
    out = img.astype(np.float32) * alpha + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def rotate_scale(img, angle_deg, scale, out_size):
    """
    Applies small rotation + scale to simulate stage/imaging misalignment
    between the reference capture and the wider search capture.
    Per PS: rotation limited to 1-3 degrees, scale relationship known
    approximately (~10x for DRAM style). We keep both within that regime
    intentionally -- the PS explicitly warns against over-engineering for
    large rotation/scale that won't occur in the test set.
    """
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return cv2.resize(rotated, out_size, interpolation=cv2.INTER_AREA)


def degrade_reference(clean_patch, seed=0):
    """Lighter degradation pipeline for the reference image (higher quality
    'captured at higher magnification' assumption -- PS: search images have
    MORE noise than reference)."""
    img = edge_brighten(clean_patch, strength=0.35)
    img = apply_blur(img, sigma=0.5, seed=seed)
    img = apply_shot_noise(img, scale=40.0, seed=seed + 1)
    img = apply_gaussian_noise(img, sigma=2.0, seed=seed + 2)
    img = apply_brightness_drift(img, seed=seed + 3)
    return img


def degrade_search(clean_tile, seed=1000, heavier=False):
    """Heavier degradation for the search image -- independent noise from
    the reference (different seed base), plus optionally boosted noise to
    emulate the official (noisier) test set."""
    img = edge_brighten(clean_tile, strength=0.4)
    img = apply_blur(img, sigma=0.7 if not heavier else 1.0, seed=seed)
    img = apply_shot_noise(img, scale=22.0 if not heavier else 14.0, seed=seed + 1)
    img = apply_gaussian_noise(img, sigma=4.0 if not heavier else 7.0, seed=seed + 2)
    img = apply_brightness_drift(img, seed=seed + 3)
    return img
