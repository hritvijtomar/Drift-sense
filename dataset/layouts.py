"""
layouts.py
----------
Generates clean, noise-free binary/grayscale layout images that mimic
top-down semiconductor die structures at CAD/mask level (before any
imaging degradation is applied).

Two architecture styles, matching the PS requirements:

  DRAM-style  : periodic horizontal word-lines + vertical bit-lines,
                crossing at right angles, with a contact/via dot at
                every intersection.
  FinFET-style: dense parallel vertical fins, crossed by 1-2 horizontal
                gate bars at the intersection region.

Design notes / why this file exists separately from degradation:
- Keeping "ideal geometry" separate from "imaging physics" (degradation.py)
  means every degradation step is independently justifiable and testable.
- All layouts are drawn on a large canvas first (the "wafer"), and both the
  reference patch and the search-image tile are CROPS of that same canvas.
  This guarantees the reference is a *true* sub-pattern of the search image
  (not a separately-generated look-alike), which is what "ground truth
  localization" actually requires.
"""

import numpy as np
import cv2


def generate_dram_canvas(canvas_size=2000, pitch=80, line_width=6, via_radius=6, seed=None):
    """
    Periodic horizontal word-lines and vertical bit-lines crossing at right
    angles, with a small contact/via dot at every intersection.

    High-contrast, fine pitch, extremely regular -- by design this is where
    periodic-pattern ambiguity will show up during localization, which is
    intentional (the PS explicitly wants this tested).
    """
    rng = np.random.default_rng(seed)
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # Slight pitch jitter so the grid isn't a perfect Fourier comb (more
    # realistic mask/etch variation, also reduces trivial autocorrelation).
    jitter = rng.integers(-1, 2, size=1000)

    x = 0
    col_idx = 0
    xs = []
    while x < canvas_size:
        xs.append(x)
        cv2.line(canvas, (x, 0), (x, canvas_size), 255, line_width)
        x += pitch + int(jitter[col_idx % len(jitter)])
        col_idx += 1

    y = 0
    row_idx = 0
    ys = []
    while y < canvas_size:
        ys.append(y)
        cv2.line(canvas, (0, y), (canvas_size, y), 255, line_width)
        y += pitch + int(jitter[(row_idx + 500) % len(jitter)])
        row_idx += 1

    for yy in ys:
        for xx in xs:
            cv2.circle(canvas, (xx, yy), via_radius, 255, -1)

    return canvas


def generate_finfet_canvas(canvas_size=2000, fin_pitch=34, fin_width=10,
                            gate_pitch=260, gate_width=26, seed=None):
    """
    Dense parallel vertical fin lines, crossed by horizontal gate bars.
    High-contrast vertical structure with distinctive gate crossings --
    the crossings are the "landmark" regions that make localization
    tractable despite the fins themselves being highly periodic.
    """
    rng = np.random.default_rng(seed)
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    x = 0
    idx = 0
    while x < canvas_size:
        jitter = int(rng.integers(-1, 2))
        cv2.line(canvas, (x, 0), (x, canvas_size), 200, fin_width)
        x += fin_pitch + jitter
        idx += 1

    y = 0
    while y < canvas_size:
        cv2.rectangle(canvas, (0, y), (canvas_size, y + gate_width), 255, -1)
        y += gate_pitch

    return canvas


def generate_canvas(style, canvas_size=2000, seed=None):
    """Dispatch helper. style in {'dram', 'finfet'}."""
    if style == "dram":
        return generate_dram_canvas(canvas_size=canvas_size, seed=seed)
    elif style == "finfet":
        return generate_finfet_canvas(canvas_size=canvas_size, seed=seed)
    else:
        raise ValueError(f"Unknown style: {style}")
