"""Display stretch for linear astro data, similar in spirit to PixInsight's
screen transfer function: clip the shadows just below the sky, then apply a
midtones transfer function that maps the sky level to a target brightness."""

from __future__ import annotations

import numpy as np

from .stars import MAD_TO_SIGMA


def _mtf(m: float, x: np.ndarray) -> np.ndarray:
    """Midtones transfer function: maps m -> 0.5, keeps 0 and 1 fixed."""
    return ((m - 1) * x) / ((2 * m - 1) * x - m)


def auto_stretch(image: np.ndarray, target: float = 0.15, shadows_sigma: float = 2.8) -> np.ndarray:
    """Return a [0, 1] display version of linear data.

    Each channel's sky level is moved to the same point (which neutralizes a
    colored, light-polluted sky); black point, white point and the midtone
    curve are shared by all channels so star colors are preserved.
    """
    img = np.nan_to_num(image.astype(np.float32), nan=0.0)
    chans = img[..., None] if img.ndim == 2 else img
    meds, sigmas = [], []
    for c in range(chans.shape[-1]):
        sample = chans[::4, ::4, c]
        med = np.median(sample)
        meds.append(med)
        sigmas.append(MAD_TO_SIGMA * np.median(np.abs(sample - med)))
    flat = chans - np.array(meds, dtype=np.float32)  # sky at 0 in every channel
    black = -shadows_sigma * float(np.mean(sigmas))
    top = float(np.percentile(flat, 99.995))
    out = np.clip((flat - black) / max(top - black, 1e-9), 0, 1)
    m0 = float(np.clip(-black / max(top - black, 1e-9), 1e-6, 0.999))  # where the sky sits now
    # Pick the midtone balance m so that _mtf(m, m0) == target.
    m = m0 * (target - 1) / (2 * m0 * target - m0 - target)
    out = np.clip(_mtf(float(np.clip(m, 1e-6, 0.999)), out), 0, 1)
    return out[..., 0] if img.ndim == 2 else out
