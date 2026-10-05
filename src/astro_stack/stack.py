"""Pixel-wise combination of aligned frames.

All functions take a cube shaped (N, ...) where NaN marks pixels a frame
does not cover, and combine along axis 0. Color data is handled per channel
simply because every channel is its own set of pixels.
"""

from __future__ import annotations

import warnings

import numpy as np

METHODS = ("mean", "median", "sigma")


def mean_stack(cube: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
        return np.nanmean(cube, axis=0)


def median_stack(cube: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(cube, axis=0)


def sigma_clip_stack(cube: np.ndarray, kappa: float = 3.0, iterations: int = 3) -> np.ndarray:
    """Mean of the values within ``kappa`` sigma of the per-pixel median.

    The first pass estimates sigma from the median absolute deviation, so a
    single very bright outlier (satellite trail, hot pixel, cosmic ray) cannot
    inflate sigma and hide itself, even with few frames. Later passes use the
    standard deviation of the surviving values, which is less noisy and so
    clips fewer good values.
    """
    data = np.array(cube, dtype=np.float32, copy=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for it in range(iterations):
            center = np.nanmedian(data, axis=0)
            if it == 0:
                sigma = 1.4826 * np.nanmedian(np.abs(data - center), axis=0)
            else:
                sigma = np.nanstd(data, axis=0)
            outliers = np.abs(data - center) > kappa * np.maximum(sigma, 1e-6)
            if not outliers.any():
                break
            data[outliers] = np.nan
        return np.nanmean(data, axis=0)


def combine(cube: np.ndarray, method: str = "sigma", kappa: float = 3.0, chunk_rows: int = 64) -> np.ndarray:
    """Combine a (possibly memory-mapped) cube in row chunks to bound memory use."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}, choose from {METHODS}")
    out = np.empty(cube.shape[1:], dtype=np.float32)
    for r in range(0, cube.shape[1], chunk_rows):
        block = np.asarray(cube[:, r : r + chunk_rows], dtype=np.float32)
        if method == "mean":
            out[r : r + chunk_rows] = mean_stack(block)
        elif method == "median":
            out[r : r + chunk_rows] = median_stack(block)
        else:
            out[r : r + chunk_rows] = sigma_clip_stack(block, kappa)
    return out
