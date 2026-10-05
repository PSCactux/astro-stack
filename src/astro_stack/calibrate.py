"""Master calibration frames: darks remove thermal signal, offset and hot
pixels; flats correct vignetting and dust shadows."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import list_images, load_image


def master_frame(folder: str | Path) -> tuple[np.ndarray, int]:
    """Median-combine every image in a folder. Returns (master, count)."""
    files = list_images(folder)
    if not files:
        raise FileNotFoundError(f"No images in {folder}")
    frames = np.stack([load_image(f) for f in files])
    return np.median(frames, axis=0).astype(np.float32), len(files)


def normalize_flat(flat: np.ndarray, dark: np.ndarray | None = None) -> np.ndarray:
    """Scale a master flat so each channel has median 1 (per-channel flat field)."""
    flat = flat.astype(np.float32)
    # The light-frame dark is only a rough stand-in for a flat-dark, so only
    # its median offset is removed; that keeps hot pixels out of the flat.
    if dark is not None:
        flat = flat - np.median(dark, axis=(0, 1))
    med = np.median(flat, axis=(0, 1))
    return np.clip(flat / med, 0.05, None)


def calibrate(light: np.ndarray, dark: np.ndarray | None, flat: np.ndarray | None) -> np.ndarray:
    """(light - dark) / flat, with shapes checked."""
    out = light
    if dark is not None:
        _check_shape(light, dark, "dark")
        out = out - dark
    if flat is not None:
        _check_shape(light, flat, "flat")
        out = out / flat
    return out.astype(np.float32)


def _check_shape(light: np.ndarray, cal: np.ndarray, name: str) -> None:
    if light.shape != cal.shape:
        raise ValueError(f"Master {name} shape {cal.shape} does not match light frame {light.shape}")
