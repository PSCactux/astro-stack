"""Background estimation, star detection and simple image statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

MAD_TO_SIGMA = 1.4826


def robust_sigma(values: np.ndarray) -> float:
    """Standard deviation estimated from the median absolute deviation."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    return float(MAD_TO_SIGMA * np.median(np.abs(values - np.median(values))))


def background_map(image: np.ndarray, block: int = 32) -> np.ndarray:
    """Smooth sky background from per-block medians (handles gradients and vignetting)."""
    h, w = image.shape
    bh, bw = max(1, h // block), max(1, w // block)
    cropped = image[: bh * block, : bw * block] if h >= block and w >= block else image
    if h < block or w < block:
        return np.full_like(image, np.nanmedian(image))
    blocks = cropped.reshape(bh, block, bw, block).swapaxes(1, 2).reshape(bh, bw, -1)
    med = np.nanmedian(blocks, axis=2)
    med = ndimage.median_filter(med, size=3, mode="nearest")
    zoom = (h / bh, w / bw)
    bkg = ndimage.zoom(med, zoom, order=1, mode="nearest")
    return bkg[:h, :w].astype(np.float32)


@dataclass
class Stars:
    """Detected stars, sorted by flux (brightest first)."""

    xy: np.ndarray  # (N, 2) centroids as (x, y) = (column, row)
    flux: np.ndarray  # (N,)
    fwhm: np.ndarray  # (N,) per-star FWHM in pixels

    def __len__(self) -> int:
        return len(self.flux)

    @property
    def median_fwhm(self) -> float:
        return float(np.median(self.fwhm)) if len(self) else float("nan")


def detect_stars(
    image: np.ndarray,
    threshold: float = 5.0,
    max_stars: int = 200,
    min_fwhm: float = 1.2,
    max_elongation: float = 2.5,
    edge: int = 8,
) -> Stars:
    """Find star-like sources in a 2-D image.

    Pixels more than ``threshold`` sigma above the local background (after a
    light Gaussian smoothing) are grouped into connected blobs. Each blob gets
    a flux-weighted centroid and a FWHM from its second moments. Hot pixels
    (too sharp) and satellite trails (too elongated) are discarded.
    """
    residual = image - background_map(image)
    smooth = ndimage.gaussian_filter(residual, 1.0)
    sigma = robust_sigma(smooth[::2, ::2])
    if sigma <= 0:
        return Stars(np.zeros((0, 2)), np.zeros(0), np.zeros(0))
    labels, n = ndimage.label(smooth > threshold * sigma)
    if n == 0:
        return Stars(np.zeros((0, 2)), np.zeros(0), np.zeros(0))

    slices = ndimage.find_objects(labels)
    h, w = image.shape
    xs, ys, fluxes, fwhms = [], [], [], []
    yy_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for idx, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        sh, sw = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if sh * sw < 4 or sh > 60 or sw > 60:  # noise speck or extended object / trail
            continue
        mask = labels[sl] == idx
        cut = np.where(mask, np.clip(residual[sl], 0, None), 0.0)
        total = cut.sum()
        if total <= 0:
            continue
        key = cut.shape
        if key not in yy_cache:
            yy_cache[key] = np.mgrid[0 : key[0], 0 : key[1]]
        yy, xx = yy_cache[key]
        cy = (cut * yy).sum() / total
        cx = (cut * xx).sum() / total
        myy = (cut * (yy - cy) ** 2).sum() / total
        mxx = (cut * (xx - cx) ** 2).sum() / total
        mxy = (cut * (xx - cx) * (yy - cy)).sum() / total
        # Eigenvalues of the second-moment matrix give the blob's axes.
        tr, det = mxx + myy, mxx * myy - mxy**2
        disc = np.sqrt(max(tr * tr / 4 - det, 0.0))
        l1, l2 = tr / 2 + disc, max(tr / 2 - disc, 1e-6)
        if np.sqrt(l1 / l2) > max_elongation:
            continue
        fwhm = 2.3548 * np.sqrt((l1 + l2) / 2)
        if fwhm < min_fwhm:
            continue
        x, y = cx + sl[1].start, cy + sl[0].start
        if x < edge or y < edge or x > w - 1 - edge or y > h - 1 - edge:
            continue
        xs.append(x)
        ys.append(y)
        fluxes.append(total)
        fwhms.append(fwhm)

    order = np.argsort(fluxes)[::-1][:max_stars]
    xy = np.column_stack([xs, ys])[order] if xs else np.zeros((0, 2))
    return Stars(xy, np.asarray(fluxes)[order], np.asarray(fwhms)[order])


def estimate_noise(image: np.ndarray) -> float:
    """Background noise (sigma) of a 2-D image, ignoring stars and gradients."""
    residual = image - background_map(image)
    sigma = robust_sigma(residual)
    sources = ndimage.binary_dilation(np.abs(residual) > 3 * sigma, iterations=2)
    return robust_sigma(residual[~sources])
