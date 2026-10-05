"""Frame registration.

The main path matches stars between a frame and the reference using
triangle asterisms (shape invariants that survive translation, rotation and
scale), then estimates a similarity transform with RANSAC and refines it by
least squares on all inliers. When that fails, a translation-only estimate
from phase correlation is tried and accepted only if the aligned frame
correlates well with the reference.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform, warp

from .stars import Stars, background_map


@dataclass
class Alignment:
    """Result of registering one frame onto the reference.

    ``transform`` maps reference (x, y) coordinates to frame coordinates,
    which is exactly the inverse map that ``skimage.transform.warp`` needs.
    """

    ok: bool
    transform: SimilarityTransform = field(default_factory=SimilarityTransform)
    method: str = "none"
    inliers: int = 0
    rms: float = float("nan")
    reason: str = ""

    def describe(self) -> dict:
        """Frame-to-reference motion in human terms (inverse of ``transform``)."""
        inv = SimilarityTransform(matrix=np.linalg.inv(self.transform.params))
        return {
            "dx": round(float(inv.translation[0]), 3),
            "dy": round(float(inv.translation[1]), 3),
            "rotation_deg": round(float(np.degrees(inv.rotation)), 4),
            "scale": round(float(inv.scale), 5),
        }


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> SimilarityTransform:
    """Least-squares similarity transform mapping src points onto dst points.

    Solves x' = a*x - b*y + tx, y' = b*x + a*y + ty for (a, b, tx, ty).
    """
    n = len(src)
    A = np.zeros((2 * n, 4))
    A[0::2] = np.column_stack([src[:, 0], -src[:, 1], np.ones(n), np.zeros(n)])
    A[1::2] = np.column_stack([src[:, 1], src[:, 0], np.zeros(n), np.ones(n)])
    a, b, tx, ty = np.linalg.lstsq(A, dst.reshape(-1), rcond=None)[0]
    return SimilarityTransform(matrix=np.array([[a, -b, tx], [b, a, ty], [0, 0, 1]]))


def _triangles(xy: np.ndarray, min_side: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """All triangles of the given points: shape invariants and ordered vertices.

    Sides are sorted L1 <= L2 <= L3; the invariants are (L1/L3, L2/L3).
    Vertices are ordered by their opposite side (shortest first), so matched
    triangles give point correspondences directly.
    """
    combos = np.array(list(itertools.combinations(range(len(xy)), 3)))
    if len(combos) == 0:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=int)
    p = xy[combos]  # (T, 3, 2)
    # side i is opposite vertex i
    sides = np.stack(
        [
            np.linalg.norm(p[:, 1] - p[:, 2], axis=1),
            np.linalg.norm(p[:, 0] - p[:, 2], axis=1),
            np.linalg.norm(p[:, 0] - p[:, 1], axis=1),
        ],
        axis=1,
    )
    order = np.argsort(sides, axis=1)
    s = np.take_along_axis(sides, order, axis=1)
    verts = np.take_along_axis(combos, order, axis=1)
    keep = s[:, 0] >= min_side
    inv = np.column_stack([s[:, 0] / s[:, 2], s[:, 1] / s[:, 2]])
    return inv[keep], verts[keep]


def match_stars(
    ref: Stars,
    frame: Stars,
    n_bright: int = 25,
    tol: float = 2.5,
    scale_range: tuple[float, float] = (0.8, 1.25),
    min_inliers: int = 6,
    max_trials: int = 400,
) -> Alignment:
    """Estimate the reference->frame similarity transform from two star lists."""
    if len(ref) < 3 or len(frame) < 3:
        return Alignment(False, reason=f"too few stars ({len(frame)})")

    ref_xy, frm_xy = ref.xy, frame.xy
    ref_inv, ref_tri = _triangles(ref_xy[:n_bright])
    frm_inv, frm_tri = _triangles(frm_xy[:n_bright])
    if len(ref_inv) == 0 or len(frm_inv) == 0:
        return Alignment(False, reason="no usable star triangles")

    dist, idx = cKDTree(frm_inv).query(ref_inv, k=1)
    candidates = np.argsort(dist)[:max_trials]
    candidates = candidates[dist[candidates] < 0.02]

    frm_tree = cKDTree(frm_xy)
    best: tuple[int, np.ndarray | None] = (0, None)
    for c in candidates:
        src, dst = ref_xy[ref_tri[c]], frm_xy[frm_tri[idx[c]]]
        t = fit_similarity(src, dst)
        if not scale_range[0] <= t.scale <= scale_range[1]:
            continue
        d, j = frm_tree.query(t(ref_xy), k=1)
        good = d < tol
        n_good = int(good.sum())
        if n_good > best[0]:
            best = (n_good, np.column_stack([np.nonzero(good)[0], j[good]]))
            if n_good >= 0.8 * min(len(ref), len(frame)):
                break

    if best[1] is None or best[0] < min_inliers:
        return Alignment(False, reason=f"star matching failed ({best[0]} inliers)")

    # Refine on all inliers, then once more with the improved transform.
    pairs = best[1]
    for _ in range(2):
        t = fit_similarity(ref_xy[pairs[:, 0]], frm_xy[pairs[:, 1]])
        d, j = frm_tree.query(t(ref_xy), k=1)
        good = d < tol
        pairs = np.column_stack([np.nonzero(good)[0], j[good]])
    resid = np.linalg.norm(t(ref_xy[pairs[:, 0]]) - frm_xy[pairs[:, 1]], axis=1)
    return Alignment(True, t, "stars", len(pairs), float(np.sqrt(np.mean(resid**2))))


def _prepare(image: np.ndarray) -> np.ndarray:
    """Background-subtracted, gently smoothed, compressed image for correlation."""
    resid = image - background_map(image)
    resid = ndimage.gaussian_filter(resid, 1.5)
    return np.arcsinh(np.clip(resid, 0, None) / (resid.std() + 1e-9))


def phase_align(ref_image: np.ndarray, image: np.ndarray, min_corr: float = 0.3) -> Alignment:
    """Translation-only fallback using phase correlation."""
    a, b = _prepare(ref_image), _prepare(image)
    shift, _, _ = phase_cross_correlation(a, b, upsample_factor=10)
    # ndimage.shift(b, shift) ~ a, so reference (x, y) maps to frame (x - sx, y - sy).
    t = SimilarityTransform(translation=(-shift[1], -shift[0]))
    aligned = warp(b, t, order=1, cval=np.nan, preserve_range=True)
    ok = np.isfinite(aligned)
    corr = float(np.corrcoef(a[ok], aligned[ok])[0, 1]) if ok.sum() > 100 else 0.0
    if corr < min_corr:
        return Alignment(False, reason=f"phase correlation too weak (r={corr:.2f})")
    return Alignment(True, t, "phase", 0, float("nan"))


def apply_alignment(image: np.ndarray, alignment: Alignment, order: int = 3) -> np.ndarray:
    """Warp a frame onto the reference grid; pixels outside the frame become NaN."""
    channels = [image] if image.ndim == 2 else [image[..., c] for c in range(image.shape[2])]
    out = []
    for ch in channels:
        warped = warp(ch, alignment.transform, order=order, mode="constant", cval=0.0, preserve_range=True)
        cover = warp(np.ones_like(ch), alignment.transform, order=0, mode="constant", cval=0.0)
        # Drop a 2 px rim where the interpolation kernel saw the zero padding.
        warped[~ndimage.binary_erosion(cover > 0.5, iterations=2)] = np.nan
        out.append(warped.astype(np.float32))
    return out[0] if image.ndim == 2 else np.stack(out, axis=-1)
