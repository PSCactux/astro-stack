"""The full stacking pipeline: calibrate, measure, register, combine, report."""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .align import Alignment, apply_alignment, match_stars, phase_align
from .calibrate import calibrate, master_frame, normalize_flat
from .io import list_images, load_image, save_png8, save_tiff16, to_luminance
from .stack import combine
from .stars import Stars, detect_stars, estimate_noise, robust_sigma
from .stretch import auto_stretch


@dataclass
class FrameReport:
    file: str
    status: str = "pending"  # "used" or "rejected"
    reason: str = ""
    n_stars: int = 0
    fwhm: float | None = None
    align_method: str | None = None
    inliers: int | None = None
    rms_px: float | None = None
    transform: dict | None = None  # frame -> reference motion
    noise: float | None = None


@dataclass
class StackResult:
    stacks: dict[str, np.ndarray]  # method -> cropped linear stack
    reference_raw: np.ndarray  # uncalibrated reference frame, same crop
    reference_aligned: np.ndarray  # calibrated reference frame, same crop
    crop: tuple[slice, slice]
    report: dict = field(default_factory=dict)


def _full_coverage_crop(coverage: np.ndarray, needed: int) -> tuple[slice, slice]:
    """Shrink the frame edge by edge until every pixel is covered by all frames."""
    bad = coverage < needed
    top, bottom, left, right = 0, bad.shape[0], 0, bad.shape[1]
    while top < bottom and left < right:
        edges = {
            "top": bad[top, left:right].sum(),
            "bottom": bad[bottom - 1, left:right].sum(),
            "left": bad[top:bottom, left].sum(),
            "right": bad[top:bottom, right - 1].sum(),
        }
        worst = max(edges, key=edges.get)
        if edges[worst] == 0:
            break
        if worst == "top":
            top += 1
        elif worst == "bottom":
            bottom -= 1
        elif worst == "left":
            left += 1
        else:
            right -= 1
    return slice(top, bottom), slice(left, right)


def run(
    lights_dir: str | Path,
    out_dir: str | Path,
    darks_dir: str | Path | None = None,
    flats_dir: str | Path | None = None,
    method: str = "sigma",
    kappa: float = 3.0,
    reference: str | None = None,
    max_fwhm_ratio: float = 1.5,
    min_stars: int = 8,
    crop: bool = True,
    extra_methods: tuple[str, ...] = (),
    log: Callable[[str], None] = print,
) -> StackResult:
    files = list_images(lights_dir)
    if len(files) < 2:
        raise ValueError(f"Need at least 2 light frames in {lights_dir}, found {len(files)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dark = flat = None
    if darks_dir:
        dark, n = master_frame(darks_dir)
        log(f"Master dark from {n} frames")
    if flats_dir:
        raw_flat, n = master_frame(flats_dir)
        flat = normalize_flat(raw_flat, dark)
        log(f"Master flat from {n} frames")

    def load(path: Path) -> np.ndarray:
        return calibrate(load_image(path), dark, flat)

    # Pass 1: detect stars and measure sharpness in every frame.
    reports = [FrameReport(f.name) for f in files]
    stars: list[Stars | None] = []
    shape = None
    for f, rep in zip(files, reports):
        img = load(f)
        shape = shape or img.shape
        if img.shape != shape:
            rep.status, rep.reason = "rejected", f"size {img.shape} differs from {shape}"
            stars.append(None)
            continue
        s = detect_stars(to_luminance(img))
        stars.append(s)
        rep.n_stars = len(s)
        rep.fwhm = round(s.median_fwhm, 3) if len(s) >= 3 else None
    fwhms = [r.fwhm for r in reports if r.fwhm is not None and r.n_stars >= min_stars]
    median_fwhm = float(np.median(fwhms)) if fwhms else None
    if median_fwhm:
        for rep in reports:
            if rep.status == "pending" and rep.fwhm and rep.fwhm > max_fwhm_ratio * median_fwhm:
                rep.status, rep.reason = "rejected", f"blurry (FWHM {rep.fwhm:.2f} px vs median {median_fwhm:.2f})"

    candidates = [i for i, r in enumerate(reports) if r.status == "pending"]
    if not candidates:
        raise RuntimeError("Every frame was rejected before alignment")
    if reference:
        names = [files[i].name for i in candidates]
        if reference not in names:
            raise ValueError(f"Reference {reference!r} is not among the usable frames")
        ref_idx = candidates[names.index(reference)]
    else:
        # Sharpest frame among those that show most of the stars.
        most = max(reports[i].n_stars for i in candidates)
        pool = [i for i in candidates if reports[i].n_stars >= 0.7 * most] or candidates
        ref_idx = min(pool, key=lambda i: reports[i].fwhm or math.inf)
    ref_stars = stars[ref_idx]
    ref_image = load(files[ref_idx])
    ref_lum = to_luminance(ref_image)
    log(f"Reference frame: {files[ref_idx].name} ({len(ref_stars)} stars, FWHM {reports[ref_idx].fwhm} px)")

    # Pass 2: register and warp every candidate into a disk-backed cube.
    tmp = tempfile.TemporaryDirectory(prefix="astro-stack-")
    try:
        cube = np.lib.format.open_memmap(
            Path(tmp.name) / "cube.npy", mode="w+", dtype=np.float32, shape=(len(candidates),) + shape
        )
        coverage = np.zeros(shape[:2], dtype=np.int32)
        used: list[int] = []
        for i in candidates:
            rep = reports[i]
            img = ref_image if i == ref_idx else load(files[i])
            if i == ref_idx:
                al = Alignment(True, method="reference")
            else:
                al = match_stars(ref_stars, stars[i])
                if not al.ok:
                    fallback = phase_align(ref_lum, to_luminance(img))
                    al = fallback if fallback.ok else Alignment(False, reason=f"{al.reason}; {fallback.reason}")
            if not al.ok:
                rep.status, rep.reason = "rejected", al.reason
                log(f"  {rep.file}: rejected ({al.reason})")
                continue
            aligned = img if i == ref_idx else apply_alignment(img, al)
            cube[len(used)] = aligned
            coverage += np.isfinite(to_luminance(aligned))
            used.append(i)
            rep.status, rep.align_method = "used", al.method
            rep.inliers = al.inliers or None
            rep.rms_px = None if math.isnan(al.rms) else round(al.rms, 3)
            rep.transform = al.describe()
            t = rep.transform
            log(f"  {rep.file}: {al.method:9s} dx={t['dx']:+7.2f} dy={t['dy']:+7.2f} rot={t['rotation_deg']:+.3f} deg")

        n = len(used)
        if n < 2:
            raise RuntimeError("Fewer than 2 frames could be aligned")
        stack_cube = cube[:n]
        whole = (slice(0, shape[0]), slice(0, shape[1]))
        rows, cols = _full_coverage_crop(coverage, n) if crop else whole
        if rows.stop - rows.start < 32 or cols.stop - cols.start < 32:
            log("Warning: frames barely overlap; skipping the crop")
            rows, cols = whole

        stacks = {}
        for m in dict.fromkeys((method, *extra_methods)):
            log(f"Combining {n} frames with method '{m}'")
            stacks[m] = combine(stack_cube, m, kappa)[rows, cols]

        # Noise is measured on the same crop of each aligned frame and of the stack.
        for k, i in enumerate(used):
            reports[i].noise = round(estimate_noise(to_luminance(np.asarray(stack_cube[k, rows, cols]))), 4)
        single_noise = float(np.median([reports[i].noise for i in used]))

        # The stack's own noise is measured from the difference between the
        # stacks of even and odd frames: the real sky (faint stars, nebula,
        # star halos) cancels and only the random part is left.
        n_a, n_b = (n + 1) // 2, n // 2
        half_a = combine(stack_cube[0::2], method, kappa)[rows, cols]
        half_b = combine(stack_cube[1::2], method, kappa)[rows, cols]
        diff_sigma = robust_sigma(to_luminance(half_a - half_b))
        stack_noise = diff_sigma * math.sqrt(n_a * n_b) / n
    finally:
        del cube
        tmp.cleanup()

    final = stacks[method]
    gain = single_noise / stack_noise if stack_noise > 0 else float("nan")

    scale = save_tiff16(out_dir / "stack.tif", final)
    save_png8(out_dir / "stack_preview.png", auto_stretch(final))
    report = {
        "lights": str(lights_dir),
        "darks": str(darks_dir) if darks_dir else None,
        "flats": str(flats_dir) if flats_dir else None,
        "method": method,
        "kappa": kappa if method == "sigma" else None,
        "reference": files[ref_idx].name,
        "frames_total": len(files),
        "frames_used": n,
        "frames_rejected": len(files) - n,
        "median_fwhm_px": round(median_fwhm, 3) if median_fwhm else None,
        "crop": {"x": [cols.start, cols.stop], "y": [rows.start, rows.stop]},
        "snr": {
            "single_frame_noise": round(single_noise, 4),
            "stack_noise": round(stack_noise, 4),
            "measured_gain": round(gain, 3),
            "ideal_gain_sqrt_n": round(math.sqrt(n), 3),
        },
        "outputs": {"tiff16": "stack.tif", "preview": "stack_preview.png", "tiff_scale": scale},
        "frames": [asdict(r) for r in reports],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    rejected = [r for r in reports if r.status == "rejected"]
    log(f"Used {n}/{len(files)} frames; rejected {len(rejected)}")
    for r in rejected:
        log(f"  - {r.file}: {r.reason}")
    log(f"Noise per frame {single_noise:.3f} -> stack {stack_noise:.3f}: SNR gain x{gain:.2f} (ideal sqrt({n}) = {math.sqrt(n):.2f})")
    log(f"Wrote {out_dir / 'stack.tif'}, {out_dir / 'stack_preview.png'}, {out_dir / 'report.json'}")

    ref_crop = ref_image[rows, cols]
    return StackResult(stacks, load_image(files[ref_idx])[rows, cols], ref_crop, (rows, cols), report)
