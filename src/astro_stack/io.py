"""Reading and writing images.

Images are handled as float32 arrays in their native ADU range
(0-255 for 8-bit files, 0-65535 for 16-bit files), shaped (H, W) for
mono data or (H, W, 3) for color data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageOps

RASTER_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
FITS_EXTENSIONS = {".fits", ".fit", ".fts"}
SUPPORTED_EXTENSIONS = RASTER_EXTENSIONS | FITS_EXTENSIONS


def list_images(folder: str | Path) -> list[Path]:
    """Return the supported image files in a folder, sorted by name."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS and not p.name.startswith(".")
    )


def load_image(path: str | Path) -> np.ndarray:
    """Load one image as float32, dropping any alpha channel."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in FITS_EXTENSIONS:
        data = _load_fits(path)
    elif suffix in {".tif", ".tiff"}:
        data = tifffile.imread(path)
    else:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)  # phones store rotation in EXIF
            if img.mode in ("I;16", "I;16B", "I;16L", "I"):
                data = np.asarray(img, dtype=np.float32)
            else:
                data = np.asarray(img.convert("RGB") if img.mode not in ("L", "RGB") else img)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 3:
        if data.shape[0] in (3, 4) and data.shape[-1] not in (3, 4):
            data = np.moveaxis(data, 0, -1)  # channel-first (FITS, some TIFFs)
        data = data[..., :3]
    if data.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape {data.shape} in {path}")
    return data


def _load_fits(path: Path) -> np.ndarray:
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("Reading FITS files needs astropy: pip install 'astro-stack[fits]'") from exc
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return np.asarray(hdu.data, dtype=np.float32)
    raise ValueError(f"No image data in {path}")


def save_tiff16(path: str | Path, image: np.ndarray, scale: float | None = None) -> float:
    """Save linear data as a 16-bit TIFF. Returns the scale factor used."""
    data = np.nan_to_num(image, nan=0.0)
    data = np.clip(data, 0, None)
    if scale is None:
        peak = float(data.max())
        scale = 65535.0 / peak if peak > 0 else 1.0
    out = np.clip(data * scale, 0, 65535).round().astype(np.uint16)
    tifffile.imwrite(path, out, photometric="rgb" if out.ndim == 3 else "minisblack")
    return scale


def save_png8(path: str | Path, image01: np.ndarray) -> None:
    """Save an already stretched image with values in [0, 1] as an 8-bit PNG."""
    out = (np.clip(np.nan_to_num(image01), 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(out).save(path)


def to_luminance(image: np.ndarray) -> np.ndarray:
    """Average the color channels (or return mono data unchanged)."""
    return image.mean(axis=2) if image.ndim == 3 else image
