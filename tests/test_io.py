import numpy as np
import pytest

from astro_stack.io import list_images, load_image, save_tiff16
from astro_stack.stretch import auto_stretch


def test_fits_and_tiff_roundtrip(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    data = np.random.default_rng(0).uniform(0, 1000, (3, 20, 30)).astype(np.float32)
    fits.PrimaryHDU(data).writeto(tmp_path / "a.fits")
    img = load_image(tmp_path / "a.fits")
    assert img.shape == (20, 30, 3)
    save_tiff16(tmp_path / "b.tif", img)
    back = load_image(tmp_path / "b.tif")
    assert back.shape == (20, 30, 3) and back.max() == 65535
    assert [p.name for p in list_images(tmp_path)] == ["a.fits", "b.tif"]


def test_auto_stretch_range_and_sky_level():
    rng = np.random.default_rng(0)
    img = rng.normal(100, 5, (200, 200, 3)).astype(np.float32) * np.array([1.2, 1.0, 0.7], np.float32)
    img[100, 100] = 5000
    out = auto_stretch(img)
    assert out.min() >= 0 and out.max() <= 1
    sky = np.median(out, axis=(0, 1))
    assert sky == pytest.approx([0.15, 0.15, 0.15], abs=0.03)  # neutral, at the target level
