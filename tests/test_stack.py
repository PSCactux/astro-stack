import math

import numpy as np
import pytest

from astro_stack.stack import combine, mean_stack, median_stack, sigma_clip_stack
from astro_stack.stars import robust_sigma

SIGMA = 10.0


def _frames(n, shape=(120, 160), seed=0):
    """A fixed smooth scene plus independent Gaussian noise in each frame."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    scene = 200 + 50 * np.exp(-((xx - 80) ** 2 + (yy - 60) ** 2) / 800)
    return scene, scene + rng.normal(0, SIGMA, (n,) + shape)


def test_sigma_clip_removes_satellite_trail_and_hot_pixels():
    scene, cube = _frames(12)
    rng = np.random.default_rng(1)
    # Satellite trail across frame 4.
    rows = np.arange(cube.shape[2]) * 0.5 + 20
    cube[4, rows.astype(int), np.arange(cube.shape[2])] += 3000
    # Hot pixels land on different sky pixels in each frame after alignment.
    for k in range(len(cube)):
        ys, xs = rng.integers(0, 120, 40), rng.integers(0, 160, 40)
        cube[k, ys, xs] += rng.uniform(500, 5000, 40)

    mean_err = np.abs(mean_stack(cube) - scene)
    clip_err = np.abs(sigma_clip_stack(cube) - scene)
    assert mean_err.max() > 200  # the mean keeps a visible ghost
    # Clipped result is within ordinary noise everywhere (5 sigma of the stack).
    assert clip_err.max() < 5 * SIGMA / math.sqrt(len(cube))


def test_sigma_clip_handles_few_frames():
    scene, cube = _frames(5)
    cube[2, 50, 50] += 4000
    assert abs(sigma_clip_stack(cube)[50, 50] - scene[50, 50]) < 4 * SIGMA


@pytest.mark.parametrize("n", [4, 9, 16, 25])
def test_snr_grows_with_sqrt_n(n):
    scene, cube = _frames(n, seed=n)
    single = robust_sigma(cube[0] - scene)
    for method, efficiency in [("mean", 1.0), ("sigma", 0.97), ("median", math.sqrt(2 / math.pi))]:
        stacked = combine(cube, method)
        gain = single / robust_sigma(stacked - scene)
        expected = math.sqrt(n) * (efficiency if n > 4 or method != "median" else 0.85)
        assert gain == pytest.approx(expected, rel=0.12), method


def test_nan_pixels_are_ignored():
    cube = np.ones((3, 4, 4), np.float32)
    cube[0, 0, 0] = np.nan
    cube[1, 0, 0] = 5
    assert mean_stack(cube)[0, 0] == pytest.approx(3)
    assert median_stack(cube)[0, 0] == pytest.approx(3)
    assert np.isfinite(sigma_clip_stack(cube)).all()


def test_color_cube_is_stacked_per_channel():
    rng = np.random.default_rng(0)
    cube = rng.normal(0, 1, (8, 10, 12, 3)).astype(np.float32) + np.array([1, 2, 3], np.float32)
    out = combine(cube, "sigma", chunk_rows=3)
    assert out.shape == (10, 12, 3)
    assert out.mean(axis=(0, 1)) == pytest.approx([1, 2, 3], abs=0.15)
