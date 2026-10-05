"""Synthetic astrophotography dataset for the demo and the tests.

A fixed "sky" (stars with Moffat PSFs, a faint galaxy and a faint red
nebula) is rendered into every frame through a different similarity
transform (drift + field rotation of an alt-az mount), then degraded the way
a small sensor degrades it: vignetting, light-polluted sky with a gradient,
photon (shot) noise, read noise, bias, dark current and hot pixels. One frame
gets a satellite trail and one is blurred by bad seeing / wind shake.
Values are in electrons with gain 1, written as 16-bit TIFF files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tifffile


@dataclass
class SynthConfig:
    width: int = 800
    height: int = 600
    n_frames: int = 20
    n_darks: int = 15
    n_flats: int = 10
    n_stars: int = 350
    fwhm: float = 3.2
    moffat_beta: float = 3.0
    sky: float = 200.0  # sky background, e-/pixel
    read_noise: float = 8.0
    bias: float = 500.0
    dark_current: float = 20.0
    n_hot_pixels: int = 80
    drift_per_frame: tuple[float, float] = (1.6, -0.9)
    jitter: float = 2.0
    rotation_per_frame_deg: float = 0.08
    trail_frame: int | None = 6
    blurry_frame: int | None = 13
    seed: int = 7


def similarity_matrix(dx: float, dy: float, rot_deg: float, scale: float, center: tuple[float, float]) -> np.ndarray:
    """3x3 matrix: rotate/scale about ``center``, then translate by (dx, dy)."""
    c, s = np.cos(np.radians(rot_deg)) * scale, np.sin(np.radians(rot_deg)) * scale
    cx, cy = center
    return np.array(
        [
            [c, -s, cx - c * cx + s * cy + dx],
            [s, c, cy - s * cx - c * cy + dy],
            [0, 0, 1],
        ]
    )


def _moffat_stamps(xy: np.ndarray, flux: np.ndarray, fwhm: float, beta: float, shape: tuple[int, int], radius: int = 12) -> np.ndarray:
    """Render point sources with a Moffat profile (what real seeing looks like)."""
    h, w = shape
    alpha = fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))
    out = np.zeros(shape, dtype=np.float64)
    off = np.arange(-radius, radius + 1)
    ox, oy = np.meshgrid(off, off)
    ix, iy = np.round(xy[:, 0]).astype(int), np.round(xy[:, 1]).astype(int)
    px = ix[:, None, None] + ox  # (S, k, k) pixel coordinates
    py = iy[:, None, None] + oy
    r2 = (px - xy[:, 0, None, None]) ** 2 + (py - xy[:, 1, None, None]) ** 2
    prof = (beta - 1) / (np.pi * alpha**2) * (1 + r2 / alpha**2) ** (-beta)
    vals = prof * flux[:, None, None]
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    np.add.at(out, (py[ok], px[ok]), vals[ok])
    return out


class SkyModel:
    """The 'true' sky, in reference coordinates."""

    def __init__(self, cfg: SynthConfig, rng: np.random.Generator):
        margin = 80
        n = cfg.n_stars
        self.xy = np.column_stack(
            [rng.uniform(-margin, cfg.width + margin, n), rng.uniform(-margin, cfg.height + margin, n)]
        )
        # Faint stars are far more common than bright ones (power law in flux).
        self.flux = np.minimum(250.0 * rng.uniform(0.0, 1.0, n) ** (-1 / 0.8), 4e6)
        t = rng.uniform(0, 1, n)  # 0 = hot/blue, 1 = cool/red
        self.color = np.column_stack([0.85 + 0.35 * t, 0.95 + 0.05 * t, 1.2 - 0.5 * t])
        self.color /= self.color.mean(axis=1, keepdims=True)
        self.cfg = cfg

    def extended(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Diffuse objects (e-/pixel per channel) at sky coordinates x, y."""
        cfg = self.cfg
        # Elliptical galaxy-like disk + bulge.
        gx, gy, ang = 0.66 * cfg.width, 0.36 * cfg.height, np.radians(32)
        u = (x - gx) * np.cos(ang) + (y - gy) * np.sin(ang)
        v = -(x - gx) * np.sin(ang) + (y - gy) * np.cos(ang)
        r = np.sqrt((u / 42) ** 2 + (v / 15) ** 2)
        galaxy = 45 * np.exp(-r) + 140 * np.exp(-(u**2 + v**2) / (2 * 4.0**2))
        # Red emission nebula made of a few overlapping blobs.
        neb = np.zeros_like(x)
        for bx, by, s, a in [(0.28, 0.62, 70, 22), (0.36, 0.70, 50, 18), (0.22, 0.52, 45, 14), (0.33, 0.55, 30, 10)]:
            neb += a * np.exp(-((x - bx * cfg.width) ** 2 + (y - by * cfg.height) ** 2) / (2 * s**2))
        out = np.empty(x.shape + (3,))
        out[..., 0] = galaxy * 1.05 + neb * 1.0
        out[..., 1] = galaxy * 1.00 + neb * 0.30
        out[..., 2] = galaxy * 0.85 + neb * 0.45
        return out

    def render(self, matrix: np.ndarray, fwhm: float) -> np.ndarray:
        """Noise-free signal (e-) of a frame whose sky->frame transform is ``matrix``."""
        cfg = self.cfg
        shape = (cfg.height, cfg.width)
        pts = np.column_stack([self.xy, np.ones(len(self.xy))]) @ matrix.T
        inside = (pts[:, 0] > -20) & (pts[:, 0] < cfg.width + 20) & (pts[:, 1] > -20) & (pts[:, 1] < cfg.height + 20)
        img = np.zeros(shape + (3,))
        for c in range(3):
            img[..., c] = _moffat_stamps(pts[inside, :2], self.flux[inside] * self.color[inside, c], fwhm, cfg.moffat_beta, shape)
        # Diffuse light: sample the sky model at each frame pixel's sky position.
        yy, xx = np.mgrid[0 : cfg.height, 0 : cfg.width].astype(np.float64)
        inv = np.linalg.inv(matrix)
        sx = inv[0, 0] * xx + inv[0, 1] * yy + inv[0, 2]
        sy = inv[1, 0] * xx + inv[1, 1] * yy + inv[1, 2]
        return img + self.extended(sx, sy)


class Sensor:
    """Fixed-pattern properties of the camera: vignetting and hot pixels."""

    def __init__(self, cfg: SynthConfig, rng: np.random.Generator):
        self.cfg = cfg
        h, w = cfg.height, cfg.width
        yy, xx = np.mgrid[0:h, 0:w]
        r2 = ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
        vig = 1 - 0.38 * r2
        # A dust shadow ("donut") on the optics.
        d = np.sqrt((xx - 0.2 * w) ** 2 + (yy - 0.25 * h) ** 2)
        vig = vig * (1 - 0.07 * np.exp(-((d - 18) ** 2) / (2 * 5**2)))
        self.flat = np.repeat(vig[..., None], 3, axis=2) * np.array([1.0, 0.97, 0.92])
        self.hot = np.zeros((h, w, 3))
        idx = rng.integers(0, h * w, cfg.n_hot_pixels)
        ch = rng.integers(0, 3, cfg.n_hot_pixels)
        self.hot[idx // w, idx % w, ch] = rng.uniform(1500, 20000, cfg.n_hot_pixels)

    def expose(self, signal: np.ndarray, rng: np.random.Generator, exposure_scale: float = 1.0) -> np.ndarray:
        """Turn an optical signal (e-) into a 16-bit raw frame."""
        cfg = self.cfg
        thermal = (cfg.dark_current + self.hot) * exposure_scale
        electrons = rng.poisson(np.clip(signal * self.flat + thermal, 0, None)).astype(np.float64)
        electrons += rng.normal(0, cfg.read_noise, electrons.shape) + cfg.bias
        return np.clip(electrons, 0, 65535).astype(np.uint16)


def _sky_glow(cfg: SynthConfig) -> np.ndarray:
    """Light-polluted sky: orange tint and a gradient towards one corner."""
    yy, xx = np.mgrid[0 : cfg.height, 0 : cfg.width]
    grad = 1 + 0.3 * (xx / cfg.width) + 0.15 * (yy / cfg.height)
    return cfg.sky * grad[..., None] * np.array([1.12, 1.0, 0.8])


def _add_trail(img: np.ndarray, rng: np.random.Generator, brightness: float = 900.0) -> None:
    """A satellite crossing the frame: a bright straight streak."""
    h, w = img.shape[:2]
    p0 = np.array([0.0, rng.uniform(0.2, 0.5) * h])
    p1 = np.array([float(w), rng.uniform(0.5, 0.9) * h])
    direction = (p1 - p0) / np.linalg.norm(p1 - p0)
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.abs((xx - p0[0]) * direction[1] - (yy - p0[1]) * direction[0])
    img += (brightness * np.exp(-(dist**2) / (2 * 1.1**2)))[..., None] * np.array([1.0, 1.0, 0.95])


def generate_dataset(out_dir: str | Path, cfg: SynthConfig | None = None) -> dict:
    """Write lights/, darks/, flats/ and truth.json into ``out_dir``."""
    cfg = cfg or SynthConfig()
    out = Path(out_dir)
    for sub in ("lights", "darks", "flats"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for old in (out / sub).glob("*.tif"):
            old.unlink()
    rng = np.random.default_rng(cfg.seed)
    sky = SkyModel(cfg, rng)
    sensor = Sensor(cfg, rng)
    glow = _sky_glow(cfg)
    center = (cfg.width / 2, cfg.height / 2)

    truth = {"config": asdict(cfg), "frames": []}
    for i in range(cfg.n_frames):
        dx = cfg.drift_per_frame[0] * i + rng.normal(0, cfg.jitter)
        dy = cfg.drift_per_frame[1] * i + rng.normal(0, cfg.jitter)
        rot = cfg.rotation_per_frame_deg * i + rng.normal(0, 0.02)
        scale = 1 + rng.normal(0, 0.0008)  # tiny focus breathing
        fwhm = cfg.fwhm * (1 + 0.06 * rng.standard_normal())
        if i == cfg.blurry_frame:
            fwhm *= 2.4
        m = similarity_matrix(dx, dy, rot, scale, center)
        signal = sky.render(m, fwhm) + glow
        if i == cfg.trail_frame:
            _add_trail(signal, rng)
        name = f"light_{i + 1:03d}.tif"
        tifffile.imwrite(out / "lights" / name, sensor.expose(signal, rng), photometric="rgb")
        truth["frames"].append(
            {
                "file": name,
                "sky_to_frame": {"dx": dx, "dy": dy, "rotation_deg": rot, "scale": scale, "center": center},
                "fwhm": fwhm,
                "satellite_trail": i == cfg.trail_frame,
                "blurry": i == cfg.blurry_frame,
            }
        )

    for i in range(cfg.n_darks):  # lens cap on, same exposure as the lights
        tifffile.imwrite(out / "darks" / f"dark_{i + 1:03d}.tif", sensor.expose(np.zeros_like(glow), rng), photometric="rgb")
    for i in range(cfg.n_flats):  # evenly lit white target, short exposure
        tifffile.imwrite(
            out / "flats" / f"flat_{i + 1:03d}.tif",
            sensor.expose(np.full_like(glow, 25000.0), rng, exposure_scale=0.02),
            photometric="rgb",
        )
    (out / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth
