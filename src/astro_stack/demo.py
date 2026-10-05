"""`astro-stack demo`: synthesize a dataset, stack it, and draw the README images."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .calibrate import master_frame, normalize_flat
from .io import to_luminance
from .pipeline import run
from .stars import robust_sigma
from .stretch import auto_stretch
from .synth import SkyModel, Sensor, SynthConfig, _sky_glow, generate_dataset, similarity_matrix


def _to_pil(image01: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(image01, 0, 1) * 255).round().astype(np.uint8))


def _label(img: Image.Image, text: str) -> Image.Image:
    font = ImageFont.load_default(size=max(14, img.width // 28))
    draw = ImageDraw.Draw(img)
    pad = 6
    box = draw.textbbox((pad, pad), text, font=font)
    draw.rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return img


def _comparison(panels: list[tuple[str, np.ndarray]], zoom_box: tuple[int, int, int, int]) -> Image.Image:
    """Full views on top, zoomed crops of the same region underneath."""
    x0, y0, x1, y1 = zoom_box
    tops = [_label(_to_pil(img), name) for name, img in panels]
    crops = [_to_pil(img[y0:y1, x0:x1]).resize(((x1 - x0) * 3, (y1 - y0) * 3), Image.Resampling.NEAREST) for _, img in panels]
    w, h = tops[0].size
    cw, ch = crops[0].size
    scale = w / cw
    crops = [c.resize((w, int(ch * scale)), Image.Resampling.NEAREST) for c in crops]
    gap = 8
    sheet = Image.new("RGB", (len(panels) * w + (len(panels) - 1) * gap, h + gap + crops[0].height), (20, 20, 20))
    for k, (top, crop) in enumerate(zip(tops, crops)):
        x = k * (w + gap)
        sheet.paste(top, (x, 0))
        ImageDraw.Draw(sheet).rectangle((x + x0, y0, x + x1, y1), outline=(255, 210, 0), width=2)
        sheet.paste(crop, (x, h + gap))
    return sheet


def _truth_noise(data_dir: Path, result, cfg: SynthConfig) -> float:
    """Noise of the stack measured against the noise-free sky (only possible here)."""
    truth = json.loads((data_dir / "truth.json").read_text())
    frame = next(f for f in truth["frames"] if f["file"] == result.report["reference"])
    p = frame["sky_to_frame"]
    rng = np.random.default_rng(cfg.seed)  # same draws as generate_dataset
    sky, sensor = SkyModel(cfg, rng), Sensor(cfg, rng)
    m = similarity_matrix(p["dx"], p["dy"], p["rotation_deg"], p["scale"], tuple(p["center"]))
    dark, _ = master_frame(data_dir / "darks")
    flat, _ = master_frame(data_dir / "flats")
    clean = (sky.render(m, frame["fwhm"]) + _sky_glow(cfg)) * sensor.flat / normalize_flat(flat, dark)
    rows, cols = result.crop
    return robust_sigma(to_luminance(result.stacks[result.report["method"]] - clean[rows, cols]))


def run_demo(out_dir: str | Path, docs_dir: str | Path | None, n_frames: int = 20, seed: int = 7, method: str = "sigma") -> dict:
    out_dir = Path(out_dir)
    data_dir, result_dir = out_dir / "data", out_dir / "result"
    cfg = SynthConfig(n_frames=n_frames, seed=seed)
    cfg.trail_frame = cfg.trail_frame if cfg.trail_frame < n_frames else None
    cfg.blurry_frame = cfg.blurry_frame if cfg.blurry_frame < n_frames else None
    print(f"Generating {n_frames} synthetic frames (+{cfg.n_darks} darks, {cfg.n_flats} flats) in {data_dir}")
    generate_dataset(data_dir, cfg)

    extra = tuple(m for m in ("mean",) if m != method)
    result = run(
        data_dir / "lights", result_dir, data_dir / "darks", data_dir / "flats", method=method, extra_methods=extra
    )

    truth_sigma = _truth_noise(data_dir, result, cfg)
    snr = result.report["snr"]
    print(
        f"Check against the noise-free truth: stack noise {truth_sigma:.3f} "
        f"(even/odd estimate {snr['stack_noise']:.3f})"
    )
    result.report["snr"]["stack_noise_vs_truth"] = round(truth_sigma, 4)
    (result_dir / "report.json").write_text(json.dumps(result.report, indent=2))

    if docs_dir is not None:
        docs = Path(docs_dir)
        docs.mkdir(parents=True, exist_ok=True)
        before = auto_stretch(result.reference_raw)
        after = auto_stretch(result.stacks[method])
        _label(_to_pil(before), "1 raw frame").save(docs / "before.png")
        _label(_to_pil(after), f"{result.report['frames_used']} frames, calibrated + {method} stack").save(docs / "after.png")

        # Zoom on the galaxy and the stretch of sky the satellite crossed.
        h, w = after.shape[:2]
        box = (int(0.50 * w), int(0.24 * h), int(0.86 * w), int(0.72 * h))
        mean_label = "mean stack (trail ghost)" if method == "sigma" else "mean stack"
        panels = [
            ("1 calibrated frame", auto_stretch(result.reference_aligned)),
            (mean_label, auto_stretch(result.stacks.get("mean", result.stacks[method]))),
            (f"{method} stack", after),
        ]
        _comparison(panels, box).save(docs / "comparison.jpg", quality=90)
        print(f"Wrote {docs / 'before.png'}, {docs / 'after.png'}, {docs / 'comparison.jpg'}")
    print(f"Measured SNR gain x{snr['measured_gain']:.2f} with {result.report['frames_used']} frames (sqrt(N) = {math.sqrt(result.report['frames_used']):.2f})")
    return result.report
