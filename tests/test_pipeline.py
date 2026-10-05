import json
import math

import numpy as np
import pytest
from PIL import Image

from astro_stack.cli import main
from astro_stack.pipeline import run
from astro_stack.synth import SynthConfig, generate_dataset, similarity_matrix


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    cfg = SynthConfig(width=480, height=360, n_frames=10, n_darks=6, n_flats=4, n_stars=200, trail_frame=3, blurry_frame=6)
    truth = generate_dataset(out, cfg)
    return out, truth


def test_pipeline_end_to_end(dataset, tmp_path):
    data, truth = dataset
    res = run(data / "lights", tmp_path, data / "darks", data / "flats", log=lambda s: None)
    rep = json.loads((tmp_path / "report.json").read_text())
    assert (tmp_path / "stack.tif").exists() and (tmp_path / "stack_preview.png").exists()

    status = {f["file"]: f for f in rep["frames"]}
    assert status["light_007.tif"]["status"] == "rejected"  # the blurry one
    assert "blurry" in status["light_007.tif"]["reason"]
    assert rep["frames_used"] == 9

    # Every reported transform matches the ground truth (frame -> reference).
    def m(f):
        p = f["sky_to_frame"]
        return similarity_matrix(p["dx"], p["dy"], p["rotation_deg"], p["scale"], tuple(p["center"]))

    ref = m(next(f for f in truth["frames"] if f["file"] == rep["reference"]))
    grid = np.array([[x, y, 1.0] for x in (40, 240, 440) for y in (40, 180, 320)])
    for f in truth["frames"]:
        got = status[f["file"]]
        if got["status"] != "used":
            continue
        t = got["transform"]
        c, s = np.cos(np.radians(t["rotation_deg"])) * t["scale"], np.sin(np.radians(t["rotation_deg"])) * t["scale"]
        est = np.array([[c, -s, t["dx"]], [s, c, t["dy"]], [0, 0, 1]])
        want = ref @ np.linalg.inv(m(f))
        # Sub-pixel agreement everywhere in the frame.
        assert np.abs(grid @ est.T - grid @ want.T)[:, :2].max() < 0.5, f["file"]

    gain, ideal = rep["snr"]["measured_gain"], rep["snr"]["ideal_gain_sqrt_n"]
    assert ideal == pytest.approx(math.sqrt(9))
    assert 0.8 * ideal < gain < 1.2 * ideal
    assert res.stacks["sigma"].ndim == 3


def test_pipeline_without_calibration(dataset, tmp_path):
    data, _ = dataset
    res = run(data / "lights", tmp_path, method="median", crop=False, log=lambda s: None)
    assert res.stacks["median"].shape == (360, 480, 3)


def test_pipeline_reads_8bit_png(dataset, tmp_path):
    data, _ = dataset
    lights = tmp_path / "png"
    lights.mkdir()
    for f in sorted((data / "lights").glob("*.tif"))[:5]:
        from astro_stack.io import load_image

        img = load_image(f)
        img8 = np.clip((img - 500) / 40, 0, 255).astype(np.uint8)
        Image.fromarray(img8).save(lights / (f.stem + ".png"))
    rep = run(lights, tmp_path / "out", log=lambda s: None).report
    assert rep["frames_used"] >= 4


def test_cli_stack_and_errors(dataset, tmp_path, capsys):
    data, _ = dataset
    assert main(["stack", str(data / "lights"), "-o", str(tmp_path), "--darks", str(data / "darks")]) == 0
    assert "SNR gain" in capsys.readouterr().out
    assert main(["stack", str(tmp_path / "missing")]) == 1


def test_cli_demo(tmp_path):
    assert main(["demo", "-o", str(tmp_path / "demo"), "--docs-dir", str(tmp_path / "docs"), "-n", "6"]) == 0
    for name in ("before.png", "after.png", "comparison.jpg"):
        assert (tmp_path / "docs" / name).exists()
