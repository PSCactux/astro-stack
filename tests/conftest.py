import numpy as np
import pytest

from astro_stack.synth import SkyModel, SynthConfig, similarity_matrix


@pytest.fixture
def star_field():
    """Render a mono star field through a given sky->frame matrix, with noise."""
    cfg = SynthConfig(width=500, height=400, n_stars=220)
    sky = SkyModel(cfg, np.random.default_rng(3))

    def render(matrix=None, noise=6.0, seed=0, fwhm=3.0):
        m = np.eye(3) if matrix is None else matrix
        img = sky.render(m, fwhm).mean(axis=2) + 100.0
        rng = np.random.default_rng(seed)
        return (img + rng.normal(0, noise, img.shape)).astype(np.float32)

    render.center = (cfg.width / 2, cfg.height / 2)
    render.matrix = similarity_matrix
    return render
