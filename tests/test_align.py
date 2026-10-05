import numpy as np
import pytest

from astro_stack.align import Alignment, apply_alignment, fit_similarity, match_stars, phase_align
from astro_stack.stars import detect_stars


@pytest.mark.parametrize(
    "dx, dy, rot, scale",
    [(15.0, -8.0, 7.0, 1.02), (-30.5, 12.25, -2.0, 1.0), (4.0, 4.0, 0.3, 0.97)],
)
def test_star_matching_recovers_similarity(star_field, dx, dy, rot, scale):
    true = star_field.matrix(dx, dy, rot, scale, star_field.center)  # reference -> frame
    ref, frame = star_field(), star_field(true, seed=1)
    al = match_stars(detect_stars(ref), detect_stars(frame))
    assert al.ok and al.method == "stars"
    est = al.transform
    assert np.degrees(est.rotation) == pytest.approx(rot, abs=0.05)
    assert est.scale == pytest.approx(scale, abs=2e-3)
    # Compare where both transforms send a grid of points across the image.
    pts = np.array([[x, y] for x in (50, 250, 450) for y in (50, 200, 350)], float)
    want = (np.column_stack([pts, np.ones(len(pts))]) @ true.T)[:, :2]
    assert np.abs(est(pts) - want).max() < 0.3
    assert al.rms < 0.5


def test_warped_frame_lines_up_with_reference(star_field):
    true = star_field.matrix(12.0, -5.0, 3.0, 1.0, star_field.center)
    ref, frame = star_field(noise=0.5), star_field(true, noise=0.5, seed=2)
    al = match_stars(detect_stars(ref), detect_stars(frame))
    aligned = apply_alignment(frame, al)
    ok = np.isfinite(aligned)
    before = np.corrcoef(ref.ravel(), frame.ravel())[0, 1]
    after = np.corrcoef(ref[ok], aligned[ok])[0, 1]
    assert before < 0.3 < 0.95 < after


def test_fit_similarity_exact():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, (10, 2))
    theta, s, t = np.radians(20), 1.1, np.array([5.0, -3.0])
    rot = s * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    tf = fit_similarity(src, src @ rot.T + t)
    assert np.degrees(tf.rotation) == pytest.approx(20)
    assert tf.scale == pytest.approx(1.1)
    assert tf.translation == pytest.approx(t)


def test_phase_correlation_fallback_recovers_translation(star_field):
    true = star_field.matrix(9.4, -6.2, 0.0, 1.0, star_field.center)
    ref, frame = star_field(), star_field(true, seed=4)
    al = phase_align(ref, frame)
    assert al.ok and al.method == "phase"
    assert al.transform.translation == pytest.approx([9.4, -6.2], abs=0.3)


def test_pure_noise_frame_is_not_aligned(star_field):
    ref = star_field()
    noise = np.random.default_rng(5).normal(100, 6, ref.shape).astype(np.float32)
    assert not match_stars(detect_stars(ref), detect_stars(noise)).ok
    assert not phase_align(ref, noise).ok


def test_identity_describe():
    assert Alignment(True).describe() == {"dx": 0.0, "dy": 0.0, "rotation_deg": 0.0, "scale": 1.0}
