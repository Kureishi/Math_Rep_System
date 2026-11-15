"""The waves and optics engine: every result is compared with a closed form it did not use."""
import math

import numpy as np
import pytest

from modules import gpu
from modules import waves as wv

LAM, L, A = 550e-9, 1.5, 50e-6


def checks(result):
    return {c.label: c for c in result.checks.checks}


def test_single_slit_first_minimum_is_lambda_L_over_a():
    p = wv.diffraction_1d("single slit", LAM, L, A)
    assert p.checks.passed and p.envelope_zero_measured == pytest.approx(LAM * L / A, rel=0.01)


@pytest.mark.parametrize("d", [70e-6, 100e-6, 250e-6, 1e-3])
def test_double_slit_fringe_spacing_envelope_and_energy(d):
    p = wv.diffraction_1d("double slit", LAM, L, A, d)
    assert p.checks.passed, [c.detail for c in p.checks.checks if not c.passed]
    assert p.fringe_spacing_measured == pytest.approx(LAM * L / d, rel=0.01)


def test_the_slit_width_is_recovered_from_the_fringe_heights():
    p = wv.diffraction_1d("double slit", LAM, L, A, 250e-6)
    assert p.envelope_zero_measured == pytest.approx(LAM * L / A, rel=0.01)


@pytest.mark.parametrize("n", [3, 5, 12])
def test_grating_principal_maxima_are_lambda_L_over_d_apart(n):
    p = wv.diffraction_1d("grating", LAM, L, A, 200e-6, n)
    assert p.checks.passed and p.features_kind == "principal maxima"
    assert p.fringe_spacing_measured == pytest.approx(LAM * L / 200e-6, rel=0.02)


def test_more_slits_make_the_maxima_narrower():
    def width(n):
        p = wv.diffraction_1d("grating", LAM, L, A, 200e-6, n)
        i = int(np.argmax(p.intensity))
        half = p.intensity[i] / 2
        j = i
        while p.intensity[j] > half:
            j += 1
        return p.x[j] - p.x[i]
    assert width(12) < width(3) / 2


def test_the_fft_matches_the_closed_form_pattern_pointwise():
    p = wv.diffraction_1d("double slit", LAM, L, A, 250e-6)
    assert float(np.max(np.abs(p.intensity - p.analytic))) < 5e-3


def test_the_aperture_is_placed_with_subsample_accuracy():
    """A separation that is not a multiple of the sample spacing still gives the right fringes."""
    p = wv.diffraction_1d("double slit", LAM, L, A, 137.3e-6)
    assert p.fringe_spacing_measured == pytest.approx(LAM * L / 137.3e-6, rel=0.005)


@pytest.mark.parametrize("kwargs, message", [
    (dict(kind="triple", wavelength=LAM, distance=L, slit_width=A), "Choose one of"),
    (dict(kind="double slit", wavelength=-1, distance=L, slit_width=A, separation=1e-4), "wavelength"),
    (dict(kind="double slit", wavelength=LAM, distance=L, slit_width=A, separation=A), "1.2 times"),
    (dict(kind="grating", wavelength=LAM, distance=L, slit_width=A, separation=1e-4, n_slits=2), "between 3 and 20"),
    (dict(kind="double slit", wavelength=LAM, distance=L, slit_width=1e-6, separation=1e-3), "too large"),
])
def test_bad_input_is_explained(kwargs, message):
    with pytest.raises(wv.WavesError, match=message):
        wv.diffraction_1d(**kwargs)


def test_a_circular_aperture_gives_the_airy_pattern():
    p = wv.diffraction_2d("circular", LAM, L, 300e-6)
    c = checks(p)
    assert p.checks.passed
    assert p.first_zero_measured == pytest.approx(wv.AIRY_FIRST_ZERO * LAM * L / 300e-6, rel=0.02)
    assert abs(float(c["83.8 % of the light falls inside the first dark ring"].passed)) == 1.0


def test_a_rectangular_aperture_has_zeros_at_lambda_L_over_each_side():
    p = wv.diffraction_2d("rectangular", LAM, L, 200e-6, 400e-6)
    assert p.checks.passed


def test_a_2d_double_slit_has_fringes_lambda_L_over_d_apart():
    p = wv.diffraction_2d("double slit", LAM, L, 50e-6, separation=250e-6)
    assert p.checks.passed and p.first_zero_measured == pytest.approx(LAM * L / 250e-6, rel=0.03)


def test_2d_input_is_validated():
    with pytest.raises(wv.WavesError, match="too large for this many pixels"):
        wv.diffraction_2d("double slit", LAM, L, 50e-6, separation=5e-3, n_pixels=256)
    with pytest.raises(wv.WavesError, match="Choose one of"):
        wv.diffraction_2d("hexagon", LAM, L, 50e-6)


def test_the_diffraction_fft_runs_the_same_on_a_pretend_gpu():
    cpu = wv.diffraction_2d("circular", LAM, L, 300e-6, n_pixels=512)
    fake = gpu.Backend(np, "GPU: pretend (CuPy)", True, "numpy")
    with gpu.pretend_gpu(fake):
        g = wv.diffraction_2d("circular", LAM, L, 300e-6, n_pixels=512, backend=gpu.select(512 * 512, "fft_pixels"))
    assert g.backend == "GPU: pretend (CuPy)" and np.allclose(cpu.intensity, g.intensity)


# ------------------------------------------------------------------------------------------ ripple tank

def test_the_numerical_wavelength_is_a_little_shorter_than_nominal_and_tends_to_it():
    assert 11.8 < wv.numerical_wavelength(12.0, 0.5) < 12.0
    assert wv.numerical_wavelength(40.0, 0.5) == pytest.approx(40.0, rel=2e-3)


def test_a_single_source_decays_as_one_over_root_r_with_the_schemes_wavelength():
    t = wv.ripple_tank(241, 12.0, [wv.Source(120, 120)])
    assert t.checks.passed, [c.detail for c in t.checks.checks if not c.passed]
    assert t.frames.shape[1:] == (241, 241) and t.period_steps == 24


@pytest.mark.parametrize("phase", [0.0, math.pi])
def test_two_sources_interfere_where_the_path_difference_says(phase):
    t = wv.ripple_tank(241, 12.0, [wv.Source(102, 120, 0.0), wv.Source(138, 120, phase)])
    assert t.checks.passed, [c.detail for c in t.checks.checks if not c.passed]
    assert len(t.maxima) >= 4
    assert max(abs(a - b) for a, b in t.maxima) < 2.5


def test_opposite_phase_swaps_maxima_and_minima_on_the_axis():
    in_phase = wv.ripple_tank(241, 12.0, [wv.Source(102, 120), wv.Source(138, 120)])
    anti = wv.ripple_tank(241, 12.0, [wv.Source(102, 120), wv.Source(138, 120, math.pi)])
    row = lambda t: float(t.amplitude[120 - 60, 120])        # on the perpendicular bisector, above the sources  # noqa: E731
    assert row(in_phase) > 3 * row(anti)


def test_sources_too_close_are_flagged_by_a_check():
    t = wv.ripple_tank(241, 12.0, [wv.Source(110, 120), wv.Source(124, 120)])
    assert not t.checks.passed and "far enough apart" in " ".join(c.label for c in t.checks.checks if not c.passed)


@pytest.mark.parametrize("kwargs, message", [
    (dict(n=100), "odd grid"), (dict(wavelength_cells=3), "cells per wavelength"), (dict(courant=0.9), "Courant"),
    (dict(sources=[wv.Source(10, 10)]), "35 cells"), (dict(sources=[]), "between 1 and 4"),
])
def test_ripple_tank_input_is_validated(kwargs, message):
    with pytest.raises(wv.WavesError, match=message):
        wv.ripple_tank(**{"n": 241, **kwargs})


def test_the_ripple_tank_runs_the_same_on_a_pretend_gpu():
    srcs = [wv.Source(102, 120), wv.Source(138, 120)]
    cpu = wv.ripple_tank(241, 12.0, srcs)
    fake = gpu.Backend(np, "GPU: pretend (CuPy)", True, "numpy")
    with gpu.pretend_gpu(fake):
        g = wv.ripple_tank(241, 12.0, srcs, backend=gpu.select(241 * 241, "wave_cells"))
    assert g.backend == "GPU: pretend (CuPy)" and np.allclose(cpu.amplitude, g.amplitude)


# ------------------------------------------------------------------------------------------ Doppler

def test_collinear_doppler_matches_the_formula_to_rounding():
    r = wv.doppler(343, 440, 30, 0, (-500, 0), (0, 0), (0, 0), 200)
    assert r.checks.passed
    assert r.checks.checks[0].detail.count("e-1") == 1
    assert wv.doppler_collinear(440, 343, 30, 0) == pytest.approx(482.173, rel=1e-5)


def test_a_moving_observer_adds_its_speed():
    r = wv.doppler(343, 440, 30, 0, (-500, 0), (100, 0), (-20, 0), 200)
    assert r.checks.passed
    assert wv.doppler_collinear(440, 343, 30, 20) == pytest.approx(440 * 363 / 313)


def test_a_receding_source_is_heard_lower():
    r = wv.doppler(343, 440, 30, 180, (-200, 0), (0, 0), (0, 0), 200)
    assert r.checks.passed and wv.doppler_collinear(440, 343, -30, 0) < 440


def test_a_passing_source_goes_from_high_to_low_through_f0():
    r = wv.doppler(343, 440, 40, 0, (-150, 0), (0, 30), (0, 0), 3300)
    assert r.checks.passed and len(r.checks.checks) == 2
    f = r.inst_frequency[np.isfinite(r.inst_frequency)]
    assert f[0] > 440 > f[-1] and f[0] == pytest.approx(440 / (1 - 40 / 343 * 0.981), rel=0.01)
    assert f[-1] == pytest.approx(440 / (1 + 40 / 343 * 0.981), rel=0.01)


def test_the_mach_cone_angle_is_arcsin_c_over_v():
    r = wv.doppler(343, 50, 600, 0, (-1500, 0), (0, 50), (0, 0), 300)
    assert r.checks.passed and r.mach_angle_theory == pytest.approx(math.degrees(math.asin(343 / 600)))
    assert r.mach_angle_measured == pytest.approx(r.mach_angle_theory, rel=0.01)


def test_at_the_wave_speed_there_is_a_note_not_a_failure():
    r = wv.doppler(343, 440, 343, 0, (-1500, 0), (0, 50), (0, 0), 300)
    assert r.notes and not r.checks.checks


def test_doppler_input_is_validated():
    with pytest.raises(wv.WavesError, match="no single Doppler"):
        wv.doppler_collinear(440, 343, 343, 0)
    with pytest.raises(wv.WavesError, match="positive"):
        wv.doppler(-1, 440, 10)
    with pytest.raises(wv.WavesError, match="slower than the wave"):
        wv.doppler(343, 440, 10, observer_velocity=(400, 0))
    with pytest.raises(wv.WavesError, match="between 20"):
        wv.doppler(343, 440, 10, n_waves=3)
