"""Tests for modules/sm_meanfield.py: mean field, Bethe, and the exact 1D / 2D solutions."""
import math

import numpy as np
import pytest

from modules import sm_meanfield as mf


def test_all_internal_cross_checks_pass():
    for z in (4, 6):
        fs = mf.failure_summary(z)
        assert fs.checks.passed, [(c.label, c.detail) for c in fs.checks.checks if not c.passed]


def test_critical_temperatures_ordered_mean_field_bethe_exact():
    assert mf.mf_tc(4) == 4.0 and mf.bethe_tc(4) == pytest.approx(2.885390, abs=1e-6) and mf.onsager_tc() == pytest.approx(2.269185, abs=1e-6)
    assert mf.mf_tc(4) > mf.bethe_tc(4) > mf.onsager_tc()
    assert mf.bethe_tc(2) == 0.0
    with pytest.raises(mf.StatMechError):
        mf.bethe_tc(1)


def test_mean_field_solves_its_own_self_consistency():
    for T in (0.5, 2.0, 3.9):
        m = mf.mf_magnetisation(T)
        assert m == pytest.approx(math.tanh(4 * m / T), abs=1e-12)
    assert mf.mf_magnetisation(4.5) == 0.0 and mf.mf_magnetisation(1.0, 0.3) > mf.mf_magnetisation(1.0, 0.0) - 1e-12


def test_mean_field_susceptibility_curie_weiss():
    assert mf.mf_susceptibility(6.0) == pytest.approx(1 / (6.0 - 4.0), rel=1e-12)


def test_landau_free_energy_double_well_below_tc_single_well_above():
    m = np.linspace(-0.99, 0.99, 3999)
    assert abs(m[np.argmin(mf.mf_landau_free_energy(m, 2.0))]) > 0.5
    assert abs(m[np.argmin(mf.mf_landau_free_energy(m, 6.0))]) < 1e-3


def test_exponents_and_jump_of_mean_field():
    assert mf.fit_mf_exponent() == pytest.approx(0.5, abs=1e-3) and mf.mf_specific_heat_jump() == pytest.approx(1.5, abs=1e-3)


# ---------------------------------------------------------------- 1D
def test_1d_transfer_matrix_matches_brute_force():
    for N, T, h in [(6, 0.8, 0.0), (10, 1.5, 0.3), (14, 2.5, -0.2)]:
        assert mf.ising_1d_finite_lnZ(N, T, h) == pytest.approx(mf.ising_1d_brute_force_lnZ(N, T, h), abs=1e-10)
    with pytest.raises(mf.StatMechError):
        mf.ising_1d_brute_force_lnZ(25, 1.0)


def test_1d_has_no_spontaneous_magnetisation_and_exact_energy():
    assert mf.ising_1d_magnetisation(0.2, 0.0) == 0.0 and mf.ising_1d_magnetisation(1.0, 1e-3) < 0.1
    assert mf.ising_1d_energy(1.0) == pytest.approx(-math.tanh(1.0))
    # u = d(beta f)/d(beta) by finite difference
    b, d = 1.0, 1e-6
    g = lambda beta: beta * mf.ising_1d_free_energy(1 / beta)  # noqa: E731
    assert (g(b + d) - g(b - d)) / (2 * d) == pytest.approx(mf.ising_1d_energy(1.0), abs=1e-7)
    # c = du/dT
    dT = 1e-6
    assert (mf.ising_1d_energy(1 + dT) - mf.ising_1d_energy(1 - dT)) / (2 * dT) == pytest.approx(mf.ising_1d_specific_heat(1.0), abs=1e-6)


def test_bethe_with_z_2_is_exact_for_the_chain_but_mean_field_is_not():
    for T, h in [(0.5, 0.2), (2.0, 0.7), (5.0, 0.1)]:
        assert mf.bethe_magnetisation(T, h, 2) == pytest.approx(mf.ising_1d_magnetisation(T, h), abs=1e-9)
    assert abs(mf.mf_magnetisation(1.0, 0.3, 2) - mf.ising_1d_magnetisation(1.0, 0.3)) > 0.01
    assert mf.mf_magnetisation(1.0, 0.0, 2) > 0.9 and mf.ising_1d_magnetisation(1.0, 0.0) == 0.0      # the false transition


# ---------------------------------------------------------------- 2D
def test_onsager_energy_closed_form_equals_derivative_of_the_free_energy():
    for T in (1.2, 2.0, 2.5, 3.5):
        assert mf.onsager_energy_from_free_energy(T) == pytest.approx(mf.onsager_energy(T), abs=1e-7)


def test_onsager_limits_and_continuity():
    assert mf.onsager_energy(0.05) == pytest.approx(-2.0, abs=1e-6) and mf.onsager_energy(200.0) == pytest.approx(-2 / 200.0, rel=1e-2)
    tc = mf.onsager_tc()
    assert mf.onsager_energy(tc) == pytest.approx(-math.sqrt(2)) and mf.onsager_energy(tc * (1 + 1e-7)) == pytest.approx(-math.sqrt(2), abs=1e-4)
    assert mf.onsager_free_energy(1e6) == pytest.approx(-1e6 * math.log(2), rel=1e-6)
    with pytest.raises(mf.StatMechError):
        mf.onsager_energy(0.0)


def test_onsager_magnetisation_vanishes_at_tc_with_exponent_one_eighth():
    tc = mf.onsager_tc()
    assert mf.onsager_magnetisation(tc * 1.01) == 0.0 and mf.onsager_magnetisation(0.1) == pytest.approx(1.0, abs=1e-12)
    t = np.array([1e-6, 1e-4])
    m = np.array([mf.onsager_magnetisation(tc * (1 - x)) for x in t])
    assert np.polyfit(np.log(t), np.log(m), 1)[0] == pytest.approx(0.125, abs=2e-3)


def test_onsager_specific_heat_log_divergence_coefficient():
    assert mf.onsager_log_coefficient() == pytest.approx(0.4945, abs=1e-4) and mf.fit_onsager_log_coefficient() == pytest.approx(0.4945, abs=0.02)


def test_exact_2d_energy_matches_an_enumerated_lattice_at_high_temperature():
    """Large-T expansion: u = -2 J^2/T ... check the exact formula against the 4x4 lattice, whose finite size matters little at T = 8."""
    from modules.sm_lattice import exact_ising_enumeration
    assert mf.onsager_energy(8.0) == pytest.approx(exact_ising_enumeration(4, 8.0).e, abs=5e-3)


def test_mean_field_overestimates_order_and_misses_short_range_order():
    cv = mf.compare_curves(np.array([1.0, 2.0, 2.2, 3.0, 4.5]))
    assert np.all(cv.m_mf >= cv.m_exact_2d - 1e-9) and cv.m_mf[3] > 0 and cv.m_exact_2d[3] == 0
    assert cv.u_mf[3] == pytest.approx(-0.5 * 4 * cv.m_mf[3] ** 2) and cv.u_exact_2d[3] < -0.5            # exact has short-range order above Tc
    assert cv.u_mf[-1] == 0.0 and cv.u_exact_2d[-1] < 0


def test_bethe_magnetisation_between_mean_field_and_exact():
    T = 2.5
    assert mf.mf_magnetisation(T) > mf.bethe_magnetisation(T) > mf.onsager_magnetisation(T) - 1e-12
    assert mf.bethe_magnetisation(2.9, 0.0) < 1e-3                                  # above its own Tc = 2.885
