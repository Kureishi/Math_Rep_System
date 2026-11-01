"""Tests for modules/sm_partition.py: partition functions, thermodynamics, closed forms, ideal gas."""
import math

import numpy as np
import pytest

from modules import sm_partition as sp_

KT = np.geomspace(0.05, 20, 60)


def _checks(spec, kT=KT, kB=1.0):
    th = sp_.thermodynamics(spec, kT, kB)
    return th, sp_.verify_thermo(spec, th)


@pytest.mark.parametrize("spec", [sp_.two_level(1.0), sp_.two_level(0.7, 2, 3), sp_.harmonic_oscillator(1.0), sp_.harmonic_oscillator(2.0, False),
                                  sp_.rigid_rotor(1.0), sp_.explicit_levels([0, 1, 1, 3], [1, 2, 1, 5]),
                                  sp_.from_expression("n^2", "2*n+1", 1, 30), sp_.from_expression("a*n", "1", 0, None, {"a": 0.5})])
def test_every_builtin_check_passes(spec):
    th, ck = _checks(spec)
    assert ck.passed, [(c.label, c.detail) for c in ck.checks if not c.passed]
    assert len(ck.checks) >= 4


def test_two_level_closed_form_values():
    th = sp_.thermodynamics(sp_.two_level(1.0), [1.0])
    z = 1 + math.exp(-1)
    assert th.lnZ[0] == pytest.approx(math.log(z)) and th.U[0] == pytest.approx(math.exp(-1) / z)
    assert th.C[0] == pytest.approx(math.exp(-1) / z ** 2)                       # (D/kT)^2 e^{-D/kT} / Z^2


def test_harmonic_oscillator_limits():
    spec = sp_.harmonic_oscillator(1.0)
    th = sp_.thermodynamics(spec, [0.02, 100.0])
    assert th.U[0] == pytest.approx(0.5, abs=1e-6)                               # zero-point energy
    assert th.C[1] == pytest.approx(1.0, abs=1e-3)                               # equipartition: C -> k
    assert th.U[1] == pytest.approx(100.0, rel=1e-3)


def test_rotor_high_temperature_is_classical():
    th = sp_.thermodynamics(sp_.rigid_rotor(1.0), [200.0])
    assert th.C[0] == pytest.approx(1.0, abs=2e-3)                               # C -> k for a linear rotor
    assert th.lnZ[0] == pytest.approx(math.log(200.0 * (1 + 1 / 600 + 1 / 15 / 200 ** 2)), abs=1e-6)


def test_third_law_with_degenerate_ground_state():
    th = sp_.thermodynamics(sp_.explicit_levels([0, 0, 0, 2.0]), [0.05])
    assert th.S[0] == pytest.approx(math.log(3), abs=1e-6) and th.g0 == 3


def test_k_B_scales_entropy_and_heat_capacity_but_not_energy():
    a = sp_.thermodynamics(sp_.two_level(1.0), [1.0], 1.0)
    b = sp_.thermodynamics(sp_.two_level(1.0), [1.0], 3.0)
    assert b.U[0] == pytest.approx(a.U[0]) and b.F[0] == pytest.approx(a.F[0])
    assert b.S[0] == pytest.approx(3 * a.S[0]) and b.C[0] == pytest.approx(3 * a.C[0])
    assert sp_.verify_thermo(sp_.two_level(1.0), b).passed


def test_schottky_peak_numerical_and_root_agree():
    xn, xr = sp_.schottky_peak(1.0)
    assert xn == pytest.approx(xr, abs=1e-4) and xr == pytest.approx(2.3994, abs=1e-3)


def test_low_temperature_does_not_overflow():
    th = sp_.thermodynamics(sp_.harmonic_oscillator(100.0), [0.01, 0.02])
    assert np.all(np.isfinite(th.F)) and np.all(np.isfinite(th.S)) and th.S[0] == pytest.approx(0.0, abs=1e-12)


def test_a_wrong_closed_form_is_caught():
    spec = sp_.two_level(1.0)
    spec.closed_form.funcs["U"] = lambda kT: np.zeros_like(kT)                    # sabotage
    th = sp_.thermodynamics(spec, KT)
    ck = sp_.verify_thermo(spec, th)
    assert not ck.passed and any("closed form" in c.label and not c.passed for c in ck.checks)


def test_a_wrong_entropy_is_caught():
    spec = sp_.explicit_levels([0, 1, 2])
    th = sp_.thermodynamics(spec, KT)
    th.S_gibbs = th.S_gibbs * 1.01
    assert not sp_.verify_thermo(spec, th).passed


# ---------------------------------------------------------------- input handling
def test_divergent_spectrum_is_refused_not_summed():
    with pytest.raises(sp_.StatMechError, match="does not converge"):
        sp_.thermodynamics(sp_.from_expression("log(n+2)"), KT)


@pytest.mark.parametrize("build", [lambda: sp_.two_level(-1), lambda: sp_.harmonic_oscillator(0), lambda: sp_.rigid_rotor(-2),
                                   lambda: sp_.explicit_levels([0, 1], [1]), lambda: sp_.explicit_levels([0, 1], [1, -1]),
                                   lambda: sp_.from_expression("n", "1", 5, 2), lambda: sp_.from_expression("1/(n-3)", "1", 0, 10),
                                   lambda: sp_.from_expression("n", "0-1", 0, 10), lambda: sp_.from_expression("__import__('os')", "1", 0, 5),
                                   lambda: sp_.from_expression("n", "1", 0, 10 ** 7)])
def test_bad_spectra_are_refused_with_a_reason(build):
    with pytest.raises(sp_.StatMechError):
        build()


def test_bad_temperatures_are_refused():
    with pytest.raises(sp_.StatMechError):
        sp_.thermodynamics(sp_.two_level(1.0), [0.0, 1.0])
    with pytest.raises(sp_.StatMechError):
        sp_.thermodynamics(sp_.two_level(1.0), [1.0], kB=0)


def test_populations_sum_to_one_and_follow_boltzmann():
    spec = sp_.explicit_levels([0, 1, 2], [1, 1, 1])
    th = sp_.thermodynamics(spec, [1.0], show_levels=3)
    assert th.populations.sum() == pytest.approx(1.0)
    assert th.populations[0, 1] / th.populations[0, 0] == pytest.approx(math.exp(-1))


# ---------------------------------------------------------------- ideal gas
def test_ideal_gas_checks_pass_and_entropy_is_sackur_tetrode():
    g = sp_.ideal_gas(1000, 1e6, [0.5, 1.0, 2.0])
    assert g.checks.passed, [(c.label, c.detail) for c in g.checks.checks if not c.passed]
    assert np.allclose(g.U, 1.5 * 1000 * g.kT) and np.allclose(g.C, 1500.0)
    assert np.max(np.abs(g.S - g.S_sackur_tetrode)) < 0.5 * math.log(2 * math.pi * 1000) + 1e-3


def test_gibbs_paradox_extensivity_needs_the_factorial():
    g = sp_.ideal_gas(500, 1e5, [1.0])
    big = sp_.ideal_gas(1000, 2e5, [1.0])
    assert abs(big.S[0] - 2 * g.S[0]) / g.S[0] < 0.01                              # extensive with 1/N!
    assert abs(big.S_without_gibbs_factor[0] - 2 * g.S_without_gibbs_factor[0]) / g.S[0] > 0.1   # not without


def test_box_sum_approaches_V_over_lambda_cubed():
    lam = float(sp_.thermal_wavelength(1.0)[0])
    for L in (20 * lam, 60 * lam):
        z = sp_.box_translational_Z1(L, 1.0)
        assert z == pytest.approx(L ** 3 / lam ** 3, rel=3 * lam / L)


def test_ideal_gas_validation():
    with pytest.raises(sp_.StatMechError):
        sp_.ideal_gas(0, 1.0, [1.0])
    with pytest.raises(sp_.StatMechError):
        sp_.ideal_gas(10, 1.0, [-1.0])
