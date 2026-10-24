"""Tests for modules/sm_distributions.py: occupations, ideal quantum gases, BEC, classical limits."""
import math

import numpy as np
import pytest

from modules import sm_distributions as sd


@pytest.mark.parametrize("n,T,g", [(0.5, 1.0, 1), (0.05, 1.0, 1), (1.0, 0.3, 2), (0.01, 5.0, 1), (3.0, 0.05, 1)])
def test_every_consistency_check_passes(n, T, g):
    rep = sd.verify_gas_tools(n, T, g)
    assert rep.checks.passed, [(c.label, c.detail) for c in rep.checks.checks if not c.passed]
    assert len(rep.checks.checks) >= 10


def test_occupations_basic_properties():
    e = np.array([0.5, 1.0, 2.0])
    assert sd.occupation(e, 0.0, 1.0, "FD") == pytest.approx(1 / (np.exp(e) + 1))
    assert sd.occupation(e, 0.0, 1.0, "BE") == pytest.approx(1 / (np.exp(e) - 1))
    assert sd.occupation(e, 0.0, 1.0, "MB") == pytest.approx(np.exp(-e))
    assert np.all(sd.occupation(e, 0.0, 1.0, "FD") <= 1) and sd.occupation(0.0, 0.0, 1.0, "FD") == pytest.approx(0.5)
    assert sd.occupation(1e4, 0.0, 1.0, "FD") == pytest.approx(0.0, abs=1e-300) and np.isfinite(sd.occupation(-1e4, 0.0, 1.0, "FD"))
    assert sd.occupation(2.0, 0.0, 1.0, "BE") > sd.occupation(2.0, 0.0, 1.0, "MB") > sd.occupation(2.0, 0.0, 1.0, "FD")
    with pytest.raises(sd.StatMechError, match="above the chemical potential"):
        sd.occupation(0.5, 1.0, 1.0, "BE")
    with pytest.raises(sd.StatMechError):
        sd.occupation(1.0, 0.0, -1.0, "FD")
    with pytest.raises(sd.StatMechError):
        sd.occupation(1.0, 0.0, 1.0, "XX")


def test_fermi_energy_and_sommerfeld():
    ef = sd.fermi_energy(1.0, 2)
    assert ef == pytest.approx(0.5 * (3 * math.pi ** 2) ** (2 / 3))
    t = 0.03 * ef
    assert sd.solve_mu(1.0, t, "FD", 2).mu == pytest.approx(sd.sommerfeld_mu(1.0, t, 2), abs=3e-5 * ef)


def test_bose_condensation_threshold_and_fraction():
    tc = sd.bec_critical_temperature(1.0)
    assert tc == pytest.approx(2 * math.pi * (1 / sd.ZETA_32) ** (2 / 3))
    below = sd.solve_mu(1.0, 0.5 * tc, "BE")
    assert below.mu == 0.0 and below.condensate_fraction == pytest.approx(1 - 0.5 ** 1.5)
    above = sd.solve_mu(1.0, 1.2 * tc, "BE")
    assert above.mu < 0 and above.condensate_fraction == 0
    assert above.degeneracy < sd.ZETA_32 < sd.solve_mu(1.0, 0.9 * tc, "BE").degeneracy
    # mu approaches 0 continuously
    mus = [sd.solve_mu(1.0, f * tc, "BE").mu for f in (2.0, 1.2, 1.02, 1.002)]
    assert mus == sorted(mus) and abs(mus[-1]) < 1e-3 * tc * 100


def test_pressure_of_condensed_gas_depends_on_T_only():
    a, b = sd.solve_mu(1.0, 1.0, "BE"), sd.solve_mu(3.0, 1.0, "BE")
    assert a.condensate_fraction > 0 and b.pressure == pytest.approx(a.pressure)     # P = g T zeta(5/2)/lambda^3, independent of n


def test_classical_limit_all_statistics_agree_and_virial_signs():
    dil = [sd.solve_mu(1e-5, 1.0, k) for k in sd.KINDS]
    mb, fd, be = dil[0].mu, dil[1].mu, dil[2].mu
    assert fd > mb > be and fd - be < 5e-4                  # all agree to O(n lambda^3) ~ 1e-4; the order is the quantum correction's sign
    eos = sd.equation_of_state(np.array([1e-3]), 1.0, 1)
    assert eos["FD"][0] > 1 > eos["BE"][0] and eos["MB"][0] == pytest.approx(1.0)
    assert (eos["FD"][0] - 1) / 1e-3 == pytest.approx(2 ** -2.5, abs=1e-3)


def test_degenerate_fermi_pressure_exceeds_classical_by_a_lot():
    ef = sd.fermi_energy(1.0)
    st = sd.solve_mu(1.0, 0.01 * ef, "FD")
    assert st.pressure / (1.0 * 0.01 * ef) > 30 and st.pressure == pytest.approx(0.4 * ef, rel=2e-3)


def test_polylog_and_integral_routes_agree():
    for kind, mu in (("FD", 0.7), ("FD", -2.0), ("FD", 5.0), ("BE", -0.3), ("BE", -3.0)):
        assert sd.density(mu, 1.0, kind) == pytest.approx(sd.density_quad(mu, 1.0, kind), rel=1e-8)
        assert sd.pressure(mu, 1.0, kind) == pytest.approx(sd.pressure_quad(mu, 1.0, kind), rel=1e-8)


def test_mu_decreases_with_temperature_at_fixed_density():
    temps = np.array([0.5, 1.0, 2.0, 4.0])
    curves = sd.mu_curves(0.05, temps)
    for k in sd.KINDS:
        assert np.all(np.diff(curves[k]) < 0)


def test_input_validation():
    with pytest.raises(sd.StatMechError):
        sd.solve_mu(-1.0, 1.0, "FD")
    with pytest.raises(sd.StatMechError):
        sd.solve_mu(1.0, 1.0, "XX")
    with pytest.raises(sd.StatMechError):
        sd.density(0.5, 1.0, "BE")
    with pytest.raises(sd.StatMechError):
        sd.pressure(0.5, 1.0, "BE")


def test_maxwell_boltzmann_pdf_and_moments():
    from scipy.integrate import quad
    T = 1.7
    v = np.linspace(0, 40, 400001)
    pdf = sd.mb_speed_pdf(v, T)
    assert np.trapezoid(pdf, v) == pytest.approx(1.0, abs=1e-8)
    mm = sd.mb_moments(T)
    assert np.trapezoid(v * pdf, v) == pytest.approx(mm["mean speed"], rel=1e-7)
    assert math.sqrt(np.trapezoid(v ** 2 * pdf, v)) == pytest.approx(mm["rms speed"], rel=1e-7)
    assert v[np.argmax(pdf)] == pytest.approx(mm["most probable speed"], abs=1e-3)
    assert mm["most probable speed"] < mm["mean speed"] < mm["rms speed"]


def test_sampled_velocities_pass_the_maxwell_tests_and_a_wrong_temperature_fails():
    ck, seed = sd.mb_sample_check(1.5, 100000, seed=3)
    assert ck.passed and seed == 3
    # the same test with samples drawn at the wrong temperature must fail: use the module's own logic by hand
    from scipy.stats import kstest, maxwell
    rng = np.random.default_rng(0)
    wrong = np.linalg.norm(rng.normal(0, math.sqrt(2.0), (100000, 3)), axis=1)
    assert kstest(wrong, maxwell(scale=math.sqrt(1.5)).cdf).pvalue < 1e-6
