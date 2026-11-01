"""Tests for modules/cc_limits.py: classical limits as checks, and large-N rates."""
import pytest
import sympy as sp

from modules import cc_limits as cl


@pytest.mark.parametrize("key", list(cl.LIMITS))
def test_every_preset_limit_check_passes(key):
    rep = cl.check_preset(key)
    assert rep.checks.passed, (key, [(c.label, c.detail) for c in rep.checks.checks if not c.passed])
    if cl.LIMITS[key].mode == "series":
        assert rep.leading_matches and rep.module_used.startswith("series_asymptotics")
        assert rep.predicted_exponent is not None and rep.observed_exponent == pytest.approx(rep.predicted_exponent, abs=0.15)


def test_relativistic_kinetic_energy_expansion_and_rate():
    rep = cl.check_preset("rel_ke")
    assert sp.simplify(rep.leading - sp.sympify("m*v**2/2")) == 0 and rep.correction_matches
    assert rep.predicted_exponent == 2.0                                                      # relative error ~ (v/c)^2
    assert all(rep.table[i][3] > rep.table[i + 1][3] for i in range(len(rep.table) - 1))


def test_planck_law_reduces_to_rayleigh_jeans_with_the_first_quantum_correction():
    rep = cl.check_preset("planck_rj")
    assert rep.leading_matches and rep.correction_matches and rep.predicted_exponent == 1.0


def test_velocity_addition_groups_terms_of_equal_order():
    rep = cl.check_preset("rel_add")
    assert rep.leading_matches and rep.correction_matches                                      # u + v is ONE leading term


def test_tunnelling_has_no_power_series_but_matches_wkb():
    rep = cl.check_preset("tunnelling")
    assert rep.series_text == "" and rep.leading_matches is None and rep.checks.passed
    assert rep.table[-1][3] < 1e-10


def test_a_wrong_expectation_is_caught():
    bad = cl.check_limit("m*c^2*(1/sqrt(1 - v^2/c^2) - 1)", "c", "oo", "m*v^2", None, {"m": 1.0, "v": 1.0, "c": 4.0})
    assert not bad.checks.passed and bad.leading_matches is False
    wrong_corr = cl.check_limit("m*c^2*(1/sqrt(1 - v^2/c^2) - 1)", "c", "oo", "m*v^2/2", "m*v^4/(8*c^2)", {"m": 1.0, "v": 1.0, "c": 4.0})
    assert wrong_corr.leading_matches and wrong_corr.correction_matches is False and not wrong_corr.checks.passed


def test_custom_limit_toward_zero():
    rep = cl.check_limit("sin(x)/x", "x", "0", "1", "-x^2/6", {"x": 1.0})
    assert rep.checks.passed and rep.predicted_exponent == 2.0


def test_input_validation():
    with pytest.raises(cl.CrossCuttingError):
        cl.check_limit("x", "x", "5", "x")
    with pytest.raises(cl.CrossCuttingError, match="does not contain"):
        cl.check_limit("x^2", "y", "0", "0")
    with pytest.raises(cl.CrossCuttingError, match="mode"):
        cl.check_limit("x", "x", "0", "0", mode="nope")
    with pytest.raises(cl.CrossCuttingError):
        cl.check_preset("nope")
    with pytest.raises(cl.CrossCuttingError):
        cl.check_large_n("nope")


@pytest.mark.parametrize("key,rate", [("stirling", 1.0), ("de_moivre", 0.5), ("harmonic", 1.0)])
def test_large_n_rates(key, rate):
    rep = cl.check_large_n(key)
    assert rep.checks.passed and -rep.observed_exponent == pytest.approx(rate, abs=0.1)


def test_stirling_remainder_is_one_over_twelve_n():
    rep = cl.check_large_n("stirling")
    assert rep.errors[-1] * rep.ns[-1] == pytest.approx(1 / 12, rel=1e-3)


def test_ising_free_energy_converges_exponentially():
    rep = cl.check_large_n("ising_1d")
    assert rep.checks.passed and rep.errors[-1] < 1e-20 * rep.errors[0] or rep.errors[-1] < 1e-30
