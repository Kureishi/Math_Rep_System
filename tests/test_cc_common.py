"""Tests for modules/cc_common.py: the secular-versus-bounded drift classifier."""
import numpy as np
import pytest

from modules import cc_common as cc

T = np.linspace(0, 100, 2000)


def test_exact_constant_and_rounding_noise_are_constant():
    assert cc.drift_report("Q", T, np.full_like(T, 3.7)).kind == "constant"
    noise = 3.7 + 1e-16 * np.random.default_rng(0).normal(size=len(T))
    assert cc.drift_report("Q", T, noise).kind == "constant"


@pytest.mark.parametrize("periods", [3, 10, 40.3])
def test_a_bounded_oscillation_is_not_called_secular(periods):
    v = 1.0 + 1e-4 * np.sin(2 * np.pi * periods * T / T[-1] + 0.4)
    r = cc.drift_report("H", T, v, reference=1.0)
    assert r.kind == "bounded oscillation" and not r.secular and r.max_abs == pytest.approx(1e-4, rel=0.05)


@pytest.mark.parametrize("slope", [1e-6, -1e-5, 3e-4])
def test_a_linear_drift_is_secular_even_with_oscillation_on_top(slope):
    v = 1.0 + slope * T + 2e-5 * np.sin(2 * np.pi * 20 * T / T[-1])
    r = cc.drift_report("H", T, v)
    if abs(slope * T[-1]) > 3 * 2e-5:
        assert r.kind == "secular drift" and np.sign(r.secular_change) == np.sign(slope)
    else:
        assert r.kind in ("bounded oscillation", "secular drift")


def test_exponential_decay_is_secular_and_monotone():
    r = cc.drift_report("norm", T, np.exp(-T / 80))
    assert r.secular and r.secular_change < 0


def test_reference_value_is_respected():
    v = 5.0 + 1e-3 * np.sin(T)
    assert cc.drift_report("Q", T, v, reference=5.0).max_abs == pytest.approx(1e-3, rel=0.02)
    assert cc.drift_report("Q", T, v, reference=5.5).max_abs > 0.4


def test_reports_to_checks_flags_size_and_secularity():
    good = cc.drift_report("Q", T, 1 + 1e-9 * np.sin(T))
    big = cc.drift_report("H", T, 1 + 1e-3 * np.sin(T))
    leak = cc.drift_report("N", T, 1 - 1e-8 * T)
    ck = cc.reports_to_checks([good, big, leak], 1e-6)
    assert [c.passed for c in ck.checks] == [True, False, False]
    assert cc.reports_to_checks([leak], 1e-3, allow_secular=True).passed


def test_input_validation():
    with pytest.raises(cc.CrossCuttingError):
        cc.drift_report("Q", T[:5], np.ones(5))
    with pytest.raises(cc.CrossCuttingError):
        cc.drift_report("Q", T, np.ones(10))


def test_parse_helpers_reject_unsafe_input():
    import sympy as sp
    x = sp.Symbol("x", real=True)
    assert cc.parse_with("x^2 + 1", {"x": x}) == x ** 2 + 1
    with pytest.raises(cc.CrossCuttingError):
        cc.parse_with("__import__('os')", {"x": x})
    assert cc.names_in("a*x + sin(b)") == {"a", "x", "b"}
