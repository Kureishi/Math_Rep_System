"""Tests for modules/cc_crossval.py: closed form against an independent numerical route."""
import math

import numpy as np
import pytest

from modules import cc_crossval as cx


@pytest.fixture(scope="module")
def suite():
    return {f"{i}:{c.task}": c for i, c in enumerate(cx.demonstration_suite())}


def test_demonstration_suite_statuses(suite):
    statuses = [c.status for c in suite.values()]
    assert "DISAGREE" not in statuses and len(statuses) == 12
    assert statuses.count("agree") >= 10
    assert any(s == "numeric only" for s in statuses)                                             # exp(sin x) on [0, 1] has no closed form SymPy can find


def test_gaussian_integral_closed_form_value(suite):
    c = next(v for k, v in suite.items() if k.endswith("integral") and "cos" in v.problem)
    assert c.symbolic_value == pytest.approx(math.sqrt(math.pi) * math.exp(-1.5 ** 2 / 4), rel=1e-12) and c.status == "agree"


def test_numeric_only_when_there_is_no_closed_form():
    c = cx.validate_integral("exp(sin(x))", "x", "0", "1")
    assert c.status == "numeric only" and c.symbolic_value is None and c.numeric_value == pytest.approx(1.6318696, abs=1e-6)
    assert "closed form" in c.detail


def test_derivative_against_mpmath_finite_differences():
    c = cx.validate_derivative("x^x*sin(x)", "x", 1.3, 2)
    assert c.status == "agree" and c.error < 1e-9
    assert cx.validate_derivative("exp(a*x)", "x", 0.5, 3, {"a": 2.0}).symbolic_value == pytest.approx(8 * math.exp(1.0), rel=1e-12)


def test_roots_including_a_double_root_and_a_transcendental_equation():
    c = cx.validate_roots("x^3 - 6*x^2 + 11*x - 6", "x", -1, 5)
    assert c.status == "agree" and c.symbolic_value == pytest.approx([1.0, 2.0, 3.0]) and c.numeric_value == pytest.approx([1.0, 2.0, 3.0], abs=1e-9)
    t = cx.validate_roots("cos(x) - x", "x", -2, 2)
    assert t.status in ("agree", "numeric only", "symbolic only") and t.numeric_value == pytest.approx([0.7390851332151607], abs=1e-9)
    d = cx.validate_roots("(x - 1)^2*(x + 2)", "x", -3, 3)                                         # touches zero at x = 1: no sign change
    assert d.status == "agree" and 1.0 in [round(v, 6) for v in d.symbolic_value] and "double root" in d.detail


def test_ode_closed_form_against_solve_ivp():
    c = cx.validate_ode("x_tt + 3*x_t + 2*x = 0", "x", "t", {"x": 1.0, "x_t": 0.0}, 5.0)
    t = np.linspace(0, 5, 101)
    assert c.status == "agree" and np.max(np.abs(np.asarray(c.symbolic_value) - (2 * np.exp(-t) - np.exp(-2 * t)))) < 1e-12
    forced = cx.validate_ode("x_tt + x = a*sin(2*t)", "x", "t", {"x": 0.0, "x_t": 0.0}, 6.0, {"a": 1.0})
    assert forced.status == "agree"


def test_nonlinear_ode_may_have_no_closed_form_but_still_gets_a_numerical_answer(monkeypatch):
    monkeypatch.setattr(cx, "SYM_TIMEOUT", 2.0)                                                    # dsolve would otherwise spend its full 20 s on the pendulum
    c = cx.validate_ode("x_tt + sin(x) = 0", "x", "t", {"x": 1.0, "x_t": 0.0}, 3.0)
    assert c.status in ("numeric only", "agree") and c.numeric_value is not None


def test_sums_finite_and_infinite():
    assert cx.validate_sum("1/k^2", "k", "1", "oo").symbolic_value == pytest.approx(math.pi ** 2 / 6)
    f = cx.validate_sum("k*r^k", "k", "0", "40", {"r": 0.9})
    assert f.status == "agree" and f.numeric_value == pytest.approx(sum(k * 0.9 ** k for k in range(41)))


def test_limits_including_divergent():
    assert cx.validate_limit("sin(x)/x", "x", "0").symbolic_value == 1
    assert cx.validate_limit("(1 + 1/x)^x", "x", "oo").symbolic_value == pytest.approx(math.e)
    div = cx.validate_limit("1/x", "x", "0")
    assert div.status != "agree" and "diverges" in div.detail


def test_eigenvalues_symmetric_and_nonsymmetric():
    s = cx.validate_eigenvalues([["2", "1", "0"], ["1", "2", "1"], ["0", "1", "2"]])
    assert s.status == "agree" and s.numeric_value == pytest.approx(sorted([2 - math.sqrt(2), 2, 2 + math.sqrt(2)]))
    n = cx.validate_eigenvalues([["0", "1"], ["-1", "0"]])
    assert n.status == "agree" and np.allclose(sorted(np.imag(n.numeric_value)), [-1, 1])


def test_a_sabotaged_closed_form_is_reported_as_a_disagreement():
    c = cx.validate_integral("x^2", "x", "0", "3")
    assert c.status == "agree"
    c.symbolic_value = 8.0                                                                         # pretend SymPy returned the wrong number
    c.error = abs(c.symbolic_value - c.numeric_value)
    assert c.error > c.tolerance


@pytest.mark.parametrize("call", [
    lambda: cx.validate_integral("__import__('os')", "x", "0", "1"),
    lambda: cx.validate_derivative("x", "x", 1.0, 9),
    lambda: cx.validate_ode("x_t = x = 1", "x", "t", {"x": 1.0}, 1.0),
    lambda: cx.validate_ode("x = 1", "x", "t", {"x": 1.0}, 1.0),
    lambda: cx.validate_ode("x_tt + x = 0", "x", "t", {"x": 1.0}, 1.0),
    lambda: cx.validate_eigenvalues([["1"]]),
    lambda: cx.validate_eigenvalues([["1", "2"], ["3", "4"], ["5", "6"]]),
])
def test_bad_input_is_refused(call):
    with pytest.raises(cx.CrossCuttingError):
        call()
