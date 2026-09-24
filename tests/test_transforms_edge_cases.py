"""
Targeted tests for modules/transforms.py branches tests/test_transforms.py
(happy-path Laplace/Fourier round trips) doesn't reach: the "SymPy left
this unevaluated" branch for Fourier and inverse Fourier, the
timeout/exception branches in all four public functions, and
_round_trip_verify's own exception/unevaluated/numeric-fallback branches
(called directly, with sympy's transform functions monkeypatched where a
real input that naturally produces the case isn't practical to find).
"""
import sympy as sp
import pytest

from modules.timeout_utils import ComputationTimeoutError
from modules.transforms import (
    _round_trip_verify, fourier_transform_expr, inverse_fourier_transform_expr,
    inverse_laplace_transform_expr, laplace_transform_expr,
)

t = sp.Symbol("t", positive=True)
s = sp.Symbol("s")
x = sp.Symbol("x", real=True)
k = sp.Symbol("k", real=True)


# ---------------------------------------------------------------- unevaluated (Fourier side)

def test_fourier_transform_unevaluated_reported_honestly():
    """1/x has no closed-form Fourier transform SymPy can find -- mirrors
    the already-covered Laplace case, for the Fourier function itself."""
    result = fourier_transform_expr("1/x")
    assert result.error is None
    assert not result.evaluated
    assert result.verification_method == "not attempted"
    assert "unevaluated" in result.verification_detail


def test_inverse_fourier_transform_unevaluated_reported_honestly():
    result = inverse_fourier_transform_expr("1/k")
    assert result.error is None
    assert not result.evaluated
    assert result.verification_method == "not attempted"


def test_inverse_laplace_transform_unevaluated_reported_honestly():
    result = inverse_laplace_transform_expr("s**s")
    assert result.error is None
    assert not result.evaluated
    assert result.verification_method == "not attempted"


def test_inverse_laplace_transform_parse_error():
    result = inverse_laplace_transform_expr("not @@ valid")
    assert result.error is not None
    assert "Could not parse" in result.error


def test_inverse_fourier_transform_parse_error():
    result = inverse_fourier_transform_expr("not @@ valid")
    assert result.error is not None
    assert "Could not parse" in result.error


# ---------------------------------------------------------------- timeout / compute-exception, all four directions

@pytest.mark.parametrize("fn, arg, sympy_target", [
    (laplace_transform_expr, "exp(-t)", "modules.transforms.sp.laplace_transform"),
    (inverse_laplace_transform_expr, "1/s", "modules.transforms.sp.inverse_laplace_transform"),
    (fourier_transform_expr, "exp(-x**2)", "modules.transforms.sp.fourier_transform"),
    (inverse_fourier_transform_expr, "exp(-k**2)", "modules.transforms.sp.inverse_fourier_transform"),
])
def test_timeout_is_reported_as_error_not_raised(monkeypatch, fn, arg, sympy_target):
    def fake_timeout(*a, **kw):
        raise ComputationTimeoutError(30.0, label="test")
    monkeypatch.setattr("modules.transforms.run_with_timeout", fake_timeout)
    result = fn(arg)
    assert result.output_expr is None
    assert "timed out" in result.error


@pytest.mark.parametrize("fn, arg", [
    (laplace_transform_expr, "exp(-t)"),
    (inverse_laplace_transform_expr, "1/s"),
    (fourier_transform_expr, "exp(-x**2)"),
    (inverse_fourier_transform_expr, "exp(-k**2)"),
])
def test_generic_compute_exception_is_reported_as_error_not_raised(monkeypatch, fn, arg):
    def fake_timeout(func, *a, **kw):
        raise ValueError("sympy blew up")
    monkeypatch.setattr("modules.transforms.run_with_timeout", fake_timeout)
    result = fn(arg)
    assert result.output_expr is None
    assert "could not compute" in result.error


# ---------------------------------------------------------------- _round_trip_verify, called directly

def test_round_trip_not_attempted_when_inverse_raises(monkeypatch):
    def fake_ilt(*a, **kw):
        raise ValueError("cannot invert")
    monkeypatch.setattr("modules.transforms.sp.inverse_laplace_transform", fake_ilt)
    verified, method, detail = _round_trip_verify("laplace", sp.exp(-t), 1 / (s + 1), t, s)
    assert verified is False
    assert method == "not attempted"
    assert "could not be computed" in detail


def test_round_trip_not_attempted_when_result_stays_unevaluated():
    """A Piecewise whose forward Fourier transform doesn't evaluate to a
    closed form -- the inverse call itself stays an inert unevaluated
    object, which _round_trip_verify must detect rather than comparing
    against it as if it were a real expression."""
    expr = sp.Piecewise((1, x > 0), (0, True))
    verified, method, detail = _round_trip_verify("fourier", expr, expr, x, k)
    assert verified is False
    assert method == "not attempted"
    assert "did not evaluate" in detail


def test_round_trip_numeric_sampling_succeeds_on_negligible_difference(monkeypatch):
    """The symbolic diff doesn't reduce to a literal 0 (a tiny epsilon term
    sp.simplify can't cancel away) but every sampled point is within
    tolerance -- exercises the numeric-sampling *success* path."""
    original = sp.exp(-t)

    def fake_ilt(*a, **kw):
        return original + sp.Float(1e-10) * sp.sin(t)
    monkeypatch.setattr("modules.transforms.sp.inverse_laplace_transform", fake_ilt)
    verified, method, detail = _round_trip_verify("laplace", original, 1 / (s + 1), t, s)
    assert verified is True
    assert method == "numeric sampling"


def test_round_trip_numeric_sampling_reports_genuine_disagreement(monkeypatch):
    original = sp.exp(-t)

    def fake_ilt(*a, **kw):
        return original + 1  # a real, non-negligible offset
    monkeypatch.setattr("modules.transforms.sp.inverse_laplace_transform", fake_ilt)
    verified, method, detail = _round_trip_verify("laplace", original, 1 / (s + 1), t, s)
    assert verified is False
    assert method == "numeric sampling"
    assert "disagreed" in detail


def test_round_trip_comparison_failure_when_result_has_a_stray_symbol(monkeypatch):
    """The (faked) inverse result contains a free symbol unrelated to the
    independent variable, so it never cancels out of the diff and can't be
    evaluated to a complex number at any sample point -- covers the
    TypeError/ValueError catch around the numeric fallback itself."""
    original = sp.exp(-t)
    stray = sp.Symbol("stray")

    def fake_ilt(*a, **kw):
        return original + stray
    monkeypatch.setattr("modules.transforms.sp.inverse_laplace_transform", fake_ilt)
    verified, method, detail = _round_trip_verify("laplace", original, 1 / (s + 1), t, s)
    assert verified is False
    assert method == "not attempted"
    assert "comparison failed" in detail
