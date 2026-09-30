"""Tests for modules/series_asymptotics.py -- Taylor/Laurent series and
asymptotic expansions, standalone from the LLM-extraction pipeline."""
import sympy as sp

from modules.series_asymptotics import taylor_series, asymptotic_expansion


# ---------------------------------------------------------------- taylor
def test_taylor_series_of_sine_at_origin():
    result = taylor_series("sin(x)", order=6)
    x = sp.Symbol("x")
    assert result.expansion_available
    assert result.truncated.equals(x - x ** 3 / 6 + x ** 5 / 120)
    assert result.verified
    assert result.order_term is not None


def test_taylor_series_around_nonzero_point():
    result = taylor_series("exp(x)", point=1, order=3)
    x = sp.Symbol("x")
    assert result.expansion_available
    # exp(x) around x=1: e + e*(x-1) + e*(x-1)**2/2 + e*(x-1)**3/6
    expected = (sp.exp(1) + sp.exp(1) * (x - 1) + sp.exp(1) * (x - 1) ** 2 / 2
                + sp.exp(1) * (x - 1) ** 3 / 6)
    assert result.truncated.equals(expected)
    assert result.verified


def test_taylor_terms_split_for_display():
    result = taylor_series("sin(x)", order=6)
    assert len(result.terms) == 3
    assert "x" in result.terms


# ---------------------------------------------------------------- laurent / exact edge case
def test_exact_laurent_series_at_simple_pole_is_not_a_failure():
    """1/(x-1) at x=1 is ALREADY the complete, exact Laurent series (a
    single term, zero remainder) -- this must be reported as a success,
    not confused with a genuine essential-singularity failure, even
    though both cases return the input expression unchanged with no O()
    term."""
    result = taylor_series("1/(x-1)", point=1, order=4)
    assert result.expansion_available
    assert result.verified
    assert "exact" in result.verification_detail.lower()


def test_essential_singularity_reported_as_unavailable():
    """exp(1/x) at x=0 has a genuine essential singularity: no
    Taylor/Laurent expansion of any order exists, and this must be
    reported honestly rather than displaying the unchanged input as if
    it were a valid expansion."""
    result = taylor_series("exp(1/x)", point=0, order=4)
    assert not result.expansion_available
    assert not result.verified
    assert "no series expansion is available" in result.verification_detail.lower()


# ---------------------------------------------------------------- asymptotic
def test_asymptotic_expansion_of_rational_function():
    result = asymptotic_expansion("(x**2+3*x+1)/(x**2-1)", order=4)
    x = sp.Symbol("x")
    assert result.expansion_available
    assert result.truncated.equals(1 + 3 / x + 2 / x ** 2 + 3 / x ** 3)
    assert result.verified


def test_asymptotic_expansion_essential_singularity_at_infinity():
    """exp(x)/x has an essential singularity as x -> oo (not capturable
    by any finite asymptotic power series) and must be reported as
    unavailable, not silently shown as its own unchanged expansion."""
    result = asymptotic_expansion("exp(x)/x", order=4)
    assert not result.expansion_available


# ---------------------------------------------------------------- errors
def test_taylor_series_parse_error():
    result = taylor_series("not @@ valid")
    assert result.error is not None
    assert result.input_expr is None


# ---------------------------------------------------------------- _verify_expansion branches
import pytest

import modules.series_asymptotics as sa
from modules.series_asymptotics import _verify_expansion
from modules.timeout_utils import ComputationTimeoutError

_x = sp.Symbol("x")
_k = sp.Symbol("k")


def test_verify_skips_when_expansion_point_is_not_numeric():
    ok, detail = _verify_expansion(sp.sin(_x), _x, _x, sp.Symbol("a"), asymptotic=False)
    assert ok is False
    assert "not numeric" in detail


def test_verify_nonzero_numeric_point_uses_points_near_it():
    # exp(x) around x=1, truncated to its exact value at 1 plus the first-order term
    p = sp.Integer(1)
    trunc = sp.E + sp.E * (_x - 1)
    ok, detail = _verify_expansion(sp.exp(_x), trunc, _x, p, asymptotic=False)
    assert ok is True and "approaching the expansion point" in detail


def test_verify_reports_unevaluable_sample_points():
    # an extra free symbol means the expression never reduces to a number,
    # so complex() raises TypeError -- reported, not propagated
    ok, detail = _verify_expansion(_x + _k, _x, _x, sp.Integer(0), asymptotic=False)
    assert ok is False
    assert "could not numerically evaluate" in detail


def test_verify_reports_nan_at_sample_points():
    ok, detail = _verify_expansion(sp.nan, sp.Integer(0), _x, sp.Integer(0), asymptotic=False)
    assert ok is False
    assert "NaN" in detail


def test_verify_rejects_error_that_grows_toward_the_expansion_point():
    # original = 1 - x, truncated = 0: errors at 0.3/0.1/0.03 are 0.7/0.9/0.97,
    # i.e. getting WORSE as we approach 0 -- a bad expansion, must not verify
    ok, detail = _verify_expansion(1 - _x, sp.Integer(0), _x, sp.Integer(0), asymptotic=False)
    assert ok is False
    assert "did not consistently shrink" in detail


def test_verify_rejects_error_that_is_flat_but_large():
    # constant error of 5 "shrinks" (non-increasing) but fails the absolute bound
    ok, detail = _verify_expansion(sp.Integer(5), sp.Integer(0), _x, sp.Integer(0), asymptotic=False)
    assert ok is False
    assert "did not consistently shrink" in detail


def test_verify_asymptotic_success_says_growing():
    ok, detail = _verify_expansion(1 + 1 / (_x + 1), 1 + 1 / _x, _x, sp.oo, asymptotic=True)
    assert ok is True and "growing" in detail


# ---------------------------------------------------------------- taylor_series failure handling

def test_taylor_timeout_is_reported_as_error(monkeypatch):
    def boom(*a, **k):
        raise ComputationTimeoutError(0.1, "series")
    monkeypatch.setattr(sa, "run_with_timeout", boom)
    r = sa.taylor_series("sin(x)")
    assert r.error and "timed out" in r.error.lower()
    assert r.input_expr is not None and not r.expansion_available


def test_taylor_sympy_failure_is_reported_as_error(monkeypatch):
    def boom(*a, **k):
        raise NotImplementedError("no series")
    monkeypatch.setattr(sa, "run_with_timeout", boom)
    r = sa.taylor_series("sin(x)")
    assert r.error and "could not expand" in r.error


def test_taylor_unchanged_detection_survives_simplify_failing(monkeypatch):
    # series() hands back exp(1/x) untouched (essential singularity at 0). If
    # simplify itself then blows up, detection falls back to structural
    # equality and must still report "unavailable" rather than crash.
    expr = sp.exp(1 / _x)
    monkeypatch.setattr(sa, "run_with_timeout", lambda *a, **k: expr)

    def boom(*a, **k):
        raise RuntimeError("simplify failed")
    monkeypatch.setattr(sp, "simplify", boom)
    r = sa.taylor_series("exp(1/x)")
    assert r.expansion_available is False
    assert "unchanged" in r.verification_detail


# ---------------------------------------------------------------- asymptotic_expansion failure handling

def test_asymptotic_parse_error():
    r = asymptotic_expansion("not @@ valid")
    assert r.error is not None and r.input_expr is None
    assert r.kind == "asymptotic"


def test_asymptotic_timeout_is_reported_as_error(monkeypatch):
    def boom(*a, **k):
        raise ComputationTimeoutError(0.1, "series_asymptotic")
    monkeypatch.setattr(sa, "run_with_timeout", boom)
    r = asymptotic_expansion("1/(x+1)")
    assert r.error and "timed out" in r.error.lower()


def test_asymptotic_sympy_failure_is_reported_as_error(monkeypatch):
    def boom(*a, **k):
        raise NotImplementedError("no series")
    monkeypatch.setattr(sa, "run_with_timeout", boom)
    r = asymptotic_expansion("1/(x+1)")
    assert r.error and "could not expand this asymptotic" in r.error


def test_asymptotic_unchanged_detection_survives_simplify_failing(monkeypatch):
    expr = sp.exp(_x) / _x
    monkeypatch.setattr(sa, "run_with_timeout", lambda *a, **k: expr)

    def boom(*a, **k):
        raise RuntimeError("simplify failed")
    monkeypatch.setattr(sp, "simplify", boom)
    r = asymptotic_expansion("exp(x)/x")
    assert r.expansion_available is False
    assert "unchanged" in r.verification_detail


@pytest.mark.parametrize("expr", ["1/x", "x", "3"])
def test_asymptotic_exact_rational_behavior_is_not_a_failure(expr):
    # a function that IS already its own complete asymptotic expansion has no
    # O() remainder -- that's exact, available, and verified, not "unchanged = failed"
    r = asymptotic_expansion(expr)
    assert r.expansion_available is True
    assert r.verified is True
    assert r.order_term is None
    assert "exact asymptotic" in r.verification_detail
