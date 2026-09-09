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
