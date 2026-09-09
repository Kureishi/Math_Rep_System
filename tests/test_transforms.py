"""Tests for modules/transforms.py -- Laplace/Fourier transforms and
their inverses, standalone from the LLM-extraction pipeline."""
import sympy as sp

from modules.transforms import (
    laplace_transform_expr, inverse_laplace_transform_expr,
    fourier_transform_expr, inverse_fourier_transform_expr,
)


# ---------------------------------------------------------------- laplace
def test_laplace_transform_of_damped_sine():
    result = laplace_transform_expr("exp(-2*t)*sin(3*t)")
    s = sp.Symbol("s")
    assert result.error is None
    assert result.evaluated
    assert result.output_expr.equals(3 / ((s + 2) ** 2 + 9))
    assert result.verified
    assert result.verification_method == "symbolic"


def test_laplace_transform_of_polynomial():
    result = laplace_transform_expr("t**2")
    s = sp.Symbol("s")
    assert result.evaluated
    assert result.output_expr.equals(2 / s ** 3)
    assert result.verified


def test_inverse_laplace_transform_round_trips():
    result = inverse_laplace_transform_expr("3/((s+2)**2+9)")
    t = sp.Symbol("t", positive=True)
    assert result.evaluated
    assert result.output_expr.equals(sp.exp(-2 * t) * sp.sin(3 * t))
    assert result.verified


def test_laplace_transform_uses_correct_symbol_assumptions():
    """Regression test: the symbol used inside the parsed expression must
    be the SAME object (same assumptions) as the one laplace_transform is
    called with, or SymPy silently computes a transform with respect to
    an unrelated 't' and returns a wrong-but-not-erroring answer."""
    result = laplace_transform_expr("t**2")
    s = sp.Symbol("s")
    # a mismatched-symbol bug previously produced t**2/s here instead
    assert not result.output_expr.equals(sp.Symbol("t") ** 2 / s)
    assert result.output_expr.equals(2 / s ** 3)


def test_laplace_transform_unevaluated_case_reported_honestly():
    result = laplace_transform_expr("tan(t)")
    assert result.error is None
    assert not result.evaluated
    assert "unevaluated" in result.verification_detail.lower() or \
           "no closed form" in result.verification_detail.lower()
    assert not result.verified


def test_laplace_transform_parse_error():
    result = laplace_transform_expr("t +* 2 $$")
    assert result.error is not None
    assert result.output_expr is None


# ---------------------------------------------------------------- fourier
def test_fourier_transform_of_gaussian_round_trips():
    result = fourier_transform_expr("exp(-x**2)")
    k = sp.Symbol("k", real=True)
    assert result.evaluated
    assert result.output_expr.equals(sp.sqrt(sp.pi) * sp.exp(-sp.pi ** 2 * k ** 2))
    assert result.verified
    assert result.verification_method == "symbolic"


def test_inverse_fourier_transform_round_trips():
    result = inverse_fourier_transform_expr("sqrt(pi)*exp(-pi**2*k**2)")
    x = sp.Symbol("x", real=True)
    assert result.evaluated
    assert result.output_expr.equals(sp.exp(-x ** 2))
    assert result.verified


def test_fourier_transform_parse_error():
    result = fourier_transform_expr("not @@ valid")
    assert result.error is not None
