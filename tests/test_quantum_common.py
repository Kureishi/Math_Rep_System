"""The shared parser and check types for the quantum modules."""
import numpy as np
import pytest
import sympy as sp

from modules.quantum_common import (
    CheckList, ExpressionError, check_expression_syntax, safe_expression, trapezoid_norm, vectorized,
)


@pytest.mark.parametrize("text", ["x", "2*x", "0.5*m*w**2*x**2", "x^2 + 1", "-x", "sin(x)*exp(-x**2)", "sech(x)**2",
                                  "Heaviside(1 - abs(x))", "sqrt(x**2 + 1)", "pi*x", "e**x", "1/x", "x % 3"])
def test_ordinary_expressions_are_accepted(text):
    safe_expression(text, ["x", "m", "w"])


@pytest.mark.parametrize("text", [
    "__import__('os').system('echo hi')", "x.__class__", "x.real", "lambda: 1", "x[0]", "[x, x]", "(x, 1)", "{1: x}",
    "'a'", "True", "None", "x if x else 1", "x > 1", "x and 1", "not x", "f(x)", "open('f')", "eval('1')", "x(1)",
    "sin(x, key=1)", "sin.__call__(x)", "print", "x; y", "2x", "x y", "",  "   ", "x" * 500, "1j", "b'x'",
])
def test_anything_beyond_arithmetic_and_whitelisted_functions_is_refused(text):
    with pytest.raises(ExpressionError):
        safe_expression(text, ["x"])


def test_an_unknown_name_is_named_in_the_error():
    with pytest.raises(ExpressionError, match="Unknown name 'q'"):
        safe_expression("q*x", ["x"])


def test_a_refused_expression_never_reaches_sympy(monkeypatch):
    import modules.quantum_common as qc
    monkeypatch.setattr(qc, "parse_expr", lambda *a, **k: (_ for _ in ()).throw(AssertionError("parsed an unsafe expression")))
    with pytest.raises(ExpressionError):
        safe_expression("__import__('os')", ["x"])


def test_check_expression_syntax_returns_the_tree_for_good_input():
    assert check_expression_syntax("x + 1", ["x"]) is not None


def test_vectorized_evaluates_with_parameters_substituted():
    f = vectorized(safe_expression("a*x**2 + b", ["x", "a", "b"]), "x", {"a": 2.0, "b": 1.0})
    np.testing.assert_allclose(f(np.array([0.0, 1.0, 2.0])), [1.0, 3.0, 9.0])


def test_a_missing_parameter_is_an_error_not_a_zero():
    with pytest.raises(ExpressionError, match="No value for: b"):
        vectorized(safe_expression("a*x + b", ["x", "a", "b"]), "x", {"a": 1.0})


def test_a_constant_expression_is_broadcast_to_the_input_shape():
    f = vectorized(safe_expression("3", ["x"]), "x", {})
    assert f(np.zeros((4,))).shape == (4,) and f(np.zeros((2, 3))).shape == (2, 3)


def test_special_functions_evaluate():
    f = vectorized(safe_expression("Heaviside(x) + sign(x) + erf(x)", ["x"]), "x", {})
    out = f(np.array([-1.0, 1.0]))
    assert out[1] > out[0]


def test_checklist_verdict_and_summary():
    c = CheckList()
    assert c.passed and c.summary() == "no checks ran"
    assert c.add("a", True, "fine") is True
    assert c.add("b", False, "broken") is False
    assert not c.passed and c.n_passed == 1 and c.summary() == "1/2 checks passed"


def test_trapezoid_norm_of_a_normalised_gaussian_is_one():
    x = np.linspace(-8, 8, 2001)
    psi = np.exp(-x ** 2 / 2) / np.pi ** 0.25
    assert trapezoid_norm(psi, x[1] - x[0]) == pytest.approx(1.0, abs=1e-9)
