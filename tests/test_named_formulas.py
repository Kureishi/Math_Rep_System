import sympy as sp
import pytest

from modules.equation_engine import Equation
from modules.named_formulas import recognize_formula


def _eq(sympy_eq, kind="equation"):
    return Equation(name="e", raw_expression=str(sympy_eq), derivation="", kind=kind, sympy_eq=sympy_eq)


# ---------------------------------------------------------------- direct-form recognition

def test_recognizes_newtons_second_law_direct_form():
    F, m, a = sp.symbols("F m a")
    result = recognize_formula(_eq(sp.Eq(F, m * a)), "force mass acceleration")
    names = [n for n, _ in result]
    assert "Newton's second law" in names


def test_recognizes_pythagorean_theorem():
    c, x, y = sp.symbols("c x y")
    result = recognize_formula(_eq(sp.Eq(c**2, x**2 + y**2)), "right triangle hypotenuse")
    names = [n for n, _ in result]
    assert "Pythagorean theorem" in names


def test_recognizes_ideal_gas_law():
    P, V, n, R, T = sp.symbols("P V n R T")
    result = recognize_formula(_eq(sp.Eq(P * V, n * R * T)), "ideal gas")
    names = [n for n, _ in result]
    assert "Ideal gas law" in names


def test_recognizes_compound_interest():
    A, P, r, n, t = sp.symbols("A P r n t")
    eq = sp.Eq(A, P * (1 + r / n) ** (n * t))
    result = recognize_formula(_eq(eq), "compound interest investment")
    names = [n for n, _ in result]
    assert "Compound interest" in names


def test_recognizes_ohms_law():
    V, I, R = sp.symbols("V I R")
    result = recognize_formula(_eq(sp.Eq(V, I * R)), "voltage current resistance circuit")
    names = [n for n, _ in result]
    assert "Ohm's law" in names


# ---------------------------------------------------------------- rearrangement recognition

def test_recognizes_newtons_law_solved_for_acceleration():
    F, m, a = sp.symbols("F m a")
    # a = F/m instead of F = m*a
    result = recognize_formula(_eq(sp.Eq(a, F / m)), "force mass acceleration")
    names = [n for n, _ in result]
    assert "Newton's second law" in names


def test_recognizes_ohms_law_solved_for_current():
    V, I, R = sp.symbols("V I R")
    result = recognize_formula(_eq(sp.Eq(I, V / R)), "voltage current resistance")
    names = [n for n, _ in result]
    assert "Ohm's law" in names


def test_recognizes_density_solved_for_mass():
    rho, m, V = sp.symbols("rho m V")
    result = recognize_formula(_eq(sp.Eq(m, rho * V)), "density mass volume")
    names = [n for n, _ in result]
    assert "Density" in names


# ---------------------------------------------------------------- variable renaming (structural, not name-based)

def test_recognition_is_symbol_name_independent():
    """Completely different symbol letters, same underlying shape --
    must still be recognized, since matching is structural."""
    x1, x2, x3 = sp.symbols("q1 q2 q3")
    result = recognize_formula(_eq(sp.Eq(x1, x2 * x3)), "force mass acceleration")
    names = [n for n, _ in result]
    assert "Newton's second law" in names


# ---------------------------------------------------------------- ambiguity handling

def test_ambiguous_shape_without_context_returns_multiple_candidates():
    """F=m*a, p=m*v, W=F*d all collide on shape -- with no disambiguating
    context, all tied candidates should come back, not a silent guess."""
    x1, x2, x3 = sp.symbols("x1 x2 x3")
    result = recognize_formula(_eq(sp.Eq(x1, x2 * x3)), "")
    assert len(result) > 1


def test_ambiguous_shape_resolved_by_specific_context():
    x1, x2, x3 = sp.symbols("x1 x2 x3")
    result_a = recognize_formula(_eq(sp.Eq(x1, x2 * x3)), "force mass acceleration newton")
    assert result_a == [("Newton's second law", "force = mass × acceleration")]

    result_b = recognize_formula(_eq(sp.Eq(x1, x2 * x3)), "momentum mass velocity")
    assert result_b == [("Momentum", "p = mv")]


def test_ambiguous_shape_with_multiple_matching_contexts_returns_all():
    """If context text happens to mention keywords for MORE than one
    candidate, that's still genuinely ambiguous -- return everything
    rather than picking arbitrarily."""
    x1, x2, x3 = sp.symbols("x1 x2 x3")
    result = recognize_formula(_eq(sp.Eq(x1, x2 * x3)), "force momentum")
    assert len(result) > 1


# ---------------------------------------------------------------- non-matches / edge cases

def test_no_match_for_an_unrelated_equation():
    x, y, z = sp.symbols("x y z")
    result = recognize_formula(_eq(sp.Eq(x, sp.sin(y) + sp.cos(z))))
    assert result == []


def test_inequality_kind_never_matched():
    x, y = sp.symbols("x y")
    result = recognize_formula(_eq(x > y, kind="inequality"))
    assert result == []


def test_equation_with_no_sympy_eq_returns_empty():
    eq = Equation(name="e", raw_expression="broken", derivation="", kind="equation",
                   sympy_eq=None, parse_error="couldn't parse")
    assert recognize_formula(eq) == []


def test_context_is_case_insensitive():
    F, m, a = sp.symbols("F m a")
    result = recognize_formula(_eq(sp.Eq(F, m * a)), "FORCE MASS ACCELERATION")
    names = [n for n, _ in result]
    assert "Newton's second law" in names
