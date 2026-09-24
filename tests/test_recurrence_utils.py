"""
Targeted tests for modules/recurrence_utils.py branches tests/test_recurrence.py
(which exercises solve_recurrence/verify_recurrence_solution end to end via
the normal extraction pipeline) doesn't reach: guard clauses inside the loop
over model.equations, the _independent_variable fallback path, and
verify_recurrence_solution's exception-handling fallbacks. Constructs
Equation/ProblemModel/InitialCondition objects directly rather than going
through the LLM-extraction JSON, so each edge case is isolated and doesn't
depend on how the (unrelated) extraction/parsing layer behaves.
"""
import sympy as sp

from modules.equation_engine import Equation, InitialCondition, ProblemModel
from modules.recurrence_utils import (
    _independent_variable, solve_recurrence, verify_recurrence_solution,
)

n = sp.Symbol("n")
a = sp.Function("a")
b = sp.Function("b")


def _model(equations, initial_conditions=()):
    return ProblemModel(problem_domain="test", variables=[], equations=list(equations),
                         solve_for=[], assumptions=[], initial_conditions=list(initial_conditions))


# ---------------------------------------------------------------- solve_recurrence guard clauses

def test_non_recurrence_equations_are_skipped():
    """A model with a mix of an algebraic equation and a recurrence: the
    algebraic one is skipped by the `e.kind != "recurrence"` guard rather
    than raising trying to rsolve() something that isn't one."""
    algebraic = Equation(name="e0", raw_expression="x = 5", derivation="",
                          kind="equation", sympy_eq=sp.Eq(sp.Symbol("x"), 5))
    recurrence = Equation(name="e1", raw_expression="a(n+1) = a(n) + 1", derivation="",
                           kind="recurrence", sympy_eq=sp.Eq(a(n + 1), a(n) + 1))
    result = solve_recurrence(_model([algebraic, recurrence]))
    assert "a" in result and "x" not in result


def test_equation_with_no_parse_result_is_skipped():
    """A recurrence-kind equation whose sympy_eq is None (failed to parse)
    is skipped rather than crashing on `.atoms()` of None."""
    unparsed = Equation(name="e0", raw_expression="garbage", derivation="",
                         kind="recurrence", sympy_eq=None)
    result = solve_recurrence(_model([unparsed]))
    assert result == {}


def test_recurrence_with_no_applied_function_is_skipped():
    """kind='recurrence' but the parsed expression has no function
    application at all (e.g. a constant equation mistakenly tagged) --
    `if not funcs: continue`."""
    no_func = Equation(name="e0", raw_expression="1 = 1", derivation="",
                        kind="recurrence", sympy_eq=sp.Eq(sp.Integer(1), sp.Integer(1)))
    result = solve_recurrence(_model([no_func]))
    assert result == {}


def test_unresolvable_independent_variable_is_skipped():
    """Every applied-function argument is itself a compound expression with
    no free symbols (a constant argument like a(1)) -- _independent_variable
    can't recover a variable to solve for, so the equation is skipped."""
    constant_arg = Equation(name="e0", raw_expression="a(1) = 5", derivation="",
                             kind="recurrence", sympy_eq=sp.Eq(a(1), 5))
    result = solve_recurrence(_model([constant_arg]))
    assert result == {}


def test_initial_condition_for_a_different_function_is_ignored():
    """Two recurrences (a and b) in one model, each with its own initial
    condition -- b's IC must not leak into a's rsolve() call (the
    `str(next(iter(lhs_funcs)).func) == func_name` guard)."""
    eq_a = Equation(name="ea", raw_expression="a(n+1) = a(n) + 1", derivation="",
                     kind="recurrence", sympy_eq=sp.Eq(a(n + 1), a(n) + 1))
    ic_a = InitialCondition(raw_expression="a(0) = 10", value=10.0, sympy_eq=sp.Eq(a(0), 10))
    ic_b = InitialCondition(raw_expression="b(0) = 999", value=999.0, sympy_eq=sp.Eq(b(0), 999))
    result = solve_recurrence(_model([eq_a], initial_conditions=[ic_a, ic_b]))
    assert result["a"].subs(n, 0) == 10


def test_initial_condition_with_no_parsed_form_is_ignored():
    """An IC whose own sympy_eq is None (failed to parse) is skipped rather
    than crashing on `.lhs`."""
    eq_a = Equation(name="ea", raw_expression="a(n+1) = a(n) + 1", derivation="",
                     kind="recurrence", sympy_eq=sp.Eq(a(n + 1), a(n) + 1))
    bad_ic = InitialCondition(raw_expression="garbage", value=0.0, sympy_eq=None)
    result = solve_recurrence(_model([eq_a], initial_conditions=[bad_ic]))
    assert "a" in result  # still solves, just without that IC applied


def test_rsolve_exception_is_caught_and_equation_skipped():
    """An equation rsolve() genuinely can't handle (a nonlinear recurrence)
    raises inside sympy -- caught by the bare `except Exception` rather than
    propagating and killing every other equation in the model."""
    nonlinear = Equation(name="e0", raw_expression="a(n+1) = a(n)**a(n)", derivation="",
                          kind="recurrence", sympy_eq=sp.Eq(a(n + 1), a(n) ** a(n)))
    result = solve_recurrence(_model([nonlinear]))
    assert "a" not in result


# ---------------------------------------------------------------- _independent_variable

def test_independent_variable_prefers_bare_symbol_argument():
    result = _independent_variable({a(n + 1), a(n)})
    assert result == n


def test_independent_variable_falls_back_to_free_symbols_of_compound_arg():
    """No applied function has a bare-symbol argument (both are shifted:
    a(n+1), a(n+2)) -- falls back to the free symbols of whichever
    argument is examined."""
    result = _independent_variable({a(n + 1), a(n + 2)})
    assert result == n


def test_independent_variable_returns_none_for_constant_arguments():
    result = _independent_variable({a(1), a(2)})
    assert result is None


# ---------------------------------------------------------------- verify_recurrence_solution

def test_verify_ignores_applied_functions_with_a_different_name():
    """The equation mentions both a(n) and an unrelated b(n) (e.g. copied
    from a coupled system) -- only a's applications get substituted; b's
    is left alone by the `if str(f.func) != func_name: continue` guard."""
    eq = sp.Eq(a(n + 1) + b(n), a(n) + 1 + b(n))  # b(n) cancels regardless
    closed_form = n + 5  # a(n) = n + 5 solves a(n+1) = a(n) + 1
    ok, residual = verify_recurrence_solution(eq, "a", closed_form, n)
    assert ok
    assert residual == 0


def test_verify_falls_back_to_numeric_sampling_on_float_noise():
    """A Fibonacci-style closed form with sqrt(5): exact sp.simplify may not
    land on a literal 0 due to how the expression is structured, but every
    sampled point should be negligibly close to zero -- exercises the
    numeric-sampling fallback path succeeding."""
    phi = (1 + sp.sqrt(5)) / 2
    psi = (1 - sp.sqrt(5)) / 2
    closed_form = (phi**n - psi**n) / sp.sqrt(5)  # Binet's formula
    eq = sp.Eq(a(n + 2), a(n + 1) + a(n))
    ok, residual = verify_recurrence_solution(eq, "a", closed_form, n)
    assert ok


def test_verify_returns_false_when_residual_is_not_numeric():
    """The residual still contains an unresolved symbol after substitution
    (a genuinely wrong closed form referencing an unrelated free variable
    that doesn't cancel out) -- complex() raises on the leftover symbol,
    caught and reported as a failed check rather than propagating."""
    stray = sp.Symbol("stray")
    eq = sp.Eq(a(n + 1), a(n) + 1)
    wrong_closed_form = n * stray  # residual = stray - 1, `stray` never cancels
    ok, residual = verify_recurrence_solution(eq, "a", wrong_closed_form, n)
    assert ok is False
    assert stray in residual.free_symbols


def test_verify_reports_false_for_a_genuinely_wrong_closed_form():
    eq = sp.Eq(a(n + 1), a(n) + 1)
    wrong_closed_form = 2 * n  # a(n+1) - a(n) = 2, not 1
    ok, residual = verify_recurrence_solution(eq, "a", wrong_closed_form, n)
    assert ok is False
