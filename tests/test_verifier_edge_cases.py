"""
Targeted tests for modules/verifier.py branches tests/test_verifier.py
(happy-path checks via verify() + fixture JSON) doesn't reach: guard
clauses, exception-handling fallbacks, and rarer structural cases in each
of the private _xxx_checks functions. Calls those private functions
directly against hand-built ProblemModel/Equation objects (or, for
_optimization_checks, against a monkeypatched solve_optimization) rather
than going through the full extract -> verify pipeline, so each branch is
isolated from the (unrelated) extraction/LLM layer.
"""
import sympy as sp
import pytest

from modules.equation_engine import Equation, Objective, ProblemModel, Variable
from modules.optimization_utils import OptimizationResult
from modules.timeout_utils import ComputationTimeoutError
from modules.units_checker import parse_unit
from modules.verifier import (
    VerificationReport, _dimension_of_side, _dimensional_checks, _domain_checks,
    _inequality_checks, _matrix_system_checks, _numeric_balance_check, _ode_checks,
    _ode_dimensional_checks, _ode_dimensional_substitute, _optimization_checks,
    _recurrence_checks, _solve_sympy, _structural_checks, _extract_final_numbers,
)

t = sp.Symbol("t", positive=True)
x, y, z = sp.symbols("x y z")


def _model(variables=(), equations=(), solve_for=(), objective=None):
    return ProblemModel(problem_domain="test", variables=list(variables), equations=list(equations),
                         solve_for=list(solve_for), assumptions=[], objective=objective)


# ================================================================== _structural_checks

def test_structural_reports_parse_failures():
    bad = Equation(name="e0", raw_expression="garbage", derivation="", kind="equation",
                    sympy_eq=None, parse_error="unexpected token")
    report = VerificationReport()
    _structural_checks(_model(equations=[bad]), report)
    check = next(c for c in report.checks if c.label == "Equation parsing")
    assert not check.passed and "1 equation(s) failed to parse" in check.detail


def test_structural_reports_missing_target_variable():
    eq = Equation(name="e0", raw_expression="Eq(x, 1)", derivation="", kind="equation",
                   sympy_eq=sp.Eq(x, 1))
    report = VerificationReport()
    _structural_checks(_model(equations=[eq], solve_for=["y"]), report)
    check = next(c for c in report.checks if c.label == "Target variable present")
    assert not check.passed and "y does not appear" in check.detail


def test_structural_reports_underdetermined_system():
    """Two unknowns (x, y), one equation -- x + y = 5 -- can't be solved
    for both without more information."""
    eq = Equation(name="e0", raw_expression="Eq(x + y, 5)", derivation="", kind="equation",
                   sympy_eq=sp.Eq(x + y, 5))
    vx = Variable(symbol="x", meaning="x", known_value=None, unit=None)
    vy = Variable(symbol="y", meaning="y", known_value=None, unit=None)
    report = VerificationReport()
    _structural_checks(_model(variables=[vx, vy], equations=[eq], solve_for=["x", "y"]), report)
    check = next(c for c in report.checks if c.label == "Determinacy")
    assert not check.passed and "underdetermined" in check.detail


# ================================================================== _numeric_balance_check

def test_numeric_balance_falls_back_to_exact_equality_on_non_numeric_residual():
    """A residual that reduces to a nonzero pure-imaginary number after
    substitution -- float() refuses to convert it (raises TypeError),
    covers the `except TypeError: ok = residual == 0` fallback, distinct
    from the normal float-comparison path."""
    eq = Equation(name="e0", raw_expression="Eq(x, I)", derivation="", kind="equation",
                   sympy_eq=sp.Eq(x, sp.I))
    vx = Variable(symbol="x", meaning="x", known_value=2 * sp.I, unit=None)
    report = VerificationReport()
    _numeric_balance_check(_model(variables=[vx], equations=[eq]), report)
    check = next(c for c in report.checks if c.label == "Numeric balance: e0")
    assert not check.passed and "DOES NOT balance" in check.detail


# ================================================================== _dimension_of_side / _dimensional_checks

def test_dimension_of_side_no_piecewise_delegates_directly():
    from modules.units_checker import dimension_of
    dim, notes = _dimension_of_side(sp.Integer(5))
    assert notes == []
    assert dim == dimension_of(sp.Integer(5))


def test_dimension_of_side_reports_an_internally_invalid_piecewise_branch():
    """One Piecewise branch adds two incompatible dimensions together --
    dimension_of() raises ValueError for JUST that branch, which is caught
    and turned into a note rather than aborting the whole comparison; the
    other (valid) branch's dimension is still used."""
    from modules.units_checker import dimension_of, parse_unit
    from modules.verifier import make_dimension_placeholder
    mass_ph = make_dimension_placeholder(dimension_of(parse_unit("g")))
    time_ph = make_dimension_placeholder(dimension_of(parse_unit("s")))
    cond = sp.Symbol("cond") > 0
    pw = sp.Piecewise((mass_ph + time_ph, cond), (mass_ph, True))
    dim, notes = _dimension_of_side(pw)
    assert dim == dimension_of(parse_unit("g"))
    assert notes and "dimensionally invalid internally" in notes[0]


def test_dimension_of_side_propagates_when_every_branch_is_invalid():
    """Every branch fails individually -- branch_dims ends up empty, so the
    function falls back to computing dimension_of() on the WHOLE
    expression, which is itself invalid and raises -- propagated to the
    caller (_dimensional_checks' own try/except ValueError) rather than
    silently returning something misleading."""
    from modules.units_checker import dimension_of, parse_unit
    from modules.verifier import make_dimension_placeholder
    mass_ph = make_dimension_placeholder(dimension_of(parse_unit("g")))
    time_ph = make_dimension_placeholder(dimension_of(parse_unit("s")))
    cond = sp.Symbol("cond") > 0
    pw = sp.Piecewise((mass_ph + time_ph, cond), (mass_ph + time_ph, True))
    with pytest.raises(ValueError):
        _dimension_of_side(pw)


def test_unit_resolution_reports_unresolved_units_informationally():
    v_ok = Variable(symbol="d", meaning="d", known_value=None, unit="m")
    v_bad = Variable(symbol="q", meaning="q", known_value=None, unit="not_a_real_unit_xyz")
    report = VerificationReport()
    _dimensional_checks(_model(variables=[v_ok, v_bad]), report)
    check = next(c for c in report.checks if c.label == "Unit resolution")
    assert check.passed  # informational only, never fails the report
    assert "q" in check.detail


def test_dimensional_checks_skips_unparsed_equation():
    v = Variable(symbol="d", meaning="d", known_value=None, unit="m")
    unparsed = Equation(name="bad", raw_expression="garbage", derivation="", kind="equation", sympy_eq=None)
    report = VerificationReport()
    _dimensional_checks(_model(variables=[v], equations=[unparsed]), report)
    assert not any("bad" in c.label for c in report.checks)


# ================================================================== _ode_dimensional_substitute / _ode_dimensional_checks

def test_ode_dimensional_substitute_skips_derivative_of_untracked_function():
    """A Derivative node for a function not present in subs_funcs (e.g.
    because it has no declared unit) is left as-is rather than KeyError-ing
    on subs_funcs[d.expr]."""
    f = sp.Function("f")
    deriv = sp.Derivative(f(t), t)
    result = _ode_dimensional_substitute(deriv, {deriv}, subs_funcs={}, subs_symbols={})
    assert result == deriv  # nothing substituted, no crash


def test_ode_dimensional_checks_skips_equation_with_unfamiliar_function():
    """The ODE references a function with no declared unit at all --
    func_names_used isn't a subset of func_units, so the equation is
    skipped rather than crashing trying to look up its unit."""
    N = sp.Function("N")
    eq = Equation(name="decay", raw_expression="Derivative(N(t), t) = -k*N(t)", derivation="",
                   kind="ode", sympy_eq=sp.Eq(sp.Derivative(N(t), t), -sp.Symbol("k") * N(t)))
    v_t = Variable(symbol="t", meaning="t", known_value=None, unit="s")
    # N is never declared as is_function=True with a unit -- func_units will be empty
    report = VerificationReport()
    _ode_dimensional_checks(_model(variables=[v_t], equations=[eq]), report, units_map={"t": parse_unit("s")})
    assert not any("decay" in c.label for c in report.checks)


def test_ode_dimensional_checks_skips_equation_with_unfamiliar_plain_symbol():
    """The function itself has a unit, but the ODE also references a plain
    symbol (k) with no declared unit -- skipped for the same reason."""
    N = sp.Function("N")
    k = sp.Symbol("k")
    eq = Equation(name="decay", raw_expression="Derivative(N(t), t) = -k*N(t)", derivation="",
                   kind="ode", sympy_eq=sp.Eq(sp.Derivative(N(t), t), -k * N(t)))
    v_n = Variable(symbol="N", meaning="N", known_value=None, unit="g", is_function=True)
    v_t = Variable(symbol="t", meaning="t", known_value=None, unit="s")
    units_map = {"N": parse_unit("g"), "t": parse_unit("s")}  # note: no "k"
    report = VerificationReport()
    _ode_dimensional_checks(_model(variables=[v_n, v_t], equations=[eq]), report, units_map=units_map)
    assert not any("decay" in c.label for c in report.checks)


def test_ode_dimensional_checks_reports_dimensionally_invalid_equation():
    """A genuinely dimensionally-inconsistent ODE: dN/dt (dimension mass/time)
    added directly to N (dimension mass) -- sympy's own dimension_of() raises
    ValueError on that incompatible addition, no monkeypatch needed. Uses a
    plain (non-positive) local `tt` rather than the module-level `t`: the
    `positive=True` assumption on `t` was observed to change how
    `Derivative(N(t), t) + N(t)` auto-simplifies, masking the exception this
    test exists to exercise."""
    N = sp.Function("N")
    tt = sp.Symbol("tt")
    eq = Equation(name="decay", raw_expression="Derivative(N(tt), tt) + N(tt) = 0", derivation="",
                   kind="ode", sympy_eq=sp.Eq(sp.Derivative(N(tt), tt) + N(tt), 0))
    v_n = Variable(symbol="N", meaning="N", known_value=None, unit="g", is_function=True)
    v_t = Variable(symbol="tt", meaning="tt", known_value=None, unit="s")
    units_map = {"N": parse_unit("g"), "tt": parse_unit("s")}
    report = VerificationReport()
    _ode_dimensional_checks(_model(variables=[v_n, v_t], equations=[eq]), report, units_map=units_map)
    check = next(c for c in report.checks if "decay" in c.label)
    assert not check.passed and "Dimensionally invalid" in check.detail


def test_ode_dimensional_checks_skips_gracefully_on_unexpected_exception(monkeypatch):
    """An unrelated exception during the dims_equivalent() comparison
    (after dimension_of() itself succeeded) is reported as a skipped,
    still-passing check rather than propagating -- distinct from the
    ValueError branch above, which fires earlier in the same try block."""
    N = sp.Function("N")
    eq = Equation(name="decay", raw_expression="Derivative(N(t), t) = -N(t)", derivation="",
                   kind="ode", sympy_eq=sp.Eq(sp.Derivative(N(t), t), -N(t)))
    v_n = Variable(symbol="N", meaning="N", known_value=None, unit="g", is_function=True)
    v_t = Variable(symbol="t", meaning="t", known_value=None, unit="s")
    units_map = {"N": parse_unit("g"), "t": parse_unit("s")}

    def fake_dims_equivalent(*a, **kw):
        raise RuntimeError("something unrelated broke")
    monkeypatch.setattr("modules.verifier.dims_equivalent", fake_dims_equivalent)
    report = VerificationReport()
    _ode_dimensional_checks(_model(variables=[v_n, v_t], equations=[eq]), report, units_map=units_map)
    check = next(c for c in report.checks if "decay" in c.label)
    assert check.passed and "Skipped" in check.detail


# ================================================================== _inequality_checks

def test_inequality_check_skips_when_result_not_a_definite_boolean(monkeypatch):
    """A relational that doesn't reduce to a definite True/False after
    substitution -- bool() raising TypeError is caught and the check
    silently skipped, not reported. Forced via monkeypatch since a
    genuinely-constructible sympy Relational normally does resolve to a
    concrete bool once every symbol is a plain number."""
    ineq = Equation(name="c0", raw_expression="x < 5", derivation="", kind="inequality",
                     sympy_eq=sp.Lt(x, 5))
    vx = Variable(symbol="x", meaning="x", known_value=1.0, unit=None)

    class Unresolvable:
        free_symbols = set()

        def __bool__(self):
            raise TypeError("cannot resolve to a definite truth value")

    def fake_subs(self, *a, **kw):
        return Unresolvable()
    monkeypatch.setattr("sympy.core.relational.Relational.subs", fake_subs)
    report = VerificationReport()
    _inequality_checks(_model(variables=[vx], equations=[ineq]), report)
    assert not any("c0" in c.label for c in report.checks)


# ================================================================== _ode_checks

def test_ode_checks_reports_when_no_closed_form_solution_found(monkeypatch):
    N = sp.Function("N")
    eq = Equation(name="hard_ode", raw_expression="Derivative(N(t), t) = N(t)**N(t)", derivation="",
                   kind="ode", sympy_eq=sp.Eq(sp.Derivative(N(t), t), N(t) ** N(t)))
    monkeypatch.setattr("modules.ode_utils.solve_ode", lambda model: {})
    report = VerificationReport()
    _ode_checks(_model(equations=[eq]), report)
    check = next(c for c in report.checks if c.label == "ODE solved: hard_ode")
    assert not check.passed and "could not find a closed-form" in check.detail


def test_ode_checks_catches_exception_from_checkodesol(monkeypatch):
    N = sp.Function("N")
    eq = Equation(name="e0", raw_expression="Derivative(N(t), t) = -N(t)", derivation="",
                   kind="ode", sympy_eq=sp.Eq(sp.Derivative(N(t), t), -N(t)))
    monkeypatch.setattr("modules.ode_utils.solve_ode", lambda model: {"N": sp.exp(-t)})

    def fake_checkodesol(*a, **kw):
        raise RuntimeError("checkodesol blew up")
    monkeypatch.setattr("modules.verifier.sp.checkodesol", fake_checkodesol)
    report = VerificationReport()
    _ode_checks(_model(equations=[eq]), report)
    check = next(c for c in report.checks if c.label == "ODE solution check: e0")
    assert not check.passed and "Could not verify" in check.detail


def test_ode_checks_catches_exception_from_coupled_verification(monkeypatch):
    A, B = sp.Function("A"), sp.Function("B")
    eq_a = Equation(name="ea", raw_expression="Derivative(A(t), t) = -A(t)", derivation="",
                     kind="ode", sympy_eq=sp.Eq(sp.Derivative(A(t), t), -A(t)))
    eq_b = Equation(name="eb", raw_expression="Derivative(B(t), t) = A(t)", derivation="",
                     kind="ode", sympy_eq=sp.Eq(sp.Derivative(B(t), t), A(t)))
    monkeypatch.setattr("modules.ode_utils.solve_ode",
                          lambda model: {"A": sp.exp(-t), "B": 1 - sp.exp(-t)})
    monkeypatch.setattr("modules.ode_utils.group_coupled_odes", lambda eqs: [[eq_a, eq_b]])

    def fake_verify_coupled(*a, **kw):
        raise RuntimeError("coupled check blew up")
    monkeypatch.setattr("modules.ode_utils.verify_coupled_solution", fake_verify_coupled)
    monkeypatch.setattr("modules.ode_utils.numerical_cross_check",
                          lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("skip bonus check")))
    report = VerificationReport()
    _ode_checks(_model(equations=[eq_a, eq_b]), report)
    check = next(c for c in report.checks if c.label.startswith("Coupled ODE system check"))
    assert not check.passed and "Could not verify" in check.detail


def test_ode_checks_swallows_exception_from_bonus_numerical_cross_check(monkeypatch):
    """The numerical cross-check is explicitly a bonus rigor pass -- an
    exception there must not remove or corrupt the primary check result
    already added above it."""
    N = sp.Function("N")
    eq = Equation(name="e0", raw_expression="Derivative(N(t), t) = -N(t)", derivation="",
                   kind="ode", sympy_eq=sp.Eq(sp.Derivative(N(t), t), -N(t)))
    monkeypatch.setattr("modules.ode_utils.solve_ode", lambda model: {"N": sp.exp(-t)})

    def fake_cross_check(*a, **kw):
        raise RuntimeError("bonus check blew up")
    monkeypatch.setattr("modules.ode_utils.numerical_cross_check", fake_cross_check)
    report = VerificationReport()
    _ode_checks(_model(equations=[eq]), report)
    primary = next(c for c in report.checks if c.label == "ODE solution check: e0")
    assert primary.passed
    assert not any(c.label.startswith("Independent numerical cross-check") for c in report.checks)


# ================================================================== _recurrence_checks

def test_recurrence_checks_skips_equation_with_no_applied_function():
    eq = Equation(name="e0", raw_expression="1 = 1", derivation="", kind="recurrence",
                   sympy_eq=sp.Eq(sp.Integer(1), sp.Integer(1)))
    report = VerificationReport()
    _recurrence_checks(_model(equations=[eq]), report)
    assert report.checks == []


def test_recurrence_checks_reports_when_no_closed_form_found(monkeypatch):
    a = sp.Function("a")
    n = sp.Symbol("n")
    eq = Equation(name="e0", raw_expression="a(n+1) = a(n)**a(n)", derivation="", kind="recurrence",
                   sympy_eq=sp.Eq(a(n + 1), a(n) ** a(n)))
    monkeypatch.setattr("modules.recurrence_utils.solve_recurrence", lambda model: {})
    report = VerificationReport()
    _recurrence_checks(_model(equations=[eq]), report)
    check = next(c for c in report.checks if c.label == "Recurrence solved: e0")
    assert not check.passed


def test_recurrence_checks_reports_when_independent_variable_unresolvable(monkeypatch):
    a = sp.Function("a")
    eq = Equation(name="e0", raw_expression="a(1) = 5", derivation="", kind="recurrence",
                   sympy_eq=sp.Eq(a(1), 5))
    monkeypatch.setattr("modules.recurrence_utils.solve_recurrence", lambda model: {"a": sp.Integer(5)})
    report = VerificationReport()
    _recurrence_checks(_model(equations=[eq]), report)
    check = next(c for c in report.checks if c.label == "Recurrence solution check: e0")
    assert not check.passed and "independent (index) variable" in check.detail


def test_recurrence_checks_catches_exception_from_verification(monkeypatch):
    a = sp.Function("a")
    n = sp.Symbol("n")
    eq = Equation(name="e0", raw_expression="a(n+1) = a(n) + 1", derivation="", kind="recurrence",
                   sympy_eq=sp.Eq(a(n + 1), a(n) + 1))
    monkeypatch.setattr("modules.recurrence_utils.solve_recurrence", lambda model: {"a": n + 1})

    def fake_verify(*a_, **kw):
        raise RuntimeError("verification blew up")
    monkeypatch.setattr("modules.recurrence_utils.verify_recurrence_solution", fake_verify)
    report = VerificationReport()
    _recurrence_checks(_model(equations=[eq]), report)
    check = next(c for c in report.checks if c.label == "Recurrence solution check: e0")
    assert not check.passed and "Could not verify" in check.detail


# ================================================================== _optimization_checks

def _objective():
    return Objective(raw_expression="x**2", direction="minimize", optimize_over=["x"],
                       sympy_expr=x**2)


def test_optimization_checks_noop_without_an_objective():
    report = VerificationReport()
    _optimization_checks(_model(objective=None), report)
    assert report.checks == []


def test_optimization_checks_noop_when_solver_returns_none(monkeypatch):
    monkeypatch.setattr("modules.optimization_utils.solve_optimization", lambda model: None)
    report = VerificationReport()
    _optimization_checks(_model(objective=_objective()), report)
    assert report.checks == []


def test_optimization_checks_reports_solver_error(monkeypatch):
    result = OptimizationResult(error="objective depends on an unresolved variable")
    monkeypatch.setattr("modules.optimization_utils.solve_optimization", lambda model: result)
    report = VerificationReport()
    _optimization_checks(_model(objective=_objective()), report)
    check = next(c for c in report.checks if c.label.startswith("Optimization solved"))
    assert not check.passed and "unresolved variable" in check.detail


def test_optimization_checks_noop_with_no_critical_points(monkeypatch):
    result = OptimizationResult(critical_points=[])
    monkeypatch.setattr("modules.optimization_utils.solve_optimization", lambda model: result)
    report = VerificationReport()
    _optimization_checks(_model(objective=_objective()), report)
    assert report.checks == []


def test_optimization_checks_gradient_typeerror_marks_unverified(monkeypatch):
    """The critical point is missing a value for one of the objective's own
    variables (a malformed solver result) -- float() on the leftover
    symbolic gradient raises, caught and reported as NOT verified rather
    than crashing."""
    obj = Objective(raw_expression="x*y", direction="minimize", optimize_over=["x", "y"],
                      sympy_expr=x * y)
    result = OptimizationResult(critical_points=[{"x": sp.Integer(0)}],  # missing "y"
                                  classifications=["minimum"])
    monkeypatch.setattr("modules.optimization_utils.solve_optimization", lambda model: result)
    report = VerificationReport()
    _optimization_checks(_model(objective=obj), report)
    check = next(c for c in report.checks if c.label.startswith("Critical point verified"))
    assert not check.passed and "NOT zero" in check.detail


def test_optimization_checks_lagrange_constraint_satisfied(monkeypatch):
    """used_lagrange=True routes through the equality-constraint-satisfied
    branch instead of the gradient/classification branch."""
    eq = Equation(name="c0", raw_expression="Eq(x, 3)", derivation="", kind="equation",
                   sympy_eq=sp.Eq(x, 3))
    result = OptimizationResult(critical_points=[{"x": sp.Integer(3)}], classifications=["minimum"],
                                  used_lagrange=True)
    monkeypatch.setattr("modules.optimization_utils.solve_optimization", lambda model: result)
    report = VerificationReport()
    _optimization_checks(_model(objective=_objective(), equations=[eq]), report)
    check = next(c for c in report.checks if c.label.startswith("Constraint satisfied at critical point"))
    assert check.passed and "satisfies" in check.detail


def test_optimization_checks_lagrange_constraint_violated(monkeypatch):
    eq = Equation(name="c0", raw_expression="Eq(x, 3)", derivation="", kind="equation",
                   sympy_eq=sp.Eq(x, 3))
    result = OptimizationResult(critical_points=[{"x": sp.Integer(99)}], classifications=["minimum"],
                                  used_lagrange=True)
    monkeypatch.setattr("modules.optimization_utils.solve_optimization", lambda model: result)
    report = VerificationReport()
    _optimization_checks(_model(objective=_objective(), equations=[eq]), report)
    check = next(c for c in report.checks if c.label.startswith("Constraint satisfied at critical point"))
    assert not check.passed and "does NOT satisfy" in check.detail


def test_optimization_checks_lagrange_constraint_unresolvable_marks_unsatisfied(monkeypatch):
    """The critical point is missing a value needed to evaluate one of the
    equality constraints (a malformed solver result, mirroring the
    gradient-TypeError test above) -- float() on the leftover symbolic
    residual raises, caught and treated as constraint-not-satisfied rather
    than crashing."""
    eq = Equation(name="c0", raw_expression="Eq(x, y)", derivation="", kind="equation",
                   sympy_eq=sp.Eq(x, y))  # "y" never gets a value in the critical point below
    result = OptimizationResult(critical_points=[{"x": sp.Integer(3)}], classifications=["minimum"],
                                  used_lagrange=True)
    monkeypatch.setattr("modules.optimization_utils.solve_optimization", lambda model: result)
    report = VerificationReport()
    _optimization_checks(_model(objective=_objective(), equations=[eq]), report)
    check = next(c for c in report.checks if c.label.startswith("Constraint satisfied at critical point"))
    assert not check.passed and "does NOT satisfy" in check.detail


def test_optimization_checks_surfaces_feasibility_notes(monkeypatch):
    result = OptimizationResult(critical_points=[{"x": sp.Integer(0)}], classifications=["minimum"],
                                  feasibility_notes=["critical point violates x >= 0"])
    monkeypatch.setattr("modules.optimization_utils.solve_optimization", lambda model: result)
    report = VerificationReport()
    _optimization_checks(_model(objective=_objective()), report)
    note = next(c for c in report.checks if c.label == "Feasibility vs. inequality constraints")
    assert not note.passed and "violates" in note.detail


# ================================================================== _matrix_system_checks

def test_matrix_system_checks_reports_singular_but_consistent():
    """A 2x2 system that's singular (det = 0) but still consistent (one
    equation is a multiple of the other) -- passes, with the singular
    determinant called out explicitly rather than silently."""
    eq1 = Equation(name="e1", raw_expression="Eq(x + y, 4)", derivation="", kind="equation",
                    sympy_eq=sp.Eq(x + y, 4))
    eq2 = Equation(name="e2", raw_expression="Eq(2*x + 2*y, 8)", derivation="", kind="equation",
                    sympy_eq=sp.Eq(2 * x + 2 * y, 8))
    vx = Variable(symbol="x", meaning="x", known_value=None, unit=None)
    vy = Variable(symbol="y", meaning="y", known_value=None, unit=None)
    report = VerificationReport()
    _matrix_system_checks(_model(variables=[vx, vy], equations=[eq1, eq2], solve_for=["x", "y"]), report)
    check = next(c for c in report.checks if c.label == "Linear system consistency")
    assert check.passed and "singular" in check.detail


# ================================================================== _domain_checks

def test_domain_checks_reports_pending_restriction_for_unknown_symbol():
    """1/x with x still unknown -- the restriction (x != 0) can't yet be
    evaluated against known values, so it's reported as a passing,
    informational "pending" note rather than violated or fully satisfied."""
    eq = Equation(name="e0", raw_expression="Eq(y, 1/x)", derivation="", kind="equation",
                   sympy_eq=sp.Eq(y, 1 / x))
    vx = Variable(symbol="x", meaning="x", known_value=None, unit=None)
    report = VerificationReport()
    _domain_checks(_model(variables=[vx], equations=[eq]), report)
    check = next(c for c in report.checks if c.label == "Domain validity: e0")
    assert check.passed and "can't fully check yet" in check.detail


# ================================================================== _solve_sympy

def test_solve_sympy_returns_empty_when_no_algebraic_equations():
    vx = Variable(symbol="x", meaning="x", known_value=None, unit=None)
    ineq = Equation(name="c0", raw_expression="x < 5", derivation="", kind="inequality",
                     sympy_eq=sp.Lt(x, 5))
    result = _solve_sympy(_model(variables=[vx], equations=[ineq], solve_for=["x"]))
    assert result == {}


def test_solve_sympy_returns_empty_when_no_equation_kind_relations_at_all():
    """solve_for names a target with no matching equation of ANY kind --
    target_kind() falls back to 'equation' by default, so algebraic_targets
    is non-empty, but the `eqs` list built from actual equation-kind
    relations is empty -- covers that specific guard, distinct from the
    algebraic_targets-empty guard above."""
    result = _solve_sympy(_model(solve_for=["x"]))
    assert result == {}


def test_solve_sympy_reraises_timeout_for_the_caller_to_report(monkeypatch):
    eq = Equation(name="e0", raw_expression="Eq(x, 5)", derivation="", kind="equation",
                   sympy_eq=sp.Eq(x, 5))
    vx = Variable(symbol="x", meaning="x", known_value=None, unit=None)

    def fake_timeout(*a, **kw):
        raise ComputationTimeoutError(30.0, label="algebraic solve")
    monkeypatch.setattr("modules.verifier.run_with_timeout", fake_timeout)
    with pytest.raises(ComputationTimeoutError):
        _solve_sympy(_model(variables=[vx], equations=[eq], solve_for=["x"]))


def test_solve_sympy_swallows_a_generic_solve_exception(monkeypatch):
    eq = Equation(name="e0", raw_expression="Eq(x, 5)", derivation="", kind="equation",
                   sympy_eq=sp.Eq(x, 5))
    vx = Variable(symbol="x", meaning="x", known_value=None, unit=None)

    def fake_timeout(*a, **kw):
        raise RuntimeError("sympy blew up internally")
    monkeypatch.setattr("modules.verifier.run_with_timeout", fake_timeout)
    result = _solve_sympy(_model(variables=[vx], equations=[eq], solve_for=["x"]))
    assert result == {}


def test_solve_sympy_skips_target_that_did_not_solve():
    """Two targets requested, but sp.solve() only manages to pin down one
    of them (the other stays as a free expression, e.g. because the system
    is underdetermined for it) -- the unsolved one is silently omitted
    rather than crashing on `.is_number`."""
    eq = Equation(name="e0", raw_expression="Eq(x, 5)", derivation="", kind="equation",
                   sympy_eq=sp.Eq(x, 5))
    vx = Variable(symbol="x", meaning="x", known_value=None, unit=None)
    vy = Variable(symbol="y", meaning="y", known_value=None, unit=None)
    result = _solve_sympy(_model(variables=[vx, vy], equations=[eq], solve_for=["x", "y"]))
    assert result == {"x": 5.0}
    assert "y" not in result


def test_solve_sympy_skips_target_whose_value_cannot_convert_to_complex(monkeypatch):
    """The solved value is a real number but genuinely can't be converted
    via complex() (a stand-in for a pathological non-numeric-but-
    `.is_number`-True sympy object) -- caught and skipped, matching how
    monte_carlo.py/goal_seek.py already treat this tolerantly."""
    eq = Equation(name="e0", raw_expression="Eq(x, 5)", derivation="", kind="equation",
                   sympy_eq=sp.Eq(x, 5))
    vx = Variable(symbol="x", meaning="x", known_value=None, unit=None)

    class Unconvertible(sp.Integer):
        def __complex__(self):
            raise TypeError("cannot convert")

    def fake_timeout(func, *a, **kw):
        return [{x: Unconvertible(5)}]
    monkeypatch.setattr("modules.verifier.run_with_timeout", fake_timeout)
    result = _solve_sympy(_model(variables=[vx], equations=[eq], solve_for=["x"]))
    assert result == {}


# ================================================================== _extract_final_numbers

def test_extract_final_numbers_skips_a_malformed_number():
    """'1.2.3' matches the permissive digits/dot/sign/exponent regex but
    isn't actually a valid float -- float() raises ValueError, caught so
    one malformed line doesn't drop every other, well-formed answer on the
    same response."""
    text = "FINAL_NUMERIC_ANSWER[x]: 1.2.3\nFINAL_NUMERIC_ANSWER[y]: 4.5"
    result = _extract_final_numbers(text)
    assert "x" not in result
    assert result["y"] == 4.5
