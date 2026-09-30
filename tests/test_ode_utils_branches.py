"""Branch-coverage tests for modules/ode_utils.py: the defensive and
failure paths that the happy-path suites (test_ode_systems.py,
test_ode_numerical_cross_check.py) never reach -- transitive group merging,
unparseable initial conditions, every way numerical_cross_check can decline
("not applicable") or fail honestly, and verify_coupled_solution rejecting a
wrong solution.

Failure paths that depend on SymPy/SciPy misbehaving (dsolve raising, a
lambdify failure, an integrator that doesn't converge) are reached by
intercepting the single call involved, because such failures can't be
provoked reliably from real input -- and a test that only passes when a
third-party library happens to fail in a particular way would be flaky.
"""
from types import SimpleNamespace

import pytest
import sympy as sp

import modules.ode_utils as ou
from modules.equation_engine import Equation, build_model
from modules.ode_utils import (
    group_coupled_odes, solve_ode, numerical_cross_check, verify_coupled_solution, _defined_function,
)
from modules.timeout_utils import ComputationTimeoutError

t = sp.Symbol("t")
A, B, C, N = (sp.Function(n) for n in "ABCN")


def _eq(name, sympy_eq):
    return Equation(name=name, raw_expression=str(sympy_eq), derivation="", kind="ode", sympy_eq=sympy_eq)


def _decay_model(ics=None, k=0.5):
    return build_model({
        "problem_domain": "physics", "problem_type": "ode", "independent_variable": "t",
        "variables": [
            {"symbol": "k", "meaning": "rate", "known_value": k, "unit": None, "is_function": False},
            {"symbol": "N", "meaning": "quantity", "known_value": None, "unit": None, "is_function": True},
        ],
        "equations": [{"name": "decay", "kind": "ode", "expression": "Eq(Derivative(N(t), t), -k*N(t))",
                       "derivation": ""}],
        "solve_for": ["N"], "assumptions": [],
        "initial_conditions": [{"expression": "N(0)", "value": 100.0}] if ics is None else ics,
    })


def _chain_model(b_ic_expr="B(0)"):
    return build_model({
        "problem_domain": "physics", "problem_type": "ode", "independent_variable": "t",
        "variables": [
            {"symbol": "k1", "meaning": "r1", "known_value": 0.3, "unit": None, "is_function": False},
            {"symbol": "k2", "meaning": "r2", "known_value": 0.1, "unit": None, "is_function": False},
            {"symbol": "A", "meaning": "A", "known_value": None, "unit": None, "is_function": True},
            {"symbol": "B", "meaning": "B", "known_value": None, "unit": None, "is_function": True},
        ],
        "equations": [
            {"name": "A decay", "kind": "ode", "expression": "Eq(Derivative(A(t), t), -k1*A(t))", "derivation": ""},
            {"name": "B prod", "kind": "ode", "expression": "Eq(Derivative(B(t), t), k1*A(t) - k2*B(t))",
             "derivation": ""},
        ],
        "solve_for": ["A", "B"], "assumptions": [],
        "initial_conditions": [{"expression": "A(0)", "value": 50.0}, {"expression": b_ic_expr, "value": 10.0}],
    })


def _group(model):
    return group_coupled_odes([e for e in model.equations if e.kind == "ode"])[0]


# ------------------------------------------------------------------ group_coupled_odes

def test_equation_without_a_parsed_sympy_form_is_left_out_of_every_group():
    broken = Equation(name="broken", raw_expression="Eq(", derivation="", kind="ode", sympy_eq=None)
    good = _eq("good", sp.Eq(A(t).diff(t), -A(t)))
    groups = group_coupled_odes([broken, good])
    assert len(groups) == 1 and groups[0] == [good]


def test_groups_that_only_overlap_later_are_merged_transitively():
    # eq1 uses {A}, eq2 uses {C} (disjoint on the first pass -> two groups), and
    # eq3 bridges them via {A, C}. The second pass must fuse everything into one.
    e1 = _eq("e1", sp.Eq(A(t).diff(t), -A(t)))
    e2 = _eq("e2", sp.Eq(C(t).diff(t), -C(t)))
    e3 = _eq("e3", sp.Eq(A(t).diff(t), C(t)))
    groups = group_coupled_odes([e1, e2, e3])
    assert len(groups) == 1
    assert {e.name for e in groups[0]} == {"e1", "e2", "e3"}


def test_disjoint_equations_stay_in_separate_groups():
    e1 = _eq("e1", sp.Eq(A(t).diff(t), -A(t)))
    e2 = _eq("e2", sp.Eq(C(t).diff(t), -C(t)))
    assert len(group_coupled_odes([e1, e2])) == 2


# ------------------------------------------------------------------ solve_ode

def test_unparseable_initial_condition_is_ignored_and_general_solution_returned():
    model = _decay_model(ics=[{"expression": "N(0 +", "value": 100.0}])
    assert model.initial_conditions[0].sympy_eq is None          # parse failed
    sol = solve_ode(model)
    assert "N" in sol
    assert sol["N"].rhs.has(sp.Symbol("C1"))                      # no IC applied -> constant left free


def test_standalone_ode_whose_dsolve_fails_is_skipped_not_fatal(monkeypatch):
    def boom(*a, **k):
        raise ComputationTimeoutError(0.1, "dsolve")
    monkeypatch.setattr(ou, "run_with_timeout", boom)
    assert solve_ode(_decay_model()) == {}


def test_coupled_system_whose_dsolve_system_fails_is_skipped_not_fatal(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("no closed form")
    monkeypatch.setattr(ou, "run_with_timeout", boom)
    assert solve_ode(_chain_model()) == {}


# ------------------------------------------------------------------ verify_coupled_solution

def test_coupled_verification_accepts_the_true_solution():
    model = _chain_model()
    ok, residual = verify_coupled_solution(_group(model), solve_ode(model))
    assert ok is True and residual == 0


def test_coupled_verification_rejects_a_wrong_solution_with_its_residual():
    model = _chain_model()
    wrong = {"A": sp.Eq(A(t), 50 * sp.exp(-0.3 * t)),
             "B": sp.Eq(B(t), 10 * sp.exp(-0.1 * t))}     # B ignores the A -> B production term
    ok, residual = verify_coupled_solution(_group(model), wrong)
    assert ok is False
    assert residual != 0


def test_coupled_verification_falls_back_to_raw_residual_when_simplify_fails(monkeypatch):
    model = _chain_model()
    wrong = {"A": sp.Eq(A(t), 50 * sp.exp(-0.3 * t)), "B": sp.Eq(B(t), sp.Integer(1))}

    def boom(*a, **k):
        raise RuntimeError("simplify failed")
    monkeypatch.setattr(ou.sp, "simplify", boom)
    ok, residual = verify_coupled_solution(_group(model), wrong)
    assert ok is False and residual != 0      # an unsimplified residual is still a valid nonzero witness


# ------------------------------------------------------------------ _defined_function

def test_defined_function_is_none_without_exactly_one_differentiated_function():
    assert _defined_function(sp.Eq(A(t), 1)) is None                       # no derivative at all
    assert _defined_function(sp.Eq(A(t).diff(t), B(t).diff(t))) is None   # two -> ambiguous, refuse to guess


# ------------------------------------------------------------------ numerical_cross_check: declining

def test_cross_check_declines_when_function_cannot_be_identified():
    model = _decay_model()
    group = [_eq("no derivative", sp.Eq(N(t), 1))]
    r = numerical_cross_check(model, group, {})
    assert not r.applicable and "Could not identify the function" in r.reason


def test_cross_check_declines_when_order_cannot_be_determined(monkeypatch):
    model = _decay_model()

    def boom(*a, **k):
        raise ValueError("bad order")
    monkeypatch.setattr(ou, "ode_order", boom)
    r = numerical_cross_check(model, _group(model), solve_ode(model))
    assert not r.applicable and r.reason == "Could not determine ODE order."


def test_cross_check_declines_when_a_function_has_no_closed_form():
    model = _decay_model()
    r = numerical_cross_check(model, _group(model), {})          # solutions dict is empty
    assert not r.applicable and "closed-form solution" in r.reason


def test_cross_check_declines_when_initial_conditions_are_at_different_points():
    model = _chain_model(b_ic_expr="B(1)")                        # A(0) but B(1)
    r = numerical_cross_check(model, _group(model), solve_ode(model))
    assert not r.applicable and "different points" in r.reason


def test_cross_check_declines_when_initial_condition_is_not_at_a_numeric_point():
    model = _decay_model(ics=[{"expression": "N(tau)", "value": 100.0}])
    r = numerical_cross_check(model, _group(model), solve_ode(model))
    assert not r.applicable and "plain" in r.reason


def test_cross_check_ignores_derivative_initial_conditions_and_still_runs():
    # an extra y'(0)-style condition is out of scope for the first-order check;
    # it must be skipped, NOT treated as a second function's IC or as an error.
    # (dsolve itself rejects that over-determined IC set, so the closed form is
    # solved from the plain model and only the cross-check sees the extra IC.)
    plain = _decay_model()
    with_derivative_ic = _decay_model(ics=[
        {"expression": "N(0)", "value": 100.0},
        {"expression": "Derivative(N(t), t).subs(t, 0)", "value": -50.0},
    ])
    assert len(with_derivative_ic.initial_conditions) == 2
    r = numerical_cross_check(with_derivative_ic, _group(with_derivative_ic), solve_ode(plain))
    assert r.applicable and r.ok


def test_cross_check_declines_when_second_function_identification_fails(monkeypatch):
    # defensive re-check while building right-hand sides: identify fine on the
    # first pass, fail on the second
    model = _decay_model()
    real = ou._defined_function
    calls = {"n": 0}

    def flaky(eq):
        calls["n"] += 1
        return real(eq) if calls["n"] == 1 else None
    monkeypatch.setattr(ou, "_defined_function", flaky)
    r = numerical_cross_check(model, _group(model), solve_ode(model))
    assert not r.applicable and "Could not identify the function in decay" in r.reason


def test_cross_check_declines_when_derivative_cannot_be_isolated(monkeypatch):
    model = _decay_model()
    sols, group = solve_ode(model), _group(model)

    def boom(*a, **k):
        raise NotImplementedError
    monkeypatch.setattr(ou.sp, "solve", boom)
    r = numerical_cross_check(model, group, sols)
    assert not r.applicable and "Could not isolate the derivative in decay" in r.reason


def test_cross_check_declines_when_solve_finds_nothing(monkeypatch):
    model = _decay_model()
    sols, group = solve_ode(model), _group(model)
    monkeypatch.setattr(ou.sp, "solve", lambda *a, **k: [])
    r = numerical_cross_check(model, group, sols)
    assert not r.applicable and "Could not isolate the derivative in decay" in r.reason


def test_cross_check_declines_when_numeric_form_cannot_be_built(monkeypatch):
    model = _decay_model()
    sols, group = solve_ode(model), _group(model)

    def boom(*a, **k):
        raise TypeError("cannot lambdify")
    monkeypatch.setattr(ou.sp, "lambdify", boom)
    r = numerical_cross_check(model, group, sols)
    assert not r.applicable and "Could not build a numeric form: cannot lambdify" in r.reason


# ------------------------------------------------------------------ numerical_cross_check: integration failures

def test_cross_check_falls_back_to_default_window_when_initial_rate_cannot_be_estimated(monkeypatch):
    model = _decay_model()
    sols, group = solve_ode(model), _group(model)
    real = ou.sp.lambdify
    state = {"first": True}

    def wrapped(*a, **k):
        fn = real(*a, **k)

        def call(*args):
            if state["first"]:
                state["first"] = False            # the very first evaluation is the rate estimate
                raise ArithmeticError("rate estimate failed")
            return fn(*args)
        return call
    monkeypatch.setattr(ou.sp, "lambdify", wrapped)
    r = numerical_cross_check(model, group, sols)
    assert r.applicable and r.ok
    assert r.sample_points[-1] == pytest.approx(5.0)   # the 5.0 fallback window, t0 = 0


def test_cross_check_reports_integration_exception(monkeypatch):
    model = _decay_model()
    sols, group = solve_ode(model), _group(model)

    def boom(*a, **k):
        raise ComputationTimeoutError(0.1, "ode_numerical_cross_check")
    monkeypatch.setattr(ou, "run_with_timeout", boom)
    r = numerical_cross_check(model, group, sols)
    assert not r.applicable and "Numerical integration failed" in r.reason


def test_cross_check_reports_integrator_that_did_not_converge(monkeypatch):
    model = _decay_model()
    sols, group = solve_ode(model), _group(model)
    monkeypatch.setattr(ou, "run_with_timeout",
                        lambda *a, **k: SimpleNamespace(success=False, message="step size too small"))
    r = numerical_cross_check(model, group, sols)
    assert not r.applicable and "did not converge: step size too small" in r.reason


def test_cross_check_reports_failure_evaluating_the_trajectory(monkeypatch):
    model = _decay_model()
    sols, group = solve_ode(model), _group(model)

    def bad_sol(_ts):
        raise ValueError("dense output unavailable")
    monkeypatch.setattr(ou, "run_with_timeout", lambda *a, **k: SimpleNamespace(success=True, sol=bad_sol))
    r = numerical_cross_check(model, group, sols)
    assert not r.applicable and "Could not evaluate for comparison" in r.reason
