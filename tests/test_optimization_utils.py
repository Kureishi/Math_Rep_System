"""
Targeted tests for modules/optimization_utils.py branches the existing
tests/test_optimization.py (which exercises the "happy path" end to end
via solve_optimization()) doesn't reach: the private helpers' edge cases,
and solve_optimization()'s error-returning branches. Kept in a separate
file (matching the module's own name) rather than added to
test_optimization.py, since these are deliberately unit-level -- calling
_eliminate_greedy/_classify_critical_point/_backfill_point directly -- while
that file stays end-to-end through solve_optimization().
"""

import sympy as sp
import math

from modules.equation_engine import build_model
from modules.optimization_utils import (
    OptimizationResult, _backfill_point, _classify_critical_point,
    _eliminate_greedy, solve_optimization, gradient_descent_path,
)

x, y, z, h, r = sp.symbols("x y z h r")


# ---------------------------------------------------------------- _eliminate_greedy

def test_eliminate_greedy_no_free_symbols_returns_as_is():
    """A constant objective has nothing to eliminate -- covers the
    `if not obj_free: return objective_expr, {}, []` short-circuit."""
    reduced, eliminated, free_vars = _eliminate_greedy(sp.Integer(5), ["x"], [])
    assert reduced == 5 and eliminated == {} and free_vars == []


def test_eliminate_greedy_single_variable_skips_elimination_when_no_constraint_touches_it():
    """With only one eligible variable, the elimination loop never runs (a
    single free variable doesn't need eliminating) -- the objective and
    variable are returned unchanged, still as the one free_var. That is only
    safe when no constraint actually constrains that variable."""
    reduced, eliminated, free_vars = _eliminate_greedy(x, ["x"], [y - 3])   # constraint is on an unrelated y
    assert reduced == x and eliminated == {} and free_vars == [x]


def test_eliminate_greedy_single_variable_with_a_constraint_on_it_fails_to_lagrange():
    """Regression: a constraint pinning the ONLY free variable used to be
    silently dropped (the loop never runs for one variable), so the caller
    then differentiated the unconstrained objective and reported a bogus
    'no critical points' for a fully-determined problem. Now it fails so the
    caller falls back to Lagrange multipliers, which honour the constraint."""
    assert _eliminate_greedy(x, ["x"], [x - 3]) == (None, {}, [])


def test_eliminate_greedy_prefers_helper_over_requested_variable():
    """V = pi*r**2*h with h a helper (not in optimize_over) and a constraint
    fixing h: h should be the one eliminated, leaving r free -- not the
    reverse, and not arbitrary (set-iteration-order) behaviour."""
    objective = sp.pi * r**2 * h
    constraint = h - 10  # h = 10
    reduced, eliminated, free_vars = _eliminate_greedy(objective, ["r"], [constraint])
    assert "h" in eliminated and "r" not in eliminated
    assert reduced == sp.pi * r**2 * 10
    assert free_vars == [r]


def test_eliminate_greedy_stops_once_one_variable_remains():
    """x + y with ONE constraint (x = 3): x is eliminated, a single free
    variable (y) remains, and the loop stops -- nothing left to eliminate or
    to honour."""
    reduced, eliminated, free_vars = _eliminate_greedy(x + y, ["x", "y"], [x - 3])
    assert eliminated == {"x": 3} and free_vars == [y]
    assert reduced == y + 3


def test_eliminate_greedy_fails_when_a_second_constraint_pins_the_last_free_variable():
    """Regression: x + y with BOTH x=3 and y=4. The loop eliminates x and
    stops with y free (it must keep one variable to differentiate) -- which
    used to silently discard y = 4. The leftover constraint on a still-free
    variable must make this fail, not return an objective that ignores it."""
    assert _eliminate_greedy(x + y, ["x", "y"], [x - 3, y - 4]) == (None, {}, [])


# ---------------------------------------------------------------- _backfill_point

def test_backfill_point_resolves_chained_eliminated_variables():
    """b was eliminated in terms of a (still-eliminated) c, and c in terms of
    the free variable a -- multi-hop resolution via the repeated-substitution
    loop, not just a single pass."""
    free_point = {sp.Symbol("a"): sp.Integer(2)}
    eliminated = {"c": sp.Symbol("a") + 1, "b": sp.Symbol("c") * 10}
    result = _backfill_point(free_point, eliminated)
    assert result == {"a": 2, "c": 3, "b": 30}


def test_backfill_point_leaves_unresolvable_entry_best_effort():
    """An eliminated variable that references a name absent from both the
    free point and the rest of `eliminated` can never fully resolve --
    covers the best-effort tail loop rather than raising."""
    free_point = {sp.Symbol("a"): sp.Integer(2)}
    eliminated = {"b": sp.Symbol("a") + sp.Symbol("unknown_var")}
    result = _backfill_point(free_point, eliminated)
    assert result["a"] == 2
    assert sp.Symbol("unknown_var") in result["b"].free_symbols


# ---------------------------------------------------------------- _classify_critical_point

def test_classify_single_variable_minimum_and_maximum():
    obj = x**2
    assert _classify_critical_point(obj, [x], {x: 0}) == "minimum"
    assert _classify_critical_point(-x**2, [x], {x: 0}) == "maximum"


def test_classify_single_variable_inconclusive_on_zero_second_derivative():
    """x**3 has a zero second derivative at x=0 (an inflection point, not an
    extremum) -- the second-derivative test is genuinely inconclusive there."""
    assert _classify_critical_point(x**3, [x], {x: 0}) == "inconclusive"


def test_classify_single_variable_inconclusive_when_not_numeric():
    """The point doesn't pin down every free symbol in the second derivative
    -- float() raises TypeError, caught and reported as inconclusive rather
    than propagating."""
    assert _classify_critical_point(x**2 * y, [x], {x: 0}) == "inconclusive"


def test_classify_multivariable_saddle_point():
    """z = x**2 - y**2 at the origin: mixed-sign Hessian eigenvalues -> saddle."""
    assert _classify_critical_point(x**2 - y**2, [x, y], {x: 0, y: 0}) == "saddle point"


def test_classify_multivariable_minimum_and_maximum():
    assert _classify_critical_point(x**2 + y**2, [x, y], {x: 0, y: 0}) == "minimum"
    assert _classify_critical_point(-x**2 - y**2, [x, y], {x: 0, y: 0}) == "maximum"


def test_classify_multivariable_inconclusive_on_degenerate_hessian():
    """z = x**2 (no y-dependence at all): the Hessian has a zero eigenvalue,
    so the test can't conclude -- covers the `len(reals) != len(eigenvals)`
    / zero-eigenvalue "inconclusive" path."""
    assert _classify_critical_point(x**2, [x, y], {x: 0, y: 0}) == "inconclusive"


# ---------------------------------------------------------------- solve_optimization edge cases

def _model(objective_expr, optimize_over, equations=(), extra_vars=(), solve_for=None):
    import re
    all_vars = set(optimize_over) | {str(s) for s in sp.sympify(objective_expr).free_symbols} | set(extra_vars)
    for eq in equations:
        all_vars |= set(re.findall(r"[A-Za-z_]\w*", eq)) - {"Eq"}
    payload = {
        "problem_domain": "test", "problem_type": "algebraic",
        "variables": [{"symbol": v, "meaning": v, "known_value": None, "unit": None}
                      for v in sorted(all_vars)],
        "equations": [{"name": f"c{i}", "kind": "equation", "expression": e, "derivation": ""}
                      for i, e in enumerate(equations)],
        "objective": {"expression": objective_expr, "direction": "minimize", "optimize_over": optimize_over},
        "solve_for": solve_for or optimize_over, "assumptions": [],
    }
    return build_model(payload)


def test_no_optimize_over_variables_returns_explicit_error():
    model = _model("x**2", optimize_over=[])
    result = solve_optimization(model)
    assert isinstance(result, OptimizationResult)
    assert result.error is not None and "optimize_over" in result.error


def test_objective_depends_on_variable_neither_known_nor_optimized_errors():
    """x**2 + z, optimize_over=[x], with a constraint linking x and y (so
    elimination is attempted -- should_try_elimination requires a constraint
    touching the objective's variables) but that constraint says nothing
    about z: z is a genuinely stuck leftover, and solve_optimization should
    report the specific error rather than silently optimizing over the
    wrong thing."""
    model = _model("x**2 + z", optimize_over=["x"], equations=["Eq(x, y)"])
    result = solve_optimization(model)
    assert result.error is not None
    assert "z" in result.error


def test_no_real_critical_points_reports_error_not_crash():
    """x**2 + 1 has no critical point where the derivative is zero for a
    complex-only solution branch is avoided here by using an objective whose
    only critical point is genuinely complex: x**2 + I is not realistic for
    this pipeline's real-valued variables, so instead use an objective with
    no stationary point at all over the reals: exp(x) has diff = exp(x),
    never zero -- solve() returns no solutions, exercising the
    'No real-valued critical points' branch without any exception."""
    model = _model("exp(x)", optimize_over=["x"])
    result = solve_optimization(model)
    assert result.error is not None
    assert "critical points" in result.error.lower() or "no solution" in result.error.lower()


# ---------------------------------------------------------------- gradient_descent_path

def test_gradient_descent_path_converges_to_the_minimum():
    path = gradient_descent_path(x**2 + y**2, [x, y], (3.0, 2.0), direction="minimize")
    assert path[0] == (3.0, 2.0)
    assert abs(path[-1][0]) < 1e-3 and abs(path[-1][1]) < 1e-3
    assert len(path) > 1


def test_gradient_descent_path_converges_to_the_maximum():
    path = gradient_descent_path(-x**2 - y**2 + 5, [x, y], (3.0, 2.0), direction="maximize")
    assert abs(path[-1][0]) < 1e-3 and abs(path[-1][1]) < 1e-3


def test_gradient_descent_path_backtracks_on_an_elongated_objective():
    """A large learning rate that would overshoot on the steep axis of an
    elongated bowl -- backtracking halves the step until it actually
    improves the objective, rather than diverging or oscillating forever."""
    path = gradient_descent_path(5*x**2 + y**2, [x, y], (4.0, 4.0), direction="minimize",
                                    learning_rate=0.3)
    assert abs(path[-1][0]) < 1e-3 and abs(path[-1][1]) < 1e-3
    assert all(math.isfinite(p) for p in path[-1])


def test_gradient_descent_path_stops_early_once_converged():
    """A starting point already essentially at the minimum should converge
    in very few iterations, not run the full max_iters budget."""
    path = gradient_descent_path(x**2 + y**2, [x, y], (1e-8, 1e-8), direction="minimize",
                                    max_iters=100)
    assert len(path) < 100


# ---------------------------------------------------------------- remaining branches (coverage round)
import pytest

import modules.optimization_utils as ou
from modules.timeout_utils import ComputationTimeoutError


def test_eliminate_greedy_skips_to_next_candidate_when_a_solve_raises(monkeypatch):
    # constraint x + y - 10 = 0 could eliminate either variable. The first
    # attempted solve (for x) fails; the heuristic must skip on to y rather
    # than abort -- this is the "best-effort elimination" contract.
    real = ou.run_with_timeout
    calls = {"n": 0}

    def flaky(func, *args, **kwargs):
        if kwargs.get("label") == "constraint elimination":
            calls["n"] += 1
            if calls["n"] == 1:
                raise ComputationTimeoutError(0.1, "constraint elimination")
        return real(func, *args, **kwargs)

    monkeypatch.setattr(ou, "run_with_timeout", flaky)
    reduced, eliminated, free_vars = _eliminate_greedy(x * y, ["x", "y"], [x + y - 10])
    assert calls["n"] == 2
    assert list(eliminated) == ["y"]                      # x failed, y succeeded
    assert sp.simplify(eliminated["y"] - (10 - x)) == 0
    assert free_vars == [x]


def test_eliminate_greedy_returns_failure_when_objective_collapses_to_a_constant():
    # x - y subject to x - y = 5: eliminating x turns the objective into the
    # constant 5, leaving no variable to differentiate -- signalled as
    # (None, {}, []) so the caller falls back to Lagrange multipliers
    reduced, eliminated, free_vars = _eliminate_greedy(x - y, ["x", "y"], [x - y - 5])
    assert reduced is None and eliminated == {} and free_vars == []


def test_classify_inconclusive_when_hessian_eigenvalues_stay_symbolic():
    k = sp.Symbol("k")
    # Hessian of x**2 + y**2 + k*x*y has eigenvalues containing the free
    # symbol k, so complex() can't convert them -> inconclusive, not a crash
    assert _classify_critical_point(x**2 + y**2 + k * x * y, [x, y], {x: 0, y: 0}) == "inconclusive"


def test_classify_inconclusive_when_hessian_eigenvalues_are_complex():
    # Hessian of I*x*y is [[0, I], [I, 0]]: eigenvalues +/- I, purely imaginary.
    # A second-derivative test is meaningless for those -> inconclusive.
    assert _classify_critical_point(sp.I * x * y, [x, y], {x: 0, y: 0}) == "inconclusive"


def test_solve_optimization_returns_none_when_model_has_no_objective(kinematics_json):
    import json
    model = build_model(json.loads(kinematics_json))
    assert model.objective is None
    assert solve_optimization(model) is None


# -- Lagrange-multiplier path. It is only reached when elimination collapses
# the objective (see test above), so x - y subject to x - y = 5 is the
# smallest real model that drives solve_optimization down it.
def _lagrange_model():
    return _model("x - y", optimize_over=["x", "y"], equations=["Eq(x - y, 5)"])


def test_lagrange_path_returns_constrained_critical_point_and_multiplier():
    result = solve_optimization(_lagrange_model())
    assert result.error is None
    assert result.used_lagrange is True
    assert len(result.critical_points) == len(result.classifications) == len(result.multiplier_values) >= 1
    assert "constrained" in result.classifications[0]         # explicitly NOT second-order classified
    lam = next(iter(result.multiplier_values[0].values()))
    assert lam == 1                                            # dL/dx = 1 - lambda = 0


def test_lagrange_solve_timeout_is_reported(monkeypatch):
    real = ou.run_with_timeout

    def boom(func, *args, **kwargs):
        # only the Lagrange solve times out -- the earlier elimination attempt
        # must still run normally or the model never reaches the Lagrange path
        if kwargs.get("label") == "Lagrange system solve":
            raise ComputationTimeoutError(0.1, "Lagrange system solve")
        return real(func, *args, **kwargs)
    monkeypatch.setattr(ou, "run_with_timeout", boom)
    result = solve_optimization(_lagrange_model())
    assert result.used_lagrange is True
    assert "Timed out solving the Lagrange system" in result.error


def test_lagrange_solve_crash_is_reported(monkeypatch):
    real = ou.run_with_timeout

    def boom(func, *args, **kwargs):
        if kwargs.get("label") == "Lagrange system solve":
            raise RuntimeError("solver exploded")
        return real(func, *args, **kwargs)
    monkeypatch.setattr(ou, "run_with_timeout", boom)
    result = solve_optimization(_lagrange_model())
    assert result.used_lagrange is True
    assert "Could not solve the Lagrange system: solver exploded" in result.error


def test_lagrange_with_no_real_solution_is_reported(monkeypatch):
    real = ou.run_with_timeout

    def none_found(func, *args, **kwargs):
        if kwargs.get("label") == "Lagrange system solve":
            return []
        return real(func, *args, **kwargs)
    monkeypatch.setattr(ou, "run_with_timeout", none_found)
    result = solve_optimization(_lagrange_model())
    assert result.used_lagrange is True
    assert "no real solution" in result.error


# -- critical-point solve failures on the ordinary (non-Lagrange) path

def test_critical_point_solve_timeout_is_reported(monkeypatch):
    def boom(*a, **k):
        raise ComputationTimeoutError(0.1, "critical point solve")
    monkeypatch.setattr(ou, "run_with_timeout", boom)
    result = solve_optimization(_model("x**2 - 4*x", optimize_over=["x"]))
    assert "Timed out solving for critical points" in result.error


def test_critical_point_solve_crash_is_reported(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("solver exploded")
    monkeypatch.setattr(ou, "run_with_timeout", boom)
    result = solve_optimization(_model("x**2 - 4*x", optimize_over=["x"]))
    assert "Could not solve for critical points: solver exploded" in result.error


def test_inequality_that_cannot_be_evaluated_is_skipped_not_fatal():
    # an inequality mentioning a variable with no value at the critical point
    # can't be decided -> skipped silently (no note, no crash); the genuinely
    # violated one next to it is still reported
    model = build_model({
        "problem_domain": "t", "problem_type": "algebraic",
        "variables": [{"symbol": s, "meaning": s, "known_value": None, "unit": None} for s in ("x", "w")],
        "equations": [
            {"name": "undecidable", "kind": "inequality", "expression": "w >= 1", "derivation": ""},
            {"name": "too small", "kind": "inequality", "expression": "x >= 5", "derivation": ""},
        ],
        "objective": {"expression": "x**2 - 4*x + 7", "direction": "minimize", "optimize_over": ["x"]},
        "solve_for": ["x"], "assumptions": [],
    })
    result = solve_optimization(model)
    assert result.error is None
    assert len(result.feasibility_notes) == 1
    assert "too small" in result.feasibility_notes[0]


# ---------------------------------------------------------------- gradient_descent_path failure handling

def _fake_lambdify(f_impl, grad_impl):
    """gradient_descent_path calls sp.lambdify twice -- objective first, then
    the gradient list -- so hand back the two stand-ins in that order."""
    impls = iter([f_impl, grad_impl])
    return lambda *a, **k: next(impls)


def test_descent_returns_just_the_start_when_objective_cannot_be_evaluated_there():
    # wrong arity for the start point -> the objective call itself raises
    path = gradient_descent_path(x**2 + y**2, [x, y], (1.0,), direction="minimize")
    assert path == [(1.0,)]


def test_descent_stops_when_gradient_evaluation_fails(monkeypatch):
    def bad_grad(*p):
        raise ValueError("gradient undefined here")
    monkeypatch.setattr(ou.sp, "lambdify", _fake_lambdify(lambda *p: 1.0, bad_grad))
    path = gradient_descent_path(x**2, [x], (2.0,), direction="minimize")
    assert path == [(2.0,)]


def test_descent_stops_when_candidate_evaluation_fails(monkeypatch):
    calls = {"n": 0}

    def f(*p):
        calls["n"] += 1
        if calls["n"] > 1:                      # fine at the start point, raises at the first candidate
            raise ValueError("domain error")
        return 4.0
    monkeypatch.setattr(ou.sp, "lambdify", _fake_lambdify(f, lambda *p: [1.0]))
    path = gradient_descent_path(x**2, [x], (2.0,), direction="minimize")
    assert path == [(2.0,)]


def test_descent_gives_up_when_backtracking_never_finds_a_finite_value(monkeypatch):
    calls = {"n": 0}

    def f(*p):
        calls["n"] += 1
        if calls["n"] == 1:
            return 4.0                          # start point
        if calls["n"] == 2:
            return 100.0                        # first candidate: worse -> enter backtracking
        raise ValueError("domain error")        # every backtracked step raises -> NaN each time
    monkeypatch.setattr(ou.sp, "lambdify", _fake_lambdify(f, lambda *p: [1.0]))
    path = gradient_descent_path(x**2, [x], (2.0,), direction="minimize")
    assert path == [(2.0,)]                     # never accepted a non-finite point
    assert calls["n"] == 2 + 20                 # backtracking is capped at 20 attempts


# ---------------------------------------------------------------- constraints must never be silently dropped

def _model_with(objective, over, equations, direction="minimize"):
    import re
    names = sorted(set(re.findall(r"[A-Za-z_]\w*", " ".join([objective, *equations]))) - {"Eq"})
    return build_model({
        "problem_domain": "t", "problem_type": "algebraic",
        "variables": [{"symbol": n, "meaning": n, "known_value": None, "unit": None} for n in names],
        "equations": [{"name": f"c{i}", "kind": "equation", "expression": e, "derivation": ""}
                      for i, e in enumerate(equations)],
        "objective": {"expression": objective, "direction": direction, "optimize_over": over},
        "solve_for": over, "assumptions": [],
    })


def test_second_constraint_is_rewritten_after_the_first_elimination():
    """Regression: minimize x^2+y^2+z^2 s.t. x+y+z=3 and x=y. After x was
    eliminated via the first constraint, the second (x = y) still mentioned
    the ELIMINATED x, so solving it for y re-introduced x into the objective
    and the answer came back circular ({'x': x/2 - y + 3/2, 'y': ...}).
    The second constraint must be rewritten without x first. True optimum
    is x = y = z = 1."""
    result = solve_optimization(_model_with(
        "x**2 + y**2 + z**2", ["x", "y", "z"], ["Eq(x + y + z, 3)", "Eq(x, y)"]))
    assert result.error is None
    assert len(result.critical_points) == 1
    assert {k: float(v) for k, v in result.critical_points[0].items()} == {"x": 1.0, "y": 1.0, "z": 1.0}
    assert result.classifications == ["minimum"]
    # no variable's value may still be expressed in terms of another symbol
    assert all(not v.free_symbols for v in result.critical_points[0].values())


def test_redundant_constraint_is_dropped_not_treated_as_unresolved():
    """x + y = 10 and 2x + 2y = 20 say the same thing. Once the first is
    substituted into the second it reduces to 0 -- redundant, harmless --
    so plain elimination must still succeed (no pointless Lagrange fallback)."""
    result = solve_optimization(_model_with(
        "x*y", ["x", "y"], ["Eq(x + y, 10)", "Eq(2*x + 2*y, 20)"], direction="maximize"))
    assert result.error is None and result.used_lagrange is False
    assert {k: float(v) for k, v in result.critical_points[0].items()} == {"x": 5.0, "y": 5.0}
    assert result.classifications == ["maximum"]


def test_inconsistent_constraints_are_not_papered_over():
    """x = 3 and x = 4 can't both hold. After substituting one into the
    other the leftover is the nonzero constant -1 -- that must NOT be
    mistaken for redundancy; the problem falls to Lagrange, which reports
    that no solution exists."""
    result = solve_optimization(_model_with("x**2 + y**2", ["x", "y"], ["Eq(x, 3)", "Eq(x, 4)"]))
    assert result.used_lagrange is True
    assert result.critical_points == []
    assert result.error is not None


def test_constraint_that_cannot_be_isolated_falls_back_to_lagrange_not_a_wrong_answer(monkeypatch):
    """Regression: when sympy can't isolate any variable from an equation
    constraint, that constraint was silently dropped, and the unconstrained
    objective's critical point was reported as success -- maximizing x*y s.t.
    x + y = 10 returned (x, y) = (0, 0), which violates the constraint.
    Elimination is forced to fail here (the real-world trigger is a
    constraint like x^5 + x + y^5 + y = 10, which sympy can't solve for
    either variable) and the Lagrange fallback must still honour it."""
    real = ou.run_with_timeout

    def no_elimination(func, *args, **kwargs):
        if kwargs.get("label") == "constraint elimination":
            return []
        return real(func, *args, **kwargs)
    monkeypatch.setattr(ou, "run_with_timeout", no_elimination)

    result = solve_optimization(_model_with("x*y", ["x", "y"], ["Eq(x + y, 10)"], direction="maximize"))
    assert result.error is None
    assert result.used_lagrange is True
    assert len(result.critical_points) == 1
    assert {k: float(v) for k, v in result.critical_points[0].items()} == {"x": 5.0, "y": 5.0}   # NOT (0, 0)


def test_genuinely_unisolable_constraint_reports_an_error_instead_of_a_wrong_point(monkeypatch):
    """The real trigger (x^5 + x + y^5 + y = 10: sympy returns [] for both
    variables). The Lagrange system is itself too hard, which is reported
    honestly -- the old behaviour was a SUCCESSFUL-looking (0, 0). The slow
    Lagrange solve is short-circuited so the test doesn't wait the full
    computation timeout."""
    assert sp.solve(x**5 + x + y**5 + y - 10, x) == []        # premise: unisolable
    real = ou.run_with_timeout

    def instant_timeout(func, *args, **kwargs):
        if kwargs.get("label") == "Lagrange system solve":
            raise ComputationTimeoutError(0.1, "Lagrange system solve")
        return real(func, *args, **kwargs)
    monkeypatch.setattr(ou, "run_with_timeout", instant_timeout)

    result = solve_optimization(_model_with("x*y", ["x", "y"], ["Eq(x**5 + x + y**5 + y, 10)"],
                                            direction="maximize"))
    assert result.used_lagrange is True
    assert result.critical_points == []
    assert "Timed out solving the Lagrange system" in result.error


def test_fully_pinned_problem_is_solved_via_lagrange_not_reported_as_having_no_critical_points():
    """Regression: maximize x + y s.t. x = 3 and y = 4 has exactly one
    feasible point. It used to report 'No real-valued critical points'."""
    result = solve_optimization(_model_with("x + y", ["x", "y"], ["Eq(x, 3)", "Eq(y, 4)"], direction="maximize"))
    assert result.error is None and result.used_lagrange is True
    assert {k: float(v) for k, v in result.critical_points[0].items()} == {"x": 3.0, "y": 4.0}


def test_is_identically_zero_cases():
    assert ou._is_identically_zero(x - x) is True                       # expands to 0
    assert ou._is_identically_zero(sp.Float(1e-17)) is True             # float noise
    assert ou._is_identically_zero(sp.Integer(-1)) is False             # nonzero constant
    assert ou._is_identically_zero((x**2 - 1) / (x - 1) - x - 1) is True    # needs simplify, not just expand
    assert ou._is_identically_zero(x - 1) is False                      # genuinely depends on x


def test_is_identically_zero_is_conservative_when_simplify_fails(monkeypatch):
    def boom(*a, **k):
        raise ComputationTimeoutError(0.1, "constraint redundancy check")
    monkeypatch.setattr(ou, "run_with_timeout", boom)
    assert ou._is_identically_zero((x**2 - 1) / (x - 1) - x - 1) is False   # "can't prove zero" == nonzero


def test_is_identically_zero_non_numeric_constant_is_not_zero():
    # a constant that can't be converted to float (e.g. zoo) must not crash or count as zero
    assert ou._is_identically_zero(sp.zoo) is False
