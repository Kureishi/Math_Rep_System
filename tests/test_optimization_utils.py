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


def test_eliminate_greedy_single_variable_skips_elimination_entirely():
    """With only one eligible variable, the elimination loop never runs (a
    single free variable doesn't need eliminating) -- the objective and
    variable are returned unchanged, still as the one free_var."""
    reduced, eliminated, free_vars = _eliminate_greedy(x, ["x"], [x - 3])
    assert reduced == x and eliminated == {} and free_vars == [x]


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
    """x + y with both x=3 and y=4 pinned by separate constraints: the loop
    eliminates one variable per pass and stops as soon as a single free
    variable remains (it doesn't need eliminating), so exactly one of the
    two constraints ends up used."""
    reduced, eliminated, free_vars = _eliminate_greedy(x + y, ["x", "y"], [x - 3, y - 4])
    assert len(eliminated) == 1 and len(free_vars) == 1
    remaining_var = free_vars[0]
    assert reduced == remaining_var + (3 if remaining_var == y else 4)


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
