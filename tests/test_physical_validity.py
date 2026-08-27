import sympy as sp

from modules.equation_engine import build_model
from modules.physical_validity import filter_physically_valid
from modules.solver import compute_steps


def _projectile_model(with_domain=True):
    domain = {"domain": "nonnegative"} if with_domain else {}
    return build_model({
        "problem_domain": "projectile motion", "problem_type": "algebraic",
        "variables": [
            {"symbol": "t", "meaning": "time since launch", "known_value": None, "unit": "s", **domain},
        ],
        "equations": [
            {"name": "height", "kind": "equation", "expression": "Eq(-4.9*t**2 + 20*t + 1.5, 0)",
             "derivation": ""},
        ],
        "solve_for": ["t"], "assumptions": [],
    })


# ---------------------------------------------------------------- filter_physically_valid


def test_filters_out_negative_root_when_domain_declared():
    model = _projectile_model(with_domain=True)
    t = sp.Symbol("t")
    eq = model.equations[0].sympy_eq
    solutions = sp.solve([eq], [t], dict=True)
    assert len(solutions) == 2  # sanity: this problem genuinely has two roots

    result = filter_physically_valid(model, solutions, [t])
    assert result.checked_any_domain is True
    assert len(result.valid) == 1
    assert float(result.valid[0][t]) > 0
    assert len(result.discarded) == 1
    assert "nonnegative" in result.discarded[0][1][0] or "\u2265 0" in result.discarded[0][1][0]


def test_no_filtering_when_no_domain_declared():
    model = _projectile_model(with_domain=False)
    t = sp.Symbol("t")
    eq = model.equations[0].sympy_eq
    solutions = sp.solve([eq], [t], dict=True)

    result = filter_physically_valid(model, solutions, [t])
    assert result.checked_any_domain is False
    assert result.valid == solutions
    assert result.discarded == []


def test_empty_solutions_handled():
    model = _projectile_model(with_domain=True)
    t = sp.Symbol("t")
    result = filter_physically_valid(model, [], [t])
    assert result.valid == []
    assert result.discarded == []


def test_all_solutions_pass_when_none_violate():
    x = sp.Symbol("x")
    model = build_model({
        "problem_domain": "test", "problem_type": "algebraic",
        "variables": [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None,
                        "domain": "positive"}],
        "equations": [{"name": "eq", "kind": "equation", "expression": "Eq(x**2, 25)", "derivation": ""}],
        "solve_for": ["x"], "assumptions": [],
    })
    solutions = [{x: sp.Integer(5)}]  # only the positive root, already
    result = filter_physically_valid(model, solutions, [x])
    assert result.valid == solutions
    assert result.discarded == []


def test_non_numeric_solution_value_not_filtered():
    # a solution that's still symbolic (e.g. in terms of another unknown)
    # shouldn't be discarded just because it's not a plain number
    x, y = sp.symbols("x y")
    model = build_model({
        "problem_domain": "test", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None, "domain": "positive"},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
        "equations": [{"name": "eq", "kind": "equation", "expression": "Eq(x, y)", "derivation": ""}],
        "solve_for": ["x"], "assumptions": [],
    })
    solutions = [{x: y}]  # x expressed symbolically in terms of y, not a number
    result = filter_physically_valid(model, solutions, [x])
    assert result.valid == solutions
    assert result.discarded == []


# ---------------------------------------------------------------- wired into solver.py


def test_solver_picks_physically_valid_root_when_domain_declared():
    model = _projectile_model(with_domain=True)
    steps = compute_steps(model)
    descriptions = [s.description for s in steps["t"]]
    assert "Discard a non-physical root" in descriptions

    numeric_step = next(s for s in steps["t"] if s.description == "Numeric result")
    value = float(numeric_step.expression.split("=")[1].strip())
    assert value > 0  # the positive root, not the negative one


def test_solver_keeps_old_behavior_when_no_domain_declared():
    """Regression/backward-compatibility check: without an explicit
    domain, behavior is unchanged from before this feature existed --
    still takes sp.solve()'s first branch, whatever sign it has."""
    model = _projectile_model(with_domain=False)
    steps = compute_steps(model)
    descriptions = [s.description for s in steps["t"]]
    assert "Discard a non-physical root" not in descriptions

    numeric_step = next(s for s in steps["t"] if s.description == "Numeric result")
    value = float(numeric_step.expression.split("=")[1].strip())
    assert value < 0  # sp.solve()'s first (negative) branch, same as before


def test_solver_single_root_problem_unaffected():
    """A problem with only one algebraic root shouldn't get any
    filtering-related steps at all -- nothing to filter."""
    model = build_model({
        "problem_domain": "algebra", "problem_type": "algebraic",
        "variables": [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None,
                        "domain": "positive"}],
        "equations": [{"name": "eq", "kind": "equation", "expression": "Eq(2*x + 3, 11)", "derivation": ""}],
        "solve_for": ["x"], "assumptions": [],
    })
    steps = compute_steps(model)
    descriptions = [s.description for s in steps["x"]]
    assert "Discard a non-physical root" not in descriptions
    assert "Multiple physically valid solutions remain" not in descriptions
