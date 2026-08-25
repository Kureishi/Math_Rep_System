import json

from modules.equation_engine import build_model, target_kind
from modules.optimization_utils import solve_optimization
from modules.verifier import verify
from modules.solver import compute_steps


def test_objective_parses_and_dispatches(optimization_unconstrained_json):
    model = build_model(json.loads(optimization_unconstrained_json))
    assert model.objective is not None
    assert model.objective.parse_error is None
    assert model.objective.direction == "minimize"
    assert target_kind(model, "x") == "optimization"


def test_unconstrained_single_variable(optimization_unconstrained_json):
    model = build_model(json.loads(optimization_unconstrained_json))
    result = solve_optimization(model)
    assert result.error is None
    assert result.critical_points[0]["x"] == 2
    assert result.classifications[0] == "minimum"


def test_unconstrained_multivariable_hessian_classification():
    model = build_model({
        "problem_domain": "min paraboloid", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
        "equations": [],
        "objective": {"expression": "x**2 + y**2 - 4*x - 6*y + 13", "direction": "minimize",
                       "optimize_over": ["x", "y"]},
        "solve_for": ["x", "y"], "assumptions": [],
    })
    result = solve_optimization(model)
    assert result.critical_points[0]["x"] == 2
    assert result.critical_points[0]["y"] == 3
    assert result.classifications[0] == "minimum"


def test_constrained_via_elimination(optimization_constrained_json):
    model = build_model(json.loads(optimization_constrained_json))
    result = solve_optimization(model)
    assert result.error is None
    assert not result.used_lagrange
    assert "h" in result.eliminated_vars
    assert len(result.critical_points) == 1  # complex roots filtered out
    assert result.classifications[0] == "minimum"


def test_elimination_is_deterministic_across_runs(optimization_constrained_json):
    """Regression test for a real bug: eliminating from an unordered set
    could nondeterministically pick the wrong variable to eliminate
    (sometimes solving the constraint for the REQUESTED variable instead
    of the helper), giving different -- and sometimes entirely complex --
    results on different runs."""
    model = build_model(json.loads(optimization_constrained_json))
    results = [solve_optimization(model) for _ in range(10)]
    first = results[0].critical_points
    for r in results[1:]:
        assert r.critical_points == first
        assert r.eliminated_vars.keys() == results[0].eliminated_vars.keys()


def test_lagrange_used_when_both_constrained_variables_requested():
    """Regression test for a real bug: when optimize_over already 'covers'
    every objective variable (e.g. both x and y requested, joined by a
    constraint), the elimination path used to never even look at the
    constraint, silently dropping it and solving the wrong (unconstrained)
    problem. This confirms the constraint is now respected either way."""
    model = build_model({
        "problem_domain": "max product", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
        "equations": [{"name": "sum constraint", "kind": "equation", "expression": "Eq(x + y, 10)", "derivation": "x"}],
        "objective": {"expression": "x*y", "direction": "maximize", "optimize_over": ["x", "y"]},
        "solve_for": ["x", "y"], "assumptions": [],
    })
    result = solve_optimization(model)
    assert result.error is None
    pt = result.critical_points[0]
    assert "x" in pt and "y" in pt
    assert float(pt["x"]) == 5.0
    assert float(pt["y"]) == 5.0


def test_feasibility_note_when_optimum_violates_inequality():
    model = build_model({
        "problem_domain": "constrained min", "problem_type": "algebraic",
        "variables": [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None}],
        "equations": [{"name": "must be large", "kind": "inequality", "expression": "x >= 5", "derivation": "x"}],
        "objective": {"expression": "x**2 - 4*x + 7", "direction": "minimize", "optimize_over": ["x"]},
        "solve_for": ["x"], "assumptions": [],
    })
    result = solve_optimization(model)
    # unconstrained minimum is at x=2, which violates x>=5
    assert result.critical_points[0]["x"] == 2
    assert len(result.feasibility_notes) == 1
    assert "violates" in result.feasibility_notes[0]


def test_full_verify_passes_for_unconstrained(optimization_unconstrained_json, fake_client_factory):
    model = build_model(json.loads(optimization_unconstrained_json))
    report = verify(model, fake_client_factory(), "x")
    assert report.passed
    grad_check = next(c for c in report.checks if "Critical point verified" in c.label)
    assert grad_check.passed
    class_check = next(c for c in report.checks if "Classification matches" in c.label)
    assert class_check.passed


def test_target_variable_present_not_falsely_flagged_for_objective_only_problem(
        optimization_unconstrained_json, fake_client_factory):
    """Regression test for a real bug: the structural 'target variable
    present' check only looked at equation-kind relations, so an
    optimization problem with NO equations (objective only) always failed
    this check even though x genuinely does appear, in the objective."""
    model = build_model(json.loads(optimization_unconstrained_json))
    report = verify(model, fake_client_factory(), "x")
    presence_check = next(c for c in report.checks if "Target variable present" in c.label)
    assert presence_check.passed


def test_determinacy_not_falsely_flagged_for_constrained_optimization(
        optimization_constrained_json, fake_client_factory):
    """Regression test: the determinacy check used to count 'r' as an
    unmatched unknown needing its own equation, even though optimization
    targets are resolved via calculus, not equation-counting."""
    model = build_model(json.loads(optimization_constrained_json))
    report = verify(model, fake_client_factory(), "x")
    determinacy_check = next(c for c in report.checks if "Determinacy" in c.label)
    assert determinacy_check.passed


def test_optimization_steps_generated(optimization_constrained_json):
    model = build_model(json.loads(optimization_constrained_json))
    steps = compute_steps(model)
    assert "r" in steps
    assert len(steps["r"]) > 0
    descriptions = " ".join(s.description for s in steps["r"])
    assert "eliminate" in descriptions.lower()
