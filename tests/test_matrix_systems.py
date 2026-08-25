import sympy as sp

from modules.equation_engine import build_model
from modules.matrix_utils import build_linear_system, analyze_linear_system, linear_system_view
from modules.verifier import verify, _known_substitutions
from modules.solver import compute_steps


def _model(equations, solve_for, variables=None):
    variables = variables or []
    return build_model({
        "problem_domain": "linear system", "problem_type": "algebraic",
        "variables": variables,
        "equations": equations,
        "solve_for": solve_for, "assumptions": [],
    })


def test_unique_solution_square_system():
    model = _model(
        equations=[
            {"name": "eq1", "kind": "equation", "expression": "Eq(2*x + 3*y, 8)", "derivation": ""},
            {"name": "eq2", "kind": "equation", "expression": "Eq(x - y, 1)", "derivation": ""},
        ],
        solve_for=["x", "y"],
        variables=[
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
    )
    result = linear_system_view(model, {})
    assert result is not None
    assert result.is_square
    assert result.consistent and result.unique
    assert result.determinant == -5
    assert result.solution["x"] == sp.Rational(11, 5)
    assert result.solution["y"] == sp.Rational(6, 5)


def test_eigenvalues_reported_for_square_system():
    model = _model(
        equations=[
            {"name": "eq1", "kind": "equation", "expression": "Eq(2*x + y, 0)", "derivation": ""},
            {"name": "eq2", "kind": "equation", "expression": "Eq(x + 3*y, 0)", "derivation": ""},
        ],
        solve_for=["x", "y"],
        variables=[
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
    )
    result = linear_system_view(model, {})
    assert result.eigenvalues is not None
    # A = [[2,1],[1,3]] -> eigenvalues 5/2 +- sqrt(5)/2
    expected = {sp.Rational(5, 2) - sp.sqrt(5) / 2, sp.Rational(5, 2) + sp.sqrt(5) / 2}
    assert set(result.eigenvalues.keys()) == expected


def test_inconsistent_system_detected():
    model = _model(
        equations=[
            {"name": "eq1", "kind": "equation", "expression": "Eq(x + y, 2)", "derivation": ""},
            {"name": "eq2", "kind": "equation", "expression": "Eq(2*x + 2*y, 5)", "derivation": ""},
        ],
        solve_for=["x", "y"],
        variables=[
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
    )
    result = linear_system_view(model, {})
    assert not result.consistent
    assert result.solution is None
    assert "inconsistent" in result.classification.lower()


def test_underdetermined_system_detected():
    model = _model(
        equations=[
            {"name": "eq1", "kind": "equation", "expression": "Eq(x + y, 2)", "derivation": ""},
            {"name": "eq2", "kind": "equation", "expression": "Eq(2*x + 2*y, 4)", "derivation": ""},
        ],
        solve_for=["x", "y"],
        variables=[
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
    )
    result = linear_system_view(model, {})
    assert result.consistent and not result.unique
    assert "infinitely many" in result.classification.lower()


def test_overdetermined_consistent_rectangular_system():
    A = sp.Matrix([[1, 1], [1, -1], [2, 0]])
    b = sp.Matrix([3, 1, 4])
    result = analyze_linear_system(A, b, ["x", "y"])
    assert not result.is_square
    assert result.determinant is None
    assert result.eigenvalues is None
    assert result.consistent and result.unique
    assert result.solution == {"x": 2, "y": 1}


def test_single_equation_is_not_treated_as_a_system():
    # only one equation -- should stay on the ordinary scalar solve path,
    # not get promoted to a "matrix system" (build_linear_system requires >=2)
    model = _model(
        equations=[{"name": "eq1", "kind": "equation", "expression": "Eq(x, 4)", "derivation": ""}],
        solve_for=["x"],
        variables=[{"symbol": "x", "meaning": "x", "known_value": None, "unit": None}],
    )
    assert linear_system_view(model, {}) is None


def test_nonlinear_system_is_not_treated_as_a_linear_system():
    model = _model(
        equations=[
            {"name": "eq1", "kind": "equation", "expression": "Eq(x**2 + y, 4)", "derivation": ""},
            {"name": "eq2", "kind": "equation", "expression": "Eq(x - y, 1)", "derivation": ""},
        ],
        solve_for=["x", "y"],
        variables=[
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
    )
    assert linear_system_view(model, {}) is None


def test_known_values_are_substituted_before_building_the_matrix():
    # z is known, so the "system" is really just 2 equations in x, y --
    # z should not appear as a spurious third column
    model = _model(
        equations=[
            {"name": "eq1", "kind": "equation", "expression": "Eq(x + y + z, 10)", "derivation": ""},
            {"name": "eq2", "kind": "equation", "expression": "Eq(x - y, 2)", "derivation": ""},
        ],
        solve_for=["x", "y"],
        variables=[
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
            {"symbol": "z", "meaning": "z", "known_value": "4", "unit": None},
        ],
    )
    knowns = _known_substitutions(model)
    result = linear_system_view(model, knowns)
    assert result.symbols == ["x", "y"]
    assert result.solution["x"] == 4
    assert result.solution["y"] == 2


def test_verifier_flags_inconsistent_linear_system():
    model = _model(
        equations=[
            {"name": "eq1", "kind": "equation", "expression": "Eq(x + y, 2)", "derivation": ""},
            {"name": "eq2", "kind": "equation", "expression": "Eq(2*x + 2*y, 5)", "derivation": ""},
        ],
        solve_for=["x", "y"],
        variables=[
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
    )

    class _Client:
        def chat(self, **kwargs):
            return "{}"

    report = verify(model, _Client(), "an inconsistent system")
    matrix_checks = [c for c in report.checks if c.label == "Linear system consistency"]
    assert len(matrix_checks) == 1
    assert matrix_checks[0].passed is False
    assert not report.passed


def test_solver_shows_matrix_representation_step():
    model = _model(
        equations=[
            {"name": "eq1", "kind": "equation", "expression": "Eq(2*x + 3*y, 8)", "derivation": ""},
            {"name": "eq2", "kind": "equation", "expression": "Eq(x - y, 1)", "derivation": ""},
        ],
        solve_for=["x", "y"],
        variables=[
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
    )
    steps = compute_steps(model)
    descriptions = [s.description for s in steps["x"]]
    assert "Represent the system as A x = b" in descriptions
    assert "Determinant of the coefficient matrix" in descriptions
    assert "Classification" in descriptions
