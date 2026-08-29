import sympy as sp

from modules.algebra_rules import classify_isolation
from modules.equation_engine import build_model
from modules.solver import compute_steps


def test_linear_equation_classified():
    x = sp.Symbol("x")
    result = classify_isolation(sp.Eq(2 * x + 3, 11), x)
    assert "linear" in result.lower()


def test_quadratic_equation_classified():
    x = sp.Symbol("x")
    result = classify_isolation(sp.Eq(x ** 2, 25), x)
    assert "quadratic" in result.lower()


def test_higher_degree_polynomial_classified():
    x = sp.Symbol("x")
    result = classify_isolation(sp.Eq(x ** 3 - 2 * x, 5), x)
    assert "degree-3" in result.lower() or "polynomial" in result.lower()


def test_root_classified():
    x = sp.Symbol("x")
    result = classify_isolation(sp.Eq(sp.sqrt(x), 5), x)
    assert "root" in result.lower()


def test_reciprocal_classified():
    x = sp.Symbol("x")
    result = classify_isolation(sp.Eq(5 / x, 2), x)
    assert "denominator" in result.lower()


def test_logarithm_classified():
    x = sp.Symbol("x")
    result = classify_isolation(sp.Eq(sp.log(x), 2), x)
    assert "log" in result.lower()


def test_exponential_classified():
    x = sp.Symbol("x")
    result = classify_isolation(sp.Eq(sp.exp(x), 10), x)
    assert "exp" in result.lower()


def test_trig_classified():
    x = sp.Symbol("x")
    result = classify_isolation(sp.Eq(sp.sin(x), sp.Rational(1, 2)), x)
    assert "sin" in result.lower()


def test_inverse_trig_classified():
    x = sp.Symbol("x")
    result = classify_isolation(sp.Eq(sp.asin(x), 1), x)
    assert "asin" in result.lower()


def test_both_sides_classified():
    x = sp.Symbol("x")
    result = classify_isolation(sp.Eq(2 * x, x + 5), x)
    assert "both sides" in result.lower()


def test_target_absent_handled_gracefully():
    x, y = sp.symbols("x y")
    result = classify_isolation(sp.Eq(y, y + 1), x)
    # y == y+1 auto-evaluates to a trivial False -- x doesn't appear at all
    assert isinstance(result, str) and len(result) > 0


def test_trivial_identity_does_not_crash():
    """sp.Eq() of two identical expressions auto-evaluates straight to
    BooleanTrue rather than staying an Eq object -- must not crash."""
    x = sp.Symbol("x")
    result = classify_isolation(sp.Eq(x, x), x)
    assert isinstance(result, str) and len(result) > 0


def test_never_raises_on_arbitrary_equations():
    x, y, z = sp.symbols("x y z")
    equations = [
        sp.Eq(x * sp.sin(x) + y, z),
        sp.Eq(x / (x + 1), 0.5),
        sp.Eq(sp.Abs(x), 3),
        sp.Eq(x ** x, 4),
    ]
    for eq in equations:
        result = classify_isolation(eq, x)
        assert isinstance(result, str) and len(result) > 0


# ---------------------------------------------------------------- wired into solver.py


def test_solver_includes_technique_step():
    model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "time", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })
    steps = compute_steps(model)
    descriptions = [s.description for s in steps["a"]]
    assert "Technique" in descriptions
    technique_step = next(s for s in steps["a"] if s.description == "Technique")
    assert "linear" in technique_step.expression.lower()


def test_solver_technique_step_for_quadratic_target():
    model = build_model({
        "problem_domain": "projectile motion", "problem_type": "algebraic",
        "variables": [
            {"symbol": "t", "meaning": "time", "known_value": None, "unit": "s"},
        ],
        "equations": [
            {"name": "height", "kind": "equation", "expression": "Eq(-4.9*t**2 + 20*t + 1.5, 0)",
             "derivation": ""},
        ],
        "solve_for": ["t"], "assumptions": [],
    })
    steps = compute_steps(model)
    technique_step = next(s for s in steps["t"] if s.description == "Technique")
    assert "quadratic" in technique_step.expression.lower()
