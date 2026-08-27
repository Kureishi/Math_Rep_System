import sympy as sp

from modules.domain_utils import (
    find_domain_restrictions, domain_restrictions_for_equation, evaluate_restriction,
)
from modules.equation_engine import build_model
from modules.verifier import verify


class _FakeClient:
    def chat(self, **kwargs):
        return "{}"


# ---------------------------------------------------------------- find_domain_restrictions


def test_division_restriction_detected():
    t, a = sp.symbols("t a")
    expr = a - 10 / t
    restrictions = find_domain_restrictions(expr)
    assert len(restrictions) == 1
    assert restrictions[0].kind == "nonzero"
    assert restrictions[0].condition == sp.Ne(t, 0)


def test_even_root_restriction_detected():
    x = sp.Symbol("x")
    expr = sp.sqrt(x - 4)
    restrictions = find_domain_restrictions(expr)
    assert len(restrictions) == 1
    assert restrictions[0].kind == "nonneg"
    assert restrictions[0].condition == (x - 4 >= 0)


def test_odd_root_has_no_restriction():
    # cube roots are defined for all reals -- shouldn't be flagged
    x = sp.Symbol("x")
    expr = x ** sp.Rational(1, 3)
    restrictions = find_domain_restrictions(expr)
    assert restrictions == []


def test_log_restriction_detected():
    x, y = sp.symbols("x y")
    expr = sp.log(x * y)
    restrictions = find_domain_restrictions(expr)
    assert len(restrictions) == 1
    assert restrictions[0].kind == "positive"


def test_inverse_trig_restriction_detected():
    x = sp.Symbol("x")
    expr = sp.asin(x / 5)
    restrictions = find_domain_restrictions(expr)
    assert len(restrictions) == 1
    assert restrictions[0].kind == "range"


def test_no_restrictions_for_polynomial():
    x = sp.Symbol("x")
    expr = 3 * x ** 2 + 2 * x - 1
    assert find_domain_restrictions(expr) == []


def test_restrictions_deduplicated_across_repeated_subexpression():
    t = sp.Symbol("t")
    expr = 10 / t + 20 / t  # same denominator twice
    restrictions = find_domain_restrictions(expr)
    assert len(restrictions) == 1


def test_numeric_denominator_not_flagged():
    x = sp.Symbol("x")
    expr = x / 5  # constant denominator, always fine
    assert find_domain_restrictions(expr) == []


def test_domain_restrictions_for_equation_combines_both_sides():
    x, t = sp.symbols("x t")
    eq = sp.Eq(sp.sqrt(x), 10 / t)
    restrictions = domain_restrictions_for_equation(eq)
    kinds = {r.kind for r in restrictions}
    assert kinds == {"nonneg", "nonzero"}


# ---------------------------------------------------------------- evaluate_restriction


def test_evaluate_restriction_true_and_false():
    t = sp.Symbol("t")
    restriction = find_domain_restrictions(10 / t)[0]
    assert evaluate_restriction(restriction, {t: 5.0}) is True
    assert evaluate_restriction(restriction, {t: 0.0}) is False


def test_evaluate_restriction_none_when_symbol_unknown():
    t, x = sp.symbols("t x")
    restriction = find_domain_restrictions(10 / t)[0]
    assert evaluate_restriction(restriction, {x: 1.0}) is None


# ---------------------------------------------------------------- wired into verifier.py


def test_verifier_flags_active_domain_violation():
    model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "time", "known_value": "0", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })
    report = verify(model, _FakeClient(), "problem with t=0")
    domain_checks = [c for c in report.checks if c.label.startswith("Domain validity")]
    assert len(domain_checks) == 1
    assert domain_checks[0].passed is False
    assert not report.passed

    assert len(report.domain_notes) == 1
    assert len(report.domain_notes[0].violated) == 1


def test_verifier_domain_check_passes_when_satisfied():
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
    report = verify(model, _FakeClient(), "normal kinematics problem")
    domain_checks = [c for c in report.checks if c.label.startswith("Domain validity")]
    assert len(domain_checks) == 1
    assert domain_checks[0].passed is True
    assert report.domain_notes[0].violated == []
    assert len(report.domain_notes[0].satisfied) == 1


def test_verifier_no_domain_checks_for_equation_without_restrictions():
    model = build_model({
        "problem_domain": "algebra", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
        ],
        "equations": [
            {"name": "eq1", "kind": "equation", "expression": "Eq(2*x + 3, 11)", "derivation": ""},
        ],
        "solve_for": ["x"], "assumptions": [],
    })
    report = verify(model, _FakeClient(), "plain algebra")
    domain_checks = [c for c in report.checks if c.label.startswith("Domain validity")]
    assert domain_checks == []
    assert report.domain_notes == []
