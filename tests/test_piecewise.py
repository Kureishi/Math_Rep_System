import json
import sympy as sp

from modules.equation_engine import build_model
from modules.verifier import verify


def test_piecewise_parses(piecewise_json):
    model = build_model(json.loads(piecewise_json))
    eq = model.equations[0]
    assert eq.parse_error is None
    assert eq.kind == "equation"  # no separate kind needed


def test_piecewise_selects_correct_branch(piecewise_json):
    model = build_model(json.loads(piecewise_json))
    eq = model.equations[0]
    x = sp.Symbol("x")
    below_threshold = float(eq.sympy_eq.rhs.subs(x, 5000))
    above_threshold = float(eq.sympy_eq.rhs.subs(x, 15000))
    assert below_threshold == 500.0  # 0.1 * 5000
    assert above_threshold == 2000.0  # 0.2 * 15000 - 1000


def test_piecewise_solves_to_correct_value(piecewise_json, fake_client_factory):
    model = build_model(json.loads(piecewise_json))
    report = verify(model, fake_client_factory(), "x")
    # tax is the unknown being solved for (x=15000 is known), so the
    # numeric-balance check correctly doesn't apply here -- instead check
    # that solving the equation gives the right branch's value directly
    assert report.sympy_numeric_answers.get("tax") == 2000.0


def test_piecewise_dimensional_check_passes_when_branches_agree():
    model = build_model({
        "problem_domain": "tiered distance", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "input", "known_value": "15", "unit": "m"},
            {"symbol": "y", "meaning": "output", "known_value": None, "unit": "m"},
        ],
        "equations": [{"name": "tiered", "kind": "equation",
                       "expression": "Eq(y, Piecewise((2*x, x <= 10), (3*x, True)))", "derivation": "x"}],
        "solve_for": ["y"], "assumptions": [],
    })

    class FakeClient:
        def chat(self, **kw):
            return "[]"

    report = verify(model, FakeClient(), "x")
    dim_check = next(c for c in report.checks if "Dimensional consistency" in c.label)
    assert dim_check.passed


def test_piecewise_dimensional_check_catches_branch_mismatch():
    """This is a real bug caught during development: sympy's own dimension
    machinery silently reports Dimension(1) for a Piecewise as a whole
    regardless of its actual contents, rather than raising or computing
    correctly -- per-branch checking is required."""
    model = build_model({
        "problem_domain": "tiered distance (broken)", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "input", "known_value": "15", "unit": "m"},
            {"symbol": "y", "meaning": "output", "known_value": None, "unit": "m"},
        ],
        "equations": [{"name": "broken tiered", "kind": "equation",
                       "expression": "Eq(y, Piecewise((2*x, x <= 10), (x**2, True)))", "derivation": "x"}],
        "solve_for": ["y"], "assumptions": [],
    })

    class FakeClient:
        def chat(self, **kw):
            return "[]"

    report = verify(model, FakeClient(), "x")
    dim_check = next(c for c in report.checks if "Dimensional consistency" in c.label)
    assert not dim_check.passed
    assert "disagree" in dim_check.detail.lower()
