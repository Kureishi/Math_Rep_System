import json
import sympy as sp

from modules.equation_engine import build_model, target_kind
from modules.recurrence_utils import solve_recurrence, verify_recurrence_solution
from modules.verifier import verify
from modules.solver import compute_steps


def test_recurrence_parses_and_dispatches(recurrence_json):
    model = build_model(json.loads(recurrence_json))
    eq = model.equations[0]
    assert eq.kind == "recurrence"
    assert eq.parse_error is None
    assert target_kind(model, "a") == "recurrence"


def test_recurrence_solves_correctly(recurrence_json):
    model = build_model(json.loads(recurrence_json))
    solutions = solve_recurrence(model)
    n = sp.Symbol("n")
    assert solutions["a"].equals(500 * n + 1000)


def test_second_order_recurrence_fibonacci():
    model = build_model({
        "problem_domain": "fibonacci", "problem_type": "recurrence", "independent_variable": "n",
        "variables": [
            {"symbol": "f", "meaning": "term", "known_value": None, "unit": None, "is_function": True},
            {"symbol": "n", "meaning": "index", "known_value": None, "unit": None, "is_function": False},
        ],
        "equations": [{"name": "fib", "kind": "recurrence", "expression": "Eq(f(n+2), f(n+1) + f(n))", "derivation": "x"}],
        "initial_conditions": [{"expression": "f(0)", "value": "0"}, {"expression": "f(1)", "value": "1"}],
        "solve_for": ["f"], "assumptions": [],
    })
    solutions = solve_recurrence(model)
    n = sp.Symbol("n")
    vals = [complex(sp.N(solutions["f"].subs(n, k))).real for k in range(6)]
    assert [round(v) for v in vals] == [0, 1, 1, 2, 3, 5]


def test_recurrence_verification_symbolic_pass(recurrence_json):
    model = build_model(json.loads(recurrence_json))
    solutions = solve_recurrence(model)
    ok, residual = verify_recurrence_solution(model.equations[0].sympy_eq, "a", solutions["a"], sp.Symbol("n"))
    assert ok
    assert residual == 0


def test_recurrence_verification_numeric_fallback_for_irrational_closed_forms():
    """Fibonacci's closed form mixes sqrt(5) with float-valued initial
    conditions, so exact symbolic zero isn't achievable -- this is a
    floating-point rounding artifact (~1e-16), not a real inconsistency,
    and the verifier should still report it as verified via numeric sampling."""
    model = build_model({
        "problem_domain": "fibonacci", "problem_type": "recurrence", "independent_variable": "n",
        "variables": [
            {"symbol": "f", "meaning": "term", "known_value": None, "unit": None, "is_function": True},
            {"symbol": "n", "meaning": "index", "known_value": None, "unit": None, "is_function": False},
        ],
        "equations": [{"name": "fib", "kind": "recurrence", "expression": "Eq(f(n+2), f(n+1) + f(n))", "derivation": "x"}],
        "initial_conditions": [{"expression": "f(0)", "value": "0"}, {"expression": "f(1)", "value": "1"}],
        "solve_for": ["f"], "assumptions": [],
    })
    solutions = solve_recurrence(model)
    ok, residual = verify_recurrence_solution(model.equations[0].sympy_eq, "f", solutions["f"], sp.Symbol("n"))
    assert ok  # would fail if the check required exact residual == 0


def test_full_verify_passes_for_recurrence(recurrence_json, fake_client_factory):
    model = build_model(json.loads(recurrence_json))
    report = verify(model, fake_client_factory(), "savings problem")
    assert report.passed
    rec_check = next(c for c in report.checks if "Recurrence solution check" in c.label)
    assert rec_check.passed


def test_recurrence_steps_generated(recurrence_json):
    model = build_model(json.loads(recurrence_json))
    steps = compute_steps(model)
    assert "a" in steps
    assert len(steps["a"]) > 0
    assert any("recurrence" in s.description.lower() for s in steps["a"])
