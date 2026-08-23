import json

from modules.equation_engine import build_model
from modules.verifier import verify
from modules.solver import compute_steps
from modules.ode_utils import solve_ode, group_coupled_odes, verify_coupled_solution


def test_coupled_equations_grouped_together(coupled_ode_json):
    model = build_model(json.loads(coupled_ode_json))
    ode_eqs = [e for e in model.equations if e.kind == "ode"]
    groups = group_coupled_odes(ode_eqs)
    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_independent_ode_not_forced_into_coupled_group():
    """A third, unrelated ODE in the same problem should form its own
    group rather than being merged in with an unrelated coupled pair."""
    model = build_model({
        "problem_domain": "mixed", "problem_type": "ode", "independent_variable": "t",
        "variables": [
            {"symbol": "A", "meaning": "x", "known_value": None, "unit": None, "is_function": True},
            {"symbol": "B", "meaning": "x", "known_value": None, "unit": None, "is_function": True},
            {"symbol": "C", "meaning": "x", "known_value": None, "unit": None, "is_function": True},
            {"symbol": "t", "meaning": "x", "known_value": None, "unit": None, "is_function": False},
            {"symbol": "k1", "meaning": "x", "known_value": "0.5", "unit": None, "is_function": False},
            {"symbol": "k2", "meaning": "x", "known_value": "0.2", "unit": None, "is_function": False},
            {"symbol": "k3", "meaning": "x", "known_value": "0.05", "unit": None, "is_function": False},
        ],
        "equations": [
            {"name": "A decay", "kind": "ode", "expression": "Eq(Derivative(A(t), t), -k1*A(t))", "derivation": "x"},
            {"name": "B production", "kind": "ode",
             "expression": "Eq(Derivative(B(t), t), k1*A(t) - k2*B(t))", "derivation": "x"},
            {"name": "C independent", "kind": "ode",
             "expression": "Eq(Derivative(C(t), t), k3*C(t))", "derivation": "x"},
        ],
        "initial_conditions": [
            {"expression": "A(0)", "value": "100"}, {"expression": "B(0)", "value": "0"},
            {"expression": "C(0)", "value": "50"},
        ],
        "solve_for": ["A", "B", "C"], "assumptions": [],
    })
    ode_eqs = [e for e in model.equations if e.kind == "ode"]
    groups = group_coupled_odes(ode_eqs)
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]


def test_coupled_system_solves_with_correct_initial_values(coupled_ode_json):
    model = build_model(json.loads(coupled_ode_json))
    solutions = solve_ode(model)
    assert set(solutions.keys()) == {"A", "B"}
    assert "1000" in str(solutions["A"])  # A(0) = 1000 applied correctly


def test_coupled_solution_verifies_via_cross_substitution(coupled_ode_json):
    model = build_model(json.loads(coupled_ode_json))
    solutions = solve_ode(model)
    ode_eqs = [e for e in model.equations if e.kind == "ode"]
    group = group_coupled_odes(ode_eqs)[0]
    ok, residual = verify_coupled_solution(group, solutions)
    assert ok
    assert residual == 0


def test_full_verify_passes_for_coupled_system(coupled_ode_json, fake_client_factory):
    model = build_model(json.loads(coupled_ode_json))
    report = verify(model, fake_client_factory(), "decay chain problem")
    assert report.passed
    coupled_check = next(c for c in report.checks if "Coupled ODE system check" in c.label)
    assert coupled_check.passed


def test_steps_generated_for_both_coupled_targets(coupled_ode_json):
    model = build_model(json.loads(coupled_ode_json))
    steps = compute_steps(model)
    assert set(steps.keys()) == {"A", "B"}
    assert len(steps["A"]) > 0
    assert len(steps["B"]) > 0
    # both targets' step traces should reference the coupled system, not
    # just their own equation in isolation
    a_descriptions = " ".join(s.description for s in steps["A"])
    assert "coupled" in a_descriptions.lower()
