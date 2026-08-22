import json

from modules.equation_engine import build_model
from modules.solver import compute_steps, narrate_steps


def test_algebraic_steps_reach_correct_numeric_answer(kinematics_json):
    model = build_model(json.loads(kinematics_json))
    steps = compute_steps(model)
    assert "a" in steps
    assert len(steps["a"]) > 0
    last_step = steps["a"][-1]
    assert "2" in last_step.expression  # a = 2.0


def test_two_target_steps_both_present(kinematics_two_target_json):
    model = build_model(json.loads(kinematics_two_target_json))
    steps = compute_steps(model)
    assert set(steps.keys()) == {"a", "d"}
    assert len(steps["a"]) > 0
    assert len(steps["d"]) > 0


def test_inequality_steps_produce_solution_set(inequality_json):
    model = build_model(json.loads(inequality_json))
    steps = compute_steps(model)
    assert "v" in steps
    assert len(steps["v"]) > 0
    # the solution should mention the target variable somewhere in the trace
    assert any("v" in s.expression for s in steps["v"])


def test_ode_steps_include_general_and_particular_solution(ode_json):
    model = build_model(json.loads(ode_json))
    steps = compute_steps(model)
    assert "N" in steps
    descriptions = [s.description for s in steps["N"]]
    assert any("differential equation" in d.lower() for d in descriptions)
    assert any("general solution" in d.lower() for d in descriptions)
    assert any("initial condition" in d.lower() for d in descriptions)
    # the initial condition description should be clean plain text, not
    # raw LaTeX control sequences like "N{\\left(0 \\right)}"
    ic_step = next(s for s in steps["N"] if "initial condition" in s.description.lower())
    assert "\\left" not in ic_step.description
    assert "N(0)" in ic_step.description


def test_ode_particular_solution_contains_initial_value(ode_json):
    model = build_model(json.loads(ode_json))
    steps = compute_steps(model)
    final_step = steps["N"][-1]
    assert "500" in final_step.expression


def test_narrate_steps_attaches_explanations(kinematics_json, fake_client_factory):
    model = build_model(json.loads(kinematics_json))
    steps = compute_steps(model)
    client = fake_client_factory(narration=json.dumps([f"explanation {i}" for i in range(len(steps["a"]))]))
    narrated = narrate_steps(client, model, steps)
    assert all(s.explanation for s in narrated["a"])


def test_narrate_steps_degrades_gracefully_on_bad_json(kinematics_json, fake_client_factory):
    """If the narration model returns garbage, steps should remain valid
    (just without explanation text) rather than crashing the pipeline."""
    model = build_model(json.loads(kinematics_json))
    steps = compute_steps(model)
    client = fake_client_factory(narration="not valid json at all")
    narrated = narrate_steps(client, model, steps)
    assert len(narrated["a"]) == len(steps["a"])  # steps preserved
    assert all(s.explanation == "" for s in narrated["a"])  # just no explanation


def test_no_solve_for_returns_empty_steps():
    model = build_model({
        "variables": [], "equations": [], "solve_for": [], "assumptions": [],
    })
    assert compute_steps(model) == {}
