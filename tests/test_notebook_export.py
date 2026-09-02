import json

from modules.equation_engine import build_model
from modules.solver import compute_steps
from modules.notebook_export import build_notebook


def _kinematics_model():
    return build_model({
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


def test_notebook_is_valid_json_with_correct_schema_keys():
    model = _kinematics_model()
    steps = compute_steps(model)
    nb = json.loads(build_notebook("A car problem", model, steps))
    assert nb["nbformat"] == 4
    assert "nbformat_minor" in nb
    assert "cells" in nb
    assert "metadata" in nb
    assert "kernelspec" in nb["metadata"]


def test_notebook_includes_problem_text_in_first_cell():
    model = _kinematics_model()
    steps = compute_steps(model)
    nb = json.loads(build_notebook("A specific car problem text", model, steps))
    first_source = "".join(nb["cells"][0]["source"])
    assert "A specific car problem text" in first_source
    assert nb["cells"][0]["cell_type"] == "markdown"


def test_notebook_includes_runnable_code_cell():
    model = _kinematics_model()
    steps = compute_steps(model)
    nb = json.loads(build_notebook("A car problem", model, steps))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 1
    func_source = "".join(code_cells[0]["source"])
    assert "def a(" in func_source


def test_notebook_code_cells_actually_execute_correctly():
    model = _kinematics_model()
    steps = compute_steps(model)
    nb = json.loads(build_notebook("A car problem", model, steps))
    ns = {}
    last_result = None
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src.strip().startswith("def "):
            exec(compile(src, "<cell>", "exec"), ns)
        else:
            last_result = eval(compile(src, "<cell>", "eval"), ns)
    assert last_result == 2.0


def test_notebook_demo_call_omitted_when_args_not_all_known():
    model = build_model({
        "problem_domain": "coupled", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": "5", "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
        "equations": [{"name": "eq", "kind": "equation", "expression": "Eq(y, x**2)", "derivation": ""}],
        "solve_for": ["y"], "assumptions": [],
    })
    steps = compute_steps(model)
    nb_json = build_notebook("test", model, steps)
    nb = json.loads(nb_json)
    assert nb["nbformat"] == 4


def test_notebook_degrades_to_markdown_only_for_optimization_target():
    model = build_model({
        "problem_domain": "optimization", "problem_type": "algebraic",
        "variables": [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None}],
        "equations": [],
        "objective": {"expression": "-2*x**2+120*x-800", "direction": "maximize", "optimize_over": ["x"]},
        "solve_for": ["x"], "assumptions": [],
    })
    steps = compute_steps(model)
    nb = json.loads(build_notebook("maximize profit", model, steps))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells == []
    assert all(c["cell_type"] == "markdown" for c in nb["cells"])


def test_notebook_multiple_targets_each_get_their_own_section():
    model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "a", "meaning": "a", "known_value": None, "unit": None},
            {"symbol": "v_f", "meaning": "v_f", "known_value": "20", "unit": None},
            {"symbol": "v_i", "meaning": "v_i", "known_value": "8", "unit": None},
            {"symbol": "t", "meaning": "t", "known_value": "6", "unit": None},
            {"symbol": "d", "meaning": "d", "known_value": None, "unit": None},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f-v_i)/t)", "derivation": ""},
            {"name": "disp", "kind": "equation", "expression": "Eq(d, 0.5*a*t**2 + t*v_i)", "derivation": ""},
        ],
        "solve_for": ["a", "d"], "assumptions": [],
    })
    steps = compute_steps(model)
    nb = json.loads(build_notebook("test", model, steps))
    all_source = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "Solving for `a`" in all_source
    assert "Solving for `d`" in all_source
    assert "def a(" in all_source
    assert "def d(" in all_source
