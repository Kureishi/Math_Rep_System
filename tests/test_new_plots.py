from modules.equation_engine import build_model
from modules.dependency_graph import build_dependency_graph
from modules.sensitivity import sweep_input, tornado_analysis
from modules.verifier import _known_substitutions
from modules.plotter import build_dependency_graph_plot, build_tornado_chart, build_sweep_chart
from modules.plot_snapshot import snapshot_dependency_graph, snapshot_tornado_chart, snapshot_sweep_chart


def _kinematics_two_eq_model():
    return build_model({
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


# ---------------------------------------------------------------- dependency graph


def test_build_dependency_graph_plot_has_traces():
    model = _kinematics_two_eq_model()
    nodes, edges = build_dependency_graph(model)
    fig = build_dependency_graph_plot(nodes, edges)
    assert len(fig.data) > len(edges)


def test_snapshot_dependency_graph_produces_png_bytes():
    model = _kinematics_two_eq_model()
    nodes, edges = build_dependency_graph(model)
    png = snapshot_dependency_graph(nodes, edges)
    assert png.startswith(b"\x89PNG")


def test_dependency_graph_plot_handles_empty_edges():
    model = build_model({
        "problem_domain": "algebra", "problem_type": "algebraic",
        "variables": [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None}],
        "equations": [{"name": "eq1", "kind": "equation", "expression": "Eq(2*x+3, 11)", "derivation": ""}],
        "solve_for": ["x"], "assumptions": [],
    })
    nodes, edges = build_dependency_graph(model)
    fig = build_dependency_graph_plot(nodes, edges)
    assert fig is not None
    png = snapshot_dependency_graph(nodes, edges)
    assert png.startswith(b"\x89PNG")


# ---------------------------------------------------------------- tornado chart


def test_build_tornado_chart_has_one_bar_trace():
    model = _kinematics_two_eq_model()
    knowns = _known_substitutions(model)
    entries = tornado_analysis(model, "a", knowns)
    fig = build_tornado_chart(entries)
    assert len(fig.data) == 1


def test_snapshot_tornado_chart_produces_png_bytes():
    model = _kinematics_two_eq_model()
    knowns = _known_substitutions(model)
    entries = tornado_analysis(model, "a", knowns)
    png = snapshot_tornado_chart(entries)
    assert png.startswith(b"\x89PNG")


def test_tornado_chart_handles_single_entry():
    model = build_model({
        "problem_domain": "algebra", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": "10", "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
        "equations": [{"name": "eq", "kind": "equation", "expression": "Eq(y, 2*x)", "derivation": ""}],
        "solve_for": ["y"], "assumptions": [],
    })
    knowns = _known_substitutions(model)
    entries = tornado_analysis(model, "y", knowns)
    assert len(entries) == 1
    fig = build_tornado_chart(entries)
    assert fig is not None
    png = snapshot_tornado_chart(entries)
    assert png.startswith(b"\x89PNG")


# ---------------------------------------------------------------- sweep chart


def test_build_sweep_chart_includes_nominal_marker():
    model = _kinematics_two_eq_model()
    knowns = _known_substitutions(model)
    result = sweep_input(model, "a", "t", knowns)
    fig = build_sweep_chart(result)
    assert len(fig.data) == 2


def test_snapshot_sweep_chart_produces_png_bytes():
    model = _kinematics_two_eq_model()
    knowns = _known_substitutions(model)
    result = sweep_input(model, "a", "t", knowns)
    png = snapshot_sweep_chart(result)
    assert png.startswith(b"\x89PNG")
