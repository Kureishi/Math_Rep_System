from modules.equation_engine import build_model
from modules.dependency_graph import build_dependency_graph


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


def test_nodes_include_all_variables_and_equations():
    model = _kinematics_two_eq_model()
    nodes, edges = build_dependency_graph(model)
    node_ids = {n.id for n in nodes}
    assert node_ids == {"var:a", "var:v_f", "var:v_i", "var:t", "var:d", "eq:accel", "eq:disp"}


def test_known_and_unknown_nodes_classified_correctly():
    model = _kinematics_two_eq_model()
    nodes, edges = build_dependency_graph(model)
    kinds = {n.id: n.kind for n in nodes}
    assert kinds["var:v_f"] == "known"
    assert kinds["var:v_i"] == "known"
    assert kinds["var:t"] == "known"
    assert kinds["var:a"] == "unknown"
    assert kinds["var:d"] == "unknown"
    assert kinds["eq:accel"] == "equation"


def test_three_column_layout_x_positions():
    model = _kinematics_two_eq_model()
    nodes, edges = build_dependency_graph(model)
    for n in nodes:
        if n.kind == "known":
            assert n.x == 0.0
        elif n.kind == "equation":
            assert n.x == 1.0
        elif n.kind == "unknown":
            assert n.x == 2.0


def test_known_inputs_flow_into_equation_not_out():
    model = _kinematics_two_eq_model()
    nodes, edges = build_dependency_graph(model)
    edge_pairs = {(e.source, e.target) for e in edges}
    assert ("var:v_f", "eq:accel") in edge_pairs
    assert ("eq:accel", "var:v_f") not in edge_pairs


def test_produced_variable_flows_out_of_its_defining_equation():
    model = _kinematics_two_eq_model()
    nodes, edges = build_dependency_graph(model)
    edge_pairs = {(e.source, e.target) for e in edges}
    # 'a' is produced by 'accel' (its equation has 'a' alone on the LHS)
    assert ("eq:accel", "var:a") in edge_pairs
    assert ("var:a", "eq:accel") not in edge_pairs


def test_variable_used_as_input_elsewhere_flows_in_not_out():
    """Regression test: 'a' is produced by 'accel' but merely USED by
    'disp' (appears on disp's RHS, not as disp's LHS) -- 'disp' should
    not also appear to 'produce' a, or a would misleadingly look
    produced by two different equations."""
    model = _kinematics_two_eq_model()
    nodes, edges = build_dependency_graph(model)
    edge_pairs = {(e.source, e.target) for e in edges}
    assert ("var:a", "eq:disp") in edge_pairs
    assert ("eq:disp", "var:a") not in edge_pairs
    # disp still correctly produces 'd'
    assert ("eq:disp", "var:d") in edge_pairs


def test_coupled_system_with_no_clean_lhs_falls_back_to_produces_all():
    model = build_model({
        "problem_domain": "circuit", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
        "equations": [
            {"name": "eq1", "kind": "equation", "expression": "Eq(2*x+3*y, 8)", "derivation": ""},
            {"name": "eq2", "kind": "equation", "expression": "Eq(x-y, 1)", "derivation": ""},
        ],
        "solve_for": ["x", "y"], "assumptions": [],
    })
    nodes, edges = build_dependency_graph(model)
    edge_pairs = {(e.source, e.target) for e in edges}
    # neither equation has a clean single-symbol LHS -- both should be
    # treated as jointly determining both unknowns
    assert ("eq:eq1", "var:x") in edge_pairs
    assert ("eq:eq1", "var:y") in edge_pairs
    assert ("eq:eq2", "var:x") in edge_pairs
    assert ("eq:eq2", "var:y") in edge_pairs


def test_single_equation_problem_has_expected_edges():
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
    nodes, edges = build_dependency_graph(model)
    node_ids = {n.id for n in nodes}
    assert node_ids == {"var:x", "eq:eq1"}
    edge_pairs = {(e.source, e.target) for e in edges}
    assert ("eq:eq1", "var:x") in edge_pairs


def test_inequality_and_optimization_kinds_excluded():
    model = build_model({
        "problem_domain": "optimization", "problem_type": "algebraic",
        "variables": [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None}],
        "equations": [],
        "objective": {"expression": "-2*x**2+120*x-800", "direction": "maximize", "optimize_over": ["x"]},
        "solve_for": ["x"], "assumptions": [],
    })
    nodes, edges = build_dependency_graph(model)
    equation_nodes = [n for n in nodes if n.kind == "equation"]
    assert equation_nodes == []


def test_no_duplicate_edges():
    model = _kinematics_two_eq_model()
    nodes, edges = build_dependency_graph(model)
    edge_pairs = [(e.source, e.target) for e in edges]
    assert len(edge_pairs) == len(set(edge_pairs))
