"""modules/dependency_graph.py: solve_order, describe_stage and replay_frames.
The order is checked against what the equations need, not against the code's
own output: an equation can only run once everything feeding into it is
known, whatever order the model happens to list them in."""
import pytest

from modules.dependency_graph import (
    build_dependency_graph, describe_stage, replay_frames, solve_order, SolveStage,
)
from modules.equation_engine import build_model
from tests.test_ode_trajectories import decay_model


def _model(equations, variables, kinds=None):
    names = {name for name, _ in variables}
    return build_model({
        "problem_domain": "p", "problem_type": "algebraic",
        "variables": [{"symbol": s, "meaning": s, "known_value": v, "unit": None} for s, v in variables],
        "equations": [{"name": n, "kind": (kinds or {}).get(n, "equation"), "expression": e, "derivation": ""}
                      for n, e in equations],
        "solve_for": [s for s, v in variables if v is None][:1] or [next(iter(names))], "assumptions": []})


KNOWNS = [("v_f", "20"), ("v_i", "8"), ("t", "6")]


def _stages(model):
    nodes, edges = build_dependency_graph(model)
    return solve_order(model, nodes, edges), nodes, edges


def _summary(stages):
    return [(s.kind, [e.removeprefix("eq:") for e in s.equations], [p.removeprefix("var:") for p in s.produces])
            for s in stages]


# ------------------------------------------------------------------ the order

def test_independent_equations_each_solve_one_variable():
    m = _model([("accel", "Eq(a, (v_f - v_i)/t)"), ("dist", "Eq(d, v_i*t)")],
               KNOWNS + [("a", None), ("d", None)])
    stages, _, _ = _stages(m)
    assert _summary(stages) == [("solve", ["accel"], ["a"]), ("solve", ["dist"], ["d"])]
    assert [s.index for s in stages] == [0, 1]


def test_an_equation_listed_first_waits_for_the_one_that_feeds_it():
    # dist needs a, which accel produces -- but dist is listed first
    m = _model([("dist", "Eq(d, a*t**2/2)"), ("accel", "Eq(a, (v_f - v_i)/t)"), ("energy", "Eq(e, d*a)")],
               KNOWNS + [("a", None), ("d", None), ("e", None)])
    stages, _, _ = _stages(m)
    assert _summary(stages) == [("solve", ["accel"], ["a"]), ("solve", ["dist"], ["d"]), ("solve", ["energy"], ["e"])]
    for earlier, later in zip(stages, stages[1:]):
        assert set(later.needs) & set(earlier.produces) or not later.needs      # each stage reads what came before


def test_stage_needs_are_the_variable_nodes_it_reads():
    m = _model([("accel", "Eq(a, (v_f - v_i)/t)")], KNOWNS + [("a", None)])
    stages, _, _ = _stages(m)
    assert sorted(stages[0].needs) == ["var:t", "var:v_f", "var:v_i"]


def test_simultaneous_equations_are_one_stage_followed_by_what_depends_on_them():
    m = _model([("e1", "Eq(2*x + 3*y, 8)"), ("e2", "Eq(x - y, 1)"), ("e3", "Eq(z, x + y)")],
               [("x", None), ("y", None), ("z", None)])
    stages, _, _ = _stages(m)
    assert _summary(stages) == [("simultaneous", ["e1", "e2"], ["x", "y"]), ("solve", ["e3"], ["z"])]


def test_equations_sharing_an_unknown_through_a_chain_of_overlaps_are_merged():
    # e1 and e2 share y; e2 and e3 share z: one block of three equations in three unknowns
    m = _model([("e1", "Eq(x + y, 3)"), ("e2", "Eq(y + z, 5)"), ("e3", "Eq(z + w, 7)")],
               [("x", None), ("y", None), ("z", None), ("w", None)])
    stages, _, _ = _stages(m)
    assert len(stages) == 1 and stages[0].kind == "underdetermined"          # 3 equations, 4 unknowns
    assert [e.removeprefix("eq:") for e in stages[0].equations] == ["e1", "e2", "e3"]
    assert [p.removeprefix("var:") for p in stages[0].produces] == ["w", "x", "y", "z"]


def test_one_equation_in_two_unknowns_is_underdetermined_not_solved():
    m = _model([("e1", "Eq(x + y, 3)")], [("x", None), ("y", None)])
    stages, nodes, _ = _stages(m)
    assert _summary(stages) == [("underdetermined", ["e1"], ["x", "y"])]
    assert "nothing pins them down" in describe_stage(stages[0], nodes)


def test_a_second_equation_for_an_already_found_variable_is_a_check():
    m = _model([("e1", "Eq(a, (v_f - v_i)/t)"), ("e2", "Eq(a, F/m)")], KNOWNS + [("F", "10"), ("m", "5"), ("a", None)])
    stages, nodes, _ = _stages(m)
    assert _summary(stages) == [("solve", ["e1"], ["a"]), ("check", ["e2"], [])]
    assert "only confirms what is already known" in describe_stage(stages[1], nodes)


def test_coupled_block_whose_unknowns_are_already_known_is_also_a_check():
    m = _model([("e1", "Eq(a, (v_f - v_i)/t)"), ("e2", "Eq(2*a + 3*b, 8)"), ("e3", "Eq(a - b, 1)")],
               KNOWNS + [("a", None), ("b", None)])
    stages, _, _ = _stages(m)
    assert stages[0].kind == "solve" and stages[0].produces == ["var:a"]
    assert stages[1].kind == "simultaneous" and stages[1].produces == ["var:b"]      # a is known by then; b is new


def test_an_equation_that_needs_something_never_determined_is_unresolved():
    m = _model([("e1", "Eq(z, w*2)")], [("z", None), ("w", None)])
    stages, nodes, _ = _stages(m)
    assert _summary(stages) == [("unresolved", ["e1"], [])]
    assert stages[0].missing == ["var:w"]
    assert "can't determine e1: still needs w" in describe_stage(stages[0], nodes)


def test_a_cycle_of_dependencies_is_unresolved():
    m = _model([("e1", "Eq(x, y + 1)"), ("e2", "Eq(y, x + 1)")], [("x", None), ("y", None)])
    stages, _, _ = _stages(m)
    assert len(stages) == 1 and stages[0].kind == "unresolved" and len(stages[0].equations) == 2
    assert stages[0].missing == ["var:x", "var:y"]


def test_resolved_equations_come_before_the_unresolved_remainder():
    m = _model([("ok", "Eq(a, (v_f - v_i)/t)"), ("stuck", "Eq(z, w*2)")],
               KNOWNS + [("a", None), ("z", None), ("w", None)])
    stages, _, _ = _stages(m)
    assert [s.kind for s in stages] == ["solve", "unresolved"]


def test_a_differential_equation_is_its_own_kind_of_stage():
    m = decay_model()
    stages, nodes, _ = _stages(m)
    assert [s.kind for s in stages] == ["differential"] and stages[0].produces == []
    assert "differential/recurrence equation" in describe_stage(stages[0], nodes)


def test_nodes_and_edges_are_built_when_not_supplied():
    m = _model([("accel", "Eq(a, (v_f - v_i)/t)")], KNOWNS + [("a", None)])
    assert _summary(solve_order(m)) == [("solve", ["accel"], ["a"])]


def test_a_model_with_no_equations_has_no_stages():
    m = _model([], KNOWNS + [("a", None)])
    assert solve_order(m) == []


def test_describe_stage_for_a_simultaneous_block():
    m = _model([("e1", "Eq(2*x + 3*y, 8)"), ("e2", "Eq(x - y, 1)")], [("x", None), ("y", None)])
    stages, nodes, _ = _stages(m)
    assert describe_stage(stages[0], nodes) == "solve x, y together from e1, e2"


def test_describe_stage_falls_back_to_the_raw_id_for_an_unknown_node():
    st = SolveStage(0, "solve", ["eq:ghost"], ["var:ghost"], [])
    assert describe_stage(st, []) == "solve var:ghost from eq:ghost"


# ------------------------------------------------------------------ replay frames

def test_replay_has_a_frame_for_the_givens_and_one_per_stage():
    m = _model([("dist", "Eq(d, a*t**2/2)"), ("accel", "Eq(a, (v_f - v_i)/t)")], KNOWNS + [("a", None), ("d", None)])
    stages, nodes, edges = _stages(m)
    frames = replay_frames(nodes, edges, stages)
    assert [f.index for f in frames] == [0, 1, 2]
    assert frames[0].title == "Givens: t, v_f, v_i" and frames[0].active_edges == []
    assert frames[1].title.startswith("Step 1 of 2: solve a from accel")
    assert frames[2].title.startswith("Step 2 of 2: solve d from dist")


def test_replay_states_progress_from_pending_to_current_to_done():
    m = _model([("dist", "Eq(d, a*t**2/2)"), ("accel", "Eq(a, (v_f - v_i)/t)")], KNOWNS + [("a", None), ("d", None)])
    stages, nodes, edges = _stages(m)
    f0, f1, f2 = replay_frames(nodes, edges, stages)
    assert f0.states["var:t"] == "known" and f0.states["var:a"] == "pending" and f0.states["eq:accel"] == "pending"
    assert (f1.states["eq:accel"], f1.states["var:a"]) == ("current", "current")
    assert f1.states["var:d"] == "pending" and f1.states["eq:dist"] == "pending"
    assert (f2.states["eq:accel"], f2.states["var:a"]) == ("done", "done")            # finished, stays lit
    assert (f2.states["eq:dist"], f2.states["var:d"]) == ("current", "current")
    assert f2.states["var:t"] == "known"                                               # the givens never dim


def test_replay_active_edges_are_the_ones_the_stage_reads_and_writes():
    m = _model([("accel", "Eq(a, (v_f - v_i)/t)")], KNOWNS + [("a", None)])
    stages, nodes, edges = _stages(m)
    _, f1 = replay_frames(nodes, edges, stages)
    assert sorted(f1.active_edges) == [("eq:accel", "var:a"), ("var:t", "eq:accel"),
                                       ("var:v_f", "eq:accel"), ("var:v_i", "eq:accel")]


def test_replay_marks_an_unresolved_stage_blocked():
    m = _model([("e1", "Eq(z, w*2)")], [("z", None), ("w", None)])
    stages, nodes, edges = _stages(m)
    _, blocked = replay_frames(nodes, edges, stages)
    assert blocked.states["eq:e1"] == "blocked" and "can't determine" in blocked.title


def test_replay_of_a_model_with_no_givens_says_so():
    m = _model([("e1", "Eq(x + y, 3)")], [("x", None), ("y", None)])
    stages, nodes, edges = _stages(m)
    assert replay_frames(nodes, edges, stages)[0].title == "Givens: none"


def test_a_simultaneous_block_whose_unknowns_were_all_found_elsewhere_only_checks():
    m = _model([("ea", "Eq(a, v_f - v_i)"), ("eb", "Eq(b, v_i*t)"), ("c1", "Eq(2*a + 3*b, 8)"), ("c2", "Eq(a - b, 1)")],
               KNOWNS + [("a", None), ("b", None)])
    stages, nodes, _ = _stages(m)
    assert _summary(stages) == [("solve", ["ea"], ["a"]), ("solve", ["eb"], ["b"]), ("check", ["c1", "c2"], [])]
    assert "only confirms what is already known" in describe_stage(stages[2], nodes)
