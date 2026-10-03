"""Plotly builders and matplotlib GIFs for the solve-order replay, the chain
value flow, and the motion-diagram upgrades (strobe ghosts, acceleration
arrows). Structure is asserted -- what each frame updates, which node is
ringed, where each arrow starts and ends -- because that is what makes the
animation correct; GIFs are checked for validity and frame count."""
import io

import numpy as np
import pytest
from PIL import Image

import modules.plot_snapshot as ps
import modules.plotter as pl
from modules.chain_flow import build_chain_flow
from modules.chains import Chain, ChainStep, InputBinding
from modules.dependency_graph import build_dependency_graph, replay_frames, solve_order
from modules.equation_engine import build_model
from modules.motion_diagram import build_kinematics_trajectory
from tests.test_chain_flow import _chain, _lit, _step, _up
from tests.test_motion_diagram import _model as motion_model, _report
from tests.test_solve_order import KNOWNS, _model


def _n_frames(gif: bytes) -> int:
    return Image.open(io.BytesIO(gif)).n_frames


# ------------------------------------------------------------------ solve-order replay

@pytest.fixture(scope="module")
def replay():
    m = _model([("dist", "Eq(d, a*t**2/2)"), ("accel", "Eq(a, (v_f - v_i)/t)")], KNOWNS + [("a", None), ("d", None)])
    nodes, edges = build_dependency_graph(m)
    return nodes, edges, replay_frames(nodes, edges, solve_order(m, nodes, edges))


def test_replay_figure_layout_and_frames(replay):
    nodes, edges, frames = replay
    fig = pl.build_solve_order_replay(nodes, edges, frames)
    assert [t.name for t in fig.data[:3]] == ["Known", "Equation", "Unknown"]            # legend swatches
    assert len(fig.data) == 3 + 1 + 2                                                    # + all edges + active edges + nodes
    assert len(fig.frames) == 3 and all(list(f.traces) == [4, 5] for f in fig.frames)    # only the two moving traces redraw
    assert [s.label for s in fig.layout.sliders[0].steps] == ["givens", "1", "2"]
    assert fig.layout.title.text == frames[0].title
    assert fig.frames[2].layout.title.text == frames[2].title


def test_replay_dims_pending_nodes_and_rings_the_current_stage(replay):
    nodes, edges, frames = replay
    fig = pl.build_solve_order_replay(nodes, edges, frames)
    ids = [n.id for n in nodes]

    def node_marker(frame_index):
        return fig.frames[frame_index].data[1].marker

    alphas = lambda marker: [float(c.rstrip(")").split(",")[-1]) for c in marker.color]
    g, s1, s2 = node_marker(0), node_marker(1), node_marker(2)
    i_acc, i_a, i_t = ids.index("eq:accel"), ids.index("var:a"), ids.index("var:t")
    assert alphas(g)[i_acc] < 0.2 and alphas(g)[i_t] == 1.0                              # pending is faint, givens are solid
    assert alphas(s1)[i_acc] == alphas(s1)[i_a] == 1.0
    assert s1.line.color[i_acc] == s1.line.color[i_a] == pl._CURRENT_RING and s1.size[i_acc] == 46
    assert s2.line.color[i_acc] == "white" and alphas(s2)[i_acc] == 1.0                  # finished: lit, no longer ringed


def test_replay_active_edges_are_drawn_thick_and_clear_between_givens_and_steps(replay):
    nodes, edges, frames = replay
    fig = pl.build_solve_order_replay(nodes, edges, frames)
    assert list(fig.frames[0].data[0].x) == []                                           # nothing active on the givens frame
    x = list(fig.frames[1].data[0].x)
    assert x.count(None) == 4                                                            # 3 inputs + 1 output edge, None-separated
    assert fig.frames[1].data[0].line.width == 5


def test_replay_marks_a_blocked_stage():
    m = _model([("e1", "Eq(z, w*2)")], [("z", None), ("w", None)])
    nodes, edges = build_dependency_graph(m)
    frames = replay_frames(nodes, edges, solve_order(m, nodes, edges))
    fig = pl.build_solve_order_replay(nodes, edges, frames)
    ids = [n.id for n in nodes]
    marker = fig.frames[1].data[1].marker
    assert marker.line.color[ids.index("eq:e1")] == pl._BLOCKED_RING and marker.line.width[ids.index("eq:e1")] == 4


def test_replay_gif_and_refusals(replay):
    nodes, edges, frames = replay
    gif = ps.snapshot_solve_order_gif(nodes, edges, frames)
    assert gif[:4] == b"GIF8" and _n_frames(gif) == 3
    m = _model([("e1", "Eq(z, w*2)")], [("z", None), ("w", None)])
    n2, e2 = build_dependency_graph(m)
    assert _n_frames(ps.snapshot_solve_order_gif(n2, e2, replay_frames(n2, e2, solve_order(m, n2, e2)))) == 2
    with pytest.raises(ValueError, match="Nothing to replay"):
        pl.build_solve_order_replay(nodes, edges, [])
    with pytest.raises(ValueError, match="Nothing to replay"):
        ps.snapshot_solve_order_gif(nodes, edges, [])


# ------------------------------------------------------------------ chain value flow

@pytest.fixture(scope="module")
def flow():
    return build_chain_flow(_chain(
        _step(0, "a", 2.0, bindings=[_lit("t", 6.0)]),
        _step(1, "d", 20.0, bindings=[_up("a", 0), _lit("t2", 10.0)]),
        _step(2, "e", None, status="error", error="Solve failed", bindings=[InputBinding("d", "upstream", None, 9, "d")])))


def test_flow_figure_reveals_one_step_per_frame_and_rings_the_current_one(flow):
    fig = pl.build_chain_flow_plot(flow)
    assert len(fig.frames) == 3 and all(list(f.traces) == list(range(7)) for f in fig.frames)
    for k, frame in enumerate(fig.frames):
        steps = frame.data[5]
        assert len(steps.x) == k + 1                                                    # steps appear cumulatively
        assert list(frame.data[6].x) == [pytest.approx(flow.steps[k].x)]                # the ring sits on step k
        assert steps.marker.line.width[k] == 5 and steps.marker.line.color[k] == pl._CURRENT_RING
    assert fig.frames[1].layout.title.text.startswith("Step 2 receives a = 2 from step 1")


def test_flow_step_nodes_show_the_value_or_the_failure(flow):
    steps = pl.build_chain_flow_plot(flow).frames[2].data[5]
    assert list(steps.text) == ["Step 1<br>a = 2", "Step 2<br>d = 20", "Step 3<br>e"]    # a failed step shows no number
    assert list(steps.marker.color) == ["#2E5EAA", "#2E5EAA", "#C0392B"]                 # ok, ok, error


def test_flow_labels_carried_values_and_draws_broken_links_dashed(flow):
    fig = pl.build_chain_flow_plot(flow)
    labels = fig.frames[2].data[2]
    assert "a = 2" in list(labels.text) and any("doesn't exist" in t for t in labels.text) is False   # no source node, no link drawn
    first = fig.frames[0].data
    assert list(first[2].text) == []                                                    # step 1 has nothing carried in
    assert list(fig.frames[1].data[2].text) == ["a = 2"]
    assert list(fig.frames[0].data[4].text) == ["t = 6"]                                # typed-in inputs are labelled
    assert list(fig.frames[2].data[4].text) == ["t = 6", "t2 = 10"]


def test_flow_with_a_broken_link_between_existing_steps_shows_it_dashed():
    f = build_chain_flow(_chain(_step(0, "a", None, status="error", error="x"),
                                 _step(1, "d", None, bindings=[_up("a", 0)])))
    frame = pl.build_chain_flow_plot(f).frames[1]
    bad = frame.data[1]
    assert bad.line.dash == "dash" and list(bad.x).count(None) == 1
    assert list(frame.data[2].text) == ["a: step 1 produced no value"]


def test_flow_single_step_and_refusals():
    one = build_chain_flow(_chain(_step(0, "a", 2.0)))
    assert len(pl.build_chain_flow_plot(one).frames) == 1
    empty = build_chain_flow(_chain())
    with pytest.raises(ValueError, match="no steps"):
        pl.build_chain_flow_plot(empty)
    with pytest.raises(ValueError, match="no steps"):
        ps.snapshot_chain_flow_gif(empty)


def test_flow_gif_has_one_frame_per_step(flow):
    gif = ps.snapshot_chain_flow_gif(flow)
    assert gif[:4] == b"GIF8" and _n_frames(gif) == 3
    broken = build_chain_flow(_chain(_step(0, "a", None, status="error", error="x"),
                                      _step(1, "d", None, bindings=[_up("a", 0)])))
    assert _n_frames(ps.snapshot_chain_flow_gif(broken)) == 2                           # the dashed, labelled-broken branch


# ------------------------------------------------------------------ motion diagram upgrades

T = np.linspace(0, 6, 61)
V0, A = 8.0, 2.0
X, V, ACC = V0 * T + 0.5 * A * T ** 2, V0 + A * T, np.full_like(T, A)


def _pt(x, y, arrow):
    """(start_x, end_x) of the first arrow segment in a trace."""
    xs = [v for v in arrow.x]
    return xs[0], xs[1]


def test_without_the_upgrades_the_figure_is_exactly_what_it_was():
    fig = pl.build_motion_diagram(T, X, V)
    assert len(fig.data) == 5 and all(len(f.data) == 5 and list(f.traces) == [0, 1, 2, 3, 4] for f in fig.frames)
    assert fig.layout.annotations[0].text == "motion"
    assert fig.layout.yaxis.range is None


def test_acceleration_adds_one_arrow_trace_under_the_object():
    fig = pl.build_motion_diagram(T, X, V, a_values=ACC)
    assert len(fig.data) == 6 and fig.data[5].name == "acceleration"
    assert "green: acceleration" in fig.layout.annotations[0].text
    frame = fig.frames[-1]
    accel = frame.data[5]
    start, end = _pt(None, None, accel)
    assert start == pytest.approx(X[-1]) and list(accel.y)[:2] == [-pl._ARROW_OFFSET] * 2
    assert accel.marker.symbol[1] == "triangle-right" and end > start                   # a > 0 points right


def test_negative_acceleration_points_left():
    fig = pl.build_motion_diagram(T, X, V, a_values=np.full_like(T, -3.0))
    accel = fig.frames[-1].data[5]
    start, end = _pt(None, None, accel)
    assert end < start and accel.marker.symbol[1] == "triangle-left"


def test_acceleration_and_velocity_are_scaled_independently():
    fig = pl.build_motion_diagram(T, X, V, a_values=ACC)
    vel_end = fig.frames[-1].data[1].x[1] - fig.frames[-1].data[1].x[0]
    acc_start, acc_end = _pt(None, None, fig.frames[-1].data[5])
    span = X.max() - X.min()
    assert vel_end == pytest.approx(0.12 * span)                                        # the fastest velocity is 12% of the track
    assert acc_end - acc_start == pytest.approx(0.12 * span)                            # ...and so is the largest acceleration


def test_strobe_ghosts_accumulate_as_time_passes():
    fig = pl.build_motion_diagram(T, X, V, n_strobes=7)
    assert len(fig.data) == 7 and fig.data[5].name == "earlier positions"
    strobes = np.unique(np.linspace(0, len(T) - 1, 7).astype(int))
    counts = [len(f.data[5].x) for f in fig.frames]
    assert counts == sorted(counts) and counts[0] == 1 and counts[-1] == 7             # a ghost is left behind each strobe instant
    last = fig.frames[-1]
    assert list(last.data[5].x) == pytest.approx([X[j] for j in strobes])               # at the position it really had then


def test_each_ghost_carries_its_own_velocity_arrow_above_and_acceleration_below():
    fig = pl.build_motion_diagram(T, X, V, a_values=ACC, n_strobes=5)
    assert len(fig.data) == 9
    last = fig.frames[-1]
    strobes = np.unique(np.linspace(0, len(T) - 1, 5).astype(int))
    vel = [v for v in last.data[7].x if v is not None]
    assert len(vel) == 2 * 5
    for k, j in enumerate(strobes):
        start, end = vel[2 * k], vel[2 * k + 1]
        assert start == pytest.approx(X[j]) and end > start
    scale = 0.12 * (X.max() - X.min()) / V.max()
    assert vel[1] - vel[0] == pytest.approx(V[strobes[0]] * scale)                      # arrows grow as the object speeds up
    assert (vel[-1] - vel[-2]) > (vel[1] - vel[0])
    assert {y for y in last.data[7].y if y is not None} == {pl._ARROW_OFFSET}
    assert {y for y in last.data[8].y if y is not None} == {-pl._ARROW_OFFSET}


def test_strobes_without_acceleration_skip_the_acceleration_arrows():
    fig = pl.build_motion_diagram(T, X, V, n_strobes=4)
    assert [t.name for t in fig.data[5:]] == ["earlier positions", "velocity at earlier positions"]
    assert "acceleration" not in fig.layout.annotations[0].text


def test_the_upgrades_leave_room_above_and_below_the_track():
    assert list(pl.build_motion_diagram(T, X, V, n_strobes=3).layout.yaxis.range) == [-0.75, 0.75]


def test_a_reversing_object_gets_left_and_right_arrows():
    t = np.linspace(0, 10, 61)
    fig = pl.build_motion_diagram(t, 20 * t - 2.5 * t ** 2, 20 - 5 * t, a_values=np.full_like(t, -5.0), n_strobes=6)
    symbols = [s for s in fig.frames[-1].data[7].marker.symbol if s != "circle"]
    assert "triangle-right" in symbols and "triangle-left" in symbols                   # velocity changes sign mid-flight


def test_strobe_indices_helper():
    assert list(pl._strobe_indices(10, 0)) == [] and list(pl._strobe_indices(10, -2)) == []
    assert list(pl._strobe_indices(3, 10)) == [0, 1, 2]                                 # never more ghosts than samples


def test_trajectory_carries_a_constant_acceleration():
    model = motion_model([
        {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
        {"symbol": "a", "meaning": "acceleration", "known_value": "2", "unit": "m/s^2"},
        {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"}])
    traj = build_kinematics_trajectory(model, _report({}))
    assert traj.a_values is not None and len(traj.a_values) == len(traj.t_values)
    assert (traj.a_values == 2.0).all()


def test_motion_gif_with_the_upgrades_is_valid_and_differs_from_the_plain_one():
    plain = ps.snapshot_motion_diagram_gif(T, X, V, n_frames=6)
    rich = ps.snapshot_motion_diagram_gif(T, X, V, n_frames=6, a_values=ACC, n_strobes=5)
    assert rich[:4] == b"GIF8" and _n_frames(rich) == 6 and rich != plain
    only_strobes = ps.snapshot_motion_diagram_gif(T, X, V, n_frames=4, n_strobes=4)
    assert _n_frames(only_strobes) == 4
