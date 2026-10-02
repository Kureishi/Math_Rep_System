"""The four time-resolved visuals, Plotly (plotter.py) and matplotlib
(plot_snapshot.py): error over time, animated phase trails, the linked time
cursor, and series partial-sum animation.

Plotly figures are checked structurally -- frame counts, which traces each
frame updates, that axis ranges are fixed, what the cursor/marker positions
are at a given frame -- because that is what makes the animation correct.
GIFs are checked for validity AND frame count. (Pixels are inspected by eye
separately; byte validity is not the same as a correct picture.)"""
import io

import numpy as np
import pytest
import sympy as sp
from PIL import Image

import modules.plot_snapshot as ps
import modules.plotter as pl
from modules.ode_trajectories import TrajectoryComparison, compare_trajectories, phase_flow
from modules.ode_utils import group_coupled_odes, solve_ode
from modules.series_animation import fourier_partial_sums, taylor_partial_sums
from modules.series_asymptotics import taylor_series
from tests.test_ode_trajectories import decay_model, spiral_model

PNG = b"\x89PNG\r\n\x1a\n"


def _n_frames(gif: bytes) -> int:
    return Image.open(io.BytesIO(gif)).n_frames


@pytest.fixture(scope="module")
def spiral():
    model = spiral_model()
    sols = solve_ode(model)
    group = group_coupled_odes([e for e in model.equations if e.kind == "ode" and e.sympy_eq is not None])[0]
    t = sp.Symbol("t")
    ts = np.linspace(0, 20, 120)
    xs = np.real(sp.lambdify(t, sols["x"].rhs, "numpy")(ts))
    ys = np.real(sp.lambdify(t, sols["y"].rhs, "numpy")(ts))
    X, Y = sp.symbols("x y")
    return dict(model=model, sols=sols, group=group, ts=ts, xs=xs, ys=ys,
                dx=sp.lambdify((X, Y), Y, "numpy"), dy=sp.lambdify((X, Y), -X - 0.3 * Y, "numpy"),
                cmp=compare_trajectories(model, group, sols, t_end=20.0, n_points=60))


# ------------------------------------------------------------------ helpers

def test_frame_indices_are_bounded_ordered_and_include_both_ends():
    assert pl._frame_indices(0, 10) == []
    assert pl._frame_indices(1, 10) == [0]
    idx = pl._frame_indices(1000, 10)
    assert len(idx) == 10 and idx[0] == 0 and idx[-1] == 999 and idx == sorted(set(idx))
    assert pl._frame_indices(5, 60) == [0, 1, 2, 3, 4]                       # fewer points than frames: keep them all
    assert ps._frame_picks(1000, 10) == idx


def test_rgba_and_play_layout():
    assert pl._rgba("#2E5EAA", 0.5) == "rgba(46,94,170,0.50)"
    layout = pl._play_layout(["a", "b", "c"], "t = ")
    assert [b["label"] for b in layout["updatemenus"][0]["buttons"]] == ["\u25b6 Play", "\u23f8 Pause"]
    steps = layout["sliders"][0]["steps"]
    assert [s["label"] for s in steps] == ["a", "b", "c"]
    assert [s["args"][0] for s in steps] == [["0"], ["1"], ["2"]]            # step i drives the frame named str(i)


# ------------------------------------------------------------------ error over time

def test_error_plot_structure_and_verdict(spiral):
    fig = pl.build_ode_error_plot(spiral["cmp"])
    assert len(fig.data) == 6                                                # per function: closed form, numerical, error
    assert fig.layout.yaxis2.type == "log"
    assert any(s.y0 == pytest.approx(spiral["cmp"].tolerance) for s in fig.layout.shapes)   # the tolerance line
    assert "agree" in fig.layout.title.text and "DISAGREE" not in fig.layout.title.text
    err_trace = fig.data[2]
    assert err_trace.y.min() >= 1e-16                                        # exact zeros are floored for the log axis


def test_error_plot_reports_a_disagreement():
    model = decay_model()
    group = group_coupled_odes([e for e in model.equations if e.kind == "ode"])[0]
    wrong = {"N": sp.Eq(sp.Function("N")(sp.Symbol("t")), 100 * sp.exp(-0.4 * sp.Symbol("t")))}
    cmp = compare_trajectories(model, group, wrong)
    assert "DISAGREE" in pl.build_ode_error_plot(cmp).layout.title.text
    assert ps.snapshot_ode_error_plot(cmp)[:8] == PNG                       # the disagreeing case renders too


def test_error_plot_refuses_an_inapplicable_comparison():
    bad = TrajectoryComparison(applicable=False, reason="no initial conditions")
    with pytest.raises(ValueError, match="no initial conditions"):
        pl.build_ode_error_plot(bad)
    with pytest.raises(ValueError, match="no initial conditions"):
        ps.snapshot_ode_error_plot(bad)
    with pytest.raises(ValueError, match="No comparison available"):
        pl.build_ode_error_plot(TrajectoryComparison(applicable=False))


def test_error_snapshot_formats(spiral):
    assert ps.snapshot_ode_error_plot(spiral["cmp"])[:8] == PNG
    assert ps.snapshot_ode_error_plot(spiral["cmp"], fmt="pdf")[:4] == b"%PDF"
    with pytest.raises(ValueError, match="Unsupported format"):
        ps.snapshot_ode_error_plot(spiral["cmp"], fmt="bmp")


# ------------------------------------------------------------------ phase trails

def _trail_fig(spiral, trajectories=None, **kw):
    paths = trajectories if trajectories is not None else [(spiral["xs"], spiral["ys"])]
    return pl.build_phase_trail_animation(spiral["dx"], spiral["dy"], (-1.5, 1.5), (-1.5, 1.5), paths,
                                            spiral["ts"], "x", "y", **kw)


def test_trail_animation_frames_update_exactly_the_two_moving_traces(spiral):
    fig = _trail_fig(spiral, max_frames=20)
    n_static = len(fig.data) - 2
    assert len(fig.frames) == 20 and fig.frames[0].name == "0" and fig.frames[-1].name == "19"
    for frame in fig.frames:
        assert list(frame.traces) == [n_static, n_static + 1]                # trail + head, nothing else is redrawn
        assert len(frame.data) == 2
    assert len(fig.layout.sliders[0].steps) == 20
    assert len(fig.layout.updatemenus[0].buttons) == 2


def test_trail_animation_head_is_at_the_current_time_and_trail_is_bounded(spiral):
    fig = _trail_fig(spiral, trail_length=8, max_frames=len(spiral["ts"]))   # one frame per sample
    k = 50
    trail, head = fig.frames[k].data
    assert list(head.x) == [pytest.approx(spiral["xs"][k])] and list(head.y) == [pytest.approx(spiral["ys"][k])]
    assert len(trail.x) == 8                                                 # exactly trail_length points
    assert trail.x[-1] == pytest.approx(spiral["xs"][k])                     # ending at the head
    alphas = [float(c.rstrip(")").split(",")[-1]) for c in trail.marker.color]
    assert alphas == sorted(alphas) and alphas[0] < alphas[-1]               # the trail FADES toward its tail
    first_trail, _ = fig.frames[0].data
    assert len(first_trail.x) == 1                                           # at t0 the trail is just the start


def test_trail_animation_axes_are_fixed(spiral):
    fig = _trail_fig(spiral)
    assert tuple(fig.layout.xaxis.range) == (-1.5, 1.5) and tuple(fig.layout.yaxis.range) == (-1.5, 1.5)
    assert fig.layout.showlegend is False


def test_trail_animation_leaves_out_points_that_are_nan(spiral):
    flows = phase_flow(lambda x, y: x * x, lambda x, y: 0.0, [(1.0, 0.0)], duration=3.0,
                       n_points=len(spiral["ts"]), blowup=1e3)               # blows up at t = 1, NaN after
    fig = _trail_fig(spiral, trajectories=[(spiral["xs"], spiral["ys"])] + flows, max_frames=len(spiral["ts"]))
    _, head_early = fig.frames[2].data
    _, head_late = fig.frames[-1].data
    assert len(head_early.x) == 2                                            # both paths alive early on
    assert len(head_late.x) == 1                                             # the blown-up one is simply not drawn


def test_trail_animation_with_no_paths_still_builds(spiral):
    fig = _trail_fig(spiral, trajectories=[])
    assert len(fig.frames) >= 1 and list(fig.frames[0].data[1].x) == []


def test_highlighted_path_is_drawn_in_the_lead_colour(spiral):
    other = (spiral["xs"] * 0.5, spiral["ys"] * 0.5)
    fig = _trail_fig(spiral, trajectories=[(spiral["xs"], spiral["ys"]), other], highlight=1)
    head = fig.frames[3].data[1]
    assert list(head.marker.color) == [pl._PALETTE[4], pl._PALETTE[0]]       # path 0 amber, highlighted path 1 blue


def test_trail_gif_is_valid_with_the_requested_frame_count(spiral):
    flows = phase_flow(spiral["dx"], spiral["dy"], [(0.5, 0.5)], 20.0, n_points=len(spiral["ts"]))
    gif = ps.snapshot_phase_trail_gif(spiral["dx"], spiral["dy"], (-1.5, 1.5), (-1.5, 1.5),
                                       [(spiral["xs"], spiral["ys"])] + flows, spiral["ts"], "x", "y", n_frames=4)
    assert gif[:4] == b"GIF8" and _n_frames(gif) == 4


# ------------------------------------------------------------------ linked time cursor

def _linked(spiral, **kw):
    return pl.build_time_linked_view(spiral["ts"], [("x", spiral["xs"]), ("y", spiral["ys"])], **kw)


def test_linked_view_needs_two_functions(spiral):
    with pytest.raises(ValueError, match="at least two"):
        pl.build_time_linked_view(spiral["ts"], [("x", spiral["xs"])])
    with pytest.raises(ValueError, match="at least two"):
        ps.snapshot_time_linked_gif(spiral["ts"], [("x", spiral["xs"])])


def test_linked_cursor_and_markers_all_sit_at_the_same_instant(spiral):
    fig = _linked(spiral, field=(spiral["dx"], spiral["dy"]), max_frames=len(spiral["ts"]))
    n_static = len(fig.data) - 3
    k = 40
    frame = fig.frames[k]
    assert list(frame.traces) == [n_static, n_static + 1, n_static + 2]
    cursor, series_markers, phase_marker = frame.data
    t_k = spiral["ts"][k]
    assert list(cursor.x) == [pytest.approx(t_k)] * 2                         # a vertical line at t_k
    assert list(series_markers.x) == [pytest.approx(t_k)] * 2
    assert list(series_markers.y) == [pytest.approx(spiral["xs"][k]), pytest.approx(spiral["ys"][k])]
    assert (phase_marker.x[0], phase_marker.y[0]) == pytest.approx((spiral["xs"][k], spiral["ys"][k]))
    assert f"t = {t_k:.3g}" in frame.layout.title.text                        # the readout names the same instant
    assert f"x = {spiral['xs'][k]:.4g}" in frame.layout.title.text


def test_linked_view_places_cursor_in_the_left_panel_and_phase_marker_in_the_right(spiral):
    fig = _linked(spiral)
    cursor, markers, phase = fig.data[-3:]
    assert cursor.xaxis == markers.xaxis == "x" and phase.xaxis == "x2"       # left panel vs right panel


def test_linked_view_with_and_without_a_direction_field(spiral):
    with_field = _linked(spiral, field=(spiral["dx"], spiral["dy"]))
    without = _linked(spiral)
    assert len(with_field.data) > len(without.data)                           # the quiver arrows are extra static traces
    assert len(without.frames) == len(with_field.frames) == 60


def test_linked_gif_is_valid_with_the_requested_frame_count(spiral):
    gif = ps.snapshot_time_linked_gif(spiral["ts"], [("x", spiral["xs"]), ("y", spiral["ys"])],
                                       field=(spiral["dx"], spiral["dy"]), n_frames=3)
    assert gif[:4] == b"GIF8" and _n_frames(gif) == 3
    no_field = ps.snapshot_time_linked_gif(spiral["ts"], [("x", spiral["xs"]), ("y", spiral["ys"])], n_frames=2)
    assert _n_frames(no_field) == 2


# ------------------------------------------------------------------ partial-sum animation

def test_taylor_animation_structure():
    frames = taylor_partial_sums(taylor_series("sin(x)", "x", 0, 7), half_width=3.0)
    fig = pl.build_partial_sum_animation(frames, title="sin(x)")
    assert [t.name for t in fig.data] == ["function", "expansion point", "partial sum"]   # the centre marker is Taylor-only
    assert len(fig.frames) == 4 and all(list(f.traces) == [2] for f in fig.frames)
    assert tuple(fig.layout.yaxis.range) == pytest.approx(frames.y_range)                  # fixed, from the true function
    for k, frame in enumerate(fig.frames):
        assert list(frame.data[0].y) == pytest.approx(list(frames.sums[k]))
        assert frames.labels[k] in frame.layout.title.text and "RMS error" in frame.layout.title.text
        assert "sin(x)" in frame.layout.title.text
    assert [s.label for s in fig.layout.sliders[0].steps] == ["1", "2", "3", "4"]


def test_fourier_animation_has_no_expansion_point_marker():
    frames = fourier_partial_sums("Piecewise((-1, x < 0), (1, True))", n_harmonics=5)
    fig = pl.build_partial_sum_animation(frames)
    assert [t.name for t in fig.data] == ["function", "partial sum"]
    assert len(fig.frames) == 5 and "over one full period" in fig.frames[0].layout.title.text


def test_partial_sum_builders_refuse_an_errored_result():
    bad = fourier_partial_sums("1/x")
    with pytest.raises(ValueError, match="must be finite"):
        pl.build_partial_sum_animation(bad)
    with pytest.raises(ValueError, match="must be finite"):
        ps.snapshot_partial_sum_gif(bad)


def test_partial_sum_gifs_have_one_frame_per_term():
    taylor = taylor_partial_sums(taylor_series("sin(x)", "x", 0, 7), half_width=3.0)
    gif = ps.snapshot_partial_sum_gif(taylor, "sin(x)", fps=4)
    assert gif[:4] == b"GIF8" and _n_frames(gif) == 4
    fourier = fourier_partial_sums("x", n_harmonics=3)
    assert _n_frames(ps.snapshot_partial_sum_gif(fourier)) == 3
