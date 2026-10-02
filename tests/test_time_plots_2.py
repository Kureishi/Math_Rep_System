"""Plotly builders (plotter.py) and matplotlib exports (plot_snapshot.py) for
the uncertainty fan, parameter morph, bifurcation diagram and space-time
view. Structure is asserted (which traces a frame updates, what the title
says, which regimes get a marker) because that is what makes them correct;
GIFs are checked for validity and frame count."""
import io

import numpy as np
import pytest
import sympy as sp
from PIL import Image

import modules.plot_snapshot as ps
import modules.plotter as pl
from modules.bifurcation import bifurcation_diagram
from modules.ode_utils import group_coupled_odes, solve_ode
from modules.parameter_morph import MorphResult, morph_family
from modules.pde_field import evaluate_field
from modules.time_uncertainty import FanResult, ode_uncertainty_fan
from tests.test_ode_trajectories import decay_model, spiral_model

PNG = b"\x89PNG\r\n\x1a\n"
t, k, w = sp.symbols("t k w")


def _n_frames(gif: bytes) -> int:
    return Image.open(io.BytesIO(gif)).n_frames


def _fan(model, sigmas, **kw):
    symbolic = solve_ode(model, symbolic_initial_conditions=True)
    group = group_coupled_odes([e for e in model.equations if e.kind == "ode" and e.sympy_eq is not None])[0]
    return ode_uncertainty_fan(model, group, symbolic, sigmas, kw.pop("t_range", (0.0, 6.0)), **kw)


@pytest.fixture(scope="module")
def fan_single():
    return _fan(decay_model(), {"k": 0.05}, n_samples=300, envelope_sigmas=2.0)


@pytest.fixture(scope="module")
def fan_pair():
    return _fan(spiral_model(), {"x(0)": 0.05}, n_samples=300)


@pytest.fixture(scope="module")
def morph():
    return morph_family(sp.exp(-t / 4) * sp.cos(w * t), t, w, 0.2, 3.0, (0, 10), {}, function="x",
                         n_values=24, nominal_value=1.0)


@pytest.fixture(scope="module")
def logistic():
    return bifurcation_diagram(lambda x, r: r * x * (1 - x), "r", (2.8, 4.0), 0.3,
                                n_params=300, n_transient=400, n_keep=32)


@pytest.fixture(scope="module")
def heat_field():
    x, tt = sp.Symbol("x", positive=True), sp.Symbol("t", positive=True)
    return evaluate_field(sp.sin(sp.pi * x) * sp.exp(-sp.pi ** 2 * tt), 1.0, 0.4, n_x=80, n_t=60)


# ------------------------------------------------------------------ uncertainty fan

def test_fan_traces_per_function_with_and_without_the_envelope(fan_single):
    with_env = pl.build_uncertainty_fan(fan_single)
    assert [tr.name for tr in with_env.data if tr.showlegend] == [
        "guaranteed envelope (\u00b12\u03c3)", "5\u201395% of samples", "25\u201375% of samples", "median",
        "nominal (inputs at their means)"]
    assert len(with_env.data) == 8                                           # 2 envelope edges + 4 band edges + median + nominal

    bare = _fan(decay_model(), {"k": 0.05}, n_samples=100)
    assert len(pl.build_uncertainty_fan(bare).data) == 6


def test_fan_bands_are_stacked_in_the_order_plotly_needs_to_fill_between_them(fan_single):
    fig = pl.build_uncertainty_fan(fan_single)
    by_fill = [(tr.fill, tr.y) for tr in fig.data if tr.fill == "tonexty"]
    assert len(by_fill) == 2                                                 # outer and inner band, each filled to the trace before it
    outer_upper = next(tr for tr in fig.data if tr.name is None and tr.line.width == 0)
    assert list(outer_upper.y) == pytest.approx(list(fan_single.p95[0]))     # the filled band runs p95 down to p5


def test_fan_has_one_panel_per_function(fan_pair):
    fig = pl.build_uncertainty_fan(fan_pair)
    assert fig.layout.height == 380 + 230
    assert [a.text for a in fig.layout.annotations] == ["x", "y"]            # subplot titles
    assert sum(1 for tr in fig.data if tr.name == "median") == 2
    assert "300 samples" in fig.layout.title.text and "seed 12345" in fig.layout.title.text


def test_fan_builders_refuse_an_inapplicable_result():
    bad = FanResult(applicable=False, reason="nothing definite")
    for builder in (pl.build_uncertainty_fan, ps.snapshot_uncertainty_fan):
        with pytest.raises(ValueError, match="nothing definite"):
            builder(bad)
        with pytest.raises(ValueError, match="No uncertainty fan available"):
            builder(FanResult(applicable=False))


def test_fan_snapshot_formats(fan_single, fan_pair):
    assert ps.snapshot_uncertainty_fan(fan_single)[:8] == PNG
    assert ps.snapshot_uncertainty_fan(fan_pair, fmt="pdf")[:4] == b"%PDF"
    with pytest.raises(ValueError, match="Unsupported format"):
        ps.snapshot_uncertainty_fan(fan_single, fmt="bmp")


# ------------------------------------------------------------------ parameter morph

def test_morph_frames_update_only_the_highlighted_curve(morph):
    fig = pl.build_parameter_morph(morph)
    assert [tr.name for tr in fig.data] == ["all values", "current setting (w = 1)", "x(t)"]
    assert len(fig.frames) == 24 and all(list(f.traces) == [2] for f in fig.frames)
    for i, frame in enumerate(fig.frames):
        assert list(frame.data[0].y) == pytest.approx(list(morph.curves[i]))
        assert f"w = {morph.values[i]:.4g}" in frame.layout.title.text
    assert tuple(fig.layout.yaxis.range) == pytest.approx(morph.y_range)    # fixed: nothing rescales mid-animation


def test_morph_titles_say_monotone_or_count_the_turning_points(morph):
    fig = pl.build_parameter_morph(morph)
    assert "monotone" in fig.frames[0].layout.title.text                     # w = 0.2: no turn within the window
    last = fig.frames[-1].layout.title.text
    assert f"{morph.turning_points[-1]} turning points" in last
    one = MorphResult(applicable=True, function="y", param_name="p", values=np.array([1.0, 2.0]),
                       t=np.array([0.0, 1.0]), curves=np.zeros((2, 2)), turning_points=[1, 1])
    assert "1 turning point " in pl.build_parameter_morph(one).frames[0].layout.title.text + " "   # singular


def test_morph_without_a_nominal_value_has_no_current_setting_trace():
    m = morph_family(sp.exp(-k * t), t, k, 0.5, 2.0, (0, 4), {}, n_values=5)
    fig = pl.build_parameter_morph(m)
    assert [tr.name for tr in fig.data] == ["all values", "y(t)"] and len(fig.frames) == 5


def test_morph_slider_labels_are_the_parameter_values(morph):
    fig = pl.build_parameter_morph(morph)
    assert fig.layout.sliders[0].steps[0].label == f"{morph.values[0]:.3g}"
    assert fig.layout.sliders[0].currentvalue.prefix == "w = "


def test_morph_gif_and_refusals(morph):
    gif = ps.snapshot_parameter_morph_gif(morph, n_frames=5)
    assert gif[:4] == b"GIF8" and _n_frames(gif) == 5
    no_nominal = morph_family(sp.exp(-k * t), t, k, 0.5, 2.0, (0, 4), {}, n_values=4)
    assert _n_frames(ps.snapshot_parameter_morph_gif(no_nominal, n_frames=3)) == 3
    bad = MorphResult(applicable=False, reason="no k in solution")
    with pytest.raises(ValueError, match="no k in solution"):
        pl.build_parameter_morph(bad)
    with pytest.raises(ValueError, match="no k in solution"):
        ps.snapshot_parameter_morph_gif(bad)
    with pytest.raises(ValueError, match="No morph available"):
        pl.build_parameter_morph(MorphResult(applicable=False))


# ------------------------------------------------------------------ bifurcation diagram

def test_bifurcation_plot_marks_the_doubling_cascade_and_the_current_value(logistic):
    fig = pl.build_bifurcation_plot(logistic, marker=3.2)
    assert len(fig.data) == 1 and fig.data[0].type == "scattergl"
    marked = sorted(s.x0 for s in fig.layout.shapes if s.line.dash == "dot")
    # periods 2, 4, 8 -- at theory's 3, 1 + sqrt(6), 3.5441, to this grid's resolution (0.004)
    assert marked[:3] == pytest.approx([3.0, 1 + np.sqrt(6), 3.5441], abs=0.012)
    current = [s for s in fig.layout.shapes if s.line.dash == "dash"]
    assert len(current) == 1 and current[0].x0 == 3.2
    assert fig.layout.xaxis.range[0] == pytest.approx(2.8) and fig.layout.xaxis.range[1] == pytest.approx(4.0)


def test_bifurcation_plot_drops_escaped_points_and_ignores_an_out_of_range_marker(logistic):
    fig = pl.build_bifurcation_plot(logistic, marker=9.9)
    assert not [s for s in fig.layout.shapes if s.line.dash == "dash"]
    assert np.isfinite(np.asarray(fig.data[0].y, dtype=float)).all()
    escaped = bifurcation_diagram(lambda x, p: 2 * x + p, "p", (1.0, 2.0), 1.0, n_params=4, n_transient=100, n_keep=3)
    assert len(pl.build_bifurcation_plot(escaped).data[0].x) == 0            # nothing finite to draw, still a valid figure


def test_bifurcation_snapshots(logistic):
    assert ps.snapshot_bifurcation_plot(logistic, marker=3.2)[:8] == PNG
    assert ps.snapshot_bifurcation_plot(logistic)[:8] == PNG
    assert ps.snapshot_bifurcation_plot(logistic, marker=99.0, fmt="pdf")[:4] == b"%PDF"
    bad = bifurcation_diagram(lambda x, p: x, "p", (1.0, 1.0), 0.5)
    with pytest.raises(ValueError, match="range for p"):
        pl.build_bifurcation_plot(bad)
    with pytest.raises(ValueError, match="range for p"):
        ps.snapshot_bifurcation_plot(bad)
    with pytest.raises(ValueError, match="No bifurcation diagram available"):
        pl.build_bifurcation_plot(type(bad)(applicable=False))


# ------------------------------------------------------------------ space-time view

def test_space_time_view_has_two_linked_panels_and_updates_profile_and_cursor(heat_field):
    fig = pl.build_space_time_view(heat_field, y_label="u(x,t)")
    assert [tr.type for tr in fig.data] == ["scatter", "heatmap", "scatter"]
    assert len(fig.frames) == 60 and all(list(f.traces) == [0, 2] for f in fig.frames)
    k_frame = 17
    profile, cursor = fig.frames[k_frame].data
    assert list(profile.y) == pytest.approx(list(heat_field.u[k_frame]))
    assert list(cursor.y) == [pytest.approx(heat_field.ts[k_frame])] * 2     # the heatmap cursor sits at the same time


def test_space_time_title_reports_the_measured_quantities(heat_field):
    fig = pl.build_space_time_view(heat_field, y_label="u")
    title = fig.frames[30].layout.title.text
    assert f"t = {heat_field.ts[30]:.3g}" in title
    assert f"{heat_field.max_abs[30]:.4g}" in title and f"{heat_field.integral[30]:.4g}" in title


def test_space_time_colour_scale_is_diverging_only_when_the_field_changes_sign(heat_field):
    unsigned = pl.build_space_time_view(heat_field)
    assert heat_field.u_min >= 0 and unsigned.data[1].colorscale[0][1].lower() != unsigned.data[1].colorscale[-1][1].lower()
    x, tt = sp.Symbol("x", positive=True), sp.Symbol("t", positive=True)
    wave = evaluate_field(sp.sin(sp.pi * x) * sp.cos(sp.pi * tt), 1.0, 2.0, n_x=40, n_t=40)
    signed = pl.build_space_time_view(wave)
    assert signed.data[1].zmid == 0 and signed.data[1].zmin == pytest.approx(-signed.data[1].zmax)   # symmetric about 0


def test_space_time_gif_and_refusals(heat_field):
    gif = ps.snapshot_space_time_gif(heat_field, n_frames=4)
    assert gif[:4] == b"GIF8" and _n_frames(gif) == 4
    x, tt = sp.Symbol("x", positive=True), sp.Symbol("t", positive=True)
    wave = evaluate_field(sp.sin(sp.pi * x) * sp.cos(sp.pi * tt), 1.0, 2.0, n_x=30, n_t=30)
    assert _n_frames(ps.snapshot_space_time_gif(wave, n_frames=3)) == 3      # the signed (diverging) branch
    bad = evaluate_field(sp.Integer(1), 0.0, 1.0)
    with pytest.raises(ValueError, match="must both be positive"):
        pl.build_space_time_view(bad)
    with pytest.raises(ValueError, match="must both be positive"):
        ps.snapshot_space_time_gif(bad)
