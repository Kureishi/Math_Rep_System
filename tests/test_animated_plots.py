"""
Tests for modules/plotter.py's animated views: build_phase_portrait,
build_cobweb_plot, build_monte_carlo_convergence_plot, build_descent_path_plot,
and build_motion_diagram. These were previously verified only manually / via
the Streamlit AppTest harness -- this file is the pytest regression coverage
that was missing for them.
"""
import numpy as np
import sympy as sp

from modules.plotter import (
    build_cobweb_plot, build_descent_path_plot, build_monte_carlo_convergence_plot,
    build_motion_diagram, build_phase_portrait,
)

x, y = sp.symbols("x y")


# ---------------------------------------------------------------- build_phase_portrait

def test_phase_portrait_builds_with_a_trajectory():
    dx = sp.lambdify((x, y), -0.5 * x, "numpy")
    dy = sp.lambdify((x, y), 0.5 * x - 0.2 * y, "numpy")
    t = np.linspace(0, 10, 50)
    traj = (1000 * np.exp(-0.5 * t), 500 * (np.exp(-0.5 * t) - np.exp(-0.2 * t)))
    fig = build_phase_portrait(dx, dy, (0, 1000), (0, 500), x_label="A", y_label="B", trajectory=traj)
    # quiver field trace(s) + trajectory line + start marker + end marker
    assert len(fig.data) >= 4
    assert fig.layout.xaxis.title.text == "A"
    assert fig.layout.yaxis.title.text == "B"


def test_phase_portrait_without_a_trajectory_still_builds():
    dx = sp.lambdify((x, y), y, "numpy")
    dy = sp.lambdify((x, y), -x, "numpy")
    fig = build_phase_portrait(dx, dy, (-2, 2), (-2, 2))
    assert len(fig.data) >= 1


def test_phase_portrait_survives_an_equilibrium_on_the_grid():
    """Regression test: create_streamline (the first implementation tried)
    integrates field lines via internal RK4 and raises at any grid point
    sitting exactly on an equilibrium (zero velocity). create_quiver draws
    one arrow per grid point with no integration, so this must not raise."""
    dx = sp.lambdify((x, y), x, "numpy")
    dy = sp.lambdify((x, y), y, "numpy")  # equilibrium exactly at (0, 0), inside this range
    fig = build_phase_portrait(dx, dy, (-2, 2), (-2, 2))
    assert len(fig.data) >= 1


# ---------------------------------------------------------------- build_cobweb_plot

def test_cobweb_plot_logistic_map_converges_toward_fixed_point():
    r = 2.8  # below the chaotic regime -- converges to a stable fixed point
    g = lambda xx: r * xx * (1 - xx)
    fig = build_cobweb_plot(g, 0.2, (0, 1), n_steps=40)
    assert len(fig.data) == 3  # y=g(x), y=x, path
    assert len(fig.frames) == 41  # n_steps + 1
    # the path's final point should be near the fixed point 1 - 1/r
    last_frame_y = fig.frames[-1].data[0].y
    assert abs(last_frame_y[-1] - (1 - 1 / r)) < 0.05


def test_cobweb_plot_linear_map_grows_without_bound():
    g = lambda xx: xx + 500
    fig = build_cobweb_plot(g, 0, (0, 5000), n_steps=5)
    last_frame_y = fig.frames[-1].data[0].y
    assert abs(last_frame_y[-1] - 5 * 500) < 1e-6


# ---------------------------------------------------------------- build_monte_carlo_convergence_plot

def test_monte_carlo_convergence_running_mean_approaches_true_mean():
    rng = np.random.default_rng(0)
    samples = list(rng.normal(10.0, 2.0, 1000))
    fig = build_monte_carlo_convergence_plot(samples, "a")
    assert len(fig.data) == 2  # shaded band + running-mean line
    assert len(fig.frames) > 1
    final_mean_trace = fig.frames[-1].data[1]
    assert abs(final_mean_trace.y[-1] - 10.0) < 0.5


def test_monte_carlo_convergence_handles_few_samples():
    fig = build_monte_carlo_convergence_plot([1.0, 2.0, 3.0, 4.0, 5.0], "x")
    assert len(fig.frames) >= 1


# ---------------------------------------------------------------- build_descent_path_plot

def test_descent_path_plot_traces_the_given_path():
    f = sp.lambdify((x, y), x ** 2 + y ** 2, "numpy")
    path = [(3.0 - 0.1 * i, 2.0 - 0.0667 * i) for i in range(30)]
    fig = build_descent_path_plot(f, path, (-1, 4), (-1, 3))
    assert len(fig.data) == 3  # contour + path line + critical-point marker
    assert len(fig.frames) == len(path)
    # the final frame's path should end exactly where the given path ends
    last_frame = fig.frames[-1].data[0]
    assert (last_frame.x[-1], last_frame.y[-1]) == path[-1]


def test_descent_path_plot_critical_point_marker_is_the_paths_last_point():
    f = sp.lambdify((x, y), (x - 1) ** 2 + (y + 1) ** 2, "numpy")
    path = [(0.0, 0.0), (0.5, -0.5), (1.0, -1.0)]
    fig = build_descent_path_plot(f, path, (-2, 2), (-2, 2))
    marker_trace = fig.data[2]
    assert (marker_trace.x[0], marker_trace.y[0]) == (1.0, -1.0)


# ---------------------------------------------------------------- build_motion_diagram

def test_motion_diagram_builds_two_panels_with_frames():
    ts = np.linspace(0, 6, 100)
    xs = 8 * ts + 0.5 * 2 * ts ** 2
    vs = 8 + 2 * ts
    fig = build_motion_diagram(ts, xs, vs, x_label="position", x_unit="m", t_unit="s")
    assert len(fig.data) == 5  # track line, velocity arrow, object dot, curve trail, curve dot
    assert len(fig.frames) > 1
    assert "position (m)" in fig.layout.xaxis2.title.text or "position (m)" in fig.layout.yaxis2.title.text


def test_motion_diagram_final_frame_matches_trajectory_end():
    ts = np.linspace(0, 6, 50)
    xs = 8 * ts + 0.5 * 2 * ts ** 2
    vs = 8 + 2 * ts
    fig = build_motion_diagram(ts, xs, vs)
    last_frame = fig.frames[-1]
    object_dot = last_frame.data[2]  # matches frame_traces()'s trace order
    assert abs(object_dot.x[0] - xs[-1]) < 1e-9


def test_motion_diagram_handles_negative_velocity():
    """A decelerating-then-reversing object -- the velocity arrow direction
    (marker symbol) must not crash when v goes negative partway through."""
    ts = np.linspace(0, 10, 60)
    xs = 20 * ts - 0.5 * 5 * ts ** 2
    vs = 20 - 5 * ts
    fig = build_motion_diagram(ts, xs, vs)
    assert len(fig.frames) > 1
