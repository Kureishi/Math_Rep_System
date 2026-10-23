"""Tests for modules/gr_plots.py: every figure builds from real results and contains what it should."""
import math

import numpy as np
import plotly.graph_objects as go
import pytest

from modules import gr_cones, gr_geodesics as gg, gr_plots, gr_potential as gp, gr_waves as gw
from modules.gr_metrics import preset


@pytest.fixture(scope="module")
def orbit():
    system = gg.build_system(preset("schwarzschild"), {"M": 1.0})
    E, L = gp.schwarzschild_orbit_from_turning_points(1.0, 100.0, 40.0)
    x0, u0 = gg.initial_from_conserved(system, 100.0, E, L, "timelike")
    return gg.integrate_geodesic(system, x0, u0, 3 * 2 * math.pi * 70 ** 1.5, n_samples=800)


def test_orbit_figures(orbit):
    f2 = gr_plots.orbit_figure_2d(orbit)
    assert isinstance(f2, go.Figure) and {t.name for t in f2.data} >= {"path", "start", "periapsis"}
    anim = gr_plots.orbit_figure_2d(orbit, animate=True, max_frames=30)
    assert 2 <= len(anim.frames) <= 30
    assert isinstance(gr_plots.orbit_figure_3d(orbit), go.Figure)


def test_conserved_figure_has_one_trace_per_quantity(orbit):
    fig = gr_plots.conserved_figure(orbit)
    assert len(fig.data) == len(orbit.conserved) and fig.layout.yaxis.type == "log"


def test_deflection_figure(orbit):
    fig = gr_plots.deflection_figure([(5.0, orbit), (6.0, orbit)], horizon=2.0)
    assert len(fig.data) == 3


def test_veff_figure_marks_features_inside_the_range():
    st = gp.static_from_metric(preset("schwarzschild"))
    r = np.linspace(2.1, 30, 200)
    curves = [(f"L = {L}", gp.veff_static(st, L, r, {"M": 1.0})) for L in (3.5, 4.5)]
    fig = gr_plots.veff_figure(r, curves, gp.static_markers(st, {"M": 1.0}), energy_sq=0.95)
    assert len(fig.data) == 2 and len(fig.layout.shapes) >= 4          # vlines for horizon, photon sphere, ISCO, bound + E^2 line


def test_cones_figure_marks_degenerate_cones():
    cones = gr_cones.light_cones(preset("minkowski"), ("x", "y"), np.array([0.0]), np.array([0.0]), time_axis="x")
    fig = gr_plots.cones_figure(cones, ("x", "y"))
    assert any(t.marker.symbol == "x" for t in fig.data if t.marker is not None and t.marker.symbol)


def test_cones_figure_with_rays():
    met = preset("schwarzschild")
    cones = gr_cones.light_cones(met, ("t", "r"), np.linspace(-3, 3, 4), np.linspace(3, 9, 3))
    ray = gr_cones.null_curve(met, ("t", "r"), (0.0, 5.0), 1, 4.0)
    fig = gr_plots.cones_figure(cones, ("t", "r"), [ray])
    assert any(t.name == "null ray 1" for t in fig.data)


@pytest.mark.parametrize("curves", [gr_cones.minkowski_penrose(), gr_cones.schwarzschild_penrose(1.0)])
def test_penrose_figures(curves):
    fig = gr_plots.penrose_figure(curves)
    assert len(fig.data) > 3 and fig.layout.yaxis.range == (-1.7, 1.7)


def test_wave_figures():
    times = np.linspace(0, 6, 60)
    ring = gr_plots.ring_animation_figure(gw.ring_positions(2.0, 0.3, 0.1, times), times, max_frames=20)
    assert len(ring.frames) <= 20 and ring.layout.updatemenus
    assert len(gr_plots.waveform_figure(times, np.sin(times), np.cos(times)).data) == 2
    assert len(gr_plots.inspiral_figure(gw.peters_inspiral(1, 1, 100.0)).data) == 2
