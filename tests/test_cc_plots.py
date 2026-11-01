"""Tests for modules/cc_plots.py: every figure builds from real results."""
import numpy as np
import plotly.graph_objects as go
import pytest

from modules import cc_common, cc_conservation as cc, cc_convergence as cv, cc_hamiltonian as ch, cc_limits as cl, cc_plots as cp


def test_drift_and_trajectory_figures():
    import sympy as sp
    x, y = sp.symbols("x y", real=True)
    inv = cc.find_invariants([y, -x], [x, y])
    run = cc.integrate_monitored([y, -x], [x, y], [1.0, 0.0], 20.0, inv)
    fig = cp.drift_figure(run.reports)
    assert isinstance(fig, go.Figure) and len(fig.data) == len(run.reports) and fig.layout.yaxis.type == "log"
    assert len(cp.trajectory_figure(run.t, run.y, ["x", "y"]).data) == 2


def test_integrator_figure_labels_symplectic_methods():
    f = cc.HamiltonianFlow(1, lambda q, p: 0.5 * p[0] ** 2 - np.cos(q[0]), lambda q, p: np.array([np.sin(q[0])]), lambda q, p: np.array([p[0]]), True)
    cmp = cc.compare_integrators(f, np.array([1.0]), np.array([0.0]), 60.0, 0.1)
    fig = cp.integrator_figure(cmp)
    assert len(fig.data) == len(cmp.runs) and any("symplectic" in t.name for t in fig.data)


def test_convergence_figure_has_the_reference_slope():
    res = cv.convergence_study(lambda h: 1 + h ** 2, 0.5, 6, 2.0, reference=1.0, expected_order=2.0)
    fig = cp.convergence_figure(res)
    assert len(fig.data) == 2 and "slope 2" in fig.data[1].name and fig.layout.xaxis.type == "log"
    no_ref = cv.convergence_study(lambda h: 1 + h ** 2, 0.5, 6, 2.0)
    assert "observed" in cp.convergence_figure(no_ref).data[1].name


def test_phase_and_poincare_figures():
    pp = ch.phase_portrait(ch.preset_system("pendulum"), (-7, 7), (-3, 3))
    fig = cp.phase_portrait_figure(pp, "th")
    assert fig.data[0].type == "contour" and len(fig.data) == 1 + len(pp.trajectories)
    s = ch.preset_system("henon_heiles")
    z = ch.initial_on_energy(s, 1 / 12, {"x": 0.0, "y": 0.1, "p_y": 0.0}, "p_x", (0.0, 1.0))
    ps = ch.poincare_section(s, [z], "x", 0.0, 40, 300)
    assert cp.poincare_figure(ps).layout.xaxis.title.text == "y"


def test_limit_and_crossval_figures():
    rep = cl.check_preset("rel_ke")
    assert len(cp.limit_figure(rep.table, "oo").data) == 1
    assert cp.limit_figure(cl.check_preset("planck_rj").table, "0").layout.xaxis.autorange == "reversed"
    t = np.linspace(0, 1, 20)
    assert len(cp.crossval_figure(t, np.sin(t), np.sin(t)).data) == 2
