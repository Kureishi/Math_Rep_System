"""
Plotly figures for the cross-cutting numerics tools -- Streamlit-free. Each takes a result object of its module and returns a plain go.Figure.
"""
import math

import numpy as np
import plotly.graph_objects as go

from modules.cc_common import DriftReport
from modules.cc_convergence import ConvergenceResult
from modules.cc_conservation import IntegratorComparison
from modules.cc_hamiltonian import PhasePortrait, PoincareSection

_PAL = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]


def drift_figure(reports: list[DriftReport], title: str = "Drift of the monitored quantities") -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(reports):
        d = np.abs(r.values - r.reference)
        d = np.where(d > 0, d, np.nan)
        fig.add_trace(go.Scatter(x=r.times, y=d, mode="lines", name=f"{r.name}: {r.kind}", line=dict(color=_PAL[i % len(_PAL)])))
    fig.update_layout(xaxis_title="t", yaxis=dict(title="|Q(t) − Q(0)|", type="log"), title=title, height=420)
    return fig


def trajectory_figure(t: np.ndarray, y: np.ndarray, names: list[str]) -> go.Figure:
    fig = go.Figure()
    for i, nm in enumerate(names):
        fig.add_trace(go.Scatter(x=t, y=y[i], mode="lines", name=nm, line=dict(color=_PAL[i % len(_PAL)])))
    fig.update_layout(xaxis_title="t", title="Trajectory", height=340)
    return fig


def integrator_figure(cmp: IntegratorComparison) -> go.Figure:
    """H(t) − H(0) for each method: bounded for the symplectic ones, a steady drift for the others."""
    fig = go.Figure()
    for i, r in enumerate(cmp.runs):
        fig.add_trace(go.Scatter(x=r.t, y=r.energy - r.energy[0], mode="lines", line=dict(color=_PAL[i % len(_PAL)]),
                                 name=f"{r.label}{' (symplectic)' if r.symplectic else ''}: {r.drift.kind}"))
    fig.update_layout(xaxis_title="t", yaxis_title="H(t) − H(0)", title="Energy error: symplectic against non-symplectic integrators", height=440)
    return fig


def convergence_figure(res: ConvergenceResult) -> go.Figure:
    fig = go.Figure()
    ok = [k for k in range(len(res.h)) if res.errors[k] > 0]
    fig.add_trace(go.Scatter(x=res.h[ok], y=res.errors[ok], mode="markers+lines", name=f"error ({res.error_basis})", line=dict(color=_PAL[0])))
    p = res.expected_order if res.expected_order is not None else (res.observed_order if np.isfinite(res.observed_order) else 1.0)
    if ok:
        k0 = ok[len(ok) // 2]
        ref = res.errors[k0] * (res.h / res.h[k0]) ** p
        fig.add_trace(go.Scatter(x=res.h, y=ref, mode="lines", name=f"slope {p:g} (expected)" if res.expected_order is not None else f"slope {p:.2f} (observed)",
                                 line=dict(color="gray", dash="dash")))
    fig.update_layout(xaxis=dict(title="h", type="log"), yaxis=dict(title="error", type="log"), title=res.name, height=400)
    return fig


def phase_portrait_figure(pp: PhasePortrait, coord: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Contour(x=pp.q, y=pp.p, z=pp.H, contours=dict(coloring="none", showlabels=True), line=dict(color="rgba(120,120,120,0.6)"), showscale=False, name="H"))
    for i, ((q, p), e) in enumerate(zip(pp.trajectories, pp.energies)):
        fig.add_trace(go.Scatter(x=q, y=p, mode="lines", name=f"E = {e:.3g}", line=dict(color=_PAL[i % len(_PAL)], width=2)))
    fig.update_layout(xaxis_title=coord, yaxis_title=f"p_{coord}", title="Phase portrait (grey: level sets of H)", height=480)
    return fig


def poincare_figure(ps: PoincareSection) -> go.Figure:
    fig = go.Figure()
    for i, (pts, lab) in enumerate(zip(ps.points, ps.labels)):
        if len(pts):
            fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="markers", name=lab, marker=dict(size=3, color=_PAL[i % len(_PAL)])))
    fig.update_layout(xaxis_title=ps.axes[0], yaxis_title=ps.axes[1], title=f"Poincaré section: {ps.section}", height=520)
    return fig


def limit_figure(table: list[tuple[float, float, float, float]], to: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r[0] for r in table], y=[max(r[3], 1e-60) for r in table], mode="markers+lines", name="|f/f_expected − 1|", line=dict(color=_PAL[0])))
    fig.update_layout(xaxis=dict(title="parameter", type="log", autorange="reversed" if to == "0" else True), yaxis=dict(title="relative difference", type="log"),
                      title="Approach to the limit", height=360)
    return fig


def crossval_figure(t: np.ndarray, symbolic: np.ndarray, numeric: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=symbolic, mode="lines", name="closed form (SymPy dsolve)", line=dict(color=_PAL[0], width=3)))
    fig.add_trace(go.Scatter(x=t, y=numeric, mode="markers", name="numerical (solve_ivp)", marker=dict(color=_PAL[1], size=5)))
    fig.update_layout(xaxis_title="t", title="Closed form against numerical integration", height=360)
    return fig


