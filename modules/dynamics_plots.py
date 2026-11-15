"""Plotly figures for the dynamics and chaos lab -- Streamlit-free, colours from modules/palette.py (never pure black or white)."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules import dynamics as dy
from modules.palette import BLUE, CATEGORICAL, INK, PURPLE, SEQUENTIAL, VERMILLION

MAX_ANIMATED_BODIES = 12


def orbit_figure(run: dy.NBodyRun, max_frames: int = 60) -> go.Figure:
    """Paths of the bodies in the x-y plane, animated with a time slider for a few bodies, and the final positions for a cluster."""
    R = run.positions
    n = R.shape[1]
    fig = go.Figure()
    for i in range(min(n, MAX_ANIMATED_BODIES) if n <= MAX_ANIMATED_BODIES else 0):
        fig.add_trace(go.Scatter(x=R[:, i, 0], y=R[:, i, 1], mode="lines", name=f"body {i + 1}", line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=2), opacity=0.5))
    if n <= MAX_ANIMATED_BODIES:
        idx = np.unique(np.linspace(0, len(run.times) - 1, min(max_frames, len(run.times))).astype(int))
        for i in range(n):
            fig.add_trace(go.Scatter(x=[R[idx[0], i, 0]], y=[R[idx[0], i, 1]], mode="markers", marker=dict(size=10 + 6 * float(np.sqrt(run.system.masses[i] / run.system.masses.max())),
                                                                                    color=CATEGORICAL[i % len(CATEGORICAL)], line=dict(width=1, color=INK)), showlegend=False))
        fig.frames = [go.Frame(data=[go.Scatter(x=[R[k, i, 0]], y=[R[k, i, 1]]) for i in range(n)], traces=list(range(n, 2 * n)), name=f"{run.times[k]:.3f}") for k in idx]
        fig.update_layout(
            updatemenus=[dict(type="buttons", x=0.0, y=-0.12, buttons=[
                dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
            sliders=[dict(active=0, x=0.18, len=0.8, pad=dict(t=40), currentvalue=dict(prefix="t = "),
                          steps=[dict(label=f"{run.times[k]:.2f}", method="animate", args=[[f"{run.times[k]:.3f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for k in idx])])
    else:
        fig.add_trace(go.Scatter(x=R[0, :, 0], y=R[0, :, 1], mode="markers", name="start", marker=dict(size=5, color=INK)))
        fig.add_trace(go.Scatter(x=R[-1, :, 0], y=R[-1, :, 1], mode="markers", name="end", marker=dict(size=6, color=BLUE)))
    lim = float(np.max(np.abs(R[..., :2]))) * 1.1 or 1.0
    fig.update_layout(height=560, title=f"{run.system.name}: {INTEGRATOR_SHORT[run.integrator]}, dt = {run.dt:g}",
                      xaxis=dict(title="x", range=[-lim, lim], scaleanchor="y", constrain="domain"), yaxis=dict(title="y", range=[-lim, lim], constrain="domain"),
                      legend=dict(orientation="h", y=-0.25))
    return fig


INTEGRATOR_SHORT = {"leapfrog": "leapfrog", "yoshida4": "Yoshida 4th order", "rk4": "Runge-Kutta 4", "euler": "explicit Euler"}


def conservation_figure(run: dy.NBodyRun) -> go.Figure:
    """The conserved quantities against time: relative energy error (signed, so a steady drift or a bounded wobble is visible), momentum and angular momentum."""
    t = run.times
    e = (run.energy - run.energy[0]) / abs(run.energy[0])
    p = np.linalg.norm(run.momentum - run.momentum[0], axis=1)
    ang = run.angular[:, 0] if run.angular.shape[1] == 1 else np.linalg.norm(run.angular, axis=1)
    lscale = max(float(np.max(np.abs(ang))), 1e-12)
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Energy: relative change", "Momentum: |ΔP|", "Angular momentum: relative change"), horizontal_spacing=0.08)
    fig.add_trace(go.Scatter(x=t, y=e, mode="lines", line=dict(color=BLUE, width=2), name="ΔE/E"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=p, mode="lines", line=dict(color=VERMILLION, width=2), name="|ΔP|"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=(ang - ang[0]) / lscale, mode="lines", line=dict(color=PURPLE, width=2), name="ΔL/L"), row=1, col=3)
    fig.update_xaxes(title_text="t")
    fig.update_layout(height=330, showlegend=False, title=f"Conserved quantities ({run.energy_drift.kind})")
    return fig


def order_study_figure(studies: list[dy.OrderStudy]) -> go.Figure:
    fig = go.Figure()
    for i, s in enumerate(studies):
        fig.add_trace(go.Scatter(x=s.dts, y=s.position_error, mode="lines+markers", name=f"{s.integrator}: slope {s.observed_order:.2f} (nominal {s.nominal_order})",
                                 line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=3), marker=dict(size=8)))
    fig.update_xaxes(type="log", title_text="time step")
    fig.update_yaxes(type="log", title_text="error in the final position (against an 8th-order reference)")
    fig.update_layout(height=420, title="How the error shrinks with the step: the slope is the order", legend=dict(orientation="h", y=-0.3))
    return fig


def pendulum_figure(run: dy.PendulumRun, max_frames: int = 90) -> go.Figure:
    p = run.params
    x1, y1, x2, y2 = dy.pendulum_positions(p, run.states)
    reach = (p.l1 + p.l2) * 1.1
    idx = np.unique(np.linspace(0, len(run.times) - 1, min(max_frames, len(run.times))).astype(int))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x2, y=y2, mode="lines", name="path of the lower bob", line=dict(color=BLUE, width=1.5), opacity=0.55))
    k0 = idx[0]
    fig.add_trace(go.Scatter(x=[0, x1[k0], x2[k0]], y=[0, y1[k0], y2[k0]], mode="lines+markers", name="pendulum", line=dict(color=INK, width=3), marker=dict(size=[4, 12, 12], color=VERMILLION)))
    fig.frames = [go.Frame(data=[go.Scatter(x=[0, x1[k], x2[k]], y=[0, y1[k], y2[k]])], traces=[1], name=f"{run.times[k]:.3f}") for k in idx]
    fig.update_layout(
        height=520, title="Double pendulum: the path of the lower bob", xaxis=dict(range=[-reach, reach], scaleanchor="y", constrain="domain", title="x"), yaxis=dict(range=[-reach, reach], title="y", constrain="domain"),
        updatemenus=[dict(type="buttons", x=0.0, y=-0.12, buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True)]),
            dict(label="⏸ Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, x=0.18, len=0.8, pad=dict(t=40), currentvalue=dict(prefix="t = "),
                      steps=[dict(label=f"{run.times[k]:.1f}", method="animate", args=[[f"{run.times[k]:.3f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for k in idx])],
        legend=dict(orientation="h", y=-0.25))
    return fig


def pendulum_energy_figure(run: dy.PendulumRun) -> go.Figure:
    fig = go.Figure(go.Scatter(x=run.times, y=(run.energy - run.energy[0]) / max(1.0, abs(run.energy[0])), mode="lines", line=dict(color=BLUE, width=2), name="ΔE/E"))
    fig.update_layout(height=300, title=f"Double pendulum: energy error ({run.energy_drift.kind})", xaxis_title="t", yaxis_title="relative change in energy")
    return fig


def lorenz_figure(ts: np.ndarray, states: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Scatter3d(x=states[:, 0], y=states[:, 1], z=states[:, 2], mode="lines", line=dict(color=ts, colorscale=SEQUENTIAL, width=3), hoverinfo="skip"))
    fig.update_layout(height=560, title="The Lorenz attractor (colour is time)", scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"), margin=dict(l=0, r=0, t=50, b=0))
    return fig


def lorenz_series_figure(ts: np.ndarray, states: np.ndarray) -> go.Figure:
    fig = go.Figure([go.Scatter(x=ts, y=states[:, i], mode="lines", name=name, line=dict(color=CATEGORICAL[i], width=1.5)) for i, name in enumerate("xyz")])
    fig.update_layout(height=300, title="The Lorenz variables against time", xaxis_title="t", legend=dict(orientation="h", y=-0.3))
    return fig


def lyapunov_figure(res: dy.LyapunovResult) -> go.Figure:
    labels = [f"λ{i + 1}" for i in range(len(res.exponents))]
    colours = [VERMILLION if e > 0 else (INK if abs(e) < 3 * s + 0.02 else BLUE) for e, s in zip(res.exponents, res.errors)]
    fig = go.Figure(go.Bar(x=labels, y=res.exponents, error_y=dict(type="data", array=3 * res.errors, visible=True), marker_color=colours))
    fig.add_hline(y=0, line_color=INK, line_width=1)
    fig.update_layout(height=340, title=f"Lyapunov exponents from {res.members:,} nearby trajectories (bars: ±3 standard errors)", yaxis_title="exponent (per unit time)")
    return fig


def divergence_figure(curve: dy.DivergenceCurve, lyapunov1: float | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.concatenate([curve.times, curve.times[::-1]]), y=np.concatenate([curve.log_separation + curve.spread, (curve.log_separation - curve.spread)[::-1]]),
                             fill="toself", fillcolor="rgba(0,114,178,0.18)", line=dict(width=0), hoverinfo="skip", name="±1 spread", showlegend=False))
    fig.add_trace(go.Scatter(x=curve.times, y=curve.log_separation, mode="lines", name="mean over the ensemble", line=dict(color=BLUE, width=3)))
    lo, hi = curve.fit_window
    sel = (curve.times >= lo) & (curve.times <= hi)
    c0 = float(np.mean(curve.log_separation[sel] - curve.slope * curve.times[sel]))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[c0 + curve.slope * lo, c0 + curve.slope * hi], mode="lines", name=f"slope {curve.slope:.3f}", line=dict(color=VERMILLION, width=2, dash="dash")))
    fig.update_layout(height=360, title="Nearby trajectories separate exponentially, then saturate at the size of the attractor", xaxis_title="t", yaxis_title="ln (separation / initial separation)",
                      legend=dict(orientation="h", y=-0.3))
    return fig


def flip_map_figure(fm: dy.FlipMap) -> go.Figure:
    z = np.log10(np.maximum(fm.flip_time, 1e-3))
    fig = go.Figure(go.Heatmap(z=z, x=fm.theta1, y=fm.theta2, colorscale=SEQUENTIAL, colorbar=dict(title="log10 t_flip"),
                               hovertemplate="θ1 %{x:.2f}<br>θ2 %{y:.2f}<br>log10 t %{z:.2f}<extra></extra>"))
    fig.update_layout(height=560, title=f"Double pendulum: time until a bob first flips over, for {fm.members:,} starting angles (at rest)", xaxis_title="θ1 (rad)", yaxis_title="θ2 (rad)",
                      yaxis=dict(scaleanchor="x", constrain="domain"))
    return fig
