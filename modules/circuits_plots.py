"""Plotly figures for the circuits and control lab -- Streamlit-free, colours from modules/palette.py (never pure black or white)."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules import circuits as ck
from modules.palette import BLUE, CATEGORICAL, GREEN, INK, PURPLE, VERMILLION


def transient_figure(tr: ck.Transient) -> go.Figure:
    """The response against its closed form, the other signals, and the energy ledger."""
    c = tr.circuit
    unit = "V" if c.topology == "series" else "V"
    fig = make_subplots(rows=3, cols=1, row_heights=[0.42, 0.3, 0.28], vertical_spacing=0.1,
                        subplot_titles=(f"{tr.primary} after a {'voltage' if c.topology == 'series' else 'current'} step: simulation and closed form",
                                        "The other signals", "Energy ledger (joules)"))
    t = tr.t
    fig.add_trace(go.Scatter(x=t, y=tr.signals[tr.primary], mode="lines", name=f"{tr.primary} (simulated)", line=dict(color=BLUE, width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t[::max(1, len(t) // 400)], y=tr.closed_form[::max(1, len(t) // 400)], mode="markers", name="closed form", marker=dict(size=4, color=VERMILLION)), row=1, col=1)
    if c.topology == "series":
        fig.add_hline(y=tr.level, line_dash="dot", line_color=INK, row=1, col=1)
    others = [k for k in tr.signals if k != tr.primary]
    for i, k in enumerate(others):
        fig.add_trace(go.Scatter(x=t, y=tr.signals[k], mode="lines", name=k, line=dict(color=CATEGORICAL[(i + 1) % len(CATEGORICAL)], width=2)), row=2, col=1)
    for name, color in (("work", INK), ("stored", GREEN), ("dissipated", PURPLE)):
        fig.add_trace(go.Scatter(x=t, y=tr.energy[name], mode="lines", name={"work": "work by the source", "stored": "stored (C and L)", "dissipated": "dissipated in R"}[name], line=dict(color=color, width=2)), row=3, col=1)
    fig.update_xaxes(title_text="time (s)", row=3, col=1)
    fig.update_yaxes(title_text=unit, row=1, col=1)
    fig.update_layout(height=760, title=f"{c.topology.capitalize()} RLC, {c.regime}: ω₀ = {c.omega0:.4g} rad/s, ζ = {c.zeta:.4g}", legend=dict(orientation="h", y=-0.12))
    return fig


def bode_figure(b: ck.Bode, c: ck.RLC) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Magnitude", "Phase"))
    f = b.omega / (2 * np.pi)
    for i, name in enumerate(b.response):
        col = CATEGORICAL[i % len(CATEGORICAL)]
        fig.add_trace(go.Scatter(x=f, y=b.magnitude_db[name], mode="lines", name=name, line=dict(color=col, width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=f, y=b.phase_deg[name], mode="lines", name=name, line=dict(color=col, width=3), showlegend=False), row=2, col=1)
        m = b.measured[name]
        fig.add_trace(go.Scatter(x=[float(m["peak_omega"] or 1.0) / (2 * np.pi)], y=[20 * np.log10(float(m["peak_gain"] or 1.0))], mode="markers", marker=dict(size=10, color=col, symbol="diamond", line=dict(width=1, color=INK)),
                                 name=f"peak of {name}", showlegend=False), row=1, col=1)
    fig.add_vline(x=c.f0, line_dash="dot", line_color=INK, annotation_text="f₀ = 1/(2π√LC)", annotation_position="top")
    fig.update_xaxes(type="log", title_text="frequency (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="dB", row=1, col=1)
    fig.update_yaxes(title_text="degrees", row=2, col=1)
    fig.update_layout(height=620, title=f"Bode plot of the {c.topology} RLC circuit (Q = {c.Q:.3g})", legend=dict(orientation="h", y=-0.15))
    return fig


def pole_zero_figure(poles: np.ndarray, zeros: np.ndarray | None = None, title: str = "Poles and zeros in the s-plane") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=poles.real, y=poles.imag, mode="markers", name="poles", marker=dict(symbol="x", size=13, color=VERMILLION, line=dict(width=3))))
    if zeros is not None and len(zeros):
        fig.add_trace(go.Scatter(x=zeros.real, y=zeros.imag, mode="markers", name="zeros", marker=dict(symbol="circle-open", size=12, color=BLUE, line=dict(width=3))))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_layout(height=360, title=title, xaxis_title="real part (1/s)", yaxis_title="imaginary part (rad/s)", legend=dict(orientation="h", y=-0.3))
    return fig


def step_figure(la: ck.LoopAnalysis) -> go.Figure:
    t, y = la.t, la.y
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=np.ones_like(t), mode="lines", name="setpoint", line=dict(color=INK, width=1, dash="dot")))
    if la.stable:
        fig.add_trace(go.Scatter(x=np.concatenate([t, t[::-1]]), y=np.concatenate([np.full_like(t, 1.02), np.full_like(t, 0.98)]), fill="toself", fillcolor="rgba(138,143,152,0.18)",
                                 line=dict(width=0), hoverinfo="skip", name="±2 % band"))
    fig.add_trace(go.Scatter(x=t, y=np.clip(y, -1e3, 1e3), mode="lines", name="output", line=dict(color=BLUE, width=3)))
    m = la.metrics
    if m is not None and m.settling_time is not None:
        fig.add_vline(x=m.settling_time, line_dash="dash", line_color=VERMILLION, annotation_text=f"settles at {m.settling_time:.3g} s", annotation_position="top")
    fig.update_layout(height=400, title="Closed-loop step response" + ("" if la.stable else " (UNSTABLE)"), xaxis_title="time (s)", yaxis_title="output", legend=dict(orientation="h", y=-0.25))
    return fig


def loop_bode_figure(la: ck.LoopAnalysis) -> go.Figure:
    w = np.logspace(-3, 3, 1500)
    h = la.loop(1j * w)
    mag, ph = 20 * np.log10(np.abs(h)), np.degrees(np.unwrap(np.angle(h)))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Open-loop magnitude", "Open-loop phase"))
    fig.add_trace(go.Scatter(x=w, y=mag, mode="lines", name="|L(jω)|", line=dict(color=BLUE, width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=w, y=ph, mode="lines", name="phase", line=dict(color=BLUE, width=3), showlegend=False), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color=INK, row=1, col=1)
    fig.add_hline(y=-180, line_dash="dot", line_color=INK, row=2, col=1)
    m = la.margins
    if m["w_gc"] is not None:
        fig.add_trace(go.Scatter(x=[m["w_gc"]], y=[0], mode="markers", marker=dict(size=11, color=VERMILLION, symbol="diamond"), name=f"phase margin {m['pm']:.1f}°"), row=1, col=1)
    if m["w_pc"] is not None:
        fig.add_trace(go.Scatter(x=[m["w_pc"]], y=[-180], mode="markers", marker=dict(size=11, color=GREEN, symbol="diamond"), name=f"gain margin {m['gm_db']:.1f} dB"), row=2, col=1)
    fig.update_xaxes(type="log", title_text="ω (rad/s)", row=2, col=1)
    fig.update_yaxes(title_text="dB", row=1, col=1)
    fig.update_yaxes(title_text="degrees", row=2, col=1)
    fig.update_layout(height=560, title="Open-loop Bode plot with the stability margins", legend=dict(orientation="h", y=-0.2))
    return fig


def root_locus_figure(plant: ck.Plant, gain: float | None = None) -> go.Figure:
    ks, roots = ck.root_locus(plant)
    fig = go.Figure()
    for j in range(roots.shape[1]):
        fig.add_trace(go.Scatter(x=roots[:, j].real, y=roots[:, j].imag, mode="lines", line=dict(color=CATEGORICAL[j % len(CATEGORICAL)], width=2), name=f"branch {j + 1}", hoverinfo="skip"))
    p0 = plant.tf.poles
    fig.add_trace(go.Scatter(x=p0.real, y=p0.imag, mode="markers", marker=dict(symbol="x", size=12, color=INK), name="open-loop poles"))
    if gain is not None:
        r = np.roots(np.polyadd(plant.tf.den, gain * plant.tf.num))
        fig.add_trace(go.Scatter(x=r.real, y=r.imag, mode="markers", marker=dict(size=11, color=VERMILLION, symbol="diamond"), name=f"poles at Kp = {gain:.3g}"))
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_layout(height=420, title="Root locus: where the closed-loop poles go as the proportional gain rises", xaxis_title="real part (1/s)", yaxis_title="imaginary part (rad/s)", legend=dict(orientation="h", y=-0.25))
    return fig
