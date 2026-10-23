"""
Plotly figures for the General-Relativity tools -- Streamlit-free. Each takes the result object of its module and
returns a plain go.Figure (the app's theme and the HTML report each apply their own styling), the same arrangement
as modules/quantum_plots.py.
"""
import math

import numpy as np
import plotly.graph_objects as go

from modules.gr_cones import LightCone
from modules.gr_geodesics import GeodesicResult, to_cartesian
from modules.gr_potential import Marker
from modules.plotter import _frame_indices, _play_layout

_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]
_MARKER_STYLE = {"horizon": ("black", "solid"), "photon sphere": ("#E45756", "dash"), "ISCO": ("#54A24B", "dash"),
                 "stability limit": ("#9D755D", "dot"), "marginally bound": ("#B279A2", "dot")}


# ------------------------------------------------------------------------------------------ geodesics

def _horizon_radii(result: GeodesicResult) -> list[float]:
    from modules.gr_geodesics import default_r_stop
    r = default_r_stop(result.system.metric, result.system.params)
    return [r / 1.002] if r else []


def orbit_figure_2d(result: GeodesicResult, animate: bool = False, max_frames: int = 120) -> go.Figure:
    """The path projected on the x-y plane of the embedding (the orbital plane for an equatorial orbit), with the
    horizon as a disc and the periapses marked."""
    metric = result.system.metric
    xyz = to_cartesian(metric, result.x, result.system.params)
    fig = go.Figure()
    for rh in _horizon_radii(result):
        ang = np.linspace(0, 2 * math.pi, 200)
        fig.add_trace(go.Scatter(x=rh * np.cos(ang), y=rh * np.sin(ang), mode="lines", fill="toself", name=f"horizon r = {rh:.3g}",
                                 line=dict(color="black"), fillcolor="rgba(0,0,0,0.85)"))
    fig.add_trace(go.Scatter(x=xyz[:, 0], y=xyz[:, 1], mode="lines", name="path", line=dict(color=_PALETTE[0], width=2)))
    fig.add_trace(go.Scatter(x=[xyz[0, 0]], y=[xyz[0, 1]], mode="markers", name="start", marker=dict(size=9, color="green")))
    if result.periapses:
        pe = to_cartesian(metric, np.array([s[:metric.dim] for _, s in result.periapses]), result.system.params)
        fig.add_trace(go.Scatter(x=pe[:, 0], y=pe[:, 1], mode="markers+text", name="periapsis", text=[str(i + 1) for i in range(len(pe))],
                                 textposition="top center", marker=dict(size=8, color="#E45756", symbol="diamond")))
    fig.update_layout(xaxis=dict(title="x", scaleanchor="y"), yaxis=dict(title="y"), title=f"{metric.name}: path ({result.kind})",
                      height=520)
    if animate:
        idx = _frame_indices(len(xyz), max_frames)
        base = len(fig.data)
        fig.add_trace(go.Scatter(x=[xyz[0, 0]], y=[xyz[0, 1]], mode="markers", name="particle", marker=dict(size=12, color="#F58518")))
        fig.frames = [go.Frame(data=[go.Scatter(x=[xyz[i, 0]], y=[xyz[i, 1]])], traces=[base], name=str(k)) for k, i in enumerate(idx)]
        fig.update_layout(**_play_layout([f"λ = {result.lam[i]:.4g}" for i in idx], "", 40))
    return fig


def orbit_figure_3d(result: GeodesicResult) -> go.Figure:
    metric = result.system.metric
    xyz = to_cartesian(metric, result.x, result.system.params)
    fig = go.Figure()
    for rh in _horizon_radii(result):
        u, v = np.meshgrid(np.linspace(0, 2 * math.pi, 40), np.linspace(0, math.pi, 20))
        fig.add_trace(go.Surface(x=rh * np.cos(u) * np.sin(v), y=rh * np.sin(u) * np.sin(v), z=rh * np.cos(v), showscale=False,
                                 colorscale=[[0, "black"], [1, "black"]], opacity=0.9, name="horizon"))
    fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="lines", name="path", line=dict(color=_PALETTE[0], width=4)))
    fig.add_trace(go.Scatter3d(x=[xyz[0, 0]], y=[xyz[0, 1]], z=[xyz[0, 2]], mode="markers", name="start", marker=dict(size=5, color="green")))
    fig.update_layout(scene=dict(aspectmode="data"), title=f"{metric.name}: path in 3D", height=560)
    return fig


def conserved_figure(result: GeodesicResult) -> go.Figure:
    """|drift| of every monitored quantity against the affine parameter, on a log axis."""
    fig = go.Figure()
    for i, cq in enumerate(result.conserved):
        ref = 0.0 if cq.name.startswith("norm") else float(cq.values[np.isfinite(cq.values)][0])
        d = np.abs(cq.values - ref)
        d = np.where(d > 0, d, np.nan)
        fig.add_trace(go.Scatter(x=result.lam, y=d, mode="lines", name=cq.name, line=dict(color=_PALETTE[i % len(_PALETTE)])))
    fig.update_layout(yaxis=dict(type="log", title="|drift|"), xaxis_title="affine parameter λ",
                      title="Conserved quantities along the geodesic", height=420)
    return fig


def deflection_figure(rays: list[tuple[float, GeodesicResult]], horizon: float | None = None) -> go.Figure:
    """Several light rays (impact parameter, result) in the x-y plane, with the undeflected straight line each would follow."""
    fig = go.Figure()
    for i, (b, res) in enumerate(rays):
        xyz = to_cartesian(res.system.metric, res.x, res.system.params)
        col = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Scatter(x=xyz[:, 0], y=xyz[:, 1], mode="lines", name=f"b = {b:.4g}", line=dict(color=col)))
    if horizon:
        ang = np.linspace(0, 2 * math.pi, 200)
        fig.add_trace(go.Scatter(x=horizon * np.cos(ang), y=horizon * np.sin(ang), fill="toself", mode="lines", name="horizon",
                                 line=dict(color="black"), fillcolor="rgba(0,0,0,0.85)"))
    fig.update_layout(xaxis=dict(title="x", scaleanchor="y"), yaxis_title="y", title="Light bending", height=520)
    return fig


# ------------------------------------------------------------------------------------------ potentials

def veff_figure(r: np.ndarray, curves: list[tuple[str, np.ndarray]], markers: list[Marker], energy_sq: float | None = None,
                y_label: str = "V_eff(r)", title: str = "Effective potential") -> go.Figure:
    fig = go.Figure()
    for i, (name, v) in enumerate(curves):
        fig.add_trace(go.Scatter(x=r, y=v, mode="lines", name=name, line=dict(color=_PALETTE[i % len(_PALETTE)], width=2)))
    ys = np.concatenate([v[np.isfinite(v)] for _, v in curves]) if curves else np.array([0.0, 1.0])
    lo, hi = float(np.min(ys)), float(np.max(ys))
    for m in markers:
        if not (r[0] <= m.r <= r[-1]):
            continue
        color, dash = _MARKER_STYLE.get(m.kind, ("gray", "dot"))
        fig.add_vline(x=m.r, line=dict(color=color, dash=dash, width=1.5), annotation_text=m.kind, annotation_position="top")
    if energy_sq is not None:
        fig.add_hline(y=energy_sq, line=dict(color="gray", dash="dash"), annotation_text="E²")
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    fig.update_layout(xaxis_title="r", yaxis=dict(title=y_label, range=[lo - pad, hi + pad]), title=title, height=480)
    return fig


# ------------------------------------------------------------------------------------------ cones and Penrose

def cones_figure(cones: list[LightCone], plane: tuple[str, str], curves: list[np.ndarray] | None = None, scale: float = 0.4,
                 aspect: float = 1.0, title: str = "Light cones") -> go.Figure:
    """Each cone as two edges (future-pointing) with the wedge between them filled; closed cones (where the metric
    restricted to the plane is not Lorentzian) are marked with an x. `scale` is the edge length in the horizontal
    axis' units; the vertical extent is scale/aspect... a purely visual choice (see gr_cones)."""
    fig = go.Figure()
    xs, ys, fx, fy = [], [], [], []
    for c in cones:
        a0, b0 = c.center
        if not c.open or len(c.edges) < 2:
            fig.add_trace(go.Scatter(x=[a0], y=[b0], mode="markers", marker=dict(symbol="x", color="red", size=7),
                                     showlegend=False, hovertext="no cone (degenerate)"))
            continue
        (d1a, d1b), (d2a, d2b) = c.edges[:2]
        p1 = (a0 + scale * d1a, b0 + scale * d1b / aspect)
        p2 = (a0 + scale * d2a, b0 + scale * d2b / aspect)
        xs += [p1[0], a0, p2[0], None]
        ys += [p1[1], b0, p2[1], None]
        fx += [a0, p1[0], p2[0], a0, None]
        fy += [b0, p1[1], p2[1], b0, None]
    fig.add_trace(go.Scatter(x=fx, y=fy, mode="lines", fill="toself", fillcolor="rgba(76,120,168,0.25)", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=_PALETTE[0], width=2), name="light cone edges"))
    for i, cu in enumerate(curves or []):
        if len(cu) > 1:
            fig.add_trace(go.Scatter(x=cu[:, 0], y=cu[:, 1], mode="lines", name=f"null ray {i + 1}",
                                     line=dict(color=_PALETTE[(i + 1) % len(_PALETTE)], width=2, dash="dot")))
    fig.update_layout(xaxis_title=plane[0], yaxis_title=plane[1], title=title, height=560)
    return fig


def penrose_figure(curves: dict[str, list[np.ndarray]], title: str = "Penrose diagram") -> go.Figure:
    style = {"horizon": ("black", 3, "solid"), "singularity": ("#E45756", 4, "solid"), "infinity": ("gray", 2, "solid"),
             "boundary": ("gray", 2, "solid"), "const_r": ("#4C78A8", 1, "solid"), "const_t": ("#54A24B", 1, "dot"),
             "rays": ("#F58518", 1, "dash")}
    labels = {"horizon": "horizons", "singularity": "singularity r = 0", "infinity": "null infinity", "boundary": "boundary",
              "const_r": "r = const", "const_t": "t = const", "rays": "radial light rays"}
    fig = go.Figure()
    for key, lines in curves.items():
        color, width, dash = style.get(key, ("gray", 1, "solid"))
        for i, ln in enumerate(lines):
            pts = ln[(np.abs(ln[:, 0]) <= math.pi / 2 + 1e-9) & (np.abs(ln[:, 1]) <= math.pi / 2 + 1e-9)]
            if len(pts) < 2:
                continue
            fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", line=dict(color=color, width=width, dash=dash),
                                     name=labels.get(key, key), legendgroup=key, showlegend=(i == 0)))
    fig.update_layout(xaxis=dict(title="X'", scaleanchor="y", range=[-1.7, 1.7]), yaxis=dict(title="T'", range=[-1.7, 1.7]),
                      title=title, height=620)
    return fig


# ------------------------------------------------------------------------------------------ waves

def ring_animation_figure(positions: np.ndarray, times: np.ndarray, max_frames: int = 100) -> go.Figure:
    """A ring of test masses deformed by a gravitational wave: positions (T, P, 2)."""
    idx = _frame_indices(len(times), max_frames)
    lim = float(np.max(np.abs(positions))) * 1.15
    closed = lambda k: np.vstack([positions[k], positions[k][:1]])  # noqa: E731
    fig = go.Figure(data=[go.Scatter(x=closed(0)[:, 0], y=closed(0)[:, 1], mode="lines+markers", marker=dict(size=7, color=_PALETTE[1]),
                                     line=dict(color=_PALETTE[0]))])
    fig.frames = [go.Frame(data=[go.Scatter(x=closed(i)[:, 0], y=closed(i)[:, 1])], name=str(k)) for k, i in enumerate(idx)]
    fig.update_layout(xaxis=dict(range=[-lim, lim], scaleanchor="y", title="x"), yaxis=dict(range=[-lim, lim], title="y"),
                      title="Ring of free test masses (wave travelling along z)", height=520,
                      **_play_layout([f"t = {times[i]:.3g}" for i in idx], "", 50))
    return fig


def waveform_figure(t: np.ndarray, h_plus: np.ndarray, h_cross: np.ndarray, title: str = "Gravitational waveform") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=h_plus, name="h₊", line=dict(color=_PALETTE[0])))
    fig.add_trace(go.Scatter(x=t, y=h_cross, name="h×", line=dict(color=_PALETTE[1])))
    fig.update_layout(xaxis_title="retarded time", yaxis_title="strain h", title=title, height=380)
    return fig


def inspiral_figure(ins) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ins.t, y=ins.f_gw, name="f_gw", line=dict(color=_PALETTE[0])))
    fig.add_trace(go.Scatter(x=ins.t, y=ins.h_amp, name="strain amplitude", yaxis="y2", line=dict(color=_PALETTE[1])))
    fig.update_layout(xaxis_title="time", yaxis=dict(title="gravitational-wave frequency", type="log"),
                      yaxis2=dict(title="strain amplitude", overlaying="y", side="right", type="log"), title="Inspiral chirp", height=420)
    return fig
