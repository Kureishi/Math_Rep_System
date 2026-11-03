"""Plotly figures for the visual showpieces -- Streamlit-free, plain go.Figure like modules/sm_plots.py."""
import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.plotter import _frame_indices, _play_layout
from modules.showcase import CriticalMovie, LensImage, Orbit, flamm_surface, flamm_z

_PAL = ["#4C78A8", "#F58518", "#E45756"]


def lens_figure(img: LensImage, with_hole: bool = True, show_shadow_edge: bool = True) -> go.Figure:
    """The rendered view (or the same sky with no hole), with the exact shadow edge as a dashed circle."""
    fig = go.Figure(go.Image(z=img.rgb if with_hole else img.rgb_flat))
    if with_hole and show_shadow_edge:
        c, r = img.n / 2 - 0.5, img.shadow_px_expected
        fig.add_shape(type="circle", x0=c - r, x1=c + r, y0=c - r, y1=c + r, line=dict(color="rgba(255,255,255,0.7)", dash="dash", width=1))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    fig.update_layout(height=640, margin=dict(l=0, r=0, t=30, b=0),
                      title="A starfield behind a black hole" if with_hole else "The same sky with no black hole")
    return fig


def orbit_figure(orb: Orbit, n_frames: int = 60) -> go.Figure:
    """The orbit lifted onto Flamm's paraboloid, with a moving body and Play / slider."""
    r_max = float(np.max(orb.r)) * 1.15
    sx, sy, sz = flamm_surface(r_max, orb.M)
    z = flamm_z(orb.r, orb.M)
    fig = go.Figure([
        go.Surface(x=sx, y=sy, z=-sz, opacity=0.55, showscale=False, colorscale=[[0, "#B8C4D6"], [1, "#4C78A8"]],
                   contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)), name="space"),
        go.Scatter3d(x=orb.x, y=orb.y, z=-z, mode="lines", line=dict(color=_PAL[1], width=5), name="orbit"),
        go.Scatter3d(x=[orb.x[0]], y=[orb.y[0]], z=[-z[0]], mode="markers", marker=dict(size=7, color=_PAL[2]), name="body"),
    ])
    idx = _frame_indices(len(orb.phi), n_frames)
    fig.frames = [go.Frame(data=[go.Scatter3d(x=[orb.x[i]], y=[orb.y[i]], z=[-z[i]], mode="markers", marker=dict(size=7, color=_PAL[2]))],
                           traces=[2], name=str(k)) for k, i in enumerate(idx)]
    fig.update_layout(height=620, title=f"Orbit with p = {orb.p:g} M, e = {orb.e:g} on the hole's embedding surface",
                      scene=dict(xaxis_title="x / M", yaxis_title="y / M", zaxis_title="embedding height",
                                 aspectmode="data", camera=dict(eye=dict(x=1.4, y=-1.4, z=0.9))),
                      **_play_layout([f"{orb.phi[i] / (2 * math.pi):.2f} orbits" for i in idx], "", 80))
    return fig


def critical_figure(movie: CriticalMovie, max_frames: int = 60) -> go.Figure:
    """Three lattices side by side, animated together."""
    runs = movie.runs
    n = min(len(r.snapshots) for r in runs)
    idx = _frame_indices(n, max_frames)
    fig = make_subplots(rows=1, cols=3, subplot_titles=movie.labels, horizontal_spacing=0.03)

    def heat(run, i):
        return go.Heatmap(z=run.snapshots[i][1], zmin=-1, zmax=1, colorscale="RdBu", showscale=False)

    for c, run in enumerate(runs, start=1):
        fig.add_trace(heat(run, idx[0]), row=1, col=c)
        fig.update_yaxes(autorange="reversed", scaleanchor=f"x{'' if c == 1 else c}", visible=False, row=1, col=c)
        fig.update_xaxes(visible=False, row=1, col=c)
    fig.frames = [go.Frame(data=[heat(run, i) for run in runs], traces=[0, 1, 2], name=str(k)) for k, i in enumerate(idx)]
    fig.update_layout(height=420, margin=dict(t=90),
                      **_play_layout([f"sweep {runs[0].snapshots[i][0]}" for i in idx], "", 110))
    return fig


def magnetisation_figure(movie: CriticalMovie) -> go.Figure:
    """|m| against sweep for the three temperatures."""
    fig = go.Figure()
    for color, label, run in zip(_PAL, movie.labels, movie.runs):
        fig.add_trace(go.Scatter(x=np.arange(1, len(run.order) + 1), y=run.order, mode="lines", name=label, line=dict(color=color)))
    fig.update_xaxes(title_text="sweep")
    fig.update_yaxes(title_text="|m|", range=[0, 1.02])
    fig.update_layout(height=320, title="Order parameter: it orders, wanders at T_c, and stays at noise level above",
                      legend=dict(orientation="h", y=-0.25))
    return fig
