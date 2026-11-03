"""Plotly figures for equation discovery -- Streamlit-free, plain go.Figure like modules/sm_plots.py."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.discover import DiscoverResult

_PAL = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]


def fit_figure(result: DiscoverResult, sim: np.ndarray) -> go.Figure:
    """Each state: the measurements, and the discovered system integrated from the first point (over the rows of `sim`;
    NaN rows, where a model blew up, are simply not drawn)."""
    n_states = len(result.names)
    fig = make_subplots(rows=n_states, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=[f"{n}(t)" for n in result.names])
    for j, name in enumerate(result.names):
        fig.add_trace(go.Scatter(x=result.t, y=result.X[:, j], mode="markers", name="data" if j == 0 else None,
                                 legendgroup="data", showlegend=(j == 0), marker=dict(size=3, color="rgba(120,120,120,0.55)")),
                      row=j + 1, col=1)
        fig.add_trace(go.Scatter(x=result.t[:len(sim)], y=sim[:, j], mode="lines", name="discovered model" if j == 0 else None,
                                 legendgroup="sim", showlegend=(j == 0), line=dict(color=_PAL[j % len(_PAL)], width=2)),
                      row=j + 1, col=1)
    fig.update_xaxes(title_text="t", row=n_states, col=1)
    fig.update_layout(height=max(320, 230 * n_states), title="Data against the discovered system run forward from t₀",
                      legend=dict(orientation="h", y=1.08))
    return fig


def path_figure(result: DiscoverResult) -> go.Figure:
    """How the number of terms and the error change with the threshold; the chosen threshold is marked."""
    thr = [p.threshold for p in result.path]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=thr, y=[p.n_terms for p in result.path], mode="lines+markers", name="terms kept",
                             line=dict(color=_PAL[0])), secondary_y=False)
    fig.add_trace(go.Scatter(x=thr, y=[p.rss for p in result.path], mode="lines+markers", name="squared error",
                             line=dict(color=_PAL[1], dash="dot")), secondary_y=True)
    fig.add_vline(x=result.threshold, line_dash="dash", line_color="gray",
                  annotation_text=f"chosen ({result.threshold_mode})", annotation_position="top")
    fig.update_xaxes(type="log", title_text="threshold")
    fig.update_yaxes(title_text="terms kept", secondary_y=False)
    fig.update_yaxes(title_text="squared derivative error", type="log", secondary_y=True)
    fig.update_layout(height=380, title="Sparsity path: error against number of terms")
    return fig


def inclusion_figure(result: DiscoverResult) -> go.Figure:
    """How often each selected term was chosen when the rows were resampled (bootstrap)."""
    rows = [r for r in result.term_table() if r["selected in resamples"] is not None]
    labels = [f"d{r['equation'][1:-3]}/dt: {r['term']}" for r in rows]
    freq = [r["selected in resamples"] for r in rows]
    fig = go.Figure(go.Bar(x=freq, y=labels, orientation="h",
                           marker_color=[_PAL[2] if f >= 0.8 else _PAL[3] for f in freq]))
    fig.add_vline(x=0.8, line_dash="dash", line_color="gray")
    fig.update_xaxes(range=[0, 1.02], title_text="fraction of resamples in which the term was selected")
    fig.update_layout(height=max(240, 40 * len(rows) + 120), title="Stability of the selected terms",
                      yaxis=dict(autorange="reversed"))
    return fig
