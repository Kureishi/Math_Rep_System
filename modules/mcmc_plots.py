"""Plotly figures for posterior sampling -- Streamlit-free, plain go.Figure like modules/sm_plots.py."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.mcmc import PosteriorResult

_BLUE, _ORANGE, _GREEN, _RED = "#4C78A8", "#F58518", "#54A24B", "#E45756"


def _thinned(samples: np.ndarray, limit: int = 3000) -> np.ndarray:
    if len(samples) <= limit:
        return samples
    return samples[np.linspace(0, len(samples) - 1, limit).astype(int)]


def corner_figure(result: PosteriorResult, truth: list[float] | None = None, max_params: int = 6) -> go.Figure:
    """The corner plot: each parameter's marginal on the diagonal (median and 68% interval marked) and every pair's joint
    density below it, with the least-squares point (red x) and, for a demo, the true values (green diamond)."""
    names = result.names[:max_params]
    k = len(names)
    data = _thinned(result.samples)
    fig = make_subplots(rows=k, cols=k, horizontal_spacing=0.025, vertical_spacing=0.025, shared_xaxes="columns")
    for i in range(k):
        for j in range(i + 1):
            row, col = i + 1, j + 1
            if i == j:
                s = result.summaries[i]
                fig.add_trace(go.Histogram(x=data[:, i], histnorm="probability density", nbinsx=40, marker_color=_BLUE, showlegend=False),
                              row=row, col=col)
                for v, dash in ((s.median, "solid"), (s.lo68, "dash"), (s.hi68, "dash")):
                    fig.add_vline(x=v, line_dash=dash, line_color="rgba(60,60,60,0.7)", line_width=1, row=row, col=col)
                fig.update_yaxes(showticklabels=False, row=row, col=col)
            else:
                fig.add_trace(go.Histogram2dContour(x=data[:, j], y=data[:, i], ncontours=7, colorscale="Blues", showscale=False,
                                                    line=dict(width=0), contours=dict(coloring="fill"), showlegend=False),
                              row=row, col=col)
                ls = result.ls_theta
                if i < len(ls) and j < len(ls):
                    fig.add_trace(go.Scatter(x=[ls[j]], y=[ls[i]], mode="markers", marker=dict(symbol="x", size=8, color=_RED),
                                             showlegend=False, hoverinfo="skip"), row=row, col=col)
                if truth is not None and i < len(truth) and j < len(truth):
                    fig.add_trace(go.Scatter(x=[truth[j]], y=[truth[i]], mode="markers", marker=dict(symbol="diamond", size=8, color=_GREEN),
                                             showlegend=False, hoverinfo="skip"), row=row, col=col)
            if i == k - 1:
                fig.update_xaxes(title_text=names[j], row=row, col=col)
            if j == 0 and i > 0:
                fig.update_yaxes(title_text=names[i], row=row, col=col)
    fig.update_layout(height=max(420, 190 * k), title="Posterior: marginals (diagonal) and pairwise joint densities", bargap=0.02)
    return fig


def trace_figure(result: PosteriorResult, max_walkers: int = 8, max_points: int = 600) -> go.Figure:
    """Each parameter's value against step for a few walkers, with the burn-in marked: a healthy run looks like noise
    around a steady level once the burn-in is over."""
    k = len(result.names)
    steps = result.chain.shape[0]
    idx = np.linspace(0, steps - 1, min(steps, max_points)).astype(int)
    fig = make_subplots(rows=k, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=result.names)
    for d in range(k):
        for w in range(min(max_walkers, result.chain.shape[1])):
            fig.add_trace(go.Scatter(x=idx, y=result.chain[idx, w, d], mode="lines", line=dict(width=0.8, color=_BLUE), opacity=0.5,
                                     showlegend=False), row=d + 1, col=1)
        fig.add_vline(x=result.burn, line_dash="dash", line_color=_ORANGE, row=d + 1, col=1)
    fig.update_xaxes(title_text="step", row=k, col=1)
    fig.update_layout(height=max(300, 170 * k), title="Walker traces (dashed line: end of burn-in)")
    return fig


def predictive_figure(result: PosteriorResult, n_grid: int = 200) -> go.Figure:
    """The data with the posterior-predictive 68% and 95% bands (noise included) and the median curve."""
    lo, hi = float(result.xs.min()), float(result.xs.max())
    pad = 0.05 * (hi - lo or 1.0)
    grid = np.linspace(lo - pad, hi + pad, n_grid)
    centre, b = result.predictive(grid, include_noise=True)
    fig = go.Figure()
    for (a, c), fill, name in (((0, 3), "rgba(76,120,168,0.18)", "95% predictive"), ((1, 2), "rgba(76,120,168,0.38)", "68% predictive")):
        fig.add_trace(go.Scatter(x=np.concatenate([grid, grid[::-1]]), y=np.concatenate([b[c], b[a][::-1]]), fill="toself", fillcolor=fill,
                                 line=dict(width=0), name=name, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=grid, y=centre, mode="lines", line=dict(color=_BLUE, width=2), name="posterior median"))
    fig.add_trace(go.Scatter(x=grid, y=result.model(grid, result.ls_theta), mode="lines", line=dict(color=_RED, width=1.5, dash="dash"),
                             name="least squares"))
    fig.add_trace(go.Scatter(x=result.xs, y=result.ys, mode="markers", marker=dict(color="black", size=6), name="data"))
    fig.update_layout(height=420, title=f"Posterior predictive: {result.predictive_coverage:.0%} of the data inside the 95% band",
                      xaxis_title="x", yaxis_title="y", legend=dict(orientation="h", y=-0.2))
    return fig
