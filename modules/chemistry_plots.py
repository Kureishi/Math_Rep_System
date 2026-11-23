"""Plotly figures for the chemistry mode -- Streamlit-free, colours from modules/palette.py (never pure black or white)."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules import chemistry as ch
from modules.palette import BLUE, CATEGORICAL, GREEN, INK, PURPLE, VERMILLION


def titration_figure(curve: ch.TitrationCurve) -> go.Figure:
    """pH against titrant volume, with the equivalence points (diamonds) and the half-equivalence points (circles) against their pKa."""
    fig = go.Figure(go.Scatter(x=curve.volume_ml, y=curve.ph, mode="lines", name="pH", line=dict(color=BLUE, width=3)))
    if curve.equivalence:
        fig.add_trace(go.Scatter(x=[v for v, _ in curve.equivalence], y=[p for _, p in curve.equivalence], mode="markers+text", name="equivalence point",
                                 marker=dict(symbol="diamond", size=12, color=VERMILLION), text=[f"pH {p:.2f}" for _, p in curve.equivalence],
                                 textposition="bottom right"))
    if curve.half_equivalence:
        fig.add_trace(go.Scatter(x=[v for v, _, _ in curve.half_equivalence], y=[p for _, p, _ in curve.half_equivalence], mode="markers", name="half-equivalence",
                                 marker=dict(symbol="circle", size=10, color=GREEN)))
        for _, _, pka in curve.half_equivalence:
            fig.add_hline(y=pka, line_dash="dot", line_color=INK, line_width=1, annotation_text=f"pKa {pka:g}", annotation_position="left")
    fig.update_layout(height=420, xaxis_title="titrant added (mL)", yaxis_title="pH", yaxis=dict(range=[0, 14]),
                      title="Titration curve: the exact charge balance solved at every point", legend=dict(orientation="h", y=-0.2))
    return fig


def distribution_figure(pkas: list[float], marker_ph: float | None = None) -> go.Figure:
    """Which form of the acid is present at each pH (the species-distribution diagram): the curves cross at each pKa."""
    ph = np.linspace(0, 14, 281)
    alphas = ch.alpha_fractions(10.0 ** (-ph), pkas)
    n = len(pkas)
    fig = go.Figure()
    for i, a in enumerate(alphas):
        name = "H" + (f"{n - i}" if n - i > 1 else "") + "A" + (f"^{i}-" if i > 1 else "^-" if i == 1 else "") if n > 1 else ("HA" if i == 0 else "A⁻")
        fig.add_trace(go.Scatter(x=ph, y=a, mode="lines", name=name, line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=3)))
    for p in pkas:
        fig.add_vline(x=p, line_dash="dot", line_color=INK, line_width=1, annotation_text=f"pKa {p:g}", annotation_position="top")
    if marker_ph is not None:
        fig.add_vline(x=marker_ph, line_color=VERMILLION, line_width=2, annotation_text=f"pH {marker_ph:.2f}", annotation_position="bottom")
    fig.update_layout(height=360, xaxis_title="pH", yaxis_title="fraction of the acid", title="Species distribution", legend=dict(orientation="h", y=-0.25))
    return fig


def kinetics_figure(report: ch.KineticsReport) -> go.Figure:
    """The data with each order's fit (the best one thick), and the best fit's residuals underneath."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.06)
    t, c = report.t, report.c
    fig.add_trace(go.Scatter(x=t, y=c, mode="markers", name="data", marker=dict(color=INK, size=8)), row=1, col=1)
    grid = np.linspace(float(t.min()), float(t.max()), 200)
    colours = {0: PURPLE, 1: BLUE, 2: VERMILLION}
    for rank, f in enumerate(report.fits):
        fig.add_trace(go.Scatter(x=grid, y=ch.rate_curve(f.order, f.c0, f.k, grid), mode="lines", name=f"{ch.ORDER_NAMES[f.order]}" + (" (best)" if rank == 0 else ""),
                                 line=dict(color=colours[f.order], width=4 if rank == 0 else 2, dash="solid" if rank == 0 else "dash")), row=1, col=1)
    best = report.best
    fig.add_trace(go.Scatter(x=t, y=c - ch.rate_curve(best.order, best.c0, best.k, t), mode="markers+lines", name="residual (best fit)",
                             marker=dict(color=colours[best.order], size=7), line=dict(color=colours[best.order], width=1)), row=2, col=1)
    fig.add_hline(y=0, line_color=INK, line_width=1, row=2, col=1)
    fig.update_xaxes(title_text="time", row=2, col=1)
    fig.update_yaxes(title_text="concentration", row=1, col=1)
    fig.update_yaxes(title_text="residual", row=2, col=1)
    fig.update_layout(height=520, title="Concentration against time: three rate laws compared", legend=dict(orientation="h", y=-0.15))
    return fig


def linearised_figure(report: ch.KineticsReport) -> go.Figure:
    """The textbook diagnostic: [A], ln[A] and 1/[A] against time. The straight one names the order (when the data allow)."""
    t, c = report.t, report.c
    fits = {f.order: f for f in report.fits}
    panels = [(0, "[A]", c), (1, "ln [A]", np.log(np.where(c > 0, c, np.nan))), (2, "1 / [A]", 1.0 / np.where(c > 0, c, np.nan))]
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.1,
                        subplot_titles=[f"{label}: R² = {fits[o].r2_linearised:.4f}" if o in fits and fits[o].r2_linearised is not None else label for o, label, _ in panels])
    for col, (order, label, y) in enumerate(panels, start=1):
        ok = np.isfinite(y)
        fig.add_trace(go.Scatter(x=t[ok], y=y[ok], mode="markers", marker=dict(color=INK, size=7), name=label, showlegend=False), row=1, col=col)
        if ok.sum() >= 2:
            slope, intercept = np.polyfit(t[ok], y[ok], 1)
            fig.add_trace(go.Scatter(x=t[ok], y=slope * t[ok] + intercept, mode="lines", line=dict(color=BLUE, width=2), showlegend=False, name=f"{label} fit"), row=1, col=col)
        fig.update_xaxes(title_text="time", row=1, col=col)
    fig.update_layout(height=320, title="Linearised plots (for reference; the nonlinear fit decides the order)")
    return fig


def arrhenius_figure(fit: ch.ArrheniusFit) -> go.Figure:
    inv = 1.0 / fit.temperatures
    grid = np.linspace(inv.min(), inv.max(), 50)
    slope = -fit.ea_kj * 1000.0 / ch.R_GAS
    fig = go.Figure([go.Scatter(x=inv, y=fit.ln_k, mode="markers", name="data", marker=dict(color=INK, size=9)),
                     go.Scatter(x=grid, y=slope * grid + fit.ln_a, mode="lines", name=f"Ea = {fit.ea_kj:.1f} ± {fit.ea_se_kj:.1f} kJ/mol", line=dict(color=BLUE, width=3))])
    fig.update_layout(height=360, xaxis_title="1 / T  (1/K)", yaxis_title="ln k", title="Arrhenius plot: the slope is −Ea / R", legend=dict(orientation="h", y=-0.25))
    return fig


def stoichiometry_figure(result: ch.StoichResult) -> go.Figure:
    """Moles of each species before and after the reaction goes to completion."""
    rows = [r for r in result.rows if r.final_mol is not None]
    labels = [r.label for r in rows]
    fig = go.Figure([go.Bar(name="before", x=labels, y=[r.initial_mol or 0.0 for r in rows], marker_color=INK),
                     go.Bar(name="after", x=labels, y=[r.final_mol or 0.0 for r in rows], marker_color=BLUE)])
    fig.update_layout(barmode="group", height=340, yaxis_title="moles", title="Amounts before and after (the limiting reagent runs out)", legend=dict(orientation="h", y=-0.2))
    return fig
