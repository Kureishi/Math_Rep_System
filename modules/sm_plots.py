"""
Plotly figures for the statistical-mechanics tools -- Streamlit-free. Each takes a result object of its module and returns a
plain go.Figure (the app's theme and the HTML report each apply their own styling), as in modules/quantum_plots.py.
"""
import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules import sm_distributions as sd
from modules import sm_randommatrix as rm
from modules.plotter import _frame_indices, _play_layout
from modules.sm_lattice import ScanResult, SingleRun
from modules.sm_meanfield import ComparisonCurves
from modules.sm_partition import Spectrum, Thermo
from modules.sm_stochastic import FPSolution, LangevinResult, msd, msd_exact_free

_PAL = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]


# ------------------------------------------------------------------------------------------ partition functions

def thermo_figure(th: Thermo, spec: Spectrum | None = None) -> go.Figure:
    """ln Z, F, U, S/k and C/k against kT; the closed form (if any) as dashed lines over the numerical sum."""
    names = ["ln Z", "F", "U", "S / k", "C / k"]
    vals = [th.lnZ, th.F, th.U, th.S / th.kB, th.C / th.kB]
    fig = make_subplots(rows=2, cols=3, subplot_titles=names + [""], horizontal_spacing=0.08, vertical_spacing=0.15)
    keys = [None, "F", "U", "S", "C"]
    for i, (v, key) in enumerate(zip(vals, keys)):
        r, c = divmod(i, 3)
        fig.add_trace(go.Scatter(x=th.kT, y=v, mode="lines", name="numerical sum", line=dict(color=_PAL[0], width=2), showlegend=(i == 0)), row=r + 1, col=c + 1)
        if spec is not None and spec.closed_form is not None and key is not None:
            ref = np.asarray(spec.closed_form.funcs[key](th.kT), dtype=float)
            fig.add_trace(go.Scatter(x=th.kT, y=ref, mode="lines", name="closed form", line=dict(color=_PAL[1], dash="dash"), showlegend=(i == 1)), row=r + 1, col=c + 1)
        fig.update_xaxes(title_text="kT", row=r + 1, col=c + 1)
    fig.update_layout(height=620, title="Thermodynamic functions from the partition function")
    return fig


def populations_figure(th: Thermo) -> go.Figure:
    fig = go.Figure()
    if th.populations is None or th.energies_shown is None:
        return fig
    for j in range(th.populations.shape[1]):
        fig.add_trace(go.Scatter(x=th.kT, y=th.populations[:, j], mode="lines", name=f"E = {th.energies_shown[j]:.4g}", line=dict(color=_PAL[j % len(_PAL)])))
    fig.update_layout(xaxis_title="kT", yaxis_title="probability of the level", title="Level populations", height=380)
    return fig


# ------------------------------------------------------------------------------------------ lattice

def lattice_scan_figure(scans: dict[int, ScanResult], tc: float | None = None) -> go.Figure:
    titles = ["energy per site e", "order parameter ⟨|m|⟩", "susceptibility χ'", "specific heat C", "Binder cumulant U₄", "autocorrelation time τ (order parameter)"]
    fig = make_subplots(rows=2, cols=3, subplot_titles=titles, horizontal_spacing=0.08, vertical_spacing=0.17)
    cols = [("e", "e_err"), ("m", "m_err"), ("chi", "chi_err"), ("C", "C_err"), ("U4", "U4_err"), ("tau_m", None)]
    for k, L in enumerate(sorted(scans)):
        sc = scans[L]
        for i, (val, err) in enumerate(cols):
            r, c = divmod(i, 3)
            fig.add_trace(go.Scatter(x=sc.temps, y=sc.column(val), mode="lines+markers", name=f"L = {L}", legendgroup=str(L), showlegend=(i == 0),
                                     line=dict(color=_PAL[k % len(_PAL)]), marker=dict(size=5),
                                     error_y=dict(type="data", array=sc.column(err), visible=True) if err else None), row=r + 1, col=c + 1)
    if tc is not None:
        for i in range(6):
            r, c = divmod(i, 3)
            fig.add_vline(x=tc, line=dict(color="gray", dash="dot"), row=r + 1, col=c + 1)
    fig.update_xaxes(title_text="T")
    fig.update_layout(height=700, title="Metropolis scan (error bars include the autocorrelation time)")
    return fig


def lattice_animation_figure(run: SingleRun, max_frames: int = 80) -> go.Figure:
    """The lattice as a heat map (spin value, Potts colour or XY angle) with Play / Pause."""
    model = run.model
    idx = _frame_indices(len(run.snapshots), max_frames)
    zmin, zmax, scale = (-1, 1, "RdBu") if model.model == "ising" else (0, model.q - 1, "Viridis") if model.model == "potts" else (0, 2 * math.pi, "HSV")
    first = run.snapshots[idx[0]][1]
    fig = go.Figure(data=[go.Heatmap(z=first, zmin=zmin, zmax=zmax, colorscale=scale, showscale=False)])
    fig.frames = [go.Frame(data=[go.Heatmap(z=run.snapshots[i][1], zmin=zmin, zmax=zmax, colorscale=scale, showscale=False)], name=str(k)) for k, i in enumerate(idx)]
    fig.update_layout(height=520, yaxis=dict(scaleanchor="x", autorange="reversed"), title=f"{model.model.upper()} {model.L}×{model.L} at T = {run.T:g}",
                      **_play_layout([f"sweep {run.snapshots[i][0]}" for i in idx], "", 120))
    return fig


def series_figure(run: SingleRun) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["energy per site", "order parameter"])
    s = np.arange(1, len(run.energy) + 1)
    fig.add_trace(go.Scatter(x=s, y=run.energy, mode="lines", line=dict(color=_PAL[0]), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=s, y=run.order, mode="lines", line=dict(color=_PAL[1]), showlegend=False), row=1, col=2)
    fig.update_xaxes(title_text="sweep")
    fig.update_layout(height=320)
    return fig


def fss_figure(fit, label: str = "") -> go.Figure:
    sizes = sorted(fit.chi_max)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["χ'_max against L (slope γ/ν)", "|m|(T_c) against L (slope −β/ν)"])
    fig.add_trace(go.Scatter(x=sizes, y=[fit.chi_max[L] for L in sizes], mode="markers+lines", showlegend=False, line=dict(color=_PAL[0])), row=1, col=1)
    fig.add_trace(go.Scatter(x=sizes, y=[fit.m_at_tc[L] for L in sizes], mode="markers+lines", showlegend=False, line=dict(color=_PAL[1])), row=1, col=2)
    fig.update_xaxes(type="log", title_text="L")
    fig.update_yaxes(type="log")
    fig.update_layout(height=340, title=label)
    return fig


# ------------------------------------------------------------------------------------------ mean field

def meanfield_figure(cv: ComparisonCurves, tc_mf: float, tc_bethe: float, tc_exact: float, mc: ScanResult | None = None) -> go.Figure:
    fig = make_subplots(rows=1, cols=3, subplot_titles=["spontaneous magnetisation m(T)", "energy per site u(T)", "specific heat c(T)"], horizontal_spacing=0.07)
    fig.add_trace(go.Scatter(x=cv.T, y=cv.m_mf, name="mean field", line=dict(color=_PAL[3])), row=1, col=1)
    fig.add_trace(go.Scatter(x=cv.T, y=cv.m_bethe, name="Bethe", line=dict(color=_PAL[1], dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=cv.T, y=cv.m_exact_2d, name="Onsager (exact)", line=dict(color=_PAL[0], width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=cv.T, y=cv.u_mf, name="mean field", showlegend=False, line=dict(color=_PAL[3])), row=1, col=2)
    fig.add_trace(go.Scatter(x=cv.T, y=cv.u_exact_2d, name="Onsager", showlegend=False, line=dict(color=_PAL[0], width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=cv.T, y=cv.u_exact_1d, name="1D exact", line=dict(color=_PAL[2], dash="dot")), row=1, col=2)
    fig.add_trace(go.Scatter(x=cv.T, y=cv.c_mf, name="mean field", showlegend=False, line=dict(color=_PAL[3])), row=1, col=3)
    fig.add_trace(go.Scatter(x=cv.T, y=np.clip(cv.c_exact_2d, 0, 6), name="Onsager", showlegend=False, line=dict(color=_PAL[0], width=3)), row=1, col=3)
    if mc is not None:
        fig.add_trace(go.Scatter(x=mc.temps, y=mc.column("m"), mode="markers", name=f"Monte Carlo L = {mc.model.L}", marker=dict(color="black", size=6),
                                 error_y=dict(type="data", array=mc.column("m_err"), visible=True)), row=1, col=1)
        fig.add_trace(go.Scatter(x=mc.temps, y=mc.column("e"), mode="markers", showlegend=False, marker=dict(color="black", size=6)), row=1, col=2)
        fig.add_trace(go.Scatter(x=mc.temps, y=mc.column("C"), mode="markers", showlegend=False, marker=dict(color="black", size=6)), row=1, col=3)
    for col in (1, 2, 3):
        for x, color in ((tc_mf, _PAL[3]), (tc_bethe, _PAL[1]), (tc_exact, _PAL[0])):
            fig.add_vline(x=x, line=dict(color=color, dash="dot", width=1), row=1, col=col)
    fig.update_xaxes(title_text="T / J")
    fig.update_layout(height=420, title="Where mean field fails (dotted lines: T_c of mean field, Bethe, exact)")
    return fig


def landau_figure(m: np.ndarray, curves: dict[float, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    for i, (t, f) in enumerate(sorted(curves.items())):
        fig.add_trace(go.Scatter(x=m, y=f - f.min(), mode="lines", name=f"T = {t:g}", line=dict(color=_PAL[i % len(_PAL)])))
    fig.update_layout(xaxis_title="m", yaxis_title="f(m) − min f", title="Mean-field (Landau) free energy", height=360, yaxis_range=[0, None])
    return fig


# ------------------------------------------------------------------------------------------ distributions

def occupation_figure(eps: np.ndarray, mu: float, T: float) -> go.Figure:
    fig = go.Figure()
    for i, k in enumerate(sd.KINDS):
        mask = eps > mu if k == "BE" else np.ones_like(eps, dtype=bool)
        fig.add_trace(go.Scatter(x=eps[mask], y=sd.occupation(eps[mask], mu, T, k), mode="lines", name=k, line=dict(color=_PAL[i], width=2)))
    fig.update_layout(xaxis_title="ε", yaxis=dict(title="mean occupation", range=[0, 3]), title=f"Occupation at μ = {mu:g}, T = {T:g}", height=400)
    return fig


def mu_figure(temps: np.ndarray, curves: dict[str, np.ndarray], t_c: float | None = None) -> go.Figure:
    fig = go.Figure()
    for i, k in enumerate(sd.KINDS):
        fig.add_trace(go.Scatter(x=temps, y=curves[k], mode="lines", name=k, line=dict(color=_PAL[i], width=2)))
    if t_c:
        fig.add_vline(x=t_c, line=dict(color="gray", dash="dot"), annotation_text="T_c (BEC)")
    fig.update_layout(xaxis_title="T", yaxis_title="μ", title="Chemical potential at fixed density", height=400)
    return fig


def eos_figure(x: np.ndarray, eos: dict[str, np.ndarray]) -> go.Figure:
    fig = go.Figure()
    for i, k in enumerate(sd.KINDS):
        fig.add_trace(go.Scatter(x=x, y=eos[k], mode="lines", name=k, line=dict(color=_PAL[i], width=2)))
    fig.update_layout(xaxis_title="n λ³ / g", yaxis_title="P / (n T)", title="Equation of state: quantum corrections to P = nT", height=400)
    return fig


def mb_figure(v: np.ndarray, T: float, samples: np.ndarray | None = None) -> go.Figure:
    fig = go.Figure()
    if samples is not None:
        fig.add_trace(go.Histogram(x=samples, histnorm="probability density", nbinsx=80, name="sample", marker_color="rgba(76,120,168,0.4)"))
    fig.add_trace(go.Scatter(x=v, y=sd.mb_speed_pdf(v, T), name="Maxwell", line=dict(color=_PAL[1], width=3)))
    fig.update_layout(xaxis_title="speed v", yaxis_title="p(v)", title="Maxwell–Boltzmann speed distribution", height=360)
    return fig


# ------------------------------------------------------------------------------------------ stochastic

def paths_figure(res: LangevinResult, n_show: int = 8) -> go.Figure:
    fig = go.Figure()
    for j in range(min(n_show, res.x.shape[1])):
        fig.add_trace(go.Scatter(x=res.t, y=res.x[:, j], mode="lines", showlegend=False, line=dict(width=1, color=_PAL[j % len(_PAL)])))
    fig.update_layout(xaxis_title="t", yaxis_title="x", title=f"{res.kind.capitalize()} Langevin paths in {res.potential.name}", height=380)
    return fig


def msd_figure(res: LangevinResult) -> go.Figure:
    m = msd(res)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res.t[1:], y=m[1:], mode="markers", name="simulation", marker=dict(color=_PAL[0], size=5)))
    if res.potential.text.strip() == "0":
        ex = msd_exact_free(res.t, res.kT, res.gamma, None if res.kind == "overdamped" else res.mass)
        fig.add_trace(go.Scatter(x=res.t[1:], y=ex[1:], mode="lines", name="exact", line=dict(color=_PAL[1], width=2)))
        fig.add_trace(go.Scatter(x=res.t[1:], y=2 * res.kT / res.gamma * res.t[1:], mode="lines", name="2Dt (diffusive)", line=dict(color="gray", dash="dash")))
        if res.kind == "underdamped":
            fig.add_trace(go.Scatter(x=res.t[1:], y=res.kT / res.mass * res.t[1:] ** 2, mode="lines", name="(kT/m)t² (ballistic)", line=dict(color="gray", dash="dot")))
    fig.update_layout(xaxis=dict(title="t", type="log"), yaxis=dict(title="⟨(x(t) − x(0))²⟩", type="log"), title="Mean-squared displacement", height=400)
    return fig


def fp_figure(sol: FPSolution, res: LangevinResult | None = None, exact: dict[int, np.ndarray] | None = None, n_curves: int = 5, bins: int = 60) -> go.Figure:
    fig = go.Figure()
    idx = np.unique(np.linspace(0, len(sol.t) - 1, n_curves).astype(int))
    for k, j in enumerate(idx):
        fig.add_trace(go.Scatter(x=sol.x, y=sol.P[j], mode="lines", name=f"Fokker–Planck, t = {sol.t[j]:.3g}", line=dict(color=_PAL[k % len(_PAL)], width=2)))
        if exact and j in exact:
            fig.add_trace(go.Scatter(x=sol.x, y=exact[j], mode="lines", name="exact Gaussian", showlegend=(k == 0), line=dict(color="black", dash="dot")))
    if res is not None:
        hist, edges = np.histogram(res.x[-1], bins=bins, range=(float(sol.x[0]), float(sol.x[-1])), density=True)
        fig.add_trace(go.Bar(x=0.5 * (edges[1:] + edges[:-1]), y=hist, name=f"Langevin ensemble, t = {res.t[-1]:.3g}", marker_color="rgba(0,0,0,0.25)"))
    fig.update_layout(xaxis_title="x", yaxis_title="P(x, t)", title="Probability density: Fokker–Planck against Langevin", height=420, barmode="overlay")
    return fig


# ------------------------------------------------------------------------------------------ random matrices

def density_figure(eigs: np.ndarray, kind: str) -> go.Figure:
    x = np.linspace(-2.2, 2.2, 600)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=eigs.ravel(), histnorm="probability density", nbinsx=70, name=f"{kind} eigenvalues", marker_color="rgba(76,120,168,0.45)"))
    fig.add_trace(go.Scatter(x=x, y=rm.semicircle_pdf(x), name="Wigner semicircle", line=dict(color=_PAL[1], width=3)))
    fig.update_layout(xaxis_title="λ", yaxis_title="density", title="Eigenvalue density", height=380)
    return fig


def spacing_figure(spacings: np.ndarray, kind: str) -> go.Figure:
    s = np.linspace(0, 4, 400)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=spacings, histnorm="probability density", xbins=dict(start=0, end=4, size=0.1), name=f"{kind} spacings", marker_color="rgba(76,120,168,0.45)"))
    for i, k in enumerate(rm.KINDS):
        fig.add_trace(go.Scatter(x=s, y=rm.surmise_pdf(k, s), name={"GOE": "GOE surmise", "GUE": "GUE surmise", "Poisson": "Poisson"}[k],
                                 line=dict(color=_PAL[(i + 1) % len(_PAL)], width=2)))
    fig.update_layout(xaxis_title="unfolded spacing s", yaxis_title="p(s)", title="Nearest-neighbour level spacings", height=400)
    return fig


def wishart_figure(eigs: np.ndarray, c: float) -> go.Figure:
    lo, hi = (1 - math.sqrt(c)) ** 2, (1 + math.sqrt(c)) ** 2
    x = np.linspace(lo * 0.9, hi * 1.05, 500)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=eigs.ravel(), histnorm="probability density", nbinsx=70, name="Wishart eigenvalues", marker_color="rgba(76,120,168,0.45)"))
    fig.add_trace(go.Scatter(x=x, y=rm.marchenko_pastur_pdf(x, c), name="Marchenko–Pastur", line=dict(color=_PAL[1], width=3)))
    fig.update_layout(xaxis_title="λ", yaxis_title="density", title=f"Wishart spectrum, c = {c:.3g}", height=380)
    return fig
