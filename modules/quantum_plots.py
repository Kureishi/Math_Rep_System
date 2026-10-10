"""
Plotly figures for the quantum-mechanics features -- Streamlit-free, and each takes the result object of its
module and nothing else, so the same figure appears in the app and in an exported report.

They are plain `go.Figure`s with no styling beyond what makes them readable, because the app's theme and the
HTML report each apply their own.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.quantum_1d import EigenResult
from modules.quantum_dynamics import DynamicsResult, transmission_curve
from modules.quantum_operators import BCHResult
from modules.quantum_perturbation import Comparison
from modules.quantum_qubits import QubitRun, bloch_to_density

_PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#EECA3B"]


# ------------------------------------------------------------------------------------------ eigenstates

def eigenstates_figure(result: EigenResult, show: str = "wavefunction", levels: list[int] | None = None,
                       scale: float | None = None) -> go.Figure:
    """The potential, each chosen level as a horizontal line at its energy, and its wavefunction (or
    probability density) drawn on that line -- the textbook ladder picture. `scale` is the amplitude in
    energy units (default: a third of the level spacing)."""
    which = list(range(result.n_levels)) if levels is None else [i for i in levels if 0 <= i < result.n_levels]
    energies = result.energies
    spacing = float(np.median(np.diff(energies))) if len(energies) > 1 else 1.0
    amp = scale if scale else 0.4 * abs(spacing)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.x, y=result.potential, mode="lines", name="V(x)", line=dict(color="gray", width=2)))
    for i in which:
        color = _PALETTE[i % len(_PALETTE)]
        psi = result.states[:, i]
        curve = psi ** 2 / np.max(psi ** 2) if show == "density" else psi / np.max(np.abs(psi))
        fig.add_trace(go.Scatter(x=[result.x[0], result.x[-1]], y=[energies[i]] * 2, mode="lines", showlegend=False,
                                 line=dict(color=color, dash="dot", width=1), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=result.x, y=energies[i] + amp * curve, mode="lines", name=f"n = {i}, E = {energies[i]:.5g}",
                                 line=dict(color=color, width=2)))
    top = max(float(energies[which[-1]]) + 1.5 * abs(spacing), float(energies[which[-1]]) + amp * 1.2) if which else None
    low = min(float(np.min(result.potential)), float(energies[0]) - amp)
    fig.update_layout(title="Energy levels and " + ("probability densities" if show == "density" else "wavefunctions"),
                      xaxis_title="x", yaxis_title="Energy", yaxis=dict(range=[low, top]) if top is not None else {})
    return fig


def convergence_figure(result: EigenResult) -> go.Figure:
    """Computed against analytic energies, as relative error per level (log scale)."""
    if result.analytic is None:
        return go.Figure()
    k = min(len(result.analytic), result.n_levels)
    rel = np.abs(result.energies[:k] - result.analytic[:k]) / np.maximum(np.abs(result.analytic[:k]), 1e-12)
    fig = go.Figure(go.Bar(x=list(range(k)), y=np.maximum(rel, 1e-17), marker_color=_PALETTE[0]))
    fig.update_layout(title="Relative error against the analytic spectrum", xaxis_title="level n",
                      yaxis_title="|E - E_analytic| / |E_analytic|", yaxis_type="log")
    return fig


# ------------------------------------------------------------------------------------------ wavepacket

def wavepacket_animation_figure(result: DynamicsResult, show_real: bool = True) -> go.Figure:
    """|ψ|² through time, with the potential on a second axis, as an animation with a time slider."""
    x = result.x
    dens, real = result.density.astype(np.float32), result.real_part.astype(np.float32)      # frames repeat per trace: keep them small
    peak = float(np.max(dens)) or 1.0
    v = result.potential
    v_span = float(np.max(v) - np.min(v))
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=x, y=v, name="V(x)", fill="tozeroy", line=dict(color="rgba(120,120,120,0.8)"),
                             fillcolor="rgba(150,150,150,0.25)"), secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=dens[0], name="|ψ|²", line=dict(color=_PALETTE[0], width=3)), secondary_y=False)
    if show_real:
        rpeak = float(np.max(np.abs(real))) or 1.0
        fig.add_trace(go.Scatter(x=x, y=real[0] * (0.5 * peak / rpeak), name="Re ψ (scaled)", line=dict(color=_PALETTE[1], width=1)),
                      secondary_y=False)
    frames = []
    for i, t in enumerate(result.frame_times):
        data = [go.Scatter(y=dens[i])]                                    # x is unchanged between frames
        if show_real:
            data.append(go.Scatter(y=real[i] * (0.5 * peak / rpeak)))
        frames.append(go.Frame(data=data, traces=[1, 2] if show_real else [1], name=f"{t:.3f}"))
    fig.frames = frames
    fig.update_layout(
        title="Wavepacket evolution", xaxis_title="x", yaxis_title="|ψ|²",
        updatemenus=[dict(type="buttons", x=0.02, y=-0.15, buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True)]),
            dict(label="⏸ Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, x=0.18, len=0.8, pad=dict(t=40),
                      steps=[dict(label=f"{t:.2f}", method="animate", args=[[f"{t:.3f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for t in result.frame_times], currentvalue=dict(prefix="t = "))])
    fig.update_yaxes(range=[-0.6 * peak if show_real else 0, 1.15 * peak], secondary_y=False)
    pad = 0.15 * v_span if v_span else 1.0
    fig.update_yaxes(range=[float(np.min(v)) - pad, float(np.max(v)) + pad * 4], title_text="V(x)", secondary_y=True, showgrid=False)
    return fig


def dynamics_diagnostics_figure(result: DynamicsResult) -> go.Figure:
    """Norm drift, energy drift, and Ehrenfest's theorem on one figure."""
    t = result.series_times
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Norm − 1", "Energy ⟨H⟩ − E₀", "Ehrenfest: ⟨x⟩ and ∫⟨p⟩/m dt"))
    fig.add_trace(go.Scatter(x=t, y=result.norm - 1.0, name="norm − 1", line=dict(color=_PALETTE[0])), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=result.energy - result.energy[0], name="E − E₀", line=dict(color=_PALETTE[1])), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=result.mean_x, name="⟨x⟩ (simulation)", line=dict(color=_PALETTE[2])), row=1, col=3)
    integral = result.mean_x[0] + np.concatenate([[0.0], np.cumsum(0.5 * (result.mean_p[1:] + result.mean_p[:-1]) / result.mass * np.diff(t))])
    fig.add_trace(go.Scatter(x=t, y=integral, name="x₀ + ∫⟨p⟩/m dt", line=dict(color=_PALETTE[3], dash="dash")), row=1, col=3)
    fig.update_xaxes(title_text="t")
    fig.update_layout(showlegend=True, height=340)
    return fig


def region_probability_figure(result: DynamicsResult) -> go.Figure:
    if result.region_probability is None:
        return go.Figure()
    t = result.frame_times
    rows = result.region_probability
    fig = go.Figure()
    for j, (name, color) in enumerate((("left of the region", _PALETTE[0]), ("in/near the region", _PALETTE[3]), ("right of the region", _PALETTE[2]))):
        fig.add_trace(go.Scatter(x=t, y=rows[:, j], name=name, stackgroup="one", line=dict(color=color)))
    if result.predicted:
        fig.add_hline(y=result.predicted["transmitted"], line_dash="dot", annotation_text="predicted transmission", line_color=_PALETTE[2])
    fig.update_layout(title="Where the probability is", xaxis_title="t", yaxis_title="probability", yaxis=dict(range=[0, 1.02]))
    return fig


def transmission_figure(result: DynamicsResult, hbar: float = 1.0, mass: float = 1.0) -> go.Figure:
    """T(E) and R(E) for the piecewise-constant potential, the packet's energy band, and the simulated T."""
    spec = result.spec
    if not spec.get("boundaries") or not spec.get("levels"):
        return go.Figure()
    k0, sigma = spec["k0"], spec["sigma"]
    levels = spec["levels"]
    e_centre = levels[0] + hbar ** 2 * k0 ** 2 / (2 * mass)
    e_hi = max(e_centre * 2.2, max(levels) * 1.6 if max(levels) > 0 else 1.0)
    energies = np.linspace(1e-3, e_hi, 400)
    t_curve = transmission_curve(energies, spec["boundaries"], levels, hbar, mass)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=energies, y=t_curve, name="T(E), exact", line=dict(color=_PALETTE[2], width=3)))
    fig.add_trace(go.Scatter(x=energies, y=1 - t_curve, name="R(E)", line=dict(color=_PALETTE[0], width=1.5, dash="dot")))
    de = hbar ** 2 * k0 / (mass * 2 * sigma)                            # dE = ħ²k0 Δk / m with Δk = 1/(2σ)
    fig.add_vrect(x0=max(e_centre - de, 0), x1=e_centre + de, fillcolor="rgba(245,133,24,0.18)", line_width=0,
                  annotation_text="packet's energy spread (±1 std)", annotation_position="top left")
    fig.add_vline(x=e_centre, line_dash="dash", line_color=_PALETTE[1])
    if result.predicted:
        fig.add_trace(go.Scatter(x=[e_centre], y=[result.predicted["transmitted"]], mode="markers", name="packet-averaged T (exact)",
                                 marker=dict(size=13, symbol="diamond", color=_PALETTE[1])))
        if "right" in result.final_probabilities:
            fig.add_trace(go.Scatter(x=[e_centre], y=[result.final_probabilities["right"]], mode="markers", name="T from the simulation",
                                     marker=dict(size=12, symbol="x", color="black")))
    fig.update_layout(title="Transmission through the potential", xaxis_title="energy E", yaxis_title="probability", yaxis=dict(range=[-0.02, 1.05]))
    return fig


# ------------------------------------------------------------------------------------------ qubits

def _sphere_wireframe() -> list[go.Scatter3d]:
    traces = []
    u = np.linspace(0, 2 * np.pi, 80)
    for lat in np.deg2rad([-60, -30, 0, 30, 60]):
        traces.append(go.Scatter3d(x=np.cos(lat) * np.cos(u), y=np.cos(lat) * np.sin(u), z=np.full_like(u, np.sin(lat)),
                                   mode="lines", line=dict(color="rgba(150,150,150,0.35)", width=1), hoverinfo="skip", showlegend=False))
    for lon in np.deg2rad(np.arange(0, 180, 30)):
        v = np.linspace(-np.pi / 2, np.pi / 2, 60)
        for sgn in (1, -1):
            traces.append(go.Scatter3d(x=sgn * np.cos(v) * np.cos(lon), y=sgn * np.cos(v) * np.sin(lon), z=np.sin(v), mode="lines",
                                       line=dict(color="rgba(150,150,150,0.25)", width=1), hoverinfo="skip", showlegend=False))
    for axis, label, pos, neg in ((0, "x", "|+⟩", "|−⟩"), (1, "y", "|+i⟩", "|−i⟩"), (2, "z", "|0⟩", "|1⟩")):
        for sgn, name in ((1, pos), (-1, neg)):
            point = np.zeros(3)
            point[axis] = sgn * 1.0
            traces.append(go.Scatter3d(x=[0, 1.12 * point[0]], y=[0, 1.12 * point[1]], z=[0, 1.12 * point[2]], mode="lines+text",
                                       text=["", name], textposition="top center", line=dict(color="rgba(90,90,90,0.8)", width=2),
                                       hoverinfo="skip", showlegend=False))
    return traces


def bloch_sphere_figure(run: QubitRun, max_frames: int = 120) -> go.Figure:
    """The Bloch sphere with the noisy path (coloured by time), the ideal path, and an animated state vector."""
    n = len(run.times)
    idx = np.unique(np.linspace(0, n - 1, min(max_frames, n)).astype(int))
    fig = go.Figure(_sphere_wireframe())
    base = len(fig.data)
    b, ideal = run.bloch, run.ideal
    fig.add_trace(go.Scatter3d(x=ideal[:, 0], y=ideal[:, 1], z=ideal[:, 2], mode="lines", name="ideal (no noise)",
                               line=dict(color="rgba(80,80,80,0.6)", width=4, dash="dash")))
    fig.add_trace(go.Scatter3d(x=b[:, 0], y=b[:, 1], z=b[:, 2], mode="lines", name="with noise",
                               line=dict(color=run.times, colorscale="Viridis", width=6)))
    fig.add_trace(go.Scatter3d(x=[0, b[0, 0]], y=[0, b[0, 1]], z=[0, b[0, 2]], mode="lines+markers", name="state",
                               line=dict(color=_PALETTE[3], width=8), marker=dict(size=[0, 7], color=_PALETTE[3])))
    frames = [go.Frame(data=[go.Scatter3d(x=[0, b[i, 0]], y=[0, b[i, 1]], z=[0, b[i, 2]])], traces=[base + 2], name=f"{run.times[i]:.3f}")
              for i in idx]
    fig.frames = frames
    fig.update_layout(
        title="Bloch sphere", margin=dict(l=0, r=0, t=40, b=0), height=520,
        scene=dict(aspectmode="cube", xaxis=dict(visible=False, range=[-1.3, 1.3]), yaxis=dict(visible=False, range=[-1.3, 1.3]),
                   zaxis=dict(visible=False, range=[-1.3, 1.3])),
        updatemenus=[dict(type="buttons", x=0.02, y=0.02, buttons=[
            dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)]),
            dict(label="⏸ Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
        sliders=[dict(active=0, x=0.15, len=0.8, currentvalue=dict(prefix="t = "),
                      steps=[dict(label=f"{run.times[i]:.2f}", method="animate", args=[[f"{run.times[i]:.3f}"], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                             for i in idx])])
    return fig


def qubit_timeseries_figure(run: QubitRun) -> go.Figure:
    """Bloch components, purity and fidelity with the ideal state against time; each gate is a shaded band."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("Bloch vector components", "Purity and fidelity with the ideal state"))
    for j, (name, color) in enumerate((("x", _PALETTE[0]), ("y", _PALETTE[1]), ("z", _PALETTE[2]))):
        fig.add_trace(go.Scatter(x=run.times, y=run.bloch[:, j], name=f"r_{name}", line=dict(color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=run.times, y=run.purity, name="purity Tr ρ²", line=dict(color=_PALETTE[3])), row=2, col=1)
    fig.add_trace(go.Scatter(x=run.times, y=run.fidelity, name="fidelity F", line=dict(color=_PALETTE[4], dash="dash")), row=2, col=1)
    # shade the gates; idle stretches stay unshaded
    clock = 0.0
    for label in run.gate_labels:
        length = run.idle_time if label == "idle" else run.gate_time
        if label != "idle":
            for row in (1, 2):
                fig.add_vrect(x0=clock, x1=clock + length, fillcolor="rgba(120,120,200,0.09)", line_width=0, row=row, col=1)
            fig.add_annotation(x=clock + length / 2, y=1.08, yref="y domain", xref="x", text=label, showarrow=False, font=dict(size=10))
        clock += length
    fig.update_yaxes(range=[-1.05, 1.05], row=1, col=1)
    fig.update_yaxes(range=[0.45, 1.05], row=2, col=1)
    fig.update_xaxes(title_text="time", row=2, col=1)
    fig.update_layout(height=520)
    return fig


def density_matrix_figure(rho: np.ndarray, title: str = "Density matrix ρ") -> go.Figure:
    """Real and imaginary parts of ρ as two annotated 2×2 heatmaps."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Re ρ", "Im ρ"))
    for col, part in ((1, rho.real), (2, rho.imag)):
        fig.add_trace(go.Heatmap(z=part[::-1], x=["|0⟩", "|1⟩"], y=["|1⟩", "|0⟩"], zmin=-1, zmax=1, colorscale="RdBu_r", showscale=(col == 2),
                                 text=[[f"{v:.3f}" for v in row] for row in part[::-1]], texttemplate="%{text}", hoverinfo="skip"), row=1, col=col)
    fig.update_layout(title=title, height=300, margin=dict(t=60))
    return fig


def relaxation_figure(times: np.ndarray, t1_values: np.ndarray, ramsey_values: np.ndarray, t1: float | None, t2: float | None,
                      fit_t1: float | None, fit_t2: float | None) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("T1: r_z after preparing |1⟩", "Ramsey: ⟨σx⟩ after (π/2)_y and a wait"))
    fig.add_trace(go.Scatter(x=times, y=t1_values, name="r_z(t)", line=dict(color=_PALETTE[2])), row=1, col=1)
    fig.add_trace(go.Scatter(x=times, y=ramsey_values, name="⟨σx⟩(t)", line=dict(color=_PALETTE[0])), row=1, col=2)
    fig.add_annotation(xref="x domain", yref="y domain", x=0.98, y=0.05, showarrow=False, row=1, col=1,
                       text=f"T1 = {t1:g} · fitted {fit_t1:.4g}" if t1 and fit_t1 else "no T1 relaxation")
    fig.add_annotation(xref="x2 domain", yref="y2 domain", x=0.98, y=0.95, showarrow=False, row=1, col=2,
                       text=f"T2 = {t2:.4g} · fitted {fit_t2:.4g}" if t2 and fit_t2 else "no decoherence")
    fig.update_xaxes(title_text="wait time")
    fig.update_layout(height=340, showlegend=False)
    return fig


# ------------------------------------------------------------------------------------------ perturbation and BCH

def perturbation_energy_figure(comp: Comparison) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comp.lambdas, y=comp.exact, name="exact", mode="lines+markers", line=dict(color="black", width=3)))
    for k, color in zip(sorted(comp.series), (_PALETTE[0], _PALETTE[1], _PALETTE[2])):
        fig.add_trace(go.Scatter(x=comp.lambdas, y=comp.series[k], name=f"through order {k}", mode="lines", line=dict(color=color, dash="dash")))
    fig.update_layout(title=f"Level {comp.level}: exact energy and perturbative orders", xaxis_title="λ", yaxis_title="E")
    return fig


def perturbation_error_figure(comp: Comparison) -> go.Figure:
    fig = go.Figure()
    for k, color in zip(sorted(comp.errors), (_PALETTE[0], _PALETTE[1], _PALETTE[2])):
        err = np.maximum(comp.errors[k], 1e-17)
        slope = comp.slopes.get(k, float("nan"))
        name = f"order {k}: slope {slope:.2f} (expected ≥ {k + 1})" if np.isfinite(slope) else f"order {k}"
        fig.add_trace(go.Scatter(x=comp.lambdas, y=err, name=name, mode="lines+markers", line=dict(color=color)))
        lam = comp.lambdas
        ref = err[0] * (lam / lam[0]) ** (k + 1)
        fig.add_trace(go.Scatter(x=lam, y=ref, mode="lines", showlegend=False, line=dict(color=color, dash="dot", width=1), hoverinfo="skip"))
    fig.update_layout(title="Error of each truncation against exact diagonalisation (dotted: λ^(k+1))", xaxis_title="λ", yaxis_title="|exact − series|",
                      xaxis_type="log", yaxis_type="log")
    return fig


def bch_error_figure(result: BCHResult) -> go.Figure:
    fig = go.Figure(go.Scatter(x=result.scales, y=np.maximum(result.errors, 1e-17), mode="lines+markers", name="error",
                               line=dict(color=_PALETTE[0])))
    ref = result.errors[0] * (result.scales / result.scales[0]) ** (result.order + 1)
    fig.add_trace(go.Scatter(x=result.scales, y=np.maximum(ref, 1e-17), mode="lines", name=f"ε^{result.order + 1}", line=dict(dash="dot", color="gray")))
    label = f", fitted exponent {result.fitted_exponent:.2f}" if np.isfinite(result.fitted_exponent) else ""
    fig.update_layout(title=f"BCH truncated at order {result.order}: error against the scale ε{label}", xaxis_title="ε", yaxis_title="‖log(e^{εX}e^{εY}) − series‖",
                      xaxis_type="log", yaxis_type="log")
    return fig


def density_of_run(run: QubitRun, index: int = -1) -> np.ndarray:
    return bloch_to_density(run.bloch[index])
