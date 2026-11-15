"""Plotly figures for the waves and optics lab -- Streamlit-free, colours from modules/palette.py (never pure black or white)."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules import waves as wv
from modules.palette import BLUE, GREEN, INK, PURPLE, SEQUENTIAL, VERMILLION


def diffraction_figure(p: wv.DiffractionPattern, log_scale: bool = False) -> go.Figure:
    """The pattern on the screen (FFT result with the closed form drawn through it), the aperture beneath it, and the screen as a strip."""
    mm = 1e3
    fig = make_subplots(rows=2, cols=1, row_heights=[0.72, 0.28], vertical_spacing=0.14,
                        subplot_titles=("Intensity on the screen (FFT) and the closed-form pattern", "The aperture that was transformed"))
    x = p.x * mm
    y_fft, y_th = p.intensity, p.analytic
    if log_scale:
        floor = 1e-6
        y_fft, y_th = np.log10(np.maximum(y_fft, floor)), np.log10(np.maximum(y_th, floor))
    fig.add_trace(go.Scatter(x=x, y=y_fft, mode="lines", name="FFT of the aperture", line=dict(color=BLUE, width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y_th, mode="lines", name="closed form", line=dict(color=VERMILLION, width=1.5, dash="dash")), row=1, col=1)
    if len(p.features_x) and not log_scale:
        ys = np.interp(p.features_x, p.x, p.intensity)
        fig.add_trace(go.Scatter(x=p.features_x * mm, y=ys, mode="markers", name=f"measured {p.features_kind}",
                                 marker=dict(size=9, color=GREEN, symbol="diamond" if p.features_kind == "principal maxima" else "x")), row=1, col=1)
    zero = p.envelope_zero_theory * mm
    for sign in (-1, 1):
        fig.add_vline(x=sign * zero, line_dash="dot", line_color=INK, line_width=1, row=1, col=1,
                      annotation_text="λL/a" if sign > 0 else None, annotation_position="top")
    fig.add_trace(go.Scatter(x=p.aperture_x * mm, y=p.aperture, mode="lines", fill="tozeroy", name="aperture", line=dict(color=PURPLE, width=2), showlegend=False), row=2, col=1)
    fig.update_xaxes(title_text="position on the screen (mm)", row=1, col=1)
    fig.update_xaxes(title_text="position in the aperture plane (mm)", row=2, col=1)
    fig.update_yaxes(title_text="log10 intensity" if log_scale else "relative intensity", row=1, col=1)
    fig.update_yaxes(title_text="transmission", row=2, col=1, range=[0, 1.1])
    fig.update_layout(height=560, title=f"Fraunhofer diffraction by a {p.kind}: λ = {p.wavelength * 1e9:.0f} nm, screen at {p.distance:g} m", legend=dict(orientation="h", y=-0.2))
    return fig


def screen_strip_figure(p: wv.DiffractionPattern) -> go.Figure:
    """What the screen looks like: a greyscale-free, perceptually uniform strip of the intensity (gamma-lifted so the faint lobes are visible)."""
    strip = np.tile(np.power(np.clip(p.intensity, 0, 1), 0.4), (24, 1))
    fig = go.Figure(go.Heatmap(z=strip, x=p.x * 1e3, colorscale=SEQUENTIAL, showscale=False, hoverinfo="skip"))
    fig.update_layout(height=170, title="The pattern as it would look on the screen (intensity^0.4, so faint lobes show)",
                      xaxis_title="position on the screen (mm)", yaxis=dict(visible=False), margin=dict(t=50, b=40))
    return fig


def diffraction_2d_figure(p: wv.DiffractionPattern2D, log_scale: bool = True) -> go.Figure:
    """The far-field image and the aperture side by side, with a cut through the centre underneath."""
    mm = 1e3
    fig = make_subplots(rows=2, cols=2, row_heights=[0.68, 0.32], vertical_spacing=0.16, horizontal_spacing=0.1,
                        subplot_titles=("Far-field intensity" + (" (log scale)" if log_scale else ""), "The aperture", "Cut through the centre along x", ""),
                        specs=[[{}, {}], [{"colspan": 2}, None]])
    z = np.log10(np.maximum(p.intensity, 1e-6)) if log_scale else p.intensity
    fig.add_trace(go.Heatmap(z=z, x=p.x * mm, y=p.x * mm, colorscale=SEQUENTIAL, showscale=False, hovertemplate="x %{x:.2f} mm<br>y %{y:.2f} mm<extra></extra>"), row=1, col=1)
    ax = np.linspace(-p.aperture_extent, p.aperture_extent, p.aperture.shape[0]) * mm
    fig.add_trace(go.Heatmap(z=p.aperture, x=ax, y=ax, colorscale=[[0, "#2B2F3A"], [1, "#E8E4D0"]], showscale=False, hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter(x=p.x * mm, y=np.maximum(p.row, 1e-6), mode="lines", line=dict(color=BLUE, width=2.5), name="intensity"), row=2, col=1)
    fig.add_vline(x=p.first_zero_theory * mm, line_dash="dot", line_color=VERMILLION, row=2, col=1, annotation_text="first zero (theory)", annotation_position="top")
    fig.add_vline(x=-p.first_zero_theory * mm, line_dash="dot", line_color=VERMILLION, row=2, col=1)
    fig.update_yaxes(type="log" if log_scale else "linear", title_text="relative intensity", range=[-6, 0.1] if log_scale else None, row=2, col=1)
    fig.update_xaxes(title_text="x (mm)", row=2, col=1)
    fig.update_xaxes(scaleanchor="y", row=1, col=1)
    fig.update_layout(height=700, title=f"Fraunhofer diffraction by a {p.kind} aperture (2-D FFT)", showlegend=False)
    return fig


def ripple_figure(t: wv.RippleTank, animate: bool = True) -> go.Figure:
    """The water surface as a diverging colour map, animated through time."""
    frames = t.frames
    peak = float(np.percentile(np.abs(frames[-1]), 99.5)) or 1.0
    fig = go.Figure(go.Heatmap(z=frames[-1], colorscale="RdBu", zmid=0, zmin=-peak, zmax=peak, showscale=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[s.x for s in t.sources], y=[s.y for s in t.sources], mode="markers", marker=dict(size=11, color=GREEN, line=dict(width=2, color=INK)),
                             name="sources", hovertemplate="source at (%{x:.0f}, %{y:.0f})<extra></extra>"))
    if animate and len(frames) > 1:
        fig.frames = [go.Frame(data=[go.Heatmap(z=f)], traces=[0], name=str(int(s))) for f, s in zip(frames, t.frame_steps)]
        fig.update_layout(
            updatemenus=[dict(type="buttons", x=0.0, y=-0.08, buttons=[
                dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate", args=[[None], dict(mode="immediate", frame=dict(duration=0, redraw=False))])])],
            sliders=[dict(active=len(frames) - 1, x=0.15, len=0.8, pad=dict(t=30), currentvalue=dict(prefix="step "),
                          steps=[dict(label=str(int(s)), method="animate", args=[[str(int(s))], dict(mode="immediate", frame=dict(duration=0, redraw=True))]) for s in t.frame_steps])])
    fig.update_layout(height=640, title=f"Ripple tank: {len(t.sources)} source{'s' if len(t.sources) != 1 else ''}, {t.wavelength_cells:g} cells per wavelength",
                      xaxis=dict(title="x (cells)", scaleanchor="y", constrain="domain"), yaxis=dict(title="y (cells)", autorange="reversed", constrain="domain"),
                      showlegend=False)
    return fig


def ripple_amplitude_figure(t: wv.RippleTank) -> go.Figure:
    """The time-averaged amplitude: bright lines are the antinodal lines, dark lines the nodal lines."""
    fig = go.Figure(go.Heatmap(z=t.amplitude, colorscale=SEQUENTIAL, showscale=True, colorbar=dict(title="rms"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[s.x for s in t.sources], y=[s.y for s in t.sources], mode="markers", marker=dict(size=10, color=VERMILLION, line=dict(width=2, color=INK)), name="sources"))
    fig.update_layout(height=560, title="Time-averaged amplitude over the last six periods (interference pattern)",
                      xaxis=dict(title="x (cells)", scaleanchor="y", constrain="domain"), yaxis=dict(title="y (cells)", autorange="reversed", constrain="domain"), showlegend=False)
    return fig


def ripple_checks_figure(t: wv.RippleTank) -> go.Figure | None:
    """Single source: amplitude against distance on log axes with the 1/sqrt(r) line. Two sources: measured against predicted maxima angles."""
    if len(t.sources) == 1 and t.probe_x is not None:
        s = t.sources[0]
        r = (t.probe_x - int(round(s.x))).astype(float)
        a = t.amplitude[int(round(s.y)), t.probe_x]
        ok = r >= 2 * t.wavelength_cells
        ref = a[ok][0] * np.sqrt(r[ok][0] / r[ok])
        fig = go.Figure([go.Scatter(x=r[ok], y=a[ok], mode="lines", name="simulation", line=dict(color=BLUE, width=3)),
                         go.Scatter(x=r[ok], y=ref, mode="lines", name="1/√r", line=dict(color=VERMILLION, dash="dash", width=2))])
        fig.update_xaxes(type="log", title_text="distance from the source (cells)")
        fig.update_yaxes(type="log", title_text="rms amplitude")
        fig.update_layout(height=340, title="Amplitude against distance for one circular wave (log-log)", legend=dict(orientation="h", y=-0.3))
        return fig
    if t.maxima:
        meas, pred = [m for m, _ in t.maxima], [p for _, p in t.maxima]
        fig = go.Figure([go.Scatter(x=pred, y=meas, mode="markers", name="maxima", marker=dict(size=11, color=BLUE)),
                         go.Scatter(x=[min(pred), max(pred)], y=[min(pred), max(pred)], mode="lines", name="agreement", line=dict(color=INK, dash="dash"))])
        fig.update_layout(height=340, title="Measured against predicted angles of the interference maxima", xaxis_title="predicted angle (degrees)", yaxis_title="measured angle (degrees)",
                          legend=dict(orientation="h", y=-0.3))
        return fig
    return None


def doppler_figure(r: wv.DopplerRun) -> go.Figure:
    """The wavefronts at the snapshot time (circles centred where each was emitted), the source's path, the observer, and the Mach cone."""
    t = r.snapshot_time
    live = np.nonzero(r.emit_times < t)[0]
    step = max(1, len(live) // 60)
    fig = go.Figure()
    theta = np.linspace(0, 2 * np.pi, 120)
    for rank, i in enumerate(live[::step]):
        rad = r.c * (t - r.emit_times[i])
        cx, cy = r.emit_positions[i]
        fig.add_trace(go.Scatter(x=cx + rad * np.cos(theta), y=cy + rad * np.sin(theta), mode="lines", line=dict(color=BLUE, width=1), opacity=0.55, hoverinfo="skip", showlegend=False))
    path = r.emit_positions[live]
    fig.add_trace(go.Scatter(x=path[:, 0], y=path[:, 1], mode="lines", line=dict(color=VERMILLION, width=2, dash="dot"), name="source path"))
    now = r.source_start + r.source_velocity * t
    fig.add_trace(go.Scatter(x=[now[0]], y=[now[1]], mode="markers", marker=dict(size=13, color=VERMILLION, symbol="star"), name="source now"))
    obs = r.observer_start + r.observer_velocity * t
    fig.add_trace(go.Scatter(x=[obs[0]], y=[obs[1]], mode="markers", marker=dict(size=12, color=GREEN, symbol="diamond"), name="observer now"))
    if len(r.cone_points):
        fig.add_trace(go.Scatter(x=r.cone_points[:, 0], y=r.cone_points[:, 1], mode="markers", marker=dict(size=4, color=PURPLE), name="Mach cone"))
    fig.update_layout(height=520, title="Wavefronts emitted by a moving source (each circle is centred where its wave was emitted)",
                      xaxis=dict(title="x (m)", scaleanchor="y", constrain="domain"), yaxis=dict(title="y (m)", constrain="domain"), legend=dict(orientation="h", y=-0.15))
    return fig


def doppler_frequency_figure(r: wv.DopplerRun) -> go.Figure:
    """The frequency the observer hears against time, measured from the arrival times and from the exact rate."""
    ok = np.isfinite(r.inst_frequency)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.inst_times[ok], y=r.inst_frequency[ok], mode="markers", name="from successive arrival times", marker=dict(size=5, color=BLUE)))
    good = np.isfinite(r.inst_theory)
    if good.any():
        fig.add_trace(go.Scatter(x=r.inst_times[good], y=r.inst_theory[good], mode="lines", name="exact rate", line=dict(color=VERMILLION, width=2, dash="dash")))
    fig.add_hline(y=r.f0, line_dash="dot", line_color=INK, annotation_text="emitted f₀", annotation_position="right")
    fig.update_layout(height=340, title="Frequency heard by the observer against time", xaxis_title="time (s)", yaxis_title="frequency (Hz)", legend=dict(orientation="h", y=-0.3))
    return fig
