"""The Waves and optics lab: slit diffraction by FFT, 2-D apertures, a ripple tank and the Doppler effect (modules/waves.py).

Every result carries the checks that were run on it (fringe spacing, the sinc^2 envelope, the Airy zero, interference maxima from the path
difference, the Mach angle). No language model involved."""
from typing import Any

import streamlit as st

from modules import gpu
from modules import waves as wv
from modules import waves_plots as wp
from ui.a11y import show_figure
from ui.answer_card import render_lab_card
from ui.cache import cached
from ui.lab_common import num, render_checks, render_notes
from ui.theme import backend_chip, badge_row


def _mm(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 1e3:.4g} mm"


def _slits_tab() -> None:
    kind = st.radio("Aperture", list(wv.KINDS_1D), horizontal=True, key="wv_kind")
    c1, c2, c3 = st.columns(3)
    with c1:
        wl = num("Wavelength (nm)", 550.0, "wv_wl", 100.0, 3000.0)
        dist = num("Screen distance L (m)", 1.5, "wv_L", 0.01, 1000.0)
    with c2:
        a = num("Slit width a (µm)", 50.0, "wv_a", 1.0, 5000.0)
        sep = num("Separation d (µm)", 250.0, "wv_d", 1.0, 100000.0, disabled=kind == "single slit")
    with c3:
        n = int(st.number_input("Number of slits", 3, 20, 5, key="wv_n", disabled=kind != "grating"))
        log = st.checkbox("Logarithmic intensity", key="wv_log")
    try:
        p = cached("wv_1d", (kind, wl, dist, a, sep, n), lambda: wv.diffraction_1d(kind, wl * 1e-9, dist, a * 1e-6, sep * 1e-6, n, backend=gpu.CPU))
    except wv.WavesError as exc:
        st.error(str(exc))
        return
    badge_row([(f"fringe spacing λL/d = {_mm(p.fringe_spacing_theory)}" if p.fringe_spacing_theory else "single slit: no fringes", "neutral"),
               (f"first zero λL/a = {_mm(p.envelope_zero_theory)}", "neutral"),
               ("✅ all checks pass" if p.checks.passed else "a check failed", "pass" if p.checks.passed else "fail")])
    show_figure(wp.diffraction_figure(p, log), width="stretch")
    show_figure(wp.screen_strip_figure(p), width="stretch")
    m1, m2 = st.columns(2)
    m1.metric("Fringe spacing, measured", _mm(p.fringe_spacing_measured), help="From the FFT result alone: " + (p.features_kind or "n/a"))
    m2.metric("First zero of the envelope, measured", _mm(p.envelope_zero_measured))
    render_notes(p.notes)
    render_checks(p.checks)
    render_lab_card(f"Fraunhofer diffraction: {kind}", [f"fringe spacing {_mm(p.fringe_spacing_measured)} (λL/d = {_mm(p.fringe_spacing_theory)})" if p.fringe_spacing_theory else f"first minimum {_mm(p.envelope_zero_measured)}",
                                                          f"first envelope zero {_mm(p.envelope_zero_measured)} (λL/a = {_mm(p.envelope_zero_theory)})"], p.checks, "wv_card_1d",
                    f"λ = {wl:g} nm, slit width {a:g} µm, separation {sep:g} µm, screen at {dist:g} m")
    st.caption("Paraxial approximation (screen position x = λL·f, small angles). The aperture is sampled at a/64 and the FFT sizes the screen resolution.")


def _aperture_tab() -> None:
    kind = st.radio("Aperture", list(wv.KINDS_2D), horizontal=True, key="wv2_kind")
    c1, c2, c3 = st.columns(3)
    with c1:
        wl = num("Wavelength (nm)", 550.0, "wv2_wl", 100.0, 3000.0)
        dist = num("Screen distance L (m)", 1.5, "wv2_L", 0.01, 1000.0)
    with c2:
        size = num("Diameter / width (µm)", 300.0 if kind == "circular" else 100.0, f"wv2_size_{kind}", 5.0, 5000.0)
        height = num("Height (µm)", 200.0, "wv2_h", 5.0, 5000.0, disabled=kind != "rectangular")
    with c3:
        sep = num("Slit separation (µm)", 400.0, "wv2_sep", 5.0, 20000.0, disabled=kind != "double slit")
        px = int(st.selectbox("Pixels per side", [512, 1024, 2048, 4096], index=1, key="wv2_px"))
    backend = gpu.select(px * px, "fft_pixels")

    def run():
        return wv.diffraction_2d(kind, wl * 1e-9, dist, size * 1e-6, height * 1e-6 if kind == "rectangular" else None, sep * 1e-6 if kind == "double slit" else 0.0,
                                 px, backend=backend)
    try:
        p = cached("wv_2d", (kind, wl, dist, size, height, sep, px, backend.name), run)
    except wv.WavesError as exc:
        st.error(str(exc))
        return
    backend_chip(p.backend)
    log = st.checkbox("Logarithmic intensity", value=True, key="wv2_log")
    show_figure(wp.diffraction_2d_figure(p, log), width="stretch")
    render_checks(p.checks)
    render_lab_card(f"2-D diffraction: {kind} aperture", [f"first zero {_mm(p.first_zero_measured)} (theory {_mm(p.first_zero_theory)})"], p.checks, "wv_card_2d",
                    f"λ = {wl:g} nm, screen at {dist:g} m, {px}² pixels")


def _tank_tab() -> None:
    preset = st.radio("Sources", ["One source", "Two sources, in phase", "Two sources, opposite phase"], horizontal=True, key="wv_tank_preset")
    c1, c2, c3 = st.columns(3)
    wl = int(c1.slider("Cells per wavelength", 8, 24, 12, key="wv_tank_wl"))
    sep = c2.slider("Source separation (wavelengths)", 1.5, 6.0, 3.0, 0.5, key="wv_tank_sep", disabled=preset == "One source")
    n = int(c3.selectbox("Grid (cells per side)", [201, 281, 361, 501], index=1, key="wv_tank_n"))
    mid = n // 2
    half = sep * wl / 2
    if preset == "One source":
        srcs = [wv.Source(mid, mid)]
    else:
        srcs = [wv.Source(int(round(mid - half)), mid, 0.0), wv.Source(int(round(mid + half)), mid, 0.0 if "in phase" in preset else 3.141592653589793)]
    backend = gpu.select(n * n, "wave_cells")
    try:
        t = cached("wv_tank", (n, wl, tuple((s.x, s.y, s.phase) for s in srcs), backend.name), lambda: wv.ripple_tank(n, float(wl), srcs, backend=backend))
    except wv.WavesError as exc:
        st.error(str(exc))
        return
    backend_chip(t.backend)
    show_figure(wp.ripple_figure(t), width="stretch")
    show_figure(wp.ripple_amplitude_figure(t), width="stretch")
    fig = wp.ripple_checks_figure(t)
    if fig is not None:
        show_figure(fig, width="stretch")
    st.caption(f"Effective wavelength of the scheme: {t.numerical_wavelength:.2f} cells (nominal {t.wavelength_cells:g}); {t.steps} time steps of Courant number {t.courant:g}.")
    render_checks(t.checks)
    render_lab_card("Ripple tank", [f"{len(t.sources)} source(s), {t.wavelength_cells:g} cells per wavelength", f"effective wavelength {t.numerical_wavelength:.2f} cells"], t.checks, "wv_card_tank",
                    f"{preset}: separation {sep:g} wavelengths" if len(srcs) > 1 else "One source")


SCENARIOS: dict[str, dict[str, Any]] = {
    "Passing by (a siren going past, 30 m off the track)": dict(angle=0.0, start=(-150.0, 0.0), obs=(0.0, 30.0), speed=40.0, auto_waves=True),
    "Head-on approach": dict(angle=0.0, start=(-500.0, 0.0), obs=(0.0, 0.0), speed=30.0),
    "Receding": dict(angle=180.0, start=(-200.0, 0.0), obs=(0.0, 0.0), speed=30.0),
    "Supersonic (Mach cone)": dict(angle=0.0, start=(-1500.0, 0.0), obs=(0.0, 50.0), speed=600.0),
}


def _doppler_tab() -> None:
    name = str(st.selectbox("Scenario", list(SCENARIOS), key="wv_dop_scn"))
    sc = SCENARIOS[name]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        c = num("Wave speed c (m/s)", 343.0, "wv_dop_c", 1.0, 1e9)
    with c2:
        f0 = num("Source frequency (Hz)", 440.0 if "Supersonic" not in name else 50.0, f"wv_dop_f_{name}", 0.1, 1e9)
    with c3:
        v = num("Source speed (m/s)", sc["speed"], f"wv_dop_v_{name}", 0.0, 1e9)
    with c4:
        vo = num("Observer speed toward the source (m/s)", 0.0, "wv_dop_vo", 0.0, 1e9, disabled="Passing" in name or "Supersonic" in name)
    obs_v = (-vo if sc["angle"] == 0.0 else vo, 0.0)
    # the passing scenario must cover the whole pass (the source travels twice its starting distance), so the number of waves follows the frequency
    n_waves = int(min(20000, max(100, f0 * 2 * abs(sc["start"][0]) / max(v, 1e-9) + 50))) if sc.get("auto_waves") else 400
    try:
        r = cached("wv_dop", (name, c, f0, v, vo, n_waves), lambda: wv.doppler(c, f0, v, sc["angle"], sc["start"], sc["obs"], obs_v, n_waves))
    except wv.WavesError as exc:
        st.error(str(exc))
        return
    if r.mach_angle_theory is not None:
        badge_row([(f"Mach number {v / c:.2f}", "neutral"), (f"cone half-angle {r.mach_angle_theory:.2f}°", "neutral")])
    else:
        import numpy as np
        ok = np.isfinite(r.inst_frequency)
        if ok.any():
            badge_row([(f"highest heard {np.nanmax(r.inst_frequency[ok]):.1f} Hz", "neutral"), (f"lowest heard {np.nanmin(r.inst_frequency[ok]):.1f} Hz", "neutral")])
    show_figure(wp.doppler_figure(r), width="stretch")
    if r.mach_angle_theory is None:
        show_figure(wp.doppler_frequency_figure(r), width="stretch")
    render_notes(r.notes)
    render_checks(r.checks)
    if r.checks.checks:
        render_lab_card("Doppler effect", [f"Mach cone half-angle {r.mach_angle_theory:.2f}°" if r.mach_angle_theory is not None else f"source {v:g} m/s at {f0:g} Hz, wave speed {c:g} m/s"], r.checks, "wv_card_dop", name)


def render_waves_tab() -> None:
    st.subheader("🌊 Waves & optics lab")
    st.caption("Double-slit and diffraction by FFT, a ripple tank, and the Doppler effect. Each result is checked against a closed form: the fringe spacing λL/d, "
               "the sinc² envelope, the Airy zero, interference maxima from the path difference, the Doppler formula and the Mach angle. No language model involved.")
    tabs = st.tabs(["🔦 Slits & diffraction", "🔵 2-D apertures", "💧 Ripple tank", "🚑 Doppler"])
    with tabs[0]:
        _slits_tab()
    with tabs[1]:
        _aperture_tab()
    with tabs[2]:
        _tank_tab()
    with tabs[3]:
        _doppler_tab()
