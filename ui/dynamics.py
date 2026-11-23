"""The Dynamics and chaos lab: symplectic N-body orbits, the double pendulum, the Lorenz butterfly and Lyapunov exponents (modules/dynamics.py).

The built-in check is conservation: energy drift (bounded oscillation for a symplectic method, steady growth otherwise), momentum, angular
momentum, the Laplace-Runge-Lenz vector, the measured order of each integrator, and the exact identities for the Lyapunov spectrum."""
import numpy as np
import streamlit as st

from modules import dynamics as dy
from modules import dynamics_plots as dp
from modules import gpu
from modules import next_steps as ns
from ui.a11y import show_figure
from ui.answer_card import render_lab_card
from ui.cache import cached
from ui.lab_common import num, render_checks, render_notes
from ui.next_steps import render_next_steps
from ui.theme import backend_chip, badge_row

INTEG_NAME = {k: str(v["label"]).split(",")[0] for k, v in dy.INTEGRATORS.items()}

PRESETS: dict[str, str] = {
    "Figure-eight (three bodies)": "figure-eight",
    "Kepler orbit (two bodies)": "kepler",
    "Lagrange equilateral triangle": "lagrange",
    "Pythagorean 3-4-5 (chaotic)": "pythagorean",
    "Random cluster": "cluster",
}


def _nbody_tab() -> None:
    c1, c2, c3 = st.columns(3)
    preset = str(c1.selectbox("System", list(PRESETS), key="dy_preset"))
    integ = str(c2.selectbox("Integrator", list(dy.INTEGRATORS), format_func=lambda k: str(dy.INTEGRATORS[k]["label"]), key="dy_integ"))
    key = PRESETS[preset]
    ecc = c3.slider("Eccentricity", 0.0, 0.9, 0.6, 0.05, key="dy_ecc", disabled=key != "kepler")
    d1, d2, d3 = st.columns(3)
    default_dt = {"figure-eight": 1e-3, "kepler": 2e-3, "lagrange": 2e-3, "pythagorean": 1e-3, "cluster": 2e-3}[key]
    dt = num("Time step", default_dt, f"dy_dt_{key}", 1e-5, 0.5)
    default_end = {"figure-eight": 3 * dy.FIGURE_EIGHT_PERIOD, "kepler": 10 * 6.2832, "lagrange": 3 * 3.6276, "pythagorean": 10.0, "cluster": 2.0}[key]
    t_end = num("Run time", round(default_end, 3), f"dy_end_{key}", 0.01, 5000.0)
    n = int(d3.number_input("Bodies (cluster only)", 3, dy.MAX_BODIES, 60, key="dy_n", disabled=key != "cluster"))
    seed = int(st.number_input("Seed (cluster)", 0, 10 ** 6, 1, key="dy_seed", disabled=key != "cluster"))
    backend = gpu.select(n if key == "cluster" else 3, "nbody_particles")

    def run():
        sys_ = dy.preset_system(key, n=n, seed=seed, eccentricity=ecc)
        return dy.run_nbody(sys_, integ, dt, t_end, backend=backend)
    try:
        r = cached("dy_nbody", (key, integ, dt, t_end, n, seed, ecc, backend.name), run)
    except dy.DynamicsError as exc:
        st.error(str(exc))
        return
    st.caption(r.system.note)
    backend_chip(r.backend)
    badge_row([(f"energy: {r.energy_drift.kind}", "pass" if r.energy_drift.max_rel < 1e-4 else "fail"), (f"max |ΔE/E| = {r.energy_drift.max_rel:.1e}", "neutral"), (f"{r.n_steps:,} steps", "neutral")])
    show_figure(dp.orbit_figure(r), width="stretch")
    show_figure(dp.conservation_figure(r), width="stretch")
    render_notes(r.notes)
    render_checks(r.checks)
    render_lab_card(r.system.name, [f"energy: {r.energy_drift.kind}", f"max |ΔE/E| = {r.energy_drift.max_rel:.1e}", f"{INTEG_NAME[integ]}, dt = {dt:g}"], r.checks, "dy_card_nbody",
                    f"{r.n_steps:,} steps of {r.system.name}")
    with st.expander("📐 How fast does each method converge? (order study on a Kepler orbit)"):
        if st.button("Run the order study", key="dy_order_btn"):
            st.session_state["dy_order_ran"] = True
        if st.session_state.get("dy_order_ran"):
            studies, ck = cached("dy_order", (ecc,), lambda: dy.order_study(dy.preset_system("kepler", eccentricity=ecc)))
            show_figure(dp.order_study_figure(studies), width="stretch")
            render_checks(ck, "Observed against nominal order")


DERIVING_MESSAGE = "Deriving the equations of motion with SymPy (first use only)…"


def _ensure_pendulum_derived() -> None:
    """The first double-pendulum computation of a process derives its equations from the Lagrangian (about 3.5 s, then cached): say so instead of leaving the page silent."""
    if not dy.pendulum_is_derived():
        with st.spinner(DERIVING_MESSAGE):
            dy.pendulum_sympy()


def _pendulum_tab() -> None:
    th1 = num("θ1 (degrees)", 120.0, "dy_th1", -180.0, 180.0)
    th2 = num("θ2 (degrees)", 90.0, "dy_th2", -180.0, 180.0)
    t_end = num("Duration (s)", 20.0, "dy_pt", 1.0, 200.0)
    m_ratio = num("m2 / m1", 1.0, "dy_mr", 0.05, 20.0)
    p = dy.PendulumParams(1.0, m_ratio, 1.0, 1.0, 9.81)
    state0 = (np.radians(th1), np.radians(th2), 0.0, 0.0)
    _ensure_pendulum_derived()
    r = cached("dy_pend", (th1, th2, t_end, m_ratio), lambda: dy.run_pendulum(p, state0, t_end))
    badge_row([(f"energy: {r.energy_drift.kind}", "pass" if r.checks.checks[0].passed else "fail"), (f"max |ΔE/E| = {r.energy_drift.max_rel:.1e}", "neutral")])
    show_figure(dp.pendulum_figure(r), width="stretch")
    show_figure(dp.pendulum_energy_figure(r), width="stretch")
    render_checks(r.checks)
    render_lab_card("Double pendulum", [f"energy: {r.energy_drift.kind}", f"max |ΔE/E| = {r.energy_drift.max_rel:.1e}"], r.checks, "dy_card_pend", f"θ1 = {th1:g}°, θ2 = {th2:g}°, m2/m1 = {m_ratio:g}, {t_end:g} s")
    render_next_steps(ns.for_trajectory(r.times, r.states, ["th1", "th2", "w1", "w2"]), "pend")
    st.markdown("**The chaos map** -- every starting angle, one batched integration")
    n = int(st.select_slider("Grid (points per side)", [32, 48, 64, 96, 128, 200, 400], value=48, key="dy_flip_n"))
    backend = gpu.select(n * n, "ensemble_trajectories")
    fm = cached("dy_flip", (m_ratio, n, backend.name), lambda: dy.flip_map(p, n=n, backend=backend))
    backend_chip(fm.backend)
    show_figure(dp.flip_map_figure(fm), width="stretch")
    render_checks(fm.checks)


def _lorenz_tab() -> None:
    sigma, rho, beta = num("σ", 10.0, "dy_sigma", 0.1, 100.0), num("ρ", 28.0, "dy_rho", 0.1, 200.0), num("β", 8 / 3, "dy_beta", 0.1, 20.0)
    p = dy.LorenzParams(sigma, rho, beta)
    ts, X = cached("dy_lorenz", (sigma, rho, beta), lambda: dy.run_lorenz(p, t_end=40.0, n_points=4000))
    show_figure(dp.lorenz_figure(ts, X), width="stretch")
    show_figure(dp.lorenz_series_figure(ts, X), width="stretch")
    render_next_steps(ns.for_trajectory(ts, X, ["x", "y", "z"]), "lorenz")
    st.markdown("**Lyapunov exponents** -- an ensemble of nearby trajectories, all advanced together")
    e1, e2, e3 = st.columns(3)
    members = int(e1.select_slider("Ensemble size", [32, 64, 128, 256, 512, 1024, 4096, 20000], value=128, key="dy_members"))
    t_run = e2.slider("Averaging time", 5.0, 60.0, 20.0, 5.0, key="dy_trun")
    seed = int(e3.number_input("Seed", 0, 10 ** 6, 1, key="dy_lseed"))
    backend = gpu.select(members, "ensemble_trajectories")
    ly = cached("dy_lyap", (sigma, rho, beta, members, t_run, seed, backend.name),
                lambda: dy.lyapunov_spectrum("lorenz", p, np.array([1.0, 1.0, 1.0]), members=members, t_run=t_run, seed=seed, backend=backend))
    backend_chip(ly.backend)
    badge_row([(f"λ1 = {ly.exponents[0]:.3f} ± {ly.errors[0]:.3f}", "pass" if ly.exponents[0] > 0 else "neutral"), (f"Kaplan-Yorke dimension {ly.kaplan_yorke:.3f}" if ly.kaplan_yorke is not None else "", "neutral")])
    show_figure(dp.lyapunov_figure(ly), width="stretch")
    dc = cached("dy_div", (sigma, rho, beta, members, seed, backend.name), lambda: dy.divergence_curve("lorenz", p, np.array([1.0, 1.0, 1.0]), members=members, seed=seed, backend=backend))
    show_figure(dp.divergence_figure(dc), width="stretch")
    st.caption(f"Second route to λ1: the slope of the mean log separation of {dc.members:,} nearby pairs is {dc.slope:.3f}, against {ly.exponents[0]:.3f} from the tangent-space method.")
    render_notes(ly.notes)
    render_checks(ly.checks)
    render_lab_card("Lorenz system", [f"λ1 = {ly.exponents[0]:.3f} ± {ly.errors[0]:.3f}", f"sum of exponents {ly.exponents.sum():.4f}", f"Kaplan-Yorke dimension {ly.kaplan_yorke:.3f}" if ly.kaplan_yorke else "Lyapunov spectrum"], ly.checks, "dy_card_lor",
                    f"σ = {sigma:g}, ρ = {rho:g}, β = {beta:.4g}; {members:,} nearby trajectories")


def _pendulum_lyapunov() -> None:
    st.markdown("**Is this pendulum chaotic?** The Lyapunov spectrum of the double pendulum for the starting angles above (an ensemble of nearby starts).")
    if not st.button("Estimate the exponents", key="dy_pl_btn"):
        return
    _ensure_pendulum_derived()
    p = dy.PendulumParams(1.0, st.session_state.get("dy_mr", 1.0), 1.0, 1.0, 9.81)
    s0 = np.array([np.radians(st.session_state.get("dy_th1", 120.0)), np.radians(st.session_state.get("dy_th2", 90.0)), 0.0, 0.0])
    ly = dy.lyapunov_spectrum("pendulum", p, s0, members=64, t_run=30.0, seed=1)
    badge_row([(f"λ1 = {ly.exponents[0]:.3f} ± {ly.errors[0]:.3f}", "pass" if ly.exponents[0] > 0.05 else "neutral")])
    show_figure(dp.lyapunov_figure(ly), width="stretch")
    render_notes(ly.notes)
    render_checks(ly.checks)


def render_dynamics_tab() -> None:
    st.subheader("🌀 Dynamics & chaos lab")
    st.caption("N-body orbits (including the figure-eight three-body solution) with symplectic integrators, a double pendulum, the Lorenz butterfly, and Lyapunov exponents "
               "estimated over whole ensembles of nearby trajectories (GPU-accelerated when it is available). The built-in check is conservation: energy drift, momentum, "
               "angular momentum, and exact identities for the exponents. No language model involved.")
    tabs = st.tabs(["🪐 N-body & symplectic integrators", "🎢 Double pendulum", "🦋 Lorenz & Lyapunov"])
    with tabs[0]:
        _nbody_tab()
    with tabs[1]:
        _pendulum_tab()
        _pendulum_lyapunov()
    with tabs[2]:
        _lorenz_tab()
