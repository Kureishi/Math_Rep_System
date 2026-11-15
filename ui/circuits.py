"""The Circuits and control lab: RLC transients, Bode plots, and step response with PID tuning (modules/circuits.py).

Every result carries its checks: the resonant frequency and damping ratio MEASURED from the simulated curve against omega_0 and zeta, the energy
balance, Routh-Hurwitz against the pole locations, an explicit feedback simulation against the closed-loop algebra, and the final value theorem."""
import numpy as np
import streamlit as st

from modules import circuits as ck
from modules import circuits_plots as cp
from modules import next_steps as ns
from ui.a11y import show_figure
from ui.answer_card import render_lab_card
from ui.cache import cached
from ui.lab_common import num, render_checks, render_notes
from ui.next_steps import render_next_steps
from ui.theme import badge_row

SI = {"R": ("Ω", 1.0), "L": ("mH", 1e-3), "C": ("µF", 1e-6)}


def _components(prefix: str, topology: str) -> ck.RLC:
    R = num("Resistance R (Ω)", 10.0, f"{prefix}_R", 1e-3, 1e9)
    L = num("Inductance L (mH)", 10.0, f"{prefix}_L", 1e-6, 1e6)
    C = num("Capacitance C (µF)", 10.0, f"{prefix}_C", 1e-6, 1e9)
    return ck.RLC(R, L * 1e-3, C * 1e-6, topology)


def _summary_badges(c: ck.RLC) -> None:
    badge_row([(f"{c.regime}", "pass" if c.regime == "underdamped" else "neutral"), (f"ω₀ = {c.omega0:.4g} rad/s ({c.f0:.4g} Hz)", "neutral"), (f"ζ = {c.zeta:.4g}", "neutral"), (f"Q = {c.Q:.4g}", "neutral")])


def _transient_tab() -> None:
    topology = st.radio("Circuit", ["series", "parallel"], horizontal=True, key="ck_topology", format_func=lambda t: "Series RLC (voltage step)" if t == "series" else "Parallel RLC (current step)")
    c = _components("ck", topology)
    level = num("Step height (V or A)", 1.0, "ck_level", 1e-6, 1e6)
    try:
        c.validate()
        tr = cached("ck_tr", (c.R, c.L, c.C, topology, level), lambda: ck.simulate_transient(c, level))
    except ck.CircuitError as exc:
        st.error(str(exc))
        return
    _summary_badges(c)
    show_figure(cp.transient_figure(tr), width="stretch")
    m, th = tr.measured, tr.theory
    cols = st.columns(4)
    if "zeta" in m:
        cols[0].metric("ζ from the log decrement", f"{m['zeta']:.4f}", help=f"circuit values: {c.zeta:.4f}")
    if "omega_d" in m:
        cols[1].metric("Ringing frequency", f"{m['omega_d'] / 2 / np.pi:.4g} Hz", help=f"ω₀√(1−ζ²) = {(th['omega_d'] or 0) / 2 / np.pi:.4g} Hz")
    if m.get("overshoot") is not None:
        cols[2].metric("Overshoot", f"{m['overshoot']:.2%}", help=f"formula: {(th['overshoot'] or 0):.2%}")
    if m.get("settling_time") is not None:
        cols[3].metric("2 % settling time", f"{m['settling_time'] * 1e3:.4g} ms")
    render_notes(tr.notes)
    render_checks(tr.checks)
    render_lab_card(f"{topology.capitalize()} RLC transient", [f"ω₀ = {c.omega0:.4g} rad/s ({c.f0:.4g} Hz)", f"ζ = {c.zeta:.4g}, Q = {c.Q:.4g}", c.regime], tr.checks, "ck_card_tr",
                    f"R = {c.R:g} Ω, L = {c.L * 1e3:g} mH, C = {c.C * 1e6:g} µF")
    key = "v_C" if topology == "series" else "v"
    render_next_steps(ns.for_trajectory(tr.t, np.stack([tr.signals[key], tr.signals["i" if topology == "series" else "i_L"]], axis=1), [key, "i" if topology == "series" else "iL"]), "rlc")


def _bode_tab() -> None:
    topology = st.radio("Circuit", ["series", "parallel"], horizontal=True, key="ck_btopology", format_func=lambda t: "Series RLC" if t == "series" else "Parallel RLC")
    c = _components("ckb", topology)
    try:
        c.validate()
        b = cached("ck_bode", (c.R, c.L, c.C, topology), lambda: ck.circuit_bode(c))
    except ck.CircuitError as exc:
        st.error(str(exc))
        return
    _summary_badges(c)
    show_figure(cp.bode_figure(b, c), width="stretch")
    show_figure(cp.pole_zero_figure(c.poles, title="The circuit's poles in the s-plane"), width="stretch")
    with st.expander("The transfer functions, derived symbolically from the impedances"):
        from modules.circuits import symbolic_transfer_latex
        for name, tex in symbolic_transfer_latex(topology).items():
            st.latex(rf"H_{{{name}}}(s) = {tex}")
    render_checks(b.checks)
    render_lab_card(f"{topology.capitalize()} RLC frequency response", [f"resonance f₀ = {c.f0:.4g} Hz", f"Q = {c.Q:.4g}, ζ = {c.zeta:.4g}"], b.checks, "ck_card_bode",
                    f"R = {c.R:g} Ω, L = {c.L * 1e3:g} mH, C = {c.C * 1e6:g} µF")


def _plant_inputs() -> ck.Plant | None:
    kind = st.selectbox("Plant", ck.PLANT_KINDS, key="ck_plant_kind")
    try:
        if kind == "First order":
            return ck.make_plant(kind, K=num("Gain K", 1.0, "ck_p_K1"), tau=num("Time constant τ (s)", 1.0, "ck_p_tau1", 1e-6))
        if kind == "Second order":
            return ck.make_plant(kind, K=num("Gain K", 1.0, "ck_p_K2"), wn=num("Natural frequency ωn (rad/s)", 2.0, "ck_p_wn", 1e-6), zeta=num("Damping ratio ζ", 0.4, "ck_p_z", 0.0))
        if kind == "First order with delay (FOPDT)":
            return ck.make_plant(kind, K=num("Gain K", 2.0, "ck_p_K3"), tau=num("Time constant τ (s)", 5.0, "ck_p_tau3", 1e-6), theta=num("Dead time θ (s)", 1.0, "ck_p_th", 0.0),
                                 pade_order=int(st.selectbox("Padé order for the delay", [1, 2, 3, 4], index=2, key="ck_p_pade")))
        if kind == "Integrator with lag":
            return ck.make_plant(kind, K=num("Gain K", 1.0, "ck_p_K4"), tau=num("Time constant τ (s)", 1.0, "ck_p_tau4", 1e-6))
        if kind == "RLC capacitor voltage":
            return ck.make_plant(kind, R=num("R (Ω)", 10.0, "ck_p_R", 1e-3), L=num("L (mH)", 10.0, "ck_p_L", 1e-6) * 1e-3, C=num("C (µF)", 10.0, "ck_p_C", 1e-6) * 1e-6)
        return ck.make_plant(kind, num=st.text_input("Numerator coefficients (highest power first)", "1", key="ck_p_num"), den=st.text_input("Denominator coefficients", "1 3 3 1", key="ck_p_den"))
    except ck.CircuitError as exc:
        st.error(str(exc))
        return None


def _tuning_buttons(plant: ck.Plant) -> None:
    st.markdown("**Tune it**")
    c1, c2, c3, c4 = st.columns(4)
    pending: ck.Tuning | None = None
    ult = ck.ultimate_gain(plant)
    fop = ck.fopdt_from_step(plant)
    try:
        if c1.button("Ziegler–Nichols (ultimate gain)", key="ck_zn1", disabled=ult is None, help="Needs a plant that a proportional gain can destabilise" if ult is None else None):
            pending = ck.zn_closed_loop(ult[0], ult[1])                                   # type: ignore[index]
            st.session_state["_ck_ult_checks"] = ult[2]                                    # type: ignore[index]
        if c2.button("Ziegler–Nichols (reaction curve)", key="ck_zn2", disabled=fop is None or fop[2] <= 1e-6):
            pending = ck.zn_open_loop(*fop)                                                # type: ignore[misc]
        if c3.button("SIMC PI", key="ck_simc", disabled=fop is None):
            pending = ck.simc_pi(*fop)                                                     # type: ignore[misc]
        if c4.button("Optimise ITAE", key="ck_itae"):
            g = (st.session_state.get("ck_kp", 1.0), st.session_state.get("ck_ki", 0.2), max(st.session_state.get("ck_kd", 0.0), 0.05))
            pending = ck.optimise_itae(plant, g, st.session_state.get("ck_tf", 0.05))
    except ck.CircuitError as exc:
        st.error(str(exc))
    if pending is not None:
        st.session_state["_ck_apply"] = pending
        st.rerun()
    if ult is not None:
        st.caption(f"Ultimate gain Ku = {ult[0]:.4g}, oscillating at ω = {ult[1]:.4g} rad/s. " + ("" if fop is None else f"First-order-plus-dead-time fit: K = {fop[0]:.4g}, τ = {fop[1]:.4g} s, θ = {fop[2]:.4g} s."))
        render_checks(ult[2], "Checks on the ultimate gain")


def _control_tab() -> None:
    plant = _plant_inputs()
    if plant is None:
        return
    applied = st.session_state.pop("_ck_apply", None)
    if applied is not None:                                    # set before the widgets below are created
        st.session_state.update(ck_kp=float(applied.kp), ck_ki=float(applied.ki), ck_kd=float(applied.kd), ck_tf=float(applied.tf) if applied.tf else st.session_state.get("ck_tf", 0.05))
        st.session_state["_ck_last_tuning"] = f"{applied.method}: {applied.detail}"
    if st.session_state.get("_ck_last_tuning"):
        st.info("Applied " + st.session_state["_ck_last_tuning"])
    kp = num("Kp", 1.0, "ck_kp", 0.0)
    ki = num("Ki", 0.0, "ck_ki", 0.0)
    kd = num("Kd", 0.0, "ck_kd", 0.0)
    tf = num("Derivative filter Tf (s)", 0.05, "ck_tf", 1e-6)
    try:
        la = cached("ck_loop", (plant.name, str(plant.params), plant.pade_order, kp, ki, kd, tf), lambda: ck.analyse_loop(plant, kp, ki, kd, tf))
    except ck.CircuitError as exc:
        st.error(str(exc))
        return
    m = la.metrics
    if la.stable and m is not None:
        badge_row([("✅ stable", "pass"), (f"overshoot {m.overshoot_pct:.1f} %", "neutral"), (f"settles in {m.settling_time:.3g} s" if m.settling_time is not None else "settling not reached", "neutral"),
                   (f"phase margin {la.margins['pm']:.1f}°" if la.margins["pm"] is not None else "no gain crossover", "neutral")])
    else:
        badge_row([("unstable closed loop", "fail")])
    show_figure(cp.step_figure(la), width="stretch")
    if m is not None:
        cols = st.columns(4)
        cols[0].metric("Rise time (10–90 %)", "n/a" if m.rise_time is None else f"{m.rise_time:.3g} s")
        cols[1].metric("Steady-state error", f"{m.steady_state_error:.3g}")
        cols[2].metric("IAE", f"{m.iae:.3g}")
        cols[3].metric("ITAE", f"{m.itae:.3g}")
    render_notes(la.notes)
    render_checks(la.checks)
    render_lab_card("Feedback loop with a PID controller", ([f"overshoot {m.overshoot_pct:.1f} %", f"settling time {m.settling_time:.3g} s" if m.settling_time is not None else "settling not reached",
                                                             f"phase margin {la.margins['pm']:.1f}°" if la.margins["pm"] is not None else "no gain crossover"] if (la.stable and m is not None) else ["UNSTABLE closed loop"]),
                    la.checks, "ck_card_loop", f"{plant.description()}; Kp = {kp:g}, Ki = {ki:g}, Kd = {kd:g}")
    _tuning_buttons(plant)
    with st.expander("Frequency response and root locus"):
        show_figure(cp.loop_bode_figure(la), width="stretch")
        show_figure(cp.root_locus_figure(plant, kp), width="stretch")
        show_figure(cp.pole_zero_figure(la.poles, title="Closed-loop poles"), width="stretch")
    st.caption("Linear analysis: no saturation, noise or sampling. A delay is a Padé approximation of the order shown.")


def render_circuits_tab() -> None:
    st.subheader("🔌 Circuits & control lab")
    st.caption("RLC transients, Bode plots, step response and PID tuning. Each result is checked against the resonant frequency and damping ratio (measured from the simulation), the "
               "energy balance, Routh–Hurwitz and an explicit feedback simulation. No language model involved.")
    tabs = st.tabs(["⚡ RLC transient", "📉 Bode plot", "🎛️ Control & PID"])
    with tabs[0]:
        _transient_tab()
    with tabs[1]:
        _bode_tab()
    with tabs[2]:
        _control_tab()
