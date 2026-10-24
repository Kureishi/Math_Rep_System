"""
🔥 Statistical mechanics: six tools, each shown together with the checks that were run on it.

    Partition functions  Z, F, U, S, C from any spectrum (two-level, oscillator, rotor, typed), ideal gas, closed-form checks
    Lattice Monte Carlo  Metropolis Ising / Potts / XY with live lattice, error bars with autocorrelation times, Binder
                         crossings, finite-size scaling, and exact 4x4 references
    Mean field vs exact  mean field, Bethe, exact 1D and Onsager 2D; where mean field fails, quantitatively
    Distributions        Maxwell-Boltzmann, Fermi-Dirac, Bose-Einstein, chemical-potential solvers, BEC, classical limits
    Stochastic dynamics  Langevin (over/underdamped), MSD, Einstein relation, Green-Kubo, Fokker-Planck against Langevin
    Random matrices      GOE / GUE / Poisson / Wishart: semicircle, level spacings, Wigner surmise, ratio statistic

The mathematics lives in modules/sm_*.py (Streamlit-free, tested); this file collects inputs and shows results.
"""
import math
from functools import lru_cache

import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp

from modules import (sm_distributions, sm_lattice, sm_meanfield, sm_partition, sm_plots, sm_randommatrix, sm_stochastic)
from modules.sm_common import StatMechError
from ui.common import persist_on_click
from ui.quantum import render_checks


def _num(label: str, value: float, key: str, **kw) -> float:
    return float(st.number_input(label, value=float(value), key=key, format="%.6g", **kw))


@lru_cache(maxsize=16)
def _cached_mu_curves(n: float, t_lo: float, t_hi: float, g: int):
    temps = np.linspace(t_lo, t_hi, 40)
    return temps, sm_distributions.mu_curves(n, temps, g)


@lru_cache(maxsize=4)
def _cached_eos(g: int):
    x = np.linspace(1e-3, 2.0, 60)
    return x, sm_distributions.equation_of_state(x, 1.0, g)


def _seed(key: str) -> int | None:
    txt = st.text_input("Seed (empty = a fresh random seed, recorded with the result)", "", key=key).strip()
    if not txt:
        return None
    if not txt.isdigit():
        st.warning("The seed must be a non-negative integer; using a random one.")
        return None
    return int(txt)


def _fresh(out, sig):
    return out[1] if out and out[0] == sig else None


def _guard(fn, *args, **kwargs):
    """(result, None) or (None, message) for a StatMechError."""
    try:
        return fn(*args, **kwargs), None
    except StatMechError as e:
        return None, str(e)


# ================================================================================ 1. partition functions

def _partition_tab() -> None:
    st.write("Give a spectrum; get Z, F, U, S and C against temperature, with the checks that tie them together (Gibbs entropy, "
             "finite-difference derivatives, third law, closed forms where they exist).")
    system = st.selectbox("System", ["Two-level system", "Harmonic oscillator", "Rigid rotor", "Explicit levels", "E_n = f(n) (typed)", "Ideal gas"], key="smp_sys")
    if system == "Ideal gas":
        _ideal_gas_panel()
        return
    spec = None
    err = None
    if system == "Two-level system":
        c1, c2, c3 = st.columns(3)
        with c1:
            gap = _num("Gap Δ", 1.0, "smp_gap", min_value=1e-6)
        with c2:
            g0 = int(st.number_input("Ground degeneracy", 1, 100, 1, key="smp_g0"))
        with c3:
            g1 = int(st.number_input("Excited degeneracy", 1, 100, 1, key="smp_g1"))
        spec, err = _guard(sm_partition.two_level, gap, g0, g1)
    elif system == "Harmonic oscillator":
        c1, c2 = st.columns(2)
        with c1:
            hw = _num("ħω", 1.0, "smp_hw", min_value=1e-6)
        with c2:
            zp = st.checkbox("Include zero-point energy ħω/2", True, key="smp_zp")
        spec, err = _guard(sm_partition.harmonic_oscillator, hw, zp)
    elif system == "Rigid rotor":
        spec, err = _guard(sm_partition.rigid_rotor, _num("Rotational constant B  (E_J = B J(J+1), g = 2J+1)", 1.0, "smp_B", min_value=1e-6))
    elif system == "Explicit levels":
        c1, c2 = st.columns(2)
        with c1:
            et = st.text_input("Energies (comma-separated)", "0, 1, 1, 3", key="smp_E")
        with c2:
            gt = st.text_input("Degeneracies (comma-separated, optional)", "1, 2, 1, 5", key="smp_G")
        try:
            e = [float(x) for x in et.split(",") if x.strip()]
            g = [float(x) for x in gt.split(",") if x.strip()] or None
            spec, err = _guard(sm_partition.explicit_levels, e, g)
        except ValueError:
            err = "Energies and degeneracies must be numbers."
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            et = st.text_input("E_n", "n^2", key="smp_en")
        with c2:
            gt = st.text_input("g_n", "2*n + 1", key="smp_gn")
        with c3:
            n_min = int(st.number_input("n from", 0, 1000, 1, key="smp_nmin"))
        with c4:
            n_max = int(st.number_input("n to (0 = infinity)", 0, 100000, 0, key="smp_nmax"))
        spec, err = _guard(sm_partition.from_expression, et, gt, n_min, n_max or None)
    if err or spec is None:
        st.error(err or "No spectrum.")
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lo = _num("kT from", 0.05, "smp_tlo", min_value=1e-4)
    with c2:
        hi = _num("kT to", 10.0, "smp_thi", min_value=1e-3)
    with c3:
        npts = int(st.number_input("Points", 20, 600, 200, key="smp_np"))
    with c4:
        kB = _num("k_B", 1.0, "smp_kb", min_value=1e-12)
    if hi <= lo:
        st.error("kT 'to' must exceed 'from'.")
        return
    th, err = _guard(sm_partition.thermodynamics, spec, np.geomspace(lo, hi, npts), kB)
    if err:
        st.error(err)
        return
    st.caption(sm_partition.tail_report(spec, hi) + f". Ground-state energy {th.ground:g}, degeneracy {th.g0:g}.")
    render_checks(sm_partition.verify_thermo(spec, th))
    st.plotly_chart(sm_plots.thermo_figure(th, spec), width="stretch")
    st.plotly_chart(sm_plots.populations_figure(th), width="stretch")
    if spec.closed_form is not None:
        with st.expander("Closed forms"):
            for k, v in spec.closed_form.latex.items():
                st.latex(f"{k} = {v}")
    if system == "Two-level system":
        xn, xr = sm_partition.schottky_peak(1.0)
        st.caption(f"Schottky anomaly: C peaks at Δ/kT = {xn:.4f} (numerical maximum) = {xr:.4f} (root of eˣ = (x+2)/(x−2)), i.e. kT = {gap / xr:.4g} for this gap.")


def _ideal_gas_panel() -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        N = int(st.number_input("N particles", 1, 10 ** 9, 1000, key="smp_N"))
    with c2:
        V = _num("Volume V", 1.0e6, "smp_V", min_value=1e-9)
    with c3:
        m = _num("Mass m", 1.0, "smp_m", min_value=1e-9)
    with c4:
        t_hi = _num("kT up to", 10.0, "smp_gt", min_value=0.01)
    gas, err = _guard(sm_partition.ideal_gas, N, V, np.geomspace(0.05, t_hi, 80), m)
    if err:
        st.error(err)
        return
    st.caption("ħ = k_B = 1. Z_N = (V/λ³)^N / N!, with λ = h/√(2π m kT).")
    render_checks(gas.checks)
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gas.kT, y=gas.S, name="S (exact, with 1/N!)"))
    fig.add_trace(go.Scatter(x=gas.kT, y=gas.S_sackur_tetrode, name="Sackur–Tetrode", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=gas.kT, y=gas.S_without_gibbs_factor, name="without the 1/N! factor", line=dict(dash="dot")))
    fig.update_layout(xaxis_title="kT", yaxis_title="S / k", title="Entropy of the ideal gas (the Gibbs paradox)", height=380)
    st.plotly_chart(fig, width="stretch")


# ================================================================================ 2. lattice Monte Carlo

def _model_inputs(prefix: str, sizes: bool = False):
    c1, c2, c3 = st.columns(3)
    with c1:
        model = st.selectbox("Model", list(sm_lattice.MODELS), key=f"{prefix}_model", format_func=lambda k: {"ising": "Ising", "potts": "Potts", "xy": "XY"}[k])
    with c2:
        q = int(st.number_input("Potts q", 2, 8, 3, key=f"{prefix}_q", disabled=model != "potts"))
    with c3:
        h = _num("Field h", 0.0, f"{prefix}_h", step=0.1)
    return model, q, h


def _lattice_tab() -> None:
    sub = st.tabs(["Live lattice", "Temperature scan & critical point", "Exact 4×4 check"])
    with sub[0]:
        _live_lattice()
    with sub[1]:
        _lattice_scan()
    with sub[2]:
        _lattice_exact()


def _live_lattice() -> None:
    model, q, h = _model_inputs("sml")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        L = int(st.number_input("L (even)", 4, 128, 48, step=2, key="sml_L"))
    with c2:
        T = _num("T", 2.0, "sml_T", min_value=0.05, step=0.1)
    with c3:
        sweeps = int(st.number_input("Sweeps", 50, 20000, 600, step=50, key="sml_sweeps"))
    with c4:
        every = int(st.number_input("Snapshot every", 1, 500, 10, key="sml_every"))
    ordered = st.checkbox("Start fully ordered (otherwise random)", False, key="sml_ord")
    seed = _seed("sml_seed")
    try:
        lm = sm_lattice.LatticeModel(model, L, 1.0, h, q)
    except StatMechError as e:
        st.error(str(e))
        return
    tc, src = sm_lattice.critical_temperature(lm)
    st.caption(f"T_c = {tc:.4f} J ({src}).")
    sig = (model, q, h, L, T, sweeps, every, ordered, seed)

    def run(progress):
        out, err = _guard(sm_lattice.run_single, lm, T, sweeps, every, seed, ordered, progress)
        return sig, out if out is not None else err

    out = persist_on_click("Run", "sml_run", "sml_result", True, run, run_label="Sweeping the lattice", stoppable=True)
    res = _fresh(out, sig)
    if res is None:
        st.info("Press **Run**. Below T_c domains coarsen and order; above it the lattice stays disordered.")
    elif isinstance(res, str):
        st.error(res)
    else:
        st.caption(f"Seed {res.seed}, acceptance rate {res.acceptance:.3f}.")
        st.plotly_chart(sm_plots.lattice_animation_figure(res), width="stretch")
        st.plotly_chart(sm_plots.series_figure(res), width="stretch")


def _lattice_scan() -> None:
    model, q, h = _model_inputs("sms")
    c1, c2 = st.columns(2)
    with c1:
        sizes = st.multiselect("Lattice sizes", [4, 6, 8, 12, 16, 24, 32], default=[8, 16], key="sms_sizes")
    with c2:
        npts = int(st.number_input("Temperatures", 5, 40, 11, key="sms_np"))
    try:
        ref = sm_lattice.critical_temperature(sm_lattice.LatticeModel(model, 4, 1.0, h, q))
    except StatMechError as e:
        st.error(str(e))
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        t_lo = _num("T from", round(ref[0] * 0.85, 2), f"sms_tlo_{model}_{q}", min_value=0.05)
    with c2:
        t_hi = _num("T to", round(ref[0] * 1.15, 2), f"sms_thi_{model}_{q}", min_value=0.06)
    with c3:
        n_th = int(st.number_input("Thermalisation sweeps", 100, 20000, 400, step=100, key="sms_th"))
    with c4:
        n_me = int(st.number_input("Measurement sweeps", 500, 200000, 3000, step=500, key="sms_me"))
    seed = _seed("sms_seed")
    if t_hi <= t_lo or not sizes:
        st.warning("Choose at least one size and T from < T to.")
        return
    temps = np.round(np.linspace(t_lo, t_hi, npts), 6)
    sig = (model, q, h, tuple(sizes), tuple(temps), n_th, n_me, seed)

    def run(progress):
        scans = {}
        try:
            for i, L in enumerate(sorted(sizes)):
                sub = None
                if progress:
                    def sub(lbl, fr, i=i):                                                    # noqa: E306
                        progress(lbl, (i + (fr or 0.0)) / len(sizes))
                scans[L] = sm_lattice.scan_temperatures(sm_lattice.LatticeModel(model, L, 1.0, h, q), temps, n_th, n_me, None if seed is None else seed + L, sub)
        except StatMechError as e:
            return sig, str(e)
        return sig, scans

    out = persist_on_click("Run scan", "sms_run", "sms_result", True, run, run_label="Scanning temperatures", stoppable=True)
    scans = _fresh(out, sig)
    if scans is None:
        st.info("Press **Run scan**. Sizes 8 and 16 with the defaults take about 20 seconds; larger lattices and longer runs sharpen the estimates.")
        return
    if isinstance(scans, str):
        st.error(scans)
        return
    lm0 = sm_lattice.LatticeModel(model, 4, 1.0, h, q)
    tc, src = sm_lattice.critical_temperature(lm0)
    st.plotly_chart(sm_plots.lattice_scan_figure(scans, tc), width="stretch")
    short = [(L, p.T) for L, sc in scans.items() for p in sc.points if not p.run_long_enough]
    if short:
        st.warning(f"{len(short)} point(s) have fewer than 50 autocorrelation times of data (near T_c the autocorrelation time grows like L^z): "
                   "their error bars are unreliable. Increase the measurement sweeps.")
    drift = [(L, round(p.T, 3)) for L, sc in scans.items() for p in sc.points if abs(p.drift_z) > 4]
    if drift:
        st.warning(f"The order parameter drifted between the first and second half of the run at {len(drift)} point(s) (|z| > 4): thermalisation may be too short.")
    if len(scans) >= 2:
        est = sm_lattice.estimate_critical_point(scans)
        if est is None:
            st.info("The Binder curves do not cross inside this temperature range: widen it.")
        else:
            st.success(f"Binder-cumulant crossing: T_c ≈ {est.tc:.4f} ± {est.tc_err:.4f}  ({src}: {tc:.4f}).")
            st.dataframe(pd.DataFrame([{"sizes": f"{a} and {b}", "T_cross": t, "error": e} for a, b, t, e in est.pairs]), hide_index=True, width="stretch")
            fit = sm_lattice.finite_size_scaling(scans, est.tc)
            if fit.gamma_over_nu is not None and model == "ising" and h == 0:
                ge = f" ± {fit.gamma_err:.3f}" if fit.gamma_err is not None else ""
                be = f" ± {fit.beta_err:.3f}" if fit.beta_err is not None else ""
                st.write(f"**Finite-size scaling at T_c = {est.tc:.3f}:**  γ/ν = {fit.gamma_over_nu:.3f}{ge} (exact 7/4 = 1.75),  β/ν = {fit.beta_over_nu:.3f}{be} (exact 1/8 = 0.125)")
                st.plotly_chart(sm_plots.fss_figure(fit), width="stretch")
    st.caption("Seeds: " + ", ".join(f"L={L}: {sc.seed}" for L, sc in sorted(scans.items())))


def _lattice_exact() -> None:
    st.write("A Metropolis run on a lattice small enough to sum EVERY state, compared with the exact answer: 4×4 Ising (all 65 536 configurations) "
             "or the 4×4 q-state Potts torus (exact transfer matrix).")
    which = st.radio("Model", ["Ising 4×4", "Potts 4×4"], horizontal=True, key="sme_which")
    c1, c2, c3 = st.columns(3)
    with c1:
        T = _num("T", 2.0 if which.startswith("Ising") else 0.9, f"sme_T_{which}", min_value=0.1, step=0.1)
    with c2:
        q = int(st.number_input("Potts q", 2, 5, 3, key="sme_q", disabled=which.startswith("Ising")))
    with c3:
        sweeps = int(st.number_input("Measurement sweeps", 5000, 400000, 60000, step=5000, key="sme_sw"))
    seed = _seed("sme_seed")
    sig = (which, T, q, sweeps, seed)

    def run(progress):
        sd = sm_common_seed(seed)
        lm = sm_lattice.LatticeModel("ising" if which.startswith("Ising") else "potts", 4, 1.0, 0.0, q)
        pt, _ = sm_lattice.run_point(lm, T, np.random.default_rng(sd), 1000, sweeps, None, False, sd, progress)
        return sig, (pt, sd)

    out = persist_on_click("Run and compare", "sme_run", "sme_result", True, run, run_label="Sampling the 4×4 lattice", stoppable=True)
    got = _fresh(out, sig)
    if got is None:
        st.info("Press **Run and compare**.")
        return
    pt, sd = got
    st.caption(f"Seed {sd}; autocorrelation times τ_e = {pt.tau_e:.2f}, τ_m = {pt.tau_m:.2f}.")
    rows = []
    if which.startswith("Ising"):
        for name, val, err, ref, ok in sm_lattice.compare_with_exact_ising(pt):
            rows.append({"quantity": name, "Monte Carlo": val, "± error": err, "exact": ref, "within 4σ": "✅" if ok else "❌"})
    else:
        e_ex, c_ex = sm_lattice.torus_exact(q, 4, 4, T)
        for name, val, err, ref in (("e", pt.e, pt.e_err, e_ex), ("C", pt.C, pt.C_err, c_ex)):
            rows.append({"quantity": name, "Monte Carlo": val, "± error": err, "exact": ref, "within 4σ": "✅" if abs(val - ref) <= 4 * err else "❌"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def sm_common_seed(seed: int | None) -> int:
    from modules.sm_common import resolve_seed
    return resolve_seed(seed)


# ================================================================================ 3. mean field vs exact

def _meanfield_tab() -> None:
    st.write("The same Ising model solved four ways: mean field, the Bethe (cavity) approximation, and the exact 1D and 2D solutions. "
             "Everything below is computed, including the failures of mean field.")
    c1, c2 = st.columns(2)
    with c1:
        z = int(st.number_input("Coordination number z (approximations)", 2, 12, 4, key="smm_z"))
    with c2:
        tmax = _num("Plot up to T / J", 5.0, "smm_tmax", min_value=1.0)
    T = np.linspace(0.3, tmax, 120)
    cv = sm_meanfield.compare_curves(T, z)
    st.plotly_chart(sm_plots.meanfield_figure(cv, sm_meanfield.mf_tc(z), sm_meanfield.bethe_tc(z), sm_meanfield.onsager_tc(), None), width="stretch")
    fs = sm_meanfield.failure_summary(z)
    st.dataframe(pd.DataFrame(fs.rows, columns=["quantity", "mean field", "Bethe", "exact"]), hide_index=True, width="stretch")
    render_checks(fs.checks, "Cross-checks of the exact solutions and the approximations")
    if z != 4:
        st.caption("The exact curves are for the square lattice (z = 4); z changes only the two approximations.")
    with st.expander("Landau free energy f(m) of mean field"):
        ts = [0.5 * sm_meanfield.mf_tc(z), 0.9 * sm_meanfield.mf_tc(z), sm_meanfield.mf_tc(z), 1.3 * sm_meanfield.mf_tc(z)]
        m = np.linspace(-0.999, 0.999, 800)
        st.plotly_chart(sm_plots.landau_figure(m, {t: sm_meanfield.mf_landau_free_energy(m, t, 0.0, z) for t in ts}), width="stretch")
    st.write("---")
    st.write("**Overlay a Monte Carlo scan** on the same axes (Ising, L = 16, zero field):")
    seed = _seed("smm_seed")
    sig = (z, seed)

    def run(progress):
        lm = sm_lattice.LatticeModel("ising", 16)
        return sig, sm_lattice.scan_temperatures(lm, np.linspace(1.2, 4.0, 15), 300, 2500, seed, progress)

    out = persist_on_click("Run Monte Carlo overlay", "smm_run", "smm_result", True, run, run_label="Monte Carlo overlay", stoppable=True)
    mc = _fresh(out, sig)
    if mc is not None:
        st.plotly_chart(sm_plots.meanfield_figure(cv, sm_meanfield.mf_tc(z), sm_meanfield.bethe_tc(z), sm_meanfield.onsager_tc(), mc), width="stretch")
        st.caption(f"Seed {mc.seed}. The 16×16 data rounds the transition; they follow the exact curve, not mean field.")


# ================================================================================ 4. distributions

def _distributions_tab() -> None:
    sub = st.tabs(["Occupation numbers", "Quantum gas & chemical potential", "Maxwell–Boltzmann"])
    with sub[0]:
        c1, c2 = st.columns(2)
        with c1:
            mu = _num("μ", -0.5, "smd_mu", step=0.1)
        with c2:
            T = _num("T", 1.0, "smd_T", min_value=0.01, step=0.1)
        eps = np.linspace(max(mu, -2) + 1e-3, 10.0 * T + max(mu, 0), 500)
        st.plotly_chart(sm_plots.occupation_figure(eps, mu, T), width="stretch")
        st.caption("Bose–Einstein is drawn only for ε > μ. Far above μ all three coincide: that is the classical limit.")
    with sub[1]:
        c1, c2, c3 = st.columns(3)
        with c1:
            n = _num("Density n", 0.3, "smd_n", min_value=1e-6, step=0.05)
        with c2:
            T = _num("Temperature T", 1.0, "smd_gT", min_value=0.01, step=0.1)
        with c3:
            g = int(st.number_input("Spin states g", 1, 8, 1, key="smd_g"))
        rep, err = _guard(sm_distributions.verify_gas_tools, n, T, g)
        if err:
            st.error(err)
        else:
            lam3 = sm_distributions.thermal_wavelength(T) ** 3
            st.caption(f"Degeneracy parameter nλ³/g = {n * lam3 / g:.4g} (≪ 1 classical).  E_F = {sm_distributions.fermi_energy(n, g):.4g},  "
                       f"T_c(BEC) = {sm_distributions.bec_critical_temperature(n, g):.4g}")
            st.dataframe(pd.DataFrame([{"statistics": k, "μ": s.mu, "P": s.pressure, "u": s.energy_density, "P/(nT)": s.pressure / (s.n * s.T),
                                        "condensate fraction": s.condensate_fraction} for k, s in rep.states.items()]), hide_index=True, width="stretch")
            render_checks(rep.checks, "Consistency checks")
            temps, mus = _cached_mu_curves(n, 0.2 * T, 3 * T, g)
            st.plotly_chart(sm_plots.mu_figure(temps, mus, sm_distributions.bec_critical_temperature(n, g)), width="stretch")
            x, eos = _cached_eos(g)
            st.plotly_chart(sm_plots.eos_figure(x, eos), width="stretch")
    with sub[2]:
        c1, c2 = st.columns(2)
        with c1:
            T = _num("T", 1.0, "smd_mbT", min_value=0.01)
        with c2:
            seed = _seed("smd_seed")
        sig = (T, seed)
        out = persist_on_click("Sample velocities", "smd_run", "smd_result", True,
                               lambda: (sig, sm_distributions.mb_sample_check(T, 200000, seed)))
        got = _fresh(out, sig)
        mm = sm_distributions.mb_moments(T)
        st.dataframe(pd.DataFrame([{"quantity": k, "value": v} for k, v in mm.items()]), hide_index=True, width="stretch")
        if got is not None:
            ck, used = got
            st.caption(f"Seed {used}.")
            render_checks(ck)
        v = np.linspace(0, 6 * math.sqrt(T), 400)
        rng = np.random.default_rng(sm_distributions.resolve_seed(seed if seed is not None else 0))
        st.plotly_chart(sm_plots.mb_figure(v, T, np.linalg.norm(rng.normal(0, math.sqrt(T), (20000, 3)), axis=1)), width="stretch")


# ================================================================================ 5. stochastic dynamics

def _potential_inputs(prefix: str, default: str = "free particle"):
    options = ["free particle", "harmonic trap", "double well", "custom U(x)"]
    kind = st.selectbox("Potential", options, index=options.index(default), key=f"{prefix}_pot")
    if kind == "free particle":
        return sm_stochastic.preset_potential("free"), None
    if kind == "harmonic trap":
        k = _num("Spring constant k", 2.0, f"{prefix}_k", min_value=1e-3)
        return sm_stochastic.preset_potential("harmonic", k=k), k
    if kind == "double well":
        c1, c2 = st.columns(2)
        with c1:
            a = _num("a  (U = a x⁴/4 − b x²/2)", 1.0, f"{prefix}_a", min_value=1e-3)
        with c2:
            b = _num("b", 1.0, f"{prefix}_b")
        return sm_stochastic.preset_potential("double_well", a=a, b=b), None
    text = st.text_input("U(x)", "x^2/2 + 0.3*x", key=f"{prefix}_ut")
    pot, err = _guard(sm_stochastic.potential_from_text, text)
    if err:
        st.error(err)
        return None, None
    return pot, None


def _stochastic_tab() -> None:
    sub = st.tabs(["Langevin & MSD", "Fluctuation–dissipation", "Fokker–Planck vs Langevin"])
    with sub[0]:
        _langevin_panel()
    with sub[1]:
        _fdt_panel()
    with sub[2]:
        _fp_panel()


def _langevin_panel() -> None:
    pot, k = _potential_inputs("sst")
    if pot is None:
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        mode = st.radio("Dynamics", ["overdamped", "underdamped"], key="sst_mode")
    with c2:
        kT = _num("kT", 1.0, "sst_kT", min_value=1e-3)
    with c3:
        gamma = _num("γ", 2.0, "sst_gam", min_value=1e-3)
    with c4:
        mass = _num("mass m (underdamped)", 1.0, "sst_m", min_value=1e-3)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        dt = _num("dt", 0.01, "sst_dt", min_value=1e-5)
    with c2:
        t_max = _num("duration", 20.0, "sst_tmax", min_value=0.1)
    with c3:
        n_paths = int(st.number_input("paths", 100, 100000, 5000, step=500, key="sst_np"))
    with c4:
        x0 = _num("x₀", 0.0, "sst_x0")
    seed = _seed("sst_seed")
    sig = (pot.text, tuple(pot.params.items()), mode, kT, gamma, mass, dt, t_max, n_paths, x0, seed)

    def run(progress):
        if mode == "overdamped":
            res, err = _guard(sm_stochastic.simulate_overdamped, pot, kT, gamma, x0, dt, t_max, n_paths, 200, 0.0, seed, progress)
        else:
            res, err = _guard(sm_stochastic.simulate_underdamped, pot, kT, gamma, mass, x0, dt, t_max, n_paths, 200, 0.0, seed, None, progress)
        return sig, res if res is not None else err

    out = persist_on_click("Simulate", "sst_run", "sst_result", True, run, run_label="Simulating Langevin paths", stoppable=True)
    res = _fresh(out, sig)
    if res is None:
        st.info("Press **Simulate**.")
        return
    if isinstance(res, str):
        st.error(res)
        return
    st.caption(f"Seed {res.seed}.")
    st.plotly_chart(sm_plots.paths_figure(res), width="stretch")
    st.plotly_chart(sm_plots.msd_figure(res), width="stretch")
    ck = sm_stochastic.equipartition_checks(res, k)
    if pot.text.strip() != "0" and res.t[-1] > 5 * gamma / (k or 1.0):
        d, p = sm_stochastic.boltzmann_ks(res.x[-1], pot, kT)
        ck.add("final positions follow the Boltzmann distribution e^(−U/kT) (Kolmogorov–Smirnov)", p > 1e-3, f"KS statistic {d:.3g}, p = {p:.3g}")
    if ck.checks:
        render_checks(ck, "Equilibrium checks")


def _fdt_panel() -> None:
    st.write("The fluctuation–dissipation theorem, measured: the diffusion constant D from the spread of a force-free ensemble, the mobility μ from the "
             "drift under a force, and their ratio must equal kT, because the noise strength and the friction are the same coupling to the bath.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kT = _num("kT", 1.0, "ssf_kT", min_value=1e-3)
    with c2:
        gamma = _num("γ", 2.0, "ssf_gam", min_value=1e-3)
    with c3:
        under = st.checkbox("Underdamped (mass 1)", True, key="ssf_under")
    with c4:
        n_paths = int(st.number_input("paths", 1000, 100000, 20000, step=1000, key="ssf_np"))
    seed = _seed("ssf_seed")
    sig = (kT, gamma, under, n_paths, seed)

    def run(progress):
        res, err = _guard(sm_stochastic.einstein_relation, kT, gamma, 1.0, 0.02 if under else 0.01, 30.0 if under else 20.0, n_paths,
                          1.0 if under else None, seed, progress)
        return sig, res if res is not None else err

    out = persist_on_click("Measure D and μ", "ssf_run", "ssf_result", True, run, run_label="Measuring D and μ", stoppable=True)
    res = _fresh(out, sig)
    if res is None:
        st.info("Press **Measure D and μ**.")
        return
    if isinstance(res, str):
        st.error(res)
        return
    st.caption(f"Seed {res.seed}.")
    render_checks(res.checks)
    if under:
        gk = persist_on_click("Also check Green–Kubo", "ssf_gk", "ssf_gk_result", True, lambda: (sig, _green_kubo(kT, gamma, seed)))
        val = _fresh(gk, sig)
        if val is not None:
            d, ref = val
            st.write(f"**Green–Kubo:** D = ∫⟨v(0)v(t)⟩dt = {d:.4g}, kT/γ = {ref:.4g}  (relative difference {abs(d - ref) / ref:.1%}; truncation and sampling error).")


def _green_kubo(kT: float, gamma: float, seed: int | None):
    res = sm_stochastic.simulate_underdamped(sm_stochastic.preset_potential("free"), kT, gamma, 1.0, 0.0, 0.01, 12.0 / gamma + 5, 40000, 150, 0.0, seed)
    return sm_stochastic.green_kubo(res)


def _fp_panel() -> None:
    st.write("The same stochastic process computed two ways: a finite-volume Fokker–Planck solver (Scharfetter–Gummel fluxes, Crank–Nicolson) and the "
             "histogram of a Langevin ensemble. For the harmonic trap both are also compared with the exact Gaussian.")
    pot, k = _potential_inputs("ssp", "harmonic trap")
    if pot is None:
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kT = _num("kT", 1.0, "ssp_kT", min_value=1e-3)
    with c2:
        gamma = _num("γ", 1.0, "ssp_gam", min_value=1e-3)
    with c3:
        x0 = _num("x₀", 1.5, "ssp_x0")
    with c4:
        s0 = _num("initial width s₀", 0.3, "ssp_s0", min_value=1e-3)
    c1, c2, c3 = st.columns(3)
    with c1:
        t_max = _num("duration", 2.0, "ssp_tmax", min_value=0.05)
    with c2:
        xr = _num("x range ±", 5.0, "ssp_xr", min_value=0.5)
    with c3:
        n_paths = int(st.number_input("ensemble size", 1000, 100000, 30000, step=1000, key="ssp_np"))
    seed = _seed("ssp_seed")
    sig = (pot.text, tuple(pot.params.items()), kT, gamma, x0, s0, t_max, xr, n_paths, seed)

    def run(progress):
        try:
            sol = sm_stochastic.solve_fokker_planck(pot, kT, gamma, (-xr, xr), 600, x0, s0, t_max, t_max / 400, 60, 0.0, progress)
            rng = np.random.default_rng(sm_stochastic.resolve_seed(seed))
            start = rng.normal(x0, s0, n_paths)
            res = sm_stochastic.simulate_overdamped(pot, kT, gamma, start, t_max / 1000, t_max, n_paths, 60, 0.0, int(rng.integers(0, 2 ** 31)), progress)
            ck, l1 = sm_stochastic.langevin_vs_fp(res, sol)
            return sig, (sol, res, ck)
        except StatMechError as e:
            return sig, str(e)

    out = persist_on_click("Solve and simulate", "ssp_run", "ssp_result", True, run, run_label="Fokker–Planck and Langevin", stoppable=True)
    got = _fresh(out, sig)
    if got is None:
        st.info("Press **Solve and simulate**.")
        return
    if isinstance(got, str):
        st.error(got)
        return
    sol, res, ck = got
    exact = None
    if k is not None:
        exact = {j: sm_stochastic.ou_exact(sol.x, float(sol.t[j]), x0, s0, kT, gamma, k) for j in range(len(sol.t))}
        err = max(float(np.max(np.abs(sol.P[j] - exact[j]))) for j in range(len(sol.t)))
        ck.add("Fokker–Planck solution = exact Ornstein–Uhlenbeck Gaussian at every output time", err < 5e-3 * float(sol.P.max()), f"max |P_FP − P_exact| = {err:.2e}")
    st.plotly_chart(sm_plots.fp_figure(sol, res, exact), width="stretch")
    render_checks(sol.checks, "Fokker–Planck solver checks")
    render_checks(ck, "Fokker–Planck against Langevin")
    st.caption(f"Langevin seed {res.seed}.")


# ================================================================================ 6. random matrices

def _rmt_tab() -> None:
    sub = st.tabs(["Ensembles & level spacings", "Wishart (Marchenko–Pastur)"])
    with sub[0]:
        c1, c2, c3 = st.columns(3)
        with c1:
            kind = st.selectbox("Ensemble", list(sm_randommatrix.KINDS), key="smr_kind", format_func=lambda k: {"GOE": "GOE (real symmetric)", "GUE": "GUE (complex Hermitian)", "Poisson": "Poisson (independent levels)"}[k])
        with c2:
            N = int(st.number_input("N", 20, 1000, 200, step=20, key="smr_N"))
        with c3:
            ns = int(st.number_input("matrices", 1, 500, 60, key="smr_ns"))
        seed = _seed("smr_seed")
        rep, err = _guard(sm_randommatrix.analyse, kind, N, ns, seed)
        if err:
            st.error(err)
            return
        st.caption(f"Seed {rep.seed}. {len(rep.spacings)} unfolded spacings from the central 80% of each spectrum.")
        render_checks(rep.checks)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(sm_plots.density_figure(rep.eigenvalues, kind), width="stretch")
        with c2:
            st.plotly_chart(sm_plots.spacing_figure(rep.spacings, kind), width="stretch")
        st.dataframe(pd.DataFrame([{"reference law": k, "KS distance": d, "nearest": "◀" if k == rep.nearest else ""} for k, d in rep.distances.items()]),
                     hide_index=True, width="stretch")
        st.write(f"**Ratio statistic** ⟨r̃⟩ = {rep.ratio:.4f}  (Poisson 0.3863 exact, GOE 0.5307, GUE 0.5996 literature)")
        ck, used = sm_randommatrix.verify_surmise_exact(100000, seed)
        render_checks(ck, f"Exactness of Wigner's surmise for 2×2 matrices (seed {used})")
    with sub[1]:
        c1, c2, c3 = st.columns(3)
        with c1:
            p = int(st.number_input("p (rows)", 20, 800, 200, step=20, key="smr_p"))
        with c2:
            n = int(st.number_input("n (columns, ≥ p)", 20, 4000, 400, step=20, key="smr_n"))
        with c3:
            ns2 = int(st.number_input("matrices", 1, 200, 20, key="smr_ns2"))
        seed2 = _seed("smr_seed2")
        out, err = _guard(sm_randommatrix.verify_wishart, p, n, ns2, seed2)
        if err:
            st.error(err)
            return
        ck, eigs, used = out
        st.caption(f"Seed {used}.")
        render_checks(ck)
        st.plotly_chart(sm_plots.wishart_figure(eigs, p / n), width="stretch")


# ================================================================================ page

def render_statmech_tab() -> None:
    st.subheader("🔥 Statistical mechanics")
    st.caption("Partition functions, lattice Monte Carlo, mean field against exact solutions, quantum gases, stochastic dynamics and random matrices. "
               "k_B = 1; every result is shown with the checks that were run on it, most of them against an exact result the method shares no code with.")
    tabs = st.tabs(["Partition functions", "Lattice Monte Carlo", "Mean field vs exact", "Distributions", "Stochastic dynamics", "Random matrices"])
    with tabs[0]:
        _partition_tab()
    with tabs[1]:
        _lattice_tab()
    with tabs[2]:
        _meanfield_tab()
    with tabs[3]:
        _distributions_tab()
    with tabs[4]:
        _stochastic_tab()
    with tabs[5]:
        _rmt_tab()
