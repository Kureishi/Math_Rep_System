"""
🧰 Numerics & verification lab: the cross-cutting tools that apply to every field.

    Conserved quantities  find invariants of an ODE system automatically, watch their drift; symplectic vs non-symplectic integrators;
                          the same monitors on the app's wavepacket and geodesic simulations
    Convergence           automatic refinement with the observed order, Richardson extrapolation and a grid-convergence index
    Hamiltonian / Lagrangian   type L or H: equations of motion, Noether charges from symmetries, phase portraits, Poincaré sections
    Nondimensionalise     the scale-free form of an equation, its dimensionless groups, checked symbolically and numerically
    Limits & large N      quantum -> classical, relativistic -> Newtonian, finite -> thermodynamic limit, as checks
    Cross-validation      the closed form and a numerical route side by side
    Reproducible notebook a run exported as a notebook/script with its parameters and seed, which verifies itself on re-run

All mathematics lives in modules/cc_*.py (Streamlit-free, tested); this file collects inputs and shows results with their checks.
"""
import json
import math

import numpy as np
import pandas as pd
import streamlit as st
from ui.a11y import show_figure
import sympy as sp

from modules import (cc_common, cc_conservation, cc_convergence, cc_crossval, cc_hamiltonian, cc_limits, cc_nondim, cc_notebook, cc_plots)
from modules.cc_common import CrossCuttingError
from modules.gr_metrics import parse_in_symbols
from ui.common import persist_on_click
from ui.quantum import render_checks


def _num(label: str, value: float, key: str, **kw) -> float:
    return float(st.number_input(label, value=float(value), key=key, format="%.6g", **kw))


def _fresh(out, sig):
    return out[1] if out and out[0] == sig else None


def _guard(fn, *a, **k):
    try:
        return fn(*a, **k), None
    except (CrossCuttingError, ValueError) as e:
        return None, str(e)


# ================================================================================ 1. conserved quantities

_ODE_PRESETS = {
    "harmonic oscillator": ("x, y", "y\n-x", "1, 0", 40.0, 2, False),
    "Lotka–Volterra (needs logs)": ("x, y", "x*(1 - y)\ny*(x - 1)", "1.5, 0.8", 40.0, 1, True),
    "SIR epidemic": ("S, I, R", "-0.5*S*I\n0.5*S*I - 0.2*I\n0.2*I", "0.9, 0.1, 0", 40.0, 1, False),
    "Euler rigid body": ("x, y, z", "-y*z\nz*x/2\n-x*y/3", "1, 0.5, 0.2", 30.0, 2, False),
    "damped oscillator (no invariant)": ("x, y", "y\n-x - 0.3*y", "1, 0", 30.0, 2, False),
}


def _invariants_panel() -> None:
    st.write("Type a system dy/dt = f(y). The app searches for **conserved quantities** (polynomials, optionally with ln y terms, up to a degree), verifies each "
             "by exact differentiation, and measures how well a numerical solution conserves them: the drift, and whether it is secular or bounded.")
    preset = st.selectbox("Example", list(_ODE_PRESETS), key="lab_inv_preset")
    names_d, rhs_d, y0_d, t_d, deg_d, logs_d = _ODE_PRESETS[preset]
    c1, c2 = st.columns(2)
    with c1:
        names_txt = st.text_input("State variables", names_d, key=f"lab_inv_names_{preset}")
        rhs_txt = st.text_area("Right-hand sides (one per line, in the same order)", rhs_d, key=f"lab_inv_rhs_{preset}", height=110)
    with c2:
        y0_txt = st.text_input("Initial values", y0_d, key=f"lab_inv_y0_{preset}")
        t_end = _num("Duration", t_d, f"lab_inv_t_{preset}", min_value=0.1)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        degree = int(st.number_input("Max degree", 1, 5, deg_d, key=f"lab_inv_deg_{preset}"))
    with c2:
        logs = st.checkbox("Include ln(y) terms", logs_d, key=f"lab_inv_logs_{preset}")
    with c3:
        method = st.selectbox("Integrator", ["DOP853", "RK45", "LSODA"], key="lab_inv_method")
    with c4:
        tol_exp = st.slider("Tolerance 10^−k", 4, 13, 10, key="lab_inv_tol")
    sig = (names_txt, rhs_txt, y0_txt, t_end, degree, logs, method, tol_exp)

    def run(progress):
        try:
            names = [n.strip() for n in names_txt.split(",") if n.strip()]
            syms = {n: sp.Symbol(n, real=True) for n in names}
            lines = [ln for ln in rhs_txt.splitlines() if ln.strip()]
            if len(lines) != len(names):
                raise CrossCuttingError(f"Give {len(names)} right-hand side(s), one per state variable.")
            pnames = sorted(set().union(*[cc_common.names_in(ln) for ln in lines]) - set(names))
            psyms = {p: sp.Symbol(p, positive=True) for p in pnames}
            rhs = [parse_in_symbols(ln, {**syms, **psyms}) for ln in lines]
            if pnames:
                raise CrossCuttingError("Replace the parameter(s) " + ", ".join(pnames) + " by numbers: invariants are found for numeric parameters.")
            y0 = [float(v) for v in y0_txt.split(",")]
            state = [syms[n] for n in names]
            invs = cc_conservation.find_invariants(rhs, state, degree, logs)
            run_ = cc_conservation.integrate_monitored(rhs, state, y0, t_end, invs, method, 10.0 ** -tol_exp, 10.0 ** -(tol_exp + 2)) if invs else None
            return sig, (names, invs, run_)
        except (CrossCuttingError, ValueError) as e:
            return sig, str(e)

    out = persist_on_click("Find invariants and monitor", "lab_inv_run", "lab_inv_result", True, run, run_label="Searching for conserved quantities", stoppable=False)
    got = _fresh(out, sig)
    if got is None:
        st.info("Press **Find invariants and monitor**.")
        return
    if isinstance(got, str):
        st.error(got)
        return
    names, invs, mon = got
    if not invs:
        st.warning("No conserved quantity of this form was found. For a damped system that is the right answer (energy is dissipated); for a conservative "
                   "one, raise the degree or add ln(y) terms.")
        return
    for k, inv in enumerate(invs, 1):
        st.latex(rf"Q_{{{k}}} = {sp.latex(inv.expr)}")
        st.caption(("✅ " if inv.verified_exactly else "⚠️ ") + inv.note)
    st.dataframe(pd.DataFrame([{"quantity": r.name, "max |drift|": r.max_abs, "relative": r.max_rel, "behaviour": r.kind} for r in mon.reports]), hide_index=True, width="stretch")
    render_checks(cc_common.reports_to_checks(mon.reports, 10.0 ** -(tol_exp - 2)), "Drift along the numerical solution")
    show_figure(cc_plots.drift_figure(mon.reports), width="stretch")
    show_figure(cc_plots.trajectory_figure(mon.t, mon.y, names), width="stretch")


def _symplectic_panel() -> None:
    st.write("A symplectic integrator conserves a nearby 'shadow' Hamiltonian, so the energy error stays **bounded** forever; a non-symplectic one of equal or higher "
             "order drifts **steadily**. Same problem, same step, side by side.")
    keys = [k for k, v in cc_hamiltonian.PRESETS.items() if k not in ("free2d", "central", "polar_central", "double_pendulum") or True]
    key = st.selectbox("System", keys, key="lab_sym_sys", index=keys.index("pendulum"))
    sys_, err = _guard(cc_hamiltonian.preset_system, key)
    if err or sys_ is None or sys_.H is None:
        st.error(err or "This system has no Hamiltonian.")
        return
    n = sys_.n
    cols = st.columns(2 * n)
    z0 = []
    defaults = {"pendulum": [1.0, 0.0], "harmonic": [1.0, 0.0], "henon_heiles": [0.0, 0.1, 0.3, 0.0]}
    d = defaults.get(key, [0.5] * (2 * n))
    for i, nm in enumerate(sys_.coords + [f"p_{c}" for c in sys_.coords]):
        with cols[i]:
            z0.append(_num(nm, d[i] if i < len(d) else 0.5, f"lab_sym_z_{key}_{i}"))
    c1, c2 = st.columns(2)
    with c1:
        t_end = _num("Duration", 300.0, "lab_sym_t", min_value=1.0)
    with c2:
        dt = _num("Time step", 0.1, "lab_sym_dt", min_value=1e-4)
    sig = (key, tuple(z0), t_end, dt)

    def run(progress):
        try:
            flow = cc_hamiltonian.to_flow(sys_)
            cmp = cc_conservation.compare_integrators(flow, np.array(z0[:n]), np.array(z0[n:]), t_end, dt)
            return sig, cmp
        except (CrossCuttingError, ValueError) as e:
            return sig, str(e)

    out = persist_on_click("Compare integrators", "lab_sym_run", "lab_sym_result", True, run, run_label="Integrating", stoppable=False)
    cmp = _fresh(out, sig)
    if cmp is None:
        st.info("Press **Compare integrators**.")
        return
    if isinstance(cmp, str):
        st.error(cmp)
        return
    st.dataframe(pd.DataFrame([{"method": r.label, "symplectic": "yes" if r.symplectic else "no", "max |ΔH|/|H|": r.drift.max_rel, "energy error": r.drift.kind,
                                "RHS evaluations": r.n_rhs, "error at t_end": r.end_error} for r in cmp.runs]), hide_index=True, width="stretch")
    for note in cmp.notes:
        st.caption(note)
    show_figure(cc_plots.integrator_figure(cmp), width="stretch")
    st.caption("The error at t_end is the distance in phase space from a DOP853 reference at rtol 1e-13. A symplectic method may have a larger PHASE error and "
               "a much better ENERGY behaviour: the two are different questions.")


def _existing_panel() -> None:
    st.write("The same monitors, applied to the app's own simulations: a quantum **wavepacket** (norm, energy, and momentum when free) and a **geodesic** (energy, "
             "angular momentum, the norm, Carter's constant).")
    which = st.radio("Simulation", ["Wavepacket in a harmonic trap", "Free wavepacket", "Schwarzschild orbit"], horizontal=True, key="lab_ex_which")
    sig = (which,)

    def run(progress):
        try:
            if which.startswith("Schwarzschild"):
                from modules import gr_geodesics as gg, gr_potential as gp
                from modules.gr_metrics import preset
                E, L = gp.schwarzschild_orbit_from_turning_points(1.0, 100.0, 40.0)
                system = gg.build_system(preset("schwarzschild"), {"M": 1.0})
                x0, u0 = gg.initial_from_conserved(system, 100.0, E, L, "timelike")
                res = gg.integrate_geodesic(system, x0, u0, 3.2 * 2 * math.pi * 70 ** 1.5, n_samples=1500)
                return sig, (cc_conservation.reports_from_geodesic(res), 1e-6, "Integrator tolerance 1e-10: the drift should be at that level.")
            from modules.quantum_dynamics import evolve_wavepacket
            if which.startswith("Free"):
                r = evolve_wavepacket("free", {}, x0=-3.0, sigma=0.6, k0=2.0, domain=(-30.0, 30.0), n_points=1024, total_time=8.0, steps=800, n_frames=2, method="split-step", cross_check=False)
            else:
                r = evolve_wavepacket("harmonic", {"w": 1.0, "center": 0.0}, x0=2.0, sigma=0.7, k0=1.0, domain=(-12.0, 12.0), n_points=512, total_time=20.0, steps=2000, n_frames=2,
                                      method="split-step", cross_check=False)
            return sig, (cc_conservation.reports_from_wavepacket(r), 1e-4, "Strang splitting is unitary (the norm is conserved to rounding) but its energy error is "
                                                                     "O(dt²): bounded, oscillating, never growing — which is what 'bounded oscillation' means here.")
        except (CrossCuttingError, ValueError) as e:
            return sig, str(e)

    out = persist_on_click("Run and monitor", "lab_ex_run", "lab_ex_result", True, run, run_label="Running the simulation", stoppable=False)
    reps = _fresh(out, sig)
    if reps is None:
        st.info("Press **Run and monitor**.")
        return
    if isinstance(reps, str):
        st.error(reps)
        return
    reps, tol, note = reps
    st.caption(note)
    render_checks(cc_common.reports_to_checks(reps, tol), f"Drift (tolerance {tol:g}, relative)")
    show_figure(cc_plots.drift_figure(reps), width="stretch")


def _conserved_tab() -> None:
    sub = st.tabs(["Find invariants of an ODE system", "Symplectic integrators", "Monitors for the app's simulations"])
    with sub[0]:
        _invariants_panel()
    with sub[1]:
        _symplectic_panel()
    with sub[2]:
        _existing_panel()


# ================================================================================ 2. convergence

def _convergence_tab() -> None:
    st.write("Refine the step or the grid again and again and read off the **observed order of convergence**. A method that does not reach its theoretical order is "
             "wrong or outside its asymptotic range, and a plot of error against h shows which.")
    study = st.selectbox("Study", ["ODE method (harmonic oscillator)", "Hamiltonian integrator (pendulum)", "Finite-difference Poisson solver", "Split-step wavepacket (Strang)",
                                   "Stochastic: Euler–Maruyama / Milstein (strong order)"], key="lab_cv_study")
    sig: list = [study]
    opts: dict = {}
    c1, c2, c3 = st.columns(3)
    if study.startswith("ODE"):
        with c1:
            opts["method"] = st.selectbox("Method", list(cc_convergence.ODE_ORDERS), index=3, key="lab_cv_m")
        with c2:
            opts["levels"] = int(st.number_input("Levels", 4, 9, 6, key="lab_cv_l"))
    elif study.startswith("Hamiltonian"):
        with c1:
            opts["method"] = st.selectbox("Method", list(cc_conservation.FIXED_STEPPERS), key="lab_cv_hm")
        with c2:
            opts["levels"] = int(st.number_input("Levels", 4, 8, 6, key="lab_cv_hl"))
    elif study.startswith("Finite"):
        with c1:
            opts["expr"] = st.text_input("Manufactured exact solution u(x, y)", "sin(pi*x)*sin(pi*y) + x*y", key="lab_cv_u")
        with c2:
            opts["levels"] = int(st.number_input("Levels", 4, 6, 4, key="lab_cv_pl"))
    elif study.startswith("Split"):
        with c1:
            opts["levels"] = int(st.number_input("Levels", 4, 7, 5, key="lab_cv_wl"))
    else:
        with c1:
            opts["scheme"] = st.selectbox("Scheme", ["euler", "milstein"], key="lab_cv_s")
        with c2:
            opts["seed"] = st.text_input("Seed (empty = fresh)", "", key="lab_cv_seed").strip()
        with c3:
            opts["levels"] = int(st.number_input("Levels", 4, 8, 6, key="lab_cv_sl"))
    sig += sorted((k, str(v)) for k, v in opts.items())
    sigt = tuple(map(str, sig))

    def run(progress):
        try:
            import numpy as _np
            if study.startswith("ODE"):
                res = cc_convergence.ode_convergence(lambda t, y: _np.array([y[1], -y[0]]), [0.0, 1.0], 3.0, opts["method"], 8, opts["levels"],
                                                     lambda t: _np.array([math.sin(t), math.cos(t)]), progress)
            elif study.startswith("Hamiltonian"):
                fl = cc_hamiltonian.to_flow(cc_hamiltonian.preset_system("pendulum"))
                res = cc_convergence.hamiltonian_convergence(fl, _np.array([1.0]), _np.array([0.0]), 5.0, opts["method"], 8, opts["levels"], progress)
            elif study.startswith("Finite"):
                res = cc_convergence.poisson_convergence(opts["expr"], n0=9, n_levels=opts["levels"], progress=progress)
            elif study.startswith("Split"):
                res = cc_convergence.wavepacket_convergence(40, opts["levels"], progress=progress)
            else:
                s = cc_convergence.sde_strong_convergence(opts["scheme"], seed=int(opts["seed"]) if opts["seed"].isdigit() else None, n_levels=opts["levels"], progress=progress)
                res = s.result
                res.name += f" (seed {s.seed})"
            return sigt, res
        except (CrossCuttingError, ValueError) as e:
            return sigt, str(e)

    out = persist_on_click("Run the study", "lab_cv_run", "lab_cv_result", True, run, run_label="Refining", stoppable=True)
    res = _fresh(out, sigt)
    if res is None:
        st.info("Press **Run the study**.")
        return
    if isinstance(res, str):
        st.error(res)
        return
    st.write(f"**{res.name}**: observed order **{res.observed_order:.3f}**" + (f" (expected {res.expected_order:g})" if res.expected_order is not None else "")
             + f";  grid-convergence index of the finest level ±{res.gci:.2e} (relative)")
    rows = [{"h": float(res.h[k]), "error": float(res.errors[k]), "pairwise order": (float(res.pairwise_orders[k - 1]) if 0 < k <= len(res.pairwise_orders) else None),
             "at rounding floor": "yes" if res.at_floor[k] else ""} for k in range(len(res.h))]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    render_checks(res.checks)
    show_figure(cc_plots.convergence_figure(res), width="stretch")
    st.caption("The 'pairwise order' needs no exact answer: it comes from the ratio of successive differences. Error basis: " + res.error_basis + ".")


# ================================================================================ 3. Hamiltonian / Lagrangian

def _hamiltonian_tab() -> None:
    st.write("Type a **Lagrangian** L(q, q̇) or a **Hamiltonian** H(q, p). Velocities are written `<name>dot`, momenta `p_<name>`; any other name is a parameter.")
    keys = list(cc_hamiltonian.PRESETS) + ["custom"]
    choice = st.selectbox("System", keys, key="lab_h_sys", format_func=lambda k: "✏️ Custom…" if k == "custom" else k)
    if choice == "custom":
        kind = st.radio("You type", ["Lagrangian L", "Hamiltonian H"], horizontal=True, key="lab_h_kind")
        c1, c2 = st.columns(2)
        with c1:
            text = st.text_input("L or H", "m*xdot^2/2 - k*x^2/2", key="lab_h_text")
        with c2:
            coords = [c.strip() for c in st.text_input("Coordinates", "x", key="lab_h_coords").split(",") if c.strip()]
        params: dict[str, float] = {}
        for nm in sorted(cc_common.names_in(text) - set(coords) - {f"{c}dot" for c in coords} - {f"p_{c}" for c in coords} - {"t"}):
            params[nm] = _num(nm, 1.0, f"lab_h_p_{nm}")
        sys_, err = _guard((cc_hamiltonian.from_lagrangian if kind.startswith("Lag") else cc_hamiltonian.from_hamiltonian), text, coords, params)
    else:
        text, coords_p, defaults, kind_p, desc = cc_hamiltonian.PRESETS[choice]
        st.caption(desc)
        params = {}
        if defaults:
            cols = st.columns(len(defaults))
            for col, (k, v) in zip(cols, defaults.items()):
                with col:
                    params[k] = _num(k, v, f"lab_h_pp_{choice}_{k}")
        sys_, err = _guard(cc_hamiltonian.preset_system, choice, params)
    if err or sys_ is None:
        st.error(err)
        return
    for note in sys_.notes[1:] if choice != "custom" else sys_.notes:
        st.caption(note)
    if sys_.L is not None and sys_.accelerations is not None:
        st.write("**Euler–Lagrange equations** (solved for the accelerations)")
        for c, a in zip(sys_.coords, sys_.accelerations):
            st.latex(rf"\ddot{{{c}}} = {sp.latex(a)}")
    if sys_.H is not None:
        st.write("**Hamiltonian**" + (" (Legendre transform)" if sys_.L is not None else ""))
        st.latex("H = " + sp.latex(sys_.H))
        if sys_.dq:
            for c, d1, d2 in zip(sys_.coords, sys_.dq, sys_.dp or []):
                st.latex(rf"\dot{{{c}}} = {sp.latex(d1)}, \qquad \dot{{p}}_{{{c}}} = {sp.latex(d2)}")
        st.caption("Separable H = T(p) + V(q): " + ("yes (leapfrog and Yoshida apply)" if sys_.separable else "no (use the implicit midpoint rule)"))
    charges = []
    if sys_.L is not None:
        charges, err2 = _guard(cc_hamiltonian.noether_charges, sys_)
        if err2:
            st.error(err2)
            charges = []
        st.write("**Conserved quantities from symmetries (Noether)**")
        if charges:
            st.dataframe(pd.DataFrame([{"kind": c.kind, "symmetry": c.description, "charge": str(c.charge), "verified": "✅ " + c.note if c.verified else "⚠️ " + c.note} for c in charges]),
                         hide_index=True, width="stretch")
        else:
            st.caption("No continuous symmetry of the searched form (translations, rotations, boosts, time translation).")
    elif sys_.H is not None:
        cyc = [c for c, q in zip(sys_.coords, sys_.q) if sp.simplify(sp.diff(sys_.H, q)) == 0]
        st.write("**Conserved quantities**: energy" + ("; cyclic coordinate(s) " + ", ".join(cyc) + " (their momenta are conserved)" if cyc else ""))
    if sys_.H is None:
        return
    st.write("---")
    n = sys_.n
    z0 = [_num(nm, v, f"lab_h_z_{choice}_{nm}") for nm, v in zip(sys_.coords + [f"p_{c}" for c in sys_.coords], ([0.5, 0.3, 0.2, 0.1] * 2)[: 2 * n])]
    t_end = _num("Duration for the checks", 20.0, "lab_h_t", min_value=1.0)
    ck, err3 = _guard(cc_hamiltonian.verify_system, sys_, np.array(z0), t_end, charges)
    if err3:
        st.error(err3)
    elif ck is not None:
        render_checks(ck, "Self-checks (Legendre consistency, H and Noether charges along a trajectory, symplectic flow)")
    if n == 1:
        c1, c2 = st.columns(2)
        with c1:
            qr = _num("q range ±", 6.0, "lab_h_qr", min_value=0.5)
        with c2:
            pr = _num("p range ±", 3.0, "lab_h_pr", min_value=0.5)
        pp, e4 = _guard(cc_hamiltonian.phase_portrait, sys_, (-qr, qr), (-pr, pr))
        if e4:
            st.error(e4)
        elif pp is not None:
            show_figure(cc_plots.phase_portrait_figure(pp, sys_.coords[0]), width="stretch")
    elif n == 2:
        _poincare_panel(sys_)


def _poincare_panel(sys_) -> None:
    st.write("**Poincaré section** (two degrees of freedom): crossings of a coordinate = value with positive momentum, started on one energy surface.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        energy = _num("Energy E", 1 / 8, "lab_h_E", step=0.01)
    with c2:
        section = st.selectbox("Section coordinate", sys_.coords, key="lab_h_sec")
    with c3:
        n_cross = int(st.number_input("Crossings per orbit", 50, 1000, 200, step=50, key="lab_h_nc"))
    with c4:
        t_max = _num("Max time", 1500.0, "lab_h_tm", min_value=50.0)
    other = [c for c in sys_.coords if c != section][0]
    starts_txt = st.text_input(f"Starting values of {other} (comma-separated), with {other}'s momentum 0", "-0.2, 0.0, 0.1, 0.3", key="lab_h_starts")
    sig = (energy, section, n_cross, t_max, starts_txt, sys_.H.__str__() if sys_.H is not None else "")

    def run(progress):
        try:
            starts = []
            skipped = []
            for v in [float(x) for x in starts_txt.split(",") if x.strip()]:
                fixed = {section: 0.0, other: v, f"p_{other}": 0.0}
                try:
                    starts.append(cc_hamiltonian.initial_on_energy(sys_, energy, fixed, f"p_{section}", (0.0, 3.0)))
                except CrossCuttingError:
                    skipped.append(v)
            if not starts:
                raise CrossCuttingError("None of these starting values lies on the energy surface; try a larger energy or other values.")
            return sig, (cc_hamiltonian.poincare_section(sys_, starts, section, 0.0, n_cross, t_max), skipped)
        except (CrossCuttingError, ValueError) as e:
            return sig, str(e)

    out = persist_on_click("Compute the section", "lab_h_run", "lab_h_result", True, run, run_label="Integrating orbits", stoppable=False)
    got = _fresh(out, sig)
    if got is None:
        st.info("Press **Compute the section**. For Hénon–Heiles, E = 1/12 gives regular curves; near E = 1/6 the orbits fill areas (chaos).")
        return
    if isinstance(got, str):
        st.error(got)
        return
    ps, skipped = got
    if skipped:
        st.caption("Not accessible at this energy: " + ", ".join(f"{v:g}" for v in skipped))
    st.caption(f"Largest energy error over all crossings: {max(ps.energy_errors):.1e} (a measure of integration accuracy).")
    show_figure(cc_plots.poincare_figure(ps), width="stretch")


# ================================================================================ 4. nondimensionalise

_ND_EXAMPLES = {
    "damped, driven oscillator": ("m*x_tt + c*x_t + k*x = F*sin(w*t)", "x", "t", "x: m\nt: s\nm: kg\nc: kg/s\nk: kg/s^2\nF: N\nw: 1/s", "m: 2\nc: 0.6\nk: 8\nF: 3\nw: 1.7", "x: 0.5\nx_t: 0"),
    "heat equation": ("u_t = D*u_xx", "u", "t, x", "u: K\nt: s\nx: m\nD: m^2/s", "", ""),
    "logistic growth": ("N_t = r*N*(1 - N/K)", "N", "t", "N: 1\nt: s\nr: 1/s\nK: 1", "r: 0.7\nK: 50", "N: 4"),
    "falling with drag": ("m*v_t = -m*g - c*v", "v", "t", "v: m/s\nt: s\nm: kg\ng: m/s^2\nc: kg/s", "m: 1\ng: 9.8\nc: 0.4", "v: 3"),
}


def _parse_kv(text: str) -> dict[str, str]:
    out = {}
    for ln in text.splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _nondim_tab() -> None:
    st.write("Turn an equation into its **scale-free form**: the app finds scales for the variables from the parameters (so each has the right dimension), uses "
             "the freedom left to make coefficients 1, and reports the dimensionless groups that remain. Notation: derivatives as subscripts (`x_tt`, `u_xx`).")
    ex = st.selectbox("Example", list(_ND_EXAMPLES), key="lab_nd_ex")
    eq_d, dep_d, ind_d, units_d, vals_d, init_d = _ND_EXAMPLES[ex]
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1:
        eq = st.text_input("Equation", eq_d, key=f"lab_nd_eq_{ex}")
    with c2:
        dep = st.text_input("Dependent variable", dep_d, key=f"lab_nd_dep_{ex}")
    with c3:
        ind = st.text_input("Independent variables", ind_d, key=f"lab_nd_ind_{ex}")
    c1, c2, c3 = st.columns(3)
    with c1:
        units = st.text_area("Units (name: unit)", units_d, key=f"lab_nd_u_{ex}", height=170)
    with c2:
        vals = st.text_area("Parameter values (optional, SI)", vals_d, key=f"lab_nd_v_{ex}", height=170)
    with c3:
        init = st.text_area("Initial values at 0 (optional)", init_d, key=f"lab_nd_i_{ex}", height=170)
    try:
        pv = {k: float(v) for k, v in _parse_kv(vals).items()} or None
        iv = {k: float(v) for k, v in _parse_kv(init).items()} or None
    except ValueError:
        st.error("Values must be numbers.")
        return
    res, err = _guard(cc_nondim.nondimensionalize, eq, dep.strip(), [s.strip() for s in ind.split(",")], _parse_kv(units), pv, iv)
    if err or res is None:
        st.error(err)
        return
    st.write("**Scales** (each has the dimensions of its variable)")
    for v, s in res.scales.items():
        st.latex(rf"{v}_c = {sp.latex(s)}")
    if res.reference_scales:
        st.info("The equation has no parameter with the dimensions of " + ", ".join(r.replace("_ref", "") for r in res.reference_scales) +
                f", so a reference scale was introduced ({', '.join(res.reference_scales)}): it is yours to choose, and the dimensionless form does not depend on it.")
    st.write("**Dimensionless groups**")
    if res.groups:
        for k, v in res.groups.items():
            st.latex(rf"\{k.replace('Pi', 'Pi_')} = {sp.latex(v)}".replace("\\Pi_", "\\Pi_"))
    else:
        st.caption("None: after scaling, the equation has no free parameter.")
    st.write("**Dimensionless equation** (hats are the scaled variables)")
    st.latex(sp.latex(res.dimensionless_equation))
    st.caption(f"{res.n_parameters} parameters, rank of the dimension matrix {res.rank}: Buckingham π allows at most {res.pi_bound} independent group(s). Balanced terms: "
               + "; ".join(res.balanced))
    if pv:
        nums = cc_nondim.groups_numeric(res, {**pv, **{r: 1.0 for r in res.reference_scales}})
        if nums:
            st.dataframe(pd.DataFrame([{"group": k, "value": v} for k, v in nums.items()]), hide_index=True, width="stretch")
    if res.numerical_note:
        st.caption(res.numerical_note)
    render_checks(res.checks)
    if res.alternatives:
        with st.expander("Alternative scalings considered"):
            st.dataframe(pd.DataFrame([{"scales": ", ".join(f"{k}_c = {v}" for k, v in c.scales.items()), "groups left": c.n_groups, "balanced": "; ".join(c.balanced)}
                                       for c in res.alternatives]), hide_index=True, width="stretch")


# ================================================================================ 5. limits and large N

def _limits_tab() -> None:
    st.write("A result must reduce to the simpler theory in the right limit. These are checked: the expansion's leading term must **be** the classical formula, the first "
             "correction must match, and a high-precision numerical sequence must approach the limit at the rate the series predicts.")
    keys = list(cc_limits.LIMITS) + ["custom"]
    choice = st.selectbox("Limit", keys, key="lab_lim_key", format_func=lambda k: "✏️ Custom…" if k == "custom" else cc_limits.LIMITS[k].title)
    if choice == "custom":
        c1, c2, c3 = st.columns(3)
        with c1:
            expr = st.text_input("Expression", "m*c^2*(1/sqrt(1 - v^2/c^2) - 1)", key="lab_lim_e")
        with c2:
            param = st.text_input("Limiting parameter", "c", key="lab_lim_p")
        with c3:
            to = st.selectbox("→", ["oo", "0"], key="lab_lim_to")
        c1, c2 = st.columns(2)
        with c1:
            expected = st.text_input("Expected limiting form", "m*v^2/2", key="lab_lim_x")
        with c2:
            corr = st.text_input("Expected first correction (optional)", "", key="lab_lim_c")
        vals = {n: _num(n, 1.0 if n != param else 4.0, f"lab_lim_v_{n}") for n in sorted(cc_common.names_in(expr))}
        rep, err = _guard(cc_limits.check_limit, expr, param, to, expected, corr or None, vals)
    else:
        spec = cc_limits.LIMITS[choice]
        st.caption(spec.description)
        rep, err = _guard(cc_limits.check_preset, choice)
    if err or rep is None:
        st.error(err)
        return
    if rep.series_text:
        st.write("**Expansion**")
        st.code(rep.series_text, language="text")
    render_checks(rep.checks)
    show_figure(cc_plots.limit_figure(rep.table, "0" if choice == "custom" and to == "0" else cc_limits.LIMITS[choice].to if choice != "custom" else "oo"), width="stretch")
    st.write("---")
    st.write("**Large-N limits**")
    key = st.selectbox("Check", list(cc_limits.LARGE_N), key="lab_ln_key", format_func=lambda k: cc_limits.LARGE_N[k].title)
    rep2, err2 = _guard(cc_limits.check_large_n, key)
    if err2 or rep2 is None:
        st.error(err2)
        return
    st.caption(cc_limits.LARGE_N[key].description)
    render_checks(rep2.checks)
    st.dataframe(pd.DataFrame({"N": rep2.ns, "|f(N) − limit|": rep2.errors}), hide_index=True, width="stretch")


# ================================================================================ 6. cross-validation

def _crossval_tab() -> None:
    st.write("Where SymPy has a closed form, the app also computes the answer **numerically by an independent route** and shows both. Agreement is the evidence; a "
             "disagreement is a bug in one of them, or a domain/branch subtlety worth knowing about.")
    task = st.selectbox("Task", ["integral", "derivative", "roots", "ode", "sum", "limit", "eigenvalues", "demonstration set (all of them)"], key="lab_cx_task")
    results: list[cc_crossval.CrossValidation] = []
    err = None
    if task == "integral":
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1:
            e = st.text_input("Integrand", "exp(-x^2)*cos(a*x)", key="lab_cx_i_e")
        with c2:
            lo = st.text_input("From", "-oo", key="lab_cx_i_lo")
        with c3:
            hi = st.text_input("To", "oo", key="lab_cx_i_hi")
        with c4:
            a = _num("a (if used)", 1.5, "lab_cx_i_a")
        r, err = _guard(cc_crossval.validate_integral, e, "x", lo, hi, {"a": a})
        results = [r] if r else []
    elif task == "derivative":
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            e = st.text_input("Expression", "x^x*sin(x)", key="lab_cx_d_e")
        with c2:
            pt = _num("At x =", 1.3, "lab_cx_d_p")
        with c3:
            order = int(st.number_input("Order", 1, 4, 2, key="lab_cx_d_o"))
        r, err = _guard(cc_crossval.validate_derivative, e, "x", pt, order)
        results = [r] if r else []
    elif task == "roots":
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            e = st.text_input("f(x) = 0", "cos(x) - x", key="lab_cx_r_e")
        with c2:
            rlo = _num("from", -2.0, "lab_cx_r_lo")
        with c3:
            rhi = _num("to", 2.0, "lab_cx_r_hi")
        r, err = _guard(cc_crossval.validate_roots, e, "x", rlo, rhi)
        results = [r] if r else []
    elif task == "ode":
        c1, c2 = st.columns([3, 1])
        with c1:
            e = st.text_input("Equation (x_tt, x_t, x)", "x_tt + 3*x_t + 2*x = 0", key="lab_cx_o_e")
        with c2:
            t_end = _num("Duration", 5.0, "lab_cx_o_t", min_value=0.1)
        init_txt = st.text_input("Initial values", "x: 1, x_t: 0", key="lab_cx_o_i")
        try:
            init = {k.strip(): float(v) for k, v in (p.split(":") for p in init_txt.split(",") if p.strip())}
        except ValueError:
            init = {}
            err = "Initial values look like 'x: 1, x_t: 0'."
        if not err:
            r, err = _guard(cc_crossval.validate_ode, e, "x", "t", init, t_end)
            results = [r] if r else []
    elif task == "sum":
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            e = st.text_input("Term", "1/k^2", key="lab_cx_s_e")
        with c2:
            lo = st.text_input("From k =", "1", key="lab_cx_s_lo")
        with c3:
            hi = st.text_input("To", "oo", key="lab_cx_s_hi")
        r, err = _guard(cc_crossval.validate_sum, e, "k", lo, hi)
        results = [r] if r else []
    elif task == "limit":
        c1, c2, c3 = st.columns([3, 1, 1])
        with c1:
            e = st.text_input("Expression", "(1 + 1/x)^x", key="lab_cx_l_e")
        with c2:
            lim_pt = st.text_input("x →", "oo", key="lab_cx_l_p")
        with c3:
            d = st.selectbox("From", ["+", "-"], key="lab_cx_l_d")
        r, err = _guard(cc_crossval.validate_limit, e, "x", lim_pt, d)
        results = [r] if r else []
    elif task == "eigenvalues":
        txt = st.text_area("Matrix (one row per line, entries separated by commas)", "2, 1, 0\n1, 2, 1\n0, 1, 2", key="lab_cx_m", height=100)
        rows = [[c.strip() for c in ln.split(",")] for ln in txt.splitlines() if ln.strip()]
        r, err = _guard(cc_crossval.validate_eigenvalues, rows)
        results = [r] if r else []
    else:
        sig = ("suite",)
        out = persist_on_click("Run the demonstration set", "lab_cx_suite", "lab_cx_suite_result", True, lambda: (sig, cc_crossval.demonstration_suite()))
        got = _fresh(out, sig)
        if got is None:
            st.info("Press **Run the demonstration set**: twelve cases, including one with no closed form.")
            return
        results = got
    if err:
        st.error(err)
        return
    icon = {"agree": "✅", "DISAGREE": "❌", "numeric only": "🔢", "symbolic only": "🔣"}
    st.dataframe(pd.DataFrame([{"": icon.get(r.status, ""), "task": r.task, "problem": r.problem, "closed form": r.symbolic or "—",
                                "closed-form value": (r.symbolic_value if isinstance(r.symbolic_value, (int, float)) else None),
                                "numerical value": (r.numeric_value if isinstance(r.numeric_value, (int, float)) else None), "error": r.error, "status": r.status}
                               for r in results]), hide_index=True, width="stretch")
    for r in results[:1] if len(results) == 1 else []:
        if r.symbolic_latex:
            st.latex(r.symbolic_latex)
        st.caption(f"Numerical route: {r.numeric_method}. " + r.detail)
        if r.task == "ode" and r.symbolic_value is not None and r.numeric_value is not None:
            num = np.asarray(r.numeric_value)
            show_figure(cc_plots.crossval_figure(np.linspace(0, 1, len(num)), np.asarray(r.symbolic_value), num), width="stretch")
        if isinstance(r.symbolic_value, list) and r.task == "roots":
            st.write("Symbolic roots in range:", r.symbolic_value, " Numerical roots:", r.numeric_value)
        render_checks(r.checks)
    if len(results) > 1:
        for r in results:
            if r.detail.strip():
                st.caption(f"{r.task}: {r.detail.strip()}")


# ================================================================================ 7. reproducible notebook

def _notebook_tab() -> None:
    st.write("Export a **computational run** as a notebook or script that repeats it from its parameters and seed, then **checks that it reproduces the numbers "
             "recorded now**. The generated code is data plus a fixed template: the only things in it that vary are the literal parameters.")
    key = st.selectbox("Computation", list(cc_notebook.RECIPES), key="lab_nb_key", format_func=lambda k: cc_notebook.RECIPES[k].title)
    recipe = cc_notebook.RECIPES[key]
    st.caption(recipe.description)
    params_txt = st.text_area("Parameters (JSON)", json.dumps(recipe.defaults, indent=2), key=f"lab_nb_params_{key}", height=190)
    c1, c2 = st.columns(2)
    with c1:
        seed_txt = st.text_input("Seed (empty = a fresh one, recorded in the notebook)", "", key="lab_nb_seed", disabled=not recipe.uses_seed).strip()
    with c2:
        narrative = st.text_area("Narrative (optional, goes in the notebook)", "", key="lab_nb_narr", height=80)
    try:
        params = json.loads(params_txt)
        if not isinstance(params, dict):
            raise ValueError
    except ValueError:
        st.error("The parameters must be a JSON object, e.g. {\"gap\": 1.0}.")
        return
    seed = int(seed_txt) if seed_txt.isdigit() else None
    sig = (key, params_txt, seed, narrative)

    def run(progress):
        try:
            rec = cc_notebook.make_record(key, params, seed, narrative)
            nb = cc_notebook.build_research_notebook(rec)
            return sig, (rec, nb, cc_notebook.build_script(rec))
        except (CrossCuttingError, ValueError, KeyError, TypeError) as e:
            return sig, f"{type(e).__name__}: {e}"

    out = persist_on_click("Run and export", "lab_nb_run", "lab_nb_result", True, run, run_label="Running the computation", stoppable=False)
    got = _fresh(out, sig)
    if got is None:
        st.info("Press **Run and export**.")
        return
    if isinstance(got, str):
        st.error(got)
        return
    rec, nb, script = got
    st.success(f"Run recorded: {len(rec.summary)} values" + (f", seed {rec.seed}" if rec.seed is not None else "") + f", {rec.created}.")
    st.dataframe(pd.DataFrame([{"quantity": k, "value": v} for k, v in rec.summary.items()]), hide_index=True, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇️ Notebook (.ipynb)", nb, file_name=f"{key}.ipynb", mime="application/x-ipynb+json", key="lab_nb_dl_nb")
    with c2:
        st.download_button("⬇️ Script (.py)", script, file_name=f"{key}.py", mime="text/x-python", key="lab_nb_dl_py")
    vsig = ("verify", sig)
    vout = persist_on_click("Verify by running the exported notebook now", "lab_nb_verify", "lab_nb_verify_result", True, lambda: (vsig, cc_notebook.verify_notebook(nb)))
    ver = _fresh(vout, vsig)
    if ver is not None:
        (st.success if ver[0] else st.error)(("✅ " if ver[0] else "❌ ") + ver[1])
    with st.expander("Preview the script"):
        st.code(script, language="python")


# ================================================================================ page

def render_lab_tab() -> None:
    st.subheader("🧰 Numerics & verification lab")
    st.caption("Tools that apply to every field: conservation monitors and symplectic integrators, convergence studies, a Hamiltonian/Lagrangian front-end with Noether "
               "symmetries, non-dimensionalisation, limiting-case checks, symbolic–numeric cross-validation, and reproducible notebooks.")
    tabs = st.tabs(["Conserved quantities", "Convergence", "Hamiltonian / Lagrangian", "Nondimensionalise", "Limits & large N", "Cross-validation", "Reproducible notebook"])
    with tabs[0]:
        _conserved_tab()
    with tabs[1]:
        _convergence_tab()
    with tabs[2]:
        _hamiltonian_tab()
    with tabs[3]:
        _nondim_tab()
    with tabs[4]:
        _limits_tab()
    with tabs[5]:
        _crossval_tab()
    with tabs[6]:
        _notebook_tab()
