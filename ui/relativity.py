"""
🌌 General relativity: six tools around the metric tensor, each of which checks itself.

    Curvature & fields   Riemann/Ricci/Einstein/Weyl, four invariants, vacuum and sourced field equations,
                         singularity classification (curvature singularity vs coordinate artefact)
    Geodesics            any metric; orbits with periapsis precession, light bending, conserved-quantity monitors
    Potentials           V_eff(r) with horizons, photon sphere, ISCO and marginally bound orbit marked live
    Coordinate changes   symbolic transformations with the curvature invariants as the check
    Cones & Penrose      light cones and null rays in any coordinate plane; Penrose diagrams where a closed form exists
    Gravitational waves  linearised gravity, TT plane waves, test-particle ring, quadrupole formula, inspiral

All the mathematics lives in modules/gr_*.py (Streamlit-free, each with its own tests); this file collects inputs
and shows results, always together with the checks that were run on them.
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp

from modules import gr_cones, gr_curvature, gr_geodesics, gr_plots, gr_potential, gr_transforms, gr_waves
from modules.gr_metrics import PRESETS, GRInputError, GRMetric, parse_user_metric, preset
from modules.quantum_common import CheckList
from modules.timeout_utils import ComputationTimeoutError
from ui.common import persist_on_click
from ui.quantum import render_checks

_CUSTOM = "✏️ Custom metric…"
_ERRORS = (GRInputError, ComputationTimeoutError)


def _number(label: str, value: float, key: str, **kw) -> float:
    return float(st.number_input(label, value=float(value), key=key, format="%.6g", **kw))


def _sig(metric: GRMetric, params: dict[str, float]) -> tuple:
    return (metric.key, str(metric.g), tuple(sorted(params.items())))


def _fresh(out, sig):
    """The result stored by persist_on_click if it belongs to the CURRENT inputs, else None."""
    return out[1] if out and out[0] == sig else None


def _metric_selector(prefix: str, *, allow_custom: bool = True, only: list[str] | None = None,
                     default: str = "schwarzschild") -> tuple[GRMetric, dict[str, float]] | None:
    """A preset or custom metric, plus its parameter values. Shows its own errors; None if unusable."""
    keys = [k for k in PRESETS if only is None or k in only]
    labels = {k: preset(k).name for k in keys}
    options = keys + ([_CUSTOM] if allow_custom else [])
    choice = st.selectbox("Metric", options, index=options.index(default) if default in options else 0,
                          format_func=lambda k: labels.get(k, k), key=f"{prefix}_metric")
    if choice == _CUSTOM:
        st.caption("Metric g_ab as a grid of expressions. Any name that is not a coordinate is a parameter. "
                   "Signature (−,+,+,+), time first.")
        coord_str = st.text_input("Coordinates (comma-separated, time first)", value="t, r, theta, phi", key=f"{prefix}_coords")
        names = [c.strip() for c in coord_str.split(",") if c.strip()]
        n = max(2, min(len(names), 5))
        default_rows = [["-(1-2*M/r)", "0", "0", "0"], ["0", "1/(1-2*M/r)", "0", "0"], ["0", "0", "r^2", "0"],
                        ["0", "0", "0", "r^2*sin(theta)^2"]]
        rows = []
        for i in range(n):
            cols = st.columns(n)
            row = []
            for j in range(n):
                dv = default_rows[i][j] if n == 4 and names == ["t", "r", "theta", "phi"] else ("-1" if i == j == 0 else "1" if i == j else "0")
                with cols[j]:
                    row.append(st.text_input(f"g[{i}][{j}]", value=dv, key=f"{prefix}_g_{n}_{i}_{j}", label_visibility="collapsed"))
            rows.append(row)
        try:
            metric = parse_user_metric(rows, names, embedding="spherical" if {"r", "theta", "phi"} <= set(names) else "none")
        except GRInputError as e:
            st.error(str(e))
            return None
        metric.sample = {}
    else:
        metric = preset(choice)
        st.caption(metric.description)
    params: dict[str, float] = {}
    if metric.params:
        cols = st.columns(len(metric.params))
        for col, (name, default_value) in zip(cols, sorted(metric.params.items())):
            with col:
                params[name] = _number(name, default_value, f"{prefix}_p_{metric.key}_{name}", step=0.1)
    return metric, params


# ================================================================================ 1. curvature

def _curvature_tab() -> None:
    st.write("Christoffel symbols → Riemann → Ricci → Einstein → Weyl, with the four scalar invariants. Every run also "
             "checks identities that hold for *any* metric (∇g = 0, the Riemann symmetries, the contracted Bianchi "
             "identity ∇ₐGᵃ_b = 0, Weyl traceless, K = C² + 2Ric² − R²/3).")
    sel = _metric_selector("grc")
    if sel is None:
        return
    metric, params = sel
    sig = _sig(metric, params)

    def run(progress):
        try:
            return sig, gr_curvature.compute_curvature(metric, progress=progress, params=params)
        except _ERRORS as e:
            return sig, str(e)

    out = persist_on_click("Compute curvature", "grc_run", "grc_result", True, run, run_label="Computing curvature",
                           stoppable=False)
    curv = _fresh(out, sig)
    if curv is None:
        st.info("Choose a metric and press **Compute curvature**. A rotating (Kerr) metric takes about half a minute.")
        return
    if isinstance(curv, str):
        st.error(curv)
        return
    st.caption("Stage times: " + ", ".join(f"{k} {v:.2g}s" for k, v in curv.seconds.items()))
    if curv.is_flat:
        st.info("The Riemann tensor is identically zero: this spacetime is flat (possibly in curvilinear coordinates).")
    elif curv.is_ricci_flat:
        st.info("Ricci-flat (a vacuum solution) but not flat: the curvature is pure Weyl curvature.")
    st.write("**Scalar invariants** (coordinate-independent)")
    point = {s: v for s, v in curv.sample_subs.items()}
    for name, expr in curv.invariants.items():
        val = gr_curvature.evaluate(expr, point)
        st.latex(rf"\text{{{name}}} = {sp.latex(expr)}")
        st.caption(f"≈ {val:.5g} at the sample point " + ", ".join(f"{k} = {v:.3g}" for k, v in point.items() if k in (metric.coords or ())))
    with st.expander("Einstein tensor G_ab (nonzero components)"):
        shown = 0
        for i in range(curv.n):
            for j in range(i, curv.n):
                if curv.einstein[i, j] != 0:
                    st.latex(rf"G_{{{i}{j}}} = {sp.latex(curv.einstein[i, j])}")
                    shown += 1
        if not shown:
            st.write("All components vanish (vacuum).")
    render_checks(curv.checks)

    st.write("---")
    st.write("**Field equations**  G_ab + Λ g_ab = 8π T_ab")
    lam = _number("Cosmological constant Λ", 0.0, "grc_lambda", step=0.01)
    source = st.radio("Source T_ab", ["Vacuum (T = 0)", "Custom T_ab", "Electromagnetic field F_ab", "Read off the effective source"],
                      horizontal=True, key="grc_source")
    n = curv.n
    try:
        if source.startswith("Vacuum"):
            res = gr_curvature.verify_field_equations(curv, None, lam)
        elif source.startswith("Custom"):
            rows = _grid("grc_T", n, "T", metric.key)
            res = gr_curvature.verify_field_equations(curv, gr_curvature.parse_tensor_rows(metric, rows), lam)
        elif source.startswith("Electromagnetic"):
            st.caption("Antisymmetric field strength F_ab (Gaussian units, G = c = 1). For Reissner–Nordström: F_tr = Q/r².")
            rows = _grid("grc_F", n, "F", metric.key, default_em=(metric.key == "reissner_nordstrom"))
            res = gr_curvature.verify_field_equations(curv, gr_curvature.electromagnetic_stress_energy(curv, rows), lam)
        else:
            res = None
            T = gr_curvature.effective_stress_energy(curv, lam)
            st.write("T_ab that would source this geometry:")
            st.latex(sp.latex(T))
            try:
                fl = gr_curvature.perfect_fluid_decomposition(curv, lam)
                st.latex(rf"\rho = {sp.latex(fl.density)}")
                for k, pr in enumerate(fl.pressures):
                    st.latex(rf"p_{{{k + 1}}} = {sp.latex(pr)}")
                st.caption(fl.note)
            except GRInputError as e:
                st.caption(str(e))
    except GRInputError as e:
        st.error(str(e))
        res = None
    if res is not None:
        if res.satisfied:
            st.success("✅ The field equations are satisfied" + (" (some components settled at sample points)" if res.numeric else " exactly") + ".")
        else:
            st.error(f"❌ Not satisfied: {len(res.nonzero)} component(s) of G_ab + Λg_ab − 8πT_ab are nonzero.")
            for i, j, residual in res.nonzero[:8]:
                st.latex(rf"E_{{{i}{j}}} = {sp.latex(residual)}")

    st.write("---")
    st.write("**Singularities: curvature or just coordinates?**")
    sing = persist_on_click("Find singularities", "grc_sing", "grc_sing_result", True,
                            lambda: (sig, _singularities(curv, params)))
    loci = _fresh(sing, sig)
    if loci is None:
        st.caption("Finds where metric components or invariants blow up, then approaches each point numerically: if an "
                   "invariant diverges it is a real singularity; if they all stay finite it is a coordinate artefact.")
    elif isinstance(loci, str):
        st.error(loci)
    elif not loci:
        st.success("No singular loci found: the metric and all invariants are regular for these parameters.")
    else:
        st.dataframe(pd.DataFrame([{"Locus": s.description, "Verdict": ("🔴 " if s.kind.startswith("curv") else "🟡 ") + s.kind,
                                    "Evidence": s.evidence} for s in loci]), hide_index=True, width="stretch")


def _singularities(curv, params):
    try:
        return gr_curvature.analyze_singularities(curv, params)
    except _ERRORS as e:
        return str(e)


def _grid(prefix: str, n: int, name: str, metric_key: str, default_em: bool = False) -> list[list[str]]:
    rows = []
    for i in range(n):
        cols = st.columns(n)
        row = []
        for j in range(n):
            dv = "0"
            if default_em and (i, j) == (0, 1):
                dv = "Q/r^2"
            if default_em and (i, j) == (1, 0):
                dv = "-Q/r^2"
            with cols[j]:
                row.append(st.text_input(f"{name}[{i}][{j}]", value=dv, key=f"{prefix}_{metric_key}_{n}_{i}_{j}",
                                         label_visibility="collapsed"))
        rows.append(row)
    return rows


# ================================================================================ 2. geodesics

def _geodesic_tab() -> None:
    sub = st.tabs(["General geodesic", "Orbit & precession", "Light bending"])
    with sub[0]:
        _general_geodesic()
    with sub[1]:
        _precession_tab()
    with sub[2]:
        _bending_tab()


def _integration_options(prefix: str) -> dict:
    c1, c2, c3 = st.columns(3)
    with c1:
        method = st.selectbox("Integrator", list(gr_geodesics.METHODS), key=f"{prefix}_method")
    with c2:
        tol_exp = st.slider("Tolerance 10^−k", 6, 13, 10, key=f"{prefix}_tol")
    with c3:
        samples = int(st.number_input("Output samples", 200, 50000, 4000, step=200, key=f"{prefix}_samples"))
    return dict(method=method, rtol=10.0 ** -tol_exp, atol=10.0 ** -(tol_exp + 2), n_samples=samples)


def _show_geodesic(res: gr_geodesics.GeodesicResult, key: str) -> None:
    st.caption(f"{res.reason}. {len(res.lam)} samples, {res.n_rhs} derivative evaluations, {res.method}, rtol {res.rtol:g}.")
    render_checks(res.checks, "Conserved quantities and run checks")
    st.plotly_chart(gr_plots.orbit_figure_2d(res, animate=st.checkbox("Animate the path", key=f"{key}_anim")), width="stretch")
    with st.expander("3D view"):
        st.plotly_chart(gr_plots.orbit_figure_3d(res), width="stretch")
    st.plotly_chart(gr_plots.conserved_figure(res), width="stretch")


def _general_geodesic() -> None:
    sel = _metric_selector("grg")
    if sel is None:
        return
    metric, params = sel
    names = metric.coord_names
    kind = st.radio("Particle", ["timelike", "null"], horizontal=True, key="grg_kind",
                    format_func=lambda k: "massive particle" if k == "timelike" else "light ray")
    can_el = names[:4] == ["t", "r", "theta", "phi"]
    modes = (["Equatorial orbit from E and L"] if can_el else []) + ["Coordinate velocities"]
    mode = st.radio("Initial conditions", modes, horizontal=True, key="grg_mode")
    x0: list[float] | None = None
    u_spatial: list[float] | None = None
    el: tuple[float, float, float, bool] | None = None
    if mode.startswith("Equatorial"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            r0 = _number("Starting radius r", 20.0, "grg_r0", step=1.0)
        with c2:
            E = _number("Energy E", 0.97 if kind == "timelike" else 1.0, "grg_E", step=0.01)
        with c3:
            L = _number("Angular momentum L", 4.2 if kind == "timelike" else 6.0, "grg_L", step=0.1)
        with c4:
            inward = st.checkbox("Moving inward", True, key="grg_inward")
        el = (r0, E, L, inward)
    else:
        st.write("Position and the spatial part of the 4-velocity (u⁰ is fixed by the norm):")
        cols = st.columns(len(names))
        defaults = metric.sample_point()
        x0 = [float(cols[i].number_input(f"{nm}", value=float(defaults[i]), key=f"grg_x_{metric.key}_{nm}", format="%.6g")) for i, nm in enumerate(names)]
        cols = st.columns(len(names) - 1)
        u_spatial = [float(cols[i].number_input(f"u^{nm}", value=0.0 if i else 0.1, key=f"grg_u_{metric.key}_{nm}", format="%.6g"))
                     for i, nm in enumerate(names[1:])]
    c1, c2 = st.columns(2)
    with c1:
        lam_max = _number("Affine parameter range λ", 400.0, "grg_lam", min_value=1e-3)
    with c2:
        auto_stop = st.checkbox("Stop just outside the horizon", True, key="grg_stop")
    opts = _integration_options("grg")
    sig = (_sig(metric, params), kind, mode, el, tuple(x0 or ()), tuple(u_spatial or ()), lam_max, auto_stop, tuple(opts.items()))

    def run(progress):
        try:
            system = gr_geodesics.build_system(metric, params)
            if el is not None:
                xs, us = gr_geodesics.initial_from_conserved(system, el[0], el[1], el[2], kind, inward=el[3])
            else:
                assert x0 is not None and u_spatial is not None
                xs = x0
                us = gr_geodesics.normalise_velocity(system, x0, u_spatial, kind)
            r_stop = gr_geodesics.default_r_stop(metric, params) if auto_stop else None
            return sig, gr_geodesics.integrate_geodesic(system, xs, us, lam_max, kind=kind, r_stop=r_stop, progress=progress, **opts)
        except _ERRORS as e:
            return sig, str(e)

    out = persist_on_click("Integrate geodesic", "grg_run", "grg_result", True, run, run_label="Integrating the geodesic", stoppable=True)
    res = _fresh(out, sig)
    if res is None:
        st.info("Set the initial conditions and press **Integrate geodesic**. Stop works between chunks of the run.")
    elif isinstance(res, str):
        st.error(res)
    else:
        _show_geodesic(res, "grg")


def _precession_tab() -> None:
    st.write("A bound Schwarzschild orbit, integrated and compared with the **exact** periapsis advance (elliptic "
             "integrals) and the first-order Einstein result 6πM/p.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        M = _number("Mass M", 1.0, "grp_M", min_value=0.01)
    with c2:
        ra = _number("Apoapsis r_a", 100.0, "grp_ra", min_value=1.0)
    with c3:
        rp = _number("Periapsis r_p", 40.0, "grp_rp", min_value=0.5)
    with c4:
        orbits = int(st.number_input("Orbits", 2, 30, 4, key="grp_orbits"))
    opts = _integration_options("grp")
    sig = (M, ra, rp, orbits, tuple(opts.items()))

    def run(progress):
        try:
            E, L = gr_potential.schwarzschild_orbit_from_turning_points(M, ra, rp)
            exact = gr_potential.schwarzschild_precession(M, ra, rp)
            weak = gr_potential.weak_field_precession(M, ra, rp)
            met = preset("schwarzschild")
            system = gr_geodesics.build_system(met, {"M": M})
            xs, us = gr_geodesics.initial_from_conserved(system, ra, E, L, "timelike", inward=True)
            a = (ra + rp) / 2
            period = 2 * math.pi * math.sqrt(a ** 3 / M)
            res = gr_geodesics.integrate_geodesic(system, xs, us, (orbits + 0.2) * period, progress=progress, **opts)
            return sig, (res, E, L, exact, weak)
        except _ERRORS as e:
            return sig, str(e)

    out = persist_on_click("Integrate orbit", "grp_run", "grp_result", True, run, run_label="Integrating the orbit", stoppable=True)
    got = _fresh(out, sig)
    if got is None:
        st.info("Press **Integrate orbit**.")
        return
    if isinstance(got, str):
        st.error(got)
        return
    res, E, L, exact, weak = got
    adv = gr_geodesics.periapsis_advance(res)
    st.caption(f"E = {E:.8g}, L = {L:.8g}")
    if adv:
        mean = float(np.mean(adv))
        err = abs(mean - exact)
        res.checks.add("measured periapsis advance matches the exact elliptic-integral result", err < 1e-6 * max(1.0, abs(exact)) + 1e-8,
                       f"measured {mean:.9g} rad/orbit over {len(adv)} orbit(s), exact {exact:.9g}, difference {err:.1e}")
        st.dataframe(pd.DataFrame([{"Quantity": "measured advance per orbit (integrated geodesic)", "radians": mean, "arcsec": mean * 206264.806},
                                   {"Quantity": "exact (elliptic integral)", "radians": exact, "arcsec": exact * 206264.806},
                                   {"Quantity": "first order 6πM/p", "radians": weak, "arcsec": weak * 206264.806}]),
                     hide_index=True, width="stretch")
    else:
        st.warning("Fewer than two periapses were reached: increase the number of orbits.")
    _show_geodesic(res, "grp")


def _bending_tab() -> None:
    st.write("Light rays from far away passing a Schwarzschild mass, compared with the **exact** deflection integral and "
             "the weak-field 4M/b + 15πM²/4b².")
    c1, c2, c3 = st.columns(3)
    with c1:
        M = _number("Mass M", 1.0, "grb_M", min_value=0.01)
    with c2:
        b_text = st.text_input("Impact parameters b (comma-separated)", "8, 12, 20, 40", key="grb_b")
    with c3:
        R = _number("Start / end radius R", 2000.0, "grb_R", min_value=10.0)
    sig = (M, b_text, R)

    def run(progress):
        try:
            bs = [float(x) for x in b_text.split(",") if x.strip()]
            if not bs:
                raise GRInputError("Give at least one impact parameter.")
            met = preset("schwarzschild")
            system = gr_geodesics.build_system(met, {"M": M})
            rows, rays = [], []
            for k, b in enumerate(bs):
                if progress:
                    progress(f"Ray {k + 1} of {len(bs)}", k / len(bs))
                xs, us = gr_geodesics.initial_from_conserved(system, R, 1.0, b, "null", inward=True)
                res = gr_geodesics.integrate_geodesic(system, xs, us, 4 * R, kind="null", r_return=R, r_stop=2.01 * M, n_samples=3000)
                captured = "stopping radius" in res.reason
                measured = None if captured else float(res.x[-1, 3] - res.x[0, 3]) - math.pi
                exact = None if b <= 3 * math.sqrt(3) * M else gr_potential.schwarzschild_deflection(M, b, R)
                rows.append({"b": b, "measured (integrator)": measured, "exact": exact, "weak field": gr_potential.weak_field_deflection(M, b),
                             "status": "captured" if captured else "escapes", "max norm drift": res.conserved[0].drift_abs if res.conserved else None})
                rays.append((b, res))
            return sig, (rows, rays)
        except (ValueError,) + _ERRORS as e:
            return sig, str(e)

    out = persist_on_click("Trace rays", "grb_run", "grb_result", True, run, run_label="Tracing light rays", stoppable=True)
    got = _fresh(out, sig)
    if got is None:
        st.info("Press **Trace rays**. b below 3√3·M ≈ 5.2 M falls into the black hole.")
        return
    if isinstance(got, str):
        st.error(got)
        return
    rows, rays = got
    ck = CheckList()
    for r in rows:
        if r["measured (integrator)"] is not None and r["exact"] is not None:
            d = abs(r["measured (integrator)"] - r["exact"])
            ck.add(f"b = {r['b']:g}: integrator vs exact deflection", d < 1e-6, f"difference {d:.1e} rad")
    render_checks(ck, "Deflection checks")
    st.caption("Radians; 'exact' is the integral between radius R and R (what a ray starting at R actually sweeps), the weak-field "
               "value is the infinite-distance deflection, so they differ by about 2b/R at finite R.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.plotly_chart(gr_plots.deflection_figure(rays, 2 * M), width="stretch")


# ================================================================================ 3. potentials

def _potential_tab() -> None:
    st.write("The effective potential governs radial motion: a particle with energy E moves where E² ≥ V(r). Radii are "
             "recomputed live as you change the parameters.")
    family = st.radio("Spacetime", ["Static, f(r)", "Kerr (equatorial plane)"], horizontal=True, key="grv_family")
    light = st.checkbox("Light (null) instead of a massive particle", False, key="grv_light")
    l_text = st.text_input("Angular momenta L (comma-separated)", "3.5, 4, 4.5, 5, 6", key="grv_L")
    try:
        Ls = [float(x) for x in l_text.split(",") if x.strip()]
    except ValueError:
        st.error("Angular momenta must be numbers.")
        return
    if family.startswith("Static"):
        options = {"schwarzschild": "Schwarzschild", "reissner_nordstrom": "Reissner–Nordström",
                   "schwarzschild_de_sitter": "Schwarzschild–de Sitter", "custom": "Custom f(r)"}
        choice = str(st.selectbox("Metric", list(options), format_func=lambda k: options[k], key="grv_static"))
        try:
            if choice == "custom":
                f_text = st.text_input("f(r)  (ds² = −f dt² + dr²/f + r² dΩ²)", "1 - 2*M/r + Q^2/r^2 - Lambda*r^2/3", key="grv_f")
                st_sp = gr_potential.static_from_text(f_text, {"M": 1.0, "Q": 0.0, "Lambda": 0.0})
            else:
                st_sp = gr_potential.static_from_metric(preset(choice))
        except GRInputError as e:
            st.error(str(e))
            return
        params = {}
        if st_sp.params:
            cols = st.columns(len(st_sp.params))
            for col, (nm, dv) in zip(cols, sorted(st_sp.params.items())):
                with col:
                    params[nm] = _number(nm, dv, f"grv_p_{choice}_{nm}", step=0.05)
        r_max = _number("Largest r shown", 30.0, "grv_rmax", min_value=3.0)
        try:
            markers = gr_potential.static_markers(st_sp, params, r_max=max(r_max * 2, 40.0))
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not analyse f(r): {e}")
            return
        r_hor = [m.r for m in markers if m.kind == "horizon"]
        r0 = (max(r_hor) if r_hor else 0.0) * 1.02 + 0.02
        r = np.linspace(max(r0, 0.05), r_max, 1500)
        curves = [(f"L = {L:g}", gr_potential.veff_static(st_sp, L, r, params, light)) for L in Ls]
        shown = [m for m in markers if not (light and m.kind in ("ISCO", "stability limit", "marginally bound"))]
        e_sq = st.number_input("Optional: show a particle energy E²", value=0.0, key="grv_E2", format="%.5g")
        st.plotly_chart(gr_plots.veff_figure(r, curves, shown, e_sq if e_sq > 0 else None,
                                             "V(r) = f (κ + L²/r²)", "Effective potential"), width="stretch")
        st.dataframe(pd.DataFrame([{"Feature": m.kind, "r": round(m.r, 6), "stable": "" if m.stable is None else ("yes" if m.stable else "no")}
                                   for m in markers]), hide_index=True, width="stretch")
        ck = CheckList()
        for m in markers:
            if m.kind in ("ISCO", "photon sphere"):
                orb = gr_potential.circular_orbit_static(st_sp, m.r, params)
                if orb is not None and m.kind == "ISCO":
                    E, L = orb
                    v = float(gr_potential.veff_static(st_sp, L, np.array([m.r]), params)[0])
                    ck.add(f"circular orbit at the ISCO r = {m.r:.4g}: V = E²", abs(v - E * E) < 1e-8, f"V − E² = {v - E * E:.1e}")
        render_checks(ck, "Cross-checks")
        if choice == "schwarzschild":
            M = params.get("M", 1.0)
            st.caption(f"Textbook values for M = {M:g}: horizon {2 * M:g}, photon sphere {3 * M:g}, marginally bound {4 * M:g}, ISCO {6 * M:g}.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            M = _number("Mass M", 1.0, "grv_kM", min_value=0.01)
        with c2:
            a = st.slider("Spin a/M", -1.0, 1.0, 0.7, 0.01, key="grv_ka")
        try:
            kc = gr_potential.kerr_characteristics(M, a * M)
        except GRInputError as e:
            st.error(str(e))
            return
        r = np.linspace(max(kc.outer_horizon or 0.1, 0.1) * 1.02, 30 * M, 1500)
        spin = a * M
        curves = [(f"L = {L:g}", gr_potential.kerr_veff(r, L, M, spin, light)) for L in Ls]
        markers = [gr_potential.Marker("outer horizon", kc.outer_horizon or 0, "horizon"),
                   gr_potential.Marker("photon orbit (prograde)", kc.photon_prograde, "photon sphere", False),
                   gr_potential.Marker("photon orbit (retrograde)", kc.photon_retrograde, "photon sphere", False)]
        if not light:
            markers += [gr_potential.Marker("ISCO (prograde)", kc.isco_prograde, "ISCO", True),
                        gr_potential.Marker("ISCO (retrograde)", kc.isco_retrograde, "ISCO", True)]
        st.plotly_chart(gr_plots.veff_figure(r, curves, markers, None, "E₊(r; L)", "Kerr equatorial effective potential (motion needs E ≥ E₊)"),
                        width="stretch")
        st.caption("Negative a is a retrograde disc; the figure uses L > 0, so a < 0 and L > 0 is the counter-rotating case. "
                   "Radii below are for |a|.")
        st.dataframe(pd.DataFrame([
            {"Feature": "outer / inner horizon", "prograde": f"{kc.outer_horizon:.6g} / {kc.inner_horizon:.6g}", "retrograde": ""},
            {"Feature": "ergosphere (equator)", "prograde": f"{kc.ergosphere_equator:.6g}", "retrograde": ""},
            {"Feature": "photon orbit", "prograde": f"{kc.photon_prograde:.6g}", "retrograde": f"{kc.photon_retrograde:.6g}"},
            {"Feature": "marginally bound", "prograde": f"{kc.bound_prograde:.6g}", "retrograde": f"{kc.bound_retrograde:.6g}"},
            {"Feature": "ISCO", "prograde": f"{kc.isco_prograde:.6g}", "retrograde": f"{kc.isco_retrograde:.6g}"}]),
            hide_index=True, width="stretch")
        ck = CheckList()
        ck.add("closed-form ISCO = numerical minimum of L along circular orbits", kc.numeric_isco_error < 2e-3 * M,
               f"difference {kc.numeric_isco_error:.1e}")
        ck.add("circular-orbit formulas satisfy R = R' = 0 at both ISCOs", kc.circular_residual < 1e-5, f"residual {kc.circular_residual:.1e}")
        render_checks(ck, "Cross-checks")


# ================================================================================ 4. coordinate changes

@dataclass
class _TransformSpec:
    source: str
    new_names: list[str]
    exprs: list[str]
    direction: str
    try_invert: bool = True


def _transform_tab() -> None:
    st.write("Transform a metric to new coordinates, symbolically. The check: the curvature invariants must come out "
             "**unchanged**, and the Ricci tensor must transform as a tensor.")
    options = list(gr_transforms.TRANSFORM_PRESETS) + ["custom"]
    choice = str(st.selectbox("Transformation", options, key="grt_choice",
                          format_func=lambda k: "✏️ Custom…" if k == "custom" else gr_transforms.TRANSFORM_PRESETS[k].label))
    if choice == "custom":
        c1, c2 = st.columns(2)
        with c1:
            src = st.selectbox("Original metric", list(PRESETS), format_func=lambda k: preset(k).name, key="grt_src")
        with c2:
            direction = st.radio("You give", ["old coordinates in terms of new", "new coordinates in terms of old"], key="grt_dir")
        base = preset(src)
        new_text = st.text_input("New coordinate names", ", ".join(base.coord_names), key=f"grt_new_{src}")
        new_names = [s.strip() for s in new_text.split(",") if s.strip()]
        st.caption("One expression per coordinate " + ("(old coordinate i as a function of the NEW names)" if direction.startswith("old") else
                                                      "(new coordinate i as a function of the OLD names)") + "; parameters as in the metric.")
        exprs = [st.text_input(f"{'old' if direction.startswith('old') else 'new'} {i}", value=(new_names[i] if i < len(new_names) else ""),
                               key=f"grt_e_{src}_{direction[:3]}_{i}") for i in range(base.dim)]
        spec = _TransformSpec(src, new_names, exprs, "old_in_terms_of_new" if direction.startswith("old") else "new_in_terms_of_old")
    else:
        p = gr_transforms.TRANSFORM_PRESETS[choice]
        spec = _TransformSpec(p.source, list(p.new_names), list(p.exprs), p.direction, p.try_invert)
        st.caption(p.note)
        st.code("\n".join(f"{n} ←→ {e}" for n, e in zip(p.new_names, p.exprs)), language="text")
    sig = (choice, tuple(map(str, spec.exprs)), tuple(spec.new_names), spec.source)

    def run(progress):
        try:
            base_metric = preset(spec.source)
            forward = spec.direction == "old_in_terms_of_new"
            return sig, gr_transforms.transform_metric(
                base_metric, list(spec.new_names), progress=progress, try_invert=spec.try_invert,
                old_in_terms_of_new=list(spec.exprs) if forward else None,
                new_in_terms_of_old=None if forward else list(spec.exprs))
        except _ERRORS as e:
            return sig, str(e)

    out = persist_on_click("Transform", "grt_run", "grt_result", True, run, run_label="Transforming the metric", stoppable=False)
    res = _fresh(out, sig)
    if res is None:
        st.info("Press **Transform**.")
        return
    if isinstance(res, str):
        st.error(res)
        return
    st.write(f"**{res.mode.capitalize()} transformation**" + (" (SymPy found the inverse, so it was made explicit)" if res.inverted else ""))
    where = "the new coordinates" if res.mode == "explicit" else "the ORIGINAL coordinates (the same tensor, written in the new basis)"
    st.write(f"New metric components, as functions of {where}:")
    st.latex(sp.latex(res.components))
    st.write("Invariants, before and after:")
    for c in res.comparisons:
        st.latex(rf"\text{{{c.name}}}:\quad {sp.latex(c.old)} \;\longrightarrow\; {sp.latex(c.new)}")
    render_checks(res.checks)


# ================================================================================ 5. cones and Penrose

def _cones_tab() -> None:
    sel = _metric_selector("grc2", default="schwarzschild")
    if sel is None:
        return
    metric, params = sel
    names = metric.coord_names
    c1, c2 = st.columns(2)
    with c1:
        ta = st.selectbox("Horizontal coordinate", names, index=min(1, len(names) - 1), key="grc2_a")
    with c2:
        tb = st.selectbox("Vertical coordinate", names, index=0, key="grc2_b")
    if ta == tb:
        st.warning("Choose two different coordinates.")
        return
    d = dict(zip(names, metric.sample_point()))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        a_lo = _number(f"{ta} from", d[ta] * 0.4 if ta == "r" else d[ta] - 4, f"grc2_alo_{metric.key}_{ta}")
    with c2:
        a_hi = _number(f"{ta} to", d[ta] * 1.6 if ta == "r" else d[ta] + 4, f"grc2_ahi_{metric.key}_{ta}")
    with c3:
        b_lo = _number(f"{tb} from", -3.0, f"grc2_blo_{metric.key}_{tb}")
    with c4:
        b_hi = _number(f"{tb} to", 3.0, f"grc2_bhi_{metric.key}_{tb}")
    if a_hi <= a_lo or b_hi <= b_lo:
        st.warning("The ranges must have from < to.")
        return
    c1, c2 = st.columns(2)
    with c1:
        n_a = int(st.slider("Cones across", 3, 15, 9, key="grc2_na"))
        n_b = int(st.slider("Cones up", 2, 10, 5, key="grc2_nb"))
    with c2:
        rays = st.checkbox("Draw null rays through each column", True, key="grc2_rays")
    time_axis = tb if names.index(tb) == 0 else ta
    fixed = {n: d[n] for n in names if n not in (ta, tb)}
    if fixed:
        with st.expander("Values of the other coordinates (held fixed)"):
            for n in list(fixed):
                fixed[n] = _number(n, fixed[n], f"grc2_fix_{metric.key}_{n}")
    try:
        ga, gb = np.linspace(a_lo, a_hi, n_a), np.linspace(b_lo, b_hi, n_b)
        aspect = (b_hi - b_lo) / (a_hi - a_lo)
        cones = gr_cones.light_cones(metric, (ta, tb), ga, gb, fixed, params, aspect=aspect, time_axis=time_axis)
        curves = []
        if rays:
            length = 0.9 * (b_hi - b_lo) if time_axis == tb else 0.9 * (a_hi - a_lo)
            for a in ga:
                for br in (0, 1):
                    curves.append(gr_cones.null_curve(metric, (ta, tb), (float(a), float(gb[0])), br, length, fixed, params,
                                                      time_axis=time_axis, bounds=((a_lo, a_hi), (b_lo, b_hi))))
    except GRInputError as e:
        st.error(str(e))
        return
    scale = 0.35 * (a_hi - a_lo) / n_a
    st.plotly_chart(gr_plots.cones_figure(cones, (ta, tb), curves, scale=scale, aspect=aspect,
                                          title=f"Light cones of {metric.name}"), width="stretch")
    n_closed = sum(1 for c in cones if not c.open)
    if n_closed:
        st.warning(f"{n_closed} cone(s) are degenerate (red ×): the metric restricted to this plane is not Lorentzian there "
                   "(a horizon in these coordinates, or the plane leaves the Lorentzian region).")
    st.caption("Axes have different units, so the picture is drawn in display units: only angles, tilts and ordering are meaningful.")

    st.write("---")
    st.write("**Penrose diagram**")
    kind = gr_cones.penrose_supported(metric)
    if kind == "minkowski":
        st.plotly_chart(gr_plots.penrose_figure(gr_cones.minkowski_penrose(), "Penrose diagram of Minkowski space"), width="stretch")
    elif kind == "schwarzschild":
        M = params.get("M", 1.0)
        st.plotly_chart(gr_plots.penrose_figure(gr_cones.schwarzschild_penrose(M), "Maximally extended Schwarzschild"), width="stretch")
        st.caption("Computed from Kruskal–Szekeres coordinates: horizons at 45°, singularities r = 0 at the top and bottom, "
                   "two asymptotic regions. Radial light rays are 45° lines.")
    else:
        st.info("A Penrose diagram needs a conformal compactification, and there is no general algorithm for an arbitrary metric. "
                "Closed forms are built in for Minkowski and Schwarzschild; for other metrics the light cones and null rays above "
                "show the causal structure in the chosen coordinates.")


# ================================================================================ 6. waves

def _waves_tab() -> None:
    sub = st.tabs(["Plane wave & test masses", "Quadrupole radiation", "Inspiral"])
    with sub[0]:
        _plane_wave_tab()
    with sub[1]:
        _quadrupole_tab()
    with sub[2]:
        _inspiral_tab()


def _plane_wave_tab() -> None:
    st.write("A transverse-traceless plane wave h_ij = A₊e₊cos(ωt − k n·x) + A×e×cos(…+δ), verified against the linearised vacuum "
             "equations, and the motion of free test masses in its field.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        direction = st.text_input("Direction n", "0, 0, 1", key="grw_n")
    with c2:
        ap = _number("A₊", 0.2, "grw_ap", step=0.05)
    with c3:
        ac = _number("A×", 0.0, "grw_ac", step=0.05)
    with c4:
        om = _number("ω", 2.0, "grw_om", min_value=0.1, step=0.1)
    c1, c2 = st.columns(2)
    with c1:
        off_shell = st.checkbox("Make it off-shell (k ≠ ω) to see the check fail", False, key="grw_off")
    with c2:
        delta = _number("Cross phase δ", 0.0, "grw_delta", step=0.1)
    try:
        n_vec = [float(v) for v in direction.split(",")]
        if len(n_vec) != 3:
            raise ValueError
        w = gr_waves.plane_wave(n_vec, ap, ac, om, wave_number=om * 1.5 if off_shell else None, cross_phase=delta)
    except (ValueError, GRInputError) as e:
        st.error(str(e) if isinstance(e, GRInputError) else "Direction must be three numbers, e.g. 0, 0, 1.")
        return
    st.latex(r"h_{ij} = " + sp.latex(w.h[1:, 1:].applyfunc(lambda e: sp.nsimplify(sp.N(e, 6), rational=False))))
    render_checks(gr_waves.check_plane_wave(w), "Gauge conditions and field equations")
    if abs(n_vec[0]) + abs(n_vec[1]) < 1e-12:
        times = np.linspace(0, 4 * math.pi / om, 120)
        pos = gr_waves.ring_positions(om, ap, ac, times, cross_phase=delta)
        st.plotly_chart(gr_plots.ring_animation_figure(pos, times), width="stretch")
        pos_num, err = gr_waves.geodesic_deviation_ring(om, ap, ac, times, cross_phase=delta)
        rc = gr_waves.free_particle_check(min(ap, 0.05), min(ac, 0.05), om, 4 * math.pi / om)
        ck = CheckList()
        ck.add("geodesic-deviation equation integrated numerically = closed-form ring", err < 1e-8, f"max difference {err:.1e}")
        ck.add("geodesic integrator: free particles stay at fixed TT coordinates", rc.max_coordinate_drift < 1e-9,
               f"largest coordinate drift {rc.max_coordinate_drift:.1e} (Γⁱ₀₀ = 0 exactly in TT gauge)")
        ck.add("proper separation = L₀√g_xx = L₀(1 + h/2) to first order", rc.max_separation_error < 5e-4,
               f"difference {rc.max_separation_error:.1e} (second order in the amplitude 0.05)")
        render_checks(ck, "Test-mass checks")
    else:
        st.caption("The test-mass ring is drawn for waves travelling along z (the ring lies in the x-y plane of the detector frame).")


def _parse_source(text: str, t: sp.Symbol):
    masses, positions = [], []
    for line in [ln for ln in text.splitlines() if ln.strip()]:
        parts = [p.strip() for p in line.split(";")]
        if len(parts) != 4:
            raise GRInputError("Each line needs: mass; x(t); y(t); z(t)")
        masses.append(gr_waves_parse(parts[0], t))
        positions.append([gr_waves_parse(p, t) for p in parts[1:]])
    if not masses:
        raise GRInputError("Give at least one mass.")
    return masses, positions


def gr_waves_parse(text: str, t: sp.Symbol) -> sp.Expr:
    from modules.gr_metrics import parse_in_symbols
    return parse_in_symbols(text, {"t": t})


def _quadrupole_tab() -> None:
    st.write("Lowest-order (Newtonian quadrupole) radiation: h^TT = (2G/c⁴D)[Q̈]^TT, P = (G/5c⁵)⟨Q⃛ᵢⱼQ⃛ᵢⱼ⟩.")
    mode = st.radio("Source", ["Circular binary", "Custom point masses"], horizontal=True, key="grq_mode")
    c1, c2, c3 = st.columns(3)
    with c3:
        incl = _number("Observer inclination ι (rad)", 0.6, "grq_incl", step=0.1)
    if mode == "Circular binary":
        units = st.radio("Units", ["Geometric (G = c = 1)", "SI (solar masses, km, Mpc)"], horizontal=True, key="grq_units")
        si = units.startswith("SI")
        with c1:
            m1 = _number("m₁" + (" (M☉)" if si else ""), 1.4 if si else 1.0, "grq_m1", min_value=1e-6)
            m2 = _number("m₂" + (" (M☉)" if si else ""), 1.4 if si else 1.0, "grq_m2", min_value=1e-6)
        with c2:
            a = _number("Separation a" + (" (km)" if si else ""), 100.0 if si else 20.0, "grq_a", min_value=1e-6)
            D = _number("Distance D" + (" (Mpc)" if si else ""), 40.0 if si else 1000.0, "grq_D", min_value=1e-6)
        G, c = (6.67430e-11, 2.99792458e8) if si else (1.0, 1.0)
        Msun, Mpc = 1.98847e30, 3.0856775814913673e22
        try:
            rad = gr_waves.circular_binary(m1 * Msun if si else m1, m2 * Msun if si else m2, a * 1e3 if si else a, incl,
                                           D * Mpc if si else D, G, c)
        except GRInputError as e:
            st.error(str(e))
            return
        st.dataframe(pd.DataFrame([
            {"Quantity": "orbital frequency", "value": rad.omega / (2 * math.pi), "unit": "Hz" if si else "1/M-units"},
            {"Quantity": "GW frequency (2 × orbital)", "value": rad.omega / math.pi, "unit": "Hz" if si else "1/M-units"},
            {"Quantity": "h₊ amplitude", "value": rad.h_plus_amplitude, "unit": ""},
            {"Quantity": "h× amplitude", "value": rad.h_cross_amplitude, "unit": ""},
            {"Quantity": "radiated power", "value": rad.power, "unit": "W" if si else "G=c=1 units"}]), hide_index=True, width="stretch")
        render_checks(rad.checks, "Symbolic quadrupole vs textbook closed forms")
        t = sp.Symbol("t", real=True)
        ts = np.linspace(0, 4 * 2 * math.pi / rad.omega, 600)
        st.plotly_chart(gr_plots.waveform_figure(ts, gr_waves.sample_series(rad.quad.h_plus, t, ts),
                                                 gr_waves.sample_series(rad.quad.h_cross, t, ts)), width="stretch")
    else:
        st.caption("One line per mass:  mass; x(t); y(t); z(t)   (t is the retarded time; example: a rotating dumbbell)")
        text = st.text_area("Point masses", "1; cos(2*t); sin(2*t); 0\n1; -cos(2*t); -sin(2*t); 0", key="grq_text", height=110)
        c1, c2 = st.columns(2)
        with c1:
            D = _number("Distance D", 100.0, "grq_cD", min_value=1e-6)
        t = sp.Symbol("t", real=True)
        try:
            masses, positions = _parse_source(text, t)
            q = gr_waves.quadrupole_radiation(masses, positions, t, inclination=incl, distance=D)
        except GRInputError as e:
            st.error(str(e))
            return
        st.write("Reduced quadrupole moment Q_ij(t):")
        st.latex(sp.latex(q.Q))
        st.latex(rf"h_+ = {sp.latex(q.h_plus)}")
        st.latex(rf"h_\times = {sp.latex(q.h_cross)}")
        st.latex(rf"P = {sp.latex(q.power)}")
        ts = np.linspace(0, 8.0, 500)
        st.plotly_chart(gr_plots.waveform_figure(ts, gr_waves.sample_series(q.h_plus, t, ts),
                                                 gr_waves.sample_series(q.h_cross, t, ts)), width="stretch")


def _inspiral_tab() -> None:
    st.write("Radiation reaction shrinks a circular orbit (Peters): the numerical integration is compared with its closed form.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        m1 = _number("m₁", 1.0, "gri_m1", min_value=1e-6)
    with c2:
        m2 = _number("m₂", 1.0, "gri_m2", min_value=1e-6)
    with c3:
        a0 = _number("Initial separation a₀", 100.0, "gri_a0", min_value=1e-3)
    with c4:
        end = _number("Stop at a/a₀ =", 0.05, "gri_end", min_value=0.001, max_value=0.99, step=0.01)
    try:
        ins = gr_waves.peters_inspiral(m1, m2, a0, a_end_factor=end)
    except GRInputError as e:
        st.error(str(e))
        return
    st.caption(f"G = c = 1. Time to merger from a₀: {ins.merger_time:.6g}. Chirp mass 𝓜 = {ins.chirp_mass:.6g}.")
    render_checks(ins.checks, "Numerical vs closed-form inspiral")
    st.plotly_chart(gr_plots.inspiral_figure(ins), width="stretch")


# ================================================================================ page

def render_relativity_tab() -> None:
    st.subheader("🌌 General relativity")
    st.caption("Curvature, geodesics, potentials, coordinate changes, causal structure and gravitational waves. Units G = c = 1 "
               "unless stated. Every result is shown with the checks that were run on it: identities that must hold, conserved "
               "quantities, and independent exact results.")
    tabs = st.tabs(["Curvature & fields", "Geodesics", "Potentials", "Coordinate changes", "Cones & Penrose", "Gravitational waves"])
    with tabs[0]:
        _curvature_tab()
    with tabs[1]:
        _geodesic_tab()
    with tabs[2]:
        _potential_tab()
    with tabs[3]:
        _transform_tab()
    with tabs[4]:
        _cones_tab()
    with tabs[5]:
        _waves_tab()
