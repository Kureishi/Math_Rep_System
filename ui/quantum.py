"""
⚛️ Quantum mechanics: five tools, each of which checks itself.

    Eigenstates     the time-independent Schrödinger equation in 1D, with a convergence study and
                    comparison against the analytic spectrum where one exists
    Wavepacket      time evolution by two independent methods, tunnelling compared with the exact result
    Operators       commutators, identities, eigen-decompositions, uncertainty relations, BCH
    Qubit           gates, noise, the Bloch sphere, purity and fidelity
    Perturbation    Rayleigh–Schrödinger corrections in exact arithmetic against exact diagonalisation

All the physics lives in modules/quantum_*.py (Streamlit-free, with their own tests); this file only
collects inputs and shows results. Every result is shown with the checks that were run on it and what they
found, which is the point of the page: a plot of a wavefunction proves nothing, a wavefunction whose
energy matches the analytic spectrum, whose nodes follow the oscillation theorem and whose grid
convergence was measured is a result.
"""
import numpy as np
import pandas as pd
import sympy as sp
import streamlit as st

from modules.quantum_1d import POTENTIALS, analytic_table, solve_eigenstates, tunnelling_splitting
from modules.quantum_common import CheckList, ExpressionError, check_expression_syntax
from modules.quantum_dynamics import DYNAMIC_POTENTIALS, METHODS, evolve_wavepacket
from modules.quantum_operators import (
    FAMILIES, MAX_DIM, STANDARD_IDENTITIES, analyze_operator, build_family, check_identity, check_standard_identities,
    evaluate, parse_matrix, parse_state, uncertainty_relations, verify_bch,
)
from modules.quantum_perturbation import matrix_perturbation, oscillator_perturbation
from modules.quantum_plots import (
    bch_error_figure, bloch_sphere_figure, convergence_figure, density_matrix_figure, dynamics_diagnostics_figure,
    eigenstates_figure, perturbation_energy_figure, perturbation_error_figure, qubit_timeseries_figure,
    region_probability_figure, relaxation_figure, transmission_figure, wavepacket_animation_figure,
)
from modules.quantum_qubits import (
    GATE_HELP, STATE_PRESETS, Noise, bloch_from_angles, bloch_to_density, fit_exponential_time, parse_sequence, ramsey_curve,
    simulate_sequence, t1_curve,
)
from ui.common import persist_on_click


# ------------------------------------------------------------------------------------------ shared display

def render_checks(checks: CheckList, title: str = "Checks run on this result") -> None:
    """The verdict first, then every check with the numbers behind it."""
    if not checks.checks:
        return
    if checks.passed:
        st.success(f"✅ {checks.summary()}")
    else:
        st.error(f"⚠️ {checks.summary()} -- read the failures below before using this result")
    with st.expander(title, expanded=not checks.passed):
        st.dataframe(pd.DataFrame([{"": "✅" if c.passed else "❌", "Check": c.label, "What was found": c.detail}
                                   for c in checks.checks]), hide_index=True, width="stretch")


def _number(label: str, value: float, key: str, **kw) -> float:
    return float(st.number_input(label, value=float(value), key=key, format="%.6g", **kw))


def _latex_matrix(m: sp.Matrix) -> None:
    st.latex(sp.latex(m))


def render_quantum_tab() -> None:
    st.subheader("⚛️ Quantum mechanics")
    st.caption("Each tool verifies its own result against an exact or independent calculation and shows what it "
               "checked. Natural units (ħ = m = 1) unless you change them.")
    tabs = st.tabs(["Eigenstates", "Wavepacket", "Operators", "Qubit", "Perturbation"])
    with tabs[0]:
        _eigenstates_tab()
    with tabs[1]:
        _wavepacket_tab()
    with tabs[2]:
        _operators_tab()
    with tabs[3]:
        _qubit_tab()
    with tabs[4]:
        _perturbation_tab()


# ------------------------------------------------------------------------------------------ 1. eigenstates

def _eigenstates_tab() -> None:
    labels = {s.label: k for k, s in POTENTIALS.items()}
    label = st.selectbox("Potential", list(labels), key="q1_potential")
    key = labels[label]
    spec = POTENTIALS[key]
    st.caption(spec.description)

    params: dict[str, float] = {}
    custom_formula = None
    cols = st.columns(max(1, len(spec.params)) if spec.params else 1)
    for col, (name, default) in zip(cols, spec.params.items()):
        with col:
            params[name] = _number(name, default, f"q1_param_{key}_{name}")
    if key == "custom":
        custom_formula = st.text_input("V(x) =", value=spec.formula, key="q1_custom",
                                       help="An expression in x. Names: hbar, m, pi, e; functions: sin, cos, exp, sqrt, "
                                            "sech, abs, Heaviside, ...")
        names = ["x", "hbar", "m"]
        try:
            check_expression_syntax(custom_formula, names)
        except ExpressionError as e:
            st.error(str(e))
            return

    c1, c2, c3 = st.columns(3)
    with c1:
        x_lo = _number("Left end of the interval", spec.domain[0], f"q1_lo_{key}")
        x_hi = _number("Right end of the interval", spec.domain[1], f"q1_hi_{key}")
    with c2:
        n_points = int(st.slider("Grid points", 100, 3000, 1000 if key != "hydrogen" else 2000, step=50, key=f"q1_n_{key}"))
        n_levels = int(st.slider("Levels", 1, 12, spec.n_levels, key=f"q1_levels_{key}"))
    with c3:
        hbar = _number("ħ", 1.0, "q1_hbar", min_value=1e-6, disabled=spec.atomic_units)
        mass = _number("Mass m", 1.0, "q1_mass", min_value=1e-6, disabled=spec.atomic_units)
        tol = _number("Tolerance vs the analytic spectrum", 1e-3, "q1_tol", min_value=1e-9)

    result = persist_on_click(
        "Solve", "q1_solve", "q1_result", True,
        lambda progress: solve_eigenstates(key, params, (x_lo, x_hi), n_points, n_levels, hbar, mass, custom_formula,
                                           tolerance=tol, progress=progress),
        run_label="Solving the Schrödinger equation", stoppable=False)
    if result is None:
        st.info("Choose a potential and press Solve.")
        return

    render_checks(result.checks)
    for note in result.notes:
        st.caption("ℹ️ " + note)

    st.dataframe(pd.DataFrame(analytic_table(result)), hide_index=True, width="stretch")
    split = tunnelling_splitting(result) if key == "double_well" else None
    if split is not None:
        st.metric("Tunnelling splitting E₁ − E₀", f"{split:.6g}")

    show = st.radio("Draw", ["wavefunction", "density"], horizontal=True, key="q1_show", format_func=lambda s: {"wavefunction": "ψ", "density": "|ψ|²"}[s])
    chosen = st.multiselect("Levels to draw", list(range(result.n_levels)), default=list(range(min(result.n_levels, 6))), key="q1_draw")
    st.plotly_chart(eigenstates_figure(result, show, chosen), width="stretch", key="q1_fig")
    if result.analytic is not None:
        st.plotly_chart(convergence_figure(result), width="stretch", key="q1_conv")

    with st.expander("Per-state observables"):
        st.dataframe(pd.DataFrame([{"n": i, "E": o.energy, "⟨x⟩": o.mean_x, "Δx": o.std_x, "Δp": o.std_p, "Δx·Δp / (ħ/2)": o.std_x * o.std_p / (result.hbar / 2),
                                    "⟨V⟩": o.mean_v, "⟨T⟩": o.mean_t, "nodes": o.nodes, "parity": o.parity or "—"}
                                   for i, o in enumerate(result.observables)]), hide_index=True, width="stretch")
    table = pd.DataFrame({"x": result.x, "V": result.potential, **{f"psi_{i}": result.states[:, i] for i in range(result.n_levels)}})
    st.download_button("⬇️ Download the states (CSV)", table.to_csv(index=False), "eigenstates.csv", "text/csv", key="q1_csv")


# ------------------------------------------------------------------------------------------ 2. wavepacket

def _wavepacket_tab() -> None:
    labels = {s.label: k for k, s in DYNAMIC_POTENTIALS.items()}
    label = st.selectbox("Potential", list(labels), index=list(labels.values()).index("barrier"), key="q2_potential")
    key = labels[label]
    spec = DYNAMIC_POTENTIALS[key]
    st.caption(spec.description)

    params: dict[str, float] = {}
    custom_formula = None
    if spec.params:
        cols = st.columns(len(spec.params))
        for col, (name, default) in zip(cols, spec.params.items()):
            with col:
                params[name] = _number(name, default, f"q2_param_{key}_{name}")
    if key == "custom":
        custom_formula = st.text_input("V(x) =", value="0.5*x**2", key="q2_custom")
        try:
            check_expression_syntax(custom_formula, ["x", "hbar", "m"])
        except ExpressionError as e:
            st.error(str(e))
            return

    st.markdown("**The packet** (a Gaussian with momentum ħk₀)")
    c1, c2, c3 = st.columns(3)
    with c1:
        x0 = _number("Start position x₀", -20.0, "q2_x0")
        sigma = _number("Width σ", 2.5, "q2_sigma", min_value=0.05)
    with c2:
        k0 = _number("Wavenumber k₀", 2.0 if key != "harmonic" else 0.0, "q2_k0")
        hbar = _number("ħ", 1.0, "q2_hbar", min_value=1e-6)
    with c3:
        mass = _number("Mass m", 1.0, "q2_mass", min_value=1e-6)
        st.caption(f"Mean energy ≈ ħ²k₀²/2m = {hbar ** 2 * k0 ** 2 / (2 * mass):.4g}")

    st.markdown("**The box and the method**")
    d1, d2, d3 = st.columns(3)
    with d1:
        lo = _number("Left end", -50.0, "q2_lo")
        hi = _number("Right end", 50.0, "q2_hi")
    with d2:
        n_points = int(st.select_slider("Grid points", [256, 512, 1024, 2048, 4096], value=2048, key="q2_n"))
        method = st.radio("Method", list(METHODS), format_func=lambda m: {"split-step": "Split-step Fourier", "crank-nicolson": "Crank–Nicolson"}[m],
                          key="q2_method", horizontal=True)
    with d3:
        n_frames = int(st.slider("Animation frames", 10, 120, 60, key="q2_frames"))
        cross = st.checkbox("Verify with the other method", value=True, key="q2_cross",
                            help="Runs the second method too and compares the final densities.")
    auto_time = st.checkbox("Choose the duration automatically (until the packet has cleared the potential)", value=True, key="q2_auto")
    total_time = None
    if not auto_time:
        total_time = _number("Duration", 15.0, "q2_time", min_value=0.01)
    tol = _number("Tolerance for T and R against the exact result", 0.02, "q2_tol", min_value=1e-4)

    def run(progress):
        return evolve_wavepacket(key, params, x0, sigma, k0, (lo, hi), n_points, total_time, None, n_frames, method, hbar, mass,
                                 custom_formula, cross, tol, progress)

    result = persist_on_click("Run", "q2_run", "q2_result", True, run, run_label="Evolving the wavepacket", stoppable=True)
    if result is None:
        st.info("Set up the packet and press Run. Stop works between chunks of time steps.")
        return

    render_checks(result.checks)
    for note in result.notes:
        st.caption("ℹ️ " + note)
    st.caption(f"{result.method}; {result.steps:,} steps of Δt = {result.dt:.3g}; total time {result.spec['total_time']:.4g}.")
    if result.predicted is not None and "right" in result.final_probabilities:
        m1, m2, m3 = st.columns(3)
        m1.metric("Transmitted (simulation)", f"{result.final_probabilities['right']:.4f}",
                  f"{result.final_probabilities['right'] - result.predicted['transmitted']:+.4f} vs exact", delta_color="off")
        m2.metric("Transmitted (exact scattering)", f"{result.predicted['transmitted']:.4f}")
        m3.metric("Reflected (simulation)", f"{result.final_probabilities['left']:.4f}")
    st.plotly_chart(wavepacket_animation_figure(result), width="stretch", key="q2_anim")
    if result.predicted is not None:
        st.plotly_chart(transmission_figure(result, result.hbar, result.mass), width="stretch", key="q2_T")
        st.plotly_chart(region_probability_figure(result), width="stretch", key="q2_region")
    with st.expander("Conservation laws and Ehrenfest's theorem"):
        st.plotly_chart(dynamics_diagnostics_figure(result), width="stretch", key="q2_diag")


# ------------------------------------------------------------------------------------------ 3. operators

def _operators_tab() -> None:
    labels = {**{f.label: k for k, f in FAMILIES.items()}, "Custom matrices A and B": "custom"}
    label = st.selectbox("Operators", list(labels), key="q3_family")
    family = labels[label]
    size = 2
    if family == "spin":
        size = int(st.slider("2j", 1, MAX_DIM - 1, 2, key="q3_2j", help="j = 1/2 → 1, j = 1 → 2, j = 3/2 → 3, ..."))
    elif family == "oscillator":
        size = int(st.slider("Dimension (truncation)", 3, MAX_DIM, 6, key="q3_dim"))

    try:
        if family == "custom":
            env = _custom_operators()
            truncated = False
        else:
            env = build_family(family, size)
            truncated = FAMILIES[family].truncated
    except (ExpressionError, ValueError) as e:
        st.error(str(e))
        return
    if not env:
        return
    dim = next(iter(env.values())).shape[0]
    st.caption(f"Operators ({dim} × {dim}, ħ = 1): " + ", ".join(f"`{n}`" for n in env) + ". Also `Id`, `I`, `pi`.")
    with st.expander("Show the matrices"):
        for name, m in env.items():
            st.markdown(f"`{name}`")
            _latex_matrix(m)
    if truncated:
        st.info("The oscillator's operators are infinite matrices cut off at this dimension. The cut-off spoils "
                "[a, a†] = 1 in the last diagonal entry; such failures are flagged as truncation artefacts.")

    ex, an, un, bch = st.tabs(["Expressions and identities", "Analyse an operator", "Uncertainty relation", "BCH expansion"])
    with ex:
        _expression_section(env, family, truncated)
    with an:
        _analysis_section(env)
    with un:
        _uncertainty_section(env, dim)
    with bch:
        _bch_section(env)


def _custom_operators() -> dict[str, sp.Matrix]:
    n = int(st.slider("Dimension", 2, 4, 2, key="q3_custom_n"))
    env = {}
    defaults = {"A": [["0", "1"], ["1", "0"]], "B": [["1", "0"], ["0", "-1"]]}
    for name in ("A", "B"):
        base = defaults[name]
        grid = [[base[i][j] if i < 2 and j < 2 else ("1" if i == j and name == "A" else "0") for j in range(n)] for i in range(n)]
        st.markdown(f"**{name}** (entries may use I, pi, sqrt(2), ...)")
        frame = st.data_editor(pd.DataFrame(grid, columns=[str(j) for j in range(n)]), key=f"q3_custom_{name}_{n}", hide_index=True, num_rows="fixed")
        env[name] = parse_matrix(frame.astype(str).values.tolist())
    return env


def _expression_section(env, family: str, truncated: bool) -> None:
    default_l, default_r = ("comm(A, B)", "A*B - B*A") if family == "custom" else \
        {"pauli": ("comm(sx, sy)", "2*I*sz"), "spin": ("comm(Jx, Jy)", "I*Jz"), "oscillator": ("comm(a, adag)", "Id")}[family]
    c1, c2 = st.columns(2)
    with c1:
        lhs = st.text_input("Left side", default_l, key=f"q3_lhs_{family}",
                            help="comm(A,B), acomm(A,B), dag(A), exp(A), tr(A), det(A), inv(A), +, -, *, **, numbers, I, pi, Id")
    with c2:
        rhs = st.text_input("Right side", default_r, key=f"q3_rhs_{family}")
    try:
        result = check_identity(lhs, rhs, env, truncated=truncated)
    except ExpressionError as e:
        st.error(str(e))
        return
    if result.holds:
        st.success("✅ The two sides are equal.")
    elif result.artefact_of_truncation:
        st.warning("⚠️ " + result.note)
    else:
        st.error("❌ The two sides differ. Their difference (left − right) is:")
    if not result.holds:
        _latex_matrix(result.difference) if isinstance(result.difference, sp.MatrixBase) else st.latex(sp.latex(result.difference))
    with st.expander("Value of the left side"):
        try:
            value = evaluate(lhs, env)
            _latex_matrix(value) if isinstance(value, sp.MatrixBase) else st.latex(sp.latex(value))
        except ExpressionError as e:
            st.error(str(e))
    if family in STANDARD_IDENTITIES and st.button("Check all the standard identities for these operators", key=f"q3_std_{family}"):
        rows = check_standard_identities(family, env)
        st.dataframe(pd.DataFrame([{"": "✅" if r.holds else ("⚠️" if r.artefact_of_truncation else "❌"), "Identity": label,
                                    "Note": r.note} for label, r in rows]), hide_index=True, width="stretch")


def _analysis_section(env) -> None:
    name = st.selectbox("Operator", list(env), key="q3_an_op")
    report = analyze_operator(env[name])
    flags = [("Hermitian", report.hermitian), ("Unitary", report.unitary), ("Normal", report.normal)]
    st.markdown("  ".join(f"{'✅' if v else '▫️'} {k}" for k, v in flags) + f"   ·   tr = {sp.N(report.trace, 6)}   ·   det = {sp.N(report.determinant, 6)}")
    st.dataframe(pd.DataFrame([{"eigenvalue": str(v), "≈": float(sp.N(sp.re(v))), "multiplicity": m} for v, m in report.eigenvalues]),
                 hide_index=True, width="stretch")
    if report.eigenvectors:
        with st.expander("Eigenvectors"):
            for val, vecs in report.eigenvectors:
                for vec in vecs:
                    st.latex(rf"\lambda = {sp.latex(val)}:\quad {sp.latex(vec)}")
    for note in report.notes:
        st.caption("ℹ️ " + note)
    render_checks(report.checks)


def _uncertainty_section(env, dim: int) -> None:
    names = list(env)
    c1, c2 = st.columns(2)
    with c1:
        a_name = st.selectbox("Observable A", names, index=0, key="q3_un_a")
    with c2:
        b_name = st.selectbox("Observable B", names, index=min(1, len(names) - 1), key="q3_un_b")
    preset = st.radio("State", ["first basis state", "equal superposition", "custom"], horizontal=True, key="q3_un_preset")
    if preset == "first basis state":
        entries = ["1"] + ["0"] * (dim - 1)
    elif preset == "equal superposition":
        entries = ["1"] * dim
    else:
        cols = st.columns(dim)
        entries = [col.text_input(f"c{i}", "1" if i == 0 else "0", key=f"q3_un_c{i}") for i, col in enumerate(cols)]
    try:
        state = parse_state(entries, dim)
        result = uncertainty_relations(env[a_name], env[b_name], state)
    except (ExpressionError, ValueError) as e:
        st.error(str(e))
        return
    m = st.columns(4)
    m[0].metric("ΔA", f"{result.std_a:.5g}")
    m[1].metric("ΔB", f"{result.std_b:.5g}")
    m[2].metric("ΔA ΔB", f"{result.product:.5g}")
    m[3].metric("Heisenberg bound ½|⟨[A,B]⟩|", f"{result.heisenberg_bound:.5g}")
    st.caption(f"Schrödinger–Robertson bound (adds the covariance {result.covariance:.4g}): {result.schrodinger_bound:.5g}")
    render_checks(result.checks, "Inequalities tested")


def _bch_section(env) -> None:
    names = list(env)
    c1, c2, c3 = st.columns(3)
    with c1:
        x_text = st.text_input("X", names[0] if names else "A", key="q3_bch_x", help="Any operator expression, e.g. 0.5*sx")
    with c2:
        y_text = st.text_input("Y", names[min(1, len(names) - 1)], key="q3_bch_y")
    with c3:
        order = int(st.slider("Order", 1, 5, 4, key="q3_bch_order"))
    try:
        x, y = evaluate(x_text, env), evaluate(y_text, env)
        if not (isinstance(x, sp.MatrixBase) and isinstance(y, sp.MatrixBase)):
            raise ExpressionError("X and Y must be operators.")
        result = verify_bch(x, y, order)
    except (ExpressionError, ValueError) as e:
        st.error(str(e))
        return
    st.markdown("**log(e^X e^Y) =**")
    for k, term in result.latex_terms:
        st.latex(("" if k == 1 else "+\\ ") + term)
    render_checks(result.checks, "Numerical verification of the expansion")
    st.plotly_chart(bch_error_figure(result), width="stretch", key="q3_bch_fig")


# ------------------------------------------------------------------------------------------ 4. qubit

def _qubit_tab() -> None:
    c1, c2 = st.columns(2)
    with c1:
        preset = st.selectbox("Initial state", list(STATE_PRESETS) + ["custom (θ, φ, length)"], key="q4_state")
        if preset == "custom (θ, φ, length)":
            theta = st.slider("θ (polar angle from |0⟩)", 0.0, float(np.pi), float(np.pi / 3), key="q4_theta")
            phi = st.slider("φ (azimuth)", 0.0, float(2 * np.pi), 0.0, key="q4_phi")
            radius = st.slider("Bloch vector length (1 = pure)", 0.0, 1.0, 1.0, key="q4_radius")
            r0 = bloch_from_angles(theta, phi, radius)
        else:
            r0 = np.array(STATE_PRESETS[preset], dtype=float)
    with c2:
        sequence = st.text_input("Gate sequence", "H, Rz(pi/4), X, T", key="q4_seq", help=GATE_HELP)
        gate_time = _number("Time per gate", 1.0, "q4_gt", min_value=1e-6)
        idle_time = _number("Idle time after each gate", 0.0, "q4_idle", min_value=0.0)

    st.markdown("**Noise** (each optional)")
    n1, n2, n3 = st.columns(3)
    with n1:
        use_t1 = st.checkbox("Amplitude damping (T1)", value=False, key="q4_use_t1")
        t1 = _number("T1", 20.0, "q4_t1", min_value=1e-6, disabled=not use_t1)
    with n2:
        use_tp = st.checkbox("Pure dephasing (Tφ)", value=False, key="q4_use_tp")
        t_phi = _number("Tφ", 15.0, "q4_tphi", min_value=1e-6, disabled=not use_tp)
    with n3:
        use_dep = st.checkbox("Depolarising", value=False, key="q4_use_dep")
        dep = _number("Rate γd", 0.02, "q4_dep", min_value=0.0, disabled=not use_dep)
    noise = Noise(t1 if use_t1 else None, t_phi if use_tp else None, dep if use_dep else 0.0)
    if noise.t2 is not None:
        st.caption(f"Resulting T2 = {noise.t2:.4g}" + (f" (T2/T1 = {noise.t2 * noise.gamma1:.3g}, never above 2)" if use_t1 else ""))

    try:
        gates = parse_sequence(sequence)
    except ExpressionError as e:
        st.error(str(e))
        return
    result = persist_on_click("Run", "q4_run", "q4_result", True,
                              lambda: simulate_sequence(gates, r0, noise, gate_time, idle_time))
    if result is None:
        st.info("Choose a state, a gate sequence and (optionally) noise, then press Run.")
        return

    render_checks(result.checks)
    m = st.columns(4)
    m[0].metric("Final purity Tr ρ²", f"{result.purity[-1]:.4f}")
    m[1].metric("Fidelity with the ideal state", f"{result.fidelity[-1]:.4f}")
    m[2].metric("|r| at the end", f"{np.linalg.norm(result.bloch[-1]):.4f}")
    m[3].metric("Total time", f"{result.times[-1]:.4g}")
    st.plotly_chart(bloch_sphere_figure(result), width="stretch", key="q4_bloch")
    st.plotly_chart(qubit_timeseries_figure(result), width="stretch", key="q4_ts")
    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(density_matrix_figure(bloch_to_density(result.initial), "Initial ρ"), width="stretch", key="q4_rho0")
    with d2:
        st.plotly_chart(density_matrix_figure(result.final_density, "Final ρ"), width="stretch", key="q4_rho1")
    with st.expander("Each gate as a rotation of the sphere"):
        st.dataframe(pd.DataFrame([{"gate": g.label, "axis (x, y, z)": ", ".join(f"{v:.3f}" for v in g.axis),
                                    "angle (rad)": g.angle, "angle (°)": np.degrees(g.angle)} for g in result.gates]),
                     hide_index=True, width="stretch")
    if not noise.noiseless:
        with st.expander("Measuring T1 and T2 with these noise settings"):
            horizon = 4.0 * max(noise.t1 or 0.0, noise.t2 or 0.0, 1.0)
            times = np.linspace(0, horizon, 80)
            t1v, rv = t1_curve(noise, times), ramsey_curve(noise, times)
            z_eq = noise.gamma1 / noise.rate_z if noise.rate_z else 0.0
            fit1 = fit_exponential_time(times, t1v, floor=z_eq)
            fit2 = fit_exponential_time(times, rv)
            st.plotly_chart(relaxation_figure(times, t1v, rv, noise.t1, noise.t2, fit1, fit2), width="stretch", key="q4_relax")


# ------------------------------------------------------------------------------------------ 5. perturbation

def _perturbation_tab() -> None:
    mode = st.radio("System", ["Anharmonic oscillator", "Your own matrix"], horizontal=True, key="q5_mode")
    if mode == "Anharmonic oscillator":
        _oscillator_perturbation_ui()
    else:
        _matrix_perturbation_ui()


def _show_levels(levels, comparison, checks, notes=()) -> None:
    render_checks(checks, "Checks (derivation and comparison)")
    for note in notes:
        st.caption("ℹ️ " + note)
    rows = []
    for lv in levels:
        if lv.e1 is None:
            rows.append({"n": lv.index, "E⁽⁰⁾": str(lv.e0), "E⁽¹⁾": "degenerate: " + ", ".join(str(b) for b in lv.first_order_branches),
                         "E⁽²⁾": "—", "E⁽³⁾": "—"})
        else:
            rows.append({"n": lv.index, "E⁽⁰⁾": str(lv.e0), "E⁽¹⁾": str(lv.e1), "E⁽²⁾": str(lv.e2), "E⁽³⁾": str(lv.e3)})
    st.markdown("**Exact corrections** (E = E⁽⁰⁾ + λE⁽¹⁾ + λ²E⁽²⁾ + λ³E⁽³⁾ + …)")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("First-order state corrections"):
        for lv in levels:
            if lv.state_correction is not None:
                st.latex(rf"|{lv.index}^{{(1)}}\rangle = {sp.latex(lv.state_correction.T)}\,\text{{ (in the unperturbed basis)}}")
    if comparison is not None:
        st.caption(f"Comparison with {comparison.exact_method}.")
        st.plotly_chart(perturbation_energy_figure(comparison), width="stretch", key="q5_e")
        st.plotly_chart(perturbation_error_figure(comparison), width="stretch", key="q5_err")


def _oscillator_perturbation_ui() -> None:
    st.caption("H = ½p² + ½x² + λ·V(x), with V = c₁x + c₂x² + c₃x³ + c₄x⁴ (ħ = m = ω = 1).")
    cols = st.columns(4)
    coeffs = {}
    for power, col in zip((1, 2, 3, 4), cols):
        with col:
            value = _number(f"c{power} (x^{power})", 1.0 if power == 4 else 0.0, f"q5_c{power}")
            if value != 0.0:
                coeffs[power] = value
    c1, c2, c3 = st.columns(3)
    with c1:
        n_levels = int(st.slider("Levels", 1, 8, 4, key="q5_levels"))
    with c2:
        level = int(st.number_input("Compare level", 0, max(n_levels - 1, 0), 0, key="q5_level"))
    with c3:
        auto = st.checkbox("Choose λ range automatically", True, key="q5_auto")
        lam_max = None if auto else _number("Largest λ", 0.05, "q5_lmax", min_value=1e-6)
    if not coeffs:
        st.info("Give at least one non-zero coefficient.")
        return
    result = persist_on_click(
        "Compute", "q5_run", "q5_result", True,
        lambda progress: oscillator_perturbation(coeffs, n_levels, min(level, n_levels - 1), lam_max, progress=progress),
        run_label="Perturbation theory", stoppable=False)
    if result is None:
        st.info("Set the perturbation and press Compute.")
        return
    _show_levels(result.levels, result.comparison, result.checks, result.notes)


def _matrix_perturbation_ui() -> None:
    n = int(st.slider("Number of levels", 2, 6, 3, key="q5_n"))
    energies_text = st.text_input("Unperturbed energies (comma-separated)", ", ".join(str(i) for i in range(n)), key=f"q5_e0_{n}",
                                  help="H0 is diagonal with these energies. Repeat a value to make a degenerate level.")
    st.markdown("**V** (Hermitian; entries may use I, pi, sqrt(2), ...)")
    base = [["1" if i == j else ("1" if abs(i - j) == 1 else "0") for j in range(n)] for i in range(n)]
    frame = st.data_editor(pd.DataFrame(base, columns=[str(j) for j in range(n)]), key=f"q5_v_{n}", hide_index=True, num_rows="fixed")
    c1, c2 = st.columns(2)
    with c1:
        level = int(st.number_input("Compare level", 0, n - 1, 0, key="q5_mlevel"))
    with c2:
        lam_max = _number("Largest λ (0 = automatic)", 0.0, "q5_mlmax", min_value=0.0)
    try:
        e0 = [sp.nsimplify(c.strip()) if c.strip() else sp.Integer(0) for c in energies_text.split(",")]
        if len(e0) != n:
            raise ValueError(f"Give exactly {n} energies.")
        v = parse_matrix(frame.astype(str).values.tolist())
    except (ExpressionError, ValueError, sp.SympifyError) as e:
        st.error(str(e))
        return
    result = persist_on_click("Compute", "q5_mrun", "q5_mresult", True,
                              lambda: matrix_perturbation(e0, v, level, lam_max or None))
    if result is None:
        st.info("Define H0 and V, then press Compute.")
        return
    levels, comparison, checks = result
    _show_levels(levels, comparison, checks)
