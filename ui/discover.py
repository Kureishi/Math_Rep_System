"""The Discover mode: find the differential equations a time series obeys (modules/discover.py). No language model."""
import numpy as np
import streamlit as st

from modules import discover as dc
from modules import discover_plots as dp
from ui.theme import badge_row

_STATE = "_discover_run"
_VERDICT_KIND = {"verified": "pass", "caution": "warn", "failed": "fail"}
_VERDICT_ICON = {"verified": "✅", "caution": "🟡", "failed": "❌"}


def _data_source() -> tuple[np.ndarray, np.ndarray, list[str], list[str] | None, float] | None:
    """(t, X, names, truth equations if known, default validation horizon) from the chosen source, or None."""
    source = st.radio("Data", ["Built-in example", "Paste or upload data"], horizontal=True, key="discover_source")
    if source == "Built-in example":
        demo = st.selectbox("System", list(dc.DEMOS), key="discover_demo")
        c1, c2 = st.columns(2)
        noise = c1.slider("Measurement noise (× each state's spread)", 0.0, 0.05, 0.0, 0.005, key="discover_noise",
                          help="Gaussian noise added to the simulated measurements; differentiation amplifies it.")
        seed = int(c2.number_input("Noise seed", 0, 10 ** 6, 0, key="discover_seed"))
        t, X, names, truth = dc.demo_data(demo, noise, seed)
        with st.expander("The true equations (what a correct answer looks like)"):
            for line in truth:
                st.code(line, language=None)
        return t, X, names, truth, 0.25 if demo.startswith("Lorenz") else 1.0
    st.caption("First column time, the rest the measured states; a header row names them (otherwise x, x1, x2, ...).")
    up = st.file_uploader("CSV or text file", type=["csv", "txt"], key="discover_file")
    pasted = st.text_area("…or paste it here", height=120, key="discover_paste", placeholder="t,x,v\n0,1.0,0.0\n0.1,0.98,-0.39\n…")
    text = up.getvalue().decode("utf-8", errors="replace") if up is not None else pasted
    if not text.strip():
        st.info("Give it a time series to look at.")
        return None
    try:
        t, X, names = dc.parse_table(text)
    except dc.DiscoverError as exc:
        st.error(str(exc))
        return None
    return t, X, names, None, 1.0


def render_discover_tab() -> None:
    st.subheader("🧭 Discover equations")
    st.caption("Give it measurements over time and it finds the differential equations they obey: the sparsest combination of "
               "candidate terms that explains the motion, then checks the result by running it forward against your data. "
               "No language model involved. The true law must be expressible with the candidate terms (polynomials, optionally "
               "sin/cos); the simulation check is what shows when it is not.")
    data = _data_source()
    if data is None:
        return
    t, X, names, truth, default_horizon = data
    st.caption(f"{len(t)} points, {len(names)} state(s): {', '.join(names)}.")

    with st.expander("Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        degree = c1.slider("Polynomial degree", 1, 3, 2, key="discover_degree")
        trig = c2.checkbox("Include sin / cos of each state", False, key="discover_trig")
        constant = c3.checkbox("Include a constant term", True, key="discover_const")
        c4, c5 = st.columns(2)
        method = c4.radio("Derivative estimate", ["savgol", "gradient"], key="discover_method", horizontal=True,
                          help="savgol smooths while differentiating (use for noisy data); gradient is plain differences.")
        auto = c5.checkbox("Choose the sparsity threshold automatically (BIC)", True, key="discover_auto")
        threshold = None if auto else float(st.number_input("Threshold (smallest coefficient kept)", 1e-6, 1e3, 0.05, format="%g",
                                                            key="discover_threshold"))
        c6, c7 = st.columns(2)
        boot = c6.slider("Stability resamples (0 = off)", 0, 100, 30, 10, key="discover_boot")
        horizon = c7.slider("Validate over the first … of the record", 0.1, 1.0, default_horizon, 0.05, key=f"discover_horizon_{len(t)}_{default_horizon}",
                            help="Chaotic systems cannot be forecast far by any model, the true one included; use a short horizon.")

    if st.button("🔎 Discover equations", type="primary", key="discover_run"):
        try:
            with st.spinner("Searching for the sparsest model…"):
                fitted = dc.fit(t, X, names, degree, trig, constant, threshold, method, None, boot, seed=0)
                err = dc.simulation_error(fitted, horizon)
                n_sim = max(3, int(round(horizon * len(fitted.t))))
                sim = dc.simulate(fitted, fitted.X[0], fitted.t[:n_sim])
            st.session_state[_STATE] = dict(result=fitted, err=err, sim=sim, truth=truth)
        except dc.DiscoverError as exc:
            st.session_state.pop(_STATE, None)
            st.error(str(exc))
            return

    run = st.session_state.get(_STATE)
    if run is None:
        return
    result: dc.DiscoverResult = run["result"]
    level, why = dc.verdict(result, run["err"])
    law = dc.recognise(result)
    badge_row([(f"{_VERDICT_ICON[level]} {level.capitalize()}", _VERDICT_KIND[level])] +
              ([(f"Looks like: {law}", "pass")] if law else []) +
              [(f"{result.n_terms} terms", "pass"), (f"simulation error {run['err']:.2f}" if np.isfinite(run["err"]) else "simulation diverges",
                                                     _VERDICT_KIND[level])])
    st.caption(why)
    st.markdown("#### Discovered equations")
    for line in result.latex():
        st.latex(line)
    if run["truth"]:
        with st.expander("Compare with the true equations"):
            for line in run["truth"]:
                st.code(line, language=None)
    for note in result.notes:
        st.caption(f"ℹ️ {note}")

    tabs = st.tabs(["Fit against data", "Terms", "Sparsity path", "Stability", "Export"])
    with tabs[0]:
        st.plotly_chart(dp.fit_figure(result, run["sim"]), width="stretch")
        st.caption("Derivative R² per state: " + ", ".join(f"{n} {r:.3f}" for n, r in zip(result.names, result.r2)))
    with tabs[1]:
        rows = result.term_table()
        st.dataframe(rows, hide_index=True, width="stretch") if rows else st.warning("No terms survived; lower the threshold.")
    with tabs[2]:
        st.plotly_chart(dp.path_figure(result), width="stretch")
        st.caption("Raising the threshold removes terms; the automatic choice is the point with the lowest BIC, which "
                   "balances fit against the number of terms.")
    with tabs[3]:
        if result.inclusion is None:
            st.info("Set 'Stability resamples' above 0 to see how often each term is selected when the data is resampled.")
        else:
            st.plotly_chart(dp.inclusion_figure(result), width="stretch")
    with tabs[4]:
        st.markdown("**Python (for `scipy.integrate.solve_ivp`)**")
        code = dc.to_python(result)
        st.code(code, language="python")
        st.download_button("⬇️ Download as .py", data=code, file_name="discovered_system.py", key="discover_dl_py")
        st.markdown("**LaTeX**")
        st.code("\n".join(result.latex()), language="latex")
