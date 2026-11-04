"""The Posterior sampling (MCMC) mode: the full uncertainty of a nonlinear fit (modules/mcmc.py). No language model."""
import io

import numpy as np
import streamlit as st

from modules import curve_fitting, mcmc
from modules import mcmc_plots as mp
from ui.theme import badge_row

_RUN = "_posterior_run"
_KINDS = {"flat (no bounds)": "flat", "uniform between": "uniform", "normal": "normal", "positive": "positive"}


def _floats(text: str) -> list[float]:
    return [float(t) for t in text.replace(";", ",").split(",") if t.strip()]


def _data_and_model() -> dict | None:
    """The data, model, initial guess and (for a demo) the true values, from the chosen source; None until usable."""
    source = st.radio("Data", ["Built-in example", "Paste or upload data"], horizontal=True, key="post_source")
    if source == "Built-in example":
        demo = st.selectbox("Example", list(mcmc.DEMOS), key="post_demo")
        seed = int(st.number_input("Noise seed", 0, 10 ** 6, 2, key="post_demo_seed"))
        d = mcmc.demo_dataset(demo, seed)
        d["key"] = demo
        return d
    st.caption("Two columns, x then y (a header row is optional).")
    up = st.file_uploader("CSV file", type=["csv", "txt"], key="post_file")
    pasted = st.text_area("…or paste it here", height=110, key="post_paste", placeholder="x,y\n0,2.5\n1,1.8\n…")
    text = up.getvalue().decode("utf-8", errors="replace") if up is not None else pasted
    if not text.strip():
        st.info("Give it some data to fit.")
        return None
    try:
        xs, ys, _, _ = curve_fitting.parse_xy_csv(text)
    except ValueError as exc:
        st.error(str(exc))
        return None
    expr = st.text_input("Model, a function of x with named parameters", "a*exp(-b*x) + c", key="post_expr")
    names = [n.strip() for n in st.text_input("Parameter names (comma separated)", "a, b, c", key="post_names").split(",") if n.strip()]
    guess = st.text_input("Initial guess (one number per parameter; a rough one is fine)", ", ".join("1" for _ in names), key="post_guess")
    try:
        p0 = _floats(guess)
    except ValueError:
        st.error("The initial guess must be numbers separated by commas.")
        return None
    return dict(xs=np.asarray(xs), ys=np.asarray(ys), expr=expr, params=names, p0=p0, truth=None, priors={}, key="custom")


def _priors_ui(d: dict) -> dict[str, mcmc.Prior]:
    out: dict[str, mcmc.Prior] = {}
    with st.expander("Priors (what is assumed before seeing the data)", expanded=bool(d["priors"])):
        st.caption("Flat is fine when the data pin a parameter down. If they do not, the run says so; bounding the parameter (uniform) or "
                   "giving it a normal prior then makes the problem well posed.")
        for name in d["params"]:
            pre = d["priors"].get(name)
            default_kind = {v: k for k, v in _KINDS.items()}[pre.kind] if pre else "flat (no bounds)"
            c1, c2, c3 = st.columns([2, 1, 1])
            kind = _KINDS[c1.selectbox(f"{name}", list(_KINDS), index=list(_KINDS).index(default_kind), key=f"post_prior_{d['key']}_{name}")]
            a = b = 0.0
            if kind == "uniform":
                a = c2.number_input("lower", value=float(pre.a) if pre and pre.kind == "uniform" else 0.0, key=f"post_pa_{d['key']}_{name}", format="%g")
                b = c3.number_input("upper", value=float(pre.b) if pre and pre.kind == "uniform" else 10.0, key=f"post_pb_{d['key']}_{name}", format="%g")
            elif kind == "normal":
                a = c2.number_input("mean", value=0.0, key=f"post_pa_{d['key']}_{name}", format="%g")
                b = c3.number_input("sd", value=1.0, min_value=1e-12, key=f"post_pb_{d['key']}_{name}", format="%g")
            out[name] = mcmc.Prior(kind, float(a), float(b))
    return out


def _samples_csv(r: mcmc.PosteriorResult) -> str:
    buf = io.StringIO()
    np.savetxt(buf, r.samples, delimiter=",", header=",".join(r.names), comments="")
    return buf.getvalue()


def _summary_markdown(r: mcmc.PosteriorResult, expr: str) -> str:
    lines = [f"# Posterior summary\n\nModel: `y = {expr}`  ·  {r.samples.shape[0]:,} samples  ·  seed {r.seed}  ·  "
             f"{'converged' if r.converged else 'NOT converged'}\n", "| parameter | median | −1σ | +1σ | 95% interval |", "|---|---|---|---|---|"]
    for s in r.summaries:
        lines.append(f"| {s.name} | {s.median:.5g} | {s.lo68 - s.median:.3g} | +{s.hi68 - s.median:.3g} | [{s.lo95:.4g}, {s.hi95:.4g}] |")
    lines += ["", "Warnings:"] + ([f"- {w}" for w in r.warnings] or ["- none"])
    return "\n".join(lines) + "\n"


def render_posterior_tab() -> None:
    st.subheader("🎲 Posterior sampling (MCMC)")
    st.caption("Least squares gives one best curve and a symmetric ±. This samples every parameter combination the data allow, so you "
               "see the real, possibly lopsided or correlated, uncertainty, and checks that the sampling has actually converged "
               "and that the model describes the data. No language model involved.")
    d = _data_and_model()
    if d is None:
        return
    if d["truth"]:
        with st.expander("The true parameter values (what a correct answer looks like)"):
            st.write(", ".join(f"{n} = {v:g}" for n, v in zip(d["params"], d["truth"])))
    priors = _priors_ui(d)
    c1, c2, c3, c4 = st.columns(4)
    noise_mode = c1.radio("Noise level σ", ["Estimate it", "I know it"], key=f"post_noise_{d['key']}")
    sigma = float(c2.number_input("σ", 1e-12, 1e12, float(d.get("noise") or 1.0), format="%g", key=f"post_sigma_{d['key']}")) \
        if noise_mode == "I know it" else None
    steps = int(c3.number_input("Steps", 1000, mcmc.MAX_STEPS, 8000, 1000, key="post_steps"))
    seed = int(c4.number_input("Seed", 0, 10 ** 6, 0, key="post_seed"))

    if st.button("🎲 Sample the posterior", type="primary", key="post_run"):
        bar = st.progress(0.0, text="Sampling…")

        def show_progress(fraction: float) -> None:
            bar.progress(min(fraction, 1.0), text="Sampling…")

        try:
            result = mcmc.sample_posterior(d["xs"], d["ys"], d["expr"], d["params"], d["p0"], priors, sigma, steps, seed=seed,
                                           progress=show_progress)
            st.session_state[_RUN] = dict(result=result, truth=d["truth"], expr=d["expr"])
        except mcmc.McmcError as exc:
            st.session_state.pop(_RUN, None)
            st.error(str(exc))
            return
        finally:
            bar.empty()

    run = st.session_state.get(_RUN)
    if run is None:
        return
    r: mcmc.PosteriorResult = run["result"]
    level = "fail" if r.unbounded else "pass" if r.converged and not r.warnings else "warn"
    badge_row([("❌ Unbounded" if r.unbounded else "✅ Converged" if r.converged else "🟡 Not converged", level),
               (f"acceptance {r.acceptance:.2f}", "pass"), (f"autocorrelation ≈ {float(np.max(r.tau)):.0f} steps" if np.isfinite(r.tau).all() else "autocorrelation unknown", "pass"),
               (f"{int(np.min(r.ess)):,} effective samples" if np.isfinite(r.ess).all() else "effective samples unknown", "pass")])
    for w in r.warnings:
        st.warning(w)
    st.markdown("#### Parameters")
    st.dataframe(r.summary_table(), hide_index=True, width="stretch")
    st.caption("−1σ / +1σ are the distances from the median to the 16th and 84th percentiles; unequal sizes mean the posterior is lopsided.")
    tabs = st.tabs(["Corner plot", "Predictive", "Walker traces", "Export"])
    with tabs[0]:
        if len(r.names) > 6:
            st.caption("Showing the first 6 parameters.")
        st.plotly_chart(mp.corner_figure(r, run["truth"]), width="stretch")
        st.caption("Contours: joint posterior of each pair. Red ×: the least-squares fit" + ("; green ◆: the true values." if run["truth"] else "."))
    with tabs[1]:
        st.plotly_chart(mp.predictive_figure(r), width="stretch")
        st.caption(f"Residual structure along x: lag-1 correlation {r.residual_lag1:+.2f} (z = {r.residual_z:+.1f}; |z| above 3 means the model "
                   "leaves a pattern unexplained).")
    with tabs[2]:
        st.plotly_chart(mp.trace_figure(r), width="stretch")
    with tabs[3]:
        st.download_button("⬇️ Samples (CSV)", data=_samples_csv(r), file_name="posterior_samples.csv", key="post_dl_csv")
        st.download_button("⬇️ Summary (Markdown)", data=_summary_markdown(r, run["expr"]), file_name="posterior_summary.md", key="post_dl_md")
