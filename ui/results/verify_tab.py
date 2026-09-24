"""
The Verify tab of a solved problem.
"""
import streamlit as st
from config import settings
from modules.llm_client import LMStudioClient
from modules.equation_engine import ProblemModel, target_kind
from modules.verifier import VerificationReport, confidence_label
from modules.paranoid import run_paranoid_check
from modules.self_consistency import run_self_consistency_check, numeric_answer_spread
from modules.adversarial_testing import run_adversarial_suite
from modules.plotter import build_spread_plot
from modules.plot_snapshot import snapshot_spread_plot


def render_verify_tab(client: LMStudioClient, model: ProblemModel, report: VerificationReport, problem_text: str, tab_verify):
    """The Verify tab: verification report, paranoid mode, self-consistency, adversarial suite."""
    with tab_verify:
        with st.expander("Verification detail (raw check list)"):
            for c in report.checks:
                icon = "✅" if c.passed else "❌"
                margin_tag = ""
                if c.passed and c.margin_ratio is not None:
                    margin_tag = f"  `confidence: {confidence_label(c.margin_ratio)}`"
                (st.write if c.passed else st.error)(f"{icon} **{c.label}**: {c.detail}{margin_tag}")
            for target, val in report.sympy_numeric_answers.items():
                st.write(f"Derived-equation answer for `{target}`: `{val:.6g}`")
            for target, val in report.llm_independent_answers.items():
                st.write(f"Independent cross-check for `{target}`: `{val:.6g}`")

        # ---- domain of validity: when does each formula's derived relation
        # actually make sense (never divide by zero, sqrt of a negative, log
        # of a non-positive value, etc.) -- shown as its own panel even for
        # restrictions that AREN'T currently violated, since knowing the
        # boundary of a formula's validity is useful on its own
        if report.domain_notes:
            with st.expander("Domain of validity", expanded=any(n.violated for n in report.domain_notes)):
                for note in report.domain_notes:
                    if note.violated:
                        st.error(f"**{note.equation}** -- undefined with the given values: " +
                                  "; ".join(r.description for r in note.violated))
                    restrictions_ok = note.satisfied + note.pending
                    if restrictions_ok:
                        descs = "; ".join(r.description for r in restrictions_ok)
                        st.write(f"**{note.equation}** requires: {descs}")

        # ---- physical plausibility: a softer, advisory-only cousin of
        # domain of validity above -- flags values that are mathematically
        # fine but land far outside what's normal for the kind of quantity
        # involved (a 500 m/s^2 acceleration, a negative mass). Never
        # affects report.passed; see plausibility.py.
        if report.plausibility_notes:
            with st.expander("⚠️ Physical plausibility check", expanded=True):
                st.caption("Advisory only -- these values are mathematically valid, just unusual for this "
                            "kind of quantity. Worth a second look, not necessarily wrong.")
                for note in report.plausibility_notes:
                    st.warning(note.message)

        # ---- paranoid mode: re-run extraction through a SECOND,
        # independently-configured model and compare its equations/answers
        # against the primary derivation -- off unless a secondary model is
        # configured in config.py, since it doubles extraction cost.
        # See paranoid.py.
        if settings.secondary_reasoning_model:
            with st.expander(f"🕵️ Paranoid mode: cross-check against {settings.secondary_reasoning_model}"):
                valid, valid_msg = client.validate_model(settings.secondary_reasoning_model)
                if not valid:
                    st.warning(f"Can't run the cross-check: {valid_msg}")
                elif st.button("Run cross-check", key="paranoid_button"):
                    with st.spinner(f"Re-solving with {settings.secondary_reasoning_model}..."):
                        st.session_state["paranoid_result"] = run_paranoid_check(
                            client, problem_text, model, report.sympy_numeric_answers,
                        )
                paranoid_result = st.session_state.get("paranoid_result")
                if paranoid_result and paranoid_result.ran:
                    if paranoid_result.error:
                        st.error(f"Secondary model failed: {paranoid_result.error}")
                    else:
                        st.write(f"Equation structure match: {paranoid_result.equations_match:.0%}")
                        if paranoid_result.disagreements:
                            for target, (primary_val, secondary_val) in paranoid_result.disagreements.items():
                                st.error(f"**{target}** disagreement: primary = {primary_val:.6g}, "
                                          f"{paranoid_result.secondary_model} = {secondary_val:.6g}")
                        elif paranoid_result.secondary_answers:
                            st.success("Both models agree on every shared answer.")

        # ---- self-consistency check: re-run extraction on the SAME model
        # 2-5 times and compare the derivations to each other -- a
        # different signal than paranoid mode: disagreement here usually
        # means the PROBLEM STATEMENT is ambiguous, not that a model is
        # specifically wrong. See self_consistency.py.
        with st.expander("🔁 Self-consistency check"):
            st.caption("Re-runs extraction on this same model several times and checks whether it "
                        "derives the same equations each time -- catches an ambiguous problem "
                        "statement, not a wrong model.")
            sc_runs = st.slider("Number of runs", 2, 5, 3, key="self_consistency_runs")
            if st.button("Run self-consistency check", key="self_consistency_button"):
                with st.spinner(f"Re-extracting {sc_runs} times..."):
                    st.session_state["self_consistency_result"] = run_self_consistency_check(
                        client, problem_text, runs=sc_runs,
                    )
            sc_result = st.session_state.get("self_consistency_result")
            if sc_result is not None:
                if sc_result.consistent is None:
                    st.warning("Couldn't get enough successful runs to compare "
                                f"({len(sc_result.errors)} failed).")
                elif sc_result.consistent:
                    st.success(f"Consistent across {sc_result.runs} runs -- "
                                f"similarity scores: {', '.join(f'{s:.0%}' for s in sc_result.shapes_match)}")
                else:
                    st.warning(f"Inconsistent across {sc_result.runs} runs -- similarity scores: "
                                f"{', '.join(f'{s:.0%}' for s in sc_result.shapes_match)}. This may mean "
                                "the problem statement is ambiguous enough that even this model can't "
                                "parse it the same way twice.")

                # numeric spread: complementary to the structural similarity
                # score above -- two runs can score a near-perfect shapes_match
                # (identical equation structure) and STILL disagree on the
                # actual number if, say, one run extracted a slightly
                # different known value. Only offered for a target every
                # usable run actually shares.
                sc_targets = sorted(set().union(*(
                    {t for t in m.solve_for if target_kind(m, t) == "equation"}
                    for m in sc_result.models if m is not None
                ))) if any(m is not None for m in sc_result.models) else []
                if sc_targets:
                    sc_target = st.selectbox("See the numeric spread for...", sc_targets,
                                               key="self_consistency_spread_target")
                    spread_values = numeric_answer_spread(sc_result, sc_target)
                    if len(spread_values) >= 2:
                        fig = build_spread_plot(spread_values, sc_target)
                        st.plotly_chart(fig, width='stretch')
                        st.caption(f"{sc_target}: mean {sum(spread_values) / len(spread_values):.4g}, "
                                    f"min {min(spread_values):.4g}, max {max(spread_values):.4g} "
                                    f"across {len(spread_values)} runs.")
                        png = snapshot_spread_plot(spread_values, sc_target)
                        st.download_button("Download plot as PNG", data=png,
                                             file_name=f"self_consistency_{sc_target}.png", mime="image/png",
                                             key="download_spread_png")
                    elif len(spread_values) == 1:
                        st.caption(f"Only one run solved for {sc_target} -- need at least 2 to show a spread.")

        # ---- adversarial edge-case testing: a QA tool for the SYSTEM
        # itself (not the student) -- perturbs this problem's own known
        # inputs to zero, negative, tiny, and huge values and runs each
        # through the real solving/plausibility pipeline, reporting
        # exactly what happened rather than pre-judging whether it's
        # acceptable. See adversarial_testing.py.
        adv_targets = [t for t in model.solve_for if target_kind(model, t) == "equation"]
        if adv_targets:
            with st.expander("🧪 Adversarial edge-case testing (developer QA)"):
                st.caption("Perturbs this problem's own known inputs to zero, negative, tiny, and "
                            "huge values and runs each through the real solving + plausibility "
                            "pipeline -- not to check if the ANSWER makes sense, but to check "
                            "whether the PIPELINE survives being handed something nasty without an "
                            "unhandled crash. A developer/debugging tool.")
                adv_target = st.selectbox("Target to test", adv_targets, key="adv_target")
                if st.button("Run adversarial suite", key="adv_run_button"):
                    with st.spinner("Running edge cases..."):
                        st.session_state["adv_outcomes"] = run_adversarial_suite(model, adv_target)
                adv_outcomes = st.session_state.get("adv_outcomes")
                if adv_outcomes:
                    status_icon = {"solved": "✅", "unsolvable": "➖", "timeout": "⏱️", "exception": "❌"}
                    n_exceptions = sum(1 for o in adv_outcomes if o.status == "exception")
                    if n_exceptions:
                        st.error(f"{n_exceptions} of {len(adv_outcomes)} edge case(s) raised an "
                                  "unhandled exception -- worth a look.")
                    else:
                        st.success(f"All {len(adv_outcomes)} edge cases handled without an "
                                    "unhandled exception (solved, correctly unsolvable, or timed out).")
                    for o in adv_outcomes:
                        icon = status_icon.get(o.status, "❓")
                        st.write(f"{icon} **{o.variant.label}** ({o.status}): {o.detail}")
