"""
Compare solves: two solved problems side by side -- the one on screen, any saved one, or a what-if of
the first with some inputs changed. All the comparing is modules/solve_compare.py; this page picks the
two sides and shows the result. No LLM is involved.

A what-if is re-solved by SymPy and checked with the deterministic checks only (the page says so), so
changing an input and comparing is instant and never touches the model server.
"""
import pandas as pd
import streamlit as st

from modules import history
from modules.compare_plots import build_category_chart, build_relative_change_chart
from modules.solve_compare import (
    SolveComparison, SolveSnapshot, chart_rows, compare, comparison_markdown, snapshot_from_history,
    what_if_snapshot,
)
from ui.cache import cached, model_signature
from ui.compare_constants import CURRENT_PROBLEM, WHAT_IF_SOURCE

_STATUS_ICON = {"same": "＝", "identical": "＝", "equivalent": "≈", "changed": "≠", "role_changed": "⇄",
                "only_a": "A only", "only_b": "B only"}


def _current_snapshot() -> SolveSnapshot | None:
    model, report = st.session_state.get("model"), st.session_state.get("report")
    if model is None or report is None:
        return None
    return SolveSnapshot(label="Current problem", problem_text=st.session_state.get("problem_text", ""),
                         model=model, report=report, steps=st.session_state.get("steps") or {})


def _history_options() -> dict[str, int]:
    return {f"[{r['id']}] {r['problem_text'].strip().splitlines()[0][:60] if r['problem_text'].strip() else ''}": r["id"]
            for r in history.list_recent()}


def _resolve(source: str, options: dict[str, int]) -> SolveSnapshot | None:
    if source == CURRENT_PROBLEM:
        return _current_snapshot()
    entry_id = options.get(source)
    return snapshot_from_history(entry_id) if entry_id is not None else None


def render_compare_tab() -> None:
    st.subheader("🆚 Compare solves")
    st.caption("Line two solves up: answers, inputs, equations and verification. Pick the problem on screen "
               "or a saved one for each side -- or let B be a what-if of A with some inputs changed.")

    options = _history_options()
    current = _current_snapshot()
    a_choices = ([CURRENT_PROBLEM] if current else []) + list(options)
    if not a_choices:
        st.info("Nothing to compare yet -- solve a problem first, or open one from the history.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        a_source = st.selectbox("A", a_choices, key="compare_a_source")
    b_choices = [WHAT_IF_SOURCE] + ([CURRENT_PROBLEM] if current else []) + list(options)
    with col_b:
        b_source = st.selectbox("B", b_choices, key="compare_b_source")

    a = _resolve(a_source, options)
    if a is None:
        st.warning("Couldn't load A.")
        return

    if b_source == WHAT_IF_SOURCE:
        b = _what_if_side(a)
        if b is None:
            return
    else:
        b = _resolve(b_source, options)
        if b is None:
            st.warning("Couldn't load B.")
            return
        b.label = f"B: {b.label}"
    a.label = f"A: {a.label}" if not a.label.startswith("A: ") else a.label

    result = cached("compare_result", (_snapshot_key(a), _snapshot_key(b)), lambda: compare(a, b))
    _render_result(result)


def _snapshot_key(snap: SolveSnapshot) -> tuple:
    return (snap.label, model_signature(snap.model), snap.deterministic_only,
            sorted(snap.report.sympy_numeric_answers.items()), [(c.label, c.passed) for c in snap.report.checks])


def _what_if_side(a: SolveSnapshot) -> SolveSnapshot | None:
    known = [v for v in a.model.variables if v.known_value is not None]
    if not known:
        st.info("A has no known inputs to change.")
        return None
    seed = st.session_state.get("compare_overrides") or {}
    rows = [{"Symbol": v.symbol, "Meaning": v.meaning, "Unit": v.unit or "", "A": float(value),
             "B": float(seed.get(v.symbol, value))} for v in known if (value := v.known_value) is not None]
    st.markdown("**Change B's inputs** (edit the B column; unchanged rows stay as in A)")
    edited = st.data_editor(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        key=f"compare_editor_{model_signature(a.model)}_{sorted(seed.items())}",
        column_config={"Symbol": st.column_config.TextColumn(disabled=True),
                       "Meaning": st.column_config.TextColumn(disabled=True),
                       "Unit": st.column_config.TextColumn(disabled=True),
                       "A": st.column_config.NumberColumn(disabled=True, format="%.6g"),
                       "B": st.column_config.NumberColumn(format="%.6g")})
    overrides = {r["Symbol"]: float(r["B"]) for _, r in edited.iterrows()
                 if r["B"] is not None and pd.notna(r["B"]) and float(r["B"]) != float(r["A"])}
    if not overrides:
        st.info("Change at least one value in the B column to see a what-if.")
        return None
    try:
        return cached("compare_whatif", (model_signature(a.model), overrides),
                      lambda: what_if_snapshot(a, overrides, label="B: what-if"))
    except ValueError as e:
        st.error(str(e))
        return None


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.6g}"


def _render_result(c: SolveComparison) -> None:
    st.divider()
    m1, m2 = st.columns(2)
    m1.metric(c.a_label, f"{c.a_score:.0%}", c.a_confidence + ("" if c.a_passed else " (failed)"),
              delta_color="off")
    m2.metric(c.b_label, f"{c.b_score:.0%}", c.b_confidence + ("" if c.b_passed else " (failed)"),
              delta_color="off")
    st.markdown("**What differs**")
    for line in c.summary:
        st.markdown(f"- {line}")
    for note in c.caveats:
        st.info(note)

    only_diff = st.checkbox("Show only differences", key="compare_only_diff")
    t_ans, t_var, t_eq, t_ver, t_chart = st.tabs(["Answers", "Inputs", "Equations", "Verification", "Charts"])

    with t_ans:
        if c.answers:
            st.dataframe(pd.DataFrame([{
                "Target": d.target, "Unit": d.unit or "", "A": d.a, "B": d.b, "Difference": d.abs_diff,
                "Relative": f"{d.rel_diff:.3%}" if d.rel_diff is not None else "—",
                "": _STATUS_ICON[d.status]} for d in c.answers
                if not only_diff or d.status != "same"]), hide_index=True, width="stretch")
        else:
            st.caption("Neither side has a numeric answer.")
    with t_var:
        var_rows = [{"Symbol": v.symbol, "Meaning": v.meaning, "A": v.a_value, "B": v.b_value,
                     "Unit A": v.a_unit or "", "Unit B": v.b_unit or "", "": _STATUS_ICON[v.status]}
                    for v in c.variables if not only_diff or v.status != "same"]
        if var_rows:
            st.dataframe(pd.DataFrame(var_rows), hide_index=True, width="stretch")
        else:
            st.caption("No differences.")
    with t_eq:
        shown = [e for e in c.equations if not only_diff or e.status not in ("identical",)]
        if not shown:
            st.caption("No differences.")
        for e in shown:
            st.markdown(f"**{_STATUS_ICON[e.status]}  {e.status.replace('_', ' ')}**")
            left, right = st.columns(2)
            with left:
                st.caption(f"A · {e.a_name or '—'}")
                if e.a_latex:
                    st.latex(e.a_latex)
            with right:
                st.caption(f"B · {e.b_name or '—'}")
                if e.b_latex:
                    st.latex(e.b_latex)
    with t_ver:
        st.dataframe(pd.DataFrame([{
            "Category": k.category,
            "A": f"{k.a_passed}/{k.a_total}" if k.a_total is not None else "—",
            "B": f"{k.b_passed}/{k.b_total}" if k.b_total is not None else "—"} for k in c.categories]),
            hide_index=True, width="stretch")
        marks = {True: "pass", False: "FAIL", None: "—"}
        check_rows = [{"Check": k.label, "A": marks[k.a_passed], "B": marks[k.b_passed]}
                      for k in c.checks if not only_diff or k.a_passed != k.b_passed]
        if check_rows:
            st.dataframe(pd.DataFrame(check_rows), hide_index=True, width="stretch")
        else:
            st.caption("Every check agrees.")
    with t_chart:
        charts = chart_rows(c)
        if charts["relative_change_pct"]:
            st.plotly_chart(build_relative_change_chart(charts["relative_change_pct"]), width="stretch",
                            key="compare_rel_chart")
        else:
            st.caption("No answer is present on both sides with a non-zero A value, so there is "
                       "nothing to chart as a change.")
        if charts["category_pass_fraction"]:
            st.plotly_chart(build_category_chart(charts["category_pass_fraction"]), width="stretch",
                            key="compare_cat_chart")

    st.download_button("⬇️ Download comparison (Markdown)", data=comparison_markdown(c),
                       file_name="solve_comparison.md", mime="text/markdown", key="compare_download",
                       on_click="ignore")
