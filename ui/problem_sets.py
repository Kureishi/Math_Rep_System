"""The Problem sets mode: make verified practice sets from a solved problem and grade the answers (modules/problem_sets.py).
No language model."""
import copy

import plotly.graph_objects as go
import streamlit as st

from modules import history
from modules import problem_sets as ps
from modules.equation_engine import build_model
from modules.problem_library import LIBRARY

_SET = "_problem_set"
_REPORT = "_problem_set_grades"
_ICON = {"correct": "✅", "close": "🟡", "wrong": "❌", "missing": "⚪", "unreadable": "❓"}


def _source() -> tuple[str, dict, str] | None:
    """(statement, model JSON, default title) from the chosen source, or None."""
    options = ["Starter library", "A problem from my history"]
    if st.session_state.get("model") is not None and st.session_state.get("problem_text"):
        options.append("The problem that is open")
    choice = st.radio("Start from", options, horizontal=True, key="pset_source")
    if choice == "Starter library":
        name = st.selectbox("Problem", list(LIBRARY), key="pset_library")
        text, raw = LIBRARY[name]
        return text, raw, name
    if choice == "The problem that is open":
        return st.session_state["problem_text"], st.session_state["model"].raw_json, "Practice problems"
    entries = [e for e in history.list_recent(60) if e["passed"]]
    if not entries:
        st.info("No verified problems in your history yet. Solve one first, or start from the library.")
        return None
    pick = st.selectbox("Problem", entries, format_func=lambda e: f"#{e['id']} · {e['problem_text'][:70]}", key="pset_history")
    loaded = history.load(pick["id"])
    if loaded is None:
        st.warning("That problem is no longer in the history.")
        return None
    return loaded[0], loaded[1].raw_json, "Practice problems"


def _create_tab() -> None:
    src = _source()
    if src is None:
        return
    text, raw, default_title = src
    st.markdown("**The problem to vary**")
    st.write(text)
    try:
        model = build_model(copy.deepcopy(raw))
        ps.check_supported(model)
    except ps.ProblemSetError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"That problem could not be read ({type(exc).__name__}).")
        return
    known = ps.known_variables(model)
    labels = {s: f"{s} = {ps.fmt(v)}" + (f" {next((x.unit for x in model.variables if x.symbol == s), '') or ''}") for s, v in known.items()}
    c1, c2 = st.columns([3, 1])
    vary = c1.multiselect("Values to change", list(known), default=ps.suggest_varied(model), format_func=lambda s: labels[s], key=f"pset_vary_{default_title}",
                          help="Physical constants and zeros are left alone by default.")
    spread = c2.slider("Spread (± of each value)", 0.1, 0.8, ps.DEFAULT_SPREAD, 0.05, key="pset_spread")
    c3, c4, c5 = st.columns([1, 1, 2])
    n = int(c3.number_input("How many problems", 1, ps.MAX_PROBLEMS, 8, key="pset_n"))
    seed = int(c4.number_input("Seed", 0, 10 ** 6, 0, key="pset_seed"))
    title = c5.text_input("Title", default_title, key=f"pset_title_{default_title}")
    if st.button("🧑‍🏫 Make the set", type="primary", key="pset_go"):
        try:
            with st.spinner("Making and verifying every problem…"):
                st.session_state[_SET] = ps.generate(text, raw, n, seed, vary, spread, title)
        except ps.ProblemSetError as exc:
            st.session_state.pop(_SET, None)
            st.error(str(exc))
            return
    pset: ps.ProblemSet | None = st.session_state.get(_SET)
    if pset is None:
        return
    st.success(f"{len(pset.instances)} problems made; each answer was computed and passed the app's verification checks"
               + (f" ({pset.discarded} candidate(s) were discarded along the way)." if pset.discarded else "."))
    for w in pset.warnings:
        st.warning(w)
    show = st.toggle("Show the answer key", False, key="pset_show_key")
    for i in pset.instances:
        with st.expander(f"Problem {i.index}", expanded=i.index <= 2):
            st.write(i.text)
            if not i.rewritten:
                st.caption("Use: " + ", ".join(f"{g['symbol']} = {ps.fmt(g['value'])}{(' ' + g['unit']) if g['unit'] else ''}" for g in i.given))
            st.caption("Find: " + "; ".join(f"{f['meaning']} ({f['symbol']})" for f in i.find))
            if show:
                st.success("  ·  ".join(f"{t} = {ps.fmt(i.answers[t])}" for t in pset.targets))
    st.markdown("#### Download")
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Worksheet (HTML, no answers)", ps.worksheet_html(pset), "worksheet.html", "text/html", key="pset_dl_ws")
    d1.download_button("⬇️ Worksheet (Markdown)", ps.worksheet_markdown(pset), "worksheet.md", key="pset_dl_wsmd")
    d2.download_button("⬇️ Answer key (HTML)", ps.worksheet_html(pset, True), "answer_key.html", "text/html", key="pset_dl_key")
    d2.download_button("⬇️ Answer key (CSV)", ps.answer_key_csv(pset), "answer_key.csv", key="pset_dl_keycsv")
    d3.download_button("⬇️ Problem set file (for grading later)", ps.to_json(pset), "problem_set.json", key="pset_dl_json")
    names = st.text_input("Student names for a submissions template (comma separated)", "", key="pset_names")
    if names.strip():
        d3.download_button("⬇️ Submissions template (CSV)", ps.submission_template_csv(pset, [n.strip() for n in names.split(",") if n.strip()]),
                           "submissions_template.csv", key="pset_dl_tpl")


def _grade_tab() -> None:
    pset: ps.ProblemSet | None = st.session_state.get(_SET)
    up = st.file_uploader("Problem set file (leave empty to use the set just made)", type=["json"], key="pset_upload")
    if up is not None:
        try:
            pset = ps.from_json(up.getvalue().decode("utf-8"))
        except ps.ProblemSetError as exc:
            st.error(str(exc))
            return
    if pset is None:
        st.info("Make a set on the first tab, or upload a problem set file.")
        return
    st.caption(f"Grading “{pset.title}”: {len(pset.instances)} problems, finding {', '.join(pset.targets)}.")
    c1, c2, c3 = st.columns(3)
    tol = c1.slider("Counts as correct within (%)", 0.1, 10.0, ps.DEFAULT_TOLERANCE * 100, 0.1, key="pset_tol") / 100
    close = c2.slider("Counts as close within (%)", 1.0, 20.0, ps.CLOSE_TOLERANCE * 100, 0.5, key="pset_close") / 100
    partial = c3.slider("Credit for close", 0.0, 1.0, ps.PARTIAL_CREDIT, 0.25, key="pset_partial")
    with st.expander("Check a single answer"):
        e1, e2, e3 = st.columns(3)
        idx = int(e1.number_input("Problem", 1, len(pset.instances), 1, key="pset_one_i"))
        tgt = e2.selectbox("Quantity", pset.targets, key="pset_one_t")
        ans = e3.text_input("Answer", "", key="pset_one_a")
        if ans.strip():
            inst = pset.instances[idx - 1]
            value, _ = ps.parse_answer(ans)
            v = ps.grade_value(value, inst.answers[tgt], tol, close, partial, other_answers=inst.answers, base_answers=pset.base_answers, target=tgt) \
                if value is not None else ps.Verdict("unreadable", 0.0, "no number found in the answer")
            st.write(f"{_ICON[v.status]} **{v.status}**" + (f": {v.diagnosis}" if v.diagnosis else "") + (f"  (the key is {ps.fmt(inst.answers[tgt])})" if v.status != "correct" else ""))
    f = st.file_uploader("Submissions (CSV with columns student, problem, target, answer)", type=["csv", "txt"], key="pset_sub_file")
    pasted = st.text_area("…or paste them", height=120, key="pset_sub_paste", placeholder="student,problem,target,answer\nAna,1,a,4.0\nAna,1,d,56")
    text = f.getvalue().decode("utf-8", errors="replace") if f is not None else pasted
    if not text.strip():
        return
    try:
        report = ps.grade_submissions(pset, ps.read_submissions(text), tol, close, partial)
    except ps.ProblemSetError as exc:
        st.error(str(exc))
        return
    st.session_state[_REPORT] = report
    for err in report.problems_with_errors[:10]:
        st.warning(f"Skipped: {err}")
    if len(report.problems_with_errors) > 10:
        st.caption(f"…and {len(report.problems_with_errors) - 10} more skipped rows.")
    for n in report.notes:
        st.info(n)
    if not report.students:
        return
    m1, m2, m3 = st.columns(3)
    m1.metric("Students", len(report.students))
    m2.metric("Class mean", f"{report.mean_percent:.0f}%")
    m3.metric("Hardest problem", min(report.problems, key=lambda p: p.facility).problem if report.problems else "—")
    st.markdown("#### Scores")
    st.dataframe([{"student": s.name, "score": s.score, "out of": s.possible, "percent": round(s.percent, 1)} for s in report.students],
                 hide_index=True, width="stretch")
    st.markdown("#### By problem")
    labels = [f"{p.problem} ({p.target})" for p in report.problems]
    fig = go.Figure(go.Bar(x=labels, y=[p.facility * 100 for p in report.problems], marker_color="#4C78A8"))
    fig.update_layout(height=300, yaxis_title="% fully correct", xaxis_title="problem (quantity)", margin=dict(t=20))
    st.plotly_chart(fig, width="stretch")
    mistakes = [p for p in report.problems if p.common_mistake]
    if mistakes:
        st.markdown("**Commonest mistakes**")
        for p in mistakes:
            st.markdown(f"- Problem {p.problem} ({p.target}): {p.common_mistake}")
    with st.expander("Every answer, with what went wrong"):
        for s in report.students:
            st.markdown(f"**{s.name}**: {s.score:g}/{s.possible}")
            for it in s.items:
                if it.verdict.status != "correct":
                    st.caption(f"{_ICON[it.verdict.status]} problem {it.problem} ({it.target}): “{it.answer_text}”, key {ps.fmt(it.key or 0)}"
                               + (f"  → {it.verdict.diagnosis}" if it.verdict.diagnosis else ""))
    st.download_button("⬇️ Scores (CSV)", report.totals_csv(), "scores.csv", key="pset_dl_scores")
    st.download_button("⬇️ Every answer (CSV)", report.results_csv(), "graded_answers.csv", key="pset_dl_graded")


def render_problem_sets_tab() -> None:
    st.subheader("🧑‍🏫 Problem sets")
    st.caption("Make a set of practice problems from a solved one, with different numbers each time and an answer key that passed the app's "
               "own verification, then grade the answers: it reads \"12.5 m/s\" or \"1.25e1\", and says what slip lies behind a wrong one "
               "(a sign, a missing ½, a power of ten). No language model involved.")
    create, grade = st.tabs(["1 · Make a set", "2 · Grade answers"])
    with create:
        _create_tab()
    with grade:
        _grade_tab()
