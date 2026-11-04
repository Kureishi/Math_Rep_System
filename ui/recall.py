"""The 'Have I solved this before?' page (modules/recall.py): search the history by meaning, wording and equation shape."""
import streamlit as st

from modules import history, recall
from modules.llm_client import LMStudioClient, pick_embedding_model
from modules.router import WORD_PROBLEM
from modules.similarity import problem_shape
from ui import view_state

_REPORT = "_recall_report"
_BADGE = {"same": "✅ Same problem", "very similar": "🟢 Very similar", "related": "🔵 Related"}
_SIGNAL_NAMES = {"meaning": "meaning (embedding model)", "wording": "wording", "shape": "equation shape"}


def _open(entry_id: int) -> None:
    """Open a past problem in the solver, the way the sidebar's history list does."""
    loaded = history.load(entry_id)
    if loaded is None:
        st.warning("That problem is no longer in the history.")
        return
    text, model, report, steps, scenarios = loaded
    st.session_state.update(problem_text=text, model=model, report=report, steps=steps, scenarios=scenarios,
                            pdf_bytes=None, plot_snapshots={})
    st.session_state["_pending_mode"] = WORD_PROBLEM        # app.py applies it before the sidebar's mode radio exists
    view_state.restore(entry_id)
    st.rerun()


def render_recall_tab(client: LMStudioClient) -> None:
    st.subheader("🔎 Have I solved this before?")
    st.caption("Searches your history of solved problems by **meaning** (needs an embedding model loaded in LM Studio), by "
               "**wording** (numbers are ignored, so the same problem with different numbers matches) and by **equation shape**. "
               "Each result says which of these matched.")
    current_model = st.session_state.get("model")
    current_text = st.session_state.get("problem_text", "")
    if current_model is not None and current_text:
        if st.button("Use the problem that is open", key="recall_use_current"):
            st.session_state["recall_query"] = current_text
    query = st.text_area("Describe the problem (or paste it)", key="recall_query", height=110,
                         placeholder="A car accelerates from 8 m/s to 20 m/s in 6 s. How far does it travel?")
    use_shape = False
    if current_model is not None:
        use_shape = st.checkbox("Also match the equation shape of the problem that is open", True, key="recall_use_shape",
                                help="Finds problems built from the same equations even when they are worded differently.")
    try:
        model_name = pick_embedding_model(client.list_models())
    except Exception:  # noqa: BLE001 -- LM Studio off: the search still works without embeddings
        model_name = None
    st.caption(f"Meaning-based matching: {'on, using ' + model_name if model_name else 'off (no embedding model loaded; load one such as nomic-embed-text)'}.")

    if st.button("🔎 Search my history", type="primary", key="recall_go"):
        if not query.strip():
            st.warning("Describe a problem first.")
        else:
            bar = st.progress(0.0, text="Looking through the history…")

            def show_progress(done: int, total: int) -> None:
                bar.progress(done / max(total, 1), text=f"Indexing past problems ({done}/{total})…")

            found = recall.search_history(query, client, problem_shape(current_model) if (use_shape and current_model is not None) else None,
                                          limit=6, on_progress=show_progress)
            bar.empty()
            st.session_state[_REPORT] = found
    report: recall.SearchReport | None = st.session_state.get(_REPORT)
    if report is None:
        return
    st.caption(f"Searched {report.searched} problem(s) using " + ", ".join(_SIGNAL_NAMES[s] for s in report.signals) + ".")
    for note in report.notes:
        st.caption(f"ℹ️ {note}")
    if not report.matches:
        st.info("Nothing in your history is close to this: it looks new.")
        return
    best = report.best
    if best is not None and best.verdict == "same":
        st.success("You have solved this before" + (" (with different numbers)." if best.same_problem_new_numbers else "."))
    for m in report.matches:
        with st.container(border=True):
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{_BADGE[m.verdict]}** · {m.score:.0%} · {m.domain or '—'} · {m.timestamp[:16].replace('T', ' ')}")
            c1.caption(f"Matched on: {m.why}")
            c1.write(m.problem_text if len(m.problem_text) < 320 else m.problem_text[:317] + "…")
            if c2.button("Open", key=f"recall_open_{m.id}"):
                _open(m.id)
