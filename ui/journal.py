"""
Research-journal mode: stitch chosen history entries into one running Markdown document.
"""
import streamlit as st
from modules.research_journal import build_journal_entry, generate_journal_markdown
from modules import history


def render_research_journal_tab():
    """Concept-based browsing across solved-problem history, plus
    stitching a chosen set of entries into one combined Markdown
    document -- see concept_index.py and research_journal.py. This is
    the research-MEMORY layer: "what has this history actually
    covered" and "give me one document covering these problems,"
    neither of which the sidebar's one-at-a-time recent-history list
    supports on its own."""
    st.subheader("📔 Research journal")
    tab_browse, tab_build = st.tabs(["Browse by concept", "Build a journal"])

    with tab_browse:
        st.caption("Every concept (named formula, or domain when nothing more specific was "
                    "recognized) that's shown up across solved problems in history, most-common "
                    "first -- a map of what this history has actually covered.")
        concepts = history.list_concepts()
        if not concepts:
            st.caption("No concept-tagged problems yet -- solve and verify a problem first.")
        else:
            search = st.text_input("🔎 Filter concepts", key="concept_search",
                                     placeholder="e.g. energy, Newton, domain: mechanics")
            filtered = [c for c in concepts if search.strip().lower() in c["concept"].lower()] \
                if search.strip() else concepts
            if search.strip() and not filtered:
                st.caption(f"No concepts match \"{search}\".")
            for c in filtered:
                with st.expander(f"{c['concept']} ({c['count']})"):
                    matches = history.problems_for_concept(c["concept"])
                    for m in matches:
                        badge = "✅" if m["passed"] else "⚠️"
                        snippet = m["problem_text"][:90] + ("..." if len(m["problem_text"]) > 90 else "")
                        st.caption(f"{badge} [{m['id']}] {snippet}")

    with tab_build:
        st.caption("Pick a set of solved problems and combine them into one Markdown document: "
                    "problem statements, derived equations, results, concept citations, and any "
                    "transfer-learning scenarios -- an actual paper trail instead of scattered "
                    "individually-loaded problems.")
        recent = history.list_recent(limit=50)
        if not recent:
            st.caption("No solved problems yet.")
            return
        options = {f"[{e['id']}] {e['problem_text'][:60]}": e["id"] for e in recent}
        selected_labels = st.multiselect("Select problems to include", list(options.keys()),
                                           key="journal_selected_problems")
        title = st.text_input("Journal title", value="Research Journal", key="journal_title")
        if st.button("Generate journal", key="journal_generate_button") and selected_labels:
            entry_ids = [options[label] for label in selected_labels]
            row_by_id = {r["id"]: r for r in recent}
            entries = []
            for eid in entry_ids:
                loaded = history.load(eid)
                if loaded is None:
                    continue
                timestamp = row_by_id.get(eid, {}).get("timestamp", "")
                entries.append(build_journal_entry(eid, loaded, timestamp=timestamp))
            st.session_state["_journal_markdown"] = generate_journal_markdown(entries, title=title)
            st.toast(f"Journal '{title}' generated", icon="📔")

        journal_md = st.session_state.get("_journal_markdown")
        if journal_md:
            st.download_button("⬇️ Download journal (Markdown)", data=journal_md,
                                 file_name="research_journal.md", mime="text/markdown",
                                 key="journal_download")
            with st.expander("Preview"):
                st.markdown(journal_md)
