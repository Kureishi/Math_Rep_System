"""
The sidebar's History list: grouped by day ("Today", "Yesterday", ...), starred entries first, a filter box that matches names and problem text,
rename, and a delete that can be undone.

  filter        every word typed must appear in an entry's name, domain or problem text (modules/history.py: matches)
  star button   star an entry: it is listed first, under "⭐ Starred", and is never pruned when the list reaches its size cap
  rename        the pencil opens a name editor under the entry (unchanged)
  delete        immediate, but the History header then shows what was deleted with an "Undo" button (Streamlit's toasts cannot hold a
                button, so the undo bar stays until the next delete, a click on its ✕, or the end of the session). Undo puts the entry back
                under its old id, so the open problem's autosave keeps working.
"""
import streamlit as st

from modules import history
from ui import view_state
from ui.actions import flash

SHOWN_UNFILTERED = 25                 # entries listed when the filter is empty (starred ones are always listed)
_FILTER_KEY = "history_filter"
_UNDO_KEY = "_history_undo"
_SHOW_ALL_KEY = "_history_show_all"


def _rename_form(entry: dict) -> None:
    """The inline name editor for one history entry: Enter or Save stores the name (an empty name goes back to the domain), Cancel leaves it as it was."""
    with st.form(key=f"renameform_{entry['id']}", border=False):
        new_name = st.text_input("Name", value=entry.get("title") or "", placeholder=entry.get("domain") or "Name this problem", max_chars=history.MAX_TITLE_CHARS,
                                 label_visibility="collapsed", key=f"renameinput_{entry['id']}")
        f1, f2 = st.columns(2)
        save = f1.form_submit_button("Save", type="primary")
        cancel = f2.form_submit_button("Cancel")
    if save:
        try:
            history.rename(entry["id"], new_name)
        except Exception as exc:  # noqa: BLE001 -- a locked database must not crash the sidebar
            st.warning(f"Could not rename it ({type(exc).__name__}: {str(exc)[:100]}).")
            return
        st.session_state.pop("_renaming_history", None)
        st.rerun()
    if cancel:
        st.session_state.pop("_renaming_history", None)
        st.rerun()


# ------------------------------------------------------------------------------------------ callbacks (they run before the rerun)

def _delete(entry_id: int) -> None:
    try:
        saved = history.delete_with_undo(entry_id)
    except Exception as exc:  # noqa: BLE001 -- a locked database must not crash the sidebar
        st.toast(f"Could not delete it ({type(exc).__name__}).", icon="⚠️")
        return
    if saved is not None:
        st.session_state[_UNDO_KEY] = saved


def _undo() -> None:
    saved = st.session_state.pop(_UNDO_KEY, None)
    if saved is None:
        return
    try:
        ok = history.restore(saved)
    except Exception as exc:  # noqa: BLE001
        st.toast(f"Could not restore it ({type(exc).__name__}).", icon="⚠️")
        return
    flash("Restored" if ok else "Could not restore it: its number has been used again.", "↩️" if ok else "⚠️")


def _dismiss_undo() -> None:
    st.session_state.pop(_UNDO_KEY, None)


def _toggle_pin(entry_id: int, pinned: bool) -> None:
    try:
        history.set_pinned(entry_id, pinned)
    except Exception as exc:  # noqa: BLE001
        st.toast(f"Could not change the star ({type(exc).__name__}).", icon="⚠️")


def _load(entry: dict) -> None:
    loaded = history.load(entry["id"])
    if loaded is None:
        return
    p_text, l_model, l_report, l_steps, l_scenarios = loaded
    st.session_state.update(problem_text=p_text, model=l_model, report=l_report, steps=l_steps, scenarios=l_scenarios, pdf_bytes=None, plot_snapshots={})
    # reopen the problem the way it was being explored (Monte Carlo inputs, sweep setup, plot axes, ...) -- see ui/view_state.py
    n_restored = view_state.restore(entry["id"])
    if n_restored:
        flash(f"Restored {n_restored} saved view setting(s)", "🗂️")        # the caller reruns right after: st.toast would be discarded


# ------------------------------------------------------------------------------------------ rendering

def _render_undo_bar() -> None:
    saved = st.session_state.get(_UNDO_KEY)
    if not saved:
        return
    name = saved.get("title") or saved.get("domain") or "entry"
    with st.container(border=True):
        st.caption(f"🗑️ Deleted **{name}**")
        left, right = st.columns([3, 1])
        left.button("Undo", key="history_undo", icon=":material/undo:", type="primary", on_click=_undo, width="stretch")
        right.button("", key="history_undo_dismiss", icon=":material/close:", type="tertiary", on_click=_dismiss_undo, help="Dismiss (the entry stays deleted)")


def _render_entry(entry: dict, renaming) -> None:
    badge = "✅" if entry["passed"] else "⚠️"
    preview = entry["problem_text"][:45] + ("..." if len(entry["problem_text"]) > 45 else "")
    st.caption(f"{badge} **{history.display_title(entry)}** -- {entry['timestamp'][11:16]}")
    st.caption(preview)
    star, load, rename, delete, _ = st.columns([1, 1, 1, 1, 4])
    # An icon, not the text glyphs ★ / ☆: those come from whatever font the browser falls back to and rendered clipped in the narrow column. A starred entry shows the
    # yellow star emoji, an unstarred one Streamlit's own outlined star, drawn in the same icon font as the load / rename / delete buttons beside it.
    star.button("", icon="⭐" if entry["pinned"] else ":material/star_outline:", key=f"pinhist_{entry['id']}", type="tertiary", on_click=_toggle_pin,
                args=(entry["id"], not entry["pinned"]),
                help="Remove the star" if entry["pinned"] else "Star this entry: it is listed first and is never removed when the history is full")
    if load.button("", icon=":material/restart_alt:", key=f"load_{entry['id']}", type="tertiary", help="Load this problem"):
        _load(entry)
        st.rerun()
    rename.button("", icon=":material/edit:", key=f"renamehist_{entry['id']}", type="tertiary", help="Rename this entry",
                  on_click=st.session_state.__setitem__, args=("_renaming_history", entry["id"]))
    delete.button("", icon=":material/delete:", key=f"delhist_{entry['id']}", type="tertiary", help="Delete from history (you can undo it)",
                  on_click=_delete, args=(entry["id"],))
    if renaming == entry["id"]:
        _rename_form(entry)           # full width, under the entry: in the entry's narrow column "Cancel" wrapped onto two lines


def render_history_panel() -> None:
    """The whole History section of the sidebar."""
    st.header("History")
    _render_undo_bar()
    everything = history.list_with_pinned(history.MAX_HISTORY_RECORDS)
    if not everything:
        st.caption("No solved problems yet -- they'll be saved here automatically.")
        return
    query = st.text_input("Filter history", key=_FILTER_KEY, placeholder="🔍 Filter by name or problem text", label_visibility="collapsed")
    filtering = bool(query.strip())
    entries = history.filter_entries(everything, query)
    if filtering:
        st.caption(f"{len(entries)} of {len(everything)} match" if entries else f"Nothing matches “{query.strip()}”.")
        entries = entries[:50]
    else:
        recent = [e for e in entries if not e["pinned"]]
        if not st.session_state.get(_SHOW_ALL_KEY) and len(recent) > SHOWN_UNFILTERED:
            keep = {e["id"] for e in recent[:SHOWN_UNFILTERED]}
            entries = [e for e in entries if e["pinned"] or e["id"] in keep]
    renaming = st.session_state.get("_renaming_history")
    for heading, group in history.sections(entries):
        st.markdown(f"**{heading}**")
        for entry in group:
            _render_entry(entry, renaming)
    if not filtering and len([e for e in everything if not e["pinned"]]) > SHOWN_UNFILTERED:
        older = len([e for e in everything if not e["pinned"]]) - SHOWN_UNFILTERED
        if st.session_state.get(_SHOW_ALL_KEY):
            st.button("Show fewer", key="history_show_fewer", type="tertiary", on_click=st.session_state.pop, args=(_SHOW_ALL_KEY, None))
        else:
            st.button(f"Show {older} older", key="history_show_older", type="tertiary", on_click=st.session_state.__setitem__, args=(_SHOW_ALL_KEY, True))
