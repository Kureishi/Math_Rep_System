"""Switching modes from inside a page: the jump itself (shared with the sidebar's "Ask anything" box) and the banner the word-problem page shows when what was typed
into it clearly belongs to another mode."""
import hashlib

import streamlit as st

from modules import mode_registry, router


def go(route: "router.Route") -> None:
    """Navigate to a route's mode and fill in what it prefills. Safe to call from a button callback (callbacks run before the page is rebuilt, so the mode radio's own
    state may be set) and from the sidebar before that radio is built."""
    st.session_state["app_mode"] = route.mode
    for key, value in route.prefill.items():
        st.session_state[key] = value
    st.session_state["_ask_notice"] = f"→ {route.mode}: {route.reason}"
    st.session_state.pop("_ask_last", None)


def _fingerprint(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).lower().encode("utf-8")).hexdigest()[:16]


def _dismiss(text_fingerprint: str) -> None:
    st.session_state.setdefault("_route_dismissed", set()).add(text_fingerprint)


SUGGESTIONS_KEY = "mode_suggestions"      # the Advanced-settings checkbox; False = never show the banner (this session)


def _stop_suggesting() -> None:
    st.session_state[SUGGESTIONS_KEY] = False


def render_mode_suggestion(text: str) -> None:
    """Under the word-problem box: if the text clearly belongs to another mode, say so in ONE line and offer to switch (carrying the text across where that mode can use it).
    The reason, the confidence and whether the mode needs the language model are in the tooltip, so the banner no longer pushes Represent & Solve down the page.

    Quiet by design: never while a solved problem is on screen, never for text the person already chose to keep here, never for short or ordinary word problems
    (router.suggest_mode's threshold was measured against ordinary word problems), never once "don't suggest again" is chosen (the button on the line, or the checkbox
    in Advanced settings), and it only suggests -- Represent & Solve still works as before."""
    if not st.session_state.get(SUGGESTIONS_KEY, True) or st.session_state.get("model") is not None:
        return
    found = router.suggest_mode(text)
    if found is None:
        return
    fp = _fingerprint(text)
    if fp in st.session_state.get("_route_dismissed", set()):
        return
    try:
        needs_model = mode_registry.discover().spec(found.mode).needs_client
    except (KeyError, AttributeError):
        needs_model = True
    why = (f"{found.reason} ({found.confidence:.0%} sure). That mode is built for it" + (" and runs without LM Studio" if not needs_model else "") +
           ". The word-problem solver will still try it if you press Represent & Solve.")
    line, switch, keep, mute = st.columns([6, 2, 2, 1], vertical_alignment="center")
    line.markdown(f"🧭 Looks like **{found.mode}**", help=why)
    switch.button("Switch", key="route_banner_go", type="primary", on_click=go, args=(found,), help=f"Go to {found.mode}", width="stretch")
    keep.button("Keep it here", key="route_banner_stay", on_click=_dismiss, args=(fp,), help="Hide this suggestion for this text", width="stretch")
    mute.button("", key="route_banner_off", icon=":material/notifications_off:", type="tertiary", on_click=_stop_suggesting,
                help="Don't suggest modes again (turn back on in Advanced settings)")
