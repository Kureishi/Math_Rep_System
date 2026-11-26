"""
Page-layout helpers for the results view: spacing, cards, and a right-hand "Contents" rail of jump links.

Why: a solved problem used to be one long column of the main solution followed directly by a dozen collapsed panels, so the extras read as part of the answer. Now each target's main solution sits in its
own bordered card, its extra tools are grouped under one "Go further" header as tabs, and an optional rail on the right lists the sections so a long page can be jumped around.

Streamlit has no right-hand sidebar, so the rail is a column of the page with a toggle (default on): it is sticky (stays in view while scrolling) where the browser allows, and hidden below about 1100 px wide, where
it would only squeeze the content. Its links are plain in-page anchors (`#sec-<key>`): each section marks its place with `anchor()`, and the rail is drawn last (into a container created first), so it lists exactly the
sections this problem produced.
"""
from contextlib import contextmanager
from typing import Iterator

import streamlit as st

SECTIONS_KEY = "_rail_sections"          # [(key, label)] collected during one run
SHOW_RAIL_KEY = "show_rail"
RAIL_CONTAINER_KEY = "results_rail"


def gap(rem: float = 1.5) -> None:
    """Vertical white space (Streamlit has no spacer that takes a size)."""
    st.markdown(f'<div class="mrs-gap" style="height:{rem}rem"></div>', unsafe_allow_html=True)


@contextmanager
def panel(label: str) -> Iterator[None]:
    """A titled card: one extra tool of the "Go further" tabs (it is always open, because it sits in a tab the person chose to open)."""
    with st.container(border=True):
        st.markdown(f"**{label}**")
        yield


def begin_sections() -> None:
    st.session_state[SECTIONS_KEY] = []


def anchor(key: str, label: str) -> None:
    """Mark a place on the page that the rail can link to."""
    st.session_state.setdefault(SECTIONS_KEY, []).append((key, label))
    st.markdown(f'<div id="sec-{key}" class="mrs-anchor"></div>', unsafe_allow_html=True)


def render_rail() -> None:
    """The rail's content: one link per anchor placed in this run."""
    sections = st.session_state.get(SECTIONS_KEY) or []
    st.markdown("**Contents**")
    if not sections:
        st.caption("Nothing to jump to yet.")
        return
    st.markdown("\n".join(f"- [{label}](#sec-{key})" for key, label in sections))
