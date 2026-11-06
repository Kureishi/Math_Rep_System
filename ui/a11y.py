"""Accessible charts: every Plotly figure is drawn through `show_figure`, which (1) themes it for the app's light or dark mode and the
colour-blind-safe series colours, and (2) gives it a text description, read out to screen readers and optionally shown on screen.

Why a wrapper and not a setting: Streamlit's st.plotly_chart has no alt-text parameter, and its own theme="streamlit" replaces
the series colours and background in a way the app's separate dark-mode switch does not follow. The wrapper is the one place that
knows both, so a chart cannot be drawn without them (tests/test_palette.py fails if any page calls st.plotly_chart directly).
"""
import html
from typing import Any

import streamlit as st

from modules import palette

SHOW_DESCRIPTIONS_KEY = "a11y_show_descriptions"


def is_dark() -> bool:
    """True if the app is in dark mode: its own 🌙 switch, or Streamlit's dark theme."""
    if st.session_state.get("dark_mode"):
        return True
    try:
        return st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001 -- older Streamlit, or no browser context (tests)
        return False


def announce(text: str) -> None:
    """Make `text` available as the chart's description: visible if the person asked for descriptions, otherwise hidden from the eye
    but read by screen readers."""
    if st.session_state.get(SHOW_DESCRIPTIONS_KEY):
        st.caption(f"📊 {text}")
    else:
        st.html(f'<span class="mrs-sr-only">{html.escape(text)}</span>')


def show_figure(fig: Any, alt: str | None = None, **kwargs: Any) -> Any:
    """Draw a Plotly figure with the app's theme, colour-blind-safe colours and a text description (`alt`, or one generated from the
    figure). Accepts everything st.plotly_chart does and RETURNS what it returns: with on_select the click or selection state, which
    the click-to-act panels (ui/plot_select.py) depend on. The theme is applied in place but idempotently, so a cached figure is safe."""
    palette.apply_theme(fig, is_dark())
    kwargs.setdefault("theme", None)                 # the theme is applied above; Streamlit's would override it
    event = st.plotly_chart(fig, **kwargs)
    announce(alt or palette.describe_figure(fig))
    return event
