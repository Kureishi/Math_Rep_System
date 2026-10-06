"""
Plots you can click.

`selectable_chart` draws a Plotly figure whose points can be clicked, and returns Streamlit's selection
state. What a click MEANS is decided by the page that draws the chart (a bar of a tornado chart selects
an input, a grid point selects parameter values, a graph node selects a step) using the parsers in
modules/plot_selection.py; this only draws.

The selection lives in `st.session_state[key]` and survives reruns, so a selected bar stays selected
while the person picks what to do with it. Figures passed here must be freshly built, not shared through
ui/cache.py: Streamlit's selection setup modifies the figure's layout, and a cached figure is shared
between reruns and must stay read-only.
"""
from typing import Any

import plotly.graph_objects as go
import streamlit as st


def selectable_chart(fig: go.Figure, key: str) -> Any:
    """Draw `fig` full-width with click-to-select points; returns the selection state (an attribute-style
    dict with ["selection"]["points"]), or whatever Streamlit returns when selection is unavailable."""
    fig.update_layout(clickmode="event+select")
    return st.plotly_chart(fig, width="stretch", key=key, on_select="rerun", selection_mode=("points",))
