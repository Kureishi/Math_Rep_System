"""
The two charts on the Compare page, drawn from `modules.solve_compare.chart_rows()`.

Kept out of plotter.py (already very large) and out of the page so they can be tested without Streamlit.
"""
import plotly.graph_objects as go

_A_COLOR = "#4C78A8"
_B_COLOR = "#F58518"


def build_relative_change_chart(rows: list[tuple[str, float]]) -> go.Figure:
    """Horizontal bars: how far each answer moved from A to B, as a percentage of A. Targets whose A value
    is zero (a percentage of zero is meaningless) are not in `rows` to begin with."""
    fig = go.Figure()
    labels = [t for t, _ in rows]
    values = [v for _, v in rows]
    fig.add_trace(go.Bar(y=labels, x=values, orientation="h",
                         marker_color=[_B_COLOR if v >= 0 else _A_COLOR for v in values],
                         hovertemplate="%{y}: %{x:+.4g}%<extra></extra>"))
    fig.update_layout(title="Change in each answer, B relative to A",
                      xaxis_title="% change from A", yaxis=dict(autorange="reversed"),
                      showlegend=False)
    fig.add_vline(x=0, line_width=1, line_color="gray")
    return fig


def build_category_chart(rows: list[tuple[str, float | None, float | None]]) -> go.Figure:
    """Grouped bars: the fraction of each verification category's checks that passed, A beside B. A
    category only one side ran has a bar for that side only."""
    cats = [c for c, _, _ in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="A", x=cats, y=[a for _, a, _ in rows], marker_color=_A_COLOR))
    fig.add_trace(go.Bar(name="B", x=cats, y=[b for _, _, b in rows], marker_color=_B_COLOR))
    fig.update_layout(title="Verification checks passed, by category", barmode="group",
                      yaxis=dict(title="fraction passed", range=[0, 1.05], tickformat=".0%"))
    return fig
