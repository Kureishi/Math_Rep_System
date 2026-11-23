"""Chart text colour is set in Python, explicitly, in both themes.

Streamlit applies its own light-theme ink over a Plotly template's text colour, so a figure that only switches template ("plotly_dark") still has dark titles, tick labels and
legends on the dark page. modules/palette.plotly_theme therefore sets layout.font.color itself; this fails if that is ever dropped."""
import plotly.graph_objects as go

from modules import palette


def test_the_theme_sets_an_explicit_font_colour_for_each_theme():
    assert palette.plotly_theme(True)["font"]["color"] == palette.CHART_TEXT_DARK_MODE
    assert palette.plotly_theme(False)["font"]["color"] == palette.CHART_TEXT_LIGHT_MODE
    assert palette.CHART_TEXT_DARK_MODE != palette.CHART_TEXT_LIGHT_MODE


def test_a_themed_figure_carries_the_colour_and_switching_back_restores_it():
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
    palette.apply_theme(fig, True)
    assert fig.layout.font.color == palette.CHART_TEXT_DARK_MODE
    palette.apply_theme(fig, False)
    assert fig.layout.font.color == palette.CHART_TEXT_LIGHT_MODE


def test_show_figure_themes_the_chart_with_the_pages_theme(monkeypatch):
    import streamlit as st
    from ui import a11y
    drawn = []
    monkeypatch.setattr(st, "plotly_chart", lambda fig, **k: drawn.append(fig))
    monkeypatch.setattr(a11y, "announce", lambda *a, **k: None)
    monkeypatch.setattr(st, "session_state", {})
    monkeypatch.setattr(a11y, "is_dark", lambda: True)
    a11y.show_figure(go.Figure(go.Scatter(x=[0, 1], y=[0, 1])))
    assert drawn[-1].layout.font.color == palette.CHART_TEXT_DARK_MODE
