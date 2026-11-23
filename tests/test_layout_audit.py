"""The static layout audit (modules/layout_audit.py) and what it found: panel gaps in pixels, and subplot titles that do not fit a narrow panel."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules import layout_audit as la
from modules import waves as wv
from modules import waves_plots as wp


def _stacked(spacing, height=450):
    fig = make_subplots(rows=2, cols=1, vertical_spacing=spacing, subplot_titles=("upper", "lower"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=1, col=1)
    fig.update_layout(height=height)
    return fig


def test_a_gap_too_small_for_the_text_in_it_is_flagged_and_a_roomy_one_is_not():
    assert any(i.kind == "stacked-gap" for i in la.audit(_stacked(0.05), 1100))
    assert la.audit(_stacked(0.25), 1100) == []


def test_side_by_side_panels_need_room_for_the_right_hand_y_axis():
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.02)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=1, col=2)
    assert any(i.kind == "side-gap" for i in la.audit(fig, 700))
    fig.update_layout(width=None)
    wide = make_subplots(rows=1, cols=2, horizontal_spacing=0.15)
    assert [i for i in la.audit(wide, 1100) if i.kind == "side-gap"] == []


def test_a_long_subplot_title_is_wrapped_to_fit_and_wrapping_is_idempotent():
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.1, subplot_titles=("short", "a rather long subplot title that cannot fit", "Ehrenfest: ⟨x⟩ and ∫⟨p⟩/m dt"))
    for c in (1, 2, 3):
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=1, col=c)
    assert any(i.kind == "subplot-title" for i in la.audit(fig, 700))
    top = fig.layout.margin.t
    n = la.fit_subplot_titles(fig)
    assert n >= 1 and [i for i in la.audit(fig, 700) if i.kind == "subplot-title"] == []
    assert "<br>" in fig.layout.annotations[1].text and fig.layout.annotations[0].text == "short"
    assert (fig.layout.margin.t or 0) > (top if top is not None else 0) - 1
    assert la.fit_subplot_titles(fig) == 0                                      # nothing left to wrap
    margin = fig.layout.margin.t
    la.fit_subplot_titles(fig)
    assert fig.layout.margin.t == margin


def test_a_single_panel_figure_is_never_touched():
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
    assert la.audit(fig, 400) == [] and la.fit_subplot_titles(fig) == 0


def test_the_diffraction_figure_has_room_between_its_two_panels_at_every_normal_width():
    """Regression: with 0.14 spacing the 53 px between the pattern and the aperture panel held about 66 px of tick labels, axis title and subplot title."""
    fig = wp.diffraction_figure(wv.diffraction_1d("double slit", 550e-9, 1.5, 50e-6, 250e-6))
    la.fit_subplot_titles(fig)
    for name in ("1500 px window", "800 px window, sidebar closed"):
        assert la.audit(fig, la.WIDTHS[name]) == [], name


def test_show_figure_wraps_titles_before_drawing(monkeypatch):
    import streamlit as st
    from ui import a11y
    drawn = []
    monkeypatch.setattr(st, "plotly_chart", lambda fig, **k: drawn.append(fig))
    monkeypatch.setattr(a11y, "announce", lambda *a, **k: None)
    monkeypatch.setattr(st, "session_state", {})
    fig = make_subplots(rows=1, cols=3, subplot_titles=("one", "a rather long subplot title that cannot fit", "three"))
    for c in (1, 2, 3):
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1]), row=1, col=c)
    a11y.show_figure(fig)
    assert "<br>" in drawn[-1].layout.annotations[1].text
