"""The colour-blind-safe palette is held to measurable criteria, charts describe themselves, and no page can bypass either."""
import itertools
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from modules import palette as pal

ROOT = Path(__file__).resolve().parent.parent
OLD_TABLEAU = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]


# ------------------------------------------------------------------------------------------ the measuring tools

def test_contrast_and_colour_difference_behave_like_the_standards():
    assert pal.contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)
    assert pal.contrast_ratio("#777777", "#777777") == pytest.approx(1.0)
    assert pal.contrast_ratio("#0072B2", "#FFFFFF") == pal.contrast_ratio("#FFFFFF", "#0072B2")
    assert pal.delta_e("#123456", "#123456") == 0.0 and pal.delta_e("#000000", "#FFFFFF") == pytest.approx(100.0, abs=0.5)


def test_simulated_vision_is_the_identity_for_normal_sight_and_collapses_red_green_for_deuteranopes():
    assert np.allclose(pal.simulate("#336699", "normal"), pal._rgb("#336699"))
    assert pal.delta_e("#E45756", "#54A24B", "normal") > 30                       # clearly different to most people...
    assert pal.delta_e("#E45756", "#54A24B", "deuteranopia") < 5                  # ...and nearly the same colour to a deuteranope


# ------------------------------------------------------------------------------------------ the palette's guarantees

@pytest.mark.parametrize("background", [pal.LIGHT_BACKGROUND, pal.DARK_BACKGROUND])
def test_every_series_colour_is_visible_on_both_the_light_and_the_dark_background(background):
    for c in pal.CATEGORICAL + [pal.INK]:
        assert pal.contrast_ratio(c, background) >= 3.0, f"{c} on {background}"


def test_the_first_four_colours_stay_apart_for_every_kind_of_colour_blindness():
    assert pal.worst_pair_distance(pal.CATEGORICAL[:4]) >= 15.0


def test_the_old_palette_failed_the_same_criteria_which_is_why_it_was_replaced():
    assert pal.worst_pair_distance(OLD_TABLEAU[:4]) < 5.0
    assert any(min(pal.contrast_ratio(c, b) for b in (pal.LIGHT_BACKGROUND, pal.DARK_BACKGROUND)) < 3.0 for c in OLD_TABLEAU)


def test_the_status_pair_is_far_apart_where_red_and_green_were_not():
    assert pal.worst_pair_distance([pal.GOOD, pal.ATTENTION]) >= 60.0
    assert pal.worst_pair_distance(["#1E7E34", "#C0392B"]) < 20.0


def test_orange_is_last_because_it_is_the_one_colour_that_collapses_onto_another():
    assert pal.CATEGORICAL.index(pal.ORANGE) > 3
    assert pal.delta_e(pal.ORANGE, pal.VERMILLION, "deuteranopia") < 15


def test_the_fifth_series_onwards_differ_by_line_style_as_well_as_colour():
    styles = [pal.line_style(i) for i in range(12)]
    assert {d for _, d in styles[:4]} == {"solid"} and styles[4][1] != "solid"
    assert len(set(styles)) == len(styles)                                         # no two of the first twelve look the same


def test_badge_text_meets_wcag_aa_on_its_tint_and_the_three_statuses_are_distinguishable():
    for kind, (bg, fg) in pal.BADGE_COLORS.items():
        assert pal.contrast_ratio(bg, fg) >= 4.5, kind
    text = [pal.BADGE_COLORS[k][1] for k in ("pass", "warn", "fail")]
    old = ["#1E7E34", "#A85D00", "#C0392B"]
    assert pal.worst_pair_distance(text) > pal.worst_pair_distance(old) and pal.worst_pair_distance(text) >= 15.0


# ------------------------------------------------------------------------------------------ theming

def test_the_theme_sets_the_series_colours_and_the_template_for_each_mode():
    light, dark = pal.plotly_theme(False), pal.plotly_theme(True)
    assert light["template"] == "plotly_white" and dark["template"] == "plotly_dark" and light["colorway"] == pal.CATEGORICAL
    assert light["paper_bgcolor"] == "rgba(0,0,0,0)"


def test_applying_the_theme_is_idempotent_leaves_data_alone_and_can_be_switched_back():
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    pal.apply_theme(fig, True)
    once = fig.to_json()
    pal.apply_theme(fig, True)
    assert fig.to_json() == once
    pal.apply_theme(fig, False)
    assert fig.layout.template.layout.paper_bgcolor != "rgb(17,17,17)" and list(fig.data[0].y) == [4, 5, 6]


# ------------------------------------------------------------------------------------------ describing a chart

def test_a_line_chart_is_described_with_its_title_axes_series_and_range():
    fig = go.Figure([go.Scatter(x=[0, 1, 2], y=[1.0, 5.5, 3.0], name="speed", mode="lines"),
                     go.Scatter(x=[0, 1, 2], y=[0, 1, 2], name="data", mode="markers")])
    fig.update_layout(title="Speed over time", xaxis_title="t (s)", yaxis_title="v (m/s)")
    d = pal.describe_figure(fig)
    assert "Speed over time" in d and "x axis: t (s)" in d and "y axis: v (m/s)" in d
    assert "2 series" in d and "speed (line, 3 values, y from 1 to 5.5)" in d and "data (points" in d


def test_other_chart_kinds_are_described_sensibly():
    heat = pal.describe_figure(go.Figure(go.Heatmap(z=np.arange(12).reshape(3, 4), name="grid")))
    assert "heat map" in heat and "z from 0 to 11" in heat
    assert "bars" in pal.describe_figure(go.Figure(go.Bar(x=list("abc"), y=[3, 1, 2], name="counts")))
    assert "3-D surface" in pal.describe_figure(go.Figure(go.Surface(z=np.ones((3, 3)))))
    band = pal.describe_figure(go.Figure(go.Scatter(x=[0, 1, 1, 0], y=[0, 0, 1, 1], fill="toself", name="95% band")))
    assert "shaded band" in band


def test_panels_and_animation_are_mentioned():
    fig = make_subplots(rows=1, cols=2, subplot_titles=["left", "right"])
    fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]), row=1, col=1)
    fig.frames = [go.Frame(name="a"), go.Frame(name="b")]
    d = pal.describe_figure(fig)
    assert "2 panels: left; right" in d and "Animated: 2 frames" in d


def test_a_long_figure_is_summarised_not_listed_and_the_text_is_capped():
    fig = go.Figure([go.Scatter(x=[1, 2], y=[1, 2], name=f"series number {i}") for i in range(20)])
    d = pal.describe_figure(fig)
    assert "20 series" in d and "and 14 more" in d and len(d) <= 900


def test_odd_figures_never_break_the_description():
    assert "No data" in pal.describe_figure(go.Figure())
    assert pal.describe_figure(go.Figure(go.Scatter(x=["a", "b"], y=[None, None]))).startswith("Chart")
    assert pal.describe_figure(None) == "Chart (no description could be generated)."
    assert "values" in pal.describe_figure(go.Figure(go.Scatter(x=["a", "b", "c"], y=["x", "y", "z"], name="categorical")))


# ------------------------------------------------------------------------------------------ nothing can bypass it

def _source_files(*folders):
    return [p for f in folders for p in (ROOT / f).rglob("*.py") if "__pycache__" not in p.parts]


def test_no_page_draws_a_plotly_chart_except_through_show_figure():
    offenders = [str(p.relative_to(ROOT)) for p in _source_files("ui") if p.name != "a11y.py" and re.search(r"\bst\.plotly_chart\(", p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_the_old_palette_and_red_green_status_colours_are_gone_from_the_code():
    banned = OLD_TABLEAU + ["#C0392B", "#1E7E34", "#2E5EAA", "#2A9D8F"]
    rx = re.compile("|".join(re.escape(b) for b in banned), re.I)
    offenders = {str(p.relative_to(ROOT)): sorted(set(rx.findall(p.read_text(encoding="utf-8")))) for p in _source_files("modules", "ui")}
    assert {k: v for k, v in offenders.items() if v} == {}


def test_plot_modules_share_the_one_palette_instead_of_keeping_copies():
    for name in ("cc_plots", "sm_plots", "gr_plots", "quantum_plots", "discover_plots"):
        text = (ROOT / "modules" / f"{name}.py").read_text(encoding="utf-8")
        assert "CATEGORICAL" in text and not re.search(r"^_PAL(ETTE)? = \[", text, re.M), name


def test_data_marks_are_never_pure_black_which_vanishes_on_the_dark_theme():
    """White outlines round coloured markers and a white guide over a heat map are fine (the fill supplies the background); a black
    data point or curve is invisible on the dark theme, so the neutral ink is used instead."""
    rx = re.compile(r"""color\s*=\s*["'](?:black|#000000?)["']""", re.I)
    skip = {"plot_snapshot.py"}                    # matplotlib figures for exported reports: always drawn on white
    offenders = [str(p.relative_to(ROOT)) for p in _source_files("modules") if "plot" in p.name and p.name not in skip
                 and rx.search(p.read_text(encoding="utf-8"))]
    assert offenders == []
