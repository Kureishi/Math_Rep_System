"""Units wherever a value of a variable is shown and the unit is known: results lines, chart axis titles and tooltips."""
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from modules import plotter
from modules.latex_text import axis_label, ratio_unit, unit_of, with_unit


def _model():
    from modules.equation_engine import build_model
    return build_model({
        "problem_domain": "geometry", "problem_type": "algebraic",
        "variables": [{"symbol": "L", "meaning": "length", "known_value": "12.4", "unit": "m", "uncertainty": "0.2"},
                      {"symbol": "W", "meaning": "width", "known_value": "5.1", "unit": "m", "uncertainty": "0.1"},
                      {"symbol": "A", "meaning": "area", "known_value": None, "unit": "m^2"}],
        "equations": [{"name": "area", "kind": "equation", "expression": "Eq(A, L*W)", "derivation": ""}],
        "solve_for": ["A"], "assumptions": [],
    })


# ------------------------------------------------------------------------------------------ helpers

def test_unit_helpers():
    model = _model()
    assert unit_of(model, "A") == "m²" and unit_of(model, "W") == "m" and unit_of(model, "nope") == ""
    assert with_unit("63.2", "m²") == "63.2 m²" and with_unit("63.2", "") == "63.2"
    assert axis_label("A", "m²") == "A (m²)" and axis_label("A", "") == "A"
    assert ratio_unit("m²", "m") == "m²/m" and ratio_unit("J", "kg·K") == "J/(kg·K)" and ratio_unit("m", "") == "" and ratio_unit("m", "m") == ""


# ------------------------------------------------------------------------------------------ chart axes and tooltips

def test_monte_carlo_plots_title_their_axes_with_the_unit():
    samples = list(np.random.default_rng(0).normal(63.2, 3.5, 300))
    hist = plotter.build_histogram_plot(samples, "A", mean=63.2, p5=57.0, p95=69.0, unit="m²")
    assert hist.layout.xaxis.title.text == "A (m²)"
    conv = plotter.build_monte_carlo_convergence_plot(samples, "A", unit="m²")
    assert conv.layout.yaxis.title.text == "A (m²)"
    assert "running mean = %{y:.6g} m²" in conv.frames[-1].data[1].hovertemplate and "running std = %{customdata[0]:.4g} m²" in conv.frames[-1].data[1].hovertemplate
    assert plotter.build_histogram_plot(samples, "A").layout.xaxis.title.text == "A"          # no unit, no brackets


def test_sweep_tornado_and_heatmap_titles_carry_units():
    heat = plotter.build_sweep_heatmap([1, 2], [3, 4], [[1, 2], [3, 4]], "L", "W", "A", x_unit="m", y_unit="m", target_unit="m²")
    assert (heat.layout.xaxis.title.text, heat.layout.yaxis.title.text, heat.data[0].colorbar.title.text) == ("L (m)", "W (m)", "A (m²)")
    from types import SimpleNamespace as NS
    entries = [NS(symbol="L", low_target=1.0, high_target=2.0), NS(symbol="W", low_target=1.5, high_target=1.7)]
    tornado = plotter.build_tornado_chart(entries, unit="m²")
    assert "(m²)" in tornado.layout.xaxis.title.text and tornado.data[0].hovertext[0].endswith("m²")
    sweep = NS(values=[1, 2], target_values=[1, 4], nominal_target=None, nominal_input=None, symbol="W")
    fig = plotter.build_sweep_chart(sweep, x_unit="m", y_unit="m²")
    assert (fig.layout.xaxis.title.text, fig.layout.yaxis.title.text) == ("W (m)", "target value (m²)")


# ------------------------------------------------------------------------------------------ results lines

def _error_page():
    from tests.test_units_everywhere import _model
    from ui.results.steps import render_analytic_error
    model = _model()
    render_analytic_error(model, [v for v in model.variables if v.known_value is not None], "A")


def _interval_page():
    from tests.test_units_everywhere import _model
    from ui.results.steps import render_interval_bounds
    model = _model()
    render_interval_bounds(model, [v for v in model.variables if v.known_value is not None], "A")


def _text(at):
    return " ".join([*(m.value for m in at.markdown), *(s.value for s in at.success), *(c.value for c in at.caption)])


def test_analytic_error_propagation_shows_units_on_the_value_and_the_partials():
    at = AppTest.from_function(_error_page, default_timeout=60)
    at.run()
    assert not at.exception
    at.multiselect(key="ep_vars_A").select("L").select("W").run()
    next(b for b in at.button if b.key == "ep_run_A").click().run()
    assert not at.exception
    text = _text(at)
    assert "A = 63.24 ± " in text and "m²**" in text                         # value ± std, then the unit
    assert "∂A/∂L = 5.1 m²/m" in text or "∂A/∂L = 5.1 m" in text


def test_interval_bounds_show_the_unit_on_the_range_and_the_width():
    at = AppTest.from_function(_interval_page, default_timeout=60)
    at.run()
    assert not at.exception
    at.multiselect(key="iv_vars_A").select("L").select("W").run()
    next(b for b in at.button if b.key == "iv_run_A").click().run()
    assert not at.exception
    text = _text(at)
    assert "guaranteed to fall within" in text and "m²**" in text and "(width " in text and text.count("m²") >= 2
