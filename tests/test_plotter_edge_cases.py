"""
Targeted tests for modules/plotter.py branches tests/test_research_plots.py,
test_feasible_regions.py, and test_new_plots.py don't reach: build_surface_plot
(entirely untested before this file), plottable_free_symbols, the
solve()-fails/residual-fallback paths in build_plot and build_contour_plot,
build_fit_plot's fit-curve branch, build_vector_plot's empty-input case, and
build_feasible_region_plot's "skipped constraint" guard clauses.
"""
import numpy as np
import pytest
import sympy as sp

from modules.equation_engine import Equation, build_model
from modules.plotter import (
    build_contour_plot, build_feasible_region_plot, build_fit_plot, build_plot,
    build_surface_plot, build_vector_plot, plottable_free_symbols,
)


def _kinematics_model():
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "v_f", "known_value": "20", "unit": None},
            {"symbol": "v_i", "meaning": "v_i", "known_value": "8", "unit": None},
            {"symbol": "t", "meaning": "t", "known_value": "6", "unit": None},
            {"symbol": "a", "meaning": "a", "known_value": None, "unit": None},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f-v_i)/t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })


def _surface_model():
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "a", "meaning": "a", "known_value": None, "unit": None},
            {"symbol": "t", "meaning": "t", "known_value": None, "unit": None},
            {"symbol": "d", "meaning": "d", "known_value": None, "unit": None},
        ],
        "equations": [
            {"name": "disp", "kind": "equation", "expression": "Eq(d, 0.5*a*t**2)", "derivation": ""},
        ],
        "solve_for": ["d"], "assumptions": [],
    })


# ---------------------------------------------------------------- plottable_free_symbols

def test_plottable_free_symbols_returns_empty_for_unparsed_equation():
    eq = Equation(name="e0", raw_expression="garbage", derivation="", kind="equation", sympy_eq=None)
    assert plottable_free_symbols(eq, set()) == []


def test_plottable_free_symbols_excludes_fixed_and_sorts():
    eq = Equation(name="e0", raw_expression="a = b + c", derivation="", kind="equation",
                   sympy_eq=sp.Eq(sp.Symbol("a"), sp.Symbol("b") + sp.Symbol("c")))
    assert plottable_free_symbols(eq, {"b"}) == ["a", "c"]


# ---------------------------------------------------------------- build_plot

def test_build_plot_raises_on_unparsed_equation():
    eq = Equation(name="e0", raw_expression="garbage", derivation="", kind="equation", sympy_eq=None)
    with pytest.raises(ValueError, match="no parsed sympy expression"):
        build_plot(None, eq, "x", {}, (0, 1))


def test_build_plot_falls_back_to_residual_when_y_target_is_none():
    model = _kinematics_model()
    eq = model.equations[0]
    fig = build_plot(model, eq, "t", {"v_f": 20.0, "v_i": 8.0}, (1, 10), y_target=None)
    assert "residual" in fig.layout.yaxis.title.text
    assert len(fig.layout.shapes) >= 1  # the dashed zero-line


def test_build_plot_falls_back_to_residual_when_solve_raises(monkeypatch):
    """solve() itself raising (rather than just returning no solutions) is
    caught, not propagated -- the plot falls back to the residual view."""
    model = _kinematics_model()
    eq = model.equations[0]

    def fake_solve(*a, **kw):
        raise NotImplementedError("sympy couldn't solve this")
    monkeypatch.setattr("modules.plotter.sp.solve", fake_solve)
    fig = build_plot(model, eq, "t", {"v_f": 20.0, "v_i": 8.0}, (1, 10), y_target="a")
    assert "residual" in fig.layout.yaxis.title.text


def test_build_plot_falls_back_to_residual_when_y_target_unsolvable():
    """y_target requested but sp.solve() returns no solutions for it (here:
    y_target equal to a symbol not actually present as a distinct unknown
    in this equation after substitution) -- falls back cleanly rather than
    crashing on an empty `solved` list."""
    model = _kinematics_model()
    eq = model.equations[0]
    fig = build_plot(model, eq, "t", {"v_f": 20.0, "v_i": 8.0}, (1, 10), y_target="nonexistent_symbol")
    assert "residual" in fig.layout.yaxis.title.text


# ---------------------------------------------------------------- build_surface_plot

def test_build_surface_plot_raises_on_unparsed_equation():
    eq = Equation(name="e0", raw_expression="garbage", derivation="", kind="equation", sympy_eq=None)
    with pytest.raises(ValueError, match="no parsed sympy expression"):
        build_surface_plot(eq, "x", "y", {}, (0, 1), (0, 1))


def test_build_surface_plot_solves_for_z_target():
    model = _surface_model()
    eq = model.equations[0]
    fig = build_surface_plot(eq, "a", "t", {}, (1, 5), (1, 5), z_target="d", resolution=10)
    assert fig.data[0].type == "surface"
    assert fig.layout.scene.zaxis.title.text == "d"
    Z = np.asarray(fig.data[0].z)
    # d = 0.5*a*t**2 -- spot-check the corner (a=1, t=1) -> d=0.5
    assert Z.shape == (10, 10)
    assert np.isclose(Z[0, 0], 0.5, atol=1e-6)


def test_build_surface_plot_falls_back_to_residual_when_z_target_unsolvable():
    """z_target requested but not solvable/present -- falls back to the
    residual surface. `d` must be pinned via param_values here since it's
    the third free symbol in the equation and isn't being plotted or solved
    for; without a value it would be left as a free symbol in the residual."""
    model = _surface_model()
    eq = model.equations[0]
    fig = build_surface_plot(eq, "a", "t", {"d": 3.0}, (1, 5), (1, 5),
                              z_target="nonexistent", resolution=10)
    assert "residual" in fig.layout.scene.zaxis.title.text


def test_build_surface_plot_falls_back_to_residual_when_z_target_is_none():
    model = _surface_model()
    eq = model.equations[0]
    fig = build_surface_plot(eq, "a", "t", {"d": 3.0}, (1, 5), (1, 5), z_target=None, resolution=10)
    assert "residual" in fig.layout.scene.zaxis.title.text
    assert fig.data[0].colorscale is not None


def test_build_surface_plot_falls_back_when_solve_raises(monkeypatch):
    model = _surface_model()
    eq = model.equations[0]

    def fake_solve(*a, **kw):
        raise NotImplementedError("nope")
    monkeypatch.setattr("modules.plotter.sp.solve", fake_solve)
    fig = build_surface_plot(eq, "a", "t", {"d": 3.0}, (1, 5), (1, 5), z_target="d", resolution=10)
    assert "residual" in fig.layout.scene.zaxis.title.text


# ---------------------------------------------------------------- build_contour_plot

def test_build_contour_plot_raises_on_unparsed_equation():
    eq = Equation(name="e0", raw_expression="garbage", derivation="", kind="equation", sympy_eq=None)
    with pytest.raises(ValueError, match="no parsed sympy expression"):
        build_contour_plot(eq, "x", "y", {}, (0, 1), (0, 1))


def test_build_contour_plot_falls_back_when_solve_raises(monkeypatch):
    model = _surface_model()
    eq = model.equations[0]

    def fake_solve(*a, **kw):
        raise NotImplementedError("nope")
    monkeypatch.setattr("modules.plotter.sp.solve", fake_solve)
    fig = build_contour_plot(eq, "a", "t", {"d": 3.0}, (1, 5), (1, 5), z_target="d", resolution=10)
    assert fig.data[0].type == "contour"


# ---------------------------------------------------------------- build_fit_plot

def test_build_fit_plot_with_no_fit_expr_only_plots_data():
    fig = build_fit_plot([1, 2, 3], [4, 5, 6], None)
    assert len(fig.data) == 1
    assert fig.data[0].mode == "markers"


def test_build_fit_plot_adds_the_fit_curve():
    x = sp.Symbol("x")
    fig = build_fit_plot([1, 2, 3], [2, 4, 6], 2 * x)
    assert len(fig.data) == 2
    assert fig.data[1].name == "fit"


def test_build_fit_plot_x_log_uses_geomspace():
    x = sp.Symbol("x")
    fig = build_fit_plot([1, 2, 3], [2, 4, 6], 2 * x, x_log=True)
    assert fig.layout.xaxis.type == "log"
    assert len(fig.data) == 2


def test_build_fit_plot_constant_x_range_still_produces_a_curve():
    """hi == lo (a single distinct x value) -- the `pad = ... if hi > lo
    else 1.0` branch, so the grid doesn't collapse to a single point."""
    x = sp.Symbol("x")
    fig = build_fit_plot([5, 5, 5], [1, 1, 1], 2 * x)
    assert len(fig.data) == 2


def test_build_fit_plot_swallows_a_bad_fit_expression():
    """A fit_expr referencing a symbol OTHER than x (a mistake upstream) --
    lambdify(x, ...) happily builds a function that just returns that other
    symbol unevaluated when called, and casting THAT to a float numpy array
    is what actually raises; caught, leaving just the data scatter rather
    than propagating."""
    y = sp.Symbol("y")  # not x -- the fit curve can never depend on it here
    fig = build_fit_plot([1, 2, 3], [4, 5, 6], y)
    assert len(fig.data) == 1


# ---------------------------------------------------------------- build_vector_plot

def test_build_vector_plot_empty_list_returns_blank_figure():
    fig = build_vector_plot([])
    assert list(fig.data) == []


# ---------------------------------------------------------------- build_feasible_region_plot guard clauses

def test_feasible_region_skips_unparsed_constraint():
    unparsed = Equation(name="bad", raw_expression="garbage", derivation="", kind="inequality", sympy_eq=None)
    fig = build_feasible_region_plot([unparsed], "x", "y", {}, (-2, 12), (-2, 12), resolution=10)
    assert fig.data[0].type == "heatmap"


def test_feasible_region_skips_constraint_with_unsubstituted_symbol():
    """A constraint that still contains a third symbol (z) not eliminated
    by param_values -- can't be evaluated over the (x, y) grid, so it's
    reported as skipped in the title rather than crashing lambdify."""
    x, y, z = sp.symbols("x y z")
    c = Equation(name="has_z", raw_expression="x + z <= 5", derivation="", kind="inequality",
                 sympy_eq=sp.Le(x + z, 5))
    fig = build_feasible_region_plot([c], "x", "y", {}, (-2, 12), (-2, 12), resolution=10)
    assert "has_z" in fig.layout.title.text
    assert "skipped" in fig.layout.title.text


def test_feasible_region_skips_constraint_that_fails_to_lambdify(monkeypatch):
    x, y = sp.symbols("x y")
    c = Equation(name="bad_lambdify", raw_expression="x <= 5", derivation="", kind="inequality",
                 sympy_eq=sp.Le(x, 5))

    def fake_lambdify(*a, **kw):
        raise TypeError("cannot lambdify this")
    monkeypatch.setattr("modules.plotter.sp.lambdify", fake_lambdify)
    fig = build_feasible_region_plot([c], "x", "y", {}, (-2, 12), (-2, 12), resolution=10)
    assert "bad_lambdify" in fig.layout.title.text
    assert "skipped" in fig.layout.title.text
