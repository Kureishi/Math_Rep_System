"""Branch-coverage tests for modules/plot_snapshot.py: the fallback, guard
and error paths the happy-path suites don't reach (a missing parsed
equation, sp.solve failing and falling back to plotting the residual, a
constant expression whose lambdified form returns a scalar rather than an
array, vector plots of every dimension, etc.).

Every test asserts on real output -- valid PNG/GIF magic bytes -- rather than
merely "didn't raise": a figure function that silently returned empty bytes
would otherwise pass. Byte validity is NOT the same as correct rendering
(see the export notes in this repo's README); these tests guard the control
flow, and the existing visual-inspection step guards the pixels.
"""
import numpy as np
import pytest
import sympy as sp

import modules.plot_snapshot as ps
from modules.equation_engine import Equation

PNG = b"\x89PNG\r\n\x1a\n"
GIF = b"GIF8"

a, t, x, y, z = sp.symbols("a t x y z")


def _eq(sympy_eq, name="eq1"):
    return Equation(name=name, raw_expression=str(sympy_eq), derivation="", kind="equation", sympy_eq=sympy_eq)


LINEAR = _eq(sp.Eq(z, 2 * x + 3 * y))          # solvable for z, x or y
CONSTANT = _eq(sp.Eq(z, sp.Integer(5)))        # solved z = 5 -> lambdify returns a bare scalar
NO_EQ = Equation(name="broken", raw_expression="Eq(", derivation="", kind="equation", sympy_eq=None)


@pytest.fixture
def solve_raises(monkeypatch):
    def boom(*a_, **k):
        raise NotImplementedError("sympy can't solve this")
    monkeypatch.setattr(ps.sp, "solve", boom)


# ------------------------------------------------------------------ line plot

def test_line_plot_without_parsed_equation_raises():
    with pytest.raises(ValueError, match="no parsed sympy expression"):
        ps.snapshot_line_plot(NO_EQ, "x", {}, (0, 1))


def test_line_plot_falls_back_to_residual_when_solve_fails(solve_raises):
    # the residual fallback needs a value for every non-axis symbol (z included,
    # since it can no longer be solved for), exactly as the UI's parameter sliders supply
    assert ps.snapshot_line_plot(LINEAR, "x", {"y": 1.0, "z": 0.0}, (0, 5), y_target="z")[:8] == PNG


def test_line_plot_without_target_plots_residual():
    assert ps.snapshot_line_plot(LINEAR, "x", {"y": 1.0, "z": 0.0}, (0, 5))[:8] == PNG


def test_line_plot_with_target_equal_to_x_plots_residual_not_a_solved_curve():
    assert ps.snapshot_line_plot(LINEAR, "x", {"y": 1.0, "z": 0.0}, (0, 5), y_target="x")[:8] == PNG


def test_line_plot_log_axes_and_vector_formats():
    png = ps.snapshot_line_plot(LINEAR, "x", {"y": 1.0}, (0, 100), y_target="z", x_log=True, y_log=True)
    assert png[:8] == PNG
    assert ps.snapshot_line_plot(LINEAR, "x", {"y": 1.0}, (0, 5), y_target="z", fmt="svg")[:5] in (b"<?xml", b"<svg ")
    assert ps.snapshot_line_plot(LINEAR, "x", {"y": 1.0}, (0, 5), y_target="z", fmt="pdf")[:4] == b"%PDF"


def test_finish_rejects_unknown_format():
    with pytest.raises(ValueError, match="Unsupported format"):
        ps.snapshot_line_plot(LINEAR, "x", {"y": 1.0}, (0, 5), y_target="z", fmt="bmp")


# ------------------------------------------------------------------ surface plot

def test_surface_plot_without_parsed_equation_raises():
    with pytest.raises(ValueError, match="no parsed sympy expression"):
        ps.snapshot_surface_plot(NO_EQ, "x", "y", {}, (0, 1), (0, 1))


def test_surface_plot_falls_back_to_residual_when_solve_fails(solve_raises):
    png = ps.snapshot_surface_plot(LINEAR, "x", "y", {"z": 0.0}, (0, 3), (0, 3), z_target="z", resolution=12)
    assert png[:8] == PNG


def test_surface_plot_without_target_plots_residual():
    assert ps.snapshot_surface_plot(LINEAR, "x", "y", {"z": 1.0}, (0, 3), (0, 3), resolution=12)[:8] == PNG


def test_surface_plot_of_constant_solution_broadcasts_scalar_to_grid():
    # z = 5 lambdifies to a function returning the scalar 5, not a 12x12 array
    assert ps.snapshot_surface_plot(CONSTANT, "x", "y", {}, (0, 3), (0, 3), z_target="z", resolution=12)[:8] == PNG


def test_surface_plot_of_constant_residual_broadcasts_scalar_to_grid():
    # residual z - 5 with z=3 substituted is the constant -2
    assert ps.snapshot_surface_plot(CONSTANT, "x", "y", {"z": 3.0}, (0, 3), (0, 3), resolution=12)[:8] == PNG


# ------------------------------------------------------------------ feasible region

def test_feasible_region_skips_unparsed_out_of_scope_and_unevaluable_constraints(monkeypatch):
    good = _eq(sp.Ge(x + y, 1), "sum")
    other_symbol = _eq(sp.Ge(x + a, 1), "needs a")      # 'a' has no value -> can't be drawn in x/y space
    png = ps.snapshot_feasible_region([NO_EQ, other_symbol, good], "x", "y", {}, (-2, 4), (-2, 4), resolution=30)
    assert png[:8] == PNG

    real = ps.sp.lambdify

    def flaky(*args, **kw):
        raise TypeError("cannot lambdify")
    monkeypatch.setattr(ps.sp, "lambdify", flaky)
    # every constraint now fails to lambdify -> all skipped -> whole plane shaded, still a valid image
    assert ps.snapshot_feasible_region([good], "x", "y", {}, (-2, 4), (-2, 4), resolution=30)[:8] == PNG
    monkeypatch.setattr(ps.sp, "lambdify", real)


# ------------------------------------------------------------------ recurrence / vector / fit

def test_recurrence_plot_draws_discrete_points():
    n = sp.Symbol("n", integer=True)
    assert ps.snapshot_recurrence_plot("a", n, 2 ** n + 1, (0, 10))[:8] == PNG


def test_vector_plot_empty_2d_3d_and_unsupported_dimension():
    assert ps.snapshot_vector_plot([])[:8] == PNG
    assert ps.snapshot_vector_plot([("F1", [1.0, 2.0]), ("F2", [-2.0, 1.0])])[:8] == PNG
    assert ps.snapshot_vector_plot([("r", [1.0, 2.0, 3.0]), ("s", [-1.0, 0.5, 2.0])])[:8] == PNG
    with pytest.raises(ValueError, match="only supports 2D or 3D"):
        ps.snapshot_vector_plot([("w", [1.0, 2.0, 3.0, 4.0])])


def test_fit_plot_variants():
    xs, ys = [1.0, 2.0, 3.0, 4.0], [2.1, 3.9, 6.2, 7.8]
    assert ps.snapshot_fit_plot(xs, ys, None)[:8] == PNG                               # data only
    assert ps.snapshot_fit_plot(xs, ys, 2 * x)[:8] == PNG                              # with fit line
    assert ps.snapshot_fit_plot(xs, ys, 2 * x, x_log=True, y_log=True)[:8] == PNG      # log axes
    assert ps.snapshot_fit_plot([3.0, 3.0], [1.0, 2.0], 2 * x)[:8] == PNG              # zero x-span -> pad fallback
    assert ps.snapshot_fit_plot(xs, ys, sp.Integer(4))[:8] == PNG                      # constant fit broadcasts


def test_fit_plot_survives_a_fit_expression_that_cannot_be_evaluated():
    # 'q' is not the plotting variable x, so evaluating the lambdified fit raises
    # NameError -- the scatter must still be produced rather than the export failing
    assert ps.snapshot_fit_plot([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], sp.Symbol("q") * 2)[:8] == PNG


# ------------------------------------------------------------------ contour / histogram / overlay

def test_contour_plot_without_parsed_equation_raises():
    with pytest.raises(ValueError, match="no parsed sympy expression"):
        ps.snapshot_contour_plot(NO_EQ, "x", "y", {}, (0, 1), (0, 1))


def test_contour_plot_falls_back_to_residual_when_solve_fails(solve_raises):
    assert ps.snapshot_contour_plot(LINEAR, "x", "y", {"z": 0.0}, (0, 3), (0, 3), z_target="z",
                                    resolution=15)[:8] == PNG


def test_contour_plot_paths_without_solve_failure():
    kw = dict(resolution=15)
    assert ps.snapshot_contour_plot(LINEAR, "x", "y", {}, (0, 3), (0, 3), z_target="z", **kw)[:8] == PNG
    assert ps.snapshot_contour_plot(LINEAR, "x", "y", {"z": 1.0}, (0, 3), (0, 3), **kw)[:8] == PNG
    # a non-constant field is required for contour levels, so use a tilted plane for the scalar-broadcast
    # branches' neighbours and a constant only where contour() tolerates it
    plane = _eq(sp.Eq(z, x * y))
    assert ps.snapshot_contour_plot(plane, "x", "y", {}, (0, 3), (0, 3), z_target="z", **kw)[:8] == PNG


def test_contour_plot_of_constant_field_broadcasts_scalar_to_grid():
    # both the solved-target path (z = 5) and the residual path (z - 5, z=3 -> -2)
    # lambdify to a bare scalar; neither may crash on the shape mismatch, and a
    # flat field (no contour levels to draw) must still yield a valid image
    assert ps.snapshot_contour_plot(CONSTANT, "x", "y", {}, (0, 3), (0, 3), z_target="z", resolution=15)[:8] == PNG
    assert ps.snapshot_contour_plot(CONSTANT, "x", "y", {"z": 3.0}, (0, 3), (0, 3), resolution=15)[:8] == PNG


def test_histogram_with_and_without_summary_lines():
    samples = list(np.random.default_rng(0).normal(10, 1, 400))
    assert ps.snapshot_histogram_plot(samples, "a")[:8] == PNG
    assert ps.snapshot_histogram_plot(samples, "a", mean=10.0, p5=8.4, p95=11.6)[:8] == PNG
    assert ps.snapshot_histogram_plot(samples, "a", p95=11.6)[:8] == PNG


def test_overlay_plot_with_and_without_title():
    series = [{"x": [0, 1, 2], "y": [0, 1, 4], "name": "sq"}, {"x": [0, 1, 2], "y": [0, 1, 2]}]
    assert ps.snapshot_overlay_plot(series)[:8] == PNG
    assert ps.snapshot_overlay_plot(series, title="Comparison")[:8] == PNG


def test_spread_plot_renders():
    assert ps.snapshot_spread_plot([1.0, 1.1, 0.9, 1.05], "v")[:8] == PNG


# ------------------------------------------------------------------ rotating GIF

def test_rotating_gif_without_parsed_equation_raises():
    with pytest.raises(ValueError, match="no parsed sympy expression"):
        ps.snapshot_rotating_surface_gif(NO_EQ, "x", "y", {}, (0, 1), (0, 1))


def test_rotating_gif_solved_target_and_residual_and_solve_failure(monkeypatch):
    kw = dict(resolution=10, n_frames=3, fps=3)
    g1 = ps.snapshot_rotating_surface_gif(LINEAR, "x", "y", {}, (0, 3), (0, 3), z_target="z", **kw)
    g2 = ps.snapshot_rotating_surface_gif(LINEAR, "x", "y", {"z": 1.0}, (0, 3), (0, 3), **kw)
    assert g1[:4] == g2[:4] == GIF

    def boom(*a_, **k):
        raise NotImplementedError
    monkeypatch.setattr(ps.sp, "solve", boom)
    g3 = ps.snapshot_rotating_surface_gif(LINEAR, "x", "y", {"z": 1.0}, (0, 3), (0, 3), z_target="z", **kw)
    assert g3[:4] == GIF


def test_rotating_gif_constant_surface_broadcasts_scalar():
    g = ps.snapshot_rotating_surface_gif(CONSTANT, "x", "y", {}, (0, 3), (0, 3), z_target="z",
                                         resolution=10, n_frames=3, fps=3)
    assert g[:4] == GIF
