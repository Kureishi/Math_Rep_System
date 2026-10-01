"""Regression tests for modules/plot_params.py and the seven plot functions
that use it (plotter.build_plot/build_surface_plot/build_contour_plot and
plot_snapshot.snapshot_line/surface/contour/rotating_gif).

Two real bugs, both found by reading how the UI calls these functions:

1. The UI builds a slider for EVERY non-axis free symbol -- the Y/Z target
   included (its selectbox comes after the sliders) -- so param_values
   contains the target. The plot functions substituted all of param_values
   BEFORE solving for the target, which turned the equation into a constant
   truth value, so sp.solve returned [] and the plot silently showed the
   residual instead. The "Y-axis target" / "Z-axis target" selector was
   effectively a no-op for any target that appears in the equation.

2. With a symbol missing from param_values, the residual fallback produced
   an object array of unevaluated sympy expressions: Plotly was handed
   garbage, and the matplotlib exports crashed with "Cannot convert
   expression to float". Now a ValueError names the missing symbol(s).
"""
import numpy as np
import pytest
import sympy as sp

import modules.plot_snapshot as ps
import modules.plotter as pl
from modules.equation_engine import Equation
from modules.plot_params import residual_expression, solve_for_target

PNG = b"\x89PNG\r\n\x1a\n"
t, u, a, v, d = sp.symbols("t u a v d")

VELOCITY = Equation(name="velocity", raw_expression="Eq(v, u + a*t)", derivation="", kind="equation",
                    sympy_eq=sp.Eq(v, u + a * t))
DISPLACEMENT = Equation(name="displacement", raw_expression="Eq(d, u*t + a*t**2/2)", derivation="",
                        kind="equation", sympy_eq=sp.Eq(d, u * t + a * t ** 2 / 2))

# exactly what the UI hands the plot functions: a value for every non-x symbol, target INCLUDED
UI_PARAMS_LINE = {"u": 8.0, "a": 2.0, "v": 999.0}


# ------------------------------------------------------------------ the helpers

def test_solve_for_target_ignores_the_targets_own_param_value():
    expr = solve_for_target(VELOCITY, "v", UI_PARAMS_LINE)      # v=999 must NOT be substituted
    assert expr is not None
    assert sp.simplify(expr - (8 + 2 * t)) == 0


def test_solve_for_target_returns_none_when_unsolvable_or_absent(monkeypatch):
    assert solve_for_target(VELOCITY, "not_in_equation", {"u": 1.0}) is None

    def boom(*a_, **k):
        raise NotImplementedError
    monkeypatch.setattr(sp, "solve", boom)
    assert solve_for_target(VELOCITY, "v", {"u": 1.0, "a": 1.0}) is None


def test_solve_for_target_skips_solutions_that_do_not_contain_the_target(monkeypatch):
    monkeypatch.setattr(sp, "solve", lambda *a_, **k: [{sp.Symbol("zzz"): 1}])
    assert solve_for_target(VELOCITY, "v", {}) is None


def test_residual_expression_substitutes_every_param_including_the_target():
    r = residual_expression(VELOCITY, {"u": 8.0, "a": 2.0, "v": 20.0}, {"t"})
    assert sp.simplify(r - (20 - 8 - 2 * t)) == 0


def test_residual_expression_names_every_missing_symbol():
    with pytest.raises(ValueError) as exc:
        residual_expression(VELOCITY, {"u": 8.0}, {"t"})
    assert "'velocity'" in str(exc.value)
    assert "a, v" in str(exc.value)                               # sorted, both named


# ------------------------------------------------------------------ bug 1: target selector now works (live Plotly plots)

def test_build_plot_solves_for_the_target_even_though_ui_params_include_it():
    fig = pl.build_plot(None, VELOCITY, "t", UI_PARAMS_LINE, (0, 10), y_target="v")
    assert fig.layout.yaxis.title.text == "v"                      # the SOLVED curve, not "...residual"
    ys = np.asarray(fig.data[0].y, dtype=float)
    assert ys[0] == pytest.approx(8.0)                             # v(t=0) = u
    assert ys[-1] == pytest.approx(8.0 + 2.0 * 10)                 # v(t=10) = u + a*10 -- NOT 999-based


def test_build_surface_plot_solves_for_the_target_even_though_ui_params_include_it():
    params = {"u": 8.0, "d": 12345.0}                              # d is the z target, with a slider value
    fig = pl.build_surface_plot(DISPLACEMENT, "a", "t", params, (0, 4), (0, 4), z_target="d", resolution=5)
    assert fig.layout.scene.zaxis.title.text == "d"
    z = np.asarray(fig.data[0].z, dtype=float)
    assert z[-1][-1] == pytest.approx(8 * 4 + 4 * 4 ** 2 / 2)      # d(a=4, t=4) = u*t + a*t^2/2 = 64


def test_build_contour_plot_solves_for_the_target_even_though_ui_params_include_it():
    params = {"u": 8.0, "d": 12345.0}
    fig = pl.build_contour_plot(DISPLACEMENT, "a", "t", params, (0, 4), (0, 4), z_target="d", resolution=5)
    assert "residual" not in (fig.layout.title.text or "")
    assert np.asarray(fig.data[0].z, dtype=float)[-1][-1] == pytest.approx(64.0)


# ------------------------------------------------------------------ bug 1: ... and in the exported snapshots

def test_snapshot_line_plot_solves_for_the_target_even_though_ui_params_include_it(monkeypatch):
    # Capture what actually gets plotted. If the solve were defeated, the
    # residual path would call axhline(0) -- the solved curve never does.
    drew_zero_line = []
    import matplotlib.axes
    real = matplotlib.axes.Axes.axhline
    monkeypatch.setattr(matplotlib.axes.Axes, "axhline",
                        lambda self, *a_, **k: (drew_zero_line.append(True), real(self, *a_, **k))[1])
    assert ps.snapshot_line_plot(VELOCITY, "t", UI_PARAMS_LINE, (0, 10), y_target="v")[:8] == PNG
    assert drew_zero_line == []


def test_snapshot_surface_and_contour_and_gif_use_the_solved_target(monkeypatch):
    seen = []
    real_solve = ps.solve_for_target
    monkeypatch.setattr(ps, "solve_for_target",
                        lambda *args, **kw: (seen.append(real_solve(*args, **kw)), seen[-1])[1])
    params = {"u": 8.0, "d": 12345.0}
    assert ps.snapshot_surface_plot(DISPLACEMENT, "a", "t", params, (0, 4), (0, 4), z_target="d",
                                    resolution=8)[:8] == PNG
    assert ps.snapshot_contour_plot(DISPLACEMENT, "a", "t", params, (0, 4), (0, 4), z_target="d",
                                    resolution=8)[:8] == PNG
    assert ps.snapshot_rotating_surface_gif(DISPLACEMENT, "a", "t", params, (0, 4), (0, 4), z_target="d",
                                            resolution=8, n_frames=2, fps=2)[:4] == b"GIF8"
    assert len(seen) == 3 and all(e is not None for e in seen)     # all three really solved for d


# ------------------------------------------------------------------ bug 2: missing value -> clear error, not garbage

@pytest.mark.parametrize("call", [
    lambda: pl.build_plot(None, VELOCITY, "t", {"u": 8.0}, (0, 10)),
    lambda: pl.build_surface_plot(DISPLACEMENT, "a", "t", {}, (0, 4), (0, 4), resolution=5),
    lambda: pl.build_contour_plot(DISPLACEMENT, "a", "t", {}, (0, 4), (0, 4), resolution=5),
    lambda: ps.snapshot_line_plot(VELOCITY, "t", {"u": 8.0}, (0, 10)),
    lambda: ps.snapshot_surface_plot(DISPLACEMENT, "a", "t", {}, (0, 4), (0, 4), resolution=5),
    lambda: ps.snapshot_contour_plot(DISPLACEMENT, "a", "t", {}, (0, 4), (0, 4), resolution=5),
    lambda: ps.snapshot_rotating_surface_gif(DISPLACEMENT, "a", "t", {}, (0, 4), (0, 4), resolution=5,
                                             n_frames=2, fps=2),
], ids=["plot", "surface", "contour", "snap-line", "snap-surface", "snap-contour", "snap-gif"])
def test_residual_fallback_with_a_missing_value_raises_a_clear_error(call):
    with pytest.raises(ValueError, match="no value for"):
        call()


def test_failed_solve_with_ui_style_params_falls_back_to_a_real_numeric_residual(monkeypatch):
    # when the solve genuinely fails, the residual IS plotted -- using the target's
    # slider value, which is exactly what that slider is still for
    monkeypatch.setattr(sp, "solve", lambda *a_, **k: [])
    fig = pl.build_plot(None, VELOCITY, "t", {"u": 8.0, "a": 2.0, "v": 20.0}, (0, 10), y_target="v")
    assert "residual" in fig.layout.yaxis.title.text
    ys = np.asarray(fig.data[0].y, dtype=float)                    # real numbers, not sympy objects
    assert ys[0] == pytest.approx(20 - 8)


def test_solved_curve_with_a_singularity_does_not_spam_numpy_warnings():
    """Now that the target selector actually works, a solved curve like
    a = (v - u)/t is plotted over a range that can include t = 0. The
    non-finite point is dropped by the plotting library; numpy's own
    'divide by zero' RuntimeWarning on every slider drag is just noise."""
    import warnings
    division = Equation(name="accel", raw_expression="Eq(a, (v - u)/t)", derivation="", kind="equation",
                        sympy_eq=sp.Eq(a, (v - u) / t))
    params = {"u": 8.0, "v": 20.0, "a": 2.0}
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # any warning becomes a test failure
        fig = pl.build_plot(None, division, "t", params, (0.0, 10.0), y_target="a")
        assert ps.snapshot_line_plot(division, "t", params, (0.0, 10.0), y_target="a")[:8] == PNG
    ys = np.asarray(fig.data[0].y, dtype=float)
    assert np.isinf(ys[0]) and ys[-1] == pytest.approx(12.0 / 10.0)      # the singular point is there, not hidden
