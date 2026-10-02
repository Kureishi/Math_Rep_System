"""modules/pde_field.py: u(x, t) on a grid, against exact solutions."""
import numpy as np
import pytest
import sympy as sp

import modules.pde_field as pf
from modules.pde_field import evaluate_field
from modules.pde_utils import solve_heat_equation_dirichlet

x = sp.Symbol("x", positive=True)
t = sp.Symbol("t", positive=True)
HEAT_MODE = sp.sin(sp.pi * x) * sp.exp(-sp.pi ** 2 * t)


def test_heat_mode_matches_the_exact_solution_and_its_conserved_summaries():
    f = evaluate_field(HEAT_MODE, 1.0, 0.4, n_x=201, n_t=41)
    assert f.error is None and f.u.shape == (41, 201) and f.xs[-1] == 1.0 and f.ts[-1] == 0.4
    assert f.u == pytest.approx(np.sin(np.pi * f.xs)[None, :] * np.exp(-np.pi ** 2 * f.ts)[:, None])
    # integral of sin(pi x) over [0, 1] is 2/pi, and the mode decays as exp(-pi^2 t)
    assert f.integral == pytest.approx(2 / np.pi * np.exp(-np.pi ** 2 * f.ts), rel=1e-4)
    assert f.max_abs == pytest.approx(np.exp(-np.pi ** 2 * f.ts), rel=1e-6)
    assert (f.u_min, f.u_max) == pytest.approx((0.0, 1.0), abs=1e-9)


def test_wave_solution_oscillates_and_conserves_nothing_it_shouldnt():
    f = evaluate_field(sp.sin(sp.pi * x) * sp.cos(sp.pi * t), 1.0, 2.0, n_x=101, n_t=101)
    assert f.u_min == pytest.approx(-1.0, abs=1e-3) and f.u_max == pytest.approx(1.0, abs=1e-3)
    assert f.integral[0] > 0 > f.integral[50]                              # cos(pi t) is +1 at t=0, -1 at t=1
    assert f.integral == pytest.approx(2 / np.pi * np.cos(np.pi * f.ts), rel=1e-3, abs=1e-4)


def test_the_real_heat_solver_output_is_evaluated_without_complaint():
    result = solve_heat_equation_dirichlet("x*(1-x)", length=1.0, alpha=0.1)
    f = evaluate_field(result.solution, 1.0, 3.0)
    assert f.error is None
    assert f.u[0] == pytest.approx(f.xs * (1 - f.xs), abs=2e-3)            # the truncated series reproduces the initial profile
    assert (np.diff(f.max_abs) <= 1e-12).all()                              # heat only ever cools toward zero here


def test_a_constant_solution_is_broadcast_over_the_grid():
    f = evaluate_field(sp.Integer(3), 2.0, 1.0, n_x=5, n_t=4)
    assert f.error is None and f.u.shape == (4, 5) and (f.u == 3).all() and f.integral == pytest.approx([6.0] * 4)


def test_a_solution_in_t_alone_still_evaluates():
    f = evaluate_field(sp.exp(-t), 1.0, 2.0, n_x=4, n_t=5)
    assert f.error is None and f.u[:, 0] == pytest.approx(np.exp(-f.ts))


@pytest.mark.parametrize("args, fragment", [
    (dict(length=0.0, t_max=1.0), "must both be positive"),
    (dict(length=1.0, t_max=-1.0), "must both be positive"),
    (dict(length=1.0, t_max=1.0, n_x=1), "at least 2x2"),
    (dict(length=1.0, t_max=1.0, n_t=1), "at least 2x2"),
    (dict(length=1.0, t_max=1.0, n_x=pf.MAX_POINTS, n_t=2), "at most"),
])
def test_invalid_grids_are_reported(args, fragment):
    f = evaluate_field(HEAT_MODE, **args)
    assert f.error and fragment in f.error and f.u.size == 0


def test_a_stray_symbol_is_reported():
    f = evaluate_field(sp.Symbol("alpha") * HEAT_MODE, 1.0, 1.0)
    assert f.error and "alpha" in f.error


def test_a_non_real_solution_is_reported():
    f = evaluate_field(sp.I * HEAT_MODE, 1.0, 1.0)
    assert f.error and "isn't real and finite" in f.error


def test_a_solution_that_blows_up_is_reported():
    f = evaluate_field(1 / x, 1.0, 1.0)                                    # infinite at x = 0, the first grid point
    assert f.error and "isn't real and finite" in f.error


def test_an_evaluation_failure_is_reported(monkeypatch):
    def boom(*a, **k):
        raise TypeError("cannot lambdify")
    monkeypatch.setattr(pf.sp, "lambdify", boom)
    assert "Could not evaluate the solution: cannot lambdify" in (evaluate_field(HEAT_MODE, 1.0, 1.0).error or "")
