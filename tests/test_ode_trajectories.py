"""modules/ode_trajectories.py: the closed-form-vs-numerical comparison over
time, and the multi-start phase flow. Reference solutions are exact where
one exists (rotation x' = -y, y' = x traces (cos t, sin t); decay is
100*exp(-k t)), so the integrator is checked against mathematics rather
than against itself."""
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

import modules.ode_trajectories as ot
from modules.equation_engine import build_model
from modules.ode_utils import solve_ode, group_coupled_odes

t = sp.Symbol("t")
N = sp.Function("N")


def _model(equations, funcs, ics, known=()):
    return build_model({
        "problem_domain": "physics", "problem_type": "ode", "independent_variable": "t",
        "variables": [{"symbol": s, "meaning": s, "known_value": v, "unit": None, "is_function": False}
                      for s, v in known]
                     + [{"symbol": f, "meaning": f, "known_value": None, "unit": None, "is_function": True}
                        for f in funcs],
        "equations": [{"name": n, "kind": "ode", "expression": e, "derivation": ""} for n, e in equations],
        "solve_for": list(funcs), "assumptions": [],
        "initial_conditions": [{"expression": ex, "value": v} for ex, v in ics],
    })


def decay_model(ics=(("N(0)", 100.0),)):
    return _model([("decay", "Eq(Derivative(N(t), t), -k*N(t))")], ["N"], ics, known=[("k", 0.5)])


def spiral_model():
    return _model([("xdot", "Eq(Derivative(x(t), t), y(t))"),
                   ("ydot", "Eq(Derivative(y(t), t), -x(t) - 0.3*y(t))")],
                  ["x", "y"], [("x(0)", 1.0), ("y(0)", 0.0)])


def _group(model):
    return group_coupled_odes([e for e in model.equations if e.kind == "ode" and e.sympy_eq is not None])[0]


def _compare(model, **kw):
    return ot.compare_trajectories(model, _group(model), solve_ode(model), **kw)


# ------------------------------------------------------------------ compare_trajectories

def test_single_ode_comparison_agrees_with_the_exact_solution():
    c = _compare(decay_model())
    assert c.applicable and c.ok and c.names == ["N"]
    assert c.t[0] == 0.0 and len(c.t) == 300 and c.numeric.shape == c.symbolic.shape == (1, 300)
    assert c.symbolic[0] == pytest.approx(100.0 * np.exp(-0.5 * c.t))        # the closed form IS 100 e^(-kt)
    assert c.numeric[0] == pytest.approx(100.0 * np.exp(-0.5 * c.t), rel=1e-6)   # ...and so is the integration
    assert c.max_rel_error < 1e-6 and c.max_rel_error < c.tolerance
    assert c.abs_error.shape == c.rel_error.shape == (1, 300)


def test_coupled_system_comparison_covers_every_function():
    c = _compare(spiral_model())
    assert c.applicable and c.ok and c.names == ["x", "y"]
    assert c.numeric.shape == (2, 300)


def test_a_wrong_closed_form_is_reported_and_its_error_grows():
    model = decay_model()
    wrong = {"N": sp.Eq(N(t), 100 * sp.exp(-0.4 * t))}                       # rate 0.4 instead of 0.5
    c = ot.compare_trajectories(model, _group(model), wrong)
    assert c.applicable and c.ok is False
    assert c.max_rel_error > 0.1
    assert c.rel_error[0][-1] > c.rel_error[0][1]                           # the disagreement accumulates with t


def test_end_time_override_sets_the_window():
    c = _compare(decay_model(), t_end=4.0, n_points=50)
    assert c.t[-1] == pytest.approx(4.0) and len(c.t) == 50


def test_end_time_before_the_initial_time_is_refused():
    c = _compare(decay_model(), t_end=0.0)
    assert not c.applicable and "must be after the initial time" in c.reason


def test_inapplicable_group_passes_the_reason_through():
    c = _compare(decay_model(ics=()))
    assert not c.applicable and "initial condition" in c.reason


def test_constant_solution_is_broadcast_over_the_time_grid():
    model = _model([("const", "Eq(Derivative(N(t), t), 0)")], ["N"], [("N(0)", 100.0)])
    c = _compare(model)
    assert c.applicable and c.ok
    assert c.symbolic.shape == (1, 300) and set(c.symbolic[0]) == {100.0}


def test_integration_exception_is_reported(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("integrator crashed")
    monkeypatch.setattr(ot, "run_with_timeout", boom)
    c = _compare(decay_model())
    assert not c.applicable and "Numerical integration failed: integrator crashed" in c.reason


def test_non_convergence_is_reported(monkeypatch):
    monkeypatch.setattr(ot, "run_with_timeout", lambda *a, **k: SimpleNamespace(success=False, message="too stiff"))
    c = _compare(decay_model())
    assert not c.applicable and "did not converge: too stiff" in c.reason


def test_failure_evaluating_the_trajectory_is_reported(monkeypatch):
    def bad_sol(_ts):
        raise ValueError("no dense output")
    monkeypatch.setattr(ot, "run_with_timeout", lambda *a, **k: SimpleNamespace(success=True, sol=bad_sol))
    c = _compare(decay_model())
    assert not c.applicable and "Could not evaluate for comparison: no dense output" in c.reason


# ------------------------------------------------------------------ ring_of_starts

def test_ring_of_starts_geometry():
    pts = ot.ring_of_starts((1.0, 2.0), (-3.0, 5.0), (0.0, 4.0), n=4, fraction=0.5)
    assert len(pts) == 4
    rx, ry = 0.5 * 8 / 2, 0.5 * 4 / 2                                       # half of the view box's half-extent
    assert pts[0] == pytest.approx((1.0 + rx, 2.0))
    assert pts[2] == pytest.approx((1.0 - rx, 2.0))
    assert pts[1] == pytest.approx((1.0, 2.0 + ry))


def test_ring_of_starts_is_empty_for_a_nonpositive_count():
    assert ot.ring_of_starts((0, 0), (-1, 1), (-1, 1), 0) == []
    assert ot.ring_of_starts((0, 0), (-1, 1), (-1, 1), -3) == []


# ------------------------------------------------------------------ phase_flow

def test_rotation_flow_traces_the_exact_unit_circle():
    paths = ot.phase_flow(lambda x, y: -y, lambda x, y: x, [(1.0, 0.0), (2.0, 0.0)], duration=6.0, n_points=61)
    assert len(paths) == 2 and all(len(xs) == len(ys) == 61 for xs, ys in paths)
    ts = np.linspace(0, 6.0, 61)
    xs, ys = paths[0]
    assert xs == pytest.approx(np.cos(ts), abs=1e-5) and ys == pytest.approx(np.sin(ts), abs=1e-5)
    xs2, ys2 = paths[1]
    assert xs2 == pytest.approx(2 * np.cos(ts), abs=1e-5)                    # same flow, a bigger circle


def test_finite_time_blowup_is_cut_off_and_the_rest_is_nan():
    # x' = x^2 from x = 1 reaches infinity at t = 1
    (xs, ys), = ot.phase_flow(lambda x, y: x * x, lambda x, y: 0.0, [(1.0, 0.0)], duration=3.0,
                               n_points=61, blowup=1e3)
    assert xs[0] == 1.0 and np.isfinite(xs[:10]).all()                       # fine well before t = 1
    assert np.isnan(xs[-1]) and np.isnan(ys[-1])
    first_nan = int(np.argmax(np.isnan(xs)))
    assert np.isnan(xs[first_nan:]).all()                                    # NaN from the cut-off onward, no gaps
    assert 15 <= first_nan <= 25                                             # cut-off is near t = 1 (index 20)


def test_integration_failure_gives_an_all_nan_path_not_a_dropped_one():
    def dx(x, y):
        raise ZeroDivisionError("singular")
    paths = ot.phase_flow(dx, lambda x, y: 0.0, [(1.0, 0.0), (2.0, 0.0)], duration=1.0, n_points=11)
    assert len(paths) == 2                                                   # the caller's index into `starts` stays valid
    for xs, ys in paths:
        assert len(xs) == 11 and np.isnan(xs).all() and np.isnan(ys).all()


def test_no_starts_gives_no_paths():
    assert ot.phase_flow(lambda x, y: -y, lambda x, y: x, [], duration=1.0) == []
