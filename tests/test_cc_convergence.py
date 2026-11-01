"""Tests for modules/cc_convergence.py: the study machinery and every ready-made adapter against the known order of its method."""
import math

import numpy as np
import pytest

from modules import cc_conservation as cc
from modules import cc_convergence as cv


# ---------------------------------------------------------------- the machinery on synthetic problems
@pytest.mark.parametrize("p", [1.0, 2.0, 3.0, 4.0])
def test_recovers_the_order_of_a_synthetic_error_law(p):
    res = cv.convergence_study(lambda h: 2.0 + 0.7 * h ** p + 0.1 * h ** (p + 1), 0.5, 6, 2.0, reference=2.0, expected_order=p)
    assert res.observed_order == pytest.approx(p, abs=0.1) and res.checks.passed and res.fit_order == pytest.approx(p, abs=0.2)


def test_works_without_a_reference_via_richardson():
    res = cv.convergence_study(lambda h: 5.0 + 3.0 * h ** 2, 0.5, 6, 2.0, expected_order=2.0)
    assert res.error_basis == "Richardson extrapolation" and res.extrapolated == pytest.approx(5.0, abs=1e-6)
    assert res.observed_order == pytest.approx(2.0, abs=0.02) and res.checks.passed


def test_wrong_expected_order_fails_the_check():
    res = cv.convergence_study(lambda h: 1.0 + h ** 2, 0.5, 6, 2.0, reference=1.0, expected_order=4.0)
    assert not res.checks.passed and any("expected order" in c.label and not c.passed for c in res.checks.checks)


def test_non_convergent_sequence_is_detected():
    res = cv.convergence_study(lambda h: 1.0 + 0.3 * math.sin(10 * h), 0.5, 6, 2.0, reference=1.0, expected_order=1.0)
    assert not res.checks.passed


def test_rounding_floor_levels_are_excluded_from_the_fit():
    res = cv.convergence_study(lambda h: 1.0 + h ** 4, 0.05, 8, 2.0, reference=1.0, expected_order=4.0)
    assert any(res.at_floor) and res.observed_order == pytest.approx(4.0, abs=0.3)
    assert "rounding floor" in " ".join(c.detail for c in res.checks.checks)


def test_gci_is_a_valid_error_bar_for_the_finest_level():
    res = cv.convergence_study(lambda h: 3.0 + 2.0 * h ** 2, 0.5, 6, 2.0, reference=3.0, expected_order=2.0)
    assert abs(res.values[-1] - 3.0) / 3.0 <= res.gci * 3 and res.gci > 0
    assert next(c for c in res.checks.checks if "grid-convergence" in c.label).passed


def test_array_valued_solutions_and_custom_norms():
    res = cv.convergence_study(lambda h: np.array([1.0 + h ** 2, 2.0 - 3 * h ** 2]), 0.4, 6, 2.0, reference=np.array([1.0, 2.0]), expected_order=2.0,
                               norm=lambda a: float(np.sqrt(np.sum(a ** 2))))
    assert res.observed_order == pytest.approx(2.0, abs=0.05)


def test_progress_hook_and_input_validation():
    seen = []
    cv.convergence_study(lambda h: 1.0 + h, 0.5, 4, progress=lambda m, f: seen.append(f))
    assert len(seen) == 4
    for kw in (dict(n_levels=3), dict(ratio=1.0), dict(h0=-1.0)):
        args = dict(solve=lambda h: 1.0, h0=0.5, n_levels=5, ratio=2.0)
        args.update(kw)
        with pytest.raises(cv.CrossCuttingError):
            cv.convergence_study(**args)


# ---------------------------------------------------------------- ready-made studies
F = lambda t, y: np.array([y[1], -y[0]])  # noqa: E731
EXACT = lambda t: np.array([math.sin(t), math.cos(t)])  # noqa: E731


@pytest.mark.parametrize("method,order", [("euler", 1), ("heun", 2), ("midpoint", 2), ("rk4", 4)])
def test_ode_methods_show_their_textbook_orders(method, order):
    res = cv.ode_convergence(F, [0.0, 1.0], 3.0, method, 8, 6, EXACT)
    assert res.observed_order == pytest.approx(order, abs=0.15) and res.checks.passed


def test_ode_without_an_exact_solution_uses_richardson():
    res = cv.ode_convergence(F, [0.0, 1.0], 3.0, "rk4", 4, 6)
    assert res.error_basis == "Richardson extrapolation" and res.observed_order == pytest.approx(4.0, abs=0.2)
    with pytest.raises(cv.CrossCuttingError):
        cv.ode_convergence(F, [0.0, 1.0], 3.0, "nope")


@pytest.mark.parametrize("method,order", [("leapfrog", 2), ("yoshida4", 4), ("midpoint", 2), ("rk4", 4)])
def test_hamiltonian_integrators_show_their_orders(method, order):
    flow = cc.HamiltonianFlow(1, lambda q, p: 0.5 * p[0] ** 2 - np.cos(q[0]), lambda q, p: np.array([np.sin(q[0])]), lambda q, p: np.array([p[0]]), True)
    res = cv.hamiltonian_convergence(flow, np.array([1.0]), np.array([0.0]), 5.0, method, 8, 6)
    assert res.observed_order == pytest.approx(order, abs=0.15) and res.checks.passed
    with pytest.raises(cv.CrossCuttingError):
        cv.hamiltonian_convergence(flow, np.array([1.0]), np.array([0.0]), 5.0, "bogus")


def test_poisson_solver_is_second_order_against_a_manufactured_solution():
    res = cv.poisson_convergence("sin(pi*x)*sin(pi*y) + x*y", n0=9, n_levels=4)
    assert res.observed_order == pytest.approx(2.0, abs=0.15) and res.checks.passed
    with pytest.raises(cv.CrossCuttingError):
        cv.poisson_convergence("x +", n0=9)
    with pytest.raises(cv.CrossCuttingError):
        cv.poisson_convergence("x*y", n0=3)


def test_split_step_wavepacket_is_second_order_in_dt():
    res = cv.wavepacket_convergence(40, 5)
    assert res.observed_order == pytest.approx(2.0, abs=0.15) and res.checks.passed


def test_sde_strong_orders_euler_half_milstein_one_and_seed_is_recorded():
    e = cv.sde_strong_convergence("euler", seed=1)
    m = cv.sde_strong_convergence("milstein", seed=1)
    assert e.seed == 1 and e.result.observed_order == pytest.approx(0.5, abs=0.12) and e.result.checks.passed
    assert m.result.observed_order == pytest.approx(1.0, abs=0.15) and m.result.checks.passed
    assert m.result.errors[-1] < e.result.errors[-1]
    with pytest.raises(cv.CrossCuttingError):
        cv.sde_strong_convergence("rk99")
    assert isinstance(cv.sde_strong_convergence("euler", n_paths=500, n_levels=4).seed, int)
