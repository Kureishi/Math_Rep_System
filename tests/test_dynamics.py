"""The dynamics and chaos engine: conservation laws, integrator orders, and exact identities for the Lyapunov spectrum."""
import math

import numpy as np
import pytest

from modules import dynamics as dy
from modules import gpu


def labels(run, passed=None):
    return {c.label: c.passed for c in run.checks.checks if passed is None or c.passed == passed}


def test_the_figure_eight_returns_after_each_period_and_conserves_everything():
    r = dy.run_nbody(dy.preset_system("figure-eight"), "leapfrog", 1e-3, 2 * dy.FIGURE_EIGHT_PERIOD)
    assert r.checks.passed and r.energy_drift.max_rel < 1e-5 and r.return_error < 1e-3
    assert r.energy_drift.kind != "secular drift"


def test_yoshida_is_far_more_accurate_than_leapfrog_at_the_same_step():
    sys_ = dy.preset_system("figure-eight")
    leap = dy.run_nbody(sys_, "leapfrog", 2e-3, dy.FIGURE_EIGHT_PERIOD)
    yosh = dy.run_nbody(sys_, "yoshida4", 2e-3, dy.FIGURE_EIGHT_PERIOD)
    assert yosh.energy_drift.max_rel < leap.energy_drift.max_rel / 1000


def test_the_lagrange_triangle_stays_equilateral():
    r = dy.run_nbody(dy.preset_system("lagrange"), "leapfrog", 2e-3, 2 * 3.6276)
    assert r.checks.passed and "The triangle stays equilateral" in labels(r)


def test_a_kepler_orbit_keeps_its_runge_lenz_vector_and_obeys_the_third_law():
    r = dy.run_nbody(dy.preset_system("kepler", eccentricity=0.5), "yoshida4", 5e-3, 6 * 2 * math.pi)
    assert r.checks.passed
    assert "The Laplace-Runge-Lenz vector is conserved (the orbit does not precess)" in labels(r)
    assert "The period obeys Kepler's third law" in labels(r)


def test_explicit_euler_leaks_energy_and_the_checks_say_so():
    r = dy.run_nbody(dy.preset_system("kepler", eccentricity=0.5), "euler", 2e-3, 4 * 2 * math.pi)
    assert not r.checks.passed and r.energy_drift.kind == "secular drift"
    assert "Energy is conserved to within the tolerance" in labels(r, passed=False)


def test_a_non_symplectic_method_gets_a_note_instead_of_a_signature_check():
    r = dy.run_nbody(dy.preset_system("kepler"), "rk4", 5e-3, 2 * 2 * math.pi)
    assert r.notes and "not symplectic" in r.notes[0]
    assert "The energy error stays bounded (the symplectic signature)" not in labels(r)


def test_the_chaotic_pythagorean_problem_fails_the_energy_check_honestly():
    r = dy.run_nbody(dy.preset_system("pythagorean"), "leapfrog", 1e-3, 8.0)
    assert not r.checks.passed and r.energy_drift.max_rel > 1e-2


def test_a_random_cluster_is_reproducible_from_its_seed_and_conserves_momentum():
    a, b = dy.preset_system("cluster", n=30, seed=5), dy.preset_system("cluster", n=30, seed=5)
    assert np.array_equal(a.positions, b.positions) and a.seed == 5
    assert abs(float((a.masses[:, None] * a.velocities).sum())) < 1e-12
    r = dy.run_nbody(a, "leapfrog", 2e-3, 1.0)
    assert r.checks.passed and r.positions.shape[1:] == (30, 3)


def test_the_nbody_run_is_the_same_on_a_pretend_gpu():
    sys_ = dy.preset_system("cluster", n=20, seed=2)
    cpu = dy.run_nbody(sys_, "leapfrog", 2e-3, 0.4)
    fake = gpu.Backend(np, "GPU: pretend (CuPy)", True, "numpy")
    with gpu.pretend_gpu(fake):
        g = dy.run_nbody(sys_, "leapfrog", 2e-3, 0.4, backend=gpu.select(2000, "nbody_particles"))
    assert g.backend == "GPU: pretend (CuPy)" and np.allclose(cpu.positions, g.positions)


def test_the_observed_order_of_each_integrator_matches_its_claim():
    studies, ck = dy.order_study(dy.preset_system("kepler", eccentricity=0.5))
    assert ck.passed
    got = {s.integrator: s.observed_order for s in studies}
    assert got["leapfrog"] == pytest.approx(2, abs=0.2) and got["yoshida4"] == pytest.approx(4, abs=0.2) and got["rk4"] == pytest.approx(4, abs=0.4)


def test_the_accelerations_obey_newtons_third_law():
    s = dy.preset_system("cluster", n=25, seed=3)
    a = dy.accelerations(s.masses, s.positions, s.softening)
    assert np.allclose((s.masses[:, None] * a).sum(0), 0, atol=1e-12)


@pytest.mark.parametrize("kwargs, message", [
    (dict(integrator="rk9"), "Choose one of"), (dict(dt=0), "positive"), (dict(t_end=-1), "positive"), (dict(dt=1e-6, t_end=10), "400,000"),
])
def test_nbody_input_is_validated(kwargs, message):
    with pytest.raises(dy.DynamicsError, match=message):
        dy.run_nbody(dy.preset_system("kepler"), **{"integrator": "leapfrog", "dt": 1e-3, "t_end": 1.0, **kwargs})


def test_presets_are_validated():
    with pytest.raises(dy.DynamicsError, match="Unknown system"):
        dy.preset_system("solar")
    with pytest.raises(dy.DynamicsError, match="between 3 and"):
        dy.preset_system("cluster", n=1)
    with pytest.raises(dy.DynamicsError, match="eccentricity"):
        dy.preset_system("kepler", eccentricity=1.2)


# ------------------------------------------------------------------------------------------ pendulum

def test_the_fast_pendulum_equations_match_the_lagrangian_derivation():
    err, _ = dy.verify_pendulum_rhs(dy.PendulumParams(1.0, 2.0, 0.7, 1.3, 9.81))
    assert err < 1e-12


def test_the_pendulum_conserves_energy_and_its_jacobian_matches_finite_differences():
    r = dy.run_pendulum(dy.PendulumParams(), (2.0, 2.5, 0.0, 0.0), 10.0)
    assert r.checks.passed and r.energy_drift.max_rel < 1e-8
    p, s = dy.PendulumParams(), np.array([0.7, -1.1, 0.4, 0.9])
    J = dy.pendulum_jacobian(s, p)
    h = 1e-6
    fd = np.stack([(dy.pendulum_rhs(s + h * e, p) - dy.pendulum_rhs(s - h * e, p)) / (2 * h) for e in np.eye(4)], axis=1)
    assert np.allclose(J, fd, atol=1e-6)


def test_pendulum_parameters_are_validated():
    with pytest.raises(dy.DynamicsError, match="m1 must be positive"):
        dy.run_pendulum(dy.PendulumParams(m1=0), (1, 1, 0, 0))
    with pytest.raises(dy.DynamicsError, match="duration"):
        dy.run_pendulum(dy.PendulumParams(), (1, 1, 0, 0), 5000)


# ------------------------------------------------------------------------------------------ Lorenz and Lyapunov

def test_the_lorenz_jacobian_matches_finite_differences():
    p, s = dy.LorenzParams(), np.array([1.3, -2.0, 20.0])
    h = 1e-6
    fd = np.stack([(dy.lorenz_rhs(s + h * e, p) - dy.lorenz_rhs(s - h * e, p)) / (2 * h) for e in np.eye(3)], axis=1)
    assert np.allclose(dy.lorenz_jacobian(s, p), fd, atol=1e-6)


def test_the_lorenz_exponents_sum_to_the_trace_and_the_largest_is_the_published_one():
    ly = dy.lyapunov_spectrum("lorenz", dy.LorenzParams(), np.array([1.0, 1.0, 1.0]), members=128, t_run=25.0, seed=1)
    assert ly.checks.passed, [c.detail for c in ly.checks.checks if not c.passed]
    assert ly.exponents.sum() == pytest.approx(-(10 + 1 + 8 / 3), abs=0.01)
    assert ly.exponents[0] == pytest.approx(dy.LORENZ_LAMBDA1_LITERATURE, abs=0.06)
    assert ly.kaplan_yorke == pytest.approx(dy.LORENZ_KAPLAN_YORKE_LITERATURE, abs=0.1)


def test_the_lyapunov_estimate_is_reproducible_from_its_seed_and_reports_it():
    a = dy.lyapunov_spectrum("lorenz", dy.LorenzParams(), np.array([1.0, 1.0, 1.0]), members=32, t_run=5.0, seed=7)
    b = dy.lyapunov_spectrum("lorenz", dy.LorenzParams(), np.array([1.0, 1.0, 1.0]), members=32, t_run=5.0, seed=7)
    assert np.array_equal(a.exponents, b.exponents) and a.seed == 7


def test_a_regular_lorenz_has_no_positive_exponent():
    ly = dy.lyapunov_spectrum("lorenz", dy.LorenzParams(rho=15.0), np.array([1.0, 1.0, 1.0]), members=32, t_run=15.0, seed=1)
    assert ly.exponents[0] < 0.05 and any("regular" in n for n in ly.notes)


def test_the_double_pendulum_spectrum_is_symmetric_and_chaotic_at_high_energy():
    ly = dy.lyapunov_spectrum("pendulum", dy.PendulumParams(), np.array([2.0, 2.5, 0.0, 0.0]), members=32, t_run=25.0, seed=1)
    assert ly.checks.passed, [c.detail for c in ly.checks.checks if not c.passed]
    assert ly.exponents[0] > 0.8 and ly.exponents[0] == pytest.approx(-ly.exponents[3], abs=0.15)


def test_a_small_swing_is_regular():
    ly = dy.lyapunov_spectrum("pendulum", dy.PendulumParams(), np.array([0.2, 0.25, 0.0, 0.0]), members=32, t_run=20.0, seed=1)
    assert ly.exponents[0] < 0.1 and any("regular" in n for n in ly.notes)


def test_the_ensemble_runs_the_same_on_a_pretend_gpu():
    cpu = dy.lyapunov_spectrum("lorenz", dy.LorenzParams(), np.array([1.0, 1.0, 1.0]), members=32, t_run=4.0, seed=3)
    fake = gpu.Backend(np, "GPU: pretend (CuPy)", True, "numpy")
    with gpu.pretend_gpu(fake):
        g = dy.lyapunov_spectrum("lorenz", dy.LorenzParams(), np.array([1.0, 1.0, 1.0]), members=32, t_run=4.0, seed=3, backend=gpu.select(30000, "ensemble_trajectories"))
    assert g.backend == "GPU: pretend (CuPy)" and np.allclose(cpu.exponents, g.exponents)


def test_two_routes_to_the_largest_exponent_agree():
    curve = dy.divergence_curve("lorenz", dy.LorenzParams(), np.array([1.0, 1.0, 1.0]), members=128, seed=2)
    assert curve.slope == pytest.approx(dy.LORENZ_LAMBDA1_LITERATURE, abs=0.1)


@pytest.mark.parametrize("kwargs, message", [
    (dict(system="rossler"), "Choose 'lorenz' or 'pendulum'"), (dict(members=2), "between 8"), (dict(t_run=0), "positive"), (dict(dt=1e-5, t_run=100), "400,000"),
])
def test_lyapunov_input_is_validated(kwargs, message):
    base = dict(system="lorenz", params=dy.LorenzParams(), state0=np.array([1.0, 1.0, 1.0]), members=16)
    with pytest.raises(dy.DynamicsError, match=message):
        dy.lyapunov_spectrum(**{**base, **kwargs})


def test_kaplan_yorke_of_a_known_spectrum():
    assert dy.kaplan_yorke(np.array([0.9056, 0.0, -14.5723])) == pytest.approx(2.0621, abs=1e-3)
    assert dy.kaplan_yorke(np.array([-1.0, -2.0])) == 0.0


def test_the_flip_map_is_symmetric_conserves_energy_and_is_the_same_on_a_pretend_gpu():
    fm = dy.flip_map(dy.PendulumParams(), n=24, t_end=4.0, dt=0.01)
    assert fm.checks.passed and fm.flip_time.shape == (24, 24) and fm.flip_time.min() > 0
    fake = gpu.Backend(np, "GPU: pretend (CuPy)", True, "numpy")
    with gpu.pretend_gpu(fake):
        g = dy.flip_map(dy.PendulumParams(), n=24, t_end=4.0, dt=0.01, backend=gpu.select(30000, "ensemble_trajectories"))
    assert np.allclose(fm.flip_time, g.flip_time)
    with pytest.raises(dy.DynamicsError, match="grid"):
        dy.flip_map(dy.PendulumParams(), n=2)
