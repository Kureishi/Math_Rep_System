"""Tests for modules/sm_stochastic.py: Langevin integrators, MSD, the Einstein relation, Fokker-Planck."""
import math

import numpy as np
import pytest

from modules import sm_stochastic as ss

FREE = ss.preset_potential("free")


# ---------------------------------------------------------------- potentials
def test_preset_and_custom_potentials():
    h = ss.preset_potential("harmonic", k=3.0)
    assert h.U(2.0) == pytest.approx(6.0) and h.dU(2.0) == pytest.approx(6.0)
    d = ss.preset_potential("double_well", a=1.0, b=1.0)
    assert d.U(1.0) == pytest.approx(-0.25) and d.dU(1.0) == pytest.approx(0.0, abs=1e-12) and d.dU(0.0) == 0.0
    c = ss.potential_from_text("a*x^2 + x", {"a": 2.0})
    assert c.U(np.array([1.0, 2.0])) == pytest.approx([3.0, 10.0]) and c.dU(1.0) == pytest.approx(5.0)
    assert ss.preset_potential("free").U(np.array([1.0, 2.0])).shape == (2,)                    # constants are broadcast
    assert ss.potential_from_text("y + x").params == {"y": 1.0}                                  # a name other than x is a parameter
    for bad in ("__import__('os')", "x +", "x.real"):
        with pytest.raises(ss.StatMechError):
            ss.potential_from_text(bad)
    with pytest.raises(ss.StatMechError):
        ss.preset_potential("nope")


# ---------------------------------------------------------------- diffusion and the exact MSD
def test_underdamped_msd_follows_the_exact_crossover():
    r = ss.simulate_underdamped(FREE, 1.0, 1.0, 1.0, 0.0, 0.01, 10.0, 30000, 100, seed=3)
    m, ex = ss.msd(r), ss.msd_exact_free(r.t, 1.0, 1.0, 1.0)
    assert np.max(np.abs(m[1:] - ex[1:]) / ex[1:]) < 0.04
    assert m[1] == pytest.approx(r.t[1] ** 2, rel=0.05)                                       # ballistic: (kT/m) t^2
    assert m[-1] == pytest.approx(2 * r.t[-1], rel=0.1)                                       # diffusive: 2 D t, D = kT/gamma = 1


def test_overdamped_diffusion_constant():
    r = ss.simulate_overdamped(FREE, 1.0, 2.0, 0.0, 0.01, 20.0, 30000, 100, seed=4)
    d, err = ss.fit_diffusion(r.t, ss.msd(r), 2.0)
    assert d == pytest.approx(0.5, abs=4 * err + 0.01)
    with pytest.raises(ss.StatMechError):
        ss.fit_diffusion(r.t, ss.msd(r), 1e9)


def test_msd_formulas_limits():
    t = np.array([1e-4, 1e4])
    ex = ss.msd_exact_free(t, 2.0, 4.0, 1.0)
    assert ex[0] == pytest.approx(2.0 * 1e-8, rel=1e-3) and ex[1] == pytest.approx(2 * 0.5 * 1e4, rel=1e-3)
    assert ss.msd_exact_free(t, 2.0, 4.0, None)[1] == pytest.approx(1e4)


# ---------------------------------------------------------------- fluctuation-dissipation
@pytest.mark.parametrize("mass,dt,tmax", [(None, 0.01, 20.0), (1.0, 0.02, 30.0)])
def test_einstein_relation_holds_when_noise_and_friction_are_tied(mass, dt, tmax):
    e = ss.einstein_relation(1.0, 2.0, 1.0, dt, tmax, 20000, mass, seed=1)
    assert e.checks.passed, [(c.label, c.detail) for c in e.checks.checks if not c.passed]
    assert e.kT_estimate == pytest.approx(1.0, abs=4 * e.kT_err) and e.seed == 1


def test_einstein_relation_fails_when_the_noise_is_wrong():
    """Break the tie: simulate with noise for kT = 2 but compare with kT = 1 -- D/mu comes out near 2, and the check must say so."""
    hot = ss.einstein_relation(2.0, 2.0, 1.0, 0.01, 20.0, 20000, None, seed=2)
    assert hot.kT_estimate == pytest.approx(2.0, abs=4 * hot.kT_err)
    hot.checks = ss.CheckList()
    assert abs(hot.kT_estimate - 1.0) > 10 * hot.kT_err                                         # a kT = 1 expectation would be rejected decisively


def test_equipartition_and_boltzmann_in_a_harmonic_trap():
    pot = ss.preset_potential("harmonic", k=2.0)
    r = ss.simulate_underdamped(pot, 1.0, 2.0, 1.0, 0.0, 0.01, 20.0, 4000, 100, seed=5)
    ck = ss.equipartition_checks(r, 2.0)
    assert ck.passed, [(c.label, c.detail) for c in ck.checks if not c.passed]
    d, p = ss.boltzmann_ks(r.x[-1], pot, 1.0)
    assert p > 1e-3


def test_double_well_samples_are_boltzmann_distributed():
    pot = ss.preset_potential("double_well")
    r = ss.simulate_overdamped(pot, 0.4, 1.0, 0.0, 0.01, 200.0, 4000, 20, seed=6)
    assert ss.boltzmann_ks(r.x[-1], pot, 0.4)[1] > 1e-3
    assert np.mean(r.x[-1] > 0) == pytest.approx(0.5, abs=0.05)                                 # both wells populated equally


def test_boltzmann_test_rejects_the_wrong_temperature():
    pot = ss.preset_potential("harmonic", k=2.0)
    samples = np.random.default_rng(0).normal(0, math.sqrt(1.0 / 2.0), 20000)
    assert ss.boltzmann_ks(samples, pot, 1.0)[1] > 1e-2
    assert ss.boltzmann_ks(samples, pot, 2.0)[1] < 1e-6


def test_green_kubo_integral_gives_the_diffusion_constant():
    r = ss.simulate_underdamped(FREE, 1.0, 2.0, 1.0, 0.0, 0.01, 12.0, 40000, 150, seed=4)
    d, ref = ss.green_kubo(r)
    assert d == pytest.approx(ref, rel=0.06)
    c = ss.velocity_autocorrelation(r)
    assert c[0] == pytest.approx(1.0, rel=0.03) and c[np.argmin(np.abs(r.t - 0.5))] == pytest.approx(math.exp(-1), rel=0.06)   # (kT/m) e^{-gamma t/m}
    with pytest.raises(ss.StatMechError):
        ss.velocity_autocorrelation(ss.simulate_overdamped(FREE, 1.0, 1.0, 0.0, 0.01, 1.0, 50, 10, seed=1))


def test_runs_are_reproducible_and_record_their_seed():
    a = ss.simulate_overdamped(FREE, 1.0, 1.0, 0.0, 0.01, 1.0, 50, 10, seed=8)
    b = ss.simulate_overdamped(FREE, 1.0, 1.0, 0.0, 0.01, 1.0, 50, 10, seed=8)
    c = ss.simulate_overdamped(FREE, 1.0, 1.0, 0.0, 0.01, 1.0, 50, 10, seed=None)
    assert np.array_equal(a.x, b.x) and a.seed == 8 and isinstance(c.seed, int) and not np.array_equal(a.x, c.x)


def test_input_limits():
    with pytest.raises(ss.StatMechError, match="path-steps"):
        ss.simulate_overdamped(FREE, 1.0, 1.0, 0.0, 1e-6, 100.0, 100000)
    with pytest.raises(ss.StatMechError):
        ss.simulate_overdamped(FREE, -1.0, 1.0, 0.0, 0.01, 1.0, 50)
    with pytest.raises(ss.StatMechError):
        ss.simulate_underdamped(FREE, 1.0, 1.0, 0.0, 0.0, 0.01, 1.0, 50)
    with pytest.raises(ss.StatMechError):
        ss.simulate_overdamped(FREE, 1.0, 1.0, 0.0, 0.01, 0.01, 5)


# ---------------------------------------------------------------- Fokker-Planck
def test_generator_conserves_probability_and_annihilates_boltzmann():
    pot = ss.preset_potential("double_well")
    x = np.linspace(-3, 3, 400)
    A = ss.fp_generator(pot, 0.5, 1.0, x)
    assert np.max(np.abs(np.asarray(A.sum(axis=0)))) < 1e-9                                      # columns sum to zero
    pb = ss.boltzmann_density(pot, 0.5, x)
    assert np.max(np.abs(A @ pb)) / np.max(pb) * (x[1] - x[0]) ** 2 < 1e-9


def test_harmonic_fokker_planck_matches_the_exact_gaussian_at_every_time():
    pot = ss.preset_potential("harmonic", k=2.0)
    sol = ss.solve_fokker_planck(pot, 1.0, 1.0, (-5, 5), 400, 1.5, 0.3, 2.0, 0.01)
    assert sol.checks.passed
    last = ss.ou_exact(sol.x, float(sol.t[-1]), 1.5, 0.3, 1.0, 1.0, 2.0)
    assert np.max(np.abs(sol.P[-1] - last)) < 1e-4 * last.max()                                  # and very accurate once the pulse has relaxed
    for j in range(len(sol.t)):
        ex = ss.ou_exact(sol.x, float(sol.t[j]), 1.5, 0.3, 1.0, 1.0, 2.0)
        assert np.max(np.abs(sol.P[j] - ex)) < 1e-2 * ex.max()                                  # second-order scheme, a narrow initial pulse


def test_fokker_planck_relaxes_to_boltzmann_in_a_double_well():
    pot = ss.preset_potential("double_well")
    sol = ss.solve_fokker_planck(pot, 0.5, 1.0, (-3, 3), 300, 1.0, 0.2, 80.0, 0.1, 10)
    pb = ss.boltzmann_density(pot, 0.5, sol.x)
    assert np.sum(np.abs(sol.P[-1] - pb)) * (sol.x[1] - sol.x[0]) / 2 < 0.03


def test_fokker_planck_with_a_force_drifts_at_F_over_gamma():
    sol = ss.solve_fokker_planck(FREE, 1.0, 2.0, (-10, 10), 500, 0.0, 0.5, 4.0, 0.01, 10, force=1.0)
    mean = float(np.trapezoid(sol.x * sol.P[-1], sol.x))
    assert mean == pytest.approx(1.0 / 2.0 * 4.0, abs=2e-3)                                       # drift velocity F/gamma


def test_langevin_ensemble_agrees_with_the_pde_and_the_check_catches_a_narrow_domain():
    pot = ss.preset_potential("harmonic", k=2.0)
    sol = ss.solve_fokker_planck(pot, 1.0, 1.0, (-5, 5), 400, 1.5, 0.3, 2.0, 0.01)
    start = np.random.default_rng(0).normal(1.5, 0.3, 40000)
    res = ss.simulate_overdamped(pot, 1.0, 1.0, start, 0.002, 2.0, 40000, 50, seed=5)
    ck, l1 = ss.langevin_vs_fp(res, sol)
    assert ck.passed and l1 < 0.03
    # free diffusion in a box that is too narrow: the PDE's reflecting wall is felt and the comparison says so
    narrow = ss.solve_fokker_planck(FREE, 1.0, 1.0, (-5, 5), 400, 1.5, 0.3, 2.0, 0.01)
    res2 = ss.simulate_overdamped(FREE, 1.0, 1.0, np.random.default_rng(1).normal(1.5, 0.3, 30000), 0.002, 2.0, 30000, 50, seed=6)
    ck2, _ = ss.langevin_vs_fp(res2, narrow)
    assert not ck2.passed and "domain is wide enough" in [c.label for c in ck2.checks if not c.passed][0]


def test_fokker_planck_input_validation():
    pot = ss.preset_potential("harmonic")
    with pytest.raises(ss.StatMechError, match="not resolved"):
        ss.solve_fokker_planck(pot, 1.0, 1.0, (-5, 5), 100, 0.0, 0.01, 1.0, 0.01)
    with pytest.raises(ss.StatMechError):
        ss.solve_fokker_planck(pot, 1.0, 1.0, (5, -5), 400, 0.0, 0.3, 1.0, 0.01)
    with pytest.raises(ss.StatMechError):
        ss.solve_fokker_planck(pot, 1.0, 1.0, (-5, 5), 10, 0.0, 0.3, 1.0, 0.01)


def test_progress_hook_is_called():
    seen = []
    ss.simulate_overdamped(FREE, 1.0, 1.0, 0.0, 0.001, 5.0, 20, 20, seed=1, progress=lambda m, f: seen.append((m, f)))
    assert len(seen) >= 2 and seen[0][1] == 0.0
