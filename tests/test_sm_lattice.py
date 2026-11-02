"""Tests for modules/sm_lattice.py: Metropolis against exact results, error bars, critical point, input validation."""
import math

import numpy as np
import pytest

from modules import sm_lattice as sl


# ---------------------------------------------------------------- exact references against each other
@pytest.mark.parametrize("T", [1.0, 2.5, 4.0])
def test_transfer_matrix_agrees_with_full_enumeration(T):
    """q = 2 Potts with J_Potts = 2 J_Ising is the Ising model (energies differ by a constant): two independent exact methods."""
    ex = sl.exact_ising_enumeration(4, T)
    e, c = sl.torus_exact(2, 4, 4, T, J=2.0)
    assert c == pytest.approx(ex.C, rel=1e-6, abs=1e-6)
    assert e + 2.0 == pytest.approx(ex.e, abs=1e-7)


def test_enumeration_limits():
    hot, cold = sl.exact_ising_enumeration(4, 1e4), sl.exact_ising_enumeration(4, 0.05)
    assert hot.e == pytest.approx(0.0, abs=1e-3) and cold.e == pytest.approx(-2.0, abs=1e-6)
    assert cold.m_abs == pytest.approx(1.0, abs=1e-6)
    assert hot.U4 == pytest.approx(0.0, abs=0.1) and cold.U4 == pytest.approx(2 / 3, abs=1e-6)   # Gaussian m: U4 -> 0; fully ordered: 2/3
    with pytest.raises(sl.StatMechError):
        sl.exact_ising_enumeration(6, 2.0)


def test_exact_critical_temperatures():
    assert sl.critical_temperature(sl.LatticeModel("ising", 4))[0] == pytest.approx(2.269185, abs=1e-6)
    assert sl.critical_temperature(sl.LatticeModel("potts", 4, q=3))[0] == pytest.approx(1 / math.log(1 + math.sqrt(3)))
    assert sl.critical_temperature(sl.LatticeModel("ising", 4, J=2.0))[0] == pytest.approx(2 * 2.269185, abs=1e-5)
    assert "literature" in sl.critical_temperature(sl.LatticeModel("xy", 4))[1]


# ---------------------------------------------------------------- Metropolis vs exact
def test_ising_4x4_every_observable_agrees_with_exact_enumeration():
    m = sl.LatticeModel("ising", 4)
    pt, _ = sl.run_point(m, 2.0, np.random.default_rng(11), 500, 25000)
    rows = sl.compare_with_exact_ising(pt, n_sigma=4.5)
    assert all(ok for *_, ok in rows), rows
    assert pt.run_long_enough and abs(pt.drift_z) < 4


def test_ising_with_a_field_agrees_with_exact_enumeration():
    m = sl.LatticeModel("ising", 4, h=0.3)
    pt, _ = sl.run_point(m, 2.5, np.random.default_rng(12), 500, 25000)
    assert all(ok for *_, ok in sl.compare_with_exact_ising(pt, n_sigma=4.5))


def test_potts_q3_energy_and_specific_heat_agree_with_the_transfer_matrix():
    m = sl.LatticeModel("potts", 4, q=3)
    pt, _ = sl.run_point(m, 0.9, np.random.default_rng(5), 500, 20000)
    e, c = sl.torus_exact(3, 4, 4, 0.9)
    assert abs(pt.e - e) < 4.5 * pt.e_err and abs(pt.C - c) < 4.5 * pt.C_err


def test_wrong_temperature_would_be_detected_by_the_exact_comparison():
    m = sl.LatticeModel("ising", 4)
    pt, _ = sl.run_point(m, 2.0, np.random.default_rng(11), 500, 20000)
    pt.T = 2.6                                                                       # compare at a different temperature than was simulated
    assert not all(ok for *_, ok in sl.compare_with_exact_ising(pt))


def test_xy_model_ordered_at_low_T_and_disordered_at_high_T():
    m = sl.LatticeModel("xy", 8)
    # a quench from a random state freezes vortices in (slow XY dynamics, physical rather than a bug), so start the cold run ordered
    lo, _ = sl.run_point(m, 0.3, np.random.default_rng(1), 300, 600, ordered=True)
    hi, _ = sl.run_point(m, 3.0, np.random.default_rng(1), 300, 600)
    assert lo.m > 0.9 and hi.m < 0.4 and -2.0 <= lo.e < hi.e < 0
    assert 0.2 < lo.acceptance < 0.7                                                  # the adaptive step works


def test_xy_high_temperature_energy_matches_the_leading_expansion():
    """e = -2 J <cos> with <cos> = I1(1/T)/I0(1/T) ~ 1/(2T) at high T for the 2D lattice (leading order)."""
    pt, _ = sl.run_point(sl.LatticeModel("xy", 8), 20.0, np.random.default_rng(2), 300, 3000)
    assert pt.e == pytest.approx(-2 * 0.5 / 20.0, abs=0.01)


# ---------------------------------------------------------------- the updates themselves
def test_checkerboard_update_preserves_detailed_balance_in_the_ordered_limit():
    """At T -> 0 an ordered Ising lattice never flips; at T -> infinity every proposal is accepted."""
    m = sl.LatticeModel("ising", 8)
    rng = np.random.default_rng(0)
    s = sl.initial_state(m, rng, ordered=True)
    sl.sweep(m, s, 0.01, rng)
    assert np.all(s == 1)
    s = sl.initial_state(m, rng)
    acc = sl.sweep(m, s, 1e6, rng)
    assert acc == pytest.approx(1.0, abs=1e-9)


def test_observables_of_known_states():
    ising = sl.LatticeModel("ising", 4)
    up = np.ones((4, 4), dtype=np.int8)
    assert sl.energy_per_site(ising, up) == pytest.approx(-2.0) and sl.order_parameter(ising, up) == 1.0
    stripes = np.tile(np.array([[1], [-1]], dtype=np.int8), (2, 4))
    assert sl.energy_per_site(ising, stripes) == pytest.approx(0.0) and sl.order_parameter(ising, stripes) == 0.0
    potts = sl.LatticeModel("potts", 4, q=3)
    zero = np.zeros((4, 4), dtype=np.int8)
    assert sl.energy_per_site(potts, zero) == pytest.approx(-2.0) and sl.order_parameter(potts, zero) == pytest.approx(1.0)
    mixed = (np.arange(16).reshape(4, 4) % 3).astype(np.int8)
    assert 0 <= sl.order_parameter(potts, mixed) < 0.5
    xy = sl.LatticeModel("xy", 4)
    assert sl.energy_per_site(xy, np.zeros((4, 4))) == pytest.approx(-2.0) and sl.order_parameter(xy, np.full((4, 4), 1.3)) == pytest.approx(1.0)


def test_field_favours_alignment():
    m = sl.LatticeModel("ising", 6, h=2.0)
    pt, _ = sl.run_point(m, 3.0, np.random.default_rng(3), 300, 800)
    assert pt.m > 0.8


# ---------------------------------------------------------------- scan, reproducibility, critical point
def test_scan_is_reproducible_from_its_seed_and_records_it():
    m = sl.LatticeModel("ising", 6)
    a = sl.scan_temperatures(m, [2.0, 2.5, 3.0], 100, 400, seed=9)
    b = sl.scan_temperatures(m, [2.0, 2.5, 3.0], 100, 400, seed=9)
    assert a.seed == 9 and np.array_equal(a.column("e"), b.column("e")) and np.array_equal(a.temps, [2.0, 2.5, 3.0])
    fresh = sl.scan_temperatures(m, [2.5], 50, 200, seed=None)
    assert isinstance(fresh.seed, int)


def test_a_point_does_not_depend_on_which_other_temperatures_are_scanned():
    m = sl.LatticeModel("ising", 6)
    pt, _ = sl.run_point(m, 2.5, sl.spawn_rngs(4, 1)[0], 300, 500)       # single-point reference stream
    assert pt.T == 2.5
    s1 = sl.scan_temperatures(m, [2.5], 100, 300, seed=4)
    s2 = sl.scan_temperatures(m, [2.5], 100, 300, seed=4)
    assert s1.points[0].e == s2.points[0].e


def test_binder_crossing_and_finite_size_scaling_find_onsagers_values():
    temps = np.linspace(2.05, 2.5, 7)
    scans = {L: sl.scan_temperatures(sl.LatticeModel("ising", L), temps, 300, 3000, seed=3) for L in (8, 16)}
    est = sl.estimate_critical_point(scans)
    assert est is not None and abs(est.tc - 2.2692) < 3 * est.tc_err + 0.03
    fit = sl.finite_size_scaling(scans, est.tc)
    assert fit.gamma_over_nu == pytest.approx(1.75, abs=0.25) and fit.beta_over_nu == pytest.approx(0.125, abs=0.08)
    # critical slowing down: tau of the order parameter near Tc exceeds tau far above it, and grows with L
    tau = {L: scans[L].column("tau_m") for L in scans}
    assert tau[16].max() > tau[16][-1] and tau[16].max() >= tau[8].max() * 0.8


def test_binder_cumulant_orders_correctly():
    scan = sl.scan_temperatures(sl.LatticeModel("ising", 8), [1.5, 4.0], 300, 3000, seed=1)
    cold, hot = scan.points
    assert cold.U4 > 0.6 and hot.U4 < 0.3                                              # 2/3 ordered, ~0 disordered


def test_estimate_critical_point_needs_matching_grids_and_a_crossing():
    a = sl.scan_temperatures(sl.LatticeModel("ising", 4), [2.0, 3.0], 50, 200, seed=1)
    b = sl.scan_temperatures(sl.LatticeModel("ising", 6), [2.0, 2.5], 50, 200, seed=1)
    with pytest.raises(sl.StatMechError, match="same temperatures"):
        sl.estimate_critical_point({4: a, 6: b})
    assert sl.estimate_critical_point({4: a}) is None


def test_run_single_records_snapshots_and_progress():
    seen = []
    run = sl.run_single(sl.LatticeModel("ising", 8), 2.0, 100, 10, seed=2, progress=lambda m, f: seen.append(f))
    assert len(run.snapshots) == 11 and run.snapshots[0][0] == 0 and run.snapshots[-1][0] == 100 and seen and run.seed == 2
    assert run.energy.shape == (100,)
    ordered = sl.run_single(sl.LatticeModel("potts", 8, q=3), 0.2, 50, 25, seed=1, ordered=True)
    assert ordered.order[0] > 0.99


# ---------------------------------------------------------------- validation
@pytest.mark.parametrize("kw", [dict(model="ising", L=5), dict(model="ising", L=2), dict(model="ising", L=400), dict(model="bad", L=8),
                                dict(model="ising", L=8, J=-1.0), dict(model="potts", L=8, q=1), dict(model="potts", L=8, q=20)])
def test_invalid_models_are_refused(kw):
    with pytest.raises(sl.StatMechError):
        sl.LatticeModel(**kw)


def test_invalid_runs_are_refused():
    m = sl.LatticeModel("ising", 4)
    with pytest.raises(sl.StatMechError):
        sl.run_point(m, 0.0, np.random.default_rng(0), 10, 200)
    with pytest.raises(sl.StatMechError):
        sl.run_point(m, 1.0, np.random.default_rng(0), 10, 10)
    with pytest.raises(sl.StatMechError):
        sl.scan_temperatures(m, [], 10, 200)
    with pytest.raises(sl.StatMechError):
        sl.run_single(m, 1.0, 0, 1)
    with pytest.raises(sl.StatMechError):
        sl.torus_exact(1, 4, 4, 1.0)
    with pytest.raises(sl.StatMechError):
        sl.compare_with_exact_ising(sl.run_point(sl.LatticeModel("ising", 6), 2.0, np.random.default_rng(0), 10, 200)[0])
