"""Tests for modules/gr_geodesics.py: the integrator against exact results, conserved quantities, initial data."""
import math

import numpy as np
import pytest

from modules import gr_geodesics as gg
from modules import gr_potential as gp
from modules.gr_metrics import GRInputError, preset


@pytest.fixture(scope="module")
def schw():
    return gg.build_system(preset("schwarzschild"), {"M": 1.0})


def _orbit(schw, ra=100.0, rp=40.0, orbits=3.2, **kw):
    E, L = gp.schwarzschild_orbit_from_turning_points(1.0, ra, rp)
    x0, u0 = gg.initial_from_conserved(schw, ra, E, L, "timelike", inward=True)
    period = 2 * math.pi * math.sqrt(((ra + rp) / 2) ** 3)
    return gg.integrate_geodesic(schw, x0, u0, orbits * period, **kw)


def test_periapsis_advance_matches_the_exact_elliptic_integral(schw):
    res = _orbit(schw)
    adv = gg.periapsis_advance(res)
    assert len(adv) >= 2
    assert adv == pytest.approx([gp.schwarzschild_precession(1.0, 100.0, 40.0)] * len(adv), abs=1e-7)


def test_all_conserved_quantities_stay_constant(schw):
    res = _orbit(schw)
    names = " ".join(c.name for c in res.conserved)
    assert "energy" in names and "angular momentum" in names and "norm" in names and "total angular momentum" in names
    assert res.checks.passed, [c for c in res.checks.checks if not c.passed]
    assert max(c.drift_abs for c in res.conserved) < 1e-6


def test_a_loose_tolerance_is_caught_by_the_monitors(schw):
    res = _orbit(schw, rtol=1e-3, atol=1e-5, method="RK45", check_tol=1e-8)
    assert not res.checks.passed                                          # the drift check exists to catch exactly this


def test_light_deflection_matches_the_exact_integral(schw):
    b, R = 30.0, 1000.0
    x0, u0 = gg.initial_from_conserved(schw, R, 1.0, b, "null", inward=True)
    res = gg.integrate_geodesic(schw, x0, u0, 5000.0, kind="null", r_return=R, r_stop=2.01)
    assert "returned" in res.reason
    measured = res.x[-1, 3] - res.x[0, 3] - math.pi
    assert measured == pytest.approx(gp.schwarzschild_deflection(1.0, b, R), abs=1e-8)
    assert res.checks.checks[0].passed                                    # the null norm stays 0


def test_ray_inside_the_capture_threshold_is_stopped_at_the_horizon(schw):
    x0, u0 = gg.initial_from_conserved(schw, 50.0, 1.0, 4.0, "null", inward=True)
    res = gg.integrate_geodesic(schw, x0, u0, 500.0, kind="null", r_stop=2.01)
    assert "stopping radius" in res.reason and res.x[-1, 1] == pytest.approx(2.01, abs=1e-6)


def test_radial_infall_from_rest_matches_the_proper_time_formula(schw):
    """Radial infall from rest at r0: tau = (r0^1.5 / sqrt(2M)) (acos(sqrt(r/r0)) + sqrt((r/r0)(1 - r/r0)))."""
    r0 = 20.0
    E = math.sqrt(1 - 2 / r0)
    x0, u0 = gg.initial_from_conserved(schw, r0, E, 0.0, "timelike", inward=True)
    # at rest: u^r = 0 exactly at the start, so give a tiny push inward by integrating from the turning point
    res = gg.integrate_geodesic(schw, x0, u0, 30.0, r_stop=4.0)
    i = int(np.argmin(np.abs(res.x[:, 1] - 10.0)))
    r = res.x[i, 1]
    tau = (r0 ** 1.5 / math.sqrt(2)) * (math.acos(math.sqrt(r / r0)) + math.sqrt((r / r0) * (1 - r / r0)))
    assert res.lam[i] == pytest.approx(tau, rel=1e-6)


def test_kerr_equatorial_orbit_keeps_energy_angular_momentum_and_norm():
    ks = gg.build_system(preset("kerr"))
    x0, u0 = gg.initial_from_conserved(ks, 12.0, 0.95, 3.2, "timelike", inward=True)
    res = gg.integrate_geodesic(ks, x0, u0, 400.0, r_stop=gg.default_r_stop(preset("kerr")), n_samples=1500)
    assert res.checks.passed, [c for c in res.checks.checks if not c.passed]


def test_kerr_carter_constant_is_conserved_off_the_equator():
    ks = gg.build_system(preset("kerr"))
    x0 = [0.0, 10.0, 1.0, 0.0]
    u0 = gg.normalise_velocity(ks, x0, [0.0, 0.03, 0.04], "timelike")
    res = gg.integrate_geodesic(ks, x0, u0, 300.0, n_samples=1500)
    carter = next(c for c in res.conserved if "Carter" in c.name)
    assert abs(carter.values[0]) > 1e-3                                    # nonzero off the equator, so the test is not vacuous
    assert carter.passed and res.checks.passed
    assert np.ptp(res.column("theta")) > 0.05                              # the orbit really leaves its initial latitude


def test_expanding_universe_conserves_comoving_momentum_not_energy():
    fs = gg.build_system(preset("frw_power"))
    x0 = [1.0, 1.0, 1.0, 0.0]
    u0 = gg.normalise_velocity(fs, x0, [0.3, 0.1, 0.2], "timelike")
    res = gg.integrate_geodesic(fs, x0, u0, 3.0, n_samples=300)
    names = " ".join(c.name for c in res.conserved)
    assert "energy" not in names                                           # ∂_t is not Killing in FRW
    assert "p_phi" in names or "angular momentum" in names
    assert res.checks.passed


def test_minkowski_geodesic_is_a_straight_line():
    sysm = gg.build_system(preset("minkowski"))
    x0 = [0.0, 1.0, 2.0, 3.0]
    u0 = gg.normalise_velocity(sysm, x0, [0.3, -0.2, 0.1], "timelike")
    res = gg.integrate_geodesic(sysm, x0, u0, 5.0, n_samples=100)
    expected = np.array(x0[1:]) + np.outer(res.lam, u0[1:])
    assert res.x[:, 1:] == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------- initial data and errors
def test_normalise_velocity_gives_the_requested_norm_and_future_direction(schw):
    for kind, kap in (("timelike", -1.0), ("null", 0.0), ("spacelike", 1.0)):
        x0 = [0.0, 10.0, 1.0, 0.5]
        spatial = [3.0, 0.5, 0.7] if kind == "spacelike" else [0.05, 0.01, 0.02]
        u = gg.normalise_velocity(schw, x0, spatial, kind)
        g = schw.g(np.array(x0))
        assert np.array(u) @ g @ np.array(u) == pytest.approx(kap, abs=1e-12)
        assert u[0] > 0


def test_impossible_or_inside_horizon_velocities_are_refused(schw):
    with pytest.raises(GRInputError, match="spacelike"):
        gg.normalise_velocity(schw, [0.0, 10.0, 1.0, 0.5], [0.01, 0.0, 0.0], "spacelike")
    with pytest.raises(GRInputError):
        gg.normalise_velocity(schw, [0.0, 2.0, 1.0, 0.5], [0.0, 0.0, 0.0], "timelike")   # g_tt = 0 at the horizon
    with pytest.raises(GRInputError, match="kind"):
        gg.normalise_velocity(schw, [0.0, 10.0, 1.0, 0.5], [0.0, 0.0, 0.0], "weird")
    with pytest.raises(GRInputError, match="position"):
        gg.normalise_velocity(schw, [0.0, 10.0], [0.0], "timelike")


def test_forbidden_energy_angular_momentum_is_refused(schw):
    with pytest.raises(GRInputError, match="forbidden"):
        gg.initial_from_conserved(schw, 5.0, 0.8, 4.0, "timelike")


def test_initial_from_conserved_needs_tr_theta_phi_coordinates():
    sysm = gg.build_system(preset("minkowski"))
    with pytest.raises(GRInputError, match="order"):
        gg.initial_from_conserved(sysm, 5.0, 1.0, 1.0)


def test_integrator_argument_validation(schw):
    x0, u0 = gg.initial_from_conserved(schw, 20.0, 0.97, 4.2, "timelike")
    with pytest.raises(GRInputError, match="method"):
        gg.integrate_geodesic(schw, x0, u0, 10.0, method="euler")
    with pytest.raises(GRInputError, match="positive"):
        gg.integrate_geodesic(schw, x0, u0, 0.0)


def test_progress_hook_is_called_between_chunks(schw):
    seen = []
    x0, u0 = gg.initial_from_conserved(schw, 20.0, 0.97, 4.2, "timelike")
    gg.integrate_geodesic(schw, x0, u0, 50.0, chunks=5, progress=lambda m, f: seen.append((m, f)))
    assert len(seen) >= 5 and seen[0][1] == 0.0 and seen[-1][1] == 1.0


def test_hitting_the_singularity_reports_why_it_stopped(schw):
    x0, u0 = gg.initial_from_conserved(schw, 20.0, 0.97, 0.0, "timelike", inward=True)
    res = gg.integrate_geodesic(schw, x0, u0, 5000.0, rtol=1e-8, atol=1e-10)
    assert not res.reason.startswith("reached the requested")
    assert not res.checks.checks[-1].passed


def test_default_r_stop_is_just_outside_the_horizon():
    assert gg.default_r_stop(preset("schwarzschild"), {"M": 1.0}) == pytest.approx(2.004, abs=1e-3)
    kerr_stop = gg.default_r_stop(preset("kerr"), {"M": 1.0, "a": 0.6})
    assert kerr_stop == pytest.approx(1.002 * (1 + math.sqrt(1 - 0.36)), abs=1e-9)
    assert gg.default_r_stop(preset("minkowski")) is None


def test_to_cartesian_embeddings():
    x = np.array([[0.0, 2.0, math.pi / 2, 0.0], [0.0, 3.0, 0.0, 0.0]])
    p = gg.to_cartesian(preset("schwarzschild"), x)
    assert p[0] == pytest.approx([2.0, 0.0, 0.0]) and p[1] == pytest.approx([0.0, 0.0, 3.0])
    ob = gg.to_cartesian(preset("kerr"), x, {"a": 0.8})
    assert ob[0, 0] == pytest.approx(math.sqrt(4 + 0.64))
    mk = gg.to_cartesian(preset("minkowski"), np.array([[0.0, 1.0, 2.0, 3.0]]))
    assert mk[0] == pytest.approx([1.0, 2.0, 3.0])
    fallback = gg.to_cartesian(preset("schwarzschild_ef"), np.array([[0.0, 1.0, 2.0, 3.0]]))
    assert fallback.shape == (1, 3)
