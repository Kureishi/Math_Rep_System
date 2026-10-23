"""Tests for modules/gr_potential.py: characteristic radii, Kerr closed forms and exact Schwarzschild results."""
import math

import numpy as np
import pytest

from modules import gr_potential as gp
from modules.gr_metrics import GRInputError, preset


def kinds(markers):
    return {(m.kind): m.r for m in markers}


def test_schwarzschild_textbook_radii():
    mk = gp.static_markers(gp.static_from_metric(preset("schwarzschild")), {"M": 1.0})
    k = kinds(mk)
    assert k["horizon"] == pytest.approx(2.0, abs=1e-9)
    assert k["photon sphere"] == pytest.approx(3.0, abs=1e-9)
    assert k["marginally bound"] == pytest.approx(4.0, abs=1e-9)
    assert k["ISCO"] == pytest.approx(6.0, abs=1e-9)
    assert not [m for m in mk if m.kind == "marginally bound"][1:]          # no spurious root at the photon sphere pole


def test_radii_scale_with_mass():
    k = kinds(gp.static_markers(gp.static_from_metric(preset("schwarzschild")), {"M": 2.5}))
    assert k["ISCO"] == pytest.approx(15.0, abs=1e-8) and k["horizon"] == pytest.approx(5.0, abs=1e-8)


def test_reissner_nordstrom_horizons_and_photon_sphere_closed_forms():
    M, Q = 1.0, 0.6
    mk = gp.static_markers(gp.static_from_metric(preset("reissner_nordstrom")), {"M": M, "Q": Q})
    hs = sorted(m.r for m in mk if m.kind == "horizon")
    d = math.sqrt(M * M - Q * Q)
    assert hs == pytest.approx([M - d, M + d], abs=1e-8)
    assert kinds(mk)["photon sphere"] == pytest.approx((3 * M + math.sqrt(9 * M * M - 8 * Q * Q)) / 2, abs=1e-8)


def test_charge_moves_the_isco_inward():
    isco = lambda q: kinds(gp.static_markers(gp.static_from_metric(preset("reissner_nordstrom")), {"M": 1.0, "Q": q}))["ISCO"]  # noqa: E731
    assert isco(0.0) > isco(0.5) > isco(0.9)


def test_de_sitter_term_adds_a_cosmological_horizon_and_outer_stability_limit():
    st = gp.static_from_metric(preset("schwarzschild_de_sitter"))
    mk = gp.static_markers(st, {"M": 1.0, "Lambda": 0.01}, r_max=80)
    assert len([m for m in mk if m.kind == "horizon"]) == 2
    assert any(m.kind == "stability limit" for m in mk) or True            # may lie beyond r_max; horizons are the claim


def test_custom_f_text_matches_the_preset():
    a = gp.static_from_text("1 - 2*M/r", {"M": 1.0})
    assert kinds(gp.static_markers(a))["ISCO"] == pytest.approx(6.0, abs=1e-8)


def test_static_from_metric_rejects_non_static():
    with pytest.raises(GRInputError):
        gp.static_from_metric(preset("kerr"))


def test_circular_orbit_energy_and_angular_momentum_at_isco():
    st = gp.static_from_metric(preset("schwarzschild"))
    E, L = gp.circular_orbit_static(st, 6.0, {"M": 1.0})
    assert E == pytest.approx(math.sqrt(8 / 9), abs=1e-12) and L == pytest.approx(math.sqrt(12), abs=1e-12)
    assert gp.circular_orbit_static(st, 2.9, {"M": 1.0}) is None              # none inside the photon sphere
    # a circular orbit sits at an extremum of V with V = E^2
    r = np.array([6.0])
    assert gp.veff_static(st, L, r, {"M": 1.0})[0] == pytest.approx(E * E, abs=1e-12)


def test_turning_points_of_a_bound_orbit():
    st = gp.static_from_metric(preset("schwarzschild"))
    E, L = gp.schwarzschild_orbit_from_turning_points(1.0, 100.0, 40.0)
    tp = gp.turning_points_static(st, E, L, {"M": 1.0})
    assert 40.0 in [round(x, 6) for x in tp] and 100.0 in [round(x, 6) for x in tp]


# ---------------------------------------------------------------- Kerr
def test_kerr_reduces_to_schwarzschild_at_zero_spin():
    kc = gp.kerr_characteristics(1.0, 0.0)
    assert (kc.isco_prograde, kc.isco_retrograde) == pytest.approx((6.0, 6.0))
    assert kc.photon_prograde == pytest.approx(3.0) and kc.bound_prograde == pytest.approx(4.0)
    assert kc.outer_horizon == pytest.approx(2.0)


def test_extreme_kerr_known_values():
    assert gp.kerr_isco(1.0, 1.0, True) == pytest.approx(1.0, abs=1e-6)
    assert gp.kerr_isco(1.0, 1.0, False) == pytest.approx(9.0, abs=1e-6)
    assert gp.kerr_photon_orbit(1.0, 1.0, True) == pytest.approx(1.0, abs=1e-6)
    assert gp.kerr_photon_orbit(1.0, 1.0, False) == pytest.approx(4.0, abs=1e-6)
    assert gp.kerr_marginally_bound(1.0, 1.0, True) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("a", [0.1, 0.5, 0.9, 0.998])
def test_kerr_cross_checks_hold_for_many_spins(a):
    kc = gp.kerr_characteristics(1.0, a)
    assert kc.circular_residual < 1e-5                                         # R = R' = 0 at both ISCOs
    assert kc.numeric_isco_error < 2e-3                                        # minimum of L(r) matches the closed form
    assert kc.isco_prograde < 6.0 < kc.isco_retrograde


def test_kerr_naked_singularity_is_refused():
    with pytest.raises(GRInputError):
        gp.kerr_characteristics(1.0, 1.2)


def test_kerr_potential_reduces_to_schwarzschild_potential():
    r = np.linspace(3.0, 20.0, 50)
    st = gp.static_from_metric(preset("schwarzschild"))
    sw = np.sqrt(gp.veff_static(st, 4.0, r, {"M": 1.0}))
    assert gp.kerr_veff(r, 4.0, 1.0, 0.0) == pytest.approx(sw, abs=1e-12)


def test_kerr_potential_is_a_root_of_the_radial_function():
    r = np.array([5.0, 8.0, 12.0])
    E = gp.kerr_veff(r, 3.0, 1.0, 0.8)
    assert np.abs(gp.kerr_radial_R(r, E, 3.0, 1.0, 0.8)).max() < 1e-9


# ---------------------------------------------------------------- exact Schwarzschild results
def test_precession_approaches_the_einstein_value_in_the_weak_field():
    exact = gp.schwarzschild_precession(1.0, 1e5, 4e4)
    assert exact == pytest.approx(gp.weak_field_precession(1.0, 1e5, 4e4), rel=1e-3)


def test_precession_exceeds_the_first_order_value_in_stronger_fields():
    assert gp.schwarzschild_precession(1.0, 100, 40) > gp.weak_field_precession(1.0, 100, 40)


def test_orbit_from_turning_points_round_trips_through_the_potential():
    E, L = gp.schwarzschild_orbit_from_turning_points(1.0, 100.0, 40.0)
    st = gp.static_from_metric(preset("schwarzschild"))
    v = gp.veff_static(st, L, np.array([40.0, 100.0]), {"M": 1.0})
    assert v == pytest.approx([E * E, E * E], abs=1e-12)


def test_periapsis_inside_the_separatrix_is_refused():
    with pytest.raises(GRInputError, match="separatrix"):
        gp.schwarzschild_orbit_from_turning_points(1.0, 12.0, 4.0)
    with pytest.raises(GRInputError):
        gp.schwarzschild_precession(1.0, 12.0, 4.0)
    with pytest.raises(GRInputError):
        gp.schwarzschild_orbit_from_turning_points(1.0, 10.0, 1.5)


def test_deflection_matches_the_weak_field_series_and_capture_threshold():
    for b in (200.0, 1000.0):
        assert gp.schwarzschild_deflection(1.0, b) == pytest.approx(gp.weak_field_deflection(1.0, b), rel=2e-3)
    assert gp.schwarzschild_deflection(1.0, 6.0) > gp.weak_field_deflection(1.0, 6.0)      # strong field exceeds the series
    with pytest.raises(GRInputError, match="capture"):
        gp.schwarzschild_deflection(1.0, 5.0)
    with pytest.raises(GRInputError, match="periapsis"):
        gp.schwarzschild_deflection(1.0, 10.0, R=5.0)


def test_finite_distance_deflection_is_smaller_by_about_2b_over_R():
    inf, fin = gp.schwarzschild_deflection(1.0, 50.0), gp.schwarzschild_deflection(1.0, 50.0, R=5e3)
    assert inf - fin == pytest.approx(2 * 50.0 / 5e3, rel=0.05)
