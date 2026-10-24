"""Tests for modules/gr_cones.py: light cones, null curves, Penrose diagrams."""
import math

import numpy as np
import pytest

from modules import gr_cones as gc
from modules import gr_geodesics as gg
from modules.gr_metrics import GRInputError, preset


def test_cone_edges_of_the_flat_2_metric_are_45_degrees():
    e = gc.cone_edges(-1.0, 0.0, 1.0)
    assert sorted(round(abs(a / b), 9) for a, b in e) == [1.0, 1.0]


def test_cone_edges_reject_a_non_lorentzian_form():
    assert gc.cone_edges(1.0, 0.0, 1.0) is None and gc.cone_edges(-1.0, 0.0, 0.0) is None


def test_schwarzschild_cones_narrow_towards_the_horizon():
    cones = gc.light_cones(preset("schwarzschild"), ("t", "r"), np.array([0.0]), np.array([2.5, 4.0, 10.0]))
    slopes = [abs(c.edges[0][1] / c.edges[0][0]) for c in cones]                   # dr/dt = f
    assert slopes == pytest.approx([1 - 2 / 2.5, 1 - 2 / 4.0, 1 - 2 / 10.0], abs=1e-12)
    assert slopes[0] < slopes[1] < slopes[2]


def test_eddington_finkelstein_cones_tilt_inward_and_one_edge_lies_along_the_horizon_at_2M():
    ef = preset("schwarzschild_ef")
    at_horizon = gc.light_cones(ef, ("v", "r"), np.array([0.0]), np.array([2.0]))[0]
    assert any(abs(e[1]) < 1e-12 for e in at_horizon.edges)                        # dr = 0: the horizon generator
    inside = gc.light_cones(ef, ("v", "r"), np.array([0.0]), np.array([1.0]))[0]
    assert all(e[1] <= 1e-12 for e in inside.edges)                                # inside, both edges point to smaller r


def test_flat_cones_in_all_planes_are_open():
    cones = gc.light_cones(preset("minkowski"), ("t", "x"), np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    assert all(c.open for c in cones)


def test_a_spacelike_plane_has_no_cones():
    cones = gc.light_cones(preset("minkowski"), ("x", "y"), np.array([0.0]), np.array([0.0]), time_axis="x")
    assert not cones[0].open


def test_plane_validation():
    with pytest.raises(GRInputError):
        gc.light_cones(preset("minkowski"), ("t", "t"), np.array([0.0]), np.array([0.0]))
    with pytest.raises(GRInputError):
        gc.light_cones(preset("minkowski"), ("t", "w"), np.array([0.0]), np.array([0.0]))


def test_null_curve_in_minkowski_is_a_straight_45_degree_line():
    cu = gc.null_curve(preset("minkowski"), ("t", "x"), (0.0, 0.0), 1, 3.0, steps=60)
    assert np.allclose(np.abs(np.diff(cu[:, 0])), np.abs(np.diff(cu[:, 1])), atol=1e-9)


def test_radial_null_ray_agrees_with_the_geodesic_integrator():
    """The cone-edge curve and a full null geodesic launched radially outward are the same curve in the (t, r) plane."""
    met = preset("schwarzschild")
    cu = gc.null_curve(met, ("t", "r"), (0.0, 10.0), 1, 20.0, fixed={"theta": math.pi / 2}, steps=400, bounds=((-5, 200), (2.1, 100)))
    system = gg.build_system(met, {"M": 1.0})
    x0 = [0.0, 10.0, math.pi / 2, 0.0]
    u0 = gg.normalise_velocity(system, x0, [1.0, 0.0, 0.0], "null")
    res = gg.integrate_geodesic(system, x0, u0, 15.0, kind="null", n_samples=300)
    # compare r at the same t by interpolation
    tmax = min(cu[-1, 0], res.x[-1, 0])
    for t in np.linspace(1.0, tmax, 6):
        assert np.interp(t, cu[:, 0], cu[:, 1]) == pytest.approx(np.interp(t, res.x[:, 0], res.x[:, 1]), rel=2e-3)


def test_null_curve_stops_at_bounds():
    cu = gc.null_curve(preset("minkowski"), ("t", "x"), (0.0, 0.0), 1, 50.0, bounds=((-1, 1), (-1, 1)))
    assert np.all(np.abs(cu) <= 1.0 + 1e-9)


# ---------------------------------------------------------------- Penrose
def _all_points(curves):
    return np.vstack([ln for lines in curves.values() for ln in lines])


def test_minkowski_penrose_fits_inside_the_diamond():
    p = gc.minkowski_penrose()
    pts = _all_points(p)
    assert np.all(np.abs(pts[:, 1]) + np.abs(pts[:, 0]) <= math.pi / 2 + 1e-9)
    assert np.all(pts[:, 0] >= -1e-12)                                               # r >= 0: the right half


def test_schwarzschild_penrose_geometry():
    p = gc.schwarzschild_penrose(1.0)
    pts = _all_points(p)
    assert np.all(np.abs(pts) <= math.pi / 2 + 1e-9)
    for hz in p["horizon"]:                                                          # horizons are 45-degree lines
        d = np.diff(hz, axis=0)
        mask = np.abs(d[:, 0]) > 1e-12
        assert np.allclose(np.abs(d[mask, 1] / d[mask, 0]), 1.0, atol=1e-6)
    for ray in p["rays"]:                                                            # so are radial light rays
        d = np.diff(ray, axis=0)
        mask = np.abs(d[:, 0]) > 1e-9
        assert np.allclose(np.abs(d[mask, 1] / d[mask, 0]), 1.0, atol=1e-5)
    for sing in p["singularity"]:                                                    # r = 0 lies strictly inside, T^2 - X^2 = 1
        assert np.all(np.abs(sing[:, 1]) <= math.pi / 2)


def test_kruskal_radius_inverts_the_defining_relation():
    for r in (0.5, 1.9, 2.1, 6.0, 15.0):
        w = (r / 2 - 1) * math.exp(r / 2)
        x = math.sqrt(w) if w > 0 else 0.0
        t = math.sqrt(-w) if w < 0 else 0.0
        assert gc.kruskal_radius(t, x, 1.0) == pytest.approx(r, rel=1e-9)
    with pytest.raises(GRInputError):
        gc.kruskal_radius(2.0, 0.0, 1.0)


def test_penrose_support_table():
    assert gc.penrose_supported(preset("minkowski")) == "minkowski"
    assert gc.penrose_supported(preset("schwarzschild")) == "schwarzschild"
    assert gc.penrose_supported(preset("kerr")) is None
