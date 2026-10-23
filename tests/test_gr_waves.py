"""Tests for modules/gr_waves.py: linearised gravity, TT plane waves, test masses, quadrupole formula, inspiral."""
import math

import numpy as np
import pytest
import sympy as sp

from modules import gr_waves as gw
from modules.gr_metrics import GRInputError

T, X, Y, Z = gw.COORDS


# ---------------------------------------------------------------- plane waves and the field equations
@pytest.mark.parametrize("n", [[0, 0, 1], [1, 0, 0], [1, 2, 2], [0, -1, 1]])
def test_null_plane_wave_in_any_direction_solves_the_vacuum_equations(n):
    w = gw.plane_wave(n, 1, 0.5, 3, cross_phase=0.7)
    ck = gw.check_plane_wave(w)
    assert ck.passed, [c for c in ck.checks if not c.passed]


def test_off_shell_wave_is_not_a_solution_and_the_check_says_so():
    w = gw.plane_wave([0, 0, 1], 1, 0.3, 2, wave_number=3)
    ck = gw.check_plane_wave(w)
    failed = [c.label for c in ck.checks if not c.passed]
    assert failed == ["linearised vacuum equations  R_ab = 0"]
    ric = gw.linearized_ricci(w.h)
    assert any(gw.is_zero(ric[i, j], gw._sample_points())[0] is False for i in range(4) for j in range(4))


def test_polarization_basis_is_orthonormal_and_right_handed():
    n, e1, e2 = gw.polarization_basis([1, 2, 2])
    assert sp.simplify(n.dot(e1)) == 0 and sp.simplify(n.dot(e2)) == 0 and sp.simplify(e1.dot(e2)) == 0
    assert sp.simplify(e1.dot(e1) - 1) == 0 and sp.simplify(e1.cross(e2) - n) == sp.zeros(3, 1)
    with pytest.raises(GRInputError):
        gw.polarization_basis([0, 0, 0])


def test_linearised_ricci_agrees_with_the_full_nonlinear_theory():
    """The linear formulas are an approximation: the first-order part of the EXACT Ricci tensor must equal them."""
    for kwargs in ({"wave_number": 3}, {"wave_number": 0.5}):
        w = gw.plane_wave([0, 0, 1], 1, 0.3, 2, **kwargs)
        agree, worst = gw.full_theory_agrees(w.h)
        assert agree, worst
        # non-vacuous: the off-shell Ricci tensor is genuinely nonzero
        assert any(gw.evaluate(gw.linearized_ricci(w.h)[i, j], gw._sample_points()[0]) != 0 for i in range(4) for j in range(4))


def test_full_theory_agreement_for_an_oblique_wave():
    w = gw.plane_wave([1, 0, 0], 1, 0.3, 2, wave_number=1)
    assert gw.full_theory_agrees(w.h)[0]


def test_linearised_riemann_is_gauge_invariant():
    w = gw.plane_wave([0, 0, 1], 1, 0, 2)
    xi = [sp.sin(T * X), Y ** 2 * T, sp.cos(Z) * X, T * Z]
    assert gw.riemann_gauge_invariant(w.h, xi, gw.COORDS)
    # and a non-pure-gauge change is detected
    h2 = w.h.copy()
    h2[0, 0] += X ** 2                                                          # a Newtonian tidal field: genuine curvature
    a, b = gw.linearized_riemann(w.h), gw.linearized_riemann(h2)
    assert any(sp.simplify(a[i][j][k][l] - b[i][j][k][l]) != 0 for i in range(4) for j in range(4) for k in range(4) for l in range(4))


def test_pure_gauge_perturbation_has_no_curvature():
    xi = [T * X, X * Y, Y * Z, Z * T]
    h = gw.gauge_transform(sp.zeros(4, 4), xi)
    r = gw.linearized_riemann(h)
    assert all(sp.simplify(r[a][b][c][d]) == 0 for a in range(4) for b in range(4) for c in range(4) for d in range(4))


def test_tt_projection_properties():
    rng = np.random.default_rng(3)
    h = rng.normal(size=(3, 3))
    h = h + h.T
    n = np.array([1.0, 2.0, -0.5])
    tt = gw.tt_project(h, n)
    nn = n / np.linalg.norm(n)
    assert abs(np.trace(tt)) < 1e-12 and np.allclose(tt @ nn, 0) and np.allclose(tt, tt.T)
    assert np.allclose(gw.tt_project(tt, n), tt)                               # a projection


# ---------------------------------------------------------------- test masses
def test_ring_is_stretched_along_x_while_squeezed_along_y_for_plus_polarisation():
    times = np.array([0.0])
    pos = gw.ring_positions(1.0, 0.4, 0.0, times, n_points=4)
    assert pos[0, 0] == pytest.approx([1.2, 0.0]) and pos[0, 1] == pytest.approx([0.0, 0.8])


def test_geodesic_deviation_integration_matches_the_closed_form():
    times = np.linspace(0, 12, 241)
    _pos, err = gw.geodesic_deviation_ring(2.0, 0.3, 0.1, times, cross_phase=0.4)
    assert err < 1e-9


def test_geodesic_integrator_keeps_particles_at_rest_in_tt_gauge():
    rc = gw.free_particle_check(0.02, 0.01, 2.0, 6.0)
    assert rc.max_coordinate_drift < 1e-9                                      # Gamma^i_00 = 0 exactly
    assert rc.max_separation_error < 2 * 0.02 ** 2                              # second order in the amplitude
    assert rc.separation.max() > 1.0 > rc.separation.min()                      # the proper separation oscillates


# ---------------------------------------------------------------- quadrupole
def test_circular_binary_symbolic_quadrupole_matches_the_textbook_formulas():
    b = gw.circular_binary(1.4, 1.4, 50.0, inclination=0.8, distance=1000.0)
    assert b.checks.passed, [c for c in b.checks.checks if not c.passed]


@pytest.mark.parametrize("incl", [0.0, math.pi / 2, 1.1])
def test_inclination_dependence_of_the_two_polarisations(incl):
    b = gw.circular_binary(1.0, 3.0, 30.0, inclination=incl, distance=500.0)
    assert b.h_plus_amplitude == pytest.approx(b.h_amplitude * (1 + math.cos(incl) ** 2) / 2)
    assert b.h_cross_amplitude == pytest.approx(b.h_amplitude * abs(math.cos(incl)))
    assert b.checks.passed


def test_face_on_binary_is_circularly_polarised_and_edge_on_is_linear():
    face, edge = gw.circular_binary(1, 1, 20, 0.0, 100), gw.circular_binary(1, 1, 20, math.pi / 2, 100)
    assert face.h_plus_amplitude == pytest.approx(face.h_cross_amplitude)
    assert edge.h_cross_amplitude == pytest.approx(0, abs=1e-12)


def test_power_scales_as_inverse_fifth_power_of_separation():
    p1, p2 = gw.circular_binary(1, 1, 20, 0.5, 100).power, gw.circular_binary(1, 1, 40, 0.5, 100).power
    assert p1 / p2 == pytest.approx(32.0, rel=1e-9)


def test_si_units_give_a_sensible_neutron_star_binary():
    G, c, Msun = 6.67430e-11, 2.99792458e8, 1.98847e30
    b = gw.circular_binary(1.4 * Msun, 1.4 * Msun, 100e3, 0.0, 40 * 3.0857e22, G, c)
    f_gw = b.omega / math.pi
    assert 100 < f_gw < 400                                                    # a ~100 km NS pair is in the LIGO band
    assert 1e-23 < b.h_plus_amplitude < 1e-21


def test_arbitrary_point_mass_source_quadrupole():
    """A mass oscillating along x: Q_xx = (2/3) m x^2, Q_yy = Q_zz = -(1/3) m x^2."""
    t = sp.Symbol("t", real=True)
    q = gw.quadrupole_radiation([1], [[sp.cos(t), 0, 0]], t, inclination=math.pi / 2, distance=10)
    assert sp.simplify(q.Q[0, 0] - sp.Rational(2, 3) * sp.cos(t) ** 2) == 0
    assert sp.simplify(q.Q[1, 1] + sp.Rational(1, 3) * sp.cos(t) ** 2) == 0 and sp.simplify(q.Q.trace()) == 0
    assert gw.sample_series(q.power, t, np.array([0.3]))[0] > 0


def test_a_static_source_does_not_radiate():
    t = sp.Symbol("t", real=True)
    q = gw.quadrupole_radiation([1, 2], [[1, 0, 0], [-0.5, 0, 0]], t, distance=10)
    assert q.h_plus == 0 and q.h_cross == 0 and q.power == 0


def test_quadrupole_input_validation():
    t = sp.Symbol("t", real=True)
    with pytest.raises(GRInputError):
        gw.quadrupole_radiation([1], [], t)
    with pytest.raises(GRInputError):
        gw.quadrupole_radiation([1], [[1, 2]], t)
    with pytest.raises(GRInputError):
        gw.circular_binary(-1, 1, 10)


def test_sample_series_broadcasts_constants():
    t = sp.Symbol("t", real=True)
    assert gw.sample_series(sp.Integer(3), t, np.arange(4.0)).tolist() == [3.0] * 4


# ---------------------------------------------------------------- inspiral
def test_inspiral_matches_peters_closed_forms():
    ins = gw.peters_inspiral(1.0, 1.0, 100.0)
    assert ins.checks.passed, [c for c in ins.checks.checks if not c.passed]
    assert ins.merger_time == pytest.approx(100.0 ** 4 / (4 * ins.beta))
    assert ins.f_gw[-1] > ins.f_gw[0] and ins.h_amp[-1] > ins.h_amp[0]


def test_merger_time_for_a_known_system_in_si():
    """Hulse-Taylor-like check of the formula: two 1.4 Msun stars at 1e9 m merge in about 1.5e5 years... computed, not looked up."""
    G, c, Msun = 6.67430e-11, 2.99792458e8, 1.98847e30
    ins = gw.peters_inspiral(1.4 * Msun, 1.4 * Msun, 1e9, G, c, 1e22)
    t_years = ins.merger_time / 3.156e7
    expected = 5 / 256 * c ** 5 * 1e9 ** 4 / (G ** 3 * (1.4 * Msun) ** 2 * (2.8 * Msun)) / 3.156e7
    assert t_years == pytest.approx(expected, rel=1e-9)


def test_inspiral_argument_validation():
    with pytest.raises(GRInputError):
        gw.peters_inspiral(1, 1, 100, a_end_factor=1.5)
    with pytest.raises(GRInputError):
        gw.peters_inspiral(0, 1, 100)
