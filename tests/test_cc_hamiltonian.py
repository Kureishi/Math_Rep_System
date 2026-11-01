"""Tests for modules/cc_hamiltonian.py: Euler-Lagrange, Legendre, Noether search, phase space, Poincare sections."""
import numpy as np
import pytest
import sympy as sp

from modules import cc_hamiltonian as ch

q = lambda n: sp.Symbol(n, real=True)  # noqa: E731


def num(expr, system):
    """The expression with the system's parameters replaced by their (default 1) values."""
    return sp.simplify(expr.subs(system.param_subs()))


@pytest.fixture(scope="module")
def systems():
    return {k: ch.preset_system(k) for k in ("harmonic", "pendulum", "gravity", "free2d", "central", "polar_central", "coupled", "henon_heiles")}


def test_euler_lagrange_for_the_pendulum_and_oscillator(systems):
    pend = systems["pendulum"]
    th = q("th")
    assert num(pend.accelerations[0] + sp.sin(th), pend) == 0                                 # th'' = -(g/l) sin th with g = l = 1
    ho = systems["harmonic"]
    assert num(ho.accelerations[0] + q("x"), ho) == 0


def test_legendre_transform_gives_the_textbook_hamiltonian(systems):
    ho = systems["harmonic"]
    assert num(ho.H - (q("p_x") ** 2 / 2 + q("x") ** 2 / 2), ho) == 0
    assert ho.separable and not ch.preset_system("double_pendulum").separable
    central = systems["polar_central"]
    r, p_th = q("r"), q("p_th")
    assert num(central.H - (q("p_r") ** 2 / 2 + p_th ** 2 / (2 * r ** 2) - 1 / r), central) == 0


def test_hamilton_equations_from_a_hamiltonian_input():
    s = ch.from_hamiltonian("p_x^2/2 + k*x^2/2", ["x"], {"k": 4.0})
    assert s.dq == [q("p_x")] and sp.simplify(s.dp[0] + sp.Symbol("k", positive=True) * q("x")) == 0 and s.separable and s.source == "hamiltonian"


def test_all_self_checks_pass_on_every_preset(systems):
    for key, s in systems.items():
        charges = ch.noether_charges(s) if s.L is not None else []
        ck = ch.verify_system(s, None, 10.0, charges)
        assert ck.passed, (key, [(c.label, c.detail) for c in ck.checks if not c.passed])


# ---------------------------------------------------------------- Noether
def kinds(charges):
    return sorted(c.kind for c in charges)


def test_cyclic_coordinate_and_energy(systems):
    ch_ = ch.noether_charges(systems["polar_central"])
    assert kinds(ch_) == ["cyclic coordinate", "energy"]
    cyc = next(c for c in ch_ if c.kind == "cyclic coordinate")
    assert num(cyc.charge - q("r") ** 2 * q("thdot"), systems["polar_central"]) == 0 and cyc.verified


def test_free_particle_symmetries_translations_rotation_boosts(systems):
    found = ch.noether_charges(systems["free2d"])
    ks = kinds(found)
    assert ks.count("boost") == 2 and "energy" in ks and ks.count("cyclic coordinate") == 2
    assert "rotation" in ks                                                                      # angular momentum x*ydot - y*xdot
    rot = next(c for c in found if c.kind == "rotation")
    ang = q("x") * q("ydot") - q("y") * q("xdot")
    assert num(rot.charge - ang, systems["free2d"]) == 0 or num(rot.charge + ang, systems["free2d"]) == 0
    assert all(c.verified for c in found)


def test_kepler_has_angular_momentum_from_rotation_symmetry_but_no_boost(systems):
    found = ch.noether_charges(systems["central"])
    assert kinds(found) == ["energy", "rotation"] and all(c.verified for c in found)


def test_uniform_gravity_boost_charge(systems):
    found = ch.noether_charges(systems["gravity"])
    gauge = next(c for c in found if c.kind == "translation")                                    # a translation, but only up to a total time derivative
    assert num(gauge.charge - (q("xdot") + q("t")), systems["gravity"]) == 0                    # m xdot + m g t with m = g = 1: the initial velocity
    assert gauge.verified and "Λ" in gauge.description


def test_no_spurious_symmetry_for_a_pendulum_or_coupled_oscillators(systems):
    assert kinds(ch.noether_charges(systems["pendulum"])) == ["energy"]
    assert kinds(ch.noether_charges(systems["coupled"])) == ["energy"]


def test_the_search_can_be_switched_off(systems):
    assert kinds(ch.noether_charges(systems["free2d"], search=False)) == ["cyclic coordinate", "cyclic coordinate", "energy"]


def test_noether_needs_a_lagrangian():
    with pytest.raises(ch.CrossCuttingError, match="Lagrangian"):
        ch.noether_charges(ch.preset_system("henon_heiles"))


# ---------------------------------------------------------------- input validation
@pytest.mark.parametrize("builder,msg", [
    (lambda: ch.from_lagrangian("x*xdot", ["x"]), "singular"),
    (lambda: ch.from_lagrangian("xdot^2/2", ["x", "x"]), "distinct"),
    (lambda: ch.from_lagrangian("xdot^2/2", []), "coordinate"),
    (lambda: ch.from_lagrangian("p_x^2", ["x"]), "p_<name>|qdot"),
    (lambda: ch.from_hamiltonian("p_x^2/2 + t*x", ["x"]), "autonomous"),
    (lambda: ch.from_hamiltonian("xdot^2/2", ["x"]), "momenta"),
    (lambda: ch.from_lagrangian("__import__('os')", ["x"]), "not allowed|Unknown|calls"),
    (lambda: ch.preset_system("nope"), "Unknown preset"),
    (lambda: ch.from_lagrangian("a*b", ["a", "b", "c", "d", "e"]), "At most 4"),
])
def test_bad_input_is_refused_with_a_reason(builder, msg):
    with pytest.raises(ch.CrossCuttingError, match=msg):
        builder()


def test_time_dependent_lagrangian_has_no_energy_conservation():
    s = ch.from_lagrangian("m*xdot^2/2 - m*x^2*(1 + t)/2", ["x"], {"m": 1.0})
    assert s.time_dependent and all(c.kind != "energy" for c in ch.noether_charges(s))


# ---------------------------------------------------------------- phase space
def test_phase_portrait_of_the_pendulum_has_closed_and_open_orbits():
    pp = ch.phase_portrait(ch.preset_system("pendulum"), (-7, 7), (-3, 3))
    assert pp.H.shape == (160, 160) and len(pp.trajectories) >= 4
    for (qs, ps), e in zip(pp.trajectories, pp.energies):
        assert np.max(np.abs(0.5 * ps ** 2 - np.cos(qs) - e)) < 1e-6                         # and starts exactly on its level set
    with pytest.raises(ch.CrossCuttingError):
        ch.phase_portrait(ch.preset_system("henon_heiles"), (-1, 1), (-1, 1))


def test_initial_on_energy_and_inaccessible_points():
    s = ch.preset_system("henon_heiles")
    z = ch.initial_on_energy(s, 1 / 8, {"x": 0.0, "y": 0.1, "p_y": 0.0}, "p_x", (0.0, 1.0))
    assert ch.to_flow(s).H(z[:2], z[2:]) == pytest.approx(1 / 8, abs=1e-12)
    with pytest.raises(ch.CrossCuttingError, match="not accessible"):
        ch.initial_on_energy(s, 1 / 8, {"x": 0.0, "y": 3.0, "p_y": 0.0}, "p_x", (0.0, 1.0))
    with pytest.raises(ch.CrossCuttingError, match="Fix every"):
        ch.initial_on_energy(s, 1 / 8, {"x": 0.0}, "p_x")


def test_poincare_section_stays_on_the_energy_surface_and_regular_orbits_trace_curves():
    s = ch.preset_system("henon_heiles")
    E = 1 / 12
    starts = [ch.initial_on_energy(s, E, {"x": 0.0, "y": y0, "p_y": 0.0}, "p_x", (0.0, 1.0)) for y0 in (0.0, 0.1)]
    ps = ch.poincare_section(s, starts, "x", 0.0, 120, 800)
    assert max(ps.energy_errors) < 1e-8 and all(len(p) >= 100 for p in ps.points)
    for pts in ps.points:                                                                         # a regular orbit: the points lie on a smooth closed curve
        y, py = pts[:, 0], pts[:, 1]
        assert np.max(y) - np.min(y) < 0.7 and np.max(np.abs(py)) < 0.5
        assert np.all(0.5 * py ** 2 + 0.5 * y ** 2 - y ** 3 / 3 <= E + 1e-8)                  # inside the energy surface (p_x^2 >= 0 on the section)
    with pytest.raises(ch.CrossCuttingError):
        ch.poincare_section(ch.preset_system("harmonic"), [np.zeros(2)], "x")
    with pytest.raises(ch.CrossCuttingError):
        ch.poincare_section(s, starts, "z")


def test_flow_is_symplectic_and_a_non_hamiltonian_flow_would_not_be():
    flow = ch.to_flow(ch.preset_system("pendulum"))
    defect, det_err = ch.symplectic_defect(flow, np.array([1.0, 0.2]), 0.5)
    assert defect < 1e-6 and det_err < 1e-6
    damped = ch.HamiltonianFlow(1, flow.H, lambda q_, p: np.array([np.sin(q_[0]) + 0.3 * p[0]]), lambda q_, p: np.array([p[0]]), True)     # friction: not Hamiltonian
    defect, det_err = ch.symplectic_defect(damped, np.array([1.0, 0.2]), 0.5)
    assert defect > 1e-2 and det_err > 1e-2                                                       # phase-space volume shrinks like exp(-0.3 t)


def test_to_flow_needs_a_hamiltonian():
    s = ch.from_lagrangian("xdot^2/2*(1 + x^2*xdot^2)", ["x"])
    if s.H is None:
        with pytest.raises(ch.CrossCuttingError):
            ch.to_flow(s)
        assert any("Legendre" in n for n in s.notes)
