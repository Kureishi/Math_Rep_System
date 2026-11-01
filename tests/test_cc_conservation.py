"""Tests for modules/cc_conservation.py: finding invariants, monitoring drift, symplectic integrators."""
import numpy as np
import pytest
import sympy as sp

from modules import cc_conservation as cc

x, y, z = sp.symbols("x y z", real=True)
S, I, R = sp.symbols("S I R", real=True)


def inv(rhs, state, **kw):
    return cc.find_invariants(rhs, state, **kw)


# ---------------------------------------------------------------- finding invariants
def test_harmonic_oscillator_energy():
    found = inv([y, -x], [x, y], max_degree=2)
    assert len(found) == 1 and sp.simplify(found[0].expr - (x ** 2 + y ** 2)) == 0 and found[0].verified_exactly


def test_scaled_oscillator_finds_the_weighted_energy():
    found = inv([y, -4 * x], [x, y], max_degree=2)
    assert len(found) == 1 and sp.simplify(found[0].expr / found[0].expr.subs({x: 1, y: 0}) - (x ** 2 + y ** 2 / 4)) == 0


def test_lotka_volterra_needs_the_log_terms():
    f = [x * (1 - y), y * (x - 1)]
    assert inv(f, [x, y], max_degree=3) == []
    found = inv(f, [x, y], max_degree=1, include_logs=True)
    assert len(found) == 1 and sp.simplify(found[0].expr - (x + y - sp.log(x) - sp.log(y))) == 0 and found[0].verified_exactly


def test_sir_conserves_the_population_and_dependent_invariants_are_dropped():
    found = inv([-S * I / 2, S * I / 2 - I / 5, I / 5], [S, I, R], max_degree=2)
    assert len(found) == 1 and sp.simplify(found[0].expr - (S + I + R)) == 0                   # (S+I+R)^2 is conserved too, but adds nothing


def test_euler_rigid_body_has_two_independent_invariants():
    found = inv([-y * z, z * x / 2, -x * y / 3], [x, y, z], max_degree=2)
    assert len(found) == 2
    energy_like = {sp.simplify(f.expr / f.expr.coeff(x, 2) if f.expr.coeff(x, 2) != 0 else f.expr) for f in found}
    assert len(energy_like) == 2


def test_dissipative_systems_have_no_invariant():
    assert inv([y, -x - y], [x, y], max_degree=3) == []
    assert inv([-x], [x], max_degree=3) == []


def test_conserved_linear_combination_of_a_decoupled_pair():
    found = inv([-x, x], [x, y], max_degree=1)                                                  # x' = -x, y' = x  ->  x + y conserved
    assert len(found) == 1 and sp.simplify(found[0].expr - (x + y)) == 0


def test_extra_basis_functions_find_the_pendulum_energy():
    # theta' = p, p' = -sin(theta):  p^2/2 - cos(theta) is conserved
    th, p = sp.symbols("th p", real=True)
    found = inv([p, -sp.sin(th)], [th, p], max_degree=2, extra_basis=[sp.cos(th)])
    assert len(found) == 1
    q = found[0].expr
    assert sp.simplify(q / q.coeff(p, 2) - (p ** 2 - 2 * sp.cos(th))) == 0
    assert found[0].verified_exactly or "numerically" in found[0].note


def test_every_returned_invariant_really_is_constant_along_a_solution():
    for rhs, st, y0, kw in [([y, -x], [x, y], [1.3, 0.2], {}), ([x * (1 - y), y * (x - 1)], [x, y], [1.5, 0.8], dict(max_degree=1, include_logs=True)),
                            ([-y * z, z * x / 2, -x * y / 3], [x, y, z], [1.0, 0.5, 0.2], {})]:
        invs = inv(rhs, st, **{"max_degree": 2, **kw})
        run = cc.integrate_monitored(rhs, st, y0, 30.0, invs)
        assert all(r.max_rel < 1e-8 and not r.secular for r in run.reports), [r.summary() for r in run.reports]


def test_a_wrong_candidate_would_be_flagged_by_the_monitor():
    wrong = cc.Invariant(x ** 2 + 2 * y ** 2, 2, False)                                          # not conserved by the oscillator
    run = cc.integrate_monitored([y, -x], [x, y], [1.0, 0.0], 20.0, [wrong])
    assert run.reports[0].max_rel > 0.1


def test_input_validation():
    with pytest.raises(cc.CrossCuttingError, match="autonomous"):
        inv([y + sp.Symbol("t"), -x], [x, y])
    with pytest.raises(cc.CrossCuttingError):
        inv([y], [x, y])
    with pytest.raises(cc.CrossCuttingError, match="max_degree"):
        inv([y, -x], [x, y], max_degree=9)
    with pytest.raises(cc.CrossCuttingError):
        cc.integrate_monitored([y, -x], [x, y], [1.0], 5.0, [])
    with pytest.raises(cc.CrossCuttingError):
        cc.integrate_monitored([y, -x], [x, y], [1.0, 0.0], -1.0, [])


# ---------------------------------------------------------------- integrators
def pendulum():
    return cc.HamiltonianFlow(1, lambda q, p: 0.5 * p[0] ** 2 - np.cos(q[0]), lambda q, p: np.array([np.sin(q[0])]), lambda q, p: np.array([p[0]]), True)


@pytest.fixture(scope="module")
def comparison():
    return cc.compare_integrators(pendulum(), np.array([1.0]), np.array([0.0]), 800.0, 0.1)


def test_symplectic_methods_have_bounded_energy_error_and_rk4_drifts(comparison):
    by = {r.method: r for r in comparison.runs}
    for m in ("leapfrog", "yoshida4", "midpoint"):
        assert by[m].symplectic and by[m].drift.kind == "bounded oscillation", (m, by[m].drift.summary())
    for m in ("rk4", "RK45"):
        assert not by[m].symplectic and by[m].drift.secular, (m, by[m].drift.summary())
    assert by["yoshida4"].drift.max_rel < by["leapfrog"].drift.max_rel                            # higher order -> smaller bounded band


def test_all_methods_agree_with_the_reference_over_a_short_time():
    cmp = cc.compare_integrators(pendulum(), np.array([1.0]), np.array([0.0]), 5.0, 0.01)
    assert all(r.end_error < 1e-2 for r in cmp.runs), [(r.method, r.end_error) for r in cmp.runs]


def test_leapfrog_second_order_and_yoshida_fourth_order_in_energy_error():
    f = pendulum()
    err = {}
    for dt in (0.1, 0.05):
        for m in ("leapfrog", "yoshida4"):
            err[(m, dt)] = cc.integrate_fixed(f, m, np.array([1.0]), np.array([0.0]), 40.0, dt).drift.max_abs
    assert err[("leapfrog", 0.1)] / err[("leapfrog", 0.05)] == pytest.approx(4.0, rel=0.25)
    assert err[("yoshida4", 0.1)] / err[("yoshida4", 0.05)] == pytest.approx(16.0, rel=0.3)


def test_harmonic_oscillator_leapfrog_conserves_the_shadow_hamiltonian():
    f = cc.HamiltonianFlow(1, lambda q, p: 0.5 * (p[0] ** 2 + q[0] ** 2), lambda q, p: q.copy(), lambda q, p: p.copy(), True)
    run = cc.integrate_fixed(f, "leapfrog", np.array([1.0]), np.array([0.0]), 2000.0, 0.1)
    assert run.drift.kind == "bounded oscillation" and run.drift.max_rel < 1e-2


def test_separable_methods_refuse_a_nonseparable_hamiltonian():
    f = cc.HamiltonianFlow(1, lambda q, p: 0.5 * p[0] ** 2 * (1 + q[0] ** 2), lambda q, p: np.array([p[0] ** 2 * q[0]]), lambda q, p: np.array([p[0] * (1 + q[0] ** 2)]), False)
    with pytest.raises(cc.CrossCuttingError, match="separable"):
        cc.integrate_fixed(f, "leapfrog", np.array([0.5]), np.array([0.3]), 10.0, 0.01)
    run = cc.integrate_fixed(f, "midpoint", np.array([0.5]), np.array([0.3]), 300.0, 0.02)         # the implicit rule handles it (over many periods)
    assert run.drift.kind == "bounded oscillation"
    cmp = cc.compare_integrators(f, np.array([0.5]), np.array([0.3]), 10.0, 0.01)
    assert {r.method for r in cmp.runs} == {"midpoint", "rk4", "RK45"}


def test_step_count_limits_and_midpoint_failure():
    f = pendulum()
    with pytest.raises(cc.CrossCuttingError, match="steps"):
        cc.integrate_fixed(f, "rk4", np.array([1.0]), np.array([0.0]), 1.0, 0.5)
    stiff = cc.HamiltonianFlow(1, lambda q, p: 1e6 * q[0] ** 2 / 2 + p[0] ** 2 / 2, lambda q, p: 1e6 * q, lambda q, p: p.copy(), True)
    with pytest.raises(cc.CrossCuttingError, match="did not converge"):
        cc.integrate_fixed(stiff, "midpoint", np.array([1.0]), np.array([0.0]), 1.0, 0.05)


def test_kepler_orbit_with_leapfrog_conserves_angular_momentum_exactly():
    """Leapfrog preserves every quadratic invariant of a central force exactly (L = x p_y - y p_x), a symmetry the integrator inherits."""
    k = 1.0
    f = cc.HamiltonianFlow(2, lambda q, p: 0.5 * float(p @ p) - k / np.linalg.norm(q), lambda q, p: k * q / np.linalg.norm(q) ** 3, lambda q, p: p.copy(), True)
    run = cc.integrate_fixed(f, "leapfrog", np.array([1.0, 0.0]), np.array([0.0, 0.8]), 100.0, 0.01)
    L = run.q[:, 0] * run.p[:, 1] - run.q[:, 1] * run.p[:, 0]
    assert np.max(np.abs(L - L[0])) < 1e-12


# ---------------------------------------------------------------- the app's ODE pipeline and other simulations
def test_ivp_monitor_finds_the_invariant_of_an_undamped_system_and_none_for_a_damped_one():
    from tests.test_ode_trajectories import _group, _model, spiral_model
    from modules.ode_utils import prepare_ivp, solve_ode
    undamped = _model([("xdot", "Eq(Derivative(x(t), t), y(t))"), ("ydot", "Eq(Derivative(y(t), t), -x(t))")], ["x", "y"], [("x(0)", 1.0), ("y(0)", 0.0)])
    setup = prepare_ivp(undamped, _group(undamped), solve_ode(undamped))
    mon = cc.monitor_ivp(setup)
    assert not isinstance(mon, str) and len(mon.invariants) == 1 and mon.reports[0].max_rel < 1e-8
    assert sp.simplify(mon.invariants[0].expr - (sp.Symbol("x", real=True) ** 2 + sp.Symbol("y", real=True) ** 2)) == 0
    damped = spiral_model()
    msg = cc.monitor_ivp(prepare_ivp(damped, _group(damped), solve_ode(damped)))
    assert isinstance(msg, str) and "no conserved" in msg


def test_ivp_monitor_declines_without_symbolic_data():
    class Bare:
        rhs_exprs = None
    assert isinstance(cc.monitor_ivp(Bare()), str)


def test_wavepacket_monitors():
    from modules.quantum_dynamics import evolve_wavepacket
    r = evolve_wavepacket("free", {}, x0=-3.0, sigma=0.6, k0=2.0, domain=(-30.0, 30.0), n_points=512, total_time=4.0, steps=200, n_frames=2, method="split-step", cross_check=False)
    reps = {x.name.split()[0]: x for x in cc.reports_from_wavepacket(r)}
    assert set(reps) == {"norm", "energy", "momentum"} and reps["norm"].max_rel < 1e-10 and reps["momentum"].max_rel < 1e-8
    h = evolve_wavepacket("harmonic", {"w": 1.0, "center": 0.0}, x0=2.0, sigma=0.7, k0=1.0, domain=(-12.0, 12.0), n_points=256, total_time=6.0, steps=300, n_frames=2, method="split-step", cross_check=False)
    assert {x.name.split()[0] for x in cc.reports_from_wavepacket(h)} == {"norm", "energy"}


def test_geodesic_monitors():
    import math
    from modules import gr_geodesics as gg, gr_potential as gp
    from modules.gr_metrics import preset
    E, L = gp.schwarzschild_orbit_from_turning_points(1.0, 100.0, 40.0)
    system = gg.build_system(preset("schwarzschild"), {"M": 1.0})
    x0, u0 = gg.initial_from_conserved(system, 100.0, E, L, "timelike")
    res = gg.integrate_geodesic(system, x0, u0, 2.2 * 2 * math.pi * 70 ** 1.5, n_samples=600)
    reps = cc.reports_from_geodesic(res)
    assert len(reps) >= 3 and all(r.max_rel < 1e-6 and not r.secular for r in reps)
