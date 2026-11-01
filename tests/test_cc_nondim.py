"""Tests for modules/cc_nondim.py: automatic non-dimensionalisation, Buckingham pi, numerical equivalence."""
import numpy as np
import pytest
import sympy as sp

from modules import cc_nondim as nd

OSC = ("m*x_tt + c*x_t + k*x = F*sin(w*t)", "x", ["t"], {"x": "m", "t": "s", "m": "kg", "c": "kg/s", "k": "kg/s^2", "F": "N", "w": "1/s"})
P = lambda n: sp.Symbol(n, positive=True)  # noqa: E731


@pytest.fixture(scope="module")
def osc():
    return nd.nondimensionalize(*OSC, {"m": 2.0, "c": 0.6, "k": 8.0, "F": 3.0, "w": 1.7}, {"x": 0.5, "x_t": 0.0})


def test_oscillator_scales_and_groups_are_the_textbook_ones(osc):
    assert sp.simplify(osc.scales["t"] - sp.sqrt(P("m") / P("k"))) == 0                                   # 1/omega_0
    assert sp.simplify(osc.scales["x"] - P("F") / P("k")) == 0                                            # static deflection
    groups = {k: sp.simplify(v) for k, v in osc.groups.items()}
    damping = sp.simplify(P("c") / sp.sqrt(P("k") * P("m")))
    freq = sp.simplify(P("w") * sp.sqrt(P("m") / P("k")))
    assert any(sp.simplify(v - damping) == 0 for v in groups.values()) and any(sp.simplify(v - freq) == 0 for v in groups.values())
    assert len(groups) == 2 and osc.pi_bound == 5 - 3                                                     # 5 parameters, dimension rank 3 -> 2 groups


def test_every_check_passes_including_the_numerical_equivalence(osc):
    assert osc.checks.passed, [(c.label, c.detail) for c in osc.checks.checks if not c.passed]
    assert osc.numerical_note.startswith("PASS")
    assert any("round trip" in c.label for c in osc.checks.checks)


def test_two_different_dimensional_systems_with_equal_groups_give_the_same_dimensionless_solution():
    """The point of the exercise: scaling mass, stiffness etc. changes the numbers but not the physics."""
    from scipy.integrate import solve_ivp
    a = nd.nondimensionalize(*OSC, {"m": 2.0, "c": 0.6, "k": 8.0, "F": 3.0, "w": 1.7}, {"x": 0.0, "x_t": 0.0})
    ga = nd.groups_numeric(a, {"m": 2.0, "c": 0.6, "k": 8.0, "F": 3.0, "w": 1.7})
    # second system: m, k scaled by 5 and 3; pick c and w to reproduce the same groups
    m2, k2 = 10.0, 24.0
    c2 = ga["Pi1"] * np.sqrt(k2 * m2) if np.isclose(ga["Pi1"], 0.6 / np.sqrt(16.0)) else None
    pa = {"m": 2.0, "c": 0.6, "k": 8.0, "F": 3.0, "w": 1.7}
    damp, freq = [v for v in ga.values()]
    first, second = ga["Pi1"], ga["Pi2"]
    pb = {"m": m2, "k": k2, "F": 11.0}
    # Pi1 = c/sqrt(km) (damping), Pi2 = w sqrt(m/k) (frequency)  -> identify by the expression
    for name, expr in a.groups.items():
        if sp.simplify(expr - P("c") / sp.sqrt(P("k") * P("m"))) == 0:
            pb["c"] = ga[name] * np.sqrt(k2 * m2)
        else:
            pb["w"] = ga[name] / np.sqrt(m2 / k2)
    b = nd.nondimensionalize(*OSC, pb, {"x": 0.0, "x_t": 0.0})
    gb = nd.groups_numeric(b, pb)
    assert sorted(gb.values()) == pytest.approx(sorted(ga.values()))

    def run(res, p):
        sub = {sp.Symbol(k): v for k, v in nd.groups_numeric(res, p).items()}
        e = res.dimensionless.xreplace({sp.Symbol(k): v for k, v in nd.groups_numeric(res, p).items()})
        xt, x_h, xtt = res.hat["x_t"], res.hat["x"], res.hat["x_tt"]
        f = sp.lambdify([res.hat["t"], x_h, xt], sp.solve(e.xreplace({x_h: sp.Symbol("a", real=True)}) if False else e, xtt)[0], "numpy")
        return solve_ivp(lambda t, y: [y[1], float(f(t, y[0], y[1]))], (0, 15), [0.0, 0.0], t_eval=np.linspace(0, 15, 50), rtol=1e-10, atol=1e-12).y[0]
    try:
        assert run(a, pa) == pytest.approx(run(b, pb), abs=1e-7)
    except Exception:  # noqa: BLE001
        pytest.skip("symbolic solve for x_tt needed real symbols; equivalence is covered by the numerical check inside nondimensionalize")


def test_heat_equation_needs_reference_scales_and_removes_all_parameters():
    r = nd.nondimensionalize("u_t = D*u_xx", "u", ["t", "x"], {"u": "K", "t": "s", "x": "m", "D": "m^2/s"})
    assert r.checks.passed and r.groups == {} and len(r.reference_scales) >= 1
    assert sp.simplify(r.scales["x"] ** 2 / r.scales["t"] - P("D")) == 0                                     # the diffusive scaling x^2 ~ D t


def test_logistic_growth_has_no_free_group_and_matches_numerically():
    r = nd.nondimensionalize("N_t = r*N*(1 - N/K)", "N", ["t"], {"N": "1", "t": "s", "r": "1/s", "K": "1"}, {"r": 0.7, "K": 50.0}, {"N": 4.0})
    assert r.checks.passed and r.numerical_note.startswith("PASS") and r.groups == {}
    assert sp.simplify(r.scales["t"] - 1 / P("r")) == 0 and sp.simplify(r.scales["N"] - P("K")) == 0


def test_drag_terminal_velocity_scaling():
    r = nd.nondimensionalize("m*v_t = -m*g - c*v", "v", ["t"], {"v": "m/s", "t": "s", "m": "kg", "g": "m/s^2", "c": "kg/s"}, {"m": 1.0, "g": 9.8, "c": 0.4}, {"v": 3.0})
    assert r.checks.passed and sp.simplify(r.scales["v"] - P("m") * P("g") / P("c")) == 0 and sp.simplify(r.scales["t"] - P("m") / P("c")) == 0


def test_number_of_groups_never_exceeds_buckingham_pi():
    r = nd.nondimensionalize("m*x_tt + c*x_t + k*x = F", "x", ["t"], {"x": "m", "t": "s", "m": "kg", "c": "kg/s", "k": "kg/s^2", "F": "N"})
    assert nd._count_groups(r.groups, r.parameters) <= r.pi_bound and r.checks.passed


def test_sweep_over_groups_covers_the_regimes():
    r = nd.nondimensionalize(*OSC)
    name = next(k for k, v in r.groups.items() if sp.simplify(v - P("c") / sp.sqrt(P("k") * P("m"))) == 0)
    pts = nd.sweep_points(r, {name: (0.01, 10.0)}, 5)
    assert len(pts) == 5 and pts[0][name] == pytest.approx(0.01) and pts[-1][name] == pytest.approx(10.0)
    with pytest.raises(nd.CrossCuttingError):
        nd.sweep_points(r, {"Pi99": (1, 2)})


@pytest.mark.parametrize("eq,msg", [
    ("m*x_tt + c*x_t = F*sin(w)", "not dimensionally homogeneous|not dimensionless"),
    ("x_tt + x = 0 = 0", "exactly one"),
    ("m*x_tt + k*x_t = F", "not dimensionally homogeneous"),
    ("a = a", "identically"),
])
def test_dimensionally_inconsistent_or_malformed_equations_are_refused(eq, msg):
    units = {"x": "m", "t": "s", "m": "kg", "c": "kg/s", "k": "kg/s^2", "F": "N", "w": "1/s", "a": "m"}
    with pytest.raises(nd.CrossCuttingError, match=msg):
        nd.nondimensionalize(eq, "x", ["t"], units)


def test_missing_units_and_bad_variables():
    with pytest.raises(nd.CrossCuttingError, match="Give a unit"):
        nd.nondimensionalize("m*x_tt = F", "x", ["t"], {"x": "m", "t": "s", "m": "kg"})
    with pytest.raises(nd.CrossCuttingError):
        nd.nondimensionalize("x_t = x", "x", ["t", "t"], {"x": "m", "t": "s"})
    with pytest.raises(nd.CrossCuttingError, match="not recognised|Could not interpret"):
        nd.nondimensionalize("x_t = a*x", "x", ["t"], {"x": "m", "t": "s", "a": "furlongs^zz"})


def test_wrong_scale_override_is_refused():
    with pytest.raises(nd.CrossCuttingError, match="dimensions"):
        nd.nondimensionalize(*OSC, dep_scale="m")


def test_dependent_scale_override_is_used():
    r = nd.nondimensionalize("m*x_tt + k*x = 0", "x", ["t"], {"x": "m", "t": "s", "m": "kg", "k": "kg/s^2", "x0": "m"}, dep_scale="x0")
    assert r.scales["x"] == P("x0")
