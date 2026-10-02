"""modules/time_uncertainty.py: the fan chart and its guaranteed envelope.
Where an exact answer exists it is used: for N(t) = N0*exp(-k t) the solution
is monotone in each input, so the sampled percentiles and the interval
envelope can both be written down in closed form."""
import numpy as np
import pytest
import sympy as sp
from scipy.stats import norm

import modules.time_uncertainty as tu
from modules.equation_engine import build_model
from modules.ode_utils import group_coupled_odes, initial_condition_symbol, solve_ode
from modules.timeout_utils import ComputationTimeoutError
from tests.test_ode_trajectories import decay_model, spiral_model

N0, K = 100.0, 0.5


def _setup(model):
    symbolic = solve_ode(model, symbolic_initial_conditions=True)
    group = group_coupled_odes([e for e in model.equations if e.kind == "ode" and e.sympy_eq is not None])[0]
    return group, symbolic


def _fan(model=None, sigmas=None, **kw):
    model = model or decay_model()
    group, symbolic = _setup(model)
    kw.setdefault("t_range", (0.0, 8.0))
    return tu.ode_uncertainty_fan(model, group, symbolic, {"k": 0.05} if sigmas is None else sigmas, **kw)


# ------------------------------------------------------------------ symbolic initial conditions

def test_symbolic_initial_conditions_leave_the_value_as_a_symbol():
    plain = solve_ode(decay_model())["N"].rhs
    symbolic = solve_ode(decay_model(), symbolic_initial_conditions=True)["N"].rhs
    n0 = initial_condition_symbol(sp.Function("N")(0))
    assert n0.name == "_ic_N_0_" and symbolic.has(n0)
    assert sp.simplify(symbolic.subs(n0, 100.0) - plain) == 0              # putting the value back gives the usual solution


def test_initial_condition_symbol_is_safe_for_awkward_arguments():
    assert initial_condition_symbol(sp.Function("y")(sp.Rational(1, 2))).name == "_ic_y_1_2_"
    assert initial_condition_symbol(sp.Function("y")(-1)).name == "_ic_y__1_"


# ------------------------------------------------------------------ candidate_parameters

def test_candidates_include_parameters_and_initial_values_with_their_nominal_values():
    group, symbolic = _setup(decay_model())
    params, unresolved = tu.candidate_parameters(decay_model(), group, symbolic)
    assert unresolved == []
    assert {(p.name, p.value, p.kind) for p in params} == {("N(0)", 100.0, "initial condition"),
                                                           ("k", 0.5, "parameter")}


def test_an_unresolved_constant_is_reported_not_offered():
    model = decay_model(ics=())                                           # no initial condition: C1 stays in the solution
    group, symbolic = _setup(model)
    params, unresolved = tu.candidate_parameters(model, group, symbolic)
    assert unresolved == ["C1"] and [p.name for p in params] == ["k"]


# ------------------------------------------------------------------ the fan, against closed forms

def test_nominal_curve_is_the_exact_solution():
    fan = _fan()
    assert fan.applicable and fan.names == ["N"]
    assert fan.nominal[0] == pytest.approx(N0 * np.exp(-K * fan.t))


def test_sampled_percentiles_match_the_exact_distribution():
    # N = N0 exp(-k t) falls as k rises, so the p-th percentile of N is the (1-p)-th percentile of k
    sd = 0.05
    fan = _fan(sigmas={"k": sd}, n_samples=20000, seed=1)
    for q, band in ((5, fan.p5), (25, fan.p25), (50, fan.median), (75, fan.p75), (95, fan.p95)):
        k_quantile = K + sd * norm.ppf(1 - q / 100)
        assert band[0] == pytest.approx(N0 * np.exp(-k_quantile * fan.t), rel=0.02)
    assert (fan.p5 <= fan.p25).all() and (fan.p25 <= fan.median).all()
    assert (fan.median <= fan.p75).all() and (fan.p75 <= fan.p95).all()


def test_the_bands_fan_out_from_a_certain_start_as_time_passes():
    fan = _fan(sigmas={"k": 0.05})                                        # only the rate is uncertain
    width = fan.p95[0] - fan.p5[0]
    assert width[0] == pytest.approx(0.0, abs=1e-9)                       # N(0) is exactly 100 for every draw
    assert width[len(width) // 4] > 0
    assert fan.finite_fraction == 1.0


def test_an_uncertain_initial_value_scales_the_whole_curve():
    fan = _fan(sigmas={"N(0)": 5.0}, n_samples=20000, seed=3)
    # N = N0' exp(-k t) is linear in N0', so the spread at t=0 is N0's own and it shrinks as exp(-k t)
    assert (fan.p95[0][0] - fan.p5[0][0]) == pytest.approx(2 * 1.645 * 5.0, rel=0.03)
    ratio = (fan.p95[0] - fan.p5[0]) / (fan.p95[0][0] - fan.p5[0][0])
    assert ratio == pytest.approx(np.exp(-K * fan.t), rel=1e-6)


def test_the_guaranteed_envelope_is_exactly_the_interval_extremes():
    sd_k, sd_n, sig = 0.05, 4.0, 2.0
    fan = _fan(sigmas={"k": sd_k, "N(0)": sd_n}, envelope_sigmas=sig)
    t = fan.t
    assert fan.envelope_lo is not None and fan.envelope_hi is not None and fan.envelope_sigmas == sig
    # N is increasing in N0 and decreasing in k: the extremes sit at opposite corners of the input box
    assert fan.envelope_lo[0] == pytest.approx((N0 - sig * sd_n) * np.exp(-(K + sig * sd_k) * t))
    assert fan.envelope_hi[0] == pytest.approx((N0 + sig * sd_n) * np.exp(-(K - sig * sd_k) * t))
    assert "Guaranteed bound" in fan.envelope_note


def test_every_sampled_value_stays_inside_the_envelope_that_covers_it():
    # the samples are normal draws, so a few exceed 2 sigma; with a 6-sigma box none can
    fan = _fan(sigmas={"k": 0.05, "N(0)": 4.0}, envelope_sigmas=6.0, n_samples=3000, seed=9)
    assert (fan.p5 >= fan.envelope_lo - 1e-9).all() and (fan.p95 <= fan.envelope_hi + 1e-9).all()


def test_no_envelope_unless_asked_for():
    fan = _fan()
    assert fan.envelope_lo is None and fan.envelope_hi is None and fan.envelope_note == ""


def test_same_seed_same_fan_different_seed_different_fan():
    a, b, c = _fan(seed=5), _fan(seed=5), _fan(seed=6)
    assert (a.p95 == b.p95).all() and a.seed == 5
    assert not np.array_equal(a.p95, c.p95)


def _symbolic_oscillator():
    """x' = y, y' = -(k^2 + w^2) x - 2 k y with k, w KNOWN (0.25, 2): dsolve_system answers in terms of
    exp(-t (k -/+ I w)) because the parameters are still symbols when it solves."""
    return build_model({
        "problem_domain": "p", "problem_type": "ode", "independent_variable": "t",
        "variables": [{"symbol": s, "meaning": s, "known_value": v, "unit": None, "is_function": False}
                      for s, v in (("k", 0.25), ("w", 2.0))]
                     + [{"symbol": f, "meaning": f, "known_value": None, "unit": None, "is_function": True}
                        for f in ("x", "y")],
        "equations": [{"name": "xd", "kind": "ode", "expression": "Eq(Derivative(x(t), t), y(t))", "derivation": ""},
                      {"name": "yd", "kind": "ode", "derivation": "",
                       "expression": "Eq(Derivative(y(t), t), -(k**2 + w**2)*x(t) - 2*k*y(t))"}],
        "solve_for": ["x", "y"], "assumptions": [],
        "initial_conditions": [{"expression": "x(0)", "value": 1.0}, {"expression": "y(0)", "value": 0.0}]})


def test_a_coupled_oscillator_with_complex_exponentials_gets_a_real_envelope():
    model = _symbolic_oscillator()
    group, symbolic = _setup(model)
    assert any(s.rhs.has(sp.I) for s in symbolic.values())                # dsolve_system writes it with exp(I*...)
    fan = tu.ode_uncertainty_fan(model, group, symbolic, {"x(0)": 0.05}, (0.0, 10.0), envelope_sigmas=2.0)
    assert fan.applicable and fan.names == ["x", "y"] and fan.finite_fraction == 1.0
    assert fan.envelope_lo is not None                                    # via real_form, not skipped
    assert (fan.p5 >= fan.envelope_lo - 1e-9).all() and (fan.p95 <= fan.envelope_hi + 1e-9).all()
    # x(0) = 1 +/- 0.05 and y(0) = 0: x(t) is linear in x(0), so x(0)'s own extremes are the envelope
    assert fan.envelope_hi[0][0] == pytest.approx(1.1) and fan.envelope_lo[0][0] == pytest.approx(0.9)


def test_draws_with_no_real_result_are_counted_and_left_out():
    # y' = sqrt(k): for k ~ N(0.01, 0.05) about 42% of draws are negative, so sqrt(k) is imaginary
    model = build_model({
        "problem_domain": "p", "problem_type": "ode", "independent_variable": "t",
        "variables": [{"symbol": "k", "meaning": "k", "known_value": 0.01, "unit": None, "is_function": False},
                      {"symbol": "y", "meaning": "y", "known_value": None, "unit": None, "is_function": True}],
        "equations": [{"name": "d", "kind": "ode", "expression": "Eq(Derivative(y(t), t), sqrt(k))", "derivation": ""}],
        "solve_for": ["y"], "assumptions": [], "initial_conditions": [{"expression": "y(0)", "value": 0.0}]})
    fan = _fan(model, sigmas={"k": 0.05}, n_samples=4000, seed=2)
    assert fan.applicable and 0.45 < fan.finite_fraction < 0.70
    assert np.isfinite(fan.median[0][1:]).all()                           # the statistics still come from the real draws


# ------------------------------------------------------------------ refusals

@pytest.mark.parametrize("kwargs, fragment", [
    (dict(n_samples=0), "sample count must be between"),
    (dict(n_samples=tu.MAX_SAMPLES + 1), "sample count must be between"),
    (dict(t_range=(5.0, 5.0)), "end time must be after"),
    (dict(sigmas={"k": 0.0}), "at least one parameter"),
    (dict(sigmas={}), "at least one parameter"),
    (dict(sigmas={"nope": 1.0}), "isn't something this solution depends on"),
    (dict(sigmas={"k": float("nan")}), "finite and not negative"),
    (dict(sigmas={"k": -0.1}), "finite and not negative"),
])
def test_invalid_requests_are_reported_not_raised(kwargs, fragment):
    fan = _fan(**kwargs)
    assert not fan.applicable and fragment in fan.reason


def test_a_solution_with_an_unresolved_constant_is_refused():
    fan = _fan(decay_model(ics=()), sigmas={"k": 0.05})
    assert not fan.applicable and "C1" in fan.reason and "nothing definite" in fan.reason


def test_a_group_with_no_closed_form_is_refused():
    model = decay_model()
    group, _ = _setup(model)
    fan = tu.ode_uncertainty_fan(model, group, {}, {"k": 0.05}, (0.0, 5.0))
    assert not fan.applicable and "closed-form solution" in fan.reason


def test_time_symbol_is_none_for_no_solutions():
    assert tu._time_symbol({}) is None


# ------------------------------------------------------------------ real_form and the envelope's failure modes

def test_real_form_rewrites_complex_exponentials_as_sin_and_cos():
    k, w, t = (sp.Symbol(n) for n in "kwt")
    expr = sp.exp(-t * (k - sp.I * w)) / 2 + sp.exp(-t * (k + sp.I * w)) / 2
    real = tu.real_form(expr)
    assert not real.has(sp.I)
    values = {k: 0.3, w: 2.0, t: 1.7}
    assert float(real.subs(values)) == pytest.approx(float(sp.re(expr.subs(values).evalf())))
    assert real.has(sp.cos)


def test_real_form_leaves_a_real_expression_alone():
    expr = sp.exp(-sp.Symbol("k") * sp.Symbol("t"))
    assert tu.real_form(expr) is expr


def test_real_form_gives_up_gracefully(monkeypatch):
    k, t = sp.symbols("k t")
    expr = sp.exp(sp.I * k * t)

    def slow(*a, **kw):
        raise ComputationTimeoutError(0.1, "real form of solution")
    monkeypatch.setattr(tu, "run_with_timeout", slow)
    tu.real_form.cache_clear()
    assert tu.real_form(expr) == expr                                     # unchanged, not an exception
    tu.real_form.cache_clear()


def test_real_form_keeps_the_original_if_the_real_part_is_still_unresolved():
    expr = sp.exp(sp.I * sp.Symbol("k")) * sp.Function("f")(sp.Symbol("k"))   # re() can't be taken of f(k)
    tu.real_form.cache_clear()
    assert tu.real_form(expr) == expr
    tu.real_form.cache_clear()


def test_an_envelope_that_interval_arithmetic_cannot_evaluate_is_explained_not_fatal(monkeypatch):
    model = decay_model()
    group, symbolic = _setup(model)
    symbolic = {"N": sp.Eq(symbolic["N"].lhs, sp.Symbol("_ic_N_0_") * sp.tanh(sp.Symbol("k") * sp.Symbol("t")))}
    fan = tu.ode_uncertainty_fan(model, group, symbolic, {"k": 0.05}, (0.0, 5.0), envelope_sigmas=2.0)
    assert fan.applicable and fan.envelope_lo is None                     # the sampled fan is still produced
    assert "Interval arithmetic can't evaluate this solution" in fan.envelope_note


def test_an_envelope_that_cannot_be_built_is_explained(monkeypatch):
    def boom(*a, **kw):
        raise TypeError("no such function")
    monkeypatch.setattr(tu.sp, "lambdify", boom)
    out = tu._interval_envelope(sp.Symbol("k") * sp.Symbol("t"), sp.Symbol("t"), np.linspace(0, 1, 3),
                                 [tu.UncertainParameter("k", sp.Symbol("k"), 1.0, "parameter")],
                                 {"k": 0.1}, 2.0, {sp.Symbol("k"): 1.0})
    assert isinstance(out, str) and "No interval form of this solution" in out


def test_real_cells_discards_complex_and_non_finite_values():
    out = tu._real_cells(np.array([1.0 + 0j, 2.0 + 1.0j, np.inf + 0j, 3.0 + 1e-14j]))
    assert out[0] == 1.0 and np.isnan(out[1]) and np.isnan(out[2]) and out[3] == pytest.approx(3.0)


def test_envelope_of_a_function_that_does_not_depend_on_the_uncertain_input_is_a_line():
    t_sym, k_sym = sp.symbols("t k")
    out = tu._interval_envelope(3 * t_sym, t_sym, np.linspace(0, 2, 5),
                                 [tu.UncertainParameter("k", k_sym, 1.0, "parameter")],
                                 {"k": 0.1}, 2.0, {k_sym: 1.0})
    lo, hi = out
    assert lo == pytest.approx(3 * np.linspace(0, 2, 5)) and hi == pytest.approx(lo)    # zero width: nothing to be uncertain about
