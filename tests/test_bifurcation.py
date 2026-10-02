"""modules/bifurcation.py, checked against the logistic map x -> r x (1 - x),
whose behaviour is classical: a stable fixed point 1 - 1/r up to r = 3,
period doubling at r = 3 and r = 1 + sqrt(6) (~3.449), a period-3 window
near r = 3.83, chaos in between."""
import numpy as np
import pytest
import sympy as sp

import modules.bifurcation as bf
from modules.bifurcation import attractor_period, bifurcation_diagram, lambdify_map, regime_changes


def logistic(x, r):
    return r * x * (1 - x)


@pytest.fixture(scope="module")
def logistic_diagram():
    return bifurcation_diagram(logistic, "r", (2.8, 4.0), 0.3, n_params=1200, n_transient=1500, n_keep=64)


def _period_at(result, r):
    return result.periods[int(np.argmin(np.abs(result.params - r)))]


def test_logistic_periods_at_known_parameters(logistic_diagram):
    d = logistic_diagram
    assert d.applicable and d.param_name == "r" and d.values.shape == (64, 1200)
    assert _period_at(d, 2.9) == 1 and _period_at(d, 3.2) == 2 and _period_at(d, 3.5) == 4
    assert _period_at(d, 3.83) == 3                          # the famous period-3 window inside the chaos
    assert _period_at(d, 3.9) is None                        # chaotic: no short cycle


def test_the_fixed_point_is_one_minus_one_over_r(logistic_diagram):
    d = logistic_diagram
    j = int(np.argmin(np.abs(d.params - 2.9)))
    assert d.values[:, j] == pytest.approx(1 - 1 / d.params[j], abs=1e-6)


def test_the_first_period_doublings_land_where_theory_puts_them(logistic_diagram):
    changes = dict((period, p) for p, period in logistic_diagram.transitions)
    assert changes[2] == pytest.approx(3.0, abs=0.01)
    assert changes[4] == pytest.approx(1 + np.sqrt(6), abs=0.01)          # 3.4495
    assert changes[8] == pytest.approx(3.5441, abs=0.01)


def test_an_orbit_that_escapes_is_dropped_not_overflowed():
    d = bifurcation_diagram(lambda x, p: 2 * x + p, "p", (1.0, 2.0), 1.0, n_params=5, n_transient=200, n_keep=4)
    assert np.isnan(d.values).all() and d.periods == [None] * 5


def test_a_start_on_an_unstable_point_is_still_handled():
    d = bifurcation_diagram(lambda x, p: p * x, "p", (0.5, 0.9), 0.0, n_params=4, n_transient=10, n_keep=3)
    assert d.periods == [1] * 4 and (d.values == 0).all()


# ------------------------------------------------------------------ attractor_period / regime_changes

def test_attractor_period_counts_distinct_values():
    assert attractor_period(np.full(20, 0.5), 1e-6) == 1
    assert attractor_period(np.array([0.3, 0.7] * 10), 1e-6) == 2
    assert attractor_period(np.array([0.3, 0.7, 0.5] * 8), 1e-6) == 3
    assert attractor_period(np.linspace(0, 1, 40), 1e-6) is None          # more than MAX_PERIOD distinct values
    assert attractor_period(np.array([0.5, np.nan, 0.5]), 1e-6) is None    # an escaped orbit
    assert attractor_period(np.array([0.5, 0.5 + 1e-9]), 1e-6) == 1        # within tolerance: the same value


def test_regime_changes_ignore_short_blips():
    params = np.linspace(0, 1, 12)
    periods = [1, 1, 1, 1, 2, 1, 2, 2, 2, 2, None, None]
    changes = regime_changes(params, periods, min_run=3)
    # the lone 2 and the lone 1 inside the run are blips; the two-wide None at the end is too short to count
    assert changes == [(pytest.approx(params[6]), 2)]


def test_regime_changes_when_nothing_is_solid():
    assert regime_changes(np.linspace(0, 1, 3), [1, 2, None], min_run=3) == []
    assert regime_changes(np.array([]), []) == []


# ------------------------------------------------------------------ lambdify_map

def test_lambdify_map_substitutes_the_fixed_symbols():
    x, r, c = sp.symbols("x r c")
    g = lambdify_map(c * r * x * (1 - x), x, r, {c: 1.0})
    assert callable(g) and g(0.5, 3.0) == pytest.approx(0.75)


def test_lambdify_map_explains_what_is_missing():
    x, r, c = sp.symbols("x r c")
    assert "No value for c" in lambdify_map(c * r * x, x, r, {})
    assert "doesn't depend on r" in lambdify_map(2 * x, x, r, {})


# ------------------------------------------------------------------ refusals

@pytest.mark.parametrize("kwargs, fragment", [
    (dict(p_range=(3.0, 3.0)), "range for r"),
    (dict(p_range=(float("nan"), 3.0)), "range for r"),
    (dict(n_params=1), "number of parameters"),
    (dict(n_params=bf.MAX_PARAMS + 1), "number of parameters"),
    (dict(n_keep=1), "at least 2 recorded"),
    (dict(n_transient=-1), "at least 2 recorded"),
])
def test_invalid_requests_are_reported_not_raised(kwargs, fragment):
    args = dict(p_range=(2.8, 4.0), n_params=10, n_transient=5, n_keep=4)
    args.update(kwargs)
    d = bifurcation_diagram(logistic, "r", args.pop("p_range"), 0.3, **args)
    assert not d.applicable and fragment in d.reason
