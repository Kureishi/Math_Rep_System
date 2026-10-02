"""modules/parameter_morph.py: a family of solution curves as one parameter varies."""
import numpy as np
import pytest
import sympy as sp

import modules.parameter_morph as pm
from modules.parameter_morph import count_turning_points, first_regime_change, morph_family

t, k, w, a = sp.symbols("t k w a")


# ------------------------------------------------------------------ turning points

def test_turning_points_of_a_cosine_match_the_exact_count():
    ts = np.linspace(0, 10, 400)
    assert count_turning_points(np.cos(2 * ts)) == 6        # extrema at t = n*pi/2 for n = 1..6 inside (0, 10)
    assert count_turning_points(np.cos(2 * ts[:200])) == 3
    assert count_turning_points(np.sin(ts)) == 3            # t = pi/2, 3pi/2, 5pi/2


def test_invisible_wiggles_are_not_counted_as_turning_points():
    ts = np.linspace(0, 10, 300)
    heavily_damped = np.exp(-1.8 * ts) * np.cos(2 * ts)     # technically oscillates, amplitude far below the plot's resolution
    assert count_turning_points(heavily_damped) == 1        # only the one dip you can actually see
    assert count_turning_points(np.exp(-ts)) == 0
    assert count_turning_points(np.linspace(0, 1, 50)) == 0


def test_turning_points_degenerate_inputs():
    assert count_turning_points(np.ones(50)) == 0
    assert count_turning_points(np.array([1.0, 2.0])) == 0
    assert count_turning_points(np.array([np.nan, np.nan, np.nan])) == 0
    assert count_turning_points(np.array([0.0, 1.0, np.nan, 0.0, 1.0])) == 2       # NaNs skipped: 0,1,0,1 = a max and a min


def test_a_falling_start_counts_its_first_minimum_and_a_flat_start_waits():
    y = np.array([0.0, 0.0, 0.0, -1.0, -2.0, -1.0, 0.0, 1.0, 0.0])
    assert count_turning_points(y) == 2                     # a minimum at -2 and a maximum at 1
    assert count_turning_points(np.array([5.0, 4.0, 3.0, 4.0, 5.0])) == 1


# ------------------------------------------------------------------ the family

def test_family_curves_are_the_formula_at_each_parameter_value():
    r = morph_family(sp.exp(-k * t), t, k, 0.5, 2.0, (0, 4), {}, function="N", n_values=7, n_points=50)
    assert r.applicable and r.function == "N" and r.param_name == "k" and r.curves.shape == (7, 50)
    assert r.values == pytest.approx(np.linspace(0.5, 2.0, 7))
    for value, curve in zip(r.values, r.curves):
        assert curve == pytest.approx(np.exp(-value * r.t))
    assert r.turning_points == [0] * 7                      # pure decay never turns
    assert r.nominal_curve is None and r.nominal_value is None


def test_other_symbols_take_their_fixed_values():
    r = morph_family(a * sp.exp(-k * t), t, k, 1.0, 2.0, (0, 3), {a: 10.0}, n_values=3, n_points=20)
    assert r.curves[0] == pytest.approx(10 * np.exp(-1.0 * r.t))


def test_the_nominal_curve_marks_the_current_setting():
    r = morph_family(sp.exp(-k * t), t, k, 0.5, 2.0, (0, 4), {}, n_values=5, n_points=30, nominal_value=1.25)
    assert r.nominal_value == 1.25 and r.nominal_curve == pytest.approx(np.exp(-1.25 * r.t))


def test_the_family_changes_character_where_oscillation_appears():
    # exp(-t/4) cos(w t): the curve is monotone-ish until w is large enough to turn within the window
    r = morph_family(sp.exp(-t / 4) * sp.cos(w * t), t, w, 0.05, 3.0, (0, 10), {}, n_values=30)
    assert r.turning_points[0] == 0 and r.turning_points[-1] >= 6
    assert r.turning_points == sorted(r.turning_points)     # more oscillation as w grows
    before, after, tp_before, tp_after = first_regime_change(r)
    assert tp_before == 0 and tp_after >= 1 and before < after
    assert r.values[0] <= before < after <= r.values[-1]


def test_no_regime_change_for_a_uniform_family():
    r = morph_family(sp.exp(-k * t), t, k, 0.5, 2.0, (0, 4), {}, n_values=6)
    assert first_regime_change(r) is None


def test_complex_exponential_solutions_come_out_real():
    expr = (sp.exp(-t * (0.2 - sp.I * w)) + sp.exp(-t * (0.2 + sp.I * w))) / 2      # = e^(-0.2 t) cos(w t)
    r = morph_family(expr, t, w, 1.0, 2.0, (0, 5), {}, n_values=3, n_points=40)
    assert r.applicable and np.isfinite(r.curves).all()
    assert r.curves[0] == pytest.approx(np.exp(-0.2 * r.t) * np.cos(1.0 * r.t))


def test_genuinely_complex_values_become_gaps():
    r = morph_family(sp.sqrt(k) * t, t, k, -1.0, 1.0, (0, 2), {}, n_values=5, n_points=10)
    assert np.isnan(r.curves[0]).all() and np.isfinite(r.curves[-1]).all()       # sqrt(-1) -> NaN; sqrt(1) fine
    assert r.applicable


def test_a_constant_in_time_is_broadcast():
    r = morph_family(k * sp.Integer(1), t, k, 1.0, 2.0, (0, 1), {}, n_values=3, n_points=5, nominal_value=1.5)
    assert r.curves.shape == (3, 5) and r.curves[2] == pytest.approx(2.0)
    assert r.nominal_curve == pytest.approx(np.full(5, 1.5))


# ------------------------------------------------------------------ y range

def test_y_range_shows_every_curve_in_full():
    r = morph_family(sp.cos(w * t) * sp.exp(-t / 8), t, w, 0.0, 3.0, (0, 10), {}, n_values=20)
    lo, hi = r.y_range
    assert lo < r.curves[np.isfinite(r.curves)].min() and hi > r.curves[np.isfinite(r.curves)].max()
    assert hi - lo < 1.2 * (r.curves.max() - r.curves.min()) + 0.2        # padded, but not wildly


def test_a_diverging_family_falls_back_to_percentile_limits():
    r = morph_family(1 / (t - 5.0) ** 3 + k, t, k, 0.0, 1.0, (0, 10), {}, n_values=4, n_points=1001)
    finite = r.curves[np.isfinite(r.curves)]
    assert finite.max() - finite.min() > 1e5                              # the pole is there (the sample at t=5 is inf -> a gap)...
    assert r.y_range[1] - r.y_range[0] < 1e-3 * (finite.max() - finite.min())   # ...but doesn't set the axis


def test_y_range_with_no_finite_values():
    assert pm._family_y_range(np.full((2, 3), np.nan)) == (-1.0, 1.0)


# ------------------------------------------------------------------ what a person sees for the parameter

def test_the_label_replaces_an_internal_symbol_name_everywhere_it_is_shown():
    """An initial value's placeholder symbol is _ic_N_0_; titles and messages must say N(0)."""
    ic = sp.Symbol("_ic_N_0_")
    r = morph_family(ic * sp.exp(-t), t, ic, 1.0, 5.0, (0, 3), {}, n_values=4, label="N(0)")
    assert r.applicable and r.param_name == "N(0)"
    bad = morph_family(ic * sp.exp(-t), t, ic, 5.0, 1.0, (0, 3), {}, label="N(0)")
    assert "range for N(0)" in bad.reason and "_ic_" not in bad.reason
    assert morph_family(sp.exp(-t), t, ic, 1.0, 5.0, (0, 3), {}, label="N(0)").reason.endswith(
        "doesn't depend on N(0).")
    assert morph_family(ic * sp.exp(-t), t, ic, 1.0, 5.0, (0, 3), {}).param_name == "_ic_N_0_"   # no label: the symbol


# ------------------------------------------------------------------ refusals

@pytest.mark.parametrize("kwargs, fragment", [
    (dict(n_values=1), "between 2 and"),
    (dict(n_values=pm.MAX_VALUES + 1), "between 2 and"),
    (dict(lo=2.0, hi=1.0), "range for k"),
    (dict(lo=float("nan"), hi=1.0), "range for k"),
    (dict(t_range=(3, 3)), "end time must be after"),
    (dict(param=w), "doesn't depend on w"),
    (dict(expr=k * t * a), "No value for a"),
])
def test_invalid_requests_are_reported_not_raised(kwargs, fragment):
    args = dict(expr=sp.exp(-k * t), param=k, lo=0.5, hi=2.0, t_range=(0, 4), n_values=5)
    args.update(kwargs)
    r = morph_family(args["expr"], t, args["param"], args["lo"], args["hi"], args["t_range"], {},
                      n_values=args["n_values"])
    assert not r.applicable and fragment in r.reason
