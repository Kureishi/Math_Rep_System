"""modules/series_animation.py: Taylor and Fourier partial sums. Checked
against known mathematics -- the Taylor polynomial of sin, the square
wave's coefficients 4/(n pi), the sawtooth/parabola coefficients, the Gibbs
overshoot of ~17.9% of the half-jump -- not against the code's own output."""
import numpy as np
import pytest
import sympy as sp

import modules.series_animation as sa
from modules.series_animation import fourier_partial_sums, taylor_partial_sums, _fixed_y_range, _rms
from modules.series_asymptotics import SeriesResult, taylor_series

x = sp.Symbol("x")
SQUARE = "Piecewise((-1, x < 0), (1, True))"


# ------------------------------------------------------------------ Taylor

def test_sine_partial_sums_are_the_taylor_polynomials_one_nonzero_term_at_a_time():
    r = taylor_partial_sums(taylor_series("sin(x)", "x", 0, 7), half_width=2.0)
    assert r.error is None and r.kind == "taylor" and r.center == 0.0
    assert r.labels == ["through x^1 \u2014 1 term", "through x^3 \u2014 2 terms",
                        "through x^5 \u2014 3 terms", "through x^7 \u2014 4 terms"]   # no repeated frames for zero coefficients
    xs = r.xs
    assert r.sums[0] == pytest.approx(xs)
    assert r.sums[1] == pytest.approx(xs - xs ** 3 / 6)
    assert r.sums[3] == pytest.approx(xs - xs ** 3 / 6 + xs ** 5 / 120 - xs ** 7 / 5040)
    assert r.target == pytest.approx(np.sin(xs))
    assert r.errors == sorted(r.errors, reverse=True) and r.errors[-1] < r.errors[0] / 100   # converges near 0
    lo, hi = r.y_range
    assert lo < -1.0 and hi > 1.0 and lo == pytest.approx(-hi, rel=1e-3)               # fixed, symmetric, bigger than sin's range
    assert "\u00b11" in r.error_region                                                 # inner half of a half-width 2


def test_expansion_about_a_nonzero_point_is_centred_there():
    r = taylor_partial_sums(taylor_series("exp(x)", "x", 1, 3), half_width=2.0)
    assert r.error is None and r.center == 1.0
    assert r.xs[0] == pytest.approx(-1.0) and r.xs[-1] == pytest.approx(3.0)
    assert len(r.sums) == 4 and "(x \u2212 1)^0" in r.labels[0]
    assert r.sums[0] == pytest.approx(np.e)                                            # the constant term e
    assert r.sums[-1] == pytest.approx(np.e * (1 + (r.xs - 1) + (r.xs - 1) ** 2 / 2 + (r.xs - 1) ** 3 / 6))


def test_a_laurent_pole_is_one_frame_with_a_finite_axis():
    r = taylor_partial_sums(taylor_series("1/(x-1)", "x", 1, 4), half_width=2.0)
    assert r.error is None and len(r.sums) == 1 and "^-1" in r.labels[0]
    assert np.isfinite(r.y_range).all()                                                # the pole can't blow up the axis
    assert r.sums[0] == pytest.approx(r.target, nan_ok=True)                           # the series IS the function


def test_laurent_terms_are_ordered_from_the_lowest_power():
    r = taylor_partial_sums(taylor_series("1/x**2 + x", "x", 0, 3), half_width=2.0)
    assert r.error is None and len(r.sums) == 2
    assert "x^-2" in r.labels[0] and "x^1" in r.labels[1]


def test_constant_function_gets_a_padded_axis_around_its_value():
    r = taylor_partial_sums(taylor_series("5", "x", 0, 3))
    assert r.error is None and len(r.sums) == 1
    assert r.y_range == pytest.approx((2.0, 8.0))                                      # hi == lo falls back to a span of |value|


def test_no_animation_for_an_expansion_that_isnt_available():
    r = taylor_partial_sums(taylor_series("x*log(x)", "x", 0, 4))
    assert r.error and "no Taylor series" in r.error and r.sums == []


def test_no_animation_for_an_asymptotic_result():
    asym = SeriesResult(kind="asymptotic", input_expr=1 / (x + 1), point=sp.oo, truncated=1 / x,
                         expansion_available=True)
    assert "no Taylor series" in (taylor_partial_sums(asym).error or "")


def test_non_power_terms_cannot_be_split_term_by_term():
    log_series = SeriesResult(kind="taylor", input_expr=x * sp.log(x), point=sp.Integer(0),
                               truncated=x * sp.log(x), expansion_available=True)
    r = taylor_partial_sums(log_series)
    assert r.error and "aren't plain powers" in r.error


def test_nonpositive_half_width_is_refused():
    assert "half-width must be positive" in (taylor_partial_sums(taylor_series("sin(x)"), half_width=0).error or "")


# ------------------------------------------------------------------ Fourier

def test_square_wave_coefficients_are_4_over_n_pi_for_odd_n():
    r = fourier_partial_sums(SQUARE, n_harmonics=9)
    assert r.error is None and r.kind == "fourier" and r.center is None
    a, b = zip(*r.coefficients)
    for n in (1, 3, 5, 7, 9):
        assert b[n] == pytest.approx(4 / (n * np.pi), abs=2e-3)
    for n in (2, 4, 6, 8):
        assert abs(b[n]) < 2e-3                                                        # even harmonics vanish
    assert all(abs(v) < 2e-3 for v in a)                                               # odd function: no cosines


def test_square_wave_partial_sums_converge_and_the_error_never_rises():
    r = fourier_partial_sums(SQUARE, n_harmonics=30)
    assert r.monotone is True
    assert all(later <= earlier + 1e-9 for earlier, later in zip(r.errors, r.errors[1:]))
    assert r.errors[-1] < r.errors[0] / 2
    assert r.labels[0] == "N = 1 harmonic" and r.labels[-1] == "N = 30 harmonics"
    assert r.error_region == "over one full period"


def test_gibbs_overshoot_is_about_9_percent_of_the_full_jump():
    r = fourier_partial_sums(SQUARE, n_harmonics=40)
    peak = float(r.sums[-1].max())
    assert 1.17 < peak < 1.19                                                          # 1 + 0.0895 * 2 = 1.179, however many terms


def test_parabola_coefficients():
    r = fourier_partial_sums("x**2", n_harmonics=6)
    a, b = zip(*r.coefficients)
    assert a[0] == pytest.approx(2 * np.pi ** 2 / 3, rel=1e-4)                        # a0/2 = pi^2/3
    for n in range(1, 7):
        assert a[n] == pytest.approx(4 * (-1) ** n / n ** 2, rel=1e-3)
        assert abs(b[n]) < 1e-3                                                        # even function: no sines


def test_a_custom_half_period_is_respected():
    r = fourier_partial_sums("sin(pi*x)", half_period=1.0, n_harmonics=3)             # period 2: b1 = 1, all else 0
    a, b = zip(*r.coefficients)
    assert b[1] == pytest.approx(1.0, abs=1e-6) and abs(b[2]) < 1e-6 and abs(b[3]) < 1e-6
    assert r.xs[0] == pytest.approx(-2.0) and r.xs[-1] == pytest.approx(2.0)          # two periods shown
    assert r.errors[0] < 1e-6                                                          # one harmonic already reproduces it


def test_an_exactly_reproduced_function_is_not_flagged_as_a_rising_error():
    """Regression: a constant (or a single sine) is reproduced to floating-point
    precision, so every partial sum's error is ~1e-16 of rounding noise. The
    monotone check compared that noise against a tolerance scaled to the (zero)
    first error and reported a Fourier series as 'suspicious' -- a false alarm."""
    for expr, L in (("3", np.pi), ("sin(pi*x)", 1.0), ("cos(2*x)", np.pi)):
        r = fourier_partial_sums(expr, half_period=L, n_harmonics=6)
        assert r.monotone is True, expr


def test_constant_function_has_only_a_mean():
    r = fourier_partial_sums("3", n_harmonics=3)
    assert r.error is None and r.coefficients[0][0] == pytest.approx(6.0)             # a0/2 = 3
    assert r.sums[0] == pytest.approx(np.full_like(r.xs, 3.0)) and r.monotone is True


def test_target_is_the_periodic_extension():
    r = fourier_partial_sums("x", n_harmonics=2)                                      # f(x) = x on [-pi, pi]
    i = int(np.argmin(np.abs(r.xs - (np.pi + 0.5))))
    assert r.target[i] == pytest.approx(r.xs[i] - 2 * np.pi, abs=1e-9)                 # beyond pi it wraps back to -pi + 0.5


@pytest.mark.parametrize("kwargs, fragment", [
    (dict(expr_str=SQUARE, half_period=0), "half-period L must be positive"),
    (dict(expr_str=SQUARE, n_harmonics=0), "between 1 and"),
    (dict(expr_str=SQUARE, n_harmonics=61), "between 1 and"),
    (dict(expr_str="x +"), "Could not parse"),
    (dict(expr_str="x*a"), "may only depend on x"),
    (dict(expr_str="1/x"), "must be finite everywhere"),
])
def test_invalid_input_is_reported_not_raised(kwargs, fragment):
    r = fourier_partial_sums(**kwargs)
    assert r.error and fragment in r.error and r.sums == []


def test_unevaluable_expression_is_reported(monkeypatch):
    def boom(*a, **k):
        raise TypeError("cannot lambdify")
    monkeypatch.setattr(sa.sp, "lambdify", boom)
    assert "Could not evaluate this expression numerically" in (fourier_partial_sums("x").error or "")


def test_a_rising_error_flips_the_monotone_check_off(monkeypatch):
    # a correct Fourier series can't do this; the check exists to catch wrong coefficients
    values = iter([1.0, 0.5, 0.9, 0.2])
    monkeypatch.setattr(sa, "_rms", lambda diff: next(values))
    assert fourier_partial_sums(SQUARE, n_harmonics=4).monotone is False


# ------------------------------------------------------------------ helpers

def test_y_range_helpers_cope_with_undefined_data():
    assert _fixed_y_range(np.array([np.nan, np.inf])) == (-1.0, 1.0)
    assert np.isnan(_rms(np.array([np.nan, np.inf])))
    # percentiles 2 and 98 of [0, 1] are 0.02 and 0.98 (spread 0.96), each side widened by 0.6 of that
    assert _fixed_y_range(np.array([0.0, 1.0])) == pytest.approx((0.02 - 0.576, 0.98 + 0.576))
