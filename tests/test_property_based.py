"""Property-based tests (Hypothesis) for a handful of the codebase's
highest-value invariants -- places where a hand-picked set of example
test cases is much weaker evidence of correctness than "this holds for
hundreds of randomly generated inputs," because the failure mode is
exactly the kind that hides in the cases nobody thought to write by
hand. Not a replacement for the example-based tests elsewhere in
tests/ -- a complement, for the specific functions where randomized
input generation can actually stress-test something real.
"""
import sympy as sp
from hypothesis import given, strategies as st, settings, assume

from modules.ode_utils import _defined_function
from modules.tensor_calculus import _is_exactly_zero
from modules.statistical_inference import regression_inference, compare_polynomial_degrees


# --------------------------------------------------------------------- _defined_function
# Regression coverage for the exact bug class caught in this codebase: a
# coupled-ODE equation like Eq(Derivative(B(t),t), k1*A(t) - k2*B(t))
# mentions BOTH A(t) and B(t) as AppliedUndef atoms, and picking "the
# first one" from a Python set is order-dependent on hash seed. This
# generates MANY different two-function equation shapes (varying which
# function is differentiated, how many extra additive terms reference
# the other function, and the order terms are written in) and checks
# _defined_function picks the right one in every case, not just the
# handful of examples in test_ode_numerical_cross_check.py.

_func_names = st.sampled_from(["A", "B", "C", "N", "P", "Q"])


@given(
    differentiated_name=_func_names,
    other_name=_func_names,
    n_extra_terms=st.integers(min_value=0, max_value=4),
    coeffs=st.lists(st.integers(min_value=-5, max_value=5).filter(lambda x: x != 0), min_size=5, max_size=5),
)
@settings(max_examples=200)
def test_defined_function_always_finds_the_differentiated_one(
        differentiated_name, other_name, n_extra_terms, coeffs):
    assume(differentiated_name != other_name)
    t = sp.Symbol("t")
    f_diff = sp.Function(differentiated_name)
    f_other = sp.Function(other_name)

    # build a right-hand side with a variable number of terms mentioning
    # the OTHER (undifferentiated) function, in varying additive order --
    # varying term count/order is what makes atoms(AppliedUndef)'s
    # underlying set construction differ across examples
    rhs = sp.Integer(coeffs[0]) * f_diff(t)
    for i in range(n_extra_terms):
        rhs = rhs + sp.Integer(coeffs[i + 1]) * f_other(t)

    eq = sp.Eq(f_diff(t).diff(t), rhs)
    result = _defined_function(eq)
    assert result == f_diff(t)


@given(n_extra_terms=st.integers(min_value=1, max_value=4))
@settings(max_examples=50)
def test_defined_function_returns_none_for_genuinely_ambiguous_equation(n_extra_terms):
    """An equation with derivatives of TWO different functions has no
    single well-defined answer -- must return None (ambiguous), not
    silently guess one of them."""
    t = sp.Symbol("t")
    A, B = sp.Function("A"), sp.Function("B")
    eq = sp.Eq(A(t).diff(t) + B(t).diff(t), sp.Integer(1))
    assert _defined_function(eq) is None


# --------------------------------------------------------------------- _is_exactly_zero
@given(
    coeffs=st.lists(st.integers(min_value=-20, max_value=20), min_size=1, max_size=5),
)
@settings(max_examples=200)
def test_is_exactly_zero_recognizes_true_zero_from_cancellation(coeffs):
    """expr - expr is always exactly zero, however complicated expr is
    -- a basic sanity floor _is_exactly_zero must never fail, since
    metric-compatibility and Riemann-flatness checks throughout
    tensor_calculus.py depend on it not producing false negatives."""
    x = sp.Symbol("x")
    expr = sum(sp.Integer(c) * x ** i for i, c in enumerate(coeffs))
    assert _is_exactly_zero(expr - expr)


@given(
    coeffs=st.lists(st.integers(min_value=1, max_value=20), min_size=1, max_size=5),
    constant_shift=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=200)
def test_is_exactly_zero_rejects_genuinely_nonzero_polynomial(coeffs, constant_shift):
    """A polynomial with strictly positive coefficients plus a strictly
    positive constant shift, evaluated as a formal expression in x, is
    never identically zero -- _is_exactly_zero must not have false
    positives either, or a broken verification check would silently
    report success."""
    x = sp.Symbol("x", positive=True)
    expr = sum(sp.Integer(c) * x ** i for i, c in enumerate(coeffs)) + sp.Integer(constant_shift)
    assert not _is_exactly_zero(expr)


# --------------------------------------------------------------------- regression_inference
@given(
    intercept=st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
    slope=st.floats(min_value=-20, max_value=20, allow_nan=False, allow_infinity=False),
    n_points=st.integers(min_value=5, max_value=15),
)
@settings(max_examples=100)
def test_confidence_interval_always_contains_the_point_estimate(intercept, slope, n_points):
    """A basic structural guarantee any correctly-computed confidence
    interval must satisfy: the point estimate itself is always inside
    its own interval. True by construction (estimate +/- margin) for a
    correct implementation, so a violation here means the CI math
    itself is broken, not just imprecise."""
    xs = list(range(n_points))
    ys = [intercept + slope * x for x in xs]  # exact line, no noise -- avoids degenerate-fit edge cases
    # perturb minimally to avoid a perfectly-zero-residual singular case
    ys = [y + 1e-6 * (i % 3 - 1) for i, y in enumerate(ys)]

    result = regression_inference(xs, ys, "polynomial", degree=1)
    assume(result.error is None)  # skip the rare degenerate case rather than failing on it
    for p in result.parameters:
        assert p.ci_lower <= p.estimate <= p.ci_upper


# --------------------------------------------------------------------- compare_polynomial_degrees
@given(
    coeffs=st.lists(st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
                     min_size=3, max_size=3),
    n_points=st.integers(min_value=10, max_value=20),
)
@settings(max_examples=100)
def test_nested_f_statistic_is_never_negative(coeffs, n_points):
    """F-statistics are, by construction, a ratio of two non-negative
    quantities (sums of squared residuals) -- never negative for a
    correct implementation, regardless of what data produced them."""
    a, b, c = coeffs
    xs = [i * 0.5 for i in range(n_points)]
    ys = [a + b * x + c * x ** 2 + 0.01 * ((i % 5) - 2) for i, x in enumerate(xs)]
    result = compare_polynomial_degrees(xs, ys, reduced_degree=1, full_degree=2)
    assume(result.error is None)
    assert result.f_statistic >= 0
