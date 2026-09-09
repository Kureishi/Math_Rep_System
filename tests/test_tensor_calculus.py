"""Tests for modules/tensor_calculus.py -- metric-based tensor calculus
(Christoffel symbols, Riemann/Ricci tensors, Ricci scalar, covariant
derivative, index raising/lowering)."""
import sympy as sp

from modules.tensor_calculus import (
    analyze_metric, nonzero_christoffel_symbols, covariant_derivative_of_vector,
    lower_index, raise_index,
)


# ---------------------------------------------------------------- known cases
def test_flat_cartesian_metric_has_zero_curvature():
    result = analyze_metric([["1", "0"], ["0", "1"]], ["x", "y"])
    assert result.error is None
    assert result.is_flat
    assert result.ricci_scalar == 0
    assert result.metric_compatible


def test_flat_polar_metric_has_zero_curvature():
    """Same flat Euclidean plane, different coordinates -- curvature is a
    coordinate-independent geometric fact, so this must also be flat."""
    result = analyze_metric([["1", "0"], ["0", "r**2"]], ["r", "theta"])
    assert result.error is None
    assert result.is_flat
    assert result.ricci_scalar == 0


def test_sphere_metric_has_known_positive_curvature():
    """The 2-sphere of radius R has Gaussian curvature K = 1/R**2, and
    Ricci scalar = 2K in 2D -- this is a known, unambiguous result any
    sign-convention bug in the Riemann tensor would break (it would
    likely still get magnitude 2/R**2 but with the wrong sign)."""
    result = analyze_metric([["R**2", "0"], ["0", "R**2*sin(theta)**2"]], ["theta", "phi"])
    R = sp.Symbol("R")
    assert result.error is None
    assert result.ricci_scalar.equals(2 / R ** 2)
    assert not result.is_flat


def test_metric_compatibility_holds_for_every_metric():
    """nabla_k g_ij == 0 is a mathematical identity that must hold for
    ANY valid metric -- it's really a self-consistency check on the
    Christoffel-symbol computation, not a property specific to one
    metric, so both the flat and curved cases above must pass it, and
    this test makes that expectation explicit."""
    for rows, coords in ([["1", "0"], ["0", "1"]], ["x", "y"]), \
                        ([["R**2", "0"], ["0", "R**2*sin(theta)**2"]], ["theta", "phi"]):
        result = analyze_metric(rows, coords)
        assert result.metric_compatible, f"metric compatibility failed for coords={coords}"


def test_nonzero_christoffel_symbols_deduplicated():
    result = analyze_metric([["R**2", "0"], ["0", "R**2*sin(theta)**2"]], ["theta", "phi"])
    symbols = nonzero_christoffel_symbols(result)
    # Gamma^0_11 and Gamma^1_01 (== Gamma^1_10) are the only nonzero ones,
    # and each should appear exactly once, not once per index ordering
    assert len(symbols) == 2


# ---------------------------------------------------------------- vector operations
def test_covariant_derivative_of_covariantly_constant_vector_field():
    """On flat space in Cartesian coordinates, an ordinary constant
    vector field has zero covariant derivative everywhere (Christoffel
    symbols vanish, so covariant derivative reduces to the ordinary
    partial derivative of a constant, which is zero)."""
    result = analyze_metric([["1", "0"], ["0", "1"]], ["x", "y"])
    nabla = covariant_derivative_of_vector(result, ["1", "0"])
    assert nabla is not None
    assert all(nabla[i, j] == 0 for i in range(2) for j in range(2))


def test_covariant_derivative_on_sphere_matches_christoffel_contraction():
    result = analyze_metric([["R**2", "0"], ["0", "R**2*sin(theta)**2"]], ["theta", "phi"])
    # constant vector field V = (0, 1) -- nabla_j V^i = Gamma^i_j1 (since dV/dcoords = 0)
    nabla = covariant_derivative_of_vector(result, ["0", "1"])
    assert nabla is not None
    theta = sp.Symbol("theta")
    assert sp.simplify(nabla[0, 1] - 1 / sp.tan(theta)) == 0  # nabla_theta V^phi = Gamma^phi_theta,phi


def test_index_raising_and_lowering_are_inverse_operations():
    result = analyze_metric([["R**2", "0"], ["0", "R**2*sin(theta)**2"]], ["theta", "phi"])
    lowered = lower_index(result, ["1", "0"])
    raised_back = raise_index(result, [str(lowered[0]), str(lowered[1])])
    assert raised_back is not None
    assert sp.simplify(raised_back[0] - 1) == 0
    assert sp.simplify(raised_back[1] - 0) == 0


# ---------------------------------------------------------------- error handling
def test_degenerate_metric_reports_error_not_crash():
    result = analyze_metric([["0", "0"], ["0", "1"]], ["x", "y"])
    assert result.error is not None
    assert "invert" in result.error.lower()


def test_metric_parse_error():
    result = analyze_metric([["not @@ valid", "0"], ["0", "1"]], ["x", "y"])
    assert result.error is not None


def test_covariant_derivative_returns_none_on_failed_metric():
    result = analyze_metric([["0", "0"], ["0", "1"]], ["x", "y"])
    assert covariant_derivative_of_vector(result, ["1", "0"]) is None
