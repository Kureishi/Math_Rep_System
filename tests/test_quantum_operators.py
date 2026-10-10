"""Operator algebra: exact identities, the expression evaluator, eigen-analysis, uncertainty relations, BCH."""
import numpy as np
import pytest
import sympy as sp

from modules.quantum_common import ExpressionError
from modules.quantum_operators import (
    FAMILIES, MAX_DIM, STANDARD_IDENTITIES, analyze_operator, anticommutator, bch_latex_terms, bch_series, build_family,
    check_identity, check_standard_identities, commutator, evaluate, parse_matrix, parse_state, uncertainty_relations, verify_bch,
)

PAULI = build_family("pauli")


# ------------------------------------------------------------------ the library

@pytest.mark.parametrize("family,size", [("pauli", 2), ("spin", 1), ("spin", 2), ("spin", 3), ("spin", 6), ("oscillator", 3), ("oscillator", 8)])
def test_every_standard_identity_holds_exactly_or_is_flagged_as_a_truncation_artefact(family, size):
    env = build_family(family, size)
    for label, result in check_standard_identities(family, env):
        if family == "oscillator":
            assert result.holds or result.artefact_of_truncation, label
        else:
            assert result.holds, (label, result.difference)


def test_the_oscillator_commutator_fails_only_in_the_last_diagonal_entry_by_exactly_the_dimension():
    env = build_family("oscillator", 5)
    diff = commutator(env["a"], env["adag"]) - sp.eye(5)
    nonzero = [(i, j) for i in range(5) for j in range(5) if diff[i, j] != 0]
    assert nonzero == [(4, 4)] and diff[4, 4] == -5


@pytest.mark.parametrize("two_j", [1, 2, 3, 4])
def test_spin_matrices_have_the_right_spectra_and_casimir(two_j):
    env = build_family("spin", two_j)
    j = sp.Rational(two_j, 2)
    assert sorted(env["Jz"].eigenvals()) == sorted(j - k for k in range(two_j + 1))
    assert sp.simplify(env["J2"] - j * (j + 1) * sp.eye(two_j + 1)).is_zero_matrix
    assert env["Jx"].is_hermitian and env["Jy"].is_hermitian and env["Jz"].is_hermitian


def test_the_pauli_matrices_square_to_one_and_are_traceless():
    for name in ("sx", "sy", "sz"):
        assert (PAULI[name] ** 2).equals(sp.eye(2)) and PAULI[name].trace() == 0


@pytest.mark.parametrize("family,size,bad", [("spin", 0, "2j"), ("spin", MAX_DIM, "2j"), ("oscillator", 1, "dimension"), ("oscillator", MAX_DIM + 1, "dimension")])
def test_sizes_outside_the_supported_range_are_refused(family, size, bad):
    with pytest.raises(ValueError, match=bad):
        build_family(family, size)


def test_an_unknown_family_is_refused():
    with pytest.raises(ValueError, match="Unknown operator family"):
        build_family("quark")


def test_every_family_lists_the_operators_it_actually_provides():
    for key, fam in FAMILIES.items():
        assert set(fam.names) == set(build_family(key, 3 if key == "spin" else 4)), key
    assert set(STANDARD_IDENTITIES) == set(FAMILIES)


# ------------------------------------------------------------------ the evaluator

def ev(text, env=PAULI):
    return evaluate(text, env)


def test_products_sums_and_scalars_follow_ordinary_rules():
    assert ev("2*sx + sz").equals(2 * PAULI["sx"] + PAULI["sz"])
    assert ev("sx*sy").equals(PAULI["sx"] * PAULI["sy"])
    assert ev("sx*sy").equals(-ev("sy*sx"))
    assert ev("sx**3").equals(PAULI["sx"]) and ev("sx**0").equals(sp.eye(2))
    assert ev("-sx").equals(-PAULI["sx"]) and ev("+sx").equals(PAULI["sx"])
    assert ev("sx/2").equals(PAULI["sx"] / 2)


def test_a_bare_number_added_to_an_operator_means_that_number_times_the_identity():
    assert ev("sx + 1").equals(PAULI["sx"] + sp.eye(2))
    assert ev("3 - sz").equals(3 * sp.eye(2) - PAULI["sz"])
    assert ev("Id").equals(sp.eye(2))


def test_the_functions_commute_anticommute_adjoint_trace_det_inverse():
    assert ev("comm(sx, sy)").equals(2 * sp.I * PAULI["sz"])
    assert ev("acomm(sx, sy)").is_zero_matrix
    assert ev("dag(sp)").equals(PAULI["sm"]) and ev("adj(sp)").equals(PAULI["sm"])
    assert ev("tr(sz)") == 0 and ev("det(sx)") == -1
    assert ev("inv(sx)").equals(PAULI["sx"]) and ev("T(sy)").equals(-PAULI["sy"])


def test_the_exponential_of_an_operator_is_exact_where_it_can_be():
    assert ev("exp(I*pi/2*sx)").equals(sp.I * PAULI["sx"])
    assert ev("exp(0*sx)").equals(sp.eye(2))
    assert ev("exp(2)") == sp.exp(2)


def test_a_larger_exponential_falls_back_to_a_numerical_matrix_exponential():
    env = build_family("spin", 5)                                  # 6 x 6: beyond the exact-exponential size
    value = ev("exp(I*0.3*Jx)", env)
    from scipy.linalg import expm
    np.testing.assert_allclose(np.array(value.evalf(), dtype=complex), expm(1j * 0.3 * np.array(env["Jx"].evalf(), dtype=complex)), atol=1e-12)


@pytest.mark.parametrize("text,match", [
    ("sx*foo", "Unknown name 'foo'"), ("sx + sx2", "Unknown name"), ("", "empty"), ("sx +", "Could not read"),
    ("sx/sy", "Cannot divide by an operator"), ("2**sx", "must be a number"), ("sx**2.5", "integers"), ("sx**20", "integers"),
    ("sx**-1", "integers"), ("comm(sx)", "takes 2"), ("tr(sx, sy)", "takes 1"), ("nope(sx)", "not available"),
    ("sin(sx)", "not available"), ("inv(sx - sx)", "singular"), ("'a'*sx", "plain numbers"), ("x" * 400, "too long"),
    ("sx if 1 else sz", "Unsupported"), ("[sx]", "Unsupported"), ("sx.T", "Unsupported"), ("__import__('os')", "plain numbers|not available"),
    ("sx; sy", "Could not read"), ("True*sx", "plain numbers"),
])
def test_bad_expressions_are_refused_with_a_message_about_what_to_fix(text, match):
    with pytest.raises(ExpressionError, match=match):
        ev(text)


def test_operators_of_different_sizes_cannot_be_combined():
    env = {**PAULI, "J": build_family("spin", 2)["Jz"]}
    with pytest.raises(ExpressionError, match="different sizes"):
        ev("sx + J", env)
    with pytest.raises(ExpressionError, match="different sizes"):
        ev("comm(sx, J)", env)


def test_evaluation_never_runs_python(monkeypatch):
    import builtins
    monkeypatch.setattr(builtins, "eval", lambda *a, **k: (_ for _ in ()).throw(AssertionError("eval used")))
    assert ev("sx*sy").equals(sp.I * PAULI["sz"])


# ------------------------------------------------------------------ identities

def test_a_true_identity_holds_and_a_false_one_shows_the_difference():
    ok = check_identity("comm(sx, sy)", "2*I*sz", PAULI)
    assert ok.holds and ok.difference.is_zero_matrix
    bad = check_identity("comm(sx, sy)", "2*sz", PAULI)
    assert not bad.holds and not bad.artefact_of_truncation
    assert bad.difference.equals(2 * sp.I * PAULI["sz"] - 2 * PAULI["sz"])


def test_identities_with_a_scalar_side_compare_it_with_the_identity_matrix():
    assert check_identity("sx**2", "1", PAULI).holds
    assert check_identity("tr(sx)", "0", PAULI).holds
    assert not check_identity("tr(sx)", "1", PAULI).holds


def test_a_failure_outside_the_last_rows_is_never_called_a_truncation_artefact():
    env = build_family("oscillator", 5)
    wrong = check_identity("comm(a, adag)", "2*Id", env, truncated=True)
    assert not wrong.holds and not wrong.artefact_of_truncation
    right = check_identity("comm(a, adag)", "Id", env, truncated=True)
    assert not right.holds and right.artefact_of_truncation and "last basis states" in right.note


def test_truncation_is_not_assumed_when_the_family_is_not_truncated():
    env = build_family("oscillator", 5)
    assert not check_identity("comm(a, adag)", "Id", env, truncated=False).artefact_of_truncation


# ------------------------------------------------------------------ parsing matrices and states

def test_parse_matrix_reads_exact_entries():
    m = parse_matrix([["0", "-I"], ["I", "sqrt(2)/2"]])
    assert m[0, 1] == -sp.I and m[1, 1] == sp.sqrt(2) / 2


@pytest.mark.parametrize("rows", [[], [[]], [["1", "2"]], [["1", "2"], ["3"]], [["1"] * 9] * 9, [["__import__('os')", "0"], ["0", "1"]], [["x", "0"], ["0", "1"]]])
def test_unusable_matrices_are_refused(rows):
    with pytest.raises(ExpressionError):
        parse_matrix(rows)


def test_parse_state_normalises_and_rejects_the_zero_vector_and_wrong_lengths():
    s = parse_state(["1", "I"], 2)
    assert sp.simplify((s.H * s)[0, 0]) == 1
    with pytest.raises(ExpressionError, match="zero vector"):
        parse_state(["0", "0"], 2)
    with pytest.raises(ExpressionError, match="2 components"):
        parse_state(["1"], 2)


# ------------------------------------------------------------------ analysing one operator

def test_pauli_operators_are_hermitian_unitary_and_have_eigenvalues_plus_and_minus_one():
    for name in ("sx", "sy", "sz"):
        r = analyze_operator(PAULI[name])
        assert r.hermitian and r.unitary and r.normal and r.exact
        assert sorted(v for v, _ in r.eigenvalues) == [-1, 1] and r.checks.passed
        assert r.trace == 0 and r.determinant == -1


def test_a_non_normal_operator_is_identified_and_a_nonhermitian_one_gets_complex_checks_only():
    r = analyze_operator(PAULI["sp"])
    assert not r.hermitian and not r.unitary and not r.normal and r.checks.passed
    assert all(not c.label.startswith("Hermitian") for c in r.checks.checks)


def test_a_unitary_has_eigenvalues_on_the_unit_circle():
    u = evaluate("exp(I*0.7*sz)", PAULI)
    r = analyze_operator(u)
    assert r.unitary and not r.hermitian and r.checks.passed
    assert any(c.label.startswith("Unitary") for c in r.checks.checks)


def test_spin_one_jz_has_eigenvalues_minus_one_zero_one():
    r = analyze_operator(build_family("spin", 2)["Jz"])
    assert sorted(v for v, _ in r.eigenvalues) == [-1, 0, 1] and r.checks.passed


def test_a_degenerate_operator_reports_multiplicity_and_keeps_orthogonality():
    r = analyze_operator(sp.diag(1, 1, 2))
    assert dict(r.eigenvalues) == {1: 2, 2: 1} and r.checks.passed


def test_eigenvectors_really_satisfy_the_eigen_equation():
    a = build_family("spin", 3)["Jx"]
    r = analyze_operator(a)
    for value, vecs in r.eigenvectors:
        for v in vecs:
            assert (a * v - value * v).applyfunc(sp.simplify).is_zero_matrix


def test_analysis_refuses_oversize_or_non_square_operators():
    with pytest.raises(ValueError):
        analyze_operator(sp.zeros(2, 3))
    with pytest.raises(ValueError):
        analyze_operator(sp.eye(MAX_DIM + 1))


# ------------------------------------------------------------------ uncertainty relations

@pytest.mark.parametrize("state", [["1", "0"], ["0", "1"], ["1", "1"], ["1", "I"], ["1", "2"], ["3", "4*I"]])
def test_the_uncertainty_bounds_hold_for_every_qubit_state(state):
    r = uncertainty_relations(PAULI["sx"], PAULI["sy"], parse_state(state, 2))
    assert r.checks.passed and r.product >= r.heisenberg_bound - 1e-9 and r.schrodinger_bound >= r.heisenberg_bound - 1e-9


def test_for_sigma_x_and_sigma_y_on_spin_up_the_bound_is_saturated():
    r = uncertainty_relations(PAULI["sx"], PAULI["sy"], parse_state(["1", "0"], 2))
    assert r.product == pytest.approx(1.0) and r.heisenberg_bound == pytest.approx(1.0)


def test_commuting_observables_have_no_lower_bound_and_can_both_be_sharp():
    r = uncertainty_relations(PAULI["sz"], PAULI["sz"], parse_state(["1", "0"], 2))
    assert r.product == pytest.approx(0.0, abs=1e-12) and r.heisenberg_bound == pytest.approx(0.0, abs=1e-12)


def test_the_stronger_schrodinger_bound_exceeds_the_heisenberg_bound_when_the_covariance_is_nonzero():
    a = PAULI["sx"]
    b = PAULI["sx"] + PAULI["sz"]
    state = parse_state(["1", "2"], 2)
    r = uncertainty_relations(a, b, state)
    assert abs(r.covariance) > 1e-6 and r.schrodinger_bound > r.heisenberg_bound + 1e-9 and r.checks.passed


def test_spin_one_state_satisfies_the_relation_for_jx_and_jy():
    env = build_family("spin", 2)
    r = uncertainty_relations(env["Jx"], env["Jy"], parse_state(["1", "1", "1"], 3))
    assert r.checks.passed


def test_a_nonhermitian_operator_is_not_an_observable():
    with pytest.raises(ValueError, match="not Hermitian"):
        uncertainty_relations(PAULI["sp"], PAULI["sx"], parse_state(["1", "0"], 2))


def test_mismatched_dimensions_are_refused():
    with pytest.raises(ValueError, match="matching dimensions"):
        uncertainty_relations(PAULI["sx"], PAULI["sz"], parse_state(["1", "0", "0"], 3))


def test_the_checks_would_fail_if_the_stronger_bound_were_computed_too_large(monkeypatch):
    from modules import quantum_operators as qo
    real, calls = qo.math.sqrt, []

    def inflated(v):
        calls.append(v)
        return real(v) * 10 if len(calls) == 3 else real(v)        # the third sqrt is the Schrödinger–Robertson bound
    monkeypatch.setattr(qo.math, "sqrt", inflated)
    # spin up saturates the bound for σx, σy (ΔA ΔB = 1 = ½|⟨[σx, σy]⟩|), so an inflated bound must fail it
    r = qo.uncertainty_relations(PAULI["sx"], PAULI["sy"], parse_state(["1", "0"], 2))
    assert not r.checks.passed and any("Schrödinger" in c.label for c in r.checks.checks if not c.passed)


# ------------------------------------------------------------------ Baker–Campbell–Hausdorff

def numeric(m):
    return np.array(m.evalf(), dtype=complex)


def test_the_bch_series_matches_log_of_the_product_to_the_stated_order_for_random_matrices():
    from scipy.linalg import expm, logm
    rng = np.random.default_rng(3)
    for _ in range(5):
        x = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        y = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        exact = logm(expm(3e-3 * x) @ expm(3e-3 * y))
        errors = [np.linalg.norm(exact - bch_series(3e-3 * x, 3e-3 * y, k), 2) for k in range(1, 6)]
        assert all(b < a or b < 1e-13 for a, b in zip(errors, errors[1:]))      # each order helps, until rounding level
        assert errors[-1] < 1e-12


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_the_fitted_error_exponent_is_order_plus_one(order):
    x = PAULI["sx"] / 2 + PAULI["sz"]
    y = PAULI["sz"] / 3 + PAULI["sy"] / 2
    r = verify_bch(x, y, order)
    assert r.checks.passed and r.fitted_exponent == pytest.approx(order + 1, abs=0.45)


def test_a_wrong_coefficient_is_caught_by_the_exponent_check(monkeypatch):
    from modules import quantum_operators as qo
    real = qo.bch_series
    def wrong(x, y, order):
        z = real(x, y, order)
        return z + (1 / 12) * qo._c(x, qo._c(x, y)) if order >= 3 else z            # double-counts the 1/12 term
    monkeypatch.setattr(qo, "bch_series", wrong)
    r = qo.verify_bch(PAULI["sx"] / 2 + PAULI["sz"], PAULI["sz"] / 3 + PAULI["sy"] / 2, 3)
    assert not r.checks.passed


def test_commuting_operators_make_the_series_exact_at_first_order():
    r = verify_bch(PAULI["sz"], 2 * PAULI["sz"], 1)
    assert r.checks.passed and np.isnan(r.fitted_exponent) and r.errors.max() < 1e-10


def test_order_one_is_x_plus_y_and_order_two_adds_half_the_commutator():
    x, y = numeric(PAULI["sx"]), numeric(PAULI["sy"])
    np.testing.assert_allclose(bch_series(x, y, 1), x + y)
    np.testing.assert_allclose(bch_series(x, y, 2), x + y + 0.5 * (x @ y - y @ x))


def test_the_latex_terms_are_nested_commutators_in_increasing_order():
    terms = bch_latex_terms(5)
    assert [k for k, _ in terms] == [1, 2, 3, 4, 5]
    assert terms[0][1] == "X + Y" and r"\left[X,\,Y\right]" in terms[1][1]
    assert terms[2][1].count(r"\left[") == 4 and terms[3][1].count(r"\left[") == 3 and terms[4][1].count(r"\left[") == 24
    assert bch_latex_terms(2) == terms[:2]


@pytest.mark.parametrize("order", [0, 6, -1])
def test_unsupported_orders_are_refused(order):
    with pytest.raises(ValueError):
        bch_series(np.eye(2), np.eye(2), order)
    with pytest.raises(ValueError):
        bch_latex_terms(order)


def test_bch_needs_operators_of_the_same_size():
    with pytest.raises(ValueError):
        verify_bch(sp.eye(2), sp.eye(3), 2)


def test_commutator_and_anticommutator_helpers():
    a, b = PAULI["sx"], PAULI["sy"]
    assert commutator(a, b).equals(2 * sp.I * PAULI["sz"]) and anticommutator(a, b).is_zero_matrix
