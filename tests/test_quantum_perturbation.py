"""Rayleigh–Schrödinger perturbation theory against closed forms, sum rules and exact diagonalisation."""
import copy

import numpy as np
import pytest
import sympy as sp

from modules.quantum_perturbation import (
    compare_with_exact, matrix_perturbation, oscillator_perturbation, oscillator_position_matrix, perturbation_levels,
    polynomial_perturbation, suggest_lambda_max, verify_series,
)

R = sp.Rational


def levels_for(coefficients, n_levels=4):
    return oscillator_perturbation(coefficients, n_levels=n_levels).levels


# ------------------------------------------------------------------ closed forms for the oscillator

@pytest.mark.parametrize("n", range(6))
def test_the_quartic_corrections_match_the_known_closed_forms_exactly(n):
    lv = levels_for({4: 1.0}, 6)[n]
    assert lv.e1 == R(3, 4) * (2 * n * n + 2 * n + 1)
    assert lv.e2 == -R(1, 8) * (34 * n ** 3 + 51 * n ** 2 + 59 * n + 21)


def test_the_ground_state_third_order_coefficient_is_333_over_16():
    assert levels_for({4: 1.0})[0].e3 == R(333, 16)               # Bender–Wu


@pytest.mark.parametrize("n", range(4))
def test_a_quadratic_perturbation_reproduces_the_exact_series_of_sqrt_one_plus_two_lambda(n):
    """H = ½p² + ½(1 + 2λ)x² has E = (n + ½) sqrt(1 + 2λ) = (n+½)(1 + λ - λ²/2 + λ³/2 - ...)."""
    lv = levels_for({2: 1.0}, 4)[n]
    base = n + R(1, 2)
    assert (lv.e1, lv.e2, lv.e3) == (base, -base / 2, base / 2)


@pytest.mark.parametrize("n", range(4))
def test_a_linear_perturbation_gives_the_exact_displaced_oscillator_shift(n):
    """H = ½p² + ½x² + λx has E = (n + ½) - λ²/2 exactly, so E1 = E3 = 0 and E2 = -1/2."""
    lv = levels_for({1: 1.0}, 4)[n]
    assert (lv.e1, lv.e2, lv.e3) == (0, -R(1, 2), 0)


def test_coefficients_scale_as_powers_of_the_coupling():
    one, three = levels_for({4: 1.0}), levels_for({4: 3.0})
    for a, b in zip(one, three):
        assert (b.e1, b.e2, b.e3) == (3 * a.e1, 9 * a.e2, 27 * a.e3)


def test_a_cubic_term_has_no_first_order_shift_and_a_negative_second_order_shift():
    for lv in levels_for({3: 1.0}):
        assert lv.e1 == 0 and lv.e2 < 0
    assert levels_for({3: 1.0})[0].e2 == -R(11, 8)                        # the known ground-state value for λx³


def test_the_polynomial_perturbation_is_exact_even_at_the_edge_of_the_basis():
    v = polynomial_perturbation({4: 1.0}, 6)
    big = oscillator_position_matrix(20)
    assert v == (big ** 4).extract(list(range(6)), list(range(6)))
    naive = oscillator_position_matrix(6) ** 4                              # forming the power in the small basis is wrong near the edge
    assert naive[5, 5] != v[5, 5]


@pytest.mark.parametrize("coefficients", [{}, {0: 1.0}, {9: 1.0}, {1.5: 1.0}])
def test_unsupported_powers_are_refused(coefficients):
    with pytest.raises(ValueError):
        polynomial_perturbation(coefficients, 5)


def test_first_order_state_corrections_connect_only_levels_the_perturbation_couples():
    lv = levels_for({4: 1.0})[0]
    nonzero = [i for i in range(lv.state_correction.shape[0]) if lv.state_correction[i] != 0]
    assert nonzero == [2, 4] and lv.state_correction[0] == 0              # x⁴|0> reaches |2> and |4> only


# ------------------------------------------------------------------ finite matrices

V3 = sp.Matrix([[0, 1, 0], [1, 0, sp.sqrt(2)], [0, sp.sqrt(2), 0]])


def test_exact_second_order_sums_for_a_three_level_system():
    lv = perturbation_levels([0, 1, 3], V3)
    assert lv[0].e1 == 0 and lv[0].e2 == -1                              # |V01|²/(0 - 1)
    assert lv[1].e2 == 1 - R(2, 2)                                        # 1/(1-0) + 2/(1-3)
    assert lv[2].e2 == R(2, 2)                                            # 2/(3 - 1)


def test_the_trace_sum_rules_hold_for_a_random_hermitian_perturbation():
    rng = np.random.default_rng(4)
    a = rng.integers(-3, 4, size=(4, 4))
    v = sp.Matrix((a + a.T).tolist())
    e0 = [0, 2, 5, 9]
    lv = perturbation_levels(e0, v)
    assert sum(l.e1 for l in lv) == v.trace()
    assert sp.simplify(sum(l.e2 for l in lv)) == 0 and sp.simplify(sum(l.e3 for l in lv)) == 0


def test_matrix_perturbation_passes_its_own_checks_and_the_measured_slopes_match():
    levels, comp, checks = matrix_perturbation([0, 1, 3], V3, comparison_level=0)
    assert checks.passed, [(c.label, c.detail) for c in checks.checks if not c.passed]
    assert comp.slopes[1] > 1.5 and comp.slopes[2] > 2.5


def test_a_level_the_series_gets_exactly_right_is_reported_as_such_not_as_a_failed_fit():
    """The middle level of this system stays at exactly 1 for every λ, so every truncation is exact and there is
    no slope to measure; that is a pass with an explanation, never a NaN-driven failure."""
    _, comp, checks = matrix_perturbation([0, 1, 3], V3, comparison_level=1)
    assert all(np.isnan(v) for v in comp.slopes.values()) and checks.passed
    assert any("rounding level" in c.detail for c in checks.checks)


def test_the_exact_energy_matches_the_series_better_with_each_order():
    _, comp, _ = matrix_perturbation([0, 1, 4], sp.Matrix([[1, 1, 0], [1, 0, 1], [0, 1, -1]]), comparison_level=0)
    for i in range(len(comp.lambdas)):
        assert comp.errors[1][i] > comp.errors[2][i] > comp.errors[3][i]


def test_a_degenerate_level_gets_the_eigenvalues_of_v_in_its_subspace_instead_of_a_division_by_zero():
    lv = perturbation_levels([0, 0, 2], sp.Matrix([[0, 1, 1], [1, 0, 0], [1, 0, 0]]))
    assert lv[0].e1 is None and lv[0].degenerate_with == [1] and lv[0].first_order_branches == [-1, 1]
    assert lv[2].e1 == 0 and lv[2].e2 == R(1, 2)                         # |V02|²/(E2 - E0) = 1/2: the non-degenerate level is untouched


def test_the_degenerate_first_order_branches_agree_with_exact_diagonalisation():
    v = sp.Matrix([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    lam = 1e-4
    exact = np.linalg.eigvalsh(np.diag([0.0, 0.0, 2.0]) + lam * np.array(v.evalf(), dtype=float))[:2]
    np.testing.assert_allclose(exact, [-lam, lam], atol=1e-7)


def test_comparing_a_degenerate_level_is_refused():
    lv = perturbation_levels([0, 0, 2], sp.Matrix([[0, 1, 1], [1, 0, 0], [1, 0, 0]]))
    with pytest.raises(ValueError, match="degenerate"):
        compare_with_exact([0, 0, 2], sp.eye(3), lv[0], [0.01, 0.02, 0.03])


def test_a_decoupled_level_has_zero_corrections():
    lv = perturbation_levels([0, 1, 2], sp.diag(0, 1, 0)[:, :])
    assert all(l.e2 == 0 for l in lv) and lv[1].e1 == 1


def test_symbolic_exactness_is_kept_for_irrational_matrix_elements():
    lv = perturbation_levels([0, 1], sp.Matrix([[0, sp.sqrt(3)], [sp.sqrt(3), 1]]))
    assert lv[0].e2 == -3 and lv[1].e1 == 1 and lv[1].e2 == 3


# ------------------------------------------------------------------ the checks must be able to fail

def test_a_wrong_second_order_coefficient_is_caught_by_the_measured_slope():
    levels, comp, _ = matrix_perturbation([0, 1, 3], V3, comparison_level=0)
    broken = copy.deepcopy(levels)
    broken[0].e2 = broken[0].e2 * 2
    comparison = compare_with_exact([0, 1, 3], V3, broken[0], comp.lambdas)
    failing = [c.label for c in verify_series([0, 1, 3], V3, broken, comparison, True).checks if not c.passed]
    assert "Order 2: error shrinks at least like λ^3" in failing and "Order 1: error shrinks at least like λ^2" not in failing


def test_a_wrong_first_order_coefficient_is_caught_at_every_order():
    levels, comp, _ = matrix_perturbation([0, 1, 4], sp.Matrix([[1, 1, 0], [1, 0, 1], [0, 1, -1]]), comparison_level=0)
    broken = copy.deepcopy(levels)
    broken[0].e1 = broken[0].e1 + 1
    comparison = compare_with_exact([0, 1, 4], sp.eye(3), broken[0], comp.lambdas)
    assert not verify_series([0, 1, 4], sp.Matrix([[1, 1, 0], [1, 0, 1], [0, 1, -1]]), broken, comparison, True).passed


def test_a_nonhermitian_perturbation_is_flagged():
    v = sp.Matrix([[0, 1], [2, 0]])
    checks = verify_series([0, 1], v, perturbation_levels([0, 1], v), None, True)
    assert "V is Hermitian" in [c.label for c in checks.checks if not c.passed]


def test_the_trace_sum_rules_fail_for_a_fabricated_set_of_levels():
    levels = perturbation_levels([0, 1, 3], V3)
    levels[0].e2 = levels[0].e2 + 1
    assert "Trace sum rule: Σ E2 = 0" in [c.label for c in verify_series([0, 1, 3], V3, levels, None, True).checks if not c.passed]


def test_a_positive_ground_state_second_order_shift_is_flagged():
    levels = perturbation_levels([0, 1, 3], V3)
    levels[0].e2 = sp.Integer(1)
    assert any("not positive" in c.label for c in verify_series([0, 1, 3], V3, levels, None, True).checks if not c.passed)


# ------------------------------------------------------------------ the oscillator comparison

def test_the_oscillator_comparison_passes_all_its_checks_including_the_independent_grid_solver():
    r = oscillator_perturbation({4: 1.0}, n_levels=3)
    assert r.checks.passed, [(c.label, c.detail) for c in r.checks.checks if not c.passed]
    labels = [c.label for c in r.checks.checks]
    assert "Agrees with the independent real-space solver" in labels and "The exact reference is converged in the basis size" in labels
    assert all(r.comparison.slopes[k] > k + 0.5 for k in (1, 2, 3))


def test_the_series_beats_the_lower_orders_only_while_lambda_is_small():
    r = oscillator_perturbation({4: 1.0}, n_levels=2, lambda_max=0.05)
    e1, e2, e3 = (r.comparison.errors[k] for k in (1, 2, 3))
    assert np.all(e3 < e2) and np.all(e2 < e1)
    large = oscillator_perturbation({4: 1.0}, n_levels=2, lambda_max=3.0)                # far outside the useful range
    assert large.comparison.errors[3][-1] > large.comparison.errors[2][-1] or large.comparison.errors[3][-1] > 0.1


def test_the_exact_energy_for_a_quadratic_perturbation_matches_the_closed_form():
    r = oscillator_perturbation({2: 1.0}, n_levels=3, comparison_level=1)
    lam = r.comparison.lambdas
    np.testing.assert_allclose(r.comparison.exact, 1.5 * np.sqrt(1 + 2 * lam), rtol=1e-10)


def test_a_mixed_polynomial_perturbation_still_passes_all_checks():
    assert oscillator_perturbation({2: 0.5, 4: 1.0}, n_levels=3).checks.passed


def test_the_asymptotic_nature_of_the_series_is_stated():
    assert any("asymptotic" in n for n in oscillator_perturbation({4: 1.0}, n_levels=2).notes)


@pytest.mark.parametrize("kwargs,match", [
    ({"n_levels": 0}, "between 1 and 12"), ({"n_levels": 13}, "between 1 and 12"), ({"comparison_level": 5, "n_levels": 3}, "must be one"),
    ({"basis_dim": 5, "n_levels": 4}, "at least"),
])
def test_unusable_settings_are_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        oscillator_perturbation({4: 1.0}, **kwargs)


def test_matrix_inputs_are_validated():
    with pytest.raises(ValueError, match="must be"):
        perturbation_levels([0, 1], sp.eye(3))
    with pytest.raises(ValueError, match="outside the basis"):
        perturbation_levels([0, 1], sp.eye(2), [5])
    with pytest.raises(ValueError, match="At most"):
        perturbation_levels(list(range(50)), sp.eye(50))


def test_the_suggested_lambda_scales_with_the_size_of_the_corrections():
    lv = levels_for({4: 1.0})
    assert suggest_lambda_max(lv[0], 1.0) > suggest_lambda_max(lv[3], 1.0) > 0           # higher levels are perturbed more
    assert suggest_lambda_max(perturbation_levels([0, 1], sp.zeros(2))[0], 1.0) == 0.1   # nothing to scale by


def test_progress_is_reported():
    seen = []
    oscillator_perturbation({4: 1.0}, n_levels=2, progress=lambda label, f: seen.append(f))
    assert seen == sorted(seen) and len(seen) >= 4
