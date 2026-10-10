"""The 1D Schrödinger solver: against exact spectra, scaling laws, and its own failure detection."""
import numpy as np
import pytest

from modules.quantum_1d import (
    MAX_LEVELS, POTENTIALS, analytic_table, level_spacings, solve_eigenstates, tunnelling_splitting,
)
from modules.quantum_common import ExpressionError


def labels(result, passed=False):
    return [c.label for c in result.checks.checks if c.passed == passed]


@pytest.mark.parametrize("key", list(POTENTIALS))
def test_every_library_potential_solves_and_passes_all_its_own_checks(key):
    result = solve_eigenstates(key, n_points=2000 if key == "hydrogen" else 800)
    assert result.checks.passed, [(c.label, c.detail) for c in result.checks.checks if not c.passed]
    assert result.n_levels == POTENTIALS[key].n_levels


# ------------------------------------------------------------------ exact spectra

def test_the_infinite_well_levels_are_n_squared_pi_squared_over_2mL_squared():
    r = solve_eigenstates("box", domain=(0.0, 2.0), n_levels=5, hbar=1.5, mass=0.7)
    expected = np.arange(1, 6) ** 2 * np.pi ** 2 * 1.5 ** 2 / (2 * 0.7 * 4.0)
    np.testing.assert_allclose(r.energies, expected, rtol=1e-6)


def test_the_oscillator_levels_are_hbar_omega_n_plus_half_in_any_units():
    r = solve_eigenstates("harmonic", {"w": 3.0}, domain=(-4, 4), hbar=2.0, mass=2.0, n_points=1000, n_levels=6)
    np.testing.assert_allclose(r.energies, 2.0 * 3.0 * (np.arange(6) + 0.5), rtol=1e-6)


def test_hydrogen_radial_energies_are_minus_one_over_two_n_squared():
    r = solve_eigenstates("hydrogen", n_points=3000, n_levels=3)
    np.testing.assert_allclose(r.energies, [-0.5, -0.125, -1 / 18], rtol=5e-4)
    r2 = solve_eigenstates("hydrogen", {"l": 1.0}, n_points=3000, n_levels=2)
    np.testing.assert_allclose(r2.energies, [-0.125, -1 / 18], rtol=5e-4)


def test_hydrogen_ignores_hbar_and_mass_because_they_are_fixed_in_atomic_units():
    a = solve_eigenstates("hydrogen", n_points=1500, hbar=5.0, mass=9.0)
    b = solve_eigenstates("hydrogen", n_points=1500)
    np.testing.assert_allclose(a.energies, b.energies)
    assert a.hbar == 1.0 and a.mass == 1.0


def test_the_morse_levels_follow_the_closed_form_and_only_bound_ones_are_compared():
    r = solve_eigenstates("morse", n_levels=14, n_points=1500)
    assert len(r.analytic) == 10                                         # n_max = floor(sqrt(2mD)/(aħ) - 1/2) = 9
    np.testing.assert_allclose(r.energies[: len(r.analytic)], r.analytic, rtol=1e-3)
    assert any("bound" in n for n in r.notes) and len(r.energies) == 14


def test_the_poschl_teller_well_has_exactly_ceil_lambda_bound_states_at_the_known_energies():
    r = solve_eigenstates("poschl_teller", {"lam": 3.0, "a": 1.0}, n_levels=3, n_points=1500)
    np.testing.assert_allclose(r.energies, [-4.5, -2.0, -0.5], rtol=1e-4)


def test_the_finite_well_bound_states_satisfy_the_matching_conditions():
    r = solve_eigenstates("finite_well", {"V0": 10.0, "a": 2.0}, n_levels=5, n_points=1500)
    assert len(r.analytic) == 3                                          # z0 = sqrt(20) ≈ 4.47 -> three bound states
    for e in r.analytic:                                                 # independent of any grid: residual of the equations
        k, kappa = np.sqrt(2 * (e + 10.0)), np.sqrt(-2 * e)
        z = k * 1.0
        even = abs(k * np.tan(z) - kappa) < 1e-7
        odd = abs(-k / np.tan(z) - kappa) < 1e-7
        assert even or odd
    # the well's sharp edges fall between grid points, so 1e-3 is the honest tolerance (and the default one)
    np.testing.assert_allclose(r.energies[:3], r.analytic, rtol=1e-3)


def test_a_very_deep_finite_well_approaches_the_infinite_well():
    r = solve_eigenstates("finite_well", {"V0": 2000.0, "a": 2.0}, n_levels=3, domain=(-3, 3), n_points=1500)
    infinite = np.arange(1, 4) ** 2 * np.pi ** 2 / (2 * 4.0) - 2000.0
    np.testing.assert_allclose(r.analytic, infinite, rtol=2e-3)


# ------------------------------------------------------------------ structure of the solution

def test_states_are_orthonormal_with_the_expected_node_counts_and_parities():
    r = solve_eigenstates("harmonic", n_levels=6)
    gram = r.states.T @ r.states * (r.x[1] - r.x[0])
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-9)
    assert [o.nodes for o in r.observables] == list(range(6))
    assert [o.parity for o in r.observables] == ["even", "odd"] * 3


def test_the_oscillator_ground_state_saturates_the_uncertainty_bound_and_is_centred():
    o = solve_eigenstates("harmonic", n_points=1500).observables[0]
    assert o.std_x * o.std_p == pytest.approx(0.5, rel=1e-4)
    assert o.mean_x == pytest.approx(0.0, abs=1e-9) and o.std_x == pytest.approx(np.sqrt(0.5), rel=1e-4)


def test_virial_theorem_for_the_oscillator_kinetic_equals_potential_energy():
    for o in solve_eigenstates("harmonic", n_points=1500).observables[:4]:
        assert o.mean_t == pytest.approx(o.mean_v, rel=1e-3)


def test_the_sign_convention_makes_every_state_positive_at_its_largest_amplitude():
    r = solve_eigenstates("harmonic")
    for j in range(r.n_levels):
        assert r.states[np.argmax(np.abs(r.states[:, j])), j] > 0


def test_the_tunnelling_splitting_shrinks_as_the_barrier_between_the_wells_grows():
    low = tunnelling_splitting(solve_eigenstates("double_well", {"lam": 1.0, "x0": 1.5}, n_levels=2))
    high = tunnelling_splitting(solve_eigenstates("double_well", {"lam": 1.0, "x0": 2.0}, domain=(-5, 5), n_levels=2))
    assert 0 < high < low < 0.5


def test_the_tunnelling_splitting_is_not_defined_for_an_asymmetric_potential():
    r = solve_eigenstates("custom", custom_formula="0.5*x**2 + 0.3*x")
    assert tunnelling_splitting(r) is None


def test_a_custom_formula_reproduces_the_library_oscillator():
    custom = solve_eigenstates("custom", custom_formula="0.5*x**2", domain=(-8, 8), n_levels=4)
    lib = solve_eigenstates("harmonic", n_levels=4)
    np.testing.assert_allclose(custom.energies, lib.energies, rtol=1e-9)


def test_a_custom_formula_can_use_hbar_m_and_its_own_parameters_by_name():
    r = solve_eigenstates("custom", {"k": 4.0}, custom_formula="0.5*k*x**2 + 0*hbar*m", mass=1.0, n_levels=2)
    np.testing.assert_allclose(r.energies, [1.0, 3.0], rtol=1e-4)           # w = sqrt(k/m) = 2


# ------------------------------------------------------------------ convergence

def test_the_measured_convergence_order_is_two_for_a_smooth_potential():
    r = solve_eigenstates("harmonic", n_points=300)
    assert np.all(np.abs(r.orders - 2) < 0.2) and r.extrapolated.all()


def test_extrapolation_improves_on_the_users_own_grid():
    r = solve_eigenstates("harmonic", n_points=150, n_levels=3)
    exact = np.array([0.5, 1.5, 2.5])
    assert np.all(np.abs(r.energies - exact) < np.abs(r.raw_energies - exact))


def test_a_singular_potential_is_not_extrapolated_with_a_formula_that_does_not_apply():
    r = solve_eigenstates("hydrogen", n_points=1000, n_levels=3)
    assert not r.extrapolated.all() or np.nanmax(np.abs(r.orders - 2)) < 0.4
    assert any("order is not close to 2" in n for n in r.notes) or r.extrapolated.all()


def test_the_energy_changes_little_when_the_grid_is_refined():
    a = solve_eigenstates("harmonic", n_points=400)
    b = solve_eigenstates("harmonic", n_points=1600)
    np.testing.assert_allclose(a.energies, b.energies, rtol=1e-6)


# ------------------------------------------------------------------ the checks catch real problems

def test_a_box_too_narrow_for_the_states_is_flagged_as_changing_the_answer():
    r = solve_eigenstates("harmonic", domain=(-1.5, 1.5))
    assert not r.checks.passed and "Bound states are contained in the box" in labels(r)


def test_a_potential_with_nothing_bound_in_the_box_is_reported_as_such():
    r = solve_eigenstates("custom", custom_formula="-0.1*x**2", domain=(-3, 3))
    failed = [c for c in r.checks.checks if not c.passed]
    assert any("nothing is bound" in c.detail for c in failed)


def test_a_coarse_grid_that_cannot_resolve_the_spectrum_fails_the_analytic_comparison():
    r = solve_eigenstates("harmonic", n_points=60, n_levels=8, domain=(-10, 10), tolerance=1e-6)
    assert any("analytic spectrum" in l for l in labels(r))


def test_the_tolerance_is_what_decides_the_analytic_comparison():
    loose = solve_eigenstates("finite_well", tolerance=1e-1)
    tight = solve_eigenstates("finite_well", tolerance=1e-12)
    pick = lambda r: next(c.passed for c in r.checks.checks if "analytic" in c.label)
    assert pick(loose) and not pick(tight)


def test_a_wrong_analytic_formula_would_be_caught(monkeypatch):
    from dataclasses import replace
    from modules import quantum_1d
    broken = replace(POTENTIALS["harmonic"], analytic=lambda n, p, h, m, d: h * p["w"] * (np.arange(n) + 0.6))
    monkeypatch.setitem(quantum_1d.POTENTIALS, "harmonic", broken)
    assert any("analytic spectrum" in l for l in labels(solve_eigenstates("harmonic")))


# ------------------------------------------------------------------ inputs

@pytest.mark.parametrize("kwargs,match", [
    ({"key": "nope"}, "Unknown potential"),
    ({"key": "box", "domain": (1.0, 0.0)}, "left end"),
    ({"key": "box", "n_points": 10}, "grid points"),
    ({"key": "box", "n_points": 10 ** 6}, "grid points"),
    ({"key": "box", "n_levels": 0}, "levels"),
    ({"key": "box", "n_levels": MAX_LEVELS + 1}, "levels"),
    ({"key": "box", "hbar": 0.0}, "positive"),
    ({"key": "box", "mass": -1.0}, "positive"),
    ({"key": "box", "hbar": float("nan")}, "positive"),
])
def test_unusable_settings_are_refused_with_a_reason(kwargs, match):
    with pytest.raises(ValueError, match=match):
        solve_eigenstates(**kwargs)


@pytest.mark.parametrize("formula", ["__import__('os')", "x.__class__", "2x", "unknown(x)", "", "y*x"])
def test_a_bad_custom_formula_is_an_expression_error(formula):
    # including an EMPTY one: clearing the box must not quietly solve the default potential instead
    with pytest.raises(ExpressionError):
        solve_eigenstates("custom", custom_formula=formula)


def test_progress_is_reported_once_per_grid_and_moves_forward():
    seen = []
    solve_eigenstates("harmonic", n_points=200, progress=lambda label, f: seen.append((label, f)))
    assert [f for _, f in seen] == sorted(f for _, f in seen) and len(seen) == 4
    assert "200 points" in seen[0][0] and "400 points" in seen[1][0] and "800 points" in seen[2][0]


def test_analytic_table_pairs_each_level_with_its_exact_value_and_error():
    rows = analytic_table(solve_eigenstates("harmonic", n_levels=3))
    assert [r["n"] for r in rows] == [0, 1, 2]
    assert all("E (analytic)" in r and r["relative error"] < 1e-6 for r in rows)
    custom_rows = analytic_table(solve_eigenstates("custom", n_levels=2))
    assert all("E (analytic)" not in r for r in custom_rows)


def test_level_spacings():
    np.testing.assert_allclose(level_spacings(np.array([1.0, 3.0, 6.0])), [2.0, 3.0])
