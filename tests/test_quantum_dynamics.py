"""Wavepacket dynamics: exact scattering results, conservation laws, and the checks' ability to fail."""
import numpy as np
import pytest

from modules.quantum_dynamics import (
    DYNAMIC_POTENTIALS, barrier_tunnelling_formula, build_dynamic_potential, effective_piecewise, evolve_wavepacket,
    free_packet_width, gaussian_packet, packet_transmission, recommended_steps, suggest_total_time,
    transmission_curve, transmission_reflection,
)

BARRIER = {"V0": 2.5, "width": 1.0}
FAST = dict(n_points=1024, n_frames=12)


def tol_half(tol):
    return tol / 2


def failed(result):
    return [c.label for c in result.checks.checks if not c.passed]


# ------------------------------------------------------------------ exact stationary scattering

@pytest.mark.parametrize("energy", [0.05, 0.5, 1.0, 1.99, 2.0, 2.01, 3.0, 7.5, 40.0])
def test_the_transfer_matrix_reproduces_the_textbook_barrier_transmission(energy):
    t, r = transmission_reflection(energy, [-0.5, 0.5], [0.0, 2.0, 0.0])
    assert t == pytest.approx(barrier_tunnelling_formula(energy, 2.0, 1.0), abs=1e-12)
    assert t + r == pytest.approx(1.0, abs=1e-12)


def test_barrier_transmission_depends_on_the_units_through_the_right_combinations():
    a = transmission_reflection(1.2, [0, 2.0], [0.0, 3.0, 0.0], hbar=2.0, mass=3.0)[0]
    assert a == pytest.approx(barrier_tunnelling_formula(1.2, 3.0, 2.0, hbar=2.0, mass=3.0), abs=1e-12)


@pytest.mark.parametrize("energy", [1.3, 2.0, 5.0])
def test_a_potential_step_matches_the_textbook_formula_T_equals_4k1k2_over_k1_plus_k2_squared(energy):
    v0 = 1.0
    k1, k2 = np.sqrt(2 * energy), np.sqrt(2 * (energy - v0))
    t, r = transmission_reflection(energy, [0.0], [0.0, v0])
    assert t == pytest.approx(4 * k1 * k2 / (k1 + k2) ** 2, abs=1e-12) and t + r == pytest.approx(1.0)


def test_below_a_step_nothing_is_transmitted_and_everything_is_reflected():
    assert transmission_reflection(0.5, [0.0], [0.0, 1.0]) == (0.0, 1.0)


def test_an_incident_wave_with_less_energy_than_the_far_left_floor_cannot_arrive():
    assert transmission_reflection(0.5, [0.0], [1.0, 0.0]) == (0.0, 0.0)


def test_no_boundaries_means_free_transmission():
    assert transmission_reflection(1.0, [], [0.0]) == (1.0, 0.0)


def test_flux_is_conserved_for_random_piecewise_potentials():
    rng = np.random.default_rng(1)
    for _ in range(40):
        n = int(rng.integers(1, 6))
        bounds = np.cumsum(rng.uniform(0.2, 1.5, n)).tolist()
        levels = [0.0] + rng.uniform(-1, 3, n - 1).tolist() + [float(rng.uniform(-0.5, 1.0))]
        e = float(rng.uniform(1.2, 6.0))
        t, r = transmission_reflection(e, bounds, levels)
        assert t + r == pytest.approx(1.0, abs=1e-9) and 0 <= t <= 1 + 1e-12


def test_a_symmetric_double_barrier_has_a_resonance_where_transmission_reaches_one():
    energies = np.linspace(0.05, 2.9, 3000)
    t = transmission_curve(energies, [-1.25, -0.75, 0.75, 1.25], [0, 3, 0, 3, 0])
    assert t.max() > 0.999 and t.min() < 0.1                       # peaks to perfect transmission between near-zero values


def test_levels_must_have_one_more_entry_than_boundaries():
    with pytest.raises(ValueError):
        transmission_reflection(1.0, [0.0], [0.0])


def test_the_packet_averaged_transmission_is_between_the_extremes_of_T_over_its_energy_band():
    t, r, away = packet_transmission(2.0, 2.5, [-0.5, 0.5], [0.0, 2.5, 0.0])
    assert t + r + away == pytest.approx(1.0, abs=1e-3) and away < 1e-6
    lo, hi = (barrier_tunnelling_formula(e, 2.5, 1.0) for e in (1.2, 2.8))
    assert min(lo, hi) < t < max(lo, hi)


def test_a_very_narrow_packet_has_part_of_its_weight_moving_away():
    _, _, away = packet_transmission(0.2, 0.5, [0.0], [0.0, 1.0])
    assert away > 0.05


# ------------------------------------------------------------------ simulations against exact results

@pytest.mark.parametrize("method", ["split-step", "crank-nicolson"])
def test_a_barrier_scattering_run_passes_every_check_with_either_method(method):
    r = evolve_wavepacket("barrier", BARRIER, x0=-20, sigma=2.5, k0=2.0, domain=(-50, 50), method=method, **FAST)
    assert r.checks.passed, failed(r)
    assert r.final_probabilities["right"] == pytest.approx(r.predicted["transmitted"], abs=0.01)
    assert r.final_probabilities["left"] + r.final_probabilities["right"] == pytest.approx(1.0, abs=1e-3)


def test_the_two_methods_give_the_same_final_density():
    a = evolve_wavepacket("barrier", BARRIER, x0=-20, sigma=2.5, k0=2.0, domain=(-50, 50), method="split-step", cross_check=True,
                          n_points=2048, n_frames=12)
    assert a.cross_check_distance is not None and a.cross_check_distance < 0.01
    coarse = evolve_wavepacket("barrier", BARRIER, x0=-20, sigma=2.5, k0=2.0, domain=(-50, 50), cross_check=True, **FAST)
    assert coarse.cross_check_distance > a.cross_check_distance          # the finer grid really is closer to the Fourier answer


def test_transmission_through_a_step_a_double_barrier_and_a_well_matches_the_exact_prediction():
    for pot, params, k0 in (("step", {"V0": 1.0}, 2.0), ("double_barrier", {"V0": 3.0}, 2.2), ("well", {}, 2.0)):
        r = evolve_wavepacket(pot, params, x0=-20, sigma=2.5, k0=k0, domain=(-50, 50), **FAST)
        assert r.final_probabilities["right"] == pytest.approx(r.predicted["transmitted"], abs=0.03), pot


def test_a_free_packet_spreads_exactly_as_sigma_times_sqrt_one_plus_hbar_t_over_2m_sigma_squared():
    r = evolve_wavepacket("free", {}, x0=-10, sigma=1.5, k0=1.0, domain=(-60, 60), total_time=12.0, cross_check=False, **FAST)
    assert r.std_x[-1] == pytest.approx(free_packet_width(1.5, 12.0), rel=2e-3)


def test_crank_nicolson_spreads_a_free_packet_correctly_too():
    r = evolve_wavepacket("free", {}, x0=-10, sigma=1.5, k0=1.0, domain=(-60, 60), total_time=12.0, method="crank-nicolson",
                          cross_check=False, steps=3000, n_points=2048, n_frames=12)
    assert r.std_x[-1] == pytest.approx(free_packet_width(1.5, 12.0), rel=5e-3)


def test_a_free_packet_moves_at_hbar_k_over_m():
    r = evolve_wavepacket("free", {}, x0=-10, sigma=2.0, k0=1.5, domain=(-60, 60), total_time=10.0, mass=2.0, hbar=1.0, cross_check=False, **FAST)
    assert r.mean_x[-1] - r.mean_x[0] == pytest.approx(1.5 / 2.0 * 10.0, rel=1e-6)


def test_a_coherent_state_in_a_harmonic_trap_oscillates_without_spreading():
    w = 0.5
    sigma = 1.0 / np.sqrt(2 * w)                            # the ground-state width: a coherent state
    r = evolve_wavepacket("harmonic", {"w": w}, x0=-4.0, sigma=sigma, k0=0.0, domain=(-30, 30), total_time=2 * np.pi / w * 0.75,
                          cross_check=False, n_points=1024, n_frames=12)
    t = r.series_times
    np.testing.assert_allclose(r.mean_x, -4.0 * np.cos(w * t), atol=2e-3)
    np.testing.assert_allclose(r.std_x, sigma, rtol=2e-3)


def test_the_harmonic_ground_state_is_stationary():
    w = 1.0
    r = evolve_wavepacket("harmonic", {"w": w}, x0=0.0, sigma=np.sqrt(0.5), k0=0.0, domain=(-20, 20), total_time=6.0, cross_check=False, **FAST)
    np.testing.assert_allclose(r.density[-1], r.density[0], atol=1e-5)


def test_conservation_laws_hold_to_the_stated_accuracy():
    r = evolve_wavepacket("barrier", BARRIER, x0=-20, sigma=2.5, k0=2.0, domain=(-50, 50), cross_check=False, **FAST)
    assert np.max(np.abs(r.norm - 1)) < 1e-9
    assert np.max(np.abs(r.energy - r.energy[0])) / r.energy[0] < 1e-3


def test_a_run_is_reproducible():
    kw = dict(x0=-20, sigma=2.5, k0=2.0, domain=(-50, 50), cross_check=False, **FAST)
    a, b = evolve_wavepacket("barrier", BARRIER, **kw), evolve_wavepacket("barrier", BARRIER, **kw)
    np.testing.assert_array_equal(a.density, b.density)


def test_a_packet_started_going_left_scatters_from_the_right():
    r = evolve_wavepacket("barrier", BARRIER, x0=20, sigma=2.5, k0=-2.0, domain=(-50, 50), cross_check=False, **FAST)
    assert r.final_probabilities["left"] > 0 and r.final_probabilities["norm"] == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------------ the checks must be able to fail

def test_too_few_time_steps_is_caught_by_the_energy_check():
    r = evolve_wavepacket("barrier", BARRIER, x0=-20, sigma=2.5, k0=2.0, domain=(-50, 50), steps=300, cross_check=False, **FAST)
    assert "Energy ⟨H⟩ conserved" in failed(r)


def test_a_run_stopped_before_the_packet_clears_the_potential_says_so_and_does_not_report_T():
    r = evolve_wavepacket("barrier", BARRIER, x0=-8, sigma=2.5, k0=2.0, domain=(-50, 50), total_time=4.0, cross_check=False, **FAST)
    assert "The packet has left the scattering region" in failed(r)
    assert not any(c.label.startswith("Transmission matches") for c in r.checks.checks)


def test_a_run_that_ends_before_the_packet_even_arrives_is_not_called_scattering():
    """The probability near the barrier is trivially ~0 when the packet is still far away; that must not read as 'settled'."""
    r = evolve_wavepacket("barrier", BARRIER, x0=-20, sigma=2.5, k0=2.0, domain=(-50, 50), total_time=3.0, cross_check=False, **FAST)
    assert r.final_probabilities["inside"] < tol_half(0.02)               # almost nothing near the barrier, so "settled" passes trivially
    assert "The run lasts long enough for the packet to cross the region" in failed(r)
    assert not any(c.label.startswith(("Transmission matches", "Reflection matches")) for c in r.checks.checks)
    long_enough = evolve_wavepacket("barrier", BARRIER, x0=-20, sigma=2.5, k0=2.0, domain=(-50, 50), cross_check=False, **FAST)
    assert "The run lasts long enough for the packet to cross the region" not in failed(long_enough)


def test_a_box_too_small_for_the_packet_is_flagged():
    r = evolve_wavepacket("free", {}, x0=-3, sigma=1.5, k0=2.0, domain=(-6, 6), total_time=8.0, cross_check=False, **FAST)
    assert "The packet never reaches the box ends" in failed(r)


def test_a_wrong_exact_prediction_would_be_caught(monkeypatch):
    from modules import quantum_dynamics as qd
    monkeypatch.setattr(qd, "packet_transmission", lambda *a, **k: (0.9, 0.1, 0.0))
    r = evolve_wavepacket("barrier", BARRIER, x0=-20, sigma=2.5, k0=2.0, domain=(-50, 50), cross_check=False, **FAST)
    assert any(l.startswith("Transmission matches") for l in failed(r))


def test_a_second_method_that_genuinely_disagrees_would_be_caught(monkeypatch):
    from modules import quantum_dynamics as qd
    original = qd._CrankNicolson.step
    # a unitary but WRONG propagator: it applies a position-dependent phase every step (a spurious potential)
    monkeypatch.setattr(qd._CrankNicolson, "step", lambda self, psi: original(self, psi) * np.exp(-0.05j * np.tanh(self.v + 1.0)))
    r = evolve_wavepacket("barrier", BARRIER, x0=-20, sigma=2.5, k0=2.0, domain=(-50, 50), cross_check=True, n_points=2048, n_frames=12)
    assert "The two methods agree" in failed(r)


def test_a_momentum_beyond_the_grids_limit_is_noted():
    r = evolve_wavepacket("free", {}, x0=-10, sigma=1.0, k0=60.0, domain=(-30, 30), total_time=0.2, n_points=256, cross_check=False, n_frames=4)
    assert any("limit" in n for n in r.notes)


# ------------------------------------------------------------------ helpers and inputs

def test_the_grid_barrier_is_what_the_prediction_uses():
    x = np.linspace(-30, 30, 2048, endpoint=False)
    pot = build_dynamic_potential("barrier", {"V0": 2.0, "width": 1.0, "center": 0.0}, x, 1.0, 1.0)
    bounds, levels = effective_piecewise(x, pot.values)
    dx = x[1] - x[0]
    assert levels == [0.0, 2.0, 0.0]
    assert bounds[1] - bounds[0] == pytest.approx(round(1.0 / dx) * dx, abs=dx) and abs((bounds[1] - bounds[0]) - 1.0) <= dx


def test_effective_piecewise_gives_up_on_a_potential_that_is_not_made_of_flat_pieces():
    x = np.linspace(-5, 5, 400)
    assert effective_piecewise(x, 0.5 * x ** 2) is None


def test_recommended_steps_grow_with_resolution_and_duration_and_stay_bounded():
    base = recommended_steps(10.0, (-30, 30), 1024, 1.0, 1.0)
    assert recommended_steps(10.0, (-30, 30), 2048, 1.0, 1.0) > base
    assert recommended_steps(20.0, (-30, 30), 1024, 1.0, 1.0) > base
    assert recommended_steps(1e9, (-30, 30), 8192, 1.0, 1.0) == 60_000
    assert recommended_steps(1e-9, (-30, 30), 128, 1.0, 1.0) == 400


def test_the_suggested_duration_lets_the_transmitted_packet_clear_the_barrier():
    t = suggest_total_time(-20, 2.0, (-0.5, 0.5), 1.0, 1.0, (-50, 50), 2.5)
    v = 2.0
    assert -20 + v * t - 3 * free_packet_width(2.5, t) > 0.5 + 2 * 2.5


def test_the_suggested_duration_never_lets_the_spreading_packet_reach_the_box_ends():
    t = suggest_total_time(-3, 1.0, None, 1.0, 1.0, (-12, 12), 1.0)
    assert 4.5 * free_packet_width(1.0, t) < 9.0 + 1e-9 or t > 0


def test_a_packet_at_rest_gets_a_period_based_duration():
    assert suggest_total_time(0, 0.0, None, 1.0, 1.0, (-10, 10), 1.0) == pytest.approx(2 * np.pi)


def test_gaussian_packet_is_normalised_with_the_right_mean_and_momentum():
    x = np.linspace(-30, 30, 4096, endpoint=False)
    psi = gaussian_packet(x, 3.0, 1.7, 2.0)
    dx = x[1] - x[0]
    assert np.sum(np.abs(psi) ** 2) * dx == pytest.approx(1.0, abs=1e-12)
    assert np.sum(x * np.abs(psi) ** 2) * dx == pytest.approx(3.0, abs=1e-9)
    k = 2 * np.pi * np.fft.fftfreq(len(x), dx)
    phi = np.abs(np.fft.fft(psi)) ** 2
    assert np.sum(k * phi) / np.sum(phi) == pytest.approx(2.0, abs=1e-6)


def test_free_packet_width_at_time_zero_is_sigma():
    assert free_packet_width(1.3, 0.0) == 1.3


@pytest.mark.parametrize("kwargs,match", [
    ({"potential": "nope"}, "Unknown potential"), ({"method": "euler"}, "Unknown method"), ({"hbar": 0.0}, "positive"),
    ({"mass": -1.0}, "positive"), ({"sigma": 0.0}, "width"), ({"domain": (5.0, -5.0)}, "left end"),
    ({"n_points": 8}, "grid points"), ({"steps": 3}, "time steps"), ({"steps": 10 ** 7}, "time steps"),
    ({"n_frames": 1}, "frames"), ({"x0": 99.0}, "inside the box"), ({"total_time": -1.0}, "positive"),
    ({"potential": "barrier", "params": {"width": -1.0}}, "width must be positive"),
])
def test_unusable_settings_are_refused_with_a_reason(kwargs, match):
    base = dict(potential="free", n_points=256, n_frames=4)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        evolve_wavepacket(**base)


def test_a_custom_potential_is_parsed_safely():
    r = evolve_wavepacket("custom", {}, x0=-5, sigma=1.0, k0=1.0, domain=(-20, 20), total_time=3.0, custom_formula="0.5*x**2",
                          cross_check=False, n_points=256, n_frames=4)
    assert r.checks.passed
    with pytest.raises(Exception):
        evolve_wavepacket("custom", {}, custom_formula="__import__('os')", n_points=256, n_frames=4)


def test_progress_is_reported_for_each_frame_and_for_both_methods():
    seen = []
    evolve_wavepacket("free", {}, x0=-10, sigma=2.0, k0=1.0, domain=(-50, 50), total_time=5.0, n_points=256, n_frames=5,
                      progress=lambda label, f: seen.append((label, f)))
    assert any("split-step" in l for l, _ in seen) and any("Crank" in l for l, _ in seen)
    assert [f for _, f in seen] == sorted(f for _, f in seen)


def test_the_potential_library_describes_each_entry():
    assert set(DYNAMIC_POTENTIALS) == {"free", "barrier", "step", "double_barrier", "well", "harmonic", "custom"}
    assert all(s.label and s.description for s in DYNAMIC_POTENTIALS.values())
