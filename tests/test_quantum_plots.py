"""Every quantum figure builds from its result object, serialises, and shows what it is supposed to."""
import json

import numpy as np
import pytest
import sympy as sp

from modules.quantum_1d import solve_eigenstates
from modules.quantum_dynamics import evolve_wavepacket
from modules.quantum_operators import build_family, verify_bch
from modules.quantum_perturbation import oscillator_perturbation
from modules.quantum_plots import (
    bch_error_figure, bloch_sphere_figure, convergence_figure, density_matrix_figure, dynamics_diagnostics_figure,
    eigenstates_figure, perturbation_energy_figure, perturbation_error_figure, qubit_timeseries_figure,
    region_probability_figure, relaxation_figure, transmission_figure, wavepacket_animation_figure,
)
from modules.quantum_qubits import Noise, bloch_to_density, ramsey_curve, simulate_sequence, t1_curve


@pytest.fixture(scope="module")
def eig():
    return solve_eigenstates("harmonic", n_levels=4)


@pytest.fixture(scope="module")
def dyn():
    return evolve_wavepacket("barrier", {"V0": 2.5, "width": 1.0}, x0=-20, sigma=2.5, k0=2.0, domain=(-50, 50), n_points=1024,
                             n_frames=10, cross_check=False)


@pytest.fixture(scope="module")
def qrun():
    return simulate_sequence("H, Rz(pi/4), X", (0, 0, 1), Noise(t1=20, t_phi=15), idle_time=0.3)


def valid_json(fig):
    return json.loads(fig.to_json())


def test_the_eigenstate_figure_has_the_potential_and_one_trace_per_level_plus_its_energy_line(eig):
    fig = eigenstates_figure(eig)
    assert fig.data[0].name == "V(x)" and len(fig.data) == 1 + 2 * eig.n_levels
    named = [t.name for t in fig.data if t.name and t.name.startswith("n = ")]
    assert named == [f"n = {i}, E = {eig.energies[i]:.5g}" for i in range(eig.n_levels)]
    valid_json(fig)


def test_each_wavefunction_is_drawn_on_its_own_energy_line(eig):
    fig = eigenstates_figure(eig, levels=[2])
    curve = next(t for t in fig.data if t.name and t.name.startswith("n = 2"))
    assert np.mean(curve.y) == pytest.approx(eig.energies[2], abs=0.2)


def test_density_mode_is_non_negative_above_the_line_and_levels_can_be_chosen(eig):
    fig = eigenstates_figure(eig, "density", [0, 3])
    assert len([t for t in fig.data if t.name and t.name.startswith("n = ")]) == 2
    low = next(t for t in fig.data if t.name and t.name.startswith("n = 0"))
    assert np.min(low.y) >= eig.energies[0] - 1e-12
    assert eigenstates_figure(eig, levels=[99]).data[0].name == "V(x)"          # out-of-range levels are ignored


def test_the_convergence_figure_is_empty_without_an_analytic_spectrum_and_log_scaled_with_one(eig):
    assert convergence_figure(solve_eigenstates("custom")).data == ()
    fig = convergence_figure(eig)
    assert fig.layout.yaxis.type == "log" and len(fig.data[0].x) == eig.n_levels


def test_the_wavepacket_animation_has_one_frame_per_stored_time_and_a_slider(dyn):
    fig = wavepacket_animation_figure(dyn)
    assert len(fig.frames) == len(dyn.frame_times) and len(fig.layout.sliders[0].steps) == len(dyn.frame_times)
    assert fig.frames[0].name == f"{dyn.frame_times[0]:.3f}"
    np.testing.assert_allclose(fig.data[1].y, dyn.density[0], rtol=1e-5, atol=1e-9)
    assert [b.label for b in fig.layout.updatemenus[0].buttons] == ["▶ Play", "⏸ Pause"]
    valid_json(fig)


def test_the_animation_can_leave_out_the_real_part(dyn):
    fig = wavepacket_animation_figure(dyn, show_real=False)
    assert len(fig.data) == 2 and all(len(f.data) == 1 for f in fig.frames)


def test_frames_carry_only_the_changing_data_so_the_figure_stays_small(dyn):
    fig = wavepacket_animation_figure(dyn)
    assert all(f.data[0].x is None for f in fig.frames)


def test_the_diagnostics_figure_has_the_three_panels_and_the_ehrenfest_pair(dyn):
    fig = dynamics_diagnostics_figure(dyn)
    names = [t.name for t in fig.data]
    assert names == ["norm − 1", "E − E₀", "⟨x⟩ (simulation)", "x₀ + ∫⟨p⟩/m dt"]
    np.testing.assert_allclose(fig.data[2].y, fig.data[3].y, atol=0.05)             # Ehrenfest, visibly


def test_the_region_figure_is_stacked_probabilities_summing_to_one(dyn):
    fig = region_probability_figure(dyn)
    total = sum(np.array(t.y) for t in fig.data)
    np.testing.assert_allclose(total, 1.0, atol=1e-9)


def test_the_transmission_figure_puts_the_simulated_value_on_the_exact_curve(dyn):
    fig = transmission_figure(dyn)
    sim = next(t for t in fig.data if t.name == "T from the simulation")
    exact = next(t for t in fig.data if t.name == "packet-averaged T (exact)")
    assert sim.y[0] == pytest.approx(exact.y[0], abs=0.02)
    curve = next(t for t in fig.data if t.name == "T(E), exact")
    assert 0 <= min(curve.y) and max(curve.y) <= 1 + 1e-12


def test_there_is_no_transmission_figure_for_a_potential_without_flat_pieces():
    r = evolve_wavepacket("harmonic", {"w": 0.5}, x0=-4, sigma=1.0, k0=0.0, domain=(-20, 20), n_points=256, n_frames=4,
                          total_time=3.0, cross_check=False)
    assert transmission_figure(r).data == () and region_probability_figure(r).data == ()


def test_the_bloch_figure_has_the_sphere_both_paths_an_arrow_and_matching_frames(qrun):
    fig = bloch_sphere_figure(qrun)
    names = [t.name for t in fig.data if t.name]
    assert names == ["ideal (no noise)", "with noise", "state"]
    assert len(fig.frames) == len(fig.layout.sliders[0].steps) <= 120
    first = fig.frames[0].data[0]
    assert (first.x[1], first.y[1], first.z[1]) == tuple(qrun.bloch[0])
    last = fig.frames[-1].data[0]
    assert (last.x[1], last.y[1], last.z[1]) == pytest.approx(tuple(qrun.bloch[-1]))
    valid_json(fig)


def test_the_bloch_figure_limits_its_frame_count():
    long_run = simulate_sequence(",".join(["X"] * 30), (0, 0, 1))
    assert len(bloch_sphere_figure(long_run, max_frames=40).frames) <= 40


def test_the_timeseries_figure_has_components_purity_fidelity_and_a_band_per_gate(qrun):
    fig = qubit_timeseries_figure(qrun)
    assert [t.name for t in fig.data] == ["r_x", "r_y", "r_z", "purity Tr ρ²", "fidelity F"]
    assert len(fig.layout.shapes) == 2 * 3                                              # 3 gates x 2 panels
    np.testing.assert_allclose(fig.data[3].y, qrun.purity)


def test_the_density_matrix_figure_shows_real_and_imaginary_parts():
    rho = bloch_to_density([0.0, 1.0, 0.0])
    fig = density_matrix_figure(rho, "demo")
    assert fig.layout.title.text == "demo" and len(fig.data) == 2
    np.testing.assert_allclose(np.array(fig.data[1].z), rho.imag[::-1])


def test_the_relaxation_figure_annotates_the_input_and_fitted_times():
    n = Noise(t1=10.0, t_phi=20.0)
    ts = np.linspace(0, 40, 40)
    fig = relaxation_figure(ts, t1_curve(n, ts), ramsey_curve(n, ts), 10.0, n.t2, 10.0, 10.0)
    texts = [a.text for a in fig.layout.annotations if a.text and "fitted" in a.text]
    assert len(texts) == 2 and "T1 = 10" in texts[0]
    assert "no decoherence" in [a.text for a in relaxation_figure(ts, ts, ts, None, None, None, None).layout.annotations][-1]


def test_the_perturbation_figures_show_exact_and_orders_with_reference_slopes():
    r = oscillator_perturbation({4: 1.0}, n_levels=2)
    fe = perturbation_energy_figure(r.comparison)
    assert [t.name for t in fe.data] == ["exact", "through order 1", "through order 2", "through order 3"]
    fr = perturbation_error_figure(r.comparison)
    assert fr.layout.xaxis.type == fr.layout.yaxis.type == "log" and len(fr.data) == 6
    assert all("slope" in t.name for t in fr.data if t.name)


def test_the_bch_figure_shows_the_error_and_the_expected_power_law():
    s = sp.Matrix
    result = verify_bch(build_family("pauli")["sx"] / 2 + build_family("pauli")["sz"], build_family("pauli")["sy"] / 3, 3)
    fig = bch_error_figure(result)
    assert fig.data[1].name == "ε^4" and "fitted exponent" in fig.layout.title.text
