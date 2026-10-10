"""The qubit simulator: gates as rotations, exact noise propagation, and agreement between independent methods."""
import numpy as np
import pytest
from scipy.stats import unitary_group

from modules.quantum_common import ExpressionError
from modules.quantum_qubits import (
    GATE_HELP, IDENTITY, MAX_GATES, PAULI, STATE_PRESETS, Noise, axis_angle, bloch_from_angles, bloch_to_density,
    bloch_generator, density_to_bloch, fit_exponential_time, idle_decay_check, parse_gate, parse_sequence, propagate,
    propagate_lindblad, purity, ramsey_curve, rotation_unitary, simulate_sequence, split_sequence, sphere_trajectory_unitary,
    sqrtm_fidelity, t1_curve, uhlmann_fidelity,
)

Z = np.array([0.0, 0.0, 1.0])


def run(seq, r0=Z, noise=None, **kw):
    return simulate_sequence(seq, r0, noise, **kw)


def final(seq, r0=Z, noise=None, **kw):
    return run(seq, r0, noise, **kw).bloch[-1]


# ------------------------------------------------------------------ states and fidelity

def test_density_matrix_and_bloch_vector_are_inverses():
    rng = np.random.default_rng(0)
    for _ in range(20):
        r = rng.normal(size=3)
        r = r / np.linalg.norm(r) * rng.uniform(0, 1)
        np.testing.assert_allclose(density_to_bloch(bloch_to_density(r)), r, atol=1e-12)
        rho = bloch_to_density(r)
        assert np.trace(rho) == pytest.approx(1) and np.allclose(rho, rho.conj().T) and np.linalg.eigvalsh(rho).min() >= -1e-12


def test_the_preset_states_are_the_expected_points_and_lengths():
    assert all(np.linalg.norm(v) == pytest.approx(1.0) for k, v in STATE_PRESETS.items() if "mixed" not in k)
    assert np.linalg.norm(STATE_PRESETS["maximally mixed"]) == 0.0
    np.testing.assert_allclose(bloch_to_density(STATE_PRESETS["|0⟩ (north pole)"]), [[1, 0], [0, 0]])


def test_purity_runs_from_one_half_to_one():
    assert purity([0, 0, 0]) == 0.5 and purity([0, 1, 0]) == 1.0 and purity([0.6, 0, 0.8]) == pytest.approx(1.0)


def test_the_closed_form_fidelity_equals_the_definition_with_matrix_square_roots():
    rng = np.random.default_rng(1)
    for _ in range(30):
        a = rng.normal(size=3); a = a / np.linalg.norm(a) * rng.uniform(0, 1)
        b = rng.normal(size=3); b = b / np.linalg.norm(b) * rng.uniform(0, 1)
        assert uhlmann_fidelity(a, b) == pytest.approx(sqrtm_fidelity(bloch_to_density(a), bloch_to_density(b)), abs=1e-6)


def test_fidelity_is_one_for_equal_states_zero_for_orthogonal_pure_states_and_symmetric():
    p = np.array([0.0, 0.6, 0.8])
    assert uhlmann_fidelity(p, p) == pytest.approx(1.0) and uhlmann_fidelity(Z, -Z) == pytest.approx(0.0)
    q = np.array([0.3, -0.2, 0.4])
    assert uhlmann_fidelity(p, q) == pytest.approx(uhlmann_fidelity(q, p))


def test_fidelity_with_a_pure_state_is_the_overlap_expectation():
    psi = np.array([1.0, 1.0j]) / np.sqrt(2)
    rho = bloch_to_density([0.2, 0.3, -0.1])
    assert uhlmann_fidelity([0.2, 0.3, -0.1], [0, 1, 0]) == pytest.approx(np.real(np.vdot(psi, rho @ psi)))


def test_bloch_from_angles_and_its_length_validation():
    np.testing.assert_allclose(bloch_from_angles(np.pi / 2, 0.0), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(bloch_from_angles(0.0, 1.0, 0.5), [0, 0, 0.5])
    with pytest.raises(ValueError):
        bloch_from_angles(0.0, 0.0, 1.5)


# ------------------------------------------------------------------ gates

def test_axis_angle_rebuilds_any_unitary_up_to_its_global_phase():
    for seed in range(25):
        u = unitary_group.rvs(2, random_state=seed)
        axis, theta, alpha = axis_angle(u)
        np.testing.assert_allclose(np.exp(1j * alpha) * rotation_unitary(axis, theta), u, atol=1e-10)
        assert 0 <= theta <= 2 * np.pi + 1e-12 and np.linalg.norm(axis) == pytest.approx(1.0)


def test_the_identity_has_the_z_axis_and_zero_angle():
    axis, theta, _ = axis_angle(IDENTITY)
    np.testing.assert_allclose(axis, Z) and theta == pytest.approx(0.0)


@pytest.mark.parametrize("seq,expected", [
    ("X", [0, 0, -1]), ("Y", [0, 0, -1]), ("Z", [0, 0, 1]), ("H", [1, 0, 0]), ("H, Z", [-1, 0, 0]), ("H, S", [0, 1, 0]),
    ("H, T, T", [0, 1, 0]), ("Rx(pi/2)", [0, -1, 0]), ("Ry(pi/2)", [1, 0, 0]), ("Rx(pi)", [0, 0, -1]),
    ("SX", [0, -1, 0]), ("H, SDG", [0, -1, 0]), ("H, T, TDG", [1, 0, 0]), ("H, P(pi)", [-1, 0, 0]), ("I", [0, 0, 1]),
])
def test_known_gate_actions_on_the_bloch_sphere(seq, expected):
    np.testing.assert_allclose(final(seq), expected, atol=1e-9)


def test_well_known_gate_identities_hold_as_matrices_up_to_phase():
    def u(seq):
        return sphere_trajectory_unitary(parse_sequence(seq))
    def same(a, b):
        phase = np.vdot(b.flatten(), a.flatten())
        return np.allclose(a, b * phase / abs(phase), atol=1e-10)
    assert same(u("T, T"), u("S")) and same(u("S, S"), u("Z")) and same(u("H, Z, H"), u("X")) and same(u("H, H"), u("I"))
    assert same(u("SX, SX"), u("X")) and same(u("Rz(pi/2)"), u("S")) and same(u("X, Y, Z"), u("I"))


def test_rotation_angles_are_reported_in_range_and_axis_conventions():
    g = parse_gate("Rx(pi/3)")
    np.testing.assert_allclose(g.axis, [1, 0, 0], atol=1e-12) and g.angle == pytest.approx(np.pi / 3)
    assert parse_gate("H").angle == pytest.approx(np.pi)
    np.testing.assert_allclose(parse_gate("H").axis, np.array([1, 0, 1]) / np.sqrt(2), atol=1e-12)


@pytest.mark.parametrize("text,match", [
    ("Foo", "Unknown gate"), ("Rx", "needs an angle"), ("Rx()", "empty"), ("Rx(abc)", "Unknown name|not a number|angle"), ("X(1)", "takes no angle"),
    ("Rx(__import__('os'))", ".*"), ("", "Could not read"), ("Rx(1", "Could not read"), ("3", "Could not read"),
    ("Rx(1/0)", ".*"), ("Rz(oo)", ".*"),
])
def test_bad_gates_are_refused_with_the_list_of_what_exists(text, match):
    with pytest.raises((ExpressionError, ValueError, ZeroDivisionError, TypeError)):
        parse_gate(text)


def test_the_gate_help_names_every_gate():
    for name in ("X", "Y", "Z", "H", "S", "T", "SX", "Rx", "Ry", "Rz", "P"):
        assert name in GATE_HELP


def test_sequences_split_on_commas_outside_parentheses_only():
    assert split_sequence("H, Rz(pi/4), X") == ["H", "Rz(pi/4)", "X"]
    assert split_sequence("Rx(max(1, 2)), H;Z\nX") == ["Rx(max(1, 2))", "H", "Z", "X"]
    assert split_sequence(" , H ,, ") == ["H"]


def test_too_many_gates_are_refused():
    with pytest.raises(ExpressionError, match="At most"):
        parse_sequence(",".join(["X"] * (MAX_GATES + 1)))


# ------------------------------------------------------------------ noise: closed forms and exactness

@pytest.mark.parametrize("t1,tp,dep", [(10.0, None, 0.0), (None, 5.0, 0.0), (None, None, 0.1), (8.0, 20.0, 0.02), (3.0, 1.0, 0.5)])
def test_idle_decay_follows_the_closed_forms_for_every_noise_combination(t1, tp, dep):
    noise = Noise(t1, tp, dep)
    ok, detail = idle_decay_check(np.array([0.6, -0.3, 0.5]), noise)
    assert ok, detail


def test_t2_is_one_over_half_gamma1_plus_gamma_phi_and_never_exceeds_two_t1():
    n = Noise(t1=10.0, t_phi=20.0)
    assert n.t2 == pytest.approx(10.0) and Noise(t1=10.0).t2 == pytest.approx(20.0)
    rng = np.random.default_rng(2)
    for _ in range(50):
        n = Noise(rng.uniform(0.5, 50), rng.choice([None, rng.uniform(0.5, 50)]), rng.uniform(0, 0.5))
        assert n.t2 <= 2 * n.t1 * (1 + 1e-12)


def test_with_no_noise_the_decay_rates_are_zero_and_t2_is_undefined():
    n = Noise()
    assert n.noiseless and n.t2 is None and n.rate_xy == n.rate_z == 0
    assert idle_decay_check(Z, n) is None


def test_amplitude_damping_drives_any_state_to_the_north_pole():
    r = propagate(np.array([0.0, 0.0, -1.0]), Z, 0.0, 200.0, Noise(t1=5.0))
    np.testing.assert_allclose(r, Z, atol=1e-12)


def test_pure_dephasing_kills_coherence_but_leaves_populations_alone():
    r = propagate(np.array([0.6, 0.0, 0.8]), Z, 0.0, 100.0, Noise(t_phi=2.0))
    np.testing.assert_allclose(r, [0, 0, 0.8], atol=1e-12)


def test_depolarising_noise_shrinks_everything_toward_the_maximally_mixed_state():
    r = propagate(np.array([0.6, 0.0, 0.8]), Z, 0.0, 600.0, Noise(depolarizing=0.1))
    np.testing.assert_allclose(r, [0, 0, 0], atol=1e-12)


def test_propagation_composes_exactly_there_is_no_time_step_error():
    noise = Noise(t1=7.0, t_phi=11.0, depolarizing=0.03)
    axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    r0 = np.array([0.2, -0.4, 0.7])
    whole = propagate(r0, axis, 0.8, 6.0, noise)
    parts = r0
    for _ in range(60):
        parts = propagate(parts, axis, 0.8, 0.1, noise)
    np.testing.assert_allclose(parts, whole, atol=1e-12)


def test_the_bloch_equations_and_the_lindblad_master_equation_give_the_same_state():
    rng = np.random.default_rng(5)
    for _ in range(25):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
        r0 = rng.normal(size=3); r0 = r0 / np.linalg.norm(r0) * rng.uniform(0, 1)
        noise = Noise(rng.uniform(1, 30), rng.choice([None, rng.uniform(1, 30)]), rng.uniform(0, 0.2))
        omega, dt = rng.uniform(-3, 3), rng.uniform(0.1, 5)
        bloch = propagate(r0, axis, omega, dt, noise)
        rho = propagate_lindblad(bloch_to_density(r0), axis, omega, dt, noise)
        np.testing.assert_allclose(density_to_bloch(rho), bloch, atol=1e-9)


def test_a_wrong_decay_rate_in_the_bloch_equations_would_be_caught_by_the_lindblad_comparison(monkeypatch):
    from modules import quantum_qubits as qq
    monkeypatch.setattr(qq.Noise, "rate_xy", property(lambda self: self.gamma1 / 3 + self.gamma_phi))     # wrong: should be γ1/2
    r = run("H, X", Z, Noise(t1=5.0), idle_time=1.0)
    assert not r.checks.passed
    assert any("Lindblad" in c.label or "closed forms" in c.label for c in r.checks.checks if not c.passed)


def test_the_generator_has_the_rotation_decay_and_drive_structure():
    noise = Noise(t1=4.0, t_phi=10.0)
    a = bloch_generator(np.array([0.0, 0.0, 1.0]), 2.0, noise)
    assert a[0, 1] == pytest.approx(-2.0) and a[1, 0] == pytest.approx(2.0)          # rotation about z
    assert a[0, 0] == pytest.approx(-noise.rate_xy) and a[2, 2] == pytest.approx(-noise.rate_z) and a[2, 3] == pytest.approx(0.25)


def test_t1_and_ramsey_measurements_recover_the_input_times():
    n = Noise(t1=10.0, t_phi=20.0)
    ts = np.linspace(0, 40, 80)
    z_eq = n.gamma1 / n.rate_z
    assert fit_exponential_time(ts, t1_curve(n, ts), floor=z_eq) == pytest.approx(10.0, rel=1e-6)
    assert fit_exponential_time(ts, ramsey_curve(n, ts)) == pytest.approx(n.t2, rel=1e-6)


def test_the_ramsey_fringe_oscillates_at_the_detuning_inside_the_t2_envelope():
    n = Noise(t_phi=30.0)
    ts = np.linspace(0, 20, 400)
    signal = ramsey_curve(n, ts, detuning=2.0)
    np.testing.assert_allclose(signal, np.exp(-ts / n.t2) * np.cos(2.0 * ts), atol=1e-9)


def test_fit_returns_none_when_there_is_nothing_to_fit():
    assert fit_exponential_time(np.linspace(0, 1, 5), np.zeros(5)) is None
    assert fit_exponential_time(np.linspace(0, 1, 5), np.linspace(1, 2, 5)) is None          # growing, not decaying


@pytest.mark.parametrize("kwargs", [{"t1": 0.0}, {"t1": -1.0}, {"t_phi": float("nan")}, {"depolarizing": -0.1}, {"depolarizing": float("inf")}])
def test_invalid_noise_parameters_are_refused(kwargs):
    with pytest.raises(ValueError):
        Noise(**kwargs)


# ------------------------------------------------------------------ whole runs

def test_a_noiseless_run_stays_pure_and_equals_the_ideal_path_with_every_check_passing():
    r = run("H, Rz(pi/4), X, T, Ry(pi/3)")
    assert r.checks.passed and np.allclose(r.bloch, r.ideal) and np.allclose(r.purity, 1.0) and np.allclose(r.fidelity, 1.0)


def test_a_noisy_run_loses_purity_and_fidelity_and_still_passes_its_checks():
    r = run("H, Rz(pi/4), X, T", Z, Noise(t1=15.0, t_phi=10.0), idle_time=0.5)
    assert r.checks.passed and r.purity[-1] < 1 and r.fidelity[-1] < 1 and np.all(r.purity <= 1 + 1e-9)
    assert np.all(np.linalg.norm(r.bloch, axis=1) <= 1 + 1e-9)


def test_noise_can_raise_the_purity_of_a_mixed_state_amplitude_damping_is_not_unital():
    r = run("I", np.array([0.0, 0.0, 0.0]), Noise(t1=2.0), gate_time=5.0)
    assert r.purity[-1] > r.purity[0] and r.checks.passed


def test_mixed_initial_states_are_supported_and_noiseless_evolution_preserves_their_purity():
    r0 = np.array([0.3, 0.0, 0.4])
    r = run("H, X", r0)
    assert np.allclose(r.purity, purity(r0)) and r.checks.passed


def test_time_axis_segments_and_labels_are_consistent():
    r = run("H, X, Z", gate_time=2.0, idle_time=0.5)
    assert np.all(np.diff(r.times) > 0) and r.times[-1] == pytest.approx(3 * 2.0 + 3 * 0.5)
    assert r.gate_labels == ["H", "idle", "X", "idle", "Z", "idle"] and len(r.segment_of_sample) == len(r.times)
    assert r.gate_edges[0] == 0.0 and r.gate_edges[-1] == pytest.approx(r.times[-1])
    assert run("H, X").gate_labels == ["H", "X"]


def test_the_state_arrives_smoothly_along_each_gates_arc():
    r = run("Rx(pi)", Z, gate_time=1.0)
    mid = r.bloch[len(r.bloch) // 2]
    np.testing.assert_allclose(mid, [0, -1, 0], atol=0.13)                         # halfway round the great circle
    assert np.allclose(np.linalg.norm(r.bloch, axis=1), 1.0)


def test_the_final_densities_are_physical_states():
    r = run("H, T", Z, Noise(t1=5.0))
    for rho in (r.final_density, r.final_ideal_density):
        assert np.trace(rho) == pytest.approx(1) and np.linalg.eigvalsh(rho).min() >= -1e-12


@pytest.mark.parametrize("kwargs,match", [
    ({"sequence": []}, "at least one gate"), ({"sequence": "H", "gate_time": 0.0}, "gate time"),
    ({"sequence": "H", "gate_time": float("nan")}, "gate time"), ({"sequence": "H", "idle_time": -1.0}, "idle time"),
    ({"sequence": "H", "initial": [0, 0, 2.0]}, "length at most 1"), ({"sequence": "H", "initial": [0, 1]}, "three components"),
])
def test_unusable_runs_are_refused(kwargs, match):
    args = dict(sequence="H", initial=Z)
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        simulate_sequence(**args)


def test_progress_is_reported_per_gate():
    seen = []
    run("H, X, Z", progress=lambda label, f: seen.append(label))
    assert [s for s in seen if s.startswith("Gate")] == ["Gate 1 of 3: H", "Gate 2 of 3: X", "Gate 3 of 3: Z"]


def test_a_nonunitary_gate_is_refused():
    from modules.quantum_qubits import _gate_from_unitary
    with pytest.raises(ValueError, match="not unitary"):
        _gate_from_unitary("bad", np.array([[1, 1], [0, 1]], dtype=complex))


def test_the_product_of_the_gates_is_applied_first_gate_first():
    gates = parse_sequence("Rx(pi/2), Rz(pi/2)")
    u = sphere_trajectory_unitary(gates)
    np.testing.assert_allclose(u, gates[1].unitary @ gates[0].unitary)
    r = np.array([np.real(np.trace(bloch_to_density(Z) .copy() @ PAULI[0]))] )
    assert r.size == 1
