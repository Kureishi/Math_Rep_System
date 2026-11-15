"""The circuits and control engine: transients against omega_0 and zeta, Bode plots against the resonance formulas, and loops against Routh-Hurwitz."""
import math

import numpy as np
import pytest
from scipy import signal

from modules import circuits as ck

R, L, C = 10.0, 10e-3, 10e-6          # zeta = 0.158, omega_0 = 3162 rad/s


def failed(result):
    return [(c.label, c.detail) for c in result.checks.checks if not c.passed]


def test_the_circuit_constants():
    c = ck.RLC(R, L, C)
    assert c.omega0 == pytest.approx(3162.2777, rel=1e-6) and c.zeta == pytest.approx(0.158114, rel=1e-5) and c.Q == pytest.approx(1 / (2 * c.zeta))
    assert ck.RLC(2 * math.sqrt(L / C), L, C).regime == "critically damped" and ck.RLC(1000, L, C).regime == "overdamped"
    assert ck.RLC(R, L, C, "parallel").zeta == pytest.approx(0.5 / R * math.sqrt(L / C))


@pytest.mark.parametrize("topology", ["series", "parallel"])
@pytest.mark.parametrize("r", [0.5, 2.0, 10.0, 31.6, 100.0, 500.0])
def test_every_regime_passes_its_transient_checks(topology, r):
    tr = ck.simulate_transient(ck.RLC(r, L, C, topology))
    assert tr.checks.passed, failed(tr)


def test_the_damping_ratio_and_frequency_are_measured_from_the_ringing():
    tr = ck.simulate_transient(ck.RLC(R, L, C))
    assert tr.measured["zeta"] == pytest.approx(0.158114, rel=0.01)
    assert tr.measured["omega_d"] == pytest.approx(3162.28 * math.sqrt(1 - 0.158114 ** 2), rel=0.005)
    assert tr.measured["overshoot"] == pytest.approx(math.exp(-math.pi * 0.158114 / math.sqrt(1 - 0.158114 ** 2)), rel=0.005)


def test_the_energy_balance_holds_through_many_cycles():
    tr = ck.simulate_transient(ck.RLC(1.0, L, C))
    imbalance = np.max(np.abs(tr.energy["work"] - tr.energy["stored"] - tr.energy["dissipated"])) / tr.energy["work"].max()
    assert imbalance < 1e-7


def test_critical_damping_is_the_fastest_non_ringing_response():
    zc = 2 * math.sqrt(L / C)
    crit, over = ck.simulate_transient(ck.RLC(zc, L, C)), ck.simulate_transient(ck.RLC(3 * zc, L, C))
    assert crit.signals["v_C"].max() <= 1.0 + 1e-9 and over.signals["v_C"].max() <= 1.0 + 1e-9

    def t99(tr):
        return tr.t[np.nonzero(tr.signals["v_C"] >= 0.99)[0][0]]
    assert t99(crit) < t99(over)


def test_overdamped_decay_follows_the_slow_pole():
    tr = ck.simulate_transient(ck.RLC(1000.0, L, C))
    assert tr.measured["decay_rate"] == pytest.approx(tr.theory["decay_rate"], rel=0.02)


def test_the_sources_voltage_is_shared_among_the_elements():
    tr = ck.simulate_transient(ck.RLC(R, L, C), level=5.0)
    total = tr.signals["v_C"] + tr.signals["v_R"] + tr.signals["v_L"]
    assert np.allclose(total, 5.0)


def test_the_symbolic_transfer_functions_are_the_expected_ones():
    tfs = ck.circuit_transfer_functions(ck.RLC(R, L, C))
    w0 = ck.RLC(R, L, C).omega0
    num, den = tfs["v_C"][1:]
    assert np.allclose(num, [w0 ** 2]) and np.allclose(den, [1, R / L, w0 ** 2])
    assert np.allclose(tfs["v_R"][1], [R / L, 0]) and np.allclose(tfs["v_L"][1], [1, 0, 0])
    assert "s" in ck.symbolic_transfer_latex("series")["v_C"]


@pytest.mark.parametrize("topology", ["series", "parallel"])
@pytest.mark.parametrize("r", [2.0, 10.0, 31.6, 200.0])
def test_bode_checks_pass(topology, r):
    b = ck.circuit_bode(ck.RLC(r, L, C, topology))
    assert b.checks.passed, failed(b)


def test_the_bandpass_resonates_at_omega0_with_bandwidth_omega0_over_Q():
    c = ck.RLC(R, L, C)
    b = ck.circuit_bode(c)
    assert b.measured["v_R"]["peak_omega"] == pytest.approx(c.omega0, rel=2e-3)
    assert b.measured["v_R"]["bandwidth"] == pytest.approx(R / L, rel=0.01)
    assert b.measured["v_C"]["peak_omega"] == pytest.approx(c.omega0 * math.sqrt(1 - 2 * c.zeta ** 2), rel=2e-3)


def test_a_heavily_damped_capacitor_voltage_has_no_resonance_peak():
    b = ck.circuit_bode(ck.RLC(200.0, L, C))
    assert b.theory["v_C"]["peak_omega"] is None


def test_the_component_values_are_validated():
    with pytest.raises(ck.CircuitError, match="R must be"):
        ck.simulate_transient(ck.RLC(-1, L, C))
    with pytest.raises(ck.CircuitError, match="topology"):
        ck.RLC(R, L, C, "ladder").validate()
    with pytest.raises(ck.CircuitError, match="non-zero"):
        ck.simulate_transient(ck.RLC(R, L, C), level=0)
    with pytest.raises(ck.CircuitError, match="duration"):
        ck.simulate_transient(ck.RLC(R, L, C), t_end=-1)


# ------------------------------------------------------------------------------------------ polynomials, Routh, margins

def test_polynomial_text_is_parsed_and_validated():
    assert np.allclose(ck.parse_poly("1, 3 3;1"), [1, 3, 3, 1]) and np.allclose(ck.parse_poly("0 0 2 1"), [2, 1])
    for bad, msg in [("a b", "numbers"), ("", "between 1"), ("0 0", "cannot all be zero"), ("1 " * 12, "between 1")]:
        with pytest.raises(ck.CircuitError, match=msg):
            ck.parse_poly(bad)


def test_routh_hurwitz_agrees_with_the_roots_on_random_polynomials():
    rng = np.random.default_rng(0)
    for _ in range(1500):
        c = rng.normal(size=rng.integers(2, 9))
        r = np.roots(c)
        if np.any(np.abs(r.real) < 1e-6):
            continue
        assert ck.routh_rhp_count(c)[0] == int(np.sum(r.real > 0))


def test_routh_finds_roots_on_the_imaginary_axis():
    rhp, imag, _ = ck.routh_rhp_count(np.array([1.0, 0.0, 1.0]))     # s^2 + 1
    assert rhp == 0 and imag == 2
    assert ck.routh_rhp_count(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))[0] == 2


def test_margins_of_a_known_loop():
    loop = ck.TF(np.array([8.0]), np.array([1.0, 3.0, 3.0, 1.0]))         # 8 / (s+1)^3: gain margin exactly 0 dB at sqrt(3)
    m = ck.margins(loop)
    assert m["w_pc"] == pytest.approx(math.sqrt(3), rel=1e-4) and m["gm_db"] == pytest.approx(0.0, abs=1e-3)
    assert m["gc_error"] < 1e-9


def test_the_ultimate_gain_of_the_classic_cubic_plant_is_eight():
    plant = ck.make_plant("Custom transfer function", num="1", den="1 3 3 1")
    ku, wu, chk = ck.ultimate_gain(plant)
    assert ku == pytest.approx(8.0, rel=1e-3) and wu == pytest.approx(math.sqrt(3), rel=1e-4) and chk.passed
    assert ck.ultimate_gain(ck.make_plant("First order")) is None


# ------------------------------------------------------------------------------------------ loops and tuning

def test_the_closed_loop_is_checked_three_ways():
    plant = ck.make_plant("First order with delay (FOPDT)", K=2.0, tau=5.0, theta=1.0)
    la = ck.analyse_loop(plant, 1.0, 0.2, 0.0, 0.0)
    assert la.stable and la.checks.passed, failed(la)
    labels = [c.label for c in la.checks.checks]
    assert any("Routh" in x for x in labels) and any("explicit feedback" in x for x in labels) and any("final value" in x for x in labels)
    assert la.metrics.steady_state_error == pytest.approx(0.0, abs=2e-3)


def test_proportional_control_leaves_a_steady_state_error_that_integral_action_removes():
    plant = ck.make_plant("Second order", K=1.0, wn=3.0, zeta=0.4)
    p_only = ck.analyse_loop(plant, 2.0, 0.0, 0.0, 0.0)
    assert p_only.metrics.final == pytest.approx(2 / 3, rel=1e-6) and any("steady-state error" in n for n in p_only.notes)
    assert ck.analyse_loop(plant, 2.0, 1.5, 0.0, 0.0).metrics.steady_state_error == pytest.approx(0.0, abs=2e-3)


def test_an_unstable_loop_is_reported_and_the_cross_checks_still_agree():
    plant = ck.make_plant("First order with delay (FOPDT)", K=2.0, tau=5.0, theta=1.0)
    la = ck.analyse_loop(plant, 10.0, 0.0, 0.0, 0.0)
    assert not la.stable and la.metrics is None and la.routh_rhp == 2 and any("UNSTABLE" in n for n in la.notes)
    assert la.checks.passed, failed(la)


def test_the_closed_loop_step_matches_scipy_for_a_known_second_order_loop():
    plant = ck.make_plant("Second order", K=1.0, wn=2.0, zeta=0.5)
    la = ck.analyse_loop(plant, 3.0, 0.0, 0.0, 0.0)
    ref = signal.step(signal.TransferFunction([12.0], [1, 2, 16]), T=la.t)[1]
    assert np.allclose(la.y, ref, atol=1e-6)


def test_step_metrics_of_a_second_order_response_match_the_formulas():
    wn, z = 4.0, 0.3
    t = np.linspace(0, 6, 6001)
    _, y = signal.step(signal.TransferFunction([wn ** 2], [1, 2 * z * wn, wn ** 2]), T=t)
    m = ck.step_metrics(t, y, 1.0)
    assert m.overshoot_pct == pytest.approx(100 * math.exp(-math.pi * z / math.sqrt(1 - z * z)), rel=0.01)
    assert m.peak_time == pytest.approx(math.pi / (wn * math.sqrt(1 - z * z)), rel=0.01)


def test_the_padé_delay_matches_the_true_delay_at_low_frequency():
    d = ck.pade_delay(1.0, 3)
    w = 0.5
    assert d(1j * w) == pytest.approx(np.exp(-1j * w), abs=1e-5) and abs(d(1j * w)) == pytest.approx(1.0, abs=1e-6)


def test_the_two_point_method_recovers_a_first_order_plus_dead_time_plant():
    K, tau, theta = ck.fopdt_from_step(ck.make_plant("First order with delay (FOPDT)", K=2.0, tau=5.0, theta=1.0))
    assert K == pytest.approx(2.0) and tau == pytest.approx(5.0, rel=0.02) and theta == pytest.approx(1.0, rel=0.1)
    assert ck.fopdt_from_step(ck.make_plant("Integrator with lag")) is None


def test_ziegler_nichols_tables_and_simc_give_stable_loops():
    plant = ck.make_plant("First order with delay (FOPDT)", K=2.0, tau=5.0, theta=1.0)
    K, tau, theta = ck.fopdt_from_step(plant)
    for tuning in (ck.zn_open_loop(K, tau, theta, "PID"), ck.zn_open_loop(K, tau, theta, "PI"), ck.simc_pi(K, tau, theta)):
        la = ck.analyse_loop(plant, tuning.kp, tuning.ki, tuning.kd, tuning.tf)
        assert la.stable and la.checks.passed, (tuning.method, failed(la))
    assert ck.analyse_loop(plant, *(lambda t: (t.kp, t.ki, 0, 0))(ck.simc_pi(K, tau, theta))).metrics.overshoot_pct < 15
    ku, wu, _ = ck.ultimate_gain(ck.make_plant("Custom transfer function", num="1", den="1 3 3 1"))
    t = ck.zn_closed_loop(ku, wu, "PID")
    assert t.kp == pytest.approx(0.6 * ku) and t.ki == pytest.approx(t.kp / (math.pi / wu)) and t.kd == pytest.approx(t.kp * (2 * math.pi / wu) / 8)
    with pytest.raises(ck.CircuitError, match="dead time"):
        ck.zn_open_loop(1.0, 1.0, 0.0)


def test_the_optimiser_improves_the_itae_of_its_starting_point():
    plant = ck.make_plant("First order with delay (FOPDT)", K=2.0, tau=5.0, theta=1.0)
    start = ck.zn_open_loop(*ck.fopdt_from_step(plant))
    best = ck.optimise_itae(plant, (start.kp, start.ki, max(start.kd, 0.1)), start.tf, max_evals=120)
    a, b = ck.analyse_loop(plant, start.kp, start.ki, start.kd, start.tf, t_end=40), ck.analyse_loop(plant, best.kp, best.ki, best.kd, best.tf, t_end=40)
    assert b.stable and b.metrics.itae < a.metrics.itae


def test_the_root_locus_branches_start_at_the_plant_poles():
    plant = ck.make_plant("Second order", K=1.0, wn=2.0, zeta=0.5)
    ks, roots = ck.root_locus(plant)
    assert np.allclose(np.sort_complex(roots[0]), np.sort_complex(plant.tf.poles), atol=0.02)


@pytest.mark.parametrize("call, message", [
    (lambda: ck.make_plant("Warp drive"), "Choose one of"), (lambda: ck.make_plant("First order", tau=-1), "positive"),
    (lambda: ck.make_plant("Second order", zeta=-1), "damping"), (lambda: ck.make_plant("Custom transfer function", num="1 2 3", den="1 1"), "proper"),
    (lambda: ck.pid_tf(-1, 0, 0, 0.1), "must not be negative"), (lambda: ck.pid_tf(1, 0, 1, 0), "positive"),
    (lambda: ck.analyse_loop(ck.make_plant("Custom transfer function", num="1 0 0", den="1 1"), 1, 0, 0, 0), "proper|improper"),
])
def test_control_input_is_validated(call, message):
    with pytest.raises(ck.CircuitError, match=message):
        call()
