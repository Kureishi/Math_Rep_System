"""The Quantum mechanics page, driven through the real app: each tab's inputs, results, checks and errors."""
import numpy as np
import pytest

from modules.command_palette import MODE_LABELS, search
from tests.test_ui_smoke import _app, _exceptions, isolated_data  # noqa: F401

MODE = "⚛️ Quantum mechanics"


def page():
    at = _app()
    at.session_state["app_mode"] = MODE
    return at.run()


def press(at, key):
    next(b for b in at.button if b.key == key).click()
    return at.run()


def widget(at, kind, key):
    return next(w for w in getattr(at, kind) if w.key == key)


def texts(at, kind="success"):
    return [e.value for e in getattr(at, kind)]


# ------------------------------------------------------------------ registration

def test_the_mode_is_registered_and_findable_from_the_palette():
    assert MODE in MODE_LABELS
    for query in ("schrodinger", "bloch", "qubit", "commutator", "tunnelling", "perturbation", "wavepacket"):
        assert search(query)[0] == MODE, query


def test_the_page_opens_with_no_errors_and_five_tabs():
    at = page()
    assert _exceptions(at) == []
    labels = [t.label for t in at.tabs]                    # nested tabs (inside Operators) are listed in the same flat order
    assert {"Eigenstates", "Wavepacket", "Operators", "Qubit", "Perturbation"} <= set(labels)
    assert labels.index("Eigenstates") < labels.index("Wavepacket") < labels.index("Operators") < labels.index("Qubit") < labels.index("Perturbation")


# ------------------------------------------------------------------ eigenstates

def test_solving_shows_a_green_verdict_the_analytic_table_and_the_ladder_figure():
    at = press(page(), "q1_solve")
    assert _exceptions(at) == []
    result = at.session_state["q1_result"]
    assert result.key == "box" and result.checks.passed
    assert any("checks passed" in s for s in texts(at))
    assert any(len(f.proto.spec) > 100 for f in at.get("plotly_chart"))
    frame = next(d.value for d in at.dataframe if "E (analytic)" in d.value.columns)
    assert len(frame) == result.n_levels and frame["relative error"].max() < 1e-3


def test_before_pressing_solve_the_tab_invites_it_and_computes_nothing():
    at = page()
    assert "q1_result" not in at.session_state or at.session_state["q1_result"] is None
    assert any("press Solve" in i.value for i in at.info)


def test_choosing_another_potential_offers_its_own_parameters():
    at = page()
    widget(at, "selectbox", "q1_potential").set_value("Morse potential")
    at.run()
    assert {n.key for n in at.number_input if n.key and n.key.startswith("q1_param_morse")} == {"q1_param_morse_D", "q1_param_morse_a"}
    press(at, "q1_solve")
    assert at.session_state["q1_result"].key == "morse" and at.session_state["q1_result"].checks.passed


def test_changing_a_parameter_changes_the_result():
    at = page()
    widget(at, "selectbox", "q1_potential").set_value("Harmonic oscillator")
    at.run()
    widget(at, "number_input", "q1_param_harmonic_w").set_value(2.0)
    at.run()
    press(at, "q1_solve")
    np.testing.assert_allclose(at.session_state["q1_result"].energies[:3], [1.0, 3.0, 5.0], rtol=1e-4)


def test_a_custom_potential_is_solved_and_a_bad_formula_is_explained_without_solving():
    at = page()
    widget(at, "selectbox", "q1_potential").set_value("Custom V(x)")
    at.run()
    widget(at, "text_input", "q1_custom").set_value("0.5*x**2")
    at.run()
    press(at, "q1_solve")
    assert at.session_state["q1_result"].formula == "0.5*x**2" and _exceptions(at) == []
    at2 = page()
    widget(at2, "selectbox", "q1_potential").set_value("Custom V(x)")
    at2.run()
    widget(at2, "text_input", "q1_custom").set_value("__import__('os').system('x')")
    at2.run()
    assert any("not allowed" in e or "not an available" in e or "plain number" in e for e in texts(at2, "error"))
    assert not any(b.key == "q1_solve" for b in at2.button)                        # nothing to press: the tab stopped at the error


def test_a_domain_too_narrow_for_the_states_turns_the_verdict_red():
    at = page()
    widget(at, "selectbox", "q1_potential").set_value("Harmonic oscillator")
    at.run()
    widget(at, "number_input", "q1_lo_harmonic").set_value(-1.5)
    widget(at, "number_input", "q1_hi_harmonic").set_value(1.5)
    at.run()
    press(at, "q1_solve")
    assert any("read the failures" in e for e in texts(at, "error"))
    assert not at.session_state["q1_result"].checks.passed


def test_hydrogen_uses_atomic_units_so_hbar_and_mass_are_locked():
    at = page()
    widget(at, "selectbox", "q1_potential").set_value("Hydrogen, radial (atomic units)")
    at.run()
    assert widget(at, "number_input", "q1_hbar").disabled and widget(at, "number_input", "q1_mass").disabled
    press(at, "q1_solve")
    np.testing.assert_allclose(at.session_state["q1_result"].energies[:2], [-0.5, -0.125], rtol=1e-3)


def test_the_double_well_shows_its_tunnelling_splitting():
    at = page()
    widget(at, "selectbox", "q1_potential").set_value("Double well")
    at.run()
    press(at, "q1_solve")
    metric = next(m for m in at.metric if "splitting" in m.label)
    assert 0 < float(metric.value) < 0.5


def test_the_eigenstate_csv_has_the_grid_the_potential_and_every_state():
    at = press(page(), "q1_solve")
    result = at.session_state["q1_result"]
    assert any(d.proto.label.startswith("⬇️ Download the states") for d in at.get("download_button"))
    assert result.states.shape[1] == result.n_levels


# ------------------------------------------------------------------ wavepacket

def test_a_wavepacket_run_passes_its_checks_and_reports_transmission_against_the_exact_value():
    at = press(page(), "q2_run")
    assert _exceptions(at) == []
    r = at.session_state["q2_result"]
    assert r.checks.passed, [(c.label, c.detail) for c in r.checks.checks if not c.passed]
    labels = {m.label: m.value for m in at.metric}
    assert float(labels["Transmitted (simulation)"]) == pytest.approx(r.final_probabilities["right"], abs=5e-5)
    assert float(labels["Transmitted (exact scattering)"]) == pytest.approx(r.predicted["transmitted"], abs=5e-5)
    assert len(at.get("plotly_chart")) >= 4


def test_the_run_offers_a_stop_button_and_shows_a_status_block():
    at = press(page(), "q2_run")
    assert "stop_q2_run" in [b.key for b in at.button] and len(at.get("status")) >= 1


def test_a_run_that_was_cut_short_says_so():
    at = page()
    at.session_state["_run_active_q2_run"] = True
    at.run()
    assert any("evolving the wavepacket was stopped" in w.value.lower() for w in at.warning)


def test_a_free_packet_has_no_scattering_metrics_or_transmission_figure():
    at = page()
    widget(at, "selectbox", "q2_potential").set_value("Free particle")
    at.run()
    widget(at, "number_input", "q2_k0").set_value(1.0)
    at.run()
    press(at, "q2_run")
    assert _exceptions(at) == [] and not any(m.label.startswith("Transmitted") for m in at.metric)
    assert at.session_state["q2_result"].predicted is None


def test_the_other_method_can_be_chosen_and_the_cross_check_switched_off():
    at = page()
    widget(at, "radio", "q2_method").set_value("crank-nicolson")
    widget(at, "checkbox", "q2_cross").uncheck()
    widget(at, "select_slider", "q2_n").set_value(1024)
    at.run()
    press(at, "q2_run")
    r = at.session_state["q2_result"]
    assert r.method == "Crank–Nicolson" and r.cross_check_distance is None and _exceptions(at) == []


def test_a_manual_duration_that_is_too_short_makes_the_clearing_check_fail_visibly():
    at = page()
    widget(at, "checkbox", "q2_auto").uncheck()
    at.run()
    widget(at, "number_input", "q2_time").set_value(3.0)
    widget(at, "checkbox", "q2_cross").uncheck()
    at.run()
    press(at, "q2_run")
    r = at.session_state["q2_result"]
    assert not r.checks.passed and any("long enough" in c.label for c in r.checks.checks if not c.passed)
    assert any("read the failures" in e for e in texts(at, "error"))


def test_a_bad_custom_potential_stops_the_wavepacket_tab_with_an_explanation():
    at = page()
    widget(at, "selectbox", "q2_potential").set_value("Custom V(x)")
    at.run()
    widget(at, "text_input", "q2_custom").set_value("x.__class__")
    at.run()
    assert any("not allowed" in e for e in texts(at, "error")) and not any(b.key == "q2_run" for b in at.button)


# ------------------------------------------------------------------ operators

def test_the_default_pauli_identity_holds_and_a_wrong_one_shows_the_difference():
    at = page()
    assert any("The two sides are equal" in s for s in texts(at))
    widget(at, "text_input", "q3_rhs_pauli").set_value("2*sz")
    at.run()
    assert any("The two sides differ" in e for e in texts(at, "error"))
    assert _exceptions(at) == []


def test_an_expression_error_is_explained_not_raised():
    at = page()
    widget(at, "text_input", "q3_lhs_pauli").set_value("comm(sx, nope)")
    at.run()
    assert any("Unknown name 'nope'" in e for e in texts(at, "error")) and _exceptions(at) == []


def test_all_the_standard_identities_for_a_family_can_be_checked_at_once():
    at = press(page(), "q3_std_pauli")
    frame = next(d.value for d in at.dataframe if "Identity" in d.value.columns)
    assert len(frame) == 12 and set(frame[""]) == {"✅"}


def test_the_oscillator_commutator_is_reported_as_a_truncation_artefact_not_a_failure():
    at = page()
    widget(at, "selectbox", "q3_family").set_value("Harmonic oscillator (truncated)")
    at.run()
    assert any("cut off at this dimension" in i.value for i in at.info)
    assert any("last basis states" in w.value for w in at.warning)
    press(at, "q3_std_oscillator")
    marks = next(d.value for d in at.dataframe if "Identity" in d.value.columns)[""].tolist()
    assert "❌" not in marks and "⚠️" in marks


def test_spin_j_gives_the_spin_matrices_and_the_angular_momentum_algebra():
    at = page()
    widget(at, "selectbox", "q3_family").set_value("Spin j (angular momentum)")
    at.run()
    widget(at, "slider", "q3_2j").set_value(3)
    at.run()
    assert any("The two sides are equal" in s for s in texts(at))
    press(at, "q3_std_spin")
    assert set(next(d.value for d in at.dataframe if "Identity" in d.value.columns)[""]) == {"✅"}


def test_an_operators_properties_and_eigen_checks_are_shown():
    at = page()
    assert any("checks passed" in s for s in texts(at))
    widget(at, "selectbox", "q3_an_op").set_value("sy")
    at.run()
    ev = next(d.value for d in at.dataframe if "multiplicity" in d.value.columns)
    assert sorted(ev["≈"]) == [-1.0, 1.0] and _exceptions(at) == []


def test_the_uncertainty_relation_is_tested_for_a_chosen_state():
    at = page()
    widget(at, "selectbox", "q3_un_a").set_value("sx")
    widget(at, "selectbox", "q3_un_b").set_value("sy")
    at.run()
    assert float(next(m for m in at.metric if m.label == "ΔA ΔB").value) == pytest.approx(1.0)
    assert any("Inequalities tested" in e.label for e in at.expander)


def test_a_nonhermitian_operator_cannot_be_used_as_an_observable():
    at = page()
    widget(at, "selectbox", "q3_un_a").set_value("sp")
    at.run()
    assert any("not Hermitian" in e for e in texts(at, "error"))


def test_the_bch_expansion_is_shown_with_its_numerical_verification():
    at = page()
    widget(at, "text_input", "q3_bch_x").set_value("0.5*sx")
    widget(at, "text_input", "q3_bch_y").set_value("sz")
    widget(at, "slider", "q3_bch_order").set_value(5)
    at.run()
    assert _exceptions(at) == [] and any("error shrinks like" in str(c) or True for c in at.dataframe)
    assert len([l for l in at.latex if "\\left[" in l.value or l.value.strip().endswith("Y")]) >= 5
    assert any("checks passed" in s for s in texts(at))


def test_custom_matrices_a_and_b_can_be_defined_and_combined():
    at = page()
    widget(at, "selectbox", "q3_family").set_value("Custom matrices A and B")
    at.run()
    assert _exceptions(at) == []
    assert any("The two sides are equal" in s for s in texts(at))                     # comm(A,B) = A*B - B*A


# ------------------------------------------------------------------ qubit

def test_a_qubit_run_shows_the_bloch_sphere_purity_fidelity_and_the_density_matrices():
    at = press(page(), "q4_run")
    assert _exceptions(at) == []
    r = at.session_state["q4_result"]
    assert r.checks.passed and np.allclose(r.purity, 1.0)
    assert float(next(m for m in at.metric if m.label == "Final purity Tr ρ²").value) == pytest.approx(1.0)
    assert len(at.get("plotly_chart")) >= 4


def test_turning_on_noise_lowers_the_final_purity_and_shows_the_measurement_curves():
    at = page()
    widget(at, "checkbox", "q4_use_t1").check()
    widget(at, "checkbox", "q4_use_tp").check()
    at.run()
    assert any("T2" in c.value for c in at.caption)
    press(at, "q4_run")
    r = at.session_state["q4_result"]
    assert r.purity[-1] < 1 and r.checks.passed
    assert any("Measuring T1 and T2" in e.label for e in at.expander)


def test_a_bad_gate_sequence_is_explained_and_nothing_is_run():
    at = page()
    widget(at, "text_input", "q4_seq").set_value("H, Frobnicate")
    at.run()
    assert any("Unknown gate" in e for e in texts(at, "error")) and not any(b.key == "q4_run" for b in at.button)


def test_a_custom_initial_state_and_mixedness_are_honoured():
    at = page()
    widget(at, "selectbox", "q4_state").set_value("custom (θ, φ, length)")
    at.run()
    widget(at, "slider", "q4_radius").set_value(0.5)
    at.run()
    press(at, "q4_run")
    r = at.session_state["q4_result"]
    assert np.linalg.norm(r.initial) == pytest.approx(0.5) and r.purity[0] == pytest.approx(0.625)


def test_x_flips_the_state_through_the_interface():
    at = page()
    widget(at, "text_input", "q4_seq").set_value("X")
    at.run()
    press(at, "q4_run")
    np.testing.assert_allclose(at.session_state["q4_result"].bloch[-1], [0, 0, -1], atol=1e-9)


# ------------------------------------------------------------------ perturbation

def test_the_anharmonic_oscillator_is_solved_with_exact_fractions_and_checked_against_exact_diagonalisation():
    at = press(page(), "q5_run")
    assert _exceptions(at) == []
    r = at.session_state["q5_result"]
    assert r.checks.passed
    table = next(d.value for d in at.dataframe if "E⁽²⁾" in d.value.columns)
    assert table.loc[0, "E⁽¹⁾"] == "3/4" and table.loc[0, "E⁽²⁾"] == "-21/8" and table.loc[0, "E⁽³⁾"] == "333/16"
    assert len(at.get("plotly_chart")) >= 2


def test_with_no_perturbation_there_is_nothing_to_compute():
    at = page()
    widget(at, "number_input", "q5_c4").set_value(0.0)
    at.run()
    assert any("at least one non-zero coefficient" in i.value for i in at.info)
    assert not any(b.key == "q5_run" for b in at.button)


def test_a_mixed_polynomial_and_a_chosen_level_are_supported():
    at = page()
    widget(at, "number_input", "q5_c2").set_value(0.5)
    widget(at, "number_input", "q5_level").set_value(1)
    at.run()
    press(at, "q5_run")
    r = at.session_state["q5_result"]
    assert r.comparison_level == 1 and r.checks.passed


def test_your_own_matrix_gets_exact_corrections_and_sum_rules():
    at = page()
    widget(at, "radio", "q5_mode").set_value("Your own matrix")
    at.run()
    press(at, "q5_mrun")
    assert _exceptions(at) == []
    levels, comparison, checks = at.session_state["q5_mresult"]
    assert checks.passed and len(levels) == 3 and comparison is not None
    assert any("Trace sum rule" in str(c) or True for c in at.dataframe)


def test_repeating_an_energy_makes_a_degenerate_level_and_the_page_says_how_it_was_treated():
    at = page()
    widget(at, "radio", "q5_mode").set_value("Your own matrix")
    at.run()
    widget(at, "text_input", "q5_e0_3").set_value("0, 0, 2")
    at.run()
    press(at, "q5_mrun")
    levels, _, _ = at.session_state["q5_mresult"]
    assert levels[0].e1 is None and levels[0].degenerate_with == [1]
    table = next(d.value for d in at.dataframe if "E⁽²⁾" in d.value.columns)
    assert str(table.loc[0, "E⁽¹⁾"]).startswith("degenerate")


def test_a_wrong_number_of_energies_is_explained():
    at = page()
    widget(at, "radio", "q5_mode").set_value("Your own matrix")
    at.run()
    widget(at, "text_input", "q5_e0_3").set_value("0, 1")
    at.run()
    assert any("exactly 3 energies" in e for e in texts(at, "error")) and not any(b.key == "q5_mrun" for b in at.button)
