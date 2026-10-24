"""The Statistical mechanics page, driven through the real app: landing state, every tool, error handling."""
from modules.command_palette import MODE_LABELS, search
from tests.test_ui_smoke import _app, _exceptions, isolated_data  # noqa: F401

MODE = "🔥 Statistical mechanics"


def page():
    at = _app()
    at.session_state["app_mode"] = MODE
    return at.run()


def press(at, key):
    next(b for b in at.button if b.key == key).click()
    return at.run()


def widget(at, kind, key):
    return next(w for w in getattr(at, kind) if w.key == key)


def csv(at):
    return " ".join(df.value.to_csv() for df in at.dataframe)


def successes(at):
    return [e.value for e in at.success]


def test_registered_in_sidebar_and_palette():
    assert MODE in MODE_LABELS
    assert search("ising")[0] == MODE and search("fokker-planck")[0] == MODE and search("partition function")[0] == MODE
    assert search("random matrix")[0] == MODE


def test_landing_renders_every_tab():
    at = page()
    assert not _exceptions(at), _exceptions(at)
    assert len(at.tabs) >= 6
    assert any("checks passed" in s for s in successes(at))


def test_partition_function_systems_and_input_errors():
    at = page()
    for system in ["Harmonic oscillator", "Rigid rotor", "Explicit levels", "E_n = f(n) (typed)", "Ideal gas"]:
        widget(at, "selectbox", "smp_sys").set_value(system).run()
        assert not _exceptions(at), (system, _exceptions(at))
    widget(at, "selectbox", "smp_sys").set_value("E_n = f(n) (typed)").run()
    widget(at, "text_input", "smp_en").set_value("log(n+2)").run()
    assert not _exceptions(at)
    assert any("does not converge" in e.value for e in at.error)
    widget(at, "text_input", "smp_en").set_value("n +").run()
    assert not _exceptions(at) and at.error


def test_live_lattice_runs():
    at = press(page(), "sml_run")
    assert not _exceptions(at), _exceptions(at)
    widget(at, "number_input", "sml_L").set_value(4).run()
    assert not _exceptions(at)


def test_temperature_scan_finds_a_critical_point():
    at = page()
    widget(at, "number_input", "sms_np").set_value(7).run()
    widget(at, "number_input", "sms_me").set_value(1000).run()
    widget(at, "number_input", "sms_th").set_value(200).run()
    at = press(at, "sms_run")
    assert not _exceptions(at), _exceptions(at)
    crossed = any("Binder-cumulant crossing" in s for s in successes(at))
    not_crossed = any("do not cross" in i.value for i in at.info)
    assert crossed or not_crossed                                              # both are legitimate outcomes of a short scan; silence is not
    if crossed:
        assert "T_cross" in csv(at)


def test_exact_comparison_for_both_models():
    at = page()
    widget(at, "number_input", "sme_sw").set_value(5000).run()
    at = press(at, "sme_run")
    assert not _exceptions(at), _exceptions(at)
    assert "exact" in csv(at) and "Monte Carlo" in csv(at)
    widget(at, "radio", "sme_which").set_value("Potts 4×4").run()
    at = press(at, "sme_run")
    assert not _exceptions(at) and "within 4σ" in csv(at)


def test_mean_field_tab_reports_the_failures():
    at = page()
    assert not _exceptions(at)
    assert "Bethe" in csv(at) and "no transition" in csv(at) and "logarithmic divergence" in csv(at)
    widget(at, "number_input", "smm_z").set_value(6).run()
    assert not _exceptions(at)
    at = press(at, "smm_run")
    assert not _exceptions(at), _exceptions(at)


def test_gas_tool_and_the_bec_threshold():
    at = page()
    assert "condensate fraction" in csv(at)
    widget(at, "number_input", "smd_n").set_value(3.0).run()                      # dense: Bose gas condenses
    assert not _exceptions(at)
    table = csv(at)
    assert "BE" in table and "FD" in table
    at = press(at, "smd_run")
    assert not _exceptions(at), _exceptions(at)


def test_stochastic_tools():
    at = press(page(), "sst_run")
    assert not _exceptions(at), _exceptions(at)
    at = press(at, "ssf_run")
    assert not _exceptions(at), _exceptions(at)
    assert any("checks passed" in s for s in successes(at))
    at = press(at, "ssp_run")
    assert not _exceptions(at), _exceptions(at)


def test_custom_potential_error_is_shown_not_raised():
    at = page()
    widget(at, "selectbox", "sst_pot").set_value("custom U(x)").run()
    widget(at, "text_input", "sst_ut").set_value("__import__('os')").run()
    assert not _exceptions(at)
    assert at.error


def test_random_matrix_tab():
    at = page()
    assert not _exceptions(at)
    assert "reference law" in csv(at)
    widget(at, "selectbox", "smr_kind").set_value("GUE").run()
    assert not _exceptions(at)
    assert "GUE" in csv(at)


def test_bad_seed_text_falls_back_with_a_warning():
    at = page()
    widget(at, "text_input", "smr_seed").set_value("abc").run()
    assert not _exceptions(at)
    assert any("non-negative integer" in w.value for w in at.warning)
