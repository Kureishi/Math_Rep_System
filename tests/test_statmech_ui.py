"""The Statistical mechanics page, driven through the real app. Each test is one session exercising several tools (see test_relativity_ui.py)."""
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


def test_registered_landing_and_the_always_live_tabs():
    assert MODE in MODE_LABELS
    for q, want in (("ising", MODE), ("fokker-planck", MODE), ("partition function", MODE), ("random matrix", MODE)):
        assert search(q)[0] == want, q
    at = page()
    assert not _exceptions(at), _exceptions(at)
    assert len(at.tabs) >= 6 and any("checks passed" in s for s in successes(at))
    table = csv(at)
    assert "Bethe" in table and "logarithmic divergence" in table                                                # mean-field tab
    assert "condensate fraction" in table and "reference law" in table                                           # gas tool and random matrices
    widget(at, "selectbox", "smr_kind").set_value("GUE").run()
    assert not _exceptions(at) and "GUE" in csv(at)
    widget(at, "text_input", "smr_seed").set_value("abc").run()
    assert not _exceptions(at) and any("non-negative integer" in w.value for w in at.warning)


def test_partition_function_systems_and_their_errors():
    at = page()
    for system in ["Harmonic oscillator", "Rigid rotor", "Explicit levels", "Ideal gas"]:
        widget(at, "selectbox", "smp_sys").set_value(system).run()
        assert not _exceptions(at), (system, _exceptions(at))
    widget(at, "selectbox", "smp_sys").set_value("E_n = f(n) (typed)").run()
    widget(at, "text_input", "smp_en").set_value("log(n+2)").run()
    assert not _exceptions(at) and any("does not converge" in e.value for e in at.error)
    widget(at, "text_input", "smp_en").set_value("n +").run()
    assert not _exceptions(at) and at.error


def test_lattice_tools_live_scan_and_exact_comparison():
    at = press(page(), "sml_run")
    assert not _exceptions(at), _exceptions(at)
    widget(at, "number_input", "sms_np").set_value(7).run()
    widget(at, "number_input", "sms_me").set_value(1000).run()
    widget(at, "number_input", "sms_th").set_value(200).run()
    at = press(at, "sms_run")
    assert not _exceptions(at), _exceptions(at)
    crossed = any("Binder-cumulant crossing" in s for s in successes(at))
    assert crossed or any("do not cross" in i.value for i in at.info)                                            # both are legitimate outcomes of a short scan
    widget(at, "number_input", "sme_sw").set_value(5000).run()
    at = press(at, "sme_run")
    assert not _exceptions(at), _exceptions(at)
    assert "exact" in csv(at) and "Monte Carlo" in csv(at)
    widget(at, "radio", "sme_which").set_value("Potts 4×4").run()
    at = press(at, "sme_run")
    assert not _exceptions(at) and "within 4σ" in csv(at)


def test_overlay_gas_sampling_and_stochastic_tools():
    at = press(page(), "smm_run")
    assert not _exceptions(at), _exceptions(at)
    widget(at, "number_input", "smd_n").set_value(3.0).run()                                                     # dense: the Bose gas condenses
    at = press(at, "smd_run")
    assert not _exceptions(at), _exceptions(at)
    at = press(at, "sst_run")
    assert not _exceptions(at), _exceptions(at)
    at = press(at, "ssf_run")
    assert not _exceptions(at) and any("checks passed" in s for s in successes(at))
    at = press(at, "ssp_run")
    assert not _exceptions(at), _exceptions(at)
    widget(at, "selectbox", "sst_pot").set_value("custom U(x)").run()
    widget(at, "text_input", "sst_ut").set_value("__import__('os')").run()                                       # refused, not executed
    assert not _exceptions(at) and at.error
