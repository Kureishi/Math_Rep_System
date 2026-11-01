"""The General relativity page, driven through the real app. Each test is one session that exercises several tools, because starting the app and
re-running every tab dominates the cost of a UI test."""
from modules.command_palette import MODE_LABELS, search
from tests.test_ui_smoke import _app, _exceptions, isolated_data  # noqa: F401

MODE = "🌌 General relativity"


def page():
    at = _app()
    at.session_state["app_mode"] = MODE
    return at.run()


def press(at, key):
    next(b for b in at.button if b.key == key).click()
    return at.run()


def widget(at, kind, key):
    return next(w for w in getattr(at, kind) if w.key == key)


def successes(at):
    return [e.value for e in at.success]


def csv(at):
    return " ".join(df.value.to_csv() for df in at.dataframe)


def test_registered_and_landing_renders_every_tab():
    assert MODE in MODE_LABELS
    assert search("black hole")[0] == MODE and search("penrose")[0] == MODE and search("geodesic")[0] == MODE   # 'geodesic' moved here from Tensor calculus
    at = page()
    assert not _exceptions(at), _exceptions(at)
    assert len(at.tabs) >= 6
    assert any("Textbook values for M = 1" in c.value for c in at.caption)                                       # potentials tab is live on landing
    assert "Bethe" not in csv(at)


def test_curvature_singularities_stale_results_and_custom_metric_errors():
    at = press(page(), "grc_run")
    assert not _exceptions(at), _exceptions(at)
    assert any("checks passed" in s for s in successes(at))
    at = press(at, "grc_sing")
    assert not _exceptions(at) and any("curvature singularity" in df.value.to_csv() for df in at.dataframe)
    widget(at, "selectbox", "grc_metric").set_value("kerr").run()                                                # a result for another metric must not be shown
    assert not _exceptions(at) and any("Compute curvature" in i.value for i in at.info)
    widget(at, "selectbox", "grc_metric").set_value("✏️ Custom metric…").run()
    widget(at, "text_input", "grc_g_4_0_1").set_value("x").run()                                                 # breaks symmetry
    assert not _exceptions(at) and any("symmetric" in e.value for e in at.error)


def test_geodesic_tools_precession_bending_general_and_their_errors():
    at = press(page(), "grp_run")
    assert not _exceptions(at), _exceptions(at)
    assert "measured periapsis advance matches" in csv(at) and "❌" not in csv(at)
    widget(at, "number_input", "grp_rp").set_value(4.0).run()                                                    # inside the separatrix: a plunge, explained
    widget(at, "number_input", "grp_ra").set_value(12.0).run()
    at = press(at, "grp_run")
    assert not _exceptions(at) and any("separatrix" in e.value for e in at.error)
    at = press(at, "grb_run")
    assert not _exceptions(at), _exceptions(at)
    widget(at, "text_input", "grb_b").set_value("4, 20").run()
    at = press(at, "grb_run")
    assert not _exceptions(at) and "captured" in csv(at)
    at = press(at, "grg_run")
    assert not _exceptions(at), _exceptions(at)
    assert any("checks passed" in s for s in successes(at))
    widget(at, "number_input", "grg_r0").set_value(5.0).run()
    widget(at, "number_input", "grg_L").set_value(6.0).run()
    at = press(at, "grg_run")
    assert any("forbidden" in e.value for e in at.error)


def test_transform_potentials_and_waves():
    at = press(page(), "grt_run")
    assert not _exceptions(at), _exceptions(at)
    assert any("checks passed" in s for s in successes(at))
    widget(at, "number_input", "grv_p_schwarzschild_M").set_value(2.0).run()
    assert not _exceptions(at)
    assert any("Textbook values for M = 2" in c.value and "ISCO 12" in c.value for c in at.caption)
    widget(at, "text_input", "grv_L").set_value("3, abc").run()
    assert not _exceptions(at) and any("numbers" in e.value for e in at.error)
    widget(at, "checkbox", "grw_off").set_value(True).run()                                                      # an off-shell wave is flagged
    assert not _exceptions(at) and "off-shell" in csv(at)
    widget(at, "text_input", "grw_n").set_value("1, 2").run()
    assert not _exceptions(at) and any("three numbers" in e.value for e in at.error)
