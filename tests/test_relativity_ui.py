"""The General relativity page, driven through the real app: landing state, every tool, errors."""
import pytest

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


def test_registered_in_sidebar_and_palette():
    assert MODE in MODE_LABELS
    assert search("black hole")[0] == MODE and search("penrose")[0] == MODE


def test_geodesic_keyword_goes_to_gr_not_tensor_calculus():
    assert search("geodesic")[0] == MODE


def test_landing_renders_every_tab_without_exceptions():
    at = page()
    assert not _exceptions(at), _exceptions(at)
    assert len(at.tabs) >= 6


def test_curvature_runs_checks_and_singularity_analysis():
    at = press(page(), "grc_run")
    assert not _exceptions(at), _exceptions(at)
    assert any("checks passed" in s for s in successes(at))
    at = press(at, "grc_sing")
    assert not _exceptions(at)
    assert any("curvature singularity" in df.value.to_csv() for df in at.dataframe)


def test_stale_curvature_result_is_not_shown_after_changing_the_metric():
    at = press(page(), "grc_run")
    widget(at, "selectbox", "grc_metric").set_value("kerr").run()
    assert not _exceptions(at)
    assert any("Compute curvature" in i.value for i in at.info)


def test_custom_metric_errors_are_shown_not_raised():
    at = page()
    widget(at, "selectbox", "grc_metric").set_value("✏️ Custom metric…").run()
    widget(at, "text_input", "grc_g_4_0_1").set_value("x").run()          # breaks symmetry
    assert not _exceptions(at)
    assert any("symmetric" in e.value for e in at.error)


def test_precession_matches_exact_value_in_the_ui():
    at = press(page(), "grp_run")
    assert not _exceptions(at), _exceptions(at)
    assert any("measured periapsis advance matches" in df.value.to_csv() for df in at.dataframe)
    assert not any("❌" in df.value.to_csv() for df in at.dataframe if "Check" in df.value.columns)


def test_precession_rejects_a_plunging_orbit_with_an_explanation():
    at = page()
    widget(at, "number_input", "grp_rp").set_value(4.0).run()
    widget(at, "number_input", "grp_ra").set_value(12.0).run()
    at = press(at, "grp_run")
    assert not _exceptions(at)
    assert any("separatrix" in e.value for e in at.error)


def test_light_bending_table_and_capture():
    at = press(page(), "grb_run")
    assert not _exceptions(at), _exceptions(at)
    widget(at, "text_input", "grb_b").set_value("4, 20").run()
    at = press(at, "grb_run")
    assert not _exceptions(at)
    assert any("captured" in df.value.to_csv() for df in at.dataframe)


def test_general_geodesic_and_forbidden_initial_condition():
    at = press(page(), "grg_run")
    assert not _exceptions(at), _exceptions(at)
    assert any("checks passed" in s for s in successes(at))
    widget(at, "number_input", "grg_r0").set_value(5.0).run()
    widget(at, "number_input", "grg_L").set_value(6.0).run()
    at = press(at, "grg_run")
    assert any("forbidden" in e.value for e in at.error)


def test_transform_preset_runs_and_reports_invariants():
    at = press(page(), "grt_run")
    assert not _exceptions(at), _exceptions(at)
    assert any("checks passed" in s for s in successes(at))


def test_potential_tab_updates_live_with_the_mass():
    at = page()
    assert any("Textbook values for M = 1" in c.value for c in at.caption)
    widget(at, "number_input", "grv_p_schwarzschild_M").set_value(2.0).run()
    assert not _exceptions(at)
    assert any("Textbook values for M = 2" in c.value and "ISCO 12" in c.value for c in at.caption)


def test_bad_angular_momentum_text_is_reported():
    at = page()
    widget(at, "text_input", "grv_L").set_value("3, abc").run()
    assert not _exceptions(at)
    assert any("numbers" in e.value for e in at.error)


def test_wave_tab_flags_an_off_shell_wave():
    at = page()
    widget(at, "checkbox", "grw_off").set_value(True).run()
    assert not _exceptions(at)
    assert any("off-shell" in df.value.to_csv() for df in at.dataframe)


def test_wave_direction_must_be_three_numbers():
    at = page()
    widget(at, "text_input", "grw_n").set_value("1, 2").run()
    assert not _exceptions(at)
    assert any("three numbers" in e.value for e in at.error)
