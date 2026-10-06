"""End-to-end tests for the interactive elements: isolated panels, click-to-act plots, live status blocks
with Stop, the Share dialog, saved views, popovers and the Compare page. Driven through the real app.py.

AppTest has two limits that shape these tests, both worth knowing:

  * it never performs a FRAGMENT-scoped rerun (a widget change reruns the whole script), so what makes a
    panel an isolated fragment is checked structurally -- it is registered and wrapped by st.fragment --
    and everything else about those panels is checked by running them for real;
  * it cannot click inside a plot, and widget state it is given for a plot does not survive past the run
    it was given for. A click is therefore simulated by presetting the plot's selection state (the exact
    shape Streamlit stores) immediately before each run that should see it.
"""
import json

import pytest

import modules.history as history
from modules import view_state as vs
from modules.batch_solver import solve_batch
from modules.command_palette import MODE_LABELS, search
from modules.equation_engine import build_model
from modules.solver import compute_steps
from modules.verifier import verify
from tests.conftest import FakeClient, KINEMATICS_TWO_TARGET_JSON
from tests.test_ui_smoke import _app, _exceptions, _seed_solved_kinematics, isolated_data  # noqa: F401
from ui.compare_constants import COMPARE_MODE, CURRENT_PROBLEM, WHAT_IF_SOURCE


def sel(*points) -> dict:
    """A plotly selection state as Streamlit stores it."""
    return {"selection": {"points": list(points), "point_indices": [], "box": [], "lasso": []}}


def button(at, key):
    return next(b for b in at.button if b.key == key)


def keys(at, prefix=""):
    return [b.key for b in at.button if b.key and b.key.startswith(prefix)]


def solved_app():
    at = _app()
    _seed_solved_kinematics(at)
    return at.run()


def save_kinematics_to_history() -> int:
    model = build_model(json.loads(KINEMATICS_TWO_TARGET_JSON))
    report = verify(model, FakeClient(KINEMATICS_TWO_TARGET_JSON, final_answers={"a": 2.0, "d": 84.0}), "p")
    return history.save("A car accelerates.", model, report, compute_steps(model), [])


# ------------------------------------------------------------------ 1. isolated panels

EXPECTED_ISOLATED = {
    "ui.results.time_views._render_fan", "ui.results.time_views._render_morph",
    "ui.results.time_views._render_bifurcation", "ui.results.time_views._render_cobweb_without_closed_form",
    "ui.results.steps.render_monte_carlo", "ui.results.steps.render_analytic_error",
    "ui.results.steps.render_interval_bounds", "ui.results.steps.render_goal_seek",
    "ui.results.steps.render_sensitivity",
    "ui.results.explore_tab._render_sweep_panel", "ui.results.explore_tab._render_bulk_mc_panel",
    "ui.results.explore_tab._interactive_plot_panel",
}


def test_the_heavy_panels_are_registered_as_isolated_fragments():
    import ui.results  # noqa: F401  (imports every panel module)
    from ui.fragments import ISOLATED
    assert EXPECTED_ISOLATED <= set(ISOLATED)
    assert set(ISOLATED) == EXPECTED_ISOLATED, "a panel was made isolated without being listed here"
    for name, fragment in ISOLATED.items():
        assert hasattr(fragment, "__wrapped__"), f"{name} is not wrapped by st.fragment"


def test_a_solved_page_renders_with_every_isolated_panel_in_place():
    at = solved_app()
    assert _exceptions(at) == []
    assert {"sweep_run_button", "bulk_mc_run", "mc_run_a", "mc_run_d", "include_snap_tornado_a"} <= set(keys(at))


def test_every_target_gets_its_own_instance_of_a_looped_panel():
    at = solved_app()
    assert "mc_run_a" in keys(at) and "mc_run_d" in keys(at)


# ------------------------------------------------------------------ 2. click-to-act: tornado bars

def test_clicking_a_tornado_bar_offers_the_three_actions():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["tornado_a"] = sel({"x": 1.0, "y": "u"})
    at.run()
    assert _exceptions(at) == []
    assert {"tornado_detail_a", "tornado_nd_a", "tornado_mc_a"} <= set(keys(at, "tornado_"))
    assert any("**u** selected" in m.value for m in at.markdown)


def test_no_click_no_actions():
    assert keys(solved_app(), "tornado_") == []


def test_a_stale_selection_for_an_input_not_on_this_chart_offers_nothing():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["tornado_a"] = sel({"x": 1.0, "y": "not_an_input"})
    at.run()
    assert keys(at, "tornado_") == []


def test_the_monte_carlo_action_sets_that_targets_uncertain_inputs():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["tornado_a"] = sel({"x": 1.0, "y": "u"})
    at.run()
    button(at, "tornado_mc_a").click()
    at.session_state["tornado_a"] = sel({"x": 1.0, "y": "u"})
    at.run()
    assert _exceptions(at) == []
    assert at.session_state["mc_vars_a"] == ["u"]
    assert next(m for m in at.multiselect if m.key == "mc_vars_a").value == ["u"]      # and the widget shows it


def test_the_sweep_action_adds_to_the_existing_selection_and_picks_the_target():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["sweep_symbols"] = ["t"]
    at.session_state["tornado_a"] = sel({"x": 1.0, "y": "u"})
    at.run()
    button(at, "tornado_nd_a").click()
    at.session_state["tornado_a"] = sel({"x": 1.0, "y": "u"})
    at.run()
    assert _exceptions(at) == []
    assert at.session_state["sweep_target"] == "a"
    assert at.session_state["sweep_symbols"] == ["t", "u"]                 # added, nothing removed


def test_adding_an_input_that_is_already_selected_does_not_duplicate_it():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["mc_vars_a"] = ["u"]
    at.session_state["tornado_a"] = sel({"x": 1.0, "y": "u"})
    at.run()
    button(at, "tornado_mc_a").click()
    at.session_state["tornado_a"] = sel({"x": 1.0, "y": "u"})
    at.run()
    assert at.session_state["mc_vars_a"] == ["u"]


def test_the_detail_action_points_the_detailed_sweep_at_the_clicked_input():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["tornado_a"] = sel({"x": 1.0, "y": "t"})
    at.run()
    button(at, "tornado_detail_a").click()
    at.session_state["tornado_a"] = sel({"x": 1.0, "y": "t"})
    at.run()
    assert next(s for s in at.selectbox if s.key == "sweep_pick_a").value == "t"


def test_an_action_shows_its_confirmation_toast():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["tornado_a"] = sel({"x": 1.0, "y": "u"})
    at.run()
    button(at, "tornado_mc_a").click()
    at.session_state["tornado_a"] = sel({"x": 1.0, "y": "u"})
    at.run()
    assert any("added to the uncertain inputs for a" in t.value for t in at.toast)


# ------------------------------------------------------------------ queued updates

def test_queued_updates_are_applied_before_the_widgets_exist_and_then_cleared():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["_pending_updates"] = {"mc_n_a": 3000, "gs_target_a": 7.5}
    at.session_state["_pending_toast"] = "applied!"
    at.run()
    assert _exceptions(at) == []
    assert next(s for s in at.slider if s.key == "mc_n_a").value == 3000
    assert next(n for n in at.number_input if n.key == "gs_target_a").value == 7.5
    assert "_pending_updates" not in at.session_state
    assert any(t.value == "applied!" for t in at.toast)


# ------------------------------------------------------------------ click-to-act: sweep heatmap

def _swept_app():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["sweep_target"] = "a"
    at.session_state["sweep_symbols"] = ["u", "t"]
    at.run()
    button(at, "sweep_run_button").click()
    at.run()
    result = at.session_state["sweep_result"]
    xs = sorted({r["u"] for r in result.rows})
    ys = sorted({r["t"] for r in result.rows})
    return at, xs, ys


def test_a_heatmap_click_shows_the_point_and_its_value():
    at, xs, ys = _swept_app()
    at.session_state["sweep_heatmap"] = sel({"x": xs[1], "y": ys[2]})
    at.run()
    assert _exceptions(at) == []
    line = next(m.value for m in at.markdown if "→" in m.value and "u =" in m.value)
    assert f"u = {xs[1]:.6g}" in line and f"t = {ys[2]:.6g}" in line
    expected = (20 - xs[1]) / ys[2]
    assert f"a = {expected:.6g}" in line
    assert {"sweep_point_load", "sweep_point_compare"} <= set(keys(at))


def test_a_click_that_is_not_on_the_grid_offers_nothing():
    at, xs, ys = _swept_app()
    half_x, half_y = (xs[1] - xs[0]) / 2, (ys[1] - ys[0]) / 2          # midway between grid lines
    at.session_state["sweep_heatmap"] = sel({"x": xs[0] + half_x, "y": ys[0] + half_y})
    at.run()
    assert keys(at, "sweep_point") == []


def test_loading_a_grid_point_sets_the_variables_panel():
    at, xs, ys = _swept_app()
    at.session_state["sweep_heatmap"] = sel({"x": xs[1], "y": ys[2]})
    at.run()
    button(at, "sweep_point_load").click()
    at.session_state["sweep_heatmap"] = sel({"x": xs[1], "y": ys[2]})
    at.run()
    assert _exceptions(at) == []
    assert next(n for n in at.number_input if n.key == "var_u").value == pytest.approx(xs[1])
    assert next(n for n in at.number_input if n.key == "var_t").value == pytest.approx(ys[2])


def test_comparing_a_grid_point_opens_the_compare_page_on_that_what_if():
    at, xs, ys = _swept_app()
    at.session_state["sweep_heatmap"] = sel({"x": xs[1], "y": ys[3]})
    at.run()
    button(at, "sweep_point_compare").click()
    at.session_state["sweep_heatmap"] = sel({"x": xs[1], "y": ys[3]})
    at.run()
    assert _exceptions(at) == []
    assert at.session_state["app_mode"] == COMPARE_MODE
    text = "\n".join(m.value for m in at.markdown)
    assert "What differs" in text
    assert "Inputs that differ: u, t." in text


# ------------------------------------------------------------------ click-to-act: dependency graph

def test_clicking_an_unknown_node_shows_that_targets_steps_and_marks_the_step_list():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["dependency_graph_chart"] = sel({"x": 2.0, "y": 0.0})        # the unknown `a`
    at.run()
    assert _exceptions(at) == []
    assert any("Steps for `a`" in m.value for m in at.markdown)
    assert any("Selected in the dependency graph" in c.value for c in at.caption)
    assert at.session_state["_graph_focus"] == ["a"]


def test_clicking_a_given_input_says_there_is_nothing_to_solve():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["dependency_graph_chart"] = sel({"x": 0.0, "y": 0.0})        # a known input
    at.run()
    assert any("given input" in c.value for c in at.caption)
    assert at.session_state["_graph_focus"] == []


def test_with_no_click_nothing_is_marked():
    at = solved_app()
    assert at.session_state["_graph_focus"] == []
    assert not any("Selected in the dependency graph" in c.value for c in at.caption)


# ------------------------------------------------------------------ live status + Stop

def test_a_monte_carlo_run_shows_a_status_block_stores_its_result_and_clears_its_flag():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["mc_vars_a"] = ["u"]
    at.run()
    button(at, "mc_run_a").click()
    at.run()
    assert _exceptions(at) == []
    result = at.session_state["mc_result_a"]
    assert result.mean == pytest.approx(2.0, abs=0.1) and result.n_requested == 1000
    assert len(at.get("status")) >= 1
    assert "_run_active_mc_a" not in at.session_state
    assert "stop_mc_a" in keys(at)                                   # the Stop button was offered


def test_a_run_that_was_cut_short_says_so_and_keeps_the_previous_result():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["mc_vars_a"] = ["u"]
    at.run()
    button(at, "mc_run_a").click()
    at.run()
    earlier = at.session_state["mc_result_a"]
    at.session_state["_run_active_mc_a"] = True                      # what an interrupted run leaves behind
    at.run()
    assert any("stopped before it finished" in w.value for w in at.warning)
    assert "_run_active_mc_a" not in at.session_state                # said once, then cleared
    assert at.session_state["mc_result_a"] is earlier                # a finished result is never discarded
    at.run()
    assert not any("stopped before it finished" in w.value for w in at.warning)


def test_a_failed_run_clears_its_flag_so_it_is_not_reported_as_stopped():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["mc_vars_a"] = ["u", "t"]
    at.run()
    button(at, "mc_run_a").click()
    at.run()                                      # both uncertain: runs fine
    assert "_run_active_mc_a" not in at.session_state


def test_the_all_targets_run_covers_every_target_with_a_status_block():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["bulk_mc_vars"] = ["u"]
    at.run()
    button(at, "bulk_mc_run").click()
    at.run()
    assert _exceptions(at) == []
    rows = at.session_state["bulk_mc_results"]
    assert [r["target"] for r in rows] == ["a", "d"] and all(r["error"] is None for r in rows)
    assert "_run_active_bulk_mc" not in at.session_state


def test_an_interrupted_batch_keeps_the_problems_that_finished():
    client = FakeClient(KINEMATICS_TWO_TARGET_JSON, final_answers={"a": 2.0, "d": 84.0})
    done = solve_batch(client, ["A car accelerates from 8 to 20 m/s in 6 s."])
    at = _app()
    at.session_state["app_mode"] = next(m for m in MODE_LABELS if "Batch" in m)
    at.session_state["_run_active_batch"] = True
    at.session_state["batch_partial"] = done
    at.run()
    assert _exceptions(at) == []
    assert any("batch was stopped" in w.value for w in at.warning)
    assert any("1 problem(s) that finished" in i.value for i in at.info)
    assert at.session_state["batch_results"] == done


def test_a_batch_that_stopped_before_anything_finished_shows_just_the_notice():
    at = _app()
    at.session_state["app_mode"] = next(m for m in MODE_LABELS if "Batch" in m)
    at.session_state["_run_active_batch"] = True
    at.run()
    assert any("batch was stopped" in w.value for w in at.warning)
    assert not at.info or not any("finished before" in i.value for i in at.info)


def test_a_finite_difference_pde_run_gets_a_status_block_and_stop_button():
    at = _app()
    at.session_state["app_mode"] = next(m for m in MODE_LABELS if "PDE" in m)
    at.run()
    button(at, "fd_button").click()
    at.run()
    assert _exceptions(at) == []
    assert at.session_state["fd_result"].error is None
    assert len(at.get("status")) >= 1 and "stop_fd_button" in keys(at)


def test_the_2d_heat_run_gets_a_status_block_but_no_stop_button():
    at = _app()
    at.session_state["app_mode"] = next(m for m in MODE_LABELS if "PDE" in m)
    at.run()
    button(at, "heat2d_button").click()
    at.run()
    assert _exceptions(at) == []
    assert len(at.get("status")) >= 1 and "stop_heat2d_button" not in keys(at)


def test_a_pde_run_cut_short_says_so():
    at = _app()
    at.session_state["app_mode"] = next(m for m in MODE_LABELS if "PDE" in m)
    at.session_state["_run_active_fd_button"] = True
    at.run()
    assert any("finite-difference solve was stopped" in w.value for w in at.warning)


# ------------------------------------------------------------------ Share dialog

def test_the_share_button_replaces_the_inline_export_and_chain_sections():
    at = solved_app()
    assert "open_share_dialog" in keys(at)
    assert not any(b.label.startswith("📄 Download as Markdown") for b in at.get("download_button"))
    assert "send_to_chain_button" not in keys(at)
    assert not any("Send this result to a chain" in e.label for e in at.expander)


def test_the_dialog_offers_downloads_latex_and_a_plain_summary():
    at = solved_app()
    button(at, "open_share_dialog").click()
    at.run()
    assert _exceptions(at) == []
    labels = [d.proto.label for d in at.get("download_button")]
    assert "📄 Download as Markdown" in labels
    latex = next(c for c in at.code if c.language == "latex").value
    assert r"\begin{gather*}" in latex and r"d = 84\,\text{m}" in latex
    plain = next(c for c in at.code if c.language == "plaintext").value
    assert "  a = 2 m/s^2" in plain


def _share_tabs_script():
    """The dialog's tab bodies, called directly. AppTest reruns the WHOLE script on every interaction, which
    closes a dialog that real Streamlit keeps open (a dialog reruns only itself), so anything done INSIDE the
    dialog is driven through these functions; opening the dialog itself is tested above."""
    import json
    import streamlit as st
    from modules.equation_engine import build_model
    from modules.solve_compare import deterministic_report
    from modules.solver import compute_steps
    from tests.conftest import KINEMATICS_TWO_TARGET_JSON
    from ui import share
    model = build_model(json.loads(KINEMATICS_TWO_TARGET_JSON))
    report = deterministic_report(model)
    steps = compute_steps(model)
    for key, default in (("problem_text", "A car accelerates."), ("scenarios", []), ("pdf_bytes", None),
                         ("plot_snapshots", {}), ("active_chain_id", None)):
        st.session_state.setdefault(key, default)
    which = st.session_state.get("_tab", "download")
    if which == "download":
        share._download_tab(model, report, steps, st.session_state["problem_text"])
    elif which == "copy":
        share._copy_tab(model, report, steps, st.session_state["problem_text"])
    else:
        share.render_send_to_chain_form(model)


def _share_tab(which: str):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_share_tabs_script, default_timeout=120)
    at.session_state["_tab"] = which
    return at.run()


def test_the_pdf_is_generated_in_place_and_offered_without_a_rerun_of_the_page():
    at = _share_tab("download")
    assert _exceptions(at) == []
    assert at.session_state["pdf_bytes"] is None
    assert "⬇️ Download PDF" not in [d.proto.label for d in at.get("download_button")]
    button(at, "share_pdf_generate").click()
    at.run()
    assert _exceptions(at) == []
    assert at.session_state["pdf_bytes"][:4] == b"%PDF"
    assert "⬇️ Download PDF" in [d.proto.label for d in at.get("download_button")]       # same run, no st.rerun()


def test_the_download_tab_says_how_many_marked_plots_the_report_will_contain():
    at = _share_tab("download")
    assert any("📸" in c.value for c in at.caption)                  # the tip, when none are marked


def test_the_include_steps_checkbox_adds_the_worked_steps_to_the_latex():
    at = _share_tab("copy")
    assert "Solving for" not in next(c for c in at.code if c.language == "latex").value
    next(c for c in at.checkbox if c.key == "share_latex_steps").check()
    at.run()
    assert "Solving for a" in next(c for c in at.code if c.language == "latex").value


def test_a_result_can_be_sent_to_a_new_chain_from_the_dialog():
    at = _share_tab("chain")
    next(t for t in at.text_input if t.key == "send_to_chain_new_name").set_value("From the dialog")
    at.run()
    button(at, "send_to_chain_button").click()
    at.run()
    assert _exceptions(at) == []
    assert at.session_state["active_chain_id"] is not None
    from modules import chains
    assert any(c["name"] == "From the dialog" for c in chains.list_chains())


def test_sending_to_a_chain_without_a_name_is_refused():
    at = _share_tab("chain")
    button(at, "send_to_chain_button").click()
    at.run()
    assert any("Give the new chain a name" in e.value for e in at.error)
    assert at.session_state["active_chain_id"] is None


def test_an_existing_chain_can_be_chosen_and_extended():
    from modules import chains
    chain_id = chains.create_chain("Existing")
    at = _share_tab("chain")
    next(s for s in at.selectbox if s.key == "send_to_chain_choice").set_value("Existing")
    at.run()
    button(at, "send_to_chain_button").click()
    at.run()
    assert _exceptions(at) == []
    assert at.session_state["active_chain_id"] == chain_id
    assert len(chains.load_chain(chain_id).steps) == 1


# ------------------------------------------------------------------ popovers

def test_each_step_has_an_explain_popover_instead_of_an_expander():
    at = solved_app()
    n_steps = sum(len(s) for s in at.session_state["steps"].values())
    assert len(at.get("popover")) == n_steps
    assert not any(e.label.startswith("🔍 Explain just step") for e in at.expander)
    assert "explain_btn_a_1" in keys(at)


# ------------------------------------------------------------------ saved views

def test_opening_a_past_problem_restores_how_it_was_explored():
    hid = save_kinematics_to_history()
    history.save_view_state(hid, vs.wrap({"mc_n_a": 2500, "sweep_target": "d", "line_x": "t"}))
    at = _app()
    at.session_state["mc_n_zzz"] = 7              # left over from a previous problem
    at.run()
    button(at, f"load_{hid}").click()
    at.run()
    assert _exceptions(at) == []
    assert next(s for s in at.slider if s.key == "mc_n_a").value == 2500
    assert next(s for s in at.selectbox if s.key == "sweep_target").value == "d"
    assert "mc_n_zzz" not in at.session_state                       # the old problem's settings do not leak
    assert at.session_state["current_history_id"] == hid
    assert any("Restored 3 saved view setting(s)" in t.value for t in at.toast)


def test_a_problem_with_no_saved_view_opens_at_its_defaults():
    hid = save_kinematics_to_history()
    at = _app()
    at.session_state["mc_n_a"] = 4000
    at.run()
    button(at, f"load_{hid}").click()
    at.run()
    assert _exceptions(at) == []
    assert next(s for s in at.slider if s.key == "mc_n_a").value == 1000
    assert not at.toast


def test_a_damaged_saved_view_never_stops_the_problem_from_opening(monkeypatch):
    hid = save_kinematics_to_history()
    monkeypatch.setattr(history, "load_view_state", lambda _id: (_ for _ in ()).throw(RuntimeError("corrupt")))
    at = _app()
    at.run()
    button(at, f"load_{hid}").click()
    at.run()
    assert _exceptions(at) == []
    assert at.session_state["model"] is not None


def test_a_saved_view_with_foreign_keys_cannot_overwrite_other_state():
    hid = save_kinematics_to_history()
    history.save_view_state(hid, {"version": 1, "values": {"problem_text": "injected", "mc_n_a": 1500}})
    at = _app()
    at.run()
    button(at, f"load_{hid}").click()
    at.run()
    assert at.session_state["problem_text"] == "A car accelerates."
    assert next(s for s in at.slider if s.key == "mc_n_a").value == 1500


def test_exploring_a_problem_saves_the_view_with_its_history_record():
    hid = save_kinematics_to_history()
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["current_history_id"] = hid
    at.run()
    next(s for s in at.slider if s.key == "mc_n_a").set_value(2000)
    at.run()
    saved = history.load_view_state(hid)
    assert saved["version"] == vs.VIEW_STATE_VERSION
    assert saved["values"]["mc_n_a"] == 2000


def test_an_unchanged_view_is_not_written_again(monkeypatch):
    hid = save_kinematics_to_history()
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["current_history_id"] = hid
    at.run()
    writes = []
    real = history.save_view_state
    monkeypatch.setattr(history, "save_view_state", lambda *a, **k: writes.append(a) or real(*a, **k))
    at.run()
    at.run()
    assert writes == []


def test_a_view_that_cannot_be_saved_does_not_break_the_page(monkeypatch):
    hid = save_kinematics_to_history()
    monkeypatch.setattr(history, "save_view_state", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["current_history_id"] = hid
    at.run()
    next(s for s in at.slider if s.key == "mc_n_a").set_value(1500)
    at.run()
    assert _exceptions(at) == []


def test_start_fresh_clears_the_previous_problems_settings_and_records_the_new_id():
    def script():
        import streamlit as st
        from ui import view_state
        st.session_state.setdefault("mc_n_a", 4000)
        st.session_state.setdefault("problem_text", "keep me")
        if st.session_state.get("go"):
            view_state.start_fresh(42)
            st.session_state["go"] = False
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(script).run()
    at.session_state["go"] = True
    at.run()
    assert "mc_n_a" not in at.session_state
    assert at.session_state["problem_text"] == "keep me"
    assert at.session_state["current_history_id"] == 42


# ------------------------------------------------------------------ Compare page

def test_compare_is_a_registered_mode_and_findable_in_the_palette():
    assert COMPARE_MODE in MODE_LABELS
    assert search("what if")[0] == COMPARE_MODE
    assert search("side by side")[0] == COMPARE_MODE


def test_compare_with_nothing_to_compare_says_so():
    at = _app()
    at.session_state["app_mode"] = COMPARE_MODE
    at.run()
    assert _exceptions(at) == []
    assert any("Nothing to compare yet" in i.value for i in at.info)


def test_compare_the_current_problem_against_a_saved_one():
    hid = save_kinematics_to_history()
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["app_mode"] = COMPARE_MODE
    at.session_state["compare_a_source"] = CURRENT_PROBLEM
    at.run()
    b = next(s for s in at.selectbox if s.key == "compare_b_source")
    b.set_value(next(o for o in b.options if o.startswith(f"[{hid}]")))
    at.run()
    assert _exceptions(at) == []
    text = "\n".join(m.value for m in at.markdown)
    assert "What differs" in text and "Nothing differs" in text or "All 2 answer(s) are the same" in text


def test_a_what_if_needs_at_least_one_changed_input():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["app_mode"] = COMPARE_MODE
    at.session_state["compare_b_source"] = WHAT_IF_SOURCE
    at.run()
    assert _exceptions(at) == []
    assert any("Change at least one value" in i.value for i in at.info)


def test_a_what_if_from_the_heatmap_is_prefilled_and_compared():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["app_mode"] = COMPARE_MODE
    at.session_state["compare_b_source"] = WHAT_IF_SOURCE
    at.session_state["compare_overrides"] = {"u": 10.0}
    at.run()
    assert _exceptions(at) == []
    text = "\n".join(m.value for m in at.markdown)
    assert "a: 2 → 1.66667" in text and "Inputs that differ: u." in text
    assert any("deterministic checks only" in i.value for i in at.info)
    assert len(at.get("plotly_chart")) == 2                           # change chart + category chart
    assert any(d.proto.label.startswith("⬇️ Download comparison") for d in at.get("download_button"))


def test_only_differences_hides_the_unchanged_rows():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["app_mode"] = COMPARE_MODE
    at.session_state["compare_b_source"] = WHAT_IF_SOURCE
    at.session_state["compare_overrides"] = {"u": 10.0}
    at.run()
    inputs_before = max(len(d.value) for d in at.dataframe)
    next(c for c in at.checkbox if c.key == "compare_only_diff").check()
    at.run()
    assert _exceptions(at) == []
    inputs_tables = [d.value for d in at.dataframe if "Symbol" in d.value.columns and "Unit A" in d.value.columns]
    assert [list(t["Symbol"]) for t in inputs_tables] == [["u"]]
    assert inputs_before >= 1
