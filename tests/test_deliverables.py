"""The in-app deliverables: provenance, saved follow-ups and tutor guesses, copy forms, cached one-click exports,
and the Share dialog's export tab. (Rendering of each format is in test_report_formats.py.)"""
import json
import threading

import pytest

import modules.chains as chains_module
import modules.history as history
from config import APP_VERSION, Settings
from modules import provenance
from modules import session_extras as se
from modules.answer_copy import copy_forms, monte_carlo_copy_forms
from modules.equation_engine import build_model
from modules.followup import FollowupAnswer
from modules.monte_carlo import MonteCarloResult
from modules.project_bundle import export_bundle, import_bundle
from modules.report_content import TutorItem
from modules.solver import compute_steps
from modules.tutor_mode import AnswerCheckResult
from modules.verifier import verify
from tests.conftest import FakeClient, KINEMATICS_TWO_TARGET_JSON
from tests.report_helpers import full_extras, kinematics_ctx, snapshot
from tests.test_interactive_elements import _share_tab, button, keys, save_kinematics_to_history, solved_app
from tests.test_ui_smoke import _app, _exceptions, _seed_solved_kinematics, isolated_data  # noqa: F401
from ui import exports


# ------------------------------------------------------------------ provenance

def test_capture_records_the_settings_and_versions_in_force():
    cfg = Settings(reasoning_model="m1", secondary_reasoning_model="m2", numeric_tolerance=1e-4)
    c = provenance.capture(cfg, used_secondary_model=True)
    assert c["reasoning_model"] == "m1" and c["secondary_model"] == "m2" and c["numeric_tolerance"] == 1e-4
    assert c["app_version"] == APP_VERSION and set(c["libraries"]) == {"sympy", "numpy", "scipy"}
    json.dumps(c)                                                       # storable as-is


def test_a_secondary_model_is_only_recorded_if_it_was_used():
    cfg = Settings(secondary_reasoning_model="m2")
    assert provenance.capture(cfg)["secondary_model"] == ""


def test_describe_uses_the_stored_snapshot_and_labels_the_export_time_separately():
    stored = provenance.capture(Settings(reasoning_model="old-model"))
    now = provenance.capture(Settings(reasoning_model="new-model"))
    rows = dict(provenance.describe(stored, now=now))
    assert "old-model" in rows["Language model"] and "new-model" not in rows["Language model"]
    assert APP_VERSION in rows["Solved"] and APP_VERSION in rows["Exported"]


def test_describe_never_presents_todays_settings_as_an_old_results_provenance():
    rows = dict(provenance.describe(None, now=provenance.capture(Settings(reasoning_model="today"))))
    assert "no record" in rows["Solved"] and "CURRENT" in rows["Solved"] and "today" in rows["Language model"]
    rows2 = dict(provenance.describe({"garbage": 1}, now=provenance.capture(Settings(reasoning_model="today"))))
    assert "no record" in rows2["Solved"]


def test_describe_lists_monte_carlo_seeds_and_states_what_is_not_deterministic():
    rows = dict(provenance.describe(None, [{"target": "a", "n": 500, "seed": 7, "inputs": ["u ~ N(1, 0.1)"]}]))
    assert "500 samples, seed 7" in rows["Monte Carlo (a)"]
    assert "not guaranteed to repeat" in rows["Reproducibility"]


# ------------------------------------------------------------------ saved follow-ups and tutor transcripts

def test_followups_round_trip_through_json():
    hist = [("what if t doubles?", FollowupAnswer("what_if", "a = 1", computed_value=1.0, target="a", symbol="t")),
            ("why?", FollowupAnswer("conceptual", "because"))]
    back = se.followups_from_json(json.loads(json.dumps(se.followups_to_json(hist))))
    assert [(q, a.kind, a.text, a.computed_value, a.target, a.symbol) for q, a in back] == \
           [(q, a.kind, a.text, a.computed_value, a.target, a.symbol) for q, a in hist]


@pytest.mark.parametrize("junk", [None, "x", 5, {"q": "a"}, [None, 3, "s", {"no_q": 1}, {"q": 5}]])
def test_damaged_followup_data_yields_nothing_rather_than_an_error(junk):
    assert se.followups_from_json(junk) == []


def test_followup_data_from_a_bundle_is_bounded_and_sanitised():
    big = [{"q": "q" * 10_000, "kind": "evil", "text": "t" * 10_000, "value": True}] * 200
    back = se.followups_from_json(big)
    assert len(back) == se.MAX_FOLLOWUPS
    q, a = back[0]
    assert len(q) <= se.MAX_TEXT and len(a.text) <= se.MAX_TEXT and a.kind == "conceptual" and a.computed_value is None


def test_tutor_activity_is_read_from_the_session():
    steps = {"a": [object()] * 7, "d": [object()] * 7}
    state = {"tutor_feedback_a": AnswerCheckResult(True, 2.0, 2.0, 0.0, "Correct!"), "tutor_guess_a": " 2 ",
             "tutor_reveal_a": 3, "tutor_reveal_d": 2}
    items = {t.target: t for t in se.tutor_from_session(state, steps)}
    assert items["a"] == TutorItem("a", "2", True, "Correct!", 3, 7)
    assert items["d"] == TutorItem("d", "", None, "", 2, 7)           # revealed steps but never guessed


def test_a_guess_that_could_not_be_marked_is_recorded_as_unmarked():
    fb = AnswerCheckResult(False, None, None, None, "", error="Couldn't read that as a number")
    item, = se.tutor_from_session({"tutor_feedback_a": fb, "tutor_guess_a": "abc"}, {"a": [object()] * 3})
    assert item.correct is None and "number" in item.feedback


def test_a_target_with_no_tutor_activity_is_not_listed():
    assert se.tutor_from_session({"tutor_on_a": True}, {"a": [object()]}) == []


def test_tutor_round_trips_and_live_activity_replaces_the_saved_for_that_target():
    saved = [TutorItem("a", "1", False, "no", 1, 7), TutorItem("d", "84", True, "yes", 7, 7)]
    assert se.tutor_from_json(json.loads(json.dumps(se.tutor_to_json(saved)))) == saved
    live = [TutorItem("a", "2", True, "yes", 7, 7)]
    assert se.merge_tutor(saved, live) == [saved[1], live[0]]
    assert se.tutor_from_json("junk") == [] and se.tutor_from_json([{"target": 5}, 3]) == []


def test_monte_carlo_runs_come_from_the_session_with_samples_only_when_asked():
    ctx = kinematics_ctx()
    r = MonteCarloResult(target="a", samples=[1.0, 2.0], n_requested=2, seed=9, mean=1.5, std=0.5, p5=1.0, p95=2.0,
                         inputs=["u ~ N(8, 1)"])
    runs = se.monte_carlo_runs({"mc_result_a": r}, ctx.model)
    assert len(runs) == 1 and runs[0]["seed"] == 9 and "samples" not in runs[0] and runs[0]["unit"] == "m/s^2"
    assert se.monte_carlo_runs({"mc_result_a": r}, ctx.model, with_samples=True)[0]["samples"] == [1.0, 2.0]
    assert se.monte_carlo_runs({}, ctx.model) == []


# ------------------------------------------------------------------ history extras and bundles

@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "h.db")
    monkeypatch.setattr(chains_module, "DB_PATH", tmp_path / "c.db")


def _record() -> int:
    model = build_model(json.loads(KINEMATICS_TWO_TARGET_JSON))
    report = verify(model, FakeClient(KINEMATICS_TWO_TARGET_JSON, final_answers={"a": 2.0, "d": 84.0}), "p")
    return history.save("A car.", model, report, compute_steps(model), [])


def test_extras_are_stored_in_the_record_and_removed_when_emptied(db):
    hid = _record()
    assert history.load_extra(hid, "followups") is None
    assert history.save_extra(hid, "followups", [{"q": "x"}]) and history.load_extra(hid, "followups") == [{"q": "x"}]
    history.save_extra(hid, "followups", [])
    assert history.load_extra(hid, "followups") is None
    assert history.load(hid) is not None                                  # the problem itself is untouched


def test_an_unknown_extra_is_refused_and_a_missing_record_reports_it(db):
    hid = _record()
    with pytest.raises(ValueError, match="Unknown"):
        history.save_extra(hid, "problem", "overwrite!")
    with pytest.raises(ValueError):
        history.load_extra(hid, "../x")
    assert history.save_extra(999, "followups", [1]) is False and history.load_extra(999, "tutor") is None


def test_followups_tutor_and_provenance_travel_in_a_project_bundle(db):
    hid = _record()
    extras = {"followups": [{"q": "why?", "kind": "conceptual", "text": "because"}],
              "tutor": [{"target": "a", "guess": "2", "correct": True, "feedback": "ok", "revealed": 3, "total": 7}],
              "provenance": {"reasoning_model": "m", "app_version": "1"}}
    for k, v in extras.items():
        history.save_extra(hid, k, v)
    bundle = export_bundle(history_ids=[hid], chain_ids=[], workspace_entries={})
    assert json.loads(bundle)["history"][0]["extras"] == extras
    for r in history.list_recent():
        history.delete(r["id"])

    class WS:
        entries: dict = {}
        def store(self, *a, **k): pass

    assert import_bundle(bundle, WS()).history_imported == 1
    new_id = history.list_recent()[0]["id"]
    assert {k: history.load_extra(new_id, k) for k in extras} == extras


def test_a_bundle_cannot_smuggle_other_payload_keys_in_through_extras(db):
    hid = _record()
    bundle = json.loads(export_bundle(history_ids=[hid], chain_ids=[], workspace_entries={}))
    bundle["history"][0]["extras"] = {"problem_text": "hijack", "followups": [{"q": "ok", "kind": "conceptual", "text": "t"}]}
    for r in history.list_recent():
        history.delete(r["id"])

    class WS:
        entries: dict = {}
        def store(self, *a, **k): pass

    import_bundle(json.dumps(bundle), WS())
    new_id = history.list_recent()[0]["id"]
    assert history.load_extra(new_id, "followups") == [{"q": "ok", "kind": "conceptual", "text": "t"}]
    assert history.load(new_id)[0] == "A car."


# ------------------------------------------------------------------ copy forms

def test_an_answer_is_offered_as_plain_text_with_units_latex_and_python():
    ctx = kinematics_ctx()
    forms = copy_forms(ctx.model, ctx.report, "a")
    assert list(forms) == ["Plain text", "With units", "LaTeX", "Python value", "Python function"]
    assert forms["Plain text"] == ("a = 2", None) and forms["With units"][0] == "a = 2 m/s^2"
    assert forms["LaTeX"] == (r"a = 2\,\text{m/s}^{2}", "latex")
    assert forms["Python value"] == ("a = 2.0  # m/s^2", "python")
    src = forms["Python function"][0]
    namespace: dict = {}
    exec(src, namespace)                                                # the function really runs
    assert namespace["a"](t=6.0, u=8.0, v=20.0) == pytest.approx(2.0)


def test_a_unitless_answer_gets_no_unit_text_and_python_keeps_full_precision():
    ctx = kinematics_ctx()
    ctx.model.variables[3].unit = "unitless"
    ctx.report.sympy_numeric_answers["a"] = 1 / 3
    forms = copy_forms(ctx.model, ctx.report, "a")
    assert forms["With units"][0] == "a = 0.333333"
    assert forms["Python value"][0] == "a = 0.3333333333333333"
    assert forms["LaTeX"][0] == "a = 0.333333"


def test_a_target_without_a_numeric_answer_has_no_copy_forms():
    ctx = kinematics_ctx()
    assert copy_forms(ctx.model, ctx.report, "nope") == {}


def test_a_monte_carlo_summary_says_how_to_reproduce_the_run():
    forms = monte_carlo_copy_forms("a", 2.0, 0.07, 1.9, 2.1, 1000, 5, "m/s^2")
    assert forms["Plain text"][0] == "a = 2 ± 0.07 m/s^2; 5th-95th percentile 1.9 to 2.1 (1,000 samples, seed 5)"
    assert r"\pm" in forms["LaTeX"][0] and "seed=5" in forms["Python value"][0]
    assert "percentile" not in monte_carlo_copy_forms("a", 2.0, 0.07, None, None, 10, 1, None)["Plain text"][0]


# ------------------------------------------------------------------ cached, deferred exports

@pytest.fixture(autouse=True)
def fresh_export_cache():
    exports.clear_cache()
    yield
    exports.clear_cache()


def test_the_deferred_download_builds_the_file_only_when_called():
    ctx = kinematics_ctx()
    data = exports.deferred("md", ctx, ["problem"])()
    assert data.startswith(b"# Math Representation System")


def test_the_same_request_is_served_from_the_cache_and_a_changed_one_is_not(monkeypatch):
    built = []
    real = exports.render_export
    monkeypatch.setattr(exports, "render_export", lambda *a, **k: built.append(a[0]) or real(*a, **k))
    ctx = kinematics_ctx()
    first = exports.build_cached("md", ctx, ["problem", "results"])
    again = exports.build_cached("md", ctx, ["problem", "results"])
    assert first is again and built == ["md"]
    exports.build_cached("md", ctx, ["problem"])                          # different sections
    exports.build_cached("html", ctx, ["problem", "results"])             # different format
    ctx.report.sympy_numeric_answers["a"] = 99.0                          # different content
    changed = exports.build_cached("md", ctx, ["problem", "results"])
    assert built == ["md", "md", "html", "md"] and b"99" in changed.data and changed is not first


def test_section_order_in_the_request_does_not_defeat_the_cache(monkeypatch):
    built = []
    real = exports.render_export
    monkeypatch.setattr(exports, "render_export", lambda *a, **k: built.append(1) or real(*a, **k))
    ctx = kinematics_ctx()
    exports.build_cached("md", ctx, ["results", "problem"])
    exports.build_cached("md", ctx, ["problem", "results"])
    assert len(built) == 1


def test_the_cache_is_bounded_by_entries_and_by_bytes():
    ctx = kinematics_ctx()
    for i in range(exports.MAX_ENTRIES + 5):
        ctx.problem_text = f"problem {i}"
        exports.build_cached("md", ctx, ["problem"])
    assert len(exports._files) == exports.MAX_ENTRIES
    exports.clear_cache()
    old = exports.MAX_BYTES
    exports.MAX_BYTES = 10
    try:
        exports.build_cached("md", ctx, ["problem", "results"])
        exports.build_cached("md", ctx, ["problem"])
        assert len(exports._files) == 1                                    # always keeps the one just built
    finally:
        exports.MAX_BYTES = old


def test_building_from_several_threads_is_safe():
    ctx = kinematics_ctx()
    results, errors = [], []

    def work():
        try:
            results.append(exports.build_cached("md", ctx, ["problem", "results"]).data)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == [] and len({bytes(r) for r in results}) == 1


def test_a_context_that_cannot_be_fingerprinted_is_rebuilt_not_failed():
    ctx = kinematics_ctx()
    ctx.model.variables[0].meaning = object()                              # not fingerprintable
    assert exports._key("md", ctx, None) is None


# ------------------------------------------------------------------ the dialog's export tab

def test_the_export_tab_offers_only_sections_that_have_content_and_ticks_the_default_preset():
    at = _share_tab()
    assert _exceptions(at) == []
    boxes = {c.key: c.value for c in at.checkbox if c.key and c.key.startswith("share_sec_")}
    assert boxes["share_sec_steps"] is True and boxes["share_sec_results"] is True
    for absent in ("matrix", "vectors", "plots", "scenarios", "followups", "tutor"):
        assert f"share_sec_{absent}" not in boxes
    assert any("Nothing to include for" in c.value for c in at.caption)


def test_every_format_has_a_button_and_a_note():
    at = _share_tab()
    assert sorted(d.proto.label for d in at.get("download_button")) == \
           sorted(["📕 PDF", "🌐 Interactive HTML", "📘 Word", "📙 PowerPoint", "📄 LaTeX", "📝 Markdown"])
    assert any("5 MB" in c.value for c in at.caption)


def test_choosing_a_preset_ticks_exactly_its_sections():
    at = _share_tab()
    next(s for s in at.selectbox if s.key == "share_preset").set_value("Quick summary")
    at.run()
    boxes = {c.key[len("share_sec_"):]: c.value for c in at.checkbox if c.key and c.key.startswith("share_sec_")}
    assert {k for k, v in boxes.items() if v} == {"problem", "results", "confidence", "provenance"} & set(boxes)
    assert at.session_state["share_selection"] == ["problem", "results", "confidence", "provenance"]


def test_changing_a_tick_switches_the_preset_to_custom_and_is_remembered():
    at = _share_tab()
    next(c for c in at.checkbox if c.key == "share_sec_verification").uncheck()
    at.run()
    assert next(s for s in at.selectbox if s.key == "share_preset").value == "Custom"
    assert "verification" not in at.session_state["share_selection"] and "steps" in at.session_state["share_selection"]


def test_with_nothing_ticked_there_is_nothing_to_download():
    at = _share_tab()
    for box in [c for c in at.checkbox if c.key and c.key.startswith("share_sec_")]:
        box.uncheck()
    at.run()
    assert any("Tick at least one section" in w.value for w in at.warning)
    assert not at.get("download_button")


# ------------------------------------------------------------------ in the running app

def test_every_answer_has_a_copy_popover_with_the_four_forms():
    at = solved_app()
    popovers = {p.children[0].key: p for p in at.get("popover") if getattr(p.children.get(0), "key", "") and
                p.children[0].key.startswith("copy_fmt_")}
    assert set(popovers) == {"copy_fmt_a", "copy_fmt_d"}
    assert popovers["copy_fmt_a"].children[0].options == ["Plain text", "With units", "LaTeX", "Python value", "Python function"]
    assert any(c.value == "a = 2" for c in at.code)


def test_choosing_a_copy_format_shows_that_form():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["copy_fmt_a"] = "LaTeX"
    at.run()
    assert any(c.language == "latex" and c.value == r"a = 2\,\text{m/s}^{2}" for c in at.code)


def test_a_monte_carlo_result_has_its_own_copy_popover_with_the_seed():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["mc_vars_a"] = ["u"]
    at.run()
    button(at, "mc_run_a").click()
    at.run()
    assert _exceptions(at) == []
    assert "copy_fmt_mc_a" in [r.key for r in at.radio]
    seed = at.session_state["mc_result_a"].seed
    assert any(f"seed {seed}" in c.value for c in at.code)
    assert at.session_state["mc_result_a"].inputs and "u ~ N(" in at.session_state["mc_result_a"].inputs[0]


def test_including_a_plot_keeps_its_live_figure_for_the_html_report():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["tornado_a"] = {"selection": {"points": [], "point_indices": [], "box": [], "lasso": []}}
    at.run()
    button(at, "include_snap_tornado_a").click()
    at.run()
    snap = at.session_state["plot_snapshots"]["tornado_a"]
    assert snap.png_bytes[:4] == b"\x89PNG" and snap.figure_json.startswith("{")
    assert json.loads(snap.figure_json)["data"]


def test_follow_ups_are_saved_with_the_problem_and_restored_when_it_is_reopened():
    hid = save_kinematics_to_history()
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["current_history_id"] = hid
    at.session_state["followup_history"] = [("what if t doubles?", FollowupAnswer("what_if", "a becomes 1", computed_value=1.0))]
    at.run()
    assert history.load_extra(hid, "followups")[0]["q"] == "what if t doubles?"
    fresh = _app()
    fresh.run()
    button(fresh, f"load_{hid}").click()
    fresh.run()
    assert _exceptions(fresh) == []
    (q, a), = fresh.session_state["followup_history"]
    assert q == "what if t doubles?" and a.text == "a becomes 1" and a.computed_value == 1.0


def test_one_problems_follow_ups_do_not_show_up_under_the_next():
    hid = save_kinematics_to_history()
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["followup_history"] = [("old question", FollowupAnswer("conceptual", "old answer"))]
    at.session_state["tutor_feedback_a"] = AnswerCheckResult(True, 2.0, 2.0, 0.0, "yes")
    at.run()
    button(at, f"load_{hid}").click()
    at.run()
    assert at.session_state["followup_history"] == []
    assert "tutor_feedback_a" not in at.session_state


def test_the_tutor_transcript_is_saved_and_comes_back_after_reopening():
    hid = save_kinematics_to_history()
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["current_history_id"] = hid
    at.session_state["tutor_on_a"] = True
    at.session_state["tutor_reveal_a"] = 2
    at.session_state["tutor_guess_a"] = "2"
    at.session_state["tutor_feedback_a"] = AnswerCheckResult(True, 2.0, 2.0, 0.0, "Correct!")
    at.run()
    saved = history.load_extra(hid, "tutor")
    assert saved == [{"target": "a", "guess": "2", "correct": True, "feedback": "Correct!", "revealed": 2, "total": 7}]
    fresh = _app()
    fresh.run()
    button(fresh, f"load_{hid}").click()
    fresh.run()
    assert fresh.session_state["tutor_saved"] == se.tutor_from_json(saved)


def test_how_a_problem_was_solved_is_stored_with_it_and_restored(db):
    from ui import view_state
    hid = _record()
    prov = provenance.capture(Settings(reasoning_model="solver-model"))

    def script():
        import streamlit as st
        from ui import view_state as vs
        if st.session_state.get("go") == "fresh":
            vs.start_fresh(st.session_state["hid"], st.session_state["prov"])
            st.session_state["go"] = ""
        elif st.session_state.get("go") == "restore":
            vs.restore(st.session_state["hid"])
            st.session_state["go"] = ""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(script)
    at.session_state["hid"], at.session_state["prov"], at.session_state["go"] = hid, prov, "fresh"
    at.run()
    assert at.session_state["provenance"] == prov and history.load_extra(hid, "provenance") == prov
    at.session_state["go"], at.session_state["provenance"] = "restore", None
    at.run()
    assert at.session_state["provenance"] == prov


def test_the_dialog_gets_follow_ups_and_provenance_sections_when_they_exist():
    at = _app(); _seed_solved_kinematics(at)
    at.session_state["followup_history"] = [("q?", FollowupAnswer("conceptual", "a."))]
    at.session_state["provenance"] = full_extras().provenance
    at.session_state["tutor_saved"] = [TutorItem("a", "2", True, "ok", 1, 7)]
    at.run()
    button(at, "open_share_dialog").click()
    at.run()
    assert _exceptions(at) == []
    keys_ = {c.key for c in at.checkbox if c.key}
    assert {"share_sec_followups", "share_sec_provenance", "share_sec_tutor"} <= keys_
