"""Saving how a problem was being explored (modules/view_state.py, history storage, bundles)."""
import glob
import json
import re

import pytest

import modules.chains as chains_module
import modules.history as history
from modules import view_state as vs
from modules.equation_engine import build_model
from modules.project_bundle import export_bundle, import_bundle
from modules.solver import compute_steps
from modules.verifier import verify
from modules.workspace import Workspace
from tests.conftest import FakeClient, KINEMATICS_TWO_TARGET_JSON


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    monkeypatch.setattr(chains_module, "DB_PATH", tmp_path / "chains.db")


def _saved_problem() -> int:
    model = build_model(json.loads(KINEMATICS_TWO_TARGET_JSON))
    report = verify(model, FakeClient(KINEMATICS_TWO_TARGET_JSON, final_answers={"a": 2.0, "d": 84.0}), "p")
    return history.save("A car accelerates.", model, report, compute_steps(model), [])


# ------------------------------------------------------------------ which keys count

@pytest.mark.parametrize("key", ["mc_n_a", "mc_seed_a", "sweep_symbols", "sweep_target", "bulk_mc_n", "var_v0",
                                 "line_x", "line_x_log", "surf_slider_t", "fan_n_x", "morph_param_x",
                                 "bif_n_x", "odeparam_y_k", "motion_show_a", "tutor_on_a", "slider_t"])
def test_exploration_keys_are_recognised(key):
    assert vs.is_exploration_key(key)


@pytest.mark.parametrize("key", ["problem_text", "model", "app_mode", "palette_query", "batch_text_input",
                                 "fit_csv_paste", "sweep_editor_u,t", "mc_editor_a_u", "_ui_cache", "", None, 5])
def test_everything_else_is_not(key):
    assert not vs.is_exploration_key(key)


def test_a_pattern_must_match_the_whole_key():
    assert not vs.is_exploration_key("xmc_n_a")
    assert not vs.is_exploration_key("var")        # "var_" needs something after it


def _widget_keys(path_glob: str) -> set[str]:
    pat = re.compile(r"st\.(slider|number_input|selectbox|multiselect|checkbox|toggle|radio|select_slider)\(")
    keys = set()
    for f in sorted(glob.glob(path_glob)):
        src = open(f, encoding="utf-8").read()
        for m in pat.finditer(src):
            i, depth = m.end(), 1
            while depth and i < len(src):
                depth += (src[i] == "(") - (src[i] == ")")
                i += 1
            k = re.search(r'key\s*=\s*f?"([^"]*)"', src[m.start():i])
            if k:
                keys.add(re.sub(r"\{[^}]*\}", "X", k.group(1)))
    return keys


# Widgets in the results page that are deliberately NOT part of a saved view: grading and worksheet inputs
# belong to the practice tab's own flow, and the self-consistency / adversarial panels re-run an LLM.
NOT_PERSISTED = {"adv_target", "grade_target", "grade_work_input_method", "grade_work_use_vision",
                 "self_consistency_runs", "self_consistency_spread_target", "worksheet_count",
                 "worksheet_difficulty", "worksheet_targeted"}


def test_every_results_widget_is_either_saved_or_deliberately_not():
    """A new widget key in ui/results/ must be a conscious decision: add it to the patterns in
    modules/view_state.py, or to NOT_PERSISTED above. Otherwise it would silently be dropped from every
    saved view."""
    undecided = {k for k in _widget_keys("ui/results/*.py") if not vs.is_exploration_key(k)} - NOT_PERSISTED
    assert undecided == set(), f"decide whether these are part of a saved view: {sorted(undecided)}"


def test_no_pattern_claims_a_key_from_a_page_outside_the_results_view():
    """A pattern that also matched, say, the curve-fitting page's keys would sweep them into a problem's
    saved view and clear them when another problem is opened."""
    keys = set()
    for f in glob.glob("ui/*.py"):
        if f.endswith("view_state.py"):
            continue
        src = open(f, encoding="utf-8").read()
        keys |= {re.sub(r"\{[^}]*\}", "X", m.group(1)) for m in re.finditer(r'key\s*=\s*f?"([^"]*)"', src)}
    assert {k for k in keys if vs.is_exploration_key(k)} == set()


# ------------------------------------------------------------------ capturing

def test_capture_keeps_only_exploration_scalars_and_short_lists_sorted_by_key():
    ss = {"mc_n_a": 500, "mc_seed_a": 99, "sweep_symbols": ["u", "t"], "problem_text": "no",
          "sweep_editor_u,t": {"edited_rows": {}}, "fan_env_x": True, "var_u": 2.5, "line_x": "t"}
    got = vs.capture(ss)
    assert list(got) == sorted(got)
    assert got == {"fan_env_x": True, "line_x": "t", "mc_n_a": 500, "mc_seed_a": 99,
                   "sweep_symbols": ["u", "t"], "var_u": 2.5}


def test_values_that_cannot_be_saved_are_dropped_not_stringified():
    ss = {"mc_n_a": float("nan"), "mc_n_b": object(), "mc_n_c": {"nested": 1}, "mc_n_d": [1, [2]],
          "mc_n_e": "x" * (vs.MAX_STRING + 1), "mc_n_f": list(range(vs.MAX_LIST + 1)), "mc_n_ok": 3}
    assert vs.capture(ss) == {"mc_n_ok": 3}


def test_numpy_scalars_are_saved_as_plain_numbers():
    import numpy as np
    got = vs.capture({"mc_n_a": np.int64(5), "var_u": np.float64(2.5), "fan_env_x": np.bool_(True)})
    assert got == {"fan_env_x": True, "mc_n_a": 5, "var_u": 2.5}
    json.dumps(got)         # genuinely JSON-serialisable


def test_a_key_that_raises_when_read_is_skipped():
    class Hostile(dict):
        def __getitem__(self, k):
            raise RuntimeError("boom")
    h = Hostile({"mc_n_a": 1})
    assert vs.capture(h) == {}


def test_digest_changes_with_the_state_and_not_with_key_order():
    assert vs.digest({"a": 1, "b": 2}) == vs.digest({"b": 2, "a": 1})
    assert vs.digest({"a": 1}) != vs.digest({"a": 2})


def test_sanitize_drops_foreign_keys_and_bad_values_from_untrusted_input():
    saved = vs.wrap({"mc_n_a": 10, "problem_text": "inject", "line_x": {"x": 1}, "var_u": 1.5})
    assert vs.sanitize_saved(saved) == {"mc_n_a": 10, "var_u": 1.5}
    assert vs.sanitize_saved("not a mapping") == {}
    assert vs.sanitize_saved({"values": "oops"}) == {}
    assert vs.sanitize_saved(None) == {}


def test_sanitize_also_accepts_a_bare_values_mapping():
    assert vs.sanitize_saved({"mc_n_a": 3}) == {"mc_n_a": 3}


def test_capture_is_capped():
    ss = {f"mc_n_{i}": i for i in range(vs.MAX_KEYS + 50)}
    assert len(vs.capture(ss)) == vs.MAX_KEYS


def test_keys_to_clear_lists_only_exploration_keys():
    assert sorted(vs.keys_to_clear({"mc_n_a": 1, "problem_text": "x", "line_x": "t"})) == ["line_x", "mc_n_a"]


# ------------------------------------------------------------------ storage with the history record

def test_a_saved_view_round_trips_through_the_history_record():
    hid = _saved_problem()
    assert history.load_view_state(hid) == {}                       # nothing yet
    state = vs.wrap({"mc_n_a": 2500, "sweep_symbols": ["u", "t"]})
    assert history.save_view_state(hid, state) is True
    assert history.load_view_state(hid) == state


def test_saving_a_view_does_not_disturb_the_rest_of_the_record():
    hid = _saved_problem()
    before = history.load(hid)
    history.save_view_state(hid, vs.wrap({"mc_n_a": 1}))
    after = history.load(hid)
    assert after[0] == before[0]
    assert after[2].sympy_numeric_answers == before[2].sympy_numeric_answers
    assert {t: [s.expression for s in steps] for t, steps in after[3].items()} == \
           {t: [s.expression for s in steps] for t, steps in before[3].items()}


def test_an_empty_view_removes_a_saved_one():
    hid = _saved_problem()
    history.save_view_state(hid, vs.wrap({"mc_n_a": 1}))
    history.save_view_state(hid, vs.wrap({}))
    assert history.load_view_state(hid) == {}


def test_a_missing_record_has_no_view_and_cannot_be_saved_to():
    assert history.load_view_state(9999) == {}
    assert history.save_view_state(9999, vs.wrap({"mc_n_a": 1})) is False


def test_a_record_saved_before_this_feature_simply_has_no_view():
    hid = _saved_problem()
    assert history.load(hid) is not None
    assert history.load_view_state(hid) == {}


# ------------------------------------------------------------------ project bundles

class _WS:
    entries: dict = {}

    def store(self, *a, **k):
        pass


def test_the_saved_view_travels_in_a_project_bundle():
    hid = _saved_problem()
    state = vs.wrap({"mc_n_a": 2500, "sweep_target": "d"})
    history.save_view_state(hid, state)
    bundle = export_bundle(history_ids=[hid], chain_ids=[], workspace_entries={})
    assert json.loads(bundle)["history"][0]["view_state"] == state

    for r in history.list_recent():
        history.delete(r["id"])
    summary = import_bundle(bundle, _WS())
    assert summary.history_imported == 1 and summary.errors == []
    new_id = history.list_recent()[0]["id"]
    assert history.load_view_state(new_id) == state


def test_a_bundle_without_views_still_imports():
    hid = _saved_problem()
    bundle = export_bundle(history_ids=[hid], chain_ids=[], workspace_entries={})
    assert "view_state" not in json.loads(bundle)["history"][0]        # additive: omitted when empty
    for r in history.list_recent():
        history.delete(r["id"])
    assert import_bundle(bundle, _WS()).history_imported == 1
