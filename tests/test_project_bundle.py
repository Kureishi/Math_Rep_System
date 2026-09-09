import json

import pytest

import modules.history as history_module
import modules.chains as chains_module
from modules.chains import InputBinding
from modules.equation_engine import build_model
from modules.verifier import verify
from modules.solver import compute_steps
from modules.workspace import Workspace
from modules.project_bundle import export_bundle, import_bundle, BUNDLE_FORMAT_VERSION


@pytest.fixture(autouse=True)
def _redirect_dbs(tmp_path, monkeypatch):
    monkeypatch.setattr(history_module, "DB_PATH", tmp_path / "history_a.db")
    monkeypatch.setattr(chains_module, "DB_PATH", tmp_path / "chains_a.db")


def _kinematics_model():
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "time", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })


def _save_a_history_entry(fake_client_factory, text="A car problem."):
    model = _kinematics_model()
    client = fake_client_factory(final_answers={"a": 2.0})
    report = verify(model, client, text)
    steps = compute_steps(model)
    return history_module.save(text, model, report, steps, [])


# ---------------------------------------------------------------- export_bundle

def test_export_includes_history_records(fake_client_factory):
    _save_a_history_entry(fake_client_factory)
    bundle = json.loads(export_bundle())
    assert bundle["format_version"] == BUNDLE_FORMAT_VERSION
    assert len(bundle["history"]) == 1
    assert bundle["history"][0]["problem_text"] == "A car problem."
    assert bundle["history"][0]["raw_json"]["problem_domain"] == "kinematics"


def test_export_includes_chains():
    model = _kinematics_model()
    cid = chains_module.create_chain("My chain")
    chains_module.add_step(cid, "A car problem.", model, "a")
    bundle = json.loads(export_bundle())
    assert len(bundle["chains"]) == 1
    assert bundle["chains"][0]["name"] == "My chain"
    assert bundle["chains"][0]["steps"][0]["output_symbol"] == "a"


def test_export_includes_workspace_entries():
    ws = Workspace({})
    ws.store("v0", 5.0, "manual entry", "m/s")
    bundle = json.loads(export_bundle(workspace_entries=ws.entries))
    assert len(bundle["workspace"]) == 1
    assert bundle["workspace"][0]["name"] == "v0"
    assert bundle["workspace"][0]["value"] == 5.0


def test_export_with_no_data_produces_empty_bundle():
    bundle = json.loads(export_bundle())
    assert bundle["history"] == []
    assert bundle["chains"] == []
    assert bundle["workspace"] == []


def test_export_specific_history_ids_only(fake_client_factory):
    id1 = _save_a_history_entry(fake_client_factory, "problem one")
    _save_a_history_entry(fake_client_factory, "problem two")
    bundle = json.loads(export_bundle(history_ids=[id1]))
    assert len(bundle["history"]) == 1
    assert bundle["history"][0]["problem_text"] == "problem one"


def test_export_specific_chain_ids_only():
    model = _kinematics_model()
    cid1 = chains_module.create_chain("chain one")
    chains_module.add_step(cid1, "p", model, "a")
    cid2 = chains_module.create_chain("chain two")
    chains_module.add_step(cid2, "p", model, "a")
    bundle = json.loads(export_bundle(chain_ids=[cid1]))
    assert len(bundle["chains"]) == 1
    assert bundle["chains"][0]["name"] == "chain one"


# ---------------------------------------------------------------- import_bundle: full round trip

def test_round_trip_history_to_a_fresh_database(fake_client_factory, tmp_path, monkeypatch):
    _save_a_history_entry(fake_client_factory, "A car problem.")
    bundle_json = export_bundle()

    monkeypatch.setattr(history_module, "DB_PATH", tmp_path / "history_b.db")
    monkeypatch.setattr(chains_module, "DB_PATH", tmp_path / "chains_b.db")
    ws = Workspace({})
    summary = import_bundle(bundle_json, ws)

    assert summary.history_imported == 1
    assert summary.errors == []
    recent = history_module.list_recent()
    assert len(recent) == 1
    assert recent[0]["problem_text"] == "A car problem."

    loaded = history_module.load(recent[0]["id"])
    problem_text, model, report, steps_by_target, scenarios = loaded
    assert report.sympy_numeric_answers["a"] == pytest.approx(2.0)


def test_round_trip_chain_to_a_fresh_database(tmp_path, monkeypatch):
    model = _kinematics_model()
    cid = chains_module.create_chain("Test chain")
    chains_module.add_step(cid, "A car problem.", model, "a")
    bundle_json = export_bundle()

    monkeypatch.setattr(history_module, "DB_PATH", tmp_path / "history_b.db")
    monkeypatch.setattr(chains_module, "DB_PATH", tmp_path / "chains_b.db")
    ws = Workspace({})
    summary = import_bundle(bundle_json, ws)

    assert summary.chains_imported == 1
    imported_chains = chains_module.list_chains()
    assert len(imported_chains) == 1
    assert imported_chains[0]["name"] == "Test chain"
    loaded_chain = chains_module.load_chain(imported_chains[0]["id"])
    assert loaded_chain.steps[0].output_value == pytest.approx(2.0)


def test_round_trip_chain_with_bindings_preserved(tmp_path, monkeypatch):
    model = _kinematics_model()
    cid = chains_module.create_chain("Chained")
    chains_module.add_step(cid, "step 0", model, "a")
    downstream_model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
            {"symbol": "t2", "meaning": "time", "known_value": "10", "unit": "s"},
            {"symbol": "d", "meaning": "distance", "known_value": None, "unit": "m"},
        ],
        "equations": [{"name": "dist", "kind": "equation", "expression": "Eq(d, a * t2)", "derivation": ""}],
        "solve_for": ["d"], "assumptions": [],
    })
    chains_module.add_step(
        cid, "step 1", downstream_model, "d",
        bindings=[InputBinding(symbol="a", source="upstream", upstream_position=0, upstream_symbol="a")],
    )
    bundle_json = export_bundle()

    monkeypatch.setattr(history_module, "DB_PATH", tmp_path / "history_b.db")
    monkeypatch.setattr(chains_module, "DB_PATH", tmp_path / "chains_b.db")
    ws = Workspace({})
    summary = import_bundle(bundle_json, ws)
    assert summary.chains_imported == 1

    imported_chains = chains_module.list_chains()
    loaded_chain = chains_module.load_chain(imported_chains[0]["id"])
    assert len(loaded_chain.steps) == 2
    assert loaded_chain.steps[1].status == "ok"
    assert loaded_chain.steps[1].output_value == pytest.approx(2.0 * 10)


def test_round_trip_workspace_merges_without_clobbering_existing(tmp_path, monkeypatch):
    ws_export = Workspace({})
    ws_export.store("v0", 5.0, "manual entry", "m/s")
    bundle_json = export_bundle(workspace_entries=ws_export.entries)

    monkeypatch.setattr(history_module, "DB_PATH", tmp_path / "history_b.db")
    monkeypatch.setattr(chains_module, "DB_PATH", tmp_path / "chains_b.db")
    ws_import = Workspace({})
    ws_import.store("already_here", 99.0, "pre-existing", None)
    summary = import_bundle(bundle_json, ws_import)

    assert summary.workspace_imported == 1
    assert "v0" in ws_import.entries
    assert "already_here" in ws_import.entries  # untouched
    assert ws_import.entries["already_here"].value == 99.0


def test_import_skips_workspace_entry_that_already_exists(tmp_path, monkeypatch):
    ws_export = Workspace({})
    ws_export.store("v0", 5.0, "manual entry", "m/s")
    bundle_json = export_bundle(workspace_entries=ws_export.entries)

    monkeypatch.setattr(history_module, "DB_PATH", tmp_path / "history_b.db")
    monkeypatch.setattr(chains_module, "DB_PATH", tmp_path / "chains_b.db")
    ws_import = Workspace({})
    ws_import.store("v0", 111.0, "already present", None)  # name collision
    summary = import_bundle(bundle_json, ws_import)

    assert summary.workspace_imported == 0
    assert any("v0" in e for e in summary.errors)
    assert ws_import.entries["v0"].value == 111.0  # NOT overwritten


# ---------------------------------------------------------------- import_bundle: error handling

def test_import_rejects_invalid_json():
    summary = import_bundle("not valid json {{{", Workspace({}))
    assert summary.history_imported == 0
    assert summary.errors
    assert "JSON" in summary.errors[0]


def test_import_rejects_unrecognized_format_version():
    bundle_json = json.dumps({"format_version": 999, "history": [], "chains": [], "workspace": []})
    summary = import_bundle(bundle_json, Workspace({}))
    assert summary.history_imported == 0
    assert any("format version" in e for e in summary.errors)


def test_import_malformed_history_record_does_not_abort_the_rest(fake_client_factory, tmp_path, monkeypatch):
    _save_a_history_entry(fake_client_factory, "good problem")
    bundle = json.loads(export_bundle())
    bundle["history"].append({"problem_text": "a broken record"})  # missing raw_json/verification/etc
    bundle_json = json.dumps(bundle)

    monkeypatch.setattr(history_module, "DB_PATH", tmp_path / "history_b.db")
    monkeypatch.setattr(chains_module, "DB_PATH", tmp_path / "chains_b.db")
    summary = import_bundle(bundle_json, Workspace({}))

    assert summary.history_imported == 1  # the good one still made it through
    assert len(summary.errors) == 1


def test_import_malformed_chain_does_not_abort_the_rest(tmp_path, monkeypatch):
    model = _kinematics_model()
    chains_module.create_chain("chain")
    chains_module.add_step(chains_module.list_chains()[0]["id"], "p", model, "a")
    bundle = json.loads(export_bundle())
    bundle["chains"].append({"name": "a broken chain"})  # missing steps
    bundle_json = json.dumps(bundle)

    monkeypatch.setattr(history_module, "DB_PATH", tmp_path / "history_b.db")
    monkeypatch.setattr(chains_module, "DB_PATH", tmp_path / "chains_b.db")
    summary = import_bundle(bundle_json, Workspace({}))

    assert summary.chains_imported == 1
    assert len(summary.errors) == 1


def test_full_round_trip_preserves_everything_together(fake_client_factory, tmp_path, monkeypatch):
    _save_a_history_entry(fake_client_factory, "A car problem.")
    model = _kinematics_model()
    chains_module.add_step(chains_module.create_chain("chain"), "p", model, "a")
    ws_export = Workspace({})
    ws_export.store("v0", 5.0, "manual entry", "m/s")

    bundle_json = export_bundle(workspace_entries=ws_export.entries)

    monkeypatch.setattr(history_module, "DB_PATH", tmp_path / "history_b.db")
    monkeypatch.setattr(chains_module, "DB_PATH", tmp_path / "chains_b.db")
    ws_import = Workspace({})
    summary = import_bundle(bundle_json, ws_import)

    assert summary.history_imported == 1
    assert summary.chains_imported == 1
    assert summary.workspace_imported == 1
    assert summary.errors == []
