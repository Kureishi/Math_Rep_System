import json

import modules.history as history_module
from modules.equation_engine import build_model
from modules.verifier import verify
from modules.solver import compute_steps


def test_save_load_delete_round_trip(tmp_path, monkeypatch, kinematics_json, fake_client_factory):
    # redirect the history DB to a temp file so tests never touch the real one
    monkeypatch.setattr(history_module, "DB_PATH", tmp_path / "test_history.db")

    model = build_model(json.loads(kinematics_json))
    client = fake_client_factory(final_answers={"a": 2.0})
    report = verify(model, client, "car problem")
    steps = compute_steps(model)

    entry_id = history_module.save("car problem", model, report, steps, [])
    assert entry_id is not None

    recent = history_module.list_recent()
    assert len(recent) == 1
    assert recent[0]["id"] == entry_id
    assert recent[0]["passed"] is True

    loaded = history_module.load(entry_id)
    assert loaded is not None
    p_text, l_model, l_report, l_steps, l_scenarios = loaded
    assert p_text == "car problem"
    assert l_model.solve_for == model.solve_for
    assert l_model.equations[0].sympy_eq == model.equations[0].sympy_eq
    assert l_report.passed == report.passed
    assert set(l_steps.keys()) == set(steps.keys())

    history_module.delete(entry_id)
    assert history_module.list_recent() == []


def test_load_nonexistent_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(history_module, "DB_PATH", tmp_path / "test_history2.db")
    assert history_module.load(99999) is None


def test_margin_ratio_survives_round_trip(tmp_path, monkeypatch, kinematics_json, fake_client_factory):
    monkeypatch.setattr(history_module, "DB_PATH", tmp_path / "test_history3.db")
    model = build_model(json.loads(kinematics_json))
    client = fake_client_factory(final_answers={"a": 2.0002})  # near-exact, nonzero margin_ratio
    report = verify(model, client, "x")
    steps = compute_steps(model)

    entry_id = history_module.save("x", model, report, steps, [])
    _, _, l_report, _, _ = history_module.load(entry_id)

    for orig, reloaded in zip(report.checks, l_report.checks):
        assert orig.margin_ratio == reloaded.margin_ratio
    history_module.delete(entry_id)
