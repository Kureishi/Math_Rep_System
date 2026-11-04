"""One lean end-to-end check of the Recall page against a real (temporary) history."""
import json

import pytest

from modules import history, recall
from modules.equation_engine import build_model
from modules.solve_compare import deterministic_report
from tests.conftest import KINEMATICS_TWO_TARGET_JSON
from tests.test_ui_smoke import _app

MODE = "🔎 Have I solved this?"
CAR = "A car accelerates from 8 m/s to 20 m/s in 6 seconds. How far does it travel?"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    monkeypatch.setattr(recall, "DB_PATH", tmp_path / "embeddings.db")
    model = build_model(json.loads(KINEMATICS_TWO_TARGET_JSON))
    history.save(CAR, model, deterministic_report(model), {}, [])


def test_the_page_finds_a_past_problem_with_different_numbers_and_can_open_it():
    at = _app()
    at.run(timeout=120)
    at.sidebar.radio[0].set_value(MODE)
    at.run(timeout=120)
    assert not at.exception
    next(t for t in at.text_area if t.key == "recall_query").set_value(CAR.replace("8", "12").replace("20", "30").replace("6", "9"))
    next(b for b in at.button if b.key == "recall_go").click()
    at.run(timeout=120)
    assert not at.exception and any("solved this before" in s.value for s in at.success)
    next(b for b in at.button if b.key.startswith("recall_open_")).click()
    at.run(timeout=120)
    assert not at.exception and at.session_state["problem_text"] == CAR
    assert at.session_state["app_mode"] == "📝 Word problem solver"           # the pending switch was applied on the rerun


def test_a_new_problem_is_reported_as_new():
    at = _app()
    at.run(timeout=120)
    at.sidebar.radio[0].set_value(MODE)
    at.run(timeout=120)
    next(t for t in at.text_area if t.key == "recall_query").set_value("Estimate the half-life of a radioactive sample")
    next(b for b in at.button if b.key == "recall_go").click()
    at.run(timeout=120)
    assert not at.exception and any("looks new" in i.value for i in at.info)
