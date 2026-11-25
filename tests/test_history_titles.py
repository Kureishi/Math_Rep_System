"""History: naming entries (storage, migration, bundles, pickers, the sidebar's rename form) and the History list filling in right after a solve."""
import json
import sqlite3
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import modules.history as history
from tests.conftest import KINEMATICS_JSON, FakeClient
from tests.report_helpers import kinematics_ctx

APP = str(Path(__file__).parent.parent / "app.py")


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    return tmp_path / "history.db"


def seed(text="A car accelerates from 8 m/s to 20 m/s in 6 s."):
    ctx = kinematics_ctx()
    return history.save(text, ctx.model, ctx.report, ctx.steps_by_target, [])


# ------------------------------------------------------------------------------------------ storage

def test_a_new_entry_has_no_title_and_shows_its_domain(db):
    entry_id = seed()
    (entry,) = history.list_recent()
    assert entry["id"] == entry_id and entry["title"] is None
    assert history.display_title(entry) == entry["domain"] == "kinematics"


def test_renaming_stores_the_name_and_survives_a_reload(db):
    entry_id = seed()
    assert history.rename(entry_id, "Car on the motorway") is True
    assert history.list_recent()[0]["title"] == "Car on the motorway" and history.get_title(entry_id) == "Car on the motorway"
    assert history.display_title(history.list_recent()[0]) == "Car on the motorway"
    assert history.load(entry_id) is not None                      # the problem itself is untouched


@pytest.mark.parametrize("raw, stored", [
    ("  Car   on\tthe\nmotorway  ", "Car on the motorway"),
    ("x" * 200, "x" * history.MAX_TITLE_CHARS),
    ("", None), ("   \n ", None), (None, None),
])
def test_names_are_cleaned_and_an_empty_name_clears_it(db, raw, stored):
    entry_id = seed()
    history.rename(entry_id, "something")
    assert history.rename(entry_id, raw) is True
    assert history.get_title(entry_id) == stored


def test_renaming_an_entry_that_does_not_exist_says_so(db):
    seed()
    assert history.rename(9999, "x") is False and history.get_title(9999) is None


def test_names_with_quotes_and_unicode_are_stored_verbatim(db):
    entry_id = seed()
    history.rename(entry_id, "Résumé's \"a\" = 2 m/s² ✓; DROP TABLE problems;--")
    assert history.get_title(entry_id) == "Résumé's \"a\" = 2 m/s² ✓; DROP TABLE problems;--" and history.list_recent()


def test_only_the_named_entry_changes(db):
    a, b = seed("first"), seed("second")
    history.rename(a, "Mine")
    titles = {e["id"]: e["title"] for e in history.list_recent()}
    assert titles == {a: "Mine", b: None}


def test_a_database_made_before_titles_existed_is_upgraded_in_place(db, monkeypatch):
    """History is the person's own data: an existing database must keep working, with its entries, when the title column is added."""
    monkeypatch.setattr(history, "_MIGRATIONS", history._MIGRATIONS[:3])
    old_id = seed("an entry saved by the previous version")
    with sqlite3.connect(db) as conn:
        assert "title" not in [r[1] for r in conn.execute("PRAGMA table_info(problems)")]
    monkeypatch.undo()
    monkeypatch.setattr(history, "DB_PATH", db)
    (entry,) = history.list_recent()
    assert entry["id"] == old_id and entry["title"] is None and entry["problem_text"].startswith("an entry saved")
    assert history.rename(old_id, "Upgraded") and history.get_title(old_id) == "Upgraded"
    with sqlite3.connect(db) as conn:
        assert "title" in [r[1] for r in conn.execute("PRAGMA table_info(problems)")]


def test_pickers_show_the_name_when_there_is_one_and_the_problem_start_otherwise():
    plain = {"id": 7, "problem_text": "A car accelerates from 8 m/s to 20 m/s in 6 s. Find a.", "title": None}
    named = {**plain, "title": "Motorway car"}
    assert history.picker_label(plain, 20) == "7 · A car accelerates fr"           # the first 20 characters of the problem
    assert history.picker_label(named) == "7 · Motorway car"


def test_a_name_survives_a_project_bundle_round_trip_and_old_bundles_still_import(db, tmp_path, monkeypatch):
    from modules import project_bundle
    from modules.workspace import Workspace
    entry_id = seed()
    history.rename(entry_id, "Keep this name")
    bundle = project_bundle.export_bundle(workspace_entries={})
    assert json.loads(bundle)["history"][0]["title"] == "Keep this name"
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "fresh.db")
    summary = project_bundle.import_bundle(bundle, Workspace({}))
    assert history.list_recent()[0]["title"] == "Keep this name" and not summary.errors
    # a bundle exported before titles existed has no "title" key
    legacy = json.loads(bundle)
    for rec in legacy["history"]:
        rec.pop("title", None)
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "legacy.db")
    project_bundle.import_bundle(json.dumps(legacy), Workspace({}))
    assert history.list_recent()[0]["title"] is None


def test_compare_lists_entries_by_name(db):
    from ui import compare
    entry_id = seed()
    history.rename(entry_id, "Named for compare")
    assert any("Named for compare" in label for label in compare._history_options())


# ------------------------------------------------------------------------------------------ the sidebar's rename form

def _app():
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    return at


def _click(at, key):
    next(b for b in at.button if b.key == key).click()
    at.run()


def _sidebar_text(at):
    return " ".join(c.value for c in at.sidebar.caption)


def test_the_sidebar_lists_a_named_entry_under_its_name(db):
    entry_id = seed()
    history.rename(entry_id, "Motorway car")
    at = _app()
    assert not at.exception and "**Motorway car**" in _sidebar_text(at) and "**kinematics**" not in _sidebar_text(at)


def test_the_rename_form_saves_a_new_name(db):
    entry_id = seed()
    at = _app()
    assert "**kinematics**" in _sidebar_text(at)
    _click(at, f"renamehist_{entry_id}")
    field = at.sidebar.text_input(key=f"renameinput_{entry_id}")
    assert field.value == "" and field.placeholder == "kinematics"            # empty field, the default shown as the placeholder
    field.set_value("  Car on the motorway ")
    next(b for b in at.sidebar.button if b.label == "Save").click()
    at.run()
    assert not at.exception and "**Car on the motorway**" in _sidebar_text(at) and history.get_title(entry_id) == "Car on the motorway"
    assert not [t for t in at.sidebar.text_input if t.key == f"renameinput_{entry_id}"]      # the form closed again


def test_cancel_leaves_the_name_alone(db):
    entry_id = seed()
    history.rename(entry_id, "Original")
    at = _app()
    _click(at, f"renamehist_{entry_id}")
    assert at.sidebar.text_input(key=f"renameinput_{entry_id}").value == "Original"         # editing starts from the current name
    at.sidebar.text_input(key=f"renameinput_{entry_id}").set_value("Changed my mind")
    next(b for b in at.sidebar.button if b.label == "Cancel").click()
    at.run()
    assert history.get_title(entry_id) == "Original" and "**Original**" in _sidebar_text(at)


def test_saving_an_empty_name_goes_back_to_the_default(db):
    entry_id = seed()
    history.rename(entry_id, "Temporary")
    at = _app()
    _click(at, f"renamehist_{entry_id}")
    at.sidebar.text_input(key=f"renameinput_{entry_id}").set_value("")
    next(b for b in at.sidebar.button if b.label == "Save").click()
    at.run()
    assert history.get_title(entry_id) is None and "**kinematics**" in _sidebar_text(at)


def test_only_one_entry_is_in_edit_mode_at_a_time(db):
    a, b = seed("first"), seed("second")
    at = _app()
    _click(at, f"renamehist_{a}")
    _click(at, f"renamehist_{b}")
    editing = [t.key for t in at.sidebar.text_input if str(t.key).startswith("renameinput_")]
    assert editing == [f"renameinput_{b}"]


def test_a_database_error_while_renaming_is_a_notice_not_a_crash(db, monkeypatch):
    entry_id = seed()
    at = _app()
    _click(at, f"renamehist_{entry_id}")
    monkeypatch.setattr(history, "rename", lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("database is locked")))
    at.sidebar.text_input(key=f"renameinput_{entry_id}").set_value("x")
    next(b for b in at.sidebar.button if b.label == "Save").click()
    at.run()
    assert not at.exception and any("Could not rename" in w.value for w in at.sidebar.warning)


# ------------------------------------------------------------------------------------------ the History list fills in right after a solve

def test_the_history_list_shows_a_problem_the_moment_it_is_solved(db, monkeypatch):
    """The sidebar is drawn before the solve saves anything: it used to say 'No solved problems yet' until the next interaction."""
    from modules.llm_client import LMStudioClient
    good = FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0})
    monkeypatch.setattr(LMStudioClient, "is_available", lambda self: (True, "ok"))
    monkeypatch.setattr(LMStudioClient, "list_models", lambda self: ["m"])
    monkeypatch.setattr(LMStudioClient, "chat", lambda self, *a, **k: good.chat(*a, **k))
    at = _app()
    assert "No solved problems yet" in _sidebar_text(at)
    at.text_area[0].set_value("A car accelerates uniformly from 8 m/s to 20 m/s over 6 seconds. What is its acceleration?")
    next(b for b in at.button if "Represent" in b.label).click()
    at.run()
    assert not at.exception, [str(e.value)[:200] for e in at.exception]
    side = _sidebar_text(at)
    assert "No solved problems yet" not in side and "**kinematics**" in side and "A car accelerates" in side
    assert at.session_state["model"] is not None and len(history.list_recent()) == 1
    assert any("Domain" in h.value for h in at.subheader)                   # the result is on screen, as before


def test_the_solved_status_box_and_the_result_survive_the_redraw(db, monkeypatch):
    from modules.llm_client import LMStudioClient
    good = FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0})
    monkeypatch.setattr(LMStudioClient, "is_available", lambda self: (True, "ok"))
    monkeypatch.setattr(LMStudioClient, "list_models", lambda self: ["m"])
    monkeypatch.setattr(LMStudioClient, "chat", lambda self, *a, **k: good.chat(*a, **k))
    at = _app()
    at.text_area[0].set_value("A car accelerates uniformly from 8 m/s to 20 m/s over 6 seconds. What is its acceleration?")
    next(b for b in at.button if "Represent" in b.label).click()
    at.run()
    status = [s for s in at.status if "Solved" in s.label]
    assert status and status[0].state == "complete"
    assert "_after_solve" not in at.session_state                           # shown once, then consumed
    at.run()                                                                # any later interaction: the result stays, the one-off status box does not
    assert at.session_state["model"] is not None and not [s for s in at.status if "Solved" in s.label]


def test_a_failed_history_save_still_shows_the_result_without_a_redraw_loop(db, monkeypatch):
    from modules.llm_client import LMStudioClient
    good = FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0})
    monkeypatch.setattr(LMStudioClient, "is_available", lambda self: (True, "ok"))
    monkeypatch.setattr(LMStudioClient, "list_models", lambda self: ["m"])
    monkeypatch.setattr(LMStudioClient, "chat", lambda self, *a, **k: good.chat(*a, **k))
    monkeypatch.setattr(history, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("database is locked")))
    at = _app()
    at.text_area[0].set_value("A car accelerates uniformly from 8 m/s to 20 m/s over 6 seconds. What is its acceleration?")
    next(b for b in at.button if "Represent" in b.label).click()
    at.run()
    assert not at.exception and at.session_state["model"] is not None and any("could not be saved to history" in w.value for w in at.warning)
