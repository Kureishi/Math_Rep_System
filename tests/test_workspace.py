from modules.workspace import Workspace


def test_store_and_retrieve():
    ws = Workspace({})
    ws.store("a", 2.0, source="test problem", unit="m/s^2")
    assert "a" in ws.entries
    assert ws.entries["a"].value == 2.0


def test_rename_success():
    ws = Workspace({})
    ws.store("d", 84.0, source="x", unit="m")
    ok, err = ws.rename("d", "distance")
    assert ok
    assert err == ""
    assert "distance" in ws.entries
    assert "d" not in ws.entries


def test_rename_to_invalid_identifier_rejected():
    ws = Workspace({})
    ws.store("a", 2.0, source="x", unit=None)
    ok, err = ws.rename("a", "2bad")
    assert not ok
    assert "a" in ws.entries  # unchanged


def test_rename_to_existing_name_rejected():
    ws = Workspace({})
    ws.store("a", 2.0, source="x", unit=None)
    ws.store("b", 3.0, source="x", unit=None)
    ok, err = ws.rename("a", "b")
    assert not ok
    assert ws.entries["a"].value == 2.0  # unchanged
    assert ws.entries["b"].value == 3.0


def test_rename_nonexistent_rejected():
    ws = Workspace({})
    ok, err = ws.rename("zzz", "foo")
    assert not ok


def test_rename_to_same_name_is_noop_success():
    ws = Workspace({})
    ws.store("a", 2.0, source="x", unit=None)
    ok, err = ws.rename("a", "a")
    assert ok


def test_remove():
    ws = Workspace({})
    ws.store("a", 2.0, source="x", unit=None)
    ws.remove("a")
    assert "a" not in ws.entries


def test_context_string_includes_all_entries():
    ws = Workspace({})
    ws.store("a", 2.0, source="problem 1", unit="m/s^2")
    ws.store("d", 84.0, source="problem 1", unit="m")
    ctx = ws.as_context_string()
    assert "a = 2" in ctx
    assert "d = 84" in ctx


def test_context_string_none_when_empty():
    ws = Workspace({})
    assert ws.as_context_string() is None
