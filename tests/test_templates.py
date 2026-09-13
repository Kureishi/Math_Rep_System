"""Tests for modules/templates.py -- named input-preset storage."""
import pytest

from modules.templates import save_template, list_templates, load_template, delete_template


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    import modules.templates as templates_module
    monkeypatch.setattr(templates_module, "DB_PATH", tmp_path / "test_templates.db")
    return templates_module


def test_save_and_load_roundtrip(isolated_db):
    tid = save_template("My metric", "tensor_metric", {"coords": "x, y", "metric": [["1", "0"], ["0", "1"]]})
    loaded = load_template(tid)
    assert loaded.name == "My metric"
    assert loaded.category == "tensor_metric"
    assert loaded.payload == {"coords": "x, y", "metric": [["1", "0"], ["0", "1"]]}


def test_list_templates_filters_by_category(isolated_db):
    save_template("A", "tensor_metric", {"x": 1})
    save_template("B", "curve_fit_model", {"y": 2})
    assert len(list_templates()) == 2
    assert len(list_templates("tensor_metric")) == 1
    assert list_templates("tensor_metric")[0].name == "A"


def test_saving_same_name_and_category_overwrites(isolated_db):
    tid1 = save_template("Same", "tensor_metric", {"v": 1})
    tid2 = save_template("Same", "tensor_metric", {"v": 2})
    assert tid1 == tid2
    assert load_template(tid1).payload == {"v": 2}
    assert len(list_templates("tensor_metric")) == 1


def test_same_name_different_category_is_a_separate_template(isolated_db):
    tid1 = save_template("Same", "tensor_metric", {"v": 1})
    tid2 = save_template("Same", "curve_fit_model", {"v": 2})
    assert tid1 != tid2
    assert len(list_templates()) == 2


def test_delete_template(isolated_db):
    tid = save_template("Gone soon", "tensor_metric", {"v": 1})
    delete_template(tid)
    assert load_template(tid) is None
    assert list_templates() == []


def test_empty_name_rejected(isolated_db):
    with pytest.raises(ValueError):
        save_template("   ", "tensor_metric", {"v": 1})


def test_load_nonexistent_returns_none(isolated_db):
    assert load_template(99999) is None
