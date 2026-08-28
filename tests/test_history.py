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


def _mk_model(vars_, eqs, sf, domain="x"):
    return build_model({"problem_domain": domain, "problem_type": "algebraic",
                         "variables": vars_, "equations": eqs, "solve_for": sf, "assumptions": []})


def test_find_similar_matches_structurally_similar_problem(tmp_path, monkeypatch, fake_client_factory):
    monkeypatch.setattr(history_module, "DB_PATH", tmp_path / "similarity_test.db")
    client = fake_client_factory(final_answers={"a": 2.0})

    m1 = _mk_model(
        [{"symbol": "a", "meaning": "a", "known_value": None, "unit": None},
         {"symbol": "v_f", "meaning": "v_f", "known_value": "20", "unit": None},
         {"symbol": "v_i", "meaning": "v_i", "known_value": "8", "unit": None},
         {"symbol": "t", "meaning": "t", "known_value": "6", "unit": None}],
        [{"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f-v_i)/t)", "derivation": ""}],
        ["a"], domain="car kinematics",
    )
    report1 = verify(m1, client, "A car problem")
    id1 = history_module.save("A car problem", m1, report1, compute_steps(m1), [])

    m2 = _mk_model(
        [{"symbol": "r", "meaning": "r", "known_value": None, "unit": None},
         {"symbol": "p", "meaning": "p", "known_value": "100", "unit": None},
         {"symbol": "q", "meaning": "q", "known_value": "40", "unit": None},
         {"symbol": "s", "meaning": "s", "known_value": "5", "unit": None}],
        [{"name": "rate", "kind": "equation", "expression": "Eq(r, (p-q)/s)", "derivation": ""}],
        ["r"], domain="chemistry rate",
    )
    report2 = verify(m2, client, "A chemistry problem")
    id2 = history_module.save("A chemistry problem", m2, report2, compute_steps(m2), [])

    m3 = _mk_model(
        [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None}],
        [{"name": "eq", "kind": "equation", "expression": "Eq(x**2, 25)", "derivation": ""}],
        ["x"], domain="algebra",
    )
    report3 = verify(m3, client, "An unrelated problem")
    history_module.save("An unrelated problem", m3, report3, compute_steps(m3), [])

    similar = history_module.find_similar(m1, exclude_id=id1)
    assert len(similar) == 1
    assert similar[0]["id"] == id2
    assert similar[0]["similarity"] == 1.0


def test_find_similar_excludes_self(tmp_path, monkeypatch, kinematics_json, fake_client_factory):
    monkeypatch.setattr(history_module, "DB_PATH", tmp_path / "self_exclude_test.db")
    model = build_model(json.loads(kinematics_json))
    client = fake_client_factory(final_answers={"a": 2.0})
    report = verify(model, client, "x")
    entry_id = history_module.save("x", model, report, compute_steps(model), [])

    similar = history_module.find_similar(model, exclude_id=entry_id)
    assert similar == []


def test_find_similar_returns_empty_for_optimization_only_problem(tmp_path, monkeypatch, fake_client_factory):
    monkeypatch.setattr(history_module, "DB_PATH", tmp_path / "opt_only_test.db")
    m = build_model({
        "problem_domain": "optimization", "problem_type": "algebraic",
        "variables": [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None}],
        "equations": [],
        "objective": {"expression": "-2*x**2+120*x-800", "direction": "maximize", "optimize_over": ["x"]},
        "solve_for": ["x"], "assumptions": [],
    })
    assert history_module.find_similar(m) == []


def test_migration_adds_equation_shapes_column_to_pre_existing_db(tmp_path, monkeypatch):
    """A DB created before equation_shapes existed (K's own local
    history.db, in practice) must not break -- ALTER TABLE ADD COLUMN
    should migrate it in place, and old rows should just have a NULL
    equation_shapes rather than causing an error."""
    import sqlite3
    db_path = tmp_path / "pre_migration.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            problem_text TEXT NOT NULL,
            domain TEXT,
            passed INTEGER,
            payload TEXT NOT NULL
        )
    """)
    conn.execute("INSERT INTO problems (timestamp, problem_text, domain, passed, payload) "
                 "VALUES ('2020-01-01', 'old problem', 'x', 1, '{}')")
    conn.commit()
    conn.close()

    monkeypatch.setattr(history_module, "DB_PATH", db_path)
    # should not raise -- migration runs transparently inside _connect()
    recent = history_module.list_recent()
    assert len(recent) == 1
    assert recent[0]["problem_text"] == "old problem"

    # calling _connect() again (as any subsequent operation would) should
    # also not raise, even though the column now already exists
    history_module._connect()


def test_find_similar_skips_rows_with_null_equation_shapes(tmp_path, monkeypatch, fake_client_factory):
    import sqlite3
    db_path = tmp_path / "null_shapes_test.db"
    monkeypatch.setattr(history_module, "DB_PATH", db_path)
    # trigger table creation + migration
    history_module._connect().close()
    with sqlite3.connect(db_path) as conn:
        conn.execute("INSERT INTO problems (timestamp, problem_text, domain, passed, payload, equation_shapes) "
                     "VALUES ('2020-01-01', 'legacy problem', 'x', 1, '{}', NULL)")

    model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "a", "meaning": "a", "known_value": None, "unit": None},
            {"symbol": "v_f", "meaning": "v_f", "known_value": "20", "unit": None},
            {"symbol": "v_i", "meaning": "v_i", "known_value": "8", "unit": None},
            {"symbol": "t", "meaning": "t", "known_value": "6", "unit": None},
        ],
        "equations": [{"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f-v_i)/t)", "derivation": ""}],
        "solve_for": ["a"], "assumptions": [],
    })
    # should not raise on the NULL-shape legacy row -- just finds nothing
    assert history_module.find_similar(model) == []
