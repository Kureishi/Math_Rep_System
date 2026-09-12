"""Tests for modules/db_migrations.py -- the versioned schema-migration
framework."""
import sqlite3

from modules.db_migrations import current_version, apply_migrations


def _fresh_conn():
    return sqlite3.connect(":memory:")


def test_current_version_starts_at_zero_for_fresh_database():
    conn = _fresh_conn()
    assert current_version(conn) == 0


def test_apply_migrations_runs_each_migration_exactly_once():
    conn = _fresh_conn()
    calls = []
    migrations = [lambda c: calls.append(1), lambda c: calls.append(2), lambda c: calls.append(3)]
    version = apply_migrations(conn, migrations)
    assert version == 3
    assert calls == [1, 2, 3]


def test_apply_migrations_is_idempotent_across_separate_calls():
    """The core guarantee: calling apply_migrations again (e.g. the next
    time the app connects) with the SAME migration list must not re-run
    anything already applied -- this is what makes it safe to call on
    every connection, the same way the old ALTER-TABLE-and-swallow
    pattern was."""
    conn = _fresh_conn()
    calls = []
    migrations = [lambda c: calls.append("a"), lambda c: calls.append("b")]
    apply_migrations(conn, migrations)
    apply_migrations(conn, migrations)  # second call, same migrations
    assert calls == ["a", "b"]  # each ran exactly once, not twice


def test_apply_migrations_only_runs_new_ones_after_a_list_grows():
    """The realistic upgrade scenario: a database migrated under an
    OLDER, shorter migrations list, then the app adds a new migration
    -- only the new one should run, not the whole list from scratch."""
    conn = _fresh_conn()
    calls = []
    first_round = [lambda c: calls.append("a"), lambda c: calls.append("b")]
    apply_migrations(conn, first_round)

    second_round = first_round + [lambda c: calls.append("c")]
    version = apply_migrations(conn, second_round)
    assert version == 3
    assert calls == ["a", "b", "c"]


def test_apply_migrations_actually_changes_the_schema():
    conn = _fresh_conn()
    migrations = [lambda c: c.execute("CREATE TABLE widgets (id INTEGER)")]
    apply_migrations(conn, migrations)
    conn.execute("INSERT INTO widgets (id) VALUES (1)")  # doesn't raise -- table exists
    row = conn.execute("SELECT id FROM widgets").fetchone()
    assert row == (1,)


def test_current_version_persists_across_reconnection_to_the_same_file(tmp_path):
    db_path = tmp_path / "test_migrations.db"
    conn1 = sqlite3.connect(db_path)
    apply_migrations(conn1, [lambda c: c.execute("CREATE TABLE t (x INTEGER)")])
    conn1.commit()
    conn1.close()

    conn2 = sqlite3.connect(db_path)
    assert current_version(conn2) == 1
