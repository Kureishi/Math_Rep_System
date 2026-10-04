"""modules/db_util.py: leaving `with _connect() as conn:` commits (or rolls back) AND closes.

Every one of the app's four SQLite modules goes through it, so each is checked: the connection is really closed
afterwards, data written inside the block survives, and an exception inside the block rolls the write back
and still closes -- plus Python's own ResourceWarning for an unclosed connection, which is what this fixes,
turned into a failure."""
import gc
import sqlite3
import warnings

import pytest

import modules.chains as chains
import modules.history as history
import modules.settings_profiles as profiles
import modules.templates as templates
from modules.db_util import ClosingConnection

MODULES = [history, chains, templates, profiles]


@pytest.fixture(params=MODULES, ids=lambda m: m.__name__.split(".")[-1])
def db(request, tmp_path, monkeypatch):
    module = request.param
    monkeypatch.setattr(module, "DB_PATH", tmp_path / "test.db")
    return module


def _is_closed(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        return True
    return False


def test_leaving_the_block_closes_the_connection(db):
    with db._connect() as conn:
        assert isinstance(conn, ClosingConnection) and not _is_closed(conn)
    assert _is_closed(conn)


def test_data_written_inside_the_block_is_committed(db):
    with db._connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS probe (n INTEGER)")
        conn.execute("INSERT INTO probe VALUES (42)")
    with db._connect() as conn:
        assert conn.execute("SELECT n FROM probe").fetchall() == [(42,)]


def test_an_exception_inside_the_block_rolls_back_and_still_closes(db):
    with db._connect() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS probe (n INTEGER)")
    with pytest.raises(RuntimeError, match="boom"):
        with db._connect() as conn:
            conn.execute("INSERT INTO probe VALUES (7)")
            raise RuntimeError("boom")
    assert _is_closed(conn)
    with db._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM probe").fetchone() == (0,)      # the insert did not survive


def test_the_exception_is_not_swallowed(db):
    with pytest.raises(ValueError):
        with db._connect():
            raise ValueError("propagates")


def test_no_connection_is_left_to_the_garbage_collector(db):
    """On Python 3.13+ a connection dropped unclosed raises ResourceWarning; with the fix there is none."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        for _ in range(5):
            with db._connect() as conn:
                conn.execute("SELECT 1")
            del conn
        gc.collect()
