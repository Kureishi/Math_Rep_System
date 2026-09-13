"""
Templates: named, savable/loadable presets of a mode's INPUT fields
(a tensor metric and its coordinates, a custom curve-fit model and its
parameter names, a PDE's boundary conditions, ...) -- the input-side
counterpart to settings_profiles.py, which covers the verification
TUNING knobs rather than the problem itself.

The recurring need this solves: Quick Start (see app.py) covers a
handful of BUILT-IN example problems, but someone doing real work
develops their OWN frequently-reused starting points -- their
department's standard metric convention, the custom model they fit
half their datasets to, a boundary-condition set they check every
assignment against. Retyping those by hand every session is exactly
the kind of repetitive friction a "save as template" button removes.

Storage is deliberately a single flexible JSON payload column, not one
column per field the way settings_profiles.py does it: a template's
shape is different per category (a tensor metric's fields have nothing
in common with a curve-fit model's), so a fixed schema would mean
either a giant mostly-NULL table or a separate table per category --
JSON keeps this as one small, simple table regardless of how many
categories of template this grows to cover later.
"""
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from modules.db_migrations import apply_migrations

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "templates.db"


@dataclass
class Template:
    id: int
    name: str
    category: str
    payload: dict
    created_at: str


def _migration_001_initial_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(name, category)
        )
    """)


_MIGRATIONS = [_migration_001_initial_schema]


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    apply_migrations(conn, _MIGRATIONS)
    conn.commit()
    return conn


def save_template(name: str, category: str, payload: dict) -> int:
    """Saves (or overwrites, if the same name+category already exists)
    a named template. `payload` is any JSON-serializable dict of field
    values -- whatever the calling UI's widgets need to restore
    themselves; this module doesn't interpret it."""
    name = name.strip()
    if not name:
        raise ValueError("Template name can't be empty.")
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO templates (name, category, payload, created_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(name, category) DO UPDATE SET payload=excluded.payload, "
            "created_at=excluded.created_at",
            (name, category, json.dumps(payload), datetime.now().isoformat(timespec="seconds")),
        )
        row = conn.execute("SELECT id FROM templates WHERE name = ? AND category = ?",
                            (name, category)).fetchone()
        assert row is not None  # just inserted/updated above, must exist
        return row[0]


def list_templates(category: str | None = None) -> list[Template]:
    """Every saved template, optionally filtered to one category,
    most-recently-saved first."""
    with _connect() as conn:
        if category is None:
            rows = conn.execute(
                "SELECT id, name, category, payload, created_at FROM templates "
                "ORDER BY created_at DESC").fetchall()
        else:
            rows = conn.execute(
                "SELECT id, name, category, payload, created_at FROM templates "
                "WHERE category = ? ORDER BY created_at DESC", (category,)).fetchall()
    return [Template(id=r[0], name=r[1], category=r[2], payload=json.loads(r[3]), created_at=r[4])
            for r in rows]


def load_template(template_id: int) -> Template | None:
    with _connect() as conn:
        row = conn.execute("SELECT id, name, category, payload, created_at FROM templates "
                            "WHERE id = ?", (template_id,)).fetchone()
    if row is None:
        return None
    return Template(id=row[0], name=row[1], category=row[2], payload=json.loads(row[3]), created_at=row[4])


def delete_template(template_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM templates WHERE id = ?", (template_id,))
