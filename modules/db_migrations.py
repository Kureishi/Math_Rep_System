"""
Lightweight, dependency-free schema-migration framework: a numbered,
ordered list of migration functions applied in sequence against a
`schema_version` table -- replacing the ad-hoc
`ALTER TABLE ... except sqlite3.OperationalError: pass` pattern
history.py used to rely on for every schema change.

Why this over that pattern: ALTER-TABLE-and-swallow-the-error works,
but it has no record of WHAT has been applied or in what order, no way
to express a migration that ISN'T a simple column addition (renaming a
column, splitting a table, backfilling a computed value from existing
rows), and no way to distinguish "genuinely already migrated" from
"some other, unrelated OperationalError" -- both look identical to a
bare except. A tracked version number fixes all three: the schema's
history is explicit and inspectable (SELECT version FROM
schema_version), migrations run in a defined order exactly once each,
and a migration failure surfaces as a real, specific error instead of
being silently absorbed.

Not a general-purpose migration TOOL (no down-migrations, no
autogeneration from model diffs, no CLI) -- just enough structure for a
handful of local SQLite files that evolve occasionally, matching the
scale of what this app actually needs.
"""
import sqlite3
from typing import Callable, Sequence

Migration = Callable[[sqlite3.Connection], None]


def current_version(conn: sqlite3.Connection) -> int:
    """Creates the schema_version bookkeeping table if it doesn't exist
    yet (a brand-new database, or one from before this framework
    existed) and returns the version currently recorded -- 0 for either
    case, so a pre-existing database (with columns already added via
    the OLD ad-hoc ALTER TABLE mechanism) and a genuinely fresh one both
    start migrating from the same place."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (0)")
        return 0
    return row[0]


def apply_migrations(conn: sqlite3.Connection, migrations: Sequence[Migration]) -> int:
    """Applies every migration in `migrations` (0-indexed; migrations[i]
    takes the schema from version i to version i+1) not yet applied to
    this connection's database, in order, updating schema_version after
    each so a later call (the next time this process, or a future one,
    connects) only runs whatever's new. Returns the final version."""
    version = current_version(conn)
    for i in range(version, len(migrations)):
        migrations[i](conn)
        conn.execute("UPDATE schema_version SET version = ?", (i + 1,))
        version = i + 1
    return version
