"""
Persistent history of solved problems, backed by a local SQLite file.

Deliberately stores the raw extraction JSON (model.raw_json) rather than
serialized SymPy objects -- ProblemModel is rebuilt from that JSON via
build_model() on load, which is cheap, deterministic, and avoids ever
needing to pickle SymPy expressions. Verification checks, solution steps
(including LLM narration), and scenarios are stored as plain JSON-safe
dicts so reloading a past problem never needs another LLM call.
"""
import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from modules.equation_engine import ProblemModel, build_model
from modules.verifier import VerificationReport, CheckResult
from modules.solver import SolutionStep
from modules.similarity import problem_shape, find_similar_shapes
from modules.concept_index import concept_tags_for_model
from modules.db_migrations import apply_migrations

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.db"

# Keeps history.db from growing unbounded over months of use, and caps
# how much a single query (list_recent, find_similar's full-table scan)
# has to work through -- enforced on every save() by pruning the oldest
# rows beyond this count, not by refusing new saves once full.
MAX_HISTORY_RECORDS = 100


def _migration_001_initial_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            problem_text TEXT NOT NULL,
            domain TEXT,
            passed INTEGER,
            payload TEXT NOT NULL
        )
    """)
    # personalized error-pattern tracking: persists grading.py's own
    # formula/arithmetic classification per "grade my work" submission,
    # separate from the `problems` table above (a submission may or may
    # not correspond to a problem that's itself been saved to history).
    # See classify_mistake() in grading.py and summarize_error_patterns()
    # below.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS grading_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            target TEXT NOT NULL,
            domain TEXT,
            category TEXT NOT NULL,
            subtype TEXT,
            detail TEXT,
            equation_shapes TEXT
        )
    """)


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, coltype: str) -> None:
    """Used only by the two migrations below that replicate schema
    changes which, for anyone with a history.db from before this
    migration framework existed, may ALREADY have been applied via the
    old ad-hoc `ALTER TABLE ... except OperationalError: pass` pattern
    -- so schema_version starting fresh at 0 on such a database must
    not choke re-adding a column that's already there. Every migration
    added AFTER this framework's introduction can use a plain
    `conn.execute("ALTER TABLE ...")` directly instead, since
    schema_version now genuinely tracks what has and hasn't run."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
    except sqlite3.OperationalError:
        pass


def _migration_002_add_equation_shapes(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "problems", "equation_shapes", "TEXT")


def _migration_003_add_concept_tags(conn: sqlite3.Connection) -> None:
    _add_column_if_missing(conn, "problems", "concept_tags", "TEXT")


# Ordered, append-only: migrations[i] takes the schema from version i to
# i+1. To change the schema going forward, ADD a new migration function
# to the end of this list -- never edit an already-shipped one (a
# database that already applied it would silently skip the new
# behavior, since apply_migrations only runs what schema_version says
# hasn't happened yet).
_MIGRATIONS = [_migration_001_initial_schema, _migration_002_add_equation_shapes,
               _migration_003_add_concept_tags]


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # WAL (write-ahead log) instead of the default rollback-journal mode:
    # a crash or kill mid-write is much less likely to leave the file in
    # a bad state, and it tolerates a second reader/writer (e.g. two
    # browser tabs open on the same session) without immediately hitting
    # "database is locked". synchronous=NORMAL is the safe pairing with
    # WAL (still durable against an OS crash, just not against a full
    # power loss mid-write, an acceptable tradeoff for a local personal
    # tool). busy_timeout makes SQLite retry for a few seconds instead of
    # raising "database is locked" immediately if a brief write overlaps
    # from another connection, rather than surfacing a raw error to the UI.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    apply_migrations(conn, _MIGRATIONS)
    # explicit commit here, not left to the caller: schema_version's own
    # bookkeeping (INSERT/UPDATE, both DML) opens an implicit transaction
    # under sqlite3's default isolation_level, and that transaction then
    # covers every migration's DDL statements too -- a caller that uses
    # `with _connect() as conn:` gets a commit for free on clean exit,
    # but one that calls `_connect().close()` directly does NOT, and
    # would silently roll back the schema setup on close(). Committing
    # here makes schema setup durable unconditionally, regardless of how
    # the connection is used afterward.
    conn.commit()
    return conn


def _prune_old_records(conn: sqlite3.Connection):
    """Keeps only the MAX_HISTORY_RECORDS most recent rows (by id, which
    is monotonically increasing) -- called after every save() so the
    table never grows past the cap, rather than needing a separate
    maintenance step someone has to remember to run."""
    conn.execute(
        "DELETE FROM problems WHERE id NOT IN "
        "(SELECT id FROM problems ORDER BY id DESC LIMIT ?)",
        (MAX_HISTORY_RECORDS,),
    )


def save(problem_text: str, model: ProblemModel, report: VerificationReport,
         steps_by_target: dict[str, list[SolutionStep]], scenarios: list[dict]) -> int:
    payload = {
        "raw_json": model.raw_json,
        "verification": {
            "passed": report.passed,
            "checks": [asdict(c) for c in report.checks],
            "sympy_numeric_answers": report.sympy_numeric_answers,
            "llm_independent_answers": report.llm_independent_answers,
        },
        "steps_by_target": {
            target: [asdict(s) for s in steps] for target, steps in steps_by_target.items()
        },
        "scenarios": scenarios,
    }
    shapes_json = json.dumps(sorted(problem_shape(model)))
    try:
        concepts_json = json.dumps(concept_tags_for_model(model))
    except Exception:  # noqa: BLE001
        # concept tagging is a provenance nicety, not core to saving a
        # solved problem -- a failure here (e.g. an unexpected shape in
        # named_formulas' matcher) shouldn't block the save itself
        concepts_json = json.dumps([])
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO problems (timestamp, problem_text, domain, passed, payload, equation_shapes, "
            "concept_tags) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), problem_text, model.problem_domain,
             int(report.passed), json.dumps(payload), shapes_json, concepts_json),
        )
        new_id = cur.lastrowid
        assert new_id is not None, "INSERT did not produce a rowid -- should be unreachable"
        _prune_old_records(conn)
        return new_id


def list_recent(limit: int = 25) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, timestamp, problem_text, domain, passed FROM problems "
            "ORDER BY id DESC LIMIT ?", (limit,),
        ).fetchall()
    return [
        {"id": r[0], "timestamp": r[1], "problem_text": r[2], "domain": r[3], "passed": bool(r[4])}
        for r in rows
    ]


def load(entry_id: int):
    """Returns (problem_text, model, report, steps_by_target, scenarios), fully
    reconstructed with no LLM calls, or None if the id doesn't exist."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT problem_text, payload FROM problems WHERE id = ?", (entry_id,)
        ).fetchone()
    if row is None:
        return None

    problem_text, payload_json = row
    payload = json.loads(payload_json)

    model = build_model(payload["raw_json"])

    v = payload["verification"]
    report = VerificationReport(
        checks=[CheckResult(**c) for c in v["checks"]],
        sympy_numeric_answers=v["sympy_numeric_answers"],
        llm_independent_answers=v["llm_independent_answers"],
        passed=v["passed"],
    )

    steps_by_target = {
        target: [SolutionStep(**s) for s in steps]
        for target, steps in payload["steps_by_target"].items()
    }
    scenarios = payload["scenarios"]

    return problem_text, model, report, steps_by_target, scenarios


def delete(entry_id: int):
    with _connect() as conn:
        conn.execute("DELETE FROM problems WHERE id = ?", (entry_id,))


def find_similar(model: ProblemModel, exclude_id: int | None = None,
                  limit: int = 5, min_similarity: float = 0.3) -> list[dict]:
    """Ranks every past problem with a stored equation_shapes fingerprint
    by structural similarity (see similarity.py) to `model`, returning
    the top matches above min_similarity. `exclude_id` skips the current
    problem itself if it's already been saved to history (so a
    freshly-solved problem doesn't just "match itself" perfectly).
    Rows saved before equation_shapes existed (NULL or missing) are
    silently skipped -- there's nothing to compare them against; they
    simply won't surface as a match, not an error."""
    target_shape = problem_shape(model)
    if not target_shape:
        return []

    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, timestamp, problem_text, domain, equation_shapes FROM problems "
            "WHERE equation_shapes IS NOT NULL"
        ).fetchall()

    candidates = []
    row_by_id = {}
    for rid, ts, text, domain, shapes_json in rows:
        if exclude_id is not None and rid == exclude_id:
            continue
        try:
            shape = frozenset(json.loads(shapes_json))
        except (TypeError, ValueError):
            continue
        if not shape:
            continue
        candidates.append((rid, shape))
        row_by_id[rid] = {"id": rid, "timestamp": ts, "problem_text": text, "domain": domain}

    ranked = find_similar_shapes(target_shape, candidates, limit=limit, min_similarity=min_similarity)
    return [{**row_by_id[rid], "similarity": score} for rid, score in ranked]


def list_concepts(limit: int = 100) -> list[dict]:
    """Every distinct concept tag across all of history, with how many
    saved problems carry it, most-common first -- the browsable index
    a research-journal view starts from ("what has this session/history
    actually covered"), as opposed to find_similar's one-problem-at-a-
    time structural lookup. Rows saved before concept_tags existed (NULL
    or missing, same as equation_shapes) simply don't contribute any
    tags -- not an error, just no signal to extract."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT concept_tags FROM problems WHERE concept_tags IS NOT NULL"
        ).fetchall()
    counts: dict[str, int] = {}
    for (tags_json,) in rows:
        try:
            tags = json.loads(tags_json)
        except (TypeError, ValueError):
            continue
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"concept": tag, "count": count} for tag, count in ranked[:limit]]


def problems_for_concept(concept: str, limit: int = 50) -> list[dict]:
    """Every saved problem tagged with `concept` (exact match against
    one of its concept_tags entries), most recent first -- the other
    half of list_concepts(): having found a concept worth pulling
    together, this is what actually retrieves its problems for a
    research journal (see research_journal.py) or just browsing."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, timestamp, problem_text, domain, passed, concept_tags FROM problems "
            "WHERE concept_tags IS NOT NULL ORDER BY id DESC"
        ).fetchall()
    matches = []
    for rid, ts, text, domain, passed, tags_json in rows:
        try:
            tags = json.loads(tags_json)
        except (TypeError, ValueError):
            continue
        if concept in tags:
            matches.append({"id": rid, "timestamp": ts, "problem_text": text, "domain": domain,
                              "passed": bool(passed), "concept_tags": tags})
        if len(matches) >= limit:
            break
    return matches


# ---------------------------------------------------------------- grading / error-pattern tracking

def _prune_old_grading_records(conn: sqlite3.Connection):
    """Same cap-and-prune approach as _prune_old_records, applied to
    grading_records so it doesn't grow unbounded either."""
    conn.execute(
        "DELETE FROM grading_records WHERE id NOT IN "
        "(SELECT id FROM grading_records ORDER BY id DESC LIMIT ?)",
        (MAX_HISTORY_RECORDS,),
    )


def record_grading(target: str, domain: str | None, category: str, subtype: str | None,
                    detail: str, equation_shapes: frozenset[str] | None = None) -> int:
    """Persists one grading.classify_mistake() result. `equation_shapes`
    (see similarity.py) is stored so a future feature could match
    patterns to structurally-similar problems, not just to a domain
    label -- optional since not every caller has a built ProblemModel
    handy."""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO grading_records (timestamp, target, domain, category, subtype, detail, "
            "equation_shapes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), target, domain, category, subtype, detail,
             json.dumps(sorted(equation_shapes)) if equation_shapes else None),
        )
        new_id = cur.lastrowid
        assert new_id is not None, "INSERT did not produce a rowid -- should be unreachable"
        _prune_old_grading_records(conn)
        return new_id


def list_recent_grading(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, timestamp, target, domain, category, subtype, detail FROM grading_records "
            "ORDER BY id DESC LIMIT ?", (limit,),
        ).fetchall()
    return [
        {"id": r[0], "timestamp": r[1], "target": r[2], "domain": r[3],
         "category": r[4], "subtype": r[5], "detail": r[6]}
        for r in rows
    ]


_PATTERN_THRESHOLD = 3  # how many times a (category, subtype) pair has to recur within the
                          # lookback window before it's called an actual "pattern" worth acting
                          # on, rather than one-off noise


def _pattern_message(category: str, subtype: str | None, count: int, days: int) -> str:
    window = "this week" if days <= 7 else f"in the last {days} days"
    if category == "formula":
        return f"You've picked the wrong starting formula {count} times {window}."
    subtype_phrase = {
        "sign_error": "a sign error",
        "subtraction": "a subtraction-step error",
        "multiplication": "a multiplication-step error",
        "division": "a division-step error",
        "addition": "an addition-step error",
    }.get(subtype or "", "an arithmetic error")
    return f"You've made {subtype_phrase} {count} times {window}."


def summarize_error_patterns(days: int = 7, min_count: int = _PATTERN_THRESHOLD) -> list[dict]:
    """Groups recent grading_records by (category, subtype) within the
    last `days` days and returns those that recurred at least
    `min_count` times, most-frequent first -- each as {"category",
    "subtype", "count", "message"}. Only "formula"/"arithmetic"
    (i.e. actual mistakes) are ever counted -- a "pattern" is about
    what keeps going wrong, so "correct"/"unverified" submissions never
    contribute to one."""
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with _connect() as conn:
        rows = conn.execute(
            "SELECT category, subtype FROM grading_records "
            "WHERE timestamp >= ? AND category IN ('formula', 'arithmetic')",
            (cutoff,),
        ).fetchall()

    counts: dict[tuple[str, str | None], int] = {}
    for category, subtype in rows:
        key = (category, subtype)
        counts[key] = counts.get(key, 0) + 1

    patterns = [
        {"category": category, "subtype": subtype, "count": count,
         "message": _pattern_message(category, subtype, count, days)}
        for (category, subtype), count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        if count >= min_count
    ]
    return patterns
