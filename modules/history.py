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
from datetime import datetime
from pathlib import Path

from modules.equation_engine import ProblemModel, build_model
from modules.verifier import VerificationReport, CheckResult
from modules.solver import SolutionStep
from modules.similarity import problem_shape, find_similar_shapes

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "history.db"

# Keeps history.db from growing unbounded over months of use, and caps
# how much a single query (list_recent, find_similar's full-table scan)
# has to work through -- enforced on every save() by pruning the oldest
# rows beyond this count, not by refusing new saves once full.
MAX_HISTORY_RECORDS = 100


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
    # equation_shapes was added after the table above already existed in
    # the wild (K's own local history.db predates it) -- ALTER TABLE ADD
    # COLUMN is the safe migration path; SQLite has no "ADD COLUMN IF NOT
    # EXISTS", so the duplicate-column error on an already-migrated DB is
    # simply swallowed rather than checked for up front.
    try:
        conn.execute("ALTER TABLE problems ADD COLUMN equation_shapes TEXT")
    except sqlite3.OperationalError:
        pass
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
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO problems (timestamp, problem_text, domain, passed, payload, equation_shapes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), problem_text, model.problem_domain,
             int(report.passed), json.dumps(payload), shapes_json),
        )
        new_id = cur.lastrowid
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
