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


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
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
        return cur.lastrowid


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
