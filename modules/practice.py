"""
Practice and mastery -- Streamlit-free, no LLM.

A review-and-retention loop built on the problem-set generator:
  * An ITEM is a solved, verified problem (from the starter library, your history, or the problem that is open).
  * A REVIEW of an item is a FRESH VARIANT of it (new numbers, a key that passed the app's own verification), so what is being
    practised is the method, never a remembered answer. The variant is a pure function of (item, day, repetition, nonce): reloading
    the page shows the same problem, and the next day shows a different one.
  * WHEN to review is decided by SM-2, the classic spaced-repetition scheduler: a success pushes the next review further out
    (1 day, 6 days, then by a growing "ease" factor); a failure resets it to tomorrow and lowers the ease. Hard items come back
    sooner, easy ones fade away.
  * MASTERY is measured per CONCEPT (a skill tag such as "projectile motion" or "acids and bases"): an exponentially-weighted
    accuracy over recent attempts (half-life 21 days), so last month's results count less than last week's, and a concept with
    too few attempts is "new", not "mastered".

NOTHING HERE READS THE CLOCK except as a default: every function that depends on the date takes `today`, so behaviour is
reproducible and testable. Data lives in its own SQLite file (data/practice.db), local to the machine.

WHAT THIS DOES NOT DO, STATED. It cannot tell whether you understood the method or got lucky on a variant: accuracy over several
variants is the evidence, which is why mastery needs several attempts. And it only covers algebraic problems (a numeric answer),
because that is what can be generated and checked automatically.
"""
import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from modules import problem_sets as ps
from modules.db_migrations import apply_migrations
from modules.db_util import ClosingConnection
from modules.problem_library import LIBRARY, LIBRARY_CONCEPTS

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "practice.db"
START_EASE = 2.5
MIN_EASE = 1.3
MAX_INTERVAL_DAYS = 365
NEW_PER_DAY = 3
HALF_LIFE_DAYS = 21.0
MASTERED_ACCURACY = 0.85
LEARNING_ACCURACY = 0.6
MIN_EFFECTIVE_ATTEMPTS = 3.0       # fewer (recency-weighted) attempts than this and a concept is not called "mastered"


class PracticeError(ValueError):
    """The problem cannot be practised (with the reason), or a request is invalid."""


# ------------------------------------------------------------------------------------------ the database

def _m001(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT NOT NULL UNIQUE, title TEXT NOT NULL, "
                 "problem_text TEXT NOT NULL, model_json TEXT NOT NULL, concepts TEXT NOT NULL, origin TEXT NOT NULL, "
                 "created TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1)")
    conn.execute("CREATE TABLE IF NOT EXISTS schedule (item_id INTEGER PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE, "
                 "ease REAL NOT NULL, interval_days INTEGER NOT NULL, reps INTEGER NOT NULL, lapses INTEGER NOT NULL, "
                 "due TEXT NOT NULL, last_reviewed TEXT, first_attempt TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY AUTOINCREMENT, item_id INTEGER NOT NULL REFERENCES items(id) "
                 "ON DELETE CASCADE, ts TEXT NOT NULL, day TEXT NOT NULL, seed INTEGER NOT NULL, score REAL NOT NULL, quality INTEGER NOT NULL, "
                 "n_targets INTEGER NOT NULL, n_correct INTEGER NOT NULL, scheduled INTEGER NOT NULL, detail TEXT NOT NULL)")
    conn.execute("CREATE INDEX IF NOT EXISTS attempts_day ON attempts(day)")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, factory=ClosingConnection)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    apply_migrations(conn, [_m001])
    conn.commit()
    return conn


# ------------------------------------------------------------------------------------------ scheduling (pure)

@dataclass(frozen=True)
class ScheduleState:
    ease: float = START_EASE
    interval: int = 0
    reps: int = 0
    lapses: int = 0
    due: date = date(1970, 1, 1)


def sm2(state: ScheduleState, quality: int, today: date) -> ScheduleState:
    """The SM-2 update. quality 0-5: below 3 is a failure (back to tomorrow, ease down); 3 or more is a success (the interval grows:
    1 day, 6 days, then the previous interval times the ease). Ease never falls below 1.3 and intervals never exceed a year."""
    if not 0 <= quality <= 5:
        raise PracticeError("quality must be between 0 and 5.")
    ease = max(MIN_EASE, state.ease + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)))
    if quality < 3:
        reps, lapses, interval = 0, state.lapses + 1, 1
    else:
        reps, lapses = state.reps + 1, state.lapses
        interval = 1 if reps == 1 else 6 if reps == 2 else max(1, round(state.interval * ease))
    interval = min(interval, MAX_INTERVAL_DAYS)
    return ScheduleState(ease, interval, reps, lapses, today + timedelta(days=interval))


# ------------------------------------------------------------------------------------------ items

@dataclass
class Item:
    id: int
    key: str
    title: str
    text: str
    raw: dict
    concepts: list[str]
    origin: str
    active: bool
    created: str
    state: ScheduleState
    last_reviewed: date | None = None
    first_attempt: date | None = None

    @property
    def is_new(self) -> bool:
        return self.state.reps == 0 and self.first_attempt is None


def item_key(text: str, raw: dict) -> str:
    """A stable identity for a problem: the same statement and model give the same key, so adding it twice does nothing."""
    blob = text.strip() + "\n" + json.dumps(raw, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]


def _iso(d: date | None) -> str | None:
    return d.isoformat() if d else None


def _item_from_row(row: tuple) -> Item:
    (iid, key, title, text, model_json, concepts, origin, created, active, ease, interval, reps, lapses, due, last, first) = row
    return Item(iid, key, title, text, json.loads(model_json), json.loads(concepts), origin, bool(active), created,
                ScheduleState(ease, interval, reps, lapses, date.fromisoformat(due)),
                date.fromisoformat(last) if last else None, date.fromisoformat(first) if first else None)


_ITEM_SQL = ("SELECT i.id, i.key, i.title, i.problem_text, i.model_json, i.concepts, i.origin, i.created, i.active, "
             "s.ease, s.interval_days, s.reps, s.lapses, s.due, s.last_reviewed, s.first_attempt "
             "FROM items i JOIN schedule s ON s.item_id = i.id")


def add_item(title: str, text: str, raw: dict, concepts: list[str] | None = None, origin: str = "history",
             today: date | None = None) -> tuple[int, bool]:
    """Add a problem to practice. Returns (id, newly_added). It must be an algebraic problem that passes its own checks and can be
    varied; otherwise PracticeError says why. Adding the same problem again does nothing (a removed one is brought back)."""
    today = today or date.today()
    try:
        ps.generate(text, raw, n=1, seed=0)
    except ps.ProblemSetError as exc:
        raise PracticeError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 -- an unreadable model is a reason, not a crash
        raise PracticeError(f"That problem could not be read ({type(exc).__name__}).") from exc
    tags = [c.strip() for c in (concepts or []) if c and c.strip()]
    if not tags:
        tags = [str(raw.get("problem_domain") or "general")]
    key = item_key(text, raw)
    with _connect() as conn:
        row = conn.execute("SELECT id, active FROM items WHERE key = ?", (key,)).fetchone()
        if row:
            if not row[1]:
                conn.execute("UPDATE items SET active = 1 WHERE id = ?", (row[0],))
                return row[0], True
            return row[0], False
        cur = conn.execute("INSERT INTO items (key, title, problem_text, model_json, concepts, origin, created, active) VALUES (?, ?, ?, ?, ?, ?, ?, 1)",
                           (key, title.strip() or "Untitled problem", text, json.dumps(raw, ensure_ascii=False), json.dumps(tags, ensure_ascii=False),
                            origin, today.isoformat()))
        iid = int(cur.lastrowid or 0)
        conn.execute("INSERT INTO schedule (item_id, ease, interval_days, reps, lapses, due) VALUES (?, ?, 0, 0, 0, ?)", (iid, START_EASE, today.isoformat()))
    return iid, True


def seed_library(today: date | None = None) -> int:
    """Add the starter problems (those not already there). Returns how many were added."""
    added = 0
    for name, (text, raw) in LIBRARY.items():
        _, created = add_item(name, text, raw, LIBRARY_CONCEPTS.get(name), "library", today)
        added += int(created)
    return added


def list_items(active_only: bool = True) -> list[Item]:
    with _connect() as conn:
        rows = conn.execute(_ITEM_SQL + (" WHERE i.active = 1" if active_only else "") + " ORDER BY i.id").fetchall()
    return [_item_from_row(r) for r in rows]


def get_item(item_id: int) -> Item | None:
    with _connect() as conn:
        row = conn.execute(_ITEM_SQL + " WHERE i.id = ?", (item_id,)).fetchone()
    return _item_from_row(row) if row else None


def remove_item(item_id: int) -> None:
    """Stop practising an item. Its past attempts are kept, so the mastery history does not change."""
    with _connect() as conn:
        conn.execute("UPDATE items SET active = 0 WHERE id = ?", (item_id,))


# ------------------------------------------------------------------------------------------ today's queue

@dataclass
class QueueEntry:
    item: Item
    kind: str               # review | new
    overdue_days: int = 0


def queue(today: date | None = None, new_limit: int = NEW_PER_DAY, limit: int = 30) -> list[QueueEntry]:
    """What to practise today: reviews that are due (the most overdue, then the hardest, first), then new items up to the daily allowance
    (less any new item already started today)."""
    today = today or date.today()
    items = list_items()
    reviews = sorted((i for i in items if not i.is_new and i.state.due <= today), key=lambda i: (i.state.due, i.state.ease))
    started_today = sum(1 for i in items if i.first_attempt == today)
    allowance = max(0, new_limit - started_today)
    fresh = [i for i in items if i.is_new][:allowance]
    entries = [QueueEntry(i, "review", (today - i.state.due).days) for i in reviews] + [QueueEntry(i, "new") for i in fresh]
    return entries[:limit]


# ------------------------------------------------------------------------------------------ one review

def variant_seed(item: Item, day: date, nonce: int = 0) -> int:
    blob = f"{item.key}:{day.isoformat()}:{item.state.reps}:{nonce}"
    return int(hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8], 16)


def variant(item: Item, day: date | None = None, nonce: int = 0) -> tuple[ps.ProblemInstance, int]:
    """(a fresh variant of the item, its seed): the same for the same item, day, repetition and nonce."""
    day = day or date.today()
    seed = variant_seed(item, day, nonce)
    try:
        pset = ps.generate(item.text, item.raw, n=1, seed=seed, title=item.title)
    except ps.ProblemSetError as exc:
        raise PracticeError(str(exc)) from exc
    return pset.instances[0], seed


@dataclass
class Grade:
    verdicts: dict[str, ps.Verdict]
    score: float                 # mean credit over the targets, 0-1
    quality: int                 # SM-2 quality, 0-5
    all_correct: bool
    n_correct: int

    @property
    def summary(self) -> str:
        return f"{self.n_correct} of {len(self.verdicts)} correct"


def quality_for(verdicts: dict[str, ps.Verdict]) -> int:
    """SM-2 quality from the per-answer verdicts: 5 all correct; 4 all correct or close (rounding); 3 at least half the credit; 2 some credit;
    1 an attempt with no credit; 0 nothing entered."""
    statuses = [v.status for v in verdicts.values()]
    score = sum(v.score for v in verdicts.values()) / max(len(verdicts), 1)
    if all(s == "correct" for s in statuses):
        return 5
    if all(s in ("correct", "close") for s in statuses):
        return 4
    if all(s == "missing" for s in statuses):
        return 0
    return 3 if score >= 0.5 else 2 if score > 0 else 1


def grade(instance: ps.ProblemInstance, answers: dict[str, str], base_answers: dict[str, float] | None = None,
          rel_tol: float = ps.DEFAULT_TOLERANCE) -> Grade:
    """Grade typed answers (one string per target) against the variant's verified key."""
    verdicts: dict[str, ps.Verdict] = {}
    for target, key in instance.answers.items():
        text = (answers.get(target) or "").strip()
        value, _unit = ps.parse_answer(text) if text else (None, "")
        if text and value is None:
            verdicts[target] = ps.Verdict("unreadable", 0.0, "no number found in the answer")
        else:
            verdicts[target] = ps.grade_value(value, key, rel_tol, other_answers=instance.answers, base_answers=base_answers, target=target)
    score = sum(v.score for v in verdicts.values()) / max(len(verdicts), 1)
    n_correct = sum(v.status == "correct" for v in verdicts.values())
    return Grade(verdicts, score, quality_for(verdicts), n_correct == len(verdicts), n_correct)


def record(item_id: int, seed: int, result: Grade, today: date | None = None, scheduled: bool = True, ts: str | None = None) -> ScheduleState:
    """Save an attempt. A scheduled attempt also updates the item's review schedule (SM-2); an extra one (a second variant tried straight
    away) does not. Returns the item's schedule afterwards."""
    today = today or date.today()
    item = get_item(item_id)
    if item is None:
        raise PracticeError("That problem is not in your practice list.")
    detail = {t: {"status": v.status, "diagnosis": v.diagnosis} for t, v in result.verdicts.items()}
    with _connect() as conn:
        conn.execute("INSERT INTO attempts (item_id, ts, day, seed, score, quality, n_targets, n_correct, scheduled, detail) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (item_id, ts or datetime.now().isoformat(timespec="seconds"), today.isoformat(), seed, result.score, result.quality,
                      len(result.verdicts), result.n_correct, int(scheduled), json.dumps(detail)))
        state = item.state
        if scheduled:
            state = sm2(item.state, result.quality, today)
            conn.execute("UPDATE schedule SET ease = ?, interval_days = ?, reps = ?, lapses = ?, due = ?, last_reviewed = ?, "
                         "first_attempt = COALESCE(first_attempt, ?) WHERE item_id = ?",
                         (state.ease, state.interval, state.reps, state.lapses, state.due.isoformat(), today.isoformat(), today.isoformat(), item_id))
        else:
            conn.execute("UPDATE schedule SET first_attempt = COALESCE(first_attempt, ?) WHERE item_id = ?", (today.isoformat(), item_id))
    return state


# ------------------------------------------------------------------------------------------ progress

@dataclass
class AttemptRow:
    item_id: int
    day: date
    score: float
    quality: int
    scheduled: bool


def attempts(since: date | None = None) -> list[AttemptRow]:
    with _connect() as conn:
        rows = conn.execute("SELECT item_id, day, score, quality, scheduled FROM attempts" + (" WHERE day >= ?" if since else "") + " ORDER BY id",
                            ((since.isoformat(),) if since else ())).fetchall()
    return [AttemptRow(r[0], date.fromisoformat(r[1]), r[2], r[3], bool(r[4])) for r in rows]


def streak(today: date | None = None) -> tuple[int, int]:
    """(current, longest) run of consecutive days with at least one attempt. The current streak is still alive if you practised yesterday but
    not yet today."""
    today = today or date.today()
    days = sorted({a.day for a in attempts() if a.day <= today})          # "as of today": a later attempt has not happened yet
    if not days:
        return 0, 0
    longest = run = 1
    for prev, cur in zip(days, days[1:]):
        run = run + 1 if (cur - prev).days == 1 else 1
        longest = max(longest, run)
    last = days[-1]
    if (today - last).days > 1:
        return 0, longest
    current = 1
    for prev, cur in zip(reversed(days[:-1]), reversed(days[1:])):
        if (cur - prev).days == 1:
            current += 1
        else:
            break
    return current, longest


@dataclass
class ConceptMastery:
    concept: str
    accuracy: float | None        # recency-weighted share of credit, None if never attempted
    effective_attempts: float     # the recency-weighted number of attempts behind it
    attempts: int
    level: str                    # new | learning | solid | mastered
    last_practised: date | None = None


def _level(accuracy: float | None, effective: float) -> str:
    if accuracy is None or effective < 1.0:
        return "new"
    if accuracy >= MASTERED_ACCURACY and effective >= MIN_EFFECTIVE_ATTEMPTS:
        return "mastered"
    if accuracy >= LEARNING_ACCURACY:
        return "solid"
    return "learning"


def mastery(today: date | None = None, half_life: float = HALF_LIFE_DAYS) -> list[ConceptMastery]:
    """Per concept: recency-weighted accuracy (an attempt's weight halves every `half_life` days), weakest first. Every item that carries
    a concept contributes its attempts to it."""
    today = today or date.today()
    items = {i.id: i for i in list_items(active_only=False)}
    by_concept: dict[str, list[AttemptRow]] = {}
    for item in items.values():
        if item.active:
            for c in item.concepts:
                by_concept.setdefault(c, [])
    for a in attempts():
        owner = items.get(a.item_id)
        if owner:
            for c in owner.concepts:
                by_concept.setdefault(c, []).append(a)
    out = []
    for concept, rows in by_concept.items():
        weights = [0.5 ** (max((today - a.day).days, 0) / half_life) for a in rows]
        total = sum(weights)
        acc = sum(w * a.score for w, a in zip(weights, rows)) / total if total > 0 else None
        out.append(ConceptMastery(concept, acc, total, len(rows), _level(acc, total), max((a.day for a in rows), default=None)))
    return sorted(out, key=lambda m: (m.accuracy is None, m.accuracy if m.accuracy is not None else 0.0, m.concept))


def heatmap(today: date | None = None, weeks: int = 8) -> tuple[list[str], list[str], list[list[float | None]]]:
    """(concepts, week labels, matrix): the average score per concept per week over the last `weeks` weeks (None where nothing was practised).
    Weeks start on Monday; the last column is this week."""
    today = today or date.today()
    this_monday = today - timedelta(days=today.weekday())
    starts = [this_monday - timedelta(weeks=w) for w in range(weeks - 1, -1, -1)]
    items = {i.id: i for i in list_items(active_only=False)}
    concepts = [m.concept for m in mastery(today)]
    cells: dict[tuple[str, int], list[float]] = {}
    for a in attempts():
        owner = items.get(a.item_id)
        if not owner:
            continue
        idx = (a.day - starts[0]).days // 7
        if 0 <= idx < weeks:
            for c in owner.concepts:
                cells.setdefault((c, idx), []).append(a.score)
    matrix = [[(sum(cells[(c, w)]) / len(cells[(c, w)]) if (c, w) in cells else None) for w in range(weeks)] for c in concepts]
    return concepts, [s.strftime("%d %b") for s in starts], matrix


def forecast(today: date | None = None, days: int = 14) -> list[tuple[date, int]]:
    """How many reviews fall due on each of the next `days` days (overdue ones are counted today)."""
    today = today or date.today()
    due = [max(i.state.due, today) for i in list_items() if not i.is_new]
    return [(today + timedelta(days=d), sum(1 for x in due if x == today + timedelta(days=d))) for d in range(days)]


def summary(today: date | None = None, new_limit: int = NEW_PER_DAY) -> dict[str, Any]:
    today = today or date.today()
    items = list_items()
    cur, longest = streak(today)
    week = attempts(today - timedelta(days=6))
    todays = queue(today, new_limit, 10 ** 6)
    return {
        "items": len(items),
        "due": sum(1 for e in todays if e.kind == "review"),
        "new": sum(1 for e in todays if e.kind == "new"),
        "streak": cur, "longest_streak": longest,
        "practised_today": any(a.day == today for a in attempts(today)),
        "accuracy_7d": (sum(a.score for a in week) / len(week)) if week else None,
        "attempts_7d": len(week),
    }
