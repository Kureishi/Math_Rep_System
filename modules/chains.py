"""
Multi-problem dependency chains: formalizes the ad-hoc "extract to
workspace" flow (workspace.py) into a NAMED, PERSISTENT sequence of
problems where an upstream problem's solved value is wired directly
into a downstream problem's known inputs -- editing an upstream input
automatically re-solves every step that depends on it, cascading
downstream, the way a cell edit in a spreadsheet ripples through
formulas that reference it. workspace.py, by contrast, is a one-shot
copy: pulling a value out once doesn't keep it wired to where it came
from, so re-solving the upstream problem again doesn't touch anything
downstream that already used the old value.

Distinct from dependency_graph.py, which only diagrams the dependency
structure WITHIN a single already-extracted problem (which equations
feed which unknowns, inside one ProblemModel). This module operates one
level up, across MULTIPLE separately-extracted problems, and actually
PERFORMS the re-solve on an edit, rather than just visualizing existing
structure.

Deliberately re-solves with plain SymPy only (verifier._solve_sympy),
never the full verify() pipeline's LLM independent cross-check -- a
chain can be re-solved many times as an upstream input changes (e.g.
"what if the initial velocity were 10 instead of 8"), and requiring an
LLM round-trip on every such edit isn't something this app should
impose. Full verification (with the LLM cross-check) still happens
once, normally, through the regular extract/verify pipeline BEFORE a
solved problem is added to a chain -- a chain step is built from an
already-trusted ProblemModel, not re-derived from scratch here.
"""
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from modules.equation_engine import ProblemModel, build_model, target_kind
from modules.verifier import _solve_sympy

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chains.db"

MAX_STEPS_PER_CHAIN = 50  # a sanity cap, not a hard product limit -- keeps a single
                            # resolve_chain() pass bounded even on a runaway chain


@dataclass
class InputBinding:
    symbol: str                            # the LOCAL variable symbol in this step being overridden
    source: str                            # "literal" | "upstream"
    literal_value: float | None = None     # used when source == "literal"
    upstream_position: int | None = None   # used when source == "upstream": which earlier step
    upstream_symbol: str | None = None     # which OUTPUT symbol of that step to pull (documentation
                                             # only -- resolution always uses that step's own
                                             # output_symbol; kept here so the UI can show it without
                                             # a second lookup)


@dataclass
class ChainStep:
    position: int                # 0-based, dense, in chain order
    problem_text: str
    raw_json: dict                # this step's own extraction JSON -- rebuilt into a
                                   # ProblemModel fresh on every resolve
    output_symbol: str            # which solved target this step exposes downstream
    bindings: list[InputBinding] = field(default_factory=list)
    output_value: float | None = None
    status: str = "stale"         # "ok" | "error" | "stale"
    error_detail: str | None = None


@dataclass
class Chain:
    id: int
    name: str
    steps: list[ChainStep] = field(default_factory=list)


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chain_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_id INTEGER NOT NULL,
            position INTEGER NOT NULL,
            problem_text TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            output_symbol TEXT NOT NULL,
            bindings TEXT NOT NULL,
            output_value REAL,
            status TEXT NOT NULL DEFAULT 'stale',
            error_detail TEXT,
            FOREIGN KEY(chain_id) REFERENCES chains(id)
        )
    """)
    return conn


# ---------------------------------------------------------------- chain CRUD

def create_chain(name: str) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO chains (name, created_at) VALUES (?, ?)",
            (name, datetime.now().isoformat(timespec="seconds")),
        )
        return cur.lastrowid


def list_chains() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT id, name, created_at FROM chains ORDER BY id DESC").fetchall()
    return [{"id": r[0], "name": r[1], "created_at": r[2]} for r in rows]


def rename_chain(chain_id: int, new_name: str):
    with _connect() as conn:
        conn.execute("UPDATE chains SET name = ? WHERE id = ?", (new_name, chain_id))


def delete_chain(chain_id: int):
    with _connect() as conn:
        conn.execute("DELETE FROM chain_steps WHERE chain_id = ?", (chain_id,))
        conn.execute("DELETE FROM chains WHERE id = ?", (chain_id,))


# ---------------------------------------------------------------- binding (de)serialization

def _bindings_to_json(bindings: list[InputBinding]) -> str:
    return json.dumps([
        {"symbol": b.symbol, "source": b.source, "literal_value": b.literal_value,
         "upstream_position": b.upstream_position, "upstream_symbol": b.upstream_symbol}
        for b in bindings
    ])


def _bindings_from_json(raw: str) -> list[InputBinding]:
    return [InputBinding(**d) for d in json.loads(raw)]


# ---------------------------------------------------------------- step CRUD

def add_step(chain_id: int, problem_text: str, model: ProblemModel, output_symbol: str,
             bindings: list[InputBinding] | None = None) -> int:
    """Appends a new step built from an already-extracted/verified
    ProblemModel. `output_symbol` must be one of model.solve_for and
    algebraic (target_kind == "equation") -- that's the only kind of
    result this module currently knows how to feed forward into a
    downstream step's known-value substitution. Triggers an immediate
    resolve_chain() so the new step (and anything wired to it) has a
    fresh output_value right away. Returns the new step's position
    (0-based, in insertion order within the chain)."""
    if output_symbol not in model.solve_for or target_kind(model, output_symbol) != "equation":
        raise ValueError(f"'{output_symbol}' isn't an algebraic solve_for target of this problem.")
    with _connect() as conn:
        existing = conn.execute(
            "SELECT COUNT(*) FROM chain_steps WHERE chain_id = ?", (chain_id,)
        ).fetchone()[0]
        if existing >= MAX_STEPS_PER_CHAIN:
            raise ValueError(f"Chain already has the maximum of {MAX_STEPS_PER_CHAIN} steps.")
        position = existing
        conn.execute(
            "INSERT INTO chain_steps (chain_id, position, problem_text, raw_json, output_symbol, "
            "bindings, status) VALUES (?, ?, ?, ?, ?, ?, 'stale')",
            (chain_id, position, problem_text, json.dumps(model.raw_json), output_symbol,
             _bindings_to_json(bindings or [])),
        )
    resolve_chain(chain_id)
    return position


def set_step_bindings(chain_id: int, position: int, bindings: list[InputBinding]):
    """Replaces a step's input bindings -- e.g. changing a literal
    override's value (the "what if an upstream input were different"
    case), or re-wiring which upstream step feeds it -- and cascades a
    full re-solve of the whole chain."""
    with _connect() as conn:
        conn.execute(
            "UPDATE chain_steps SET bindings = ? WHERE chain_id = ? AND position = ?",
            (_bindings_to_json(bindings), chain_id, position),
        )
    resolve_chain(chain_id)


def remove_step(chain_id: int, position: int):
    """Removes a step and shifts every later step's position down by
    one, so positions stay a dense 0..n-1 sequence. A binding elsewhere
    in the chain that pointed at the removed step (or at a step whose
    position just shifted) is NOT rewritten -- upstream_position
    references are not renumbered here, so a now-broken binding is left
    for the next resolve_chain() to catch and report as that step's
    error, rather than silently re-pointing it at a different step."""
    with _connect() as conn:
        conn.execute("DELETE FROM chain_steps WHERE chain_id = ? AND position = ?", (chain_id, position))
        rows = conn.execute(
            "SELECT id, position FROM chain_steps WHERE chain_id = ? AND position > ? ORDER BY position",
            (chain_id, position),
        ).fetchall()
        for step_id, pos in rows:
            conn.execute("UPDATE chain_steps SET position = ? WHERE id = ?", (pos - 1, step_id))
    resolve_chain(chain_id)


def load_chain(chain_id: int) -> Chain | None:
    with _connect() as conn:
        row = conn.execute("SELECT id, name FROM chains WHERE id = ?", (chain_id,)).fetchone()
        if row is None:
            return None
        step_rows = conn.execute(
            "SELECT position, problem_text, raw_json, output_symbol, bindings, output_value, "
            "status, error_detail FROM chain_steps WHERE chain_id = ? ORDER BY position",
            (chain_id,),
        ).fetchall()
    steps = [
        ChainStep(
            position=r[0], problem_text=r[1], raw_json=json.loads(r[2]), output_symbol=r[3],
            bindings=_bindings_from_json(r[4]), output_value=r[5], status=r[6], error_detail=r[7],
        )
        for r in step_rows
    ]
    return Chain(id=row[0], name=row[1], steps=steps)


# ---------------------------------------------------------------- resolving

def resolve_chain(chain_id: int) -> Chain | None:
    """Re-solves every step in position order, feeding each step's
    resolved output_value forward into any downstream step whose
    bindings reference it. A plain single top-to-bottom pass is
    sufficient (not a general dependency-graph solve): a binding can
    only legitimately reference an EARLIER position, so by the time a
    step is reached every value it could depend on has already been
    computed this same pass. Nothing enforces "earlier position only"
    at the DB level, so a binding pointing at position >= itself, or at
    a position that no longer exists / didn't solve, is caught here and
    reported as THAT step's own error rather than crashing the whole
    pass or silently propagating a stale/wrong number."""
    chain = load_chain(chain_id)
    if chain is None:
        return None

    resolved_outputs: dict[int, float] = {}  # position -> its output_value, for steps already
                                                # solved earlier in this same pass

    with _connect() as conn:
        for step in chain.steps:
            model = build_model(step.raw_json)
            overrides: dict[str, float] = {}
            error = None

            for b in step.bindings:
                if b.source == "literal":
                    if b.literal_value is None:
                        error = f"Binding for '{b.symbol}' has no literal value set."
                        break
                    overrides[b.symbol] = b.literal_value
                elif b.source == "upstream":
                    if b.upstream_position is None or b.upstream_position >= step.position:
                        error = f"Binding for '{b.symbol}' references an invalid upstream step."
                        break
                    if b.upstream_position not in resolved_outputs:
                        error = (f"Binding for '{b.symbol}' depends on step "
                                  f"{b.upstream_position + 1}, which hasn't solved successfully.")
                        break
                    overrides[b.symbol] = resolved_outputs[b.upstream_position]
                else:
                    error = f"Unknown binding source '{b.source}' for '{b.symbol}'."
                    break

            answers: dict[str, float] = {}
            if error is None:
                for var in model.variables:
                    if var.symbol in overrides:
                        var.known_value = overrides[var.symbol]
                try:
                    answers = _solve_sympy(model)
                except Exception as e:  # noqa: BLE001
                    error = f"Solve failed: {e}"
                else:
                    if step.output_symbol not in answers:
                        error = f"Couldn't solve for '{step.output_symbol}' with these inputs."

            if error is None:
                value = answers[step.output_symbol]
                resolved_outputs[step.position] = value
                conn.execute(
                    "UPDATE chain_steps SET output_value = ?, status = 'ok', error_detail = NULL "
                    "WHERE chain_id = ? AND position = ?",
                    (value, chain_id, step.position),
                )
                step.output_value, step.status, step.error_detail = value, "ok", None
            else:
                conn.execute(
                    "UPDATE chain_steps SET output_value = NULL, status = 'error', error_detail = ? "
                    "WHERE chain_id = ? AND position = ?",
                    (error, chain_id, step.position),
                )
                step.output_value, step.status, step.error_detail = None, "error", error

    return chain


# ---------------------------------------------------------------- convenience

def suggest_bindings(chain: Chain, model: ProblemModel) -> list[InputBinding]:
    """Best-effort auto-wiring for a new step about to be appended to
    `chain`: for each of the new step's own STILL-UNKNOWN variables
    (known_value is None in its own extraction) whose symbol name
    matches an existing step's output_symbol exactly, proposes an
    upstream binding to that step. This only saves the common case of
    matching variable names -- it doesn't try to be clever about units
    or meaning, so anything not auto-matched is left for the caller/UI
    to bind manually (as a literal override or a differently-named
    upstream symbol)."""
    suggestions: list[InputBinding] = []
    for var in model.variables:
        if var.known_value is not None:
            continue
        for step in chain.steps:
            if step.output_symbol == var.symbol:
                suggestions.append(InputBinding(
                    symbol=var.symbol, source="upstream",
                    upstream_position=step.position, upstream_symbol=step.output_symbol,
                ))
                break
    return suggestions


# ---------------------------------------------------------------- research: parameter sweep

MAX_SWEEP_POINTS = 200  # each sweep point re-resolves the ENTIRE chain (a real sp.solve() per
                          # downstream step), so this is a much tighter cap than monte_carlo.py's,
                          # which solves symbolically only once regardless of sample count


def sweep_step_binding(chain_id: int, position: int, symbol: str,
                         values: list[float]) -> list[dict]:
    """Sweeps ONE literal input binding on step `position` across
    `values`, re-resolving the WHOLE chain at each swept value (via
    resolve_chain()'s ordinary top-to-bottom cascade), and returns the
    resulting output value of every step at every swept point --
    the "how does everything downstream change as I vary this one
    upstream input" research view. Restores the chain's stored bindings
    to their original values before returning (a sweep is exploratory,
    not a step's real state -- it shouldn't leave the persisted chain
    sitting on the last swept value).

    Returns a list of {"value": <swept value>, "outputs": {position:
    output_value_or_None}} rows, one per entry in `values`, suitable for
    plotter.build_chain_sweep_plot() / plot_snapshot.snapshot_chain_sweep_plot().
    """
    if len(values) > MAX_SWEEP_POINTS:
        raise ValueError(f"Sweep has {len(values)} points; the cap is {MAX_SWEEP_POINTS}.")
    if not values:
        raise ValueError("Need at least one value to sweep over.")

    chain = load_chain(chain_id)
    if chain is None:
        raise ValueError(f"No chain with id {chain_id}.")
    step = next((s for s in chain.steps if s.position == position), None)
    if step is None:
        raise ValueError(f"Chain {chain_id} has no step at position {position}.")

    original_bindings = step.bindings
    matching = [b for b in original_bindings if b.symbol == symbol and b.source == "literal"]
    if not matching:
        raise ValueError(f"Step {position + 1} has no literal binding for '{symbol}' to sweep.")

    rows: list[dict] = []
    try:
        for value in values:
            swept_bindings = [
                InputBinding(symbol=b.symbol, source=b.source, literal_value=value,
                              upstream_position=b.upstream_position, upstream_symbol=b.upstream_symbol)
                if b.symbol == symbol and b.source == "literal" else b
                for b in original_bindings
            ]
            set_step_bindings(chain_id, position, swept_bindings)
            resolved = load_chain(chain_id)
            outputs = {s.position: s.output_value for s in resolved.steps}
            rows.append({"value": value, "outputs": outputs})
    finally:
        # always restore the chain to its pre-sweep state, even if a
        # swept value raised partway through
        set_step_bindings(chain_id, position, original_bindings)

    return rows
