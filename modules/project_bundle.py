"""
Session/project export & import: bundles EVERYTHING that otherwise
lives tied to one machine -- history.py's solved-problem records,
chains.py's chains, and the session-only Variable Workspace -- into ONE
portable JSON file. Something that can be archived alongside a paper,
emailed to a collaborator, or reloaded on a different machine to pick
up exactly where a session left off. None of that currently survives
moving to a different machine: history.db and chains.db are both local
SQLite files tied to wherever they happen to sit on disk, and the
workspace lives only in Streamlit's in-memory session_state, gone the
moment the browser tab closes.

Reuses history.py's/chains.py's own already-round-trippable storage
format directly (the same payload their own load()/load_chain()
functions reconstruct a ProblemModel from with no LLM calls needed) --
this module doesn't invent a second serialization scheme, just wraps
theirs into one combined file.

Import is ADDITIVE, never destructive: importing a bundle INSERTS its
problems/chains as brand-new rows in the importing machine's own
history.db/chains.db (new autoincrement ids -- no attempt to preserve
or collide with the exporting machine's own ids) and merges workspace
entries into whatever's already in the current session. Nothing already
present is ever overwritten or deleted just because a bundle was
imported; a workspace entry whose name collides with one already
present is skipped, not silently clobbered.
"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime

from modules import history
from modules import chains as chains_module
from modules.chains import InputBinding
from modules.equation_engine import build_model
from modules.verifier import VerificationReport, CheckResult
from modules.solver import SolutionStep

BUNDLE_FORMAT_VERSION = 1


@dataclass
class ImportSummary:
    history_imported: int = 0
    chains_imported: int = 0
    workspace_imported: int = 0
    errors: list = field(default_factory=list)


def export_bundle(history_ids: list | None = None, chain_ids: list | None = None,
                   workspace_entries: dict | None = None) -> str:
    """Returns a JSON string bundling history records, chains, and
    workspace entries. Defaults to EVERY currently-stored history
    record and chain when `history_ids`/`chain_ids` aren't given;
    `workspace_entries` (a name -> WorkspaceEntry dict, i.e.
    Workspace.entries) is passed in explicitly since the workspace
    itself is session-scoped, not something this module can query on
    its own the way history/chains can."""
    if history_ids is None:
        history_ids = [r["id"] for r in history.list_recent(limit=history.MAX_HISTORY_RECORDS)]

    history_records = []
    for hid in history_ids:
        loaded = history.load(hid)
        if loaded is None:
            continue
        problem_text, model, report, steps_by_target, scenarios = loaded
        history_records.append({
            "problem_text": problem_text,
            "raw_json": model.raw_json,
            "verification": {
                "passed": report.passed,
                "checks": [asdict(c) for c in report.checks],
                "sympy_numeric_answers": report.sympy_numeric_answers,
                "llm_independent_answers": report.llm_independent_answers,
            },
            "steps_by_target": {t: [asdict(s) for s in steps] for t, steps in steps_by_target.items()},
            "scenarios": scenarios,
        })

    if chain_ids is None:
        chain_ids = [c["id"] for c in chains_module.list_chains()]

    chain_records = []
    for cid in chain_ids:
        chain = chains_module.load_chain(cid)
        if chain is None:
            continue
        chain_records.append({
            "name": chain.name,
            "steps": [
                {
                    "position": s.position,
                    "problem_text": s.problem_text,
                    "raw_json": s.raw_json,
                    "output_symbol": s.output_symbol,
                    "bindings": [asdict(b) for b in s.bindings],
                }
                for s in chain.steps
            ],
        })

    workspace_records = [asdict(e) for e in (workspace_entries or {}).values()]

    bundle = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "history": history_records,
        "chains": chain_records,
        "workspace": workspace_records,
    }
    return json.dumps(bundle, indent=2)


def import_bundle(bundle_json: str, workspace) -> ImportSummary:
    """Merges a bundle produced by export_bundle() into the CURRENT
    machine's history.db/chains.db and the given Workspace's in-session
    entries. `workspace` is a modules.workspace.Workspace instance (or
    anything duck-typing its `.entries`/`.store()`). Every history/chain
    record is always inserted as a brand-new row; a workspace entry
    whose name already exists is skipped rather than overwritten. A
    malformed individual record is skipped with its error recorded in
    `.errors` -- one bad record shouldn't abort importing the rest of
    an otherwise-good bundle."""
    try:
        bundle = json.loads(bundle_json)
    except json.JSONDecodeError as e:
        return ImportSummary(errors=[f"Not valid JSON: {e}"])

    if bundle.get("format_version") != BUNDLE_FORMAT_VERSION:
        return ImportSummary(errors=[
            f"Unrecognized bundle format version {bundle.get('format_version')!r} -- "
            f"expected {BUNDLE_FORMAT_VERSION}.",
        ])

    errors = []

    n_history = 0
    for rec in bundle.get("history", []):
        try:
            model = build_model(rec["raw_json"])
            v = rec["verification"]
            report = VerificationReport(
                checks=[CheckResult(**c) for c in v["checks"]],
                sympy_numeric_answers=v["sympy_numeric_answers"],
                llm_independent_answers=v["llm_independent_answers"],
                passed=v["passed"],
            )
            steps_by_target = {t: [SolutionStep(**s) for s in steps]
                                 for t, steps in rec["steps_by_target"].items()}
            history.save(rec["problem_text"], model, report, steps_by_target, rec.get("scenarios", []))
            n_history += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"Skipped a history record ('{rec.get('problem_text', '?')[:40]}'): {e}")

    n_chains = 0
    for rec in bundle.get("chains", []):
        try:
            new_chain_id = chains_module.create_chain(rec["name"])
            for step in sorted(rec["steps"], key=lambda s: s["position"]):
                step_model = build_model(step["raw_json"])
                bindings = [InputBinding(**b) for b in step["bindings"]]
                chains_module.add_step(new_chain_id, step["problem_text"], step_model,
                                          step["output_symbol"], bindings)
            n_chains += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"Skipped a chain ('{rec.get('name', '?')}'): {e}")

    n_workspace = 0
    for entry in bundle.get("workspace", []):
        name = entry.get("name")
        if not name:
            errors.append("Skipped a workspace entry with no name.")
            continue
        if name in workspace.entries:
            errors.append(f"Skipped workspace entry '{name}' -- already present.")
            continue
        workspace.store(name, entry.get("value"), entry.get("source", "imported"), entry.get("unit"))
        n_workspace += 1

    return ImportSummary(history_imported=n_history, chains_imported=n_chains,
                           workspace_imported=n_workspace, errors=errors)
