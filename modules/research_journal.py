"""
Research journal: stitches a chosen set of history entries into ONE
running Markdown document -- problem text, derived equations, variable
tables, verification status, concept citations, and any transfer-
learning scenarios -- rather than leaving a research session as
scattered, individually-loaded problems with no combined record.

Deliberately takes already-loaded entries (from history.load()) as
input rather than reaching into history.py's database itself: this
module is a pure formatter, decoupled from persistence, the same
separation dependency_graph.py keeps from equation_engine.py (operates
on a ProblemModel it's handed, never fetches one). That keeps this
testable without a database and reusable regardless of where the
entries came from (history.py today; conceivably a future in-session
list of "everything solved this sitting," not yet persisted, later).

Concept tags are recomputed via concept_index.concept_tags_for_model()
rather than read back from history.py's stored concept_tags column --
recomputing is cheap (structural pattern matching, no LLM call) and
sidesteps needing to widen history.load()'s existing return signature
(several call sites already depend on its exact shape) just to also
carry tags through.

Markdown, not a second .ipynb-style hand-built format: a research
journal is meant to be READ start to finish as a document (headers,
prose, embedded LaTeX), which is squarely what Markdown is for --
notebook_export.py's cell-based approach is for something meant to be
RUN, a different job entirely.
"""
from dataclasses import dataclass, field

import sympy as sp

from modules.equation_engine import ProblemModel
from modules.verifier import VerificationReport
from modules.solver import SolutionStep
from modules.concept_index import concept_tags_for_model


@dataclass
class JournalEntry:
    entry_id: int
    problem_text: str
    model: ProblemModel
    report: VerificationReport
    steps_by_target: dict[str, list[SolutionStep]] = field(default_factory=dict)
    scenarios: list[dict] = field(default_factory=list)
    timestamp: str = ""


def build_journal_entry(entry_id: int, loaded: tuple, timestamp: str = "") -> JournalEntry:
    """Wraps the 5-tuple history.load() returns into a JournalEntry --
    call history.load(entry_id) yourself and pass its result straight
    through, keeping this module's only dependency on history.py's
    return SHAPE, not the module itself."""
    problem_text, model, report, steps_by_target, scenarios = loaded
    return JournalEntry(entry_id=entry_id, problem_text=problem_text, model=model, report=report,
                          steps_by_target=steps_by_target, scenarios=scenarios, timestamp=timestamp)


def _entry_heading(entry: JournalEntry) -> str:
    snippet = entry.problem_text.strip().replace("\n", " ")
    if len(snippet) > 70:
        snippet = snippet[:70].rstrip() + "..."
    return f"Problem {entry.entry_id}: {snippet}"


def _render_entry(entry: JournalEntry) -> list[str]:
    lines = [f"## {_entry_heading(entry)}", ""]

    badge = "✅ Verified" if entry.report.passed else "⚠️ Verification did not pass"
    meta_bits = [badge]
    if entry.timestamp:
        meta_bits.append(entry.timestamp.replace("T", " "))
    if entry.model.problem_domain:
        meta_bits.append(f"domain: {entry.model.problem_domain}")
    lines.append(" · ".join(meta_bits))
    lines.append("")

    lines.append("**Problem statement:**")
    lines.append("")
    lines.append(f"> {entry.problem_text.strip()}")
    lines.append("")

    try:
        tags = concept_tags_for_model(entry.model)
    except Exception:  # noqa: BLE001
        tags = []
    if tags:
        lines.append(f"**Concepts:** {', '.join(tags)}")
        lines.append("")

    if entry.model.variables:
        lines.append("**Variables:**")
        lines.append("")
        lines.append("| Symbol | Meaning | Value | Unit |")
        lines.append("|---|---|---|---|")
        for v in entry.model.variables:
            value = f"{v.known_value:g}" if v.known_value is not None else "*(solved for)*"
            lines.append(f"| {v.symbol} | {v.meaning or '—'} | {value} | {v.unit or '—'} |")
        lines.append("")

    if entry.model.equations:
        lines.append("**Equations:**")
        lines.append("")
        for eq in entry.model.equations:
            if eq.sympy_eq is not None:
                lines.append(f"$${sp.latex(eq.sympy_eq)}$$")
            else:
                lines.append(f"*(failed to parse: `{eq.raw_expression}`)*")
            if eq.derivation:
                lines.append(f"— {eq.derivation}")
            lines.append("")

    if entry.report.sympy_numeric_answers:
        lines.append("**Results:**")
        lines.append("")
        for name, value in entry.report.sympy_numeric_answers.items():
            lines.append(f"- {name} = {value:g}")
        lines.append("")

    failed_checks = [c for c in entry.report.checks if not c.passed]
    if failed_checks:
        lines.append("**Verification notes:**")
        lines.append("")
        for c in failed_checks:
            lines.append(f"- ⚠️ {c.label}: {c.detail}")
        lines.append("")

    if entry.scenarios:
        lines.append("**Other contexts this structure applies to:**")
        lines.append("")
        for s in entry.scenarios:
            scenario_text = s.get("scenario", "")
            mapping_text = s.get("mapping", "")
            lines.append(f"- {scenario_text}" + (f" ({mapping_text})" if mapping_text else ""))
        lines.append("")

    lines.append("---")
    lines.append("")
    return lines


def generate_journal_markdown(entries: list[JournalEntry], title: str = "Research Journal") -> str:
    """Assembles `entries` (in the order given -- callers should sort
    however makes sense for the journal, e.g. chronological or grouped
    by concept before calling this) into one Markdown document: a
    header, a table of contents, a "concepts covered" summary (the
    research-memory payoff -- one place showing everything this session
    touched), then one section per entry."""
    lines = [f"# {title}", ""]

    if not entries:
        lines.append("*No entries selected.*")
        return "\n".join(lines)

    all_tags: dict[str, int] = {}
    for entry in entries:
        try:
            for tag in concept_tags_for_model(entry.model):
                all_tags[tag] = all_tags.get(tag, 0) + 1
        except Exception:  # noqa: BLE001
            continue

    if all_tags:
        lines.append("**Concepts covered in this journal:** " +
                      ", ".join(f"{tag} ({count})" for tag, count in
                                 sorted(all_tags.items(), key=lambda kv: (-kv[1], kv[0]))))
        lines.append("")

    lines.append("**Contents:**")
    lines.append("")
    for entry in entries:
        lines.append(f"- {_entry_heading(entry)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for entry in entries:
        lines.extend(_render_entry(entry))

    return "\n".join(lines)
