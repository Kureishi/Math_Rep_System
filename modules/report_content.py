"""
A solved problem as structured report content -- the one thing every export format renders from.

Before this, Markdown and PDF each walked the model and report by hand, so a new section (or a fix to
one) had to be made twice, and any further format would have made it three or four times. Now
`build_sections()` turns a solved problem into an ordered list of `Section`s made of a few kinds of
`Block` (paragraph, equation, bullets, table, image, ...), and each format is only a renderer over that:
modules/exporter.py (Markdown, PDF), report_html.py, report_tex.py, report_office.py (Word, PowerPoint).
Streamlit-free.

WHAT GOES IN is the caller's choice. `include` is the set of section ids to keep (`SECTIONS` lists them,
in report order); `PRESETS` are named selections ("Full audit", "Student handout", ...). A section that
would be empty (no matrix, no plots, no follow-ups) is left out whether or not it was asked for, so a
report never carries a heading over nothing.

Content that is not part of the solved model is passed in as `ReportExtras`: the follow-up Q&A, the
tutor-mode transcripts, the Monte Carlo runs, and the provenance (how the problem was solved).
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import sympy as sp

from modules import provenance as provenance_module
from modules.equation_engine import ProblemModel, target_kind
from modules.matrix_utils import linear_system_view
from modules.sensitivity import tornado_analysis
from modules.solver import SolutionStep
from modules.unit_conversion import sweep_conversions
from modules.vector_utils import vector_summary
from modules.verifier import VerificationReport, _known_substitutions

REPORT_TITLE = "Math Representation System -- Solved Problem"


# ------------------------------------------------------------------ the content model

@dataclass
class PlotSnapshot:
    """A user-chosen static capture of an interactive plot, for inclusion in an exported report -- the
    interactive version lives only in the browser session, so this is the opt-in way to get a specific
    view (with whatever parameter values were selected at the time) into a document. `caption` should
    describe exactly what is shown, since a static image alone doesn't carry that context."""
    title: str
    caption: str
    png_bytes: bytes
    figure_json: str = ""      # the live Plotly figure (fig.to_json()), when one was available -- see Block


@dataclass
class Block:
    """One piece of a section. `kind` decides which of the other fields are used:

        text       text (a paragraph); style "plain" | "italic" | "bold"
        label      text (a bold line introducing what follows)
        equation   text (LaTeX); style "plain" | "step" | "matrix" -- a size hint for formats that typeset to images
        bullets    items
        table      headers, rows
        status     rows of (ok, label, detail) -- a pass/fail list; ok is True/False
        kv         rows of (label, value) -- a short list of facts
        image      png, text (the caption), label_text (the title); figure_json, when present, is the
                   same plot as a live Plotly figure (only the HTML report uses it)
        figure     figure_json, text (caption), label_text (title) -- a live plot with no static image
                   at all, so every format except HTML leaves it out
        subheading text
    """
    kind: str
    text: str = ""
    style: str = "plain"
    items: list[str] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    png: bytes = b""
    label_text: str = ""
    figure_json: str = ""


@dataclass
class Section:
    id: str
    title: str
    blocks: list[Block]


@dataclass
class FollowupItem:
    question: str
    kind: str          # "what_if" | "conceptual" | "error"
    answer: str


@dataclass
class TutorItem:
    target: str
    guess: str
    correct: bool | None
    feedback: str
    steps_revealed: int
    steps_total: int


@dataclass
class ReportExtras:
    """What a report needs beyond the solved model itself."""
    followups: list[FollowupItem] = field(default_factory=list)
    tutor: list[TutorItem] = field(default_factory=list)
    monte_carlo: list[dict[str, Any]] = field(default_factory=list)   # one dict per run, see mc_run_dict()
    provenance: dict[str, Any] | None = None
    include_provenance: bool = True       # a footer is only added when asked for AND there is something to say
    now: dict[str, Any] | None = None     # override "the current settings" (tests)
    live_figures: bool = False            # also build live Plotly figures (tornado charts, the dependency
                                          # graph, Monte Carlo histograms) -- only the HTML report can show them


def mc_run_dict(result: Any, unit: str | None = None) -> dict[str, Any]:
    """A MonteCarloResult as the plain dict the report uses (and bundles could store)."""
    return {"target": result.target, "n": result.n_requested, "n_failed": result.n_failed, "seed": result.seed,
            "mean": result.mean, "std": result.std, "p5": result.p5, "p95": result.p95,
            "inputs": list(getattr(result, "inputs", []) or []), "unit": unit or ""}


# ------------------------------------------------------------------ which sections exist

# (id, title, what it holds) in REPORT ORDER. Titles are the headings the Markdown has always used.
SECTIONS: list[tuple[str, str, str]] = [
    ("problem", "Problem", "The problem text, its domain and the verification verdict"),
    ("results", "Results", "The answer for each target, with units"),
    ("equations", "Derived equations", "Each derived equation with its derivation, and the assumptions"),
    ("variables", "Variables", "Every symbol, its meaning, known value and unit"),
    ("matrix", "Matrix representation", "The linear system as A x = b, with determinant and eigenvalues"),
    ("vectors", "Vectors", "Vector variables with components and magnitude"),
    ("units", "Results in other units", "Each answer converted to common alternative units"),
    ("confidence", "Confidence report", "The overall score and each verification category"),
    ("domain", "Domain of validity", "Where the equations are defined, for the given values"),
    ("verification", "Verification detail", "Every individual check and its outcome"),
    ("steps", "Step-by-step solution", "The worked steps for each target"),
    ("sensitivity", "Sensitivity and uncertainty", "How far each answer moves with each input, and any Monte Carlo runs"),
    ("plots", "Plots", "The plots you marked with 📸 \"Include this plot\""),
    ("scenarios", "Where else this applies", "Other situations the same equations describe"),
    ("followups", "Follow-up questions", "The follow-up Q&A from this session"),
    ("tutor", "Tutor-mode practice", "Your guesses in tutor mode and how they were marked"),
    ("provenance", "Reproducibility", "Model, settings, versions and seeds this was produced with"),
]
SECTION_IDS = [s[0] for s in SECTIONS]
SECTION_TITLES = {s[0]: s[1] for s in SECTIONS}
SECTION_DESCRIPTIONS = {s[0]: s[2] for s in SECTIONS}

PRESETS: dict[str, list[str]] = {
    "Full audit": list(SECTION_IDS),
    "Student handout": ["problem", "results", "equations", "variables", "units", "steps", "plots", "scenarios"],
    "Peer review": ["problem", "results", "equations", "variables", "confidence", "domain", "verification",
                    "sensitivity", "provenance"],
    "Quick summary": ["problem", "results", "confidence", "provenance"],
}
DEFAULT_PRESET = "Full audit"


def unknown_sections(include: list[str] | set[str]) -> list[str]:
    return sorted(set(include) - set(SECTION_IDS))


# ------------------------------------------------------------------ building the sections

def _unit(model: ProblemModel, symbol: str) -> str | None:
    return next((v.unit for v in model.variables if v.symbol == symbol), None)


def _eq_latex(eq) -> str:
    return sp.latex(eq.sympy_eq) if eq.sympy_eq is not None else eq.raw_expression


def _problem(problem_text, model, report) -> Section:
    status = "✅ Passed" if report.passed else "⚠️ Issues found -- review before trusting"
    return Section("problem", SECTION_TITLES["problem"], [
        Block("text", problem_text.strip()),
        Block("kv", rows=[["Domain", model.problem_domain], ["Self-verification", status]]),
    ])


def _results(model, report) -> Section | None:
    if not report.sympy_numeric_answers:
        return None
    rows = [[t, f"{v:.6g}", _unit(model, t) or ""] for t, v in report.sympy_numeric_answers.items()]
    return Section("results", SECTION_TITLES["results"], [Block("table", headers=["Target", "Value", "Unit"], rows=rows)])


def _equations(model) -> Section | None:
    if not model.equations:
        return None
    blocks: list[Block] = []
    for eq in model.equations:
        blocks += [Block("label", eq.name),
                   Block("equation", _eq_latex(eq), style="plain" if eq.sympy_eq is not None else "raw")]
        if eq.derivation:
            blocks.append(Block("text", eq.derivation))
    if model.assumptions:
        blocks += [Block("label", "Assumptions:"), Block("bullets", items=list(model.assumptions))]
    return Section("equations", SECTION_TITLES["equations"], blocks)


def _variables(model) -> Section | None:
    if not model.variables:
        return None
    rows = [[v.symbol, v.meaning, v.known_value if v.known_value is not None else "(solved)", v.unit or ""]
            for v in model.variables]
    return Section("variables", SECTION_TITLES["variables"],
                   [Block("table", headers=["Symbol", "Meaning", "Known value", "Unit"], rows=rows)])


def _matrix(model) -> Section | None:
    result = linear_system_view(model, _known_substitutions(model))
    if result is None:
        return None
    x_latex = sp.latex(sp.Matrix([sp.Symbol(s) for s in result.symbols]))
    blocks = [Block("equation", f"{sp.latex(result.A)} {x_latex} = {sp.latex(result.b)}", style="matrix")]
    if result.is_square:
        blocks.append(Block("text", f"det(A) = {result.determinant}", style="bold"))
        if result.eigenvalues:
            eig = ", ".join(f"{val}" + (f" (x{mult})" if mult > 1 else "") for val, mult in result.eigenvalues.items())
            blocks.append(Block("text", f"Eigenvalues: {eig}"))
    blocks.append(Block("text", result.classification))
    return Section("matrix", SECTION_TITLES["matrix"], blocks)


def _vectors(model) -> Section | None:
    vector_vars = [v for v in model.variables if v.is_vector and v.components]
    if not vector_vars:
        return None
    knowns = _known_substitutions(model)
    items = []
    for v in vector_vars:
        summary = vector_summary(v.symbol, v.components, knowns)
        if summary:
            comp = ", ".join(f"{c}={val:g}" for c, val in summary["components"].items())
            items.append(f"{v.symbol} ({v.meaning}): {comp} -- magnitude = {summary['magnitude']:.6g} {v.unit or ''}".rstrip())
        else:
            items.append(f"{v.symbol} ({v.meaning}): components {', '.join(v.components)}")
    return Section("vectors", SECTION_TITLES["vectors"], [Block("bullets", items=items)])


def _units(model, report) -> Section | None:
    items = []
    for target, val in report.sympy_numeric_answers.items():
        unit = _unit(model, target)
        alternates = sweep_conversions(val, unit)
        if alternates:
            items.append(f"{target} = {val:.6g} {unit} = " + ", ".join(f"{av:.6g} {au}" for au, av in alternates))
    return Section("units", SECTION_TITLES["units"], [Block("bullets", items=items)]) if items else None


def _confidence(report) -> Section:
    cr = report.confidence_report()
    blocks = [Block("text", f"Overall score: {cr.score:.0%} ({cr.label}) -- {cr.passed_count}/{cr.total_count} checks passed.",
                    style="bold"),
              Block("status", rows=[[s.all_passed, cat, f"{s.passed}/{s.total}"] for cat, s in cr.categories.items()])]
    if cr.critical_failures:
        blocks += [Block("label", "Critical failures:"),
                   Block("bullets", items=[f"{c.label}: {c.detail}" for c in cr.critical_failures])]
    return Section("confidence", SECTION_TITLES["confidence"], blocks)


def _domain(report) -> Section | None:
    if not report.domain_notes:
        return None
    rows: list[list[Any]] = []
    for note in report.domain_notes:
        if note.violated:
            rows.append([False, note.equation, "undefined with the given values: " + "; ".join(r.description for r in note.violated)])
        ok = note.satisfied + note.pending
        if ok:
            rows.append([True, note.equation, "requires: " + "; ".join(r.description for r in ok)])
    return Section("domain", SECTION_TITLES["domain"], [Block("status", rows=rows)]) if rows else None


def _verification(report) -> Section:
    return Section("verification", SECTION_TITLES["verification"],
                   [Block("status", rows=[[c.passed, c.label, c.detail] for c in report.checks])])


def _steps(steps_by_target) -> Section | None:
    if not steps_by_target:
        return None
    blocks: list[Block] = []
    for target, steps in steps_by_target.items():
        blocks.append(Block("subheading", f"Solving for {target}"))
        for i, s in enumerate(steps, 1):
            blocks += [Block("label", f"Step {i}: {s.description}"), Block("equation", s.expression, style="step")]
            if s.explanation:
                blocks.append(Block("text", s.explanation, style="italic"))
    return Section("steps", SECTION_TITLES["steps"], blocks)


def _sensitivity(model, report, extras: ReportExtras, pct_range: float = 0.2) -> Section | None:
    blocks: list[Block] = []
    knowns = _known_substitutions(model)
    for target in report.sympy_numeric_answers:
        if target_kind(model, target) != "equation":
            continue
        try:
            entries = tornado_analysis(model, target, knowns, pct_range=pct_range)
        except Exception:  # noqa: BLE001 -- a sensitivity table is a bonus; never let it sink the export
            entries = []
        if not entries:
            continue
        blocks += [Block("label", f"{target}: each input varied by ±{pct_range:.0%}, others held at their given values")]
        if extras.live_figures:
            from modules.plotter import build_tornado_chart
            blocks.append(Block("figure", label_text=f"Sensitivity of {target}", text="Hover for values; zoom and pan with the toolbar.",
                                figure_json=build_tornado_chart(entries).to_json()))
        blocks += [Block("table", headers=["Input", "Input range", f"{target} at low", f"{target} at high", "Swing"],
                         rows=[[e.symbol, f"{e.low_value:.6g} to {e.high_value:.6g}", f"{e.low_target:.6g}",
                                f"{e.high_target:.6g}", f"{e.swing:.6g}"] for e in entries])]
    if extras.monte_carlo:
        def fmt(v):
            return "—" if v is None else f"{v:.6g}"
        blocks += [Block("label", "Monte Carlo runs (from this session)")]
        if extras.live_figures:
            import plotly.graph_objects as go
            for r in extras.monte_carlo:
                if r.get("samples"):
                    hist = go.Figure(go.Histogram(x=r["samples"], nbinsx=40))
                    hist.update_layout(title=f"Monte Carlo: {r['target']} ({r.get('n', len(r['samples']))} samples, seed {r.get('seed', '?')})",
                                       xaxis_title=r["target"] + (f" ({r['unit']})" if r.get("unit") else ""), yaxis_title="count",
                                       bargap=0.02)
                    blocks.append(Block("figure", label_text=f"Distribution of {r['target']}", text="; ".join(r.get("inputs", [])),
                                        figure_json=hist.to_json()))
        blocks += [
                   Block("table", headers=["Target", "Mean", "Std", "5th pct", "95th pct", "Samples", "Seed", "Uncertain inputs"],
                         rows=[[r["target"], fmt(r.get("mean")), fmt(r.get("std")), fmt(r.get("p5")), fmt(r.get("p95")),
                                r.get("n", ""), r.get("seed", ""), "; ".join(r.get("inputs", []))] for r in extras.monte_carlo])]
    return Section("sensitivity", SECTION_TITLES["sensitivity"], blocks) if blocks else None


def _plots(snapshots) -> Section | None:
    if not snapshots:
        return None
    return Section("plots", SECTION_TITLES["plots"],
                   [Block("image", text=s.caption, png=s.png_bytes, label_text=s.title,
                                                            figure_json=s.figure_json) for s in snapshots])


def _scenarios(scenarios) -> Section | None:
    if not scenarios or any("error" in s for s in scenarios):
        return None
    return Section("scenarios", SECTION_TITLES["scenarios"], [Block("bullets", items=[
        f"{s.get('scenario', '')} -- {s.get('mapping', '')}" if s.get("mapping") else str(s.get("scenario", ""))
        for s in scenarios])])


def _followups(extras: ReportExtras) -> Section | None:
    if not extras.followups:
        return None
    blocks: list[Block] = []
    for f in extras.followups:
        blocks += [Block("label", f"Q: {f.question}"),
                   Block("text", ("(could not be answered) " if f.kind == "error" else "") + f.answer)]
    return Section("followups", SECTION_TITLES["followups"], blocks)


def _tutor(extras: ReportExtras) -> Section | None:
    if not extras.tutor:
        return None
    rows = []
    for t in extras.tutor:
        verdict = "not marked" if t.correct is None else ("correct" if t.correct else "incorrect")
        rows.append([t.target, t.guess or "—", verdict, t.feedback or "", f"{t.steps_revealed}/{t.steps_total}"])
    return Section("tutor", SECTION_TITLES["tutor"],
                   [Block("table", headers=["Target", "Your guess", "Marked", "Feedback", "Steps revealed"], rows=rows)])


def _provenance(extras: ReportExtras) -> Section | None:
    if not extras.include_provenance:
        return None
    rows = provenance_module.describe(extras.provenance, extras.monte_carlo, now=extras.now)
    return Section("provenance", SECTION_TITLES["provenance"], [Block("kv", rows=[[k, v] for k, v in rows])])


def build_sections(problem_text: str, model: ProblemModel, report: VerificationReport,
                   steps_by_target: dict[str, list[SolutionStep]], scenarios: list[dict] | None = None,
                   plot_snapshots: list[PlotSnapshot] | None = None, extras: ReportExtras | None = None,
                   include: list[str] | set[str] | None = None) -> list[Section]:
    """The report as ordered sections. `include` None keeps every section that has content; otherwise only
    those ids (ValueError for an id that doesn't exist -- a typo must not silently drop a section).

    Without `extras`, the sections that need one (follow-ups, tutor, reproducibility) are simply absent --
    what `build_markdown()`/`build_pdf_bytes()` did before they existed."""
    bad = unknown_sections(include) if include is not None else []
    if bad:
        raise ValueError(f"Unknown report section(s): {', '.join(bad)}. Choose from: {', '.join(SECTION_IDS)}.")
    wanted = set(SECTION_IDS) if include is None else set(include)
    ex = extras or ReportExtras(include_provenance=False)

    builders = {
        "problem": lambda: _problem(problem_text, model, report),
        "results": lambda: _results(model, report),
        "equations": lambda: _equations(model),
        "variables": lambda: _variables(model),
        "matrix": lambda: _matrix(model),
        "vectors": lambda: _vectors(model),
        "units": lambda: _units(model, report),
        "confidence": lambda: _confidence(report),
        "domain": lambda: _domain(report),
        "verification": lambda: _verification(report),
        "steps": lambda: _steps(steps_by_target),
        "sensitivity": lambda: _sensitivity(model, report, ex),
        "plots": lambda: _plots(plot_snapshots),
        "scenarios": lambda: _scenarios(scenarios),
        "followups": lambda: _followups(ex),
        "tutor": lambda: _tutor(ex),
        "provenance": lambda: _provenance(ex),
    }
    out = []
    for sid in SECTION_IDS:
        if sid in wanted:
            section = builders[sid]()
            if section is not None and section.blocks:
                out.append(section)
    return out


def generated_stamp() -> str:
    return f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
