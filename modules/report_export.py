"""
One entry point for every export format -- Streamlit-free, so the app, cli.py and tests all go through it.

    ctx = ExportContext(problem_text, model, report, steps_by_target, scenarios, snapshots, extras)
    exported = render_export("html", ctx, include=PRESETS["Student handout"])
    exported.name, exported.data, exported.mime

`FORMATS` lists what exists and what each costs the reader. Each format is rendered from the same sections
(modules/report_content.py), so choosing sections means the same thing in all of them.

The live-figure HTML report alone asks for the extras that only it can show (histograms of the Monte Carlo
samples), so `render_export` turns `live_figures` on for it and for nothing else.
"""
from dataclasses import dataclass, field, replace

from modules import exporter, report_office, report_tex
from modules.equation_engine import ProblemModel
from modules.report_content import PlotSnapshot, ReportExtras, build_sections
from modules.report_html import render_html
from modules.solver import SolutionStep
from modules.verifier import VerificationReport


@dataclass(frozen=True)
class FormatSpec:
    key: str
    label: str
    icon: str
    ext: str
    mime: str
    note: str


FORMATS: list[FormatSpec] = [
    FormatSpec("pdf", "PDF", "📕", ".pdf", "application/pdf", "Typeset equations; fixed layout for printing or sending."),
    FormatSpec("html", "Interactive HTML", "🌐", ".html", "text/html",
               "One file that opens in any browser: live, zoomable plots and clickable equations. About 5 MB when it holds live plots."),
    FormatSpec("docx", "Word", "📘", ".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
               "Editable; equations are pictures."),
    FormatSpec("pptx", "PowerPoint", "📙", ".pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation",
               "One slide per step for teaching; equations are pictures."),
    FormatSpec("tex", "LaTeX", "📄", ".tex", "application/x-tex",
               "Compilable source; a zip with figures/ when plots are included."),
    FormatSpec("md", "Markdown", "📝", ".md", "text/markdown", "Plain text with $$ equations; plots embedded."),
]
FORMAT_KEYS = [f.key for f in FORMATS]


@dataclass
class ExportContext:
    problem_text: str
    model: ProblemModel
    report: VerificationReport
    steps_by_target: dict[str, list[SolutionStep]]
    scenarios: list[dict] = field(default_factory=list)
    snapshots: list[PlotSnapshot] = field(default_factory=list)
    extras: ReportExtras = field(default_factory=ReportExtras)


@dataclass
class ExportedFile:
    name: str
    data: bytes
    mime: str


def render_export(fmt: str, ctx: ExportContext, include: list[str] | set[str] | None = None,
                  stem: str = "solved_problem") -> ExportedFile:
    """Build `fmt` ("pdf", "html", "docx", "pptx", "tex" or "md") for the chosen sections. ValueError for an
    unknown format or section id."""
    if fmt not in FORMAT_KEYS:
        raise ValueError(f"Unknown export format {fmt!r}; expected one of {', '.join(FORMAT_KEYS)}.")
    spec = next(f for f in FORMATS if f.key == fmt)
    extras = replace(ctx.extras, live_figures=(fmt == "html"))
    sections = build_sections(ctx.problem_text, ctx.model, ctx.report, ctx.steps_by_target, ctx.scenarios,
                              ctx.snapshots, extras, include)
    if fmt == "pdf":
        return ExportedFile(stem + spec.ext, exporter.render_pdf(sections), spec.mime)
    if fmt == "md":
        return ExportedFile(stem + spec.ext, exporter.render_markdown(sections).encode("utf-8"), spec.mime)
    if fmt == "html":
        return ExportedFile(stem + spec.ext, render_html(sections).encode("utf-8"), spec.mime)
    if fmt == "docx":
        return ExportedFile(stem + spec.ext, report_office.render_docx(sections), spec.mime)
    if fmt == "pptx":
        return ExportedFile(stem + spec.ext, report_office.render_pptx(sections), spec.mime)
    name, data, mime = report_tex.package_tex(sections)
    return ExportedFile(stem + (".tex" if name.endswith(".tex") else "_tex.zip"), data, mime)


def available_sections(ctx: ExportContext) -> list[str]:
    """The section ids that would have content for this problem (the rest would be left out anyway), so a
    chooser offers only what exists. Built the cheap way: sections are built with live figures off."""
    return [s.id for s in build_sections(ctx.problem_text, ctx.model, ctx.report, ctx.steps_by_target,
                                          ctx.scenarios, ctx.snapshots, replace(ctx.extras, live_figures=False))]


def expected_file(fmt: str, ctx: ExportContext, include: list[str] | set[str] | None = None,
                  stem: str = "solved_problem") -> tuple[str, str]:
    """(file name, mime type) `render_export` will produce, WITHOUT building it -- a download button needs both
    before anything is generated. The one format whose name depends on content is LaTeX: a .tex, or a zip
    when marked plots (which a lone .tex cannot carry) are included."""
    spec = next((f for f in FORMATS if f.key == fmt), None)
    if spec is None:
        raise ValueError(f"Unknown export format {fmt!r}; expected one of {', '.join(FORMAT_KEYS)}.")
    if fmt == "tex":
        wants_plots = include is None or "plots" in include
        if wants_plots and any(s.png_bytes for s in ctx.snapshots):
            return stem + "_tex.zip", "application/zip"
    return stem + spec.ext, spec.mime
