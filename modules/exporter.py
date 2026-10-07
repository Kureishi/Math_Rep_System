"""
Exports a solved problem to Markdown or PDF.

Markdown is the simpler, lossless format -- LaTeX equations go in as
$$...$$ blocks, which render natively in most markdown viewers (GitHub,
Obsidian, VS Code) that support MathJax/KaTeX.

PDF rendering needs equations turned into actual images since PDF has no
native math typesetting without a full LaTeX toolchain (which we're
deliberately not requiring, for portability). matplotlib's mathtext engine
renders a large, practically-relevant subset of LaTeX without needing a
system TeX installation, so each equation/step is rendered to a small
in-memory PNG and embedded in the PDF.
"""
import base64
import io
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # headless -- no display needed, safe in a server context
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from modules.equation_engine import ProblemModel
from modules.report_content import (
    REPORT_TITLE, PlotSnapshot, ReportExtras, Section, build_sections, generated_stamp,
)
from modules.verifier import VerificationReport
from modules.solver import SolutionStep


# ---------------------------------------------------------------- Markdown

def _md_cell(value) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(sections: list[Section], title: str = REPORT_TITLE) -> str:
    """Markdown for already-built sections. LaTeX goes in as $$...$$ blocks."""
    L: list[str] = [f"# {title}", f"*{generated_stamp()}*", ""]
    for sec in sections:
        L += [f"## {sec.title}", ""]
        for b in sec.blocks:
            if b.kind == "text":
                L += [{"italic": f"_{b.text}_", "bold": f"**{b.text}**"}.get(b.style, b.text), ""]
            elif b.kind == "label":
                L += [f"**{b.text}**", ""]
            elif b.kind == "subheading":
                m = re.fullmatch(r"Solving for (.+)", b.text)
                L += [f"### Solving for `{m.group(1)}`" if m else f"### {b.text}", ""]
            elif b.kind == "equation":
                L += [f"$$ {b.text} $$", ""]
            elif b.kind == "bullets":
                L += [f"- {item}" for item in b.items] + [""]
            elif b.kind == "table":
                L += ["| " + " | ".join(b.headers) + " |", "|" + "---|" * len(b.headers)]
                L += ["| " + " | ".join(_md_cell(c) for c in row) + " |" for row in b.rows] + [""]
            elif b.kind == "status":
                L += [f"- {'✅' if ok else '❌'} **{label}:** {detail}" for ok, label, detail in b.rows] + [""]
            elif b.kind == "kv":
                L += [f"**{k}:** {v}  " for k, v in b.rows] + [""]
            elif b.kind == "image":
                b64 = base64.b64encode(b.png).decode("ascii")
                L += [f"**{b.label_text}**", "", f"![{b.label_text}](data:image/png;base64,{b64})", "", f"_{b.text}_", ""]
    return "\n".join(L)


def build_markdown(problem_text: str, model: ProblemModel, report: VerificationReport,
                    steps_by_target: dict[str, list[SolutionStep]], scenarios: list[dict],
                    plot_snapshots: list[PlotSnapshot] | None = None,
                    extras: ReportExtras | None = None, include: list[str] | set[str] | None = None) -> str:
    """The whole report as Markdown. `include` limits it to some sections (see report_content.SECTIONS)."""
    return render_markdown(build_sections(problem_text, model, report, steps_by_target, scenarios,
                                          plot_snapshots, extras, include))


# ---------------------------------------------------------------- PDF

_LATIN1_REPLACEMENTS = {
    "—": "--", "–": "-", "→": "->", "•": "-", "’": "'", "‘": "'",
    "“": '"', "”": '"', "×": "x", "·": "*", "✅": "[PASS]", "❌": "[FAIL]",
    "⚠️": "[!]", "🧮": "",
}


def _safe(text: str) -> str:
    """fpdf2's core (non-embedded) fonts are Latin-1 only. Swap common
    Unicode punctuation/symbols for ASCII equivalents, then hard-fallback
    any remaining unencodable character rather than raising."""
    for k, v in _LATIN1_REPLACEMENTS.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def _render_latex_image(latex_str: str, fontsize: int = 13) -> io.BytesIO | None:
    """Renders a LaTeX-ish string to an in-memory transparent PNG via
    matplotlib's mathtext. Returns None (caller should fall back to plain
    text) if the string uses LaTeX matplotlib's limited mathtext engine
    doesn't support."""
    text = latex_str.strip()
    if not (text.startswith("$") and text.endswith("$")):
        text = f"${text}$"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0)
    try:
        fig.text(0, 0, text, fontsize=fontsize)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                     pad_inches=0.06, transparent=True)
        buf.seek(0)
        return buf
    except Exception:  # noqa: BLE001 -- mathtext can't parse everything sympy emits
        return None
    finally:
        plt.close(fig)


def _add_equation(pdf: FPDF, latex_str: str, fontsize: int = 13, max_h: float = 10):
    img = _render_latex_image(latex_str, fontsize=fontsize)
    if img is not None:
        try:
            pdf.image(img, h=max_h)
            return
        except Exception:  # noqa: BLE001
            pass
    # fallback: plain monospace text if image rendering/placement failed
    pdf.set_font("Courier", "", 10)
    pdf.multi_cell(0, 5, _safe(latex_str), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


_EQ_SIZES = {"plain": (13, 9), "step": (11, 7), "matrix": (12, 16)}      # (font size, max image height)


def _pdf_line(pdf: FPDF, text: str, font: str = "Helvetica", style: str = "", size: int = 10, h: float = 5) -> None:
    pdf.set_font(font, style, size)
    pdf.multi_cell(0, h, _safe(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def render_pdf(sections: list[Section]) -> bytes:
    """PDF for already-built sections. Equations are typeset to images by matplotlib's mathtext."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    _pdf_line(pdf, "Math Representation System", style="B", size=18, h=10)
    pdf.set_text_color(120, 120, 120)
    _pdf_line(pdf, generated_stamp(), size=9)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    for sec in sections:
        _pdf_line(pdf, sec.title, style="B", size=13, h=8)
        for b in sec.blocks:
            if b.kind == "text":
                style = {"italic": "I", "bold": "B"}.get(b.style, "")
                _pdf_line(pdf, b.text, style=style, size=9 if b.style == "italic" else 10,
                          h=6 if b.style == "plain" else 5)
            elif b.kind == "label":
                _pdf_line(pdf, b.text, style="B", size=10 if b.text.startswith("Step ") else 11, h=5 if b.text.startswith("Step ") else 6)
            elif b.kind == "subheading":
                _pdf_line(pdf, b.text, style="B", size=11, h=6)
            elif b.kind == "equation":
                size, height = _EQ_SIZES.get(b.style, _EQ_SIZES["plain"])
                _add_equation(pdf, b.text, fontsize=size, max_h=height)
            elif b.kind == "bullets":
                for item in b.items:
                    _pdf_line(pdf, f"- {item}")
            elif b.kind == "table":
                pdf.set_font("Helvetica", "", 9)
                with pdf.table(first_row_as_headings=True, text_align="LEFT", line_height=5) as table:
                    for row_cells in [b.headers] + b.rows:
                        row = table.row()
                        for cell in row_cells:
                            row.cell(_safe(str(cell)))
                pdf.ln(1)
            elif b.kind == "status":
                for ok, label, detail in b.rows:
                    _pdf_line(pdf, f"{'[PASS]' if ok else '[FAIL]'} {label}: {detail}", size=9)
            elif b.kind == "kv":
                for k, v in b.rows:
                    _pdf_line(pdf, f"{k}: {v}")
            elif b.kind == "image":
                _pdf_line(pdf, b.label_text, style="B", size=11, h=6)
                try:
                    pdf.image(io.BytesIO(b.png), w=170)
                except Exception as e:  # noqa: BLE001
                    _pdf_line(pdf, f"(couldn't embed image: {e})", style="I", size=9)
                _pdf_line(pdf, b.text, style="I", size=9)
        pdf.ln(2)
    return bytes(pdf.output())


def build_pdf_bytes(problem_text: str, model: ProblemModel, report: VerificationReport,
                     steps_by_target: dict[str, list[SolutionStep]], scenarios: list[dict],
                     plot_snapshots: list[PlotSnapshot] | None = None,
                     extras: ReportExtras | None = None, include: list[str] | set[str] | None = None) -> bytes:
    """The whole report as a PDF. `include` limits it to some sections (see report_content.SECTIONS)."""
    return render_pdf(build_sections(problem_text, model, report, steps_by_target, scenarios,
                                     plot_snapshots, extras, include))


def build_batch_markdown(results: list) -> str:
    """Combines a whole batch's worth of BatchItemResult (see
    batch_solver.py) into one Markdown document: a summary table up
    front, then each problem's own build_markdown() output verbatim
    under its own numbered heading. Problems that errored out entirely
    (no model/report at all) get a short "couldn't be solved" note
    instead of a blank section, rather than being silently omitted."""
    from modules.batch_solver import batch_summary  # local import: avoids a module-load cycle
                                                        # (batch_solver imports verifier/solver,
                                                        # which this module also imports)

    L = ["# Math Representation System -- Batch Report",
         f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*", ""]
    summary = batch_summary(results)
    L.append(f"**{summary['solved']}/{summary['total']} solved** "
              f"({summary['needed_retry']} needed a retry, {summary['failed']} failed)")
    L.append("")
    L.append("| # | Problem | Status |")
    L.append("|---|---|---|")
    for r in results:
        preview = r.problem_text.strip().replace("\n", " ")[:70]
        if r.error:
            status = f"❌ Error: {r.error[:60]}"
        elif r.report and r.report.passed:
            status = "✅ Solved" + (f" ({r.retries} retr{'y' if r.retries==1 else 'ies'})" if r.retries else "")
        else:
            status = "⚠️ Verification failed"
        L.append(f"| {r.index + 1} | {preview}{'...' if len(r.problem_text.strip()) > 70 else ''} | {status} |")
    L.append("")

    for r in results:
        L.append("---")
        L.append("")
        L.append(f"# Problem {r.index + 1} of {len(results)}")
        L.append("")
        if r.error or r.model is None or r.report is None:
            L.append(f"**Could not be solved.** {r.error or ''}")
            L.append("")
            L.append("Original text:")
            L.append("")
            L.append(r.problem_text.strip())
            L.append("")
            continue
        # strip the per-problem title line (build_markdown's own "#
        # Math Representation System" header) so each problem doesn't
        # repeat the document title inside the batch report
        single_md = build_markdown(r.problem_text, r.model, r.report, r.steps, [])
        body = "\n".join(single_md.splitlines()[2:])  # drop title + timestamp lines
        L.append(body)

    return "\n".join(L)


def build_batch_pdf_bytes(results: list) -> bytes:
    """Merges each problem's own build_pdf_bytes() output (each already
    a complete, valid single-problem PDF) into one combined document via
    pypdf, with a one-page summary cover prepended. Byte-concatenating
    separate PDFs would produce a corrupt file -- a real merge (appending
    parsed pages) is required, hence the pypdf dependency."""
    from pypdf import PdfWriter, PdfReader
    from modules.batch_solver import batch_summary

    summary = batch_summary(results)
    cover = FPDF()
    cover.set_auto_page_break(auto=True, margin=15)
    cover.add_page()
    cover.set_font("Helvetica", "B", 16)
    cover.multi_cell(0, 10, "Math Representation System -- Batch Report",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    cover.set_font("Helvetica", "", 10)
    cover.multi_cell(0, 6, _safe(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    cover.ln(4)
    cover.set_font("Helvetica", "B", 12)
    cover.multi_cell(0, 6, _safe(f"{summary['solved']}/{summary['total']} solved "
                                   f"({summary['needed_retry']} needed a retry, "
                                   f"{summary['failed']} failed)"),
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    cover.ln(2)
    cover.set_font("Helvetica", "", 10)
    for r in results:
        preview = r.problem_text.strip().replace("\n", " ")[:80]
        status = ("Error" if r.error else "Solved" if (r.report and r.report.passed) else "Verification failed")
        cover.multi_cell(0, 6, _safe(f"{r.index + 1}. [{status}] {preview}"),
                           new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    writer = PdfWriter()
    writer.append(PdfReader(io.BytesIO(bytes(cover.output()))))
    for r in results:
        if r.error or r.model is None or r.report is None:
            continue
        try:
            single_pdf = build_pdf_bytes(r.problem_text, r.model, r.report, r.steps, [])
            writer.append(PdfReader(io.BytesIO(single_pdf)))
        except Exception:  # noqa: BLE001
            continue  # one bad problem's PDF shouldn't sink the whole batch export

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()
