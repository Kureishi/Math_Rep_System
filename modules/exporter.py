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
from dataclasses import dataclass
from datetime import datetime

import sympy as sp
import matplotlib
matplotlib.use("Agg")  # headless -- no display needed, safe in a server context
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from modules.equation_engine import ProblemModel
from modules.verifier import VerificationReport, _known_substitutions
from modules.solver import SolutionStep
from modules.matrix_utils import linear_system_view
from modules.vector_utils import vector_summary
from modules.unit_conversion import sweep_conversions


@dataclass
class PlotSnapshot:
    """A user-chosen static capture of an interactive plot, for inclusion
    in an exported report -- the interactive version lives only in the
    browser session, so this is the opt-in way to get a specific view
    (with whatever parameter values were selected at the time) into a
    document. `caption` should describe exactly what's shown, e.g. which
    equation, which axes, and what any fixed slider values were, since a
    static image alone doesn't carry that context."""
    title: str
    caption: str
    png_bytes: bytes


# ---------------------------------------------------------------- Markdown

def build_markdown(problem_text: str, model: ProblemModel, report: VerificationReport,
                    steps_by_target: dict[str, list[SolutionStep]], scenarios: list[dict],
                    plot_snapshots: list[PlotSnapshot] | None = None) -> str:
    L = []
    L.append("# Math Representation System -- Solved Problem")
    L.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    L.append("")
    L.append("## Problem")
    L.append("")
    L.append(problem_text.strip())
    L.append("")
    L.append(f"**Domain:** {model.problem_domain}  ")
    status = "✅ Passed" if report.passed else "⚠️ Issues found -- review before trusting"
    L.append(f"**Self-verification:** {status}")
    L.append("")

    L.append("## Derived equations")
    L.append("")
    for eq in model.equations:
        latex_str = sp.latex(eq.sympy_eq) if eq.sympy_eq is not None else eq.raw_expression
        L.append(f"**{eq.name}**")
        L.append("")
        L.append(f"$$ {latex_str} $$")
        L.append("")
        L.append(eq.derivation)
        L.append("")

    if model.assumptions:
        L.append("**Assumptions:**")
        L.extend(f"- {a}" for a in model.assumptions)
        L.append("")

    L.append("## Variables")
    L.append("")
    L.append("| Symbol | Meaning | Known value | Unit |")
    L.append("|---|---|---|---|")
    for v in model.variables:
        kv = v.known_value if v.known_value is not None else "_(solved)_"
        L.append(f"| {v.symbol} | {v.meaning} | {kv} | {v.unit or ''} |")
    L.append("")

    matrix_result = linear_system_view(model, _known_substitutions(model))
    if matrix_result is not None:
        L.append("## Matrix representation")
        L.append("")
        x_latex = sp.latex(sp.Matrix([sp.Symbol(s) for s in matrix_result.symbols]))
        L.append(f"$$ {sp.latex(matrix_result.A)} {x_latex} = {sp.latex(matrix_result.b)} $$")
        L.append("")
        if matrix_result.is_square:
            L.append(f"**det(A) = {sp.latex(matrix_result.determinant)}**")
            L.append("")
            if matrix_result.eigenvalues:
                eig_text = ", ".join(
                    f"{sp.latex(val)}" + (f" (x{mult})" if mult > 1 else "")
                    for val, mult in matrix_result.eigenvalues.items())
                L.append(f"Eigenvalues: {eig_text}")
                L.append("")
        L.append(matrix_result.classification)
        L.append("")

    vector_vars = [v for v in model.variables if v.is_vector and v.components]
    if vector_vars:
        L.append("## Vectors")
        L.append("")
        knowns = _known_substitutions(model)
        for v in vector_vars:
            summary = vector_summary(v.symbol, v.components, knowns)
            if summary:
                comp_str = ", ".join(f"{c}={val:g}" for c, val in summary["components"].items())
                L.append(f"- **{v.symbol}** ({v.meaning}): {comp_str} -- "
                          f"magnitude = {summary['magnitude']:.6g} {v.unit or ''}")
            else:
                L.append(f"- **{v.symbol}** ({v.meaning}): components {', '.join(v.components)}")
        L.append("")

    if report.sympy_numeric_answers:
        conv_lines = []
        for target, val in report.sympy_numeric_answers.items():
            unit = next((v.unit for v in model.variables if v.symbol == target), None)
            alternates = sweep_conversions(val, unit)
            if alternates:
                alt_text = ", ".join(f"{av:.6g} {au}" for au, av in alternates)
                conv_lines.append(f"- **{target}** = {val:.6g} {unit} = {alt_text}")
        if conv_lines:
            L.append("## Results in other units")
            L.append("")
            L.extend(conv_lines)
            L.append("")

    cr = report.confidence_report()
    L.append("## Confidence report")
    L.append("")
    L.append(f"**Overall score: {cr.score:.0%}** ({cr.label}) -- {cr.passed_count}/{cr.total_count} "
              f"checks passed.")
    L.append("")
    for cat, summary in cr.categories.items():
        mark = "✅" if summary.all_passed else "❌"
        L.append(f"- {mark} **{cat}:** {summary.passed}/{summary.total}")
    L.append("")
    if cr.critical_failures:
        L.append("**Critical failures:**")
        for c in cr.critical_failures:
            L.append(f"- {c.label}: {c.detail}")
        L.append("")

    if report.domain_notes:
        L.append("## Domain of validity")
        L.append("")
        for note in report.domain_notes:
            if note.violated:
                L.append(f"- ❌ **{note.equation}** -- undefined with the given values: " +
                          "; ".join(r.description for r in note.violated))
            ok = note.satisfied + note.pending
            if ok:
                L.append(f"- **{note.equation}** requires: " + "; ".join(r.description for r in ok))
        L.append("")

    L.append("## Verification detail")
    L.append("")
    for c in report.checks:
        mark = "✅" if c.passed else "❌"
        L.append(f"- {mark} **{c.label}:** {c.detail}")
    L.append("")

    if steps_by_target:
        L.append("## Step-by-step solution")
        L.append("")
        for target, steps in steps_by_target.items():
            L.append(f"### Solving for `{target}`")
            L.append("")
            for i, s in enumerate(steps, 1):
                L.append(f"**Step {i}: {s.description}**")
                L.append("")
                L.append(f"$$ {s.expression} $$")
                if s.explanation:
                    L.append("")
                    L.append(f"_{s.explanation}_")
                L.append("")

    if plot_snapshots:
        L.append("## Plots")
        L.append("")
        for snap in plot_snapshots:
            b64 = base64.b64encode(snap.png_bytes).decode("ascii")
            L.append(f"**{snap.title}**")
            L.append("")
            L.append(f"![{snap.title}](data:image/png;base64,{b64})")
            L.append("")
            L.append(f"_{snap.caption}_")
            L.append("")

    if scenarios and not any("error" in s for s in scenarios):
        L.append("## Where else this applies")
        L.append("")
        for s in scenarios:
            L.append(f"- **{s.get('scenario', '')}** -- {s.get('mapping', '')}")
        L.append("")

    return "\n".join(L)


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


def build_pdf_bytes(problem_text: str, model: ProblemModel, report: VerificationReport,
                     steps_by_target: dict[str, list[SolutionStep]], scenarios: list[dict],
                     plot_snapshots: list[PlotSnapshot] | None = None) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(0, 10, _safe("Math Representation System"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, _safe(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(0, 7, "Problem", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6, _safe(problem_text.strip()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)

    pdf.set_font("Helvetica", "B", 11)
    status = "PASSED" if report.passed else "ISSUES FOUND -- review before trusting"
    pdf.multi_cell(0, 6, _safe(f"Domain: {model.problem_domain}    |    Self-verification: {status}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 8, "Derived equations", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for eq in model.equations:
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 6, _safe(eq.name), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        latex_str = sp.latex(eq.sympy_eq) if eq.sympy_eq is not None else eq.raw_expression
        _add_equation(pdf, latex_str, fontsize=13, max_h=9)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _safe(eq.derivation), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    if model.assumptions:
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 6, "Assumptions", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        for a in model.assumptions:
            pdf.multi_cell(0, 5, _safe(f"- {a}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 8, "Variables", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    for v in model.variables:
        kv = v.known_value if v.known_value is not None else "(solved)"
        pdf.multi_cell(0, 5, _safe(f"{v.symbol} -- {v.meaning}: {kv} {v.unit or ''}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    matrix_result = linear_system_view(model, _known_substitutions(model))
    if matrix_result is not None:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, "Matrix representation", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        x_latex = sp.latex(sp.Matrix([sp.Symbol(s) for s in matrix_result.symbols]))
        _add_equation(pdf, f"{sp.latex(matrix_result.A)} {x_latex} = {sp.latex(matrix_result.b)}",
                       fontsize=12, max_h=16)
        pdf.set_font("Helvetica", "", 10)
        if matrix_result.is_square:
            pdf.multi_cell(0, 5, _safe(f"det(A) = {matrix_result.determinant}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if matrix_result.eigenvalues:
                eig_text = ", ".join(f"{val}" + (f" (x{mult})" if mult > 1 else "")
                                       for val, mult in matrix_result.eigenvalues.items())
                pdf.multi_cell(0, 5, _safe(f"Eigenvalues: {eig_text}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.multi_cell(0, 5, _safe(matrix_result.classification), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    vector_vars = [v for v in model.variables if v.is_vector and v.components]
    if vector_vars:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, "Vectors", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        knowns = _known_substitutions(model)
        for v in vector_vars:
            summary = vector_summary(v.symbol, v.components, knowns)
            if summary:
                comp_str = ", ".join(f"{c}={val:g}" for c, val in summary["components"].items())
                pdf.multi_cell(0, 5, _safe(f"{v.symbol} ({v.meaning}): {comp_str} -- "
                                            f"magnitude = {summary['magnitude']:.6g} {v.unit or ''}"),
                                new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            else:
                pdf.multi_cell(0, 5, _safe(f"{v.symbol} ({v.meaning}): components "
                                            f"{', '.join(v.components)}"),
                                new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    if report.sympy_numeric_answers:
        conv_written = False
        for target, val in report.sympy_numeric_answers.items():
            unit = next((v.unit for v in model.variables if v.symbol == target), None)
            alternates = sweep_conversions(val, unit)
            if alternates:
                if not conv_written:
                    pdf.set_font("Helvetica", "B", 13)
                    pdf.multi_cell(0, 8, "Results in other units", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.set_font("Helvetica", "", 10)
                    conv_written = True
                alt_text = ", ".join(f"{av:.6g} {au}" for au, av in alternates)
                pdf.multi_cell(0, 5, _safe(f"{target} = {val:.6g} {unit} = {alt_text}"),
                                new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        if conv_written:
            pdf.ln(2)

    cr = report.confidence_report()
    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 8, "Confidence report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _safe(f"Overall score: {cr.score:.0%} ({cr.label}) -- "
                                f"{cr.passed_count}/{cr.total_count} checks passed."),
                    new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for cat, summary in cr.categories.items():
        mark = "[OK]" if summary.all_passed else "[!!]"
        pdf.multi_cell(0, 5, _safe(f"{mark} {cat}: {summary.passed}/{summary.total}"),
                        new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if cr.critical_failures:
        pdf.set_font("Helvetica", "B", 10)
        pdf.multi_cell(0, 5, "Critical failures:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        for c in cr.critical_failures:
            pdf.multi_cell(0, 5, _safe(f"- {c.label}: {c.detail}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    if report.domain_notes:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, "Domain of validity", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        for note in report.domain_notes:
            if note.violated:
                pdf.multi_cell(0, 5, _safe(f"[FAIL] {note.equation} -- undefined with the given "
                                            "values: " + "; ".join(r.description for r in note.violated)),
                                new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            ok = note.satisfied + note.pending
            if ok:
                pdf.multi_cell(0, 5, _safe(f"{note.equation} requires: " +
                                            "; ".join(r.description for r in ok)),
                                new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 8, "Verification detail", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    for c in report.checks:
        mark = "[PASS]" if c.passed else "[FAIL]"
        pdf.multi_cell(0, 5, _safe(f"{mark} {c.label}: {c.detail}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    if steps_by_target:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, "Step-by-step solution", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for target, steps in steps_by_target.items():
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 6, _safe(f"Solving for {target}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            for i, s in enumerate(steps, 1):
                pdf.set_font("Helvetica", "B", 10)
                pdf.multi_cell(0, 5, _safe(f"Step {i}: {s.description}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                _add_equation(pdf, s.expression, fontsize=11, max_h=7)
                if s.explanation:
                    pdf.set_font("Helvetica", "I", 9)
                    pdf.multi_cell(0, 5, _safe(s.explanation), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(1)

    if plot_snapshots:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, "Plots", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for snap in plot_snapshots:
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(0, 6, _safe(snap.title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            try:
                pdf.image(io.BytesIO(snap.png_bytes), w=170)
            except Exception as e:  # noqa: BLE001
                pdf.set_font("Helvetica", "I", 9)
                pdf.multi_cell(0, 5, _safe(f"(couldn't embed image: {e})"),
                                new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(0, 5, _safe(snap.caption), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(2)

    if scenarios and not any("error" in s for s in scenarios):
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, "Where else this applies", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        for s in scenarios:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, _safe(f"- {s.get('scenario', '')}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if s.get("mapping"):
                pdf.set_font("Helvetica", "I", 9)
                pdf.multi_cell(0, 5, _safe(f"  {s['mapping']}"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())
