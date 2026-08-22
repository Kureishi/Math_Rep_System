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
import io
from datetime import datetime

import sympy as sp
import matplotlib
matplotlib.use("Agg")  # headless -- no display needed, safe in a server context
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.enums import XPos, YPos

from modules.equation_engine import ProblemModel
from modules.verifier import VerificationReport
from modules.solver import SolutionStep


# ---------------------------------------------------------------- Markdown

def build_markdown(problem_text: str, model: ProblemModel, report: VerificationReport,
                    steps_by_target: dict[str, list[SolutionStep]], scenarios: list[dict]) -> str:
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
                     steps_by_target: dict[str, list[SolutionStep]], scenarios: list[dict]) -> bytes:
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
