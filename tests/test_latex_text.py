"""A sentence handed to a math renderer loses its spaces ("linearinthetarget:..."): prose steps are wrapped in \\text{...} (modules/latex_text.py)."""
import re

import pytest
import sympy as sp

from modules import algebra_rules
from modules.latex_text import display_latex, looks_like_prose, prose_to_latex

SENTENCE = "linear in the target: move every other term to the other side, then divide by the target's coefficient"


def test_a_sentence_is_wrapped_as_text_and_keeps_its_spaces():
    out = display_latex(SENTENCE)
    assert out.startswith(r"\begin{gathered}") and r"\text{" in out
    assert re.sub(r"\\text\{|\}|\\begin\{gathered\}|\\end\{gathered\}|\\\\", "", out).replace("  ", " ").count("move every other term") == 1


def test_real_math_is_left_alone():
    for expr in (r"A = \frac{1581}{25}", "x = 2", r"\sqrt{x^2 + 1}", "F = m a", r"v = \text{a sentence} + 1", r"\text{already text}", ""):
        assert display_latex(expr) == expr and not looks_like_prose(expr)


def test_it_is_idempotent_and_escapes_special_characters():
    once = display_latex(SENTENCE)
    assert display_latex(once) == once
    assert prose_to_latex("rate is 5% of the total & more_stuff here") == r"\text{rate is 5\% of the total \& more\_stuff here}"
    assert prose_to_latex("short", width=None) == r"\text{short}" and "gathered" not in prose_to_latex("two words only here")


def test_every_technique_the_solver_can_name_is_detected_as_prose():
    x, y = sp.symbols("x y")
    equations = [sp.Eq(x, 3), sp.Eq(2 * x + y, 5), sp.Eq(x**2 + x, 1), sp.Eq(x**3, 8), sp.Eq(sp.sqrt(x), y), sp.Eq(sp.sin(x), y), sp.Eq(x, x + 1),
                 sp.Eq(1 / x, y), sp.Eq(x * y, 1), sp.Eq(sp.exp(x), y), sp.Eq(sp.log(x), y), sp.Eq(x**2, y), sp.Eq(x, x)]
    seen = set()
    for eq in equations:
        text = algebra_rules.classify_isolation(eq, x)
        seen.add(text)
        assert looks_like_prose(text) or len(text.split()) < 3, text
        assert " " in display_latex(text) or "text" in display_latex(text)
    assert len(seen) >= 4


def test_streamlit_steps_and_the_exports_use_it():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for rel in ("ui/results/steps.py", "ui/results/explore_tab.py"):
        src = (root / rel).read_text(encoding="utf-8")
        assert "st.latex(step.expression)" not in src and "display_latex(step.expression)" in src, rel


def test_the_report_draws_a_prose_step_as_plain_text_not_typeset_math():
    from modules import report_content
    from modules.solver import SolutionStep
    section = report_content._steps({"A": [SolutionStep(description="Technique", expression=SENTENCE),
                                          SolutionStep(description="Isolate", expression=r"A = \frac{1581}{25}")]})
    kinds = [(b.kind, b.style, b.text) for b in section.blocks if b.kind in ("text", "equation")]
    assert kinds == [("equation", "raw", SENTENCE), ("equation", "step", r"A = \frac{1581}{25}")]


def test_the_latex_share_export_wraps_prose_in_text():
    from modules.share_text import display_latex as in_share
    assert in_share(SENTENCE, None) == r"\text{" + SENTENCE + "}"
