"""Copy-ready LaTeX and plain text (modules/share_text.py)."""
import json
import re

import pytest

from modules.equation_engine import build_model
from modules.share_text import _unit_tex, build_latex, build_plain_text, tex_escape
from modules.solve_compare import deterministic_report
from modules.solver import compute_steps
from tests.conftest import KINEMATICS_TWO_TARGET_JSON


@pytest.fixture
def solved():
    model = build_model(json.loads(KINEMATICS_TWO_TARGET_JSON))
    return model, deterministic_report(model)


def test_latex_has_the_derived_equations_and_the_answers_with_units(solved):
    model, report = solved
    tex = build_latex(model, report)
    assert tex.count(r"\begin{gather*}") == 2 and tex.count(r"\end{gather*}") == 2
    assert "v = a t + u" in tex
    assert r"a = 2\,\text{m/s}^{2}" in tex
    assert r"d = 84\,\text{m}" in tex


def test_latex_environments_are_balanced(solved):
    model, report = solved
    tex = build_latex(model, report, compute_steps(model), include_steps=True)
    assert len(re.findall(r"\\begin\{", tex)) == len(re.findall(r"\\end\{", tex))
    assert tex.count(r"\begin{gather*}") == 4          # equations, answers, one per target's steps


def test_steps_are_only_included_when_asked(solved):
    model, report = solved
    steps = compute_steps(model)
    assert "Solving for" not in build_latex(model, report, steps)
    assert "Solving for a" in build_latex(model, report, steps, include_steps=True)


def test_an_empty_model_says_so_instead_of_emitting_empty_environments():
    model = build_model({"problem_domain": "x", "variables": [], "equations": [], "solve_for": [], "assumptions": []})
    from modules.verifier import VerificationReport
    tex = build_latex(model, VerificationReport())
    assert r"\begin" not in tex and "nothing to export" in tex


@pytest.mark.parametrize("text,expected", [
    ("a_b", r"a\_b"), ("50%", r"50\%"), ("R&D", r"R\&D"), ("{x}", r"\{x\}"), ("$5", r"\$5"),
    ("a\\b", r"a\textbackslash{}b"), ("plain", "plain"),
])
def test_special_characters_are_escaped(text, expected):
    assert tex_escape(text) == expected


@pytest.mark.parametrize("unit,expected", [
    (None, ""), ("", ""), ("unitless", ""), ("dimensionless", ""), ("1", ""),
    ("m/s^2", r"\,\text{m/s}^{2}"), ("m**2", r"\,\text{m}^{2}"), ("s", r"\,\text{s}"),
    ("m_x", r"\,\text{m\_x}"),
])
def test_units_become_latex_text_with_exponents(unit, expected):
    assert _unit_tex(unit) == expected


def test_large_and_small_numbers_use_scientific_notation_in_latex():
    from modules.share_text import _number_tex
    assert _number_tex(1.5e-7) == r"1.5\times 10^{-7}"
    assert _number_tex(2.5e9) == r"2.5\times 10^{9}"
    assert _number_tex(12.5) == "12.5"


def test_plain_text_lists_answers_units_verdict_and_assumptions(solved):
    model, report = solved
    text = build_plain_text(model, report, "A car accelerates.")
    assert text.startswith("A car accelerates.")
    assert "  a = 2 m/s^2" in text and "  d = 84 m" in text
    assert "Verification: passed" in text and "Assumptions: acceleration is uniform" in text


def test_plain_text_without_numeric_answers_says_so(solved):
    model, report = solved
    report.sympy_numeric_answers = {}
    assert "no single numeric answer" in build_plain_text(model, report)
