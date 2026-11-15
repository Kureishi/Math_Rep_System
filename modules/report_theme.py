"""
Report themes -- Streamlit-free. A PRESET (modules/report_content.PRESETS) says which sections go in; a THEME says how the report reads:
its title, the order and the headings of its sections, whether figures are numbered and captioned automatically, and whether it is compact.

    Lab report         Objective, Theory, Method and analysis, Results, Figures, Uncertainty, Verification, Appendix (proof card + replay manifest)
    Homework solution  Problem, Given, Equations used, Solution, Answer, Check, Appendix
    One-page summary   Problem, Answer, Key equations, Verdict with a one-line proof reference; short enough for one page

A theme only reorders, renames and trims what the report already contains; it never adds a claim that is not in the underlying sections.
"""
from dataclasses import dataclass, field

from modules.report_content import Block, Section


@dataclass(frozen=True)
class ReportTheme:
    name: str
    title: str
    order: tuple[str, ...]
    titles: dict[str, str] = field(default_factory=dict)
    auto_captions: bool = True
    compact: bool = False
    note: str = ""


LAB_REPORT = ReportTheme(
    "Lab report", "Lab Report",
    ("problem", "equations", "variables", "steps", "results", "units", "plots", "sensitivity", "confidence", "domain", "verification", "scenarios", "followups", "tutor", "vectors", "matrix",
     "provenance", "appendix"),
    {"problem": "Objective", "equations": "Theory", "variables": "Quantities and units", "steps": "Method and analysis", "results": "Results", "plots": "Figures",
     "sensitivity": "Uncertainty analysis", "confidence": "Confidence in the result", "verification": "Verification and error checks", "provenance": "Software and reproducibility",
     "appendix": "Appendix A: Proof card and replay manifest"},
    note="Reads like a lab write-up: objective, theory, method, results, uncertainty, verification, and an appendix that lets anyone re-verify the result.")

HOMEWORK = ReportTheme(
    "Homework solution", "Homework Solution",
    ("problem", "variables", "equations", "steps", "results", "units", "confidence", "verification", "plots", "appendix"),
    {"variables": "Given", "equations": "Equations used", "steps": "Solution", "results": "Answer", "units": "Answer in other units", "confidence": "Check",
     "verification": "How each check came out", "appendix": "Appendix: Proof card and replay manifest"},
    note="Problem, what is given, the equations, the worked solution and the answer, with a short check at the end.")

ONE_PAGE = ReportTheme(
    "One-page summary", "Summary",
    ("problem", "results", "equations", "confidence", "appendix"),
    {"results": "Answer", "equations": "Key equations", "confidence": "Verdict", "appendix": "Proof reference"},
    auto_captions=False, compact=True,
    note="The problem, the answer and the verdict on a single page, with a one-line reference to the proof.")

THEMES: dict[str, ReportTheme] = {t.name: t for t in (LAB_REPORT, HOMEWORK, ONE_PAGE)}

# the section selection each theme starts from (the Share dialog's preset list gets these entries)
THEME_SECTIONS: dict[str, list[str]] = {
    "Lab report": ["problem", "equations", "variables", "steps", "results", "units", "plots", "sensitivity", "confidence", "verification", "provenance", "appendix"],
    "Homework solution": ["problem", "variables", "equations", "steps", "results", "units", "confidence", "appendix"],
    "One-page summary": ["problem", "results", "equations", "confidence", "appendix"],
}

MAX_COMPACT_EQUATIONS = 3
MAX_COMPACT_PROBLEM = 420


def _compact(section: Section) -> Section:
    if section.id == "problem":
        blocks = []
        for b in section.blocks:
            if b.kind == "text" and len(b.text) > MAX_COMPACT_PROBLEM:
                b = Block("text", b.text[:MAX_COMPACT_PROBLEM].rstrip() + " …", style=b.style)
            blocks.append(b)
        return Section(section.id, section.title, blocks)
    if section.id == "equations":
        keep: list[Block] = []
        taken = 0
        for i, b in enumerate(section.blocks):
            if b.kind == "equation" and taken < MAX_COMPACT_EQUATIONS:
                prev = section.blocks[i - 1] if i else None
                if prev is not None and prev.kind == "label":
                    keep.append(prev)
                keep.append(b)
                taken += 1
        return Section(section.id, section.title, keep)
    if section.id == "confidence":
        keep = [b for b in section.blocks if b.kind == "text"][:1]
        return Section(section.id, section.title, keep or section.blocks[:1])
    return section


def apply_theme(sections: list[Section], theme: ReportTheme) -> list[Section]:
    """The sections in the theme's order, under its headings (and trimmed, for a compact theme)."""
    rank = {sid: i for i, sid in enumerate(theme.order)}
    ordered = sorted(sections, key=lambda s: rank.get(s.id, len(rank)))
    out = []
    for s in ordered:
        s = Section(s.id, theme.titles.get(s.id, s.title), s.blocks)
        out.append(_compact(s) if theme.compact else s)
    return out
