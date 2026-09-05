"""
Extraction diff: given two ProblemModels extracted from two DIFFERENT
(but intended-to-mean-the-same) wordings of a problem, produces a
structural, side-by-side diff -- which variables matched, which
appeared in only one extraction, which equation SHAPES matched (via
similarity.py's canonicalize_equation, so a variable rename alone never
counts as a difference) and which didn't -- rather than only the single
similarity score self_consistency.py already computes.

Distinct from self_consistency.py, which repeats extraction on the SAME
wording multiple times to check whether the extractor agrees with
itself, and reports one Jaccard shapes_match number per run. This
module answers the debugging question THAT raises but doesn't answer:
given two SPECIFIC wordings, what EXACTLY differs between their
extractions? A developer-facing debugging tool for understanding why
the extractor is fragile to phrasing, not a student-facing feature.

Variables are matched primarily by their normalized MEANING text (e.g.
"initial velocity"), not by symbol name -- two extractions of the same
underlying quantity commonly pick different symbol letters (v_i vs v0),
and a name-only match would report that as a spurious difference every
single time, drowning out the differences that actually matter.
"""
import re
from dataclasses import dataclass, field

from modules.equation_engine import ProblemModel
from modules.similarity import canonicalize_equation, problem_shape, jaccard_similarity


def _normalize_meaning(meaning: str) -> str:
    return re.sub(r"\s+", " ", (meaning or "").strip().lower())


@dataclass
class VariableDiffEntry:
    status: str                  # "matched" | "changed" | "only_in_a" | "only_in_b"
    symbol_a: str | None = None
    symbol_b: str | None = None
    detail: str = ""


@dataclass
class EquationDiffEntry:
    status: str                  # "matched" | "only_in_a" | "only_in_b"
    shape: str
    name_a: str | None = None
    name_b: str | None = None


@dataclass
class ExtractionDiff:
    domain_a: str
    domain_b: str
    domain_matches: bool
    variables: list[VariableDiffEntry] = field(default_factory=list)
    equations: list[EquationDiffEntry] = field(default_factory=list)
    solve_for_meanings_a: set = field(default_factory=set)
    solve_for_meanings_b: set = field(default_factory=set)
    solve_for_matches: bool = False
    equation_shape_similarity: float = 0.0    # same Jaccard score self_consistency.py uses


def diff_variables(model_a: ProblemModel, model_b: ProblemModel) -> list[VariableDiffEntry]:
    a_by_meaning = {_normalize_meaning(v.meaning): v for v in model_a.variables}
    b_by_meaning = {_normalize_meaning(v.meaning): v for v in model_b.variables}

    entries: list[VariableDiffEntry] = []
    for norm, va in a_by_meaning.items():
        vb = b_by_meaning.get(norm)
        if vb is None:
            entries.append(VariableDiffEntry("only_in_a", symbol_a=va.symbol, detail=va.meaning))
            continue
        # a different symbol LETTER between two independently-named
        # extractions is expected and not itself a meaningful
        # difference (that's the whole reason matching is done by
        # meaning, not by symbol) -- only substantive attributes count
        # toward "changed"; the symbol mapping is still shown, just as
        # context rather than as a flagged diff
        diffs = []
        if va.known_value != vb.known_value:
            diffs.append(f"known value {va.known_value!r} vs {vb.known_value!r}")
        if (va.unit or None) != (vb.unit or None):
            diffs.append(f"unit '{va.unit}' vs '{vb.unit}'")
        if va.domain != vb.domain:
            diffs.append(f"domain '{va.domain}' vs '{vb.domain}'")
        status = "changed" if diffs else "matched"
        if va.symbol != vb.symbol:
            diffs.insert(0, f"symbol '{va.symbol}' vs '{vb.symbol}'")
        entries.append(VariableDiffEntry(status, va.symbol, vb.symbol, "; ".join(diffs)))

    for norm, vb in b_by_meaning.items():
        if norm not in a_by_meaning:
            entries.append(VariableDiffEntry("only_in_b", symbol_b=vb.symbol, detail=vb.meaning))

    return entries


def diff_equations(model_a: ProblemModel, model_b: ProblemModel) -> list[EquationDiffEntry]:
    def shapes_by_name(model: ProblemModel) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for eq in model.equations:
            if eq.kind not in ("equation", "ode", "recurrence") or eq.sympy_eq is None:
                continue
            try:
                shape = f"{eq.kind}:{canonicalize_equation(eq.sympy_eq)}"
            except Exception:  # noqa: BLE001
                continue
            out.setdefault(shape, []).append(eq.name)
        return out

    shapes_a = shapes_by_name(model_a)
    shapes_b = shapes_by_name(model_b)

    entries: list[EquationDiffEntry] = []
    for shape, names_a in shapes_a.items():
        if shape in shapes_b:
            entries.append(EquationDiffEntry("matched", shape, names_a[0], shapes_b[shape][0]))
        else:
            entries.append(EquationDiffEntry("only_in_a", shape, name_a=names_a[0]))
    for shape, names_b in shapes_b.items():
        if shape not in shapes_a:
            entries.append(EquationDiffEntry("only_in_b", shape, name_b=names_b[0]))
    return entries


def _solve_for_meanings(model: ProblemModel) -> set[str]:
    var_by_symbol = {v.symbol: v for v in model.variables}
    return {_normalize_meaning(var_by_symbol[s].meaning) for s in model.solve_for if s in var_by_symbol}


def diff_extractions(model_a: ProblemModel, model_b: ProblemModel) -> ExtractionDiff:
    """The main entry point: a full structural diff between two
    independently-extracted models of (nominally) the same problem."""
    solve_for_a = _solve_for_meanings(model_a)
    solve_for_b = _solve_for_meanings(model_b)
    shape_a = problem_shape(model_a)
    shape_b = problem_shape(model_b)

    return ExtractionDiff(
        domain_a=model_a.problem_domain, domain_b=model_b.problem_domain,
        domain_matches=_normalize_meaning(model_a.problem_domain) == _normalize_meaning(model_b.problem_domain),
        variables=diff_variables(model_a, model_b),
        equations=diff_equations(model_a, model_b),
        solve_for_meanings_a=solve_for_a, solve_for_meanings_b=solve_for_b,
        solve_for_matches=solve_for_a == solve_for_b,
        equation_shape_similarity=jaccard_similarity(shape_a, shape_b),
    )
