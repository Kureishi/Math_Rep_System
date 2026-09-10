"""
Concept index: tags a solved problem by the named CONCEPTS it touches
(via named_formulas.py's recognizer, plus a domain-label fallback),
rather than by raw equation SHAPE the way similarity.py does.

The distinction matters for a research workflow specifically:
similarity.py answers "what past problem looked structurally like THIS
one" (useful for reusing a derivation). A research session more often
wants the other direction -- "show me every solved problem that
touched conservation of energy" -- across however many different
SHAPES that concept actually took (KE=1/2mv^2 alone, a work-energy
theorem application, a spring-block oscillator, ...), which shape-only
similarity can't answer since those don't canonicalize to the same
fingerprint at all. This module is what makes that second kind of
query possible, and history.py is where the tags actually get stored
and queried across a session's worth of solved problems.

A problem can (and often does) carry more than one tag -- e.g. a single
multi-equation problem that derives BOTH kinetic energy and momentum.
A problem with no recognized named formula still gets a
"domain: <problem_domain>" fallback tag, so it's still browsable by
concept rather than silently falling out of the index.
"""
from modules.equation_engine import ProblemModel
from modules.named_formulas import recognize_formula


def concept_tags_for_model(model: ProblemModel) -> list[str]:
    """Every distinct named-formula match across the model's
    equation-kind relations, using the SAME context-building approach
    (domain label + the equation's own variable MEANINGS) the
    equation-detail view already uses to disambiguate shape collisions
    -- see named_formulas.py's module docstring for why domain alone
    isn't enough (F=m*a, p=m*v, and W=F*d, e.g., are all "y = a*b" once
    canonicalized). Falls back to a "domain: X" tag when nothing is
    recognized, so every problem is tagged with SOMETHING browsable."""
    var_by_symbol = {v.symbol: v for v in model.variables}
    tags: set[str] = set()
    for eq in model.equations:
        if eq.kind != "equation" or eq.sympy_eq is None:
            continue
        eq_symbols = {s.name for s in eq.sympy_eq.free_symbols}
        meanings_text = " ".join(
            var_by_symbol[s].meaning for s in eq_symbols
            if s in var_by_symbol and var_by_symbol[s].meaning
        )
        context_text = f"{model.problem_domain or ''} {meanings_text}"
        for name, _ in recognize_formula(eq, context_text):
            tags.add(name)

    if not tags and model.problem_domain:
        tags.add(f"domain: {model.problem_domain}")

    return sorted(tags)
