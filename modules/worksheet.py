"""
Reverse generation: inverts scenarios.py's "explain where else this
structure shows up" into "write me a NEW problem that uses this same
structure" -- useful for a teacher building worksheet variants of a
problem they already have verified.

Deliberately does NOT ask the LLM to also produce the new problem's
answer or equations -- an LLM's own stated numeric answer for a problem
it just invented isn't trustworthy on its own (this app's whole premise
is that LLM output needs independent symbolic verification). Instead
this module only generates the new problem TEXT; the caller is expected
to feed each generated problem right back through the app's own
extract_model()/verify() pipeline -- the exact same one used for any
user-submitted problem -- so a generated worksheet problem is verified
by the same standard as everything else, not taken on faith.
"""
from modules.llm_client import LMStudioClient, extract_json
from modules.equation_engine import ProblemModel
import sympy as sp

WORKSHEET_PROMPT = """Here is a mathematical model derived from a solved problem:

Domain: {domain}
Equations:
{equations}

Write {count} NEW word problem(s) that use this EXACT SAME underlying mathematical \
structure/equations, but with a DIFFERENT concrete scenario/story and DIFFERENT numbers -- \
not the same numbers, not the same wording. Each new problem must be fully solvable using \
the same equations, just with new values substituted in. {difficulty_note}

Respond as a JSON array of plain strings, one per problem, with no other text: \
["first new problem text...", "second new problem text..."]
"""

_DIFFICULTY_NOTES = {
    "easier": "Make the numbers simpler/rounder than the reference and avoid adding complications.",
    "harder": "Make the numbers less round and/or add one small extra step or twist versus the reference.",
    "similar": "Keep the difficulty comparable to the reference problem.",
}


def generate_worksheet_problems(client: LMStudioClient, model: ProblemModel,
                                  count: int = 3, difficulty: str = "similar") -> list[str]:
    """Returns up to `count` new word-problem TEXTS sharing model's own
    verified equation structure. Returns [] (not an exception) on any
    failure -- an API error, a malformed response, whatever -- since
    worksheet generation is a bonus feature built on top of an already-
    verified problem, the same "degrade gracefully" principle used for
    scenarios.py and step narration."""
    count = max(1, min(count, 5))
    eq_text = "\n".join(f"- {e.name}: {sp.latex(e.sympy_eq) if e.sympy_eq is not None else e.raw_expression}"
                         for e in model.equations)
    difficulty_note = _DIFFICULTY_NOTES.get(difficulty, _DIFFICULTY_NOTES["similar"])

    try:
        raw = client.chat(
            system="You write clear, solvable word problems for a math worksheet.",
            user=WORKSHEET_PROMPT.format(domain=model.problem_domain, equations=eq_text,
                                           count=count, difficulty_note=difficulty_note),
            temperature=0.8,
        )
    except Exception:  # noqa: BLE001
        return []

    try:
        parsed = extract_json(raw)
        if not isinstance(parsed, list):
            return []
        return [p.strip() for p in parsed if isinstance(p, str) and p.strip()][:count]
    except Exception:  # noqa: BLE001
        return []
