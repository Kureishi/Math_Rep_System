"""
Given a derived equation, asks the LLM for other real-world contexts
where the same mathematical structure applies -- a lightweight
"transfer learning" prompt, kept separate from extraction/verification
so it can't influence the math itself.
"""
import sympy as sp

from config import settings
from modules.llm_client import LMStudioClient, extract_json
from modules.equation_engine import ProblemModel

SCENARIO_PROMPT = """Here is a mathematical model derived from a problem:

Domain: {domain}
Equations:
{equations}

List 3 DIFFERENT real-world scenarios (from different fields if possible) where this exact \
same mathematical structure would apply, even though the surface details differ. For each, \
give a one-sentence scenario and one sentence on what each symbol would represent in that \
new context. Respond as a JSON array of objects: [{{"scenario": "...", "mapping": "..."}}]
"""


def generate_alternative_scenarios(client: LMStudioClient, model: ProblemModel) -> list[dict]:
    eq_text = "\n".join(f"- {e.name}: {sp.latex(e.sympy_eq) if e.sympy_eq is not None else e.raw_expression}"
                         for e in model.equations)
    raw = client.chat(
        system="You are creative but mathematically precise.",
        user=SCENARIO_PROMPT.format(domain=model.problem_domain, equations=eq_text),
        temperature=0.7,
    )
    try:
        parsed = extract_json(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON array, got {type(parsed).__name__}")
        # Tolerate entries missing a key rather than failing the whole batch.
        return [
            {"scenario": item.get("scenario", "(no scenario text returned)"),
             "mapping": item.get("mapping", "")}
            for item in parsed if isinstance(item, dict)
        ]
    except Exception as e:  # noqa: BLE001
        # Surface a clear, honest failure instead of silently rendering the
        # raw/fenced model output as if it were a real scenario.
        return [{"error": f"Couldn't parse the model's scenario suggestions ({e}). "
                           "Raw response is available for debugging.", "raw": raw}]
