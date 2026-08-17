"""
Given a derived equation, asks the LLM for other real-world contexts
where the same mathematical structure applies -- a lightweight
"transfer learning" prompt, kept separate from extraction/verification
so it can't influence the math itself.
"""
import sympy as sp

from config import settings
from modules.llm_client import LMStudioClient
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
        import json
        text = raw.strip().strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        start, end = text.find("["), text.rfind("]")
        return json.loads(text[start:end + 1])
    except Exception:  # noqa: BLE001
        return [{"scenario": raw, "mapping": ""}]
