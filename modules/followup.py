"""
Grounded follow-up Q&A: after a problem is solved, lets someone ask
"what happens if t doubles?" or "why this formula and not X?" with the
answer constrained to the problem's OWN verified equations/values,
instead of free-floating chat that could invent new numbers or
formulas the person never asked to verify.

Two-tier handling, splitting "compute something" from "explain
something" -- because this app's whole premise is that LLM arithmetic
isn't trustworthy on its own:

1. NUMERIC "what if" questions ("what if t doubles", "what if v_i were
   15 instead") are classified via one small LLM call into a
   STRUCTURED intent (which known symbol, which operation: multiply /
   add / set, what operand) -- the LLM only extracts INTENT from
   natural language, exactly the kind of task it's suited for. The
   actual arithmetic (applying that operation, then re-evaluating the
   verified formula) is done by SymPy via
   uncertainty.solve_symbolic_for_target, the same machinery
   sensitivity.py/uncertainty.py already use -- so a what-if answer is
   exactly as verified as the original solve, not a fresh LLM guess.
2. CONCEPTUAL questions ("why this formula", "what does v_i mean", "is
   there a simpler way") get a grounded LLM answer: the system prompt
   includes the problem's actual equations, known values, and solved
   answers, with an explicit instruction to answer using ONLY that
   information and say so rather than invent a fact that isn't there.
   This grounds the answer in verified facts but the PROSE explanation
   itself isn't independently re-verified the way a numeric answer is --
   an honest limitation of natural-language explanation, not something
   this module claims to solve.
"""
from dataclasses import dataclass

import sympy as sp

from modules.equation_engine import ProblemModel
from modules.llm_client import LMStudioClient, extract_json
from modules.uncertainty import solve_symbolic_for_target
from modules.verifier import VerificationReport

INTENT_PROMPT = """This problem's known variables are: {variables}

Student question: "{question}"

If the question asks a numeric "what if [variable] changes" question, respond with ONLY this \
JSON (no other text): {{"intent": "what_if", "symbol": "<one of the variable names above>", \
"operation": "multiply"|"add"|"set", "operand": <a number>}}
- operation "multiply": operand is the multiplier (e.g. "doubles" -> operand 2, "increases by \
20%" -> operand 1.2, "is cut in half" -> operand 0.5)
- operation "add": operand is the amount added (e.g. "increases by 3" -> operand 3, "decreases \
by 2" -> operand -2)
- operation "set": operand is the new absolute value (e.g. "if t were 10" -> operand 10)

Otherwise (a conceptual question, not asking to change a specific known value), respond with \
ONLY this JSON: {{"intent": "conceptual"}}
"""

GROUNDED_SYSTEM_PROMPT = """You are answering a follow-up question about an ALREADY-SOLVED and \
verified math problem. Use ONLY the equations, known values, and answers given below -- if the \
question needs a fact or number not given here, say so plainly instead of inventing one. Keep \
the answer concise (2-4 sentences unless the question genuinely needs more)."""


@dataclass
class FollowupAnswer:
    kind: str          # "what_if" | "conceptual" | "error"
    text: str          # the answer to show
    computed_value: float | None = None   # set for a successful what_if recompute
    target: str | None = None
    symbol: str | None = None


def _build_grounding(model: ProblemModel, report: VerificationReport) -> str:
    lines = [f"Domain: {model.problem_domain}", "Equations:"]
    for e in model.equations:
        expr_text = sp.latex(e.sympy_eq) if e.sympy_eq is not None else e.raw_expression
        lines.append(f"- {e.name}: {expr_text}")
    lines.append("Known values:")
    for v in model.variables:
        if v.known_value is not None:
            lines.append(f"- {v.symbol} ({v.meaning}) = {v.known_value} {v.unit or ''}")
    if report.sympy_numeric_answers:
        lines.append("Solved answers:")
        for target, val in report.sympy_numeric_answers.items():
            lines.append(f"- {target} = {val:.6g}")
    return "\n".join(lines)


def _classify_intent(client: LMStudioClient, model: ProblemModel, question: str) -> dict:
    var_list = ", ".join(v.symbol for v in model.variables if v.known_value is not None)
    try:
        raw = client.chat(
            system="You classify student follow-up questions into a structured intent.",
            user=INTENT_PROMPT.format(variables=var_list, question=question),
            temperature=0.0, json_mode=True,
        )
        parsed = extract_json(raw)
        return parsed if isinstance(parsed, dict) else {"intent": "conceptual"}
    except Exception:  # noqa: BLE001
        return {"intent": "conceptual"}


def _apply_operation(nominal: float, operation: str, operand: float) -> float | None:
    if operation == "multiply":
        return nominal * operand
    if operation == "add":
        return nominal + operand
    if operation == "set":
        return operand
    return None


def answer_followup(client: LMStudioClient, model: ProblemModel, report: VerificationReport,
                     question: str, target_name: str | None = None) -> FollowupAnswer:
    """Answers one follow-up question, grounded in model/report. If
    target_name isn't given, uses the first solve_for target with a
    verified numeric answer for any what-if recompute."""
    if not question.strip():
        return FollowupAnswer(kind="error", text="No question was entered.")

    target_name = target_name or next(iter(report.sympy_numeric_answers), None)
    intent = _classify_intent(client, model, question)

    if intent.get("intent") == "what_if" and target_name is not None:
        symbol_name = intent.get("symbol")
        operation = intent.get("operation")
        operand = intent.get("operand")
        known_var = next((v for v in model.variables
                           if v.symbol == symbol_name and v.known_value is not None), None)
        if known_var is not None and operation in ("multiply", "add", "set") and isinstance(operand, (int, float)):
            new_value = _apply_operation(known_var.known_value, operation, operand)
            expr = solve_symbolic_for_target(model, target_name)
            if expr is not None and new_value is not None:
                knowns = {sp.Symbol(v.symbol): v.known_value for v in model.variables
                          if v.known_value is not None}
                knowns[sp.Symbol(symbol_name)] = new_value
                try:
                    result = expr.subs(knowns)
                    if not result.free_symbols:
                        computed = float(result)
                        original = report.sympy_numeric_answers.get(target_name)
                        change_text = (f" (was {original:.6g})" if original is not None else "")
                        return FollowupAnswer(
                            kind="what_if",
                            text=(f"If {symbol_name} becomes {new_value:.6g} "
                                  f"(from {known_var.known_value:.6g}), then "
                                  f"{target_name} = {computed:.6g}{change_text}."),
                            computed_value=computed, target=target_name, symbol=symbol_name,
                        )
                except (TypeError, ValueError):
                    pass
        # intent was "what_if" but couldn't be resolved to a real
        # recompute (unknown symbol, target has no closed formula,
        # etc.) -- fall through to a grounded conceptual answer instead
        # of silently failing

    grounding = _build_grounding(model, report)
    try:
        raw = client.chat(
            system=GROUNDED_SYSTEM_PROMPT,
            user=f"{grounding}\n\nQuestion: {question}",
            temperature=0.3,
        )
        return FollowupAnswer(kind="conceptual", text=raw.strip())
    except Exception as e:  # noqa: BLE001
        return FollowupAnswer(kind="error", text=f"Couldn't reach the model to answer ({e}).")
