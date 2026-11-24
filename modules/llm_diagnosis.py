"""
Turning a failure into an explanation a person can act on -- Streamlit-free, no LLM.

Before this, every way the model could fail to produce a usable derivation ended in one of two sentences ("The model didn't return any JSON ... try rephrasing, or
use a stronger model"), whatever had actually gone wrong. A diagnosis says:

    * WHAT happened, in one sentence, and at which stage of the pipeline;
    * the likely CAUSES, ranked, each with the specific FIX (a named setting, a LM Studio action, a way to reword the problem);
    * the FACTS the diagnosis rests on (model, finish reason, tokens used against the limit, how long the reply was), so a person who knows LM Studio can
      judge it for themselves, and the exact reply the model sent.

The facts come from `LMStudioClient.chat()` (`LLMText.meta`) and from the classified `LLMOutputError` (`kind`, `diagnostics`); the causes are rules over those facts,
not guesses made by a language model. When the facts are not enough to rank the causes the diagnosis says so rather than inventing a confident answer.

`diagnose_exception()` covers the rest of the pipeline too (the server dropping out mid-request, a request that timed out, a context-length error, a symbolic solve
that timed out, an equation the model wrote that SymPy cannot read, a missing package), and `diagnose_verification()` explains a derivation that was extracted but
never passed its checks, from WHICH checks failed.
"""
import re
from dataclasses import dataclass, field

from modules.llm_client import LLMOutputError
from modules.optional_deps import explain_import_error


@dataclass(frozen=True)
class Cause:
    text: str
    fix: str
    likelihood: str = "possible"          # "most likely" | "possible" | "less likely"


@dataclass
class Diagnosis:
    kind: str
    title: str
    summary: str
    causes: list[Cause] = field(default_factory=list)
    details: list[tuple[str, str]] = field(default_factory=list)
    stage: str | None = None
    raw_output: str = ""
    reply_excerpt: str = ""
    technical: str = ""
    severity: str = "error"               # "error" | "warning"

    @property
    def top_fix(self) -> str:
        return self.causes[0].fix if self.causes else ""


_RANK = {"most likely": 0, "possible": 1, "less likely": 2}


def _ranked(causes: list[Cause]) -> list[Cause]:
    return sorted(causes, key=lambda c: _RANK.get(c.likelihood, 1))


def _settings():
    from config import settings
    return settings


def _facts(diag: dict, cfg, reply: str = "", stage: str | None = None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [("Model asked for", str(cfg.reasoning_model))]
    if diag.get("model") and diag["model"] != cfg.reasoning_model:
        rows.append(("Model that answered", str(diag["model"])))
    rows.append(("LM Studio address", str(cfg.lm_studio_base_url)))
    if stage:
        rows.append(("Pipeline stage", stage))
    if diag.get("finish_reason") is not None:
        rows.append(("Finish reason", f"{diag['finish_reason']}" + (" (stopped because it ran out of tokens)" if diag["finish_reason"] == "length" else "")))
    if diag.get("completion_tokens") is not None:
        limit = diag.get("max_tokens") or cfg.max_tokens
        rows.append(("Tokens generated", f"{diag['completion_tokens']} of a {limit}-token limit"))
    if diag.get("prompt_tokens") is not None:
        rows.append(("Prompt size", f"{diag['prompt_tokens']} tokens"))
    rows.append(("Extraction temperature", f"{cfg.temperature_extraction:g}"))
    rows.append(("Reply length", f"{len(reply)} characters"))
    if diag.get("reasoning_chars"):
        rows.append(("Separate reasoning text", f"{diag['reasoning_chars']} characters (not part of the reply)"))
    attempts = diag.get("attempts") or []
    if attempts:
        tried = "; ".join(f"{a['step']}: " + ("stopped early (reasoning used its budget)" if a.get("stopped_early") else f"finished '{a.get('finish_reason')}'") + f", {a.get('content_chars', 0)} reply chars"
                          for a in attempts)
        rows.append(("What the app tried", tried))
    return rows


def _out_of_tokens(diag: dict) -> bool:
    comp, limit = diag.get("completion_tokens"), diag.get("max_tokens")
    return diag.get("finish_reason") == "length" or bool(comp and limit and comp >= 0.95 * limit)


_QUESTION_WORDS = re.compile(r"\b(could you|can you|please (provide|clarify|specify)|which (one|value)|do you mean|what (is|are) the (value|units?)|need more)\b", re.I)
_REFUSAL = re.compile(r"^\W*(i('| a)?m sorry|sorry|i can(no|')t|i cannot|i('| a)?m unable|unable to|as an ai)", re.I)


def diagnose_llm_output(err: LLMOutputError, stage: str | None = None, cfg=None) -> Diagnosis:
    """Explain a reply the pipeline could not use."""
    cfg = cfg or _settings()
    diag = err.diagnostics
    reply = (err.raw_output or "").strip()
    excerpt = reply[:420] + (" …" if len(reply) > 420 else "")
    kind = err.kind
    limit_hint = f"Raise “Max response tokens” in the sidebar's Advanced settings (now {diag.get('max_tokens') or cfg.max_tokens}) to 8192 or more"
    d = Diagnosis(kind, "", "", details=_facts(diag, cfg, reply, stage), stage=stage, raw_output=err.raw_output, reply_excerpt=excerpt, technical=str(err))
    causes: list[Cause] = []

    if kind == "empty":
        d.title = "The model sent back an empty reply"
        d.summary = "LM Studio answered, but there was no text in the answer, so there was nothing to turn into equations."
        if diag.get("attempts") and diag.get("reasoning_chars"):
            causes.append(Cause("The model is a “thinking” model that reasons at length before answering. The app asked it not to think, stopped the reasoning when it used up its budget "
                                 f"({diag.get('reasoning_budget_tokens') or cfg.reasoning_budget_tokens} tokens), and retried under a JSON schema, but this model kept reasoning.",
                                 "Do NOT raise “Max response tokens”: that only lets it think for longer. In LM Studio turn thinking off when you load the model (its model settings, "
                                 "“Enable Thinking”), or load a non-thinking (instruct) variant, then press Represent & Solve again.", "most likely"))
        if _out_of_tokens(diag) and not diag.get("attempts"):
            causes.append(Cause(f"The reply hit the length limit ({diag.get('completion_tokens')} of {diag.get('max_tokens') or cfg.max_tokens} tokens) before any text was produced. "
                                 "This is typical of “thinking” models, which spend their whole budget reasoning first.",
                                 limit_hint + ", or load a non-thinking (instruct) model for extraction.", "most likely"))
        if diag.get("reasoning_chars"):
            causes.append(Cause(f"The model wrote {diag['reasoning_chars']} characters into a separate reasoning field and none into the reply.",
                                 "Turn off “thinking” for this model in LM Studio (or use an instruct variant of it); or raise the token limit so it can finish thinking and then answer.",
                                 "most likely" if not _out_of_tokens(diag) else "possible"))
        comp = diag.get("completion_tokens")
        if comp is not None and comp <= 3 and diag.get("finish_reason") in ("stop", None):
            causes.append(Cause("The model stopped immediately, after almost no tokens. Usually the prompt does not fit the model's context window, or the model is a base model "
                                 "that has no chat template.",
                                 "In LM Studio reload the model with a context length of at least 4096, and make sure it is an instruct/chat model.", "most likely"))
        if not diag:
            causes.append(Cause("LM Studio did not report why (no finish reason or token counts came back).",
                                 "Open LM Studio's Developer tab and read the server log for this request; the reason is usually printed there.", "possible"))
        causes.append(Cause("The model was still loading, was unloaded, or crashed while answering.",
                             "Check that the model shows as loaded in LM Studio, then press Represent & Solve again.", "possible"))
        causes.append(Cause("A very long problem statement plus the instructions filled the context window.",
                             "Shorten the problem to the quantities and the question, or raise the context length in LM Studio.", "less likely"))

    elif kind == "unclosed_thinking":
        d.title = "The model ran out of room while still thinking"
        d.summary = "Its reply starts with a <think> block that never ends, so the JSON answer was never written."
        causes += [Cause("The token limit was spent on reasoning.", limit_hint + ", or use a model that does not think out loud.", "most likely"),
                   Cause("The problem is hard enough to send the model into a long deliberation.", "Split it into steps, or state the intended method in the problem.", "possible")]

    elif kind == "no_json":
        d.title = "The model answered in words instead of the structured format"
        d.summary = "The reply contains no JSON, which is the only form the app can turn into equations."
        if reply and _REFUSAL.search(reply):
            causes.append(Cause("The model refused or apologised instead of extracting the problem.", "Rephrase the problem as a plain maths/physics statement; if it keeps refusing, try a different model.", "most likely"))
        if reply and (reply.rstrip().endswith("?") or _QUESTION_WORDS.search(reply)):
            causes.append(Cause("The model asked a clarifying question instead of answering.",
                                 "Put the missing information in the problem (every number with its unit, and what is being asked for). If it refers to a stored value, check the workspace name.", "most likely"))
        causes.append(Cause("The model solved or explained the problem in prose instead of following the output format.",
                             "Use a stronger instruction-following model for extraction (set it under Advanced settings), or lower the extraction temperature.", "most likely" if not causes else "possible"))
        causes.append(Cause("The problem statement is ambiguous or incomplete, so the model could not decide what to extract.",
                             "State the unknown explicitly (“find the acceleration”) and give every known value with its unit.", "possible"))

    elif kind == "truncated":
        d.title = "The model's answer was cut off part-way"
        d.summary = "The JSON starts but has no closing bracket, so it cannot be read."
        causes.append(Cause("The reply was cut off at the length limit." + (f" ({diag.get('completion_tokens')} of {diag.get('max_tokens') or cfg.max_tokens} tokens.)" if diag.get("completion_tokens") else ""),
                            limit_hint + ".", "most likely" if _out_of_tokens(diag) or not diag else "possible"))
        causes.append(Cause("The problem needs many equations, so the structured answer is long.", "Solve it in parts (a chain), or raise the token limit.", "possible"))
        causes.append(Cause("The server stopped the stream early (the model was unloaded or crashed).", "Check LM Studio's Developer log, then try again.", "less likely"))

    elif kind == "bad_json":
        line = diag.get("line")
        d.title = "The model's JSON is malformed"
        d.summary = f"The reply contains JSON, but it does not parse ({diag.get('error', err)})."
        if line and reply:
            lines = reply.splitlines()
            if 1 <= line <= len(lines):
                d.details.append(("Offending line", f"line {line}: {lines[line - 1].strip()[:160]}"))
        causes += [Cause("Smaller models often add trailing commas, comments, or unescaped backslashes (from LaTeX) to JSON.",
                         "Press Represent & Solve again (the app retries with a hint), or use a stronger model for extraction.", "most likely"),
                   Cause("The sampling temperature is high enough to make the formatting unreliable.", f"Lower “Extraction temperature” (now {cfg.temperature_extraction:g}) toward 0.", "possible")]

    elif kind == "schema":
        probs = diag.get("problems") or []
        d.title = "The model's JSON has the wrong structure"
        d.summary = "The reply is valid JSON but is missing required parts, so it cannot be turned into a model."
        for p in probs[:8]:
            d.details.append(("Problem", f"{p.get('field', '?')}: {p.get('message', '')}"))
        causes += [Cause("The model used its own field names instead of the ones the app asks for.", "Press Represent & Solve again; if it repeats, use a stronger instruction-following model.", "most likely"),
                   Cause("The problem has an unusual shape (no equations, or no unknowns), so the model left parts out.", "State the quantities and what to find explicitly.", "possible")]

    else:
        d.title = "The model's reply could not be used"
        d.summary = str(err)
        causes.append(Cause("Unclassified failure.", "Open “What the model sent” below; the reply usually shows what went wrong.", "possible"))

    d.causes = _ranked(causes)
    return d


def _status_code(exc: BaseException) -> int | None:
    return getattr(exc, "status_code", None)


def diagnose_exception(exc: BaseException, stage: str | None = None, cfg=None) -> Diagnosis:
    """Explain ANY exception from the solve pipeline."""
    cfg = cfg or _settings()
    stage = stage or getattr(exc, "pipeline_stage", None)
    if isinstance(exc, LLMOutputError):
        return diagnose_llm_output(exc, stage, cfg)

    base = [("Model asked for", str(cfg.reasoning_model)), ("LM Studio address", str(cfg.lm_studio_base_url))] + ([("Pipeline stage", stage)] if stage else [])
    technical = f"{type(exc).__name__}: {exc}"
    msg = str(exc)

    hint = explain_import_error(exc)
    if hint:
        return Diagnosis("missing_package", "A required package is not installed", hint.replace("**", ""),
                         [Cause("The Python environment running the app is missing a package.", hint.replace("**", ""), "most likely")], base, stage, technical=technical)

    from openai import APIConnectionError, APIStatusError, APITimeoutError

    if isinstance(exc, APITimeoutError):
        return Diagnosis("timeout", "LM Studio took too long to answer", "The request was sent but no answer came back in time.",
                         _ranked([Cause("The model is large for this machine (for example running mostly on the CPU), so a long structured answer takes minutes.", "Load a smaller or more heavily quantised model, or offload more layers to the GPU in LM Studio.", "most likely"),
                                  Cause("The first request after loading a model also loads it into memory.", "Wait for LM Studio to finish loading, then try again.", "possible"),
                                  Cause("A very long problem or a very high token limit.", "Shorten the problem or lower “Max response tokens”.", "less likely")]),
                         base, stage, technical=technical)
    if isinstance(exc, APIConnectionError):
        return Diagnosis("connection", "Lost contact with LM Studio", f"The app could not reach {cfg.lm_studio_base_url} while working on the problem.",
                         _ranked([Cause("The server was stopped, or LM Studio closed or crashed during the request.", "Open LM Studio, load a model, and click Start Server on the Developer tab; then try again.", "most likely"),
                                  Cause("The address is wrong.", f"Check the server address (LM_STUDIO_BASE_URL; now {cfg.lm_studio_base_url}) and the port shown in LM Studio.", "possible"),
                                  Cause("A firewall or VPN is blocking local connections.", "Allow the connection to localhost, or pause the VPN.", "less likely")]),
                         base, stage, technical=technical)
    if isinstance(exc, APIStatusError):
        code = _status_code(exc)
        low = msg.lower()
        if code == 404 or "model" in low and ("not found" in low or "not loaded" in low or "no models loaded" in low):
            return Diagnosis("model_missing", "LM Studio does not have the requested model loaded", f"The server answered {code or ''}: the model “{cfg.reasoning_model}” is not available.".replace("answered :", "answered:"),
                             _ranked([Cause(f"“{cfg.reasoning_model}” is not loaded.", "Load it in LM Studio, or set the model name under Advanced settings (or LM_REASONING_MODEL) to one that is loaded.", "most likely")]),
                             base + [("Server said", msg[:200])], stage, technical=technical)
        if code == 400 and any(w in low for w in ("context", "n_ctx", "too long", "maximum", "tokens")):
            return Diagnosis("context_length", "The request does not fit the model's context window", "The prompt plus the reply budget is larger than the context the model was loaded with.",
                             _ranked([Cause("The context length in LM Studio is smaller than the prompt and the reply limit together.", "Reload the model with a larger context length, or lower “Max response tokens”.", "most likely"),
                                      Cause("The problem text (or the stored workspace values) is very long.", "Shorten the problem.", "possible")]),
                             base + [("Server said", msg[:200])], stage, technical=technical)
        return Diagnosis("server_error", f"LM Studio returned an error ({code})", msg[:300] or "The server reported an error.",
                         _ranked([Cause("The server rejected or failed the request.", "Read LM Studio's Developer log for the reason; reloading the model often clears it.", "most likely")]),
                         base + [("Server said", msg[:200])], stage, technical=technical)

    from modules.timeout_utils import ComputationTimeoutError
    if isinstance(exc, ComputationTimeoutError):
        return Diagnosis("symbolic_timeout", "The equations were too hard to solve symbolically in time",
                         f"The symbolic engine gave up after {cfg.computation_timeout_seconds:g} seconds so the app would not freeze.",
                         _ranked([Cause("The system is nonlinear or transcendental, which has no closed form (or a very long one).", "Ask for a numerical answer, give an initial guess, or simplify the model.", "most likely"),
                                  Cause("The model produced many coupled equations where fewer would do.", "State the problem with only the quantities that matter.", "possible"),
                                  Cause("The limit is too low for a slow machine.", "Raise COMPUTATION_TIMEOUT_SECONDS (an environment variable) and restart.", "less likely")]),
                         base, stage, technical=technical)

    name = type(exc).__name__
    if name in ("SympifyError", "TokenError", "ParseError", "SyntaxError") or (isinstance(exc, (ValueError, KeyError)) and any(w in msg.lower() for w in ("parse", "sympif", "could not interpret", "invalid expression"))):
        return Diagnosis("equation_syntax", "An equation the model wrote cannot be read", "The model's JSON was fine, but one of its equations is not valid for the symbolic engine.",
                         _ranked([Cause("The model used notation SymPy does not accept (^ for powers, implicit multiplication such as 2x, LaTeX, or a function name that does not exist).",
                                        "Press Represent & Solve again (the retry tells the model what failed), or state the formula in the problem using * and ** .", "most likely"),
                                  Cause("The problem mixes symbols that look alike (l and 1, O and 0).", "Use distinct names for the quantities.", "possible")]),
                         base + [("Error", msg[:240])], stage, technical=technical)

    return Diagnosis("unexpected", "Something unexpected went wrong", f"{name}: {msg[:300]}",
                     _ranked([Cause("This looks like a bug rather than a problem with the model or the input.",
                                    "The full traceback is below and in data/app.log. Press Represent & Solve again; if it repeats, report it with the technical details.", "most likely")]),
                     base + [("Error type", name)], stage, technical=technical)


# ------------------------------------------------------------------------------------------ a derivation that never verified

_CHECK_ADVICE: list[tuple[str, str, str, str]] = [
    # (label starts with, what it means, likely cause, fix)
    ("Equation parsing", "An equation could not be read.", "The model wrote notation the symbolic engine does not accept.", "Press Represent & Solve again, or write the formula in the problem with * and ** ."),
    ("Target variable present", "The quantity you asked for does not appear in any equation.", "The model extracted a different unknown than the one you meant.", "Name the quantity wanted explicitly (“find the acceleration a”), using the same words as the problem."),
    ("Determinacy", "There are not as many independent equations as unknowns (or there are too many).", "A given value was missed, or an equation was repeated.", "Make sure every known value is in the problem with its unit; remove statements that say the same thing twice."),
    ("Symbolic solve", "The symbolic solver found no solution.", "The equations contradict each other, or are nonlinear with no closed form.", "Check the numbers for a typo; for nonlinear systems ask for a numerical answer."),
    ("Dimensional consistency", "An equation adds or equates quantities with different units.", "A unit is missing or mixed (minutes with seconds, km with m).", "Convert to consistent units in the problem text, or state each unit explicitly."),
    ("Numeric balance", "Putting the answer back into an equation does not make both sides equal.", "The model's equation does not describe the situation, or a number was mis-copied.", "Compare the “Derived equations” with the problem statement; correct the wording and solve again."),
    ("Independent cross-check", "A second, independent solve gave a different answer.", "The model's two attempts disagree: one of them mis-modelled the problem.", "Check which equation is wrong in “Derived equations”; rephrase the ambiguous part."),
    ("Independent numerical cross-check", "A numerical solve disagrees with the symbolic one.", "The system may be stiff, or the initial conditions were extracted differently.", "Check the initial conditions and the time span in “Variables”."),
    ("ODE", "The differential equation's solution did not check.", "The equation or its initial condition was extracted incorrectly.", "State the rate law and the starting value explicitly."),
    ("Recurrence", "The recurrence's closed form did not check.", "The recurrence or its starting term was extracted incorrectly.", "State the first term and the rule for the next term explicitly."),
    ("Optimization", "The optimisation could not be solved or verified.", "The objective or the constraint was extracted incorrectly.", "State what is to be maximised or minimised and every constraint."),
    ("Feasibility", "The answer breaks one of the stated inequalities.", "An inequality was extracted with the wrong direction.", "Check the inequality statements in the problem and in “Derived equations”."),
    ("Coupled ODE", "The coupled system's solution did not check.", "One of the coupled equations is wrong or missing.", "Write each rate on its own line with its initial value."),
]


def diagnose_verification(report, retries: int | None = None) -> Diagnosis | None:
    """Why a derivation that WAS extracted never passed its checks, from which checks failed. None if it passed."""
    if report.passed:
        return None
    cfg = _settings()
    failing = [c for c in report.checks if not c.passed]
    causes: list[Cause] = []
    seen: set[str] = set()
    for c in failing:
        for prefix, means, cause, fix in _CHECK_ADVICE:
            if c.label.startswith(prefix) and prefix not in seen:
                seen.add(prefix)
                causes.append(Cause(f"{means} {cause}", fix, "most likely" if not causes else "possible"))
                break
    if not causes:
        causes.append(Cause("A check failed that has no specific advice yet.", "Read the failed checks below and compare the derived equations with the problem statement.", "possible"))
    causes.append(Cause("The extraction model is too small for this kind of problem.", "Try a stronger model for extraction (Advanced settings), or raise the number of verification retries.", "less likely"))
    limit = cfg.max_verification_retries if retries is None else retries
    d = Diagnosis("unverified", "This derivation did not pass its checks",
                  f"The model produced equations, but after up to {limit} retries at least one check still fails, so the answer below is shown without a guarantee.",
                  _ranked(causes), [(c.label, c.detail[:200]) for c in failing[:8]], severity="warning")
    return d
