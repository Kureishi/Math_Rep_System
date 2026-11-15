"""
"What next?": suggest the sensible next move from a result -- Streamlit-free.

With this many modes, the likeliest reason a feature goes unused is that nobody is pointed at it. After a result is shown, this module
looks at what the result IS (a verified algebraic problem, a curve fit, a kinetics run) and proposes the few moves that follow from it:
make a practice set from the problem, add it to your daily practice, check whether you have solved it before, sample the posterior of
a fit, or open the mode that covers its subject.

A suggestion is either a NAVIGATION (open another mode with its inputs already filled in) or an ACTION done in place (add this problem
to practice). Suggestions are decided from the state, not shown blindly: a problem that failed verification is not offered as practice
material, and a problem whose answer is not a number (a differential equation) is not offered a practice set it cannot have.

Destination modes are found through the mode registry by their page module, so renaming a mode's label or adding a plugin cannot break
this.
"""
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modules import problem_sets as ps
from modules.mode_registry import REGISTRY

MAX_CSV_ROWS = 500

PAGE_PROBLEM_SETS = "ui.problem_sets"
PAGE_RECALL = "ui.recall"
PAGE_POSTERIOR = "ui.posterior"
PAGE_PRACTICE = "ui.practice"
PAGE_CHEMISTRY = "ui.chemistry"
PAGE_COMPARE = "ui.compare"
PAGE_DISCOVER = "ui.discover"

ADD_TO_PRACTICE = "add_practice"


def mode_for(page_module: str) -> str:
    """The label of the mode whose page lives in `page_module`."""
    for spec in REGISTRY.specs:
        if spec.module == page_module:
            return spec.label
    raise KeyError(page_module)


@dataclass(frozen=True)
class NextStep:
    key: str                                  # unique within one page, for widget keys
    label: str                                # the button text
    reason: str                               # why it is suggested (the tooltip)
    mode: str | None = None                   # navigate to this mode ...
    prefill: dict[str, Any] = field(default_factory=dict)     # ... with these widget values already set
    action: str = ""                          # ... or do this in place (ADD_TO_PRACTICE)
    payload: dict[str, Any] = field(default_factory=dict)


def _step(key: str, label: str, reason: str, page: str, prefill: dict[str, Any] | None = None) -> NextStep | None:
    try:
        return NextStep(key, label, reason, mode_for(page), prefill or {})
    except KeyError:                                    # the destination mode is not installed: simply do not offer it
        return None


def for_problem(text: str, raw: dict, title: str = "", concepts: list[str] | None = None) -> list[NextStep]:
    """Moves that apply to a solved problem as a thing to practise: only when it can be (algebraic, numeric answer)."""
    return [NextStep("add_practice", "🧠 Add to my daily practice", "reviewed on a spaced schedule, with fresh numbers each time",
                     action=ADD_TO_PRACTICE, payload={"title": title or text[:60], "text": text, "raw": raw, "concepts": concepts or []})]


def for_solution(model: Any, report: Any, problem_text: str) -> list[NextStep]:
    """What to suggest after a problem has been solved and verified (at most four)."""
    steps: list[NextStep | None] = []
    practisable = False
    if getattr(report, "passed", False):
        try:
            ps.check_supported(model)
            practisable = True
        except ps.ProblemSetError:
            practisable = False
    if practisable:
        steps.append(_step("pset", "📝 Practice set from this", "the same problem with new numbers, and an answer key that passed the same verification",
                           PAGE_PROBLEM_SETS, {"pset_source": "The problem that is open"}))
        steps += for_problem(problem_text, model.raw_json, concepts=[getattr(model, "problem_domain", "") or "general"])
    steps.append(_step("recall", "🔎 Have I solved this before?", "search your history by meaning, wording and equation shape",
                       PAGE_RECALL, {"recall_query": problem_text}))
    if not getattr(report, "passed", True):
        steps.append(_step("compare", "🆚 Compare two solves", "the checks failed: solve it a second way and see where the answers disagree", PAGE_COMPARE))
    if getattr(model, "problem_domain", "") == "chemistry":
        steps.append(_step("chem", "🧪 Chemistry lab", "balance reactions, run titrations, fit rate laws", PAGE_CHEMISTRY))
    return [s for s in steps if s is not None][:4]


def csv_for(xs: Any, ys: Any, max_rows: int = MAX_CSV_ROWS) -> str:
    """Two-column CSV (x,y) for the posterior page, evenly thinned to at most `max_rows` rows."""
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if len(x) > max_rows:
        keep = np.linspace(0, len(x) - 1, max_rows).round().astype(int)
        x, y = x[keep], y[keep]
    return "x,y\n" + "\n".join(f"{a:.10g},{b:.10g}" for a, b in zip(x, y))


def fit_to_model(family: str, degree: int, expr_str: str | None, param_names: list[str] | None,
                 param_values: dict[str, float]) -> tuple[str, list[str], list[float]] | None:
    """(expression in x, parameter names, starting values) for a curve fit, in the form the posterior sampler takes, or None if the fit
    has no explicit form to hand over."""
    if family == "custom":
        if not expr_str or not param_names:
            return None
        return expr_str, list(param_names), [float(param_values.get(n, 1.0)) for n in param_names]
    if family in ("linear", "polynomial"):
        found = sorted(int(k[1:]) for k in param_values if k[:1] == "c" and k[1:].isdigit())
        n = 1 if family == "linear" else (found[-1] if found else degree)         # the fit's own coefficients say its degree
        names = [f"c{i}" for i in range(n + 1)]
        expr = " + ".join(["c0"] + [f"c{i}*x" + (f"**{i}" if i > 1 else "") for i in range(1, n + 1)])
        return expr, names, [float(param_values.get(k, 0.0)) for k in names]
    forms = {"exponential": "a*exp(b*x)", "power": "a*x**b", "logarithmic": "a*log(x) + b"}
    if family in forms:
        names = ["a", "b"]
        return forms[family], names, [float(param_values.get(k, 1.0)) for k in names]
    return None


def for_fit(xs: Any, ys: Any, expr: str, names: list[str], p0: list[float], label: str = "🎲 Sample the posterior of this fit") -> list[NextStep]:
    """Hand a fit (data + model + a starting point) to the posterior sampler, so the uncertainty is the full posterior, not just a ±."""
    step = _step("posterior", label, "least squares gives one best curve and a symmetric ±; sampling shows every parameter combination the data allow",
                 PAGE_POSTERIOR, {"post_source": "Paste or upload data", "post_paste": csv_for(xs, ys), "post_expr": expr,
                                  "post_names": ", ".join(names), "post_guess": ", ".join(f"{v:.6g}" for v in p0)})
    return [step] if step else []


def for_trajectory(t: Any, X: Any, names: list[str], label: str = "🧭 Discover the equations behind this trajectory", max_rows: int = 400) -> list[NextStep]:
    """Hand a simulated trajectory (time and a few state variables) to the equation-discovery mode: can the dynamics be recovered from the data alone?"""
    tt, XX = np.asarray(t, dtype=float), np.asarray(X, dtype=float)
    if XX.ndim != 2 or XX.shape[1] != len(names) or len(tt) != len(XX):
        return []
    if len(tt) > max_rows:
        keep = np.linspace(0, len(tt) - 1, max_rows).round().astype(int)
        tt, XX = tt[keep], XX[keep]
    text = "t," + ",".join(names) + "\n" + "\n".join(",".join(f"{v:.8g}" for v in [a, *row]) for a, row in zip(tt, XX))
    step = _step("discover", label, "SINDy fits sparse equations to the data: with clean simulated data it should rediscover the equations that produced it",
                 PAGE_DISCOVER, {"discover_source": "Paste or upload data", "discover_paste": text})
    return [step] if step else []
