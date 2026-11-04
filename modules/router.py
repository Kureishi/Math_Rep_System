"""
"Ask anything": route a free-form request to the right mode and prefill it -- Streamlit-free.

With this many modes the hard part for a new user is knowing where to go. The router reads what was typed or pasted and answers
three questions: WHICH mode (with a confidence), WHY (so the choice can be checked rather than trusted), and WHAT TO PREFILL
there (the problem text, the two expressions to compare, the pasted data table, the sides of a triangle).

HOW IT DECIDES. Not by a language model: a set of weighted signals, so it is instant, works with LM Studio switched off, and
gives the same answer every time.
  * the keywords each mode already registers in the command palette (modules/command_palette.py), longer phrases weighing more;
  * structural cues that no keyword captures: a block of rows of numbers (data), "A == B" or "is A equivalent to B",
    a triangle with three numbers, an ask to "show / animate / visualise" a black hole or the Ising model;
  * a sentence with a number and a question word is the default home of the app: the word-problem solver.
The top mode's share of the total score is the confidence. Below AUTO_GO_CONFIDENCE the UI shows the choices instead of
jumping, and `route_with_llm` can break the tie with the language model -- whose answer is accepted only if it names a real mode.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any

from modules.command_palette import MODE_LABELS, _ENTRIES

AUTO_GO_CONFIDENCE = 0.6

WORD_PROBLEM = "📝 Word problem solver"
EQUIVALENCE = "🔁 Check equivalence"
CURVE_FIT = "📈 Curve fitting"
DISCOVER = "🧭 Discover equations"
SHOWCASE = "🎬 Visual showcase"
GEOMETRY = "📐 Geometry"
RELATIVITY = "🌌 General relativity"
STATMECH = "🔥 Statistical mechanics"
QUANTUM = "⚛️ Quantum mechanics"
DIMENSIONAL = "📐 Dimensional analysis"
RECALL = "🔎 Have I solved this?"

# Words naming a mode outright (heavier than an incidental keyword).
_MODE_WORDS: dict[str, list[str]] = {
    QUANTUM: ["quantum"], RELATIVITY: ["relativity", "general relativity", "spacetime"], STATMECH: ["statistical mechanics", "thermodynamics"],
    GEOMETRY: ["geometry", "triangle"], DISCOVER: ["sindy", "equation discovery"], "🌡️ PDE solver": ["pde"],
    "🧮 Tensor calculus": ["tensor"], "🧰 Numerics & verification lab": ["numerics"],
}


@dataclass(frozen=True)
class Route:
    mode: str
    confidence: float
    reason: str
    prefill: dict[str, Any] = field(default_factory=dict)
    alternatives: list[tuple[str, float]] = field(default_factory=list)

    @property
    def sure(self) -> bool:
        return self.confidence >= AUTO_GO_CONFIDENCE

    @property
    def needs_language_model(self) -> bool:
        return self.mode == WORD_PROBLEM


# ------------------------------------------------------------------------------------------ extracting structure

_NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
_ROW = re.compile(rf"^\s*{_NUM}(?:\s*[,;\t ]\s*{_NUM})+\s*$")
_EQUIV_WORDS = re.compile(r"equivalent|identical|the same|same as|equal to|equals|prove|show that|verify|check (?:whether|if)|simplif", re.I)


def find_data_block(text: str, min_rows: int = 4) -> str | None:
    """The longest run of consecutive numeric rows (with an optional header line directly above), or None."""
    lines = text.splitlines()
    best: tuple[int, int] | None = None
    i = 0
    while i < len(lines):
        if _ROW.match(lines[i]):
            j = i
            while j < len(lines) and _ROW.match(lines[j]):
                j += 1
            if j - i >= min_rows and (best is None or j - i > best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1
    if best is None:
        return None
    start = best[0] - 1 if best[0] > 0 and lines[best[0] - 1].strip() and not _ROW.match(lines[best[0] - 1]) and \
        re.fullmatch(r"[\w\s,;\t().^/-]+", lines[best[0] - 1]) and len(lines[best[0] - 1]) < 80 else best[0]
    return "\n".join(l.strip() for l in lines[start:best[1]])


def split_equivalence(text: str) -> tuple[str, str] | None:
    """The two expressions in 'A == B', 'is A equivalent to B', 'show that A = B' ... or None."""
    t = text.strip().rstrip("?.! ")
    if "==" in t:
        a, b = t.split("==", 1)
    else:
        m = re.search(r"(?:^|\b(?:is|are|does|show that|prove that|prove|verify that|verify|check (?:whether|if|that)))\s*(.+?)\s+"
                      r"(?:is |are )?(?:equivalent to|identical to|equal to|the same as|same as|equals)\s+(.+)$", t, re.I)
        if not m:
            m = re.search(r"(?:show that|prove that|prove|verify that|verify)\s+(.+?)\s*=\s*(.+)$", t, re.I)
        if not m:
            return None
        a, b = m.group(1), m.group(2)
    a, b = a.strip().strip(",:;"), b.strip().strip(",:;")
    a = re.sub(r"^(?:is|are|does|check|that)\s+", "", a, flags=re.I)
    return (a, b) if a and b and len(a) < 200 and len(b) < 200 else None


def triangle_sides(text: str) -> tuple[str, str, str] | None:
    if "triangle" not in text.lower():
        return None
    m = re.search(rf"sides?\D{{0,12}}({_NUM})\D{{1,6}}({_NUM})\D{{1,6}}({_NUM})", text, re.I)
    return (m.group(1), m.group(2), m.group(3)) if m else None


# ------------------------------------------------------------------------------------------ the decision

def _phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def route(query: str) -> Route:
    """Where `query` should go, how sure that is, why, and what to prefill."""
    raw = query.strip()
    if not raw:
        return Route(WORD_PROBLEM, 0.0, "Nothing was asked.")
    text = raw.lower()
    scores: dict[str, float] = {}
    why: dict[str, list[str]] = {}
    prefill: dict[str, dict[str, Any]] = {}

    def add(mode: str, weight: float, reason: str) -> None:
        scores[mode] = scores.get(mode, 0.0) + weight
        why.setdefault(mode, []).append(reason)

    for entry in _ENTRIES:
        for kw in entry.keywords:
            if len(kw) >= 2 and _phrase(text, kw.lower()):
                add(entry.mode, 1.0 + 0.4 * (len(kw.split()) - 1), f"mentions “{kw}”")
    for mode, words in _MODE_WORDS.items():
        for w in words:
            if _phrase(text, w):
                add(mode, 1.5, f"names {w}")

    data = find_data_block(raw)
    dynamics = re.search(r"differential|ode\b|dynamics|time series|discover|find the (?:law|equation)|governing|sindy|model the|identify", text)
    fitting = re.search(r"\bfit\b|regress|trend|best fit|curve", text)
    if data:
        rows = len(data.splitlines())
        prefill[DISCOVER] = {"discover_source": "Paste or upload data", "discover_paste": data}
        if dynamics:
            add(DISCOVER, 3.0, f"a table of {rows} rows and you asked for the dynamics")
        if fitting:
            add(CURVE_FIT, 2.5, f"a table of {rows} rows and you asked for a fit")
        if not dynamics and not fitting:
            add(DISCOVER, 1.2, f"a table of {rows} rows")
            add(CURVE_FIT, 1.2, f"a table of {rows} rows")
    elif dynamics and re.search(r"from (?:the )?data|measurements|observations", text):
        add(DISCOVER, 2.0, "asks to infer equations from data")

    pair = split_equivalence(raw) if ("=" in raw or _EQUIV_WORDS.search(raw)) else None
    if pair and (_EQUIV_WORDS.search(raw) or "==" in raw):
        add(EQUIVALENCE, 3.0, "compares two expressions")
        prefill[EQUIVALENCE] = {"equiv_expr1": pair[0], "equiv_expr2": pair[1]}

    sides = triangle_sides(raw)
    if sides:
        add(GEOMETRY, 2.5, "a triangle with three given sides")
        prefill[GEOMETRY] = {"tri_a": sides[0], "tri_b": sides[1], "tri_c": sides[2]}

    show = re.search(r"\b(?:show|animat\w*|movie|visuali[sz]e\w*|render|picture|draw|plot)\b", text)
    subject = re.search(r"black hole|lensing|shadow|photon ring|orbit|precess\w*|ising|critical point|embedding|flamm", text)
    if show and subject:
        add(SHOWCASE, 2.5, "asks to see a black hole, an orbit or the critical point")

    if re.search(r"(?:have|did) i (?:already |ever )?(?:solved?|done|seen)|solved (?:this|that|it|something like)[^.?!]{0,20}before|"
                 r"seen this (?:problem|before)|(?:similar|duplicate|old) (?:problems?|questions?) (?:to|in|from) (?:this|my)", text):
        add(RECALL, 3.0, "asks whether you have solved something like this before")
        prefill[RECALL] = {"recall_query": raw}

    if re.search(r"\bconvert\b|\bunits?\b|dimensionless|dimension of", text):
        add(DIMENSIONAL, 0.8, "about units")

    sentence = len(raw.split()) >= 6
    has_number = re.search(r"\d", raw) is not None
    asks = re.search(r"\b(?:how|what|find|calculate|compute|determine|solve|when|where)\b", text) is not None
    if sentence and has_number and asks and not data:
        add(WORD_PROBLEM, 2.0, "a sentence with numbers and a question")
        prefill[WORD_PROBLEM] = {"problem_text": raw}
    elif sentence and has_number and not data:
        add(WORD_PROBLEM, 0.8, "a sentence with numbers")
        prefill[WORD_PROBLEM] = {"problem_text": raw}

    if not scores:
        from modules.command_palette import search
        guess = search(raw, limit=3) if len(raw.split()) <= 2 else []        # a name-like request only; a sentence has no "closest mode name"
        if guess:
            return Route(guess[0], 0.25, "closest match by name", {}, [(m, 0.1) for m in guess[1:]])
        return Route(WORD_PROBLEM, 0.2, "no clear signal; the word-problem solver is the general-purpose home",
                     {"problem_text": raw} if len(raw.split()) >= 3 else {})

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top, top_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = round(top_score / (top_score + second + 1.0), 3)
    reason = "; ".join(dict.fromkeys(why[top]))
    return Route(top, confidence, reason, prefill.get(top, {}), [(m, round(s / (top_score + second + 1.0), 3)) for m, s in ranked[1:4]])


# ------------------------------------------------------------------------------------------ the language-model tie-break

_LLM_SYSTEM = ("You route a request to one mode of a mathematics and physics app. Reply with ONLY a JSON object "
               '{"mode": "<one of the listed modes, copied exactly>", "reason": "<one short sentence>"}.')


def route_with_llm(query: str, client: Any, first: Route | None = None) -> Route:
    """Let the language model choose among the modes when the heuristic is unsure. The model's answer is used only if it names
    a real mode; anything else (an invented mode, bad JSON, an error) leaves the heuristic's answer in place."""
    first = first or route(query)
    try:
        raw = client.chat(system=_LLM_SYSTEM, user="Modes:\n" + "\n".join(MODE_LABELS) + f"\n\nRequest:\n{query[:1500]}", temperature=0.0)
        data = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
        mode = str(data["mode"]).strip()
    except Exception:  # noqa: BLE001 -- the tie-break is a convenience; any failure keeps the heuristic answer
        return first
    if mode not in MODE_LABELS:
        return first
    prefill = first.prefill if mode == first.mode else ({"problem_text": query.strip()} if mode == WORD_PROBLEM else {})
    return Route(mode, max(first.confidence, AUTO_GO_CONFIDENCE), f"chosen by the language model: {str(data.get('reason', ''))[:160]}".rstrip(": "),
                 prefill, [(first.mode, first.confidence)] if first.mode != mode else first.alternatives)
