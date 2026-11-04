"""
Problem sets with auto-grading -- Streamlit-free, no LLM.

GENERATING. Take one solved problem and make `n` variants by re-drawing the known values. The answer key is only worth
having if it is right, so every variant is put through the app's own solve-and-verify pipeline
(solve_compare.deterministic_report): a variant whose equations fail a check, give a non-finite answer or violate a
declared domain (a negative time) is discarded and replaced, and the key is the SymPy solution, not a model's guess.
  * Values are drawn STRATIFIED: each variable's range is cut into n bins and each variant takes a different bin, in an
    independent random order per variable, so n variants really do span the range instead of clustering.
  * Numbers look like a person wrote them: an integer stays an integer, a one-decimal value stays one decimal.
  * Physical constants (g, a gas constant) and zeros are never varied.
  * The statement is rewritten ONLY when every changed number can be located in the text unambiguously (it appears once,
    and no other variable shares that value). Otherwise the original wording is kept and the new values are listed under
    it, so a statement can never silently disagree with its key.
  * Only problems whose answer is a number are supported (algebraic problems); a differential equation, for example, has a
    function for an answer and is refused with a reason.

GRADING. Answers are parsed leniently ("12.5 m/s", "1.25e1", "1.25 x 10^1", "12,5") and compared with the key by relative
tolerance. Within the tolerance is correct; within a looser one is "close" (partial credit: it is usually rounding); anything
else is wrong, with a best-guess diagnosis of the slip: a sign error, a factor of 2 (the forgotten ½), a power of ten (a unit
conversion), a reciprocal, degrees against radians, the answer to a different quantity, or the original problem's numbers.
A class report gives each student's score and, per problem, the share who got it right and the commonest mistake, so a
question nobody gets is visible as a question, not as a class.
"""
import copy
import csv
import html
import io
import json
import math
import random
import re
from dataclasses import dataclass, field
from datetime import datetime

from modules.equation_engine import ProblemModel, build_model, target_kind
from modules.solve_compare import deterministic_report

FORMAT_VERSION = 1
MAX_PROBLEMS = 60
DEFAULT_SPREAD = 0.4
DEFAULT_TOLERANCE = 0.01          # relative, for "correct"
CLOSE_TOLERANCE = 0.05            # relative, for "close"
PARTIAL_CREDIT = 0.5
ANSWER_RANGE_FACTOR = 100.0       # a variant whose answer is this many times bigger/smaller than the original's is rejected
_CONSTANT_WORDS = ("constant", "gravitational", "speed of light", "boltzmann", "planck", "permittivity", "permeability", "avogadro",
                   "acceleration due to gravity")
_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(%?)(?![\w])")


class ProblemSetError(ValueError):
    """The source problem or the request cannot produce a trustworthy problem set (the message says why)."""


# ------------------------------------------------------------------------------------------ data

@dataclass
class ProblemInstance:
    index: int                              # 1-based
    values: dict[str, float]                # the known values of this variant (all knowns, varied or not)
    answers: dict[str, float]               # the verified answer for each target
    text: str                               # the statement
    rewritten: bool                         # True if the original wording was rewritten with the new numbers
    given: list[dict] = field(default_factory=list)       # {symbol, meaning, value, unit}
    find: list[dict] = field(default_factory=list)        # {symbol, meaning, unit}


@dataclass
class ProblemSet:
    title: str
    source_text: str
    domain: str
    targets: list[str]
    instances: list[ProblemInstance]
    base_values: dict[str, float]
    base_answers: dict[str, float]
    varied: list[str]
    seed: int
    spread: float
    discarded: int = 0
    created: str = ""
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------------------------ formatting helpers

def fmt(value: float, sig: int = 4) -> str:
    """A number as a person would write it: significant figures, no trailing noise, no exponent for ordinary sizes."""
    if value == 0 or not math.isfinite(value):
        return "0" if value == 0 else str(value)
    if 1e-3 <= abs(value) < 1e6:
        s = f"{value:.{sig}g}"
        if "e" in s:
            s = f"{value:.{sig}f}".rstrip("0").rstrip(".")
        return s
    return f"{value:.{sig}g}"


def _decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def _round_like(base: float, value: float, decimals: int | None = None) -> float:
    """Round `value` the way `base` is written: to `decimals` places if known (how the number appears in the statement), else an
    integer stays whole and otherwise up to 3 decimals of the base are kept."""
    if decimals is None:
        decimals = 0 if base == round(base) else min(max(_decimals(repr(base)), 1), 3)
    return float(round(value)) if decimals == 0 else round(value, min(decimals, 6))


def _format_like(token: str, value: float) -> str:
    d = _decimals(token.rstrip("%"))
    suffix = "%" if token.endswith("%") else ""
    v = value * 100 if suffix else value
    return (f"{v:.{d}f}" if d else str(int(round(v)))) + suffix


def _token_value(token: str) -> float:
    return float(token.rstrip("%")) / (100.0 if token.endswith("%") else 1.0)


def _matches(token: str, value: float) -> bool:
    return abs(_token_value(token) - value) <= 1e-12 * max(1.0, abs(value))


def written_decimals(text: str, base: dict[str, float], symbols: list[str]) -> dict[str, int | None]:
    """How many decimal places each variable's value is written with in the statement (a percentage counts two more places for the
    variable itself: 5% is 0.05), or None when the number cannot be located exactly once."""
    out: dict[str, int | None] = {}
    for sym in symbols:
        hits = [m.group() for m in _NUMBER.finditer(text) if _matches(m.group(), base[sym])]
        out[sym] = (_decimals(hits[0].rstrip("%")) + (2 if hits[0].endswith("%") else 0)) if len(hits) == 1 else None
    return out


# ------------------------------------------------------------------------------------------ choosing what to vary

def known_variables(model: ProblemModel) -> dict[str, float]:
    return {v.symbol: float(v.known_value) for v in model.variables
            if v.known_value is not None and not v.is_function and not v.is_vector}


def suggest_varied(model: ProblemModel) -> list[str]:
    """The knowns worth varying: not physical constants, not zero."""
    out = []
    for v in model.variables:
        if v.known_value is None or v.is_function or v.is_vector or float(v.known_value) == 0.0:
            continue
        if any(w in v.meaning.lower() for w in _CONSTANT_WORDS):
            continue
        out.append(v.symbol)
    return out


def check_supported(model: ProblemModel) -> None:
    if model.problem_type != "algebraic":
        raise ProblemSetError(f"Problem sets need an answer that is a number; this is a {model.problem_type} problem, whose answer is a "
                              "function.")
    if not model.solve_for:
        raise ProblemSetError("The problem has nothing to solve for.")
    bad = [t for t in model.solve_for if target_kind(model, t) != "equation"]
    if bad:
        raise ProblemSetError("These targets are not plain numeric unknowns: " + ", ".join(bad))


# ------------------------------------------------------------------------------------------ solving a variant

def _solve_variant(raw: dict, values: dict[str, float], base_answers: dict[str, float], model: ProblemModel) -> dict[str, float] | None:
    """The verified answers for `values`, or None if this variant is not acceptable."""
    trial = copy.deepcopy(raw)
    for var in trial["variables"]:
        if var["symbol"] in values:
            var["known_value"] = format(values[var["symbol"]], ".10g")
    try:
        report = deterministic_report(build_model(trial))
    except Exception:  # noqa: BLE001 -- an unbuildable variant is simply not usable
        return None
    if not report.passed:
        return None
    found = {t: report.sympy_numeric_answers.get(t) for t in model.solve_for}
    if any(a is None for a in found.values()):
        return None
    answers = {t: float(a) for t, a in found.items() if a is not None}
    if any(not math.isfinite(a) or abs(a) > 1e12 for a in answers.values()):
        return None
    domains = {v.symbol: v.domain for v in model.variables}
    for t, a in answers.items():
        dom = domains.get(t)
        if (dom == "positive" and a <= 0) or (dom == "nonnegative" and a < 0) or (dom == "negative" and a >= 0) or (dom == "nonpositive" and a > 0):
            return None
    for t, a in answers.items():
        base = base_answers[t]
        if base != 0 and not (1 / ANSWER_RANGE_FACTOR <= abs(a / base) <= ANSWER_RANGE_FACTOR):
            return None
    return answers


# ------------------------------------------------------------------------------------------ the statement

def render_statement(text: str, base: dict[str, float], new: dict[str, float], varied: list[str]) -> tuple[str, bool]:
    """(statement, rewritten). Rewrites the original wording with the new numbers only if every varied value can be found in
    it exactly once and no two varied variables share a value; otherwise returns the original text unchanged."""
    tokens = [(m.start(), m.end(), m.group()) for m in _NUMBER.finditer(text)]
    edits = []
    for sym in varied:
        hits = [t for t in tokens if _matches(t[2], base[sym])]
        sharing = [s for s in varied if base[s] == base[sym]]
        if len(hits) != 1 or len(sharing) != 1:
            return text, False
        edits.append((hits[0], sym))
    out = text
    for (start, end, tok), sym in sorted(edits, key=lambda e: -e[0][0]):
        out = out[:start] + _format_like(tok, new[sym]) + out[end:]
    return out, True


def _given_find(model: ProblemModel, values: dict[str, float]) -> tuple[list[dict], list[dict]]:
    by = {v.symbol: v for v in model.variables}
    given = [{"symbol": s, "meaning": by[s].meaning, "value": values[s], "unit": by[s].unit or ""} for s in values if s in by]
    find = [{"symbol": t, "meaning": by[t].meaning, "unit": by[t].unit or ""} for t in model.solve_for if t in by]
    return given, find


# ------------------------------------------------------------------------------------------ generating

def generate(problem_text: str, raw_json: dict, n: int = 8, seed: int = 0, vary: list[str] | None = None, spread: float = DEFAULT_SPREAD,
             title: str = "Practice problems", include_original: bool = False) -> ProblemSet:
    """`n` verified variants of a solved problem."""
    if not 1 <= n <= MAX_PROBLEMS:
        raise ProblemSetError(f"Choose between 1 and {MAX_PROBLEMS} problems.")
    if not 0.02 <= spread <= 0.9:
        raise ProblemSetError("The spread must be between 2% and 90% of each value.")
    raw = copy.deepcopy(raw_json)
    model = build_model(copy.deepcopy(raw))
    check_supported(model)
    base_report = deterministic_report(model)
    if not base_report.passed:
        failed = [c.label for c in base_report.checks if not c.passed][:3]
        raise ProblemSetError("The source problem does not pass its own checks (" + "; ".join(failed) + "), so variants of it could not "
                              "be trusted.")
    base_answers = {t: base_report.sympy_numeric_answers[t] for t in model.solve_for if t in base_report.sympy_numeric_answers}
    if set(base_answers) != set(model.solve_for):
        raise ProblemSetError("The source problem could not be solved for every target.")
    knowns = known_variables(model)
    chosen = [s for s in (vary if vary is not None else suggest_varied(model)) if s in knowns and knowns[s] != 0.0]
    if not chosen:
        raise ProblemSetError("There is no known value to vary (constants and zeros are not varied).")

    decimals = written_decimals(problem_text, knowns, chosen)
    rng = random.Random(seed)
    ranges = {s: tuple(sorted((knowns[s] * (1 - spread), knowns[s] * (1 + spread)))) for s in chosen}
    order = {s: rng.sample(range(n), n) for s in chosen}                     # an independent bin order per variable

    def draw(k: int | None) -> dict[str, float]:
        out = dict(knowns)
        for s in chosen:
            lo, hi = ranges[s]
            if k is None:
                v = rng.uniform(lo, hi)
            else:
                width = (hi - lo) / n
                v = lo + (order[s][k] + rng.random()) * width
            out[s] = _round_like(knowns[s], v, decimals[s])
        return out

    instances: list[ProblemInstance] = []
    seen: set[tuple] = {tuple(knowns[s] for s in chosen)} if not include_original else set()
    discarded = attempts = 0
    pending: list[int | None] = list(range(n))
    while len(instances) < n and attempts < 40 * n:
        attempts += 1
        slot = pending.pop(0) if pending else None                           # stratified first, then plain random refills
        values = draw(slot)
        key = tuple(values[s] for s in chosen)
        if key in seen:
            discarded += 1
            continue
        answers = _solve_variant(raw, values, base_answers, model)
        if answers is None:
            discarded += 1
            continue
        seen.add(key)
        text, rewritten = render_statement(problem_text, knowns, values, chosen)
        given, find = _given_find(model, values)
        instances.append(ProblemInstance(len(instances) + 1, values, answers, text, rewritten, given, find))
    warnings = []
    if len(instances) < n:
        warnings.append(f"Only {len(instances)} of the {n} requested problems could be made: many random draws failed the checks. "
                        "Try a smaller spread or vary fewer values.")
    if not any(i.rewritten for i in instances):
        warnings.append("The wording could not be rewritten unambiguously (a number appears twice, or not at all), so each problem keeps the "
                        "original sentence and lists its own values underneath.")
    return ProblemSet(title, problem_text, model.problem_domain, list(model.solve_for), instances, knowns, base_answers, chosen, seed,
                      spread, discarded, datetime.now().isoformat(timespec="seconds"), warnings)


# ------------------------------------------------------------------------------------------ exporting

def _units(ps: ProblemSet) -> dict[str, str]:
    inst = ps.instances[0] if ps.instances else None
    return {f["symbol"]: f["unit"] for f in (inst.find if inst else [])}


def _given_line(i: ProblemInstance) -> str:
    return ", ".join(f"{g['symbol']} = {fmt(g['value'])}{(' ' + g['unit']) if g['unit'] else ''}" for g in i.given)


def worksheet_markdown(ps: ProblemSet, with_answers: bool = False) -> str:
    lines = [f"# {ps.title}", "", "Give each answer to 3 significant figures, with its unit.", ""]
    for i in ps.instances:
        lines.append(f"**{i.index}.** {i.text}")
        if not i.rewritten:
            lines.append(f"   *Use: {_given_line(i)}*")
        lines.append("   Find: " + "; ".join(f"{f['meaning']} ({f['symbol']})" + (f" in {f['unit']}" if f["unit"] else "") for f in i.find))
        lines.append("")
    if with_answers:
        lines += ["---", "", "## Answer key", ""]
        for i in ps.instances:
            lines.append(f"{i.index}. " + "; ".join(f"{t} = {fmt(i.answers[t])}" + (f" {_units(ps)[t]}" if _units(ps).get(t) else "") for t in ps.targets))
    return "\n".join(lines) + "\n"


def worksheet_html(ps: ProblemSet, with_answers: bool = False) -> str:
    e = html.escape
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{e(ps.title)}</title><style>"
             "body{font-family:Georgia,serif;max-width:46rem;margin:2rem auto;line-height:1.5}li{margin-bottom:1.2rem}"
             ".use{color:#555;font-style:italic}.find{color:#333}.key td{padding:.15rem .8rem}@media print{body{margin:1cm}}"
             f"</style></head><body><h1>{e(ps.title)}</h1><p>Give each answer to 3 significant figures, with its unit.</p><ol>"]
    for i in ps.instances:
        parts.append(f"<li>{e(i.text)}")
        if not i.rewritten:
            parts.append(f"<div class='use'>Use: {e(_given_line(i))}</div>")
        parts.append("<div class='find'>Find: " + e("; ".join(f"{f['meaning']} ({f['symbol']})" + (f" in {f['unit']}" if f["unit"] else "") for f in i.find)) + "</div></li>")
    parts.append("</ol>")
    if with_answers:
        parts.append("<h2>Answer key</h2><table class='key'>")
        units = _units(ps)
        for i in ps.instances:
            parts.append(f"<tr><td>{i.index}</td><td>" + e("; ".join(f"{t} = {fmt(i.answers[t])}" + (f" {units[t]}" if units.get(t) else "") for t in ps.targets)) + "</td></tr>")
        parts.append("</table>")
    return "".join(parts) + "</body></html>"


def answer_key_csv(ps: ProblemSet) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["problem", "target", "answer", "unit"] + [f"given_{s}" for s in ps.varied])
    units = _units(ps)
    for i in ps.instances:
        for t in ps.targets:
            w.writerow([i.index, t, repr(i.answers[t]), units.get(t, "")] + [i.values[s] for s in ps.varied])
    return buf.getvalue()


def submission_template_csv(ps: ProblemSet, students: list[str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["student", "problem", "target", "answer"])
    for s in students:
        for i in ps.instances:
            for t in ps.targets:
                w.writerow([s, i.index, t, ""])
    return buf.getvalue()


def to_json(ps: ProblemSet) -> str:
    d = {"format_version": FORMAT_VERSION, "title": ps.title, "source_text": ps.source_text, "domain": ps.domain, "targets": ps.targets,
         "base_values": ps.base_values, "base_answers": ps.base_answers, "varied": ps.varied, "seed": ps.seed, "spread": ps.spread,
         "discarded": ps.discarded, "created": ps.created, "warnings": ps.warnings,
         "instances": [vars(i) for i in ps.instances]}
    return json.dumps(d, indent=1)


def from_json(text: str) -> ProblemSet:
    try:
        d = json.loads(text)
        if d.get("format_version") != FORMAT_VERSION:
            raise ProblemSetError(f"Unrecognized problem-set format version {d.get('format_version')!r}.")
        inst = [ProblemInstance(**i) for i in d["instances"]]
        return ProblemSet(d["title"], d["source_text"], d["domain"], d["targets"], inst, d["base_values"], d["base_answers"], d["varied"],
                          d["seed"], d["spread"], d.get("discarded", 0), d.get("created", ""), d.get("warnings", []))
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
        raise ProblemSetError(f"This is not a problem-set file ({type(exc).__name__}: {exc}).") from exc


# ------------------------------------------------------------------------------------------ parsing an answer

_NUM_TOKEN = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:\s*(?:[eE][-+]?\d+|(?:x|×|\*)\s*10\s*\^?\s*[-+]?\d+))?")


def parse_answer(text: str) -> tuple[float | None, str]:
    """(value, unit text) from what a student typed, or (None, '') if there is no number. Accepts 12.5, 1.25e1,
    1.25 x 10^1, 1.25×10^1, 1,234.5 and 12,5 (a lone comma followed by 1-2 digits is a decimal comma)."""
    s = str(text).strip().replace("−", "-").replace("–", "-")
    m = _NUM_TOKEN.search(s)
    if not m:
        return None, ""
    tok = m.group().replace(" ", "")
    mult = re.search(r"(?:x|×|\*)10\^?([-+]?\d+)$", tok)
    exponent = 0
    if mult:
        exponent = int(mult.group(1))
        tok = tok[:mult.start()]
    if "," in tok and "." in tok:
        tok = tok.replace(",", "")
    elif tok.count(",") == 1 and len(tok.split(",")[1]) in (1, 2):
        tok = tok.replace(",", ".")
    else:
        tok = tok.replace(",", "")
    try:
        value = float(tok) * 10.0 ** exponent
    except ValueError:
        return None, ""
    return (value, s[m.end():].strip()) if math.isfinite(value) else (None, "")


# ------------------------------------------------------------------------------------------ grading one answer

@dataclass
class Verdict:
    status: str                  # correct | close | wrong | missing | unreadable
    score: float
    diagnosis: str = ""
    student_value: float | None = None


def diagnose(student: float, key: float, other_answers: dict[str, float] | None = None, base_answers: dict[str, float] | None = None,
             target: str | None = None) -> str:
    """A best guess at the slip behind a wrong answer, or '' if nothing recognisable."""
    if key == 0 or student == 0:
        return ""
    close = lambda a, b, tol=0.02: abs(a - b) <= tol * max(abs(b), 1e-300)           # noqa: E731
    for name, val in (other_answers or {}).items():
        if name != target and close(student, val):
            return f"this is the value of {name}, not {target or 'the quantity asked for'}"
    if base_answers and target in base_answers and close(student, base_answers[target]) and not close(key, base_answers[target]):
        return "this is the answer to the original problem: the new numbers were not used"
    if close(student, -key):
        return "sign error"
    ratio = student / key
    if close(ratio, 2.0, 0.03):
        return "off by a factor of 2 (an extra 2, or a missing ½)"
    if close(ratio, 0.5, 0.03):
        return "off by a factor of 2 (a missing 2, or an extra ½)"
    if ratio > 0:
        exp = math.log10(ratio)
        if abs(exp - round(exp)) < 0.02 and 1 <= abs(round(exp)) <= 6:
            return f"off by a factor of 10^{round(exp)}: check the unit conversion or decimal place"
    if close(student, 1 / key, 0.03):
        return "the reciprocal of the right answer"
    if close(ratio, math.pi / 180, 0.02) or close(ratio, 180 / math.pi, 0.02):
        return "degrees and radians confused"
    if close(student, key ** 2, 0.03) and abs(key) != 1:
        return "the square of the right answer (a missing square root?)"
    if key > 0 and close(student, math.sqrt(key), 0.03) and key != 1:
        return "the square root of the right answer (a missing square?)"
    return ""


def grade_value(student: float | None, key: float, rel_tol: float = DEFAULT_TOLERANCE, close_tol: float = CLOSE_TOLERANCE,
                partial: float = PARTIAL_CREDIT, **diag) -> Verdict:
    if student is None:
        return Verdict("missing", 0.0)
    err = abs(student - key)
    scale = max(abs(key), 1e-12)
    if err <= rel_tol * scale or err <= 1e-9:
        return Verdict("correct", 1.0, student_value=student)
    if err <= close_tol * scale:
        return Verdict("close", partial, f"within {err / scale:.1%}: check rounding or significant figures", student)
    return Verdict("wrong", 0.0, diagnose(student, key, **diag), student)


# ------------------------------------------------------------------------------------------ grading a class

@dataclass
class ItemResult:
    student: str
    problem: int
    target: str
    answer_text: str
    key: float | None
    verdict: Verdict


@dataclass
class StudentResult:
    name: str
    score: float
    possible: int
    items: list[ItemResult]

    @property
    def percent(self) -> float:
        return 100.0 * self.score / self.possible if self.possible else 0.0


@dataclass
class ProblemStats:
    problem: int
    target: str
    attempted: int
    correct: int
    facility: float                    # share of the attempts fully correct
    common_mistake: str


@dataclass
class GradeReport:
    students: list[StudentResult] = field(default_factory=list)
    problems: list[ProblemStats] = field(default_factory=list)
    problems_with_errors: list[str] = field(default_factory=list)     # unusable rows, each with its reason
    notes: list[str] = field(default_factory=list)

    @property
    def mean_percent(self) -> float:
        return sum(s.percent for s in self.students) / len(self.students) if self.students else 0.0

    def results_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["student", "problem", "target", "answer", "key", "status", "score", "note"])
        for s in self.students:
            for it in s.items:
                w.writerow([it.student, it.problem, it.target, it.answer_text, "" if it.key is None else repr(it.key), it.verdict.status,
                            it.verdict.score, it.verdict.diagnosis])
        return buf.getvalue()

    def totals_csv(self) -> str:
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["student", "score", "possible", "percent"])
        for s in self.students:
            w.writerow([s.name, s.score, s.possible, f"{s.percent:.1f}"])
        return buf.getvalue()


def read_submissions(text: str) -> list[dict]:
    """Rows of {student, problem, target, answer} from CSV text (a header row naming those columns; `target` is optional)."""
    sample = text.strip()
    if not sample:
        raise ProblemSetError("The submissions are empty.")
    rows = list(csv.DictReader(io.StringIO(sample)))
    if not rows or not {"student", "problem", "answer"} <= {(k or "").strip().lower() for k in rows[0]}:
        raise ProblemSetError("Submissions need a header row with the columns: student, problem, answer (and optionally target).")
    return [{(k or "").strip().lower(): (v or "").strip() for k, v in r.items()} for r in rows]


def grade_submissions(ps: ProblemSet, rows: list[dict], rel_tol: float = DEFAULT_TOLERANCE, close_tol: float = CLOSE_TOLERANCE,
                      partial: float = PARTIAL_CREDIT) -> GradeReport:
    report = GradeReport()
    by_index = {i.index: i for i in ps.instances}
    answers: dict[tuple[str, int, str], str] = {}
    order: list[str] = []
    for n, r in enumerate(rows, start=2):                             # row 1 is the header
        student = r.get("student", "").strip()
        if not student:
            report.problems_with_errors.append(f"row {n}: no student name")
            continue
        try:
            idx = int(float(r.get("problem", "")))
        except ValueError:
            report.problems_with_errors.append(f"row {n}: '{r.get('problem', '')}' is not a problem number")
            continue
        if idx not in by_index:
            report.problems_with_errors.append(f"row {n}: there is no problem {idx} in this set")
            continue
        target = r.get("target", "").strip() or (ps.targets[0] if len(ps.targets) == 1 else "")
        if target not in ps.targets:
            report.problems_with_errors.append(f"row {n}: '{target}' is not one of the quantities to find ({', '.join(ps.targets)})")
            continue
        key = (student, idx, target)
        if key in answers:
            report.notes.append(f"{student} answered problem {idx} ({target}) twice; the last answer was used.")
        answers[key] = r.get("answer", "")
        if student not in order:
            order.append(student)

    mistakes: dict[tuple[int, str], dict[str, int]] = {}
    stats: dict[tuple[int, str], list[Verdict]] = {}
    for student in order:
        items = []
        for inst in ps.instances:
            for t in ps.targets:
                text = answers.get((student, inst.index, t), "")
                value, _unit = parse_answer(text) if text.strip() else (None, "")
                if text.strip() and value is None:
                    verdict = Verdict("unreadable", 0.0, "no number found in the answer")
                else:
                    verdict = grade_value(value, inst.answers[t], rel_tol, close_tol, partial, other_answers=inst.answers,
                                          base_answers=ps.base_answers, target=t)
                items.append(ItemResult(student, inst.index, t, text, inst.answers[t], verdict))
                stats.setdefault((inst.index, t), []).append(verdict)
                if verdict.status in ("wrong", "close") and verdict.diagnosis:
                    mistakes.setdefault((inst.index, t), {}).setdefault(verdict.diagnosis, 0)
                    mistakes[(inst.index, t)][verdict.diagnosis] += 1
        report.students.append(StudentResult(student, sum(i.verdict.score for i in items), len(items), items))

    for (idx, t), verdicts in sorted(stats.items()):
        attempted = sum(v.status != "missing" for v in verdicts)
        correct = sum(v.status == "correct" for v in verdicts)
        common = max(mistakes.get((idx, t), {}).items(), key=lambda kv: kv[1], default=("", 0))
        report.problems.append(ProblemStats(idx, t, attempted, correct, correct / len(verdicts), common[0] if common[1] >= 2 or
                                            (common[1] == 1 and len(verdicts) <= 3) else ""))
    if report.students:
        none_right = [f"{p.problem} ({p.target})" for p in report.problems if p.attempted and p.correct == 0]
        if none_right:
            report.notes.append("Nobody got " + ", ".join(none_right) + " right: the key is verified, so check the wording or whether it "
                                "was taught.")
    return report
