"""
The answer card: one shareable image -- the problem, the answer, the verdict -- for a chat or a class forum. Streamlit-free.

    card = from_solution(problem_text, model, report)        # a solved word problem
    card = from_checks("Double slit", ["fringe spacing 3.3 mm"], checks)   # a result of one of the labs
    png = render_png(card)                                    # 1200 x 630, the proportions chat and forum previews crop to
    card.alt_text, card.caption                               # to post with the image

HONESTY. The verdict on the card is the same one the app shows (modules/proof_card.trust_summary for a solved problem, the lab's own checks for a lab
result), in words as well as colour: the symbol (check mark, exclamation mark, cross) and the text carry the meaning, so it reads in greyscale and for colour-blind
readers. A failing result is NOT dressed up: the card says "Not verified". A card carries a short proof id (the hash of what was checked), the same one in the
report's appendix, so a card and a report can be matched. The footer says where a language model was involved (it reads the equations from a word problem's text; the checks are computed) or that none was (the labs).
"""
import hashlib
import io
import textwrap
from dataclasses import dataclass
from datetime import datetime

from config import APP_VERSION
from modules.palette import BADGE_COLORS, BLUE, GRAY, VERMILLION

WIDTH, HEIGHT = 1200, 630
SUPERSCRIPTS = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")
MAX_PROBLEM_CHARS = 230
MAX_ANSWERS = 4

_VERDICT = {
    "verified": ("✓", "Verified", BADGE_COLORS["pass"], BLUE),
    "caution": ("!", "Passed, with less behind it", BADGE_COLORS["warn"], "#C48700"),
    "failed": ("✕", "Not verified", BADGE_COLORS["fail"], VERMILLION),
}


@dataclass
class AnswerCard:
    title: str
    problem: str
    answers: list[str]
    level: str                              # verified | caution | failed
    verdict_detail: str = ""
    proof_id: str = ""
    kind: str = "Solved problem"
    footnote: str = "computed and checked; no language model involved"

    @property
    def verdict(self) -> str:
        return _VERDICT[self.level][1]

    @property
    def alt_text(self) -> str:
        ans = "; ".join(self.answers) if self.answers else "no single numeric answer"
        return (f"Answer card. {self.kind}: {self.problem} Answer: {ans}. Verdict: {self.verdict}"
                + (f" ({self.verdict_detail})" if self.verdict_detail else "") + (f". Proof id {self.proof_id}." if self.proof_id else "."))

    @property
    def caption(self) -> str:
        sym = _VERDICT[self.level][0]
        lines = [self.problem, "", *(f"  {a}" for a in self.answers), "", f"{sym} {self.verdict}" + (f" -- {self.verdict_detail}" if self.verdict_detail else "")]
        if self.proof_id:
            lines.append(f"proof #{self.proof_id}")
        return "\n".join(lines)


def _pretty_unit(unit: str | None) -> str:
    if not unit or unit.strip().lower() in ("unitless", "dimensionless", "none", "1"):
        return ""
    out, i, u = [], 0, unit.strip().replace("**", "^")
    while i < len(u):
        if u[i] == "^":
            j = i + 1
            if j < len(u) and u[j] in "+-":
                j += 1
            while j < len(u) and u[j].isdigit():
                j += 1
            out.append(u[i + 1:j].translate(SUPERSCRIPTS))
            i = j
        else:
            out.append(u[i])
            i += 1
    return " " + "".join(out).replace("*", "·")


def from_solution(problem_text: str, model, report, max_answers: int = MAX_ANSWERS) -> AnswerCard:
    """A card for a solved word problem: its numeric answers with units and the proof card's verdict."""
    from modules import proof_card as pc
    units = {v.symbol: v.unit for v in model.variables}
    answers = [f"{t} = {v:.6g}{_pretty_unit(units.get(t))}" for t, v in list(report.sympy_numeric_answers.items())[:max_answers]]
    if not answers:
        answers = ["no single numeric answer: see the derived equations"]
    summary = pc.trust_summary(model, report)
    manifest = pc.build_manifest(problem_text, model, report)
    detail = summary.reasons[0].rstrip(".") if summary.reasons else ""
    return AnswerCard("Math Representation System", " ".join(problem_text.split()), answers, summary.level, detail, manifest["content_sha256"][:12],
                      footnote="equations read from the text by a language model; every check computed")


def from_checks(title: str, answers: list[str], checks, problem: str = "") -> AnswerCard:
    """A card for a lab result: lines worth sharing (the headline numbers) and the verdict from the checks that ran on it."""
    n = len(checks.checks)
    if n == 0:
        level, detail = "caution", "no checks ran on this result"
    elif checks.passed:
        level, detail = "verified", f"{checks.n_passed} of {n} checks passed"
    else:
        level, detail = "failed", f"{checks.n_passed} of {n} checks passed"
    ident = hashlib.sha256("|".join([title, problem, *answers, *(f"{c.label}:{c.passed}" for c in checks.checks)]).encode("utf-8")).hexdigest()[:12]
    return AnswerCard("Math Representation System", problem or title, answers[:MAX_ANSWERS], level, detail, ident, kind=title)


def _wrap(text: str, width: int, max_lines: int) -> list[str]:
    lines = textwrap.wrap(text, width=width) or [""]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" .,;:")[: max(0, width - 2)] + " …"
    return lines


def render_png(card: AnswerCard, dark: bool = False) -> bytes:
    """The card as a 1200 x 630 PNG (light by default; `dark` for dark-themed forums)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    bg, ink, muted = ("#0E1117", "#E8E8E8", "#9AA0AA") if dark else ("#FBFBF8", "#1D2230", "#5B6170")
    symbol, label, (tint, text_col), accent = _VERDICT[card.level]
    if dark:
        tint, text_col = "#1C2230", accent
    dpi = 100
    fig = plt.figure(figsize=(WIDTH / dpi, HEIGHT / dpi), dpi=dpi, facecolor=bg)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 18, HEIGHT, color=accent, lw=0))
    ax.text(60, 52, card.title, fontsize=15, color=muted, va="center", fontweight="bold")
    ax.text(WIDTH - 50, 52, card.kind, fontsize=15, color=muted, va="center", ha="right")
    ax.plot([60, WIDTH - 50], [82, 82], color=GRAY, lw=1, alpha=0.5)
    y = 125
    for line in _wrap(card.problem if len(card.problem) <= MAX_PROBLEM_CHARS else card.problem[:MAX_PROBLEM_CHARS].rsplit(" ", 1)[0] + " …", 62, 4):
        ax.text(60, y, line, fontsize=19, color=ink, va="center")
        y += 34
    y = max(y + 28, 285)
    size = 40 if len(card.answers) <= 2 else 32 if len(card.answers) == 3 else 27
    step = 62 if len(card.answers) <= 2 else 52 if len(card.answers) == 3 else 44
    for ans in card.answers:
        shown = ans if len(ans) <= 44 else ans[:41] + "…"
        ax.text(60, y, shown, fontsize=size, color=ink, va="center", fontweight="bold")
        y += step
    ax.add_patch(FancyBboxPatch((60, 510), 690, 62, boxstyle="round,pad=0,rounding_size=14", fc=tint, ec=accent, lw=2))
    ax.text(86, 541, symbol, fontsize=28, color=text_col, va="center", ha="center", fontweight="bold")
    ax.text(120, 541, label, fontsize=23, color=text_col, va="center", fontweight="bold")
    if card.verdict_detail:
        ax.text(WIDTH - 50, 541, _wrap(card.verdict_detail, 40, 2)[0], fontsize=14, color=muted, va="center", ha="right")
    foot = f"proof #{card.proof_id}   ·   app {APP_VERSION}   ·   {datetime.now().strftime('%Y-%m-%d')}" if card.proof_id else f"app {APP_VERSION}   ·   {datetime.now().strftime('%Y-%m-%d')}"
    ax.text(60, 604, foot, fontsize=12, color=muted, va="center")
    ax.text(WIDTH - 50, 604, card.footnote, fontsize=12, color=muted, va="center", ha="right")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=bg)
    plt.close(fig)
    return buf.getvalue()
