"""
Sig-fig discipline check: tracks the PRECISION implied by the problem's
own given inputs (e.g. "8 m/s" implies 1 significant figure, "8.0 m/s"
implies 2) and flags when a reported final answer carries implausibly
MORE precision than those inputs could actually support -- a classic
thing intro science/engineering grading cares about that nothing else
in this pipeline checks. plausibility.py flags a value that's the wrong
MAGNITUDE; this flags a value that's stated with the wrong PRECISION,
an entirely different (and much more commonly graded) kind of mistake.

Sig-fig counting is done on the ORIGINAL TEXT of each known value (the
digit string as typed/extracted), not on the parsed float -- a float
has already lost the distinction between "8" (1 sig fig) and "8.0" (2
sig figs); both parse to the same Python float 8.0. This means an
accurate count depends on the raw string surviving from extraction
through to here, which is why this module takes the raw known-value
strings directly rather than trying to reverse-engineer precision from
an already-parsed Variable.known_value.

The combination rule used here -- for multiplication/division, the
result's sig figs are capped at the FEWEST any input has; for
addition/subtraction, the result's DECIMAL PLACES are capped at the
fewest any input has -- is the standard rule taught alongside the
concept itself, not a rule this module invents. Like plausibility.py,
this is advisory only: it flags a mismatch worth a second look, it
doesn't (and structurally can't, without re-deriving which operations
combined which inputs) claim to compute the exact propagated
uncertainty the way error_propagation.py does.
"""
import re
from dataclasses import dataclass


@dataclass
class SigFigNote:
    value_text: str          # the answer as it would be displayed, e.g. "2.667"
    reported_figures: int
    supported_figures: int   # the fewest sig figs among the given inputs
    message: str


_SIG_FIG_DIGITS_RE = re.compile(r"[0-9]")
_TRAILING_ZEROS_NO_DECIMAL_RE = re.compile(r"^-?[1-9][0-9]*?(0+)$")


def count_significant_figures(raw: str) -> int | None:
    """Counts significant figures in a raw numeric string using the
    standard textbook rules:
      - leading zeros never count ("0.0034" -> 2)
      - zeros between nonzero digits always count ("1002" -> 4)
      - trailing zeros count ONLY if there's a decimal point
        ("100" -> 1, ambiguous by convention; "100." -> 3; "100.0" -> 4)
      - a bare integer with trailing zeros and no decimal point is the
        one genuinely AMBIGUOUS case in this whole scheme (by long-
        standing convention "100" is read as 1 sig fig, the conservative
        reading, even though a writer might have meant more)
    Returns None if `raw` doesn't look like a plain decimal number at
    all (scientific notation, a fraction, non-numeric text) -- this
    module simply has no opinion on those rather than guessing wrong."""
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None
    if re.search(r"[eE]", text):
        return None  # scientific notation not handled -- see docstring
    if not re.fullmatch(r"-?[0-9]*\.?[0-9]+", text) and not re.fullmatch(r"-?[0-9]+\.?[0-9]*", text):
        return None
    if not _SIG_FIG_DIGITS_RE.search(text):
        return None

    sign_stripped = text.lstrip("-")
    if "." in sign_stripped:
        left, right = sign_stripped.split(".", 1)
        digits = (left + right).lstrip("0")
        if not digits:
            # e.g. "0.000" -- all zeros; conservatively count the zeros
            # right of the decimal point
            return max(1, len(right))
        first_nonzero = next(i for i, c in enumerate(left + right) if c != "0")
        return len(left + right) - first_nonzero
    else:
        stripped = sign_stripped.lstrip("0")
        if not stripped:
            return 1  # "0" itself
        m = _TRAILING_ZEROS_NO_DECIMAL_RE.match(stripped)
        if m:
            return len(stripped) - len(m.group(1))
        return len(stripped)


def raw_known_value_strings(model) -> dict[str, str]:
    """Pulls each known variable's ORIGINAL known_value text straight
    from the model's own raw_json -- the parsed Variable.known_value
    float has already lost the "8" vs "8.0" distinction that sig-fig
    counting depends on, but the raw extraction JSON (model.raw_json)
    hasn't. Skips any variable whose raw known_value isn't a plain
    string (null, missing, or an unexpected type)."""
    texts: dict[str, str] = {}
    for entry in model.raw_json.get("variables", []):
        symbol = entry.get("symbol")
        raw_value = entry.get("known_value")
        if symbol and isinstance(raw_value, str) and raw_value.strip():
            texts[symbol] = raw_value
    return texts


def check_sig_figs(known_value_texts: dict[str, str], answer_value: float,
                    answer_symbol: str) -> SigFigNote | None:
    """Compares the precision an answer is displayed with against what
    the given inputs actually support. `known_value_texts` maps variable
    symbol -> its ORIGINAL raw text (not the parsed float -- see module
    docstring for why). Returns None when there's nothing to compare (no
    known values with countable precision) rather than a note claiming
    false confidence."""
    input_figures_raw = [count_significant_figures(text) for text in known_value_texts.values()]
    input_figures = [f for f in input_figures_raw if f is not None]
    if not input_figures:
        return None

    supported = min(input_figures)
    value_text = f"{answer_value:.10g}"
    reported = count_significant_figures(value_text)
    if reported is None:
        return None

    # allow one extra digit of slack -- flagging every single-digit
    # overage would be noisy for a rule that's already an approximation
    # (the real propagated uncertainty, see error_propagation.py, is
    # rarely EXACTLY at the sig-fig-rule boundary)
    if reported <= supported + 1:
        return None

    rounded = f"{answer_value:.{max(supported, 1)}g}"
    return SigFigNote(
        value_text=value_text, reported_figures=reported, supported_figures=supported,
        message=(f"{answer_symbol} = {value_text} is reported to {reported} significant figures, "
                  f"but the given inputs only support {supported} -- consider rounding to "
                  f"{answer_symbol} \u2248 {rounded}."),
    )
