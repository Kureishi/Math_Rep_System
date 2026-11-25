"""
Prose that ends up where LaTeX is expected.

A solution step's `expression` is LaTeX, but the "Technique" step holds a sentence ("linear in the target: move every other term to the other side, ..."). Typeset as math, a
sentence loses its spaces (math mode ignores them) and its letters turn italic, so it came out as "linearinthetarget:moveeveryothertermtotheotherside...". `display_latex` wraps such a
sentence in \\text{...} (wrapped onto several lines when it is long) and leaves real math alone, so every place that typesets a step can pass the expression through it. It is idempotent,
and it works on steps saved in History long ago, whose stored expression is still the bare sentence. Streamlit-free.
"""
import re
import textwrap

_WORD = re.compile(r"(?<![A-Za-z])[A-Za-z]{3,}(?![A-Za-z])")
_MATH_MARKS = set("\\={}^_$")
_ESCAPES = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_", "^": r"\^{}", "~": r"\~{}"}


def looks_like_prose(expr: str) -> bool:
    """A sentence rather than math: at least three words of three or more letters, and none of LaTeX's marks (a backslash, =, braces, ^, _, $). Real math in a step has an
    '=' or a command; the technique descriptions have neither."""
    text = (expr or "").strip()
    return bool(text) and not (_MATH_MARKS & set(text)) and len(_WORD.findall(text)) >= 3 and " " in text


def prose_to_latex(text: str, width: int | None = 64) -> str:
    """`text` as \\text{...}; with `width`, wrapped into lines of about that many characters inside \\begin{gathered}. Special characters are escaped."""
    def esc(s: str) -> str:
        return "".join(_ESCAPES.get(c, c) for c in s)
    text = " ".join(text.split())
    lines = textwrap.wrap(text, width=width) if width else [text]
    if len(lines) <= 1:
        return r"\text{" + esc(text) + "}"
    return r"\begin{gathered}" + r" \\ ".join(r"\text{" + esc(line) + "}" for line in lines) + r"\end{gathered}"


def display_latex(expr: str, width: int | None = 64) -> str:
    """What to hand to st.latex (width=64), or to a LaTeX document (width=None): prose becomes \\text{...}, anything else is returned unchanged."""
    return prose_to_latex(expr, width) if looks_like_prose(expr) else expr


_TEX_SPECIALS = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                 "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex_escape(text: str) -> str:
    """`text` made safe to use inside \\text{...} (or running LaTeX text)."""
    return "".join(_TEX_SPECIALS.get(ch, ch) for ch in text)


def unit_latex(unit: str | None) -> str:
    """A unit string as LaTeX math: 'm/s^2' -> '\\,\\mathrm{m/s}^{2}'. Falls back to escaped text for
    anything that is not a plain product/quotient of unit names with integer exponents."""
    if not unit or unit.strip().lower() in ("unitless", "dimensionless", "none", "1"):
        return ""
    u = unit.strip().replace("**", "^")
    m = re.fullmatch(r"([A-Za-z°µΩ/\*\s\.\-]+?)\^(-?\d+)", u)
    if m:
        return rf"\,\text{{{tex_escape(m.group(1).strip())}}}^{{{m.group(2)}}}"
    # exponents inside a quotient, e.g. "kg*m/s^2": convert each name^n
    parts = re.split(r"(\^-?\d+)", u)
    out = ""
    for p in parts:
        if re.fullmatch(r"\^-?\d+", p):
            out += "^{" + p[1:] + "}"
        else:
            out += rf"\text{{{tex_escape(p)}}}" if p else ""
    return r"\," + out


