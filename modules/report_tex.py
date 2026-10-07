"""
The report as a LaTeX document (.tex) you can compile and edit -- Streamlit-free.

`render_tex()` returns the source and a dict of the image files it refers to (`figures/plot-1.png`, ...):
a single .tex file cannot carry pictures, so a report with marked plots is delivered as a small zip
(see `package_tex()`), and one without any is just the .tex.

It uses only common packages (amsmath, amssymb, graphicx, booktabs, tabularx, hyperref) and compiles with
plain pdflatex. Text is escaped (`share_text.tex_escape`) and the Unicode the app uses in prose (arrows,
plus-minus, dashes, the pass/fail marks) is mapped to LaTeX equivalents, because pdflatex's default
fonts have no glyph for many of them. Equations are SymPy's own LaTeX, unchanged; an equation that never
parsed (so only its raw text exists) is set as monospaced text rather than as math, since raw text like
`Eq(v, u + a*t)` is not valid math and would stop the compile.

Live-only figures (the HTML report's interactive charts) have no image and are left out.
"""
import io
import re
import zipfile

from modules.report_content import REPORT_TITLE, Section
from modules.share_text import tex_escape

_UNICODE = {"→": r"$\to$", "←": r"$\leftarrow$", "±": r"$\pm$", "×": r"$\times$", "≈": r"$\approx$", "≤": r"$\leq$",
            "≥": r"$\geq$", "≠": r"$\neq$", "—": "---", "–": "--", "…": r"\ldots{}", "°": r"$^\circ$",
            "µ": r"$\mu$", "Ω": r"$\Omega$", "’": "'", "‘": "`", "“": "``", "”": "''", "·": r"$\cdot$",
            "✅": "[PASS]", "❌": "[FAIL]", "⚠️": "[!]", "✔": "[PASS]", "✖": "[FAIL]", "📸": ""}

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs,tabularx}
\usepackage[margin=2.2cm]{geometry}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.5em}
\sloppy
"""


def tex_text(text: str) -> str:
    """Prose made safe for LaTeX: special characters escaped, common Unicode mapped."""
    out = []
    for ch in str(text):
        if ch in _UNICODE:
            out.append(_UNICODE[ch])
        else:
            out.append(tex_escape(ch))
    return "".join(out)


def _table(headers: list[str], rows: list[list]) -> list[str]:
    n = len(headers)
    spec = "l" * (n - 1) + "X"
    lines = [r"{\small", rf"\begin{{tabularx}}{{\linewidth}}{{{spec}}}", r"\toprule",
             " & ".join(rf"\textbf{{{tex_text(h)}}}" for h in headers) + r" \\", r"\midrule"]
    lines += [" & ".join(tex_text(c) for c in row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabularx}", "}", ""]
    return lines


def render_tex(sections: list[Section], title: str = REPORT_TITLE) -> tuple[str, dict[str, bytes]]:
    """(LaTeX source, {relative path: PNG bytes} for each \\includegraphics)."""
    from modules.report_content import generated_stamp
    files: dict[str, bytes] = {}
    L = [PREAMBLE, rf"\title{{{tex_text(title)}}}", r"\date{" + tex_text(generated_stamp().replace("Generated ", "")) + "}",
         r"\author{}", r"\begin{document}", r"\maketitle", ""]
    for sec in sections:
        L += [rf"\section*{{{tex_text(sec.title)}}}", ""]
        for b in sec.blocks:
            if b.kind == "text":
                body = tex_text(b.text)
                L += [{"italic": rf"\emph{{{body}}}", "bold": rf"\textbf{{{body}}}"}.get(b.style, body), ""]
            elif b.kind == "label":
                L += [rf"\textbf{{{tex_text(b.text)}}}\par\nopagebreak", ""]
            elif b.kind == "subheading":
                L += [rf"\subsection*{{{tex_text(b.text)}}}", ""]
            elif b.kind == "equation":
                if b.style == "raw":
                    L += [rf"\texttt{{{tex_escape(b.text)}}}", ""]
                else:
                    L += [r"\begin{equation*}", b.text, r"\end{equation*}", ""]
            elif b.kind == "bullets":
                L += [r"\begin{itemize}"] + [rf"\item {tex_text(i)}" for i in b.items] + [r"\end{itemize}", ""]
            elif b.kind == "table":
                L += _table(b.headers, b.rows)
            elif b.kind == "status":
                L += [r"\begin{itemize}"] + [
                    rf"\item \textbf{{[{'PASS' if ok else 'FAIL'}]}} {tex_text(label)}: {tex_text(detail)}"
                    for ok, label, detail in b.rows] + [r"\end{itemize}", ""]
            elif b.kind == "kv":
                L += [r"\begin{description}"] + [rf"\item[{tex_text(k)}] {tex_text(v)}" for k, v in b.rows] + [r"\end{description}", ""]
            elif b.kind == "image" and b.png:
                name = f"figures/plot-{len(files) + 1}.png"
                files[name] = b.png
                L += [r"\begin{figure}[h]", r"\centering",
                      rf"\includegraphics[width=0.9\linewidth]{{{name}}}",
                      rf"\caption{{\textbf{{{tex_text(b.label_text)}}} --- {tex_text(b.text)}}}", r"\end{figure}", ""]
    L.append(r"\end{document}")
    return "\n".join(L) + "\n", files


def package_tex(sections: list[Section], title: str = REPORT_TITLE) -> tuple[str, bytes, str]:
    """(file name, bytes, mime type) to hand to a download button: the bare .tex if it needs no images,
    else a zip holding report.tex and figures/."""
    source, files = render_tex(sections, title)
    if not files:
        return "report.tex", source.encode("utf-8"), "application/x-tex"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("report.tex", source)
        for name, data in files.items():
            z.writestr(name, data)
    return "report_tex.zip", buf.getvalue(), "application/zip"


def looks_balanced(source: str) -> bool:
    """A cheap sanity check used by tests: every \\begin has its \\end, in order."""
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", source):
        if m.group(1) == "begin":
            stack.append(m.group(2))
        elif not stack or stack.pop() != m.group(2):
            return False
    return not stack
