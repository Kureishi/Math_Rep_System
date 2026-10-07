"""
The report as ONE self-contained HTML file: equations typeset, plots live.

Open it in any browser, offline, with nothing else: no server, no CDN, no installed fonts. What makes that
true, and what each choice costs:

  * Equations are inline SVG drawn by matplotlib's mathtext (the engine the PDF already uses), with the
    glyphs converted to paths -- so they need no font and no JavaScript, scale without blurring, and take
    the page's text colour (so they stay readable in dark mode). Clicking one copies its LaTeX. mathtext
    cannot typeset everything SymPy emits; an equation it cannot do is shown as its LaTeX source in a
    code box rather than dropped.
  * Plots are live Plotly figures (zoom, pan, hover, toggle traces), not screenshots: every plot you marked
    with "Include this plot" that was captured live, plus a sensitivity (tornado) chart per target and a
    histogram per Monte Carlo run. A plot with no live form (a matplotlib-only figure) is embedded as its
    image. Making plots live means embedding the Plotly library itself, about 5 MB, once; a report with
    no live plots has no JavaScript library at all and is a few kilobytes.
  * Everything else (tables, pass/fail lists, the reproducibility footer) is plain HTML and CSS, with a
    print stylesheet.

No scripts are loaded from anywhere; the one script block holds the figures' JSON and the code that draws
them. Text is HTML-escaped throughout, since problem text, step explanations and follow-up answers come from
a language model and from the user.
"""
import base64
import functools
import html
import io
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.report_content import REPORT_TITLE, Section, generated_stamp

_CSS = """
:root{--bg:#fff;--fg:#1d2230;--muted:#667;--card:#f6f8fb;--line:#d9dfe8;--ok:#1a7f45;--bad:#c0392b;--accent:#2b5fd9}
@media (prefers-color-scheme:dark){:root{--bg:#14171f;--fg:#e6e9f0;--muted:#9aa3b5;--card:#1d222e;--line:#313a4d;--ok:#4cc17e;--bad:#ff7a6b;--accent:#7aa2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header,main,footer{max-width:60rem;margin:0 auto;padding:0 1.2rem}header{padding-top:2rem}h1{margin:.2rem 0}
.stamp{color:var(--muted);font-size:.9rem}nav{margin:1rem 0;padding:.6rem .9rem;background:var(--card);border:1px solid var(--line);border-radius:10px;font-size:.92rem}
nav a{color:var(--accent);text-decoration:none;margin-right:.9rem;white-space:nowrap}nav a:hover{text-decoration:underline}
section{margin:2rem 0}h2{border-bottom:2px solid var(--line);padding-bottom:.25rem}h3{margin-bottom:.2rem}
.lbl{font-weight:600;margin:.9rem 0 .15rem}.italic{font-style:italic;color:var(--muted)}
.eq{margin:.35rem 0;overflow-x:auto;cursor:copy}.eq svg{max-width:100%;height:auto;vertical-align:middle}
.eq code{white-space:pre-wrap}.eq.copied{outline:2px solid var(--accent);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:.93rem;margin:.5rem 0;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:.35rem .6rem;text-align:left;vertical-align:top}th{background:var(--card)}
ul.st{list-style:none;padding-left:0}ul.st li{padding:.12rem 0}.ok{color:var(--ok);font-weight:700}.bad{color:var(--bad);font-weight:700}
dl{display:grid;grid-template-columns:max-content 1fr;gap:.2rem 1rem}dt{font-weight:600}dd{margin:0}
figure{margin:1rem 0}figcaption{color:var(--muted);font-size:.9rem}.plot{background:#fff;border:1px solid var(--line);border-radius:10px;min-height:340px}
figure img{max-width:100%;border-radius:8px;background:#fff}
footer{color:var(--muted);font-size:.85rem;padding-bottom:2rem}
@media print{nav,.noprint{display:none}body{background:#fff;color:#000}.plot{break-inside:avoid}}
"""

_JS_COPY = """
document.querySelectorAll('.eq[data-latex]').forEach(function(el){
  el.title='Click to copy the LaTeX';
  var done=function(){el.classList.add('copied');setTimeout(function(){el.classList.remove('copied')},700)};
  var legacy=function(t){
    // the async clipboard API is often refused on file:// pages; a hidden textarea and execCommand still works
    var a=document.createElement('textarea');a.value=t;a.style.position='fixed';a.style.opacity='0';
    document.body.appendChild(a);a.select();try{document.execCommand('copy')}catch(e){}document.body.removeChild(a);done();
  };
  el.addEventListener('click',function(){
    var t=el.getAttribute('data-latex');
    if(navigator.clipboard&&navigator.clipboard.writeText){navigator.clipboard.writeText(t).then(done,function(){legacy(t)})}else{legacy(t)}
  });
});
"""


@functools.lru_cache(maxsize=512)
def latex_to_svg(latex: str, fontsize: int = 14) -> str | None:
    """`latex` typeset as an inline SVG string that takes the surrounding text colour, or None if
    matplotlib's mathtext cannot parse it."""
    text = latex.strip()
    if not (text.startswith("$") and text.endswith("$")):
        text = f"${text}$"
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0)
    try:
        fig.text(0, 0, text, fontsize=fontsize, color="#000000")
        buf = io.BytesIO()
        with matplotlib.rc_context({"svg.fonttype": "path"}):
            fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.04, transparent=True)
        svg = buf.getvalue().decode("utf-8")
    except Exception:  # noqa: BLE001 -- mathtext can't parse everything SymPy emits
        return None
    finally:
        plt.close(fig)
    svg = svg[svg.index("<svg"):]                       # drop the XML prolog and DOCTYPE
    svg = re.sub(r"<metadata>.*?</metadata>", "", svg, flags=re.S)       # matplotlib's credit and date: noise
    # The glyph paths name no fill, so they would be black whatever the page's colours are. A fill on the
    # root is inherited by all of them, and `currentColor` is the surrounding text colour.
    return svg.replace("<svg ", '<svg fill="currentColor" ', 1)


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


def _json_for_script(obj_json: str) -> str:
    """JSON safe to place inside a <script> element: nothing in it can close the tag or start a comment."""
    return obj_json.replace("</", "<\\/").replace("<!--", "<\\!--")


def _slug(sec: Section) -> str:
    return f"sec-{sec.id}"


def _plotly_js() -> str:
    from plotly.offline import get_plotlyjs
    return get_plotlyjs()


def render_html(sections: list[Section], title: str = REPORT_TITLE) -> str:
    """The sections as a complete HTML document."""
    body: list[str] = []
    figures: list[tuple[str, str]] = []        # (element id, figure JSON)
    for sec in sections:
        body.append(f'<section id="{_slug(sec)}"><h2>{_esc(sec.title)}</h2>')
        for b in sec.blocks:
            if b.kind == "text":
                cls = {"italic": ' class="italic"', "bold": ' style="font-weight:600"'}.get(b.style, "")
                body.append(f"<p{cls}>{_esc(b.text)}</p>")
            elif b.kind == "label":
                body.append(f'<div class="lbl">{_esc(b.text)}</div>')
            elif b.kind == "subheading":
                body.append(f"<h3>{_esc(b.text)}</h3>")
            elif b.kind == "equation":
                svg = None if b.style == "raw" else latex_to_svg(b.text)
                inner = svg if svg else f"<code>{_esc(b.text)}</code>"
                body.append(f'<div class="eq" data-latex="{_esc(b.text)}">{inner}</div>')
            elif b.kind == "bullets":
                body.append("<ul>" + "".join(f"<li>{_esc(i)}</li>" for i in b.items) + "</ul>")
            elif b.kind == "table":
                head = "".join(f"<th>{_esc(h)}</th>" for h in b.headers)
                rows = "".join("<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in row) + "</tr>" for row in b.rows)
                body.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>")
            elif b.kind == "status":
                items = "".join(
                    f'<li><span class="{"ok" if ok else "bad"}">{"✔" if ok else "✖"}</span> <b>{_esc(label)}:</b> {_esc(detail)}</li>'
                    for ok, label, detail in b.rows)
                body.append(f'<ul class="st">{items}</ul>')
            elif b.kind == "kv":
                body.append("<dl>" + "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in b.rows) + "</dl>")
            elif b.kind in ("image", "figure"):
                caption = f"<figcaption><b>{_esc(b.label_text)}</b>" + (f" -- {_esc(b.text)}" if b.text else "") + "</figcaption>"
                if b.figure_json:
                    fid = f"plot-{len(figures)}"
                    figures.append((fid, b.figure_json))
                    body.append(f'<figure><div class="plot" id="{fid}" role="img" aria-label="{_esc(b.label_text)}"></div>{caption}</figure>')
                elif b.png:
                    b64 = base64.b64encode(b.png).decode("ascii")
                    body.append(f'<figure><img alt="{_esc(b.label_text)}" src="data:image/png;base64,{b64}">{caption}</figure>')
        body.append("</section>")

    nav = " ".join(f'<a href="#{_slug(s)}">{_esc(s.title)}</a>' for s in sections)
    scripts = f"<script>{_JS_COPY}</script>"
    if figures:
        payload = "[" + ",".join(f'{{"id":{json.dumps(fid)},"fig":{fj}}}' for fid, fj in figures) + "]"
        scripts = (f"<script>{_plotly_js()}</script>\n"
                   f"<script>var PLOTS={_json_for_script(payload)};"
                   "PLOTS.forEach(function(p){Plotly.newPlot(p.id,p.fig.data,p.fig.layout,{responsive:true,displaylogo:false});});"
                   f"{_JS_COPY}</script>")
    return (f'<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_esc(title)}</title>'
            f"<style>{_CSS}</style></head><body>"
            f'<header><h1>{_esc(title)}</h1><div class="stamp">{_esc(generated_stamp())}</div>'
            f'<nav class="noprint" aria-label="Sections">{nav}</nav></header>'
            f"<main>{''.join(body)}</main>"
            f'<footer>Click any equation to copy its LaTeX. Plots are interactive.</footer>{scripts}</body></html>')


def count_live_figures(html_text: str) -> int:
    """How many live plots a rendered report contains (for the UI to say what the file holds)."""
    return len(re.findall(r'class="plot" id="plot-\d+"', html_text))
