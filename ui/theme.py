"""
A single place for the app's visual polish, so "make it look more
professional" is one file to touch rather than CSS strings scattered
across every page. Two entry points:

- inject_base_styles(): always-on layout/typography/component polish
  (card-style containers, refined buttons, badge pills, tighter spacing).
  Called once from app.py, before anything else renders.
- dark_mode_css(): the CSS override string for the sidebar's existing
  "🌙 Dark mode" toggle (moved here from ui/sidebar.py -- same data-testid
  targeting, same limitations, just kept alongside the rest of the app's
  styling instead of inline in the sidebar's own render function).

Neither function touches Streamlit's [theme] section in .streamlit/config.toml
(primaryColor etc.) -- that's the base palette; this module only adds CSS
Streamlit's theme system has no knobs for (card borders, badge pills, button
hover motion).
"""
from modules.palette import BADGE_COLORS
import streamlit as st

# Small, reusable color tokens for the status badges below -- kept in one
# place rather than repeated per call site, and distinct from the
# Streamlit-level theme colors in config.toml.
_BADGE_COLORS = BADGE_COLORS       # (tint, text) per kind: chosen for text contrast and colour-blind distinguishability (modules/palette.py)
_SYMBOL = {"pass": "✓", "warn": "!", "fail": "✕"}      # a cue that does not depend on colour


def inject_base_styles() -> None:
    """Always-on CSS: called once per script run from app.py. Idempotent
    and cheap (Streamlit dedupes identical <style> blocks across reruns),
    so there's no need to guard against calling it more than once."""
    st.markdown("""
        <style>
        /* ---- layout: capped width so lines of text/equations don't
           stretch uncomfortably wide on a large monitor with
           layout="wide". padding-top must stay generous (this matches
           Streamlit's own default): Streamlit's header toolbar is
           position:fixed, sitting on top of the page content rather than
           pushing it down, so anything shorter than its height gets
           visually overlapped/clipped by it -- a smaller value here
           previously clipped the page title. */
        .block-container {
            padding-top: 6rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        h1, h2, h3 { font-weight: 650; letter-spacing: -0.01em; }

        /* ---- hero header (see ui/theme.py's render_hero()) */
        .mrs-hero {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            margin-bottom: 0.1rem;
            padding-top: 0.2rem;
        }
        .mrs-hero-title {
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: -0.015em;
            /* Explicit, generous line-height: the emoji in the title
               renders taller than this font's normal alphabetic line box,
               and Streamlit's inherited line-height (tuned for plain text)
               isn't tall enough for it -- without this, the top of the
               emoji gets clipped by the surrounding block's line box. */
            line-height: 1.35;
        }
        /* opacity 0.67, not the rounder-looking 0.65 -- 0.65 measures at
           4.26:1 against a white background, just under WCAG AA's 4.5:1
           minimum for normal-size text; 0.67 clears it (~4.5:1). */
        .mrs-hero-sub { color: rgba(49, 51, 63, 0.67); font-size: 0.98rem; margin-bottom: 1.2rem; }

        /* ---- cards: expanders and bordered containers get a consistent,
           slightly-rounded, subtly-shadowed treatment instead of
           Streamlit's bare-rule default, so grouped content (an
           expander's contents, a bordered st.container) reads as one
           distinct block rather than just more text on the page. Content
           gets explicit breathing room from the border itself -- text
           sitting flush against a border reads as cramped/unfinished
           regardless of how the border itself looks. */
        [data-testid="stExpander"] {
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 10px;
            overflow: hidden;
        }
        [data-testid="stExpander"] summary {
            font-weight: 600;
            padding: 0.7rem 1rem;
        }
        [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            padding: 0.4rem 1rem 1rem 1rem;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 10px !important;
            padding: 0.3rem 0.2rem;
        }

        /* ---- buttons: rounded corners, a small lift on hover -- the
           kind of micro-interaction a hand-built page usually skips,
           which is exactly what reads as "unpolished". Deliberately NOT
           forcing font-weight here: some buttons (e.g. the history row's
           "reload"/"delete" icon buttons) are a single Unicode symbol
           character (\u21ba, \u2715) rather than a word, and forcing a
           non-default weight on those caused them to render blank --
           the base font's bold cut has no glyph for that exact
           codepoint, and the browser didn't fall back to a font that
           does once a specific weight was demanded. */
        .stButton > button, .stDownloadButton > button {
            border-radius: 8px;
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.10);
        }
        .stButton > button[data-testid^="stBaseButton-primary"] {
            box-shadow: 0 1px 4px rgba(46, 94, 170, 0.35);
        }

        /* ---- status badges: small pill labels used for the confidence
           banner and anywhere else a pass/warn/fail state is shown
           inline rather than as a full-width st.success/warning block. */
        .mrs-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.22rem 0.7rem;
            border-radius: 999px;
            font-size: 0.83rem;
            font-weight: 600;
            white-space: nowrap;
        }
        /* chart descriptions for screen readers: invisible, but read out (the usual "visually hidden" pattern) */
        .mrs-sr-only {
            position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0; overflow: hidden;
            clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
        }
        .mrs-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            align-items: center;
            margin: 0.3rem 0 0.6rem 0;
        }

        /* ---- sidebar: a hairline separator from the main content, and
           slightly denser section headers so a long sidebar (mode list +
           connection status + history) scans faster. The mode radio list
           in particular is long (a dozen-plus options) and Streamlit's
           default radio spacing packs them close enough that adjacent
           options are easy to misclick -- extra vertical gap between
           options fixes that without needing to touch the radio markup. */
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(49, 51, 63, 0.08);
        }
        [data-testid="stSidebar"] h2 {
            font-size: 1.02rem;
            margin-top: 0.4rem;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            padding: 0.22rem 0;
        }
        [data-testid="stSidebar"] hr {
            margin: 0.9rem 0;
        }

        /* ---- st.status: align its internal spacing with the card style
           above, so the pipeline-progress box (see ui/word_problem.py)
           reads as part of the same design language, not a leftover
           default widget. */
        [data-testid="stStatusWidget"] {
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str) -> None:
    """The page header: a title/subtitle pair styled via .mrs-hero rather
    than a bare st.title()/st.caption(), so it reads as a designed header
    rather than the top of a plain document."""
    st.markdown(f"""
        <div class="mrs-hero"><span class="mrs-hero-title">{title}</span></div>
        <div class="mrs-hero-sub">{subtitle}</div>
    """, unsafe_allow_html=True)


def _with_symbol(text: str, kind: str) -> str:
    """Prefix a status symbol unless the text already starts with a symbol or emoji, so pass / warn / fail never depend on colour alone."""
    symbol = _SYMBOL.get(kind)
    first = text.lstrip()[:1]
    if symbol is None or not first or (ord(first) > 127 and not first.isalpha()):
        return text
    return f"{symbol} {text}"


def badge(text: str, kind: str = "neutral") -> str:
    """Returns the HTML for one .mrs-badge pill. Callers batch several
    into one st.markdown(..., unsafe_allow_html=True) call (see
    badge_row()) rather than one st.markdown() per badge, since each
    st.markdown() call is its own layout element with its own margin --
    several separate calls would stack vertically instead of forming a row.
    `kind` is one of "pass", "warn", "fail", "neutral"."""
    bg, fg = _BADGE_COLORS.get(kind, _BADGE_COLORS["neutral"])
    return f'<span class="mrs-badge mrs-badge-{kind}" style="background:{bg};color:{fg};">{_with_symbol(text, kind)}</span>'


def backend_chip(backend: str) -> None:
    """A small chip saying where a result was computed: the GPU (highlighted) or the CPU. Results record this themselves
    (modules/gpu.py), so it is what actually ran, not what was requested: a GPU job that fell back to the CPU shows CPU."""
    on_gpu = backend.startswith("GPU")
    badge_row([(f"⚡ {backend}" if on_gpu else f"🖥️ {backend}", "pass" if on_gpu else "neutral")])


def badge_row(badges: list[tuple[str, str]]) -> None:
    """Renders several badge(text, kind) pairs as one horizontal row."""
    html = '<div class="mrs-badge-row">' + "".join(badge(t, k) for t, k in badges) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def dark_mode_css() -> str:
    """The dark-mode override string -- unchanged from its previous home
    inline in ui/sidebar.py, just relocated so all of the app's CSS lives
    in one module. See the call site in ui/sidebar.py for why this targets
    data-testid attributes rather than CSS custom properties."""
    return """
        <style>
        [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
            background-color: #0E1117;
            color: #E8E8E8;
        }
        [data-testid="stSidebar"] {
            background-color: #1C1F26;
            color: #E8E8E8;
        }
        [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
        [data-testid="stCaptionContainer"], label, .stMarkdown, h1, h2, h3, h4, h5, h6 {
            color: #E8E8E8 !important;
        }
        [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
        [data-testid="stTextArea"] textarea, [data-baseweb="select"] {
            background-color: #262B36 !important;
            color: #E8E8E8 !important;
        }
        /* Placeholder text (e.g. the sidebar's "Jump to..." search box)
           defaults to a low-opacity DARK gray tuned for a light
           background -- against this dark input background that's nearly
           invisible. Both the explicit color and opacity:1 are needed:
           some browsers additionally dim ::placeholder by default
           regardless of the color's own alpha. */
        [data-testid="stTextInput"] input::placeholder,
        [data-testid="stTextArea"] textarea::placeholder,
        [data-testid="stNumberInput"] input::placeholder {
            color: rgba(232, 232, 232, 0.6) !important;
            opacity: 1 !important;
        }
        /* The sidebar's right-edge separator was still the LIGHT-mode
           rgba(49,51,63,...) -- a dark gray at low opacity, which against
           the dark sidebar background is essentially the same color as
           the background itself (measured contrast ~1.02:1, i.e.
           imperceptible). This is the dark-mode equivalent at a similar
           "hairline" subtlety to the light-mode original, not a stark
           divider. */
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(232, 232, 232, 0.18) !important;
        }
        /* Secondary buttons (anything not the primary "Solve" button)
           otherwise keep rendering with the LIGHT theme's button chrome
           (a near-white fill), since Streamlit bakes button colors from
           the fixed startup theme rather than exposing them as CSS
           variables this override could redefine in one place -- left
           alone, every secondary button would show up as a bright white
           pill floating on the dark page.

           The actual attribute Streamlit puts on a button is
           data-testid="stBaseButton-<kind>" (verified against the
           installed package's own frontend bundle), NOT a `kind="..."`
           HTML attribute -- an earlier version of this rule used
           `button[kind="secondary"]`, which matched nothing at all, so
           this override silently never applied: the button kept its
           light-theme background while a separate, broader rule
           elsewhere in this file still recolored its text/icon light,
           producing an invisible light-on-light icon (this is what the
           history row's reload/delete buttons looked like: blank pills).
           :not(...primary) covers "secondary" and any other non-primary
           kind Streamlit might use (e.g. a form-submit variant) without
           needing to enumerate each one by name. */
        [data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-primary"]) {
            background-color: #262B36 !important;
            color: #E8E8E8 !important;
            border-color: rgba(232, 232, 232, 0.25) !important;
        }
        /* SURFACES THAT STAY LIGHT IN DARK MODE. Streamlit paints a few things from the light theme it started with (the page background and most text are
           re-coloured above, these are not), and the broad "text is light grey" rule above then lands on them: pale text on a pale surface (measured
           contrast 1.1-1.2 : 1, i.e. unreadable). Each is given a dark text colour instead of being repainted, so they keep their look. Every rule here is
           limited to the state in which the surface really is light: a popover's trigger turns dark-tinted on hover, focus and while open, and dark text on
           that would be the same mistake the other way round. */

        /* a popover's trigger ("Explain step 3", "Copy a..."): near-white at rest, dark-tinted when hovered, focused (it keeps focus after its popover closes)
           or open -- there the light text above is correct and is left alone */
        [data-testid="stPopoverButton"]:not(:hover):not(:focus):not(:focus-visible):not(:active):not([aria-expanded="true"]),
        [data-testid="stPopoverButton"]:not(:hover):not(:focus):not(:focus-visible):not(:active):not([aria-expanded="true"]) p,
        [data-testid="stPopoverButton"]:not(:hover):not(:focus):not(:focus-visible):not(:active):not([aria-expanded="true"]) span {
            color: #2B3040 !important;
        }
        [data-testid="stPopoverButton"]:not(:hover):not(:focus):not(:focus-visible):not(:active):not([aria-expanded="true"]) svg {
            fill: #2B3040 !important;
            color: #2B3040 !important;
        }
        /* the header of an OPEN expander (and of st.status, which is one): light fill when open and at rest, dark-tinted on hover and keyboard focus (there the
           light text above is right) */
        [data-testid="stExpander"] details[open] > summary:not(:hover):not(:focus-visible),
        [data-testid="stExpander"] details[open] > summary:not(:hover):not(:focus-visible) p,
        [data-testid="stExpander"] details[open] > summary:not(:hover):not(:focus-visible) span,
        [data-testid="stExpander"] details[open] > summary:not(:hover):not(:focus-visible) svg {
            color: #2B3040 !important;
            fill: #2B3040 !important;
        }
        /* toasts */
        [data-testid="stToast"], [data-testid="stToast"] p, [data-testid="stToast"] span, [data-testid="stToast"] div,
        [data-testid="stToast"] [data-testid="stMarkdownContainer"] p, [data-testid="stToast"] button svg {
            color: #2B3040 !important;
            fill: #2B3040 !important;
        }
        /* the panel a popover opens: text, captions, labels and headings in it */
        [data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"],
        [data-testid="stPopoverBody"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stPopoverBody"] [data-testid="stCaptionContainer"],
        [data-testid="stPopoverBody"] label, [data-testid="stPopoverBody"] p,
        [data-testid="stPopoverBody"] h1, [data-testid="stPopoverBody"] h2, [data-testid="stPopoverBody"] h3, [data-testid="stPopoverBody"] h4 {
            color: #2B3040 !important;
        }
        [data-testid="stPopoverBody"] a { color: #0B57D0 !important; }
        /* the open list of a selectbox / multiselect (they render in a portal, outside the containers styled above) */
        [data-testid$="Dropdown"], [data-testid$="Dropdown"] [role="option"], [data-testid$="Dropdown"] p, [data-testid$="Dropdown"] span {
            color: #2B3040 !important;
        }
        /* code blocks: the syntax colours are tuned for a white page and several (greens, oranges, light blues) fall to 2.6-3.7 : 1 on it; darken each token
           by the same factor so the colours keep their meaning and clear 4.5 : 1 */
        [data-testid="stCode"] pre code span, [data-testid="stCode"] code span { filter: brightness(0.62); }
        /* segmented controls / button groups: an unselected option is a white pill (selected and hovered ones are tinted, where the light text is right) */
        [data-testid="stButtonGroup"] button[aria-checked="false"]:not(:hover):not(:focus-visible) p,
        [data-testid="stButtonGroup"] button[aria-checked="false"]:not(:hover):not(:focus-visible) span {
            color: #2B3040 !important;
        }
        /* a date field's "/" separators */
        [data-testid="stDateInputField"] span { color: #4B5160 !important; }
        /* st.json: its colour scheme is tuned for white too (blue 3.7 : 1, violet 4.4 : 1); darken the text-bearing spans and the leaf elements (a span nested in a span darkens a little more, which only helps) */
        [data-testid="stJson"] span, [data-testid="stJson"] :not(:has(*)) { filter: brightness(0.72); }
        /* the file uploader's "limit 200MB per file" hint */
        [data-testid="stFileUploaderDropzoneInstructions"] span, [data-testid="stFileUploaderDropzoneInstructions"] small { color: #4B5160 !important; }
        /* an icon in a (dark) button takes the button's light text colour -- the sidebar's collapse arrow was dark grey on dark grey */
        [data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-primary"]) [data-testid="stIconMaterial"] { color: #E8E8E8 !important; }
        /* the assistant's avatar icon is white on an orange fill */
        [data-testid^="stChatMessageAvatar"] [data-testid="stIconMaterial"] { color: #1D2230 !important; }
        /* links in text: the light theme's dark blue is 2.5 : 1 on the dark page */
        [data-testid="stMarkdownContainer"] a, [data-testid="stCaptionContainer"] a { color: #8AB4F8 !important; }
        /* A button's hover-help tooltip (st.button(..., help="...")) also
           renders in a portal with its own light-theme background and a
           low-opacity gray hint text -- both unreadable against a dark
           page without an explicit override. */
        [data-testid="stTooltipContent"] {
            background-color: #2F3542 !important;
            color: #F2F2F2 !important;
            border: 1px solid rgba(232, 232, 232, 0.25) !important;
        }
        /* the hint text itself can sit in a child element with its own
           (dimmer) color/opacity, so color the descendants too */
        [data-testid="stTooltipContent"] * {
            color: #F2F2F2 !important;
            opacity: 1 !important;
        }
        /* The open dropdown list for a selectbox/multiselect renders in a
           portal (appended near the end of <body>, outside the sidebar/
           main containers above), so it needs its own rule -- it is NOT
           a descendant of anything already covered here. Same reasoning
           for a multiselect's selected-item "tag" chips. */
        [data-baseweb="popover"] [data-baseweb="menu"] {
            background-color: #262B36 !important;
        }
        [data-baseweb="popover"] [data-baseweb="menu"] li {
            color: #E8E8E8 !important;
        }
        [data-baseweb="tag"] {
            background-color: #33394a !important;
            color: #E8E8E8 !important;
        }
        [data-baseweb="tag"] span, [data-baseweb="tag"] svg {
            color: #E8E8E8 !important;
            fill: #E8E8E8 !important;
        }
        .mrs-hero-sub { color: rgba(232, 232, 232, 0.65) !important; }
        [data-testid="stExpander"] { border-color: rgba(232, 232, 232, 0.15) !important; }
        </style>
    """
