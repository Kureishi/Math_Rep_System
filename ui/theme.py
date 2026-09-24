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
import streamlit as st

# Small, reusable color tokens for the status badges below -- kept in one
# place rather than repeated per call site, and distinct from the
# Streamlit-level theme colors in config.toml.
_BADGE_COLORS = {
    "pass": ("#E6F4EA", "#1E7E34"),
    "warn": ("#FFF4E0", "#A85D00"),
    "fail": ("#FDEAEA", "#C0392B"),
    "neutral": ("#EEF1F6", "#3B4252"),
}


def inject_base_styles() -> None:
    """Always-on CSS: called once per script run from app.py. Idempotent
    and cheap (Streamlit dedupes identical <style> blocks across reruns),
    so there's no need to guard against calling it more than once."""
    st.markdown("""
        <style>
        /* ---- layout: a touch tighter than Streamlit's defaults, and
           capped width so lines of text/equations don't stretch
           uncomfortably wide on a large monitor with layout="wide". */
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }
        h1, h2, h3 { font-weight: 650; letter-spacing: -0.01em; }

        /* ---- hero header (see ui/theme.py's render_hero()) */
        .mrs-hero {
            display: flex;
            align-items: baseline;
            gap: 0.6rem;
            margin-bottom: 0.1rem;
        }
        .mrs-hero-title { font-size: 1.9rem; font-weight: 700; letter-spacing: -0.015em; }
        .mrs-hero-sub { color: rgba(49, 51, 63, 0.65); font-size: 0.98rem; margin-bottom: 1.1rem; }

        /* ---- cards: expanders and bordered containers get a consistent,
           slightly-rounded, subtly-shadowed treatment instead of
           Streamlit's bare-rule default, so grouped content (an
           expander's contents, a bordered st.container) reads as one
           distinct block rather than just more text on the page. */
        [data-testid="stExpander"] {
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 10px;
            overflow: hidden;
        }
        [data-testid="stExpander"] summary {
            font-weight: 600;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 10px !important;
        }

        /* ---- buttons: rounded, slightly bolder label, a small lift on
           hover -- the kind of micro-interaction a hand-built page
           usually skips, which is exactly what reads as "unpolished". */
        .stButton > button, .stDownloadButton > button {
            border-radius: 8px;
            font-weight: 600;
            transition: transform 0.12s ease, box-shadow 0.12s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.10);
        }
        .stButton > button[kind="primary"] {
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
        .mrs-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            align-items: center;
            margin: 0.3rem 0 0.6rem 0;
        }

        /* ---- sidebar: a hairline separator from the main content, and
           slightly denser section headers so a long sidebar (mode list +
           connection status + history) scans faster. */
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(49, 51, 63, 0.08);
        }
        [data-testid="stSidebar"] h2 {
            font-size: 1.02rem;
            margin-top: 0.4rem;
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


def badge(text: str, kind: str = "neutral") -> str:
    """Returns the HTML for one .mrs-badge pill. Callers batch several
    into one st.markdown(..., unsafe_allow_html=True) call (see
    badge_row()) rather than one st.markdown() per badge, since each
    st.markdown() call is its own layout element with its own margin --
    several separate calls would stack vertically instead of forming a row.
    `kind` is one of "pass", "warn", "fail", "neutral"."""
    bg, fg = _BADGE_COLORS.get(kind, _BADGE_COLORS["neutral"])
    return f'<span class="mrs-badge" style="background:{bg};color:{fg};">{text}</span>'


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
        .mrs-hero-sub { color: rgba(232, 232, 232, 0.65) !important; }
        [data-testid="stExpander"] { border-color: rgba(232, 232, 232, 0.15) !important; }
        </style>
    """
