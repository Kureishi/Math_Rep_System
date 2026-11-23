"""
A single place for the app's visual polish, so "make it look more
professional" is one file to touch rather than CSS strings scattered
across every page. Entry points:

- inject_base_styles(): always-on layout/typography/component polish
  (card-style containers, refined buttons, badge pills, tighter spacing,
  the grouped mode list), written to work on BOTH Streamlit themes, plus
  a block of contrast fixes that is scoped to the light theme only.
- page_helper_iframe(): a 1 px frame whose script (a) focuses the jump
  box on Ctrl/Cmd+K and (b) marks the page with data-mrs-theme="light" or
  "dark" so the light-only block can switch itself off in dark.

LIGHT AND DARK ARE STREAMLIT'S OWN. .streamlit/config.toml defines [theme.light] and
[theme.dark] (supported since Streamlit 1.5x; this project pins 1.62), so Streamlit paints every
surface -- popovers, dialogs, inputs, open expanders -- itself, in both themes and in every state.
The person chooses in the app menu (Settings -> Choose app theme: Light, Dark, or Use system
setting, which is the default). The old "🌙 Dark mode" checkbox, and the ~150 lines of CSS
that recoloured text on the surfaces Streamlit left white, are gone with it: that override
could only ever be a patch over a theme Streamlit had already painted light.
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
        /* The subtitle is the page's own text colour at 74 % opacity rather than a fixed grey, so it follows the theme: about 6:1 on white in
           the light theme and 8:1 on the dark one (a fixed rgba(49,51,63,0.67) was right for light only; 0.65 measured 4.26:1, under WCAG AA's 4.5:1). */
        .mrs-hero-sub { opacity: 0.74; font-size: 0.98rem; margin-bottom: 1.2rem; }

        /* ---- cards: expanders and bordered containers get a consistent,
           slightly-rounded, subtly-shadowed treatment instead of
           Streamlit's bare-rule default, so grouped content (an
           expander's contents, a bordered st.container) reads as one
           distinct block rather than just more text on the page. Content
           gets explicit breathing room from the border itself -- text
           sitting flush against a border reads as cramped/unfinished
           regardless of how the border itself looks. */
        [data-testid="stExpander"] {
            border: 1px solid rgba(128, 128, 128, 0.3);
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
        /* Narrow windows (about 800 px or less): the main area is already tight, so give it back some of its side padding, and let a row of columns wrap
           instead of squeezing four buttons into 80 px each. Only below 900 px, so the normal layout is untouched. The sidebar is excluded. */
        @media (max-width: 900px) {
            [data-testid="stMain"] .block-container, [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; }
            [data-testid="stMain"] [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
            [data-testid="stMain"] [data-testid="stColumn"] { min-width: min(100%, 11rem); }
        }
        /* People who ask their operating system for less motion get none from the app's own CSS either (the hover lift on buttons and cards, the expander arrow);
           the charts' animations have their own "Reduce motion" setting (Advanced settings), which removes the Play button. */
        @media (prefers-reduced-motion: reduce) {
            * { transition: none !important; animation: none !important; scroll-behavior: auto !important; }
        }
        /* A visible keyboard focus ring on every button, including the icon-only tertiary ones (history rows, close buttons), whose default focus
           cue is faint. currentColor is the button's own text colour, so the ring contrasts with whatever surface the button is on, in both themes. */
        [data-testid^="stBaseButton-"]:focus-visible, .stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
            outline: 2px solid currentColor;
            outline-offset: 2px;
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
           slightly denser section headers so a long sidebar scans faster.
           The mode list (ui/mode_nav.py) is a column of collapsible groups of
           buttons: left-aligned like a menu, tighter than a default button,
           with the group headers a little more compact than a content expander. */
        [data-testid="stSidebar"] {
            border-right: 1px solid rgba(128, 128, 128, 0.22);
        }
        [data-testid="stSidebar"] h2 {
            font-size: 1.02rem;
            margin-top: 0.4rem;
        }
        [data-testid="stSidebar"] hr {
            margin: 0.9rem 0;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary { padding: 0.5rem 0.8rem; }
        [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] { padding: 0.15rem 0.4rem 0.45rem 0.4rem; }
        [data-testid="stSidebar"] [class*="st-key-nav_"] button { justify-content: flex-start; text-align: left; min-height: 2rem; padding: 0.2rem 0.6rem; }
        [data-testid="stSidebar"] [class*="st-key-nav_"] button p { text-align: left; }
        /* the star / load / rename / delete buttons of a history entry sit in a tight row */
        [data-testid="stSidebar"] [class*="st-key-pinhist_"] button,
        [data-testid="stSidebar"] [class*="st-key-load_"] button,
        [data-testid="stSidebar"] [class*="st-key-renamehist_"] button,
        [data-testid="stSidebar"] [class*="st-key-delhist_"] button { min-height: 1.8rem; padding: 0.1rem 0.4rem; }

        /* ---- st.status: align its internal spacing with the card style
           above, so the pipeline-progress box (see ui/word_problem.py)
           reads as part of the same design language, not a leftover
           default widget. */
        [data-testid="stStatusWidget"] {
            border-radius: 10px;
        }
        </style>
    """, unsafe_allow_html=True)
    # A SEPARATE call, never concatenated onto the string above: Streamlit dedents a markdown body by its smallest indent, and this block starts at column 0, so joining
    # them left the indented block above at 8 spaces, which Markdown renders as a code block -- the stylesheet showed up as text at the top of the page.
    st.markdown(_light_only_css(), unsafe_allow_html=True)


# Contrast fixes measured in a browser (scripts/dark_mode_contrast_scan.py) on Streamlit's LIGHT theme: each of these fell below WCAG AA's 4.5 : 1. They are colours
# chosen for light surfaces, so they must not reach the dark theme (a darkened code token on a dark background is the same mistake the other way round): every
# selector is scoped to `:root:not([data-mrs-theme="dark"])`, where the attribute is set by the page helper script below. Until that script has run (a fraction of a
# second after load) the page counts as light, which is the safe side for the default theme.
LIGHT_ONLY_RULES: list[tuple[list[str], str]] = [
    # code blocks: Streamlit's syntax colours (greens, oranges, light blues, grey punctuation) are 2.6-3.7 : 1 on the near-white code background;
    # darken each token by the same factor so the colours keep their meaning
    (['[data-testid="stCode"] pre code span', '[data-testid="stCode"] code span'], "filter: brightness(0.62);"),
    # st.json: the solarized scheme is blue 3.7 : 1 and violet 4.4 : 1 on white (the span rule catches a span with a child; the leaf rule the rest)
    (['[data-testid="stJson"] span', '[data-testid="stJson"] :not(:has(*))'], "filter: brightness(0.72);"),
    # secondary text Streamlit draws at 60 % of its ink colour (4.4 : 1 on its light-grey surfaces): the uploader's size hint, a slider's end labels,
    # the sidebar's collapse arrow
    (['[data-testid="stFileUploaderDropzoneInstructions"] span', '[data-testid="stFileUploaderDropzoneInstructions"] small',
      '[data-testid="stSliderTickBar"] p', '[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]'], "color: #4B5160;"),
    (['[data-testid="stDateInputField"] span'], "color: #4B5160 !important;"),
    # the assistant's chat avatar: a white icon on an orange fill is 2 : 1
    (['[data-testid^="stChatMessageAvatar"] [data-testid="stIconMaterial"]'], "color: #1D2230;"),
    # Streamlit's success green is 4.1-4.5 : 1 on its tinted fills; the delta under a metric is green when it rises and red (3.9 : 1) when it falls
    (['[data-testid="stAlertContentSuccess"] p', '[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"])',
      '[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"]) p'], "color: #0B5D27;"),
    (['[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"])', '[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"]) p'],
     "color: #A61B1B;"),
    (['[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) [data-testid="stAlertDynamicIcon"]'], "color: #0B5D27;"),
    # an error alert's red text is 4.2 : 1 on its pink fill (it is the message shown whenever LM Studio is not running)
    (['[data-testid="stAlertContentError"]', '[data-testid="stAlertContentError"] p'], "color: #8B1A1A;"),
]
_LIGHT_SCOPE = ':root:not([data-mrs-theme="dark"])'


def _light_only_css() -> str:
    rules = "\n".join(f"{', '.join(f'{_LIGHT_SCOPE} {sel}' for sel in selectors)} {{ {declarations} }}" for selectors, declarations in LIGHT_ONLY_RULES)
    return f"<style>\n{rules}\n</style>"


# The script in the 1 px frame. It reaches out of its sandboxed iframe into the parent page (the usual, slightly fragile trick: Streamlit has no API for a global keyboard shortcut
# or for reading the theme into CSS). If a future Streamlit changes the DOM it simply stops working: the shortcut does nothing, and the page is treated as light.
#   Ctrl/Cmd+K   focus the "Jump to..." box. The handler is replaced, not stacked, each time the frame reloads.
#   theme        every second, read the app's background colour; dark if its luminance is below one half. Polling (one getComputedStyle call) rather than an observer
#                because the theme can change by the app menu, by the operating system, or by Streamlit re-applying it, and all of them end up as the same background colour.
PAGE_HELPER_HTML = """
    <script>
    (function() {
        const win = window.parent, doc = win.document;
        if (win.__mrsKeyHandler) { doc.removeEventListener('keydown', win.__mrsKeyHandler); }
        win.__mrsKeyHandler = function(e) {
            const isK = e.key === 'k' || e.key === 'K';
            if ((e.metaKey || e.ctrlKey) && isK) {
                e.preventDefault();
                const inputs = doc.querySelectorAll('input[type="text"]');
                for (const el of inputs) {
                    if (el.placeholder && el.placeholder.includes('heat equation')) { el.focus(); break; }
                }
            }
        };
        doc.addEventListener('keydown', win.__mrsKeyHandler);

        function paintTheme() {
            const app = doc.querySelector('[data-testid="stApp"]') || doc.body;
            const m = win.getComputedStyle(app).backgroundColor.match(/[0-9.]+/g);
            if (!m || m.length < 3) { return; }
            const luminance = (0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2]) / 255;
            const theme = luminance < 0.5 ? 'dark' : 'light';
            if (doc.documentElement.getAttribute('data-mrs-theme') !== theme) { doc.documentElement.setAttribute('data-mrs-theme', theme); }
        }
        paintTheme();
        if (win.__mrsThemeTimer) { win.clearInterval(win.__mrsThemeTimer); }
        win.__mrsThemeTimer = win.setInterval(paintTheme, 1000);
    })();
    </script>
"""


def page_helper_iframe() -> None:
    """Draw the 1 px frame that runs PAGE_HELPER_HTML (called once per run, from the sidebar)."""
    st.iframe(src=PAGE_HELPER_HTML, height=1)


# The theme is Streamlit's own, and its frontend reads the person's choice (System / Light / Dark) from the browser's localStorage when the page LOADS: there is no way to switch
# it live from Python, and no storage listener to nudge. So the sidebar's toggle does what Streamlit's own Settings dialog does -- stores the choice under Streamlit's key --
# and reloads the page. The reload starts a new session (whatever was on screen is gone; the History list still has every solved problem), which is why the toggle says so.
# The key's "-v2" is Streamlit's CACHED_THEME_VERSION in 1.62; if a later Streamlit changes it the toggle falls back to a harmless reload (the app menu still works).
THEME_STORAGE_KEY_JS = "'stActiveTheme-' + win.location.pathname + '-v2'"
THEME_TOGGLE_KEY = "theme_dark_toggle"
_THEME_REQUEST_KEY = "_theme_request"


def theme_switch_html(choice: str) -> str:
    """The script (for the 1 px frame) that stores `choice` ('Dark', 'Light' or 'System') as the theme and reloads the page."""
    assert choice in ("Dark", "Light", "System")
    return ("<script>(function(){ try { const win = window.parent; "
            f"win.localStorage.setItem({THEME_STORAGE_KEY_JS}, JSON.stringify('{choice}')); win.location.reload(); }} catch (e) {{}} }})();</script>")


def _request_theme() -> None:
    st.session_state[_THEME_REQUEST_KEY] = "Dark" if st.session_state.get(THEME_TOGGLE_KEY) else "Light"


def render_theme_toggle() -> None:
    """The sidebar's 🌙 Dark mode toggle: on when the page is in the dark theme, and switching it stores the choice and reloads (see above)."""
    from ui.a11y import is_dark
    st.toggle("🌙 Dark mode", value=is_dark(), key=THEME_TOGGLE_KEY, on_change=_request_theme,
              help="Switches the whole app between Streamlit's light and dark themes. The page reloads (a problem on screen is gone; History keeps it). "
                   "To follow your system again, use the menu at the top right: Settings, then Use system setting.")
    choice = st.session_state.pop(_THEME_REQUEST_KEY, None)
    if choice:
        st.iframe(src=theme_switch_html(choice), height=1)


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
