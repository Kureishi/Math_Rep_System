"""Light and dark themes are Streamlit's own ([theme.light] / [theme.dark] in .streamlit/config.toml), not a CSS override.

What is checked here is text and arithmetic: the config defines both themes with readable colours, the contrast fixes that were measured on the LIGHT theme are scoped
so they cannot reach the dark one, and the helper script that tells the stylesheet which theme is showing is wired in. What the pages LOOK like is measured in a browser by
scripts/dark_mode_contrast_scan.py (a dev tool: it needs Playwright and a Chromium, which CI does not have).
"""
import re
import tomllib
from pathlib import Path

import pytest

from ui import theme

ROOT = Path(__file__).resolve().parent.parent
CONFIG = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))


def luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda v: v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4          # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(a: str, b: str) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# ------------------------------------------------------------------------------------------ the themes are Streamlit's own

def test_both_themes_are_defined_so_streamlit_offers_light_dark_and_system():
    assert {"light", "dark"} <= CONFIG["theme"].keys()
    for name in ("light", "dark"):
        assert {"primaryColor", "backgroundColor", "secondaryBackgroundColor", "textColor"} <= CONFIG["theme"][name].keys(), name


def test_there_is_no_base_key_in_the_light_or_dark_section():
    """Streamlit 1.62 rejects it there ('is not a valid config option'): base belongs to [theme] only."""
    assert "base" not in CONFIG["theme"]["light"] and "base" not in CONFIG["theme"]["dark"]


@pytest.mark.parametrize("name", ["light", "dark"])
def test_text_is_readable_on_the_page_and_on_the_widget_surface(name):
    t = CONFIG["theme"][name]
    assert contrast(t["textColor"], t["backgroundColor"]) >= 7
    assert contrast(t["textColor"], t["secondaryBackgroundColor"]) >= 7


@pytest.mark.parametrize("name", ["light", "dark"])
def test_a_primary_button_label_is_readable(name):
    """Streamlit draws a primary button's label in white."""
    assert contrast("#FFFFFF", CONFIG["theme"][name]["primaryColor"]) >= 4.5


def test_links_are_readable_on_the_dark_page():
    t = CONFIG["theme"]["dark"]
    assert contrast(t["linkColor"], t["backgroundColor"]) >= 4.5


def test_the_developer_toolbar_is_hidden_but_the_theme_switch_is_kept():
    """'minimal' would hide the whole menu, Settings -> Choose app theme included."""
    assert CONFIG["client"]["toolbarMode"] == "viewer"


def test_the_apps_own_dark_mode_override_is_gone():
    assert not hasattr(theme, "dark_mode_css")


# ------------------------------------------------------------------------------------------ the light-only fixes stay out of the dark theme

def _all_css() -> str:
    import streamlit as st
    seen: list[str] = []
    real = st.markdown
    st.markdown = lambda body, *a, **k: seen.append(body)                 # type: ignore[assignment]
    try:
        theme.inject_base_styles()
    finally:
        st.markdown = real                                                  # type: ignore[assignment]
    return seen[0]


def test_every_light_only_selector_is_scoped_to_not_dark():
    css = _all_css()
    light_block = css[css.index(theme.LIGHT_ONLY_RULES[0][0][0].split(" ")[0]) - 40:]
    for selectors, _ in theme.LIGHT_ONLY_RULES:
        for sel in selectors:
            assert f'{theme._LIGHT_SCOPE} {sel}' in light_block, sel
    # and no unscoped copy of those colours is left in the always-on rules
    always_on = css[:css.index("<style>", 10)] if css.count("<style>") > 1 else css
    for colour in ("#0B5D27", "#A61B1B", "#8B1A1A", "#4B5160", "brightness(0.62)"):
        assert colour not in always_on, colour


def test_the_always_on_rules_use_no_fixed_light_theme_colours():
    css = _all_css()
    always_on = css[:css.index("<style>", 10)]
    assert "rgba(49, 51, 63" not in always_on                            # the light theme's ink as a fixed border / text colour


def test_the_helper_script_marks_the_theme_and_keeps_the_jump_shortcut():
    html = theme.PAGE_HELPER_HTML
    assert "data-mrs-theme" in html and "stApp" in html
    assert "removeEventListener" in html and "heat equation" in html      # the handler is replaced, not stacked
    assert re.search(r"setInterval\(paintTheme", html)
