"""Dark mode: text must stay readable on every surface Streamlit leaves light, in every state that surface can be in.

The stylesheet is checked here as text and by arithmetic. What it DOES to a real page is measured in a browser by scripts/dark_mode_contrast_scan.py (a dev tool: it
needs Playwright and a Chromium, which CI does not have): it reported 25 low-contrast spots (down to 1.1 : 1) before these rules and none after.
"""
import re
from pathlib import Path

import pytest

from ui.theme import dark_mode_css, inject_base_styles

CSS = dark_mode_css()
ROOT = Path(__file__).resolve().parent.parent


def _base_css() -> str:
    """The always-on stylesheet (what inject_base_styles hands to st.markdown), captured."""
    import streamlit as st
    seen: list[str] = []
    real = st.markdown
    st.markdown = lambda body, *a, **k: seen.append(body)                 # type: ignore[assignment]
    try:
        inject_base_styles()
    finally:
        st.markdown = real                                                  # type: ignore[assignment]
    return seen[0]


BASE = _base_css()


def rule_for(selector_fragment: str) -> str:
    """The declaration block of the first rule whose selector list contains `selector_fragment`."""
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", CSS):
        if selector_fragment in m.group(1):
            return m.group(2)
    raise AssertionError(f"no rule for {selector_fragment!r}")


def selectors_with(fragment: str) -> list[str]:
    out = []
    for m in re.finditer(r"([^{}]+)\{([^{}]*)\}", CSS):
        out += [s.strip() for s in m.group(1).split(",") if fragment in s and "#2B3040" in m.group(2)]
    return out


def luminance(hex_or_rgb) -> float:
    r, g, b = (int(hex_or_rgb[i:i + 2], 16) for i in (1, 3, 5)) if isinstance(hex_or_rgb, str) else hex_or_rgb
    f = lambda v: (v / 255) / 12.92 if v / 255 <= 0.03928 else (((v / 255) + 0.055) / 1.055) ** 2.4          # noqa: E731
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(a, b) -> float:
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


# ------------------------------------------------------------------------------------------ state: dark text only where the surface is light

def test_a_popover_trigger_has_dark_text_only_at_rest():
    """It is near-white at rest but dark-tinted when hovered, focused (it keeps focus once its popover closes) or open: dark text there is unreadable."""
    sels = selectors_with("stPopoverButton")
    assert sels
    for sel in sels:
        for state in (":not(:hover)", ":not(:focus)", ":not(:focus-visible)", ":not(:active)", ':not([aria-expanded="true"])'):
            assert state in sel, (state, sel)


def test_an_open_expander_header_has_dark_text_but_not_while_hovered_or_keyboard_focused():
    sels = selectors_with('details[open] > summary')
    assert len(sels) >= 4
    assert all(":not(:hover)" in s and ":not(:focus-visible)" in s for s in sels)


def test_an_unselected_segment_has_dark_text_but_not_while_hovered():
    sels = selectors_with("stButtonGroup")
    assert sels and all('[aria-checked="false"]' in s and ":not(:hover)" in s for s in sels)


@pytest.mark.parametrize("fragment", [
    '[data-testid="stToast"]', '[data-testid="stPopoverBody"]', '[data-testid$="Dropdown"]', '[data-testid="stDateInputField"]',
    '[data-testid="stFileUploaderDropzoneInstructions"]', '[data-testid^="stChatMessageAvatar"]', '[data-testid="stSidebarCollapseButton"]',
])
def test_every_light_surface_has_a_rule(fragment):
    if fragment == '[data-testid="stSidebarCollapseButton"]':        # covered by the generic "icon in a dark button takes the button's text colour" rule
        assert '[data-testid^="stBaseButton-"]:not([data-testid="stBaseButton-primary"]) [data-testid="stIconMaterial"]' in CSS
    else:
        assert fragment in CSS


def test_links_in_captions_get_the_light_link_colour_too():
    assert '[data-testid="stCaptionContainer"] a' in CSS


def test_code_blocks_and_json_are_darkened_leaf_by_leaf_not_wholesale():
    """In the always-on stylesheet, so light mode gets it too (the same colours are 2.6-3.7 : 1 on white in either mode)."""
    assert re.search(r'\[data-testid="stCode"\] pre code span[^{]*\{\s*filter: brightness\(0\.\d+\)', BASE)
    assert re.search(r'\[data-testid="stJson"\] span[^{]*\{\s*filter: brightness\(0\.\d+\)', BASE)
    assert "filter: brightness" not in CSS                      # not duplicated in the dark block


@pytest.mark.parametrize("fragment", [
    '[data-testid="stAlertContentSuccess"] p', '[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) [data-testid="stAlertDynamicIcon"]', '[data-testid="stAlertContentError"]', '[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Up"])',
    '[data-testid="stMetricDelta"]:has([data-testid="stMetricDeltaIcon-Down"])', '[data-testid="stSliderTickBar"] p', '[data-testid="stFileUploaderDropzoneInstructions"] span',
    '[data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"]', '[data-testid^="stChatMessageAvatar"] [data-testid="stIconMaterial"]',
])
def test_light_mode_has_a_rule_for_each_surface_the_scan_found(fragment):
    assert fragment in BASE


def test_the_always_on_rules_never_use_important_so_dark_mode_can_always_override_them():
    start = BASE.index("*/", BASE.index("readable text on the light surfaces")) + 2          # after the comment that introduces the block
    block = re.sub(r"/\*.*?\*/", "", BASE[start:BASE.index("</style>")], flags=re.S)        # declarations only, not the comments
    offenders = [line.strip() for line in block.splitlines() if "!important" in line and "stDateInputField" not in line]
    assert not offenders, offenders                           # the one exception, the date field, is the colour dark mode uses too


# ------------------------------------------------------------------------------------------ arithmetic: the colours chosen clear WCAG AA (4.5 : 1)

@pytest.mark.parametrize("fg, bg, minimum", [
    ("#2B3040", (255, 255, 255), 7), ("#2B3040", (248, 250, 252), 7),               # dark text on the light surfaces (toast, popover, menus, expander header, trigger)
    ("#E8E8E8", (38, 43, 54), 7),                                                 # the same text where the surface goes dark-tinted
    ("#8AB4F8", (14, 17, 23), 7), ("#0B57D0", (255, 255, 255), 4.5),                # links: on the dark page, and inside a light popover
    ("#4B5160", (240, 244, 249), 4.5), ("#1D2230", (255, 164, 33), 7),              # uploader hint, avatar icon
    ("#0B5D27", (219, 239, 233), 6.5), ("#0B5D27", (233, 249, 238), 7), ("#0B5D27", (255, 255, 255), 7),        # light mode: success text and a rising metric delta
    ("#6FD39A", (16, 35, 29), 7),                                                                                # dark mode: the success alert's icon
    ("#A61B1B", (255, 255, 255), 7), ("#8B1A1A", (242, 224, 228), 7),                                         # a falling delta, error text on its pink fill
])
def test_the_chosen_colours_clear_wcag_aa(fg, bg, minimum):
    assert contrast(fg, bg) >= minimum


def test_light_mode_text_colours_are_the_ones_the_stylesheet_declares():
    for hexcolour in ("#0B5D27", "#A61B1B", "#8B1A1A", "#4B5160", "#1D2230"):
        assert hexcolour in BASE


def test_a_code_token_colour_darkened_by_the_filter_clears_aa():
    """Streamlit's lightest-contrast tokens on its near-white code background (rgb 9,171,59 green; 237,111,19 orange; 128,132,149 grey) after brightness(0.62)."""
    k = float(re.search(r'stCode"\] pre code span[^{]*\{\s*filter: brightness\(([\d.]+)\)', BASE).group(1))
    for rgb in ((9, 171, 59), (237, 111, 19), (128, 132, 149), (28, 131, 225), (41, 176, 157)):
        assert contrast(tuple(c * k for c in rgb), (248, 250, 252)) >= 4.5, rgb


# ------------------------------------------------------------------------------------------ the tool that measures it

def test_the_contrast_tooling_is_in_the_repo_and_documented():
    for name in ("dark_mode_gallery.py", "dark_mode_contrast_scan.py"):
        compile((ROOT / "scripts" / name).read_text(encoding="utf-8"), name, "exec")
    assert "dark_mode_contrast_scan" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "dark_mode_contrast_scan" in (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "playwright" in (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
