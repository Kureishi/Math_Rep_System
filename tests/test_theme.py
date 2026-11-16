

def test_dark_mode_gives_popover_buttons_dark_text_but_leaves_hover_alone():
    """The popover triggers ("Explain step 3", "Copy ...") keep a near-white fill in dark mode, so their label must not be the page's light-grey text."""
    from ui.theme import dark_mode_css
    css = dark_mode_css()
    assert '[data-testid="stPopoverButton"]:not(:hover)' in css
    rule = css.split('[data-testid="stPopoverButton"]:not(:hover),')[1].split("}")[0]
    assert "color: #2B3040 !important" in rule
    assert 'stPopoverButton"]:hover' not in css            # the hover appearance is untouched
