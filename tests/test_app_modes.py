"""
Guards the parity that ui/command_palette.py's and ui/__init__.py's own
docstrings both claim: the sidebar's mode list (modules.command_palette.
MODE_LABELS, which ui/sidebar.py's radio is built from) and the page
dispatch table (ui.PAGES, plus the one mode -- word problem solver -- that
isn't in PAGES because it's the fallthrough default) must cover exactly the
same set of modes. Losing sync either way is a real, silent-failure bug:
a mode in MODE_LABELS but missing from PAGES would render nothing when
selected (app.py falls through to the word-problem page under whatever
mode is actually selected); a stale entry in PAGES with no MODE_LABELS
counterpart is unreachable dead code.
"""
from modules.command_palette import MODE_LABELS
from ui import PAGES, WORD_PROBLEM_MODE


def test_every_mode_label_is_routable():
    """Every sidebar mode is either the word-problem default or has a page
    in PAGES -- nothing silently falls through to the wrong page."""
    unrouted = [m for m in MODE_LABELS if m != WORD_PROBLEM_MODE and m not in PAGES]
    assert not unrouted, f"modes with no page: {unrouted}"


def test_no_stale_dispatch_entries():
    """Every PAGES entry corresponds to a real sidebar mode -- no dead
    entry left behind by a renamed or removed mode."""
    stale = [m for m in PAGES if m not in MODE_LABELS]
    assert not stale, f"PAGES entries with no matching mode: {stale}"


def test_word_problem_mode_is_not_also_in_pages():
    """The default page is reached by falling through PAGES.get() returning
    None, not by an explicit PAGES entry -- both existing would mean two
    different code paths could handle the same mode."""
    assert WORD_PROBLEM_MODE not in PAGES
    assert WORD_PROBLEM_MODE in MODE_LABELS


def test_mode_labels_has_no_duplicates():
    assert len(MODE_LABELS) == len(set(MODE_LABELS))
