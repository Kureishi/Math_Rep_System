"""Tests for modules/command_palette.py -- fuzzy search over sidebar
modes for the command-palette-style quick-jump feature."""
from modules.command_palette import search


def test_keyword_not_in_label_still_matches():
    """"heat" doesn't appear in "PDE solver" at all -- only findable via
    the keywords list, which is the whole point of having one."""
    results = search("heat")
    assert results[0] == "🌡️ PDE solver"


def test_exact_domain_term_matches_tensor_calculus():
    results = search("curvature")
    assert results[0] == "🧮 Tensor calculus"


def test_partial_word_matches():
    results = search("stat")
    assert "📈 Curve fitting" in results


def test_typo_tolerant_match():
    results = search("tensr calclus")
    assert results[0] == "🧮 Tensor calculus"


def test_unrelated_gibberish_returns_no_results():
    """The central quality bar this module has to clear: a nonsense
    query must return NOTHING, not a list of plausible-looking wrong
    guesses -- a navigation tool that confidently suggests the wrong
    place is worse than one that admits it doesn't know."""
    assert search("xyz123nonsense") == []


def test_empty_query_returns_default_list_not_empty():
    results = search("")
    assert len(results) > 0


def test_exact_label_substring_matches_itself():
    results = search("curve fitting")
    assert results[0] == "📈 Curve fitting"


def test_results_respect_limit():
    results = search("", limit=3)
    assert len(results) == 3


def test_all_returned_modes_are_valid_sidebar_values():
    """Every result must be a string usable directly as app_mode's
    value -- if this drifts out of sync with app.py's actual mode list,
    a "jump" would silently select a mode that doesn't exist."""
    from modules.command_palette import _ENTRIES
    valid_modes = {e.mode for e in _ENTRIES}
    for query in ["heat", "curve", "bayes", "", "tensor"]:
        for mode in search(query):
            assert mode in valid_modes
