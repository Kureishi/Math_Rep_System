"""
Command-palette-style fuzzy search over the app's navigable targets --
see app.py's sidebar for where this gets wired to an actual search box
plus a Ctrl/Cmd+K keyboard shortcut that focuses it.

Scoped to MODE-level jumps (every item in the sidebar radio), not
sub-tabs within a mode: Streamlit's st.tabs() has no programmatic
"jump to tab index" API -- each tab's content is shown/hidden entirely
client-side, with no server-side concept of "which tab is active" to
set from Python. That's a real capability limit, not an oversight, and
it's why this is a fuzzy-searchable jump list rather than a floating
overlay that can land you mid-way into, say, the PDE solver's Robin-BC
sub-tab specifically -- Streamlit genuinely cannot do that without a
custom-built JS component, which is a much heavier undertaking than a
search box justifies here. Landing on the right MODE and letting the
person pick the sub-tab themselves is still a real improvement over
scanning the mode list by eye, especially once a query's keywords
(below) cover terms that don't appear in the sidebar label at all --
"heat equation" finding the PDE solver, e.g.
"""
import difflib
from dataclasses import dataclass


@dataclass
class PaletteEntry:
    mode: str                    # exact string used as app_mode's value
    keywords: tuple[str, ...] = ()


# One entry per sidebar mode. `keywords` are extra searchable terms that
# don't appear in the mode's own label -- e.g. "heat"/"robin" for the
# PDE solver, so a person thinking in domain terms (not this app's menu
# labels) still finds the right mode.
_ENTRIES = [
    PaletteEntry("📝 Word problem solver", ("solve", "kinematics", "physics", "word problem", "extract")),
    PaletteEntry("📈 Curve fitting", ("regression", "fit", "statistics", "bayesian", "confidence interval")),
    PaletteEntry("🔁 Check equivalence", ("proof", "identity", "same", "equal")),
    PaletteEntry("📚 Batch solver", ("multiple", "worksheet", "many problems")),
    PaletteEntry("🔗 Problem chains", ("chain", "sequence", "multi-step", "pipeline")),
    PaletteEntry("🔬 Extraction diff", ("wording", "compare", "rephrase")),
    PaletteEntry("📐 Dimensional analysis", ("units", "dimensions", "si", "imperial")),
    PaletteEntry("🔄 Transforms & series", ("laplace", "fourier", "taylor", "asymptotic", "expansion")),
    PaletteEntry("🌡️ PDE solver", ("heat", "wave", "laplace equation", "robin", "neumann", "dirichlet",
                                    "partial differential")),
    PaletteEntry("🧮 Tensor calculus", ("metric", "christoffel", "ricci", "curvature", "riemann", "geodesic")),
    PaletteEntry("📔 Research journal", ("history", "concepts", "journal", "past problems")),
    PaletteEntry("🚀 Quick start", ("examples", "tutorial", "gallery", "getting started")),
    PaletteEntry("📐 Geometry", ("triangle", "circle", "angle", "law of sines", "law of cosines",
                                  "schematic", "sss", "sas", "asa")),
]


# The single source of truth for the sidebar's mode list, in display order.
# app.py's mode radio is built from this (not from its own copy of the
# labels), and ui/__init__.py's dispatch table is checked against it in
# tests/test_app_modes.py -- so adding a mode means adding one
# PaletteEntry here plus one dispatch entry, and forgetting either fails
# a test instead of silently shipping a mode the palette can't find (or a
# palette entry that jumps to a mode with no page behind it).
MODE_LABELS: list[str] = [e.mode for e in _ENTRIES]


def search(query: str, limit: int = 5) -> list[str]:
    """Fuzzy-matches `query` against each mode's label + keywords,
    ranked best-first, returning just the mode strings (ready to drop
    straight into app_mode). An empty query returns the first `limit`
    entries as-listed, so opening the palette before typing anything
    still shows something rather than a blank list."""
    if not query.strip():
        return [e.mode for e in _ENTRIES[:limit]]

    q = query.strip().lower()
    scored = []
    for entry in _ENTRIES:
        haystack = [entry.mode.lower()] + [k.lower() for k in entry.keywords]
        best_score = 0.0
        for h in haystack:
            if q in h:
                # substring match: prefer a query that covers more of the
                # matched term (typing "heat" should rank above a query
                # that only weakly substring-matches a much longer keyword)
                score = 0.9 + 0.1 * (len(q) / max(len(h), 1))
            else:
                score = difflib.SequenceMatcher(None, q, h).ratio()
            best_score = max(best_score, score)
        scored.append((best_score, entry.mode))

    scored.sort(key=lambda pair: -pair[0])
    # 0.5 was calibrated empirically: a genuine keyword/label match
    # (even a typo'd one, e.g. "tensr calclus") consistently scores
    # 0.85+ here, while a truly unrelated query's best-scoring entry
    # tops out in the 0.3-0.45 range purely from incidental character
    # overlap (difflib's ratio() is lenient on short strings) -- a
    # lower threshold let genuinely irrelevant "matches" through for a
    # query like "xyz123nonsense", which is worse for a navigation
    # tool than just admitting nothing matched.
    return [mode for score, mode in scored[:limit] if score > 0.5]
