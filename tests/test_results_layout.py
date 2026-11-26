"""The results page layout: the main solution in its own card, the extra tools grouped as tabs under one header, and the optional right-hand Contents rail (ui/layout.py)."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from modules import history, practice
from modules.equation_engine import build_model
from modules.solver import compute_steps
from modules.solve_compare import deterministic_report
from tests.conftest import KINEMATICS_TWO_TARGET_JSON
from ui import layout

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TEXT = "A car accelerates from 8 m/s to 20 m/s in 6 seconds. Find its acceleration and distance."


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(practice, "DB_PATH", tmp_path / "practice.db")
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")


def solved_app():
    at = AppTest.from_file(APP, default_timeout=180)
    model = build_model(json.loads(KINEMATICS_TWO_TARGET_JSON))
    for key, value in dict(model=model, report=deterministic_report(model), steps=compute_steps(model), scenarios=[], pdf_bytes=None,
                           plot_snapshots={}, problem_text=TEXT).items():
        at.session_state[key] = value
    at.run(timeout=180)
    assert not at.exception
    return at


def _text(at):
    return "\n".join(m.value for m in at.markdown)


def test_the_extra_tools_are_grouped_under_one_go_further_header_as_tabs():
    at = solved_app()
    text = _text(at)
    assert "#### 🔬 Go further with" in text
    labels = [t.label for t in at.tabs]
    for label in ("🎲 Uncertainty", "🎚️ Sensitivity", "🎯 Goal seek", "🔁 Another way"):
        assert label in labels, labels
    # the five tools are cards inside those tabs now, not a column of collapsed expanders
    assert not [e for e in at.expander if e.label.startswith(("🎲 Uncertainty propagation for", "📐 Analytic", "📏 Guaranteed", "🎯 Goal seek", "🎚️ Sensitivity"))]
    assert "**🎯 Goal seek: find the input for a target" in text and "**🎚️ Sensitivity analysis for" in text


def test_the_contents_rail_lists_the_sections_this_page_produced():
    at = solved_app()
    assert at.toggle(key=layout.SHOW_RAIL_KEY).value is True
    text = _text(at)
    assert "**Contents**" in text
    for key, label in ((("summary"), "Summary and proof"), ("equations", "Derived equations"), ("variables", "Variables"), ("solution", "Step-by-step solution"),
                       ("next", "What next?"), ("followup", "Follow-up questions")):
        assert f"[{label}](#sec-{key})" in text, label
        assert f'id="sec-{key}"' in text                                     # every link has somewhere to land
    assert any(f"(#sec-tools-" in m.value for m in at.markdown)               # and the "Go further" group of each target


def test_every_rail_link_has_an_anchor_and_anchors_are_unique():
    at = solved_app()
    links = [part.split("(#sec-")[1].rstrip(")") for m in at.markdown for part in m.value.splitlines() if "(#sec-" in part]
    anchors = [seg.split('"')[0] for m in at.markdown for seg in m.value.split('id="sec-')[1:]]
    assert links and set(links) <= set(anchors) and len(anchors) == len(set(anchors))


def test_turning_the_rail_off_removes_it_and_keeps_the_page():
    at = solved_app()
    at.toggle(key=layout.SHOW_RAIL_KEY).set_value(False).run()
    assert not at.exception
    assert "**Contents**" not in _text(at) and "#### 🔬 Go further with" in _text(at)


def test_the_rail_and_the_spacing_have_css_and_the_rail_hides_in_narrow_windows():
    import streamlit as st
    from ui import theme
    seen: list[str] = []
    real = st.markdown
    st.markdown = lambda body, *a, **k: seen.append(body)                     # type: ignore[assignment]
    try:
        theme.inject_base_styles()
    finally:
        st.markdown = real                                                      # type: ignore[assignment]
    css = seen[0]
    assert "st-key-results_rail" in css and "position: sticky" in css and ".mrs-anchor" in css and "scroll-margin-top" in css
    assert "@media (max-width: 1100px)" in css and "st-key-show_rail" in css
    # sticky must sit on the COLUMN (a rail whose parent is as tall as itself has nowhere to stick) and the column must not stretch
    sticky = css[css.index('[data-testid="stColumn"]:has([class*="st-key-results_rail"]) {'):]
    sticky = sticky[:sticky.index("}")]
    assert "position: sticky" in sticky and "align-self: flex-start" in sticky and "top: 4.5rem" in sticky


def test_the_layout_helpers_work_standalone():
    from streamlit.testing.v1 import AppTest as T

    def page():
        from ui import layout as lo
        lo.begin_sections()
        lo.anchor("one", "First")
        lo.anchor("two", "Second")
        lo.gap(2)
        with lo.panel("A card"):
            import streamlit as st
            st.write("inside")
        lo.render_rail()

    at = T.from_function(page, default_timeout=30)
    at.run()
    assert not at.exception
    text = "\n".join(m.value for m in at.markdown)
    assert '<div id="sec-one"' in text and "[First](#sec-one)" in text and "[Second](#sec-two)" in text and "**A card**" in text and 'height:2rem' in text
