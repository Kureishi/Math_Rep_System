"""The 'What next?' bar on the real results page: it appears, navigates with the inputs filled in, and acts in place."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from modules import history, next_steps as ns, practice
from modules.equation_engine import build_model
from modules.solve_compare import deterministic_report
from modules.solver import compute_steps
from tests.conftest import KINEMATICS_TWO_TARGET_JSON
from tests.test_ui_smoke import _app

ROOT = str(Path(__file__).resolve().parent.parent)
TEXT = "A car accelerates from 8 m/s to 20 m/s in 6 seconds. Find its acceleration and distance."


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(practice, "DB_PATH", tmp_path / "practice.db")
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")


def solved_app(passed=True):
    at = _app()
    model = build_model(json.loads(KINEMATICS_TWO_TARGET_JSON))
    report = deterministic_report(model)
    if not passed:
        report.add("A forced failure", False, "for the test")
    for key, value in dict(model=model, report=report, steps=compute_steps(model), scenarios=[], pdf_bytes=None, plot_snapshots={}, problem_text=TEXT).items():
        at.session_state[key] = value                                          # (the test harness's session state has no update())
    at.run(timeout=180)
    assert not at.exception
    return at


def ns_keys(at):
    return {b.key for b in at.button if b.key and b.key.startswith("ns_solve_")}


def test_a_solved_and_verified_problem_shows_the_bar_with_practice_and_recall():
    at = solved_app()
    assert ns_keys(at) == {"ns_solve_pset", "ns_solve_add_practice", "ns_solve_recall"}
    assert any("What next?" in m.value for m in at.markdown)


def test_a_problem_that_failed_verification_is_not_offered_as_practice_material():
    assert ns_keys(solved_app(passed=False)) == {"ns_solve_recall", "ns_solve_compare"}


def test_the_practice_set_button_opens_problem_sets_on_the_open_problem():
    at = solved_app()
    next(b for b in at.button if b.key == "ns_solve_pset").click()
    at.run(timeout=180)
    assert not at.exception
    assert at.session_state["app_mode"] == ns.mode_for(ns.PAGE_PROBLEM_SETS)
    assert at.radio(key="pset_source").value == "The problem that is open"
    assert any("The problem to vary" in m.value for m in at.markdown)


def test_the_recall_button_opens_recall_with_the_problem_text():
    at = solved_app()
    next(b for b in at.button if b.key == "ns_solve_recall").click()
    at.run(timeout=180)
    assert not at.exception and at.session_state["app_mode"] == ns.mode_for(ns.PAGE_RECALL)
    assert at.text_area(key="recall_query").value == TEXT


def test_adding_to_practice_acts_in_place_once_and_says_so_the_second_time():
    at = solved_app()
    next(b for b in at.button if b.key == "ns_solve_add_practice").click()
    at.run(timeout=180)
    assert not at.exception and any("Added." in s.value for s in at.success)
    (item,) = practice.list_items()
    assert item.text == TEXT and item.concepts == ["kinematics"] and item.origin == "open"
    next(b for b in at.button if b.key == "ns_solve_add_practice").click()
    at.run(timeout=180)
    assert any("already in your daily practice" in s.value for s in at.success) and len(practice.list_items()) == 1


def test_the_added_problem_is_then_in_todays_practice_queue():
    at = solved_app()
    next(b for b in at.button if b.key == "ns_solve_add_practice").click()
    at.run(timeout=180)
    assert [e.item.text for e in practice.queue()] == [TEXT]


# ------------------------------------------------------------------------------------------ robustness

def test_suggestions_that_cannot_be_built_never_hide_the_results(monkeypatch):
    monkeypatch.setattr(ns, "for_solution", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    at = solved_app()
    assert not at.exception and ns_keys(at) == set()


def test_an_add_to_practice_that_fails_explains_why_instead_of_crashing():
    script = f'''
import sys; sys.path.insert(0, {ROOT!r})
from modules import next_steps as ns
from ui.next_steps import render_next_steps
render_next_steps(ns.for_problem("junk", {{"nonsense": True}}, "junk"), "t")
'''
    at = AppTest.from_string(script)
    at.run()
    next(b for b in at.button if b.key == "ns_t_add_practice").click()
    at.run()
    assert not at.exception and any("cannot be added to practice" in w.value and "nothing to solve for" in w.value for w in at.warning)


def test_an_empty_list_of_steps_draws_nothing():
    at = AppTest.from_string(f'''
import sys; sys.path.insert(0, {ROOT!r})
from ui.next_steps import render_next_steps
render_next_steps([], "none")
''')
    at.run()
    assert not at.exception and not at.button and not at.markdown


# ------------------------------------------------------------------------------------------ a curve fit hands over to the posterior sampler

def fitted_app(family):
    at = _app()
    at.run(timeout=120)
    at.session_state["app_mode"] = "📈 Curve fitting"
    at.run(timeout=120)
    data = "x,y\n" + "\n".join(f"{x},{2.0 * 1.3 ** x + (0.05 if i % 2 else -0.05):.4f}" for i, x in enumerate(range(0, 10)))
    at.text_area(key="fit_csv_paste").set_value(data)
    at.run(timeout=120)
    at.selectbox(key="fit_family").set_value(family)
    at.run(timeout=120)
    next(b for b in at.button if "fit" in b.label.lower()).click()
    at.run(timeout=180)
    assert not at.exception
    return at


def test_an_exponential_fit_offers_to_sample_its_posterior_with_the_model_and_data_filled_in():
    at = fitted_app("exponential")
    assert any(b.key == "ns_fit_posterior" for b in at.button)
    next(b for b in at.button if b.key == "ns_fit_posterior").click()
    at.run(timeout=180)
    assert not at.exception and at.session_state["app_mode"] == ns.mode_for(ns.PAGE_POSTERIOR)
    assert at.text_input(key="post_expr").value == "a*exp(b*x)" and at.text_input(key="post_names").value == "a, b"
    assert at.text_area(key="post_paste").value.startswith("x,y\n0,")
    guess = [float(v) for v in at.text_input(key="post_guess").value.split(",")]
    assert guess[0] == pytest.approx(2.0, rel=0.15) and guess[1] == pytest.approx(0.262, rel=0.15)          # ln 1.3: the fit's own values start the sampler


def test_a_polynomial_fit_hands_over_its_coefficients_as_the_model():
    at = fitted_app("polynomial")
    assert any(b.key == "ns_fit_posterior" for b in at.button)


def test_what_next_sits_between_the_step_by_step_solution_and_the_follow_up_box():
    """Layout: the worked solution first, then "What next?", then "Ask a follow-up question"."""
    from tests.report_helpers import kinematics_ctx
    ctx = kinematics_ctx()
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=200)
    at.run()
    at.session_state["model"], at.session_state["report"] = ctx.model, ctx.report
    at.session_state["steps"], at.session_state["problem_text"] = ctx.steps_by_target, ctx.problem_text
    at.run()
    assert not at.exception

    def walk(node):
        """Every text-bearing element in the order it appears on the page."""
        for child in getattr(node, "children", {}).values():
            if getattr(child, "type", "") in ("markdown", "subheader", "header", "caption"):
                yield child.value
            yield from walk(child)
    text = list(walk(at.main))
    where = lambda needle: next(i for i, t in enumerate(text) if needle in t)          # noqa: E731
    assert where("Step-by-step") < where("What next?") < where("follow-up")
    # and it is no longer up near the top, between the confidence banner and the tabs
    assert where("What next?") > where("Domain:")
