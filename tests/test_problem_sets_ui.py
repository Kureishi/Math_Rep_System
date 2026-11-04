"""One lean end-to-end check of the Problem sets page: make a set, then grade a class from pasted answers."""
from modules import problem_sets as ps
from modules.problem_library import LIBRARY
from tests.test_ui_smoke import _app

MODE = "🧑‍🏫 Problem sets"


def _open():
    at = _app()
    at.run(timeout=120)
    at.sidebar.radio[0].set_value(MODE)
    at.run(timeout=120)
    assert not at.exception
    return at


def test_a_set_is_made_and_then_a_class_is_graded_with_a_diagnosed_mistake():
    at = _open()
    next(b for b in at.button if b.key == "pset_go").click()
    at.run(timeout=180)
    assert not at.exception and any("problems made" in s.value for s in at.success)
    pset = at.session_state["_problem_set"]
    first = pset.instances[0]
    rows = ["student,problem,target,answer",
            f"Ana,1,a,{first.answers['a']:.6g} m/s^2", f"Ana,1,d,{first.answers['d']:.6g}",
            f"Ben,1,a,{-first.answers['a']:.6g}", f"Ben,1,d,{first.answers['d']:.6g}"]
    next(t for t in at.text_area if t.key == "pset_sub_paste").set_value("\n".join(rows))
    at.run(timeout=120)
    assert not at.exception
    report = at.session_state["_problem_set_grades"]
    assert [s.name for s in report.students] == ["Ana", "Ben"]
    ben = next(s for s in report.students if s.name == "Ben")
    assert any(it.verdict.diagnosis == "sign error" for it in ben.items)


def test_choosing_another_library_problem_reloads_the_page_without_error():
    at = _app()
    at.run(timeout=120)
    at.sidebar.radio[0].set_value(MODE)
    at.run(timeout=120)
    next(s for s in at.selectbox if s.key == "pset_library").set_value("Pendulum period")
    at.run(timeout=120)
    assert not at.exception and not [e for e in at.error if "LM Studio" not in e.value]
