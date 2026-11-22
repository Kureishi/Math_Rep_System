"""The Practice & mastery page end to end. The date is injected (`_practice_today`), so nothing here reads the real clock."""
from datetime import date

import pytest

from modules import history, practice
from tests.test_ui_smoke import _app

MODE = "🧠 Practice & mastery"
TODAY = date(2026, 3, 2)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(practice, "DB_PATH", tmp_path / "practice.db")
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")


@pytest.fixture
def at():
    app = _app()
    app.session_state["_practice_today"] = TODAY
    app.run(timeout=120)
    app.session_state["app_mode"] = MODE
    app.run(timeout=120)
    assert not app.exception
    return app


def seeded(app):
    next(b for b in app.button if b.key == "prac_seed").click()
    app.run(timeout=120)
    assert not app.exception
    return app


def answer(app, item, nonce=0, correct=True):
    inst, _ = practice.variant(item, TODAY, nonce)
    for target, value in inst.answers.items():
        app.text_input(key=f"prac_ans_{item.id}_{item.state.reps}_{nonce}_{target}").set_value(repr(value if correct else value * 7.3 + 1.1))
    next(b for b in app.button if b.key == f"prac_check_{item.id}_{nonce}").click()
    app.run(timeout=120)
    assert not app.exception


def current(app_):
    return practice.queue(TODAY)[0].item


def test_with_nothing_to_practise_it_offers_the_starter_problems(at):
    assert any("no problems to practise" in i.value for i in at.info)
    seeded(at)
    assert len(practice.list_items()) == 10 and not any(b.key == "prac_seed" for b in at.button)


def test_a_new_problem_is_asked_with_a_fresh_variant_and_inputs_for_every_answer(at):
    seeded(at)
    item = current(at)
    inst, _ = practice.variant(item, TODAY)
    assert any(inst.text in m.value for m in at.markdown) or inst.text in " ".join(m.value for m in at.markdown)
    assert {t.key for t in at.text_input if t.key and t.key.startswith("prac_ans_")} == {f"prac_ans_{item.id}_0_0_{t}" for t in inst.answers}


def test_a_correct_answer_is_marked_scheduled_and_the_next_problem_follows(at):
    seeded(at)
    first = current(at)
    answer(at, first)
    last = at.session_state["_prac_last"]
    assert last["grade"].all_correct and last["state"].reps == 1
    assert any("2 of 2 correct" in s.value for s in at.success) and any("Next review in 1 day" in c.value for c in at.caption)
    assert len(practice.attempts()) == 1
    next(b for b in at.button if b.key == "prac_next").click()
    at.run(timeout=120)
    assert "_prac_last" not in at.session_state and current(at).id != first.id


def test_a_wrong_answer_says_why_and_comes_back_tomorrow(at):
    seeded(at)
    first = current(at)
    inst, _ = practice.variant(first, TODAY)
    answer(at, first, correct=False)
    assert any("0 of 2 correct" in w.value for w in at.warning)
    shown = " ".join(m.value for m in at.markdown)
    assert "the key is" in shown and f"{inst.answers['a']:.4g}"[:3] in shown
    assert practice.get_item(first.id).state.lapses == 1 and practice.get_item(first.id).state.due.isoformat() == "2026-03-03"


def test_an_extra_variant_is_recorded_but_does_not_move_the_schedule(at):
    seeded(at)
    first = current(at)
    answer(at, first)
    before = practice.get_item(first.id).state
    next(b for b in at.button if b.key == "prac_extra").click()
    at.run(timeout=120)
    answer(at, practice.get_item(first.id), nonce=1)
    assert practice.get_item(first.id).state == before and len(practice.attempts()) == 2 and not practice.attempts()[1].scheduled


def test_skipping_hides_a_problem_for_now_without_recording_anything(at):
    seeded(at)
    first = current(at)
    next(b for b in at.button if b.key == f"prac_skip_{first.id}").click()
    at.run(timeout=120)
    assert not at.exception and practice.attempts() == []
    assert not any(t.key and t.key.startswith(f"prac_ans_{first.id}_") for t in at.text_input)


def test_after_the_daily_allowance_it_says_you_are_done(at):
    seeded(at)
    for _ in range(practice.NEW_PER_DAY):
        answer(at, current(at))
        next(b for b in at.button if b.key == "prac_next").click()
        at.run(timeout=120)
    assert any("done for today" in s.value for s in at.success) and practice.queue(TODAY) == []


def test_tomorrows_queue_has_the_reviews_back_in_it(at):
    seeded(at)
    answer(at, current(at))
    at.session_state["_practice_today"] = date(2026, 3, 3)
    at.session_state["_prac_last"] = None                                 # (feedback from today's answer is dismissed)
    at.run(timeout=120)
    assert not at.exception and any(e.kind == "review" for e in practice.queue(date(2026, 3, 3)))


def test_the_mastery_tab_shows_concepts_a_heatmap_and_a_forecast_after_some_practice(at):
    seeded(at)
    answer(at, current(at))
    assert not at.exception
    body = " ".join(m.value for m in at.markdown)
    assert "kinematics" in body and len(at.get("plotly_chart")) >= 2
    described = " ".join(h.proto.body for h in at.get("html"))
    assert "Average score per concept per week" in described and "Reviews coming due" in described


def test_my_problems_lists_items_and_removing_one_stops_it_being_asked(at):
    seeded(at)
    assert any(df.value.shape[0] == 10 for df in at.dataframe)
    victim = practice.list_items()[0]
    at.selectbox(key="prac_remove_pick").set_value(victim)
    at.run(timeout=120)
    next(b for b in at.button if b.key == "prac_remove").click()
    at.run(timeout=120)
    assert not at.exception and len(practice.list_items()) == 9 and victim.id not in {i.id for i in practice.list_items()}
