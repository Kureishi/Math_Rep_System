"""Practice and mastery: SM-2 scheduling, fresh verified variants, the daily queue, and honest mastery. No test reads the clock."""
import copy
from datetime import date, timedelta

import pytest

from modules import practice as pr
from modules import problem_sets as ps
from modules.problem_library import LIBRARY, LIBRARY_CONCEPTS

D0 = date(2026, 3, 2)                      # a Monday
KIN_TEXT, KIN_RAW = LIBRARY["Constant acceleration (kinematics)"]
CIRC_TEXT, CIRC_RAW = LIBRARY["Series circuit"]


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(pr, "DB_PATH", tmp_path / "practice.db")


def add(name="Constant acceleration (kinematics)", today=D0, concepts=None):
    text, raw = LIBRARY[name]
    iid, _ = pr.add_item(name, text, raw, concepts or LIBRARY_CONCEPTS.get(name), "library", today)
    return pr.get_item(iid)


def perfect(instance):
    return {t: repr(v) for t, v in instance.answers.items()}


def wrong(instance):
    return {t: repr(v * 7.3 + 1.1) for t, v in instance.answers.items()}


def attempt(item, day, correct=True, scheduled=True):
    inst, seed = pr.variant(item, day)
    result = pr.grade(inst, perfect(inst) if correct else wrong(inst))
    return pr.record(item.id, seed, result, day, scheduled, ts=f"{day.isoformat()}T10:00:00")


# ------------------------------------------------------------------------------------------ SM-2

def test_successful_reviews_stretch_out_one_day_then_six_then_by_the_ease():
    s, today = pr.ScheduleState(), D0
    seen = []
    for _ in range(5):
        s = pr.sm2(s, 5, today)
        today = s.due
        seen.append(s.interval)
    assert seen[:2] == [1, 6] and seen[2] == round(6 * 2.8) and seen[3] > seen[2] > seen[1]
    assert s.reps == 5 and s.lapses == 0 and s.ease == pytest.approx(2.5 + 5 * 0.1)


def test_the_due_date_is_the_review_date_plus_the_interval():
    s = pr.sm2(pr.ScheduleState(), 4, D0)
    assert s.due == D0 + timedelta(days=1)
    assert pr.sm2(s, 4, s.due).due == s.due + timedelta(days=6)


def test_a_failure_goes_back_to_tomorrow_lowers_the_ease_and_counts_a_lapse():
    grown = pr.ScheduleState(ease=2.5, interval=20, reps=4, lapses=0, due=D0)
    s = pr.sm2(grown, 1, D0)
    assert s.interval == 1 and s.reps == 0 and s.lapses == 1 and s.due == D0 + timedelta(days=1)
    assert s.ease == pytest.approx(2.5 - 0.54)


def test_ease_never_falls_below_the_floor_however_often_it_fails():
    s = pr.ScheduleState()
    for _ in range(12):
        s = pr.sm2(s, 0, D0)
    assert s.ease == pr.MIN_EASE and s.lapses == 12


def test_a_hard_success_lowers_the_ease_and_a_perfect_one_raises_it():
    assert pr.sm2(pr.ScheduleState(), 3, D0).ease == pytest.approx(2.5 - 0.14)
    assert pr.sm2(pr.ScheduleState(), 5, D0).ease == pytest.approx(2.6)
    assert pr.sm2(pr.ScheduleState(), 4, D0).ease == pytest.approx(2.5)


def test_intervals_are_capped_at_a_year():
    huge = pr.ScheduleState(ease=3.0, interval=300, reps=9, lapses=0, due=D0)
    assert pr.sm2(huge, 5, D0).interval == pr.MAX_INTERVAL_DAYS


@pytest.mark.parametrize("q", [-1, 6])
def test_quality_outside_zero_to_five_is_refused(q):
    with pytest.raises(pr.PracticeError, match="between 0 and 5"):
        pr.sm2(pr.ScheduleState(), q, D0)


# ------------------------------------------------------------------------------------------ items

def test_adding_a_problem_twice_does_nothing_and_a_removed_one_comes_back():
    item = add()
    assert item.is_new and item.state.due == D0 and item.concepts == ["kinematics", "uniform acceleration"]
    again, created = pr.add_item("again", KIN_TEXT, KIN_RAW, None, "history", D0)
    assert again == item.id and not created and len(pr.list_items()) == 1
    pr.remove_item(item.id)
    assert pr.list_items() == [] and len(pr.list_items(active_only=False)) == 1
    back, created = pr.add_item("back", KIN_TEXT, KIN_RAW, None, "history", D0)
    assert back == item.id and created and len(pr.list_items()) == 1


def test_the_same_statement_with_a_different_model_is_a_different_problem():
    other = copy.deepcopy(KIN_RAW)
    other["assumptions"] = ["something else"]
    assert pr.item_key(KIN_TEXT, KIN_RAW) != pr.item_key(KIN_TEXT, other) and pr.item_key(KIN_TEXT, KIN_RAW) == pr.item_key("  " + KIN_TEXT + " ", KIN_RAW)


def test_a_problem_without_concepts_gets_its_domain_as_the_concept():
    iid, _ = pr.add_item("untagged", KIN_TEXT, KIN_RAW, None, "history", D0)
    assert pr.get_item(iid).concepts == ["kinematics"]


def test_problems_that_cannot_be_practised_are_refused_with_the_reason():
    ode = copy.deepcopy(KIN_RAW)
    ode["problem_type"] = "ode"
    with pytest.raises(pr.PracticeError, match="ode problem"):
        pr.add_item("ode", KIN_TEXT, ode, None, "history", D0)
    with pytest.raises(pr.PracticeError, match="nothing to solve for"):
        pr.add_item("empty", "text", {"nonsense": True}, None, "history", D0)
    broken = copy.deepcopy(KIN_RAW)
    broken["equations"][0]["expression"] = "Eq(v, (("                       # an equation that cannot be parsed
    with pytest.raises(pr.PracticeError, match=r"does not pass its own checks \(Equation parsing"):
        pr.add_item("broken", KIN_TEXT, broken, None, "history", D0)


def test_the_starter_library_is_added_once():
    assert pr.seed_library(D0) == len(LIBRARY) and pr.seed_library(D0) == 0
    items = pr.list_items()
    assert {i.title for i in items} == set(LIBRARY) and all(i.origin == "library" and len(i.concepts) == 2 for i in items)


# ------------------------------------------------------------------------------------------ the queue

def test_new_items_are_introduced_a_few_at_a_time():
    pr.seed_library(D0)
    entries = pr.queue(D0)
    assert len(entries) == pr.NEW_PER_DAY and all(e.kind == "new" for e in entries)
    assert len(pr.queue(D0, new_limit=5)) == 5


def test_a_new_item_started_today_uses_up_the_daily_allowance():
    pr.seed_library(D0)
    for e in pr.queue(D0)[:2]:
        attempt(e.item, D0)
    todays = pr.queue(D0)
    assert sum(e.kind == "new" for e in todays) == pr.NEW_PER_DAY - 2 and not any(e.kind == "review" for e in todays)


def test_reviews_that_are_due_come_first_most_overdue_first_and_future_ones_wait():
    a, b, c = add("Projectile range"), add("Series circuit"), add("Pendulum period")
    attempt(a, D0)                                         # due D0 + 1
    attempt(b, D0 - timedelta(days=5))                     # due D0 - 4 : the most overdue
    attempt(c, D0 + timedelta(days=1))                     # due D0 + 2 : not yet
    entries = pr.queue(D0 + timedelta(days=1))
    assert [e.item.id for e in entries] == [b.id, a.id] and [e.kind for e in entries] == ["review", "review"]
    assert entries[0].overdue_days == 5 and entries[1].overdue_days == 0


def test_a_failed_item_comes_back_tomorrow_not_today():
    item = add()
    attempt(item, D0, correct=False)
    assert pr.queue(D0) == [] and [e.item.id for e in pr.queue(D0 + timedelta(days=1))] == [item.id]


def test_removed_items_leave_the_queue():
    item = add()
    pr.remove_item(item.id)
    assert pr.queue(D0) == []


# ------------------------------------------------------------------------------------------ fresh variants

def test_the_variant_is_the_same_on_reload_and_different_on_another_day():
    item = add()
    a, sa = pr.variant(item, D0)
    b, sb = pr.variant(item, D0)
    assert a.values == b.values and a.text == b.text and sa == sb
    others = [pr.variant(item, D0 + timedelta(days=d))[0].values for d in range(1, 8)]
    assert any(o != a.values for o in others) and len({tuple(sorted(o.items())) for o in others}) > 3


def test_a_nonce_or_a_new_repetition_gives_a_new_variant():
    item = add()
    base = pr.variant(item, D0)[1]
    assert pr.variant(item, D0, nonce=1)[1] != base
    attempt(item, D0)
    assert pr.variant(pr.get_item(item.id), D0)[1] != base


def test_every_variant_has_a_verified_key_from_the_problem_set_engine():
    item = add("Projectile range")
    for d in range(10):
        inst, _ = pr.variant(item, D0 + timedelta(days=d))
        assert inst.answers["R"] == pytest.approx(inst.values["v"] ** 2 * __import__("math").sin(__import__("math").radians(2 * inst.values["theta"])) / 9.8)


# ------------------------------------------------------------------------------------------ grading

def test_quality_follows_the_credit_earned():
    inst, _ = pr.variant(add(), D0)
    assert pr.grade(inst, perfect(inst)).quality == 5
    a, d = inst.answers["a"], inst.answers["d"]
    assert pr.grade(inst, {"a": repr(a), "d": repr(d * 1.03)}).quality == 4                      # one answer only close (rounding)
    assert pr.grade(inst, {"a": repr(a), "d": repr(d * 9)}).quality == 3                         # half the credit
    assert pr.grade(inst, {"a": repr(a * 9), "d": repr(d * 9)}).quality == 1
    assert pr.grade(inst, {}).quality == 0


def test_answers_are_read_with_units_and_formats_and_a_diagnosis_is_kept():
    inst, _ = pr.variant(add(), D0)
    g = pr.grade(inst, {"a": f"{inst.answers['a']:.6g} m/s^2", "d": f"{-inst.answers['d']:.6g}"})
    assert g.verdicts["a"].status == "correct" and g.verdicts["d"].status == "wrong" and g.verdicts["d"].diagnosis == "sign error"
    assert not g.all_correct and g.n_correct == 1 and g.summary == "1 of 2 correct"
    assert pr.grade(inst, {"a": "no idea", "d": "?"}).verdicts["a"].status == "unreadable"


# ------------------------------------------------------------------------------------------ recording

def test_a_correct_attempt_schedules_the_next_review_and_remembers_the_first_day():
    item = add()
    state = attempt(item, D0)
    got = pr.get_item(item.id)
    assert state.due == D0 + timedelta(days=1) and got.state.reps == 1 and got.first_attempt == D0 and got.last_reviewed == D0 and not got.is_new


def test_an_extra_attempt_does_not_change_the_schedule():
    item = add()
    attempt(item, D0)
    before = pr.get_item(item.id).state
    attempt(pr.get_item(item.id), D0, correct=False, scheduled=False)
    assert pr.get_item(item.id).state == before and len(pr.attempts()) == 2


def test_recording_for_an_unknown_item_is_an_error():
    inst, seed = pr.variant(add(), D0)
    with pytest.raises(pr.PracticeError, match="not in your practice list"):
        pr.record(9999, seed, pr.grade(inst, perfect(inst)), D0)


def test_a_month_of_perfect_practice_spaces_the_reviews_out():
    item = add()
    day, due_days = D0, []
    for _ in range(6):
        attempt(pr.get_item(item.id), day)
        day = pr.get_item(item.id).state.due
        due_days.append((day - D0).days)
    assert due_days[0] == 1 and due_days[1] == 7 and due_days[2] > 20 and all(b > a for a, b in zip(due_days, due_days[1:]))


# ------------------------------------------------------------------------------------------ streaks

def test_the_streak_counts_consecutive_days_and_survives_until_a_day_is_missed():
    item = add()
    for d in (0, 1, 2, 3):
        attempt(pr.get_item(item.id), D0 + timedelta(days=d), scheduled=False)
    assert pr.streak(D0 + timedelta(days=3)) == (4, 4)
    assert pr.streak(D0 + timedelta(days=4)) == (4, 4)                       # not practised yet today: still alive
    assert pr.streak(D0 + timedelta(days=5)) == (0, 4)                       # a day was missed


def test_a_gap_resets_the_current_streak_but_not_the_longest():
    item = add()
    for d in (0, 1, 2, 5, 6):
        attempt(pr.get_item(item.id), D0 + timedelta(days=d), scheduled=False)
    assert pr.streak(D0 + timedelta(days=6)) == (2, 3)
    assert pr.streak(D0 + timedelta(days=2)) == (3, 3)                       # "as of" an earlier day: only attempts up to then count
    assert pr.streak(D0) == (1, 1)


def test_no_practice_means_no_streak():
    assert pr.streak(D0) == (0, 0)


# ------------------------------------------------------------------------------------------ mastery

def concept(name, today):
    return next(m for m in pr.mastery(today) if m.concept == name)


def test_a_concept_with_no_attempts_is_new_and_has_no_accuracy():
    add()
    m = concept("kinematics", D0)
    assert m.level == "new" and m.accuracy is None and m.attempts == 0


def test_a_few_good_attempts_are_solid_and_several_are_mastered():
    item = add()
    for d in range(2):
        attempt(pr.get_item(item.id), D0 + timedelta(days=d), scheduled=False)
    assert concept("kinematics", D0 + timedelta(days=1)).level == "solid"          # accurate, but too few attempts to call it mastered
    for d in range(2, 6):
        attempt(pr.get_item(item.id), D0 + timedelta(days=d), scheduled=False)
    m = concept("kinematics", D0 + timedelta(days=5))
    assert m.level == "mastered" and m.accuracy == pytest.approx(1.0) and m.attempts == 6


def test_failures_make_a_concept_learning():
    item = add()
    for d in range(4):
        attempt(pr.get_item(item.id), D0 + timedelta(days=d), correct=(d == 0), scheduled=False)
    m = concept("kinematics", D0 + timedelta(days=3))
    assert m.level == "learning" and m.accuracy < pr.LEARNING_ACCURACY


def test_old_results_count_for_less_than_recent_ones():
    item = add()
    for d in range(3):                                                         # three failures long ago ...
        attempt(pr.get_item(item.id), D0 + timedelta(days=d), correct=False, scheduled=False)
    for d in range(60, 63):                                                    # ... then three successes two months later
        attempt(pr.get_item(item.id), D0 + timedelta(days=d), scheduled=False)
    recent = concept("kinematics", D0 + timedelta(days=62))
    flat = sum(a.score for a in pr.attempts()) / 6
    assert recent.accuracy > flat and recent.accuracy > 0.8


def test_an_item_with_two_concepts_counts_towards_both_and_a_removed_item_keeps_its_history():
    item = add()
    attempt(item, D0, scheduled=False)
    assert concept("kinematics", D0).attempts == 1 and concept("uniform acceleration", D0).attempts == 1
    pr.remove_item(item.id)
    assert concept("kinematics", D0).attempts == 1


def test_weakest_concepts_come_first_and_untried_ones_last():
    good, bad = add("Series circuit"), add("Pendulum period")
    add("Ideal gas")
    attempt(good, D0, scheduled=False)
    attempt(bad, D0, correct=False, scheduled=False)
    order = [m.concept for m in pr.mastery(D0)]
    assert order.index("oscillations") < order.index("circuits") < order.index("thermodynamics")


# ------------------------------------------------------------------------------------------ the heatmap and forecast

def test_the_heatmap_has_a_row_per_concept_a_column_per_week_and_gaps_where_nothing_was_done():
    item = add()
    attempt(item, D0, scheduled=False)                                         # this week (D0 is a Monday)
    attempt(pr.get_item(item.id), D0 - timedelta(days=14), correct=False, scheduled=False)
    concepts, weeks, matrix = pr.heatmap(D0, weeks=4)
    row = matrix[concepts.index("kinematics")]
    assert len(weeks) == 4 and len(matrix) == len(concepts) == 2
    assert row[3] == pytest.approx(1.0) and row[1] == pytest.approx(0.0) and row[0] is None and row[2] is None


def test_attempts_outside_the_window_are_not_shown():
    item = add()
    attempt(item, D0 - timedelta(days=100), scheduled=False)
    _, _, matrix = pr.heatmap(D0, weeks=4)
    assert all(cell is None for row in matrix for cell in row)


def test_the_forecast_counts_reviews_per_day_with_overdue_ones_today():
    a, b, c = add("Projectile range"), add("Series circuit"), add("Pendulum period")
    attempt(a, D0 - timedelta(days=10))                                        # due long ago -> counted today
    attempt(b, D0)                                                             # due tomorrow
    add("Ideal gas")                                                           # new: not a review
    counts = dict(pr.forecast(D0, days=5))
    assert counts[D0] == 1 and counts[D0 + timedelta(days=1)] == 1 and sum(counts.values()) == 2


def test_the_summary_pulls_it_together():
    item = add()
    add("Series circuit")
    attempt(item, D0)
    s = pr.summary(D0)
    assert s["items"] == 2 and s["new"] == 1 and s["due"] == 0 and s["practised_today"] and s["streak"] == 1
    assert s["accuracy_7d"] == pytest.approx(1.0) and s["attempts_7d"] == 1
    assert pr.summary(D0 + timedelta(days=1))["due"] == 1
