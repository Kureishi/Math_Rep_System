"""Problem sets: variants whose keys are verified, statements that cannot disagree with their keys, and grading that explains mistakes."""
import copy
import json
import math

import pytest

from modules import problem_sets as ps
from modules.equation_engine import build_model
from modules.problem_library import LIBRARY
from modules.solve_compare import deterministic_report
from tests.conftest import KINEMATICS_BUGGY_JSON

KIN_TEXT, KIN_RAW = LIBRARY["Constant acceleration (kinematics)"]


@pytest.fixture(scope="module")
def kin():
    return ps.generate(KIN_TEXT, KIN_RAW, n=8, seed=3)


def _answers_from_values(raw, values):
    trial = copy.deepcopy(raw)
    for v in trial["variables"]:
        if v["symbol"] in values:
            v["known_value"] = repr(values[v["symbol"]])
    return deterministic_report(build_model(trial)).sympy_numeric_answers


# ------------------------------------------------------------------------------------------ the keys are right

def test_every_key_is_the_right_answer_by_an_independent_formula(kin):
    assert len(kin.instances) == 8
    for i in kin.instances:
        u, v, t = i.values["u"], i.values["v"], i.values["t"]
        a = (v - u) / t
        assert i.answers["a"] == pytest.approx(a) and i.answers["d"] == pytest.approx(0.5 * a * t * t + u * t)


def test_every_library_problem_makes_a_set_whose_keys_reverify_from_the_stored_values():
    for name, (text, raw) in LIBRARY.items():
        s = ps.generate(text, raw, n=4, seed=1)
        assert len(s.instances) == 4, name
        for i in s.instances:
            fresh = _answers_from_values(raw, i.values)
            assert all(fresh[t] == pytest.approx(i.answers[t], rel=1e-9) for t in s.targets), name


def test_projectile_and_circuit_keys_match_their_formulas():
    text, raw = LIBRARY["Projectile range"]
    for i in ps.generate(text, raw, n=5, seed=2).instances:
        assert i.answers["R"] == pytest.approx(i.values["v"] ** 2 * math.sin(math.radians(2 * i.values["theta"])) / 9.8)
    text, raw = LIBRARY["Series circuit"]
    for i in ps.generate(text, raw, n=5, seed=2).instances:
        current = i.values["V"] / (i.values["R1"] + i.values["R2"])
        assert i.answers["I"] == pytest.approx(current) and i.answers["P"] == pytest.approx(i.values["V"] * current)


def test_it_is_deterministic_for_a_seed_and_differs_for_another():
    a, b = ps.generate(KIN_TEXT, KIN_RAW, n=6, seed=9), ps.generate(KIN_TEXT, KIN_RAW, n=6, seed=9)
    c = ps.generate(KIN_TEXT, KIN_RAW, n=6, seed=10)
    assert [i.values for i in a.instances] == [i.values for i in b.instances] and [i.values for i in a.instances] != [i.values for i in c.instances]


# ------------------------------------------------------------------------------------------ what varies

def test_constants_and_zeros_are_never_varied():
    text, raw = LIBRARY["Projectile range"]
    s = ps.generate(text, raw, n=6, seed=1)
    assert "g" not in s.varied and all(i.values["g"] == 9.8 for i in s.instances)
    text, raw = LIBRARY["Ideal gas"]
    assert "R" not in ps.generate(text, raw, n=4).varied
    zero = copy.deepcopy(KIN_RAW)
    next(v for v in zero["variables"] if v["symbol"] == "u")["known_value"] = "0"
    assert "u" not in ps.generate(KIN_TEXT.replace("8 m/s", "0 m/s"), zero, n=4).varied


def test_the_variants_spread_across_the_range_and_none_repeats_or_equals_the_original(kin):
    for sym in kin.varied:
        vals = [i.values[sym] for i in kin.instances]
        base = kin.base_values[sym]
        assert (max(vals) - min(vals)) >= 0.5 * (2 * ps.DEFAULT_SPREAD * base)
        assert all(abs(v - base) <= ps.DEFAULT_SPREAD * base + 1 for v in vals)
    keys = [tuple(i.values[s] for s in kin.varied) for i in kin.instances]
    assert len(set(keys)) == len(keys) and tuple(kin.base_values[s] for s in kin.varied) not in keys


def test_numbers_are_written_the_way_the_original_writes_them():
    text, raw = LIBRARY["Ideal gas"]
    s = ps.generate(text, raw, n=6, seed=4)
    for i in s.instances:
        assert round(i.values["T"]) == i.values["T"]                                 # "300 K": whole
        assert round(i.values["n"], 1) == i.values["n"]                              # "2.0 mol": one decimal
        assert round(i.values["V"], 3) == i.values["V"]                              # "0.050 m^3": three decimals


def test_choosing_which_values_to_vary_is_respected():
    s = ps.generate(KIN_TEXT, KIN_RAW, n=5, seed=1, vary=["t"])
    assert s.varied == ["t"] and all(i.values["u"] == 8 and i.values["v"] == 20 for i in s.instances)


# ------------------------------------------------------------------------------------------ the statement agrees with the key

def test_a_rewritten_statement_contains_exactly_the_values_of_its_key(kin):
    for i in kin.instances:
        assert i.rewritten
        found = [float(m.group()) for m in ps._NUMBER.finditer(i.text)]
        assert sorted(found) == sorted([i.values["u"], i.values["v"], i.values["t"]])


def test_percentages_are_rewritten_as_percentages():
    text, raw = LIBRARY["Compound interest"]
    s = ps.generate(text, raw, n=5, seed=1)
    for i in s.instances:
        assert i.rewritten and f"{round(i.values['r'] * 100)}%" in i.text


def test_an_ambiguous_number_means_the_wording_is_kept_and_the_values_are_listed():
    raw = copy.deepcopy(KIN_RAW)
    text = "A car goes from 8 m/s to 20 m/s, then another car goes 8 m/s faster, over 6 seconds. Find its acceleration and distance."
    s = ps.generate(text, raw, n=4, seed=1)
    assert all(not i.rewritten and i.text == text for i in s.instances) and any("could not be rewritten" in w for w in s.warnings)
    assert "Use: u = " in ps.worksheet_markdown(s)


def test_two_variables_sharing_a_value_are_not_guessed_between():
    raw = copy.deepcopy(KIN_RAW)
    next(v for v in raw["variables"] if v["symbol"] == "t")["known_value"] = "8"
    text, ok = ps.render_statement("From 8 m/s to 20 m/s in 8 seconds.", {"u": 8.0, "v": 20.0, "t": 8.0}, {"u": 9.0, "v": 22.0, "t": 7.0}, ["u", "v", "t"])
    assert not ok and text.startswith("From 8 m/s")


def test_a_statement_can_be_rewritten_when_the_numbers_are_unambiguous():
    text, ok = ps.render_statement("From 8 m/s to 20.5 m/s in 6 s.", {"u": 8.0, "v": 20.5, "t": 6.0}, {"u": 9.0, "v": 22.5, "t": 7.0}, ["u", "v", "t"])
    assert ok and text == "From 9 m/s to 22.5 m/s in 7 s."


# ------------------------------------------------------------------------------------------ what it refuses

def test_a_differential_equation_is_refused_with_a_reason():
    raw = copy.deepcopy(KIN_RAW)
    raw["problem_type"] = "ode"
    with pytest.raises(ps.ProblemSetError, match="ode problem"):
        ps.generate(KIN_TEXT, raw)


def test_a_source_that_fails_its_own_checks_is_refused():
    with pytest.raises(ps.ProblemSetError, match="does not pass its own checks"):
        ps.generate("buggy", json.loads(KINEMATICS_BUGGY_JSON))


@pytest.mark.parametrize("kwargs, message", [(dict(n=0), "between 1"), (dict(n=500), "between 1"), (dict(spread=0.0), "spread"),
                                              (dict(spread=2.0), "spread"), (dict(vary=["zzz"]), "no known value to vary")])
def test_bad_requests_are_refused(kwargs, message):
    with pytest.raises(ps.ProblemSetError, match=message):
        ps.generate(KIN_TEXT, KIN_RAW, **kwargs)


def test_variants_that_would_be_nonsense_are_replaced_not_shipped():
    """A large spread on a time makes some variants invalid or wildly different: they are discarded and replaced."""
    s = ps.generate(KIN_TEXT, KIN_RAW, n=10, seed=1, spread=0.85)
    assert len(s.instances) == 10 and all(i.answers["d"] > 0 and i.values["t"] > 0 for i in s.instances)


def test_forty_problems_take_a_moment_not_a_minute():
    import time
    t0 = time.perf_counter()
    assert len(ps.generate(KIN_TEXT, KIN_RAW, n=40, seed=1).instances) == 40
    assert time.perf_counter() - t0 < 20


# ------------------------------------------------------------------------------------------ exports

def test_the_worksheet_hides_the_answers_unless_asked(kin):
    plain, keyed = ps.worksheet_markdown(kin), ps.worksheet_markdown(kin, with_answers=True)
    assert "Answer key" not in plain and "Answer key" in keyed
    assert f"a = {ps.fmt(kin.instances[0].answers['a'])}" in keyed and f"a = {ps.fmt(kin.instances[0].answers['a'])}" not in plain


def test_the_html_worksheet_escapes_its_text_and_prints_a_key_only_on_request():
    text, raw = LIBRARY["Pendulum period"]
    s = ps.generate("<b>A</b> " + text, raw, n=2, seed=1, title="Quiz <1> & more")
    page = ps.worksheet_html(s)
    assert "&lt;b&gt;A&lt;/b&gt;" in page and "Quiz &lt;1&gt; &amp; more" in page and "<b>A</b>" not in page and "Answer key" not in page
    assert "Answer key" in ps.worksheet_html(s, with_answers=True)


def test_the_answer_key_csv_has_one_row_per_problem_and_target(kin):
    rows = ps.answer_key_csv(kin).strip().splitlines()
    assert len(rows) == 1 + len(kin.instances) * len(kin.targets) and rows[0].startswith("problem,target,answer,unit")


def test_the_set_round_trips_through_json(kin):
    again = ps.from_json(ps.to_json(kin))
    assert [i.values for i in again.instances] == [i.values for i in kin.instances] and again.targets == kin.targets
    assert [i.answers for i in again.instances] == [i.answers for i in kin.instances]


@pytest.mark.parametrize("text", ["not json", "{}", json.dumps({"format_version": 99}), json.dumps({"format_version": 1})])
def test_a_file_that_is_not_a_problem_set_is_refused(text):
    with pytest.raises(ps.ProblemSetError):
        ps.from_json(text)


def test_the_submission_template_lists_every_student_problem_and_target(kin):
    rows = ps.submission_template_csv(kin, ["Ana", "Ben"]).strip().splitlines()
    assert len(rows) == 1 + 2 * len(kin.instances) * len(kin.targets) and rows[0] == "student,problem,target,answer"


# ------------------------------------------------------------------------------------------ reading an answer

@pytest.mark.parametrize("text, value, unit", [
    ("12.5", 12.5, ""), ("12.5 m/s", 12.5, "m/s"), ("-3.2e2", -320.0, ""), ("1.25e1 m", 12.5, "m"), ("1.25 x 10^1", 12.5, ""),
    ("1.25×10^-2 kg", 0.0125, "kg"), ("1,234.5", 1234.5, ""), ("12,5 m", 12.5, "m"), ("1,234", 1234.0, ""), ("−4", -4.0, ""),
    ("  7  ", 7.0, ""), (".5", 0.5, ""),
])
def test_answers_are_read_leniently(text, value, unit):
    got, u = ps.parse_answer(text)
    assert got == pytest.approx(value) and u == unit


@pytest.mark.parametrize("text", ["", "no idea", "m/s", "???"])
def test_text_with_no_number_is_unreadable(text):
    assert ps.parse_answer(text)[0] is None


# ------------------------------------------------------------------------------------------ grading one answer

def test_correct_close_wrong_and_missing():
    assert ps.grade_value(100.5, 100.0).status == "correct"
    close = ps.grade_value(103.0, 100.0)
    assert close.status == "close" and close.score == ps.PARTIAL_CREDIT and "rounding" in close.diagnosis
    assert ps.grade_value(150.0, 100.0).status == "wrong" and ps.grade_value(None, 100.0).status == "missing"


def test_a_tolerance_the_teacher_sets_is_honoured():
    assert ps.grade_value(103.0, 100.0, rel_tol=0.05).status == "correct"
    assert ps.grade_value(100.5, 100.0, rel_tol=0.001).status != "correct"


def test_an_answer_that_is_exactly_zero_is_graded_without_dividing_by_zero():
    assert ps.grade_value(0.0, 0.0).status == "correct" and ps.grade_value(1.0, 0.0).status == "wrong"


@pytest.mark.parametrize("student, key, expected", [
    (-42.0, 42.0, "sign error"), (84.0, 42.0, "factor of 2"), (21.0, 42.0, "factor of 2"), (4200.0, 42.0, "10^2"),
    (0.042, 42.0, "10^-3"), (1 / 42.0, 42.0, "reciprocal"), (42.0 * math.pi / 180, 42.0, "degrees and radians"),
    (42.0 ** 2, 42.0, "square"), (math.sqrt(42.0), 42.0, "square root"), (57.3, 42.0, ""),
])
def test_the_slip_behind_a_wrong_answer_is_named(student, key, expected):
    assert expected in ps.diagnose(student, key)


def test_an_answer_to_a_different_quantity_or_the_original_problem_is_recognised():
    assert "value of d" in ps.diagnose(84.0, 2.0, other_answers={"a": 2.0, "d": 84.0}, target="a")
    assert "original problem" in ps.diagnose(5.0, 9.0, base_answers={"a": 5.0}, target="a")


# ------------------------------------------------------------------------------------------ grading a class

def _rows(kin, who, answers):
    """answers: {(problem, target): text}"""
    return [dict(student=who, problem=str(p), target=t, answer=a) for (p, t), a in answers.items()]


def _key(kin):
    return {(i.index, t): repr(i.answers[t]) for i in kin.instances for t in kin.targets}


def test_a_perfect_student_scores_full_marks_and_a_blank_one_zero(kin):
    rows = _rows(kin, "Ana", _key(kin))
    report = ps.grade_submissions(kin, rows + [dict(student="Ben", problem="1", target="a", answer="")])
    ana, ben = report.students
    assert ana.percent == 100.0 and ana.score == ana.possible == len(kin.instances) * len(kin.targets)
    assert ben.score == 0 and report.mean_percent == pytest.approx(50.0, abs=0.01)


def test_units_and_formats_in_answers_do_not_matter(kin):
    i = kin.instances[0]
    rows = [dict(student="Ana", problem="1", target="a", answer=f"{i.answers['a']:.4g} m/s^2"),
            dict(student="Ana", problem="1", target="d", answer=f"{i.answers['d']:.3e} m")]
    verdicts = {it.target: it.verdict.status for it in ps.grade_submissions(kin, rows).students[0].items if it.problem == 1}
    assert verdicts == {"a": "correct", "d": "correct"}


def test_mistakes_are_diagnosed_per_item_and_summarised_per_problem(kin):
    i = kin.instances[0]
    rows = []
    for who in ("Ana", "Ben", "Cy"):
        rows.append(dict(student=who, problem="1", target="a", answer=repr(-i.answers["a"])))           # all three slip on the sign
    rows.append(dict(student="Ana", problem="1", target="d", answer=repr(i.answers["d"])))
    report = ps.grade_submissions(kin, rows)
    stat = next(p for p in report.problems if p.problem == 1 and p.target == "a")
    assert stat.correct == 0 and stat.common_mistake == "sign error"
    assert any("Nobody got" in n and "1 (a)" in n for n in report.notes)
    first = report.students[0].items[0]
    assert first.verdict.status == "wrong" and first.verdict.diagnosis == "sign error"


def test_facility_is_the_share_of_the_class_who_got_it_right(kin):
    i = kin.instances[1]
    rows = [dict(student="Ana", problem="2", target="a", answer=repr(i.answers["a"])),
            dict(student="Ben", problem="2", target="a", answer=repr(i.answers["a"] * 3)),
            dict(student="Cy", problem="2", target="a", answer=repr(i.answers["a"]))]
    stat = next(p for p in ps.grade_submissions(kin, rows).problems if p.problem == 2 and p.target == "a")
    assert stat.facility == pytest.approx(2 / 3) and stat.correct == 2


def test_unusable_rows_are_listed_with_their_reason_and_do_not_stop_the_rest(kin):
    rows = [dict(student="", problem="1", target="a", answer="1"), dict(student="Ana", problem="x", target="a", answer="1"),
            dict(student="Ana", problem="99", target="a", answer="1"), dict(student="Ana", problem="1", target="zz", answer="1"),
            dict(student="Ana", problem="1", target="a", answer=repr(kin.instances[0].answers["a"]))]
    report = ps.grade_submissions(kin, rows)
    assert len(report.problems_with_errors) == 4 and report.students[0].score >= 1
    joined = " ".join(report.problems_with_errors)
    assert "no student name" in joined and "not a problem number" in joined and "no problem 99" in joined and "not one of the quantities" in joined


def test_a_repeated_answer_uses_the_last_and_says_so(kin):
    i = kin.instances[0]
    rows = [dict(student="Ana", problem="1", target="a", answer="0"), dict(student="Ana", problem="1", target="a", answer=repr(i.answers["a"]))]
    report = ps.grade_submissions(kin, rows)
    assert any("twice" in n for n in report.notes)
    assert next(it for it in report.students[0].items if it.problem == 1 and it.target == "a").verdict.status == "correct"


def test_text_that_is_not_a_number_is_unreadable_not_wrong(kin):
    rows = [dict(student="Ana", problem="1", target="a", answer="I don't know")]
    assert ps.grade_submissions(kin, rows).students[0].items[0].verdict.status == "unreadable"


def test_the_target_can_be_left_out_when_there_is_only_one():
    text, raw = LIBRARY["Pendulum period"]
    s = ps.generate(text, raw, n=3, seed=1)
    rows = [dict(student="Ana", problem=str(i.index), answer=repr(i.answers["T"])) for i in s.instances]
    assert ps.grade_submissions(s, rows).students[0].percent == 100.0


def test_results_export_as_csv(kin):
    report = ps.grade_submissions(kin, _rows(kin, "Ana", _key(kin)))
    assert report.results_csv().splitlines()[0] == "student,problem,target,answer,key,status,score,note"
    assert report.totals_csv().splitlines()[1].startswith("Ana,") and report.totals_csv().strip().endswith("100.0")


def test_a_submissions_file_is_read_and_validated():
    rows = ps.read_submissions("student,problem,target,answer\nAna,1,a,2.0\nBen,1,a,3\n")
    assert rows[0] == {"student": "Ana", "problem": "1", "target": "a", "answer": "2.0"}
    with pytest.raises(ps.ProblemSetError, match="header"):
        ps.read_submissions("name,q,ans\nAna,1,2\n")
    with pytest.raises(ps.ProblemSetError, match="empty"):
        ps.read_submissions("   ")
