"""'What next?': suggestions follow the state of the result, destinations come from the registry, and handoffs are accepted by their targets."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from modules import mcmc
from modules import next_steps as ns
from modules.equation_engine import build_model
from modules.mode_registry import REGISTRY
from modules.problem_library import LIBRARY
from modules.solve_compare import deterministic_report

TEXT, RAW = LIBRARY["Constant acceleration (kinematics)"]


def solved(name="Constant acceleration (kinematics)"):
    text, raw = LIBRARY[name]
    model = build_model(copy.deepcopy(raw))
    return model, deterministic_report(model), text


def keys(steps):
    return [s.key for s in steps]


# ------------------------------------------------------------------------------------------ the registry lookup

def test_destinations_are_found_through_the_registry_by_page_module():
    for page in (ns.PAGE_PROBLEM_SETS, ns.PAGE_RECALL, ns.PAGE_POSTERIOR, ns.PAGE_PRACTICE, ns.PAGE_CHEMISTRY, ns.PAGE_COMPARE):
        assert ns.mode_for(page) in REGISTRY.labels
    with pytest.raises(KeyError):
        ns.mode_for("ui.no_such_page")


def test_a_destination_that_is_not_installed_is_simply_not_offered(monkeypatch):
    monkeypatch.setattr(ns, "mode_for", lambda page: (_ for _ in ()).throw(KeyError(page)))
    model, report, text = solved()
    assert keys(ns.for_solution(model, report, text)) == ["add_practice"]            # the in-place action needs no destination mode
    assert ns.for_fit([1, 2], [1, 2], "a*x", ["a"], [1.0]) == []


# ------------------------------------------------------------------------------------------ after a solved problem

def test_a_verified_numeric_problem_offers_practice_and_recall():
    model, report, text = solved()
    steps = ns.for_solution(model, report, text)
    assert keys(steps) == ["pset", "add_practice", "recall"]
    pset = steps[0]
    assert pset.mode == ns.mode_for(ns.PAGE_PROBLEM_SETS) and pset.prefill == {"pset_source": "The problem that is open"}
    add = steps[1]
    assert add.mode is None and add.action == ns.ADD_TO_PRACTICE and add.payload["text"] == text and add.payload["raw"] == model.raw_json
    assert add.payload["concepts"] == ["kinematics"]
    recall = steps[2]
    assert recall.mode == ns.mode_for(ns.PAGE_RECALL) and recall.prefill == {"recall_query": text}


def test_a_problem_that_failed_verification_is_not_offered_as_practice_but_can_be_compared():
    model, report, text = solved()
    failed = SimpleNamespace(passed=False)
    steps = ns.for_solution(model, failed, text)
    assert keys(steps) == ["recall", "compare"] and steps[1].mode == ns.mode_for(ns.PAGE_COMPARE)


def test_a_problem_whose_answer_is_not_a_number_is_not_offered_a_practice_set_it_cannot_have():
    ode = SimpleNamespace(problem_type="ode", problem_domain="dynamics", raw_json={}, solve_for=["y"])
    assert keys(ns.for_solution(ode, SimpleNamespace(passed=True), "y' = -y")) == ["recall"]


def test_a_chemistry_problem_also_points_at_the_chemistry_lab():
    model, report, text = solved("pH of a strong acid")
    steps = ns.for_solution(model, report, text)
    assert "chem" in keys(steps) and next(s for s in steps if s.key == "chem").mode == ns.mode_for(ns.PAGE_CHEMISTRY)
    assert next(s for s in steps if s.key == "add_practice").payload["concepts"] == ["chemistry"]


def test_at_most_four_suggestions_are_made():
    model, report, text = solved("pH of a strong acid")
    assert len(ns.for_solution(model, report, text)) <= 4


def test_every_suggestion_says_why():
    model, report, text = solved()
    assert all(s.label and s.reason for s in ns.for_solution(model, report, text))


# ------------------------------------------------------------------------------------------ handoff of a fit to the posterior sampler

@pytest.mark.parametrize("family, degree, values, expr, names", [
    ("linear", 1, {"c1": 2.0, "c0": 1.0}, "c0 + c1*x", ["c0", "c1"]),
    ("polynomial", 3, {"c3": 4.0, "c2": 3.0, "c1": 2.0, "c0": 1.0}, "c0 + c1*x + c2*x**2 + c3*x**3", ["c0", "c1", "c2", "c3"]),
    ("exponential", 1, {"a": 2.0, "b": 0.5}, "a*exp(b*x)", ["a", "b"]),
    ("power", 1, {"a": 2.0, "b": 1.5}, "a*x**b", ["a", "b"]),
    ("logarithmic", 1, {"a": 3.0, "b": 1.0}, "a*log(x) + b", ["a", "b"]),
])
def test_each_builtin_fit_has_an_explicit_model_the_sampler_accepts(family, degree, values, expr, names):
    got = ns.fit_to_model(family, degree, None, None, values)
    assert got is not None and got[0] == expr and got[1] == names
    model = mcmc.compile_model(*got[:2])                                                  # the sampler really can use it
    assert model(np.array([1.0, 2.0]), np.array([values[n] for n in names])).shape == (2,)


def test_a_polynomials_degree_comes_from_its_own_coefficients_not_the_pages_placeholder():
    got = ns.fit_to_model("polynomial", 2, None, None, {"c0": 1.0, "c1": 2.0, "c2": 3.0, "c3": 4.0, "c4": 5.0})
    assert got[1] == ["c0", "c1", "c2", "c3", "c4"]                                       # degree 4, whatever the page said


def test_a_custom_fit_hands_over_its_own_expression_and_values():
    assert ns.fit_to_model("custom", 2, "A*exp(-k*x)", ["A", "k"], {"A": 3.0, "k": 0.2}) == ("A*exp(-k*x)", ["A", "k"], [3.0, 0.2])
    assert ns.fit_to_model("custom", 2, None, None, {}) is None and ns.fit_to_model("mystery", 1, None, None, {}) is None


def test_the_posterior_handoff_fills_exactly_the_widgets_the_posterior_page_reads():
    steps = ns.for_fit([0.0, 1.0, 2.0], [1.0, 0.5, 0.25], "a*exp(b*x)", ["a", "b"], [1.0, -0.7])
    (step,) = steps
    assert step.mode == ns.mode_for(ns.PAGE_POSTERIOR)
    assert set(step.prefill) == {"post_source", "post_paste", "post_expr", "post_names", "post_guess"}
    assert step.prefill["post_names"] == "a, b" and step.prefill["post_guess"] == "1, -0.7" and step.prefill["post_paste"].startswith("x,y\n0,1\n1,0.5")


def test_long_data_is_thinned_evenly_for_the_handoff():
    x = np.linspace(0, 10, 5000)
    csv = ns.csv_for(x, x ** 2).splitlines()
    assert len(csv) == ns.MAX_CSV_ROWS + 1 and csv[1].startswith("0,0") and csv[-1].startswith("10,100")
    assert len(ns.csv_for([1, 2, 3], [1, 4, 9]).splitlines()) == 4


def test_the_add_to_practice_step_for_a_problem_set_source():
    (step,) = ns.for_problem(TEXT, RAW, "My problem", ["kinematics"])
    assert step.action == ns.ADD_TO_PRACTICE and step.payload["title"] == "My problem" and step.payload["concepts"] == ["kinematics"]
