"""ui/cache.py: content fingerprints and the per-session result cache.

The property that matters is "never stale": two inputs that differ in any way
that could change the result must not share an entry. So the fingerprint tests
are mostly about telling similar-looking things apart, and anything that can't
be told apart by content must be REFUSED rather than guessed at."""
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

import ui.cache as cache
from modules.equation_engine import build_model
from tests.test_ode_trajectories import decay_model


@pytest.fixture
def session(monkeypatch):
    state: dict = {}
    monkeypatch.setattr(cache, "st", SimpleNamespace(session_state=state))
    return state


# ------------------------------------------------------------------ fingerprint: same content, same hash

def test_equal_content_gives_equal_fingerprints_however_it_was_built():
    a = np.array([1.0, 2.0, 3.0])
    assert cache.fingerprint(a) == cache.fingerprint(np.array([1.0, 2.0, 3.0]))
    assert cache.fingerprint({"b": 1, "a": 2}) == cache.fingerprint({"a": 2, "b": 1})        # dict order is not content
    assert cache.fingerprint({3, 1, 2}) == cache.fingerprint({2, 3, 1})
    x = sp.Symbol("x")
    assert cache.fingerprint(x ** 2 + 1) == cache.fingerprint(sp.Symbol("x") ** 2 + 1)
    assert cache.fingerprint(np.arange(6).reshape(2, 3)) == cache.fingerprint(np.arange(6).reshape(2, 3).copy(order="F"))


def test_non_contiguous_arrays_hash_by_content():
    base = np.arange(12.0).reshape(3, 4)
    assert cache.fingerprint(base[:, ::2]) == cache.fingerprint(np.ascontiguousarray(base[:, ::2]))


# ------------------------------------------------------------------ fingerprint: anything that could change a result differs

@pytest.mark.parametrize("a, b", [
    (np.array([1.0, 2.0]), np.array([1.0, 2.5])),                       # one value
    (np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])),                  # length
    (np.arange(6).reshape(2, 3), np.arange(6).reshape(3, 2)),           # same bytes, different shape
    (np.array([1, 2], dtype=np.int64), np.array([1.0, 2.0])),           # dtype
    (1, 1.0), (1, True), (0.0, -0.0), ("1", 1),
    ([1, 2], (1, 2)),                                                   # list vs tuple
    ([[1, 2], [3]], [[1], [2, 3]]),                                     # same items, different grouping
    ("ab", "a b"), (("a", "b"), ("ab",)), ({"a": 1}, {"a": 2}), ({"a": 1}, {"b": 1}),
    (None, 0), (None, ""), (b"x", "x"),
    (sp.Symbol("x") ** 2, sp.Symbol("x") ** 3), (sp.Symbol("x"), sp.Symbol("x", positive=True)),
])
def test_anything_that_could_change_a_result_changes_the_fingerprint(a, b):
    assert cache.fingerprint(a) != cache.fingerprint(b)


def test_nan_and_infinity_fingerprint_stably():
    assert cache.fingerprint(np.array([np.nan, np.inf])) == cache.fingerprint(np.array([np.nan, np.inf]))
    assert cache.fingerprint(float("nan")) == cache.fingerprint(float("nan"))


def test_numpy_scalars_and_python_scalars_are_kept_apart_not_conflated():
    assert cache.fingerprint(np.float64(1.5)) != cache.fingerprint(1.5)         # a miss, never a wrong hit
    assert cache.fingerprint(np.float64(1.5)) == cache.fingerprint(np.float64(1.5))
    assert cache.fingerprint(np.bool_(True)) == cache.fingerprint(np.bool_(True))


@dataclasses.dataclass
class _Point:
    x: float
    tags: list


@dataclasses.dataclass
class _OtherPoint:
    x: float
    tags: list


def test_dataclasses_hash_by_class_and_every_field():
    assert cache.fingerprint(_Point(1.0, ["a"])) == cache.fingerprint(_Point(1.0, ["a"]))
    assert cache.fingerprint(_Point(1.0, ["a"])) != cache.fingerprint(_Point(1.0, ["b"]))
    assert cache.fingerprint(_Point(1.0, ["a"])) != cache.fingerprint(_Point(2.0, ["a"]))
    assert cache.fingerprint(_Point(1.0, ["a"])) != cache.fingerprint(_OtherPoint(1.0, ["a"]))
    assert cache.fingerprint(_Point(1.0, [np.array([1.0])])) != cache.fingerprint(_Point(1.0, [np.array([2.0])]))


def test_a_dataclass_class_itself_is_not_an_instance():
    with pytest.raises(TypeError):
        cache.fingerprint(_Point)


@pytest.mark.parametrize("thing", [lambda: 1, object(), np.array([object()], dtype=object), sum, SimpleNamespace(a=1)])
def test_things_that_cannot_be_hashed_by_content_are_refused(thing):
    with pytest.raises(TypeError, match="can't fingerprint|can't be fingerprinted"):
        cache.fingerprint(thing)
    with pytest.raises(TypeError):
        cache.fingerprint([1, [2, {"k": thing}]])                     # also when buried inside a structure


def test_the_parts_are_ordered_and_delimited():
    assert cache.fingerprint("a", "b") != cache.fingerprint("b", "a")
    assert cache.fingerprint("ab") != cache.fingerprint("a", "b")


# ------------------------------------------------------------------ cached()

def test_computes_once_while_the_inputs_are_unchanged(session):
    calls = []
    compute = lambda: calls.append(1) or object()
    first = cache.cached("k", (1, "a", np.array([1.0])), compute)
    second = cache.cached("k", (1, "a", np.array([1.0])), compute)
    assert len(calls) == 1 and first is second                        # the SAME object, not a rebuilt equal one


def test_recomputes_when_any_input_changes(session):
    calls = []
    compute = lambda: calls.append(1) or len(calls)
    assert cache.cached("k", (1,), compute) == 1
    assert cache.cached("k", (2,), compute) == 2
    assert cache.cached("k", (np.array([1.0, 2.0]),), compute) == 3
    assert cache.cached("k", (np.array([1.0, 2.5]),), compute) == 4
    assert cache.cached("k", (np.array([1.0, 2.5]),), compute) == 4          # unchanged again: reused


def test_one_entry_per_key_so_going_back_to_old_inputs_recomputes(session):
    calls = []
    compute = lambda: calls.append(1) or len(calls)
    cache.cached("k", ("A",), compute)
    cache.cached("k", ("B",), compute)
    assert cache.cached("k", ("A",), compute) == 3                    # the A entry was replaced by B


def test_different_keys_do_not_interfere(session):
    assert cache.cached("one", (1,), lambda: "from one") == "from one"
    assert cache.cached("two", (1,), lambda: "from two") == "from two"
    assert cache.cached("one", (1,), lambda: "recomputed?") == "from one"


def test_unfingerprintable_parts_fall_back_to_computing_every_time(session):
    calls = []
    compute = lambda: calls.append(1) or len(calls)
    assert cache.cached("k", (lambda: 1,), compute) == 1
    assert cache.cached("k", (lambda: 1,), compute) == 2             # correct, merely not cached
    assert "_ui_cache" not in session


def test_a_failed_computation_leaves_nothing_behind(session):
    def boom():
        raise RuntimeError("builder failed")
    with pytest.raises(RuntimeError, match="builder failed"):
        cache.cached("k", (1,), boom)
    assert cache.cached("k", (1,), lambda: "ok") == "ok"


def test_the_oldest_entries_are_evicted_and_use_keeps_an_entry_alive(session, monkeypatch):
    monkeypatch.setattr(cache, "MAX_ENTRIES", 3)
    for name in "abc":
        cache.cached(name, (name,), lambda n=name: n)
    cache.cached("a", ("a",), lambda: "recomputed")                  # touching a makes it the most recent
    cache.cached("d", ("d",), lambda: "d")                           # over the limit: the oldest untouched (b) goes
    assert set(session["_ui_cache"]) == {"c", "a", "d"}
    assert cache.cached("a", ("a",), lambda: "recomputed") == "a"    # survived, original value
    assert cache.cached("b", ("b",), lambda: "b again") == "b again"


# ------------------------------------------------------------------ model signature

def test_model_signature_is_stable_and_follows_what_a_derived_result_depends_on(session):
    assert cache.model_signature(decay_model()) == cache.model_signature(decay_model())
    base = cache.model_signature(decay_model())
    assert cache.model_signature(decay_model(ics=(("N(0)", 50.0),))) != base                      # initial value
    assert cache.model_signature(decay_model(ics=())) != base                                      # no initial condition
    other_rate = build_model({
        "problem_domain": "physics", "problem_type": "ode", "independent_variable": "t",
        "variables": [{"symbol": "k", "meaning": "k", "known_value": 0.9, "unit": None, "is_function": False},
                      {"symbol": "N", "meaning": "N", "known_value": None, "unit": None, "is_function": True}],
        "equations": [{"name": "decay", "kind": "ode", "expression": "Eq(Derivative(N(t), t), -k*N(t))", "derivation": ""}],
        "solve_for": ["N"], "assumptions": [], "initial_conditions": [{"expression": "N(0)", "value": 100.0}]})
    assert cache.model_signature(other_rate) != base                                               # a parameter value
    other_equation = build_model({
        "problem_domain": "physics", "problem_type": "ode", "independent_variable": "t",
        "variables": [{"symbol": "k", "meaning": "k", "known_value": 0.5, "unit": None, "is_function": False},
                      {"symbol": "N", "meaning": "N", "known_value": None, "unit": None, "is_function": True}],
        "equations": [{"name": "decay", "kind": "ode", "expression": "Eq(Derivative(N(t), t), -2*k*N(t))", "derivation": ""}],
        "solve_for": ["N"], "assumptions": [], "initial_conditions": [{"expression": "N(0)", "value": 100.0}]})
    assert cache.model_signature(other_equation) != base                                           # the equation itself


def test_cached_by_model_recomputes_for_a_different_model_or_extra_input(session):
    calls = []
    compute = lambda: calls.append(1) or len(calls)
    assert cache.cached_by_model("solve", decay_model(), compute) == 1
    assert cache.cached_by_model("solve", decay_model(), compute) == 1                    # an equal model, rebuilt: reused
    assert cache.cached_by_model("solve", decay_model(ics=(("N(0)", 7.0),)), compute) == 2
    assert cache.cached_by_model("solve", decay_model(ics=(("N(0)", 7.0),)), compute, "symbolic") == 3
