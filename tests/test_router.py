"""The "Ask anything" router: where requests go, why, what gets prefilled, and what it does when unsure."""
import pytest

from modules import router as rt
from modules.command_palette import MODE_LABELS


def modes(q):
    return rt.route(q).mode


# ------------------------------------------------------------------------------------------ where things go

@pytest.mark.parametrize("query, mode", [
    ("A car accelerates from 8 m/s to 20 m/s in 6 seconds. How far does it travel?", rt.WORD_PROBLEM),
    ("is sin(x)**2 + cos(x)**2 equivalent to 1?", rt.EQUIVALENCE),
    ("(x+1)**2 == x**2 + 2*x + 1", rt.EQUIVALENCE),
    ("show that tan(x) = sin(x)/cos(x)", rt.EQUIVALENCE),
    ("a triangle with sides 3, 4 and 5", rt.GEOMETRY),
    ("show me the shadow of a black hole", rt.SHOWCASE),
    ("animate the Ising model at the critical point", rt.SHOWCASE),
    ("render gravitational lensing", rt.SHOWCASE),
    ("solve the heat equation with Dirichlet boundaries", "🌡️ PDE solver"),
    ("tunneling through a potential barrier", rt.QUANTUM),
    ("compute the christoffel symbols of this metric", "🧮 Tensor calculus"),
    ("partition function of a two level system", rt.STATMECH),
    ("laplace transform of exp(-t)", "🔄 Transforms & series"),
])
def test_requests_go_to_the_right_mode(query, mode):
    assert modes(query) == mode


def test_a_table_of_numbers_with_a_dynamics_request_goes_to_discover_with_the_data_prefilled():
    q = "find the differential equation behind this data\nt,x\n0,1\n0.1,0.9\n0.2,0.82\n0.3,0.74\n0.4,0.67"
    r = rt.route(q)
    assert r.mode == rt.DISCOVER and r.sure
    assert r.prefill["discover_source"] == "Paste or upload data" and r.prefill["discover_paste"].startswith("t,x\n0,1")
    assert r.prefill["discover_paste"].count("\n") == 5


def test_a_table_with_a_fit_request_goes_to_curve_fitting():
    assert modes("fit a line to\n1 2\n2 4.1\n3 5.9\n4 8.2\n5 9.9") == rt.CURVE_FIT


def test_a_bare_table_is_ambiguous_between_discover_and_curve_fitting_and_says_so():
    r = rt.route("0,1\n1,2\n2,5\n3,10\n4,17")
    assert {r.mode, r.alternatives[0][0]} == {rt.DISCOVER, rt.CURVE_FIT} and not r.sure


# ------------------------------------------------------------------------------------------ prefill

def test_equivalence_prefills_both_expressions():
    assert rt.route("is sin(x)**2 + cos(x)**2 equivalent to 1?").prefill == {"equiv_expr1": "sin(x)**2 + cos(x)**2", "equiv_expr2": "1"}
    assert rt.route("(x+1)**2 == x**2 + 2*x + 1").prefill == {"equiv_expr1": "(x+1)**2", "equiv_expr2": "x**2 + 2*x + 1"}


def test_geometry_prefills_the_three_sides():
    assert rt.route("a triangle with sides 3, 4 and 5").prefill == {"tri_a": "3", "tri_b": "4", "tri_c": "5"}


def test_a_word_problem_is_prefilled_with_the_whole_text_and_flagged_as_needing_the_model():
    q = "A ball is thrown at 12 m/s at 30 degrees. What is its range?"
    r = rt.route(q)
    assert r.prefill == {"problem_text": q} and r.needs_language_model


# ------------------------------------------------------------------------------------------ confidence and reasons

def test_every_answer_says_why():
    for q in ["tunneling through a barrier", "is x == x", "show me a black hole shadow", "what is 5 + 5 in 3 steps please"]:
        assert rt.route(q).reason


def test_a_single_weak_keyword_is_not_sure_but_two_signals_are():
    assert not rt.route("curvature").sure
    assert rt.route("black hole geodesic and the schwarzschild precession").sure


def test_confidence_is_a_share_of_the_total_and_stays_in_range():
    for q in ["heat equation", "ising", "orbit", "hamiltonian", "a b c"]:
        assert 0.0 <= rt.route(q).confidence < 1.0


def test_an_empty_or_meaningless_request_falls_back_gently():
    assert rt.route("   ").confidence == 0.0
    r = rt.route("zxqv")
    assert r.mode in MODE_LABELS and not r.sure


def test_the_router_is_deterministic():
    q = "a ball falls 20 m in how many seconds"
    assert rt.route(q) == rt.route(q)


def test_every_mode_the_router_can_return_exists():
    for q in ["is a == b", "triangle sides 3 4 5", "show orbit", "fit\n1,2\n2,3\n3,4\n4,5", "x"]:
        assert rt.route(q).mode in MODE_LABELS
    assert {rt.WORD_PROBLEM, rt.EQUIVALENCE, rt.CURVE_FIT, rt.DISCOVER, rt.SHOWCASE, rt.GEOMETRY} <= set(MODE_LABELS)


# ------------------------------------------------------------------------------------------ the language-model tie-break

class _Model:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def chat(self, system, user, temperature=0.0, **kw):
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_the_model_can_break_a_tie_and_its_reason_is_kept():
    first = rt.route("curvature")
    out = rt.route_with_llm("curvature", _Model('Sure: {"mode": "🧮 Tensor calculus", "reason": "it asks about curvature tensors"}'), first)
    assert out.mode == "🧮 Tensor calculus" and out.sure and "curvature tensors" in out.reason


@pytest.mark.parametrize("reply", ['{"mode": "🧪 Invented mode", "reason": "x"}', "no json at all", '{"reason": "missing mode"}', RuntimeError("down")])
def test_a_bad_model_answer_leaves_the_heuristic_answer_unchanged(reply):
    first = rt.route("curvature")
    assert rt.route_with_llm("curvature", _Model(reply), first) == first


def test_choosing_the_word_problem_solver_prefills_the_text():
    out = rt.route_with_llm("my weird question", _Model(f'{{"mode": "{rt.WORD_PROBLEM}", "reason": "general"}}'))
    assert out.mode == rt.WORD_PROBLEM and out.prefill == {"problem_text": "my weird question"}


def test_quantum_questions_that_name_no_quantum_word_still_find_quantum():
    assert modes("energy levels of a particle in a box of width 1 nm") == rt.QUANTUM


def test_a_sentence_with_no_signal_does_not_get_a_fuzzy_name_match():
    r = rt.route("plot the function sin of x please")
    assert r.reason != "closest match by name" and not r.sure


@pytest.mark.parametrize("query", ["have I solved this before? a car accelerates from 8 m/s to 20 m/s in 6 s",
                                   "did I already solve something like this", "is there a similar problem in my history"])
def test_questions_about_the_history_go_to_recall_with_the_text_prefilled(query):
    r = rt.route(query)
    assert r.mode == rt.RECALL and r.prefill == {"recall_query": query}


@pytest.mark.parametrize("query, mode", [
    ("make a worksheet of 10 practice problems with an answer key", "🧑‍🏫 Problem sets"),
    ("posterior of my nonlinear fit with a corner plot", "🎲 Posterior sampling (MCMC)"),
    ("what is the credible interval of the decay rate", "🎲 Posterior sampling (MCMC)"),
    ("lm studio is not working, diagnose it", "🩺 Health check"),
])
def test_the_newer_modes_are_reachable_by_asking(query, mode):
    assert mode in MODE_LABELS and modes(query) == mode


def test_a_quiz_request_at_least_offers_problem_sets_among_its_choices():
    r = rt.route("generate a quiz on kinematics")
    assert "🧑‍🏫 Problem sets" in [r.mode] + [m for m, _ in r.alternatives]
