import sympy as sp

from modules.equivalence import check_equivalence
from modules.proof import build_proof


# ---------------------------------------------------------------- build_proof


def test_proof_for_trig_identity_ends_at_zero():
    result = check_equivalence("sin(x)**2 + cos(x)**2", "1")
    proof = build_proof(result)
    assert proof is not None
    assert proof[0][0] == "Start from the difference of the two expressions"
    # last step's resulting expression should literally be "0"
    assert proof[-1][1] == "0"


def test_proof_for_trig_identity_uses_trig_technique():
    result = check_equivalence("sin(x)**2 + cos(x)**2", "1")
    proof = build_proof(result)
    technique_names = [name for name, _ in proof]
    assert any("trig" in name.lower() for name in technique_names)


def test_proof_for_binomial_expansion_uses_expand():
    result = check_equivalence("(x+1)**2", "x**2 + 2*x + 1")
    proof = build_proof(result)
    technique_names = [name for name, _ in proof]
    assert "Expand" in technique_names
    assert proof[-1][1] == "0"


def test_proof_for_factored_quadratic():
    result = check_equivalence("x**2 - 1", "(x-1)*(x+1)")
    proof = build_proof(result)
    assert proof is not None
    assert proof[-1][1] == "0"


def test_proof_none_for_non_equivalent_expressions():
    result = check_equivalence("x**2", "x**3")
    assert build_proof(result) is None


def test_proof_none_for_numeric_sampling_only_equivalence():
    """equivalence.py's numeric-sampling fallback is evidence, not
    proof (see its own docstring) -- there should be no fabricated
    'proof' for a case SymPy couldn't symbolically confirm."""
    result = check_equivalence("sqrt(x**2)", "x")
    if result.method == "numeric sampling":
        assert build_proof(result) is None


def test_proof_never_pads_with_no_op_steps():
    """Every step in a returned proof must represent an actual
    structural change -- no step should repeat the previous expression."""
    result = check_equivalence("(x+1)**2", "x**2 + 2*x + 1")
    proof = build_proof(result)
    seen_latex = [latex for _, latex in proof]
    assert len(seen_latex) == len(set(seen_latex))  # no duplicate consecutive states


def test_proof_starts_from_raw_unsimplified_difference():
    """Regression test: proof.py must use raw_difference, not
    difference_simplified -- the simplified field is ALREADY fully
    reduced by equivalence.py itself, which would make every proof
    trivially one step long with nothing to show."""
    result = check_equivalence("(x+1)**2", "x**2 + 2*x + 1")
    proof = build_proof(result)
    # a genuine multi-step proof should exist: starting point + at
    # least one real transformation + reaching zero
    assert len(proof) >= 2


def test_equivalence_result_carries_raw_difference_field():
    result = check_equivalence("(x+1)**2", "x**2 + 2*x + 1")
    assert result.raw_difference is not None
    x = sp.Symbol("x")
    # the raw difference should NOT already be simplified to 0
    assert result.raw_difference != 0


# ---------------------------------------------------------------- build_recurrence_induction_proof
from modules.proof import build_recurrence_induction_proof


def test_induction_proof_linear_recurrence_valid():
    n = sp.Symbol("n", integer=True)
    a = sp.Function("a")
    eq = sp.Eq(a(n + 1), a(n) + 1)  # a(n) = n + 1, given a(0) = 1
    result = build_recurrence_induction_proof(eq, "a", n + 1, n, {0: 1})
    assert result.valid
    assert result.error is None
    assert len(result.steps) == 2  # one base case + one inductive step
    assert all(s.verified for s in result.steps)


def test_induction_proof_fibonacci_binet_formula_valid():
    """A genuinely nontrivial case: Binet's closed form for Fibonacci
    needs TWO base cases (second-order recurrence) and involves
    irrational numbers (sqrt(5)) in the inductive step's algebra."""
    n = sp.Symbol("n", integer=True)
    a = sp.Function("a")
    phi = (1 + sp.sqrt(5)) / 2
    psi = (1 - sp.sqrt(5)) / 2
    closed_form = (phi ** n - psi ** n) / sp.sqrt(5)
    eq = sp.Eq(a(n + 2), a(n + 1) + a(n))
    result = build_recurrence_induction_proof(eq, "a", closed_form, n, {0: 0, 1: 1})
    assert result.valid
    assert len(result.steps) == 3  # two base cases + one inductive step


def test_induction_proof_catches_wrong_base_case():
    n = sp.Symbol("n", integer=True)
    a = sp.Function("a")
    eq = sp.Eq(a(n + 1), a(n) + 1)
    result = build_recurrence_induction_proof(eq, "a", n + 1, n, {0: 5})  # should be 1, not 5
    assert not result.valid
    assert not result.steps[0].verified
    assert result.steps[-1].verified  # inductive step itself is still fine on its own


def test_induction_proof_catches_wrong_closed_form():
    n = sp.Symbol("n", integer=True)
    a = sp.Function("a")
    eq = sp.Eq(a(n + 1), a(n) + 1)
    result = build_recurrence_induction_proof(eq, "a", n ** 2, n, {0: 0})
    assert not result.valid
    assert not result.steps[-1].verified


def test_induction_proof_requires_at_least_one_base_case():
    n = sp.Symbol("n", integer=True)
    a = sp.Function("a")
    eq = sp.Eq(a(n + 1), a(n) + 1)
    result = build_recurrence_induction_proof(eq, "a", n + 1, n, {})
    assert result.error is not None


# ---------------------------------------------------------------- defensive branches in build_proof
import pytest

import modules.proof as proofmod
from modules.equivalence import EquivalenceResult
from modules.timeout_utils import ComputationTimeoutError

_x = sp.Symbol("x")


def _confirmed(raw_difference):
    return EquivalenceResult(True, "symbolic", sp.Integer(0), "ok", raw_difference=raw_difference)


def test_proof_none_when_raw_difference_missing():
    # a confirmed symbolic equivalence that somehow carries no raw difference
    # (e.g. a result built by older code) has nothing to walk through
    assert build_proof(_confirmed(None)) is None


def test_proof_skips_a_step_that_times_out_and_keeps_going(monkeypatch):
    real = proofmod.run_with_timeout

    def fake(func, *args, **kwargs):
        if kwargs.get("label") == "proof step: Expand":
            raise ComputationTimeoutError(0.1, kwargs["label"])
        return real(func, *args, **kwargs)

    monkeypatch.setattr(proofmod, "run_with_timeout", fake)
    steps = build_proof(_confirmed((_x + 1) ** 2 - (_x ** 2 + 2 * _x + 1)))
    names = [n for n, _ in steps]
    assert "Expand" not in names                 # the timed-out pass is skipped, not faked
    assert steps[-1][1] == "0"                   # ...but a later pass still reaches zero
    assert len(steps) >= 2


def test_proof_skips_a_step_that_raises_and_keeps_going(monkeypatch):
    real = proofmod.run_with_timeout

    def fake(func, *args, **kwargs):
        if kwargs.get("label") == "proof step: Expand":
            raise RuntimeError("sympy internal error")
        return real(func, *args, **kwargs)

    monkeypatch.setattr(proofmod, "run_with_timeout", fake)
    steps = build_proof(_confirmed((_x + 1) ** 2 - (_x ** 2 + 2 * _x + 1)))
    assert "Expand" not in [n for n, _ in steps]
    assert steps[-1][1] == "0"


def test_proof_is_honest_when_named_steps_never_reach_zero(monkeypatch):
    # every pass in the chain is a no-op, so the expression never reduces to 0
    # -- the proof must say so explicitly instead of claiming it ended at zero
    monkeypatch.setattr(proofmod, "_STEPS", [("Identity pass", lambda e: e)])
    steps = build_proof(_confirmed(_x + 1))
    assert len(steps) == 2                        # start + the honesty note; no padded no-op step
    assert steps[0][0].startswith("Start from the difference")
    assert "didn't fully reduce" in steps[-1][0]
    assert steps[-1][1] == sp.latex(_x + 1)


def test_proof_where_raw_difference_is_already_zero_has_only_the_start_step():
    steps = build_proof(_confirmed(sp.Integer(0)))
    # current == 0 from the start: every pass is a no-op (skipped), and since
    # current IS zero no "didn't fully reduce" note is appended
    assert [n for n, _ in steps] == ["Start from the difference of the two expressions"]


# ---------------------------------------------------------------- induction proof: failure handling

def test_induction_base_case_that_cannot_be_evaluated_is_marked_unverified():
    n = sp.Symbol("n", integer=True)
    k = sp.Symbol("k")
    a = sp.Function("a")
    eq = sp.Eq(a(n + 1), a(n) + 1)
    # a closed form with a stray free symbol (n + k) doesn't reduce to a number
    # at n=0, so complex() raises TypeError -- that must become an unverified
    # step with an explanatory message, not an uncaught crash
    result = build_recurrence_induction_proof(eq, "a", n + k, n, {0: 1})
    assert not result.valid
    assert result.steps[0].verified is False
    assert "could not evaluate" in result.steps[0].detail
    assert "does NOT match" in result.steps[0].detail


def test_induction_base_case_with_divergent_closed_form_is_unverified():
    n = sp.Symbol("n", integer=True)
    a = sp.Function("a")
    eq = sp.Eq(a(n + 1), a(n) + 1)
    # 1/n at n=0 is zoo: evaluates without raising, but can't equal the base value
    result = build_recurrence_induction_proof(eq, "a", 1 / n, n, {0: 1})
    assert not result.valid
    assert result.steps[0].verified is False
    assert "zoo" in result.steps[0].detail


def test_induction_inductive_step_exception_is_reported_not_raised(monkeypatch):
    import modules.recurrence_utils as ru

    def boom(*a, **k):
        raise RuntimeError("simplify blew up")

    monkeypatch.setattr(ru, "verify_recurrence_solution", boom)
    n = sp.Symbol("n", integer=True)
    a = sp.Function("a")
    eq = sp.Eq(a(n + 1), a(n) + 1)
    result = build_recurrence_induction_proof(eq, "a", n + 1, n, {0: 1})
    assert not result.valid
    assert result.steps[0].verified is True       # base case still stands on its own
    assert result.steps[-1].verified is False
    assert "could not verify" in result.steps[-1].detail
    assert "does NOT go through" in result.conclusion


def test_induction_conclusion_names_smallest_base_index_on_success():
    n = sp.Symbol("n", integer=True)
    a = sp.Function("a")
    eq = sp.Eq(a(n + 2), a(n + 1) + a(n))
    phi, psi = (1 + sp.sqrt(5)) / 2, (1 - sp.sqrt(5)) / 2
    result = build_recurrence_induction_proof(eq, "a", (phi ** n - psi ** n) / sp.sqrt(5), n, {1: 1, 0: 0})
    assert result.valid
    assert "at or above 0" in result.conclusion   # min of {0, 1}, regardless of dict order
