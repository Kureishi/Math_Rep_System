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
