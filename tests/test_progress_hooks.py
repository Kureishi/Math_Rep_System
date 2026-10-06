"""The progress hook (modules/progress.py) and the long computations that report through it."""
import numpy as np
import pytest

from modules.equation_engine import build_model
from modules.monte_carlo import EVAL_CHUNK, UncertainVariable, run_monte_carlo
from modules.progress import chunk_bounds, report, scaled
from modules.pde_utils import solve_heat_equation_2d_dirichlet, solve_pde_finite_difference_2d


# ------------------------------------------------------------------ the helpers

def test_report_with_no_hook_is_a_no_op():
    report(None, "anything", 0.5)          # must not raise


def test_report_clamps_the_fraction_into_zero_to_one():
    seen = []
    report(lambda label, f: seen.append((label, f)), "x", 1.7)
    report(lambda label, f: seen.append((label, f)), "y", -3)
    report(lambda label, f: seen.append((label, f)), "z", None)
    assert seen == [("x", 1.0), ("y", 0.0), ("z", None)]


@pytest.mark.parametrize("total,chunk,expected", [
    (0, 10, []), (-5, 10, []), (10, 10, [(0, 10)]), (25, 10, [(0, 10), (10, 20), (20, 25)]),
    (3, 0, [(0, 1), (1, 2), (2, 3)]),          # a non-positive chunk is treated as 1, never an infinite loop
])
def test_chunk_bounds(total, chunk, expected):
    assert chunk_bounds(total, chunk) == expected


def test_chunk_bounds_cover_the_range_exactly_once():
    covered = [i for s, e in chunk_bounds(1013, 100) for i in range(s, e)]
    assert covered == list(range(1013))


def test_scaled_maps_each_step_onto_its_slice_and_prefixes_the_label():
    seen = []
    hook = scaled(lambda label, f: seen.append((label, f)), index=1, count=4, prefix="v: ")
    hook("half way", 0.5)
    hook("unknown", None)
    assert seen[0] == ("v: half way", pytest.approx(0.375))     # (1 + 0.5) / 4
    assert seen[1] == ("v: unknown", None)


def test_scaled_of_nothing_is_nothing():
    assert scaled(None, 0, 3) is None


# ------------------------------------------------------------------ Monte Carlo

def _model():
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "time", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [{"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""}],
        "solve_for": ["a"], "assumptions": [],
    })


def test_a_progress_hook_does_not_change_the_numbers_for_a_given_seed():
    """Chunking only splits the EVALUATION; the draws are made once, up front, so a seed still means
    exactly the same samples -- reproducibility is the point of recording the seed."""
    uv = [UncertainVariable("v_f", 20.0, 1.0), UncertainVariable("t", 6.0, 0.2)]
    n = EVAL_CHUNK * 2 + 123
    plain = run_monte_carlo(_model(), "a", uv, n_samples=n, seed=7)
    hooked = run_monte_carlo(_model(), "a", uv, n_samples=n, seed=7, progress=lambda *_: None)
    assert hooked.samples == plain.samples
    assert (hooked.mean, hooked.std, hooked.p5, hooked.p95) == (plain.mean, plain.std, plain.p5, plain.p95)


def test_monte_carlo_reports_checkpoints_that_only_move_forward_and_finish_near_one():
    seen: list[tuple[str, float | None]] = []
    n = EVAL_CHUNK * 3
    run_monte_carlo(_model(), "a", [UncertainVariable("v_f", 20.0, 1.0)], n_samples=n, seed=1,
                    progress=lambda label, f: seen.append((label, f)))
    fractions = [f for _, f in seen if f is not None]
    assert fractions == sorted(fractions)
    assert fractions[-1] > 0.9
    evaluated = [label for label, _ in seen if label.startswith("Evaluated")]
    assert len(evaluated) == 3                        # one checkpoint per chunk
    assert any("symbolically" in label for label, _ in seen)


def test_a_hook_that_raises_stops_the_run_at_that_checkpoint():
    """This is the mechanism Stop relies on: an exception from the hook (Streamlit raises one inside its
    own calls when a rerun was requested) propagates out of the computation instead of being swallowed."""
    class Stop(BaseException):
        pass

    calls = []

    def hook(label, fraction):
        calls.append(label)
        if label.startswith("Evaluated"):
            raise Stop

    with pytest.raises(Stop):
        run_monte_carlo(_model(), "a", [UncertainVariable("v_f", 20.0, 1.0)],
                        n_samples=EVAL_CHUNK * 4, seed=1, progress=hook)
    assert sum(1 for c in calls if c.startswith("Evaluated")) == 1       # it did not carry on to the other chunks


def test_a_constant_target_with_a_hook_still_returns_cleanly():
    result = run_monte_carlo(_model(), "a", [UncertainVariable("v_i", 8.0, 0.0001)], n_samples=50, seed=2,
                             progress=lambda *_: None)
    assert result.mean is not None


# ------------------------------------------------------------------ PDE solvers

def test_finite_difference_solver_checkpoints_before_the_expensive_refined_solve():
    seen = []
    result = solve_pde_finite_difference_2d("0", "1", "x**2+y**2<=1", (-1.2, 1.2), (-1.2, 1.2), 21, 21,
                                            progress=lambda label, f: seen.append((label, f)))
    assert result.error is None
    labels = [label for label, _ in seen]
    assert labels[0].startswith("Solving on a 21 x 21")
    assert labels[1].startswith("Re-solving on a 41 x 41")        # reported BEFORE that solve starts
    fractions = [f for _, f in seen]
    assert fractions == sorted(fractions)


def test_finite_difference_solver_without_a_hook_is_unchanged():
    a = solve_pde_finite_difference_2d("0", "1", "x**2+y**2<=1", (-1.2, 1.2), (-1.2, 1.2), 21, 21)
    b = solve_pde_finite_difference_2d("0", "1", "x**2+y**2<=1", (-1.2, 1.2), (-1.2, 1.2), 21, 21,
                                       progress=lambda *_: None)
    assert np.allclose(np.nan_to_num(a.solution_grid), np.nan_to_num(b.solution_grid))


def test_heat_2d_reports_its_stages_in_the_calling_thread():
    import threading
    threads = set()
    seen = []

    def hook(label, fraction):
        threads.add(threading.current_thread())
        seen.append(label)

    result = solve_heat_equation_2d_dirichlet("sin(pi*x)*sin(pi*y)", "0", nx=11, ny=11, progress=hook)
    assert result.error is None
    assert threads == {threading.current_thread()}      # never from the timeout wrapper's worker thread
    assert any(label.startswith("Time-stepping") for label in seen)
