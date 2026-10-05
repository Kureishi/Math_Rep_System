import time
import pytest

from modules.timeout_utils import run_with_timeout, ComputationTimeoutError


# ---------------------------------------------------------------- core mechanism


def test_fast_function_returns_normally():
    assert run_with_timeout(lambda: 2 + 2, timeout=1.0) == 4


def test_slow_function_raises_computation_timeout_error():
    def slow():
        time.sleep(2)
        return "done"

    with pytest.raises(ComputationTimeoutError):
        run_with_timeout(slow, timeout=0.2)


def test_timeout_returns_control_promptly_not_after_full_duration():
    def slow():
        time.sleep(2)

    t0 = time.time()
    with pytest.raises(ComputationTimeoutError):
        run_with_timeout(slow, timeout=0.2)
    elapsed = time.time() - t0
    assert elapsed < 1.0  # should return around 0.2s, not wait for the full 2s sleep


def test_non_timeout_exception_propagates_unchanged():
    def raises():
        raise ValueError("a real error")

    with pytest.raises(ValueError, match="a real error"):
        run_with_timeout(raises, timeout=1.0)


def test_args_and_kwargs_passed_through():
    def add(a, b, c=0):
        return a + b + c

    assert run_with_timeout(add, 1, 2, timeout=1.0, c=3) == 6


def test_uses_default_timeout_from_settings(monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "computation_timeout_seconds", 0.2)

    def slow():
        time.sleep(2)

    with pytest.raises(ComputationTimeoutError):
        run_with_timeout(slow)  # no explicit timeout -- should use the patched setting


def test_error_message_includes_seconds_and_label():
    def slow():
        time.sleep(2)

    try:
        run_with_timeout(slow, timeout=0.2, label="my computation")
        assert False, "expected ComputationTimeoutError"
    except ComputationTimeoutError as e:
        assert "0.2" in str(e)
        assert "my computation" in str(e)


def test_error_message_omits_label_when_not_given():
    def slow():
        time.sleep(2)

    try:
        run_with_timeout(slow, timeout=0.2)
        assert False, "expected ComputationTimeoutError"
    except ComputationTimeoutError as e:
        assert "(" not in str(e)


# ---------------------------------------------------------------- wired-in call sites


def test_verifier_solve_reports_timeout_as_explicit_check(monkeypatch):
    import modules.verifier as verifier_module
    from modules.equation_engine import build_model

    def _raise_timeout(model):
        raise ComputationTimeoutError(10.0, label="algebraic solve")
    monkeypatch.setattr(verifier_module, "_solve_sympy", _raise_timeout)

    class FakeClient:
        def chat(self, **kwargs):
            return "{}"

    model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "time", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })
    report = verifier_module.verify(model, FakeClient(), "x")
    timeout_checks = [c for c in report.checks if c.label == "Symbolic solve"]
    assert len(timeout_checks) == 1
    assert timeout_checks[0].passed is False
    assert "timed out" in timeout_checks[0].detail.lower()
    assert report.passed is False
    assert report.sympy_numeric_answers == {}


def test_matrix_analyze_partial_degradation_on_timeout(monkeypatch):
    """Regression test: a timeout on eigenvalues/linsolve must not lose
    the still-fast rank-based classification -- genuine partial
    degradation, not all-or-nothing failure."""
    import sympy as sp
    from config import settings
    from modules.matrix_utils import analyze_linear_system

    monkeypatch.setattr(settings, "computation_timeout_seconds", 0.0)
    A = sp.Matrix([[2, 3], [1, -1]])
    b = sp.Matrix([8, 1])
    result = analyze_linear_system(A, b, ["x", "y"])
    assert result.rank_A == 2
    assert result.consistent is True
    assert result.unique is True
    assert "Unique solution" in result.classification


def test_uncertainty_solve_symbolic_never_raises_on_timeout(monkeypatch):
    from config import settings
    from modules.equation_engine import build_model
    from modules.uncertainty import solve_symbolic_for_target

    monkeypatch.setattr(settings, "computation_timeout_seconds", 0.0)
    model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "time", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })
    # with an essentially-zero timeout, this should never raise, whether
    # or not the solve happened to squeak in under the wire
    solve_symbolic_for_target(model, "a")


def test_equivalence_check_degrades_to_undetermined_on_timeout(monkeypatch):
    from config import settings
    from modules.equivalence import check_equivalence

    monkeypatch.setattr(settings, "computation_timeout_seconds", 0.0)
    result = check_equivalence("sin(x)**2 + cos(x)**2", "1")
    assert result.equivalent is None
    assert result.method == "undetermined"
    assert "timed out" in result.detail.lower()
    assert result.error is None  # this is a timeout, not a parse error


def test_proof_build_returns_none_when_equivalence_check_timed_out(monkeypatch):
    from config import settings
    from modules.equivalence import check_equivalence
    from modules.proof import build_proof

    monkeypatch.setattr(settings, "computation_timeout_seconds", 0.0)
    result = check_equivalence("sin(x)**2 + cos(x)**2", "1")
    assert build_proof(result) is None


# ---------------------------------------------------------------- one thread per call; bounded, explicit pile-up
import subprocess
import sys
import textwrap
import threading

import modules.timeout_utils as tu
from modules.timeout_utils import ComputationBusyError, abandoned_computations


def _wait_until_nothing_is_abandoned(limit: float = 6.0) -> None:
    deadline = time.time() + limit
    while abandoned_computations() and time.time() < deadline:
        time.sleep(0.02)


@pytest.fixture
def hang():
    """A computation that stays 'stuck' until released -- a stand-in for a SymPy call that never returns."""
    _wait_until_nothing_is_abandoned()           # other tests' sleeping threads must not count against this one
    release = threading.Event()

    def stuck():
        release.wait(30)

    yield stuck
    release.set()
    _wait_until_nothing_is_abandoned()


def test_hung_computations_do_not_make_later_ones_time_out(hang):
    """The failure this design replaced: four hung computations filled a shared pool of four workers, and from then
    on a trivial `1 + 1` was reported as 'timed out' too, until the hung work happened to finish."""
    for i in range(5):                                           # more than the old pool's four workers
        with pytest.raises(ComputationTimeoutError):
            run_with_timeout(hang, timeout=0.05, label=f"stuck-{i}")
    assert abandoned_computations() == 5

    started = time.time()
    assert run_with_timeout(lambda: 1 + 1, timeout=1.0) == 2
    assert time.time() - started < 0.5                           # answered at once, not after the full timeout


def test_abandoned_computations_are_counted_and_forgotten_once_they_finish(hang):
    assert abandoned_computations() == 0
    with pytest.raises(ComputationTimeoutError):
        run_with_timeout(hang, timeout=0.05)
    assert abandoned_computations() == 1


def test_count_drops_back_when_the_abandoned_work_ends():
    release = threading.Event()
    _wait_until_nothing_is_abandoned()
    with pytest.raises(ComputationTimeoutError):
        run_with_timeout(lambda: release.wait(30), timeout=0.05)
    assert abandoned_computations() == 1
    release.set()
    _wait_until_nothing_is_abandoned()
    assert abandoned_computations() == 0


def test_new_work_is_refused_at_once_once_too_many_are_still_running(hang, monkeypatch):
    monkeypatch.setattr(tu, "MAX_ABANDONED", 3)
    for _ in range(3):
        with pytest.raises(ComputationTimeoutError):
            run_with_timeout(hang, timeout=0.05)

    started = time.time()
    with pytest.raises(ComputationBusyError) as refused:
        run_with_timeout(lambda: 1 + 1, timeout=5.0, label="innocent bystander")
    assert time.time() - started < 0.5                           # refused immediately, not after waiting out its timeout
    message = str(refused.value)
    assert "3 earlier computations" in message and "innocent bystander" in message and "restart" in message
    assert isinstance(refused.value, ComputationTimeoutError)    # so every existing `except ComputationTimeoutError` copes
    assert refused.value.abandoned == 3 and refused.value.label == "innocent bystander"


def test_work_is_accepted_again_as_soon_as_the_backlog_clears(monkeypatch):
    monkeypatch.setattr(tu, "MAX_ABANDONED", 2)
    _wait_until_nothing_is_abandoned()
    release = threading.Event()
    for _ in range(2):
        with pytest.raises(ComputationTimeoutError):
            run_with_timeout(lambda: release.wait(30), timeout=0.05)
    with pytest.raises(ComputationBusyError):
        run_with_timeout(lambda: 1, timeout=1.0)
    release.set()
    _wait_until_nothing_is_abandoned()
    assert run_with_timeout(lambda: 1, timeout=1.0) == 1


def test_a_refused_call_is_logged(hang, monkeypatch, caplog):
    monkeypatch.setattr(tu, "MAX_ABANDONED", 1)
    with pytest.raises(ComputationTimeoutError):
        run_with_timeout(hang, timeout=0.05)
    seen = []
    monkeypatch.setattr(tu.logger, "warning", lambda msg, *a: seen.append(msg % a))
    with pytest.raises(ComputationBusyError):
        run_with_timeout(lambda: 1, timeout=1.0, label="queued")
    assert any("Refusing to start (queued)" in m and "1 timed-out" in m for m in seen)


def test_the_functions_own_timeout_error_is_not_mistaken_for_a_computation_timeout():
    """A socket timeout, say, raised BY the computation is a real error from the function, not 'this took too long'.
    (concurrent.futures.TimeoutError is the builtin TimeoutError on Python 3.11+, so the old
    `except futures.TimeoutError` could not tell them apart.)"""
    def raises_builtin_timeout():
        raise TimeoutError("the socket timed out")

    with pytest.raises(TimeoutError, match="the socket timed out") as caught:
        run_with_timeout(raises_builtin_timeout, timeout=5.0)
    assert not isinstance(caught.value, ComputationTimeoutError)


def test_keyboard_interrupt_style_base_exceptions_reach_the_caller():
    class Stop(BaseException):
        pass

    def stops():
        raise Stop("halt")

    with pytest.raises(Stop, match="halt"):
        run_with_timeout(stops, timeout=5.0)


def test_calls_from_many_threads_run_side_by_side_not_four_at_a_time():
    """With the old shared pool of four workers, twelve 0.3 s calls took at least three rounds (0.9 s)."""
    results, errors = [], []

    def caller(i):
        try:
            results.append(run_with_timeout(lambda: (time.sleep(0.3), i)[1], timeout=5.0))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    callers = [threading.Thread(target=caller, args=(i,)) for i in range(12)]
    started = time.time()
    for t in callers:
        t.start()
    for t in callers:
        t.join()
    elapsed = time.time() - started
    assert not errors and sorted(results) == list(range(12))
    assert elapsed < 0.8


def test_worker_threads_are_daemons_named_for_the_app():
    seen = {}

    def peek():
        t = threading.current_thread()
        seen["daemon"], seen["name"] = t.daemon, t.name

    run_with_timeout(peek, timeout=5.0)
    assert seen == {"daemon": True, "name": "mrs-computation"}


def test_a_hung_computation_does_not_keep_the_process_from_exiting():
    """A ThreadPoolExecutor's workers are joined at interpreter shutdown, so one stuck SymPy call would have made
    the app impossible to quit. Daemon threads are not."""
    script = textwrap.dedent("""
        import sys, time
        sys.path.insert(0, ".")
        from modules.timeout_utils import run_with_timeout, ComputationTimeoutError
        try:
            run_with_timeout(lambda: time.sleep(60), timeout=0.1)
        except ComputationTimeoutError:
            print("timed out, now exiting", flush=True)
    """)
    started = time.time()
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0 and "timed out, now exiting" in proc.stdout
    assert time.time() - started < 20                            # not the 60 s the stuck thread would hold it for


def test_zero_timeout_still_returns_a_result_if_the_function_was_instant():
    # the suite relies on timeout=0 meaning "it may or may not squeak in": neither outcome may crash
    try:
        assert run_with_timeout(lambda: 7, timeout=0.0) == 7
    except ComputationTimeoutError:
        pass
