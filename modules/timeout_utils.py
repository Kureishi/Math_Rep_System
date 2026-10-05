"""
Computation timeouts: wraps a callable so it can't hang a Streamlit
session indefinitely. SymPy operations (solve, dsolve, rsolve, eigenvalue
extraction, equivalence checking, the sequential simplification passes
in proof.py, etc.) can, on pathological input, run for a very long time
or effectively never return -- and this app has grown enough SymPy-heavy
features (matrix systems, ODEs, optimization, curve fitting, an 8-pass
proof-mode simplification chain) that the risk surface for this is real,
not hypothetical.

Built on a plain worker thread rather than signal.alarm: this app targets
Windows as a first-class environment (conda + PowerShell, per its own setup
notes), and signal.alarm/SIGALRM simply doesn't exist there -- a
signal-based timeout would silently do nothing on the platform it's most
meant to protect.

The real tradeoff, stated plainly rather than glossed over: Python cannot
forcibly kill a running thread. If a computation genuinely hangs,
run_with_timeout() returns control to the caller (so the UI stops waiting
and shows a timeout message) but the abandoned worker thread keeps running in
the background until it naturally finishes, consuming CPU meanwhile. This
protects the session from LOOKING hung -- it doesn't reclaim the CPU from a
truly runaway computation. A multiprocessing-based approach could forcibly
terminate the work, but at the cost of process-spawn overhead on every single
call (including the overwhelming majority that finish in milliseconds), which
isn't the right tradeoff for an interactive app where most solves are instant.

Why one thread per call and not a shared pool: this used to be a shared pool
of four workers. A computation that timed out kept its worker, so after four
hangs the pool was full, every later call sat in the queue and was reported as
timed out -- including a trivial `1 + 1`. The app looked permanently broken
until the hung work happened to finish. A thread per call (about 70 microseconds
more than a pool, nothing next to a SymPy call) means an abandoned computation
costs only itself. The cost it WOULD otherwise have had -- unbounded pile-up --
is bounded explicitly instead: once MAX_ABANDONED timed-out computations are
still running, new calls are refused immediately with ComputationBusyError and a
message saying why, rather than quietly adding more CPU load. The workers are
daemon threads, so a hung one can never keep the process from exiting (a
ThreadPoolExecutor's workers are joined at interpreter shutdown, and would).
"""
import threading
from concurrent.futures import Future, wait
from typing import Callable, TypeVar

from config import settings
from modules.app_logging import logger

T = TypeVar("T")

# How many timed-out computations may still be running in the background before
# new ones are refused. Each one is a thread that may be burning a CPU core, so
# this is the bound on the cost of abandoning work.
MAX_ABANDONED = 8

_abandoned: set[threading.Thread] = set()
_abandoned_lock = threading.Lock()


def abandoned_computations() -> int:
    """How many computations that timed out are still running in the
    background (a thread cannot be killed, only abandoned)."""
    with _abandoned_lock:
        _abandoned.difference_update({t for t in _abandoned if not t.is_alive()})
        return len(_abandoned)


class ComputationTimeoutError(Exception):
    """Raised as a plain, catchable exception (not left as SymPy's own
    internals or a raw concurrent.futures.TimeoutError) when a wrapped
    computation exceeds its timeout, so callers can catch this
    specifically and report "timed out" as distinct from any other
    failure mode."""
    def __init__(self, seconds: float, label: str | None = None):
        self.seconds = seconds
        self.label = label
        msg = f"Computation timed out after {seconds:g}s"
        if label:
            msg += f" ({label})"
        super().__init__(msg)


class ComputationBusyError(ComputationTimeoutError):
    """Raised, immediately, instead of starting a new computation while
    MAX_ABANDONED earlier ones that timed out are still running. It IS a
    ComputationTimeoutError, so every existing `except ComputationTimeoutError`
    handles it, but its message says what is actually wrong."""
    def __init__(self, abandoned: int, label: str | None = None):
        Exception.__init__(
            self,
            f"Not started{f' ({label})' if label else ''}: {abandoned} earlier computations that timed out "
            "are still running in the background. Wait for them to finish, or restart the app to reclaim them.")
        self.seconds = 0.0
        self.label = label
        self.abandoned = abandoned


def run_with_timeout(func: Callable[..., T], *args, timeout: float | None = None,
                      label: str | None = None, **kwargs) -> T:
    """Runs func(*args, **kwargs) on its own worker thread and waits up to
    `timeout` seconds (settings.computation_timeout_seconds if not
    given) for it to finish. Returns the result on success; raises
    ComputationTimeoutError if it doesn't finish in time; re-raises
    whatever exception func itself raised, unchanged, if it fails for
    any other reason -- this function only adds a time bound, it
    doesn't change func's own error behavior (including when func itself
    raises the builtin TimeoutError, which is NOT mistaken for this
    function's own timeout)."""
    timeout = timeout if timeout is not None else settings.computation_timeout_seconds
    still_running = abandoned_computations()
    if still_running >= MAX_ABANDONED:
        logger.warning("Refusing to start%s: %d timed-out computations are still running",
                        f" ({label})" if label else "", still_running)
        raise ComputationBusyError(still_running, label)

    future: Future[T] = Future()

    def runner() -> None:
        try:
            future.set_result(func(*args, **kwargs))
        except BaseException as exc:  # noqa: BLE001 -- handed to the caller unchanged below
            future.set_exception(exc)

    worker = threading.Thread(target=runner, name="mrs-computation", daemon=True)
    worker.start()
    done, _ = wait([future], timeout=timeout)
    if done:
        return future.result()                      # the result, or func's own exception, unchanged
    with _abandoned_lock:
        _abandoned.add(worker)
    # Logged here -- the one gateway every timeout-protected symbolic
    # computation across the app (matrix_utils, ode_utils,
    # recurrence_utils, optimization_utils, equivalence.py, proof.py,
    # solver.py, uncertainty.py, verifier.py) already goes through --
    # so a pattern of "this keeps timing out" is visible in the log
    # regardless of which specific call site it came from, without
    # needing every one of those call sites to log it separately.
    logger.warning("Computation timed out after %gs%s", timeout, f" ({label})" if label else "")
    raise ComputationTimeoutError(timeout, label)
