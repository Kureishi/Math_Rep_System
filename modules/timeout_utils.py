"""
Computation timeouts: wraps a callable so it can't hang a Streamlit
session indefinitely. SymPy operations (solve, dsolve, rsolve, eigenvalue
extraction, equivalence checking, the sequential simplification passes
in proof.py, etc.) can, on pathological input, run for a very long time
or effectively never return -- and this app has grown enough SymPy-heavy
features (matrix systems, ODEs, optimization, curve fitting, an 8-pass
proof-mode simplification chain) that the risk surface for this is real,
not hypothetical.

Deliberately built on concurrent.futures.ThreadPoolExecutor rather than
signal.alarm: this app targets Windows as a first-class environment
(conda + PowerShell, per its own setup notes), and signal.alarm/SIGALRM
simply doesn't exist there -- a signal-based timeout would silently do
nothing on the platform it's most meant to protect. ThreadPoolExecutor
works identically on every platform since it doesn't touch any
platform-specific API.

The real tradeoff, stated plainly rather than glossed over: Python
cannot forcibly kill a running thread. If a computation genuinely hangs,
run_with_timeout() returns control to the caller (so the UI stops
waiting and shows a timeout message) but the orphaned worker thread
keeps running in the background until it naturally finishes, consuming
CPU meanwhile. This protects the session from LOOKING hung -- it
doesn't reclaim the CPU from a truly runaway computation. A
multiprocessing-based approach could forcibly terminate the work, but
at the cost of process-spawn overhead on every single call (including
the overwhelming majority that finish in milliseconds), which isn't the
right tradeoff for an interactive app where most solves are instant.
"""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeoutError
from typing import Callable, TypeVar

from config import settings

T = TypeVar("T")

# A small shared pool rather than spawning a new thread per call --
# max_workers caps how many timed-out-but-still-running computations can
# pile up in the background at once (a soft bound on the "orphaned
# thread" cost described above), without adding per-call thread-creation
# overhead for the common case of a fast, well-behaved computation.
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="mrs-computation")


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


def run_with_timeout(func: Callable[..., T], *args, timeout: float | None = None,
                      label: str | None = None, **kwargs) -> T:
    """Runs func(*args, **kwargs) on a worker thread and waits up to
    `timeout` seconds (settings.computation_timeout_seconds if not
    given) for it to finish. Returns the result on success; raises
    ComputationTimeoutError if it doesn't finish in time; re-raises
    whatever exception func itself raised, unchanged, if it fails for
    any other reason -- this function only adds a time bound, it
    doesn't change func's own error behavior."""
    timeout = timeout if timeout is not None else settings.computation_timeout_seconds
    future = _executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except _FutureTimeoutError:
        raise ComputationTimeoutError(timeout, label) from None
