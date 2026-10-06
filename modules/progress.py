"""
A tiny, Streamlit-free progress hook shared by the long-running computations.

A computation that can take a while (Monte Carlo sampling, a two-pass PDE solve) accepts an optional
`progress` callable and calls it at its natural checkpoints: `progress("Sampling", 0.4)`. The UI passes
a callable that updates a status block; a test passes a list's `append`; everyone else passes nothing.

Why this is also what makes "Stop" work: Streamlit cancels a running script by raising an exception at
its NEXT `st.*` call once a rerun has been requested. A computation that never touches `st` between its
first and last line therefore cannot be interrupted at all. Every checkpoint that calls `progress` is a
place where the UI's callback talks to Streamlit, and so a place where a Stop (or any other widget
interaction) can take effect. That is why the checkpoints sit BETWEEN chunks of work and never inside a
single vectorised numpy call or a worker thread: calling `st.*` from the timeout wrapper's worker thread
has no script context, and splitting a vectorised call would change nothing about its result.

The hook never raises on its own; a computation must not depend on it being present.
"""
from collections.abc import Callable

# (what is happening now, fraction of the whole run done in 0..1 -- or None when unknown)
ProgressFn = Callable[[str, float | None], None]


def report(progress: ProgressFn | None, label: str, fraction: float | None = None) -> None:
    """Call `progress` if there is one, with `fraction` clamped to [0, 1]."""
    if progress is None:
        return
    if fraction is not None:
        fraction = min(1.0, max(0.0, float(fraction)))
    progress(label, fraction)


def chunk_bounds(total: int, chunk: int) -> list[tuple[int, int]]:
    """[(start, stop), ...] covering range(total) in pieces of at most `chunk`. Empty for total <= 0."""
    if total <= 0:
        return []
    chunk = max(1, int(chunk))
    return [(s, min(s + chunk, total)) for s in range(0, total, chunk)]


def scaled(progress: ProgressFn | None, index: int, count: int, prefix: str = "") -> ProgressFn | None:
    """A hook for step `index` (0-based) of `count` equal steps: it maps the step's own 0..1 fraction onto
    its slice of the whole run and prefixes the label -- so a loop over several targets can pass each one
    its own `progress` and the overall bar still moves from 0 to 1. None stays None."""
    if progress is None or count <= 0:
        return None

    def hook(label: str, fraction: float | None) -> None:
        whole = None if fraction is None else (index + min(1.0, max(0.0, fraction))) / count
        progress(f"{prefix}{label}", whole)
    return hook
