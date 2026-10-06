"""
A long computation shown as a live status block, with a Stop button where one can take effect.

    notice_interrupted("mc_v")                       # before the Run button: was the last run cut short?
    if st.button("Run"):
        with tracked_run("mc_v", "Monte Carlo for v") as progress:
            result = run_monte_carlo(..., progress=progress)

`progress` is the hook modules/progress.py defines: the computation calls it at checkpoints, and each call
updates the block's label and bar.

How Stop works, because Streamlit has no cancel API: clicking ANY widget while a script runs requests a
rerun, and Streamlit abandons the running script at its next `st.*` call. The Stop button is just a
widget that is always there to click, and every progress checkpoint is an `st.*` call -- so the run ends
at the next checkpoint (between chunks of sampling, between the two solves of a PDE run), never in the
middle of a single numpy or SymPy call, which cannot be interrupted. The aborted run leaves nothing on
screen, so `tracked_run` raises a flag when it starts and clears it only when the run finishes or fails;
a flag still set at the next run means the last one was cut short, and `notice_interrupted` says so. An
earlier finished result is never touched: results are stored only when a run completes.

`stoppable=False` omits the button for a computation whose only checkpoints are before and after one
indivisible call, where Stop would have nothing to stop.
"""
from collections.abc import Iterator
from contextlib import contextmanager

import streamlit as st

from modules.progress import ProgressFn


def _flag(key: str) -> str:
    return f"_run_active_{key}"


def notice_interrupted(key: str, what: str = "run") -> bool:
    """If the previous `tracked_run(key, ...)` never finished, say so and clear the flag. Returns whether
    it was cut short."""
    if st.session_state.pop(_flag(key), False):
        st.warning(f"The previous {what} was stopped before it finished, so it produced no new result.")
        return True
    return False


@contextmanager
def tracked_run(key: str, label: str, *, stoppable: bool = True,
                done_label: str | None = None) -> Iterator[ProgressFn]:
    st.session_state[_flag(key)] = True
    with st.status(label, expanded=True) as status:
        bar = st.progress(0.0, text="Starting")
        if stoppable:
            st.button("⏹ Stop", key=f"stop_{key}",
                      help="Stops at the next checkpoint (between chunks of work), not mid-calculation.")

        def on_progress(message: str, fraction: float | None) -> None:
            status.update(label=message)
            if fraction is not None:
                bar.progress(min(1.0, max(0.0, fraction)), text=message)

        try:
            yield on_progress
        except Exception:
            status.update(label=f"{label} failed", state="error", expanded=True)
            st.session_state.pop(_flag(key), None)
            raise
        status.update(label=done_label or f"{label}: done", state="complete", expanded=False)
        st.session_state.pop(_flag(key), None)
