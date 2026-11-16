"""
Keeping one failing part of a page from taking the rest down.

A Streamlit page is one script: an exception in any panel used to end the script there, so the red traceback replaced everything that would have rendered after
it (a missing python-docx in the report exporter hid the whole results view). `guarded()` and `run_panel()` catch an ordinary exception in one panel, say in plain
language what that panel could not do and what to try (a missing package gets its exact `pip install`), keep the traceback in an expander and in the log, and let the
rest of the page render.

What is NOT caught, on purpose: Streamlit's own control flow (`st.stop()` and `st.rerun()` are implemented as exceptions), and KeyboardInterrupt / SystemExit.
"""
import traceback
from contextlib import contextmanager
from typing import Any, Callable, Iterator

import streamlit as st
from streamlit.runtime.scriptrunner import RerunException, StopException

from modules.app_logging import logger
from modules.optional_deps import explain_import_error

_PASS_THROUGH = (RerunException, StopException)


def explain(exc: BaseException) -> str:
    """The sentence shown for an exception: the package fix for an import error, otherwise the error itself."""
    return explain_import_error(exc) or f"{type(exc).__name__}: {str(exc)[:300]}"


def show_failure(label: str, exc: BaseException, hint: str = "") -> None:
    """The in-page notice for a panel that failed (the rest of the page is unaffected)."""
    logger.exception("panel %r failed", label, exc_info=exc)
    st.warning(f"**{label}** could not be shown. {explain(exc)}" + (f"  \n{hint}" if hint else "") + "  \nThe rest of this page is not affected.", icon="⚠️")
    with st.expander(f"Technical details: {label}"):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), language="text")


@contextmanager
def guarded(label: str, hint: str = "") -> Iterator[None]:
    """`with guarded("Share / export"): ...` -- if the block raises, show a notice and carry on."""
    try:
        yield
    except _PASS_THROUGH:
        raise
    except Exception as exc:  # noqa: BLE001
        show_failure(label, exc, hint)


def run_panel(label: str, fn: Callable[..., Any], *args: Any, default: Any = None, hint: str = "", **kwargs: Any) -> Any:
    """Call `fn(*args, **kwargs)`; on an ordinary exception show a notice and return `default` (for panels whose return value later panels use)."""
    try:
        return fn(*args, **kwargs)
    except _PASS_THROUGH:
        raise
    except Exception as exc:  # noqa: BLE001
        show_failure(label, exc, hint)
        return default
