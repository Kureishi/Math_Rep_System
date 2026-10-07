"""
Building exports for the Share dialog: the session -> an ExportContext, and a cached, deferred build.

DEFERRED: a download button given a callable builds nothing until it is clicked (Streamlit runs the callable
on the click, on its own thread), so opening the dialog costs nothing and there is no "generate, then
download" second step. That thread has no script context, so the callable closes over plain data collected
here in the script thread, and its cache is this module's own, not session state.

CACHED: building a PDF or deck takes a second or two, and people click twice (or switch to another format
and back). Results are kept by the CONTENT of everything they were built from -- the same fingerprint the
rest of the UI cache uses -- plus the format and the chosen sections, so a changed answer, plot or section
choice can never be served a stale file. The minute is part of the key because every export is stamped with
when it was generated: a click a minute later gets a fresh stamp, not an old file. Content that cannot be
fingerprinted is simply rebuilt each time. The cache holds the most recent few files under a byte budget.
"""
import threading
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime

import streamlit as st

from modules import session_extras
from modules.report_export import ExportContext, ExportedFile, expected_file, render_export
from ui.cache import fingerprint

MAX_ENTRIES = 8
MAX_BYTES = 64 * 1024 * 1024

_lock = threading.Lock()
_files: "OrderedDict[str, ExportedFile]" = OrderedDict()


def collect_context() -> ExportContext:
    """The open problem and everything about it that a report can hold, read from the session."""
    model = st.session_state["model"]
    steps = st.session_state.get("steps") or {}
    return ExportContext(
        problem_text=st.session_state.get("problem_text", ""),
        model=model,
        report=st.session_state["report"],
        steps_by_target=steps,
        scenarios=st.session_state.get("scenarios") or [],
        snapshots=list(st.session_state.get("plot_snapshots", {}).values()),
        extras=session_extras.build_extras(st.session_state, model, steps, live_figures=True),
    )


def _key(fmt: str, ctx: ExportContext, include: list[str] | None) -> str | None:
    try:
        return fingerprint(fmt, sorted(include) if include is not None else None, ctx,
                           datetime.now().strftime("%Y-%m-%d %H:%M"))
    except TypeError:
        return None


def build_cached(fmt: str, ctx: ExportContext, include: list[str] | None) -> ExportedFile:
    key = _key(fmt, ctx, include)
    if key is not None:
        with _lock:
            hit = _files.get(key)
            if hit is not None:
                _files.move_to_end(key)
                return hit
    built = render_export(fmt, ctx, include)
    if key is not None:
        with _lock:
            _files[key] = built
            _files.move_to_end(key)
            while len(_files) > MAX_ENTRIES or sum(len(f.data) for f in _files.values()) > MAX_BYTES:
                if len(_files) <= 1:
                    break
                _files.popitem(last=False)
    return built


def clear_cache() -> None:
    with _lock:
        _files.clear()


def download_button(fmt: str, label: str, ctx: ExportContext, include: list[str] | None, key: str) -> None:
    """A download button for `fmt` that builds the file only when clicked."""
    name, mime = expected_file(fmt, ctx, include)
    chosen = list(include) if include is not None else None

    def build() -> bytes:
        return build_cached(fmt, ctx, chosen).data

    st.download_button(label, data=build, file_name=name, mime=mime, key=key, on_click="ignore")


def deferred(fmt: str, ctx: ExportContext, include: list[str] | None) -> Callable[[], bytes]:
    """The callable a deferred download would run (exposed so it can be tested without a browser)."""
    chosen = list(include) if include is not None else None
    return lambda: build_cached(fmt, ctx, chosen).data
