"""
Panels that rerun on their own.

Streamlit re-executes the whole script whenever any widget changes, so moving a slider inside one
analysis panel used to re-run every other panel on the page too. `@isolated` makes a function an
`st.fragment`: a widget interaction inside it reruns only that function. Everything the panel reads from
outside (the model, the report) is passed in as arguments, which a fragment keeps from the last full run.

Three rules keep a fragment correct, and each is why the functions decorated with this look the way they
do:

  * a fragment may only write into containers it creates itself, so the `with tab:` block lives at the
    CALL site and the decorated function draws straight into whatever container it is called in;
  * anything that must change the REST of the page (adding a plot to the report, setting a widget that
    lives in another panel) has to trigger a full rerun -- `st.rerun()` does, and ui/actions.py's
    `action_button` does it for a click;
  * a fragment must not write to the sidebar.

`isolated` also autosaves the exploration state afterwards (ui/view_state.py), because a fragment
rerun does not reach the end of the results page where that normally happens.

ISOLATED lists every decorated function, so a test can pin which panels are isolated and a new panel
cannot be added to (or dropped from) the list unnoticed.
"""
import functools
from collections.abc import Callable
from typing import Any, TypeVar

import streamlit as st

from ui.view_state import autosave

F = TypeVar("F", bound=Callable[..., Any])

ISOLATED: dict[str, Callable[..., Any]] = {}


def isolated(fn: F) -> F:
    """Decorator: run `fn` as its own fragment (see the module docstring)."""

    @functools.wraps(fn)
    def body(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        autosave()
        return result

    wrapped = st.fragment(body)
    ISOLATED[f"{fn.__module__}.{fn.__qualname__}"] = wrapped
    return wrapped  # type: ignore[return-value]
