"""
The Streamlit front end. app.py used to be one ~4,000-line script; it is now a thin entrypoint over
this package:

    app.py                    page config, session defaults, sidebar, dispatch
    ui/__init__.py            PAGES: mode label -> page function, derived from modules/modes/. Pages are
                              imported LAZILY, on first use -- see the note below
    ui/common.py              helpers shared by several pages (upload guard, snapshot buttons, ...)
    ui/sidebar.py             the sidebar
    ui/word_problem.py        the default page: input, Solve, pipeline, then the results view
    ui/results/               the results view for a solved problem, one function per section
    ui/<mode>.py              one module per standalone mode

Nothing in modules/ imports from here (modules/ stays Streamlit-free), and the mode list itself lives
in modules/modes/ (one declaration file per mode, discovered by modules/mode_registry.py), from which the sidebar radio, the
palette, the router and this dispatch table are all derived, so they cannot drift apart.

Why lazy: this file used to import all twelve page modules up front, and those pull in pandas
(-> pyarrow), openai, matplotlib, scipy, ... -- several seconds of imports that all had to finish
before the first pixel of the app could be drawn, even though a given session only ever opens one or
two modes. Now a page's module is imported the first time its mode is selected, so importing `ui`
(which app.py does before drawing the title) is essentially free. The trade-off is that a broken
page module no longer fails at import of this package; tests/test_app_modes.py compensates by
importing every entry in PAGE_TARGETS and checking the function exists.
"""
import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING

from modules.mode_registry import REGISTRY

if TYPE_CHECKING:  # annotation only -- importing it for real would drag in the openai SDK
    from modules.llm_client import LMStudioClient

# The default mode. It is not in PAGES because it is not gated behind an early st.stop(): app.py falls
# through to it when the selected mode has no entry in PAGES.
WORD_PROBLEM_MODE = "📝 Word problem solver"

Page = Callable[["LMStudioClient"], None]

# mode label -> (module, function, takes_client), DERIVED from the mode declarations in modules/modes/ (and plugins/): to add a mode
# you add a file there, not a line here. The default mode has no page entry. takes_client is False for pages that never talk to the
# model; `grep needs_client=True modules/modes` lists the ones that do.
PAGE_TARGETS: dict[str, tuple[str, str, bool]] = {
    spec.label: (spec.module, spec.function, spec.needs_client) for spec in REGISTRY.specs if spec.module
}


def _lazy(module: str, func: str, takes_client: bool) -> Page:
    """A page function that imports its module on first call."""
    def run(client: "LMStudioClient") -> None:
        page = getattr(importlib.import_module(module), func)
        page(client) if takes_client else page()
    return run


PAGES: dict[str, Page] = {label: _lazy(*target) for label, target in PAGE_TARGETS.items()}

