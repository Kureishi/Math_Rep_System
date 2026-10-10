"""
The Streamlit front end. app.py used to be one ~4,000-line script; it is now a thin entrypoint over
this package:

    app.py                    page config, session defaults, sidebar, dispatch
    ui/__init__.py            PAGES: mode label -> page function (this file). Pages are imported
                              LAZILY, on first use -- see the note below
    ui/common.py              helpers shared by several pages (upload guard, snapshot buttons, ...)
    ui/sidebar.py             the sidebar
    ui/word_problem.py        the default page: input, Solve, pipeline, then the results view
    ui/results/               the results view for a solved problem, one function per section
    ui/<mode>.py              one module per standalone mode

Nothing in modules/ imports from here (modules/ stays Streamlit-free), and the mode list itself lives
in modules/command_palette.py (MODE_LABELS) so the sidebar radio, the palette and this dispatch table
cannot drift apart -- tests/test_app_modes.py fails if they do.

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

if TYPE_CHECKING:  # annotation only -- importing it for real would drag in the openai SDK
    from modules.llm_client import LMStudioClient

# The default mode. It is not in PAGES because it is not gated behind an early st.stop(): app.py falls
# through to it when the selected mode has no entry in PAGES.
WORD_PROBLEM_MODE = "📝 Word problem solver"

Page = Callable[["LMStudioClient"], None]

# mode label -> (module, function, takes_client). takes_client is False for pages that never talk to the
# model, so which pages DO use the LLM stays visible in this one table.
PAGE_TARGETS: dict[str, tuple[str, str, bool]] = {
    "📈 Curve fitting": ("ui.curve_fitting", "render_curve_fitting_tab", False),
    "🔁 Check equivalence": ("ui.equivalence", "render_equivalence_tab", False),
    "📚 Batch solver": ("ui.batch", "render_batch_solver_tab", True),
    "🔗 Problem chains": ("ui.chains", "render_chains_tab", True),
    "🔬 Extraction diff": ("ui.extraction_diff", "render_extraction_diff_tab", True),
    "🆚 Compare solves": ("ui.compare", "render_compare_tab", False),
    "⚛️ Quantum mechanics": ("ui.quantum", "render_quantum_tab", False),
    "📐 Dimensional analysis": ("ui.dimensional", "render_dimensional_analysis_tab", False),
    "🔄 Transforms & series": ("ui.transforms_series", "render_transforms_series_tab", False),
    "🌡️ PDE solver": ("ui.pde", "render_pde_tab", False),
    "🧮 Tensor calculus": ("ui.tensor", "render_tensor_calculus_tab", False),
    "📔 Research journal": ("ui.journal", "render_research_journal_tab", False),
    "🚀 Quick start": ("ui.quick_start", "render_quick_start_tab", False),
    "📐 Geometry": ("ui.geometry", "render_geometry_tab", False),
}


def _lazy(module: str, func: str, takes_client: bool) -> Page:
    """A page function that imports its module on first call."""
    def run(client: "LMStudioClient") -> None:
        page = getattr(importlib.import_module(module), func)
        page(client) if takes_client else page()
    return run


PAGES: dict[str, Page] = {label: _lazy(*target) for label, target in PAGE_TARGETS.items()}

