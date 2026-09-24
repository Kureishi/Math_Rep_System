"""
The Streamlit front end. app.py used to be one ~4,000-line script; it is now a thin entrypoint over
this package:

    app.py                    page config, session defaults, sidebar, dispatch
    ui/__init__.py            PAGES: mode label -> page function (this file)
    ui/common.py              helpers shared by several pages (upload guard, snapshot buttons, ...)
    ui/sidebar.py             the sidebar
    ui/word_problem.py        the default page: input, Solve, pipeline, then the results view
    ui/results/               the results view for a solved problem, one function per section
    ui/<mode>.py              one module per standalone mode

Nothing in modules/ imports from here (modules/ stays Streamlit-free), and the mode list itself lives
in modules/command_palette.py (MODE_LABELS) so the sidebar radio, the palette and this dispatch table
cannot drift apart -- tests/test_app_modes.py fails if they do.
"""
from collections.abc import Callable

from modules.llm_client import LMStudioClient
from ui.batch import render_batch_solver_tab
from ui.chains import render_chains_tab
from ui.curve_fitting import render_curve_fitting_tab
from ui.dimensional import render_dimensional_analysis_tab
from ui.equivalence import render_equivalence_tab
from ui.extraction_diff import render_extraction_diff_tab
from ui.geometry import render_geometry_tab
from ui.journal import render_research_journal_tab
from ui.pde import render_pde_tab
from ui.quick_start import render_quick_start_tab
from ui.tensor import render_tensor_calculus_tab
from ui.transforms_series import render_transforms_series_tab

# The default mode. It is not in PAGES because it is not gated behind an early st.stop(): app.py falls
# through to it when the selected mode has no entry in PAGES.
WORD_PROBLEM_MODE = "📝 Word problem solver"

Page = Callable[[LMStudioClient], None]


def _no_llm(page: Callable[[], None]) -> Page:
    """Adapts a page that never talks to the model to the uniform Page signature, so which pages DO
    use the LLM is visible in the table below (the ones not wrapped in _no_llm)."""
    def run(client: LMStudioClient) -> None:
        page()
    return run


PAGES: dict[str, Page] = {
    "📈 Curve fitting": _no_llm(render_curve_fitting_tab),
    "🔁 Check equivalence": _no_llm(render_equivalence_tab),
    "📚 Batch solver": render_batch_solver_tab,
    "🔗 Problem chains": render_chains_tab,
    "🔬 Extraction diff": render_extraction_diff_tab,
    "📐 Dimensional analysis": _no_llm(render_dimensional_analysis_tab),
    "🔄 Transforms & series": _no_llm(render_transforms_series_tab),
    "🌡️ PDE solver": _no_llm(render_pde_tab),
    "🧮 Tensor calculus": _no_llm(render_tensor_calculus_tab),
    "📔 Research journal": _no_llm(render_research_journal_tab),
    "🚀 Quick start": _no_llm(render_quick_start_tab),
    "📐 Geometry": _no_llm(render_geometry_tab),
}
