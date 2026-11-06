from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="📐 Dimensional analysis",
    keywords=("units", "dimensions", "si", "imperial"),
    module="ui.dimensional", function="render_dimensional_analysis_tab",
    needs_client=False,
    order=90,
)
