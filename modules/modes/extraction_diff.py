from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🔬 Extraction diff",
    keywords=("wording", "compare", "rephrase"),
    module="ui.extraction_diff", function="render_extraction_diff_tab",
    needs_client=True,
    order=60,
)
