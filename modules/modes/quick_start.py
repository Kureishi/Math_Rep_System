from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🚀 Quick start",
    keywords=("examples", "tutorial", "gallery", "getting started"),
    module="ui.quick_start", function="render_quick_start_tab",
    needs_client=False,
    order=230,
)
