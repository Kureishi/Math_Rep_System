from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🆚 Compare solves",
    keywords=("diff", "versus", "difference", "what if", "side by side", "two problems"),
    module="ui.compare", function="render_compare_tab",
    needs_client=False,
    order=80,
)
