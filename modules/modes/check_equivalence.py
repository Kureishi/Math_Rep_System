from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🔁 Check equivalence",
    keywords=("proof", "identity", "same", "equal"),
    module="ui.equivalence", function="render_equivalence_tab",
    needs_client=False,
    order=30,
)
