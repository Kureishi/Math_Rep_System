from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🔗 Problem chains",
    keywords=("chain", "sequence", "multi-step", "pipeline"),
    module="ui.chains", function="render_chains_tab",
    needs_client=True,
    order=50,
)
