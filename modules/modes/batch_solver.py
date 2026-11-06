from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="📚 Batch solver",
    keywords=("multiple", "worksheet", "many problems"),
    module="ui.batch", function="render_batch_solver_tab",
    needs_client=True,
    order=40,
)
