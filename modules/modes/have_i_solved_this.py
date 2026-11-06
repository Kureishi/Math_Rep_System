from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🔎 Have I solved this?",
    keywords=("solved before", "have i solved", "did i already", "similar problem", "similar problems", "past problems", "history search", "search history", "find an old problem", "seen this", "recall", "duplicate"),
    module="ui.recall", function="render_recall_tab",
    needs_client=True,
    order=190,
)
