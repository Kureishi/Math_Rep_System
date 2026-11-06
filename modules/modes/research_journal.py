from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="📔 Research journal",
    keywords=("history", "concepts", "journal", "past problems"),
    module="ui.journal", function="render_research_journal_tab",
    needs_client=False,
    order=220,
)
