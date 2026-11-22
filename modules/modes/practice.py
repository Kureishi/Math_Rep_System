from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🧠 Practice & mastery",
    keywords=("daily practice", "spaced repetition", "review queue", "mastery", "streak", "flashcards", "study", "revise", "revision",
              "retention", "due today", "learn", "practice every day"),
    module="ui.practice", function="render_practice_tab",
    needs_client=False,
    order=215,
    group="Tools",
)
