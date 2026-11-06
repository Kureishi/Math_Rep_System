from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🧑‍🏫 Problem sets",
    keywords=("problem set", "worksheet", "quiz", "practice problems", "homework", "answer key", "grade", "grading", "exam", "assignment", "teacher", "variants", "auto-grade"),
    module="ui.problem_sets", function="render_problem_sets_tab",
    needs_client=False,
    order=210,
)
