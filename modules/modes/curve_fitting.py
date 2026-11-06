from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="📈 Curve fitting",
    keywords=("regression", "fit", "statistics", "bayesian", "confidence interval"),
    module="ui.curve_fitting", function="render_curve_fitting_tab",
    needs_client=False,
    order=20,
)
