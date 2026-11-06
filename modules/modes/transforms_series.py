from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🔄 Transforms & series",
    keywords=("laplace", "fourier", "taylor", "asymptotic", "expansion"),
    module="ui.transforms_series", function="render_transforms_series_tab",
    needs_client=False,
    order=100,
)
