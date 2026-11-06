from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🧮 Tensor calculus",
    keywords=("metric", "christoffel", "ricci", "curvature", "riemann", "covariant derivative"),
    module="ui.tensor", function="render_tensor_calculus_tab",
    needs_client=False,
    order=120,
)
