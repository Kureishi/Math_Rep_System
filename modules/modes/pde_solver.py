from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🌡️ PDE solver",
    keywords=("heat", "wave", "laplace equation", "robin", "neumann", "dirichlet", "partial differential"),
    module="ui.pde", function="render_pde_tab",
    needs_client=False,
    order=110,
)
