from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🌌 General relativity",
    keywords=("geodesic", "schwarzschild", "kerr", "black hole", "einstein", "kretschmann", "precession", "light bending", "effective potential", "isco", "horizon", "penrose", "light cone", "gravitational waves", "quadrupole", "eddington", "kruskal", "singularity", "weyl"),
    module="ui.relativity", function="render_relativity_tab",
    needs_client=False,
    order=150,
)
