from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="📐 Geometry",
    keywords=("triangle", "circle", "angle", "law of sines", "law of cosines", "schematic", "sss", "sas", "asa"),
    module="ui.geometry", function="render_geometry_tab",
    needs_client=False,
    order=240,
)
