from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🌊 Waves & optics lab",
    keywords=("double slit", "diffraction", "interference", "fringe", "fringes", "fraunhofer", "airy disk", "grating", "ripple tank", "doppler",
              "doppler effect", "mach cone", "sonic boom", "wave optics", "young's experiment", "single slit", "slit"),
    module="ui.waves", function="render_waves_tab",
    needs_client=False,
    order=177,
)
