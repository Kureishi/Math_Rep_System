from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🔌 Circuits & control lab",
    keywords=("rlc", "rlc circuit", "resonance", "resonant frequency", "damping ratio", "q factor", "bode", "bode plot", "step response", "transient", "pid", "pid tuning",
              "controller", "control system", "transfer function", "phase margin", "gain margin", "ziegler-nichols", "routh", "root locus", "overshoot", "settling time", "circuit"),
    module="ui.circuits", function="render_circuits_tab",
    needs_client=False,
    order=179,
)
