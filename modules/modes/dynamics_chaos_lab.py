from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🌀 Dynamics & chaos lab",
    keywords=("n-body", "nbody", "three-body", "three body", "figure-eight", "figure eight", "orbit", "kepler", "symplectic", "leapfrog", "verlet", "yoshida", "double pendulum",
              "pendulum", "lorenz", "butterfly", "chaos", "chaotic", "lyapunov", "lyapunov exponent", "strange attractor", "energy drift", "integrator"),
    module="ui.dynamics", function="render_dynamics_tab",
    needs_client=False,
    order=178,
    group="Simulation labs",
)
