from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🔥 Statistical mechanics",
    keywords=("partition function", "free energy", "entropy", "heat capacity", "ising", "potts", "xy model", "metropolis", "monte carlo lattice", "binder cumulant", "critical point", "onsager", "mean field", "fermi-dirac", "bose-einstein", "maxwell-boltzmann", "bec", "chemical potential", "langevin", "brownian", "fokker-planck", "diffusion", "fluctuation dissipation", "random matrix", "wigner semicircle", "level spacing"),
    module="ui.statmech", function="render_statmech_tab",
    needs_client=False,
    order=130,
)
