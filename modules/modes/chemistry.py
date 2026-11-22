from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🧪 Chemistry",
    keywords=("chemistry", "balance equation", "balancing", "stoichiometry", "limiting reagent", "molar mass", "molarity", "ph", "titration",
              "buffer", "acid", "base", "kinetics", "rate law", "reaction order", "half-life", "arrhenius", "activation energy", "mole",
              "yield", "equilibrium", "chemical"),
    module="ui.chemistry", function="render_chemistry_tab",
    needs_client=False,
    order=175,
    group="Science",
)
