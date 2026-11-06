from modules.mode_spec import ModeSpec

MODE = ModeSpec(
    label="🎲 Posterior sampling (MCMC)",
    keywords=("mcmc", "posterior", "bayesian fit", "corner plot", "credible interval", "markov chain", "emcee", "parameter uncertainty", "nonlinear fit uncertainty", "sampling", "walkers"),
    module="ui.posterior", function="render_posterior_tab",
    needs_client=False,
    order=200,
)
