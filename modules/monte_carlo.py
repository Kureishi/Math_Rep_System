"""
Monte Carlo uncertainty propagation: given one or more known variables
with an uncertainty attached (a measurement's ± error, say), samples
each uncertain variable many times and reports the resulting SPREAD of
the target's value -- not just a single point estimate.

Distinct from sensitivity.py's tornado/sweep analysis, which varies ONE
input at a time across its own range to see how much the target moves
(a partial-derivative-flavored view). This module varies ALL uncertain
inputs SIMULTANEOUSLY, each drawn from its own distribution, and looks
at the resulting output distribution as a whole -- the standard way a
researcher would ask "given these measurement uncertainties, how
uncertain is my final answer?" rather than "which single input matters
most?" (that second question is what the tornado chart is for; the two
are complementary, not redundant).

Deliberately solves the system SYMBOLICALLY ONLY ONCE (substituting
every FIXED known value in, leaving the uncertain variables as free
symbols) and then lambdifies+vectorizes that one closed-form expression
across every sample, rather than calling verifier._solve_sympy() -- and
therefore sp.solve() -- once per sample. That distinction matters more
here than it does in chains.py: chains.py re-solves once per user EDIT,
but a single Monte Carlo run needs hundreds to thousands of evaluations,
and verifier._known_substitutions() runs every known value through
sp.nsimplify() to look for a "nice" exact form -- fine for the small
number of LLM-extracted values it normally sees, but on arbitrary
sampled floats it can occasionally hit sympy's (very slow) algebraic-
number reconstruction path. Solving once symbolically and evaluating
numerically after sidesteps that entirely, the same "solve symbolically
once, sweep numerically" pattern plotter.py already uses for line/
surface plots.
"""
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from modules.equation_engine import ProblemModel, target_kind
from modules.timeout_utils import run_with_timeout, ComputationTimeoutError

MAX_SAMPLES = 20000  # a sanity cap -- generous, since (unlike chains.py) this module only ever
                       # does ONE symbolic solve regardless of sample count, so cost scales with
                       # cheap numpy evaluation, not with repeated sp.solve() calls


@dataclass
class UncertainVariable:
    symbol: str
    mean: float
    std: float             # standard deviation of a normal distribution around `mean`;
                             # must be > 0 -- a variable with no real uncertainty shouldn't be here


@dataclass
class MonteCarloResult:
    target: str
    samples: list[float] = field(default_factory=list)   # only the samples that solved successfully
    n_requested: int = 0
    n_failed: int = 0        # samples where the symbolic expression evaluated to a complex or
                               # non-finite value at that particular random draw (e.g. a sqrt of a
                               # negative number at an extreme sample) -- excluded from `samples`/stats
    mean: float | None = None
    std: float | None = None
    p5: float | None = None
    p95: float | None = None


def run_monte_carlo(model: ProblemModel, target: str, uncertain_vars: list[UncertainVariable],
                     n_samples: int = 1000, seed: int | None = None) -> MonteCarloResult:
    """Draws `n_samples` joint samples (each uncertain variable sampled
    independently from its own Normal(mean, std)) and evaluates a single
    closed-form solution for `target` at every sample. A sample that
    evaluates to a complex or non-finite value is dropped from the
    result rather than raising -- an occasional failed sample at an
    extreme random draw is expected, but see `n_failed` to catch a run
    where MOST samples fail (usually a sign the given std is too large
    relative to the model's valid domain)."""
    if target not in model.solve_for or target_kind(model, target) != "equation":
        raise ValueError(f"'{target}' isn't an algebraic solve_for target of this model.")
    if not uncertain_vars:
        raise ValueError("Need at least one uncertain variable to propagate.")
    for uv in uncertain_vars:
        if uv.std <= 0:
            raise ValueError(f"'{uv.symbol}' has a non-positive std ({uv.std}) -- not actually uncertain.")

    n_samples = max(10, min(n_samples, MAX_SAMPLES))
    rng = np.random.default_rng(seed)
    draws = {uv.symbol: rng.normal(uv.mean, uv.std, size=n_samples) for uv in uncertain_vars}

    # solve ONCE: substitute every fixed (non-uncertain) known value in,
    # leaving the uncertain variables as free symbols in the solution
    uncertain_symbols = {uv.symbol for uv in uncertain_vars}
    fixed_subs = {
        sp.Symbol(v.symbol): sp.nsimplify(v.known_value)
        for v in model.variables
        if v.known_value is not None and v.symbol not in uncertain_symbols
    }
    eqs = [e.sympy_eq.subs(fixed_subs) for e in model.equations
           if e.kind == "equation" and e.sympy_eq is not None]
    if not eqs:
        raise ValueError("This model has no algebraic equations to solve.")

    algebraic_targets = [t for t in model.solve_for if target_kind(model, t) == "equation"]
    target_syms = [sp.Symbol(t) for t in algebraic_targets]
    try:
        sol = run_with_timeout(sp.solve, eqs, target_syms, dict=True, label="monte carlo solve")
    except ComputationTimeoutError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Couldn't symbolically solve for '{target}': {e}") from e
    if not sol:
        raise ValueError(f"Couldn't symbolically solve for '{target}' given the fixed inputs.")

    target_expr = sol[0].get(sp.Symbol(target))
    if target_expr is None:
        raise ValueError(f"'{target}' didn't appear in the symbolic solution.")

    # any uncertain variable that dropped out algebraically (target
    # doesn't actually depend on it) still needs a lambdify argument --
    # sp.lambdify requires every named arg to appear, so substituting a
    # harmless local placeholder for genuinely-absent symbols would be
    # more complexity than it's worth; instead just lambdify over
    # whichever uncertain symbols the solution actually contains, and
    # feed only those samples through
    used_symbols = target_expr.free_symbols
    active_vars = [uv for uv in uncertain_vars if sp.Symbol(uv.symbol) in used_symbols]
    if not active_vars:
        # the target doesn't depend on any uncertain input -- deterministic
        value = float(target_expr)
        return MonteCarloResult(target=target, samples=[value] * n_samples, n_requested=n_samples,
                                  n_failed=0, mean=value, std=0.0, p5=value, p95=value)

    f = sp.lambdify([sp.Symbol(uv.symbol) for uv in active_vars], target_expr, "numpy")
    arg_arrays = [draws[uv.symbol] for uv in active_vars]

    try:
        raw = np.broadcast_to(np.asarray(f(*arg_arrays), dtype=complex), (n_samples,))
    except Exception:  # noqa: BLE001
        # lambdify's vectorized path can't always handle every expression
        # (piecewise conditionals, certain special functions) -- fall
        # back to evaluating one sample at a time rather than failing
        # the whole run; an individual sample that itself raises just
        # becomes NaN, same as any other failed sample
        vals = []
        for i in range(n_samples):
            try:
                vals.append(complex(f(*[arr[i] for arr in arg_arrays])))
            except Exception:  # noqa: BLE001
                vals.append(complex("nan"))
        raw = np.array(vals)

    is_real = np.abs(raw.imag) < 1e-9
    real_vals = raw.real
    finite_mask = is_real & np.isfinite(real_vals)
    samples = real_vals[finite_mask].tolist()
    n_failed = n_samples - len(samples)

    if not samples:
        return MonteCarloResult(target=target, samples=[], n_requested=n_samples, n_failed=n_failed)

    arr = np.array(samples)
    return MonteCarloResult(
        target=target, samples=samples, n_requested=n_samples, n_failed=n_failed,
        mean=float(np.mean(arr)), std=float(np.std(arr)),
        p5=float(np.percentile(arr, 5)), p95=float(np.percentile(arr, 95)),
    )
