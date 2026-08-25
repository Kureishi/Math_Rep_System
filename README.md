# Math Representation System

Turns a plain-language problem (typed or photographed) into verified symbolic
equations, a step-by-step solution, and suggestions for where else the same
math applies -- running entirely on your machine via LM Studio + SymPy.

## 1. Install LM Studio and a model

1. Download LM Studio: https://lmstudio.ai (Windows/Mac/Linux)
2. In LM Studio, download a reasoning-capable model (e.g. Qwen2.5-14B-Instruct,
   Llama-3.1-8B-Instruct -- anything decent at instruction following).
3. Optional but recommended: also download a **vision** model
   (e.g. Qwen2-VL-7B-Instruct) if you want to solve problems from photos
   without a separate OCR install.
4. Go to the **Developer** tab in LM Studio, load your model, and click
   **Start Server**. Note the port (default `1234`).

## 2. Install this app

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you want image-OCR as a fallback for non-vision models, also install the
Tesseract binary:
- macOS: `brew install tesseract`
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- Windows: https://github.com/UB-Mannheim/tesseract/wiki

## 3. Point the app at your LM Studio models

Edit `config.py`, or set environment variables before launching:

```bash
export LM_REASONING_MODEL="qwen2.5-14b-instruct"   # must match the model ID shown in LM Studio
export LM_VISION_MODEL="qwen2-vl-7b-instruct"
export LM_STUDIO_BASE_URL="http://localhost:1234/v1"   # change if you used a different port
```

## 4. Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` in your browser. Everything (LLM inference,
math, plotting) runs locally -- nothing leaves your machine.

## How it works

See `ARCHITECTURE.md` for the full design. In short:

1. **Extraction** -- your problem text (or LLM-transcribed image) is sent to
   the reasoning model with a strict JSON schema, producing variables and
   relations, plus an optional **objective** for optimization problems.
   Each relation is tagged as a plain **equation** (which may use
   `Piecewise` for tiered/conditional formulas like tax brackets, with no
   separate kind needed), an **inequality** (a constraint like `v <= 25`),
   an **ODE** (a differential equation for a declared function like
   `y(t)`), or a **recurrence** (a difference equation like `a(n+1) =
   a(n) + 5`) -- each kind is parsed, verified, and solved differently.
2. **Verification** -- SymPy independently checks that relations parse,
   balance numerically against known values (equations -- Piecewise-aware,
   selecting the correct branch), are actually satisfied by the given
   numbers (inequalities), or symbolically satisfy the original
   differential/difference equation(s) -- via `sp.checkodesol` for a
   standalone ODE, substituting every function's solution into every
   equation simultaneously for a coupled ODE system, or substituting a
   recurrence's closed form back into itself (numeric-sampling fallback
   when the exact symbolic residual doesn't land on zero due to floating-
   point noise, e.g. Fibonacci-style solutions mixing `sqrt(5)` with float
   initial conditions). Optimization problems get their own two-part check:
   the gradient is confirmed to actually be zero at the claimed critical
   point, and the min/max classification (via the second-derivative test
   or the Hessian's eigenvalue signs) is confirmed to match what was
   requested -- plus an honest feasibility check against any inequality
   constraints, since finding the true constrained optimum against
   inequalities (not just equalities) is out of scope (full KKT analysis)
   and the app says so rather than silently presenting an unconstrained
   answer as the constrained one. Plus a physical-unit consistency check
   via `sympy.physics.units` for equations, inequalities, and ODEs (the
   dimension of d^n(f)/dx^n is dim(f)/dim(x)^n; Piecewise equations get
   each branch checked separately and required to agree, since sympy's own
   dimension machinery silently mishandles a Piecewise as a whole). Every
   passing check also reports *how close* it came to its own tolerance
   (`essentially exact` vs `borderline`), not just pass/fail -- a residual
   of 1e-9 and one that barely cleared the bar look different in the UI.
   For algebraic equations, a *second*, independent LLM call re-solves the
   original word problem from scratch; if its numeric answer disagrees
   with SymPy's solve by more than ~2%, the derivation is considered
   flawed and automatically retried (up to `max_verification_retries`
   times) with the discrepancy fed back to the model.
3. **Step-by-step** -- SymPy computes the actual steps (substitute -> isolate
   -> simplify for equations; `reduce_inequalities` for constraints;
   `dsolve`/`dsolve_system` general solution -> apply initial conditions
   for ODEs, coupled or standalone; `rsolve` for recurrences; direct
   calculus or Lagrange multipliers for optimization, showing any
   constraint-elimination substitution as its own step); the LLM only
   narrates those already-verified steps in plain language, rather than
   being trusted to invent
   the math live.
4. **Alternative scenarios** -- a separate, low-stakes LLM call suggests
   other real-world contexts with the same mathematical structure.
5. **Optimization** -- given an objective to minimize/maximize, critical
   points are found via direct calculus (unconstrained) or Lagrange
   multipliers (equality-constrained, when the constraint can't be cleanly
   substituted away), classified min/max/saddle via the second-derivative
   test or the Hessian's eigenvalue signs, and checked against any
   inequality constraints for feasibility.
6. **Workspace & plotting** -- solved values can be pushed into a session
   workspace for reuse in later calculations (reference them by name in a
   new problem statement -- the model is automatically told their value).
   Any equation with one free parameter gets a live 2D line plot; with two
   free parameters, a live 3D surface plot. Multiple inequality constraints
   sharing two free variables get a shaded feasible-region plot (every
   constraint satisfied simultaneously). ODE solutions get their own plot
   of the solution curve plus an "evaluate at a specific point" control
   (e.g. population after 10 years); recurrence solutions get the discrete
   analog -- a stem/marker plot rather than a connected line, since a
   sequence is only defined at integer indices.
7. **Dimensional checking** -- alongside numeric-balance checking, each
   equation, inequality, ODE, and recurrence is independently checked for
   physical-unit consistency via `sympy.physics.units`, catching errors a
   numeric check can't (e.g. equating a distance to a velocity, or giving
   a decay-rate constant in the wrong dimension -- both could pass
   numerically by pure coincidence but never pass dimensionally).
8. **History** -- every solved problem is saved to a local SQLite file
   (`data/history.db`, gitignored) and listed in the sidebar. Loading a
   past problem restores everything -- equations, verification, steps,
   scenarios -- with no new LLM calls.
9. **Export** -- download any solved problem as Markdown (LaTeX equations
   included) or a typeset PDF (equations rendered via matplotlib's
   mathtext -- no system LaTeX install needed). Any plot (2D line, 3D
   surface, feasible region, ODE solution curve, or recurrence sequence)
   has a "📸 Include this plot in the report" button -- the interactive
   version only lives in the browser session, so this opts a specific view
   (with whatever parameter values were selected at the time) into the
   exported document, with a caption spelling out exactly which values
   were used. Static images are re-rendered via matplotlib rather than a
   screenshot of the Plotly figure, deliberately avoiding `kaleido`: as of
   Plotly's current
   release, `kaleido>=1.0` requires a separately-installed Chrome browser,
   which conflicts with this app's "pip install and go" portability goal
   the same way requiring a system LaTeX install would have.

## Extending it

- Swap Streamlit for a desktop shell (e.g. `pywebview` wrapping the same
  Streamlit app, or a PyQt front end calling the same `modules/`) if you want
  a native window instead of a browser tab -- the `modules/` package has no
  Streamlit dependency, so it's reusable as-is.
- The verification tolerance, retry count, and temperatures are all in
  `config.py` -- or tune them live from the app's "⚙️ Advanced settings"
  sidebar expander without restarting (see below).
- `modules/ode_utils.py` centralizes ODE-solving (used by both
  `solver.py` and `verifier.py`) specifically to avoid a circular import
  between those two -- keep that pattern in mind if you add another
  cross-cutting solve step. It also handles coupled SYSTEMS of ODEs (e.g. a
  decay chain A -> B): `group_coupled_odes()` groups ode-kind equations by
  shared function names, and equations that turn out to be coupled are
  solved together via `dsolve_system` rather than one at a time.
- **Dimensional checking substitutes fresh placeholder quantities, not the
  raw parsed unit, per distinct symbol** (`make_dimension_placeholder()` in
  `units_checker.py`). This mattered in practice: substituting the same
  canonical unit object (e.g. `u.meter`) for two DIFFERENT symbols that
  happen to share a unit causes SymPy to treat them as literally
  interchangeable, so checking something like `a - b` (both in meters)
  would silently collapse to a bare `0` before the dimension was ever
  computed -- correctly reporting "dimensionless" instead of "length" and
  false-failing the check. A coupled ODE system (`k1*A - k2*B`, where both
  rate constants and both functions happened to share units) hit exactly
  this during development. Each distinct symbol now gets its own
  uniquely-named placeholder with the right dimension, so same-dimension
  terms still validate correctly but don't falsely cancel.
- `modules/plot_snapshot.py` is the static (matplotlib) counterpart to
  `plotter.py`'s interactive Plotly figures -- kept as a separate module
  since they serve different purposes (one for the live browser session,
  one for exported documents) and deliberately don't share a rendering
  path, so a change to the interactive figures can't accidentally break
  what gets embedded in a report, or vice versa.
- Currently unsupported: nonlinear coupled ODE systems (true predator-prey
  dynamics, for instance) mostly have no closed-form solution even in
  principle -- `dsolve_system` will fail on those and the app reports it
  honestly rather than falling back to numeric integration (a further
  scope decision, not free); and multi-variable inequality regions are
  only visualized in 2D (pick any two free variables as axes; more than
  two requires fixing the rest via sliders, same pattern as the 3D surface
  plot for equations).

## Advanced settings (in-app)

The sidebar's "⚙️ Advanced settings" expander exposes the knobs that would
otherwise only be editable in `config.py`, live, without a restart:

- **Extraction / narration temperature** -- how much freedom the model has
  when converting text into equations vs. writing step explanations.
- **Max verification retries** -- how many times to re-prompt with the
  failure reason before giving up.
- **Numeric balance tolerance** -- how close a residual must be to zero to
  count as "balances."
- **Independent cross-check tolerance** -- how far the derived answer and
  the independent re-solve can disagree before verification flags it.

A "Reset to defaults" button restores `config.py`'s original values.
Settings apply to the *next* problem you solve, and persist for the
lifetime of the running app process (they're not saved back to disk).

## Running the tests

A pytest suite covers the extraction/verification/solving core -- the same
mocked-client pattern (`tests/conftest.py`'s `FakeClient`) used throughout
development, so no live LM Studio server is needed to run it:

```bash
pip install -r requirements-dev.txt
pytest
```

Coverage includes: all three equation kinds (algebraic, inequality, ODE)
end-to-end through extraction -> verification -> solving -> export;
the confidence/margin system; the JSON-extraction robustness helpers
(fenced code blocks, prose-wrapped JSON, truncated/invalid JSON); the
dimensional-unit checker (including the "unrecognized unit silently
mis-parsed" bug caught during development); workspace rename validation;
and the history save/load/delete round trip. `tests/conftest.py` has the
sample payloads and fixtures if you want to add more.