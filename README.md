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

The LLM translates and narrates; SymPy does and checks the math. The full design
is in [ARCHITECTURE.md](ARCHITECTURE.md).

1. **Extraction.** The reasoning model turns the problem text (or a photo) into
   JSON, validated against a pydantic schema: variables, relations, and an
   optional objective. Each relation is an **equation** (`Piecewise` allowed),
   **inequality**, **ODE** or **recurrence**, and each kind is parsed, verified
   and solved differently.
2. **Verification.** SymPy checks every relation independently, and each check
   reports how close it came to its tolerance (*essentially exact* vs
   *borderline*), not just pass/fail:
   - equations balance numerically; inequalities hold for the given numbers;
   - ODE and recurrence solutions satisfy the original equation (coupled
     systems are solved together);
   - optimisation results have zero gradient and the right min/max/saddle
     classification, plus a feasibility check against inequality constraints
     (full KKT analysis is out of scope, and the app says so);
   - every relation passes a physical-unit consistency check;
   - for algebraic equations, a second, independent LLM call re-solves the
     problem from scratch. If it disagrees with SymPy by more than about 2%, the
     derivation is retried (up to `max_verification_retries`) with the
     discrepancy fed back to the model.
3. **Solution steps.** SymPy computes the steps (substitute, isolate, simplify;
   `dsolve` plus initial conditions; `rsolve`; calculus or Lagrange multipliers),
   and the LLM only narrates them. If `sp.solve` finds no closed form, a clearly
   labelled numerical fallback is used instead.
4. **Scenarios.** A separate, low-stakes LLM call suggests other real-world
   contexts with the same mathematical structure.

Solved problems are saved to a local SQLite history and can be exported as
Markdown, PDF, Python, or a Jupyter notebook.

### What else it does

**Solving and checking**
- **Matrix systems** (`matrix_utils.py`): a genuine linear system is also shown
  as `A x = b`, with its determinant, eigenvalues, and a rank-based verdict
  (unique, infinitely many, or inconsistent).
- **Vectors** (`vector_utils.py`): variables can be vectors, used through `dot`,
  `cross`, `magnitude`, `unit`, `angle_between`, `distance` and `Point`.
- **Numerical fallback** (`numerical_fallback.py`): approximate roots for
  equations with no closed form, labelled as approximations. One equation, one
  unknown.
- **Physical-validity filtering** (`physical_validity.py`): a variable can declare
  a domain (`nonnegative`, `positive`, ...), and non-physical roots are discarded
  with a visible step. Opt-in.
- **Domain of validity** (`domain_utils.py`): finds where a formula is undefined
  (denominators, even roots, logs, `asin`/`acos`) and fails verification if the
  given values hit one.
- **Advisory checks**: physical plausibility (`plausibility.py`) and
  significant-figure discipline (`sig_figs.py`) flag results worth a second look.
  They never fail verification.
- **Confidence report**: all individual checks aggregated into a category-grouped
  score, capped below 0.5 as soon as any check fails outright.
- **Paranoid mode** (`paranoid.py`): re-runs extraction through a second model
  (`secondary_reasoning_model`) and compares the two derivations. Off by default.
- **Self-consistency** (`self_consistency.py`): re-extracts with the same model
  2-5 times, comparing equation shapes and the numeric answers. Disagreement
  usually means the problem statement is ambiguous.
- **Other aids**: unit-conversion sweep (`unit_conversion.py`), algebra-technique
  tagging, a "show me another way" method (back-substitution, Cramer's rule),
  named-formula recognition, and Python/Jupyter code export.

**Uncertainty and sensitivity**
- **Sensitivity / tornado** (`sensitivity.py`), first-order **error propagation**
  (`uncertainty.py`, `error_propagation.py`), **Monte Carlo** (`monte_carlo.py`,
  seeded and reproducible), **interval arithmetic** (`interval_arithmetic.py`,
  guaranteed bounds), and **goal seek** (`goal_seek.py`, the inverse solve).
- **Bulk analysis**: N-dimensional parameter sweeps (`parameter_sweep.py`) and
  Monte Carlo across every target at once. These solve symbolically once and
  evaluate vectorised.

**Modes** (chosen in the sidebar; `Ctrl/Cmd+K` is a command palette)
- **Word problem solver**, **Quick start** (example gallery), and **Batch solver**
  (pasted or PDF problem sets; results as Markdown, PDF, CSV or Excel).
- **Curve fitting** (`curve_fitting.py`, `statistical_inference.py`): linear,
  polynomial, exponential, power, logarithmic and custom linear-in-parameters
  models, with R², RMSE, residuals, confidence intervals and hypothesis tests.
- **Check equivalence** (`equivalence.py`, `proof.py`): symbolic equivalence with
  sampled evidence when inconclusive, and the actual simplification passes as a
  proof.
- **Problem chains** (`chains.py`): named, persistent sequences where one step's
  output feeds the next and everything downstream re-solves, like spreadsheet
  cells. Includes sweeps across a chain.
- **Geometry**, **PDE solver** (heat, wave, Laplace, first-order), **Tensor
  calculus**, **Transforms & series** (Laplace, Fourier, Taylor, Laurent,
  asymptotic, Fourier series), **Dimensional analysis** (Buckingham-Pi-style
  exploration from units alone), **Extraction diff**, **Compare solves**
  (`solve_compare.py`: two solves, or a what-if of one with some inputs changed, lined
  up on answers, inputs, equations and verification; no LLM call), **Quantum
  mechanics** (below), **General relativity** (below), **Statistical mechanics** (below), **Numerics & verification lab** (below), and **Research journal**.

**Plots and animations**
- Interactive 2D line, 3D surface, contour, feasible-region, ODE, recurrence and
  curve-fit plots, with log axes. Plot target selectors solve the equation for the
  chosen variable.
- Time-resolved views, each with Play/Pause and a GIF or PNG export:

  | View | Shows |
  |---|---|
  | Closed form vs numerical integration | the ODE solution and an independent integration over time, with relative error against the verifier's tolerance |
  | Phase-portrait flow, time-linked view | points flowing along the direction field with trails; one time cursor driving the series and the phase plane |
  | Uncertainty fan | median and percentile bands from sampled parameters or initial values, plus a guaranteed interval-arithmetic envelope |
  | Parameter morph | the whole solution family as one parameter varies, with visible turning points counted |
  | Bifurcation diagram, cobweb | long-run values of a one-parameter map (no closed form needed) |
  | PDE evolution | the profile u(x, t) beside a heatmap of the whole evolution |
  | Series convergence | Taylor and Fourier partial sums added term by term |
  | Solve-order replay | the dependency graph lit up in the order quantities become determined |
  | Chain value flow | each chain step and the value it passed on |
  | Motion diagram | velocity and acceleration arrows, with a strobe trail |

  The solve-order replay shows the order implied by the *dependencies*, not a
  trace of the solver's own internal steps.

**Learning and practice**
- **Grade my work** (`grading.py`): checks a student's formula, arithmetic and
  final answer separately, from typed text or a photo, and tracks recurring
  mistake patterns. **Worksheet variants** (`worksheet.py`) generate new problem
  text, which is solved through the normal pipeline rather than trusting the LLM's
  own answer.
- **Tutor mode**, per-step **"explain just this"**, grounded **follow-up Q&A**
  (numeric what-ifs are computed by SymPy, not the LLM), **similar past problems**
  (by equation shape, not wording), and a **concept index**.

**Sessions, data and access**
- **History**, **templates**, **settings profiles** and the **variable workspace**.
  **Project bundle** export/import moves everything between machines.
- **`cli.py`** runs batch solves and Monte Carlo without Streamlit. A **REST API**
  (`api_server.py`) covers `/solve`, `/fit`, `/equivalence` and
  `/dimensional-analysis`. Both talk to LM Studio the same way the app does.
- **Export:** one **Share / export** button opens a dialog. Choose what goes in (a
  preset -- *Full audit*, *Student handout*, *Peer review*, *Quick summary* -- or tick
  sections: problem, results, equations, steps, sensitivity and Monte Carlo, plots,
  follow-up Q&A, tutor practice, reproducibility ...) and click a format; the file is
  built on that click. Formats: **PDF**, an **interactive HTML** report (one offline
  file, live zoomable plots, click an equation to copy its LaTeX), **Word**,
  **PowerPoint** (one slide per worked step, explanations in the speaker notes),
  **LaTeX** source, and **Markdown**. Every export can carry a reproducibility
  footer (model, settings, library versions, Monte Carlo seeds). The dialog also has
  copy-ready LaTeX and plain-text summaries and "send to a chain". Plots go into a
  report through an explicit "Include this plot" button. Static images deliberately
  avoid `kaleido`, whose current releases need a separate Chrome install.
- **Numerics & verification lab** (`🧰` mode; no LLM involved). The cross-cutting tools that apply to every field. As everywhere in the
  app, results come with the checks that were run on them.
  - *Conserved quantities*: type an ODE system and the app searches for conserved quantities (polynomials, optionally with ln y terms,
    or your own basis functions), verifies each by exact differentiation (dQ/dt ≡ 0), and measures the **drift along a numerical solution**,
    classified as *bounded oscillation* or *secular drift* — the distinction that matters. It finds x²+y² for an oscillator, the
    Lotka–Volterra invariant, SIR's population sum and the two invariants of Euler's rigid body, and correctly finds nothing for a damped system.
    The same monitor runs **automatically on the ODE results page** for any autonomous coupled system, and on the wavepacket (norm,
    energy, momentum) and geodesic (energy, angular momentum, norm, Carter constant) simulations. **Symplectic integrators**
    (leapfrog, Yoshida 4th order, implicit midpoint) are compared with RK4 and RK45 on the same problem: bounded energy error against steady drift.
  - *Convergence*: automatic refinement with the **observed order** (from successive differences, so no exact answer is needed),
    Richardson extrapolation and a grid-convergence index. Ready-made studies for ODE methods (Euler 1, Heun/midpoint 2, RK4 4),
    Hamiltonian integrators, the app's finite-difference Poisson solver (against a manufactured solution), the split-step wavepacket
    solver (second order in dt) and Euler–Maruyama / Milstein strong convergence (1/2 and 1) — every method's measured order matches theory.
  - *Hamiltonian / Lagrangian*: type L(q, q̇) or H(q, p) and get the Euler–Lagrange equations, the Legendre transform, Hamilton's equations,
    and **conserved quantities from Noether's theorem**: cyclic coordinates, energy, and an automatic search over translations, rotations, Galilean
    boosts and gauge terms (it finds the angular momentum of the Kepler problem and the m·ẋ + m·g·t charge of uniform gravity). Phase
    portraits for one degree of freedom and **Poincaré sections** for two (Hénon–Heiles: regular curves at E = 1/12, chaos near 1/6).
    Self-checks include Legendre consistency, H and the Noether charges conserved along a trajectory, and symplecticity of the flow (JᵀΩJ = Ω).
  - *Nondimensionalise*: the scale-free form of an ODE or PDE (`m*x_tt + c*x_t + k*x = F*sin(w*t)`), with scales built from the parameters
    (dimension-checked), the leftover dimensionless groups named and counted against Buckingham's π bound, an exact symbolic round trip,
    and, for ODEs, a **numerical equivalence check** (the original with numbers and its dimensionless form give the same solution).
    A variable no parameter can set the scale of gets an explicit reference scale, and says so.
  - *Limits & large N*: quantum → classical (ħ → 0: Planck → Rayleigh–Jeans, equipartition, WKB tunnelling), relativistic → Newtonian
    (c → ∞), and large-N rates (Stirling 1/12N, de Moivre–Laplace N^(−1/2), harmonic numbers, the exponentially fast 1D-Ising
    thermodynamic limit), using the series/asymptotics module: the leading term must BE the classical result, the first correction must match, and
    a 50-digit numerical sequence must approach it at the rate the series predicts.
  - *Cross-validation*: integral, derivative, roots, ODE, sum, limit and eigenvalue tasks, each with its SymPy closed form and an independent
    numerical route side by side (agree / DISAGREE / numeric only / symbolic only); a twelve-case demonstration set includes one with no closed form.
  - *Reproducible notebook*: export a computation (partition function, lattice scan, Einstein relation, random matrices, geodesic precession,
    integrator comparison, convergence, nondimensionalisation, cross-validation, limit check) as a `.ipynb` or `.py` that re-runs it from its
    **parameters and seed** and then **asserts it reproduces the numbers recorded at export**; it fails loudly if they differ. Library versions are recorded.
- **Statistical mechanics** (`🔥` mode; no LLM involved; k_B = 1). Six tools; every result is shown with the checks that
  were run on it, and nearly every check compares the method with an exact result it shares no code with.
  - *Partition functions*: Z, F, U, S and C against temperature for a two-level system, harmonic oscillator, rigid
    rotor, explicit levels, or any E_n and g_n you type (infinite spectra are truncated where the sum has converged,
    and a spectrum whose partition function diverges is refused). The Gibbs entropy −kΣp ln p must equal (U−F)/T,
    U and C must equal finite-difference derivatives of ln Z, S → k ln g₀ as T → 0, the numerical sum must match the
    closed form where one exists, the rotor matches its high-T series, and the Schottky peak is located two ways.
    The ideal gas shows the Gibbs paradox: entropy is extensive only with the 1/N! factor.
  - *Lattice Monte Carlo*: Metropolis for Ising, q-state Potts and XY on an L×L lattice (checkerboard updates, vectorised),
    with a live lattice animation. Error bars carry the **integrated autocorrelation time** (Sokal windowing) and
    χ', C and the Binder cumulant are jackknife estimates; the page warns when a point has fewer than 50τ of data
    or the first and second halves disagree. A scan over temperatures and sizes gives the Binder crossing (T_c) and
    finite-size-scaling exponents (γ/ν, β/ν). Verified against the **exact** 4×4 Ising lattice (all 65 536 states: e, ⟨|m|⟩,
    χ', C, U₄ within their errors) and against an exact transfer matrix for the Potts torus (q = 3, 4).
  - *Mean field vs exact*: mean field, the Bethe approximation, exact 1D (transfer matrix) and Onsager's exact 2D solution on
    one set of axes, with the failures computed rather than asserted: T_c (4J against 2.269J), β (1/2 against 1/8), a
    specific-heat *jump* against a logarithmic divergence, no short-range order above T_c. The Onsager energy is obtained
    twice (closed form with the elliptic integral, and as d(βf)/dβ of the free-energy integral) and Bethe with z = 2 must
    reproduce the exact chain. Optional Monte Carlo overlay.
  - *Distributions*: Maxwell–Boltzmann, Fermi–Dirac, Bose–Einstein; chemical-potential solvers at fixed density; Bose–Einstein
    condensation (μ sticks at 0, condensate fraction 1 − (T/T_c)^(3/2)); the classical limit checked two ways
    (occupations → MB with deviation ∓e^(−y), second virial coefficient ±2^(−5/2)); the degenerate Fermi limit against the
    Sommerfeld expansion. The solvers use fast integral forms; the polylogarithm is the independent reference.
  - *Stochastic dynamics*: overdamped (Euler–Maruyama) and underdamped (BAOAB) Langevin in a free, harmonic, double-well or
    typed potential; MSD against its exact ballistic-to-diffusive crossover; the **Einstein relation / fluctuation–
    dissipation theorem measured** (D from a force-free ensemble, μ from the drift under a force, D/μ = kT; breaking the
    noise–friction tie makes it fail); Green–Kubo; a Fokker–Planck solver (Scharfetter–Gummel fluxes, Crank–Nicolson)
    compared with the Langevin ensemble and, for the harmonic trap, with the exact Gaussian at every time.
  - *Random matrices*: GOE, GUE, Poisson and Wishart. Eigenvalue density against Wigner's semicircle (moments are the Catalan
    numbers; the second moment is checked against its exact finite-N value) and Marchenko–Pastur; unfolded nearest-neighbour
    spacings classified against the GOE/GUE surmises and Poisson; the surmise is tested as *exact* for 2×2 matrices; the
    ratio statistic ⟨r̃⟩.
  Every stochastic result stores the **seed** that produced it (a fresh one is drawn when the seed box is empty), so any run can
  be reproduced by typing it back in.
- **General relativity** (`🌌` mode; no LLM involved; G = c = 1, signature (−,+,+,+), MTW sign convention). Six
  tools around the metric tensor, each shown with the checks that were run on it:
  - *Curvature & fields*: Christoffel symbols → Riemann → Ricci → Einstein → Weyl for a preset (Minkowski,
    Schwarzschild, Reissner–Nordström, Kerr, Schwarzschild–de Sitter, flat FRW, de Sitter, Eddington–Finkelstein)
    or any metric you type. Four invariants (R, R_abR^ab, Kretschmann, Weyl²); the field equations
    G_ab + Λg_ab = 8πT_ab checked for vacuum, a custom T_ab, or an electromagnetic field F_ab (Reissner–Nordström
    solves Einstein–Maxwell, a wrong charge does not); an effective-fluid reading that returns the Friedmann
    equations for FRW; and a *singularity classifier* that finds where metric or invariants blow up and tells a
    curvature singularity (Schwarzschild r = 0, the Kerr ring, the big bang) from a coordinate artefact (r = 2M,
    the polar axis, the Kerr horizons). Every run checks identities valid for any metric (∇g = 0, Riemann
    symmetries, the contracted Bianchi identity, traceless Weyl, K = C² + 2Ric² − R²/3). A full Kerr tensor takes
    about half a minute.
  - *Geodesics*: any metric; integrate from coordinate velocities or, for (t, r, θ, φ) metrics, from E and L.
    Energy, angular momentum, the norm, total L² (spherical symmetry) and the **Carter constant** (Kerr) are
    monitored and their drift reported. Dedicated tabs compare an integrated Schwarzschild orbit with the *exact*
    elliptic-integral periapsis advance and an integrated light ray with the *exact* deflection integral (both
    agree to ~1e-10). Animated 2D path, 3D view, Stop button.
  - *Potentials*: V_eff(r) for several L with horizons, photon sphere, ISCO and marginally bound orbit marked,
    live as M, Q, Λ or the Kerr spin change; any f(r) you type; Kerr's closed-form radii cross-checked against
    numerical circular-orbit analysis.
  - *Coordinate changes*: Schwarzschild → Eddington–Finkelstein / isotropic / Kruskal–Szekeres, Minkowski →
    Rindler / spherical, de Sitter → conformal time, or your own map (either direction). The curvature
    invariants must be unchanged, and the Ricci tensor must transform as a tensor, which is checked against the
    engine's direct computation.
  - *Cones & Penrose*: light cones and null rays in any coordinate plane (Schwarzschild coordinates pinch the
    cones at r = 2M, Eddington–Finkelstein ones tilt through it); Penrose diagrams for Minkowski and the
    maximally extended Schwarzschild spacetime (there is no general algorithm for other metrics, and the app says
    so instead of guessing).
  - *Gravitational waves*: TT plane waves in any direction, verified against the linearised vacuum equations
    (an off-shell wave is correctly reported as not a solution; the linear Ricci tensor is also checked against
    the first-order part of the full nonlinear one); an animated ring of test masses; the quadrupole formula for
    a circular binary or your own point-mass trajectories (symbolic, compared with the textbook amplitudes and
    Peters' power); and an inspiral chirp integrated numerically against its closed form.
- **Quantum mechanics** (`⚛️` mode; no LLM involved; ħ = m = 1 unless you change them). Five tools, each
  of which checks its own result and shows what it checked:
  - *Eigenstates*: the 1D Schrödinger equation for a library of potentials (box, oscillator, finite well,
    Morse, Pöschl–Teller, double well, hydrogen radial) or a formula you type. The energy ladder with
    wavefunctions, a convergence study (the grid order is measured on three grids, not assumed), and
    comparison with the analytic spectrum where one exists.
  - *Wavepacket*: a Gaussian packet evolved by split-step Fourier and, independently, Crank–Nicolson, with
    an animation. Tunnelling is compared with the *exact* transfer-matrix result averaged over the packet's
    momentum distribution; norm, energy and Ehrenfest's theorem are checked.
  - *Operators*: commutators, anticommutators and identities (exact), eigen-decompositions, the
    Heisenberg–Robertson and Schrödinger–Robertson uncertainty relations on a state, and the
    Baker–Campbell–Hausdorff expansion to fifth order with its error exponent measured. Spin-j,
    Pauli and (truncated) oscillator operators are built in; so are two matrices of your own.
  - *Qubit*: gate sequences on the Bloch sphere (animated), amplitude damping, dephasing and depolarising
    noise (exact propagation, cross-checked against the Lindblad master equation), purity and fidelity,
    density matrices, and T1/Ramsey measurements.
  - *Perturbation*: Rayleigh–Schrödinger corrections through third order in exact arithmetic (E¹ = 3/4,
    E² = −21/8, E³ = 333/16 for the quartic oscillator's ground state), compared with exact diagonalisation;
    the measured error slope per order is what catches a wrong coefficient.
- **Copy buttons:** each answer, and each Monte Carlo summary, has a 📋 popover to copy
  it as plain text, with units, as LaTeX, or as Python (a value, or the function).
- **Follow-up questions and tutor-mode guesses are saved with the problem**, so they
  survive reopening it and appear in exports.

**Reliability**
- Every SymPy-heavy call has a configurable timeout (default 10 s). A thread
  cannot be killed, so a timed-out computation is abandoned rather than stopped,
  and at most 8 may be running before new calls are refused with a clear message.
- Dependencies are pinned to the versions the suite has been run against. SQLite
  runs in WAL mode, prunes history to the latest 100 records, and closes its
  connections. Uploads are capped at 500 MB. Recurring failures go to a rotating
  log (`data/app.log`). Details are in
  [ARCHITECTURE.md](ARCHITECTURE.md#design-notes).

### Interface

- A solved problem shows its secondary panels in three tabs: **Verify**,
  **Explore** and **Practice**. The core content (equations, steps, ODE and
  recurrence solutions, follow-up Q&A) stays below them.
- Mode navigation, the active chain and recent error patterns live in the
  sidebar. The LM Studio connection block is collapsed unless there is a problem.
- Phone-friendly: camera capture for both photo inputs, and compact
  `st.data_editor` tables instead of one widget per variable.
- Moving a widget only rebuilds what it feeds (`ui/cache.py`), and the heavy analysis
  panels (Monte Carlo, sweeps, sensitivity, the interactive plots, the time views) are
  isolated `st.fragment`s that rerun on their own.
- **Plots you can click:** a tornado bar can be swept in detail, added to the
  N-dimensional sweep or given a Monte Carlo uncertainty; a sweep-heatmap point can be
  loaded into the Variables panel or opened in Compare; a dependency-graph node shows
  its steps.
- **Long runs show live progress** (Monte Carlo, batch solves, PDE solves) with a
  **Stop** button that takes effect at the next checkpoint. A single SymPy or numpy call
  cannot be interrupted, so Stop acts between chunks of work, not mid-calculation.
- **Reopening a past problem restores how you were exploring it** (sample counts and
  seeds, sweep setup, plot axes and sliders). Opening or solving a problem starts from
  that problem's own settings rather than the previous one's.
- Each step's "Explain just this step" is a popover rather than an expander.

## Extending it

- **A new sidebar mode:** add a `PaletteEntry` to `modules/command_palette.py`'s
  `_ENTRIES` (it builds `MODE_LABELS`, which the sidebar radio uses), write the
  page as `ui/<name>.py`, and add it to `ui/__init__.py`'s `PAGES`.
  `tests/test_app_modes.py` fails if either half is missing.
- **A different front end:** `modules/` has no Streamlit dependency, so a desktop
  shell or another UI can call it as-is.
- **Tuning:** verification tolerance, retry count and temperatures are in
  `config.py`, or live in the sidebar's "⚙️ Advanced settings".
- **Known limits:** nonlinear coupled ODE systems (predator-prey, say) mostly have
  no closed form, so `dsolve_system` fails and the app says so rather than
  integrating numerically. Inequality regions are only plotted in 2D. A
  `solve_for` target can't be a vector variable; solve for a component, or for a
  scalar equation defined with `dot`/`cross`/`magnitude`.

## Advanced settings (in-app)

The sidebar's "⚙️ Advanced settings" expander changes these live, without a
restart, for the *next* problem you solve (they aren't saved to disk, but can be
saved as a named profile): extraction and narration temperature, maximum
verification retries, numeric balance tolerance, independent cross-check
tolerance, and the computation timeout. "Reset to defaults" restores
`config.py`'s values.

## Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

<<<<<<< HEAD
The suite (about 2,900 tests; roughly seven minutes on a single core, far less in parallel) uses a
mocked LLM client (`tests/conftest.py`'s `FakeClient`), so no LM Studio server is needed. Use `pytest -n auto` to
spread it over every core, and `pytest -m "not slow"` to skip the four engine tests that take about 10 s or more
=======
The suite (about 2,900 tests; roughly eight minutes on a single core with coverage, far less in parallel) uses a
mocked LLM client (`tests/conftest.py`'s `FakeClient`), so no LM Studio server is needed. Use `pytest -n auto` to
spread it over every core, and `pytest -m "not slow"` to skip the five engine tests that take more than ~8 s each
>>>>>>> d95db13d68c97241e08eb7228a86210f7f019305
(Kerr curvature, the large Monte Carlo and finite-size-scaling runs).

- **CI** (`.github/workflows/tests.yml`) runs on every push and pull request:
  **ubuntu-latest and windows-latest** x Python **3.12 and 3.14** (the oldest and
  newest supported) x two hash seeds. Windows is included deliberately, since this
  app targets it and some code (the timeout wrapper) exists because the obvious
  approach, `signal.alarm`, does nothing there. CI also byte-compiles the project
  and runs `mypy modules/ ui/ app.py cli.py config.py api_server.py`. One cell (ubuntu, 3.12,
  seed 0) runs the full suite with coverage; the other seven skip the `slow` tests. All use `-n auto`.
- **Coverage floor:** `pytest` fails below 90% line coverage of `modules/`
  (measured: mid-90s). It exists to catch new code with no tests at all, not to
  chase 100%. Coverage uses `sys.monitoring` (`core = "sysmon"`), which reports the
  same lines as the default tracer at about a third of the overhead.
- **UI tests** (`tests/test_ui_smoke.py`) drive the real `app.py` through
  Streamlit's `AppTest`, with the SQLite databases redirected to a temp directory.
  Booting a page is slow, so read-only checks share one boot (`_shared_*_page`,
  never touched with a widget), a page's controls are tested together, and GIF
  buttons use the `gif_stub` recorder; the renderers themselves are tested for
  real elsewhere.
- **Hypothesis deadlines are disabled** (`tests/conftest.py`): SymPy's first call
  on a new expression is several times slower than later ones, so the default
  deadline failed at random without meaning anything.
- **Pre-commit hook** (optional, `.pre-commit-config.yaml`): runs the suite before
  each commit. It takes about two minutes, so you may prefer
  `pre-commit install --hook-type pre-push` with `stages: [pre-push]`. Skip it
  once with `git commit --no-verify`.
