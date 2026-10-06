# Architecture

## Design principle

The LLM is treated as a **translator and narrator**, never as the sole
authority on correctness. Every number in the final answer traces back to
SymPy, a deterministic symbolic math engine. The LLM's outputs are only
trusted after they've been checked two independent ways -- structurally
(does the math even parse and balance) and empirically (does an independent
re-derivation of the original word problem agree numerically). This is the
answer to "how does it check its own work" -- it isn't the same model
grading its own homework, it's a second, blind attempt compared numerically
against the first.

## Data flow

```
Input (text | image)
   │
   ▼
[extraction: LM Studio reasoning model, temp=0.1, JSON-schema constrained,
   response validated against llm_schema.py's pydantic model before use]
   │  -> {domain, variables[], equations[], solve_for, assumptions[]}
   ▼
[equation_engine.build_model]  -- parses each equation string into a
   │                               SymPy Eq() object; records parse errors
   ▼
[verifier.verify]
   ├─ structural checks (parsing, target-symbol presence, determinacy)
   ├─ numeric balance check (substitute all-known equations, must reduce to 0)
   └─ independent cross-check:
        - SymPy solves the derived system for `solve_for`          -> A
        - LLM re-solves the ORIGINAL word problem from scratch     -> B
        - |A - B| / |A| < 2%  required to pass
   │
   ├─ FAIL -> retry extraction with failure_reason appended to the prompt
   │           (bounded by config.max_verification_retries)
   ▼
[solver.compute_steps]  -- deterministic SymPy trace: state equation ->
   │                        substitute knowns -> isolate target -> simplify
   ▼
[solver.narrate_steps]  -- LLM explains (does not compute) each already-
   │                        verified step, one sentence per step
   ▼
[scenarios.generate_alternative_scenarios] -- separate LLM call, given only
   │                        the final verified equations, suggests other
   │                        domains with the same structure
   ▼
Streamlit UI
   ├─ LaTeX-rendered equations + derivation text
   ├─ verification report (pass/fail per check, both numeric answers shown)
   ├─ editable variable panel
   ├─ step-by-step accordion with narration
   ├─ "extract to workspace" -> stores solved value in session_state for
   │   reuse in a later, unrelated problem
   └─ live Plotly plot: pick a free symbol as x-axis, others become sliders,
       equation is solved symbolically for the target and lambdified with
       numpy for fast redraw on every slider move
```

Everything from extraction through scenario generation is one function,
`modules/pipeline.run_pipeline()` -- both the interactive app (ui/word_problem.py)
and batch mode (modules/batch_solver.py) call it, so there is exactly one
implementation of "extract, verify, retry with the failure fed back, compute
steps" rather than two copies that could drift. See that module's own
docstring for what's deliberately NOT routed through it (cli.py and the
problem-chains tab, which both want a single un-retried attempt).

## Why these specific tool choices

- **Streamlit over Flask/Django+JS, or PyQt/Tkinter**: one Python file runs
  as a full interactive app with no separate frontend build, and it's
  identical across OSes -- directly serves the "portable, Python-first"
  requirement. `modules/` has zero Streamlit imports, so if a native desktop
  shell is wanted later (PyQt, pywebview), the math/LLM logic is reused as-is.
- **LM Studio's OpenAI-compatible server**: means the official `openai`
  python SDK works unmodified by just repointing `base_url` -- no bespoke
  client, and the same code would work against any other OpenAI-compatible
  local server (Ollama's compat endpoint, vLLM, etc.) with a one-line config
  change.
- **SymPy over trusting LLM arithmetic**: LLMs are unreliable at multi-step
  algebra/arithmetic; SymPy is exact. Splitting "propose the model" (LLM)
  from "solve/verify the model" (SymPy) plays to each one's strength.
- **Plotly over Matplotlib for the interactive plot**: Streamlit's
  slider-triggered rerun + Plotly's fast redraw gives smooth "real-time"
  feeling adjustment without any custom JS callback code.

## File map

```
math-rep-system/
├── app.py                    # Streamlit entrypoint -- thin: page config, session
│                              #   defaults, sidebar, mode dispatch. The pipeline and
│                              #   every page's actual UI now live in ui/ (below);
│                              #   this file is deliberately small enough to read in
│                              #   one sitting. Draws the title and a first-load notice
│                              #   BEFORE importing the heavy stack (the LLM client,
│                              #   workspace, sidebar, pages), so the page shows
│                              #   something immediately instead of staying blank while
│                              #   those imports finish.
├── cli.py                     # non-interactive: `python cli.py solve "..."` -- a single
│                              #   extract -> verify attempt with NO retry loop (reports
│                              #   the first attempt honestly rather than re-prompting)
├── api_server.py               # FastAPI REST surface: /solve, /fit, /equivalence,
│                              #   /dimensional-analysis -- deliberately NOT covering
│                              #   every one of the Streamlit app's modes, see its
│                              #   own module docstring for which and why
├── config.py                 # LM Studio endpoint/model settings, tunables
├── requirements.txt
├── requirements-dev.txt      # + pytest, pre-commit (dev/test only)
├── requirements-api.txt      # + fastapi/uvicorn, for api_server.py only
├── .github/workflows/tests.yml  # CI -- runs the suite on every push/PR,
│                              #   ubuntu-latest AND windows-latest x Python 3.12/3.14
├── .pre-commit-config.yaml   # optional local hook: runs the suite before each commit
├── .streamlit/config.toml    # server.maxUploadSize=500 (MB) -- committed despite
│                              #   .streamlit/ being gitignored (a narrow exception
│                              #   carves this one file out; secrets.toml etc. stay ignored)
├── README.md                 # setup + run instructions
├── ARCHITECTURE.md           # this file
│
├── ui/                        # Streamlit front end -- pages and rendering only, no
│   │                          #   math/LLM logic of its own. See ui/__init__.py's own
│   │                          #   docstring for the full package map.
│   ├── __init__.py             # PAGES: mode label -> page function (the dispatch table
│   │                          #   app.py falls through to), built lazily -- a page's own
│   │                          #   module is only imported the first time its mode is
│   │                          #   selected, so opening the app doesn't pay for every
│   │                          #   page's imports up front; command_palette.MODE_LABELS
│   │                          #   is the actual source of truth for the mode list itself,
│   │                          #   with tests/test_app_modes.py enforcing the two can't drift
│   ├── common.py                # helpers shared by several pages (upload-size guard,
│   │                          #   snapshot/download buttons, query-param syncing, ...)
│   ├── cache.py                 # per-session cache keyed by the CONTENT of a result's inputs, so
│   │                          #   a rerun rebuilds only what its widget changed (figures, ODE solves)
│   ├── fragments.py             # @isolated: a panel that reruns on its own (st.fragment), plus a
│   │                          #   registry of which panels those are (a test pins the list)
│   ├── actions.py               # buttons that set up OTHER widgets: queue the values in a callback,
│   │                          #   apply them at the top of app.py before any widget exists
│   ├── plot_select.py           # selectable_chart(): a Plotly figure whose points can be clicked
│   ├── progress.py              # tracked_run(): a long computation as a live status block with a
│   │                          #   Stop button, and notice_interrupted() for a run that was cut short
│   ├── view_state.py            # save/restore how a problem was being explored (with its history record)
│   ├── share.py                 # the Share / export dialog: downloads, copy-as-LaTeX, send to a chain
│   ├── compare.py, compare_constants.py   # the Compare solves mode (and the names panels link to it by)
│   ├── theme.py                   # visual design system: inject_base_styles() (cards,
│   │                          #   badges, button/spacing polish -- called once from app.py),
│   │                          #   render_hero(), badge()/badge_row(), and dark_mode_css()
│   │                          #   (the sidebar dark-mode toggle's CSS, kept here so all of
│   │                          #   the app's styling lives in one file)
│   ├── sidebar.py                # the whole left sidebar as one function
│   ├── word_problem.py            # the default page: input, Solve button, calls
│   │                          #   modules/pipeline.py, then ui/results/
│   ├── batch.py, chains.py, curve_fitting.py, dimensional.py, equivalence.py,
│   │   extraction_diff.py, geometry.py, journal.py, pde.py, quick_start.py,
│   │   tensor.py, transforms_series.py    # one module per standalone mode
│   └── results/                  # the results view for a solved problem, one function
│       ├── __init__.py             #   per section, called in display order by render_results()
│       ├── summary.py              # confidence banner, derived equations, variables,
│       │                          #   vector summary, follow-up Q&A, scenarios, ...
│       ├── steps.py                # the step-by-step section: step list, per-target answer
│       │                          #   extras, and the uncertainty/bounds/goal-seek/
│       │                          #   sensitivity expanders (split per feature)
│       ├── verify_tab.py, explore_tab.py, practice_tab.py, solutions.py
│
└── modules/                   # math/LLM logic -- zero Streamlit imports, so it's
    │                          #   reusable as-is from cli.py, api_server.py, or a future
    │                          #   non-Streamlit shell (PyQt, pywebview, ...)
    ├── pipeline.py              # THE shared extract -> verify -> retry -> steps ->
    │                          #   narrate -> scenarios sequence -- ui/word_problem.py
    │                          #   and batch_solver.py both call this; see its own
    │                          #   docstring for what's deliberately NOT routed through it
    ├── llm_client.py         # LM Studio (OpenAI-compatible) client wrapper -- the
    │                         #   connection check (is_available/list_models) does a
    │                         #   fast, hard-timeout TCP probe before ever making an HTTP
    │                         #   call, and uses a separate no-retry client for that call,
    │                         #   so "LM Studio isn't running" is reported in well under a
    │                         #   second instead of the several seconds the OpenAI SDK's
    │                         #   own connect-timeout/retry defaults would otherwise take
    │                         #   (those defaults are right for a real completion request,
    │                         #   wrong for a bare status ping)
    ├── llm_schema.py           # pydantic validation of the LLM's extraction JSON, at
    │                          #   the exact boundary before equation_engine.build_model
    │                          #   ever sees it
    ├── ocr.py                 # pytesseract fallback for non-vision models
    ├── equation_engine.py     # LLM extraction prompt + JSON -> SymPy parsing
    │                          #   (equations / inequalities / ODEs / recurrences,
    │                          #   plus an optional optimization objective --
    │                          #   see target_kind() for kind dispatch)
    ├── units_checker.py        # sympy.physics.units-based dimensional checks
    ├── ode_utils.py             # shared dsolve()/dsolve_system() helper (solver.py +
    │                            #   verifier.py both need it; lives here to avoid a
    │                            #   circular import)
    ├── db_util.py                 # ClosingConnection: `with _connect() as conn:` commits AND closes
    │                              #   (used by history, chains, templates, settings_profiles)
    ├── chain_flow.py              # a problem chain as a cascade: what each step received and carried on
    ├── time_uncertainty.py        # fan chart: sample uncertain parameters/initial values
    │                              #   through the symbolic solution + interval-arithmetic envelope
    ├── parameter_morph.py         # a family of solution curves as one parameter varies,
    │                              #   with visible-turning-point counting
    ├── bifurcation.py             # long-run values of a one-parameter map vs the parameter
    ├── pde_field.py               # u(x, t) on a grid (vectorised) + max|u| and integral per time
    ├── ode_trajectories.py        # the closed-form-vs-numerical ODE comparison as a curve
    │                              #   over time (shares prepare_ivp with ode_utils'
    │                              #   numerical_cross_check) + multi-start phase flow
    ├── recurrence_utils.py       # shared rsolve() helper, same circular-import reason
    ├── pde_utils.py                # partial differential equations: first-order PDEs,
    │                              #   heat/wave with Dirichlet/Neumann/Robin boundary
    │                              #   conditions, Laplace on a rectangle, and numerical
    │                              #   (finite-difference) fallbacks for both 1D and 2D
    │                              #   heat when no closed form exists
    ├── geometry_solver.py          # triangle solving (SSS/SAS/ASA/AAS/SSA -- including
    │                              #   the genuinely ambiguous SSA case, which returns
    │                              #   both valid triangles rather than picking one, plus
    │                              #   build_ssa_ambiguity_animation() morphing between them
    │                              #   via the shared "swinging compass" construction) --
    │                              #   also reachable from word-problem extraction via
    │                              #   ProblemModel.geometry, not just the standalone mode
    ├── tensor_calculus.py          # classical (index-based) tensor calculus on a
    │                              #   Riemannian manifold given a metric: Christoffel
    │                              #   symbols, curvature, covariant derivatives, and
    │                              #   index raising/lowering
    ├── transforms.py               # integral transforms (Laplace and Fourier, and
    │                              #   their inverses), each independently verified
    ├── series_asymptotics.py       # Taylor/Maclaurin/Laurent series expansions and
    │                              #   asymptotic expansions
    ├── series_animation.py         # cumulative partial sums of a Taylor result, and a
    │                              #   Fourier series by harmonic, for convergence animations
    ├── statistical_inference.py    # the statistics layer on top of curve_fitting.py:
    │                              #   parameter confidence intervals, hypothesis tests,
    │                              #   and related inference on a fitted model
    ├── matrix_utils.py            # A x = b representation + rank-based classification
    │                              #   (unique/infinite/inconsistent) + eigenvalues for
    │                              #   genuine linear systems (>=2 equations, >=2 shared
    │                              #   unknowns) -- an additional structural view, not a
    │                              #   separate solve path (sp.solve() still produces the answer)
    ├── vector_utils.py             # dot/cross/magnitude/unit/angle_between/distance/Point --
    │                              #   bound into equation_engine's local_dict so vector
    │                              #   quantities (forces, displacements) parse and reduce to
    │                              #   scalars via genuine sp.Matrix objects, not LLM pre-decomposition
    ├── curve_fitting.py            # sibling pipeline: CSV/pasted (x,y) data -> fitted symbolic
    │                              #   model + R²/RMSE/residuals. numpy-only (polyfit after
    │                              #   linearizing, or lstsq for linear-in-parameters custom
    │                              #   models) -- deliberately no scipy dependency
    ├── equivalence.py               # standalone "are these two expressions the same" utility,
    │                              #   built on sp.Expr.equals() with a numeric-sampling fallback
    │                              #   that reports WHERE two expressions agree/disagree
    ├── uncertainty.py               # first-order error propagation for algebraic targets --
    │                              #   re-solves the un-substituted system symbolically to get a
    │                              #   formula to differentiate against each known's uncertainty
    ├── domain_utils.py              # domain-of-validity: walks an expression once, finds
    │                              #   division/even-root/log/inverse-trig restrictions, checks
    │                              #   them against the problem's specific known values
    ├── code_export.py               # "get this as Python" -- renders algebraic/ODE/recurrence
    │                              #   closed-form targets as standalone Python source (sp.pycode,
    │                              #   not sp.lambdify) that can be saved/read/reused elsewhere
    ├── physical_validity.py         # filters multi-root sp.solve() results against each
    │                              #   variable's declared domain (nonnegative/positive/etc.) --
    │                              #   fixes the "silently returns the negative time root" bug
    ├── unit_conversion.py           # "also equals..." -- offers a numeric answer in common
    │                              #   alternate units for the same dimension, built on
    │                              #   units_checker.py's existing parse/dimension machinery
    ├── grading.py                    # "grade my work" -- formula/arithmetic/final-answer checks
    │                              #   on a student's own attempted steps, reusing
    │                              #   equivalence.py's tested logic rather than diffing steps
    ├── tutor_mode.py                  # guided/tutor mode: turns a solved problem's step-by-step
    │                              #   derivation into a Socratic, one-question-at-a-time walkthrough
    ├── worksheet.py                  # reverse generation -- new problem TEXT (not answers)
    │                              #   sharing a solved problem's verified equation structure,
    │                              #   meant to be re-solved through the normal pipeline
    ├── batch_solver.py               # batch mode -- solves a whole pasted problem set in
    │                              #   one pass via modules/pipeline.py, one failure per
    │                              #   problem rather than all-or-nothing
    ├── similarity.py                 # structural "find similar past problems" -- canonicalizes
    │                              #   equations (symbol names anonymized, structure/coefficients
    │                              #   kept) and compares by Jaccard similarity of equation shapes
    ├── concept_index.py              # tags a solved problem by the named CONCEPTS it touches
    │                              #   (e.g. "conservation of energy"), for browsing history by
    │                              #   concept rather than only by domain/keyword
    ├── sensitivity.py                # what-if / tornado analysis -- sweeps one known input at a
    │                              #   time (others fixed) to see which one moves the answer most
    ├── algebra_rules.py              # structurally classifies WHICH technique (linear/quadratic/
    │                              #   root/inverse-function/etc.) isolates a target, since
    │                              #   sp.solve() doesn't expose its own internal step trace
    ├── dependency_graph.py           # three-column known/equation/unknown diagram of which
    │                              #   variables feed into which equations, plus solve_order():
    │                              #   the order they become determined, and replay_frames()
    ├── followup.py                   # grounded Q&A -- numeric "what if" questions get a REAL
    │                              #   SymPy recompute (LLM only classifies intent); conceptual
    │                              #   questions get an LLM answer grounded in the actual equations
    ├── paranoid.py                   # "paranoid mode" -- re-runs extraction through a SECOND,
    │                              #   independently-configured model and compares equation
    │                              #   shape + numeric answers against the primary derivation
    ├── proof.py                      # symbolic proof mode -- renders the actual sequence of
    │                              #   SymPy simplification passes that prove an equivalence,
    │                              #   not just the final True/False
    ├── timeout_utils.py              # configurable timeout wrapper (a daemon thread per call,
    │                              #   Windows-safe -- no signal.alarm) around every SymPy-heavy
    │                              #   call, so pathological input can't hang the session; at most
    │                              #   MAX_ABANDONED timed-out computations may still be running
    ├── app_logging.py                # rotating WARNING+ log file (data/app.log) -- wired in at
    │                              #   3 gateway points (chat(), extract_json(), run_with_timeout())
    │                              #   for near-complete failure coverage without touching every
    │                              #   individual try/except across the app
    ├── numerical_fallback.py         # mpmath.findroot fallback for single-equation/single-unknown
    │                              #   cases sp.solve() can't handle symbolically (e.g. x+sin(x)=5)
    │                              #   -- explicitly labeled as an approximation, never blended
    │                              #   in with exact symbolic answers
    ├── self_consistency.py           # re-runs extraction on the SAME model 2-5 times and compares
    │                              #   via similarity.py -- catches an ambiguous PROBLEM STATEMENT,
    │                              #   distinct from paranoid.py's cross-model disagreement check
    ├── notebook_export.py            # hand-built nbformat v4 JSON -- narrative as markdown cells +
    │                              #   code_export.py's runnable functions as code cells, no new
    │                              #   dependency on the nbformat package
    ├── optimization_utils.py      # calculus/Lagrange optimization solver (elimination
    │                              #   that substitutes each eliminated variable out of the
    │                              #   remaining constraints too, and that FAILS -- falling
    │                              #   back to Lagrange multipliers -- rather than ever
    │                              #   dropping a constraint it couldn't use; imports
    │                              #   verifier._known_substitutions
    │                              #   at module level -- safe because verifier only
    │                              #   imports back from here inside a function body)
    ├── verifier.py               # structural + numeric + dimensional + inequality +
    │                             #   ODE/recurrence (substitute-and-check) + optimization
    │                             #   (gradient=0 + classification) + independent
    │                             #   cross-check verification
    ├── solver.py                  # SymPy step trace per kind + LLM narration
    ├── scenarios.py                # alternative real-world context generator
    ├── plotter.py                   # 2D line / 3D surface / feasible-region Plotly figures,
    │                                 #   plus add_camera_rotation() (an auto-orbit wrapper for
    │                                 #   any 3D scene) and five animated views (frames +
    │                                 #   play/pause, the same pattern ui/pde.py's time-evolution
    │                                 #   animation established): ODE phase portraits, recurrence
    │                                 #   cobweb diagrams, Monte Carlo convergence, optimization
    │                                 #   descent paths, and kinematics motion diagrams
    ├── motion_diagram.py              # detects a SUVAT-style 1D kinematics setup among a
    │                                 #   solved problem's variables (by each variable's own
    │                                 #   `meaning` text, NOT problem_domain -- an LLM's free-text
    │                                 #   domain label isn't reliable on its own) and, when
    │                                 #   resolvable, builds the x(t)/v(t) trajectory
    │                                 #   plotter.build_motion_diagram() animates
    ├── plot_params.py                # shared "solve for the target, else plot the residual"
    │                                 #   parameter handling for plotter.py AND plot_snapshot.py
    │                                 #   (the target's own slider value must not be substituted
    │                                 #   before solving; a missing residual value is a clear
    │                                 #   ValueError, not a garbage figure)
    ├── plot_snapshot.py              # matplotlib static re-renders of the above, for
    │                                 #   the "include this plot in the report" export feature --
    │                                 #   plus animated GIF exports (matplotlib.animation +
    │                                 #   PillowWriter, no ffmpeg/kaleido/Chrome needed) for the
    │                                 #   rotating 3D surface, the motion diagram, and cobweb
    │                                 #   diagrams, so those can go in a report too, not just
    │                                 #   live in the browser
    ├── templates.py                   # named, savable/loadable presets of a mode's INPUT
    │                                 #   fields (SQLite-backed, like history.py)
    ├── command_palette.py             # fuzzy search over the app's navigable targets;
    │                                 #   MODE_LABELS here is the single source of truth for
    │                                 #   the sidebar's mode list (see ui/__init__.py)
    ├── db_migrations.py               # lightweight, dependency-free schema-migration
    │                                 #   framework: a numbered, idempotent migration list
    │                                 #   applied to every SQLite-backed module (history,
    │                                 #   templates, settings_profiles, chains) at startup
    ├── research_journal.py            # stitches a chosen set of history entries into ONE
    │                                 #   running Markdown document
    ├── workspace.py                  # cross-problem variable memory (session_state)
    ├── history.py                     # SQLite-backed solved-problem history (and, inside each
    │                                 #   record's JSON, the saved exploration view)
    ├── progress.py                    # the optional progress hook long computations call at their
    │                                 #   checkpoints (and chunk_bounds / scaled helpers)
    ├── plot_selection.py              # reads a click on a plot: which bar, grid point or graph node
    ├── view_state.py                  # WHICH widget settings form a problem's saved view, and how to
    │                                 #   capture and sanitise them (pure; takes any mapping)
    ├── solve_compare.py               # compare two solves: answers, inputs, equations (equivalence
    │                                 #   decided by SymPy), verification; what-if snapshots
    ├── compare_plots.py               # the two charts on the Compare page
    ├── share_text.py                  # copy-ready LaTeX and plain-text summaries
    └── exporter.py                     # Markdown + PDF (matplotlib mathtext) export
```

## Coverage: equation kinds

Every relation the extraction step produces is tagged with a `kind`:
`"equation"`, `"inequality"`, or `"ode"`. This tag drives three separate
code paths, not just a display label:

- **Parsing** (`equation_engine.py`): equations parse via `sp.Eq`;
  inequalities parse as a raw SymPy `Relational` (rejecting anything that
  isn't a genuine comparison); ODEs parse with the unknown function bound
  to `sp.Function(name)` instead of `sp.Symbol(name)`, so `Derivative(y(t), t)`
  parses correctly.
- **Verification** (`verifier.py`): equations get numeric-balance +
  dimensional checks; inequalities get a "does the constraint actually
  hold" check once all symbols are known; ODEs get `sp.checkodesol` --
  an exact symbolic check that the solution satisfies the original
  differential equation, not a numeric approximation.
- **Solving** (`solver.py`): equations solve via `sp.solve` (whole system
  at once, so coupled targets resolve correctly); inequalities solve via
  `sp.reduce_inequalities` to produce a solution set; ODEs solve via
  `sp.dsolve`, first for the general solution (shown as its own step),
  then with initial conditions applied for the particular solution.

`target_kind(model, name)` (in `equation_engine.py`) determines which path
a given `solve_for` target actually takes, based on which kind of relation
defines it -- this is what lets `compute_steps()` and `verify()` dispatch
correctly even when a single problem mixes kinds (e.g. an equation and a
constraint together).

## Design notes

Decisions and operational details that don't belong in the README.

**Verification and solving**
- **Dimensional checking uses a fresh placeholder per symbol**
  (`units_checker.make_dimension_placeholder`). Substituting the same unit object
  for two different symbols that share a unit makes SymPy treat them as
  interchangeable, so `a - b` (both lengths) collapses to `0` and is reported
  dimensionless. A `Piecewise` is checked branch by branch, because SymPy's own
  dimension machinery mishandles it whole. The dimension of `d^n f / dx^n` is
  `dim(f) / dim(x)^n`.
- **Optimisation never drops an equality constraint.** Each is used to eliminate a
  variable (and is then substituted out of every other constraint), found
  redundant, or -- if SymPy can't isolate a variable, or it pins the last free one
  -- the whole problem falls back to Lagrange multipliers. An unsolvable Lagrange
  system is an error, never an answer that ignores the constraint.
- **A matrix view is shown only for a genuine system**, i.e. one that can't be
  solved one unknown at a time (`matrix_utils._is_sequentially_solvable`). The
  "show me another way" toggle bypasses that heuristic with `force=True`.
- **The numerical fallback is surfaced in the step trace only.** It isn't threaded
  into the independent cross-check or the confidence report, which stay scoped to
  exact symbolic answers.
- **Numeric answers are computed once and reused.** Monte Carlo, parameter sweeps,
  interval arithmetic and error propagation solve the system symbolically once and
  evaluate the closed form vectorised. An earlier per-sample version hit
  `sp.nsimplify`'s slow path on arbitrary floats (100 samples took 14 s).
- **`ode_utils.py` is shared by `solver.py` and `verifier.py`** to avoid a circular
  import between them. Coupled ODEs are grouped by shared function names and
  solved together with `dsolve_system`.
- **Vector variables carry no `known_value`**; only their components do.

**Plots and exports**
- **`plot_snapshot.py` (matplotlib) is deliberately separate from `plotter.py`
  (Plotly)**, so a change to the live figures can't break exported documents.
  `plot_params.py` is the one shared piece (solve for the target, else plot the
  residual), so a live plot and its snapshot always agree.
- **No `kaleido`.** Its current releases need a separately installed Chrome, which
  conflicts with a "pip install and go" app. PDF equations use matplotlib
  mathtext, so no system LaTeX either. Byte validity is not correct rendering, so
  exports are inspected by eye.
- **Animations** use Plotly frames with a Play/Pause button and slider, with a
  matplotlib `PillowWriter` GIF as the export. Axis ranges and colour scales are
  fixed up front so nothing rescales mid-animation.

**Operations**
- **Timeouts** (`timeout_utils.py`) use a daemon thread per call, not
  `signal.alarm` (absent on Windows) and not a shared pool. Python can't kill a
  thread, so a timed-out computation keeps running; that is bounded by
  `MAX_ABANDONED` (8), after which calls are refused with `ComputationBusyError`.
  A shared pool of four used to fill up after four hangs and report even `1 + 1`
  as timed out. Call sites degrade according to importance: the primary solve
  reports a visible failure, while secondary features (uncertainty, alternate
  method, constraint elimination) quietly become "unavailable".
- **SQLite** (`history.py` and three siblings): WAL journal, `synchronous=NORMAL`,
  a 5 s `busy_timeout`, and history pruned to `MAX_HISTORY_RECORDS` (100).
  `db_util.ClosingConnection` makes `with _connect() as conn:` commit *and* close
  (the plain context manager only commits, which leaks a connection and warns on
  Python 3.13+). Numbered, idempotent migrations (`db_migrations.py`) run at
  startup.
- **Uploads** are capped at 500 MB twice: `server.maxUploadSize` in
  `.streamlit/config.toml`, and `ui/common.check_upload_size()` for a clearer
  message. `.gitignore` carries a narrow exception so that one secret-free file is
  tracked.
- **Logging** (`app_logging.py`): a rotating WARNING-and-above log (5 MB x 3),
  wired into three gateways that nearly every failure passes through:
  `LMStudioClient.chat()`, `extract_json()` and `run_with_timeout()`. The handler
  is guarded so Streamlit's script reruns don't add duplicates.
- **Dependencies are pinned** to the versions the suite has run against, because
  an open-ended range can pull in a breaking release unseen. To upgrade, bump one
  line and let CI run on both platforms.
- **Reruns reuse work** (`ui/cache.py`). Streamlit re-executes the script on every
  widget change, and building one 60-frame Plotly figure takes a second or more.
  `cached(key, parts, compute)` returns the previous result unless `parts` changed.
  Inputs are fingerprinted by content (array bytes and shape, expression structure,
  dataclass fields), never identity or `repr`, and anything unhashable recomputes:
  slower, never stale. Cached figures are shared, so they are never modified.

- **Isolated panels** (`ui/fragments.py`). The heavy analysis panels (the per-target
  Monte Carlo / error / bounds / goal-seek / sensitivity expanders, the N-dimensional
  sweep, the all-targets Monte Carlo, the interactive plots, and the four time views)
  are `st.fragment`s: moving a widget inside one reruns only that panel. This was
  first judged not worth it, because a fragment reruns alone and so cannot update the
  rest of the page. That is still true, and is handled rather than avoided: three
  rules, stated in the module, keep it correct. (1) A fragment may only write into a
  container it creates, so each `with tab:` lives at the call site and the decorated
  function draws into whatever container it is called in. (2) Anything that must
  change another panel goes through `ui/actions.py`: the click's callback queues the
  values, a fragment-scoped click then asks for a full rerun (`st.rerun()`), and
  `apply_pending_updates()` writes them at the top of app.py, before any widget
  exists, which is the only time Streamlit lets code set a widget. The "include in
  report" buttons already did `st.rerun()` for the same reason. (3) A fragment never
  writes to the sidebar. `isolated` also autosaves the saved view afterwards, since a
  fragment rerun never reaches the end of the results page. **What the tests can and
  cannot show:** `AppTest` never performs a fragment-scoped rerun (every interaction
  reruns the whole script), so which panels are isolated is pinned structurally
  (`ISOLATED`), and everything else about them is tested by running them for real;
  the speed-up itself needs a browser to observe.
- **Click-to-act plots** (`ui/plot_select.py`, `modules/plot_selection.py`). Selecting
  a point reruns the page with the selection in `st.session_state[key]`; the page then
  offers actions for it (a clicked tornado bar -> sweep it, add it to the N-D sweep, or
  give it a Monte Carlo uncertainty; a clicked heatmap grid point -> load its values
  into the Variables panel or open it in Compare; a clicked dependency-graph node ->
  that node's steps, shown under the graph and marked in the step list). A click is
  identified by what it IS on the figure (a bar's label, a grid point's x and y, a
  node's coordinates), not by trace or point index, which shift whenever a figure
  gains a trace. Plotly cannot select heatmap cells, so the grid points are drawn as
  small visible markers over the heatmap, and those are what is clicked. The page
  cannot scroll itself, so "jump to that step" shows the steps in place instead.
  Selectable figures are built fresh each run, never taken from `ui/cache.py`.
- **Live status and Stop** (`ui/progress.py`, `modules/progress.py`). Streamlit has no
  cancel API; clicking any widget during a run requests a rerun, and Streamlit abandons
  the running script at its next `st.*` call. So a long computation takes an optional
  `progress` hook and calls it at checkpoints *between chunks of work* (Monte Carlo
  evaluates in chunks of 2,500 after drawing all samples at once, so a seed gives the
  same numbers with or without a hook; the finite-difference PDE solver checkpoints
  before its refined solve). Each checkpoint updates the status block, and is where a
  Stop takes effect. A single numpy/SymPy call cannot be interrupted, and a timeout
  wrapper's worker thread has no Streamlit context, so the 2D heat solver (one
  indivisible stepping call) gets stages but no Stop button. An aborted run leaves
  nothing on screen, so `tracked_run` raises a flag that only a finished or failed run
  clears; a flag found at the next run is reported as "stopped before it finished". A
  finished result is stored only on completion, so Stop never discards an earlier one,
  and a stopped batch keeps the problems that finished.
- **Saved views** (`modules/view_state.py`, `ui/view_state.py`). Reopening a problem
  restores how it was explored: sample counts and seeds, sweep setup, plot axes and
  sliders, time-view settings. It is a whitelist of key patterns, never all of
  `session_state`; only scalars and short lists are kept (a `data_editor`'s value is
  edits against one particular table, so those tables are rebuilt from the remembered
  selections instead). The view is stored in the history record's existing JSON payload
  (no schema change; an older record simply has none) and travels in project bundles.
  Widget keys are built from symbol and target names, which different problems share,
  so opening or solving a problem first clears the previous problem's keys: a setting
  from one problem no longer leaks into the next. A test checks that every widget key
  in `ui/results/` is either covered or deliberately excluded.
- **Share dialog** (`ui/share.py`). The old inline Export section and the "send to
  chain" expander are one dialog. A dialog reruns only itself, so generating the PDF
  must not call `st.rerun()` (that would close it). Nothing is built until it opens,
  which also stops the full Markdown report being rebuilt on every rerun.
  `AppTest` closes a dialog on any interaction, so the dialog's tab bodies are tested
  directly and only the opening of the dialog goes through the app.
- **Compare** (`modules/solve_compare.py`, `ui/compare.py`). Two solves from the
  screen, history or a what-if (known inputs changed, re-solved by SymPy). A what-if is
  verified with the deterministic checks only and says so; the independent LLM re-solve
  is not repeated. Equations are paired as identical, equivalent (SymPy decides,
  under the usual timeout), changed or one-sided.
