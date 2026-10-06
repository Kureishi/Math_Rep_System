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
  up on answers, inputs, equations and verification; no LLM call), and **Research
  journal**.

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
- **Export:** one **Share / export** button opens a dialog with the Markdown and PDF
  downloads, copy-ready **LaTeX** and plain-text summaries (`share_text.py`), and
  "send to a chain". PDF equations are rendered with matplotlib mathtext, and plots go
  into a report through an explicit "Include this plot" button. Static images
  deliberately avoid `kaleido`, whose current releases need a separate Chrome
  install.

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

The suite (about 1,850 tests, roughly two minutes) uses a mocked LLM client
(`tests/conftest.py`'s `FakeClient`), so no LM Studio server is needed.

- **CI** (`.github/workflows/tests.yml`) runs on every push and pull request:
  **ubuntu-latest and windows-latest** x Python **3.12 and 3.14** (the oldest and
  newest supported) x two hash seeds. Windows is included deliberately, since this
  app targets it and some code (the timeout wrapper) exists because the obvious
  approach, `signal.alarm`, does nothing there. CI also byte-compiles the project
  and runs `mypy modules/ ui/ app.py cli.py config.py api_server.py`.
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
