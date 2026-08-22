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
[extraction: LM Studio reasoning model, temp=0.1, JSON-schema constrained]
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
eqsolver/
├── app.py                    # Streamlit UI, orchestrates the pipeline
├── config.py                 # LM Studio endpoint/model settings, tunables
├── requirements.txt
├── README.md                 # setup + run instructions
├── ARCHITECTURE.md           # this file
└── modules/
    ├── llm_client.py         # LM Studio (OpenAI-compatible) client wrapper
    ├── ocr.py                 # pytesseract fallback for non-vision models
    ├── equation_engine.py     # LLM extraction prompt + JSON -> SymPy parsing
    │                          #   (equations / inequalities / ODEs, each kind
    │                          #   parsed differently -- see target_kind())
    ├── units_checker.py        # sympy.physics.units-based dimensional checks
    ├── ode_utils.py             # shared dsolve() helper (solver.py + verifier.py
    │                            #   both need it; lives here to avoid a circular import)
    ├── verifier.py               # structural + numeric + dimensional + inequality +
    │                             #   ODE (checkodesol) + independent cross-check verification
    ├── solver.py                  # SymPy step trace per kind + LLM narration
    ├── scenarios.py                # alternative real-world context generator
    ├── plotter.py                   # 2D line + 3D surface Plotly figure builders
    ├── workspace.py                  # cross-problem variable memory (session_state)
    ├── history.py                     # SQLite-backed solved-problem history
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
