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

10. **Matrix systems** -- whenever the algebraic part of a problem is a
    genuine linear system (two or more `equation`-kind relations sharing
    two or more unknowns -- circuits, coupled springs, Markov chains,
    input-output economic models), it's additionally represented
    explicitly as `A x = b` (`modules/matrix_utils.py`), not just handed
    to `sp.solve()` as an opaque list. The app shows the coefficient
    matrix itself, its determinant and eigenvalues when square (useful
    for stability/vibration problems), and classifies the system via
    rank comparison (`rank(A)` vs `rank([A|b])`) as having a unique
    solution, infinitely many solutions, or being outright inconsistent
    -- rather than `sp.solve()`'s opaque "found nothing" for the latter
    two cases. This is purely an additional structural VIEW alongside
    the existing scalar solve path; the numeric answer for each target
    is unchanged. "Genuine system" specifically excludes anything
    sequentially solvable by plain substitution -- e.g. `a = (v_f-v_i)/t`
    followed by `d = 0.5*a*t^2 + t*v_i` merely shares the model with two
    unknowns (`a`, `d`), but the first equation determines `a` completely
    on its own, so no matrix view is shown for it. Only equations that
    genuinely can't be resolved one unknown at a time (checked via
    `matrix_utils._is_sequentially_solvable`, e.g. `2x+3y=8, x-y=1`,
    where neither equation alone isolates either variable) get the A x=b
    treatment.
11. **Vectors and basic geometry** -- a variable can be declared
    `"is_vector": true` with a `"components"` list (e.g. a force `F`
    with components `Fx`, `Fy`), and used directly in equations via
    `dot()`, `cross()`, `magnitude()`/`norm()`, `unit()`,
    `angle_between()`/`angle_between_deg()`, `distance()`, and `Point()`
    (`modules/vector_utils.py`) -- backed by a genuine SymPy column
    `Matrix`, not a scalar the LLM had to pre-decompose by hand. A
    force/displacement/velocity stays a real vector object all the way
    up to the point a dot/cross product collapses it to the scalar the
    equation actually needs (e.g. `Eq(W, dot(F, d))` for work, or
    `Eq(tau, cross(r, F))` for 2D torque) -- so it flows through the
    existing equation/verification/solving pipeline unchanged; no new
    equation kind was needed. The app shows each declared vector's
    numeric components, magnitude, and direction once its components
    are filled in.
12. **Curve/data fitting** (`modules/curve_fitting.py`) -- a genuinely
    different pipeline, reachable via the "📈 Curve fitting" mode at the
    top of the app: input is a table of (x, y) numbers (pasted or
    uploaded as a 2-column CSV), not an LLM-extracted word problem, and
    the output is a fitted symbolic model plus fit-quality metrics
    (R², RMSE, per-point residuals) rather than a verified derivation --
    there's no independent-derivation cross-check to run against a
    curve fit, so the metrics themselves are the honesty check. Built-in
    families (linear, polynomial, exponential, power, logarithmic) are
    all solved via `numpy.polyfit` after linearizing (e.g. exponential
    fits `ln(y)` against `x`) -- exact, non-iterative, and deliberately
    avoids adding scipy as a dependency. A "custom" family lets you fit
    any expression that's linear in its named parameters (e.g.
    `a*sin(x) + b*x + c`) via plain linear least squares
    (`numpy.linalg.lstsq`) -- genuinely nonlinear custom models (e.g.
    `a*sin(b*x)`) are explicitly rejected with a specific explanation
    rather than silently attempted or requiring scipy; that's a
    conscious portability tradeoff, not a hidden limitation.
13. **Equivalence/simplification checking** (`modules/equivalence.py`)
    -- "are these two expressions the same" as a standalone utility
    (reachable via "🔁 Check equivalence"), not a new representation
    capability. Built on `sp.Expr.equals()`, which tries symbolic
    simplification first and falls back to numeric sampling; when that's
    still inconclusive (e.g. `sqrt(x**2)` vs `x`, which is only equal
    for `x >= 0`), this module does its own targeted sampling and
    reports *which* tested points agreed and disagreed rather than a
    bare "undetermined."
14. **Uncertainty/error propagation** (`modules/uncertainty.py`) -- a
    variable can carry an optional `"uncertainty"` field (an absolute
    +/- tolerance -- see the extraction prompt's rules for converting a
    stated percentage into one). For an algebraic target that depends on
    such a variable, the solve steps get one extra "Propagate
    measurement uncertainty" step: standard first-order error
    propagation, `sigma_f^2 = sum_i (df/dxi)^2 * sigma_xi^2`, computed by
    re-solving the UN-substituted system symbolically (see
    `solve_symbolic_for_target`) so there's a formula left to
    differentiate against each known. Reports not just the combined
    uncertainty but which input dominates it (`dominant_source`) --
    useful for knowing which measurement to tighten if the answer needs
    to be more precise. Deliberately scoped to algebraic targets only,
    same as the matrix-system detection: ODE/recurrence/optimization
    solutions don't have an equally clean closed form to differentiate
    in general.
15. **Domain-of-validity tracking** (`modules/domain_utils.py`) -- walks
    each derived equation's expression tree once, structurally
    identifying where it's undefined (denominators that must not be
    zero, even roots requiring a nonnegative argument, logs requiring a
    positive argument, inverse sine/cosine requiring an argument in
    [-1, 1]), then checks those conditions against the SPECIFIC known
    values given in the problem. An actively-violated restriction (e.g.
    dividing by a variable that happens to be 0) is reported as a
    genuine verification FAILURE via `_domain_checks` in `verifier.py`
    -- not a silent NaN three steps later. A restriction that's
    satisfied, or can't yet be checked (involves a still-unknown
    symbol), is still surfaced as an informational note in a dedicated
    "Domain of validity" panel, since knowing a formula's boundary of
    validity is useful even when today's inputs don't cross it.
    Deliberately limited to these four structural sources rather than a
    full domain solve (`sp.calculus.util.continuous_domain` only handles
    one free variable at a time, and most formulas here have several) --
    and deliberately excludes `tan()`-style infinitely-repeating
    singularities, which would be more noise than signal for word
    problems.
16. **Confidence report** (`VerificationReport.confidence_report()` in
    `modules/verifier.py`) -- aggregates the (often 10-20+) individual
    checks that already run into one category-grouped view: a 0-1
    overall score, a pass count per category (structural, dimensional,
    independent cross-check, domain validity, matrix/system consistency,
    and any ODE/recurrence/optimization/inequality-specific checks that
    ran), and an explicit list of critical failures -- rather than
    making a person scan a flat list of 15 checks to get a sense of "how
    much should I trust this." The score is 1.0 only when every
    margin-bearing check passed with an essentially-exact margin, and is
    capped below 0.5 the moment ANY check fails outright, regardless of
    how many others passed. Categories are inferred from each check's
    existing label text (`_infer_category`) rather than requiring every
    `report.add(...)` call site across the codebase to be updated with
    an explicit category -- a lighter-touch way to get the aggregation
    without an invasive refactor.
17. **Runnable code export** (`modules/code_export.py`) -- "get this as
    Python" for any algebraic, ODE-closed-form, or recurrence-closed-
    form target: renders the derived formula as actual Python SOURCE
    TEXT via `sp.pycode` (not `sp.lambdify`, which builds a live
    compiled closure that can't be saved to a file or read by a
    person). Reuses `uncertainty.solve_symbolic_for_target` to get a
    formula purely in terms of named inputs (re-solving the
    UN-substituted system, since by the time the ordinary step trace
    calls `sp.solve()` the knowns are already plain numbers with no
    symbolic trace left). A coupled target (e.g. displacement depending
    on an also-unknown acceleration) exports correctly reduced to only
    the genuinely-known inputs, since the whole system is solved
    together. "Get all formulas as one Python file" bundles every
    target into one module with a demo `__main__` block calling each
    function with the problem's own known values. Deliberately excludes
    optimization results (a single numeric critical point isn't a
    general-purpose formula to export).
18. **Physical-validity filtering** (`modules/physical_validity.py`) --
    when `sp.solve()` returns more than one root (a quadratic
    time-of-flight equation is the textbook case), the raw result has no
    notion of which branch is physically meaningful; the solver used to
    always take branch `[0]` unconditionally, which for the classic
    `-4.9*t**2 + 20*t + 1.5 = 0` genuinely returns a NEGATIVE time
    first. A variable can now be declared with a `"domain"` field
    (`"nonnegative"`, `"positive"`, `"nonpositive"`, `"negative"`); when
    a target has multiple roots, each is checked against every declared
    domain, non-conforming roots are discarded with an explicit reason
    shown as a solve step ("Discard a non-physical root"), and the
    remaining (physically valid) root is used as the answer. If no
    domain is declared, behavior is unchanged from before this feature
    existed -- this is opt-in, matching how vectors/uncertainty are
    opt-in, not a silent behavior change for every existing problem.
19. **Unit conversion sweep** (`modules/unit_conversion.py`) -- once an
    answer's unit is known, offers it in a handful of common alternate
    units for the same dimension (m/s <-> km/h <-> mph <-> ft/s, m <->
    ft <-> mi <-> km, kg <-> lb <-> g, etc.), shown in an "Also equals
    ..." expander and included in exported reports. Built entirely on
    `units_checker.py`'s existing unit-parsing/dimension
    infrastructure (`parse_unit`, `dimension_of`, `dims_equivalent`) --
    no second unit-string format to keep in sync. Deliberately excludes
    Celsius/Fahrenheit/Kelvin conversions: those are AFFINE (value *
    scale + offset), not pure multiplicative scale factors, and
    `sympy.physics.units.convert_to()` only handles the multiplicative
    case -- silently applying it would produce a wrong number that
    looks plausible, so temperature-scale conversion is left out rather
    than faked.
20. **"Grade my work"** (`modules/grading.py`) -- a student (or teacher)
    pastes their own attempted work for an algebraic target, one step
    per line, and gets back three separate diagnoses instead of just a
    right/wrong verdict: a FORMULA check (is their starting equation
    mathematically equivalent to one of the problem's own verified
    relations -- solved both for the target symbol and compared via
    equivalence.py's tested logic, so a correctly *rearranged* equation
    like `a*t = v_f - v_i` instead of `a = (v_f-v_i)/t` is recognized as
    the same formula, not flagged as different), an ARITHMETIC check per
    line (does each side agree numerically once known values are
    substituted), and a FINAL-ANSWER check against the system's own
    verified value. Deliberately not a literal step-by-step diff against
    the system's own derivation -- two valid derivations of the same
    formula can look completely different, so diffing them directly
    would flag correct-but-differently-ordered work as wrong. Caught two
    real bugs during development: SymPy auto-evaluates `Eq()` of two
    pure numbers straight to a bare `True`/`False` rather than staying
    an `Eq` object (so `Eq(12/6, 3)` needed its own detection path, not
    just an `isinstance(Eq)` check), and a line whose left side is just
    the (still-unknown) target symbol itself was wrongly being marked
    "not checkable" before that got fixed.
21. **Reverse generation / worksheet variants** (`modules/worksheet.py`)
    -- "Generate worksheet variants" asks the LLM to write NEW word
    problem TEXT sharing the current problem's own verified equation
    structure, with different numbers and a different story --
    inverting `scenarios.py`'s "where else does this apply" into "give
    me a fresh problem to hand a student." Deliberately does NOT ask the
    LLM for the new problem's answer or equations too -- an LLM's own
    stated numeric answer for a problem it just invented isn't
    trustworthy on its own, which is this whole app's premise. Instead
    each generated problem is meant to be pasted right back into the
    main text input and solved through the exact same extract/verify
    pipeline as any other problem, so a generated worksheet problem gets
    verified by the same standard as everything else.
22. **Multiple-method toggle** (`alternate_method_steps()` in
    `solver.py`) -- an explicit "show me another way" per algebraic
    target, shown only on request (never part of the default step
    list). Always includes a back-substitution check: plug the already-
    solved answer into the original equation(s) and confirm both sides
    agree. When the target is part of a square coupled linear system, it
    ALSO shows Cramer's rule (`x_i = det(A_i)/det(A)`) -- even if the
    default view used plain substitution because that was the simplest
    path for THAT target (see point 10's sequential-solvability note).
    Required adding a `force=True` option to
    `matrix_utils.build_linear_system`/`linear_system_view` so this
    toggle can get at the coefficient matrix regardless of what the
    default-view heuristic decided -- "show me a second way, on
    request" is a different question from "what should the default
    view be," so it gets its own bypass rather than fighting the
    heuristic.
23. **Worksheet/batch mode** (`modules/batch_solver.py`, "📚 Batch
    solver") -- solve a whole problem set (pasted, separated by a blank
    line or a `---` line) in one pass instead of running each problem
    through the app individually and reassembling the results by hand.
    Mirrors (rather than imports) the exact extract -> verify -> retry
    -> compute_steps pipeline the single-problem flow uses, since that
    flow is inline Streamlit script code, not an importable function --
    mirroring it keeps batch mode from risking any change to the
    already-working single-problem path. Narration/scenario generation
    are off by default (each is an extra LLM round trip per problem,
    and neither changes whether the math is right) but toggleable. One
    bad problem in a batch of 20 doesn't take the other 19 down with it
    -- every failure mode is caught per-problem. Produces a combined
    Markdown report or a combined PDF (merged via `pypdf`, a new
    dependency -- byte-concatenating separately-generated PDFs produces
    a corrupt file; a real page-level merge is required).
24. **"Find similar past problems"** (`modules/similarity.py` +
    `history.find_similar()`) -- structural similarity based on
    equation SHAPE, not problem-text wording: `a = (v_f-v_i)/t` and
    `r = (p-q)/s` are recognized as the same underlying formula (used
    with different variable names/domains) via `canonicalize_equation()`
    replacing every plain `Symbol` with an anonymous placeholder while
    leaving numeric coefficients and operator structure untouched, then
    comparing problems by the Jaccard similarity of their canonicalized
    equation-shape sets. A plain text/keyword search wouldn't catch a
    car-acceleration problem and a chemistry-rate problem sharing the
    same math; a literal equation-string match wouldn't either
    (different variable names -> different strings). Every solved
    problem's shape fingerprint is stored alongside it in SQLite (a new
    `equation_shapes` column, added via a safe `ALTER TABLE` migration
    so an existing local `history.db` from before this feature existed
    doesn't break), and a "similar past problems" panel surfaces matches
    for whatever's currently being solved. Scoped to plain-Symbol
    equation/ode/recurrence relations -- ODE/recurrence FUNCTION names
    (the "T" in `T(t)`) aren't themselves canonicalized, only symbols
    appearing as plain `Symbol` nodes are, so two ODEs with the same
    structure but different function names won't match quite as
    strongly. Documented as a scope limitation, not a silent gap.
25. **Sensitivity / what-if analysis** (`modules/sensitivity.py`) --
    reuses `uncertainty.py`'s trick of re-solving the un-substituted
    system symbolically to get the target as a formula in the knowns,
    then sweeps ONE input across a range (default ±20%, holding every
    other input fixed) to see how the answer moves.
    `tornado_analysis()` ranks every input by how much it swings the
    answer across its own range -- the classic "which input matters
    most" tornado-chart view -- and a per-input sweep chart shows the
    full curve, not just the two endpoints. Distinct from `uncertainty.py`'s
    error propagation: that asks "how much does the answer move given
    each input's STATED measurement error"; this asks "if I could
    deliberately change this input, how much would the answer move,"
    independent of whether any input carries a stated uncertainty at
    all. Scoped to algebraic targets, same as `uncertainty.py`.
26. **Algebra-rule tagging** (`modules/algebra_rules.py`) -- names which
    algebraic TECHNIQUE was needed to isolate a target (linear,
    quadratic, root, reciprocal, an inverse function, or "target on both
    sides") as a "Technique" solve step. `sp.solve()` doesn't expose an
    internal trace of the moves it makes, so rather than fabricate a
    step sequence it never actually took, this does a structural
    classification of the equation's shape instead -- honest about
    being a classification, not a derivation, while still naming the
    technique the way a textbook section heading would.
27. **PDF/document batch import** (`extract_text_from_pdf()` +
    numbered-list detection in `split_batch_text()`, both in
    `batch_solver.py`) -- upload a PDF worksheet instead of retyping
    every problem; text is pulled out via `pypdf` (already a
    dependency -- no OCR, so a scanned/image-only PDF with no text
    layer comes back empty rather than raising). Splitting now tries a
    numbered-list pattern first ("1.", "2)", "Problem 3:") before
    falling back to the `---`/blank-line delimiters, since PDF-extracted
    text often loses blank-line spacing between problems even when the
    original document had it, and worksheets are conventionally
    numbered anyway. Verified the pattern doesn't false-positive on a
    problem that happens to start with a decimal number ("3.5 kg of
    ice...").
28. **Dependency graph visualization** (`modules/dependency_graph.py`)
    -- a diagram of which known/unknown variables feed into which
    equations, useful once a problem has enough equations that it's not
    obvious at a glance which pieces depend on which (a matrix system, a
    coupled ODE pair, a chain of substitutions). Fixed three-column
    layout (known inputs -> equations -> unknowns) rather than a
    generic force-directed graph, since that's literally the
    information flow the solving pipeline follows, and it avoids a
    graph-layout dependency (e.g. networkx) for graphs that are always
    small and naturally three-tiered. Uses each equation's LHS shape
    (`Eq(single_symbol, expr)`) to distinguish a variable an equation
    PRODUCES from one it merely DEPENDS ON -- important for a variable
    used in more than one equation (e.g. an acceleration computed by one
    equation and then consumed by a displacement formula), so it doesn't
    misleadingly appear to be "produced" by every equation that
    references it.
29. **Grounded follow-up Q&A** (`modules/followup.py`) -- ask a question
    about an already-solved problem via "Ask a follow-up question."
    Splits "compute something" from "explain something," since this
    app's whole premise is that LLM arithmetic isn't trustworthy on its
    own: a numeric "what if" question ("what if t doubles?") is
    classified via one small LLM call into a STRUCTURED intent (which
    known symbol, which operation -- multiply/add/set, what operand) --
    the LLM only extracts intent from natural language, the actual
    arithmetic (applying that operation, then re-evaluating the
    verified formula via `uncertainty.solve_symbolic_for_target`, the
    same machinery `sensitivity.py` uses) is done by SymPy, so a
    what-if answer is exactly as verified as the original solve. A
    conceptual question ("why this formula", "what does v_i mean")
    gets an LLM answer grounded in the problem's actual equations/known
    values/solved answers via the system prompt, with an explicit
    instruction to say so rather than invent a fact that isn't given --
    though the prose explanation itself isn't independently
    re-verified the way a numeric answer is, an honest limitation of
    natural-language explanation this module doesn't claim to solve.
30. **Multi-model cross-verification / "paranoid mode"**
    (`modules/paranoid.py`) -- the existing independent cross-check in
    `verifier.py` re-asks an LLM to solve the problem from scratch, but
    by default with the SAME model as extraction; a model can be wrong
    in a way that's entirely self-consistent (it misreads the problem
    the same way whether asked to extract equations or asked to just
    "give a number"), which single-model verification can't
    structurally catch. Setting `config.settings.secondary_reasoning_model`
    to a second loaded model enables a "🕵️ Paranoid mode" panel that
    re-runs the FULL extraction pipeline through that second model and
    compares the two derivations two ways: structural equation-shape
    similarity (reusing `similarity.py`'s canonicalization -- the same
    trick "find similar past problems" uses, just comparing two live
    derivations against each other instead of one against history) and
    numeric-answer agreement within the normal cross-check tolerance.
    Off by default -- it doubles the extraction cost of a problem, so
    it's opt-in, not something every solve pays for.
31. **Symbolic proof mode** (`modules/proof.py`) -- for an equivalence
    check that comes back symbolically confirmed True, "📐 Show proof"
    renders the ACTUAL sequence of SymPy simplification passes (expand,
    combine into a fraction, apply trig identities, combine powers,
    combine logs, simplify radicals, factor, general simplification)
    that reduce the difference of the two expressions to zero -- the
    real transformation SymPy applies at each stage, not a fabricated
    derivation, just reported incrementally instead of only the final
    True the way `equivalence.py` does on its own. Required adding a
    `raw_difference` field to `EquivalenceResult`: the existing
    `difference_simplified` field is already FULLY reduced by
    `equivalence.py` itself before `proof.py` ever sees it, which would
    make every "proof" trivially one step long with nothing to show --
    caught during development by actually running the proof builder
    against real identities rather than assuming the design worked.
    Scoped to symbolically-confirmed equivalences only; there's no
    proof to walk through for something only confirmed by numeric-
    sampling evidence (see `equivalence.py`'s own docstring on why
    that's evidence, not proof) or that isn't equivalent at all.
32. **Configurable computation timeouts** (`modules/timeout_utils.py`)
    -- every genuinely SymPy-heavy call in the app (algebraic solve,
    matrix determinant/eigenvalues/linsolve, ODE `dsolve`, recurrence
    `rsolve`, optimization critical points, equivalence checking, and
    each pass of `proof.py`'s simplification chain) runs under a
    configurable time bound (`config.settings.computation_timeout_seconds`,
    default 10s -- adjustable live from "⚙️ Advanced settings", no
    restart needed) instead of being able to hang the session
    indefinitely on pathological input. Built on
    `concurrent.futures.ThreadPoolExecutor`, deliberately NOT
    `signal.alarm`: this app targets Windows as a first-class
    environment, and `SIGALRM` doesn't exist there -- a signal-based
    timeout would silently do nothing on the platform it's most meant
    to protect. The honest tradeoff, stated rather than glossed over:
    Python can't forcibly kill a running thread, so a genuinely hung
    computation's worker thread keeps running in the background
    (consuming CPU) even after the app has moved on and shown a timeout
    message -- this protects the UI from LOOKING hung, it doesn't
    reclaim the CPU from truly runaway work. A multiprocessing-based
    approach could forcibly terminate it, at the cost of process-spawn
    overhead on every single call including the overwhelming majority
    that finish in milliseconds -- not the right tradeoff for an
    interactive app. Degradation is calibrated per call site: the
    primary algebraic solve and matrix analysis report a timeout as an
    explicit, visible failure (verifier.py's "Symbolic solve" check, or
    a "Computation timed out" solve step) rather than silently looking
    like "no answer found"; secondary/bonus features (uncertainty
    propagation, the "show me another way" alternate method, constraint
    elimination in `optimization_utils.py`) degrade silently to
    "unavailable," matching how those features already handle any other
    failure. `matrix_utils.MatrixSystemResult` gained a
    `computation_notes` field specifically so a timeout on, say,
    eigenvalues alone doesn't lose the still-fast rank-based
    classification -- genuine partial degradation, not all-or-nothing.
33. **Pinned dependencies** (`requirements.txt`) -- every package is
    pinned to the exact version this project's full test suite has
    actually been run against (`streamlit==1.62.0`, `sympy==1.14.0`,
    etc.), not an open-ended `>=` range. An unpinned range looks
    convenient but means a fresh install months from now could silently
    pull in a breaking release of any dependency and fail in ways that
    have never been seen or tested here. Upgrade deliberately: bump one
    line, run `pytest` (or push and let CI run it across both platforms),
    and only commit once it's green.
34. **SQLite hardening** (`modules/history.py`) -- `_connect()` now sets
    `PRAGMA journal_mode=WAL` (a crash or kill mid-write is far less
    likely to leave the file in a bad state, and it tolerates a second
    reader/writer -- e.g. two browser tabs on the same session --
    without immediately hitting "database is locked"), paired with
    `synchronous=NORMAL` (the standard safe pairing with WAL) and a
    5-second `busy_timeout` (retries briefly instead of raising an error
    the moment two connections briefly overlap). Every `save()` also
    prunes the table down to the `MAX_HISTORY_RECORDS` most recent rows
    (100 by default) -- keeps `history.db` from growing unbounded over
    months of use, and keeps `list_recent()`/`find_similar()`'s
    full-table scans bounded, without needing a separate maintenance
    step someone has to remember to run.
35. **Upload size limits** -- every `st.file_uploader` (CSV for curve
    fitting, PDF for batch worksheet import, problem photos) is capped
    at 500 MB, enforced twice: `.streamlit/config.toml` sets
    `server.maxUploadSize = 500` so Streamlit rejects an over-limit
    upload server-side before it's even fully received, and
    `check_upload_size()` in `app.py` re-checks the file's own
    `.size` as a second, defense-in-depth layer with a clearer,
    upload-specific error message than Streamlit's generic rejection.
    `.streamlit/` is otherwise gitignored (it can hold a local
    `secrets.toml`); `.gitignore` carries a narrow `!.streamlit/config.toml`
    exception so this one file -- which holds no secrets -- is still
    tracked and ships with the project.
36. **Basic logging** (`modules/app_logging.py`) -- a small rotating log
    file (`data/app.log`, 5 MB × 3 backups, WARNING level and above
    only -- not a full request/access log, which would be noisy and
    mostly pointless overhead for a personal local tool) so a recurring
    failure is visible after the fact instead of only ever showing up
    as a message in the UI that's gone the moment the page reruns.
    Directly motivated by an earlier incident in this project where
    telling "a rare one-off" apart from "this keeps happening" for an
    LM Studio engine error required manually digging back through the
    conversation rather than checking a log. Wired in at just three
    gateway points essentially every failure of its class already
    funnels through, rather than touching every individual try/except
    scattered across the app: `LMStudioClient.chat()` (every LLM call,
    whichever module made it), `extract_json()` (every JSON-parsing
    failure), and `timeout_utils.run_with_timeout()` (every symbolic
    computation timeout, across matrix analysis, ODEs, recurrences,
    optimization, equivalence checking, and proof mode alike). Three
    edits, near-complete coverage. `logging.getLogger()`'s process-wide
    singleton-by-name behavior is what keeps Streamlit's constant script
    reruns from re-adding a handler (and duplicating every log line) on
    every interaction -- guarded explicitly rather than assumed.
37. **Config/connection validation** (`LMStudioClient.validate_model()`
    in `modules/llm_client.py`) -- distinguishes two genuinely different
    failure modes that would otherwise both just look like a similar
    raw API error the moment a call is attempted: LM Studio isn't
    reachable at all, versus it IS reachable but the specific model
    requested isn't one it currently has loaded (a typo, or a model
    unloaded since it was configured). The primary/vision models are
    already implicitly validated by construction -- their selectors in
    the sidebar are populated FROM `list_models()`, so nothing outside
    that list can be chosen -- but "paranoid mode"'s secondary model
    (set via `config.py`/an env var, no dropdown) had no such guarantee.
    The "🕵️ Paranoid mode" panel now validates it up front and shows a
    specific, actionable message before attempting the (more expensive)
    cross-check, rather than launching into an extraction call destined
    to fail confusingly.
38. **Numerical fallback** (`modules/numerical_fallback.py`) -- when
    `sp.solve()` can't find a closed form (common for equations mixing
    polynomial and transcendental terms, e.g. `x + sin(x) = 5` or
    `x*exp(x) = 10` -- both ordinary physics/engineering "solve for x"
    problems that SymPy's symbolic solver genuinely can't handle, not
    edge cases), falls back to numerical root-finding via
    `mpmath.findroot` -- already a SymPy dependency, so no new one
    added -- tried from several starting points to catch multiple
    distinct roots. Unmistakably labeled as an approximation the whole
    way through (`is_numerical=True` on every result, a distinct
    "No exact symbolic solution -- numerical approximation" step rather
    than blending in with exact answers): this app's whole premise is
    verification-first, so a numerical fallback that looked identical to
    a verified symbolic answer would undermine that. Deliberately scoped
    to a single equation in a single remaining unknown -- coupled
    numerical solving across several unknowns simultaneously is a much
    less reliable problem and isn't attempted. Currently surfaced in the
    step-by-step solve trace only; it isn't yet threaded back into
    `verifier.py`'s independent-cross-check/confidence-report pipeline,
    which stays scoped to exact symbolic answers for now -- a known,
    stated limitation rather than a silent gap.
39. **Self-consistency check** (`modules/self_consistency.py`, "🔁
    Self-consistency check") -- re-runs extraction on the SAME model
    2-5 times and compares the derivations via `similarity.py`'s
    equation-shape canonicalization, the same trick "find similar past
    problems" and "paranoid mode" both use. A genuinely different
    signal from paranoid mode's cross-MODEL check: two different models
    disagreeing suggests one of them is specifically wrong, but the SAME
    model disagreeing with ITSELF across repeated runs of the identical
    prompt usually means the PROBLEM STATEMENT is ambiguous or
    underspecified enough that even one model can't parse it the same
    way twice -- a property of the input, not of any one model's
    competence, worth surfacing regardless of which derivation ends up
    being used.
40. **Jupyter notebook export** (`modules/notebook_export.py`, "⬇️ Get
    as Jupyter notebook") -- bundles the step-by-step narrative (as
    markdown cells) with the runnable Python formula(s) (as executable
    code cells, reusing `code_export.py`'s exact same `sp.pycode`-
    rendered functions) into a single `.ipynb` -- a more natural
    deliverable than a bare `.py` script for further work in a notebook
    environment. Built by hand-constructing the nbformat v4 JSON
    structure directly rather than adding a dependency on the
    `nbformat` package: the schema needed here (a flat list of
    markdown/code cells, no stored outputs) is small and stable enough
    that a new dependency for it isn't worth it. Verified the exported
    notebook is genuinely runnable, not just well-formed JSON -- its
    code cells were executed in sequence exactly as Jupyter would, and
    produce the correct numeric answer.
41. **Physical plausibility check** (`modules/plausibility.py`,
    "⚠️ Physical plausibility check") -- a softer, advisory-only cousin
    of Domain of validity above. `domain_utils.py` catches values that
    are mathematically *undefined* (a division by zero, a negative
    even-root argument); this catches values that are mathematically
    fine but land far outside what's normal for the kind of quantity
    involved -- a car's acceleration coming out to 500 m/s², a computed
    mass that's negative even though nothing declared a domain
    restriction on it. A small curated table of typical magnitude
    ranges per domain category (kinematics, mechanics, finance,
    thermodynamics, electricity), inferred from the problem's own
    `problem_domain` label, plus an independent meaning-based
    heuristic ("mass", "distance", "age", ... shouldn't be negative)
    for variables with no declared domain at all. Deliberately never
    affects `report.passed` -- an out-of-range magnitude isn't proof
    the math is wrong (a problem CAN legitimately be about a rocket
    sled), only worth a second look.
42. **Personalized error-pattern tracking** (`modules/grading.py`'s
    `classify_mistake`, `history.py`'s `grading_records` table +
    `summarize_error_patterns`, wired into "📝 Grade my work" and "📄
    Generate worksheet variants") -- connects three already-built
    pieces into an actual learning loop. Every "Grade my work"
    submission gets classified (correct / wrong formula / arithmetic
    slip, with a best-effort subtype like sign error, subtraction,
    division) and persisted alongside history.py's existing records.
    When a (category, subtype) pair recurs 3+ times within the last
    week, it surfaces as a plain-English pattern ("You've made a sign
    error 3 times this week") right in the grading panel, and the
    worksheet generator gets a "🎯 Target my recent mistake pattern(s)"
    option that biases new practice problems toward exercising exactly
    that step -- turning generic worksheet variants into targeted
    practice.
43. **Multi-problem dependency chains** (`modules/chains.py`, "🔗
    Problem chains" mode) -- formalizes the ad-hoc "extract to
    workspace" flow into a NAMED, PERSISTENT sequence of problems where
    a downstream step's input is wired directly to an upstream step's
    solved output: change an upstream value (or edit a fixed input on
    an early step) and everything downstream automatically re-solves,
    the way a spreadsheet cell ripples through formulas that reference
    it. Distinct from `dependency_graph.py`, which only diagrams
    structure WITHIN a single already-extracted problem -- this spans
    MULTIPLE separately-extracted problems and actually performs the
    re-solve, not just visualizes it. Deliberately re-solves with plain
    SymPy only (`verifier._solve_sympy`), skipping the LLM independent
    cross-check that the normal `verify()` pipeline does -- a chain step
    can be re-solved many times as inputs change, and an LLM round trip
    on every edit isn't something this should require. A step is only
    ever added to a chain from an already-extracted-and-verified
    `ProblemModel` in the first place, so full verification still
    happens once, upstream of this feature.
44. **Log/log-log axis toggle** (`plotter.build_plot`/`build_fit_plot`,
    "Log X-axis"/"Log Y-axis" checkboxes on the main interactive plot
    and the curve-fitting tab) -- a power-law relationship renders as a
    straight line on log-log axes, an exponential one as a straight
    line with only the Y-axis logged, which is usually a clearer visual
    sanity check of a fit or trend than the default linear view. A log
    axis uses a geometric (not linear) sweep grid, floored just above
    zero.
45. **Contour plots** (`plotter.build_contour_plot`, "Contour" plot type
    alongside "2D line"/"3D surface" whenever a plottable equation has 2+
    free symbols) -- the flat, labeled-level-lines counterpart of the
    existing 3D surface plot: the same (x, y) → z evaluation, but
    without a viewing angle to fight with, and usually easier to read
    exact values off of.
46. **Overlay/comparison plots** (`plotter.build_overlay_plot`, used in
    the curve-fitting tab's "📊 Compare every candidate fit on one plot")
    -- a generic multi-series plot for putting several curves on the
    same axes at once, rather than only ever seeing one result at a
    time. `best_fit()` already tries every built-in family; this makes
    it possible to actually SEE why one family won, not just read its
    higher R² in a list.
47. **Vector plot export** (`plot_snapshot.py`'s `fmt` parameter --
    `"png"`/`"svg"`/`"pdf"` -- threaded through every snapshot function,
    surfaced as a Format dropdown + download button next to the main
    interactive plots and the curve-fit plot) -- PNG is fine for the
    exported Markdown/PDF report, but a figure headed into a paper or
    slide deck usually wants a vector format that doesn't pixelate when
    scaled up. Comes for free from matplotlib's own `savefig()` -- no
    new dependency.
48. **Chain-driven parameter sweeps** (`chains.sweep_step_binding`,
    `plotter.build_chain_sweep_plot`, "📊 Sweep `<symbol>` across a
    range" inside a chain step's fixed-input editor) -- sweeps ONE fixed
    input on one chain step across a range, cascading the WHOLE chain at
    every swept value, and plots every downstream step's output as its
    own line against the swept value. The natural next step once
    `chains.py` existed: rather than testing "what if this input were
    different" one value at a time by hand, sweep it and see the whole
    curve. Restores the step's original binding when the sweep finishes
    -- a sweep is exploratory, not a change to the chain's real state.
49. **Monte Carlo uncertainty propagation** (`modules/monte_carlo.py`,
    "🎲 Uncertainty propagation for `<target>`") -- give one or more known
    inputs a measurement uncertainty (mean ± std) and see the resulting
    SPREAD in the target, sampled JOINTLY across every uncertain input
    at once. Distinct from the existing sensitivity/tornado analysis,
    which varies ONE input at a time deterministically (a
    partial-derivative-flavored "which input matters most" view) rather
    than propagating a joint distribution (a "given these measurement
    uncertainties, how uncertain is my final answer" view) -- the two
    are complementary. Solves the system SYMBOLICALLY ONLY ONCE
    (substituting every fixed known value, leaving the uncertain
    variables as free symbols) and evaluates that one closed-form
    expression vectorized across every sample via NumPy, rather than
    calling `verifier._solve_sympy()` per sample -- an earlier version
    did the latter and, because `_known_substitutions()` runs every
    known value through `sp.nsimplify()` looking for an exact form, hit
    sympy's occasionally very slow algebraic-number-reconstruction path
    on arbitrary sampled floats (100 samples took 14+ seconds). The
    symbolic-once/numeric-after rewrite runs 5,000 samples in well
    under a second.
50. **Self-consistency numeric spread** (`self_consistency.numeric_answer_spread`,
    `plotter.build_spread_plot`, shown inside the existing "🔁
    Self-consistency check" expander) -- self-consistency's own
    `shapes_match` score is a STRUCTURAL similarity between repeated
    re-extractions; it says nothing about whether they land on the same
    NUMBER. Two runs can score a near-perfect shapes_match and still
    disagree numerically if, say, one run's extraction assigned a
    different known value to some variable. This solves each usable
    run's own re-derived model for a chosen target and plots the actual
    numeric answers as a box-and-strip spread, making that kind of
    disagreement directly visible rather than only inferable from a
    similarity percentage.

## Rigor & analysis

Three additions that give a solved problem's uncertainty/sensitivity a
harder mathematical treatment than the existing Monte Carlo panel alone,
each shown as its own expander inside a target's per-target analysis
section:

- **Analytic (closed-form) error propagation**
  (`modules/error_propagation.py`, "📐 Analytic error propagation for
  `<target>`") -- the textbook first-order propagation-of-uncertainty
  formula, `σ_f² = Σ (∂f/∂xᵢ)² σᵢ²`, computed instantly with no
  sampling. Exact when the target is linear in its uncertain inputs, a
  good local approximation otherwise -- the standard alternative
  `monte_carlo.py`'s sampling-based approach, and the one most intro
  physics/chem courses actually grade against by name. Solves the
  system symbolically once (the same "solve once, evaluate the closed
  form" pattern `monte_carlo.py` and `chains.py` both use), then
  differentiates that one expression with respect to each uncertain
  input and evaluates every partial at the given central values --
  shown alongside a tornado-style breakdown of which input's
  uncertainty actually dominates the total variance.
- **Interval arithmetic / guaranteed bounds**
  (`modules/interval_arithmetic.py`, "📏 Guaranteed bounds for
  `<target>`") -- a genuinely different flavor of "how wrong could this
  be" than either of the above: given each uncertain input as a hard
  range (not a probability distribution), computes a range for the
  target that's PROVABLY guaranteed to contain every possible result --
  "cannot be outside this band," not "95% likely to be in this band."
  Implemented as a small, dependency-free `Interval` type with the
  standard (conservative) interval-arithmetic rules for +, -, *, /, and
  a few elementary functions; because Python's operator overloading
  means `sp.lambdify()`'s generated `+`/`-`/`*`/`/`/`**` expression works
  unmodified against `Interval` operands, the same "solve symbolically
  once" approach applies here too, just evaluated with different
  arithmetic underneath. Careful about the well-known interval-
  arithmetic gotchas: multiplication/division always check all four
  corner combinations rather than assuming positive operands (wrong
  whenever a range spans zero), and an even power of a range spanning
  zero has its minimum AT zero, not at either endpoint.
- **Goal-seek / inverse solve** (`modules/goal_seek.py`, "🎯 Goal seek:
  find the input for a target `<target>`") -- the inverse of
  `chains.sweep_step_binding`'s "vary this input and see what happens":
  "what value of this input makes the target hit a SPECIFIC number,"
  solved directly rather than by sweeping a range and reading a chart.
  Works by substituting the DESIRED value in place of the target's own
  symbol -- turning "solve for target, given inputs" into "solve for
  one input, given the desired target" -- and inverting that system.
  Tries an exact symbolic `sp.solve()` first (and shows the resulting
  formula, worth seeing as its own small derivation), falling back to
  `numerical_fallback.py`'s `mpmath.findroot` machinery -- the same
  fallback `verifier.py` itself uses whenever `sp.solve` can't invert an
  equation symbolically -- when the system doesn't yield to that
  (implicit or transcendental relationships, mainly). A declared domain
  restriction on the variable being sought (see
  `equation_engine.Variable.domain`) narrows a multi-root result (a
  quadratic goal, e.g., commonly has two) down to the physically
  sensible one or ones, without ever filtering the result down to
  nothing.

## Robustness / QA

Two developer-facing tools aimed at hardening the pipeline itself,
distinct from every student-facing feature above -- neither cares
whether an ANSWER looks sensible, both care whether the SYSTEM survives
what it's handed:

- **Adversarial edge-case generator** (`modules/adversarial_testing.py`,
  "🧪 Adversarial edge-case testing (developer QA)" in the Verify tab)
  -- takes an already-solved problem's own known inputs and generates
  deliberately nasty variants of each one (zero, a flipped sign, an
  extremely large or extremely small magnitude), then runs every
  variant through the real solving + plausibility pipeline and reports
  exactly what happened: solved cleanly, correctly recognized as
  unsolvable, timed out, or raised an exception. Wrapped in its own
  short timeout separate from the normal computation timeout, since an
  extreme magnitude can (through the same `sp.nsimplify()` performance
  cliff `monte_carlo.py`'s docstring describes hitting and routing
  around) occasionally make a solve pathologically slow rather than
  fast-failing -- that slowness is itself a finding worth surfacing, not
  something this tool should silently wait out. **This immediately
  found a real bug on first use**: `verifier._solve_sympy()` was calling
  a bare `float()` on every numeric solution, which raises an unhandled
  `TypeError` for a target that solves to a complex number (e.g. `sqrt`
  of a negative input) -- meaning ANY real problem whose given inputs
  happened to produce a non-real root could crash the whole app around
  it. Fixed at the source with the same tolerant complex-to-real
  conversion `monte_carlo.py`/`goal_seek.py` already use, with two
  regression tests locking it in.
- **Extraction diff mode** (`modules/extraction_diff.py`, "🔬 Extraction
  diff" mode) -- paste two DIFFERENT wordings of the same underlying
  problem and get back a structural, side-by-side diff of their
  independent extractions: which variables matched, which equations
  matched (via the same variable-name-independent canonicalization
  `similarity.py` already uses), and what changed. Distinct from
  `self_consistency.py`, which re-extracts the SAME wording several
  times and reports one aggregate similarity score -- this answers the
  debugging question that raises but doesn't answer: given two SPECIFIC
  wordings, what EXACTLY differs? Variables are matched by normalized
  MEANING text, not symbol name, since two independent extractions
  routinely pick different symbol letters for the same quantity (v_i vs
  v0) -- matching by name alone would report that mismatch as a
  "difference" on every single run, drowning out the differences that
  actually matter.

## UI streamlining

A few changes aimed purely at making the interface easier to navigate as
the feature list above has grown, with no changes to the underlying
solving/verification logic:

- **Secondary panels are grouped into tabs.** Right after the confidence
  banner, a solved problem now shows three tabs -- **🔎 Verify**
  (verification detail, domain of validity, physical plausibility,
  paranoid mode, self-consistency check), **📊 Explore** (dependency
  graph, the interactive plot/contour/feasible-region section), and
  **🎯 Practice** (grade my work, generate worksheet variants) -- instead
  of all of those expanders stacking in one long vertical scroll. The
  core content everyone always wants (derived equations, variables,
  step-by-step solution, ODE/recurrence solutions, follow-up Q&A) stays
  in the main flow below the tabs, always visible.
- **Mode navigation moved to the sidebar.** The word-problem-solver /
  curve-fitting / equivalence-checking / batch-solver / problem-chains
  selector used to be a horizontal radio competing for attention right
  above the main input box; it's now the first thing in the sidebar, so
  switching tools doesn't require scrolling past whatever's currently in
  the main content area.
- **Recent error patterns and the active chain are now visible in the
  sidebar at all times**, not just inside the one problem's own tabs/
  expanders where they'd disappear once you moved to a different
  problem -- both matter across an entire session, not just the problem
  currently on screen. The active-chain panel includes an "Open in
  Problem chains" button that switches modes directly.
- **A "🔗 Send this result to a chain" shortcut** on every solved
  problem (right below the confidence banner) creates a new chain -- or
  adds a step to an existing one -- from the CURRENT solved model in one
  click, rather than needing to re-paste the problem's text into the
  separate Problem chains mode.

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
- Vector variables' own `known_value` is always null -- only their declared
  `components` (ordinary scalar variables in their own right) carry numbers.
  `solve_for` can never name a vector variable directly (`sp.solve` needs a
  scalar target) -- solve for a component, or for a scalar equation's LHS
  that's defined via `dot`/`cross`/`magnitude` on the vector instead.
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

### Automated testing on every change

- **CI** (`.github/workflows/tests.yml`): the full suite runs automatically
  on every push and pull request, on a matrix of **ubuntu-latest AND
  windows-latest** × Python 3.11/3.12. Windows is included deliberately,
  not just as a formality -- this app targets Windows as a first-class
  local-run environment, and at least one module
  (`modules/timeout_utils.py`) exists specifically because a naive
  implementation (`signal.alarm`) would silently do nothing there;
  testing only on Linux would never catch that class of bug for real, it
  would just look green. The workflow also byte-compiles the whole
  project (including `app.py` itself, which the pytest suite never
  imports directly since it's a Streamlit script) as a cheap first check
  before running the actual suite. Trigger it manually from the Actions
  tab any time via `workflow_dispatch`.
- **Optional local pre-commit hook** (`.pre-commit-config.yaml`): runs
  the same suite before each commit, for immediate feedback rather than
  finding out something broke only after pushing -- the suite runs in
  roughly 10-15 seconds, so this isn't a meaningful slowdown. Opt in
  with:
  ```bash
  pip install pre-commit
  pre-commit install
  ```
  Skip it for a single commit with `git commit --no-verify`.