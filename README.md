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