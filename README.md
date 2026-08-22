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
   relations. Each relation is tagged as a plain **equation**, an
   **inequality** (a constraint like `v <= 25`), or an **ODE** (a
   differential equation for a declared function like `y(t)`) -- each kind
   is parsed, verified, and solved differently.
2. **Verification** -- SymPy independently checks that relations parse,
   balance numerically against known values (equations), are actually
   satisfied by the given numbers (inequalities), or symbolically satisfy
   the original differential equation via `sp.checkodesol` (ODEs) -- plus a
   physical-unit consistency check via `sympy.physics.units` for equations
   and inequalities. Every passing check also reports *how close* it came
   to its own tolerance (`essentially exact` vs `borderline`), not just
   pass/fail -- a residual of 1e-9 and one that barely cleared the bar look
   different in the UI. For algebraic equations, a *second*, independent
   LLM call re-solves the original word problem from scratch; if its
   numeric answer disagrees with SymPy's solve by more than ~2%, the
   derivation is considered flawed and automatically retried (up to
   `max_verification_retries` times) with the discrepancy fed back to the
   model.
3. **Step-by-step** -- SymPy computes the actual steps (substitute -> isolate
   -> simplify for equations; `reduce_inequalities` for constraints;
   `dsolve` general solution -> apply initial conditions for ODEs); the LLM
   only narrates those already-verified steps in plain language, rather
   than being trusted to invent the math live.
4. **Alternative scenarios** -- a separate, low-stakes LLM call suggests
   other real-world contexts with the same mathematical structure.
5. **Workspace & plotting** -- solved values can be pushed into a session
   workspace for reuse in later calculations (reference them by name in a
   new problem statement -- the model is automatically told their value).
   Any equation with one free parameter gets a live 2D line plot; with two
   free parameters, a live 3D surface plot. ODE solutions get their own
   plot of the solution curve plus a "evaluate at a specific point"
   control (e.g. population after 10 years).
6. **Dimensional checking** -- alongside numeric-balance checking, each
   equation/inequality is independently checked for physical-unit
   consistency via `sympy.physics.units`, catching errors a numeric check
   can't (e.g. equating a distance to a velocity, which could pass
   numerically by pure coincidence but never passes dimensionally).
7. **History** -- every solved problem is saved to a local SQLite file
   (`data/history.db`, gitignored) and listed in the sidebar. Loading a
   past problem restores everything -- equations, verification, steps,
   scenarios -- with no new LLM calls.
8. **Export** -- download any solved problem as Markdown (LaTeX equations
   included) or a typeset PDF (equations rendered via matplotlib's
   mathtext -- no system LaTeX install needed).

## Extending it

- Swap Streamlit for a desktop shell (e.g. `pywebview` wrapping the same
  Streamlit app, or a PyQt front end calling the same `modules/`) if you want
  a native window instead of a browser tab -- the `modules/` package has no
  Streamlit dependency, so it's reusable as-is.
- The verification tolerance, retry count, and temperatures are all in
  `config.py`.
- `modules/ode_utils.py` centralizes ODE-solving (used by both
  `solver.py` and `verifier.py`) specifically to avoid a circular import
  between those two -- keep that pattern in mind if you add another
  cross-cutting solve step.
- Currently unsupported: systems of ODEs (only single first-order ODEs per
  function), inequalities with more than one free symbol after
  substitution (multi-variable feasible regions aren't visualized), and
  dimensional checking for ODEs (Derivative dimensional analysis isn't
  implemented, so ODE equations skip that check).
