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
   equations.
2. **Verification** -- SymPy independently checks the equations parse,
   balance numerically against known values, and aren't under/over-determined.
   A *second*, independent LLM call re-solves the original word problem from
   scratch; if its numeric answer disagrees with SymPy's solve of the derived
   equations by more than ~2%, the derivation is considered flawed and
   automatically retried (up to `max_verification_retries` times) with the
   discrepancy fed back to the model.
3. **Step-by-step** -- SymPy computes the actual algebraic steps (substitute
   → isolate → simplify); the LLM only narrates those already-verified steps
   in plain language, rather than being trusted to invent the math live.
4. **Alternative scenarios** -- a separate, low-stakes LLM call suggests
   other real-world contexts with the same mathematical structure.
5. **Workspace & plotting** -- solved values can be pushed into a session
   workspace for reuse in later calculations (reference them by name in a
   new problem statement -- the model is automatically told their value);
   any equation with a free parameter can be plotted with Streamlit sliders
   controlling the other variables live.
6. **Dimensional checking** -- alongside numeric-balance checking, each
   equation is independently checked for physical-unit consistency via
   `sympy.physics.units`, catching errors a numeric check can't (e.g.
   equating a distance to a velocity, which could pass numerically by pure
   coincidence but never passes dimensionally).
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
