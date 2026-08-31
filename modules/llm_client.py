"""
Thin wrapper around LM Studio's local OpenAI-compatible server.

LM Studio (Developer tab -> "Start Server") exposes:
    POST http://localhost:1234/v1/chat/completions
which is schema-identical to OpenAI's chat completions endpoint, so we
reuse the official `openai` python SDK and just repoint base_url.
"""
import base64
import json
from openai import OpenAI, APIConnectionError

from config import settings
from modules.app_logging import logger


class LMStudioClient:
    def __init__(self):
        self._client = OpenAI(
            base_url=settings.lm_studio_base_url,
            api_key=settings.lm_studio_api_key,
        )

    def is_available(self) -> tuple[bool, str]:
        """Ping the server so the UI can show a clear connection status."""
        try:
            self._client.models.list()
            return True, "Connected to LM Studio."
        except APIConnectionError:
            return False, (
                "Could not reach LM Studio at "
                f"{settings.lm_studio_base_url}. Open LM Studio, load a model, "
                "and click 'Start Server' on the Developer tab."
            )
        except Exception as e:  # noqa: BLE001
            return False, f"LM Studio responded with an error: {e}"

    def list_models(self) -> list[str]:
        """Models LM Studio currently has loaded/served -- ground truth for
        what's actually runnable, as opposed to config.py's defaults."""
        try:
            return sorted(m.id for m in self._client.models.list().data)
        except Exception:  # noqa: BLE001
            return []

    def validate_model(self, model_name: str) -> tuple[bool, str]:
        """Checks model_name against what's actually loaded, distinguishing
        the two genuinely different failure modes that would otherwise both
        just surface as a similar-looking raw API error the moment a chat()
        call is attempted: the server isn't reachable at all, versus the
        server IS reachable but this particular model isn't one of the ones
        it has loaded (e.g. a typo, or a model that was unloaded since
        config.py/an env var was set). Used by paranoid.py's secondary-model
        check before attempting a real (more expensive, more confusing to
        fail) extraction call against it."""
        ok, msg = self.is_available()
        if not ok:
            return False, msg
        loaded = self.list_models()
        if not loaded:
            return False, "Connected to LM Studio, but no models are loaded at all."
        if model_name not in loaded:
            return False, (
                f"'{model_name}' isn't currently loaded in LM Studio. "
                f"Loaded models: {', '.join(loaded)}."
            )
        return True, f"'{model_name}' is loaded and ready."

    def chat(self, system: str, user: str, temperature: float,
              json_mode: bool = False, model: str | None = None) -> str:
        # NOTE: LM Studio's OpenAI-compat server is stricter than OpenAI's
        # own API here -- it rejects response_format={"type": "json_object"}
        # with a 400 ("must be 'json_schema' or 'text'"). Rather than build
        # a full json_schema per call site (extraction, narration, and
        # scenarios each have different shapes), we just ask for JSON in the
        # prompt and rely on extract_json()'s forgiving parser below, which
        # already strips code fences/prose. json_mode is kept as a parameter
        # so call sites can still signal intent even though it's a no-op here.
        try:
            resp = self._client.chat.completions.create(
                model=model or settings.reasoning_model,
                temperature=temperature,
                max_tokens=settings.max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            # Logged HERE, at the one gateway every LLM call in the app
            # goes through, rather than in each of the many try/except
            # blocks scattered across scenarios.py/solver.py/worksheet.py/
            # paranoid.py/followup.py/batch_solver.py that already catch
            # and gracefully recover from this -- recovering gracefully in
            # the UI is good, but it also means a recurring failure would
            # otherwise leave no trace once the page reruns. Re-raised
            # unchanged; this only adds a log line, not new behavior.
            logger.warning("LM Studio chat() call failed (model=%s): %s",
                             model or settings.reasoning_model, e)
            raise

    def vision_extract(self, image_bytes: bytes, mime_type: str = "image/png",
                         model: str | None = None) -> str:
        """Ask a multimodal model to transcribe + lightly describe an image's
        problem statement (handles handwritten/photographed word problems)."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        resp = self._client.chat.completions.create(
            model=model or settings.vision_model,
            temperature=0.0,
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You transcribe math/science problem statements from images "
                        "exactly as written, including any diagrams described in words. "
                        "Do not solve the problem. Output plain text only."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe the problem in this image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                        },
                    ],
                },
            ],
        )
        return resp.choices[0].message.content


class LLMOutputError(Exception):
    """Raised when the model's response couldn't be parsed as expected.
    Carries the raw output so the UI can show it for debugging instead of
    just crashing with a bare traceback."""
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output


def extract_json(raw: str) -> dict | list:
    """Best-effort JSON extraction for models that wrap JSON in prose/fences.
    Handles both object ({...}) and array ([...]) roots, since different
    call sites in this app expect different shapes (extraction wants an
    object, scenarios/narration want an array)."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.lstrip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.rstrip("`")
    text = text.strip()

    first_curly = text.find("{")
    first_square = text.find("[")
    candidates = [i for i in (first_curly, first_square) if i != -1]
    if not candidates:
        logger.warning("extract_json(): model returned no JSON at all (%d chars of raw output)", len(raw))
        raise LLMOutputError(
            "The model didn't return any JSON -- it may have replied with a "
            "clarifying question or plain prose instead of the structured "
            "format the app requires.",
            raw_output=raw,
        )
    start = min(candidates)
    closing = "}" if text[start] == "{" else "]"
    end = text.rfind(closing)
    if end == -1 or end < start:
        logger.warning("extract_json(): model's JSON output looks truncated (%d chars of raw output)", len(raw))
        raise LLMOutputError("The model's JSON output looks truncated (no closing bracket found).",
                              raw_output=raw)
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        logger.warning("extract_json(): JSON decode failed: %s", e)
        raise LLMOutputError(f"The model's JSON didn't parse ({e}).", raw_output=raw) from e
