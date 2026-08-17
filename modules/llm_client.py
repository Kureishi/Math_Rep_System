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

    def chat(self, system: str, user: str, temperature: float, json_mode: bool = False) -> str:
        kwargs = {}
        if json_mode:
            # Most LM Studio-hosted models honor this; if a model doesn't,
            # equation_engine.py has a fallback JSON extractor.
            kwargs["response_format"] = {"type": "json_object"}

        resp = self._client.chat.completions.create(
            model=settings.reasoning_model,
            temperature=temperature,
            max_tokens=settings.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )
        return resp.choices[0].message.content

    def vision_extract(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
        """Ask a multimodal model to transcribe + lightly describe an image's
        problem statement (handles handwritten/photographed word problems)."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        resp = self._client.chat.completions.create(
            model=settings.vision_model,
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


def extract_json(raw: str) -> dict:
    """Best-effort JSON extraction for models that wrap JSON in prose/fences."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output:\n{raw}")
    return json.loads(raw[start : end + 1])
