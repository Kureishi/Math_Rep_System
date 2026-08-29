"""
Central configuration for the Math Representation System.
Edit these values (or override with environment variables) to point at
your local LM Studio server / preferred models.
"""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # LM Studio exposes an OpenAI-compatible server. Default port is 1234.
    lm_studio_base_url: str = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    lm_studio_api_key: str = os.getenv("LM_STUDIO_API_KEY", "lm-studio")  # LM Studio ignores the value but the SDK requires one

    # Model identifiers as they appear in LM Studio's "My Models" / server tab.
    # Any text-capable model works for reasoning_model. vision_model must be
    # a multimodal model (e.g. a Qwen2-VL / LLaVA / InternVL family model)
    # if you want image problems solved without a separate OCR step.
    reasoning_model: str = os.getenv("LM_REASONING_MODEL", "qwen2.5-14b-instruct")
    vision_model: str = os.getenv("LM_VISION_MODEL", "qwen2-vl-7b-instruct")

    # Optional second model for "paranoid mode" multi-model cross-verification
    # (modules/paranoid.py) -- leave blank (the default) to leave that
    # feature off; set to another model already loaded in LM Studio to
    # enable it.
    secondary_reasoning_model: str = os.getenv("LM_SECONDARY_MODEL", "")

    # Verification behavior
    max_verification_retries: int = 2
    numeric_tolerance: float = 1e-6
    cross_check_tolerance: float = 0.02  # 2% -- how far the independent re-solve
                                          # can differ from the derived answer before
                                          # verification flags a disagreement

    # Generation behavior
    temperature_extraction: float = 0.1   # low temp: we want faithful, reproducible math
    temperature_narration: float = 0.4    # slightly higher for readable explanations
    max_tokens: int = 2048


settings = Settings()
