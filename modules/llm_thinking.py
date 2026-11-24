"""
Helpers for "thinking" (reasoning) models served by LM Studio. SDK-free and Streamlit-free, so they are unit-testable without a server.

What goes wrong with them: a thinking model spends its token budget reasoning before it writes the answer. The OpenAI-compatible endpoint returns the reasoning in a separate
`reasoning_content` field, so when the budget runs out the answer field is EMPTY ("The model sent back an empty reply"), and raising the budget only lets it think for longer
(some models loop). LM Studio does not reliably honour a per-request "don't think" switch (it is a model-load setting in its own UI for several Qwen releases), and when a JSON schema
is requested it applies that schema to the reasoning stream, so the finished JSON can arrive in the reasoning field with the answer field empty. So the app:

  1. asks for no thinking anyway (harmless where ignored),
  2. streams the reply and stops it once the reasoning alone has used a budget with no answer text started (`collect_stream`), instead of waiting for the full token limit,
  3. retries once with a JSON schema, which forces the output to start as JSON, and
  4. reads a finished JSON object out of the reasoning text when the answer field is empty (`recover_json`).

`LMStudioClient.chat` (modules/llm_client.py) runs that ladder.
"""
import json
from dataclasses import dataclass, field
from typing import Any

# What a reply must look like when no per-call-site schema is known: any JSON object or array (every call that uses json_mode expects one of the two).
GENERIC_JSON_SCHEMA: dict[str, Any] = {"anyOf": [{"type": "object"}, {"type": "array"}]}


@dataclass
class Collected:
    """One model call, however it was made (streamed or not)."""
    content: str = ""
    reasoning: str = ""
    finish_reason: str | None = None
    chunks: int = 0                      # streamed deltas received: about one per token
    aborted: bool = False                # we stopped the stream: the reasoning used its budget and no answer text had begun
    extra: dict[str, Any] = field(default_factory=dict)


def _delta_text(delta: Any, *names: str) -> str:
    for name in names:
        value = getattr(delta, name, None)
        if isinstance(value, str) and value:
            return value
    return ""


def collect_stream(stream: Any, reasoning_budget_tokens: int) -> Collected:
    """Read an OpenAI-style chat completion stream into one result. If more than `reasoning_budget_tokens` chunks of reasoning (or an unfinished inline <think> block) have
    arrived and no answer text has started, close the stream, which makes the server stop generating, and report `aborted`. A budget of 0 means no limit."""
    out = Collected()
    reasoning_chunks = 0
    try:
        for chunk in stream:
            out.chunks += 1
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is not None:
                piece = _delta_text(delta, "content")
                thought = _delta_text(delta, "reasoning_content", "reasoning")
                if piece:
                    out.content += piece
                if thought:
                    out.reasoning += thought
                    reasoning_chunks += 1
            if getattr(choice, "finish_reason", None):
                out.finish_reason = choice.finish_reason
            if reasoning_budget_tokens > 0:
                inline_open = "<think>" in out.content and "</think>" not in out.content
                answered = bool(out.content.strip()) and not inline_open
                spent = reasoning_chunks + (out.chunks if inline_open else 0)
                if not answered and spent > reasoning_budget_tokens:
                    out.aborted = True
                    break
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:                                                # noqa: BLE001 -- closing a stream must never raise over the real result
                pass
    return out


def recover_json(text: str) -> str | None:
    """The last complete JSON object (or array) found in `text` that has real content, or None. Used on the reasoning field when the answer field came back empty: a thinking
    model that finished (or was forced by a schema) often wrote the finished JSON there. 'Last', because drafts come before the final version; 'real content' because a
    stray `{}` or `{x}` in a formula is not an answer: an object needs at least two keys, an array at least one element."""
    if not text:
        return None
    decoder = json.JSONDecoder()
    best: str | None = None
    i = 0
    while i < len(text):
        if text[i] in "{[":
            try:
                value, end = decoder.raw_decode(text, i)
            except ValueError:
                i += 1
                continue
            if (isinstance(value, dict) and len(value) >= 2) or (isinstance(value, list) and len(value) >= 1):
                best = text[i:end]
            i = end
        else:
            i += 1
    return best
