from __future__ import annotations

from dataclasses import dataclass

_NETWORK_MODEL_SIGNALS = (
    "llm request failed",
    "llm error",
    "network",
    "timeout",
    "timed out",
    "connection",
    "dns",
    "tls",
    "certificate",
    "api key",
    "unauthorized",
    "forbidden",
    "rate limit",
    "too many requests",
    "base_url",
    "chat/completions",
)

_TOOL_TRANSCRIPT_SIGNAL_GROUPS = (
    (
        "assistant message with 'tool_calls' must be followed by tool messages",
        'assistant message with "tool_calls" must be followed by tool messages',
    ),
    (
        "tool_call_id",
        "did not have response messages",
    ),
    (
        "tool transcript",
        "missing",
    ),
    (
        "tool transcript",
        "malformed",
    ),
    (
        "tool transcript",
        "incomplete",
    ),
)


@dataclass(frozen=True)
class LLMErrorDisplay:
    title: str
    guidance_lines: tuple[str, ...]


def is_tool_transcript_error(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    return any(all(token in lowered for token in group) for group in _TOOL_TRANSCRIPT_SIGNAL_GROUPS)


def is_network_or_model_error(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered or is_tool_transcript_error(lowered):
        return False
    return any(token in lowered for token in _NETWORK_MODEL_SIGNALS)


def classify_llm_error_display(text: str) -> LLMErrorDisplay:
    # Transcript integrity errors often arrive inside a generic "LLM error 400" wrapper,
    # so classify them before broader network/model guidance.
    if is_tool_transcript_error(text):
        return LLMErrorDisplay(
            title="Tool Transcript Error",
            guidance_lines=(
                "This session's tool transcript is incomplete or malformed.",
                "Retry in a new session. If it repeats, inspect resume/compaction around missing tool responses.",
            ),
        )
    if is_network_or_model_error(text):
        return LLMErrorDisplay(
            title="Network/Model Error",
            guidance_lines=(
                "Check base URL, API key, and network connectivity.",
                "Retry when ready. Use /status for current session settings.",
            ),
        )
    return LLMErrorDisplay(title="LLM Error", guidance_lines=())
