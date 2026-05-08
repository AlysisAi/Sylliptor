from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .llm.openai_compat import strip_provider_metadata_from_message
from .token_budget import estimate_tokens

_MEMORY_MARKER = "<<<SYLLIPTOR_CONVERSATION_MEMORY_JSON>>>"
_PINS_MARKER = "<<<SYLLIPTOR_CONVERSATION_PINS_JSON>>>"


@dataclass(frozen=True)
class RequestTokenBreakdown:
    bootstrap_prompt_tokens: int = 0
    tool_schema_tokens: int = 0
    live_conversation_history_tokens: int = 0
    inline_tool_transcript_tokens: int = 0
    memory_summary_tokens: int = 0
    pins_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.bootstrap_prompt_tokens
            + self.tool_schema_tokens
            + self.live_conversation_history_tokens
            + self.inline_tool_transcript_tokens
            + self.memory_summary_tokens
            + self.pins_tokens
        )

    def to_payload(self) -> dict[str, int]:
        return {
            "bootstrap_prompt_tokens": self.bootstrap_prompt_tokens,
            "tool_schema_tokens": self.tool_schema_tokens,
            "live_conversation_history_tokens": self.live_conversation_history_tokens,
            "inline_tool_transcript_tokens": self.inline_tool_transcript_tokens,
            "memory_summary_tokens": self.memory_summary_tokens,
            "pins_tokens": self.pins_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> RequestTokenBreakdown | None:
        if not isinstance(payload, dict):
            return None

        def _as_non_negative_int(key: str) -> int:
            value = payload.get(key)
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                return 0
            return max(0, parsed)

        breakdown = cls(
            bootstrap_prompt_tokens=_as_non_negative_int("bootstrap_prompt_tokens"),
            tool_schema_tokens=_as_non_negative_int("tool_schema_tokens"),
            live_conversation_history_tokens=_as_non_negative_int(
                "live_conversation_history_tokens"
            ),
            inline_tool_transcript_tokens=_as_non_negative_int("inline_tool_transcript_tokens"),
            memory_summary_tokens=_as_non_negative_int("memory_summary_tokens"),
            pins_tokens=_as_non_negative_int("pins_tokens"),
        )
        if breakdown.total_tokens == 0 and not any(
            key in payload
            for key in (
                "bootstrap_prompt_tokens",
                "tool_schema_tokens",
                "live_conversation_history_tokens",
                "inline_tool_transcript_tokens",
                "memory_summary_tokens",
                "pins_tokens",
            )
        ):
            return None
        return breakdown


def _normalize_message_content_for_estimation(content: Any) -> Any:
    if not isinstance(content, list):
        return content

    normalized: list[Any] = []
    for part in content:
        if not isinstance(part, dict):
            normalized.append(part)
            continue
        copied = dict(part)
        if copied.get("type") == "image_url" and isinstance(copied.get("image_url"), dict):
            image_url = dict(copied["image_url"])
            url = image_url.get("url")
            if isinstance(url, str):
                if url.startswith("data:"):
                    prefix = url.split(",", 1)[0]
                    image_url["url"] = f"{prefix},<omitted>"
                elif len(url) > 160:
                    image_url["url"] = url[:160] + "...<omitted>"
            copied["image_url"] = image_url
        normalized.append(copied)
    return normalized


def sanitize_messages_for_estimation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        copied = strip_provider_metadata_from_message(msg)
        copied["content"] = _normalize_message_content_for_estimation(copied.get("content"))
        sanitized.append(copied)
    return sanitized


def estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    if not messages:
        return 0
    payload = json.dumps(
        sanitize_messages_for_estimation(messages),
        ensure_ascii=False,
        sort_keys=True,
    )
    return estimate_tokens(payload)


def estimate_tool_schema_tokens(tool_list: list[dict[str, Any]] | None) -> int:
    if not tool_list:
        return 0
    tool_payload = json.dumps(tool_list, ensure_ascii=False, sort_keys=True)
    return estimate_tokens(tool_payload)


def estimate_request_tokens(
    messages: list[dict[str, Any]],
    tool_list: list[dict[str, Any]] | None,
) -> int:
    return estimate_message_tokens(messages) + estimate_tool_schema_tokens(tool_list)


def estimate_request_token_breakdown(
    *,
    messages: list[dict[str, Any]],
    tool_list: list[dict[str, Any]] | None,
    pinned_prefix_len: int = 0,
) -> RequestTokenBreakdown:
    sanitized_messages = sanitize_messages_for_estimation(messages)
    normalized_pinned_prefix_len = max(0, min(int(pinned_prefix_len), len(sanitized_messages)))

    bootstrap_messages: list[dict[str, Any]] = []
    live_history_messages: list[dict[str, Any]] = []
    inline_tool_messages: list[dict[str, Any]] = []
    memory_messages: list[dict[str, Any]] = []
    pins_messages: list[dict[str, Any]] = []

    for idx, message in enumerate(sanitized_messages):
        content = message.get("content")
        content_text = content if isinstance(content, str) else ""

        if content_text.startswith(_MEMORY_MARKER):
            memory_messages.append(message)
            continue
        if content_text.startswith(_PINS_MARKER):
            pins_messages.append(message)
            continue

        if str(message.get("role") or "") == "tool" or message.get("tool_calls"):
            inline_tool_messages.append(message)
            continue

        if idx < normalized_pinned_prefix_len:
            bootstrap_messages.append(message)
            continue

        live_history_messages.append(message)

    return RequestTokenBreakdown(
        bootstrap_prompt_tokens=estimate_message_tokens(bootstrap_messages),
        tool_schema_tokens=estimate_tool_schema_tokens(tool_list),
        live_conversation_history_tokens=estimate_message_tokens(live_history_messages),
        inline_tool_transcript_tokens=estimate_message_tokens(inline_tool_messages),
        memory_summary_tokens=estimate_message_tokens(memory_messages),
        pins_tokens=estimate_message_tokens(pins_messages),
    )
