from __future__ import annotations

import copy
import json
from typing import Any

PROVIDER_METADATA_KEY = "_sylliptor_provider_metadata"
OPENAI_RESPONSES_PROVIDER_METADATA_KEY = "openai_responses"
ANTHROPIC_MESSAGES_PROVIDER_METADATA_KEY = "anthropic_messages"
GEMINI_GENERATE_CONTENT_PROVIDER_METADATA_KEY = "gemini_generate_content"
GEMINI_INTERACTIONS_PROVIDER_METADATA_KEY = "gemini_interactions"
TOOL_CALL_PROVIDER_METADATA_KEY = "_tool_calls"
DEEPSEEK_REASONING_CONTENT_KEY = "reasoning_content"
OPENROUTER_REASONING_KEY = "reasoning"
OPENROUTER_REASONING_DETAILS_KEY = "reasoning_details"

STATEFUL_PROVIDER_METADATA_KEYS = frozenset(
    {
        OPENAI_RESPONSES_PROVIDER_METADATA_KEY,
        ANTHROPIC_MESSAGES_PROVIDER_METADATA_KEY,
        GEMINI_GENERATE_CONTENT_PROVIDER_METADATA_KEY,
        GEMINI_INTERACTIONS_PROVIDER_METADATA_KEY,
    }
)


def merge_provider_metadata(*items: dict[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, dict) and value:
                merged[str(key)] = copy.deepcopy(value)
    return merged or None


def tool_call_metadata_entries(response: Any) -> list[dict[str, Any]]:
    tool_calls = getattr(response, "tool_calls", None)
    if not isinstance(tool_calls, list):
        return []
    entries: list[dict[str, Any]] = []
    for index, tool_call in enumerate(tool_calls):
        metadata = getattr(tool_call, "provider_metadata", None)
        merged = merge_provider_metadata(metadata)
        if not merged:
            continue
        entry: dict[str, Any] = {
            "index": index,
            "metadata": merged,
        }
        tool_call_id = str(getattr(tool_call, "id", "") or "")
        if tool_call_id:
            entry["id"] = tool_call_id
        entries.append(entry)
    return entries


def strip_provider_metadata_from_message(message: dict[str, Any]) -> dict[str, Any]:
    copied = dict(message)
    copied.pop(PROVIDER_METADATA_KEY, None)
    copied.pop(DEEPSEEK_REASONING_CONTENT_KEY, None)
    copied.pop(OPENROUTER_REASONING_KEY, None)
    copied.pop(OPENROUTER_REASONING_DETAILS_KEY, None)
    return copied


def attach_provider_metadata_to_assistant_message(
    message: dict[str, Any],
    response: Any,
) -> dict[str, Any]:
    if str(message.get("role") or "") != "assistant":
        return message
    message_metadata = merge_provider_metadata(getattr(response, "provider_metadata", None))
    if not message.get("tool_calls") and not (
        isinstance(message_metadata, dict)
        and any(key in message_metadata for key in STATEFUL_PROVIDER_METADATA_KEYS)
    ):
        return message
    tool_call_metadata = tool_call_metadata_entries(response)
    if not message_metadata and not tool_call_metadata:
        return message
    copied = dict(message)
    merged = dict(message_metadata or {})
    if tool_call_metadata:
        merged[TOOL_CALL_PROVIDER_METADATA_KEY] = tool_call_metadata
    copied[PROVIDER_METADATA_KEY] = merged
    return copied


def assistant_message_from_response(
    response: Any,
    *,
    content: str | None = None,
) -> dict[str, Any]:
    message_content = getattr(response, "content", "") if content is None else content
    message: dict[str, Any] = {
        "role": "assistant",
        "content": str(message_content or ""),
    }
    tool_calls = getattr(response, "tool_calls", None)
    if isinstance(tool_calls, list) and tool_calls:
        message["tool_calls"] = [
            {
                "id": str(getattr(tool_call, "id", "") or ""),
                "type": "function",
                "function": {
                    "name": str(getattr(tool_call, "name", "") or ""),
                    "arguments": json.dumps(getattr(tool_call, "arguments", {}) or {}),
                },
            }
            for tool_call in tool_calls
        ]
    return attach_provider_metadata_to_assistant_message(message, response)
