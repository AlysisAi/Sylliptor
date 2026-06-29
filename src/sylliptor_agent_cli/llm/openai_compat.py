from __future__ import annotations

import copy
import json
import logging
import re
import threading
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import httpx

from ..failure_category import provider_unavailable_retry_reason
from ..provider_telemetry import ProviderCallTelemetryRecorder
from .metadata import (
    DEEPSEEK_REASONING_CONTENT_KEY as _DEEPSEEK_REASONING_CONTENT_KEY,
)
from .metadata import (
    OPENROUTER_REASONING_DETAILS_KEY as _OPENROUTER_REASONING_DETAILS_KEY,
)
from .metadata import (
    OPENROUTER_REASONING_KEY as _OPENROUTER_REASONING_KEY,
)
from .metadata import (
    PROVIDER_METADATA_KEY,
    strip_provider_metadata_from_message,
)
from .metadata import (
    TOOL_CALL_PROVIDER_METADATA_KEY as _TOOL_CALL_PROVIDER_METADATA_KEY,
)
from .metadata import (
    assistant_message_from_response as assistant_message_from_response,
)
from .metadata import (
    attach_provider_metadata_to_assistant_message as attach_provider_metadata_to_assistant_message,
)
from .metadata import (
    merge_provider_metadata as _merge_provider_metadata,
)
from .provider_limits import (
    DEFAULT_PROVIDER_CONCURRENCY_CAPS,
    ProviderRetrySettings,
    best_effort_provider_key,
    run_provider_limited_call,
)
from .types import LLMError, LLMResponse, LLMUsage, ToolCall

_TEXT_LIKE_CONTENT_PART_TYPES = {"text", "output_text"}
_DEEPSEEK_PROVIDER_KEY = "deepseek"
_OPENROUTER_PROVIDER_KEY = "openrouter"
_GEMINI_PROVIDER_KEY = "gemini"
_GEMINI_EXTRA_CONTENT_KEY = "extra_content"
_OPENAI_STYLE_REASONING_EFFORT_PROVIDERS = frozenset({"openai", "azure", "mistral"})
_GEMINI_REASONING_EFFORTS = frozenset({"minimal", "low", "medium", "high"})
_DEFAULT_ACCEPT_ENCODING = "identity"
_LOGGER = logging.getLogger(__name__)
_TEMPERATURE_DEFAULT_VALUE = 1.0
_TEMPERATURE_COMPAT_MODE_DEFAULT = "default_temperature"
_TEMPERATURE_COMPAT_MODE_OMIT = "omit_temperature"
_TEMPERATURE_COMPAT_MODES = {
    _TEMPERATURE_COMPAT_MODE_DEFAULT,
    _TEMPERATURE_COMPAT_MODE_OMIT,
}
_TEMPERATURE_UNSUPPORTED_STATUS_CODES = {400, 422}
_TEMPERATURE_UNSUPPORTED_TOKENS = (
    "allowed",
    "greater than",
    "invalid",
    "unsupported",
    "not support",
    "not supported",
    "not allowed",
    "out of range",
    "range",
    "deprecated",
    "only the default",
    "cannot be set",
    "must be omitted",
)


def _headers_with_default_accept_encoding(headers: dict[str, str]) -> dict[str, str]:
    request_headers = dict(headers)
    if not any(key.lower() == "accept-encoding" for key in request_headers):
        request_headers["Accept-Encoding"] = _DEFAULT_ACCEPT_ENCODING
    return request_headers


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and left[idx] == right[idx]:
        idx += 1
    return idx


def _suffix_prefix_overlap_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    for size in range(limit, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


def _strip_cumulative_restart_suffix(*, previous: str, incoming: str) -> str | None:
    candidate = incoming
    while candidate:
        trimmed = candidate.lstrip("\r\n ")
        if trimmed.startswith(previous):
            remainder = trimmed[len(previous) :]
            trimmed_remainder = remainder.lstrip("\r\n ")
            if trimmed_remainder.startswith(previous):
                candidate = trimmed_remainder
                continue
            return remainder
        if trimmed != candidate:
            candidate = trimmed
            continue
        break
    return None


def _looks_like_alternate_cumulative_restart(*, previous: str, incoming: str) -> bool:
    if len(previous) < 120 or len(incoming) < 80:
        return False
    if incoming.startswith(previous) or previous.startswith(incoming):
        return False
    common_prefix = _common_prefix_length(previous, incoming)
    if common_prefix < 24:
        return False
    return common_prefix < (min(len(previous), len(incoming)) // 2)


def _stream_delta_suffix(*, previous: str, incoming: str) -> str:
    if not incoming:
        return ""
    if not previous:
        return incoming
    if incoming == previous:
        return ""
    if _looks_like_alternate_cumulative_restart(previous=previous, incoming=incoming):
        return ""
    cumulative_suffix = _strip_cumulative_restart_suffix(previous=previous, incoming=incoming)
    if cumulative_suffix is not None:
        return cumulative_suffix
    common_prefix = _common_prefix_length(previous, incoming)
    if common_prefix >= max(16, min(len(previous), len(incoming)) // 2):
        return incoming[common_prefix:]
    previous_restart = incoming.rfind(previous)
    if previous_restart > 0:
        prefix = incoming[:previous_restart]
        if not prefix.strip():
            return incoming[previous_restart + len(previous) :]
    overlap = _suffix_prefix_overlap_length(previous, incoming)
    if overlap > 0:
        restarted_suffix = _strip_cumulative_restart_suffix(
            previous=previous,
            incoming=incoming[overlap:],
        )
        if restarted_suffix is not None:
            return restarted_suffix
    if overlap >= max(4, len(incoming) // 2):
        return incoming[overlap:]
    return incoming


def _sanitize_transport_text(text: str) -> str:
    if not text:
        return text
    if not any(0xD800 <= ord(ch) <= 0xDFFF for ch in text):
        return text
    try:
        # Recover surrogate-escaped terminal bytes when possible, and replace
        # genuinely invalid sequences so JSON transport never crashes.
        return text.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
    except Exception:
        return text.encode("utf-8", errors="replace").decode("utf-8")


def _sanitize_transport_value(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_transport_text(value)
    if isinstance(value, list):
        return [_sanitize_transport_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_transport_value(item) for item in value)
    if isinstance(value, dict):
        return {
            _sanitize_transport_text(key)
            if isinstance(key, str)
            else key: _sanitize_transport_value(item)
            for key, item in value.items()
        }
    return value


def _normalize_provider_key(provider_key: str | None) -> str:
    normalized = str(provider_key or "").strip().casefold()
    return "".join(char if char.isalnum() else "_" for char in normalized).strip("_")


_SYLLIPTOR_TRIAL_PROXY_PATH_MARKER = "/functions/v1/llm"


def _is_sylliptor_trial_proxy(*, host: str, path: str) -> bool:
    """True for the hosted Sylliptor MiMo trial proxy (a Supabase Edge Function).

    Match the ``/functions/v1/llm`` proxy path so unrelated ``*.supabase.co``
    apps are never misclassified.
    """
    if _SYLLIPTOR_TRIAL_PROXY_PATH_MARKER not in path:
        return False
    return host == "supabase.co" or host.endswith(".supabase.co")


def _provider_key_from_base_url(base_url: str | None) -> str | None:
    raw = str(base_url or "").strip()
    if not raw:
        return None
    try:
        parsed = urlparse(raw)
        host = (parsed.hostname or "").rstrip(".").casefold()
        path = (parsed.path or "").casefold()
    except Exception:
        return None
    if not host:
        return None
    if _is_sylliptor_trial_proxy(host=host, path=path):
        # Hosted Sylliptor MiMo trial: a Supabase Edge Function that forwards to
        # OpenRouter/Xiaomi and speaks the OpenRouter reasoning shape. Classifying
        # it as openrouter activates reasoning capture/round-trip so a MiMo turn
        # that answers only in the reasoning channel is not lost.
        return _OPENROUTER_PROVIDER_KEY
    if "dashscope" in host:
        return "qwen"
    if host == "openrouter.ai" or host.endswith(".openrouter.ai"):
        return _OPENROUTER_PROVIDER_KEY
    if host == "api.openai.com":
        return "openai"
    if (
        host.endswith(".openai.azure.com")
        or host.endswith(".cognitiveservices.azure.com")
        or host.endswith(".services.ai.azure.com")
    ):
        return "azure"
    if host == "api.deepseek.com" or host.endswith(".deepseek.com"):
        return _DEEPSEEK_PROVIDER_KEY
    if host == "generativelanguage.googleapis.com":
        return "gemini"
    if host == "api.mistral.ai" or host.endswith(".mistral.ai"):
        return "mistral"
    if host == "api.x.ai" or host == "x.ai" or host.endswith(".x.ai"):
        return "xai"
    return None


def _transport_provider_key(
    *,
    base_url: str | None,
    provider_key: str | None,
    model: str | None,
) -> str:
    from_url = _provider_key_from_base_url(base_url)
    if from_url:
        return from_url
    normalized_provider = _normalize_provider_key(provider_key)
    if normalized_provider:
        if normalized_provider in {"dashscope", "qwen", "aliyun", "aliyuncs"}:
            return "qwen"
        return normalized_provider
    return _normalize_provider_key(best_effort_provider_key(base_url=base_url, model=model))


def _is_deepseek_provider(provider_key: str | None) -> bool:
    return _normalize_provider_key(provider_key) == _DEEPSEEK_PROVIDER_KEY


def _is_openrouter_provider(provider_key: str | None) -> bool:
    return _normalize_provider_key(provider_key) == _OPENROUTER_PROVIDER_KEY


def _is_gemini_provider(provider_key: str | None) -> bool:
    return _normalize_provider_key(provider_key) == _GEMINI_PROVIDER_KEY


def _is_dashscope_provider(provider_key: str | None) -> bool:
    return _normalize_provider_key(provider_key) in {"qwen", "dashscope"}


def _uses_reasoning_effort(provider_key: str | None) -> bool:
    return _normalize_provider_key(provider_key) in _OPENAI_STYLE_REASONING_EFFORT_PROVIDERS


def _model_name_parts(model: str | None) -> set[str]:
    normalized = str(model or "").strip().casefold()
    return {part for part in re.split(r"[^a-z0-9]+", normalized) if part}


def _gemini_model_allows_none_reasoning_effort(model: str | None) -> bool:
    parts = _model_name_parts(model)
    if "gemini" not in parts or "2" not in parts or "5" not in parts:
        return False
    if "pro" in parts:
        return False
    return "flash" in parts


def _gemini_reasoning_effort(
    *,
    model: str | None,
    reasoning_effort: str | None,
) -> str | None:
    effort = str(reasoning_effort or "").strip().casefold()
    if not effort:
        return None
    if effort in _GEMINI_REASONING_EFFORTS:
        return effort
    if effort == "none" and _gemini_model_allows_none_reasoning_effort(model):
        return effort
    return None


def _reasoning_effort_enables_thinking(reasoning_effort: str | None) -> bool | None:
    normalized = _normalize_provider_key(reasoning_effort)
    if not normalized:
        return None
    return normalized != "none"


def _deepseek_reasoning_payload_enabled(
    *,
    enable_thinking: bool | None,
    reasoning_effort: str | None,
) -> bool | None:
    if enable_thinking is not None:
        return enable_thinking
    return _reasoning_effort_enables_thinking(reasoning_effort)


def _openrouter_reasoning_payload(
    *,
    enable_thinking: bool | None,
    reasoning_effort: str | None,
) -> dict[str, Any] | None:
    normalized_effort = str(reasoning_effort or "").strip().lower()
    if normalized_effort:
        return {"effort": normalized_effort}
    if enable_thinking is True:
        return {"enabled": True}
    if enable_thinking is False:
        return {"enabled": False}
    return None


def _deepseek_reasoning_provider_metadata(reasoning_content: str) -> dict[str, Any] | None:
    reasoning = str(reasoning_content or "")
    if not reasoning:
        return None
    return {
        _DEEPSEEK_PROVIDER_KEY: {
            _DEEPSEEK_REASONING_CONTENT_KEY: reasoning,
        }
    }


def _openrouter_reasoning_provider_metadata(
    *,
    reasoning: str | None = None,
    reasoning_details: Any = None,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    reasoning_text = str(reasoning or "")
    if reasoning_text:
        payload[_OPENROUTER_REASONING_KEY] = reasoning_text
    if isinstance(reasoning_details, list) and reasoning_details:
        payload[_OPENROUTER_REASONING_DETAILS_KEY] = reasoning_details
    if not payload:
        return None
    return {_OPENROUTER_PROVIDER_KEY: payload}


def _provider_metadata_for_reasoning(
    *,
    provider_key: str | None,
    message: dict[str, Any],
) -> dict[str, Any] | None:
    if _is_deepseek_provider(provider_key):
        reasoning = message.get(_DEEPSEEK_REASONING_CONTENT_KEY)
        return _deepseek_reasoning_provider_metadata(
            reasoning if isinstance(reasoning, str) else ""
        )
    if _is_openrouter_provider(provider_key):
        reasoning = message.get(_OPENROUTER_REASONING_KEY)
        reasoning_details = message.get(_OPENROUTER_REASONING_DETAILS_KEY)
        return _openrouter_reasoning_provider_metadata(
            reasoning=reasoning if isinstance(reasoning, str) else None,
            reasoning_details=reasoning_details,
        )
    return None


def _deepseek_reasoning_from_provider_metadata(metadata: Any) -> str:
    if not isinstance(metadata, dict):
        return ""
    deepseek = metadata.get(_DEEPSEEK_PROVIDER_KEY)
    if not isinstance(deepseek, dict):
        return ""
    reasoning = deepseek.get(_DEEPSEEK_REASONING_CONTENT_KEY)
    return reasoning if isinstance(reasoning, str) else ""


def _openrouter_reasoning_from_provider_metadata(metadata: Any) -> tuple[str, list[Any] | None]:
    if not isinstance(metadata, dict):
        return "", None
    openrouter = metadata.get(_OPENROUTER_PROVIDER_KEY)
    if not isinstance(openrouter, dict):
        return "", None
    reasoning = openrouter.get(_OPENROUTER_REASONING_KEY)
    reasoning_details = openrouter.get(_OPENROUTER_REASONING_DETAILS_KEY)
    return (
        reasoning if isinstance(reasoning, str) else "",
        list(reasoning_details) if isinstance(reasoning_details, list) else None,
    )


def _gemini_tool_call_provider_metadata(tool_call: dict[str, Any]) -> dict[str, Any] | None:
    extra_content = tool_call.get(_GEMINI_EXTRA_CONTENT_KEY)
    if not isinstance(extra_content, dict) or not extra_content:
        return None
    return {
        _GEMINI_PROVIDER_KEY: {
            _GEMINI_EXTRA_CONTENT_KEY: copy.deepcopy(extra_content),
        }
    }


def _copy_transport_tool_calls(
    tool_calls: Any,
    *,
    preserve_extra_content: bool,
) -> list[Any] | Any:
    if not isinstance(tool_calls, list):
        return tool_calls
    copied_tool_calls: list[Any] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            copied_tool_calls.append(tool_call)
            continue
        copied_tool_call = copy.deepcopy(tool_call)
        if not preserve_extra_content:
            copied_tool_call.pop(_GEMINI_EXTRA_CONTENT_KEY, None)
        copied_tool_calls.append(copied_tool_call)
    return copied_tool_calls


def _gemini_extra_content_indexes(metadata: Any) -> tuple[dict[str, Any], dict[int, Any]]:
    if not isinstance(metadata, dict):
        return {}, {}
    entries = metadata.get(_TOOL_CALL_PROVIDER_METADATA_KEY)
    if not isinstance(entries, list):
        return {}, {}
    by_id: dict[str, Any] = {}
    by_index: dict[int, Any] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_metadata = entry.get("metadata")
        if not isinstance(entry_metadata, dict):
            continue
        gemini = entry_metadata.get(_GEMINI_PROVIDER_KEY)
        if not isinstance(gemini, dict):
            continue
        extra_content = gemini.get(_GEMINI_EXTRA_CONTENT_KEY)
        if not isinstance(extra_content, dict) or not extra_content:
            continue
        tool_call_id = entry.get("id")
        if isinstance(tool_call_id, str) and tool_call_id:
            by_id[tool_call_id] = copy.deepcopy(extra_content)
        index = entry.get("index")
        if isinstance(index, int):
            by_index[index] = copy.deepcopy(extra_content)
    return by_id, by_index


def _reattach_gemini_tool_call_extra_content(
    message: dict[str, Any],
    metadata: Any,
) -> None:
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return
    by_id, by_index = _gemini_extra_content_indexes(metadata)
    if not by_id and not by_index:
        return
    reattached: list[Any] = []
    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            reattached.append(tool_call)
            continue
        copied_tool_call = dict(tool_call)
        tool_call_id = str(copied_tool_call.get("id") or "")
        extra_content = by_id.get(tool_call_id) if tool_call_id else None
        if extra_content is None:
            extra_content = by_index.get(index)
        if isinstance(extra_content, dict) and extra_content:
            copied_tool_call[_GEMINI_EXTRA_CONTENT_KEY] = copy.deepcopy(extra_content)
        reattached.append(copied_tool_call)
    message["tool_calls"] = reattached


def _message_for_transport(
    message: dict[str, Any],
    *,
    provider_key: str | None,
) -> dict[str, Any]:
    metadata = message.get(PROVIDER_METADATA_KEY)
    copied = strip_provider_metadata_from_message(message)
    if str(copied.get("role") or "") != "assistant" or not copied.get("tool_calls"):
        return copied
    copied["tool_calls"] = _copy_transport_tool_calls(
        copied.get("tool_calls"),
        preserve_extra_content=_is_gemini_provider(provider_key),
    )
    if _is_deepseek_provider(provider_key):
        reasoning = _deepseek_reasoning_from_provider_metadata(metadata)
        if reasoning:
            copied[_DEEPSEEK_REASONING_CONTENT_KEY] = reasoning
    elif _is_openrouter_provider(provider_key):
        reasoning, reasoning_details = _openrouter_reasoning_from_provider_metadata(metadata)
        if reasoning:
            copied[_OPENROUTER_REASONING_KEY] = reasoning
        if reasoning_details:
            copied[_OPENROUTER_REASONING_DETAILS_KEY] = reasoning_details
    elif _is_gemini_provider(provider_key):
        _reattach_gemini_tool_call_extra_content(copied, metadata)
    return copied


def _messages_for_transport(
    messages: list[dict[str, Any]],
    *,
    provider_key: str | None,
) -> list[dict[str, Any]]:
    return [
        _message_for_transport(
            message,
            provider_key=provider_key,
        )
        for message in messages
        if isinstance(message, dict)
    ]


def _normalize_assistant_content_to_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "".join(_normalize_assistant_content_to_text(item) for item in raw)
    if isinstance(raw, dict):
        part_type = raw.get("type")
        text = raw.get("text")
        if isinstance(text, str) and (
            part_type in _TEXT_LIKE_CONTENT_PART_TYPES or part_type is None
        ):
            return text
        return ""
    return ""


def _reasoning_fallback_content(message: dict[str, Any]) -> str:
    """Reasoning text to use as content when the assistant ``content`` is empty.

    Some reasoning models (e.g. Xiaomi MiMo served through the hosted Sylliptor
    trial proxy) put their whole answer in the reasoning channel and return an
    empty/whitespace ``content`` for a simple turn like "hi". Left as-is that
    empty content silently degrades the chat turn to the generic clarification
    fallback. We surface the reasoning text instead, but only when there is no
    real content and no tool calls, so normal completions are never altered.
    """
    for key in (_OPENROUTER_REASONING_KEY, _DEEPSEEK_REASONING_CONTENT_KEY):
        reasoning = _normalize_assistant_content_to_text(message.get(key))
        if reasoning.strip():
            return reasoning
    return ""


def _parse_arguments(args_s: str) -> dict[str, Any]:
    try:
        args = json.loads(args_s)
    except json.JSONDecodeError:
        return {"_raw_arguments": args_s}
    if not isinstance(args, dict):
        return {"_raw_arguments": args_s}
    return args


def _parse_tool_calls(tool_calls_raw: list[dict[str, Any]]) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    for tc in tool_calls_raw:
        try:
            tc_id = tc["id"]
            fn = tc["function"]
            name = fn["name"]
            args_s = fn.get("arguments") or "{}"
            if not isinstance(args_s, str):
                args_s = json.dumps(args_s)
            tool_calls.append(
                ToolCall(
                    id=tc_id,
                    name=name,
                    arguments=_parse_arguments(args_s),
                    provider_metadata=_gemini_tool_call_provider_metadata(tc),
                )
            )
        except Exception:
            continue
    return tool_calls


def _parse_stream_tool_calls(tool_chunks: dict[int, dict[str, Any]]) -> list[ToolCall]:
    out: list[ToolCall] = []
    for idx in sorted(tool_chunks):
        chunk = tool_chunks[idx]
        name = chunk.get("name") or ""
        if not name:
            continue
        tc_id = chunk.get("id") or f"call_{idx}"
        args_s = chunk.get("arguments") or "{}"
        metadata = chunk.get("provider_metadata")
        out.append(
            ToolCall(
                id=tc_id,
                name=name,
                arguments=_parse_arguments(args_s),
                provider_metadata=dict(metadata)
                if isinstance(metadata, dict) and metadata
                else None,
            )
        )
    return out


def _parse_usage(raw: Any) -> LLMUsage | None:
    if not isinstance(raw, dict):
        return None
    prompt = raw.get("prompt_tokens")
    completion = raw.get("completion_tokens")
    total = raw.get("total_tokens")
    try:
        prompt_i = int(prompt) if prompt is not None else None
    except (TypeError, ValueError):
        prompt_i = None
    try:
        completion_i = int(completion) if completion is not None else None
    except (TypeError, ValueError):
        completion_i = None
    try:
        total_i = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_i = None
    cached_prompt_tokens_raw = raw.get("cached_prompt_tokens")
    prompt_tokens_details = raw.get("prompt_tokens_details")
    if cached_prompt_tokens_raw is None and isinstance(prompt_tokens_details, dict):
        cached_prompt_tokens_raw = prompt_tokens_details.get("cached_tokens")
    try:
        cached_prompt_tokens_i = (
            int(cached_prompt_tokens_raw) if cached_prompt_tokens_raw is not None else None
        )
    except (TypeError, ValueError):
        cached_prompt_tokens_i = None
    if (
        prompt_i is None
        and completion_i is None
        and total_i is None
        and cached_prompt_tokens_i is None
    ):
        return None
    return LLMUsage(
        prompt_tokens=prompt_i if (prompt_i is None or prompt_i >= 0) else None,
        completion_tokens=completion_i if (completion_i is None or completion_i >= 0) else None,
        total_tokens=total_i if (total_i is None or total_i >= 0) else None,
        cached_prompt_tokens=(
            cached_prompt_tokens_i
            if (cached_prompt_tokens_i is None or cached_prompt_tokens_i >= 0)
            else None
        ),
    )


def _is_stream_options_unsupported_error(err: LLMError) -> bool:
    msg = str(err).lower()
    if "stream_options" not in msg:
        return False
    return any(token in msg for token in ("unsupported", "unknown", "invalid", "not allowed"))


def _llm_error_status_code(err: LLMError) -> int | None:
    match = re.match(r"LLM error\s+(\d{3}):", str(err or "").strip())
    if match is None:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _llm_error_body(err: LLMError) -> str:
    _prefix, sep, body = str(err or "").partition(":")
    return body.strip() if sep else str(err or "").strip()


def _json_error_payload(body: str) -> dict[str, Any]:
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


# Friendly, user-facing copy for the Sylliptor MiMo trial proxy's error codes.
# The proxy (Supabase Edge Function) returns an OpenAI-shaped envelope
# {"error": {"message": ..., "type": ..., "code": "<reason>"}} with these string
# codes; upstream/other-provider errors use different codes and fall through.
_SYLLIPTOR_PROXY_ERROR_MESSAGES: dict[str, str] = {
    "invalid_key": (
        "Your Sylliptor session is invalid or has been reset. "
        "Run `sylliptor login` to reconnect your account."
    ),
    "trial_expired": (
        "Your 10-day free MiMo trial has ended. "
        "See your options at https://sylliptor.alysisai.com/account"
    ),
    "quota_exhausted": (
        "You've used all of your free MiMo trial tokens. "
        "See your options at https://sylliptor.alysisai.com/account"
    ),
    "email_not_verified": (
        "Please confirm your email to use the MiMo trial — "
        "check your inbox for the verification link."
    ),
    "plan_inactive": (
        "Your Sylliptor plan is not active. "
        "Visit https://sylliptor.alysisai.com/account to continue."
    ),
    "rate_limit_exceeded": (
        "You're sending requests too quickly. Please wait a moment and try again."
    ),
    "global_budget_exceeded": (
        "The free MiMo trial is at capacity right now. Please try again shortly."
    ),
    "proxy_unconfigured": ("The MiMo service is temporarily unavailable. Please try again later."),
}


def sylliptor_trial_error_message(err: LLMError) -> str | None:
    """Friendly message for a Sylliptor MiMo proxy error, or None if not ours.

    Maps the proxy's known error ``code`` (trial_expired, quota_exhausted, ...) to
    human copy so a user whose trial ended sees a clear next step instead of a raw
    ``LLM error 402: {...}`` dump. Returns None for any other failure (including
    upstream OpenRouter errors, which use numeric codes), so non-proxy errors
    render unchanged.
    """
    payload = _json_error_payload(_llm_error_body(err))
    error = payload.get("error")
    if not isinstance(error, dict):
        return None
    code = str(error.get("code") or "").strip()
    return _SYLLIPTOR_PROXY_ERROR_MESSAGES.get(code)


def _temperature_error_fields(err: LLMError) -> tuple[str, str, str] | None:
    status_code = _llm_error_status_code(err)
    if status_code not in _TEMPERATURE_UNSUPPORTED_STATUS_CODES:
        return None

    body = _llm_error_body(err)
    payload = _json_error_payload(body)
    if not payload:
        message = body.strip().casefold()
        if "temperature" in message and any(
            token in message for token in _TEMPERATURE_UNSUPPORTED_TOKENS
        ):
            return "", "", message
        return None

    raw_error = payload.get("error")
    error: dict[str, Any] = raw_error if isinstance(raw_error, dict) else payload
    param = str(error.get("param") or "").strip().casefold()
    code = str(error.get("code") or "").strip().casefold()
    message = str(error.get("message") or "").strip().casefold()
    combined = f"{code} {message}"
    has_unsupported_marker = any(token in combined for token in _TEMPERATURE_UNSUPPORTED_TOKENS)
    if param == "temperature" and (code or message) and has_unsupported_marker:
        return param, code, message
    if "temperature" in message and has_unsupported_marker:
        return param, code, message
    return None


def _temperature_unsupported_error(err: LLMError) -> bool:
    return _temperature_error_fields(err) is not None


def _is_temperature_default_value(value: Any) -> bool:
    try:
        return float(value) == _TEMPERATURE_DEFAULT_VALUE
    except (TypeError, ValueError):
        return False


def _temperature_compat_mode_for_error(
    err: LLMError,
    *,
    current_temperature: Any,
) -> str | None:
    fields = _temperature_error_fields(err)
    if fields is None:
        return None
    _param, _code, message = fields
    if "deprecated" in message:
        return _TEMPERATURE_COMPAT_MODE_OMIT
    if not _is_temperature_default_value(current_temperature):
        return _TEMPERATURE_COMPAT_MODE_DEFAULT
    return _TEMPERATURE_COMPAT_MODE_OMIT


def _merge_transport_metadata(
    response: LLMResponse,
    *,
    transport_metadata: dict[str, Any] | None,
) -> LLMResponse:
    if not transport_metadata:
        return response
    provider_metadata = _merge_provider_metadata(
        response.provider_metadata,
        {"transport": transport_metadata},
    )
    return LLMResponse(
        content=response.content,
        tool_calls=list(response.tool_calls),
        raw=response.raw,
        response_model=response.response_model,
        usage=response.usage,
        provider_metadata=provider_metadata,
    )


def _response_with_stream_restart_metadata(
    response: LLMResponse,
    *,
    count: int,
    reason: str,
) -> LLMResponse:
    raw = dict(response.raw) if isinstance(response.raw, dict) else {}
    raw["stream_restart_count"] = max(0, int(count))
    if reason:
        raw["stream_restart_reason"] = str(reason)
    return LLMResponse(
        content=response.content,
        tool_calls=list(response.tool_calls),
        raw=raw,
        response_model=response.response_model,
        usage=response.usage,
        provider_metadata=response.provider_metadata,
    )


class OpenAICompatClient:
    supports_forced_tool_choice = True

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 60.0,
        temperature: float = 1.0,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        transport: httpx.BaseTransport | None = None,
        extra_headers: dict[str, str] | None = None,
        provider_key: str | None = None,
        provider_concurrency_caps: dict[str, int] | None = None,
        provider_retry_settings: ProviderRetrySettings | None = None,
        provider_sleep_fn: Callable[[float], None] | None = None,
        provider_random_fn: Callable[[], float] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.prompt_cache_key = str(prompt_cache_key or "").strip() or None
        self.prompt_cache_retention = str(prompt_cache_retention or "").strip() or None
        self.enable_thinking = enable_thinking
        self.reasoning_effort = str(reasoning_effort or "").strip().lower() or None
        self._transport = transport
        self.extra_headers = {
            str(key): str(value)
            for key, value in (extra_headers or {}).items()
            if str(key).strip() and str(value).strip()
        }
        self.provider_key = str(provider_key or "").strip() or None
        self.provider_concurrency_caps = dict(
            DEFAULT_PROVIDER_CONCURRENCY_CAPS
            if provider_concurrency_caps is None
            else provider_concurrency_caps
        )
        self.provider_retry_settings = provider_retry_settings or ProviderRetrySettings()
        self._provider_sleep_fn = provider_sleep_fn
        self._provider_random_fn = provider_random_fn
        self._provider_retry_deadline_allows: Callable[[float], bool] | None = None
        self._provider_finalization_retry_used = False
        self._temperature_compat_modes: dict[tuple[str, str], str] = {}
        self._temperature_compat_lock = threading.Lock()

    def _temperature_compat_key(self, provider_key: str | None) -> tuple[str, str]:
        provider = _normalize_provider_key(provider_key) or _normalize_provider_key(self.base_url)
        model = str(self.model or "").strip().casefold()
        return provider, model

    def _temperature_compat_mode_for(self, key: tuple[str, str]) -> str | None:
        with self._temperature_compat_lock:
            return self._temperature_compat_modes.get(key)

    def _mark_temperature_compat_mode(self, key: tuple[str, str], mode: str) -> None:
        if mode not in _TEMPERATURE_COMPAT_MODES:
            raise ValueError(f"Unknown temperature compatibility mode: {mode}")
        with self._temperature_compat_lock:
            self._temperature_compat_modes[key] = mode

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        on_text_delta: Callable[[str], None] | None = None,
        on_reasoning_delta: Callable[[str], None] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cancellation_token: Any | None = None,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "sylliptor-agent-cli/0.1.0",
        }
        headers.update(self.extra_headers)
        headers = _headers_with_default_accept_encoding(headers)
        resolved_temperature = self.temperature if temperature is None else float(temperature)
        transport_provider_key = _transport_provider_key(
            base_url=self.base_url,
            provider_key=self.provider_key,
            model=self.model,
        )
        temperature_key = self._temperature_compat_key(transport_provider_key)
        cached_temperature_compat_mode = self._temperature_compat_mode_for(temperature_key)
        transport_metadata: dict[str, Any] = {}
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages_for_transport(
                messages,
                provider_key=transport_provider_key,
            ),
        }
        if cached_temperature_compat_mode == _TEMPERATURE_COMPAT_MODE_DEFAULT:
            payload["temperature"] = _TEMPERATURE_DEFAULT_VALUE
            transport_metadata["temperature_adjusted"] = True
            transport_metadata["temperature_adjustment"] = _TEMPERATURE_COMPAT_MODE_DEFAULT
            transport_metadata["temperature_adjustment_reason"] = "cached_provider_rejection"
        elif cached_temperature_compat_mode == _TEMPERATURE_COMPAT_MODE_OMIT:
            transport_metadata["temperature_adjusted"] = True
            transport_metadata["temperature_adjustment"] = _TEMPERATURE_COMPAT_MODE_OMIT
            transport_metadata["temperature_adjustment_reason"] = "cached_provider_rejection"
            transport_metadata["temperature_omitted"] = True
            transport_metadata["temperature_omit_reason"] = "cached_provider_rejection"
        else:
            payload["temperature"] = resolved_temperature
        if self.prompt_cache_key:
            payload["prompt_cache_key"] = self.prompt_cache_key
        if self.prompt_cache_retention:
            payload["prompt_cache_retention"] = self.prompt_cache_retention
        if _is_dashscope_provider(transport_provider_key):
            enable_thinking = self.enable_thinking
            if enable_thinking is None:
                enable_thinking = _reasoning_effort_enables_thinking(self.reasoning_effort)
            if enable_thinking is not None:
                payload["enable_thinking"] = enable_thinking
        elif _is_deepseek_provider(transport_provider_key):
            thinking_enabled = _deepseek_reasoning_payload_enabled(
                enable_thinking=self.enable_thinking,
                reasoning_effort=self.reasoning_effort,
            )
            if thinking_enabled is not None:
                payload["thinking"] = {"type": "enabled" if thinking_enabled else "disabled"}
        elif _is_openrouter_provider(transport_provider_key):
            reasoning = _openrouter_reasoning_payload(
                enable_thinking=self.enable_thinking,
                reasoning_effort=self.reasoning_effort,
            )
            if reasoning is not None:
                payload["reasoning"] = reasoning
        elif _is_gemini_provider(transport_provider_key):
            reasoning_effort = _gemini_reasoning_effort(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
            )
            if reasoning_effort:
                payload["reasoning_effort"] = reasoning_effort
        elif _uses_reasoning_effort(transport_provider_key):
            if self.reasoning_effort:
                payload["reasoning_effort"] = self.reasoning_effort
        elif self.enable_thinking is not None and not self.reasoning_effort:
            payload["enable_thinking"] = self.enable_thinking
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto" if tool_choice is None else tool_choice
        elif tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        payload = _sanitize_transport_value(payload)

        provider_key = (
            transport_provider_key
            or self.provider_key
            or best_effort_provider_key(
                base_url=self.base_url,
                model=self.model,
            )
        )
        telemetry = ProviderCallTelemetryRecorder(
            provider_key=provider_key,
            protocol="openai_compat",
            model=self.model,
            base_url=self.base_url,
            stream=stream,
            tools=tools,
            operation="chat_completions",
        )
        telemetry_on_text_delta = telemetry.wrap_text_delta(on_text_delta)
        stream_restart_count = 0
        stream_restart_reason = ""

        def _send_request() -> LLMResponse:
            nonlocal stream_restart_count, stream_restart_reason
            temperature_retry_count = 0
            temperature_retry_modes: set[str] = set()
            try:
                with httpx.Client(timeout=self.timeout_s, transport=self._transport) as client:
                    while True:
                        try:
                            if stream:
                                with client.stream(
                                    "POST", url, headers=headers, json=payload
                                ) as resp:
                                    attempt_deltas: list[str] = []
                                    forward_attempt_deltas = cancellation_token is not None

                                    def _on_attempt_delta(
                                        delta: str,
                                        *,
                                        _attempt_deltas: list[str] = attempt_deltas,
                                        _forward_attempt_deltas: bool = forward_attempt_deltas,
                                    ) -> None:
                                        _attempt_deltas.append(delta)
                                        if (
                                            _forward_attempt_deltas
                                            and telemetry_on_text_delta is not None
                                        ):
                                            telemetry_on_text_delta(delta)

                                    try:
                                        response = self._parse_stream_response(
                                            resp,
                                            on_text_delta=_on_attempt_delta,
                                            on_reasoning_delta=on_reasoning_delta,
                                            provider_key=transport_provider_key,
                                            cancellation_token=cancellation_token,
                                        )
                                    except Exception as stream_error:
                                        if attempt_deltas:
                                            stream_restart_count += 1
                                            stream_restart_reason = (
                                                provider_unavailable_retry_reason(stream_error)
                                                or "provider_stream_interrupted"
                                            )
                                            transport_metadata["stream_restart_count"] = (
                                                stream_restart_count
                                            )
                                            transport_metadata["stream_restart_reason"] = (
                                                stream_restart_reason
                                            )
                                        raise
                                    if (
                                        telemetry_on_text_delta is not None
                                        and not forward_attempt_deltas
                                    ):
                                        for delta in attempt_deltas:
                                            telemetry_on_text_delta(delta)
                                    if stream_restart_count:
                                        response = _response_with_stream_restart_metadata(
                                            response,
                                            count=stream_restart_count,
                                            reason=stream_restart_reason,
                                        )
                            else:
                                resp = client.post(url, headers=headers, json=payload)
                                if resp.status_code >= 400:
                                    raise self._error_from_response(resp)
                                response = self._parse_non_stream_response(
                                    resp,
                                    provider_key=transport_provider_key,
                                )
                        except LLMError as e:
                            if stream and _is_stream_options_unsupported_error(e):
                                payload.pop("stream_options", None)
                                continue
                            if "temperature" in payload and _temperature_unsupported_error(e):
                                rejected_temperature = payload.get("temperature")
                                compat_mode = _temperature_compat_mode_for_error(
                                    e,
                                    current_temperature=rejected_temperature,
                                )
                                if compat_mode is None or compat_mode in temperature_retry_modes:
                                    raise
                                temperature_retry_modes.add(compat_mode)
                                temperature_retry_count += 1
                                self._mark_temperature_compat_mode(temperature_key, compat_mode)
                                transport_metadata["temperature_adjusted"] = True
                                transport_metadata["temperature_adjustment"] = compat_mode
                                transport_metadata["temperature_adjustment_reason"] = (
                                    "provider_rejected_parameter"
                                )
                                transport_metadata["temperature_retry_used"] = True
                                transport_metadata["temperature_retry_count"] = (
                                    temperature_retry_count
                                )
                                if compat_mode == _TEMPERATURE_COMPAT_MODE_DEFAULT:
                                    payload["temperature"] = _TEMPERATURE_DEFAULT_VALUE
                                else:
                                    payload.pop("temperature", None)
                                    transport_metadata["temperature_omitted"] = True
                                    transport_metadata["temperature_omit_reason"] = (
                                        "provider_rejected_parameter"
                                    )
                                _LOGGER.info(
                                    "llm_temperature_parameter_rejected_retrying_with_compat_mode",
                                    extra={
                                        "provider_key": provider_key,
                                        "model": self.model,
                                        "temperature": rejected_temperature,
                                        "temperature_compat_mode": compat_mode,
                                    },
                                )
                                continue
                            raise
                        return _merge_transport_metadata(
                            response,
                            transport_metadata=transport_metadata,
                        )
            except LLMError:
                raise
            except httpx.DecodingError as e:
                raise LLMError(f"LLM response decompression failed: {e}") from e
            except Exception as e:  # noqa: BLE001 - network errors vary
                raise LLMError(f"LLM request failed: {e}") from e

        return telemetry.run(
            lambda: run_provider_limited_call(
                call=_send_request,
                provider_key=provider_key,
                provider_concurrency_caps=self.provider_concurrency_caps,
                retry_settings=self.provider_retry_settings,
                operation="chat_completions",
                sleep_fn=self._provider_sleep_fn,
                random_fn=self._provider_random_fn,
                on_retry=telemetry.on_retry,
                retry_deadline_allows=getattr(self, "_provider_retry_deadline_allows", None),
            )
        )

    @staticmethod
    def _parse_non_stream_response(
        resp: httpx.Response,
        *,
        provider_key: str | None,
    ) -> LLMResponse:
        try:
            data = resp.json()
        except Exception as e:  # noqa: BLE001
            raise LLMError("LLM returned non-JSON response") from e

        try:
            choice0 = data["choices"][0]
            msg = choice0["message"]
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"Unexpected LLM response shape: {data!r}") from e

        content = _normalize_assistant_content_to_text(msg.get("content"))
        tool_calls_raw = msg.get("tool_calls") or []
        tool_calls = _parse_tool_calls(tool_calls_raw)
        if not content.strip() and not tool_calls:
            content = _reasoning_fallback_content(msg) or content
        response_model = data.get("model") if isinstance(data.get("model"), str) else None
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            raw=data,
            response_model=response_model,
            usage=_parse_usage(data.get("usage")),
            provider_metadata=_provider_metadata_for_reasoning(
                provider_key=provider_key,
                message=msg,
            ),
        )

    @staticmethod
    def _error_from_response(resp: httpx.Response) -> LLMError:
        body = resp.text
        if len(body) > 1000:
            body = body[:1000] + "...(truncated)"
        return LLMError(f"LLM error {resp.status_code}: {body}")

    def _parse_stream_response(
        self,
        resp: httpx.Response,
        *,
        on_text_delta: Callable[[str], None] | None,
        on_reasoning_delta: Callable[[str], None] | None = None,
        provider_key: str | None,
        cancellation_token: Any | None = None,
    ) -> LLMResponse:
        if resp.status_code >= 400:
            body = self._safe_error_body(resp)
            raise LLMError(f"LLM error {resp.status_code}: {body}")

        content_parts: list[str] = []
        tool_chunks: dict[int, dict[str, Any]] = {}
        event_count = 0
        response_model: str | None = None
        usage: LLMUsage | None = None
        accumulated_content = ""
        reasoning_parts: list[str] = []
        reasoning_details: list[Any] = []
        saw_done = False

        # Make the (possibly long) initial read interruptible: register the live
        # response's close so a cancel from another thread unblocks iter_lines, and
        # re-check the flag per line. A close mid-read surfaces as a read error that
        # we translate into a clean interrupt below.
        _set_abort = getattr(cancellation_token, "set_abort_callback", None)
        _clear_abort = getattr(cancellation_token, "clear_abort_callback", None)
        if callable(_set_abort):
            _set_abort(resp.close)
        _stream_iter = resp.iter_lines()

        def _is_cancelled() -> bool:
            return cancellation_token is not None and getattr(
                cancellation_token, "is_cancelled", False
            )

        try:
            while True:
                try:
                    line = next(_stream_iter)
                except StopIteration:
                    break
                except Exception:
                    if _is_cancelled():
                        raise KeyboardInterrupt("cancelled_by_user") from None
                    raise
                if _is_cancelled():
                    raise KeyboardInterrupt("cancelled_by_user")
                if not line:
                    continue
                if isinstance(line, bytes):
                    text = line.decode("utf-8", errors="ignore")
                else:
                    text = line
                if not text.startswith("data:"):
                    continue
                payload = text[5:].strip()
                if not payload:
                    continue
                if payload == "[DONE]":
                    saw_done = True
                    break

                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                event_count += 1
                model = event.get("model")
                if isinstance(model, str) and model:
                    response_model = model
                parsed_usage = _parse_usage(event.get("usage"))
                if parsed_usage is not None:
                    usage = parsed_usage

                choices = event.get("choices") or []
                if not isinstance(choices, list) or not choices:
                    continue
                choice0 = choices[0]
                if not isinstance(choice0, dict):
                    continue
                delta = choice0.get("delta") or {}
                if not isinstance(delta, dict):
                    continue

                reasoning_delta = delta.get(_DEEPSEEK_REASONING_CONTENT_KEY)
                if not isinstance(reasoning_delta, str):
                    reasoning_delta = delta.get(_OPENROUTER_REASONING_KEY)
                if isinstance(reasoning_delta, str) and reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                    if on_reasoning_delta is not None:
                        on_reasoning_delta(reasoning_delta)
                    if _is_cancelled():
                        raise KeyboardInterrupt("cancelled_by_user")
                details_delta = delta.get(_OPENROUTER_REASONING_DETAILS_KEY)
                if isinstance(details_delta, list) and details_delta:
                    reasoning_details.extend(details_delta)

                content_delta = _normalize_assistant_content_to_text(delta.get("content"))
                if content_delta:
                    content_suffix = _stream_delta_suffix(
                        previous=accumulated_content,
                        incoming=content_delta,
                    )
                    if content_suffix:
                        content_parts.append(content_suffix)
                        accumulated_content += content_suffix
                        if on_text_delta is not None:
                            on_text_delta(content_suffix)
                        if _is_cancelled():
                            raise KeyboardInterrupt("cancelled_by_user")

                tc_delta = delta.get("tool_calls") or []
                if not isinstance(tc_delta, list):
                    continue
                for raw_tc in tc_delta:
                    if not isinstance(raw_tc, dict):
                        continue
                    idx = raw_tc.get("index")
                    if not isinstance(idx, int):
                        continue
                    entry = tool_chunks.setdefault(
                        idx,
                        {"id": "", "name": "", "arguments": "", "provider_metadata": None},
                    )

                    tc_id = raw_tc.get("id")
                    if isinstance(tc_id, str) and tc_id:
                        entry["id"] = tc_id

                    provider_metadata = _gemini_tool_call_provider_metadata(raw_tc)
                    if provider_metadata:
                        existing_metadata = entry.get("provider_metadata")
                        entry["provider_metadata"] = _merge_provider_metadata(
                            existing_metadata if isinstance(existing_metadata, dict) else None,
                            provider_metadata,
                        )

                    fn = raw_tc.get("function")
                    if not isinstance(fn, dict):
                        continue
                    name = fn.get("name")
                    if isinstance(name, str) and name:
                        entry["name"] = name
                    args_piece = fn.get("arguments")
                    if isinstance(args_piece, str):
                        entry["arguments"] += _stream_delta_suffix(
                            previous=entry["arguments"],
                            incoming=args_piece,
                        )
        finally:
            if callable(_clear_abort):
                _clear_abort()
        if _is_cancelled():
            # Stream ended because the abort closed it (clean EOF, not an error).
            raise KeyboardInterrupt("cancelled_by_user")
        if not saw_done:
            raise LLMError("LLM stream truncated before [DONE]")
        streamed_tool_calls = _parse_stream_tool_calls(tool_chunks)
        streamed_content = "".join(content_parts)
        if not streamed_content.strip() and not streamed_tool_calls:
            # Reasoning-only completion (see _reasoning_fallback_content): the
            # model streamed its answer in the reasoning channel and emitted no
            # content deltas. Use the accumulated reasoning so the turn does not
            # degrade to the static clarification fallback.
            reasoning_text = "".join(reasoning_parts)
            if reasoning_text.strip():
                streamed_content = reasoning_text
        return LLMResponse(
            content=streamed_content,
            tool_calls=streamed_tool_calls,
            raw={"stream": True, "events": event_count},
            response_model=response_model,
            usage=usage,
            provider_metadata=_provider_metadata_for_reasoning(
                provider_key=provider_key,
                message={
                    _DEEPSEEK_REASONING_CONTENT_KEY: "".join(reasoning_parts),
                    _OPENROUTER_REASONING_KEY: "".join(reasoning_parts),
                    _OPENROUTER_REASONING_DETAILS_KEY: reasoning_details,
                },
            ),
        )

    @staticmethod
    def _safe_error_body(resp: httpx.Response) -> str:
        try:
            resp.read()
        except Exception:
            pass
        try:
            body = resp.text
        except Exception:
            body = "<unable to read response body>"
        if len(body) > 1000:
            body = body[:1000] + "...(truncated)"
        return body
