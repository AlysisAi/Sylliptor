from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from ..provider_telemetry import ProviderCallTelemetryRecorder
from ..web_search_adapters import AUTO_WEB_SEARCH_ADAPTER, GEMINI_GROUNDING_ADAPTER
from .metadata import (
    GEMINI_GENERATE_CONTENT_PROVIDER_METADATA_KEY,
    PROVIDER_METADATA_KEY,
    TOOL_CALL_PROVIDER_METADATA_KEY,
)
from .provider_limits import (
    DEFAULT_PROVIDER_CONCURRENCY_CAPS,
    ProviderRetrySettings,
    best_effort_provider_key,
    run_provider_limited_call,
)
from .streaming import SSEFrame, iter_sse_frames, parse_sse_json_frame
from .types import LLMError, LLMResponse, LLMUsage, ToolCall

_DEFAULT_ACCEPT_ENCODING = "identity"
_GEMINI_METADATA_KEY = GEMINI_GENERATE_CONTENT_PROVIDER_METADATA_KEY
_SYLLIPTOR_WEB_SEARCH_FUNCTION_NAME = "web_search"
_WEB_SEARCH_MODES_ALLOWING_GEMINI_GROUNDING = frozenset({"auto", "native"})
_INCLUDE_SERVER_SIDE_TOOL_INVOCATIONS = "includeServerSideToolInvocations"
_TOOL_CALL_PROVIDER_METADATA_KEY = TOOL_CALL_PROVIDER_METADATA_KEY
_DUMMY_IMPORTED_FUNCTION_CALL_THOUGHT_SIGNATURE = "skip_thought_signature_validator"
_GEMINI_THINKING_LEVELS = frozenset({"minimal", "low", "high"})


def _headers_with_default_accept_encoding(headers: dict[str, str]) -> dict[str, str]:
    request_headers = dict(headers)
    if not any(key.lower() == "accept-encoding" for key in request_headers):
        request_headers["Accept-Encoding"] = _DEFAULT_ACCEPT_ENCODING
    return request_headers


def _gemini_native_base_url(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if normalized.endswith("/openai"):
        return normalized.removesuffix("/openai")
    return normalized


def _content_to_text(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    if isinstance(raw, dict):
        text = raw.get("text") or raw.get("content")
        return text if isinstance(text, str) else json.dumps(raw, ensure_ascii=False)
    return str(raw)


def _json_arguments(args: Any) -> dict[str, Any]:
    if isinstance(args, dict):
        return dict(args)
    if args is None:
        return {}
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return {"_raw_arguments": args}
        if isinstance(parsed, dict):
            return parsed
        return {"_raw_arguments": args}
    return {"_raw_arguments": json.dumps(args, ensure_ascii=False)}


def _json_response_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"result": value}
        if isinstance(parsed, dict):
            return parsed
        return {"result": parsed}
    return {"result": value}


def _gemini_parts_from_content(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [{"text": raw}] if raw else []
    if not isinstance(raw, list):
        text = _content_to_text(raw)
        return [{"text": text}] if text else []

    parts: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            if item:
                parts.append({"text": item})
            continue
        if not isinstance(item, dict):
            continue
        part_type = str(item.get("type") or "").strip()
        text = item.get("text") or item.get("content")
        if part_type in {"text", "input_text", "output_text"} and isinstance(text, str):
            parts.append({"text": text})
            continue
        if part_type == "image_url":
            image_url = item.get("image_url")
            url = ""
            if isinstance(image_url, dict):
                url = str(image_url.get("url") or "").strip()
            elif isinstance(image_url, str):
                url = image_url.strip()
            if url:
                parts.append({"fileData": {"fileUri": url}})
            continue
        if any(
            key in item for key in ("functionCall", "functionResponse", "inlineData", "fileData")
        ):
            parts.append(copy.deepcopy(item))
    return parts


def _metadata_content(message: dict[str, Any]) -> dict[str, Any] | None:
    metadata = message.get(PROVIDER_METADATA_KEY)
    if not isinstance(metadata, dict):
        return None
    gemini_metadata = metadata.get(_GEMINI_METADATA_KEY)
    if not isinstance(gemini_metadata, dict):
        return None
    content = gemini_metadata.get("content")
    if isinstance(content, dict):
        return _normalize_gemini_content_wire_keys(content)
    return None


def _normalize_gemini_content_wire_keys(content: dict[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(content)
    parts = copied.get("parts")
    if not isinstance(parts, list):
        return copied
    for part in parts:
        if not isinstance(part, dict):
            continue
        if "thought_signature" in part:
            if "thoughtSignature" not in part:
                part["thoughtSignature"] = copy.deepcopy(part["thought_signature"])
            part.pop("thought_signature", None)
    return copied


def _function_name_from_tool(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "").strip()
    if str(tool.get("type") or "") == "function":
        return str(tool.get("name") or "").strip()
    return ""


def _is_sylliptor_web_search_function(tool: dict[str, Any]) -> bool:
    return _function_name_from_tool(tool) == _SYLLIPTOR_WEB_SEARCH_FUNCTION_NAME


def _is_gemini_grounding_tool(tool: dict[str, Any]) -> bool:
    return isinstance(tool.get("google_search"), dict) or isinstance(tool.get("googleSearch"), dict)


def _gemini_grounding_allowed(*, mode: str, adapter: str) -> bool:
    normalized_mode = str(mode or "").strip().lower()
    normalized_adapter = str(adapter or "").strip().lower() or AUTO_WEB_SEARCH_ADAPTER
    if normalized_mode not in _WEB_SEARCH_MODES_ALLOWING_GEMINI_GROUNDING:
        return False
    return normalized_adapter in {AUTO_WEB_SEARCH_ADAPTER, GEMINI_GROUNDING_ADAPTER}


@dataclass(frozen=True)
class _GeminiToolMapping:
    tools: list[dict[str, Any]]
    added_google_search: bool
    removed_sylliptor_web_search: bool
    include_server_side_tool_invocations: bool


def _gemini_function_declaration_from_chat_tool(tool: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    tool_type = str(tool.get("type") or "").strip()
    if tool_type != "function":
        if _is_gemini_grounding_tool(tool):
            return None
        raise LLMError(f"Gemini GenerateContent does not support tool type {tool_type!r}")
    function = tool.get("function")
    source = function if isinstance(function, dict) else tool
    name = str(source.get("name") or "").strip()
    if not name:
        return None
    declaration: dict[str, Any] = {
        "name": name,
        "parameters": copy.deepcopy(source.get("parameters") or {"type": "object"}),
    }
    description = str(source.get("description") or "").strip()
    if description:
        declaration["description"] = description
    return declaration


def _gemini_tools(
    tools: list[dict[str, Any]] | None,
    *,
    mode: str,
    adapter: str,
) -> _GeminiToolMapping:
    normalized_mode = str(mode or "off").strip().lower()
    normalized_adapter = (
        str(adapter or AUTO_WEB_SEARCH_ADAPTER).strip().lower() or AUTO_WEB_SEARCH_ADAPTER
    )
    raw_tools = [tool for tool in tools or [] if isinstance(tool, dict)]
    sylliptor_web_search_present = any(
        _is_sylliptor_web_search_function(tool) for tool in raw_tools
    )
    use_google_search = sylliptor_web_search_present and _gemini_grounding_allowed(
        mode=normalized_mode, adapter=normalized_adapter
    )
    if normalized_mode == "native" and sylliptor_web_search_present and not use_google_search:
        raise LLMError(
            "web_search_mode=native with protocol=gemini_generate_content requires "
            "web_search_adapter='auto' or 'gemini_grounding' for Gemini Google Search grounding; "
            f"got {normalized_adapter!r}"
        )

    function_declarations: list[dict[str, Any]] = []
    mapped_tools: list[dict[str, Any]] = []
    removed_sylliptor_web_search = False
    for tool in raw_tools:
        if _is_sylliptor_web_search_function(tool):
            if normalized_mode in {"off", "native"} or use_google_search:
                removed_sylliptor_web_search = True
                continue
        if _is_gemini_grounding_tool(tool):
            if normalized_mode in {"off", "external"}:
                continue
            mapped_tools.append(copy.deepcopy(tool))
            continue
        declaration = _gemini_function_declaration_from_chat_tool(tool)
        if declaration is not None:
            function_declarations.append(declaration)
    if function_declarations:
        mapped_tools.insert(0, {"functionDeclarations": function_declarations})
    if use_google_search and not any(_is_gemini_grounding_tool(tool) for tool in mapped_tools):
        mapped_tools.append({"google_search": {}})
    include_server_side_tool_invocations = bool(function_declarations) and any(
        _is_gemini_grounding_tool(tool) for tool in mapped_tools
    )
    return _GeminiToolMapping(
        tools=mapped_tools,
        added_google_search=use_google_search,
        removed_sylliptor_web_search=removed_sylliptor_web_search,
        include_server_side_tool_invocations=include_server_side_tool_invocations,
    )


def _gemini_tool_choice(
    tool_choice: Any,
    *,
    removed_sylliptor_web_search: bool,
    added_google_search: bool,
) -> dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        normalized = tool_choice.strip()
        if normalized == "auto":
            return {"functionCallingConfig": {"mode": "AUTO"}}
        if normalized == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        if normalized == "required":
            return {"functionCallingConfig": {"mode": "ANY"}}
        raise LLMError(f"Gemini GenerateContent does not support tool_choice={tool_choice!r}")
    if not isinstance(tool_choice, dict):
        raise LLMError("Gemini GenerateContent tool_choice must be a string or object")

    choice_type = str(tool_choice.get("type") or "").strip()
    if choice_type == "function":
        if "name" in tool_choice:
            name = str(tool_choice.get("name") or "").strip()
        else:
            function = tool_choice.get("function")
            name = str(function.get("name") or "").strip() if isinstance(function, dict) else ""
        if not name:
            raise LLMError("Gemini GenerateContent forced function tool_choice is missing name")
        if (
            name == _SYLLIPTOR_WEB_SEARCH_FUNCTION_NAME
            and removed_sylliptor_web_search
            and not added_google_search
        ):
            raise LLMError(
                "Gemini GenerateContent removed the Sylliptor web_search function for the "
                "selected web_search_mode; do not force tool_choice to function web_search"
            )
        if name == _SYLLIPTOR_WEB_SEARCH_FUNCTION_NAME and added_google_search:
            raise LLMError(
                "Gemini GenerateContent cannot force Google Search grounding via tool_choice"
            )
        return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}
    if choice_type in {"auto", "any", "none"}:
        mode = "ANY" if choice_type == "any" else choice_type.upper()
        return {"functionCallingConfig": {"mode": mode}}
    raise LLMError(f"Gemini GenerateContent does not support tool_choice type {choice_type!r}")


def _system_instruction_from_parts(parts: list[str]) -> dict[str, Any] | None:
    text = "\n\n".join(part for part in parts if part).strip()
    if not text:
        return None
    return {"parts": [{"text": text}]}


def _tool_call_provider_metadata_indexes(
    message: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, Any]]]:
    metadata = message.get(PROVIDER_METADATA_KEY)
    if not isinstance(metadata, dict):
        return {}, {}
    entries = metadata.get(_TOOL_CALL_PROVIDER_METADATA_KEY)
    if not isinstance(entries, list):
        return {}, {}
    by_id: dict[str, dict[str, Any]] = {}
    by_index: dict[int, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_metadata = entry.get("metadata")
        if not isinstance(entry_metadata, dict):
            continue
        gemini_metadata = entry_metadata.get(_GEMINI_METADATA_KEY)
        if not isinstance(gemini_metadata, dict):
            continue
        copied = copy.deepcopy(gemini_metadata)
        tool_call_id = entry.get("id")
        if isinstance(tool_call_id, str) and tool_call_id.strip():
            by_id[tool_call_id.strip()] = copied
        index = entry.get("index")
        if isinstance(index, int):
            by_index[index] = copied
    return by_id, by_index


def _apply_gemini_thought_signature(
    part: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> None:
    if isinstance(metadata, dict):
        if "thoughtSignature" in metadata:
            part["thoughtSignature"] = copy.deepcopy(metadata["thoughtSignature"])
            return
        if "thought_signature" in metadata:
            part["thoughtSignature"] = copy.deepcopy(metadata["thought_signature"])
            return
    part["thoughtSignature"] = _DUMMY_IMPORTED_FUNCTION_CALL_THOUGHT_SIGNATURE


def _tool_call_parts_from_message(message: dict[str, Any]) -> list[dict[str, Any]]:
    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return []
    parts: list[dict[str, Any]] = []
    metadata_by_id, metadata_by_index = _tool_call_provider_metadata_indexes(message)
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            continue
        tool_call_index = len(parts)
        call_id = str(raw_tool_call.get("id") or raw_tool_call.get("call_id") or "").strip()
        function = raw_tool_call.get("function")
        if isinstance(function, dict):
            name = str(function.get("name") or "").strip()
            args = _json_arguments(function.get("arguments"))
        else:
            name = str(raw_tool_call.get("name") or "").strip()
            args = _json_arguments(raw_tool_call.get("arguments"))
        if not name:
            continue
        function_call: dict[str, Any] = {"name": name, "args": args}
        if call_id:
            function_call["id"] = call_id
        part = {"functionCall": function_call}
        metadata = metadata_by_id.get(call_id) if call_id else None
        if metadata is None:
            metadata = metadata_by_index.get(tool_call_index)
        _apply_gemini_thought_signature(part, metadata)
        parts.append(part)
    return parts


def _collect_function_call_names(content: dict[str, Any], call_names: dict[str, str]) -> None:
    parts = content.get("parts")
    if not isinstance(parts, list):
        return
    for part in parts:
        if not isinstance(part, dict):
            continue
        function_call = part.get("functionCall")
        if not isinstance(function_call, dict):
            continue
        call_id = str(function_call.get("id") or "").strip()
        name = str(function_call.get("name") or "").strip()
        if call_id and name:
            call_names[call_id] = name


def _is_function_response_user_content(content: dict[str, Any]) -> bool:
    if str(content.get("role") or "") != "user":
        return False
    parts = content.get("parts")
    return (
        isinstance(parts, list)
        and bool(parts)
        and all(
            isinstance(part, dict) and isinstance(part.get("functionResponse"), dict)
            for part in parts
        )
    )


def _gemini_contents_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        if role in {"system", "developer"}:
            text = _content_to_text(message.get("content")).strip()
            if text:
                system_parts.append(text)
            continue
        if role == "user":
            parts = _gemini_parts_from_content(message.get("content"))
            if parts:
                contents.append({"role": "user", "parts": parts})
            continue
        if role == "assistant":
            metadata_content = _metadata_content(message)
            if metadata_content is not None:
                contents.append(metadata_content)
                _collect_function_call_names(metadata_content, call_names)
                continue
            parts = _gemini_parts_from_content(message.get("content"))
            parts.extend(_tool_call_parts_from_message(message))
            if parts:
                content = {"role": "model", "parts": parts}
                contents.append(content)
                _collect_function_call_names(content, call_names)
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id") or message.get("call_id") or "").strip()
            if not call_id:
                raise LLMError("Gemini GenerateContent function response is missing tool_call_id")
            name = str(message.get("name") or call_names.get(call_id) or "").strip()
            if not name:
                raise LLMError(
                    "Gemini GenerateContent function response is missing function name for "
                    f"tool_call_id={call_id!r}"
                )
            response_payload = _json_response_payload(message.get("content"))
            function_response_part = {
                "functionResponse": {
                    "id": call_id,
                    "name": name,
                    "response": response_payload,
                }
            }
            if contents and _is_function_response_user_content(contents[-1]):
                parts = contents[-1].setdefault("parts", [])
                if isinstance(parts, list):
                    parts.append(function_response_part)
                else:
                    contents.append({"role": "user", "parts": [function_response_part]})
            else:
                contents.append({"role": "user", "parts": [function_response_part]})
            continue
        raise LLMError(f"Gemini GenerateContent cannot send message role {role!r}")
    return _system_instruction_from_parts(system_parts), contents


def _parse_usage(raw: Any) -> LLMUsage | None:
    if not isinstance(raw, dict):
        return None

    def _as_non_negative_int(value: Any) -> int | None:
        try:
            parsed = int(value) if value is not None else None
        except (TypeError, ValueError):
            return None
        if parsed is None or parsed >= 0:
            return parsed
        return None

    prompt_tokens = _as_non_negative_int(raw.get("promptTokenCount"))
    completion_tokens = _as_non_negative_int(raw.get("candidatesTokenCount"))
    total_tokens = _as_non_negative_int(raw.get("totalTokenCount"))
    cached_tokens = _as_non_negative_int(raw.get("cachedContentTokenCount"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    usage = LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_prompt_tokens=cached_tokens,
    )
    if (
        usage.prompt_tokens is None
        and usage.completion_tokens is None
        and usage.total_tokens is None
        and usage.cached_prompt_tokens is None
    ):
        return None
    return usage


def _extract_error_message(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    error_obj = data.get("error")
    if isinstance(error_obj, dict):
        message = str(error_obj.get("message") or "").strip()
        if message:
            status = str(error_obj.get("status") or "").strip()
            return f"{status}: {message}" if status else message
    return None


def _candidate(data: dict[str, Any]) -> dict[str, Any] | None:
    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        return None
    return next((item for item in candidates if isinstance(item, dict)), None)


def _candidate_content(candidate: dict[str, Any]) -> dict[str, Any] | None:
    content = candidate.get("content")
    return content if isinstance(content, dict) else None


def _extract_text(parts: list[Any]) -> str:
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, dict) and isinstance(part.get("text"), str):
            text_parts.append(part["text"])
    return "".join(text_parts)


def _parse_tool_calls(parts: list[Any]) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    for index, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        function_call = part.get("functionCall")
        if not isinstance(function_call, dict):
            continue
        name = str(function_call.get("name") or "").strip()
        if not name:
            continue
        call_id = str(function_call.get("id") or f"call_{index}").strip()
        raw_args = function_call.get("args")
        args = dict(raw_args) if isinstance(raw_args, dict) else _json_arguments(raw_args)
        metadata: dict[str, Any] = {"part_index": index}
        for key in ("thoughtSignature", "thought_signature"):
            if key in part:
                metadata[key] = part.get(key)
        tool_calls.append(
            ToolCall(
                id=call_id,
                name=name,
                arguments=args,
                provider_metadata={_GEMINI_METADATA_KEY: metadata},
            )
        )
    return tool_calls


def _source_from_grounding_chunk(chunk: Any) -> dict[str, Any] | None:
    if not isinstance(chunk, dict):
        return None
    payload = chunk.get("web") if isinstance(chunk.get("web"), dict) else chunk
    url = str(payload.get("uri") or payload.get("url") or "").strip()
    if not url:
        return None
    return {"url": url, "title": str(payload.get("title") or "").strip()}


def _grounding_metadata_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    grounding = candidate.get("groundingMetadata")
    if not isinstance(grounding, dict):
        return {}
    raw_chunks = grounding.get("groundingChunks")
    chunks = raw_chunks if isinstance(raw_chunks, list) else []
    sources: list[dict[str, Any]] = []
    for chunk in chunks:
        source = _source_from_grounding_chunk(chunk)
        if source is not None and source not in sources:
            sources.append(source)
    citations: list[dict[str, Any]] = []
    supports = grounding.get("groundingSupports")
    if isinstance(supports, list):
        for support in supports:
            if not isinstance(support, dict):
                continue
            segment = support.get("segment") if isinstance(support.get("segment"), dict) else {}
            for raw_index in support.get("groundingChunkIndices", []):
                try:
                    source = _source_from_grounding_chunk(chunks[int(raw_index)])
                except (TypeError, ValueError, IndexError):
                    continue
                if source is None:
                    continue
                citations.append(
                    {
                        "title": source["title"],
                        "url": source["url"],
                        "start_index": segment.get("startIndex"),
                        "end_index": segment.get("endIndex"),
                        "text": segment.get("text"),
                    }
                )
    raw_queries = grounding.get("webSearchQueries")
    queries = (
        [str(query).strip() for query in raw_queries if str(query).strip()]
        if isinstance(raw_queries, list)
        else []
    )
    payload: dict[str, Any] = {
        "groundingMetadata": copy.deepcopy(grounding),
    }
    if sources:
        payload["sources"] = sources
    if citations:
        payload["citations"] = citations
    if queries:
        payload["queries"] = queries
    return payload


def _gemini_provider_metadata(
    data: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any] | None:
    metadata: dict[str, Any] = {}
    response_id = str(data.get("responseId") or "").strip()
    if response_id:
        metadata["response_id"] = response_id
    model_version = str(data.get("modelVersion") or "").strip()
    if model_version:
        metadata["model_version"] = model_version
    finish_reason = str(candidate.get("finishReason") or "").strip()
    if finish_reason:
        metadata["finish_reason"] = finish_reason
    for key in ("finishMessage", "safetyRatings", "citationMetadata"):
        value = candidate.get(key)
        if value is not None:
            metadata[_camel_to_snake(key)] = copy.deepcopy(value)
    content = _candidate_content(candidate)
    if isinstance(content, dict):
        metadata["content"] = copy.deepcopy(content)
    metadata.update(_grounding_metadata_payload(candidate))
    usage = data.get("usageMetadata")
    if isinstance(usage, dict):
        metadata["usage"] = copy.deepcopy(usage)
    stream_metadata = data.get("streamMetadata")
    if isinstance(stream_metadata, dict):
        metadata["stream_metadata"] = copy.deepcopy(stream_metadata)
    return {_GEMINI_METADATA_KEY: metadata} if metadata else None


def _camel_to_snake(value: str) -> str:
    result = []
    for index, char in enumerate(value):
        if char.isupper() and index > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def _response_from_json(data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=data)


def _append_unique(target: list[Any], values: list[Any]) -> None:
    for value in values:
        copied = copy.deepcopy(value)
        if copied not in target:
            target.append(copied)


def _merge_stream_grounding_metadata(
    current: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(current)
    for key, value in incoming.items():
        if isinstance(value, list):
            existing = merged.setdefault(key, [])
            if isinstance(existing, list):
                _append_unique(existing, value)
            else:
                merged[key] = copy.deepcopy(value)
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_stream_grounding_metadata(merged[key], value)
            continue
        merged[key] = copy.deepcopy(value)
    return merged


class _GeminiStreamAccumulator:
    def __init__(self, *, on_text_delta: Callable[[str], None] | None) -> None:
        self.on_text_delta = on_text_delta
        self.response_id: str | None = None
        self.model_version: str | None = None
        self.usage_metadata: dict[str, Any] = {}
        self.parts: list[dict[str, Any]] = []
        self.role = "model"
        self.finish_reason: str | None = None
        self.finish_message: str | None = None
        self.safety_ratings: list[Any] | None = None
        self.citation_metadata: dict[str, Any] | None = None
        self.grounding_metadata: dict[str, Any] = {}
        self.stream_metadata: dict[str, Any] = {"chunks": 0}
        self.seen_candidate = False

    def handle(self, frame: SSEFrame, data: dict[str, Any]) -> None:
        _ = frame
        error_message = _extract_error_message(data)
        if error_message:
            raise LLMError(f"Gemini GenerateContent stream error: {error_message}")
        self.stream_metadata["chunks"] = int(self.stream_metadata["chunks"]) + 1
        if not data:
            self._record_unknown_chunk(data)
            return

        response_id = data.get("responseId")
        if isinstance(response_id, str) and response_id.strip():
            self.response_id = response_id
        model_version = data.get("modelVersion")
        if isinstance(model_version, str) and model_version.strip():
            self.model_version = model_version
        usage = data.get("usageMetadata")
        if isinstance(usage, dict):
            self.usage_metadata.update(copy.deepcopy(usage))

        candidates = data.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            self._record_unknown_chunk(data)
            return
        candidate = next((item for item in candidates if isinstance(item, dict)), None)
        if candidate is None:
            self._record_unknown_chunk(data)
            return
        self.seen_candidate = True
        self._handle_candidate(candidate)

    def _handle_candidate(self, candidate: dict[str, Any]) -> None:
        finish_reason = candidate.get("finishReason")
        if isinstance(finish_reason, str) and finish_reason.strip():
            self.finish_reason = finish_reason
        finish_message = candidate.get("finishMessage")
        if isinstance(finish_message, str) and finish_message.strip():
            self.finish_message = finish_message
        safety_ratings = candidate.get("safetyRatings")
        if isinstance(safety_ratings, list):
            self.safety_ratings = copy.deepcopy(safety_ratings)
        citation_metadata = candidate.get("citationMetadata")
        if isinstance(citation_metadata, dict):
            self.citation_metadata = copy.deepcopy(citation_metadata)
        grounding_metadata = candidate.get("groundingMetadata")
        if isinstance(grounding_metadata, dict):
            self.grounding_metadata = _merge_stream_grounding_metadata(
                self.grounding_metadata,
                grounding_metadata,
            )

        content = candidate.get("content")
        if not isinstance(content, dict):
            return
        role = content.get("role")
        if isinstance(role, str) and role.strip():
            self.role = role
        parts = content.get("parts")
        if not isinstance(parts, list):
            return
        for part in parts:
            if not isinstance(part, dict):
                continue
            copied = copy.deepcopy(part)
            self.parts.append(copied)
            text = copied.get("text")
            if isinstance(text, str) and text and self.on_text_delta is not None:
                self.on_text_delta(text)

    def _record_unknown_chunk(self, data: dict[str, Any]) -> None:
        unknown = self.stream_metadata.setdefault("unknown_chunks", [])
        if isinstance(unknown, list):
            unknown.append(copy.deepcopy(data))

    def finish(self) -> dict[str, Any]:
        if int(self.stream_metadata["chunks"]) <= 0:
            raise LLMError("Gemini GenerateContent stream returned no chunks")
        if not self.seen_candidate:
            raise LLMError("Gemini GenerateContent stream returned no candidate chunks")

        candidate: dict[str, Any] = {
            "content": {
                "role": self.role,
                "parts": copy.deepcopy(self.parts),
            }
        }
        if self.finish_reason:
            candidate["finishReason"] = self.finish_reason
        if self.finish_message:
            candidate["finishMessage"] = self.finish_message
        if self.safety_ratings is not None:
            candidate["safetyRatings"] = copy.deepcopy(self.safety_ratings)
        if self.citation_metadata is not None:
            candidate["citationMetadata"] = copy.deepcopy(self.citation_metadata)
        if self.grounding_metadata:
            candidate["groundingMetadata"] = copy.deepcopy(self.grounding_metadata)

        data: dict[str, Any] = {
            "candidates": [candidate],
            "streamMetadata": copy.deepcopy(self.stream_metadata),
        }
        if self.response_id:
            data["responseId"] = self.response_id
        if self.model_version:
            data["modelVersion"] = self.model_version
        if self.usage_metadata:
            data["usageMetadata"] = copy.deepcopy(self.usage_metadata)
        return data


class GeminiGenerateContentClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 20.0,
        temperature: float = 1.0,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        thinking_level: str | None = None,
        thinking_budget: int | None = None,
        transport: httpx.BaseTransport | None = None,
        extra_headers: dict[str, str] | None = None,
        provider_key: str | None = None,
        web_search_mode: str = "off",
        web_search_adapter: str = AUTO_WEB_SEARCH_ADAPTER,
        provider_concurrency_caps: dict[str, int] | None = None,
        provider_retry_settings: ProviderRetrySettings | None = None,
        provider_sleep_fn: Callable[[float], None] | None = None,
        provider_random_fn: Callable[[], float] | None = None,
    ) -> None:
        self.base_url = _gemini_native_base_url(base_url)
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.prompt_cache_key = str(prompt_cache_key or "").strip() or None
        self.prompt_cache_retention = str(prompt_cache_retention or "").strip() or None
        self.enable_thinking = enable_thinking
        self.reasoning_effort = str(reasoning_effort or "").strip().lower() or None
        self.thinking_level = str(thinking_level or "").strip().lower() or None
        self.thinking_budget = thinking_budget
        self._transport = transport
        self.extra_headers = {
            str(key): str(value)
            for key, value in (extra_headers or {}).items()
            if str(key).strip() and str(value).strip()
        }
        self.provider_key = str(provider_key or "").strip() or None
        self.web_search_mode = str(web_search_mode or "off").strip().lower()
        self.web_search_adapter = (
            str(web_search_adapter or AUTO_WEB_SEARCH_ADAPTER).strip().lower()
            or AUTO_WEB_SEARCH_ADAPTER
        )
        self.provider_concurrency_caps = dict(
            DEFAULT_PROVIDER_CONCURRENCY_CAPS
            if provider_concurrency_caps is None
            else provider_concurrency_caps
        )
        self.provider_retry_settings = provider_retry_settings or ProviderRetrySettings()
        self._provider_sleep_fn = provider_sleep_fn
        self._provider_random_fn = provider_random_fn

    def _headers(self) -> dict[str, str]:
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "sylliptor-agent-cli/0.1.0",
        }
        headers.update(self.extra_headers)
        return _headers_with_default_accept_encoding(headers)

    @staticmethod
    def _llm_error_from_response(response: httpx.Response) -> LLMError:
        try:
            data = response.json()
        except Exception:
            body = response.text
            if len(body) > 1000:
                body = body[:1000] + "...(truncated)"
            return LLMError(f"LLM error {response.status_code}: {body}")
        error_message = _extract_error_message(data)
        if error_message:
            return LLMError(f"LLM error {response.status_code}: {error_message}")
        return LLMError(f"LLM error {response.status_code}: {data!r}")

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
    ) -> LLMResponse:
        _ = on_reasoning_delta
        if self.prompt_cache_key or self.prompt_cache_retention:
            raise LLMError("Gemini GenerateContent does not support prompt_cache_key settings")

        system_instruction, contents = _gemini_contents_from_messages(messages)
        tool_mapping = _gemini_tools(
            tools,
            mode=self.web_search_mode,
            adapter=self.web_search_adapter,
        )
        generation_config: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else float(temperature),
        }
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = int(max_tokens)
        text_config = _gemini_response_format(response_format)
        generation_config.update(text_config)
        thinking_config = _gemini_thinking_config(
            model=self.model,
            enable_thinking=self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
            thinking_level=self.thinking_level,
            thinking_budget=self.thinking_budget,
        )
        if thinking_config:
            generation_config["thinkingConfig"] = thinking_config

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": generation_config,
        }
        if system_instruction is not None:
            payload["systemInstruction"] = system_instruction
        if tool_mapping.tools:
            payload["tools"] = tool_mapping.tools
            mapped_tool_choice = _gemini_tool_choice(
                tool_choice,
                removed_sylliptor_web_search=tool_mapping.removed_sylliptor_web_search,
                added_google_search=tool_mapping.added_google_search,
            )
            tool_config = dict(mapped_tool_choice or {})
            if tool_mapping.include_server_side_tool_invocations:
                tool_config[_INCLUDE_SERVER_SIDE_TOOL_INVOCATIONS] = True
            if tool_config:
                payload["toolConfig"] = tool_config
        elif tool_choice is not None:
            raise LLMError(
                "Gemini GenerateContent tool_choice requires at least one available tool"
            )

        provider_key = self.provider_key or best_effort_provider_key(
            base_url=self.base_url,
            model=self.model,
        )
        telemetry = ProviderCallTelemetryRecorder(
            provider_key=provider_key,
            protocol="gemini_generate_content",
            model=self.model,
            base_url=self.base_url,
            stream=stream,
            tools=tools,
            web_search_mode=self.web_search_mode,
            web_search_adapter=self.web_search_adapter,
            native_web_search=tool_mapping.added_google_search,
            operation="gemini_generate_content_chat",
        )
        telemetry_on_text_delta = telemetry.wrap_text_delta(on_text_delta)

        def _send_request() -> LLMResponse:
            encoded_model = quote(self.model, safe="")
            operation = "streamGenerateContent?alt=sse" if stream else "generateContent"
            url = f"{self.base_url}/models/{encoded_model}:{operation}"
            try:
                with httpx.Client(timeout=self.timeout_s, transport=self._transport) as client:
                    if stream:
                        with client.stream(
                            "POST",
                            url,
                            headers=self._headers(),
                            json=payload,
                        ) as response:
                            if response.status_code >= 400:
                                response.read()
                                raise self._llm_error_from_response(response)
                            return self._parse_stream_response(
                                response,
                                on_text_delta=telemetry_on_text_delta,
                            )
                    response = client.post(url, headers=self._headers(), json=payload)
            except httpx.DecodingError as e:
                raise LLMError(f"Gemini GenerateContent decompression failed: {e}") from e
            except Exception as e:  # noqa: BLE001
                if isinstance(e, LLMError):
                    raise
                raise LLMError(f"Gemini GenerateContent request failed: {e}") from e
            if response.status_code >= 400:
                raise self._llm_error_from_response(response)
            return self._parse_chat_response(response)

        return telemetry.run(
            lambda: run_provider_limited_call(
                call=_send_request,
                provider_key=provider_key,
                provider_concurrency_caps=self.provider_concurrency_caps,
                retry_settings=self.provider_retry_settings,
                operation="gemini_generate_content_chat",
                sleep_fn=self._provider_sleep_fn,
                random_fn=self._provider_random_fn,
                on_retry=telemetry.on_retry,
            )
        )

    @staticmethod
    def _parse_stream_response(
        response: httpx.Response,
        *,
        on_text_delta: Callable[[str], None] | None,
    ) -> LLMResponse:
        accumulator = _GeminiStreamAccumulator(on_text_delta=on_text_delta)
        for frame in iter_sse_frames(response.iter_lines()):
            raw_event = parse_sse_json_frame(frame, stream_name="Gemini GenerateContent stream")
            if not isinstance(raw_event, dict):
                raise LLMError("Gemini GenerateContent stream emitted non-object JSON event")
            accumulator.handle(frame, raw_event)
        data = accumulator.finish()
        return GeminiGenerateContentClient._parse_chat_response(_response_from_json(data))

    @staticmethod
    def _parse_chat_response(response: httpx.Response) -> LLMResponse:
        try:
            data = response.json()
        except Exception as e:  # noqa: BLE001
            raise LLMError("Gemini GenerateContent returned non-JSON response") from e
        if not isinstance(data, dict):
            raise LLMError("Unexpected Gemini GenerateContent payload: expected JSON object")
        candidate = _candidate(data)
        if candidate is None:
            raise LLMError("Unexpected Gemini GenerateContent payload: missing candidates")
        content = _candidate_content(candidate)
        if content is None:
            finish_reason = str(candidate.get("finishReason") or "").strip()
            suffix = f" (finish_reason={finish_reason})" if finish_reason else ""
            raise LLMError(f"Gemini GenerateContent returned no candidate content{suffix}")
        parts = content.get("parts")
        if not isinstance(parts, list):
            raise LLMError("Unexpected Gemini GenerateContent payload: missing content parts")

        text = _extract_text(parts)
        tool_calls = _parse_tool_calls(parts)
        if not text and not tool_calls:
            finish_reason = str(candidate.get("finishReason") or "").strip()
            suffix = f" (finish_reason={finish_reason})" if finish_reason else ""
            raise LLMError(
                f"Gemini GenerateContent returned no assistant text or tool calls{suffix}"
            )
        return LLMResponse(
            content=text,
            tool_calls=tool_calls,
            raw=data,
            response_model=data.get("modelVersion")
            if isinstance(data.get("modelVersion"), str)
            else None,
            usage=_parse_usage(data.get("usageMetadata")),
            provider_metadata=_gemini_provider_metadata(data, candidate),
        )


def _gemini_response_format(response_format: dict[str, Any] | None) -> dict[str, Any]:
    if not response_format:
        return {}
    response_type = str(response_format.get("type") or "").strip()
    if response_type == "json_object":
        return {"responseMimeType": "application/json"}
    if response_type == "json_schema":
        raw_json_schema = response_format.get("json_schema")
        json_schema = raw_json_schema if isinstance(raw_json_schema, dict) else response_format
        schema = json_schema.get("schema")
        if not isinstance(schema, dict):
            raise LLMError(
                "Gemini GenerateContent json_schema response_format requires schema object"
            )
        return {
            "responseMimeType": "application/json",
            "responseSchema": copy.deepcopy(schema),
        }
    if response_type == "text":
        return {}
    raise LLMError(
        f"Gemini GenerateContent does not support response_format type {response_type!r}"
    )


def _thinking_budget(reasoning_effort: str) -> int:
    effort = str(reasoning_effort or "").strip().lower()
    if effort in {"none", "minimal"}:
        return 0
    if effort == "low":
        return 1024
    if effort == "medium":
        return 4096
    if effort == "high":
        return 8192
    if effort == "xhigh":
        return 16384
    raise LLMError(f"Gemini GenerateContent reasoning_effort is not supported: {effort}")


def _is_gemini_3_model(model: str | None) -> bool:
    return "gemini-3" in str(model or "").strip().lower()


def _thinking_level_from_reasoning_effort(reasoning_effort: str) -> str:
    effort = str(reasoning_effort or "").strip().lower()
    if effort in {"none", "minimal"}:
        return "minimal"
    if effort == "low":
        return "low"
    if effort in {"medium", "high", "xhigh"}:
        return "high"
    raise LLMError(f"Gemini GenerateContent reasoning_effort is not supported: {effort}")


def _gemini_thinking_config(
    *,
    model: str,
    enable_thinking: bool | None,
    reasoning_effort: str | None,
    thinking_level: str | None,
    thinking_budget: int | None,
) -> dict[str, Any]:
    if thinking_level is not None and thinking_budget is not None:
        raise LLMError("Gemini GenerateContent cannot set both thinking_level and thinking_budget")
    if thinking_level is not None and not _is_gemini_3_model(model):
        raise LLMError("Gemini GenerateContent thinking_level requires a Gemini 3 model")
    if thinking_level is not None:
        normalized_level = str(thinking_level or "").strip().lower()
        if normalized_level not in _GEMINI_THINKING_LEVELS:
            raise LLMError(
                "Gemini GenerateContent thinking_level must be one of: high, low, minimal"
            )
        return {"thinkingLevel": normalized_level}
    if thinking_budget is not None:
        return {"thinkingBudget": int(thinking_budget)}
    if reasoning_effort:
        if _is_gemini_3_model(model):
            return {"thinkingLevel": _thinking_level_from_reasoning_effort(reasoning_effort)}
        return {"thinkingBudget": _thinking_budget(reasoning_effort)}
    if enable_thinking is False:
        if _is_gemini_3_model(model):
            return {"thinkingLevel": "minimal"}
        return {"thinkingBudget": 0}
    return {}
