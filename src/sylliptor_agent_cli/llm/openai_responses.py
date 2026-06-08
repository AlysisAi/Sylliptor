from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from ..provider_telemetry import ProviderCallTelemetryRecorder
from ..web_search_adapters import AUTO_WEB_SEARCH_ADAPTER, OPENAI_RESPONSES_ADAPTER
from .metadata import OPENAI_RESPONSES_PROVIDER_METADATA_KEY, PROVIDER_METADATA_KEY
from .provider_limits import (
    DEFAULT_PROVIDER_CONCURRENCY_CAPS,
    ProviderRetrySettings,
    best_effort_provider_key,
    run_provider_limited_call,
)
from .streaming import SSEFrame, iter_sse_frames, parse_sse_json_frame
from .types import LLMError, LLMResponse, LLMUsage, ToolCall


class ResponsesError(RuntimeError):
    pass


_DEFAULT_ACCEPT_ENCODING = "identity"
_OPENAI_RESPONSES_METADATA_KEY = OPENAI_RESPONSES_PROVIDER_METADATA_KEY
_WEB_SEARCH_MODES_ALLOWING_OPENAI_BUILTIN = frozenset({"auto", "native"})
_RESPONSES_TOOL_CHOICE_STRINGS = frozenset({"auto", "none", "required"})
_RESPONSES_REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
_RESPONSES_JSON_SCHEMA_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SYLLIPTOR_WEB_SEARCH_FUNCTION_NAME = "web_search"
_RESPONSES_HOSTED_WEB_SEARCH_TYPES = frozenset({"web_search", "web_search_preview"})


def _headers_with_default_accept_encoding(headers: dict[str, str]) -> dict[str, str]:
    request_headers = dict(headers)
    if not any(key.lower() == "accept-encoding" for key in request_headers):
        request_headers["Accept-Encoding"] = _DEFAULT_ACCEPT_ENCODING
    return request_headers


@dataclass(frozen=True)
class WebSearchCitation:
    title: str
    url: str
    start_index: int | None = None
    end_index: int | None = None


@dataclass(frozen=True)
class WebSearchSource:
    url: str
    title: str = ""


@dataclass(frozen=True)
class WebSearchResponse:
    answer: str
    citations: list[WebSearchCitation]
    sources: list[WebSearchSource]
    queries: list[str]
    raw: dict[str, Any]
    response_id: str | None = None
    model: str | None = None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_error_message(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    error_obj = data.get("error")
    if isinstance(error_obj, dict):
        message = str(error_obj.get("message") or "").strip()
        if message:
            return message
    return None


def _parse_arguments(args_s: Any) -> dict[str, Any]:
    if isinstance(args_s, dict):
        return dict(args_s)
    if not isinstance(args_s, str):
        args_s = json.dumps(args_s if args_s is not None else {})
    try:
        args = json.loads(args_s)
    except json.JSONDecodeError:
        return {"_raw_arguments": args_s}
    if not isinstance(args, dict):
        return {"_raw_arguments": args_s}
    return args


def _json_arguments(args: Any) -> str:
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except json.JSONDecodeError:
            return args
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    if args is None:
        return "{}"
    return json.dumps(args, ensure_ascii=False, separators=(",", ":"))


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
        return text if isinstance(text, str) else ""
    return str(raw)


def _responses_message_content(raw: Any, *, role: str) -> str | list[dict[str, Any]]:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, list):
        return _content_to_text(raw)

    parts: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            text = item
            if text:
                parts.append(
                    {"type": "output_text" if role == "assistant" else "input_text", "text": text}
                )
            continue
        if not isinstance(item, dict):
            continue
        part_type = str(item.get("type") or "").strip()
        text = item.get("text") or item.get("content")
        if part_type in {"text", "input_text", "output_text"} and isinstance(text, str):
            parts.append(
                {
                    "type": "output_text" if role == "assistant" else "input_text",
                    "text": text,
                }
            )
            continue
        if role != "user":
            continue
        if part_type == "image_url":
            image_url = item.get("image_url")
            url = ""
            if isinstance(image_url, dict):
                url = str(image_url.get("url") or "").strip()
            elif isinstance(image_url, str):
                url = image_url.strip()
            if url:
                parts.append({"type": "input_image", "image_url": url})
            continue
        if part_type == "input_image":
            copied = {key: copy.deepcopy(value) for key, value in item.items()}
            if copied.get("image_url") or copied.get("file_id"):
                parts.append(copied)
    return parts if parts else ""


def _chat_tool_call_parts(raw_tool_call: Any) -> tuple[str, str, str] | None:
    if not isinstance(raw_tool_call, dict):
        return None
    call_id = str(raw_tool_call.get("id") or raw_tool_call.get("call_id") or "").strip()
    function = raw_tool_call.get("function")
    if isinstance(function, dict):
        name = str(function.get("name") or "").strip()
        arguments = _json_arguments(function.get("arguments"))
    else:
        name = str(raw_tool_call.get("name") or "").strip()
        arguments = _json_arguments(raw_tool_call.get("arguments"))
    if not name:
        return None
    if not call_id:
        call_id = f"call_{name}"
    return call_id, name, arguments


def _metadata_output_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = message.get(PROVIDER_METADATA_KEY)
    if not isinstance(metadata, dict):
        return []
    responses_metadata = metadata.get(_OPENAI_RESPONSES_METADATA_KEY)
    if not isinstance(responses_metadata, dict):
        return []
    output_items = responses_metadata.get("output_items")
    if not isinstance(output_items, list):
        return []
    copied: list[dict[str, Any]] = []
    for item in output_items:
        if isinstance(item, dict):
            copied.append(copy.deepcopy(item))
    return copied


def _responses_input_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    input_items: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        if role in {"system", "developer", "user"}:
            input_items.append(
                {
                    "role": role,
                    "content": _responses_message_content(message.get("content"), role=role),
                }
            )
            continue
        if role == "assistant":
            metadata_items = _metadata_output_items(message)
            if metadata_items:
                input_items.extend(metadata_items)
                continue

            content = _responses_message_content(message.get("content"), role=role)
            if content:
                input_items.append({"role": "assistant", "content": content})
            raw_tool_calls = message.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                for raw_tool_call in raw_tool_calls:
                    parts = _chat_tool_call_parts(raw_tool_call)
                    if parts is None:
                        continue
                    call_id, name, arguments = parts
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": name,
                            "arguments": arguments,
                        }
                    )
            continue
        if role == "tool":
            call_id = str(message.get("tool_call_id") or message.get("call_id") or "").strip()
            if not call_id:
                raise LLMError("OpenAI Responses tool message is missing tool_call_id")
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": _content_to_text(message.get("content")),
                }
            )
            continue
        raise LLMError(f"OpenAI Responses cannot send message role {role!r}")
    return input_items


def _assistant_output_parts(data: dict[str, Any]) -> list[dict[str, Any]]:
    output = data.get("output")
    if not isinstance(output, list):
        return []
    parts: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") != "message":
            continue
        if str(item.get("role") or "") != "assistant":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict):
                parts.append(part)
    return parts


def _extract_answer_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    text_parts: list[str] = []
    for part in _assistant_output_parts(data):
        part_type = str(part.get("type") or "")
        if part_type not in {"output_text", "text"}:
            continue
        text = part.get("text")
        if isinstance(text, str):
            text_parts.append(text)
    return "".join(text_parts)


def _extract_citations(data: dict[str, Any]) -> list[WebSearchCitation]:
    citations: list[WebSearchCitation] = []
    for part in _assistant_output_parts(data):
        annotations = part.get("annotations")
        if not isinstance(annotations, list):
            continue
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            if str(annotation.get("type") or "") != "url_citation":
                continue
            url = str(annotation.get("url") or "").strip()
            if not url:
                continue
            citations.append(
                WebSearchCitation(
                    title=str(annotation.get("title") or "").strip(),
                    url=url,
                    start_index=_coerce_int(annotation.get("start_index")),
                    end_index=_coerce_int(annotation.get("end_index")),
                )
            )
    raw_citations = data.get("citations")
    if isinstance(raw_citations, list):
        for raw_citation in raw_citations:
            citation = _coerce_citation(raw_citation)
            if citation is not None:
                citations.append(citation)
    return _dedupe_citations(citations)


def _coerce_citation(raw_citation: Any) -> WebSearchCitation | None:
    if isinstance(raw_citation, str):
        url = raw_citation.strip()
        if not url:
            return None
        return WebSearchCitation(title="", url=url)
    if not isinstance(raw_citation, dict):
        return None

    citation_payload = raw_citation
    for nested_key in ("url_citation", "web_citation", "x_citation"):
        nested = raw_citation.get(nested_key)
        if isinstance(nested, dict):
            citation_payload = nested
            break

    url = str(
        citation_payload.get("url")
        or citation_payload.get("uri")
        or citation_payload.get("link")
        or ""
    ).strip()
    if not url:
        return None
    start_index = citation_payload.get("start_index")
    if start_index is None:
        start_index = citation_payload.get("startIndex")
    end_index = citation_payload.get("end_index")
    if end_index is None:
        end_index = citation_payload.get("endIndex")
    return WebSearchCitation(
        title=str(citation_payload.get("title") or citation_payload.get("name") or "").strip(),
        url=url,
        start_index=_coerce_int(start_index),
        end_index=_coerce_int(end_index),
    )


def _dedupe_citations(citations: list[WebSearchCitation]) -> list[WebSearchCitation]:
    deduped: list[WebSearchCitation] = []
    seen: set[str] = set()
    for citation in citations:
        url = str(citation.url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(citation)
    return deduped


def _extract_sources_and_queries(data: dict[str, Any]) -> tuple[list[WebSearchSource], list[str]]:
    output = data.get("output")
    if not isinstance(output, list):
        return [], []

    sources: list[WebSearchSource] = []
    queries: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") != "web_search_call":
            continue
        action = item.get("action")
        if not isinstance(action, dict):
            continue

        raw_sources = action.get("sources")
        if isinstance(raw_sources, list):
            for raw_source in raw_sources:
                if not isinstance(raw_source, dict):
                    continue
                url = str(raw_source.get("url") or "").strip()
                if not url:
                    continue
                sources.append(
                    WebSearchSource(
                        url=url,
                        title=str(raw_source.get("title") or "").strip(),
                    )
                )

        raw_queries = action.get("queries")
        if isinstance(raw_queries, list):
            for raw_query in raw_queries:
                query = str(raw_query or "").strip()
                if query:
                    queries.append(query)
        raw_query = action.get("query")
        if isinstance(raw_query, str) and raw_query.strip():
            queries.append(raw_query.strip())

    return sources, queries


def _merge_citation_sources(
    sources: list[WebSearchSource],
    citations: list[WebSearchCitation],
) -> list[WebSearchSource]:
    merged = list(sources)
    seen = {str(source.url or "").strip() for source in merged if str(source.url or "").strip()}
    for citation in citations:
        url = str(citation.url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        merged.append(WebSearchSource(url=url, title=str(citation.title or "").strip()))
    return merged


def _function_name_from_tool(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "").strip()
    if str(tool.get("type") or "") == "function":
        return str(tool.get("name") or "").strip()
    return ""


def _responses_tool_from_chat_tool(tool: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    tool_type = str(tool.get("type") or "").strip()
    if tool_type == "function":
        function = tool.get("function")
        if isinstance(function, dict):
            name = str(function.get("name") or "").strip()
            if not name:
                return None
            mapped: dict[str, Any] = {
                "type": "function",
                "name": name,
                "parameters": copy.deepcopy(function.get("parameters") or {"type": "object"}),
            }
            description = str(function.get("description") or "").strip()
            if description:
                mapped["description"] = description
            strict = function.get("strict", tool.get("strict"))
            if strict is not None:
                mapped["strict"] = bool(strict)
            return mapped

        name = str(tool.get("name") or "").strip()
        if not name:
            return None
        mapped = copy.deepcopy(tool)
        mapped["type"] = "function"
        return mapped
    if tool_type in {"web_search", "web_search_preview"}:
        return copy.deepcopy(tool)
    raise LLMError(f"OpenAI Responses does not support tool type {tool_type!r}")


def _tools_contain_function(tools: list[dict[str, Any]] | None, name: str) -> bool:
    if not tools:
        return False
    return any(_function_name_from_tool(tool) == name for tool in tools if isinstance(tool, dict))


def _is_sylliptor_web_search_function(tool: dict[str, Any]) -> bool:
    return _function_name_from_tool(tool) == _SYLLIPTOR_WEB_SEARCH_FUNCTION_NAME


def _is_responses_hosted_web_search_tool(tool: dict[str, Any]) -> bool:
    return str(tool.get("type") or "").strip() in _RESPONSES_HOSTED_WEB_SEARCH_TYPES


def _openai_builtin_web_search_allowed(*, mode: str, adapter: str) -> bool:
    normalized_mode = str(mode or "").strip().lower()
    normalized_adapter = str(adapter or "").strip().lower() or AUTO_WEB_SEARCH_ADAPTER
    if normalized_mode not in _WEB_SEARCH_MODES_ALLOWING_OPENAI_BUILTIN:
        return False
    return normalized_adapter in {AUTO_WEB_SEARCH_ADAPTER, OPENAI_RESPONSES_ADAPTER}


@dataclass(frozen=True)
class _ResponsesToolMapping:
    tools: list[dict[str, Any]]
    added_builtin_web_search: bool
    removed_sylliptor_web_search: bool


def _responses_tools(
    tools: list[dict[str, Any]] | None,
    *,
    mode: str,
    adapter: str,
) -> _ResponsesToolMapping:
    normalized_mode = str(mode or "off").strip().lower()
    normalized_adapter = (
        str(adapter or AUTO_WEB_SEARCH_ADAPTER).strip().lower() or AUTO_WEB_SEARCH_ADAPTER
    )
    raw_tools = [tool for tool in tools or [] if isinstance(tool, dict)]
    sylliptor_web_search_present = any(
        _is_sylliptor_web_search_function(tool) for tool in raw_tools
    )
    use_openai_builtin_web_search = (
        sylliptor_web_search_present
        and _openai_builtin_web_search_allowed(
            mode=normalized_mode,
            adapter=normalized_adapter,
        )
    )
    if (
        normalized_mode == "native"
        and sylliptor_web_search_present
        and not use_openai_builtin_web_search
    ):
        raise LLMError(
            "web_search_mode=native with protocol=openai_responses requires "
            "web_search_adapter='auto' or 'openai_responses' for OpenAI hosted web_search; "
            f"got {normalized_adapter!r}"
        )

    mapped_tools: list[dict[str, Any]] = []
    removed_sylliptor_web_search = False
    for tool in raw_tools:
        if _is_sylliptor_web_search_function(tool):
            if normalized_mode in {"off", "native"} or use_openai_builtin_web_search:
                removed_sylliptor_web_search = True
                continue
        if _is_responses_hosted_web_search_tool(tool) and normalized_mode in {"off", "external"}:
            continue
        mapped = _responses_tool_from_chat_tool(tool)
        if mapped is not None:
            mapped_tools.append(mapped)
    if use_openai_builtin_web_search and not any(
        _is_responses_hosted_web_search_tool(tool) for tool in mapped_tools
    ):
        mapped_tools.append({"type": "web_search", "external_web_access": True})
    return _ResponsesToolMapping(
        tools=mapped_tools,
        added_builtin_web_search=use_openai_builtin_web_search,
        removed_sylliptor_web_search=removed_sylliptor_web_search,
    )


def _tool_choice_for_mapped_tools(
    tool_choice: Any,
    *,
    removed_sylliptor_web_search: bool,
) -> Any:
    if not removed_sylliptor_web_search:
        return _responses_tool_choice(tool_choice)
    if isinstance(tool_choice, dict) and str(tool_choice.get("type") or "").strip() == "function":
        if "name" in tool_choice:
            name = str(tool_choice.get("name") or "").strip()
        else:
            function = tool_choice.get("function")
            name = str(function.get("name") or "").strip() if isinstance(function, dict) else ""
        if name == _SYLLIPTOR_WEB_SEARCH_FUNCTION_NAME:
            raise LLMError(
                "OpenAI Responses removed the Sylliptor web_search function for the selected "
                "web_search_mode; do not force tool_choice to function web_search"
            )
    return _responses_tool_choice(tool_choice)


def _responses_tool_choice(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        normalized = tool_choice.strip()
        if normalized in _RESPONSES_TOOL_CHOICE_STRINGS:
            return normalized
        raise LLMError(f"OpenAI Responses does not support tool_choice={tool_choice!r}")
    if not isinstance(tool_choice, dict):
        raise LLMError("OpenAI Responses tool_choice must be a string or object")

    choice_type = str(tool_choice.get("type") or "").strip()
    if choice_type == "function":
        if "name" in tool_choice:
            name = str(tool_choice.get("name") or "").strip()
        else:
            function = tool_choice.get("function")
            name = str(function.get("name") or "").strip() if isinstance(function, dict) else ""
        if not name:
            raise LLMError("OpenAI Responses forced function tool_choice is missing name")
        return {"type": "function", "name": name}
    if choice_type == "allowed_tools":
        return copy.deepcopy(tool_choice)
    if choice_type in {"web_search", "web_search_preview"}:
        return copy.deepcopy(tool_choice)
    raise LLMError(f"OpenAI Responses does not support tool_choice type {choice_type!r}")


def _responses_text_config(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    if not response_format:
        return None
    if "format" in response_format and isinstance(response_format.get("format"), dict):
        return copy.deepcopy(response_format)

    response_type = str(response_format.get("type") or "").strip()
    if response_type == "json_schema":
        raw_json_schema = response_format.get("json_schema")
        json_schema = raw_json_schema if isinstance(raw_json_schema, dict) else response_format
        name = str(json_schema.get("name") or "").strip()
        schema = json_schema.get("schema")
        if not name or not _RESPONSES_JSON_SCHEMA_NAME_RE.fullmatch(name):
            raise LLMError(
                "OpenAI Responses json_schema response_format requires a valid name "
                "(letters, digits, underscores, or dashes; max 64 chars)"
            )
        if not isinstance(schema, dict):
            raise LLMError("OpenAI Responses json_schema response_format requires schema object")
        fmt: dict[str, Any] = {
            "type": "json_schema",
            "name": name,
            "schema": copy.deepcopy(schema),
        }
        description = str(json_schema.get("description") or "").strip()
        if description:
            fmt["description"] = description
        if "strict" in json_schema:
            fmt["strict"] = bool(json_schema.get("strict"))
        return {"format": fmt}
    if response_type in {"json_object", "text"}:
        return {"format": {"type": response_type}}
    raise LLMError(f"OpenAI Responses does not support response_format type {response_type!r}")


def _responses_reasoning(
    *,
    enable_thinking: bool | None,
    reasoning_effort: str | None,
) -> dict[str, Any] | None:
    effort = str(reasoning_effort or "").strip().lower()
    if effort:
        if effort not in _RESPONSES_REASONING_EFFORTS:
            raise LLMError(f"OpenAI Responses reasoning_effort is not supported: {effort}")
        return {"effort": effort}
    if enable_thinking is False:
        return {"effort": "none"}
    return None


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

    input_tokens = raw.get("input_tokens", raw.get("prompt_tokens"))
    output_tokens = raw.get("output_tokens", raw.get("completion_tokens"))
    total_tokens = raw.get("total_tokens")
    cached_tokens = raw.get("cached_prompt_tokens")
    input_details = raw.get("input_tokens_details")
    if cached_tokens is None and isinstance(input_details, dict):
        cached_tokens = input_details.get("cached_tokens")
    prompt_details = raw.get("prompt_tokens_details")
    if cached_tokens is None and isinstance(prompt_details, dict):
        cached_tokens = prompt_details.get("cached_tokens")

    usage = LLMUsage(
        prompt_tokens=_as_non_negative_int(input_tokens),
        completion_tokens=_as_non_negative_int(output_tokens),
        total_tokens=_as_non_negative_int(total_tokens),
        cached_prompt_tokens=_as_non_negative_int(cached_tokens),
    )
    if (
        usage.prompt_tokens is None
        and usage.completion_tokens is None
        and usage.total_tokens is None
        and usage.cached_prompt_tokens is None
    ):
        return None
    return usage


def _citation_to_dict(citation: WebSearchCitation) -> dict[str, Any]:
    return {
        "title": citation.title,
        "url": citation.url,
        "start_index": citation.start_index,
        "end_index": citation.end_index,
    }


def _source_to_dict(source: WebSearchSource) -> dict[str, Any]:
    return {"title": source.title, "url": source.url}


def _responses_provider_metadata(data: dict[str, Any]) -> dict[str, Any] | None:
    metadata: dict[str, Any] = {}
    response_id = str(data.get("id") or "").strip()
    if response_id:
        metadata["response_id"] = response_id
    output = data.get("output")
    if isinstance(output, list):
        metadata["output_items"] = copy.deepcopy(output)
        web_search_calls = [
            copy.deepcopy(item)
            for item in output
            if isinstance(item, dict) and str(item.get("type") or "") == "web_search_call"
        ]
        if web_search_calls:
            metadata["web_search_calls"] = web_search_calls
    citations = [_citation_to_dict(citation) for citation in _extract_citations(data)]
    if citations:
        metadata["citations"] = citations
    sources, queries = _extract_sources_and_queries(data)
    sources = _merge_citation_sources(sources, _extract_citations(data))
    if sources:
        metadata["sources"] = [_source_to_dict(source) for source in sources]
    if queries:
        metadata["queries"] = list(queries)
    stream_metadata = data.get("stream_metadata")
    if isinstance(stream_metadata, dict):
        metadata["stream_metadata"] = copy.deepcopy(stream_metadata)
    return {_OPENAI_RESPONSES_METADATA_KEY: metadata} if metadata else None


def _extract_refusal(data: dict[str, Any]) -> str:
    output = data.get("output")
    if not isinstance(output, list):
        return ""
    refusals: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            refusal = part.get("refusal")
            if isinstance(refusal, str) and refusal.strip():
                refusals.append(refusal.strip())
            if str(part.get("type") or "") == "refusal":
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    refusals.append(text.strip())
    return "\n".join(refusals)


def _parse_response_tool_calls(data: dict[str, Any]) -> list[ToolCall]:
    output = data.get("output")
    if not isinstance(output, list):
        return []
    tool_calls: list[ToolCall] = []
    for index, item in enumerate(output):
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") != "function_call":
            continue
        call_id = str(item.get("call_id") or item.get("id") or f"call_{index}").strip()
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        provider_metadata: dict[str, Any] = {
            _OPENAI_RESPONSES_METADATA_KEY: {
                "item_id": item.get("id"),
                "output_index": index,
                "status": item.get("status"),
            }
        }
        tool_calls.append(
            ToolCall(
                id=call_id,
                name=name,
                arguments=_parse_arguments(item.get("arguments") or "{}"),
                provider_metadata=provider_metadata,
            )
        )
    return tool_calls


def _response_from_json(data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=data)


def _event_output_index(data: dict[str, Any], *, event_type: str) -> int:
    index = data.get("output_index")
    if isinstance(index, int) and index >= 0:
        return index
    raise LLMError(f"OpenAI Responses stream {event_type} event is missing output_index")


def _event_content_index(data: dict[str, Any]) -> int:
    index = data.get("content_index")
    if isinstance(index, int) and index >= 0:
        return index
    return 0


def _stream_error_message(data: dict[str, Any]) -> str:
    message = _extract_error_message(data)
    if message:
        return message
    response = data.get("response")
    if isinstance(response, dict):
        message = _extract_error_message(response)
        if message:
            return message
        status = str(response.get("status") or "").strip()
        incomplete = response.get("incomplete_details")
        if isinstance(incomplete, dict):
            reason = str(incomplete.get("reason") or "").strip()
            if reason:
                return f"status={status or 'incomplete'} reason={reason}"
        if status:
            return f"status={status}"
    return repr(data)


class _OpenAIResponsesStreamAccumulator:
    def __init__(self, *, on_text_delta: Callable[[str], None] | None) -> None:
        self.on_text_delta = on_text_delta
        self.response: dict[str, Any] = {"object": "response", "output": []}
        self.output_items: dict[int, dict[str, Any]] = {}
        self.text_parts: dict[tuple[int, int], dict[str, Any]] = {}
        self.argument_chunks: dict[int, list[str]] = {}
        self.unknown_events: list[dict[str, Any]] = []
        self.event_count = 0
        self.seen_final = False
        self.final_response: dict[str, Any] | None = None

    def handle(self, frame: SSEFrame, data: dict[str, Any]) -> None:
        event_type = str(data.get("type") or frame.event or "").strip()
        if not event_type:
            self._append_unknown(frame=frame, data=data)
            return
        self.event_count += 1

        if event_type == "error":
            raise LLMError(f"OpenAI Responses stream error: {_stream_error_message(data)}")
        if event_type in {"response.failed", "response.incomplete"}:
            raise LLMError(f"OpenAI Responses stream {event_type}: {_stream_error_message(data)}")
        if event_type in {"response.created", "response.in_progress", "response.queued"}:
            self._merge_response(data.get("response"))
            return
        if event_type in {"response.completed", "response.done"}:
            self._handle_final_response(data)
            return
        if event_type in {"response.output_item.added", "response.output_item.done"}:
            self._handle_output_item(data)
            return
        if event_type in {"response.content_part.added", "response.content_part.done"}:
            self._handle_content_part(data)
            return
        if event_type == "response.output_text.delta":
            self._handle_output_text_delta(data)
            return
        if event_type == "response.output_text.done":
            self._handle_output_text_done(data)
            return
        if event_type == "response.output_text.annotation.added":
            self._handle_output_text_annotation(data)
            return
        if event_type == "response.function_call_arguments.delta":
            self._handle_function_arguments_delta(data)
            return
        if event_type == "response.function_call_arguments.done":
            self._handle_function_arguments_done(data)
            return
        if event_type.startswith("response.web_search_call."):
            self._handle_web_search_call_state(event_type, data)
            return

        self._append_unknown(frame=frame, data=data)

    def finish(self) -> dict[str, Any]:
        if self.event_count <= 0:
            raise LLMError("OpenAI Responses stream returned no events")
        if not self.seen_final:
            raise LLMError("OpenAI Responses stream ended before response.completed")
        data = copy.deepcopy(self.final_response or self.response)
        output = data.get("output")
        if not isinstance(output, list) or not output:
            output = self._ordered_output_items()
            data["output"] = output
        elif self.output_items:
            data["output"] = self._merge_ordered_items(output)
        if "output_text" not in data:
            text = _extract_answer_text(data)
            if text:
                data["output_text"] = text
        stream_metadata: dict[str, Any] = {"events": self.event_count}
        if self.unknown_events:
            stream_metadata["unknown_events"] = copy.deepcopy(self.unknown_events)
        data["stream_metadata"] = stream_metadata
        return data

    def _merge_response(self, raw_response: Any) -> None:
        if not isinstance(raw_response, dict):
            return
        response = copy.deepcopy(raw_response)
        output = response.pop("output", None)
        self.response.update(response)
        if isinstance(output, list):
            for index, item in enumerate(output):
                if isinstance(item, dict):
                    self._set_output_item(index, item)

    def _handle_final_response(self, data: dict[str, Any]) -> None:
        self._merge_response(data.get("response"))
        response = data.get("response")
        self.final_response = (
            copy.deepcopy(response) if isinstance(response, dict) else copy.deepcopy(self.response)
        )
        self.seen_final = True

    def _handle_output_item(self, data: dict[str, Any]) -> None:
        index = _event_output_index(data, event_type=str(data.get("type") or "output_item"))
        item = data.get("item")
        if not isinstance(item, dict):
            return
        self._set_output_item(index, item)

    def _set_output_item(self, index: int, item: dict[str, Any]) -> dict[str, Any]:
        copied = copy.deepcopy(item)
        existing = self.output_items.get(index)
        if existing is not None:
            merged = copy.deepcopy(existing)
            merged.update(copied)
            copied = merged
        self.output_items[index] = copied
        return copied

    def _ensure_message_item(self, output_index: int, item_id: str | None = None) -> dict[str, Any]:
        item = self.output_items.get(output_index)
        if not isinstance(item, dict):
            item = {
                "type": "message",
                "role": "assistant",
                "content": [],
            }
            if item_id:
                item["id"] = item_id
            self.output_items[output_index] = item
        else:
            item.setdefault("type", "message")
            item.setdefault("role", "assistant")
            if item_id and not item.get("id"):
                item["id"] = item_id
            content = item.get("content")
            if not isinstance(content, list):
                item["content"] = []
        return item

    def _ensure_text_part(
        self, output_index: int, content_index: int, item_id: str
    ) -> dict[str, Any]:
        key = (output_index, content_index)
        part = self.text_parts.get(key)
        if part is None:
            part = {"type": "output_text", "text": ""}
            self.text_parts[key] = part
            item = self._ensure_message_item(output_index, item_id)
            content = item.setdefault("content", [])
            if isinstance(content, list):
                while len(content) <= content_index:
                    content.append({"type": "output_text", "text": ""})
                content[content_index] = part
        return part

    def _handle_content_part(self, data: dict[str, Any]) -> None:
        output_index = _event_output_index(data, event_type=str(data.get("type") or "content_part"))
        content_index = _event_content_index(data)
        item_id = str(data.get("item_id") or "").strip()
        raw_part = data.get("part")
        if not isinstance(raw_part, dict):
            return
        item = self._ensure_message_item(output_index, item_id or None)
        content = item.setdefault("content", [])
        if not isinstance(content, list):
            content = []
            item["content"] = content
        while len(content) <= content_index:
            content.append({"type": "output_text", "text": ""})
        part = copy.deepcopy(raw_part)
        if str(part.get("type") or "") in {"text", "output_text"}:
            part["type"] = "output_text"
            part.setdefault("text", "")
            self.text_parts[(output_index, content_index)] = part
        content[content_index] = part

    def _handle_output_text_delta(self, data: dict[str, Any]) -> None:
        output_index = _event_output_index(data, event_type="response.output_text.delta")
        content_index = _event_content_index(data)
        item_id = str(data.get("item_id") or "").strip()
        delta = data.get("delta")
        if not isinstance(delta, str) or not delta:
            return
        part = self._ensure_text_part(output_index, content_index, item_id)
        existing = part.get("text")
        part["text"] = (existing if isinstance(existing, str) else "") + delta
        if self.on_text_delta is not None:
            self.on_text_delta(delta)

    def _handle_output_text_done(self, data: dict[str, Any]) -> None:
        output_index = _event_output_index(data, event_type="response.output_text.done")
        content_index = _event_content_index(data)
        item_id = str(data.get("item_id") or "").strip()
        text = data.get("text")
        if not isinstance(text, str):
            return
        part = self._ensure_text_part(output_index, content_index, item_id)
        part["text"] = text
        part["type"] = "output_text"

    def _handle_output_text_annotation(self, data: dict[str, Any]) -> None:
        output_index = _event_output_index(
            data,
            event_type="response.output_text.annotation.added",
        )
        content_index = _event_content_index(data)
        item_id = str(data.get("item_id") or "").strip()
        annotation = data.get("annotation")
        if not isinstance(annotation, dict):
            return
        part = self._ensure_text_part(output_index, content_index, item_id)
        annotations = part.setdefault("annotations", [])
        if isinstance(annotations, list):
            annotations.append(copy.deepcopy(annotation))

    def _handle_function_arguments_delta(self, data: dict[str, Any]) -> None:
        output_index = _event_output_index(
            data,
            event_type="response.function_call_arguments.delta",
        )
        delta = data.get("delta")
        if isinstance(delta, str):
            self.argument_chunks.setdefault(output_index, []).append(delta)
        item = self.output_items.get(output_index)
        if isinstance(item, dict) and str(item.get("type") or "") == "function_call":
            existing = item.get("arguments")
            item["arguments"] = (existing if isinstance(existing, str) else "") + (
                delta if isinstance(delta, str) else ""
            )

    def _handle_function_arguments_done(self, data: dict[str, Any]) -> None:
        output_index = _event_output_index(
            data,
            event_type="response.function_call_arguments.done",
        )
        item = self.output_items.get(output_index)
        if not isinstance(item, dict):
            item = {"type": "function_call"}
            self.output_items[output_index] = item
        item["type"] = "function_call"
        for key in ("item_id", "call_id", "name", "status"):
            value = data.get(key)
            if value is not None:
                item["id" if key == "item_id" else key] = copy.deepcopy(value)
        arguments = data.get("arguments")
        if isinstance(arguments, str):
            item["arguments"] = arguments
        elif output_index in self.argument_chunks:
            item["arguments"] = "".join(self.argument_chunks[output_index])

    def _handle_web_search_call_state(self, event_type: str, data: dict[str, Any]) -> None:
        try:
            output_index = _event_output_index(data, event_type=event_type)
        except LLMError:
            self._append_unknown(frame=SSEFrame(event=event_type, data=json.dumps(data)), data=data)
            return
        item = self.output_items.get(output_index)
        if not isinstance(item, dict):
            item = {"type": "web_search_call"}
            self.output_items[output_index] = item
        item["type"] = "web_search_call"
        item_id = data.get("item_id")
        if isinstance(item_id, str) and item_id.strip():
            item["id"] = item_id
        status = event_type.rsplit(".", 1)[-1]
        if status:
            item["status"] = status
        for key in ("action", "results"):
            value = data.get(key)
            if value is not None:
                item[key] = copy.deepcopy(value)

    def _ordered_output_items(self) -> list[dict[str, Any]]:
        return [
            copy.deepcopy(item)
            for _index, item in sorted(self.output_items.items(), key=lambda pair: pair[0])
            if isinstance(item, dict)
        ]

    def _merge_ordered_items(self, output: list[Any]) -> list[dict[str, Any]]:
        merged: dict[int, dict[str, Any]] = {}
        for index, item in enumerate(output):
            if isinstance(item, dict):
                merged[index] = copy.deepcopy(item)
        for index, item in self.output_items.items():
            if not isinstance(item, dict):
                continue
            if index in merged:
                copied = copy.deepcopy(item)
                copied.update(copy.deepcopy(merged[index]))
                merged[index] = copied
            else:
                merged[index] = copy.deepcopy(item)
        return [item for _index, item in sorted(merged.items(), key=lambda pair: pair[0])]

    def _append_unknown(self, *, frame: SSEFrame, data: dict[str, Any]) -> None:
        self.unknown_events.append(
            {
                "event": frame.event,
                "data": copy.deepcopy(data),
            }
        )


class OpenAIResponsesClient:
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
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "sylliptor-agent-cli/0.1.0",
        }
        headers.update(self.extra_headers)
        return _headers_with_default_accept_encoding(headers)

    @staticmethod
    def _error_from_response(response: httpx.Response) -> ResponsesError:
        try:
            data = response.json()
        except Exception:
            body = response.text
            if len(body) > 1000:
                body = body[:1000] + "...(truncated)"
            return ResponsesError(f"Responses error {response.status_code}: {body}")
        if isinstance(data, dict):
            error_message = _extract_error_message(data)
            if error_message:
                lower = error_message.lower()
                if "unsupported" in lower or "not support" in lower:
                    return ResponsesError(f"Responses web_search unsupported: {error_message}")
                return ResponsesError(f"Responses error {response.status_code}: {error_message}")
        return ResponsesError(f"Responses error {response.status_code}: {data!r}")

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
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        tool_mapping = _responses_tools(
            tools,
            mode=self.web_search_mode,
            adapter=self.web_search_adapter,
        )
        mapped_tools = tool_mapping.tools
        payload: dict[str, Any] = {
            "model": self.model,
            "input": _responses_input_from_messages(messages),
            "temperature": self.temperature if temperature is None else float(temperature),
        }
        if self.prompt_cache_key:
            payload["prompt_cache_key"] = self.prompt_cache_key
        if self.prompt_cache_retention:
            payload["prompt_cache_retention"] = self.prompt_cache_retention
        reasoning = _responses_reasoning(
            enable_thinking=self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
        )
        if reasoning is not None:
            payload["reasoning"] = reasoning
        if mapped_tools:
            payload["tools"] = mapped_tools
            payload["tool_choice"] = (
                "auto"
                if tool_choice is None
                else _tool_choice_for_mapped_tools(
                    tool_choice,
                    removed_sylliptor_web_search=tool_mapping.removed_sylliptor_web_search,
                )
            )
            if tool_mapping.added_builtin_web_search:
                payload["include"] = ["web_search_call.action.sources"]
        elif tool_choice is not None:
            payload["tool_choice"] = _tool_choice_for_mapped_tools(
                tool_choice,
                removed_sylliptor_web_search=tool_mapping.removed_sylliptor_web_search,
            )
        text_config = _responses_text_config(response_format)
        if text_config is not None:
            payload["text"] = text_config
        if max_tokens is not None:
            payload["max_output_tokens"] = int(max_tokens)
        if stream:
            payload["stream"] = True

        provider_key = self.provider_key or best_effort_provider_key(
            base_url=self.base_url,
            model=self.model,
        )
        telemetry = ProviderCallTelemetryRecorder(
            provider_key=provider_key,
            protocol="openai_responses",
            model=self.model,
            base_url=self.base_url,
            stream=stream,
            tools=tools,
            web_search_mode=self.web_search_mode,
            web_search_adapter=self.web_search_adapter,
            native_web_search=tool_mapping.added_builtin_web_search,
            operation="responses_chat",
        )
        telemetry_on_text_delta = telemetry.wrap_text_delta(on_text_delta)

        def _send_request() -> LLMResponse:
            url = f"{self.base_url}/responses"
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
                raise LLMError(f"OpenAI Responses decompression failed: {e}") from e
            except Exception as e:  # noqa: BLE001
                if isinstance(e, LLMError):
                    raise
                raise LLMError(f"OpenAI Responses request failed: {e}") from e
            if response.status_code >= 400:
                raise self._llm_error_from_response(response)
            return self._parse_chat_response(response)

        return telemetry.run(
            lambda: run_provider_limited_call(
                call=_send_request,
                provider_key=provider_key,
                provider_concurrency_caps=self.provider_concurrency_caps,
                retry_settings=self.provider_retry_settings,
                operation="responses_chat",
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
        accumulator = _OpenAIResponsesStreamAccumulator(on_text_delta=on_text_delta)
        for frame in iter_sse_frames(response.iter_lines()):
            raw_event = parse_sse_json_frame(frame, stream_name="OpenAI Responses stream")
            if not isinstance(raw_event, dict):
                raise LLMError("OpenAI Responses stream emitted non-object JSON event")
            accumulator.handle(frame, raw_event)
        data = accumulator.finish()
        return OpenAIResponsesClient._parse_chat_response(_response_from_json(data))

    @staticmethod
    def _parse_chat_response(response: httpx.Response) -> LLMResponse:
        try:
            data = response.json()
        except Exception as e:  # noqa: BLE001
            raise LLMError("OpenAI Responses returned non-JSON response") from e
        if not isinstance(data, dict):
            raise LLMError("Unexpected OpenAI Responses payload: expected JSON object")

        content = _extract_answer_text(data)
        tool_calls = _parse_response_tool_calls(data)
        if not content and not tool_calls:
            refusal = _extract_refusal(data)
            if refusal:
                raise LLMError(f"OpenAI Responses refusal: {refusal}")
            status = str(data.get("status") or "").strip()
            suffix = f" (status={status})" if status else ""
            raise LLMError(f"OpenAI Responses returned no assistant text or tool calls{suffix}")

        response_model = data.get("model") if isinstance(data.get("model"), str) else None
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            raw=data,
            response_model=response_model,
            usage=_parse_usage(data.get("usage")),
            provider_metadata=_responses_provider_metadata(data),
        )

    def web_search(
        self,
        *,
        query: str,
        allowed_domains: list[str] | None = None,
        external_web_access: bool | None = None,
        include_source_details: bool = True,
        tool_choice: str | dict[str, Any] | None = "required",
    ) -> WebSearchResponse:
        url = f"{self.base_url}/responses"
        headers = self._headers()

        tool_spec: dict[str, Any] = {"type": "web_search"}
        if allowed_domains:
            tool_spec["filters"] = {"allowed_domains": list(allowed_domains)}
        if external_web_access is not None:
            tool_spec["external_web_access"] = bool(external_web_access)

        payload: dict[str, Any] = {
            "model": self.model,
            "input": query,
            "tools": [tool_spec],
        }
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if include_source_details:
            payload["include"] = ["web_search_call.action.sources"]

        provider_key = self.provider_key or best_effort_provider_key(
            base_url=self.base_url,
            model=self.model,
        )

        def _send_request() -> httpx.Response:
            try:
                with httpx.Client(timeout=self.timeout_s, transport=self._transport) as client:
                    response = client.post(url, headers=headers, json=payload)
            except httpx.DecodingError as e:
                raise ResponsesError(f"Responses response decompression failed: {e}") from e
            except Exception as e:  # noqa: BLE001
                raise ResponsesError(f"Responses request failed: {e}") from e
            if response.status_code >= 400:
                raise self._error_from_response(response)
            return response

        response = run_provider_limited_call(
            call=_send_request,
            provider_key=provider_key,
            provider_concurrency_caps=self.provider_concurrency_caps,
            retry_settings=self.provider_retry_settings,
            operation="responses_web_search",
            sleep_fn=self._provider_sleep_fn,
            random_fn=self._provider_random_fn,
        )

        try:
            data = response.json()
        except Exception as e:  # noqa: BLE001
            raise ResponsesError("Responses API returned non-JSON response") from e

        if not isinstance(data, dict):
            raise ResponsesError("Unexpected Responses API payload: expected JSON object")

        if response.status_code >= 400:
            error_message = _extract_error_message(data)
            if error_message:
                lower = error_message.lower()
                if "unsupported" in lower or "not support" in lower:
                    raise ResponsesError(f"Responses web_search unsupported: {error_message}")
                raise ResponsesError(f"Responses error {response.status_code}: {error_message}")
            raise ResponsesError(f"Responses error {response.status_code}: {data!r}")

        answer = _extract_answer_text(data)
        citations = _extract_citations(data)
        sources, queries = _extract_sources_and_queries(data)
        sources = _merge_citation_sources(sources, citations)
        if not sources:
            raise ResponsesError("Responses web_search did not return sources")

        response_id = str(data.get("id") or "").strip() or None
        response_model = str(data.get("model") or "").strip() or None
        return WebSearchResponse(
            answer=answer,
            citations=citations,
            sources=sources,
            queries=queries,
            raw=data,
            response_id=response_id,
            model=response_model,
        )
