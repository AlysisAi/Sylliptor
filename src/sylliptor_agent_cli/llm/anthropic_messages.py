from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from ..provider_telemetry import ProviderCallTelemetryRecorder
from ..web_search_adapters import ANTHROPIC_MESSAGES_ADAPTER, AUTO_WEB_SEARCH_ADAPTER
from .metadata import ANTHROPIC_MESSAGES_PROVIDER_METADATA_KEY, PROVIDER_METADATA_KEY
from .provider_limits import (
    DEFAULT_PROVIDER_CONCURRENCY_CAPS,
    ProviderRetrySettings,
    best_effort_provider_key,
    run_provider_limited_call,
)
from .streaming import SSEFrame, iter_sse_frames, parse_sse_json_frame
from .types import LLMError, LLMResponse, LLMUsage, ToolCall

_DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_ACCEPT_ENCODING = "identity"
_ANTHROPIC_METADATA_KEY = ANTHROPIC_MESSAGES_PROVIDER_METADATA_KEY
_SYLLIPTOR_WEB_SEARCH_FUNCTION_NAME = "web_search"
_ANTHROPIC_WEB_SEARCH_TOOL_TYPE = "web_search_20260209"
_ANTHROPIC_WEB_SEARCH_TOOL_TYPES = frozenset(
    {
        "web_search_20250305",
        "web_search_20260209",
    }
)
_WEB_SEARCH_MODES_ALLOWING_ANTHROPIC_BUILTIN = frozenset({"auto", "native"})


def _headers_with_default_accept_encoding(headers: dict[str, str]) -> dict[str, str]:
    request_headers = dict(headers)
    if not any(key.lower() == "accept-encoding" for key in request_headers):
        request_headers["Accept-Encoding"] = _DEFAULT_ACCEPT_ENCODING
    return request_headers


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


def _anthropic_blocks_from_content(raw: Any, *, role: str) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [{"type": "text", "text": raw}] if raw else []
    if not isinstance(raw, list):
        text = _content_to_text(raw)
        return [{"type": "text", "text": text}] if text else []

    blocks: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            if item:
                blocks.append({"type": "text", "text": item})
            continue
        if not isinstance(item, dict):
            continue
        block_type = str(item.get("type") or "").strip()
        text = item.get("text") or item.get("content")
        if block_type in {"text", "input_text", "output_text"} and isinstance(text, str):
            blocks.append({"type": "text", "text": text})
            continue
        if role == "user" and block_type == "image_url":
            image_url = item.get("image_url")
            url = ""
            if isinstance(image_url, dict):
                url = str(image_url.get("url") or "").strip()
            elif isinstance(image_url, str):
                url = image_url.strip()
            if url:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
            continue
        if block_type in {
            "text",
            "image",
            "tool_result",
            "tool_use",
            "server_tool_use",
            "web_search_tool_result",
        }:
            blocks.append(copy.deepcopy(item))
    return blocks


def _function_name_from_tool(tool: dict[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "").strip()
    if str(tool.get("type") or "") == "function":
        return str(tool.get("name") or "").strip()
    return ""


def _is_sylliptor_web_search_function(tool: dict[str, Any]) -> bool:
    return _function_name_from_tool(tool) == _SYLLIPTOR_WEB_SEARCH_FUNCTION_NAME


def _is_anthropic_hosted_web_search_tool(tool: dict[str, Any]) -> bool:
    return str(tool.get("type") or "").strip() in _ANTHROPIC_WEB_SEARCH_TOOL_TYPES


def _anthropic_builtin_web_search_allowed(*, mode: str, adapter: str) -> bool:
    normalized_mode = str(mode or "").strip().lower()
    normalized_adapter = str(adapter or "").strip().lower() or AUTO_WEB_SEARCH_ADAPTER
    if normalized_mode not in _WEB_SEARCH_MODES_ALLOWING_ANTHROPIC_BUILTIN:
        return False
    return normalized_adapter in {AUTO_WEB_SEARCH_ADAPTER, ANTHROPIC_MESSAGES_ADAPTER}


@dataclass(frozen=True)
class _AnthropicToolMapping:
    tools: list[dict[str, Any]]
    added_builtin_web_search: bool
    removed_sylliptor_web_search: bool


def _anthropic_tool_from_chat_tool(tool: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(tool, dict):
        return None
    tool_type = str(tool.get("type") or "").strip()
    if tool_type == "function":
        function = tool.get("function")
        source = function if isinstance(function, dict) else tool
        name = str(source.get("name") or "").strip()
        if not name:
            return None
        mapped: dict[str, Any] = {
            "name": name,
            "input_schema": copy.deepcopy(source.get("parameters") or {"type": "object"}),
        }
        description = str(source.get("description") or "").strip()
        if description:
            mapped["description"] = description
        return mapped
    if _is_anthropic_hosted_web_search_tool(tool):
        return copy.deepcopy(tool)
    raise LLMError(f"Anthropic Messages does not support tool type {tool_type!r}")


def _anthropic_tools(
    tools: list[dict[str, Any]] | None,
    *,
    mode: str,
    adapter: str,
) -> _AnthropicToolMapping:
    normalized_mode = str(mode or "off").strip().lower()
    normalized_adapter = (
        str(adapter or AUTO_WEB_SEARCH_ADAPTER).strip().lower() or AUTO_WEB_SEARCH_ADAPTER
    )
    raw_tools = [tool for tool in tools or [] if isinstance(tool, dict)]
    sylliptor_web_search_present = any(
        _is_sylliptor_web_search_function(tool) for tool in raw_tools
    )
    use_builtin_web_search = sylliptor_web_search_present and _anthropic_builtin_web_search_allowed(
        mode=normalized_mode,
        adapter=normalized_adapter,
    )
    if normalized_mode == "native" and sylliptor_web_search_present and not use_builtin_web_search:
        raise LLMError(
            "web_search_mode=native with protocol=anthropic_messages requires "
            "web_search_adapter='auto' or 'anthropic_messages' for Anthropic hosted web_search; "
            f"got {normalized_adapter!r}"
        )

    mapped_tools: list[dict[str, Any]] = []
    removed_sylliptor_web_search = False
    for tool in raw_tools:
        if _is_sylliptor_web_search_function(tool):
            if normalized_mode in {"off", "native"} or use_builtin_web_search:
                removed_sylliptor_web_search = True
                continue
        if _is_anthropic_hosted_web_search_tool(tool) and normalized_mode in {"off", "external"}:
            continue
        mapped = _anthropic_tool_from_chat_tool(tool)
        if mapped is not None:
            mapped_tools.append(mapped)

    if use_builtin_web_search and not any(
        _is_anthropic_hosted_web_search_tool(tool) for tool in mapped_tools
    ):
        mapped_tools.append(
            {
                "type": _ANTHROPIC_WEB_SEARCH_TOOL_TYPE,
                "name": _SYLLIPTOR_WEB_SEARCH_FUNCTION_NAME,
                "max_uses": 5,
            }
        )
    return _AnthropicToolMapping(
        tools=mapped_tools,
        added_builtin_web_search=use_builtin_web_search,
        removed_sylliptor_web_search=removed_sylliptor_web_search,
    )


def _anthropic_tool_choice(
    tool_choice: Any,
    *,
    removed_sylliptor_web_search: bool,
    added_builtin_web_search: bool,
) -> dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        normalized = tool_choice.strip()
        if normalized == "auto":
            return {"type": "auto"}
        if normalized == "none":
            return {"type": "none"}
        if normalized == "required":
            return {"type": "any"}
        raise LLMError(f"Anthropic Messages does not support tool_choice={tool_choice!r}")
    if not isinstance(tool_choice, dict):
        raise LLMError("Anthropic Messages tool_choice must be a string or object")

    choice_type = str(tool_choice.get("type") or "").strip()
    if choice_type == "function":
        if "name" in tool_choice:
            name = str(tool_choice.get("name") or "").strip()
        else:
            function = tool_choice.get("function")
            name = str(function.get("name") or "").strip() if isinstance(function, dict) else ""
        if not name:
            raise LLMError("Anthropic Messages forced function tool_choice is missing name")
        if (
            name == _SYLLIPTOR_WEB_SEARCH_FUNCTION_NAME
            and removed_sylliptor_web_search
            and not added_builtin_web_search
        ):
            raise LLMError(
                "Anthropic Messages removed the Sylliptor web_search function for the selected "
                "web_search_mode; do not force tool_choice to function web_search"
            )
        return {"type": "tool", "name": name}
    if choice_type in {"auto", "any", "none"}:
        return {"type": choice_type}
    if choice_type == "tool":
        name = str(tool_choice.get("name") or "").strip()
        if not name:
            raise LLMError("Anthropic Messages forced tool_choice is missing name")
        return {"type": "tool", "name": name}
    raise LLMError(f"Anthropic Messages does not support tool_choice type {choice_type!r}")


def _metadata_content_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = message.get(PROVIDER_METADATA_KEY)
    if not isinstance(metadata, dict):
        return []
    anthropic_metadata = metadata.get(_ANTHROPIC_METADATA_KEY)
    if not isinstance(anthropic_metadata, dict):
        return []
    blocks = anthropic_metadata.get("content_blocks")
    if not isinstance(blocks, list):
        return []
    copied: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, dict):
            copied.append(copy.deepcopy(block))
    return copied


def _tool_use_blocks_from_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return blocks
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            continue
        call_id = str(raw_tool_call.get("id") or raw_tool_call.get("call_id") or "").strip()
        function = raw_tool_call.get("function")
        if isinstance(function, dict):
            name = str(function.get("name") or "").strip()
            arguments = _json_arguments(function.get("arguments"))
        else:
            name = str(raw_tool_call.get("name") or "").strip()
            arguments = _json_arguments(raw_tool_call.get("arguments"))
        if not name:
            continue
        if not call_id:
            call_id = f"toolu_{name}"
        blocks.append(
            {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": arguments,
            }
        )
    return blocks


def _is_tool_result_user_message(message: dict[str, Any]) -> bool:
    if str(message.get("role") or "") != "user":
        return False
    content = message.get("content")
    return (
        isinstance(content, list)
        and bool(content)
        and all(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)
    )


def _anthropic_messages_from_messages(
    messages: list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    anthropic_messages: list[dict[str, Any]] = []
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
            blocks = _anthropic_blocks_from_content(message.get("content"), role="user")
            if blocks:
                anthropic_messages.append({"role": "user", "content": blocks})
            continue
        if role == "assistant":
            metadata_blocks = _metadata_content_blocks(message)
            if metadata_blocks:
                anthropic_messages.append({"role": "assistant", "content": metadata_blocks})
                continue
            blocks = _anthropic_blocks_from_content(message.get("content"), role="assistant")
            blocks.extend(_tool_use_blocks_from_tool_calls(message))
            if blocks:
                anthropic_messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            tool_use_id = str(message.get("tool_call_id") or message.get("call_id") or "").strip()
            if not tool_use_id:
                raise LLMError("Anthropic Messages tool result is missing tool_call_id")
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": _content_to_text(message.get("content")),
            }
            if anthropic_messages and _is_tool_result_user_message(anthropic_messages[-1]):
                content = anthropic_messages[-1].setdefault("content", [])
                if isinstance(content, list):
                    content.append(tool_result_block)
                else:
                    anthropic_messages.append({"role": "user", "content": [tool_result_block]})
            else:
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [tool_result_block],
                    }
                )
            continue
        raise LLMError(f"Anthropic Messages cannot send message role {role!r}")
    system = "\n\n".join(system_parts).strip() or None
    return system, anthropic_messages


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

    input_tokens = _as_non_negative_int(raw.get("input_tokens"))
    output_tokens = _as_non_negative_int(raw.get("output_tokens"))
    total_tokens = None
    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    cached_tokens = _as_non_negative_int(raw.get("cache_read_input_tokens"))
    return LLMUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_prompt_tokens=cached_tokens,
    )


def _extract_error_message(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    error_obj = data.get("error")
    if isinstance(error_obj, dict):
        message = str(error_obj.get("message") or "").strip()
        if message:
            error_type = str(error_obj.get("type") or "").strip()
            return f"{error_type}: {message}" if error_type else message
    return None


def _text_from_content_blocks(content: list[Any]) -> str:
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if str(block.get("type") or "") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _parse_tool_calls(content: list[Any]) -> list[ToolCall]:
    tool_calls: list[ToolCall] = []
    for index, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        if str(block.get("type") or "") != "tool_use":
            continue
        tool_use_id = str(block.get("id") or f"toolu_{index}").strip()
        name = str(block.get("name") or "").strip()
        if not name:
            continue
        raw_input = block.get("input")
        arguments = (
            dict(raw_input) if isinstance(raw_input, dict) else {"_raw_arguments": raw_input}
        )
        tool_calls.append(
            ToolCall(
                id=tool_use_id,
                name=name,
                arguments=arguments,
                provider_metadata={
                    _ANTHROPIC_METADATA_KEY: {
                        "content_index": index,
                    }
                },
            )
        )
    return tool_calls


def _collect_web_search_metadata(content: list[Any]) -> dict[str, Any]:
    server_tool_uses: list[dict[str, Any]] = []
    web_search_results: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    queries: list[str] = []

    def _append_source(raw: dict[str, Any]) -> None:
        url = str(raw.get("url") or "").strip()
        if not url:
            return
        source: dict[str, Any] = {
            "url": url,
            "title": str(raw.get("title") or "").strip(),
        }
        for key in ("page_age", "encrypted_content", "encrypted_index", "cited_text"):
            value = raw.get(key)
            if value is not None:
                source[key] = value
        sources.append(source)

    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "")
        if block_type == "server_tool_use" and str(block.get("name") or "") == "web_search":
            copied = copy.deepcopy(block)
            server_tool_uses.append(copied)
            raw_input = block.get("input")
            if isinstance(raw_input, dict):
                query = str(raw_input.get("query") or "").strip()
                if query:
                    queries.append(query)
            continue
        if block_type == "web_search_tool_result":
            web_search_results.append(copy.deepcopy(block))
            result_content = block.get("content")
            if isinstance(result_content, list):
                for result in result_content:
                    if (
                        isinstance(result, dict)
                        and str(result.get("type") or "") == "web_search_result"
                    ):
                        web_search_results.append(copy.deepcopy(result))
                        _append_source(result)
            continue
        if block_type != "text":
            continue
        raw_citations = block.get("citations")
        if not isinstance(raw_citations, list):
            continue
        for citation in raw_citations:
            if not isinstance(citation, dict):
                continue
            if str(citation.get("type") or "") != "web_search_result_location":
                continue
            url = str(citation.get("url") or "").strip()
            if not url:
                continue
            citation_payload = {
                "url": url,
                "title": str(citation.get("title") or "").strip(),
                "encrypted_index": citation.get("encrypted_index"),
                "cited_text": citation.get("cited_text"),
            }
            citations.append(citation_payload)
            _append_source(citation)

    if queries:
        queries = list(dict.fromkeys(queries))
    deduped_sources: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for source in sources:
        url = str(source.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        deduped_sources.append(source)
    metadata: dict[str, Any] = {}
    if server_tool_uses:
        metadata["server_tool_uses"] = server_tool_uses
    if web_search_results:
        metadata["web_search_results"] = web_search_results
    if citations:
        metadata["citations"] = citations
    if deduped_sources:
        metadata["sources"] = deduped_sources
    if queries:
        metadata["queries"] = queries
    return metadata


def _anthropic_provider_metadata(data: dict[str, Any]) -> dict[str, Any] | None:
    content = data.get("content")
    metadata: dict[str, Any] = {}
    message_id = str(data.get("id") or "").strip()
    if message_id:
        metadata["message_id"] = message_id
    stop_reason = str(data.get("stop_reason") or "").strip()
    if stop_reason:
        metadata["stop_reason"] = stop_reason
    stop_sequence = data.get("stop_sequence")
    if stop_sequence is not None:
        metadata["stop_sequence"] = stop_sequence
    if isinstance(content, list):
        metadata["content_blocks"] = copy.deepcopy(content)
        metadata.update(_collect_web_search_metadata(content))
    usage = data.get("usage")
    if isinstance(usage, dict):
        metadata["usage"] = copy.deepcopy(usage)
    stream_metadata = data.get("stream_metadata")
    if isinstance(stream_metadata, dict):
        metadata["stream_metadata"] = copy.deepcopy(stream_metadata)
    return {_ANTHROPIC_METADATA_KEY: metadata} if metadata else None


def _response_from_json(data: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=data)


def _event_index(data: dict[str, Any], *, event_type: str) -> int:
    index = data.get("index")
    if isinstance(index, int) and index >= 0:
        return index
    raise LLMError(f"Anthropic Messages stream {event_type} event is missing a valid index")


class _AnthropicStreamAccumulator:
    def __init__(self, *, on_text_delta: Callable[[str], None] | None) -> None:
        self.on_text_delta = on_text_delta
        self.message: dict[str, Any] = {
            "type": "message",
            "role": "assistant",
            "content": [],
        }
        self.content_blocks: dict[int, dict[str, Any]] = {}
        self.input_json_chunks: dict[int, list[str]] = {}
        self.usage: dict[str, Any] = {}
        self.unknown_events: list[dict[str, Any]] = []
        self.event_count = 0
        self.seen_message_start = False
        self.seen_message_stop = False

    def handle(self, frame: SSEFrame, data: dict[str, Any]) -> None:
        event_type = str(data.get("type") or frame.event or "").strip()
        if not event_type:
            self._append_unknown(frame=frame, data=data)
            return
        self.event_count += 1

        if event_type == "message_start":
            self._handle_message_start(data)
            return
        if event_type == "content_block_start":
            self._handle_content_block_start(data)
            return
        if event_type == "content_block_delta":
            self._handle_content_block_delta(data)
            return
        if event_type == "content_block_stop":
            self._handle_content_block_stop(data)
            return
        if event_type == "message_delta":
            self._handle_message_delta(data)
            return
        if event_type == "message_stop":
            self.seen_message_stop = True
            return
        if event_type == "ping":
            return
        if event_type == "error":
            message = _extract_error_message(data)
            raise LLMError(f"Anthropic Messages stream error: {message or data!r}")

        self._append_unknown(frame=frame, data=data)

    def _handle_message_start(self, data: dict[str, Any]) -> None:
        raw_message = data.get("message")
        if not isinstance(raw_message, dict):
            raise LLMError("Anthropic Messages stream message_start missing message object")
        message = copy.deepcopy(raw_message)
        content = message.get("content")
        if not isinstance(content, list):
            message["content"] = []
        else:
            message["content"] = []
        self.message.update(message)
        usage = message.get("usage")
        if isinstance(usage, dict):
            self.usage.update(copy.deepcopy(usage))
        self.seen_message_start = True

    def _handle_content_block_start(self, data: dict[str, Any]) -> None:
        index = _event_index(data, event_type="content_block_start")
        raw_block = data.get("content_block")
        block = copy.deepcopy(raw_block) if isinstance(raw_block, dict) else {}
        if not str(block.get("type") or "").strip():
            block["type"] = "unknown"
        block.setdefault("_stream_index", index)
        self.content_blocks[index] = block
        if str(block.get("type") or "") in {"tool_use", "server_tool_use"}:
            self.input_json_chunks.setdefault(index, [])

    def _handle_content_block_delta(self, data: dict[str, Any]) -> None:
        index = _event_index(data, event_type="content_block_delta")
        block = self.content_blocks.setdefault(index, {"type": "unknown", "_stream_index": index})
        raw_delta = data.get("delta")
        if not isinstance(raw_delta, dict):
            self._append_block_delta(block, raw_delta)
            return
        delta_type = str(raw_delta.get("type") or "").strip()

        if delta_type == "text_delta":
            text = raw_delta.get("text")
            if isinstance(text, str) and text:
                existing = block.get("text")
                block["text"] = (existing if isinstance(existing, str) else "") + text
                if self.on_text_delta is not None:
                    self.on_text_delta(text)
            return

        if delta_type == "input_json_delta":
            partial_json = raw_delta.get("partial_json")
            if isinstance(partial_json, str):
                self.input_json_chunks.setdefault(index, []).append(partial_json)
            return

        if delta_type == "thinking_delta":
            thinking = raw_delta.get("thinking")
            if isinstance(thinking, str) and thinking:
                existing = block.get("thinking")
                block["thinking"] = (existing if isinstance(existing, str) else "") + thinking
            return

        if delta_type == "signature_delta":
            signature = raw_delta.get("signature")
            if isinstance(signature, str) and signature:
                existing = block.get("signature")
                block["signature"] = (existing if isinstance(existing, str) else "") + signature
            return

        if delta_type in {"citations_delta", "citation_delta"}:
            self._append_citation_delta(block, raw_delta)
            return

        self._append_block_delta(block, raw_delta)

    def _handle_content_block_stop(self, data: dict[str, Any]) -> None:
        index = _event_index(data, event_type="content_block_stop")
        block = self.content_blocks.get(index)
        if block is None:
            raise LLMError(
                "Anthropic Messages stream content_block_stop before content_block_start"
            )
        chunks = self.input_json_chunks.get(index)
        if chunks is None:
            return
        joined = "".join(chunks).strip()
        if not joined:
            if "input" not in block:
                block["input"] = {}
            return
        try:
            parsed = json.loads(joined)
        except json.JSONDecodeError as exc:
            raise LLMError(
                "Anthropic Messages stream emitted malformed tool input JSON "
                f"for content block {index}: {exc.msg}"
            ) from exc
        block["input"] = parsed if isinstance(parsed, dict) else {"_raw_arguments": parsed}

    def _handle_message_delta(self, data: dict[str, Any]) -> None:
        delta = data.get("delta")
        if isinstance(delta, dict):
            for key in ("stop_reason", "stop_sequence"):
                if key in delta:
                    self.message[key] = copy.deepcopy(delta[key])
        usage = data.get("usage")
        if isinstance(usage, dict):
            self.usage.update(copy.deepcopy(usage))

    def finish(self) -> dict[str, Any]:
        if not self.seen_message_start:
            raise LLMError("Anthropic Messages stream returned no message_start event")
        if not self.seen_message_stop:
            raise LLMError("Anthropic Messages stream ended before message_stop")
        for index in tuple(self.input_json_chunks):
            self._handle_content_block_stop({"index": index})
        content = [
            self._public_content_block(block)
            for _index, block in sorted(self.content_blocks.items(), key=lambda item: item[0])
        ]
        self.message["content"] = content
        if self.usage:
            self.message["usage"] = copy.deepcopy(self.usage)
        stream_metadata: dict[str, Any] = {"events": self.event_count}
        if self.unknown_events:
            stream_metadata["unknown_events"] = copy.deepcopy(self.unknown_events)
        self.message["stream_metadata"] = stream_metadata
        return copy.deepcopy(self.message)

    @staticmethod
    def _append_citation_delta(block: dict[str, Any], delta: dict[str, Any]) -> None:
        citations = block.setdefault("citations", [])
        if not isinstance(citations, list):
            citations = []
            block["citations"] = citations
        citation = delta.get("citation")
        if isinstance(citation, dict):
            citations.append(copy.deepcopy(citation))
            return
        raw_citations = delta.get("citations")
        if isinstance(raw_citations, list):
            citations.extend(
                copy.deepcopy(item) for item in raw_citations if isinstance(item, dict)
            )

    @staticmethod
    def _append_block_delta(block: dict[str, Any], delta: Any) -> None:
        deltas = block.setdefault("_stream_deltas", [])
        if isinstance(deltas, list):
            deltas.append(copy.deepcopy(delta))

    def _append_unknown(self, *, frame: SSEFrame, data: dict[str, Any]) -> None:
        self.unknown_events.append(
            {
                "event": frame.event,
                "data": copy.deepcopy(data),
            }
        )

    @staticmethod
    def _public_content_block(block: dict[str, Any]) -> dict[str, Any]:
        copied = copy.deepcopy(block)
        copied.pop("_stream_index", None)
        return copied


class AnthropicMessagesClient:
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
        default_max_tokens: int = 4096,
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
        self.default_max_tokens = int(default_max_tokens)

    def _headers(self) -> dict[str, str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": _DEFAULT_ANTHROPIC_VERSION,
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
        if response_format is not None:
            raise LLMError("Anthropic Messages does not support response_format")
        if self.reasoning_effort:
            raise LLMError("Anthropic Messages does not support reasoning_effort")
        if self.enable_thinking:
            raise LLMError("Anthropic Messages does not support enable_thinking yet")

        system, anthropic_messages = _anthropic_messages_from_messages(messages)
        tool_mapping = _anthropic_tools(
            tools,
            mode=self.web_search_mode,
            adapter=self.web_search_adapter,
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": int(max_tokens) if max_tokens is not None else self.default_max_tokens,
            "messages": anthropic_messages,
            "temperature": self.temperature if temperature is None else float(temperature),
        }
        if stream:
            payload["stream"] = True
        if system:
            payload["system"] = system
        if tool_mapping.tools:
            payload["tools"] = tool_mapping.tools
            mapped_tool_choice = _anthropic_tool_choice(
                tool_choice,
                removed_sylliptor_web_search=tool_mapping.removed_sylliptor_web_search,
                added_builtin_web_search=tool_mapping.added_builtin_web_search,
            )
            if mapped_tool_choice is not None:
                payload["tool_choice"] = mapped_tool_choice
        elif tool_choice is not None:
            raise LLMError("Anthropic Messages tool_choice requires at least one available tool")

        provider_key = self.provider_key or best_effort_provider_key(
            base_url=self.base_url,
            model=self.model,
        )
        telemetry = ProviderCallTelemetryRecorder(
            provider_key=provider_key,
            protocol="anthropic_messages",
            model=self.model,
            base_url=self.base_url,
            stream=stream,
            tools=tools,
            web_search_mode=self.web_search_mode,
            web_search_adapter=self.web_search_adapter,
            native_web_search=tool_mapping.added_builtin_web_search,
            operation="anthropic_messages_chat",
        )
        telemetry_on_text_delta = telemetry.wrap_text_delta(on_text_delta)

        def _send_request() -> LLMResponse:
            url = f"{self.base_url}/messages"
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
                                raise self._llm_error_from_response(response)
                            return self._parse_stream_response(
                                response,
                                on_text_delta=telemetry_on_text_delta,
                            )
                    response = client.post(url, headers=self._headers(), json=payload)
            except httpx.DecodingError as e:
                raise LLMError(f"Anthropic Messages decompression failed: {e}") from e
            except Exception as e:  # noqa: BLE001
                if isinstance(e, LLMError):
                    raise
                raise LLMError(f"Anthropic Messages request failed: {e}") from e
            if response.status_code >= 400:
                raise self._llm_error_from_response(response)
            return self._parse_chat_response(response)

        return telemetry.run(
            lambda: run_provider_limited_call(
                call=_send_request,
                provider_key=provider_key,
                provider_concurrency_caps=self.provider_concurrency_caps,
                retry_settings=self.provider_retry_settings,
                operation="anthropic_messages_chat",
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
        accumulator = _AnthropicStreamAccumulator(on_text_delta=on_text_delta)
        for frame in iter_sse_frames(response.iter_lines()):
            raw_event = parse_sse_json_frame(frame, stream_name="Anthropic Messages stream")
            if not isinstance(raw_event, dict):
                raise LLMError("Anthropic Messages stream emitted non-object JSON event")
            accumulator.handle(frame, raw_event)
        data = accumulator.finish()
        return AnthropicMessagesClient._parse_chat_response(response=_response_from_json(data))

    @staticmethod
    def _parse_chat_response(response: httpx.Response) -> LLMResponse:
        try:
            data = response.json()
        except Exception as e:  # noqa: BLE001
            raise LLMError("Anthropic Messages returned non-JSON response") from e
        if not isinstance(data, dict):
            raise LLMError("Unexpected Anthropic Messages payload: expected JSON object")
        content = data.get("content")
        if not isinstance(content, list):
            raise LLMError("Unexpected Anthropic Messages payload: missing content list")

        stop_reason = str(data.get("stop_reason") or "").strip()
        if stop_reason == "refusal":
            raise LLMError("Anthropic Messages refusal")

        for block in content:
            if isinstance(block, dict) and str(block.get("type") or "") == "refusal":
                text = str(block.get("text") or block.get("refusal") or "").strip()
                suffix = f": {text}" if text else ""
                raise LLMError(f"Anthropic Messages refusal{suffix}")

        text = _text_from_content_blocks(content)
        tool_calls = _parse_tool_calls(content)
        if not text and not tool_calls:
            suffix = f" (stop_reason={stop_reason})" if stop_reason else ""
            raise LLMError(f"Anthropic Messages returned no assistant text or tool calls{suffix}")

        response_model = data.get("model") if isinstance(data.get("model"), str) else None
        return LLMResponse(
            content=text,
            tool_calls=tool_calls,
            raw=data,
            response_model=response_model,
            usage=_parse_usage(data.get("usage")),
            provider_metadata=_anthropic_provider_metadata(data),
        )
