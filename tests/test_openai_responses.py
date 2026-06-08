from __future__ import annotations

import json

import httpx
import pytest

from sylliptor_agent_cli.llm.metadata import (
    PROVIDER_METADATA_KEY,
    attach_provider_metadata_to_assistant_message,
    strip_provider_metadata_from_message,
)
from sylliptor_agent_cli.llm.openai_responses import (
    OpenAIResponsesClient,
    ResponsesError,
)
from sylliptor_agent_cli.llm.types import LLMError


def _client(transport: httpx.BaseTransport) -> OpenAIResponsesClient:
    return OpenAIResponsesClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="search-model",
        transport=transport,
    )


def _sse_event(event_type: str, data: dict[str, object]) -> bytes:
    return (f"event: {event_type}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n").encode()


def _sse_response(events: list[tuple[str, dict[str, object]]]) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=b"".join(_sse_event(event, data) for event, data in events),
    )


def test_chat_maps_messages_tools_response_format_and_reasoning() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content.decode("utf-8"))
        captured.update(body)
        return httpx.Response(
            200,
            json={
                "id": "resp_chat_1",
                "model": "gpt-5.5",
                "output_text": "Done.",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Done."}],
                    }
                ],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 3,
                    "total_tokens": 14,
                    "input_tokens_details": {"cached_tokens": 5},
                },
            },
        )

    client = OpenAIResponsesClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5.5",
        temperature=0.4,
        reasoning_effort="low",
        transport=httpx.MockTransport(handler),
    )

    response = client.chat(
        messages=[
            {"role": "system", "content": "System prompt."},
            {"role": "developer", "content": "Developer prompt."},
            {"role": "user", "content": "Read it."},
            {
                "role": "assistant",
                "content": "I will read.",
                "tool_calls": [
                    {
                        "id": "call_read",
                        "type": "function",
                        "function": {
                            "name": "fs_read",
                            "arguments": json.dumps({"path": "README.md"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_read", "content": "README contents"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    {"type": "text", "text": "Describe the image."},
                ],
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "fs_read",
                    "description": "Read a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "fs_read"}},
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "result",
                "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
                "strict": True,
            },
        },
        max_tokens=123,
    )

    assert captured["model"] == "gpt-5.5"
    assert captured["temperature"] == 0.4
    assert captured["reasoning"] == {"effort": "low"}
    assert captured["max_output_tokens"] == 123
    assert captured["input"] == [
        {"role": "system", "content": "System prompt."},
        {"role": "developer", "content": "Developer prompt."},
        {"role": "user", "content": "Read it."},
        {"role": "assistant", "content": "I will read."},
        {
            "type": "function_call",
            "call_id": "call_read",
            "name": "fs_read",
            "arguments": '{"path":"README.md"}',
        },
        {"type": "function_call_output", "call_id": "call_read", "output": "README contents"},
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,abc"},
                {"type": "input_text", "text": "Describe the image."},
            ],
        },
    ]
    assert captured["tools"] == [
        {
            "type": "function",
            "name": "fs_read",
            "description": "Read a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]
    assert captured["tool_choice"] == {"type": "function", "name": "fs_read"}
    assert captured["text"] == {
        "format": {
            "type": "json_schema",
            "name": "result",
            "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
            "strict": True,
        }
    }
    assert response.content == "Done."
    assert response.usage is not None
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 3
    assert response.usage.total_tokens == 14
    assert response.usage.cached_prompt_tokens == 5


def test_chat_parses_function_calls_and_usage() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "resp_tool",
                "model": "gpt-5.5",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_shell",
                        "name": "shell_run",
                        "arguments": '{"cmd":"pytest -q"}',
                        "status": "completed",
                    }
                ],
                "usage": {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
            },
        )

    response = _client(httpx.MockTransport(handler)).chat(
        messages=[{"role": "user", "content": "Run tests."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "shell_run",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    assert response.content == ""
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_shell"
    assert response.tool_calls[0].name == "shell_run"
    assert response.tool_calls[0].arguments == {"cmd": "pytest -q"}
    assert response.usage is not None
    assert response.usage.prompt_tokens == 8
    assert response.provider_metadata is not None
    assert response.provider_metadata["openai_responses"]["response_id"] == "resp_tool"
    assert response.provider_metadata["openai_responses"]["output_items"][0]["id"] == "fc_1"


def _web_search_function_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Standalone Sylliptor web search.",
            "parameters": {"type": "object"},
        },
    }


def test_chat_native_mode_uses_only_openai_hosted_web_search() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.update(body)
        return httpx.Response(
            200,
            json={
                "id": "resp_search_chat",
                "model": "gpt-5.5",
                "output_text": "Use the cited docs.",
                "output": [
                    {
                        "type": "web_search_call",
                        "id": "ws_1",
                        "status": "completed",
                        "action": {
                            "type": "search",
                            "query": "Sylliptor docs",
                            "sources": [
                                {"title": "Docs", "url": "https://docs.example.com/sylliptor"}
                            ],
                        },
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Use the cited docs.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "title": "Docs",
                                        "url": "https://docs.example.com/sylliptor",
                                        "start_index": 8,
                                        "end_index": 18,
                                    }
                                ],
                            }
                        ],
                    },
                ],
            },
        )

    client = OpenAIResponsesClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5.5",
        web_search_mode="native",
        web_search_adapter="openai_responses",
        transport=httpx.MockTransport(handler),
    )

    response = client.chat(
        messages=[{"role": "user", "content": "Find current docs."}],
        tools=[_web_search_function_tool()],
    )

    assert captured["tools"] == [
        {"type": "web_search", "external_web_access": True},
    ]
    assert captured["include"] == ["web_search_call.action.sources"]
    assert response.content == "Use the cited docs."
    assert response.provider_metadata is not None
    metadata = response.provider_metadata["openai_responses"]
    assert metadata["citations"][0]["url"] == "https://docs.example.com/sylliptor"
    assert metadata["sources"][0]["url"] == "https://docs.example.com/sylliptor"
    assert metadata["queries"] == ["Sylliptor docs"]


def test_chat_auto_mode_prefers_openai_hosted_web_search_for_auto_adapter() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.update(body)
        return httpx.Response(
            200,
            json={"id": "resp_no_search", "output_text": "ok", "output": []},
        )

    client = OpenAIResponsesClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5.5",
        web_search_mode="auto",
        web_search_adapter="auto",
        transport=httpx.MockTransport(handler),
    )

    response = client.chat(
        messages=[{"role": "user", "content": "hello"}],
        tools=[_web_search_function_tool()],
    )

    assert captured["tools"] == [{"type": "web_search", "external_web_access": True}]
    assert captured["include"] == ["web_search_call.action.sources"]
    assert response.content == "ok"


def test_chat_external_mode_uses_only_sylliptor_web_search_function() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.update(body)
        return httpx.Response(
            200,
            json={"id": "resp_external_search", "output_text": "ok", "output": []},
        )

    client = OpenAIResponsesClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5.5",
        web_search_mode="external",
        web_search_adapter="tavily",
        transport=httpx.MockTransport(handler),
    )

    response = client.chat(
        messages=[{"role": "user", "content": "hello"}],
        tools=[_web_search_function_tool()],
    )

    assert captured["tools"] == [
        {
            "type": "function",
            "name": "web_search",
            "description": "Standalone Sylliptor web search.",
            "parameters": {"type": "object"},
        }
    ]
    assert "include" not in captured
    assert response.content == "ok"


def test_chat_off_mode_removes_all_web_search_tools() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.update(body)
        return httpx.Response(
            200,
            json={"id": "resp_no_search", "output_text": "ok", "output": []},
        )

    client = OpenAIResponsesClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5.5",
        web_search_mode="off",
        web_search_adapter="openai_responses",
        transport=httpx.MockTransport(handler),
    )

    response = client.chat(
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            _web_search_function_tool(),
            {"type": "web_search", "external_web_access": True},
        ],
    )

    assert "tools" not in captured
    assert "include" not in captured
    assert response.content == "ok"


def test_chat_auto_mode_with_external_adapter_uses_sylliptor_web_search_function() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.update(body)
        return httpx.Response(
            200,
            json={"id": "resp_auto_external", "output_text": "ok", "output": []},
        )

    client = OpenAIResponsesClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5.5",
        web_search_mode="auto",
        web_search_adapter="tavily",
        transport=httpx.MockTransport(handler),
    )

    response = client.chat(
        messages=[{"role": "user", "content": "hello"}],
        tools=[_web_search_function_tool()],
    )

    assert captured["tools"] == [
        {
            "type": "function",
            "name": "web_search",
            "description": "Standalone Sylliptor web search.",
            "parameters": {"type": "object"},
        }
    ]
    assert "include" not in captured
    assert response.content == "ok"


def test_chat_native_mode_rejects_non_openai_search_adapter() -> None:
    client = OpenAIResponsesClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5.5",
        web_search_mode="native",
        web_search_adapter="tavily",
        transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
    )

    with pytest.raises(LLMError, match="web_search_mode=native.*openai_responses"):
        client.chat(
            messages=[{"role": "user", "content": "hello"}],
            tools=[_web_search_function_tool()],
        )


def test_chat_rejects_forced_removed_web_search_tool_choice() -> None:
    client = OpenAIResponsesClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5.5",
        web_search_mode="native",
        web_search_adapter="openai_responses",
        transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
    )

    with pytest.raises(LLMError, match="removed the Sylliptor web_search function"):
        client.chat(
            messages=[{"role": "user", "content": "hello"}],
            tools=[_web_search_function_tool()],
            tool_choice={"type": "function", "function": {"name": "web_search"}},
        )


def test_chat_surfaces_provider_errors() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad responses request"}})

    with pytest.raises(LLMError, match="LLM error 400: bad responses request"):
        _client(httpx.MockTransport(handler)).chat(
            messages=[{"role": "user", "content": "hello"}],
        )


def test_chat_rejects_refusal_or_empty_output() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "resp_refusal",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "refusal", "text": "I cannot help with that."}],
                    }
                ],
            },
        )

    with pytest.raises(LLMError, match="OpenAI Responses refusal: I cannot help"):
        _client(httpx.MockTransport(handler)).chat(
            messages=[{"role": "user", "content": "hello"}],
        )


def test_chat_streaming_emits_text_deltas_and_preserves_metadata() -> None:
    captured: dict[str, object] = {}
    deltas: list[str] = []
    message_item = {
        "type": "message",
        "id": "msg_stream",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Hello stream."}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.update(body)
        return _sse_response(
            [
                (
                    "response.created",
                    {
                        "type": "response.created",
                        "response": {
                            "id": "resp_stream",
                            "model": "gpt-5.5",
                            "status": "in_progress",
                            "output": [],
                        },
                    },
                ),
                (
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "response_id": "resp_stream",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": "msg_stream",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                ),
                (
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "response_id": "resp_stream",
                        "item_id": "msg_stream",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": "Hello ",
                    },
                ),
                (
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "response_id": "resp_stream",
                        "item_id": "msg_stream",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": "stream.",
                    },
                ),
                (
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "response_id": "resp_stream",
                        "item_id": "msg_stream",
                        "output_index": 0,
                        "content_index": 0,
                        "text": "Hello stream.",
                    },
                ),
                (
                    "response.reasoning_summary.delta",
                    {
                        "type": "response.reasoning_summary.delta",
                        "response_id": "resp_stream",
                        "delta": "hidden",
                    },
                ),
                (
                    "response.completed",
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp_stream",
                            "model": "gpt-5.5",
                            "status": "completed",
                            "output_text": "Hello stream.",
                            "output": [message_item],
                            "usage": {
                                "input_tokens": 5,
                                "output_tokens": 3,
                                "total_tokens": 8,
                            },
                        },
                    },
                ),
            ]
        )

    response = _client(httpx.MockTransport(handler)).chat(
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        on_text_delta=deltas.append,
    )

    assert captured["stream"] is True
    assert response.content == "Hello stream."
    assert deltas == ["Hello ", "stream."]
    assert response.usage is not None
    assert response.usage.total_tokens == 8
    assert response.provider_metadata is not None
    metadata = response.provider_metadata["openai_responses"]
    assert metadata["response_id"] == "resp_stream"
    assert metadata["output_items"] == [message_item]
    assert metadata["stream_metadata"]["events"] == 7
    assert metadata["stream_metadata"]["unknown_events"][0]["event"] == (
        "response.reasoning_summary.delta"
    )


def test_chat_streaming_parses_function_call_argument_deltas() -> None:
    function_item = {
        "type": "function_call",
        "id": "fc_stream",
        "call_id": "call_stream",
        "name": "fs_read",
        "arguments": '{"path":"README.md"}',
        "status": "completed",
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                (
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "response_id": "resp_tool_stream",
                        "output_index": 0,
                        "item": {
                            "type": "function_call",
                            "id": "fc_stream",
                            "call_id": "call_stream",
                            "name": "fs_read",
                            "arguments": "",
                        },
                    },
                ),
                (
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "response_id": "resp_tool_stream",
                        "item_id": "fc_stream",
                        "output_index": 0,
                        "delta": '{"path":',
                    },
                ),
                (
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "response_id": "resp_tool_stream",
                        "item_id": "fc_stream",
                        "output_index": 0,
                        "delta": '"README.md"}',
                    },
                ),
                (
                    "response.function_call_arguments.done",
                    {
                        "type": "response.function_call_arguments.done",
                        "response_id": "resp_tool_stream",
                        "item_id": "fc_stream",
                        "output_index": 0,
                        "call_id": "call_stream",
                        "name": "fs_read",
                        "arguments": '{"path":"README.md"}',
                    },
                ),
                (
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "response_id": "resp_tool_stream",
                        "output_index": 0,
                        "item": function_item,
                    },
                ),
                (
                    "response.completed",
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp_tool_stream",
                            "model": "gpt-5.5",
                            "status": "completed",
                            "output": [function_item],
                        },
                    },
                ),
            ]
        )

    response = _client(httpx.MockTransport(handler)).chat(
        messages=[{"role": "user", "content": "read"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "fs_read", "parameters": {"type": "object"}},
            }
        ],
        stream=True,
    )

    assert response.content == ""
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_stream"
    assert response.tool_calls[0].name == "fs_read"
    assert response.tool_calls[0].arguments == {"path": "README.md"}
    assert response.provider_metadata is not None
    assert response.provider_metadata["openai_responses"]["output_items"] == [function_item]


def test_chat_streaming_preserves_hosted_web_search_citations_and_sources() -> None:
    web_search_item = {
        "type": "web_search_call",
        "id": "ws_stream",
        "status": "completed",
        "action": {
            "type": "search",
            "query": "current docs",
            "sources": [{"title": "Docs", "url": "https://docs.example.com/current"}],
        },
    }
    message_item = {
        "type": "message",
        "id": "msg_search_stream",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "Use current docs.",
                "annotations": [
                    {
                        "type": "url_citation",
                        "title": "Docs",
                        "url": "https://docs.example.com/current",
                        "start_index": 4,
                        "end_index": 16,
                    }
                ],
            }
        ],
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                (
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "response_id": "resp_search_stream",
                        "output_index": 0,
                        "item": {
                            "type": "web_search_call",
                            "id": "ws_stream",
                            "status": "in_progress",
                        },
                    },
                ),
                (
                    "response.web_search_call.searching",
                    {
                        "type": "response.web_search_call.searching",
                        "response_id": "resp_search_stream",
                        "item_id": "ws_stream",
                        "output_index": 0,
                    },
                ),
                (
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "response_id": "resp_search_stream",
                        "output_index": 0,
                        "item": web_search_item,
                    },
                ),
                (
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "response_id": "resp_search_stream",
                        "item_id": "msg_search_stream",
                        "output_index": 1,
                        "content_index": 0,
                        "delta": "Use current docs.",
                    },
                ),
                (
                    "response.output_text.annotation.added",
                    {
                        "type": "response.output_text.annotation.added",
                        "response_id": "resp_search_stream",
                        "item_id": "msg_search_stream",
                        "output_index": 1,
                        "content_index": 0,
                        "annotation_index": 0,
                        "annotation": message_item["content"][0]["annotations"][0],
                    },
                ),
                (
                    "response.completed",
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp_search_stream",
                            "model": "gpt-5.5",
                            "status": "completed",
                            "output_text": "Use current docs.",
                            "output": [web_search_item, message_item],
                        },
                    },
                ),
            ]
        )

    response = OpenAIResponsesClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5.5",
        web_search_mode="native",
        web_search_adapter="openai_responses",
        transport=httpx.MockTransport(handler),
    ).chat(
        messages=[{"role": "user", "content": "search"}],
        tools=[_web_search_function_tool()],
        stream=True,
    )

    assert response.content == "Use current docs."
    assert response.provider_metadata is not None
    metadata = response.provider_metadata["openai_responses"]
    assert metadata["web_search_calls"] == [web_search_item]
    assert metadata["citations"][0]["url"] == "https://docs.example.com/current"
    assert metadata["sources"][0]["url"] == "https://docs.example.com/current"
    assert metadata["queries"] == ["current docs"]


def test_chat_streaming_error_event_is_clear() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                (
                    "error",
                    {
                        "type": "error",
                        "error": {"message": "stream exploded"},
                    },
                )
            ]
        )

    with pytest.raises(LLMError, match="OpenAI Responses stream error: stream exploded"):
        _client(httpx.MockTransport(handler)).chat(
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )


def test_chat_streaming_malformed_event_is_clear() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"event: response.output_text.delta\ndata: {bad-json}\n\n",
        )

    with pytest.raises(LLMError, match="OpenAI Responses stream emitted malformed JSON"):
        _client(httpx.MockTransport(handler)).chat(
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )


def test_chat_streaming_early_termination_is_clear() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            [
                (
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "response_id": "resp_short",
                        "item_id": "msg_short",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": "partial",
                    },
                )
            ]
        )

    with pytest.raises(LLMError, match="ended before response.completed"):
        _client(httpx.MockTransport(handler)).chat(
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )


def test_chat_text_only_response_preserves_provider_metadata_for_next_turn() -> None:
    calls: list[dict[str, object]] = []
    output_items = [
        {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "First answer."}],
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "resp_text",
                    "model": "gpt-5.5",
                    "output_text": "First answer.",
                    "output": output_items,
                },
            )
        return httpx.Response(
            200,
            json={"id": "resp_second", "output_text": "Second answer.", "output": []},
        )

    client = OpenAIResponsesClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5.5",
        transport=httpx.MockTransport(handler),
    )
    first = client.chat(messages=[{"role": "user", "content": "first"}])
    assistant_message = attach_provider_metadata_to_assistant_message(
        {"role": "assistant", "content": first.content},
        first,
    )

    assert PROVIDER_METADATA_KEY in assistant_message
    assert strip_provider_metadata_from_message(assistant_message) == {
        "role": "assistant",
        "content": "First answer.",
    }

    second = client.chat(
        messages=[
            {"role": "user", "content": "first"},
            assistant_message,
            {"role": "user", "content": "second"},
        ],
    )

    assert second.content == "Second answer."
    assert calls[1]["input"] == [
        {"role": "user", "content": "first"},
        output_items[0],
        {"role": "user", "content": "second"},
    ]


def test_chat_hosted_web_search_output_round_trips_from_provider_metadata() -> None:
    calls: list[dict[str, object]] = []
    output_items = [
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {
                "type": "search",
                "query": "current docs",
                "sources": [{"title": "Docs", "url": "https://docs.example.com/current"}],
            },
        },
        {
            "type": "message",
            "id": "msg_search",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Use current docs."}],
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "resp_search",
                    "model": "gpt-5.5",
                    "output_text": "Use current docs.",
                    "output": output_items,
                },
            )
        return httpx.Response(
            200,
            json={"id": "resp_after_search", "output_text": "ok", "output": []},
        )

    client = OpenAIResponsesClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5.5",
        transport=httpx.MockTransport(handler),
    )
    first = client.chat(messages=[{"role": "user", "content": "search"}])
    assistant_message = attach_provider_metadata_to_assistant_message(
        {"role": "assistant", "content": first.content},
        first,
    )

    assert PROVIDER_METADATA_KEY in assistant_message
    assert first.provider_metadata is not None
    assert first.provider_metadata["openai_responses"]["web_search_calls"][0]["id"] == "ws_1"

    client.chat(
        messages=[
            {"role": "user", "content": "search"},
            assistant_message,
            {"role": "user", "content": "continue"},
        ],
    )

    assert calls[1]["input"] == [
        {"role": "user", "content": "search"},
        output_items[0],
        output_items[1],
        {"role": "user", "content": "continue"},
    ]


def test_chat_function_call_and_output_round_trip_uses_exact_call_id() -> None:
    calls: list[dict[str, object]] = []
    function_call_item = {
        "type": "function_call",
        "id": "fc_exact",
        "call_id": "call_exact_123",
        "name": "fs_read",
        "arguments": '{"path":"README.md"}',
        "status": "completed",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "resp_call",
                    "model": "gpt-5.5",
                    "output": [function_call_item],
                },
            )
        return httpx.Response(
            200,
            json={"id": "resp_done", "output_text": "Read it.", "output": []},
        )

    client = OpenAIResponsesClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-5.5",
        transport=httpx.MockTransport(handler),
    )
    first = client.chat(
        messages=[{"role": "user", "content": "read"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "fs_read", "parameters": {"type": "object"}},
            }
        ],
    )
    assistant_message = attach_provider_metadata_to_assistant_message(
        {
            "role": "assistant",
            "content": first.content,
            "tool_calls": [
                {
                    "id": first.tool_calls[0].id,
                    "type": "function",
                    "function": {
                        "name": first.tool_calls[0].name,
                        "arguments": json.dumps(first.tool_calls[0].arguments),
                    },
                }
            ],
        },
        first,
    )

    client.chat(
        messages=[
            {"role": "user", "content": "read"},
            assistant_message,
            {"role": "tool", "tool_call_id": "call_exact_123", "content": "file contents"},
        ],
    )

    assert calls[1]["input"] == [
        {"role": "user", "content": "read"},
        function_call_item,
        {
            "type": "function_call_output",
            "call_id": "call_exact_123",
            "output": "file contents",
        },
    ]


def test_chat_empty_output_without_refusal_is_explicit() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "resp_empty", "status": "completed", "output": []})

    with pytest.raises(
        LLMError,
        match="OpenAI Responses returned no assistant text or tool calls",
    ):
        _client(httpx.MockTransport(handler)).chat(
            messages=[{"role": "user", "content": "hello"}],
        )


def test_web_search_parses_output_text_citations_and_sources() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "search-model"
        assert body["input"] == "latest httpx release notes"
        assert body["tool_choice"] == "required"
        assert body["include"] == ["web_search_call.action.sources"]
        assert body["tools"] == [
            {
                "type": "web_search",
                "filters": {"allowed_domains": ["github.com", "python.org"]},
                "external_web_access": False,
            }
        ]
        return httpx.Response(
            200,
            json={
                "id": "resp_123",
                "model": "search-model",
                "output_text": "Use the official release notes page.",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Use the official release notes page.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "title": "httpx Releases",
                                        "url": "https://github.com/encode/httpx/releases",
                                        "start_index": 8,
                                        "end_index": 29,
                                    }
                                ],
                            }
                        ],
                    },
                    {
                        "type": "web_search_call",
                        "action": {
                            "queries": ["latest httpx release notes"],
                            "sources": [
                                {
                                    "title": "httpx Releases",
                                    "url": "https://github.com/encode/httpx/releases",
                                },
                                {
                                    "title": "httpx Docs",
                                    "url": "https://www.python-httpx.org/",
                                },
                            ],
                        },
                    },
                ],
            },
        )

    response = _client(httpx.MockTransport(handler)).web_search(
        query="latest httpx release notes",
        allowed_domains=["github.com", "python.org"],
        external_web_access=False,
    )

    assert response.response_id == "resp_123"
    assert response.model == "search-model"
    assert response.answer == "Use the official release notes page."
    assert response.queries == ["latest httpx release notes"]
    assert len(response.citations) == 1
    assert response.citations[0].title == "httpx Releases"
    assert response.citations[0].url == "https://github.com/encode/httpx/releases"
    assert response.citations[0].start_index == 8
    assert response.citations[0].end_index == 29
    assert [source.url for source in response.sources] == [
        "https://github.com/encode/httpx/releases",
        "https://www.python-httpx.org/",
    ]


def test_web_search_falls_back_to_assistant_message_text_when_output_text_missing() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "resp_124",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Read the changelog."},
                            {"type": "output_text", "text": " Then fetch the API docs."},
                        ],
                    },
                    {
                        "type": "web_search_call",
                        "action": {
                            "sources": [
                                {
                                    "title": "Changelog",
                                    "url": "https://github.com/encode/httpx/releases",
                                }
                            ]
                        },
                    },
                ],
            },
        )

    response = _client(httpx.MockTransport(handler)).web_search(query="httpx changelog")

    assert response.answer == "Read the changelog. Then fetch the API docs."
    assert response.citations == []
    assert [source.url for source in response.sources] == [
        "https://github.com/encode/httpx/releases"
    ]


def test_web_search_can_omit_openai_source_include_for_provider_compatibility() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert "include" not in body
        assert body["tool_choice"] == "required"
        assert body["tools"] == [{"type": "web_search"}]
        return httpx.Response(
            200,
            json={
                "id": "resp_xai",
                "model": "grok-4",
                "output_text": "Use xAI citations.",
                "citations": [
                    {
                        "title": "xAI Web Search",
                        "url": "https://docs.x.ai/developers/tools/web-search",
                        "start_index": 0,
                        "end_index": 7,
                    }
                ],
            },
        )

    response = _client(httpx.MockTransport(handler)).web_search(
        query="xAI search docs",
        include_source_details=False,
    )

    assert response.answer == "Use xAI citations."
    assert response.citations[0].title == "xAI Web Search"
    assert response.citations[0].url == "https://docs.x.ai/developers/tools/web-search"
    assert response.citations[0].start_index == 0
    assert response.citations[0].end_index == 7
    assert response.sources[0].url == "https://docs.x.ai/developers/tools/web-search"


def test_web_search_http_error_surfaces_provider_message() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "This provider does not support web_search."}},
        )

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(ResponsesError, match="Responses web_search unsupported"):
        client.web_search(query="httpx docs")


def test_web_search_non_json_response_is_explicit() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(ResponsesError, match="non-JSON"):
        client.web_search(query="httpx docs")


def test_web_search_decompression_error_is_explicit() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip", "content-type": "application/json"},
            content=b'{"output_text":"ok","sources":[{"url":"https://example.com"}]}',
        )

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(ResponsesError, match="response decompression failed"):
        client.web_search(query="httpx docs")


def test_web_search_rejects_unsupported_response_shape() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "resp_125", "model": "search-model", "output": []})

    client = _client(httpx.MockTransport(handler))
    with pytest.raises(ResponsesError, match="did not return sources"):
        client.web_search(query="httpx docs")
