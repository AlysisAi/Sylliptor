from __future__ import annotations

import json

import httpx
import pytest

from sylliptor_agent_cli.llm.anthropic_messages import AnthropicMessagesClient
from sylliptor_agent_cli.llm.metadata import (
    PROVIDER_METADATA_KEY,
    attach_provider_metadata_to_assistant_message,
    strip_provider_metadata_from_message,
)
from sylliptor_agent_cli.llm.types import LLMError


def _client(transport: httpx.BaseTransport) -> AnthropicMessagesClient:
    return AnthropicMessagesClient(
        base_url="https://api.anthropic.com/v1",
        api_key="test-key",
        model="claude-sonnet-4-6",
        transport=transport,
    )


def _web_search_function_tool() -> dict[str, object]:
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Standalone Sylliptor web search.",
            "parameters": {"type": "object"},
        },
    }


def _sse_event(event: str, data: dict[str, object]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _sse_response(*events: str) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=httpx.ByteStream("".join(events).encode("utf-8")),
    )


def test_chat_maps_messages_tools_tool_choice_and_usage() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == "test-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        body = json.loads(request.content.decode("utf-8"))
        captured.update(body)
        return httpx.Response(
            200,
            json={
                "id": "msg_text",
                "model": "claude-sonnet-4-6",
                "role": "assistant",
                "type": "message",
                "content": [{"type": "text", "text": "Done."}],
                "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 4,
                    "cache_read_input_tokens": 3,
                },
            },
        )

    client = AnthropicMessagesClient(
        base_url="https://api.anthropic.com/v1",
        api_key="test-key",
        model="claude-sonnet-4-6",
        temperature=0.4,
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
                        "id": "toolu_read",
                        "type": "function",
                        "function": {
                            "name": "fs_read",
                            "arguments": json.dumps({"path": "README.md"}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "toolu_read", "content": "README contents"},
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
        max_tokens=123,
    )

    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["temperature"] == 0.4
    assert captured["max_tokens"] == 123
    assert captured["system"] == "System prompt.\n\nDeveloper prompt."
    assert captured["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Read it."}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I will read."},
                {
                    "type": "tool_use",
                    "id": "toolu_read",
                    "name": "fs_read",
                    "input": {"path": "README.md"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_read",
                    "content": "README contents",
                }
            ],
        },
    ]
    assert captured["tools"] == [
        {
            "name": "fs_read",
            "description": "Read a file.",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
    ]
    assert captured["tool_choice"] == {"type": "tool", "name": "fs_read"}
    assert response.content == "Done."
    assert response.usage is not None
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 4
    assert response.usage.total_tokens == 16
    assert response.usage.cached_prompt_tokens == 3


def test_chat_parses_multiple_tool_use_blocks() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_tools",
                "model": "claude-sonnet-4-6",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "fs_read",
                        "input": {"path": "README.md"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "shell_run",
                        "input": {"cmd": "pytest -q"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 7, "output_tokens": 5},
            },
        )

    response = _client(httpx.MockTransport(handler)).chat(
        messages=[{"role": "user", "content": "Inspect."}],
        tools=[
            {"type": "function", "function": {"name": "fs_read", "parameters": {"type": "object"}}},
            {
                "type": "function",
                "function": {"name": "shell_run", "parameters": {"type": "object"}},
            },
        ],
    )

    assert response.content == ""
    assert [(tc.id, tc.name, tc.arguments) for tc in response.tool_calls] == [
        ("toolu_1", "fs_read", {"path": "README.md"}),
        ("toolu_2", "shell_run", {"cmd": "pytest -q"}),
    ]
    assert response.provider_metadata is not None
    assert response.provider_metadata["anthropic_messages"]["stop_reason"] == "tool_use"


def test_chat_function_tool_use_and_tool_result_round_trip_exact_id() -> None:
    calls: list[dict[str, object]] = []
    tool_use_block = {
        "type": "tool_use",
        "id": "toolu_exact_123",
        "name": "fs_read",
        "input": {"path": "README.md"},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "msg_tool_roundtrip",
                    "model": "claude-sonnet-4-6",
                    "content": [tool_use_block],
                    "stop_reason": "tool_use",
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "msg_done",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "Read it."}],
                "stop_reason": "end_turn",
            },
        )

    client = _client(httpx.MockTransport(handler))
    first = client.chat(
        messages=[{"role": "user", "content": "read"}],
        tools=[
            {"type": "function", "function": {"name": "fs_read", "parameters": {"type": "object"}}}
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
            {"role": "tool", "tool_call_id": "toolu_exact_123", "content": "file contents"},
        ],
    )

    assert calls[1]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "read"}]},
        {"role": "assistant", "content": [tool_use_block]},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_exact_123",
                    "content": "file contents",
                }
            ],
        },
    ]


def test_chat_text_only_response_preserves_provider_metadata_for_next_turn() -> None:
    calls: list[dict[str, object]] = []
    content_blocks = [{"type": "text", "text": "First answer."}]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "msg_text_metadata",
                    "model": "claude-sonnet-4-6",
                    "content": content_blocks,
                    "stop_reason": "end_turn",
                },
            )
        return httpx.Response(
            200,
            json={
                "id": "msg_second",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "Second answer."}],
                "stop_reason": "end_turn",
            },
        )

    client = _client(httpx.MockTransport(handler))
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

    client.chat(
        messages=[
            {"role": "user", "content": "first"},
            assistant_message,
            {"role": "user", "content": "second"},
        ],
    )

    assert calls[1]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "first"}]},
        {"role": "assistant", "content": content_blocks},
        {"role": "user", "content": [{"type": "text", "text": "second"}]},
    ]


def test_chat_native_mode_uses_only_anthropic_hosted_web_search() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.update(body)
        return httpx.Response(
            200,
            json={
                "id": "msg_search",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
            },
        )

    client = AnthropicMessagesClient(
        base_url="https://api.anthropic.com/v1",
        api_key="test-key",
        model="claude-sonnet-4-6",
        web_search_mode="native",
        web_search_adapter="anthropic_messages",
        transport=httpx.MockTransport(handler),
    )
    response = client.chat(
        messages=[{"role": "user", "content": "find docs"}],
        tools=[_web_search_function_tool()],
    )

    assert captured["tools"] == [
        {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}
    ]
    assert response.content == "ok"


def test_chat_auto_mode_prefers_anthropic_hosted_web_search_for_auto_adapter() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.update(body)
        return httpx.Response(
            200,
            json={"id": "msg_auto", "content": [{"type": "text", "text": "ok"}]},
        )

    client = AnthropicMessagesClient(
        base_url="https://api.anthropic.com/v1",
        api_key="test-key",
        model="claude-sonnet-4-6",
        web_search_mode="auto",
        web_search_adapter="auto",
        transport=httpx.MockTransport(handler),
    )
    client.chat(
        messages=[{"role": "user", "content": "find docs"}], tools=[_web_search_function_tool()]
    )

    assert captured["tools"] == [
        {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}
    ]


def test_chat_external_mode_uses_only_sylliptor_web_search_function() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.update(body)
        return httpx.Response(
            200,
            json={"id": "msg_external", "content": [{"type": "text", "text": "ok"}]},
        )

    client = AnthropicMessagesClient(
        base_url="https://api.anthropic.com/v1",
        api_key="test-key",
        model="claude-sonnet-4-6",
        web_search_mode="external",
        web_search_adapter="tavily",
        transport=httpx.MockTransport(handler),
    )
    client.chat(
        messages=[{"role": "user", "content": "find docs"}], tools=[_web_search_function_tool()]
    )

    assert captured["tools"] == [
        {
            "name": "web_search",
            "description": "Standalone Sylliptor web search.",
            "input_schema": {"type": "object"},
        }
    ]


def test_chat_off_mode_removes_all_web_search_tools() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.update(body)
        return httpx.Response(
            200, json={"id": "msg_off", "content": [{"type": "text", "text": "ok"}]}
        )

    client = AnthropicMessagesClient(
        base_url="https://api.anthropic.com/v1",
        api_key="test-key",
        model="claude-sonnet-4-6",
        web_search_mode="off",
        web_search_adapter="anthropic_messages",
        transport=httpx.MockTransport(handler),
    )
    client.chat(
        messages=[{"role": "user", "content": "find docs"}],
        tools=[_web_search_function_tool(), {"type": "web_search_20260209", "name": "web_search"}],
    )

    assert "tools" not in captured


def test_chat_auto_mode_with_external_adapter_uses_sylliptor_web_search_function() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.update(body)
        return httpx.Response(
            200,
            json={"id": "msg_auto_external", "content": [{"type": "text", "text": "ok"}]},
        )

    client = AnthropicMessagesClient(
        base_url="https://api.anthropic.com/v1",
        api_key="test-key",
        model="claude-sonnet-4-6",
        web_search_mode="auto",
        web_search_adapter="tavily",
        transport=httpx.MockTransport(handler),
    )
    client.chat(
        messages=[{"role": "user", "content": "find docs"}], tools=[_web_search_function_tool()]
    )

    assert captured["tools"][0]["name"] == "web_search"
    assert "type" not in captured["tools"][0]


def test_chat_native_mode_rejects_non_anthropic_search_adapter() -> None:
    client = AnthropicMessagesClient(
        base_url="https://api.anthropic.com/v1",
        api_key="test-key",
        model="claude-sonnet-4-6",
        web_search_mode="native",
        web_search_adapter="tavily",
        transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
    )

    with pytest.raises(LLMError, match="web_search_mode=native.*anthropic_messages"):
        client.chat(
            messages=[{"role": "user", "content": "find docs"}],
            tools=[_web_search_function_tool()],
        )


def test_chat_extracts_web_search_citations_sources_and_queries_metadata() -> None:
    content_blocks = [
        {
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": {"query": "anthropic docs"},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [
                {
                    "type": "web_search_result",
                    "title": "Anthropic Docs",
                    "url": "https://docs.anthropic.com/",
                    "encrypted_content": "encrypted-content",
                    "page_age": "May 21, 2026",
                }
            ],
        },
        {
            "type": "text",
            "text": "Use the docs.",
            "citations": [
                {
                    "type": "web_search_result_location",
                    "title": "Anthropic Docs",
                    "url": "https://docs.anthropic.com/",
                    "encrypted_index": "encrypted-index",
                    "cited_text": "Anthropic documentation excerpt",
                }
            ],
        },
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_search_metadata",
                "model": "claude-sonnet-4-6",
                "content": content_blocks,
                "stop_reason": "end_turn",
            },
        )

    response = _client(httpx.MockTransport(handler)).chat(
        messages=[{"role": "user", "content": "find docs"}]
    )

    assert response.content == "Use the docs."
    assert response.provider_metadata is not None
    metadata = response.provider_metadata["anthropic_messages"]
    assert metadata["content_blocks"] == content_blocks
    assert metadata["server_tool_uses"][0]["id"] == "srvtoolu_1"
    assert metadata["queries"] == ["anthropic docs"]
    assert metadata["sources"][0]["encrypted_content"] == "encrypted-content"
    assert metadata["citations"][0]["encrypted_index"] == "encrypted-index"


def test_chat_streams_text_deltas_usage_and_hidden_thinking_metadata() -> None:
    captured: dict[str, object] = {}
    deltas: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode("utf-8")))
        return _sse_response(
            _sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_stream_text",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-4-6",
                        "content": [],
                        "usage": {"input_tokens": 11, "output_tokens": 0},
                    },
                },
            ),
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "private reasoning"},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "sig-abc"},
                },
            ),
            _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "Hello "},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "world."},
                },
            ),
            _sse_event("content_block_stop", {"type": "content_block_stop", "index": 1}),
            _sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 4},
                },
            ),
            _sse_event("message_stop", {"type": "message_stop"}),
        )

    response = _client(httpx.MockTransport(handler)).chat(
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        on_text_delta=deltas.append,
    )

    assert captured["stream"] is True
    assert response.content == "Hello world."
    assert deltas == ["Hello ", "world."]
    assert response.usage is not None
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 4
    assert response.usage.total_tokens == 15
    metadata = response.provider_metadata["anthropic_messages"]  # type: ignore[index]
    assert metadata["content_blocks"][0]["thinking"] == "private reasoning"
    assert metadata["content_blocks"][0]["signature"] == "sig-abc"
    assert "private reasoning" not in response.content


def test_chat_streams_tool_use_from_partial_json_deltas() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            _sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_stream_tool",
                        "role": "assistant",
                        "model": "claude-sonnet-4-6",
                        "content": [],
                    },
                },
            ),
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_stream_1",
                        "name": "fs_read",
                        "input": {},
                    },
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"path": '},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '"README.md"}'},
                },
            ),
            _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _sse_event(
                "message_delta",
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            ),
            _sse_event("message_stop", {"type": "message_stop"}),
        )

    response = _client(httpx.MockTransport(handler)).chat(
        messages=[{"role": "user", "content": "read"}],
        tools=[
            {"type": "function", "function": {"name": "fs_read", "parameters": {"type": "object"}}}
        ],
        stream=True,
    )

    assert response.content == ""
    assert [(tool.id, tool.name, tool.arguments) for tool in response.tool_calls] == [
        ("toolu_stream_1", "fs_read", {"path": "README.md"})
    ]
    assert response.provider_metadata is not None
    assert response.provider_metadata["anthropic_messages"]["content_blocks"][0]["input"] == {
        "path": "README.md"
    }


def test_chat_streams_parallel_tool_use_blocks_in_index_order() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            _sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {"id": "msg_parallel_tools", "role": "assistant", "content": []},
                },
            ),
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_read",
                        "name": "fs_read",
                        "input": {},
                    },
                },
            ),
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_shell",
                        "name": "shell_run",
                        "input": {},
                    },
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"cmd":"pytest -q"}'},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"path":"README.md"}'},
                },
            ),
            _sse_event("content_block_stop", {"type": "content_block_stop", "index": 1}),
            _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _sse_event(
                "message_delta",
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            ),
            _sse_event("message_stop", {"type": "message_stop"}),
        )

    response = _client(httpx.MockTransport(handler)).chat(
        messages=[{"role": "user", "content": "inspect"}],
        tools=[
            {"type": "function", "function": {"name": "fs_read", "parameters": {"type": "object"}}},
            {
                "type": "function",
                "function": {"name": "shell_run", "parameters": {"type": "object"}},
            },
        ],
        stream=True,
    )

    assert [(tool.id, tool.name, tool.arguments) for tool in response.tool_calls] == [
        ("toolu_read", "fs_read", {"path": "README.md"}),
        ("toolu_shell", "shell_run", {"cmd": "pytest -q"}),
    ]


def test_chat_stream_preserves_hosted_web_search_and_citation_metadata() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            _sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {"id": "msg_stream_search", "role": "assistant", "content": []},
                },
            ),
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": "srvtoolu_stream",
                        "name": "web_search",
                        "input": {},
                    },
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"query":"Anthropic Messages streaming"}',
                    },
                },
            ),
            _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_stream",
                        "content": [
                            {
                                "type": "web_search_result",
                                "title": "Anthropic Docs",
                                "url": "https://docs.anthropic.com/",
                                "encrypted_content": "encrypted-content",
                            }
                        ],
                    },
                },
            ),
            _sse_event("content_block_stop", {"type": "content_block_stop", "index": 1}),
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "text_delta", "text": "Use the Anthropic docs."},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {
                        "type": "citations_delta",
                        "citation": {
                            "type": "web_search_result_location",
                            "title": "Anthropic Docs",
                            "url": "https://docs.anthropic.com/",
                            "encrypted_index": "encrypted-index",
                            "cited_text": "docs excerpt",
                        },
                    },
                },
            ),
            _sse_event("content_block_stop", {"type": "content_block_stop", "index": 2}),
            _sse_event(
                "message_delta",
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
            ),
            _sse_event("message_stop", {"type": "message_stop"}),
        )

    response = _client(httpx.MockTransport(handler)).chat(
        messages=[{"role": "user", "content": "search"}],
        stream=True,
    )

    assert response.content == "Use the Anthropic docs."
    metadata = response.provider_metadata["anthropic_messages"]  # type: ignore[index]
    assert metadata["server_tool_uses"][0]["id"] == "srvtoolu_stream"
    assert metadata["queries"] == ["Anthropic Messages streaming"]
    assert metadata["sources"][0]["encrypted_content"] == "encrypted-content"
    assert metadata["citations"][0]["encrypted_index"] == "encrypted-index"


def test_chat_stream_preserves_unknown_events_without_crashing() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            _sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {"id": "msg_unknown_event", "role": "assistant", "content": []},
                },
            ),
            _sse_event(
                "model_context_window_delta",
                {"type": "model_context_window_delta", "payload": {"remaining": 123}},
            ),
            _sse_event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            _sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "ok"},
                },
            ),
            _sse_event("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _sse_event("message_stop", {"type": "message_stop"}),
        )

    response = _client(httpx.MockTransport(handler)).chat(
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )

    metadata = response.provider_metadata["anthropic_messages"]  # type: ignore[index]
    assert response.content == "ok"
    assert metadata["stream_metadata"]["unknown_events"][0]["event"] == "model_context_window_delta"


def test_chat_stream_surfaces_error_events_and_early_termination() -> None:
    def error_handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            _sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {"id": "msg_stream_error", "role": "assistant", "content": []},
                },
            ),
            _sse_event(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "overloaded_error",
                        "message": "provider overloaded",
                    },
                },
            ),
        )

    with pytest.raises(LLMError, match="overloaded_error: provider overloaded"):
        _client(httpx.MockTransport(error_handler)).chat(
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )

    def truncated_handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            _sse_event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {"id": "msg_truncated", "role": "assistant", "content": []},
                },
            )
        )

    with pytest.raises(LLMError, match="ended before message_stop"):
        _client(httpx.MockTransport(truncated_handler)).chat(
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )

    with pytest.raises(LLMError, match="no message_start"):
        _client(httpx.MockTransport(lambda _request: _sse_response())).chat(
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )


def test_chat_surfaces_provider_errors() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"type": "invalid_request_error", "message": "bad request"}},
        )

    with pytest.raises(LLMError, match="LLM error 400: invalid_request_error: bad request"):
        _client(httpx.MockTransport(handler)).chat(
            messages=[{"role": "user", "content": "hello"}],
        )


def test_chat_rejects_empty_malformed_refusal_and_unsupported_options() -> None:
    with pytest.raises(LLMError, match="does not support response_format"):
        _client(httpx.MockTransport(lambda _request: httpx.Response(500))).chat(
            messages=[{"role": "user", "content": "hello"}],
            response_format={"type": "json_object"},
        )

    with pytest.raises(LLMError, match="does not support reasoning_effort"):
        AnthropicMessagesClient(
            base_url="https://api.anthropic.com/v1",
            api_key="test-key",
            model="claude-sonnet-4-6",
            reasoning_effort="high",
            transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
        ).chat(messages=[{"role": "user", "content": "hello"}])

    def empty_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"id": "msg_empty", "content": [], "stop_reason": "end_turn"}
        )

    with pytest.raises(LLMError, match="returned no assistant text or tool calls"):
        _client(httpx.MockTransport(empty_handler)).chat(
            messages=[{"role": "user", "content": "hello"}],
        )

    def malformed_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "msg_bad"})

    with pytest.raises(LLMError, match="missing content list"):
        _client(httpx.MockTransport(malformed_handler)).chat(
            messages=[{"role": "user", "content": "hello"}],
        )

    def refusal_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_refusal",
                "content": [{"type": "refusal", "text": "I cannot help"}],
                "stop_reason": "refusal",
            },
        )

    with pytest.raises(LLMError, match="Anthropic Messages refusal"):
        _client(httpx.MockTransport(refusal_handler)).chat(
            messages=[{"role": "user", "content": "hello"}],
        )
