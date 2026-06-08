from __future__ import annotations

import json

import httpx
import pytest

from sylliptor_agent_cli.llm.gemini_generate_content import GeminiGenerateContentClient
from sylliptor_agent_cli.llm.metadata import (
    PROVIDER_METADATA_KEY,
    attach_provider_metadata_to_assistant_message,
    strip_provider_metadata_from_message,
)
from sylliptor_agent_cli.llm.types import LLMError


def _client(transport: httpx.BaseTransport) -> GeminiGenerateContentClient:
    return GeminiGenerateContentClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="test-key",
        model="gemini-3-flash-preview",
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


def _fs_read_tool() -> dict[str, object]:
    return {
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


def _sse_event(data: dict[str, object]) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _sse_response(*events: str) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=httpx.ByteStream("".join(events).encode("utf-8")),
    )


def test_chat_maps_messages_tools_tool_choice_response_format_reasoning_and_usage() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-3.1-pro-preview:generateContent"
        )
        assert request.headers["x-goog-api-key"] == "test-key"
        body = json.loads(request.content.decode("utf-8"))
        captured.update(body)
        return httpx.Response(
            200,
            json={
                "responseId": "resp_text",
                "modelVersion": "gemini-3.1-pro-preview",
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "Done."}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 14,
                    "cachedContentTokenCount": 5,
                },
            },
        )

    client = GeminiGenerateContentClient(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="test-key",
        model="gemini-3.1-pro-preview",
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
                    {"type": "image_url", "image_url": {"url": "gs://bucket/image.png"}},
                    {"type": "text", "text": "Describe the image."},
                ],
            },
        ],
        tools=[_fs_read_tool()],
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

    assert captured["systemInstruction"] == {
        "parts": [{"text": "System prompt.\n\nDeveloper prompt."}]
    }
    assert captured["contents"] == [
        {"role": "user", "parts": [{"text": "Read it."}]},
        {
            "role": "model",
            "parts": [
                {"text": "I will read."},
                {
                    "functionCall": {
                        "name": "fs_read",
                        "args": {"path": "README.md"},
                        "id": "call_read",
                    },
                    "thoughtSignature": "skip_thought_signature_validator",
                },
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "id": "call_read",
                        "name": "fs_read",
                        "response": {"result": "README contents"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {"fileData": {"fileUri": "gs://bucket/image.png"}},
                {"text": "Describe the image."},
            ],
        },
    ]
    assert captured["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "fs_read",
                    "description": "Read a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ]
        }
    ]
    assert captured["toolConfig"] == {
        "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["fs_read"]}
    }
    assert captured["generationConfig"] == {
        "temperature": 0.4,
        "maxOutputTokens": 123,
        "responseMimeType": "application/json",
        "responseSchema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
        "thinkingConfig": {"thinkingLevel": "low"},
    }
    assert response.content == "Done."
    assert response.response_model == "gemini-3.1-pro-preview"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 11
    assert response.usage.completion_tokens == 3
    assert response.usage.total_tokens == 14
    assert response.usage.cached_prompt_tokens == 5
    assert response.provider_metadata is not None
    assert response.provider_metadata["gemini_generate_content"]["response_id"] == "resp_text"


def test_chat_imported_parallel_function_calls_get_dummy_thought_signatures() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}]},
        )

    _client(httpx.MockTransport(handler)).chat(
        messages=[
            {"role": "user", "content": "Inspect."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_read",
                        "type": "function",
                        "function": {
                            "name": "fs_read",
                            "arguments": json.dumps({"path": "README.md"}),
                        },
                    },
                    {
                        "id": "call_shell",
                        "type": "function",
                        "function": {
                            "name": "shell_run",
                            "arguments": json.dumps({"cmd": "pytest -q"}),
                        },
                    },
                ],
            },
        ],
        tools=[
            _fs_read_tool(),
            {
                "type": "function",
                "function": {"name": "shell_run", "parameters": {"type": "object"}},
            },
        ],
    )

    assert captured["contents"][1]["parts"] == [  # type: ignore[index]
        {
            "functionCall": {
                "name": "fs_read",
                "args": {"path": "README.md"},
                "id": "call_read",
            },
            "thoughtSignature": "skip_thought_signature_validator",
        },
        {
            "functionCall": {
                "name": "shell_run",
                "args": {"cmd": "pytest -q"},
                "id": "call_shell",
            },
            "thoughtSignature": "skip_thought_signature_validator",
        },
    ]


def test_chat_parses_parallel_function_calls_exact_ids_and_thought_signatures() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "responseId": "resp_tools",
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {
                                        "id": "call_read",
                                        "name": "fs_read",
                                        "args": {"path": "README.md"},
                                    },
                                    "thoughtSignature": "thought-read",
                                },
                                {
                                    "functionCall": {
                                        "id": "call_shell",
                                        "name": "shell_run",
                                        "args": {"cmd": "pytest -q"},
                                    },
                                    "thought_signature": "thought-shell",
                                },
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 5},
            },
        )

    response = _client(httpx.MockTransport(handler)).chat(
        messages=[{"role": "user", "content": "Inspect."}],
        tools=[
            _fs_read_tool(),
            {
                "type": "function",
                "function": {"name": "shell_run", "parameters": {"type": "object"}},
            },
        ],
    )

    assert response.content == ""
    assert [(tc.id, tc.name, tc.arguments) for tc in response.tool_calls] == [
        ("call_read", "fs_read", {"path": "README.md"}),
        ("call_shell", "shell_run", {"cmd": "pytest -q"}),
    ]
    assert response.tool_calls[0].provider_metadata == {
        "gemini_generate_content": {"part_index": 0, "thoughtSignature": "thought-read"}
    }
    assert response.tool_calls[1].provider_metadata == {
        "gemini_generate_content": {"part_index": 1, "thought_signature": "thought-shell"}
    }
    assert response.provider_metadata is not None
    metadata = response.provider_metadata["gemini_generate_content"]
    assert metadata["content"]["parts"][0]["thoughtSignature"] == "thought-read"
    assert response.usage is not None
    assert response.usage.total_tokens == 12


def test_chat_function_call_and_function_response_round_trip_replays_metadata() -> None:
    calls: list[dict[str, object]] = []
    function_call_part = {
        "functionCall": {
            "id": "call_exact_123",
            "name": "fs_read",
            "args": {"path": "README.md"},
        },
        "thoughtSignature": "thought-exact",
    }
    provider_content = {"role": "model", "parts": [function_call_part]}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "responseId": "resp_call",
                    "candidates": [{"content": provider_content, "finishReason": "STOP"}],
                },
            )
        return httpx.Response(
            200,
            json={
                "responseId": "resp_done",
                "candidates": [{"content": {"role": "model", "parts": [{"text": "Read it."}]}}],
            },
        )

    client = _client(httpx.MockTransport(handler))
    first = client.chat(
        messages=[{"role": "user", "content": "read"}],
        tools=[_fs_read_tool()],
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

    assert calls[1]["contents"] == [
        {"role": "user", "parts": [{"text": "read"}]},
        provider_content,
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "id": "call_exact_123",
                        "name": "fs_read",
                        "response": {"result": "file contents"},
                    }
                }
            ],
        },
    ]


def test_chat_replays_snake_case_thought_signature_as_gemini_wire_key() -> None:
    calls: list[dict[str, object]] = []
    provider_content = {
        "role": "model",
        "parts": [
            {
                "functionCall": {
                    "id": "call_snake_sig",
                    "name": "fs_read",
                    "args": {"path": "README.md"},
                },
                "thought_signature": "thought-from-provider",
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content.decode("utf-8")))
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "responseId": "resp_call",
                    "candidates": [{"content": provider_content, "finishReason": "STOP"}],
                },
            )
        return httpx.Response(
            200,
            json={
                "responseId": "resp_done",
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
            },
        )

    client = _client(httpx.MockTransport(handler))
    first = client.chat(messages=[{"role": "user", "content": "read"}], tools=[_fs_read_tool()])
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
            {"role": "tool", "tool_call_id": "call_snake_sig", "content": "README"},
        ]
    )

    replay_part = calls[1]["contents"][1]["parts"][0]  # type: ignore[index]
    assert replay_part["thoughtSignature"] == "thought-from-provider"
    assert "thought_signature" not in replay_part


def test_chat_text_only_response_preserves_provider_metadata_for_next_turn() -> None:
    calls: list[dict[str, object]] = []
    provider_content = {"role": "model", "parts": [{"text": "First answer."}]}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "responseId": "resp_text_metadata",
                    "candidates": [{"content": provider_content, "finishReason": "STOP"}],
                },
            )
        return httpx.Response(
            200,
            json={
                "responseId": "resp_second",
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "Second answer."}]}}
                ],
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

    second = client.chat(
        messages=[
            {"role": "user", "content": "first"},
            assistant_message,
            {"role": "user", "content": "second"},
        ],
    )

    assert second.content == "Second answer."
    assert calls[1]["contents"] == [
        {"role": "user", "parts": [{"text": "first"}]},
        provider_content,
        {"role": "user", "parts": [{"text": "second"}]},
    ]


def test_chat_native_mode_uses_only_gemini_google_search_grounding() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.update(body)
        return httpx.Response(
            200,
            json={
                "responseId": "resp_search",
                "candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}],
            },
        )

    client = GeminiGenerateContentClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="test-key",
        model="gemini-3-flash-preview",
        web_search_mode="native",
        web_search_adapter="gemini_grounding",
        transport=httpx.MockTransport(handler),
    )
    response = client.chat(
        messages=[{"role": "user", "content": "find docs"}],
        tools=[_web_search_function_tool(), _fs_read_tool()],
    )

    assert captured["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "fs_read",
                    "description": "Read a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ]
        },
        {"google_search": {}},
    ]
    assert captured["toolConfig"] == {"includeServerSideToolInvocations": True}
    assert response.content == "ok"


def test_chat_auto_mode_prefers_google_search_grounding_for_auto_adapter() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}]},
        )

    client = GeminiGenerateContentClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="test-key",
        model="gemini-3-flash-preview",
        web_search_mode="auto",
        web_search_adapter="auto",
        transport=httpx.MockTransport(handler),
    )
    client.chat(
        messages=[{"role": "user", "content": "find docs"}], tools=[_web_search_function_tool()]
    )

    assert captured["tools"] == [{"google_search": {}}]
    assert "toolConfig" not in captured


def test_chat_combined_grounding_and_tool_choice_merges_tool_config() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}]},
        )

    client = GeminiGenerateContentClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="test-key",
        model="gemini-3-flash-preview",
        web_search_mode="native",
        web_search_adapter="gemini_grounding",
        transport=httpx.MockTransport(handler),
    )
    client.chat(
        messages=[{"role": "user", "content": "find docs"}],
        tools=[_web_search_function_tool(), _fs_read_tool()],
        tool_choice="auto",
    )

    assert captured["toolConfig"] == {
        "functionCallingConfig": {"mode": "AUTO"},
        "includeServerSideToolInvocations": True,
    }


def test_chat_external_mode_uses_only_sylliptor_web_search_function() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}]},
        )

    client = GeminiGenerateContentClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="test-key",
        model="gemini-3-flash-preview",
        web_search_mode="external",
        web_search_adapter="tavily",
        transport=httpx.MockTransport(handler),
    )
    client.chat(
        messages=[{"role": "user", "content": "find docs"}], tools=[_web_search_function_tool()]
    )

    assert captured["tools"] == [
        {
            "functionDeclarations": [
                {
                    "name": "web_search",
                    "description": "Standalone Sylliptor web search.",
                    "parameters": {"type": "object"},
                }
            ]
        }
    ]


def test_chat_auto_mode_with_external_adapter_uses_sylliptor_web_search_function() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}]},
        )

    client = GeminiGenerateContentClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="test-key",
        model="gemini-3-flash-preview",
        web_search_mode="auto",
        web_search_adapter="tavily",
        transport=httpx.MockTransport(handler),
    )
    client.chat(
        messages=[{"role": "user", "content": "find docs"}], tools=[_web_search_function_tool()]
    )

    assert captured["tools"][0]["functionDeclarations"][0]["name"] == "web_search"


def test_chat_off_mode_removes_all_web_search_tools() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}]},
        )

    client = GeminiGenerateContentClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="test-key",
        model="gemini-3-flash-preview",
        web_search_mode="off",
        web_search_adapter="gemini_grounding",
        transport=httpx.MockTransport(handler),
    )
    client.chat(
        messages=[{"role": "user", "content": "find docs"}],
        tools=[_web_search_function_tool(), {"google_search": {}}],
    )

    assert "tools" not in captured


def test_chat_native_mode_rejects_non_gemini_search_adapter() -> None:
    client = GeminiGenerateContentClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="test-key",
        model="gemini-3-flash-preview",
        web_search_mode="native",
        web_search_adapter="tavily",
        transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
    )

    with pytest.raises(LLMError, match="web_search_mode=native.*gemini_grounding"):
        client.chat(
            messages=[{"role": "user", "content": "find docs"}],
            tools=[_web_search_function_tool()],
        )


def test_chat_rejects_forced_google_search_tool_choice() -> None:
    client = GeminiGenerateContentClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="test-key",
        model="gemini-3-flash-preview",
        web_search_mode="native",
        web_search_adapter="gemini_grounding",
        transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
    )

    with pytest.raises(LLMError, match="cannot force Google Search grounding"):
        client.chat(
            messages=[{"role": "user", "content": "find docs"}],
            tools=[_web_search_function_tool()],
            tool_choice={"type": "function", "function": {"name": "web_search"}},
        )


def test_chat_thinking_level_and_budget_are_mapped_deterministically() -> None:
    captured: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}]},
        )

    GeminiGenerateContentClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="test-key",
        model="gemini-3-flash-preview",
        thinking_level="high",
        transport=httpx.MockTransport(handler),
    ).chat(messages=[{"role": "user", "content": "hello"}])
    assert captured[-1]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "high"}

    GeminiGenerateContentClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="test-key",
        model="gemini-2.5-pro",
        thinking_budget=-1,
        transport=httpx.MockTransport(handler),
    ).chat(messages=[{"role": "user", "content": "hello"}])
    assert captured[-1]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": -1}

    with pytest.raises(LLMError, match="thinking_level requires a Gemini 3 model"):
        GeminiGenerateContentClient(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="test-key",
            model="gemini-2.5-pro",
            thinking_level="low",
            transport=httpx.MockTransport(handler),
        ).chat(messages=[{"role": "user", "content": "hello"}])

    with pytest.raises(LLMError, match="cannot set both thinking_level and thinking_budget"):
        GeminiGenerateContentClient(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="test-key",
            model="gemini-3-flash-preview",
            thinking_level="low",
            thinking_budget=1024,
            transport=httpx.MockTransport(handler),
        ).chat(messages=[{"role": "user", "content": "hello"}])


def test_chat_extracts_grounding_metadata_citations_sources_queries_and_tool_calls() -> None:
    grounding = {
        "webSearchQueries": ["gemini docs"],
        "groundingChunks": [
            {"web": {"uri": "https://ai.google.dev/gemini-api/docs", "title": "Gemini Docs"}}
        ],
        "groundingSupports": [
            {
                "segment": {"startIndex": 0, "endIndex": 8, "text": "Use docs"},
                "groundingChunkIndices": [0],
            }
        ],
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "responseId": "resp_grounded",
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "Use docs."},
                                {
                                    "functionCall": {
                                        "id": "call_read",
                                        "name": "fs_read",
                                        "args": {"path": "README.md"},
                                    }
                                },
                            ],
                        },
                        "groundingMetadata": grounding,
                        "finishReason": "STOP",
                    }
                ],
            },
        )

    response = _client(httpx.MockTransport(handler)).chat(
        messages=[{"role": "user", "content": "find docs"}],
        tools=[_fs_read_tool()],
    )

    assert response.content == "Use docs."
    assert response.tool_calls[0].id == "call_read"
    assert response.provider_metadata is not None
    metadata = response.provider_metadata["gemini_generate_content"]
    assert metadata["content"]["parts"][1]["functionCall"]["id"] == "call_read"
    assert metadata["groundingMetadata"] == grounding
    assert metadata["queries"] == ["gemini docs"]
    assert metadata["sources"] == [
        {"url": "https://ai.google.dev/gemini-api/docs", "title": "Gemini Docs"}
    ]
    assert metadata["citations"] == [
        {
            "title": "Gemini Docs",
            "url": "https://ai.google.dev/gemini-api/docs",
            "start_index": 0,
            "end_index": 8,
            "text": "Use docs",
        }
    ]


def test_chat_grounded_function_call_round_trip_replays_provider_content() -> None:
    calls: list[dict[str, object]] = []
    grounding = {
        "webSearchQueries": ["gemini docs"],
        "groundingChunks": [
            {"web": {"uri": "https://ai.google.dev/gemini-api/docs", "title": "Gemini Docs"}}
        ],
    }
    provider_content = {
        "role": "model",
        "parts": [
            {"text": "I found docs."},
            {
                "functionCall": {
                    "id": "call_grounded_read",
                    "name": "fs_read",
                    "args": {"path": "README.md"},
                },
                "thoughtSignature": "grounded-thought",
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "responseId": "resp_grounded_call",
                    "candidates": [
                        {
                            "content": provider_content,
                            "groundingMetadata": grounding,
                            "finishReason": "STOP",
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}}]},
        )

    client = GeminiGenerateContentClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="test-key",
        model="gemini-3-flash-preview",
        web_search_mode="native",
        web_search_adapter="gemini_grounding",
        transport=httpx.MockTransport(handler),
    )
    first = client.chat(
        messages=[{"role": "user", "content": "find and read"}],
        tools=[_web_search_function_tool(), _fs_read_tool()],
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

    assert first.provider_metadata is not None
    metadata = first.provider_metadata["gemini_generate_content"]
    assert metadata["groundingMetadata"] == grounding
    assert metadata["content"] == provider_content

    client.chat(
        messages=[
            {"role": "user", "content": "find and read"},
            assistant_message,
            {"role": "tool", "tool_call_id": "call_grounded_read", "content": "file contents"},
        ],
        tools=[_web_search_function_tool(), _fs_read_tool()],
    )

    assert calls[0]["toolConfig"] == {"includeServerSideToolInvocations": True}
    assert calls[1]["toolConfig"] == {"includeServerSideToolInvocations": True}
    assert calls[1]["contents"] == [
        {"role": "user", "parts": [{"text": "find and read"}]},
        provider_content,
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "id": "call_grounded_read",
                        "name": "fs_read",
                        "response": {"result": "file contents"},
                    }
                }
            ],
        },
    ]


def test_chat_streams_text_usage_and_metadata() -> None:
    captured: dict[str, object] = {}
    deltas: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == (
            "https://generativelanguage.googleapis.com/v1beta/"
            "models/gemini-3-flash-preview:streamGenerateContent?alt=sse"
        )
        captured.update(json.loads(request.content.decode("utf-8")))
        return _sse_response(
            _sse_event(
                {
                    "responseId": "resp_stream_text",
                    "modelVersion": "gemini-3-flash-preview",
                    "candidates": [{"content": {"role": "model", "parts": [{"text": "Hello "}]}}],
                }
            ),
            _sse_event(
                {
                    "responseId": "resp_stream_text",
                    "modelVersion": "gemini-3-flash-preview",
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "world."}]},
                            "finishReason": "STOP",
                            "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT"}],
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 10,
                        "candidatesTokenCount": 3,
                        "totalTokenCount": 13,
                    },
                }
            ),
        )

    response = _client(httpx.MockTransport(handler)).chat(
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        on_text_delta=deltas.append,
    )

    assert captured["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]
    assert response.content == "Hello world."
    assert deltas == ["Hello ", "world."]
    assert response.response_model == "gemini-3-flash-preview"
    assert response.usage is not None
    assert response.usage.total_tokens == 13
    metadata = response.provider_metadata["gemini_generate_content"]  # type: ignore[index]
    assert metadata["response_id"] == "resp_stream_text"
    assert metadata["model_version"] == "gemini-3-flash-preview"
    assert metadata["finish_reason"] == "STOP"
    assert metadata["safety_ratings"] == [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT"}]
    assert metadata["content"]["parts"] == [{"text": "Hello "}, {"text": "world."}]


def test_chat_streams_single_function_call_with_id_and_thought_signature() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            _sse_event(
                {
                    "responseId": "resp_stream_call",
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "functionCall": {
                                            "id": "call_stream_read",
                                            "name": "fs_read",
                                            "args": {"path": "README.md"},
                                        },
                                        "thoughtSignature": "stream-thought",
                                    }
                                ],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                }
            )
        )

    response = _client(httpx.MockTransport(handler)).chat(
        messages=[{"role": "user", "content": "read"}],
        tools=[_fs_read_tool()],
        stream=True,
    )

    assert response.content == ""
    assert [(tool.id, tool.name, tool.arguments) for tool in response.tool_calls] == [
        ("call_stream_read", "fs_read", {"path": "README.md"})
    ]
    assert response.tool_calls[0].provider_metadata == {
        "gemini_generate_content": {"part_index": 0, "thoughtSignature": "stream-thought"}
    }
    metadata = response.provider_metadata["gemini_generate_content"]  # type: ignore[index]
    assert metadata["content"]["parts"][0]["thoughtSignature"] == "stream-thought"


def test_chat_streams_parallel_function_calls_in_exact_part_order() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            _sse_event(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "functionCall": {
                                            "id": "call_read",
                                            "name": "fs_read",
                                            "args": {"path": "README.md"},
                                        },
                                        "thoughtSignature": "thought-read",
                                    },
                                    {
                                        "functionCall": {
                                            "id": "call_shell",
                                            "name": "shell_run",
                                            "args": {"cmd": "pytest -q"},
                                        },
                                        "thoughtSignature": "thought-shell",
                                    },
                                ],
                            },
                            "finishReason": "STOP",
                        }
                    ]
                }
            )
        )

    response = _client(httpx.MockTransport(handler)).chat(
        messages=[{"role": "user", "content": "inspect"}],
        tools=[
            _fs_read_tool(),
            {
                "type": "function",
                "function": {"name": "shell_run", "parameters": {"type": "object"}},
            },
        ],
        stream=True,
    )

    assert [(tool.id, tool.name, tool.arguments) for tool in response.tool_calls] == [
        ("call_read", "fs_read", {"path": "README.md"}),
        ("call_shell", "shell_run", {"cmd": "pytest -q"}),
    ]
    metadata = response.provider_metadata["gemini_generate_content"]  # type: ignore[index]
    assert [part["functionCall"]["id"] for part in metadata["content"]["parts"]] == [
        "call_read",
        "call_shell",
    ]
    assert metadata["content"]["parts"][1]["thoughtSignature"] == "thought-shell"


def test_chat_streams_mixed_text_function_call_and_replays_function_response() -> None:
    calls: list[dict[str, object]] = []
    provider_content = {
        "role": "model",
        "parts": [
            {"text": "I will read."},
            {
                "functionCall": {
                    "id": "call_stream_mixed",
                    "name": "fs_read",
                    "args": {"path": "README.md"},
                },
                "thoughtSignature": "mixed-thought",
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content.decode("utf-8")))
        if len(calls) == 1:
            return _sse_response(
                _sse_event(
                    {
                        "candidates": [
                            {"content": {"role": "model", "parts": [{"text": "I will read."}]}}
                        ]
                    }
                ),
                _sse_event(
                    {
                        "responseId": "resp_stream_mixed",
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "id": "call_stream_mixed",
                                                "name": "fs_read",
                                                "args": {"path": "README.md"},
                                            },
                                            "thoughtSignature": "mixed-thought",
                                        }
                                    ],
                                },
                                "finishReason": "STOP",
                            }
                        ],
                    }
                ),
            )
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "done"}]}}]},
        )

    client = _client(httpx.MockTransport(handler))
    first = client.chat(
        messages=[{"role": "user", "content": "read"}],
        tools=[_fs_read_tool()],
        stream=True,
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
            {"role": "tool", "tool_call_id": "call_stream_mixed", "content": "file contents"},
        ],
        tools=[_fs_read_tool()],
    )

    assert calls[1]["contents"] == [
        {"role": "user", "parts": [{"text": "read"}]},
        provider_content,
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "id": "call_stream_mixed",
                        "name": "fs_read",
                        "response": {"result": "file contents"},
                    }
                }
            ],
        },
    ]


def test_chat_stream_preserves_google_search_grounding_metadata() -> None:
    captured: dict[str, object] = {}
    grounding = {
        "webSearchQueries": ["gemini stream grounding"],
        "groundingChunks": [
            {"web": {"uri": "https://ai.google.dev/gemini-api/docs", "title": "Gemini Docs"}}
        ],
        "groundingSupports": [
            {
                "segment": {"startIndex": 0, "endIndex": 8, "text": "Use docs"},
                "groundingChunkIndices": [0],
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode("utf-8")))
        return _sse_response(
            _sse_event(
                {
                    "responseId": "resp_stream_grounded",
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "Use docs."}]},
                            "groundingMetadata": grounding,
                            "finishReason": "STOP",
                        }
                    ],
                }
            )
        )

    response = GeminiGenerateContentClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="test-key",
        model="gemini-3-flash-preview",
        web_search_mode="native",
        web_search_adapter="gemini_grounding",
        transport=httpx.MockTransport(handler),
    ).chat(
        messages=[{"role": "user", "content": "find docs"}],
        tools=[_web_search_function_tool()],
        stream=True,
    )

    assert captured["tools"] == [{"google_search": {}}]
    assert response.content == "Use docs."
    metadata = response.provider_metadata["gemini_generate_content"]  # type: ignore[index]
    assert metadata["groundingMetadata"] == grounding
    assert metadata["queries"] == ["gemini stream grounding"]
    assert metadata["sources"] == [
        {"url": "https://ai.google.dev/gemini-api/docs", "title": "Gemini Docs"}
    ]
    assert metadata["citations"] == [
        {
            "title": "Gemini Docs",
            "url": "https://ai.google.dev/gemini-api/docs",
            "start_index": 0,
            "end_index": 8,
            "text": "Use docs",
        }
    ]
    assert "groundingMetadata" not in response.content


def test_chat_stream_preserves_grounding_and_function_call_metadata_together() -> None:
    grounding = {
        "webSearchQueries": ["gemini docs"],
        "groundingChunks": [
            {"web": {"uri": "https://ai.google.dev/gemini-api/docs", "title": "Gemini Docs"}}
        ],
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return _sse_response(
            _sse_event(
                {
                    "responseId": "resp_grounded_stream_call",
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {"text": "I found docs."},
                                    {
                                        "functionCall": {
                                            "id": "call_grounded_stream",
                                            "name": "fs_read",
                                            "args": {"path": "README.md"},
                                        },
                                        "thoughtSignature": "grounded-stream-thought",
                                    },
                                ],
                            },
                            "groundingMetadata": grounding,
                            "finishReason": "STOP",
                        }
                    ],
                }
            )
        )

    response = _client(httpx.MockTransport(handler)).chat(
        messages=[{"role": "user", "content": "find and read"}],
        tools=[_web_search_function_tool(), _fs_read_tool()],
        stream=True,
    )

    assert response.content == "I found docs."
    assert response.tool_calls[0].id == "call_grounded_stream"
    metadata = response.provider_metadata["gemini_generate_content"]  # type: ignore[index]
    assert metadata["groundingMetadata"] == grounding
    assert metadata["content"]["parts"][1]["thoughtSignature"] == "grounded-stream-thought"


def test_chat_stream_surfaces_malformed_error_and_empty_streams() -> None:
    with pytest.raises(LLMError, match="LLM error 400: INVALID_ARGUMENT: bad stream request"):
        _client(
            httpx.MockTransport(
                lambda _request: httpx.Response(
                    400,
                    json={
                        "error": {
                            "status": "INVALID_ARGUMENT",
                            "message": "bad stream request",
                        }
                    },
                )
            )
        ).chat(messages=[{"role": "user", "content": "hello"}], stream=True)

    with pytest.raises(LLMError, match="malformed JSON"):
        _client(
            httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=httpx.ByteStream(b"data: {bad\n\n"),
                )
            )
        ).chat(messages=[{"role": "user", "content": "hello"}], stream=True)

    with pytest.raises(LLMError, match="INVALID_ARGUMENT: bad stream"):
        _client(
            httpx.MockTransport(
                lambda _request: _sse_response(
                    _sse_event({"error": {"status": "INVALID_ARGUMENT", "message": "bad stream"}})
                )
            )
        ).chat(messages=[{"role": "user", "content": "hello"}], stream=True)

    with pytest.raises(LLMError, match="returned no chunks"):
        _client(httpx.MockTransport(lambda _request: _sse_response())).chat(
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )

    with pytest.raises(LLMError, match="returned no candidate chunks"):
        _client(httpx.MockTransport(lambda _request: _sse_response(_sse_event({})))).chat(
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
        )


def test_chat_surfaces_provider_errors() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"status": "INVALID_ARGUMENT", "message": "bad request"}},
        )

    with pytest.raises(LLMError, match="LLM error 400: INVALID_ARGUMENT: bad request"):
        _client(httpx.MockTransport(handler)).chat(
            messages=[{"role": "user", "content": "hello"}],
        )


def test_chat_rejects_empty_malformed_and_unsupported_options() -> None:
    with pytest.raises(LLMError, match="does not support prompt_cache_key"):
        GeminiGenerateContentClient(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="test-key",
            model="gemini-3-flash-preview",
            prompt_cache_key="cache-key",
            transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
        ).chat(messages=[{"role": "user", "content": "hello"}])

    with pytest.raises(LLMError, match="does not support response_format type"):
        _client(httpx.MockTransport(lambda _request: httpx.Response(500))).chat(
            messages=[{"role": "user", "content": "hello"}],
            response_format={"type": "xml_object"},
        )

    with pytest.raises(LLMError, match="reasoning_effort is not supported"):
        GeminiGenerateContentClient(
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key="test-key",
            model="gemini-3-flash-preview",
            reasoning_effort="extreme",
            transport=httpx.MockTransport(lambda _request: httpx.Response(500)),
        ).chat(messages=[{"role": "user", "content": "hello"}])

    def empty_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": []}}]},
        )

    with pytest.raises(LLMError, match="returned no assistant text or tool calls"):
        _client(httpx.MockTransport(empty_handler)).chat(
            messages=[{"role": "user", "content": "hello"}],
        )

    def malformed_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"responseId": "resp_bad"})

    with pytest.raises(LLMError, match="missing candidates"):
        _client(httpx.MockTransport(malformed_handler)).chat(
            messages=[{"role": "user", "content": "hello"}],
        )
