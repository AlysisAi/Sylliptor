from __future__ import annotations

import json

import httpx
import pytest

from sylliptor_agent_cli.llm import types as shared_types
from sylliptor_agent_cli.llm.openai_compat import (
    PROVIDER_METADATA_KEY,
    LLMError,
    LLMResponse,
    LLMUsage,
    OpenAICompatClient,
    ToolCall,
    _provider_key_from_base_url,
    attach_provider_metadata_to_assistant_message,
    sylliptor_trial_error_message,
)
from sylliptor_agent_cli.llm.provider_limits import ProviderRetrySettings

_SYLLIPTOR_TRIAL_BASE_URL = "https://vzigujbcjjmpntxhmyvr.supabase.co/functions/v1/llm/v1"


def test_openai_compat_reexports_shared_llm_types() -> None:
    assert LLMError is shared_types.LLMError
    assert ToolCall is shared_types.ToolCall
    assert LLMUsage is shared_types.LLMUsage
    assert LLMResponse is shared_types.LLMResponse


def test_sylliptor_trial_error_message_maps_known_codes() -> None:
    err = LLMError(
        "LLM error 402: "
        + json.dumps({"error": {"message": "Trial window passed.", "code": "trial_expired"}})
    )
    msg = sylliptor_trial_error_message(err)
    assert msg is not None
    assert "trial has ended" in msg
    assert "sylliptor.alysisai.com/account" in msg


def test_sylliptor_trial_error_message_handles_each_proxy_code() -> None:
    cases = {
        "invalid_key": "sylliptor login",
        "quota_exhausted": "trial tokens",
        "email_not_verified": "confirm your email",
        "plan_inactive": "not active",
        "rate_limit_exceeded": "wait a moment",
        "global_budget_exceeded": "at capacity",
        "proxy_unconfigured": "temporarily unavailable",
    }
    for code, needle in cases.items():
        err = LLMError("LLM error 400: " + json.dumps({"error": {"code": code}}))
        msg = sylliptor_trial_error_message(err)
        assert msg is not None, code
        assert needle in msg, code


def test_sylliptor_trial_error_message_ignores_non_proxy_errors() -> None:
    # Upstream OpenRouter error: numeric code, not one of ours.
    upstream = LLMError(
        "LLM error 401: " + json.dumps({"error": {"message": "User not found.", "code": 401}})
    )
    assert sylliptor_trial_error_message(upstream) is None

    # Unknown string code.
    unknown = LLMError("LLM error 500: " + json.dumps({"error": {"code": "kaboom"}}))
    assert sylliptor_trial_error_message(unknown) is None

    # Non-JSON body (e.g. a plain-text 500 from an edge/CDN layer).
    plain = LLMError("LLM error 500: Internal Server Error")
    assert sylliptor_trial_error_message(plain) is None


def _surrogate_escaped_text(text: str) -> str:
    return text.encode("utf-8").decode("ascii", errors="surrogateescape")


def test_parses_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "test-model"
        assert body["temperature"] == 1.0
        data = {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "fs_read",
                                    "arguments": json.dumps({"path": "README.md", "max_bytes": 10}),
                                },
                            }
                        ],
                    }
                }
            ]
        }
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.content == "ok"
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.name == "fs_read"
    assert tc.arguments["path"] == "README.md"


def test_sends_custom_temperature() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["temperature"] == 0.2
        data = {"choices": [{"message": {"content": "ok"}}]}
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        temperature=0.2,
        transport=transport,
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.content == "ok"


def test_sends_extra_headers() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        extra_headers={"anthropic-version": "2023-06-01"},
        transport=transport,
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.content == "ok"


def test_chat_decompression_error_is_explicit() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip", "content-type": "application/json"},
            content=b'{"choices":[{"message":{"content":"ok"}}]}',
        )

    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMError, match="response decompression failed"):
        client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])


def test_chat_temperature_override_takes_precedence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["temperature"] == 0.9
        data = {"choices": [{"message": {"content": "ok"}}]}
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        temperature=0.2,
        transport=transport,
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[], temperature=0.9)
    assert resp.content == "ok"


def test_retries_with_default_temperature_when_provider_accepts_only_default() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(body)
        if len(requests) == 1:
            assert body["temperature"] == 0.0
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": (
                            "Unsupported value: 'temperature' does not support 0 with "
                            "this model. Only the default (1) value is supported."
                        ),
                        "type": "invalid_request_error",
                        "param": "temperature",
                        "code": "unsupported_value",
                    }
                },
            )
        assert body["temperature"] == 1.0
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatClient(
        base_url="https://api.openai.com/v1",
        api_key="test",
        model="gpt-5.5",
        temperature=0.0,
        transport=httpx.MockTransport(handler),
    )

    first = client.chat(messages=[{"role": "user", "content": "hi"}])
    second = client.chat(messages=[{"role": "user", "content": "hi again"}])

    assert first.content == "ok"
    assert second.content == "ok"
    assert len(requests) == 3
    assert requests[0]["temperature"] == 0.0
    assert requests[1]["temperature"] == 1.0
    assert requests[2]["temperature"] == 1.0
    assert first.provider_metadata == {
        "transport": {
            "temperature_adjusted": True,
            "temperature_adjustment": "default_temperature",
            "temperature_adjustment_reason": "provider_rejected_parameter",
            "temperature_retry_used": True,
            "temperature_retry_count": 1,
        }
    }
    assert second.provider_metadata == {
        "transport": {
            "temperature_adjusted": True,
            "temperature_adjustment": "default_temperature",
            "temperature_adjustment_reason": "cached_provider_rejection",
        }
    }


def test_omits_temperature_when_default_temperature_is_also_rejected() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(body)
        if len(requests) <= 2:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "temperature is not supported for this model.",
                        "type": "invalid_request_error",
                        "param": "temperature",
                        "code": "unsupported_parameter",
                    }
                },
            )
        assert "temperature" not in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="no-temperature-model",
        temperature=0.2,
        transport=httpx.MockTransport(handler),
    )

    first = client.chat(messages=[{"role": "user", "content": "hi"}])
    second = client.chat(messages=[{"role": "user", "content": "hi again"}])

    assert first.content == "ok"
    assert second.content == "ok"
    assert len(requests) == 4
    assert requests[0]["temperature"] == 0.2
    assert requests[1]["temperature"] == 1.0
    assert "temperature" not in requests[2]
    assert "temperature" not in requests[3]
    assert first.provider_metadata == {
        "transport": {
            "temperature_adjusted": True,
            "temperature_adjustment": "omit_temperature",
            "temperature_adjustment_reason": "provider_rejected_parameter",
            "temperature_retry_used": True,
            "temperature_retry_count": 2,
            "temperature_omitted": True,
            "temperature_omit_reason": "provider_rejected_parameter",
        }
    }
    assert second.provider_metadata == {
        "transport": {
            "temperature_adjusted": True,
            "temperature_adjustment": "omit_temperature",
            "temperature_adjustment_reason": "cached_provider_rejection",
            "temperature_omitted": True,
            "temperature_omit_reason": "cached_provider_rejection",
        }
    }


def test_retries_with_default_temperature_for_plain_text_temperature_error() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(400, text="temperature is not supported for this model")
        assert body["temperature"] == 1.0
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="plain-text-error-model",
        temperature=0.2,
        transport=httpx.MockTransport(handler),
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}])

    assert resp.content == "ok"
    assert len(requests) == 2
    assert requests[0]["temperature"] == 0.2
    assert requests[1]["temperature"] == 1.0


def test_retries_with_default_temperature_for_temperature_range_error() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "temperature must be in range (0.0, 1.0]",
                        "type": "invalid_request_error",
                        "param": "temperature",
                        "code": "invalid_parameter",
                    }
                },
            )
        assert body["temperature"] == 1.0
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="temperature-range-model",
        temperature=0.0,
        transport=httpx.MockTransport(handler),
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}])

    assert resp.content == "ok"
    assert len(requests) == 2
    assert requests[0]["temperature"] == 0.0
    assert requests[1]["temperature"] == 1.0


def test_omits_temperature_when_provider_allows_only_non_default_temperature() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(body)
        if len(requests) <= 2:
            return httpx.Response(400, text="invalid temperature: only 0.6 is allowed")
        assert "temperature" not in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="fixed-temperature-model",
        temperature=0.2,
        transport=httpx.MockTransport(handler),
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}])

    assert resp.content == "ok"
    assert len(requests) == 3
    assert requests[0]["temperature"] == 0.2
    assert requests[1]["temperature"] == 1.0
    assert "temperature" not in requests[2]


def test_retries_without_temperature_for_deprecated_temperature_error_without_param() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(body)
        if len(requests) == 1:
            assert body["temperature"] == 0.2
            return httpx.Response(
                400,
                json={
                    "error": {
                        "code": "invalid_request_error",
                        "message": "`temperature` is deprecated for this model.",
                        "type": "invalid_request_error",
                        "param": None,
                    }
                },
            )
        assert "temperature" not in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatClient(
        base_url="https://api.anthropic.com/v1",
        api_key="test",
        model="claude-opus-4-7",
        temperature=0.2,
        transport=httpx.MockTransport(handler),
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}])

    assert resp.content == "ok"
    assert len(requests) == 2


def test_temperature_retry_does_not_hide_unrelated_provider_errors() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            404,
            json={
                "error": {
                    "code": "not_found_error",
                    "message": "model: missing-model",
                    "type": "invalid_request_error",
                    "param": None,
                }
            },
        )

    client = OpenAICompatClient(
        base_url="https://api.anthropic.com/v1",
        api_key="test",
        model="missing-model",
        temperature=0.2,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMError, match="not_found_error"):
        client.chat(messages=[{"role": "user", "content": "hi"}])

    assert len(requests) == 1
    assert requests[0]["temperature"] == 0.2


def test_stream_retries_with_default_temperature_when_provider_accepts_only_default() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": (
                            "Unsupported value: 'temperature' does not support 0 with "
                            "this model. Only the default (1) value is supported."
                        ),
                        "type": "invalid_request_error",
                        "param": "temperature",
                        "code": "unsupported_value",
                    }
                },
            )
        assert body["temperature"] == 1.0
        assert body["stream"] is True
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"content":"o"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    client = OpenAICompatClient(
        base_url="https://api.openai.com/v1",
        api_key="test",
        model="gpt-5.5",
        temperature=0.0,
        transport=httpx.MockTransport(handler),
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], stream=True)

    assert resp.content == "ok"
    assert len(requests) == 2
    assert requests[0]["temperature"] == 0.0
    assert requests[1]["temperature"] == 1.0


def test_stream_cancellation_token_interrupts_midstream() -> None:
    # A cancel mid-stream must raise KeyboardInterrupt (so the TUI shows
    # "Interrupted.") and stop consuming further deltas — not finish the response.
    import pytest

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"c"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    client = OpenAICompatClient(
        base_url="https://api.openai.com/v1",
        api_key="test",
        model="m",
        transport=httpx.MockTransport(handler),
    )

    class _Tok:
        def __init__(self) -> None:
            self._cancelled = False
            self._abort = None

        def set_abort_callback(self, fn):
            self._abort = fn

        def clear_abort_callback(self):
            self._abort = None

        @property
        def is_cancelled(self) -> bool:
            return self._cancelled

        def cancel(self) -> None:
            self._cancelled = True
            if self._abort is not None:
                self._abort()

    tok = _Tok()
    seen: list[str] = []

    def on_text(delta: str) -> None:
        seen.append(delta)
        tok.cancel()  # user interrupts right after the first delta

    with pytest.raises(KeyboardInterrupt):
        client.chat(
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            on_text_delta=on_text,
            cancellation_token=tok,
        )
    assert seen == ["a"]  # stopped immediately, did not consume "b"/"c"


def test_chat_sends_tool_choice_and_response_format() -> None:
    tool_choice = {"type": "function", "function": {"name": "emit_json"}}
    tools = [
        {
            "type": "function",
            "function": {
                "name": "emit_json",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["tools"] == tools
        assert body["tool_choice"] == tool_choice
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    resp = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        tool_choice=tool_choice,
        response_format={"type": "json_object"},
    )

    assert resp.content == "{}"


def test_tool_call_arguments_non_json() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        data = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "shell_run", "arguments": "not-json"},
                            }
                        ],
                    }
                }
            ]
        }
        return httpx.Response(200, json=data)

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.tool_calls[0].arguments["_raw_arguments"] == "not-json"


def test_normalizes_non_stream_content_arrays_to_text() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "Hello"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "https://example.com/image.png"},
                                },
                                {"type": "output_text", "text": " world"},
                            ]
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.content == "Hello world"


def test_non_stream_content_array_keeps_tool_calls() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": [{"type": "text", "text": "ok"}],
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "fs_read",
                                        "arguments": json.dumps({"path": "README.md"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.content == "ok"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "fs_read"


def test_deepseek_reasoning_content_round_trips_for_tool_calls() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            assert body["thinking"] == {"type": "enabled"}
            assert "enable_thinking" not in body
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "reasoning_content": "hidden reasoning state",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "fs_read",
                                            "arguments": json.dumps({"path": "README.md"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )

        assistant_message = body["messages"][1]
        assert PROVIDER_METADATA_KEY not in assistant_message
        assert assistant_message["reasoning_content"] == "hidden reasoning state"
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://api.deepseek.com",
        api_key="test",
        model="deepseek-v4-pro",
        enable_thinking=True,
        transport=transport,
    )

    first = client.chat(messages=[{"role": "user", "content": "read"}], tools=[])
    assert first.content == ""
    assert first.provider_metadata == {"deepseek": {"reasoning_content": "hidden reasoning state"}}
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
    assert PROVIDER_METADATA_KEY in assistant_message

    second = client.chat(
        messages=[
            {"role": "user", "content": "read"},
            assistant_message,
            {"role": "tool", "tool_call_id": "call_1", "content": '{"content":"x"}'},
        ],
        tools=[],
    )
    assert second.content == "ok"


def test_provider_metadata_is_stripped_for_non_deepseek_transports() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assistant_message = body["messages"][0]
        assert PROVIDER_METADATA_KEY not in assistant_message
        assert "reasoning_content" not in assistant_message
        assert "reasoning" not in assistant_message
        assert "reasoning_details" not in assistant_message
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    resp = client.chat(
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [],
                "reasoning_content": "must not leak",
                "reasoning": "must not leak",
                "reasoning_details": [{"type": "reasoning.encrypted", "data": "must not leak"}],
                PROVIDER_METADATA_KEY: {"deepseek": {"reasoning_content": "must not leak"}},
            }
        ],
        tools=[],
    )
    assert resp.content == "ok"


def test_gemini_tool_call_extra_content_round_trips_for_tool_calls() -> None:
    calls: list[dict[str, object]] = []
    extra_content = {"google": {"thought_signature": "sig-1"}}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "extra_content": extra_content,
                                        "function": {
                                            "name": "fs_read",
                                            "arguments": json.dumps({"path": "README.md"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )

        assistant_message = body["messages"][1]
        assert PROVIDER_METADATA_KEY not in assistant_message
        assert assistant_message["tool_calls"][0]["extra_content"] == extra_content
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatClient(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key="test",
        model="gemini-3.1-pro-preview",
        transport=httpx.MockTransport(handler),
    )

    first = client.chat(messages=[{"role": "user", "content": "read"}], tools=[])
    assert first.tool_calls[0].provider_metadata == {"gemini": {"extra_content": extra_content}}
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
    assert PROVIDER_METADATA_KEY in assistant_message
    assert "extra_content" not in assistant_message["tool_calls"][0]

    second = client.chat(
        messages=[
            {"role": "user", "content": "read"},
            assistant_message,
            {"role": "tool", "tool_call_id": "call_1", "content": '{"content":"x"}'},
        ],
        tools=[],
    )
    assert second.content == "ok"


def test_gemini_tool_call_extra_content_is_stripped_for_other_transports() -> None:
    extra_content = {"google": {"thought_signature": "sig-1"}}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assistant_message = body["messages"][0]
        assert PROVIDER_METADATA_KEY not in assistant_message
        assert "extra_content" not in assistant_message["tool_calls"][0]
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    resp = client.chat(
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "extra_content": extra_content,
                        "function": {
                            "name": "fs_read",
                            "arguments": json.dumps({"path": "README.md"}),
                        },
                    }
                ],
                PROVIDER_METADATA_KEY: {
                    "_tool_calls": [
                        {
                            "id": "call_1",
                            "index": 0,
                            "metadata": {"gemini": {"extra_content": extra_content}},
                        }
                    ]
                },
            }
        ],
        tools=[],
    )
    assert resp.content == "ok"


@pytest.mark.parametrize(
    ("base_url", "model"),
    [
        ("https://api.openai.com/v1", "gpt-5"),
        ("https://example-resource.openai.azure.com/openai/deployments/main", "gpt-5"),
        ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.1-pro-preview"),
        ("https://api.mistral.ai/v1", "mistral-large-latest"),
    ],
)
def test_reasoning_effort_is_sent_for_supported_openai_style_providers(
    base_url: str,
    model: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["reasoning_effort"] == "high"
        assert "enable_thinking" not in body
        assert "thinking" not in body
        assert "reasoning" not in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatClient(
        base_url=base_url,
        api_key="test",
        model=model,
        reasoning_effort="high",
        transport=httpx.MockTransport(handler),
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.content == "ok"


@pytest.mark.parametrize(
    ("model", "effort", "expected"),
    [
        ("gemini-3.1-pro-preview", "minimal", "minimal"),
        ("gemini-3.1-pro-preview", "medium", "medium"),
        ("gemini-3.1-pro-preview", "high", "high"),
        ("gemini-3.1-pro-preview", "none", None),
        ("gemini-3-flash-preview", "minimal", "minimal"),
        ("gemini-3-flash-preview", "medium", "medium"),
        ("gemini-2.5-flash", "none", "none"),
        ("gemini-2.5-flash-lite", "none", "none"),
        ("gemini-2.5-pro", "none", None),
        ("gemini-3.1-pro-preview", "xhigh", None),
    ],
)
def test_gemini_reasoning_effort_is_model_safe(
    model: str,
    effort: str,
    expected: str | None,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        if expected is None:
            assert "reasoning_effort" not in body
        else:
            assert body["reasoning_effort"] == expected
        assert "enable_thinking" not in body
        assert "thinking" not in body
        assert "reasoning" not in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatClient(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key="test",
        model=model,
        reasoning_effort=effort,
        transport=httpx.MockTransport(handler),
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.content == "ok"


@pytest.mark.parametrize(
    ("base_url", "model"),
    [
        ("https://api.x.ai/v1", "grok-4"),
        ("https://example.com/v1", "unknown-reasoning-model"),
    ],
)
def test_reasoning_effort_is_omitted_for_unsupported_transports(
    base_url: str,
    model: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert "reasoning_effort" not in body
        assert "enable_thinking" not in body
        assert "thinking" not in body
        assert "reasoning" not in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatClient(
        base_url=base_url,
        api_key="test",
        model=model,
        enable_thinking=True,
        reasoning_effort="high",
        transport=httpx.MockTransport(handler),
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.content == "ok"


def test_openrouter_reasoning_round_trips_for_tool_calls() -> None:
    calls: list[dict[str, object]] = []
    reasoning_details = [{"type": "reasoning.encrypted", "data": "encrypted-state"}]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            assert body["reasoning"] == {"effort": "high"}
            assert "reasoning_effort" not in body
            assert "enable_thinking" not in body
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "reasoning": "hidden openrouter reasoning",
                                "reasoning_details": reasoning_details,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "fs_read",
                                            "arguments": json.dumps({"path": "README.md"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )

        assistant_message = body["messages"][1]
        assert PROVIDER_METADATA_KEY not in assistant_message
        assert assistant_message["reasoning"] == "hidden openrouter reasoning"
        assert assistant_message["reasoning_details"] == reasoning_details
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatClient(
        base_url="https://openrouter.ai/api/v1",
        api_key="test",
        model="openai/gpt-5",
        reasoning_effort="high",
        transport=httpx.MockTransport(handler),
    )

    first = client.chat(messages=[{"role": "user", "content": "read"}], tools=[])
    assert first.provider_metadata == {
        "openrouter": {
            "reasoning": "hidden openrouter reasoning",
            "reasoning_details": reasoning_details,
        }
    }
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
    assert PROVIDER_METADATA_KEY in assistant_message

    second = client.chat(
        messages=[
            {"role": "user", "content": "read"},
            assistant_message,
            {"role": "tool", "tool_call_id": "call_1", "content": '{"content":"x"}'},
        ],
        tools=[],
    )
    assert second.content == "ok"


def test_http_error_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="nope")

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    with pytest.raises(LLMError):
        client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])


def test_parses_usage_and_response_model() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-5-nano-2026-01-01",
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 7,
                    "total_tokens": 19,
                },
                "choices": [{"message": {"content": "ok"}}],
            },
        )

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="gpt-5-nano",
        transport=transport,
    )
    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.response_model == "gpt-5-nano-2026-01-01"
    assert resp.usage is not None
    assert resp.usage.prompt_tokens == 12
    assert resp.usage.completion_tokens == 7
    assert resp.usage.total_tokens == 19


def test_parses_cached_prompt_tokens_from_usage_details() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-5-nano-2026-01-01",
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 4,
                    "total_tokens": 24,
                    "prompt_tokens_details": {
                        "cached_tokens": 14,
                    },
                },
                "choices": [{"message": {"content": "ok"}}],
            },
        )

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="gpt-5-nano",
        transport=transport,
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.usage is not None
    assert resp.usage.cached_prompt_tokens == 14


def test_omits_prompt_cache_fields_by_default() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert "prompt_cache_key" not in body
        assert "prompt_cache_retention" not in body
        assert "enable_thinking" not in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.content == "ok"


def test_includes_prompt_cache_fields_when_configured() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["prompt_cache_key"] == "repo-main"
        assert body["prompt_cache_retention"] == "24h"
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        prompt_cache_key="repo-main",
        prompt_cache_retention="24h",
        transport=transport,
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.content == "ok"


def test_includes_enable_thinking_when_configured() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["enable_thinking"] is False
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key="test",
        model="qwen3.5-plus",
        enable_thinking=False,
        transport=transport,
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[])
    assert resp.content == "ok"


def test_chat_sanitizes_surrogate_escaped_message_content() -> None:
    original = "θελω να φτιαξουμε ενα website με AI news"
    surrogate_text = _surrogate_escaped_text(original)
    assert surrogate_text != original
    assert any(0xD800 <= ord(ch) <= 0xDFFF for ch in surrogate_text)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["messages"][0]["content"] == original
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    resp = client.chat(messages=[{"role": "user", "content": surrogate_text}], tools=[])
    assert resp.content == "ok"


def test_stream_parses_content_and_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}

        events = [
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "fs_read", "arguments": '{"path":"REA'},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": 'DME.md","max_bytes":10}'}}
                            ]
                        }
                    }
                ]
            },
        ]
        body_sse = "".join(f"data: {json.dumps(e)}\n" for e in events) + "data: [DONE]\n"
        return httpx.Response(200, content=body_sse.encode("utf-8"))

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    chunks: list[str] = []
    resp = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        stream=True,
        on_text_delta=chunks.append,
    )
    assert resp.content == "Hello"
    assert chunks == ["Hel", "lo"]
    assert len(resp.tool_calls) == 1
    tc = resp.tool_calls[0]
    assert tc.id == "call_1"
    assert tc.name == "fs_read"
    assert tc.arguments == {"path": "README.md", "max_bytes": 10}


def test_stream_gemini_tool_call_extra_content_round_trips() -> None:
    calls: list[dict[str, object]] = []
    extra_content = {"google": {"thought_signature": "sig-stream"}}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            event = {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "extra_content": extra_content,
                                    "function": {
                                        "name": "fs_read",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
            body_sse = f"data: {json.dumps(event)}\n" + "data: [DONE]\n"
            return httpx.Response(200, content=body_sse.encode("utf-8"))

        assistant_message = body["messages"][1]
        assert PROVIDER_METADATA_KEY not in assistant_message
        assert assistant_message["tool_calls"][0]["extra_content"] == extra_content
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatClient(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        api_key="test",
        model="gemini-3.1-pro-preview",
        transport=httpx.MockTransport(handler),
    )

    first = client.chat(messages=[{"role": "user", "content": "read"}], tools=[], stream=True)
    assert first.tool_calls[0].provider_metadata == {"gemini": {"extra_content": extra_content}}
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

    second = client.chat(
        messages=[
            {"role": "user", "content": "read"},
            assistant_message,
            {"role": "tool", "tool_call_id": "call_1", "content": '{"content":"x"}'},
        ],
        tools=[],
    )
    assert second.content == "ok"


def test_stream_normalizes_content_arrays_and_ignores_non_text_parts() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        events = [
            {
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {"type": "text", "text": "Hel"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "https://example.com/image.png"},
                                },
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {"type": "output_text", "text": "lo"},
                            ]
                        }
                    }
                ]
            },
        ]
        body_sse = "".join(f"data: {json.dumps(e)}\n" for e in events) + "data: [DONE]\n"
        return httpx.Response(200, content=body_sse.encode("utf-8"))

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    chunks: list[str] = []
    resp = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        stream=True,
        on_text_delta=chunks.append,
    )
    assert resp.content == "Hello"
    assert chunks == ["Hel", "lo"]


def test_stream_dedupes_cumulative_content_and_tool_argument_deltas() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        events = [
            {"choices": [{"delta": {"content": "I'll help you build an AI news website."}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "content": (
                                "I'll help you build an AI news website. "
                                "Let me start by reviewing the existing web assets."
                            )
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "fs_read",
                                        "arguments": '{"path":"README.md"',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '{"path":"README.md","max_bytes":10}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
        ]
        body_sse = "".join(f"data: {json.dumps(e)}\n" for e in events) + "data: [DONE]\n"
        return httpx.Response(200, content=body_sse.encode("utf-8"))

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    chunks: list[str] = []
    resp = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        stream=True,
        on_text_delta=chunks.append,
    )
    assert resp.content == (
        "I'll help you build an AI news website. Let me start by reviewing the existing web assets."
    )
    assert chunks == [
        "I'll help you build an AI news website.",
        " Let me start by reviewing the existing web assets.",
    ]
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].arguments == {"path": "README.md", "max_bytes": 10}


def test_stream_dedupes_cumulative_content_with_trailing_whitespace_drift() -> None:
    first = (
        "Now I understand the repository structure. It's a simple content repository "
        "with Markdown files. Let me "
    )
    second = (
        "Now I understand the repository structure. It's a simple content repository "
        "with Markdown files. Let me\ncreate the AI news website section following the approved plan."
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        events = [
            {"choices": [{"delta": {"content": first}}]},
            {"choices": [{"delta": {"content": second}}]},
        ]
        body_sse = "".join(f"data: {json.dumps(e)}\n" for e in events) + "data: [DONE]\n"
        return httpx.Response(200, content=body_sse.encode("utf-8"))

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    chunks: list[str] = []
    resp = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        stream=True,
        on_text_delta=chunks.append,
    )
    assert resp.content == (
        "Now I understand the repository structure. It's a simple content repository "
        "with Markdown files. Let me \ncreate the AI news website section following the approved plan."
    )
    assert chunks == [
        first,
        "\ncreate the AI news website section following the approved plan.",
    ]


def test_stream_dedupes_cumulative_content_restarted_after_leading_newline() -> None:
    first = "Let me verify the website renders correctly by checking the generated HTML export:"
    second = (
        "\nLet me verify the website renders correctly by checking the generated HTML export:\n"
        "Step 11: Read File"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        events = [
            {"choices": [{"delta": {"content": first}}]},
            {"choices": [{"delta": {"content": second}}]},
        ]
        body_sse = "".join(f"data: {json.dumps(e)}\n" for e in events) + "data: [DONE]\n"
        return httpx.Response(200, content=body_sse.encode("utf-8"))

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    chunks: list[str] = []
    resp = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        stream=True,
        on_text_delta=chunks.append,
    )
    assert resp.content == first + "\nStep 11: Read File"
    assert chunks == [first, "\nStep 11: Read File"]


def test_stream_dedupes_tail_overlap_followed_by_restart() -> None:
    first = (
        "Now I understand the repository structure. It's a simple content repository "
        "with Markdown files. Let me"
    )
    second = (
        "Let meNow I understand the repository structure. It's a simple content repository "
        "with Markdown files. Let me\ncreate the AI news website section following the approved plan."
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        events = [
            {"choices": [{"delta": {"content": first}}]},
            {"choices": [{"delta": {"content": second}}]},
        ]
        body_sse = "".join(f"data: {json.dumps(e)}\n" for e in events) + "data: [DONE]\n"
        return httpx.Response(200, content=body_sse.encode("utf-8"))

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    chunks: list[str] = []
    resp = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        stream=True,
        on_text_delta=chunks.append,
    )
    assert resp.content == (
        "Now I understand the repository structure. It's a simple content repository "
        "with Markdown files. Let me\ncreate the AI news website section following the approved plan."
    )
    assert chunks == [
        first,
        "\ncreate the AI news website section following the approved plan.",
    ]


def test_stream_dedupes_duplicate_full_sentence_inside_cumulative_chunk() -> None:
    first = (
        "Both directories are empty. Now I'll create the AI news website structure. Let me create:"
    )
    second = (
        first
        + "\n"
        + first
        + "\n\n1 Sample AI news markdown files\n2 An HTML file to display the news"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        events = [
            {"choices": [{"delta": {"content": first}}]},
            {"choices": [{"delta": {"content": second}}]},
        ]
        body_sse = "".join(f"data: {json.dumps(e)}\n" for e in events) + "data: [DONE]\n"
        return httpx.Response(200, content=body_sse.encode("utf-8"))

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    chunks: list[str] = []
    resp = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        stream=True,
        on_text_delta=chunks.append,
    )
    assert resp.content == (
        first + "\n\n1 Sample AI news markdown files\n2 An HTML file to display the news"
    )
    assert chunks == [
        first,
        "\n\n1 Sample AI news markdown files\n2 An HTML file to display the news",
    ]


def test_stream_dedupes_exact_duplicate_progress_sentence_from_cumulative_chunk() -> None:
    first = "I found the issues! Let me fix both:"
    second = first + "\n" + first

    def handler(_request: httpx.Request) -> httpx.Response:
        events = [
            {"choices": [{"delta": {"content": first}}]},
            {"choices": [{"delta": {"content": second}}]},
        ]
        body_sse = "".join(f"data: {json.dumps(e)}\n" for e in events) + "data: [DONE]\n"
        return httpx.Response(200, content=body_sse.encode("utf-8"))

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    chunks: list[str] = []
    resp = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        stream=True,
        on_text_delta=chunks.append,
    )
    assert resp.content == first
    assert chunks == [first]


def test_stream_suppresses_alternate_full_answer_restart() -> None:
    first = (
        "Yes, I can see the image. It shows the **Recycle Bin** icon with a blue "
        'recycling symbol and the label "Recycle Bin" below it.\n\n'
        "Do you need help with this image, or should we work in the repository?"
    )
    second = (
        'Yes, I can see the image. It shows the "Recycle Bin" icon with the blue '
        "recycling symbol.\n\n"
        "Is there something specific you want me to do with it, or should we work "
        "in the repository?"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        events = [
            {"choices": [{"delta": {"content": first}}]},
            {"choices": [{"delta": {"content": second}}]},
        ]
        body_sse = "".join(f"data: {json.dumps(e)}\n" for e in events) + "data: [DONE]\n"
        return httpx.Response(200, content=body_sse.encode("utf-8"))

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    chunks: list[str] = []
    resp = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        stream=True,
        on_text_delta=chunks.append,
    )
    assert resp.content == first
    assert chunks == [first]


def test_stream_http_error_raises() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="stream unsupported")

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    with pytest.raises(LLMError):
        client.chat(messages=[{"role": "user", "content": "hi"}], tools=[], stream=True)


def test_stream_http_error_handles_unread_stream_body() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="stream body unread")

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    with pytest.raises(LLMError) as exc:
        client.chat(messages=[{"role": "user", "content": "hi"}], tools=[], stream=True)
    assert "stream body unread" in str(exc.value)
    assert "without having called `read()`" not in str(exc.value)


def test_stream_retries_without_stream_options_when_unsupported() -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        if len(calls) == 1:
            assert body.get("stream_options") == {"include_usage": True}
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unsupported field: stream_options",
                    }
                },
            )
        assert "stream_options" not in body
        events = [
            {
                "model": "gpt-5-nano",
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
                "choices": [{"delta": {"content": "ok"}}],
            }
        ]
        body_sse = "".join(f"data: {json.dumps(e)}\n" for e in events) + "data: [DONE]\n"
        return httpx.Response(200, content=body_sse.encode("utf-8"))

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[], stream=True)
    assert resp.content == "ok"
    assert resp.usage is not None
    assert resp.usage.total_tokens == 3
    assert len(calls) == 2


class _TruncatedSseStream(httpx.SyncByteStream):
    def __iter__(self):  # type: ignore[no-untyped-def]
        yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
        raise httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body"
        )


def test_stream_transport_truncation_retries_without_partial_tool_batch() -> None:
    attempts = 0
    chunks: list[str] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, stream=_TruncatedSseStream())
        body = 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, content=body.encode("utf-8"))

    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=httpx.MockTransport(handler),
        provider_retry_settings=ProviderRetrySettings(max_retries=1),
        provider_sleep_fn=lambda _seconds: None,
        provider_random_fn=lambda: 0.5,
    )

    resp = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        stream=True,
        on_text_delta=chunks.append,
    )

    assert attempts == 2
    assert resp.content == "ok"
    assert resp.tool_calls == []
    assert chunks == ["ok"]
    assert resp.raw["stream_restart_count"] == 1


def test_stream_parses_cached_prompt_tokens() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        events = [
            {
                "model": "gpt-5-nano",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "prompt_tokens_details": {"cached_tokens": 6},
                },
                "choices": [{"delta": {"content": "ok"}}],
            }
        ]
        body_sse = "".join(f"data: {json.dumps(e)}\n" for e in events) + "data: [DONE]\n"
        return httpx.Response(200, content=body_sse.encode("utf-8"))

    transport = httpx.MockTransport(handler)
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="test-model",
        transport=transport,
    )

    resp = client.chat(messages=[{"role": "user", "content": "hi"}], tools=[], stream=True)
    assert resp.usage is not None
    assert resp.usage.cached_prompt_tokens == 6


def _content_handler(message: dict[str, object]):  # type: ignore[no-untyped-def]
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": message}]})

    return handler


def _trial_client(handler) -> OpenAICompatClient:  # type: ignore[no-untyped-def]
    return OpenAICompatClient(
        base_url=_SYLLIPTOR_TRIAL_BASE_URL,
        api_key="test",
        model="mimo-v2.5-pro",
        transport=httpx.MockTransport(handler),
    )


def test_non_stream_folds_reasoning_into_empty_content() -> None:
    # MiMo reasoning models can answer entirely in the reasoning channel with an
    # empty `content`. The fold surfaces that text so the turn does not degrade
    # to the generic clarification fallback.
    handler = _content_handler({"content": "", "reasoning": "Hi! How can I help?"})
    resp = _trial_client(handler).chat(messages=[{"role": "user", "content": "hi"}])
    assert resp.content == "Hi! How can I help?"


def test_non_stream_folds_reasoning_content_into_whitespace_content() -> None:
    handler = _content_handler({"content": "   ", "reasoning_content": "Hello there."})
    resp = _trial_client(handler).chat(messages=[{"role": "user", "content": "hi"}])
    assert resp.content == "Hello there."


def test_non_stream_keeps_empty_when_no_reasoning() -> None:
    # Genuinely empty completion with no reasoning must stay empty so the
    # legitimate clarification fallback still triggers upstream.
    handler = _content_handler({"content": ""})
    resp = _trial_client(handler).chat(messages=[{"role": "user", "content": "hi"}])
    assert resp.content == ""


def test_non_stream_does_not_overwrite_real_content_with_reasoning() -> None:
    handler = _content_handler({"content": "real answer", "reasoning": "scratch work"})
    resp = _trial_client(handler).chat(messages=[{"role": "user", "content": "hi"}])
    assert resp.content == "real answer"


def test_non_stream_empty_content_with_tool_calls_is_not_folded() -> None:
    # Tool-call turns legitimately carry empty content; the fold must not run.
    message = {
        "content": "",
        "reasoning": "deciding to call a tool",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "fs_read", "arguments": json.dumps({"path": "README.md"})},
            }
        ],
    }
    resp = _trial_client(_content_handler(message)).chat(
        messages=[{"role": "user", "content": "hi"}], tools=[]
    )
    assert resp.content == ""
    assert len(resp.tool_calls) == 1


def test_stream_folds_reasoning_into_empty_content() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        events = [
            {"choices": [{"delta": {"reasoning": "Hi! "}}]},
            {"choices": [{"delta": {"reasoning": "How can I help?"}}]},
        ]
        body_sse = "".join(f"data: {json.dumps(e)}\n" for e in events) + "data: [DONE]\n"
        return httpx.Response(200, text=body_sse)

    resp = _trial_client(handler).chat(messages=[{"role": "user", "content": "hi"}], stream=True)
    assert resp.content == "Hi! How can I help?"


def test_stream_keeps_content_over_reasoning() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        events = [
            {"choices": [{"delta": {"reasoning": "thinking"}}]},
            {"choices": [{"delta": {"content": "real "}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
        ]
        body_sse = "".join(f"data: {json.dumps(e)}\n" for e in events) + "data: [DONE]\n"
        return httpx.Response(200, text=body_sse)

    resp = _trial_client(handler).chat(messages=[{"role": "user", "content": "hi"}], stream=True)
    assert resp.content == "real answer"


def test_sylliptor_trial_proxy_resolves_to_openrouter_provider_key() -> None:
    assert _provider_key_from_base_url(_SYLLIPTOR_TRIAL_BASE_URL) == "openrouter"
    # A plain supabase host without the LLM proxy path must NOT be captured.
    assert _provider_key_from_base_url("https://other.supabase.co/rest/v1") != "openrouter"


def test_trial_proxy_captures_reasoning_into_provider_metadata() -> None:
    # Because the trial proxy is classified as openrouter, a reasoning field on a
    # normal (non-empty content) completion is preserved in provider_metadata.
    handler = _content_handler({"content": "ok", "reasoning": "hidden reasoning"})
    resp = _trial_client(handler).chat(messages=[{"role": "user", "content": "hi"}])
    assert resp.content == "ok"
    assert resp.provider_metadata == {"openrouter": {"reasoning": "hidden reasoning"}}
