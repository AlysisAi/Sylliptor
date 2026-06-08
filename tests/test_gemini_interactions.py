from __future__ import annotations

import json

import httpx
import pytest

from sylliptor_agent_cli.config import AppConfig, set_config_value
from sylliptor_agent_cli.llm.gemini_interactions import (
    GEMINI_INTERACTIONS_CONFIG_FLAG,
    GEMINI_INTERACTIONS_EXPERIMENT_ENV,
    GeminiInteractionsClient,
    gemini_interactions_enabled,
)
from sylliptor_agent_cli.llm.metadata import GEMINI_INTERACTIONS_PROVIDER_METADATA_KEY
from sylliptor_agent_cli.llm.types import LLMError


def test_gemini_interactions_feature_flag_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv(GEMINI_INTERACTIONS_EXPERIMENT_ENV, raising=False)

    assert gemini_interactions_enabled(AppConfig()) is False


def test_gemini_interactions_feature_flag_accepts_env(monkeypatch) -> None:
    cfg = AppConfig()
    cfg.experimental_gemini_interactions_enabled = False
    monkeypatch.setenv(GEMINI_INTERACTIONS_EXPERIMENT_ENV, "1")

    assert gemini_interactions_enabled(cfg) is True


def test_gemini_interactions_feature_flag_accepts_config(monkeypatch) -> None:
    monkeypatch.delenv(GEMINI_INTERACTIONS_EXPERIMENT_ENV, raising=False)
    cfg = set_config_value(AppConfig(), GEMINI_INTERACTIONS_CONFIG_FLAG, "true")

    assert cfg.experimental_gemini_interactions_enabled is True
    assert gemini_interactions_enabled(cfg) is True


def test_gemini_interactions_text_only_client_maps_request_and_response() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["api_revision"] = request.headers.get("Api-Revision")
        captured["api_key_header"] = request.headers.get("x-goog-api-key")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "id": "interaction-123",
                "model": "gemini-2.5-flash",
                "outputs": [{"type": "text", "text": "hello from interactions"}],
                "usage": {
                    "total_input_tokens": 3,
                    "total_output_tokens": 5,
                    "total_tokens": 8,
                },
            },
        )

    client = GeminiInteractionsClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="gemini-secret-key",
        model="gemini-2.5-flash",
        transport=httpx.MockTransport(handler),
    )

    response = client.chat(
        messages=[
            {"role": "user", "content": "Say hello."},
        ],
        temperature=0.0,
        max_tokens=20,
    )

    assert captured["method"] == "POST"
    assert captured["url"] == "https://generativelanguage.googleapis.com/v1beta/interactions"
    assert "gemini-secret-key" not in str(captured["url"])
    assert captured["api_revision"] == "2026-05-20"
    assert captured["api_key_header"] == "gemini-secret-key"
    assert captured["body"] == {
        "input": "Say hello.",
        "model": "gemini-2.5-flash",
        "temperature": 0.0,
        "max_output_tokens": 20,
    }
    assert response.content == "hello from interactions"
    assert response.tool_calls == []
    assert response.response_model == "gemini-2.5-flash"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 8
    assert response.provider_metadata == {
        GEMINI_INTERACTIONS_PROVIDER_METADATA_KEY: {
            "interaction_id": "interaction-123",
        }
    }


def test_gemini_interactions_text_only_client_rejects_unimplemented_surfaces() -> None:
    client = GeminiInteractionsClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="gemini-secret-key",
        model="gemini-2.5-flash",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={})),
    )

    with pytest.raises(LLMError, match="does not support streaming"):
        client.chat(messages=[{"role": "user", "content": "hi"}], stream=True)

    with pytest.raises(LLMError, match="does not support tools"):
        client.chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
        )


def test_gemini_interactions_client_raises_clear_provider_error() -> None:
    client = GeminiInteractionsClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="gemini-secret-key",
        model="gemini-2.5-flash",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                400,
                json={"error": {"message": "Api-Revision header is required"}},
            )
        ),
    )

    with pytest.raises(LLMError, match="Api-Revision header is required"):
        client.chat(messages=[{"role": "user", "content": "hi"}])


def test_gemini_interactions_client_rejects_multi_turn_history() -> None:
    client = GeminiInteractionsClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="gemini-secret-key",
        model="gemini-2.5-flash",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={})),
    )

    with pytest.raises(LLMError, match="previous_interaction_id continuation"):
        client.chat(
            messages=[
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
                {"role": "user", "content": "second"},
            ]
        )


def test_gemini_interactions_client_rejects_system_instructions_until_mapped() -> None:
    client = GeminiInteractionsClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="gemini-secret-key",
        model="gemini-2.5-flash",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={})),
    )

    with pytest.raises(LLMError, match="system/developer instructions"):
        client.chat(
            messages=[
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ]
        )


def test_gemini_interactions_client_rejects_unhandled_actions() -> None:
    client = GeminiInteractionsClient(
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="gemini-secret-key",
        model="gemini-2.5-flash",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "id": "interaction-123",
                    "outputs": [{"type": "text", "text": "I need a tool."}],
                    "actions": [{"type": "function_call", "name": "lookup", "args": {}}],
                },
            )
        ),
    )

    with pytest.raises(LLMError, match="only supports text output"):
        client.chat(messages=[{"role": "user", "content": "hi"}])
