from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from typer.testing import CliRunner

from sylliptor_agent_cli.cli import app as sylliptor_app
from sylliptor_agent_cli.config import AppConfig, save_config
from sylliptor_agent_cli.llm.openai_compat import OpenAICompatClient
from sylliptor_agent_cli.llm.openai_responses import (
    WebSearchCitation,
    WebSearchResponse,
    WebSearchSource,
)
from sylliptor_agent_cli.llm.provider_limits import ProviderRetrySettings
from sylliptor_agent_cli.provider_telemetry import (
    last_provider_call_summary,
    last_web_search_summary,
    record_web_search_call,
    reset_provider_telemetry_for_tests,
)
from sylliptor_agent_cli.tools.web_search import web_search


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "SYLLIPTOR_CONFIG_DIR": str(tmp_path),
        "SYLLIPTOR_API_KEY": "",
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "GEMINI_API_KEY": "",
        "TAVILY_API_KEY": "",
    }


def test_provider_call_telemetry_redacts_secrets_and_hidden_metadata() -> None:
    reset_provider_telemetry_for_tests()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "deepseek-chat",
                "choices": [
                    {
                        "message": {
                            "content": "visible answer",
                            "reasoning_content": "hidden provider reasoning",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            },
        )

    client = OpenAICompatClient(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-secret-provider-key",
        model="deepseek-chat",
        provider_key="deepseek",
        transport=httpx.MockTransport(handler),
    )
    response = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "look up sk-tool-argument-secret",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    assert response.provider_metadata
    summary = last_provider_call_summary()
    assert summary is not None
    assert summary["provider_key"] == "deepseek"
    assert summary["protocol"] == "openai_compat"
    assert summary["base_url_host"] == "api.deepseek.com"
    assert summary["web_search"]["backend_kind"] == "external"
    assert summary["provider_metadata_present"] is True
    assert summary["usage"]["total_tokens"] == 5
    rendered = json.dumps(summary, sort_keys=True)
    assert "sk-secret-provider-key" not in rendered
    assert "sk-tool-argument-secret" not in rendered
    assert "hidden provider reasoning" not in rendered
    assert "https://api.deepseek.com/v1" not in rendered


def test_streaming_telemetry_counts_events_deltas_and_first_token_latency(monkeypatch) -> None:
    reset_provider_telemetry_for_tests()
    timestamps = iter([1000.0, 1012.0, 1050.0])
    monkeypatch.setattr(
        "sylliptor_agent_cli.provider_telemetry.telemetry_clock_ms",
        lambda: next(timestamps),
    )
    deltas: list[str] = []

    def sse_event(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    body = "".join(
        [
            sse_event({"model": "test-model", "choices": [{"delta": {"content": "Hel"}}]}),
            sse_event(
                {
                    "choices": [{"delta": {"content": "lo"}}],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 2,
                        "total_tokens": 3,
                    },
                }
            ),
            "data: [DONE]\n\n",
        ]
    )
    client = OpenAICompatClient(
        base_url="https://example.com/v1",
        api_key="test-key",
        model="test-model",
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=body)),
    )

    response = client.chat(
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        on_text_delta=deltas.append,
    )

    assert response.content == "Hello"
    assert deltas == ["Hel", "lo"]
    summary = last_provider_call_summary()
    assert summary is not None
    assert summary["stream"] is True
    assert summary["streaming"]["event_count"] == 2
    assert summary["streaming"]["text_delta_count"] == 2
    assert summary["streaming"]["first_token_latency_ms"] == 12
    assert summary["streaming"]["final_latency_ms"] == 50
    assert summary["usage"]["total_tokens"] == 3


def test_provider_retry_telemetry_uses_fake_clock_without_sleep(monkeypatch) -> None:
    reset_provider_telemetry_for_tests()
    timestamps = iter([2000.0, 2033.0])
    monkeypatch.setattr(
        "sylliptor_agent_cli.provider_telemetry.telemetry_clock_ms",
        lambda: next(timestamps),
    )
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, text="rate limit")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = OpenAICompatClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-test",
        provider_key="openai",
        transport=httpx.MockTransport(handler),
        provider_retry_settings=ProviderRetrySettings(
            max_retries=1,
            base_delay_seconds=0.0,
            max_delay_seconds=0.0,
        ),
        provider_sleep_fn=lambda _seconds: None,
        provider_random_fn=lambda: 0.5,
    )

    response = client.chat(messages=[{"role": "user", "content": "hi"}])

    assert response.content == "ok"
    assert attempts == 2
    summary = last_provider_call_summary()
    assert summary is not None
    assert summary["retry_count"] == 1
    assert summary["retry_reasons"] == ["provider_throttled"]
    assert summary["status_category"] == "success"
    assert summary["latency_ms"] == 33


def test_web_search_telemetry_records_native_and_external_labels(monkeypatch) -> None:
    reset_provider_telemetry_for_tests()
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)

    class _FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def web_search(self, **_kwargs: object) -> WebSearchResponse:
            return WebSearchResponse(
                answer="answer",
                citations=[
                    WebSearchCitation(title="Docs", url="https://example.test/docs"),
                ],
                sources=[WebSearchSource(url="https://example.test/docs", title="Docs")],
                queries=["docs query"],
                raw={"id": "resp_search"},
                response_id="resp_search",
                model="search-model",
            )

    result = web_search(
        query="docs query",
        cfg=AppConfig(
            model="gpt-test",
            base_url="https://api.openai.com/v1",
            web_search_mode="native",
            web_search_adapter="openai_responses",
        ),
        api_key="main-key",
        client_factory=_FakeClient,
    )

    assert result["provider_hosted_search"] is True
    native = last_web_search_summary()
    assert native is not None
    assert native["provider_hosted_search"] is True
    assert native["external_provider_name"] == ""
    assert native["source_count"] == 1
    assert native["citation_count"] == 1
    assert native["query_count"] == 1

    record_web_search_call(
        protocol="openai_compat",
        provider_key="tavily",
        model=None,
        web_search_mode="auto",
        web_search_adapter="tavily",
        provider_hosted_search=False,
        external_provider_name="tavily",
        source_count=2,
        citation_count=2,
        query_count=1,
        fallback_occurred=True,
    )
    external = last_web_search_summary()
    assert external is not None
    assert external["provider_hosted_search"] is False
    assert external["external_provider_name"] == "tavily"
    assert external["fallback_occurred"] is True


def test_doctor_bundle_is_redacted_and_excludes_hidden_values(
    monkeypatch,
    tmp_path: Path,
) -> None:
    reset_provider_telemetry_for_tests()
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", str(tmp_path))
    cfg = AppConfig(
        model="gpt-test",
        base_url="https://api.openai.com/v1",
        api_key_env="OPENAI_API_KEY",
    )
    save_config(cfg)
    record_web_search_call(
        protocol="openai_compat",
        provider_key="tavily",
        model="gpt-test",
        web_search_mode="external",
        web_search_adapter="tavily",
        provider_hosted_search=False,
        external_provider_name="tavily",
        source_count=1,
        citation_count=1,
        query_count=1,
        fallback_occurred=False,
    )

    result = CliRunner().invoke(
        sylliptor_app,
        ["doctor", "bundle", "--redacted"],
        env={**_env(tmp_path), "OPENAI_API_KEY": "sk-openai-secret-value"},
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["redacted"] is True
    assert "provider_diagnostics" in payload
    assert "recent_web_search_calls" in payload
    rendered = json.dumps(payload, sort_keys=True)
    assert "sk-openai-secret-value" not in rendered
    assert "provider_metadata" not in rendered
    assert "tool arguments" in rendered
