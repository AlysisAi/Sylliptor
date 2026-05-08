from __future__ import annotations

import json

import httpx
import pytest

from sylliptor_agent_cli.llm.openai_responses import (
    OpenAIResponsesClient,
    ResponsesError,
)


def _client(transport: httpx.BaseTransport) -> OpenAIResponsesClient:
    return OpenAIResponsesClient(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="search-model",
        transport=transport,
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
