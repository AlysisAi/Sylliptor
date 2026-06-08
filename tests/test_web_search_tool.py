from __future__ import annotations

import json

import httpx
import pytest

from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.llm.openai_responses import (
    WebSearchCitation,
    WebSearchResponse,
    WebSearchSource,
)
from sylliptor_agent_cli.tools.web_search import (
    WebSearchError,
    resolve_web_search_runtime,
    resolve_web_search_runtime_status,
    web_search,
)


def _public_resolver(_host: str, _port: int) -> list[str]:
    return ["93.184.216.34"]


def _configured_cfg(
    *,
    mode: str = "auto",
    base_url: str = "https://api.openai.com/v1",
    web_search_base_url: str | None = None,
) -> AppConfig:
    return AppConfig(
        model="main-model",
        base_url=base_url,
        web_search_mode=mode,
        web_search_base_url=web_search_base_url,
    )


def test_web_search_returns_structured_openai_result_and_dedupes_sources() -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            captured["init"] = kwargs

        def web_search(
            self,
            *,
            query: str,
            allowed_domains: list[str] | None = None,
            external_web_access: bool | None = None,
            include_source_details: bool = True,
        ) -> WebSearchResponse:
            captured["query"] = query
            captured["allowed_domains"] = allowed_domains
            captured["external_web_access"] = external_web_access
            captured["include_source_details"] = include_source_details
            return WebSearchResponse(
                answer="Read the official docs.",
                citations=[
                    WebSearchCitation(
                        title="Docs",
                        url="https://docs.example.com/start",
                        start_index=0,
                        end_index=12,
                    )
                ],
                sources=[
                    WebSearchSource(url="https://docs.example.com/start", title="Docs"),
                    WebSearchSource(url="https://docs.example.com/start", title="Docs duplicate"),
                    WebSearchSource(
                        url="https://changelog.example.com/release", title="Release Notes"
                    ),
                ],
                queries=["docs example"],
                raw={"id": "resp"},
                response_id="resp_1",
                model="search-model",
            )

    result = web_search(
        query="docs example",
        cfg=_configured_cfg(),
        api_key="main-key",
        allowed_domains=["docs.example.com"],
        max_sources=8,
        external_web_access=False,
        client_factory=_FakeClient,
    )

    assert captured["query"] == "docs example"
    assert captured["allowed_domains"] == ["docs.example.com"]
    assert captured["external_web_access"] is False
    assert captured["include_source_details"] is True
    assert result["query"] == "docs example"
    assert result["answer"] == "Read the official docs."
    assert result["backend"] == "openai_responses"
    assert result["protocol"] == "openai_compat"
    assert result["chat_protocol"] == "openai_compat"
    assert result["search_protocol"] == "openai_responses"
    assert result["backend_adapter"] == "openai_responses"
    assert result["provider_hosted_search"] is True
    assert result["external_search_provider"] is None
    assert result["citation_count"] == 1
    assert result["source_count"] == 2
    assert result["model"] == "search-model"
    assert result["response_id"] == "resp_1"
    assert result["sources_truncated"] is False
    assert result["sources"] == [
        {"url": "https://docs.example.com/start", "title": "Docs"},
        {"url": "https://changelog.example.com/release", "title": "Release Notes"},
    ]
    assert result["citations"] == [
        {
            "title": "Docs",
            "url": "https://docs.example.com/start",
            "start_index": 0,
            "end_index": 12,
        }
    ]


def test_web_search_auto_status_prefers_openai_when_conservatively_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    status = resolve_web_search_runtime_status(cfg=_configured_cfg(), api_key="main-key")
    runtime = resolve_web_search_runtime(cfg=_configured_cfg(), api_key="main-key", strict=True)

    assert status.registration_ready is True
    assert status.provider == "openai_responses"
    assert status.availability_label == "available"
    assert runtime is not None
    assert runtime.provider == "openai_responses"


def test_web_search_auto_status_falls_back_to_tavily_when_openai_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    cfg = _configured_cfg(base_url="https://example-proxy.invalid/v1")
    status = resolve_web_search_runtime_status(cfg=cfg, api_key="main-key")
    runtime = resolve_web_search_runtime(cfg=cfg, api_key="main-key", strict=True)

    assert status.registration_ready is True
    assert status.provider == "tavily"
    assert status.base_url is None
    assert status.model is None
    assert runtime is not None
    assert runtime.provider == "tavily"


def test_web_search_auto_status_prefers_native_when_external_is_also_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    status = resolve_web_search_runtime_status(cfg=_configured_cfg(), api_key="main-key")
    runtime = resolve_web_search_runtime(cfg=_configured_cfg(), api_key="main-key", strict=True)

    assert status.registration_ready is True
    assert status.provider == "openai_responses"
    assert runtime is not None
    assert runtime.provider == "openai_responses"


def test_web_search_native_status_never_selects_tavily_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    cfg = _configured_cfg(mode="native", base_url="https://example-proxy.invalid/v1")
    status = resolve_web_search_runtime_status(cfg=cfg, api_key="main-key")

    assert status.mode == "native"
    assert status.registration_ready is False
    assert status.provider is None
    assert status.availability_label == "native-unavailable"
    assert any("Native mode never falls back to Tavily" in note for note in status.notes)
    assert not any("TAVILY_API_KEY" in note for note in status.notes)

    with pytest.raises(
        WebSearchError,
        match="web_search is not available in native mode.*Native mode never falls back to Tavily",
    ):
        resolve_web_search_runtime(cfg=cfg, api_key="main-key", strict=True)


def test_web_search_external_status_selects_tavily_and_ignores_ready_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    cfg = _configured_cfg(mode="external")
    status = resolve_web_search_runtime_status(cfg=cfg, api_key="main-key")
    runtime = resolve_web_search_runtime(cfg=cfg, api_key="main-key", strict=True)

    assert status.mode == "external"
    assert status.registration_ready is True
    assert status.provider == "tavily"
    assert runtime is not None
    assert runtime.provider == "tavily"


def test_web_search_auto_status_uses_dashscope_chat_for_qwen_coding_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    cfg = AppConfig(
        model="qwen3.5-plus",
        base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
        web_search_mode="auto",
    )
    status = resolve_web_search_runtime_status(cfg=cfg, api_key="main-key")
    runtime = resolve_web_search_runtime(cfg=cfg, api_key="main-key", strict=True)

    assert status.registration_ready is True
    assert status.provider == "dashscope_chat"
    assert status.base_url == "https://coding-intl.dashscope.aliyuncs.com/v1"
    assert status.model == "qwen3.5-plus"
    assert runtime is not None
    assert runtime.provider == "dashscope_chat"


def test_web_search_auto_status_uses_dashscope_chat_for_qwen_us_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    cfg = AppConfig(
        model="qwen3.5-plus",
        base_url="https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        web_search_mode="auto",
    )
    status = resolve_web_search_runtime_status(cfg=cfg, api_key="main-key")
    runtime = resolve_web_search_runtime(cfg=cfg, api_key="main-key", strict=True)

    assert status.registration_ready is True
    assert status.provider == "dashscope_chat"
    assert status.base_url == "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
    assert status.model == "qwen3.5-plus"
    assert runtime is not None
    assert runtime.provider == "dashscope_chat"


@pytest.mark.parametrize(
    ("base_url", "model", "expected_provider"),
    [
        ("https://api.moonshot.cn/v1", "kimi-k2.6", "moonshot_kimi"),
        ("https://open.bigmodel.cn/api/paas/v4/", "glm-4.6", "zhipu_web_search"),
        (
            "https://ark.cn-beijing.volces.com/api/v3",
            "doubao-seed-1-6-250615",
            "volcengine_web_search",
        ),
    ],
)
def test_web_search_auto_status_uses_native_chinese_provider_adapters(
    monkeypatch: pytest.MonkeyPatch,
    base_url: str,
    model: str,
    expected_provider: str,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    cfg = AppConfig(model=model, base_url=base_url, web_search_mode="auto")
    status = resolve_web_search_runtime_status(cfg=cfg, api_key="main-key")
    runtime = resolve_web_search_runtime(cfg=cfg, api_key="main-key", strict=True)

    assert status.registration_ready is True
    assert status.provider == expected_provider
    assert runtime is not None
    assert runtime.provider == expected_provider


def test_web_search_explicit_openai_override_uses_only_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", "openai_responses")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    status = resolve_web_search_runtime_status(cfg=_configured_cfg(), api_key="main-key")

    assert status.provider == "openai_responses"
    assert status.registration_ready is True
    assert "explicit adapter selected openai_responses" in status.notes


def test_web_search_explicit_tavily_override_uses_only_tavily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    status = resolve_web_search_runtime_status(cfg=_configured_cfg(), api_key="main-key")

    assert status.provider == "tavily"
    assert status.registration_ready is True
    assert "explicit adapter selected tavily" in status.notes


def test_web_search_native_mode_rejects_explicit_external_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    cfg = AppConfig(
        model="main-model",
        base_url="https://api.openai.com/v1",
        web_search_mode="native",
        web_search_adapter="tavily",
    )
    status = resolve_web_search_runtime_status(cfg=cfg, api_key="main-key")

    assert status.mode == "native"
    assert status.registration_ready is False
    assert status.provider is None
    assert "explicit adapter selected tavily" in status.notes
    assert any("web_search_mode=native is incompatible" in note for note in status.notes)

    with pytest.raises(
        WebSearchError,
        match="web_search_mode=native is incompatible with web_search_adapter=tavily",
    ):
        resolve_web_search_runtime(cfg=cfg, api_key="main-key", strict=True)


def test_web_search_external_mode_rejects_explicit_native_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    cfg = AppConfig(
        model="main-model",
        base_url="https://api.openai.com/v1",
        web_search_mode="external",
        web_search_adapter="openai_responses",
    )
    status = resolve_web_search_runtime_status(cfg=cfg, api_key="main-key")

    assert status.mode == "external"
    assert status.registration_ready is False
    assert status.provider is None
    assert "explicit adapter selected openai_responses" in status.notes
    assert any("web_search_mode=external is incompatible" in note for note in status.notes)

    with pytest.raises(
        WebSearchError,
        match="web_search_mode=external is incompatible with web_search_adapter=openai_responses",
    ):
        resolve_web_search_runtime(cfg=cfg, api_key="main-key", strict=True)


def test_web_search_explicit_dashscope_override_uses_only_dashscope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", "dashscope_chat")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    cfg = AppConfig(
        model="qwen3.5-plus",
        base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
        web_search_mode="auto",
    )
    status = resolve_web_search_runtime_status(cfg=cfg, api_key="main-key")

    assert status.provider == "dashscope_chat"
    assert status.registration_ready is True
    assert "explicit adapter selected dashscope_chat" in status.notes


def test_web_search_explicit_tavily_override_without_key_is_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    cfg = _configured_cfg(base_url="https://example-proxy.invalid/v1")
    status = resolve_web_search_runtime_status(cfg=cfg, api_key="main-key")

    assert status.registration_ready is False
    assert status.provider is None
    assert status.availability_label == "auto-unavailable"
    assert "explicit adapter selected tavily" in status.notes
    assert "missing TAVILY_API_KEY" in status.summary

    with pytest.raises(WebSearchError, match="adapter tavily is not ready"):
        resolve_web_search_runtime(cfg=cfg, api_key="main-key", strict=True)


def test_web_search_auto_can_fall_back_from_openai_to_tavily_in_same_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.openai.com":
            return httpx.Response(
                400,
                json={"error": {"message": "This provider does not support web_search."}},
            )
        if request.url.host == "api.tavily.com":
            return httpx.Response(
                200,
                json={
                    "request_id": "tavily_req_2",
                    "answer": "Use the Tavily fallback answer.",
                    "results": [
                        {
                            "title": "Fallback docs",
                            "url": "https://docs.example.com/fallback",
                            "content": "Fallback snippet",
                        }
                    ],
                },
            )
        raise AssertionError(f"unexpected host: {request.url.host}")

    result = web_search(
        query="fallback docs",
        cfg=_configured_cfg(),
        api_key="main-key",
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["backend"] == "tavily"
    assert result["protocol"] == "openai_compat"
    assert result["chat_protocol"] == "openai_compat"
    assert result["search_protocol"] == "tavily"
    assert result["backend_adapter"] == "tavily"
    assert result["provider_hosted_search"] is False
    assert result["external_search_provider"] == "tavily"
    assert result["citation_count"] == 1
    assert result["source_count"] == 1
    assert result["answer"] == "Use the Tavily fallback answer."
    assert result["response_id"] == "tavily_req_2"
    assert result["citations"][0]["start_index"] is None
    assert result["citations"][0]["end_index"] is None
    assert result["sources"][0]["snippet"] == "Fallback snippet"


def test_web_search_native_mode_does_not_fallback_to_tavily_in_same_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")
    seen_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(str(request.url.host))
        if request.url.host == "api.tavily.com":
            raise AssertionError("native mode must not call Tavily")
        return httpx.Response(400, json={"error": {"message": "OpenAI failure"}})

    with pytest.raises(WebSearchError, match="OpenAI failure"):
        web_search(
            query="failing docs",
            cfg=_configured_cfg(mode="native"),
            api_key="main-key",
            transport=httpx.MockTransport(handler),
            resolver=_public_resolver,
        )

    assert seen_hosts == ["api.openai.com"]


def test_web_search_external_mode_calls_only_tavily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")
    seen_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(str(request.url.host))
        if request.url.host == "api.openai.com":
            raise AssertionError("external mode must not call native provider search")
        return httpx.Response(
            200,
            json={
                "request_id": "external_tavily",
                "answer": "External Tavily answer.",
                "results": [
                    {
                        "title": "External docs",
                        "url": "https://docs.example.com/external",
                        "content": "External snippet",
                    }
                ],
            },
        )

    result = web_search(
        query="external docs",
        cfg=_configured_cfg(mode="external"),
        api_key="main-key",
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert seen_hosts == ["api.tavily.com"]
    assert result["backend"] == "tavily"
    assert result["response_id"] == "external_tavily"


def test_web_search_auto_reports_combined_error_when_all_backends_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.openai.com":
            return httpx.Response(400, json={"error": {"message": "OpenAI failure"}})
        if request.url.host == "api.tavily.com":
            return httpx.Response(502, json={"error": "Tavily failure"})
        raise AssertionError(f"unexpected host: {request.url.host}")

    with pytest.raises(
        WebSearchError,
        match="web_search failed across auto backends: openai_responses: Responses error 400: OpenAI failure; tavily: Tavily error 502: Tavily failure",
    ):
        web_search(
            query="failing docs",
            cfg=_configured_cfg(),
            api_key="main-key",
            transport=httpx.MockTransport(handler),
            resolver=_public_resolver,
        )


def test_web_search_tavily_output_contract_stays_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "request_id": "tavily_req_3",
                "answer": "Read the migration guide.",
                "results": [
                    {
                        "title": "Guide",
                        "url": "https://docs.example.com/guide",
                        "content": "Guide snippet",
                    }
                ],
            },
        )

    result = web_search(
        query="migration guide",
        cfg=_configured_cfg(base_url="https://example-proxy.invalid/v1"),
        api_key="main-key",
        allowed_domains=["docs.example.com"],
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result == {
        "query": "migration guide",
        "answer": "Read the migration guide.",
        "citations": [
            {
                "title": "Guide",
                "url": "https://docs.example.com/guide",
                "start_index": None,
                "end_index": None,
            }
        ],
        "sources": [
            {
                "url": "https://docs.example.com/guide",
                "title": "Guide",
                "snippet": "Guide snippet",
            }
        ],
        "queries": ["migration guide"],
        "model": None,
        "backend": "tavily",
        "protocol": "openai_compat",
        "chat_protocol": "openai_compat",
        "search_protocol": "tavily",
        "backend_adapter": "tavily",
        "provider_hosted_search": False,
        "external_search_provider": "tavily",
        "citation_count": 1,
        "source_count": 1,
        "allowed_domains": ["docs.example.com"],
        "external_web_access": True,
        "response_id": "tavily_req_3",
        "sources_truncated": False,
    }


def test_web_search_rejects_external_web_access_false_for_tavily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", "tavily")
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    with pytest.raises(
        WebSearchError,
        match="external_web_access=false is supported only by the openai_responses web_search backend; tavily always uses external web access",
    ):
        web_search(
            query="migration guide",
            cfg=_configured_cfg(base_url="https://example-proxy.invalid/v1"),
            api_key="main-key",
            external_web_access=False,
        )


def test_web_search_rejects_invalid_query() -> None:
    with pytest.raises(WebSearchError, match="query must be a non-empty string"):
        web_search(query="   ", cfg=_configured_cfg(), api_key="main-key")


def test_web_search_rejects_invalid_allowed_domains() -> None:
    with pytest.raises(WebSearchError, match="allowed_domains must be an array"):
        web_search(
            query="httpx docs",
            cfg=_configured_cfg(),
            api_key="main-key",
            allowed_domains="python.org",  # type: ignore[arg-type]
        )
    with pytest.raises(
        WebSearchError, match="allowed_domains must contain only non-empty domain strings"
    ):
        web_search(
            query="httpx docs",
            cfg=_configured_cfg(),
            api_key="main-key",
            allowed_domains=["python.org", ""],
        )


def test_web_search_plumbs_allowed_domains_and_external_access_to_openai_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        tool_spec = body["tools"][0]
        assert tool_spec["filters"] == {"allowed_domains": ["docs.python.org"]}
        assert tool_spec["external_web_access"] is False
        return httpx.Response(
            200,
            json={
                "id": "resp_3",
                "model": "main-model",
                "output_text": "Use the official docs.",
                "output": [
                    {
                        "type": "web_search_call",
                        "action": {
                            "queries": ["python pathlib docs"],
                            "sources": [
                                {
                                    "title": "pathlib",
                                    "url": "https://docs.python.org/3/library/pathlib.html",
                                }
                            ],
                        },
                    }
                ],
            },
        )

    result = web_search(
        query="python pathlib docs",
        cfg=_configured_cfg(),
        api_key="main-key",
        allowed_domains=["docs.python.org"],
        external_web_access=False,
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["backend"] == "openai_responses"
    assert result["allowed_domains"] == ["docs.python.org"]
    assert result["external_web_access"] is False
    assert result["sources"] == [
        {
            "url": "https://docs.python.org/3/library/pathlib.html",
            "title": "pathlib",
        }
    ]


def test_web_search_dispatches_to_dashscope_chat_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://coding-intl.dashscope.aliyuncs.com/v1/chat/completions"
        body = json.loads(request.content.decode("utf-8"))
        assert body["enable_search"] is True
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_dashscope_2",
                "model": "qwen3.5-plus",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "DashScope searched the web.",
                                    "sources": [
                                        {
                                            "title": "Docs",
                                            "url": "https://docs.example.com/search",
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ],
            },
        )

    result = web_search(
        query="dashscope search docs",
        cfg=AppConfig(
            model="qwen3.5-plus",
            base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
            web_search_mode="auto",
        ),
        api_key="main-key",
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["backend"] == "dashscope_chat"
    assert result["answer"] == "DashScope searched the web."
    assert result["sources"] == [{"url": "https://docs.example.com/search", "title": "Docs"}]


def test_web_search_dispatches_to_xai_responses_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.x.ai/v1/responses"
        body = json.loads(request.content.decode("utf-8"))
        assert body["tools"] == [{"type": "web_search"}]
        assert body["tool_choice"] == "required"
        assert "include" not in body
        return httpx.Response(
            200,
            json={
                "id": "resp_xai_1",
                "model": "grok-4",
                "output_text": "xAI searched the web.",
                "citations": ["https://docs.x.ai/docs"],
                "output": [
                    {
                        "type": "web_search_call",
                        "action": {
                            "queries": ["xai docs"],
                        },
                    }
                ],
            },
        )

    result = web_search(
        query="xai docs",
        cfg=AppConfig(
            model="grok-4",
            base_url="https://api.x.ai/v1",
            web_search_adapter="xai_responses",
        ),
        api_key="xai-key",
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["backend"] == "xai_responses"
    assert result["sources"] == [{"url": "https://docs.x.ai/docs", "title": ""}]


def test_web_search_dispatches_to_anthropic_messages_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == "ant-key"
        assert request.headers["accept-encoding"] == "identity"
        body = json.loads(request.content.decode("utf-8"))
        assert body["tools"][0]["type"] == "web_search_20260209"
        assert body["tools"][0]["allowed_domains"] == ["docs.example.com"]
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "model": "claude-sonnet-4-6",
                "content": [
                    {"type": "server_tool_use", "input": {"query": "anthropic search"}},
                    {
                        "type": "web_search_tool_result",
                        "content": [
                            {
                                "type": "web_search_result",
                                "title": "Anthropic Docs",
                                "url": "https://docs.example.com/anthropic",
                            }
                        ],
                    },
                    {
                        "type": "text",
                        "text": "Anthropic searched the web.",
                        "citations": [
                            {
                                "type": "web_search_result_location",
                                "title": "Anthropic Docs",
                                "url": "https://docs.example.com/anthropic",
                            }
                        ],
                    },
                ],
            },
        )

    result = web_search(
        query="anthropic search",
        cfg=AppConfig(
            model="claude-sonnet-4-6",
            base_url="https://api.anthropic.com/v1/",
            web_search_adapter="anthropic_messages",
        ),
        api_key="ant-key",
        allowed_domains=["docs.example.com"],
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["backend"] == "anthropic_messages"
    assert result["queries"] == ["anthropic search"]
    assert result["sources"] == [
        {"url": "https://docs.example.com/anthropic", "title": "Anthropic Docs"}
    ]


def test_web_search_dispatches_to_gemini_grounding_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert (
            str(request.url)
            == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        )
        assert request.headers["x-goog-api-key"] == "gem-key"
        body = json.loads(request.content.decode("utf-8"))
        assert body["tools"] == [{"google_search": {}}]
        return httpx.Response(
            200,
            json={
                "responseId": "gem_resp_1",
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "Gemini grounded the answer."}],
                            "role": "model",
                        },
                        "groundingMetadata": {
                            "webSearchQueries": ["gemini search"],
                            "groundingChunks": [
                                {
                                    "web": {
                                        "uri": "https://docs.example.com/gemini",
                                        "title": "Gemini Docs",
                                    }
                                }
                            ],
                            "groundingSupports": [
                                {
                                    "segment": {"startIndex": 0, "endIndex": 6},
                                    "groundingChunkIndices": [0],
                                }
                            ],
                        },
                    }
                ],
            },
        )

    result = web_search(
        query="gemini search",
        cfg=AppConfig(
            model="gemini-2.5-flash",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            web_search_adapter="gemini_grounding",
        ),
        api_key="gem-key",
        allowed_domains=["docs.example.com"],
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["backend"] == "gemini_grounding"
    assert result["queries"] == ["gemini search"]
    assert result["citations"][0]["url"] == "https://docs.example.com/gemini"


def test_web_search_dispatches_to_openrouter_web_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        body = json.loads(request.content.decode("utf-8"))
        assert body["tools"][0]["type"] == "openrouter:web_search"
        assert body["tools"][0]["parameters"]["allowed_domains"] == ["docs.example.com"]
        return httpx.Response(
            200,
            json={
                "id": "or_1",
                "model": "anthropic/claude-sonnet-4.6",
                "choices": [
                    {
                        "message": {
                            "content": "OpenRouter searched the web.",
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "url_citation": {
                                        "title": "OpenRouter Docs",
                                        "url": "https://docs.example.com/openrouter",
                                        "start_index": 0,
                                        "end_index": 10,
                                    },
                                },
                                {
                                    "type": "url_citation",
                                    "url_citation": {
                                        "title": "Filtered",
                                        "url": "https://outside.example.com/openrouter",
                                        "start_index": 12,
                                        "end_index": 20,
                                    },
                                },
                            ],
                        }
                    }
                ],
            },
        )

    result = web_search(
        query="openrouter search",
        cfg=AppConfig(
            model="anthropic/claude-sonnet-4.6",
            base_url="https://openrouter.ai/api/v1",
            web_search_adapter="openrouter_web",
        ),
        api_key="or-key",
        allowed_domains=["docs.example.com"],
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["backend"] == "openrouter_web"
    assert result["sources"] == [
        {"url": "https://docs.example.com/openrouter", "title": "OpenRouter Docs"}
    ]
    assert [citation["url"] for citation in result["citations"]] == [
        "https://docs.example.com/openrouter"
    ]


def test_web_search_dispatches_to_moonshot_kimi_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.moonshot.cn/v1/chat/completions"
        body = json.loads(request.content.decode("utf-8"))
        calls.append(body)
        assert body["model"] == "kimi-k2.6"
        assert body["tools"] == [{"type": "builtin_function", "function": {"name": "$web_search"}}]
        assert body["thinking"] == {"type": "disabled"}
        if len(calls) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "kimi_tool_1",
                    "model": "kimi-k2.6",
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "$web_search",
                                            "arguments": json.dumps(
                                                {
                                                    "query": "kimi search",
                                                    "results": [
                                                        {
                                                            "title": "Kimi Docs",
                                                            "url": "https://docs.example.com/kimi",
                                                            "snippet": "Kimi snippet",
                                                        }
                                                    ],
                                                }
                                            ),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            )
        assert body["messages"][-1]["role"] == "tool"
        assert body["messages"][-1]["tool_call_id"] == "call_1"
        return httpx.Response(
            200,
            json={
                "id": "kimi_final",
                "model": "kimi-k2.6",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Kimi searched the web."},
                    }
                ],
            },
        )

    result = web_search(
        query="kimi search",
        cfg=AppConfig(
            model="kimi-k2",
            base_url="https://api.moonshot.cn/v1",
            web_search_adapter="moonshot_kimi",
            web_search_model="kimi-k2.6",
        ),
        api_key="moonshot-key",
        allowed_domains=["docs.example.com"],
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["backend"] == "moonshot_kimi"
    assert result["response_id"] == "kimi_final"
    assert result["queries"] == ["kimi search"]
    assert result["sources"] == [
        {
            "url": "https://docs.example.com/kimi",
            "title": "Kimi Docs",
            "snippet": "Kimi snippet",
        }
    ]


def test_web_search_dispatches_to_zhipu_web_search_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "glm-4.6"
        assert body["stream"] is False
        assert body["tools"] == [
            {
                "type": "web_search",
                "web_search": {
                    "enable": True,
                    "search_engine": "search_pro",
                    "search_query": "zhipu search",
                    "search_result": True,
                    "count": 5,
                    "search_recency_filter": "noLimit",
                    "content_size": "medium",
                    "search_domain_filter": "docs.example.com",
                },
            }
        ]
        return httpx.Response(
            200,
            json={
                "id": "glm_1",
                "model": "glm-4.6",
                "choices": [{"message": {"content": "Zhipu searched the web."}}],
                "web_search": [
                    {
                        "title": "GLM Docs",
                        "link": "https://docs.example.com/glm",
                        "content": "GLM snippet",
                        "media": "Docs",
                    }
                ],
            },
        )

    result = web_search(
        query="zhipu search",
        cfg=AppConfig(
            model="glm-4.6",
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            web_search_adapter="zhipu_web_search",
        ),
        api_key="glm-key",
        max_sources=5,
        allowed_domains=["docs.example.com"],
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["backend"] == "zhipu_web_search"
    assert result["answer"] == "Zhipu searched the web."
    assert result["sources"] == [
        {
            "url": "https://docs.example.com/glm",
            "title": "GLM Docs",
            "snippet": "GLM snippet",
        }
    ]


def test_web_search_dispatches_to_volcengine_web_search_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://ark.cn-beijing.volces.com/api/v3/responses"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "doubao-seed-1-6-250615"
        assert body["tools"] == [{"type": "web_search"}]
        assert body["stream"] is False
        return httpx.Response(
            200,
            json={
                "id": "ark_resp_1",
                "model": "doubao-seed-1-6-250615",
                "output": [
                    {
                        "type": "web_search_call",
                        "action": {"queries": ["doubao search"]},
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Doubao searched the web.",
                                "annotations": [
                                    {
                                        "type": "url_citation",
                                        "title": "Doubao Docs",
                                        "url": "https://docs.example.com/doubao",
                                        "content": "Doubao snippet",
                                    }
                                ],
                            }
                        ],
                    },
                ],
            },
        )

    result = web_search(
        query="doubao search",
        cfg=AppConfig(
            model="doubao-1.5-pro",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            web_search_adapter="volcengine_web_search",
            web_search_model="doubao-seed-1-6-250615",
        ),
        api_key="ark-key",
        allowed_domains=["docs.example.com"],
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["backend"] == "volcengine_web_search"
    assert result["model"] == "doubao-seed-1-6-250615"
    assert result["sources"] == [
        {
            "url": "https://docs.example.com/doubao",
            "title": "Doubao Docs",
            "snippet": "Doubao snippet",
        }
    ]


def test_web_search_openrouter_answer_without_sources_falls_back_to_tavily_in_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "openrouter.ai":
            return httpx.Response(
                200,
                json={
                    "id": "or_no_search",
                    "model": "openai/gpt-5.2",
                    "choices": [{"message": {"content": "Answer without search evidence."}}],
                },
            )
        if request.url.host == "api.tavily.com":
            return httpx.Response(
                200,
                json={
                    "request_id": "tavily_after_openrouter",
                    "answer": "Tavily searched instead.",
                    "results": [
                        {
                            "title": "Fallback Source",
                            "url": "https://docs.example.com/fallback",
                            "content": "fallback snippet",
                        }
                    ],
                },
            )
        raise AssertionError(f"unexpected host: {request.url.host}")

    result = web_search(
        query="current docs",
        cfg=AppConfig(
            model="openai/gpt-5.2",
            base_url="https://openrouter.ai/api/v1",
            web_search_adapter="auto",
        ),
        api_key="or-key",
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["backend"] == "tavily"
    assert result["response_id"] == "tavily_after_openrouter"
    assert result["sources"][0]["url"] == "https://docs.example.com/fallback"


def test_web_search_openrouter_answer_without_sources_errors_when_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.tavily.com":
            raise AssertionError("explicit adapter should not fall back to Tavily")
        return httpx.Response(
            200,
            json={
                "id": "or_no_search",
                "model": "openai/gpt-5.2",
                "choices": [{"message": {"content": "Answer without search evidence."}}],
            },
        )

    with pytest.raises(WebSearchError, match="OpenRouter web_search did not return sources"):
        web_search(
            query="current docs",
            cfg=AppConfig(
                model="openai/gpt-5.2",
                base_url="https://openrouter.ai/api/v1",
                web_search_adapter="openrouter_web",
            ),
            api_key="or-key",
            transport=httpx.MockTransport(handler),
            resolver=_public_resolver,
        )


def test_web_search_dispatches_to_perplexity_sonar_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.perplexity.ai/v1/sonar"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "sonar"
        assert body["search_domain_filter"] == ["docs.example.com"]
        return httpx.Response(
            200,
            json={
                "id": "pplx_1",
                "model": "sonar",
                "choices": [{"message": {"content": "Perplexity searched the web."}}],
                "search_results": [
                    {
                        "title": "Perplexity Docs",
                        "url": "https://docs.example.com/perplexity",
                        "snippet": "PPLX snippet",
                    }
                ],
            },
        )

    result = web_search(
        query="perplexity search",
        cfg=AppConfig(
            model="sonar-pro",
            base_url="https://api.perplexity.ai",
            web_search_adapter="perplexity_sonar",
            web_search_model="sonar",
        ),
        api_key="pplx-key",
        allowed_domains=["docs.example.com"],
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["backend"] == "perplexity_sonar"
    assert result["sources"][0]["snippet"] == "PPLX snippet"


def test_web_search_dispatches_to_groq_compound_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.groq.com/openai/v1/chat/completions"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "groq/compound-mini"
        assert body["search_settings"] == {"include_domains": ["docs.example.com"]}
        return httpx.Response(
            200,
            json={
                "id": "groq_1",
                "model": "groq/compound-mini",
                "choices": [
                    {
                        "message": {
                            "content": "Groq searched the web.",
                            "executed_tools": [
                                {
                                    "search_results": {
                                        "results": [
                                            {
                                                "title": "Groq Docs",
                                                "url": "https://docs.example.com/groq",
                                                "content": "Groq snippet",
                                            }
                                        ]
                                    }
                                }
                            ],
                        }
                    }
                ],
            },
        )

    result = web_search(
        query="groq search",
        cfg=AppConfig(
            model="llama-3.3-70b-versatile",
            base_url="https://api.groq.com/openai/v1",
            web_search_adapter="groq_compound",
        ),
        api_key="groq-key",
        allowed_domains=["docs.example.com"],
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["backend"] == "groq_compound"
    assert result["model"] == "groq/compound-mini"
    assert result["sources"][0]["snippet"] == "Groq snippet"


def test_web_search_dispatches_to_mistral_conversations_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.mistral.ai/v1/conversations"
        body = json.loads(request.content.decode("utf-8"))
        assert body["tools"] == [{"type": "web_search"}]
        assert body["store"] is False
        return httpx.Response(
            200,
            json={
                "conversation_id": "conv_1",
                "outputs": [
                    {
                        "type": "message.output",
                        "content": [
                            {"type": "text", "text": "Mistral searched the web."},
                            {
                                "type": "tool_reference",
                                "title": "Mistral Docs",
                                "url": "https://docs.example.com/mistral",
                            },
                        ],
                    }
                ],
            },
        )

    result = web_search(
        query="mistral search",
        cfg=AppConfig(
            model="mistral-large-latest",
            base_url="https://api.mistral.ai/v1",
            web_search_adapter="mistral_conversations",
            web_search_model="mistral-medium-latest",
        ),
        api_key="mistral-key",
        allowed_domains=["docs.example.com"],
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )

    assert result["backend"] == "mistral_conversations"
    assert result["sources"] == [
        {"url": "https://docs.example.com/mistral", "title": "Mistral Docs"}
    ]


def test_web_search_rejects_external_web_access_false_for_dashscope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(
        WebSearchError,
        match="external_web_access=false is supported only by the openai_responses web_search backend; dashscope_chat always uses external web access",
    ):
        web_search(
            query="dashscope search docs",
            cfg=AppConfig(
                model="qwen3.5-plus",
                base_url="https://coding-intl.dashscope.aliyuncs.com/v1",
                web_search_mode="auto",
            ),
            api_key="main-key",
            external_web_access=False,
        )


def test_web_search_does_not_fall_back_to_tavily_when_external_web_access_is_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.tavily.com":
            raise AssertionError("should not fall back to Tavily")
        return httpx.Response(
            400,
            json={"error": {"message": "OpenAI failure"}},
        )

    with pytest.raises(WebSearchError, match="OpenAI failure"):
        web_search(
            query="failing docs",
            cfg=_configured_cfg(),
            api_key="main-key",
            external_web_access=False,
            transport=httpx.MockTransport(handler),
            resolver=_public_resolver,
        )


def test_resolve_web_search_runtime_status_reports_off_mode_as_disabled() -> None:
    status = resolve_web_search_runtime_status(
        cfg=AppConfig(web_search_mode="off"),
        api_key=None,
    )

    assert status.mode == "off"
    assert status.provider is None
    assert status.registration_ready is False
    assert status.availability_label == "disabled"
    assert status.summary.startswith("disabled by policy:")


def test_resolve_web_search_runtime_status_reports_auto_unavailable_notes_are_informative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    cfg = AppConfig(
        base_url="https://example-proxy.invalid/v1",
        web_search_mode="auto",
    )

    status = resolve_web_search_runtime_status(cfg=cfg, api_key=None)

    assert status.mode == "auto"
    assert status.provider is None
    assert status.registration_ready is False
    assert status.availability_label == "auto-unavailable"
    assert status.base_url == "https://example-proxy.invalid/v1"
    assert status.model is None
    assert status.api_key_available is False
    assert any(
        "OpenAI auto readiness requires explicit web_search_base_url" in note
        for note in status.notes
    )
    assert any("missing model" in note for note in status.notes)
    assert any("missing API key" in note for note in status.notes)
    assert any("missing TAVILY_API_KEY" in note for note in status.notes)
    assert "TAVILY_API_KEY" in status.setup_hint
    assert "provider-agnostic fallback" in status.setup_hint


def test_resolve_web_search_runtime_status_reports_ready_setup_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.setenv("TAVILY_API_KEY", "tavily-key")

    status = resolve_web_search_runtime_status(
        cfg=_configured_cfg(base_url="https://example-proxy.invalid/v1"),
        api_key="main-key",
    )

    assert status.registration_ready is True
    assert status.provider == "tavily"
    assert "Provider-agnostic web search is ready" in status.setup_hint
    assert status.to_payload()["setup_hint"] == status.setup_hint


def test_resolve_web_search_runtime_status_legacy_on_maps_to_auto_ready_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    status = resolve_web_search_runtime_status(
        cfg=_configured_cfg(
            mode="on",
            base_url="https://example-proxy.invalid/v1",
            web_search_base_url="https://search.example.com/v1",
        ),
        api_key="main-key",
    )

    assert status.mode == "auto"
    assert status.registration_ready is True
    assert status.provider == "openai_responses"
    assert status.availability_label == "available"
    assert status.base_url == "https://search.example.com/v1"


@pytest.mark.parametrize(
    ("cfg", "api_key", "env", "message"),
    [
        (
            AppConfig(
                web_search_mode="auto", model="main-model", base_url="", web_search_timeout_s=20.0
            ),
            "main-key",
            {},
            "missing OpenAI search base URL",
        ),
        (
            AppConfig(
                web_search_mode="auto",
                model="",
                base_url="https://api.openai.com/v1",
                web_search_timeout_s=20.0,
            ),
            "main-key",
            {},
            "missing model",
        ),
        (
            AppConfig(
                web_search_mode="auto",
                model="main-model",
                base_url="https://api.openai.com/v1",
                web_search_timeout_s=20.0,
            ),
            None,
            {},
            "missing API key",
        ),
        (
            AppConfig(
                web_search_mode="auto",
                model="main-model",
                base_url="https://api.openai.com/v1",
                web_search_timeout_s=0.0,
            ),
            "main-key",
            {},
            "web_search_timeout_s must be > 0",
        ),
        (
            AppConfig(
                web_search_mode="auto",
                model="main-model",
                base_url="https://example-proxy.invalid/v1",
                web_search_timeout_s=20.0,
            ),
            "main-key",
            {"SYLLIPTOR_WEB_SEARCH_PROVIDER": "tavily"},
            "adapter tavily is not ready",
        ),
    ],
)
def test_resolve_web_search_runtime_strict_mode_rejects_missing_requirements(
    monkeypatch: pytest.MonkeyPatch,
    cfg: AppConfig,
    api_key: str | None,
    env: dict[str, str],
    message: str,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_PROVIDER", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    with pytest.raises(WebSearchError, match=message):
        resolve_web_search_runtime(cfg=cfg, api_key=api_key, strict=True)
