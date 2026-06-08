from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx

from ..llm.provider_limits import (
    ProviderRetrySettings,
    best_effort_provider_key,
    run_provider_limited_call,
)
from ..safety import SafeHttpError, safe_http_request
from ..safety.safe_http import Resolver
from ..web_research import extract_public_web_urls, normalize_web_url
from .http_timeout import build_http_timeout_budget, format_http_timeout_error


class DashScopeChatSearchError(RuntimeError):
    pass


def _extract_error_message(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    error_obj = data.get("error")
    if isinstance(error_obj, dict):
        message = str(error_obj.get("message") or "").strip()
        if message:
            code = str(error_obj.get("code") or "").strip()
            return f"{code}: {message}" if code else message
    message = str(data.get("message") or "").strip()
    if message:
        return message
    return None


def _dashscope_error_from_response(response: httpx.Response) -> DashScopeChatSearchError:
    try:
        data = response.json()
    except Exception:
        body = response.text
        if len(body) > 1000:
            body = body[:1000] + "...(truncated)"
        return DashScopeChatSearchError(
            f"DashScope chat search error {response.status_code}: {body}"
        )
    if isinstance(data, dict):
        error_message = _extract_error_message(data)
        if error_message:
            return DashScopeChatSearchError(
                f"DashScope chat search error {response.status_code}: {error_message}"
            )
    return DashScopeChatSearchError(f"DashScope chat search error {response.status_code}: {data!r}")


def _choice_message_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list):
        return ""
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    return ""


def _sse_json_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return events


def _coalesce_streaming_response(text: str) -> dict[str, Any] | None:
    events = _sse_json_events(text)
    if not events:
        return None

    content_parts: list[str] = []
    response_id: str | None = None
    response_model: str | None = None
    sources: list[dict[str, str]] = []
    for event in events:
        if response_id is None:
            response_id = str(event.get("id") or "").strip() or None
        if response_model is None:
            response_model = str(event.get("model") or "").strip() or None
        sources.extend(_sources_from_json_payload(event))

        choices = event.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = choice.get("message")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str):
                content_parts.append(content)

    return {
        "id": response_id,
        "model": response_model,
        "choices": [{"message": {"content": "".join(content_parts)}}],
        "search_info": {"search_results": sources},
    }


def _coerce_source(raw: Any) -> dict[str, str] | None:
    if not isinstance(raw, dict):
        return None
    url = str(raw.get("url") or raw.get("link") or "").strip()
    if not url:
        return None
    title = str(raw.get("title") or raw.get("name") or "").strip()
    snippet = str(raw.get("snippet") or raw.get("content") or raw.get("summary") or "").strip()
    source = {"url": url, "title": title}
    normalized_url = normalize_web_url(url)
    if normalized_url:
        source["normalized_url"] = normalized_url
    if snippet:
        source["snippet"] = snippet[:500]
    return source


def _sources_from_json_payload(value: Any) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key in ("sources", "citations", "search_results", "results"):
            raw_sources = value.get(key)
            if isinstance(raw_sources, list):
                for raw_source in raw_sources:
                    source = _coerce_source(raw_source)
                    if source is not None:
                        sources.append(source)
        search_info = value.get("search_info")
        if isinstance(search_info, dict):
            sources.extend(_sources_from_json_payload(search_info))
    return sources


def _strip_json_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 2 or not lines[-1].strip().startswith("```"):
        return stripped
    first_line = lines[0].strip().lower()
    if first_line not in {"```", "```json", "```javascript", "```js"}:
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _json_answer_and_sources(text: str) -> tuple[str, list[dict[str, str]]]:
    stripped = _strip_json_code_fence(text)
    if not stripped:
        return "", []
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped, []
    if not isinstance(parsed, dict):
        return stripped, []
    answer = str(
        parsed.get("answer")
        or parsed.get("summary")
        or parsed.get("content")
        or parsed.get("text")
        or ""
    ).strip()
    sources = _sources_from_json_payload(parsed)
    return answer or stripped, sources


def _sources_from_urls(text: str) -> list[dict[str, str]]:
    return [
        {
            "url": entry["url"],
            "normalized_url": entry["normalized_url"],
            "title": "",
        }
        for entry in extract_public_web_urls(text)
    ]


def _effective_source_url(source: dict[str, str]) -> str | None:
    normalized_url = normalize_web_url(source.get("normalized_url"))
    if normalized_url:
        return normalized_url
    return normalize_web_url(source.get("url"))


def _host_matches_allowed_domains(url: str, allowed_domains: list[str] | None) -> bool:
    if not allowed_domains:
        return True
    try:
        host = (urlsplit(url).hostname or "").rstrip(".").lower()
    except ValueError:
        return False
    for domain in allowed_domains:
        normalized = str(domain or "").strip().rstrip(".").lower()
        if host == normalized or host.endswith(f".{normalized}"):
            return True
    return False


def _dedupe_sources(
    sources: list[dict[str, str]],
    *,
    allowed_domains: list[str] | None,
) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for source in sources:
        raw_url = str(source.get("url") or "").strip()
        if not raw_url:
            continue
        effective_url = _effective_source_url(source)
        dedupe_key = effective_url or raw_url
        if dedupe_key in seen:
            continue
        if not _host_matches_allowed_domains(effective_url or raw_url, allowed_domains):
            continue
        seen.add(dedupe_key)
        cleaned = {
            "url": effective_url or raw_url,
            "title": str(source.get("title") or "").strip(),
        }
        snippet = str(source.get("snippet") or "").strip()
        if snippet:
            cleaned["snippet"] = snippet
        deduped.append(cleaned)
    return deduped


def _search_prompt(query: str, allowed_domains: list[str] | None) -> str:
    domain_text = ""
    if allowed_domains:
        domain_text = (
            "\nOnly use results from these domains: "
            + ", ".join(str(domain).strip() for domain in allowed_domains)
            + "."
        )
    return (
        "Use live web search to answer the query. Return only JSON with this shape: "
        '{"answer":"...", "sources":[{"title":"...", "url":"..."}]}. '
        "Include source URLs whenever the search backend provides them."
        f"{domain_text}\nQuery: {query}"
    )


def dashscope_chat_search(
    *,
    query: str,
    base_url: str,
    api_key: str,
    model: str,
    max_results: int = 8,
    include_domains: list[str] | None = None,
    timeout_s: float = 45.0,
    transport: httpx.BaseTransport | None = None,
    resolver: Resolver | None = None,
    provider_key: str | None = None,
    provider_concurrency_caps: dict[str, int] | None = None,
    provider_retry_settings: ProviderRetrySettings | None = None,
    provider_sleep_fn: Callable[[float], None] | None = None,
    provider_random_fn: Callable[[], float] | None = None,
) -> dict[str, Any]:
    if not str(query or "").strip():
        raise DashScopeChatSearchError("query must be a non-empty string.")
    if not str(api_key or "").strip():
        raise DashScopeChatSearchError("API key is required for DashScope chat search.")
    if not str(base_url or "").strip():
        raise DashScopeChatSearchError("base_url is required for DashScope chat search.")
    if not str(model or "").strip():
        raise DashScopeChatSearchError("model is required for DashScope chat search.")

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": _search_prompt(query, include_domains)}],
        "enable_search": True,
        "search_options": {
            "forced_search": True,
            "search_strategy": "agent",
            "enable_source": True,
        },
        "enable_thinking": False,
        "stream": True,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "sylliptor-agent-cli/0.1.0",
    }
    timeout_budget = build_http_timeout_budget(timeout_s, profile="search")

    resolved_provider_key = provider_key or best_effort_provider_key(
        base_url=base_url,
        model=model,
    )

    def _send_request() -> httpx.Response:
        try:
            response = asyncio.run(
                safe_http_request(
                    "POST",
                    url,
                    timeout=timeout_s,
                    headers=headers,
                    json=payload,
                    _transport=transport,  # type: ignore[arg-type]
                    _resolver=resolver,
                )
            )
        except httpx.TimeoutException as e:
            raise DashScopeChatSearchError(
                format_http_timeout_error(
                    operation="DashScope chat search",
                    budget=timeout_budget,
                    error=e,
                )
            ) from e
        except SafeHttpError as e:
            raise DashScopeChatSearchError(f"DashScope chat search request blocked: {e}") from e
        except Exception as e:  # noqa: BLE001
            raise DashScopeChatSearchError(f"DashScope chat search request failed: {e}") from e
        if response.status_code >= 400:
            raise _dashscope_error_from_response(response)
        return response

    response = run_provider_limited_call(
        call=_send_request,
        provider_key=resolved_provider_key,
        provider_concurrency_caps=provider_concurrency_caps,
        retry_settings=provider_retry_settings,
        operation="dashscope_chat_search",
        sleep_fn=provider_sleep_fn,
        random_fn=provider_random_fn,
    )

    try:
        data = response.json()
    except Exception as e:  # noqa: BLE001
        stream_data = _coalesce_streaming_response(response.text)
        if stream_data is None:
            raise DashScopeChatSearchError(
                "DashScope chat search returned non-JSON response"
            ) from e
        data = stream_data

    if not isinstance(data, dict):
        raise DashScopeChatSearchError("Unexpected DashScope chat search payload")

    if response.status_code >= 400:
        error_message = _extract_error_message(data)
        if error_message:
            raise DashScopeChatSearchError(
                f"DashScope chat search error {response.status_code}: {error_message}"
            )
        raise DashScopeChatSearchError(
            f"DashScope chat search error {response.status_code}: {data!r}"
        )

    raw_answer = _choice_message_text(data)
    answer, json_sources = _json_answer_and_sources(raw_answer)
    payload_sources = _sources_from_json_payload(data)
    url_sources = _sources_from_urls(answer)
    deduped_sources = _dedupe_sources(
        [*json_sources, *payload_sources, *url_sources],
        allowed_domains=include_domains,
    )
    sources_truncated = len(deduped_sources) > max_results
    final_sources = deduped_sources[:max_results]
    citations = [
        {
            "title": str(source.get("title") or "").strip(),
            "url": str(source.get("url") or "").strip(),
            "start_index": None,
            "end_index": None,
        }
        for source in final_sources
    ]

    if not answer and not final_sources:
        raise DashScopeChatSearchError("DashScope chat search payload missing answer and sources")

    response_id = str(data.get("id") or "").strip() or None
    response_model = str(data.get("model") or "").strip() or None
    return {
        "query": query,
        "answer": answer,
        "citations": citations,
        "sources": final_sources,
        "queries": [query],
        "model": response_model or model,
        "backend": "dashscope_chat",
        "allowed_domains": include_domains or [],
        "external_web_access": True,
        "response_id": response_id,
        "sources_truncated": sources_truncated,
    }
