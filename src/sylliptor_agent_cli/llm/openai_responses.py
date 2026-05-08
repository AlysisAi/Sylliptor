from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

from .provider_limits import (
    DEFAULT_PROVIDER_CONCURRENCY_CAPS,
    ProviderRetrySettings,
    best_effort_provider_key,
    run_provider_limited_call,
)


class ResponsesError(RuntimeError):
    pass


_DEFAULT_ACCEPT_ENCODING = "identity"


def _headers_with_default_accept_encoding(headers: dict[str, str]) -> dict[str, str]:
    request_headers = dict(headers)
    if not any(key.lower() == "accept-encoding" for key in request_headers):
        request_headers["Accept-Encoding"] = _DEFAULT_ACCEPT_ENCODING
    return request_headers


@dataclass(frozen=True)
class WebSearchCitation:
    title: str
    url: str
    start_index: int | None = None
    end_index: int | None = None


@dataclass(frozen=True)
class WebSearchSource:
    url: str
    title: str = ""


@dataclass(frozen=True)
class WebSearchResponse:
    answer: str
    citations: list[WebSearchCitation]
    sources: list[WebSearchSource]
    queries: list[str]
    raw: dict[str, Any]
    response_id: str | None = None
    model: str | None = None


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_error_message(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    error_obj = data.get("error")
    if isinstance(error_obj, dict):
        message = str(error_obj.get("message") or "").strip()
        if message:
            return message
    return None


def _assistant_output_parts(data: dict[str, Any]) -> list[dict[str, Any]]:
    output = data.get("output")
    if not isinstance(output, list):
        return []
    parts: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") != "message":
            continue
        if str(item.get("role") or "") != "assistant":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict):
                parts.append(part)
    return parts


def _extract_answer_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    text_parts: list[str] = []
    for part in _assistant_output_parts(data):
        part_type = str(part.get("type") or "")
        if part_type not in {"output_text", "text"}:
            continue
        text = part.get("text")
        if isinstance(text, str):
            text_parts.append(text)
    return "".join(text_parts)


def _extract_citations(data: dict[str, Any]) -> list[WebSearchCitation]:
    citations: list[WebSearchCitation] = []
    for part in _assistant_output_parts(data):
        annotations = part.get("annotations")
        if not isinstance(annotations, list):
            continue
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            if str(annotation.get("type") or "") != "url_citation":
                continue
            url = str(annotation.get("url") or "").strip()
            if not url:
                continue
            citations.append(
                WebSearchCitation(
                    title=str(annotation.get("title") or "").strip(),
                    url=url,
                    start_index=_coerce_int(annotation.get("start_index")),
                    end_index=_coerce_int(annotation.get("end_index")),
                )
            )
    raw_citations = data.get("citations")
    if isinstance(raw_citations, list):
        for raw_citation in raw_citations:
            citation = _coerce_citation(raw_citation)
            if citation is not None:
                citations.append(citation)
    return _dedupe_citations(citations)


def _coerce_citation(raw_citation: Any) -> WebSearchCitation | None:
    if isinstance(raw_citation, str):
        url = raw_citation.strip()
        if not url:
            return None
        return WebSearchCitation(title="", url=url)
    if not isinstance(raw_citation, dict):
        return None

    citation_payload = raw_citation
    for nested_key in ("url_citation", "web_citation", "x_citation"):
        nested = raw_citation.get(nested_key)
        if isinstance(nested, dict):
            citation_payload = nested
            break

    url = str(
        citation_payload.get("url")
        or citation_payload.get("uri")
        or citation_payload.get("link")
        or ""
    ).strip()
    if not url:
        return None
    start_index = citation_payload.get("start_index")
    if start_index is None:
        start_index = citation_payload.get("startIndex")
    end_index = citation_payload.get("end_index")
    if end_index is None:
        end_index = citation_payload.get("endIndex")
    return WebSearchCitation(
        title=str(citation_payload.get("title") or citation_payload.get("name") or "").strip(),
        url=url,
        start_index=_coerce_int(start_index),
        end_index=_coerce_int(end_index),
    )


def _dedupe_citations(citations: list[WebSearchCitation]) -> list[WebSearchCitation]:
    deduped: list[WebSearchCitation] = []
    seen: set[str] = set()
    for citation in citations:
        url = str(citation.url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(citation)
    return deduped


def _extract_sources_and_queries(data: dict[str, Any]) -> tuple[list[WebSearchSource], list[str]]:
    output = data.get("output")
    if not isinstance(output, list):
        return [], []

    sources: list[WebSearchSource] = []
    queries: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") != "web_search_call":
            continue
        action = item.get("action")
        if not isinstance(action, dict):
            continue

        raw_sources = action.get("sources")
        if isinstance(raw_sources, list):
            for raw_source in raw_sources:
                if not isinstance(raw_source, dict):
                    continue
                url = str(raw_source.get("url") or "").strip()
                if not url:
                    continue
                sources.append(
                    WebSearchSource(
                        url=url,
                        title=str(raw_source.get("title") or "").strip(),
                    )
                )

        raw_queries = action.get("queries")
        if isinstance(raw_queries, list):
            for raw_query in raw_queries:
                query = str(raw_query or "").strip()
                if query:
                    queries.append(query)

    return sources, queries


def _merge_citation_sources(
    sources: list[WebSearchSource],
    citations: list[WebSearchCitation],
) -> list[WebSearchSource]:
    merged = list(sources)
    seen = {str(source.url or "").strip() for source in merged if str(source.url or "").strip()}
    for citation in citations:
        url = str(citation.url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        merged.append(WebSearchSource(url=url, title=str(citation.title or "").strip()))
    return merged


class OpenAIResponsesClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 20.0,
        transport: httpx.BaseTransport | None = None,
        provider_key: str | None = None,
        provider_concurrency_caps: dict[str, int] | None = None,
        provider_retry_settings: ProviderRetrySettings | None = None,
        provider_sleep_fn: Callable[[float], None] | None = None,
        provider_random_fn: Callable[[], float] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self._transport = transport
        self.provider_key = str(provider_key or "").strip() or None
        self.provider_concurrency_caps = dict(
            DEFAULT_PROVIDER_CONCURRENCY_CAPS
            if provider_concurrency_caps is None
            else provider_concurrency_caps
        )
        self.provider_retry_settings = provider_retry_settings or ProviderRetrySettings()
        self._provider_sleep_fn = provider_sleep_fn
        self._provider_random_fn = provider_random_fn

    @staticmethod
    def _error_from_response(response: httpx.Response) -> ResponsesError:
        try:
            data = response.json()
        except Exception:
            body = response.text
            if len(body) > 1000:
                body = body[:1000] + "...(truncated)"
            return ResponsesError(f"Responses error {response.status_code}: {body}")
        if isinstance(data, dict):
            error_message = _extract_error_message(data)
            if error_message:
                lower = error_message.lower()
                if "unsupported" in lower or "not support" in lower:
                    return ResponsesError(f"Responses web_search unsupported: {error_message}")
                return ResponsesError(f"Responses error {response.status_code}: {error_message}")
        return ResponsesError(f"Responses error {response.status_code}: {data!r}")

    def web_search(
        self,
        *,
        query: str,
        allowed_domains: list[str] | None = None,
        external_web_access: bool | None = None,
        include_source_details: bool = True,
        tool_choice: str | dict[str, Any] | None = "required",
    ) -> WebSearchResponse:
        url = f"{self.base_url}/responses"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "sylliptor-agent-cli/0.1.0",
        }
        headers = _headers_with_default_accept_encoding(headers)

        tool_spec: dict[str, Any] = {"type": "web_search"}
        if allowed_domains:
            tool_spec["filters"] = {"allowed_domains": list(allowed_domains)}
        if external_web_access is not None:
            tool_spec["external_web_access"] = bool(external_web_access)

        payload: dict[str, Any] = {
            "model": self.model,
            "input": query,
            "tools": [tool_spec],
        }
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if include_source_details:
            payload["include"] = ["web_search_call.action.sources"]

        provider_key = self.provider_key or best_effort_provider_key(
            base_url=self.base_url,
            model=self.model,
        )

        def _send_request() -> httpx.Response:
            try:
                with httpx.Client(timeout=self.timeout_s, transport=self._transport) as client:
                    response = client.post(url, headers=headers, json=payload)
            except httpx.DecodingError as e:
                raise ResponsesError(f"Responses response decompression failed: {e}") from e
            except Exception as e:  # noqa: BLE001
                raise ResponsesError(f"Responses request failed: {e}") from e
            if response.status_code >= 400:
                raise self._error_from_response(response)
            return response

        response = run_provider_limited_call(
            call=_send_request,
            provider_key=provider_key,
            provider_concurrency_caps=self.provider_concurrency_caps,
            retry_settings=self.provider_retry_settings,
            operation="responses_web_search",
            sleep_fn=self._provider_sleep_fn,
            random_fn=self._provider_random_fn,
        )

        try:
            data = response.json()
        except Exception as e:  # noqa: BLE001
            raise ResponsesError("Responses API returned non-JSON response") from e

        if not isinstance(data, dict):
            raise ResponsesError("Unexpected Responses API payload: expected JSON object")

        if response.status_code >= 400:
            error_message = _extract_error_message(data)
            if error_message:
                lower = error_message.lower()
                if "unsupported" in lower or "not support" in lower:
                    raise ResponsesError(f"Responses web_search unsupported: {error_message}")
                raise ResponsesError(f"Responses error {response.status_code}: {error_message}")
            raise ResponsesError(f"Responses error {response.status_code}: {data!r}")

        answer = _extract_answer_text(data)
        citations = _extract_citations(data)
        sources, queries = _extract_sources_and_queries(data)
        sources = _merge_citation_sources(sources, citations)
        if not sources:
            raise ResponsesError("Responses web_search did not return sources")

        response_id = str(data.get("id") or "").strip() or None
        response_model = str(data.get("model") or "").strip() or None
        return WebSearchResponse(
            answer=answer,
            citations=citations,
            sources=sources,
            queries=queries,
            raw=data,
            response_id=response_id,
            model=response_model,
        )
