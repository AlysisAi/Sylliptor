from __future__ import annotations

import copy
import json
import logging
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit

from .failure_category import is_provider_throttling_error, is_provider_unavailable_error
from .llm.types import LLMResponse, LLMUsage

_LOGGER = logging.getLogger(__name__)
_MAX_HISTORY = 50
_HISTORY_LOCK = threading.Lock()
_PROVIDER_CALL_HISTORY: deque[dict[str, Any]] = deque(maxlen=_MAX_HISTORY)
_WEB_SEARCH_HISTORY: deque[dict[str, Any]] = deque(maxlen=_MAX_HISTORY)
_SYLLIPTOR_WEB_SEARCH_TOOL_NAME = "web_search"
_SENSITIVE_EXACT_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "bearer_token",
    "credential",
    "id_token",
    "password",
    "refresh_token",
    "secret",
    "token",
    "x_api_key",
    "x_goog_api_key",
}
_SENSITIVE_KEY_FRAGMENTS = ("client_secret", "private_key")
_SAFE_REDACTION_KEYS = {"api_key_present", "api_key_source"}
_HIDDEN_PAYLOAD_KEYS = {
    "arguments",
    "body",
    "content",
    "contents",
    "headers",
    "input",
    "messages",
    "parameters",
    "provider_metadata",
    "raw",
    "tool_calls",
    "tools",
}
_SECRET_VALUE_RE = re.compile(
    r"(?i)\b(?:sk|tvly|ghp|github_pat|xoxb|xapp|ya29|AIza|key|token)[-_A-Za-z0-9]{10,}\b"
)


def telemetry_clock_ms() -> float:
    return time.monotonic() * 1000.0


def base_url_host(base_url: str | None) -> str:
    raw = str(base_url or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except Exception:
        return ""
    host = (parsed.hostname or "").rstrip(".").casefold()
    return host


def reset_provider_telemetry_for_tests() -> None:
    with _HISTORY_LOCK:
        _PROVIDER_CALL_HISTORY.clear()
        _WEB_SEARCH_HISTORY.clear()


def provider_call_history_snapshot(*, limit: int | None = None) -> list[dict[str, Any]]:
    with _HISTORY_LOCK:
        items = list(_PROVIDER_CALL_HISTORY)
    if limit is not None:
        items = items[-max(0, int(limit)) :]
    return [copy.deepcopy(item) for item in items]


def web_search_history_snapshot(*, limit: int | None = None) -> list[dict[str, Any]]:
    with _HISTORY_LOCK:
        items = list(_WEB_SEARCH_HISTORY)
    if limit is not None:
        items = items[-max(0, int(limit)) :]
    return [copy.deepcopy(item) for item in items]


def last_provider_call_summary() -> dict[str, Any] | None:
    with _HISTORY_LOCK:
        item = _PROVIDER_CALL_HISTORY[-1] if _PROVIDER_CALL_HISTORY else None
    return copy.deepcopy(item) if item is not None else None


def last_web_search_summary() -> dict[str, Any] | None:
    with _HISTORY_LOCK:
        item = _WEB_SEARCH_HISTORY[-1] if _WEB_SEARCH_HISTORY else None
    return copy.deepcopy(item) if item is not None else None


class ProviderCallTelemetryRecorder:
    """Accumulates one safe provider-call summary.

    The recorder stores only derived fields: counts, booleans, host names, latency,
    and token totals. It never stores request bodies, tool arguments, raw provider
    payloads, or hidden provider metadata.
    """

    def __init__(
        self,
        *,
        provider_key: str | None,
        protocol: str,
        model: str,
        base_url: str,
        stream: bool,
        tools: list[dict[str, Any]] | None,
        web_search_mode: str | None = None,
        web_search_adapter: str | None = None,
        native_web_search: bool = False,
        operation: str = "chat",
    ) -> None:
        self.provider_key = _safe_label(provider_key)
        self.protocol = _safe_label(protocol)
        self.model = _safe_label(model)
        self.base_url_host = base_url_host(base_url)
        self.stream = bool(stream)
        self.operation = _safe_label(operation)
        self.tool_count = len([tool for tool in tools or [] if isinstance(tool, dict)])
        self.web_search_exposed = tools_expose_web_search(tools) or bool(native_web_search)
        self.web_search_mode = _safe_label(web_search_mode or "off")
        self.web_search_adapter = _safe_label(web_search_adapter or "")
        self.provider_hosted_search = bool(native_web_search)
        self.external_search_provider = _external_search_provider(
            exposed=self.web_search_exposed,
            native=native_web_search,
            adapter=web_search_adapter,
        )
        self.web_search_backend_kind = _chat_web_search_backend_kind(
            exposed=self.web_search_exposed,
            native=native_web_search,
            mode=web_search_mode,
        )
        self._started_ms = telemetry_clock_ms()
        self._first_text_delta_ms: float | None = None
        self._text_delta_count = 0
        self._retry_count = 0
        self._retry_reasons: list[str] = []

    def wrap_text_delta(
        self,
        callback: Callable[[str], None] | None,
    ) -> Callable[[str], None] | None:
        if not self.stream:
            return callback

        def _wrapped(delta: str) -> None:
            if delta:
                self._text_delta_count += 1
                if self._first_text_delta_ms is None:
                    self._first_text_delta_ms = telemetry_clock_ms()
            if callback is not None:
                callback(delta)

        return _wrapped

    def on_retry(self, attempt: int, reason: str, _wait_seconds: float) -> None:
        try:
            self._retry_count = max(self._retry_count, int(attempt))
        except (TypeError, ValueError):
            self._retry_count += 1
        normalized_reason = _safe_label(reason)
        if normalized_reason and normalized_reason not in self._retry_reasons:
            self._retry_reasons.append(normalized_reason)

    def run(self, call: Callable[[], LLMResponse]) -> LLMResponse:
        try:
            response = call()
        except Exception as exc:
            self.record_error(exc)
            raise
        self.record_success(response)
        return response

    def record_success(self, response: LLMResponse) -> None:
        latency_ms = _duration_ms(self._started_ms)
        payload = self._base_payload(
            latency_ms=latency_ms,
            status_category="success",
        )
        payload["usage"] = _usage_payload(response.usage)
        payload["provider_metadata_present"] = bool(response.provider_metadata)
        payload["tool_call_provider_metadata_count"] = sum(
            1 for tool_call in response.tool_calls if tool_call.provider_metadata
        )
        payload["streaming"] = self._streaming_payload(
            raw=response.raw,
            final_latency_ms=latency_ms,
        )
        payload["web_search"].update(_provider_metadata_web_search_counts(response))
        record_provider_call(payload)

    def record_error(self, exc: Exception) -> None:
        latency_ms = _duration_ms(self._started_ms)
        payload = self._base_payload(
            latency_ms=latency_ms,
            status_category=_status_category(exc),
        )
        payload["error_type"] = type(exc).__name__
        payload["usage"] = _usage_payload(None)
        payload["provider_metadata_present"] = False
        payload["tool_call_provider_metadata_count"] = 0
        payload["streaming"] = self._streaming_payload(
            raw=None,
            final_latency_ms=latency_ms,
            error_type=type(exc).__name__ if self.stream else None,
        )
        record_provider_call(payload)

    def _base_payload(self, *, latency_ms: int, status_category: str) -> dict[str, Any]:
        return {
            "kind": "provider_call",
            "operation": self.operation,
            "provider_key": self.provider_key,
            "protocol": self.protocol,
            "model": self.model,
            "base_url_host": self.base_url_host,
            "stream": self.stream,
            "tool_count": self.tool_count,
            "web_search_exposed": self.web_search_exposed,
            "web_search": {
                "web_search_mode": self.web_search_mode,
                "web_search_adapter": self.web_search_adapter,
                "backend_kind": self.web_search_backend_kind,
                "provider_hosted_search": self.provider_hosted_search,
                "external_provider_name": self.external_search_provider,
                "source_count": 0,
                "citation_count": 0,
                "query_count": 0,
                "fallback_occurred": False,
            },
            "retry_count": self._retry_count,
            "retry_reasons": list(self._retry_reasons),
            "status_category": status_category,
            "latency_ms": latency_ms,
        }

    def _streaming_payload(
        self,
        *,
        raw: Mapping[str, Any] | None,
        final_latency_ms: int,
        error_type: str | None = None,
    ) -> dict[str, Any]:
        if not self.stream:
            return {
                "enabled": False,
                "event_count": 0,
                "text_delta_count": 0,
                "first_token_latency_ms": None,
                "final_latency_ms": final_latency_ms,
                "stream_error_type": None,
                "unknown_event_count": 0,
                "stream_restart_count": 0,
                "stream_restart_reason": "",
            }
        return {
            "enabled": True,
            "event_count": _stream_event_count(raw),
            "text_delta_count": self._text_delta_count,
            "first_token_latency_ms": _first_token_latency_ms(
                started_ms=self._started_ms,
                first_delta_ms=self._first_text_delta_ms,
            ),
            "final_latency_ms": final_latency_ms,
            "stream_error_type": error_type,
            "unknown_event_count": _stream_unknown_event_count(raw),
            "stream_restart_count": _stream_restart_count(raw),
            "stream_restart_reason": _stream_restart_reason(raw),
        }


def record_provider_call(payload: Mapping[str, Any]) -> None:
    safe_payload = redact_telemetry_payload(payload)
    with _HISTORY_LOCK:
        _PROVIDER_CALL_HISTORY.append(safe_payload)
    _LOGGER.info("provider_call", extra={"sylliptor_provider_call": safe_payload})


def record_web_search_call(
    *,
    protocol: str | None,
    provider_key: str | None,
    model: str | None,
    web_search_mode: str,
    web_search_adapter: str,
    provider_hosted_search: bool,
    external_provider_name: str | None,
    source_count: int,
    citation_count: int,
    query_count: int,
    fallback_occurred: bool,
    status_category: str = "success",
) -> None:
    payload = redact_telemetry_payload(
        {
            "kind": "web_search",
            "protocol": _safe_label(protocol),
            "provider_key": _safe_label(provider_key),
            "model": _safe_label(model),
            "web_search_mode": _safe_label(web_search_mode),
            "web_search_adapter": _safe_label(web_search_adapter),
            "provider_hosted_search": bool(provider_hosted_search),
            "external_provider_name": _safe_label(external_provider_name),
            "source_count": max(0, int(source_count)),
            "citation_count": max(0, int(citation_count)),
            "query_count": max(0, int(query_count)),
            "fallback_occurred": bool(fallback_occurred),
            "status_category": _safe_label(status_category),
        }
    )
    with _HISTORY_LOCK:
        _WEB_SEARCH_HISTORY.append(payload)
    _LOGGER.info("web_search_call", extra={"sylliptor_web_search": payload})


def redact_telemetry_payload(payload: Any) -> Any:
    return _redact_value(payload)


def tools_expose_web_search(tools: list[dict[str, Any]] | None) -> bool:
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        if _tool_name(tool) == _SYLLIPTOR_WEB_SEARCH_TOOL_NAME:
            return True
    return False


def diagnostic_bundle_payload(*, provider_diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    return redact_telemetry_payload(
        {
            "redacted": True,
            "provider_diagnostics": dict(provider_diagnostics),
            "last_provider_call": last_provider_call_summary(),
            "recent_provider_calls": provider_call_history_snapshot(limit=10),
            "last_web_search": last_web_search_summary(),
            "recent_web_search_calls": web_search_history_snapshot(limit=10),
            "notes": [
                "Request bodies, tool arguments, secrets, raw provider payloads, and hidden provider metadata are excluded.",
                "Provider call history is process-local and may be empty in a fresh CLI process.",
            ],
        }
    )


def _tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, Mapping):
        return str(function.get("name") or "").strip()
    return str(tool.get("name") or "").strip()


def _chat_web_search_backend_kind(
    *,
    exposed: bool,
    native: bool,
    mode: str | None,
) -> str:
    if native:
        return "native/provider-hosted"
    if not exposed:
        return "off"
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode == "off":
        return "off"
    return "external"


def _external_search_provider(
    *,
    exposed: bool,
    native: bool,
    adapter: str | None,
) -> str | None:
    if native or not exposed:
        return None
    normalized_adapter = str(adapter or "").strip().lower()
    if normalized_adapter and normalized_adapter != "auto":
        return normalized_adapter
    return "sylliptor_web_search_tool"


def _duration_ms(started_ms: float) -> int:
    return max(0, int(round(telemetry_clock_ms() - started_ms)))


def _first_token_latency_ms(
    *,
    started_ms: float,
    first_delta_ms: float | None,
) -> int | None:
    if first_delta_ms is None:
        return None
    return max(0, int(round(first_delta_ms - started_ms)))


def _usage_payload(usage: LLMUsage | None) -> dict[str, int | None]:
    return {
        "prompt_tokens": usage.prompt_tokens if usage is not None else None,
        "completion_tokens": usage.completion_tokens if usage is not None else None,
        "total_tokens": usage.total_tokens if usage is not None else None,
        "cached_prompt_tokens": usage.cached_prompt_tokens if usage is not None else None,
    }


def _status_category(exc: Exception) -> str:
    if is_provider_throttling_error(exc):
        return "rate_limited"
    if is_provider_unavailable_error(exc):
        return "provider_unavailable"
    lowered = str(exc).lower()
    if any(token in lowered for token in ("timeout", "connect", "network", "dns")):
        return "network_error"
    return "provider_error"


def _stream_event_count(raw: Mapping[str, Any] | None) -> int:
    if not isinstance(raw, Mapping):
        return 0
    direct = raw.get("events")
    if isinstance(direct, int):
        return max(0, direct)
    stream_metadata = raw.get("stream_metadata")
    if isinstance(stream_metadata, Mapping):
        events = stream_metadata.get("events")
        if isinstance(events, int):
            return max(0, events)
    gemini_metadata = raw.get("streamMetadata")
    if isinstance(gemini_metadata, Mapping):
        chunks = gemini_metadata.get("chunks")
        if isinstance(chunks, int):
            return max(0, chunks)
    return 0


def _stream_unknown_event_count(raw: Mapping[str, Any] | None) -> int:
    if not isinstance(raw, Mapping):
        return 0
    stream_metadata = raw.get("stream_metadata")
    if isinstance(stream_metadata, Mapping):
        unknown = stream_metadata.get("unknown_events")
        if isinstance(unknown, list):
            return len(unknown)
    gemini_metadata = raw.get("streamMetadata")
    if isinstance(gemini_metadata, Mapping):
        unknown_chunks = gemini_metadata.get("unknown_chunks")
        if isinstance(unknown_chunks, list):
            return len(unknown_chunks)
    return 0


def _stream_restart_count(raw: Mapping[str, Any] | None) -> int:
    if not isinstance(raw, Mapping):
        return 0
    try:
        return max(0, int(raw.get("stream_restart_count") or 0))
    except (TypeError, ValueError):
        return 0


def _stream_restart_reason(raw: Mapping[str, Any] | None) -> str:
    if not isinstance(raw, Mapping):
        return ""
    return _safe_label(raw.get("stream_restart_reason"))


def _provider_metadata_web_search_counts(response: LLMResponse) -> dict[str, int]:
    metadata = response.provider_metadata if isinstance(response.provider_metadata, Mapping) else {}
    return {
        "source_count": _count_list_keys(metadata, {"sources", "groundingChunks"}),
        "citation_count": _count_list_keys(
            metadata,
            {"citations", "groundingSupports", "citationMetadata"},
        ),
        "query_count": _count_list_keys(metadata, {"queries", "webSearchQueries"}),
    }


def _count_list_keys(value: Any, keys: set[str]) -> int:
    if isinstance(value, Mapping):
        count = 0
        for key, item in value.items():
            if str(key) in keys and isinstance(item, list):
                count += len(item)
            else:
                count += _count_list_keys(item, keys)
        return count
    if isinstance(value, list):
        return sum(_count_list_keys(item, keys) for item in value)
    return 0


def _safe_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return _redact_string(text)


def _redact_value(value: Any, *, key: str | None = None) -> Any:
    normalized_key = str(key or "").strip().casefold().replace("-", "_")
    if normalized_key:
        if normalized_key not in _SAFE_REDACTION_KEYS and (
            normalized_key in _SENSITIVE_EXACT_KEYS
            or normalized_key.endswith("_api_key")
            or any(fragment in normalized_key for fragment in _SENSITIVE_KEY_FRAGMENTS)
        ):
            return "[redacted]"
        if normalized_key in _HIDDEN_PAYLOAD_KEYS:
            return "[omitted]"
    if isinstance(value, Mapping):
        return {str(k): _redact_value(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    try:
        json.dumps(value)
    except TypeError:
        return _redact_string(repr(value))
    return value


def _redact_string(value: str) -> str:
    text = str(value)
    if not text:
        return text
    return _SECRET_VALUE_RE.sub("[redacted]", text)
