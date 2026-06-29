from __future__ import annotations

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import httpx

from ..branding import env_get
from ..provider_telemetry import ProviderCallTelemetryRecorder
from .metadata import GEMINI_INTERACTIONS_PROVIDER_METADATA_KEY
from .provider_limits import (
    DEFAULT_PROVIDER_CONCURRENCY_CAPS,
    ProviderRetrySettings,
    best_effort_provider_key,
    run_provider_limited_call,
)
from .types import LLMError, LLMResponse, LLMUsage

if TYPE_CHECKING:
    from ..config import AppConfig

GEMINI_INTERACTIONS_EXPERIMENT_ENV = "SYLLIPTOR_EXPERIMENTAL_GEMINI_INTERACTIONS"
GEMINI_INTERACTIONS_CONFIG_FLAG = "experimental_gemini_interactions_enabled"
_INTERACTIONS_API_REVISION = "2026-05-20"


def gemini_interactions_enabled(cfg: AppConfig | None = None) -> bool:
    """Return whether the experimental Gemini Interactions protocol is explicitly enabled."""
    env_value = str(env_get(GEMINI_INTERACTIONS_EXPERIMENT_ENV, "") or "").strip().lower()
    if env_value in {"1", "true", "yes", "on"}:
        return True
    if env_value in {"0", "false", "no", "off"}:
        return False
    return bool(getattr(cfg, GEMINI_INTERACTIONS_CONFIG_FLAG, False))


def gemini_interactions_disabled_message() -> str:
    return (
        "protocol='gemini_interactions' is experimental and disabled by default. "
        f"Set {GEMINI_INTERACTIONS_EXPERIMENT_ENV}=1 or run "
        f"`sylliptor config set {GEMINI_INTERACTIONS_CONFIG_FLAG} true` to enable the "
        "text-only prototype. Gemini GenerateContent remains the stable native Gemini protocol."
    )


class GeminiInteractionsClient:
    """Minimal text-only prototype for Google's beta Gemini Interactions API."""

    supports_forced_tool_choice = False

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_s: float = 60.0,
        temperature: float = 0.2,
        prompt_cache_key: str | None = None,
        prompt_cache_retention: str | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        transport: httpx.BaseTransport | None = None,
        extra_headers: dict[str, str] | None = None,
        provider_key: str | None = None,
        provider_concurrency_caps: dict[str, int] | None = None,
        provider_retry_settings: ProviderRetrySettings | None = None,
        provider_sleep_fn: Callable[[float], None] | None = None,
        provider_random_fn: Callable[[], float] | None = None,
    ) -> None:
        self.base_url = _gemini_interactions_base_url(base_url)
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.prompt_cache_key = str(prompt_cache_key or "").strip() or None
        self.prompt_cache_retention = str(prompt_cache_retention or "").strip() or None
        self.enable_thinking = enable_thinking
        self.reasoning_effort = str(reasoning_effort or "").strip().lower() or None
        self._transport = transport
        self.extra_headers = {
            str(key): str(value)
            for key, value in (extra_headers or {}).items()
            if str(key).strip() and str(value).strip()
        }
        self.provider_key = str(provider_key or "").strip() or None
        self.provider_concurrency_caps = dict(
            DEFAULT_PROVIDER_CONCURRENCY_CAPS
            if provider_concurrency_caps is None
            else provider_concurrency_caps
        )
        self.provider_retry_settings = provider_retry_settings or ProviderRetrySettings()
        self._provider_sleep_fn = provider_sleep_fn
        self._provider_random_fn = provider_random_fn

    def _headers(self) -> dict[str, str]:
        headers = {
            "Api-Revision": _INTERACTIONS_API_REVISION,
            "Content-Type": "application/json",
            "User-Agent": "sylliptor-agent-cli/0.1.0",
            "x-goog-api-key": self.api_key,
        }
        headers.update(self.extra_headers)
        return headers

    @staticmethod
    def _llm_error_from_response(response: httpx.Response) -> LLMError:
        try:
            data = response.json()
        except Exception:
            body = response.text
            if len(body) > 1000:
                body = body[:1000] + "...(truncated)"
            return LLMError(f"Gemini Interactions error {response.status_code}: {body}")
        message = _extract_error_message(data)
        if message:
            return LLMError(f"Gemini Interactions error {response.status_code}: {message}")
        return LLMError(f"Gemini Interactions error {response.status_code}: {data!r}")

    def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
        stream: bool = False,
        on_text_delta: Callable[[str], None] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        del on_text_delta
        if stream:
            raise LLMError("Gemini Interactions prototype does not support streaming yet")
        if tools:
            raise LLMError("Gemini Interactions prototype does not support tools yet")
        if tool_choice is not None:
            raise LLMError("Gemini Interactions prototype does not support tool_choice yet")
        if response_format is not None:
            raise LLMError("Gemini Interactions prototype does not support response_format yet")
        if self.prompt_cache_key or self.prompt_cache_retention:
            raise LLMError("Gemini Interactions does not support prompt_cache_key settings")
        if self.enable_thinking is not None or self.reasoning_effort:
            raise LLMError("Gemini Interactions prototype does not support thinking settings yet")

        prompt = _last_user_text(messages)
        if not prompt:
            raise LLMError("Gemini Interactions prototype requires a non-empty user message")
        _validate_text_only_single_turn_history(messages)

        payload: dict[str, Any] = {
            "input": prompt,
            "model": self.model,
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        elif self.temperature is not None:
            payload["temperature"] = self.temperature
        if max_tokens is not None:
            payload["max_output_tokens"] = int(max_tokens)

        provider_key = self.provider_key or best_effort_provider_key(
            base_url=self.base_url,
            model=self.model,
        )
        telemetry = ProviderCallTelemetryRecorder(
            provider_key=provider_key,
            protocol="gemini_interactions",
            model=self.model,
            base_url=self.base_url,
            stream=False,
            tools=tools,
            web_search_mode="off",
            web_search_adapter="gemini_grounding",
            native_web_search=False,
            operation="gemini_interactions_chat",
        )

        def _send_request() -> LLMResponse:
            try:
                with httpx.Client(timeout=self.timeout_s, transport=self._transport) as client:
                    response = client.post(
                        f"{self.base_url}/interactions",
                        headers=self._headers(),
                        json=payload,
                    )
            except Exception as exc:  # noqa: BLE001
                if isinstance(exc, LLMError):
                    raise
                raise LLMError(f"Gemini Interactions request failed: {exc}") from exc
            if response.status_code >= 400:
                raise self._llm_error_from_response(response)
            return self._parse_chat_response(response)

        return telemetry.run(
            lambda: run_provider_limited_call(
                call=_send_request,
                provider_key=provider_key,
                provider_concurrency_caps=self.provider_concurrency_caps,
                retry_settings=self.provider_retry_settings,
                operation="gemini_interactions_chat",
                sleep_fn=self._provider_sleep_fn,
                random_fn=self._provider_random_fn,
                on_retry=telemetry.on_retry,
                retry_deadline_allows=getattr(self, "_provider_retry_deadline_allows", None),
            )
        )

    def _parse_chat_response(self, response: httpx.Response) -> LLMResponse:
        try:
            data = response.json()
        except Exception as exc:
            raise LLMError(f"Gemini Interactions returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMError("Gemini Interactions returned non-object JSON")
        text = _interaction_output_text(data).strip()
        actions = data.get("actions")
        if isinstance(actions, list) and actions:
            raise LLMError(
                "Gemini Interactions returned actions, but this prototype only supports text output"
            )
        metadata: dict[str, Any] = {}
        interaction_id = data.get("id") or data.get("interaction_id") or data.get("interactionId")
        if isinstance(interaction_id, str) and interaction_id:
            metadata["interaction_id"] = interaction_id
        response_payload = data.get("response")
        if isinstance(response_payload, dict):
            metadata["response"] = copy.deepcopy(response_payload)

        return LLMResponse(
            content=text,
            tool_calls=[],
            raw=data,
            response_model=_response_model(data) or self.model,
            usage=_usage_from_response(data),
            provider_metadata=(
                {GEMINI_INTERACTIONS_PROVIDER_METADATA_KEY: metadata} if metadata else None
            ),
        )


def _gemini_interactions_base_url(base_url: str) -> str:
    normalized = str(base_url or "").strip().rstrip("/")
    if normalized.endswith("/openai"):
        normalized = normalized.removesuffix("/openai")
    return normalized or "https://generativelanguage.googleapis.com/v1beta"


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").strip() != "user":
            continue
        text = _content_text(message.get("content"))
        if text.strip():
            return text.strip()
    return ""


def _content_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_content_text(item) for item in value)
    if isinstance(value, dict):
        text = value.get("text") or value.get("content")
        return text if isinstance(text, str) else ""
    return str(value)


def _interaction_output_text(data: dict[str, Any]) -> str:
    output = data.get("output")
    if isinstance(output, str):
        return output
    text = _collect_text(output)
    if text:
        return text
    text = _collect_text(data.get("outputs"))
    if text:
        return text
    return _collect_text(data.get("response"))


def _collect_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_collect_text(item) for item in value)
    if isinstance(value, dict):
        parts: list[str] = []
        text = value.get("text")
        if isinstance(text, str):
            parts.append(text)
        for key in ("content", "parts", "candidates", "output", "response"):
            child = value.get(key)
            if child is not None:
                parts.append(_collect_text(child))
        return "".join(parts)
    return ""


def _usage_from_response(data: dict[str, Any]) -> LLMUsage | None:
    usage = data.get("usageMetadata") or data.get("usage_metadata") or data.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt_tokens = _int_or_none(
        usage.get("promptTokenCount")
        or usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("total_input_tokens")
    )
    completion_tokens = _int_or_none(
        usage.get("candidatesTokenCount")
        or usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("total_output_tokens")
    )
    total_tokens = _int_or_none(usage.get("totalTokenCount") or usage.get("total_tokens"))
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None
    return LLMUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _response_model(data: dict[str, Any]) -> str | None:
    for key in ("modelVersion", "model_version", "model"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    response = data.get("response")
    if isinstance(response, dict):
        for key in ("modelVersion", "model_version", "model"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _validate_text_only_single_turn_history(messages: list[dict[str, Any]]) -> None:
    user_message_count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").strip()
        if role in {"system", "developer"}:
            raise LLMError(
                "Gemini Interactions prototype does not support system/developer instructions yet"
            )
        if role == "user":
            user_message_count += 1
            continue
        raise LLMError(
            "Gemini Interactions prototype supports only a single text-only user turn; "
            "server-side previous_interaction_id continuation is not implemented yet"
        )
    if user_message_count != 1:
        raise LLMError(
            "Gemini Interactions prototype supports exactly one user message until "
            "previous_interaction_id continuation is implemented"
        )


def _extract_error_message(data: Any) -> str:
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        message = data.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    return ""
