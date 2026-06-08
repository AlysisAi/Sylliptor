from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from ..config import (
    resolve_llm_enable_thinking,
    resolve_llm_reasoning_effort,
    resolve_web_search_adapter,
    resolve_web_search_mode,
)
from ..model_registry import resolve_model_provider_key
from ..profiles import ProfileSpec, get_active_profile, resolve_effective_base_url
from . import (
    anthropic_messages,
    gemini_generate_content,
    gemini_interactions,
    openai_compat,
    openai_responses,
)
from .base import ChatClient
from .protocols import (
    ANTHROPIC_MESSAGES_PROTOCOL,
    GEMINI_GENERATE_CONTENT_PROTOCOL,
    GEMINI_INTERACTIONS_PROTOCOL,
    OPENAI_COMPAT_PROTOCOL,
    OPENAI_RESPONSES_PROTOCOL,
    UnsupportedProtocolError,
)
from .provider_limits import resolve_provider_retry_settings

if TYPE_CHECKING:
    from ..config import AppConfig


def make_llm_client(
    *,
    cfg: AppConfig,
    api_key: str,
    model: str,
    timeout_s: float | None = None,
    temperature: float = 0.2,
    prompt_cache_key: str | None = None,
    prompt_cache_retention: str | None = None,
    enable_thinking: bool | None = None,
    reasoning_effort: str | None = None,
    transport: httpx.BaseTransport | None = None,
    profile: ProfileSpec | None = None,
) -> ChatClient:
    """Build an LLM client using the resolved provider profile protocol."""
    resolved_profile = profile or get_active_profile(cfg)
    protocol = str(resolved_profile.protocol or OPENAI_COMPAT_PROTOCOL).strip()
    base_url = _resolve_base_url(cfg=cfg, profile=resolved_profile)
    resolved_enable_thinking = (
        resolve_llm_enable_thinking(cfg) if enable_thinking is None else enable_thinking
    )
    resolved_reasoning_effort = (
        resolve_llm_reasoning_effort(cfg) if reasoning_effort is None else reasoning_effort
    )
    provider_key = resolve_model_provider_key(
        cfg=cfg,
        model_name=model,
        base_url=base_url,
        profile_name=resolved_profile.name,
    )
    if protocol == OPENAI_RESPONSES_PROTOCOL:
        return openai_responses.OpenAIResponsesClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=60.0 if timeout_s is None else timeout_s,
            temperature=temperature,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            enable_thinking=resolved_enable_thinking,
            reasoning_effort=resolved_reasoning_effort,
            transport=transport,
            extra_headers=resolved_profile.extra_headers,
            provider_key=provider_key,
            web_search_mode=resolve_web_search_mode(cfg),
            web_search_adapter=resolve_web_search_adapter(cfg),
            provider_concurrency_caps=cfg.provider_concurrency_caps,
            provider_retry_settings=resolve_provider_retry_settings(cfg),
        )

    if protocol == ANTHROPIC_MESSAGES_PROTOCOL:
        return anthropic_messages.AnthropicMessagesClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=60.0 if timeout_s is None else timeout_s,
            temperature=temperature,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            enable_thinking=resolved_enable_thinking,
            reasoning_effort=resolved_reasoning_effort,
            transport=transport,
            extra_headers=resolved_profile.extra_headers,
            provider_key=provider_key,
            web_search_mode=resolve_web_search_mode(cfg),
            web_search_adapter=resolve_web_search_adapter(cfg),
            provider_concurrency_caps=cfg.provider_concurrency_caps,
            provider_retry_settings=resolve_provider_retry_settings(cfg),
        )

    if protocol == GEMINI_GENERATE_CONTENT_PROTOCOL:
        return gemini_generate_content.GeminiGenerateContentClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=60.0 if timeout_s is None else timeout_s,
            temperature=temperature,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            enable_thinking=resolved_enable_thinking,
            reasoning_effort=resolved_reasoning_effort,
            transport=transport,
            extra_headers=resolved_profile.extra_headers,
            provider_key=provider_key,
            web_search_mode=resolve_web_search_mode(cfg),
            web_search_adapter=resolve_web_search_adapter(cfg),
            provider_concurrency_caps=cfg.provider_concurrency_caps,
            provider_retry_settings=resolve_provider_retry_settings(cfg),
        )

    if protocol == GEMINI_INTERACTIONS_PROTOCOL:
        if not gemini_interactions.gemini_interactions_enabled(cfg):
            raise UnsupportedProtocolError(
                gemini_interactions.gemini_interactions_disabled_message()
            )
        return gemini_interactions.GeminiInteractionsClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout_s=60.0 if timeout_s is None else timeout_s,
            temperature=temperature,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_retention=prompt_cache_retention,
            enable_thinking=resolved_enable_thinking,
            reasoning_effort=resolved_reasoning_effort,
            transport=transport,
            extra_headers=resolved_profile.extra_headers,
            provider_key=provider_key,
            provider_concurrency_caps=cfg.provider_concurrency_caps,
            provider_retry_settings=resolve_provider_retry_settings(cfg),
        )

    if protocol != OPENAI_COMPAT_PROTOCOL:
        raise UnsupportedProtocolError(
            f"Profile {resolved_profile.name!r} uses recognized LLM protocol {protocol!r}, "
            "but Sylliptor does not implement a native chat client for that protocol yet. "
            "Use protocol='openai_compat', protocol='openai_responses', "
            "protocol='anthropic_messages', protocol='gemini_generate_content', or enable the "
            "experimental protocol='gemini_interactions' text-only prototype."
        )
    return openai_compat.OpenAICompatClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_s=60.0 if timeout_s is None else timeout_s,
        temperature=temperature,
        prompt_cache_key=prompt_cache_key,
        prompt_cache_retention=prompt_cache_retention,
        enable_thinking=resolved_enable_thinking,
        reasoning_effort=resolved_reasoning_effort,
        transport=transport,
        extra_headers=resolved_profile.extra_headers,
        provider_key=provider_key,
        provider_concurrency_caps=cfg.provider_concurrency_caps,
        provider_retry_settings=resolve_provider_retry_settings(cfg),
    )


def _resolve_base_url(*, cfg: AppConfig, profile: ProfileSpec) -> str:
    return resolve_effective_base_url(cfg=cfg, profile=profile)
