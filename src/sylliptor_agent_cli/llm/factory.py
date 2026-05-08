from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from ..config import resolve_llm_enable_thinking, resolve_llm_reasoning_effort
from ..model_registry import resolve_model_provider_key
from ..profiles import ProfileSpec, get_active_profile, resolve_effective_base_url
from . import openai_compat
from .provider_limits import resolve_provider_retry_settings

if TYPE_CHECKING:
    from ..config import AppConfig
    from .openai_compat import OpenAICompatClient


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
) -> OpenAICompatClient:
    """Build an OpenAI-compatible client using the resolved provider profile."""
    resolved_profile = profile or get_active_profile(cfg)
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
