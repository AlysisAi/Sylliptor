from __future__ import annotations

import re

from sylliptor_agent_cli.profile_presets import (
    PROFILE_PRESETS,
    find_preset_for_base_url,
    find_preset_for_profile,
    get_preset,
    make_profile_from_preset,
)
from sylliptor_agent_cli.profiles import ProfileSpec


def test_at_least_15_presets_registered() -> None:
    assert len(PROFILE_PRESETS) >= 15


def test_each_preset_has_non_empty_label_and_protocol() -> None:
    for preset in PROFILE_PRESETS:
        assert preset.label
        assert preset.protocol == "openai_compat"


def test_each_non_custom_preset_has_valid_base_url_and_suggested_models() -> None:
    url_pattern = re.compile(r"^https?://[^/].*$")
    for preset in PROFILE_PRESETS:
        if preset.key == "custom":
            continue
        assert url_pattern.match(preset.base_url), preset.key
        assert preset.suggested_models, preset.key


def test_openai_preset_uses_working_chat_completion_models() -> None:
    preset = get_preset("openai")

    assert preset is not None
    assert preset.base_url == "https://api.openai.com/v1"
    assert preset.suggested_models == (
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    )


def test_provider_presets_use_current_openai_compatible_base_urls() -> None:
    expected_base_urls = {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1/",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "deepseek": "https://api.deepseek.com",
        "qwen-intl": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "qwen-us": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        "qwen-cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "zhipu": "https://open.bigmodel.cn/api/paas/v4/",
        "moonshot": "https://api.moonshot.cn/v1",
        "minimax": "https://api.minimax.io/v1",
        "bytedance": "https://ark.cn-beijing.volces.com/api/v3",
        "01ai": "https://api.lingyiwanwu.com/v1",
        "groq": "https://api.groq.com/openai/v1",
        "cerebras": "https://api.cerebras.ai/v1",
        "mistral": "https://api.mistral.ai/v1",
        "xai": "https://api.x.ai/v1",
        "cohere": "https://api.cohere.ai/compatibility/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "perplexity": "https://api.perplexity.ai",
        "together": "https://api.together.ai/v1",
        "fireworks": "https://api.fireworks.ai/inference/v1",
        "ollama": "http://localhost:11434/v1",
        "lm-studio": "http://localhost:1234/v1",
        "vllm": "http://localhost:8000/v1",
        "custom": "",
    }

    actual_base_urls = {preset.key: preset.base_url for preset in PROFILE_PRESETS}

    assert actual_base_urls == expected_base_urls


def test_anthropic_preset_uses_openai_compat_v1_endpoint_and_current_models() -> None:
    preset = get_preset("anthropic")

    assert preset is not None
    assert preset.base_url == "https://api.anthropic.com/v1/"
    assert preset.base_url.rstrip("/").endswith("/v1")
    assert preset.extra_headers == {}
    assert preset.suggested_models == (
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    )


def test_launch_provider_presets_use_supported_chat_models() -> None:
    expected_models = {
        "deepseek": ("deepseek-v4-pro", "deepseek-v4-flash"),
        "gemini": (
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gemini-3.1-flash-lite-preview",
        ),
        "groq": (
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
        ),
        "mistral": ("mistral-medium-3-5", "devstral-2512", "mistral-small-2603"),
        "openrouter": (
            "openai/gpt-5.5",
            "deepseek/deepseek-v4-pro",
            "mistralai/mistral-medium-3-5",
        ),
        "together": (
            "zai-org/GLM-5.1",
            "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
            "openai/gpt-oss-120b",
        ),
        "xai": (
            "grok-4.3",
            "grok-4.20-0309-reasoning",
            "grok-4.20-0309-non-reasoning",
        ),
    }

    for provider, models in expected_models.items():
        preset = get_preset(provider)

        assert preset is not None
        assert preset.suggested_models == models
        assert all(model in preset.suggested_model_descriptions for model in models)


def test_deepseek_preset_does_not_offer_legacy_or_retired_aliases() -> None:
    preset = get_preset("deepseek")

    assert preset is not None
    assert "deepseek-coder" not in preset.suggested_models
    assert "deepseek-chat" not in preset.suggested_models
    assert "deepseek-reasoner" not in preset.suggested_models


def test_find_preset_for_profile_matches_base_url_without_requiring_env_var() -> None:
    profile = ProfileSpec(name="legacy", base_url="https://api.openai.com/v1")

    preset = find_preset_for_profile(profile)

    assert preset is not None
    assert preset.key == "openai"


def test_find_preset_for_base_url_matches_known_provider() -> None:
    preset = find_preset_for_base_url("https://api.anthropic.com/v1/")

    assert preset is not None
    assert preset.key == "anthropic"


def test_custom_preset_has_empty_base_url() -> None:
    preset = get_preset("custom")

    assert preset is not None
    assert preset.base_url == ""


def test_make_profile_from_preset_uses_preset_key_as_name_default() -> None:
    preset = get_preset("openai")

    assert preset is not None
    assert make_profile_from_preset(preset).name == "openai"
