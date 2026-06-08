from __future__ import annotations

import re

from sylliptor_agent_cli.profile_presets import (
    PROFILE_PRESETS,
    canonical_model_alias_for_preset,
    convert_profile_to_preset,
    find_preset_for_base_url,
    find_preset_for_profile,
    get_preset,
    make_profile_from_preset,
    target_preset_for_profile_conversion,
)
from sylliptor_agent_cli.profiles import ProfileSpec


def test_at_least_15_presets_registered() -> None:
    assert len(PROFILE_PRESETS) >= 15


def test_each_preset_has_non_empty_label_and_protocol() -> None:
    for preset in PROFILE_PRESETS:
        assert preset.label
        assert preset.protocol in {
            "anthropic_messages",
            "gemini_generate_content",
            "gemini_interactions",
            "openai_compat",
            "openai_responses",
        }


def test_openai_responses_preset_is_explicit_native_opt_in() -> None:
    preset = get_preset("openai-responses")

    assert preset is not None
    assert preset.label == "OpenAI Responses"
    assert preset.protocol == "openai_responses"
    assert preset.base_url == "https://api.openai.com/v1"
    assert preset.web_search_adapter == "openai_responses"


def test_anthropic_preset_is_native_by_default() -> None:
    preset = get_preset("anthropic")

    assert preset is not None
    assert preset.label == "Anthropic Claude"
    assert preset.protocol == "anthropic_messages"
    assert preset.base_url == "https://api.anthropic.com/v1"
    assert preset.web_search_adapter == "anthropic_messages"


def test_gemini_preset_is_native_by_default() -> None:
    preset = get_preset("gemini")

    assert preset is not None
    assert preset.label == "Google Gemini"
    assert preset.protocol == "gemini_generate_content"
    assert preset.base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert preset.web_search_adapter == "gemini_grounding"


def test_gemini_interactions_is_not_a_normal_provider_preset() -> None:
    assert all(preset.protocol != "gemini_interactions" for preset in PROFILE_PRESETS)


def test_first_party_compatibility_presets_are_explicit_legacy_fallbacks() -> None:
    anthropic = get_preset("anthropic-compat")
    gemini = get_preset("gemini-compat")

    assert anthropic is not None
    assert anthropic.protocol == "openai_compat"
    assert anthropic.base_url == "https://api.anthropic.com/v1/"
    assert anthropic.web_search_adapter == "anthropic_messages"
    assert gemini is not None
    assert gemini.protocol == "openai_compat"
    assert gemini.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert gemini.web_search_adapter == "gemini_grounding"


def test_compatibility_presets_stay_openai_compatible() -> None:
    for preset in PROFILE_PRESETS:
        if preset.key in {
            "openai-responses",
            "anthropic",
            "anthropic-native",
            "gemini",
            "gemini-native",
        }:
            continue
        assert preset.protocol == "openai_compat", preset.key


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
        "gpt-5.5-pro",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.4",
    )


def test_openai_preset_preserves_legacy_nano_alias_without_hiding_current_nano() -> None:
    preset = get_preset("openai")

    assert preset is not None
    assert canonical_model_alias_for_preset(preset, "gpt-5-nano") == "gpt-5.4-nano"
    assert canonical_model_alias_for_preset(preset, "gpt-5.4-nano") == "gpt-5.4-nano"


def test_provider_presets_use_current_openai_compatible_base_urls() -> None:
    expected_base_urls = {
        "openai": "https://api.openai.com/v1",
        "openai-responses": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com/v1",
        "anthropic-compat": "https://api.anthropic.com/v1/",
        "anthropic-native": "https://api.anthropic.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta",
        "gemini-compat": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "gemini-native": "https://generativelanguage.googleapis.com/v1beta",
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


def test_anthropic_preset_uses_native_messages_endpoint_and_current_models() -> None:
    preset = get_preset("anthropic")

    assert preset is not None
    assert preset.protocol == "anthropic_messages"
    assert preset.base_url == "https://api.anthropic.com/v1"
    assert preset.base_url.rstrip("/").endswith("/v1")
    assert preset.extra_headers == {}
    assert preset.suggested_models == (
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-8",
    )


def test_first_party_presets_do_not_default_to_preview_only_models() -> None:
    for key in (
        "openai",
        "openai-responses",
        "anthropic",
        "anthropic-compat",
        "gemini",
        "gemini-compat",
    ):
        preset = get_preset(key)
        assert preset is not None
        assert preset.suggested_models
        default_model = preset.suggested_models[0]
        assert "preview" not in default_model
        assert not default_model.endswith("-latest")


def test_launch_provider_presets_use_supported_chat_models() -> None:
    expected_models = {
        "deepseek": ("deepseek-v4-pro", "deepseek-v4-flash"),
        "gemini": (
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-3.1-pro-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
        ),
        "groq": (
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
        ),
        "mistral": ("mistral-medium-3-5", "devstral-2512", "mistral-small-2603"),
        "openrouter": (
            "openai/gpt-5.5",
            "anthropic/claude-opus-4.8",
            "google/gemini-3.5-flash",
            "qwen/qwen3.7-plus",
            "deepseek/deepseek-v4-pro",
        ),
        "together": (
            "zai-org/GLM-5.1",
            "moonshotai/Kimi-K2.6",
            "deepseek-ai/DeepSeek-V4-Pro",
            "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
            "openai/gpt-oss-120b",
        ),
        "xai": (
            "grok-4.3",
            "grok-4.20-0309-reasoning",
            "grok-code-fast-1",
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


def test_find_preset_for_profile_is_protocol_aware_for_native_variants() -> None:
    cases = [
        ("openai-responses", "openai_responses", "https://api.openai.com/v1"),
        ("anthropic", "anthropic_messages", "https://api.anthropic.com/v1"),
        ("anthropic-native", "anthropic_messages", "https://api.anthropic.com/v1"),
        (
            "gemini",
            "gemini_generate_content",
            "https://generativelanguage.googleapis.com/v1beta",
        ),
        (
            "gemini-native",
            "gemini_generate_content",
            "https://generativelanguage.googleapis.com/v1beta",
        ),
    ]

    for name, protocol, base_url in cases:
        preset = find_preset_for_profile(
            ProfileSpec(name=name, protocol=protocol, base_url=base_url)
        )

        assert preset is not None
        assert preset.key == name


def test_find_preset_for_profile_prefers_protocol_base_url_over_compatibility_base_match() -> None:
    profile = ProfileSpec(
        name="work-openai",
        protocol="openai_responses",
        base_url="https://api.openai.com/v1",
    )

    preset = find_preset_for_profile(profile)

    assert preset is not None
    assert preset.key == "openai-responses"


def test_find_preset_for_profile_keeps_legacy_base_url_only_profiles_compatibility() -> None:
    anthropic = find_preset_for_profile(
        ProfileSpec(name="legacy-claude", base_url="https://api.anthropic.com/v1")
    )
    gemini = find_preset_for_profile(
        ProfileSpec(
            name="legacy-gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    )

    assert anthropic is not None
    assert anthropic.key == "anthropic-compat"
    assert gemini is not None
    assert gemini.key == "gemini-compat"


def test_profile_conversion_targets_first_party_native_and_compatibility_presets() -> None:
    profile = ProfileSpec(
        name="claude",
        protocol="openai_compat",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-4-6",
        web_search_adapter="anthropic_messages",
    )

    native_preset = target_preset_for_profile_conversion(profile, target="native")
    assert native_preset is not None
    assert native_preset.key == "anthropic"

    converted = convert_profile_to_preset(profile, native_preset)
    assert converted.name == "claude"
    assert converted.protocol == "anthropic_messages"
    assert converted.base_url == "https://api.anthropic.com/v1"
    assert converted.default_model == "claude-sonnet-4-6"

    compat_preset = target_preset_for_profile_conversion(converted, target="compatibility")
    assert compat_preset is not None
    assert compat_preset.key == "anthropic-compat"


def test_profile_conversion_replaces_known_incompatible_model() -> None:
    profile = ProfileSpec(
        name="gemini",
        protocol="openai_compat",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        default_model="claude-sonnet-4-6",
        web_search_adapter="gemini_grounding",
    )

    native_preset = target_preset_for_profile_conversion(profile, target="native")
    assert native_preset is not None

    converted = convert_profile_to_preset(profile, native_preset)

    assert converted.protocol == "gemini_generate_content"
    assert converted.default_model == "gemini-3.5-flash"


def test_profile_conversion_replaces_provider_qualified_model_ids() -> None:
    profile = ProfileSpec(
        name="anthropic",
        protocol="openai_compat",
        base_url="https://api.anthropic.com/v1",
        api_key_env="ANTHROPIC_API_KEY",
        default_model="anthropic/claude-sonnet-4-6",
        web_search_adapter="anthropic_messages",
    )

    native_preset = target_preset_for_profile_conversion(profile, target="native")
    assert native_preset is not None

    converted = convert_profile_to_preset(profile, native_preset)

    assert converted.protocol == "anthropic_messages"
    assert converted.default_model == "claude-sonnet-4-6"


def test_profile_conversion_replaces_known_stale_alias_for_target_preset() -> None:
    profile = ProfileSpec(
        name="gemini",
        protocol="openai_compat",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        default_model="gemini-3-flash-preview",
        web_search_adapter="gemini_grounding",
    )

    native_preset = target_preset_for_profile_conversion(profile, target="native")
    assert native_preset is not None

    converted = convert_profile_to_preset(profile, native_preset)

    assert converted.protocol == "gemini_generate_content"
    assert converted.default_model == "gemini-3.5-flash"


def test_find_preset_for_base_url_matches_known_provider() -> None:
    preset = find_preset_for_base_url("https://api.anthropic.com/v1/")

    assert preset is not None
    assert preset.key == "anthropic-compat"


def test_custom_preset_has_empty_base_url() -> None:
    preset = get_preset("custom")

    assert preset is not None
    assert preset.base_url == ""


def test_make_profile_from_preset_uses_preset_key_as_name_default() -> None:
    preset = get_preset("openai")

    assert preset is not None
    assert make_profile_from_preset(preset).name == "openai"
