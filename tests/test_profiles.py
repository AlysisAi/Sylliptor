from __future__ import annotations

import os
from pathlib import Path

import pytest

from sylliptor_agent_cli.config import AppConfig, ConfigError, load_config, save_config
from sylliptor_agent_cli.llm.protocols import SUPPORTED_LLM_PROTOCOLS
from sylliptor_agent_cli.profile_presets import get_preset, make_profile_from_preset
from sylliptor_agent_cli.profiles import (
    ProfileSpec,
    add_profile,
    get_active_profile,
    list_profiles,
    remove_profile,
    set_active_profile,
    sync_active_profile_to_config,
    update_active_profile_defaults,
)


def test_profile_spec_roundtrip_to_dict() -> None:
    profile = ProfileSpec(
        name="anthropic",
        base_url="https://api.anthropic.com/v1/openai",
        api_key_env="ANTHROPIC_API_KEY",
        extra_headers={"anthropic-version": "2023-06-01"},
        default_model="claude-sonnet-4-6",
        web_search_adapter="anthropic_messages",
        web_search_model="claude-sonnet-4-6",
        notes="compat",
    )

    assert ProfileSpec.from_dict("anthropic", profile.to_dict()) == profile


def test_profile_name_validation_rejects_invalid_chars() -> None:
    with pytest.raises(ConfigError):
        ProfileSpec(name="Bad Name", base_url="https://example.com/v1")


def test_protocol_validation_rejects_unknown() -> None:
    with pytest.raises(ConfigError):
        ProfileSpec(name="test", protocol="anthropic_native", base_url="https://example.com/v1")


@pytest.mark.parametrize("protocol", sorted(SUPPORTED_LLM_PROTOCOLS))
def test_profile_spec_accepts_supported_llm_protocols(protocol: str) -> None:
    profile = ProfileSpec(name="test", protocol=protocol, base_url="https://example.com/v1")

    assert profile.protocol == protocol


def test_profile_from_dict_defaults_missing_protocol_to_openai_compat() -> None:
    profile = ProfileSpec.from_dict(
        "legacy",
        {
            "base_url": "https://example.com/v1",
            "default_model": "model",
        },
    )

    assert profile.protocol == "openai_compat"


def test_openai_responses_preset_uses_native_protocol_without_changing_openai_default() -> None:
    compat = get_preset("openai")
    native = get_preset("openai-responses")

    assert compat is not None
    assert compat.protocol == "openai_compat"
    assert native is not None
    assert native.protocol == "openai_responses"

    profile = make_profile_from_preset(native)
    assert profile.name == "openai-responses"
    assert profile.protocol == "openai_responses"
    assert profile.base_url == "https://api.openai.com/v1"
    assert profile.web_search_adapter == "openai_responses"


def test_anthropic_preset_uses_messages_protocol_by_default() -> None:
    native = get_preset("anthropic")
    compat = get_preset("anthropic-compat")

    assert compat is not None
    assert compat.protocol == "openai_compat"
    assert native is not None
    assert native.protocol == "anthropic_messages"

    profile = make_profile_from_preset(native)
    assert profile.name == "anthropic"
    assert profile.protocol == "anthropic_messages"
    assert profile.base_url == "https://api.anthropic.com/v1"
    assert profile.web_search_adapter == "anthropic_messages"


def test_gemini_preset_uses_generate_content_protocol_by_default() -> None:
    native = get_preset("gemini")
    compat = get_preset("gemini-compat")

    assert compat is not None
    assert compat.protocol == "openai_compat"
    assert native is not None
    assert native.protocol == "gemini_generate_content"

    profile = make_profile_from_preset(native)
    assert profile.name == "gemini"
    assert profile.protocol == "gemini_generate_content"
    assert profile.base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert profile.web_search_adapter == "gemini_grounding"


def test_base_url_validation_rejects_malformed_url() -> None:
    with pytest.raises(ConfigError, match="valid http:// or https:// URL"):
        ProfileSpec(name="bad", base_url="https://api.deepseek.com]")


def test_web_search_adapter_validation_rejects_unknown() -> None:
    with pytest.raises(ConfigError):
        ProfileSpec(name="test", web_search_adapter="unknown")


def test_add_profile_persists_in_extra_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    cfg = load_config()
    add_profile(cfg, ProfileSpec(name="test", base_url="https://example.com/v1"))
    save_config(cfg)

    profiles = {profile.name: profile for profile in list_profiles(load_config())}
    assert profiles["test"].base_url == "https://example.com/v1"
    assert profiles["test"].web_search_adapter == "auto"


def test_update_active_profile_defaults_preserves_web_search_fields() -> None:
    cfg = AppConfig()
    cfg.extra_fields = {"profiles": {}, "active_profile": ""}
    add_profile(
        cfg,
        ProfileSpec(
            name="anthropic",
            base_url="https://api.anthropic.com/v1",
            default_model="claude-sonnet-4-6",
            web_search_adapter="anthropic_messages",
            web_search_model="claude-sonnet-4-6",
        ),
    )
    set_active_profile(cfg, "anthropic")

    changed = update_active_profile_defaults(cfg, default_model="claude-opus-4-7")

    assert changed is True
    profile = get_active_profile(cfg)
    assert profile.default_model == "claude-opus-4-7"
    assert profile.web_search_adapter == "anthropic_messages"
    assert profile.web_search_model == "claude-sonnet-4-6"


def test_remove_active_profile_switches_to_first_remaining() -> None:
    cfg = AppConfig()
    cfg.extra_fields = {"profiles": {}, "active_profile": ""}
    add_profile(cfg, ProfileSpec(name="b", base_url="https://b.example/v1"))
    add_profile(cfg, ProfileSpec(name="a", base_url="https://a.example/v1"))
    set_active_profile(cfg, "b")

    remove_profile(cfg, "b")

    assert get_active_profile(cfg).name == "a"


def test_remove_only_profile_clears_active() -> None:
    cfg = AppConfig()
    cfg.extra_fields = {"profiles": {}, "active_profile": ""}
    add_profile(cfg, ProfileSpec(name="only", base_url="https://example.com/v1"))
    set_active_profile(cfg, "only")

    remove_profile(cfg, "only")

    assert cfg.extra_fields.get("active_profile") is None


def test_get_active_profile_raises_if_none_configured() -> None:
    cfg = AppConfig()
    cfg.extra_fields = {"profiles": {}, "active_profile": ""}

    with pytest.raises(ConfigError):
        get_active_profile(cfg)


def test_sync_active_profile_to_config_repairs_stale_top_level_base_url() -> None:
    cfg = AppConfig(model="DeepSeek-V4-Flash", base_url="https://api.deepseek.com]")
    cfg.extra_fields = {"profiles": {}, "active_profile": ""}
    add_profile(
        cfg,
        ProfileSpec(
            name="deepseek",
            base_url="https://api.deepseek.com",
            default_model="deepseek-v4-flash",
        ),
    )
    set_active_profile(cfg, "deepseek")
    cfg.base_url = "https://api.deepseek.com]"

    changed = sync_active_profile_to_config(cfg)

    assert changed is True
    assert cfg.base_url == "https://api.deepseek.com"
    assert cfg.model == "deepseek-v4-flash"


def test_update_active_profile_defaults_persists_model_and_base_url() -> None:
    cfg = AppConfig(model="old", base_url="https://old.example/v1")
    cfg.extra_fields = {"profiles": {}, "active_profile": ""}
    add_profile(cfg, ProfileSpec(name="deepseek", base_url="https://api.deepseek.com"))
    set_active_profile(cfg, "deepseek")

    assert update_active_profile_defaults(
        cfg,
        base_url="https://api.deepseek.com",
        default_model="deepseek-v4-flash",
    )

    profile = get_active_profile(cfg)
    assert cfg.base_url == "https://api.deepseek.com"
    assert cfg.model == "deepseek-v4-flash"
    assert profile.default_model == "deepseek-v4-flash"
