from __future__ import annotations

import os
from pathlib import Path

from typer.testing import CliRunner

from sylliptor_agent_cli.cli import app as sylliptor_app
from sylliptor_agent_cli.config import (
    AppConfig,
    load_config,
    load_persisted_profile_keys,
    save_config,
)
from sylliptor_agent_cli.profiles import ProfileSpec, add_profile, set_active_profile


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "SYLLIPTOR_CONFIG_DIR": os.fspath(tmp_path),
        "SYLLIPTOR_API_KEY": "",
        "OPENAI_API_KEY": "",
    }


def _seed_profiles(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    cfg = AppConfig(model="gpt-test")
    cfg.extra_fields = {"profiles": {}, "active_profile": ""}
    add_profile(cfg, ProfileSpec(name="openai", base_url="https://api.openai.com/v1"))
    add_profile(cfg, ProfileSpec(name="anthropic", base_url="https://api.anthropic.com/v1"))
    set_active_profile(cfg, "openai")
    save_config(cfg)


def test_profile_list_shows_profiles_with_active_marker(monkeypatch, tmp_path: Path) -> None:
    _seed_profiles(tmp_path, monkeypatch)

    result = CliRunner().invoke(sylliptor_app, ["profile", "list"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "openai" in result.output
    assert "✓" in result.output


def test_profile_use_switches_active(monkeypatch, tmp_path: Path) -> None:
    _seed_profiles(tmp_path, monkeypatch)

    result = CliRunner().invoke(sylliptor_app, ["profile", "use", "anthropic"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert load_config().extra_fields["active_profile"] == "anthropic"


def test_profile_use_unknown_errors(tmp_path: Path) -> None:
    result = CliRunner().invoke(sylliptor_app, ["profile", "use", "missing"], env=_env(tmp_path))

    assert result.exit_code == 2
    assert "Profile not found" in result.output


def test_profile_add_creates_profile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    result = CliRunner().invoke(
        sylliptor_app,
        ["profile", "add", "custom", "--base-url", "https://example.com/v1"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "custom" in load_config().extra_fields["profiles"]


def test_profile_remove_with_yes_skips_confirm(monkeypatch, tmp_path: Path) -> None:
    _seed_profiles(tmp_path, monkeypatch)

    result = CliRunner().invoke(
        sylliptor_app,
        ["profile", "remove", "anthropic", "--yes"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "anthropic" not in load_config().extra_fields["profiles"]


def test_profile_preset_clones_into_named_profile(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    result = CliRunner().invoke(
        sylliptor_app,
        ["profile", "preset", "anthropic", "--as", "claude"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    profile = load_config().extra_fields["profiles"]["claude"]
    assert profile["protocol"] == "anthropic_messages"
    assert profile["base_url"] == "https://api.anthropic.com/v1"
    assert profile["default_model"] == "claude-sonnet-4-6"
    assert profile["extra_headers"] == {}


def test_profile_set_key_persists_per_profile(monkeypatch, tmp_path: Path) -> None:
    _seed_profiles(tmp_path, monkeypatch)

    result = CliRunner().invoke(
        sylliptor_app,
        ["profile", "set-key", "anthropic", "--key", "sk-ant-test"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert load_persisted_profile_keys()["anthropic"] == "sk-ant-test"


def test_profile_presets_lists_known_presets(tmp_path: Path) -> None:
    result = CliRunner().invoke(sylliptor_app, ["profile", "presets"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "anthropic" in result.output
    assert "compatibility" in result.output
    assert "native" in result.output
    assert "openai_responses" in result.output
    assert "openrouter" in result.output
    assert "qwen-us" in result.output


def test_qwen_us_preset_uses_dashscope_virginia_endpoint(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))

    result = CliRunner().invoke(
        sylliptor_app,
        ["profile", "preset", "qwen-us"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    profile = load_config().extra_fields["profiles"]["qwen-us"]
    assert profile["base_url"] == "https://dashscope-us.aliyuncs.com/compatible-mode/v1"
    assert profile["default_model"] == "qwen3.7-plus"


def test_profile_convert_to_native_updates_protocol_without_exposing_key(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _seed_profiles(tmp_path, monkeypatch)
    cfg = load_config()
    add_profile(
        cfg,
        ProfileSpec(
            name="anthropic",
            base_url="https://api.anthropic.com/v1",
            default_model="claude-sonnet-4-6",
            web_search_adapter="anthropic_messages",
            web_search_model="legacy-search-model",
        ),
    )
    save_config(cfg)
    set_key = CliRunner().invoke(
        sylliptor_app,
        ["profile", "set-key", "anthropic", "--key", "sk-ant-test"],
        env=_env(tmp_path),
    )
    assert set_key.exit_code == 0

    result = CliRunner().invoke(
        sylliptor_app,
        ["profile", "convert", "anthropic", "--to", "native", "--yes"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert "anthropic_messages" in result.output
    assert "web_search_model" in result.output
    assert "legacy-search-model" in result.output
    assert "sk-ant-test" not in result.output
    profile = load_config().extra_fields["profiles"]["anthropic"]
    assert profile["protocol"] == "anthropic_messages"
    assert profile["base_url"] == "https://api.anthropic.com/v1"
    assert profile["web_search_adapter"] == "anthropic_messages"
    assert profile["web_search_model"] == ""
    assert load_persisted_profile_keys()["anthropic"] == "sk-ant-test"


def test_profile_convert_to_compatibility_updates_gemini_native_profile(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    cfg = AppConfig(model="gemini-3-flash-preview")
    cfg.extra_fields = {"profiles": {}, "active_profile": ""}
    add_profile(
        cfg,
        ProfileSpec(
            name="gemini-native",
            protocol="gemini_generate_content",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GEMINI_API_KEY",
            default_model="gemini-3-flash-preview",
            web_search_adapter="gemini_grounding",
        ),
    )
    set_active_profile(cfg, "gemini-native")
    save_config(cfg)

    result = CliRunner().invoke(
        sylliptor_app,
        ["profile", "convert", "--to", "compatibility", "--yes"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0, result.output
    profile = load_config().extra_fields["profiles"]["gemini-native"]
    assert profile["protocol"] == "openai_compat"
    assert profile["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert profile["default_model"] == "gemini-3.5-flash"
    assert profile["web_search_adapter"] == "gemini_grounding"
