from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from sylliptor_agent_cli.cli_impl import config_menu as config_menu_mod
from sylliptor_agent_cli.cli_impl.chat import loop as chat_loop
from sylliptor_agent_cli.cli_impl.config_menu import (
    ConfigMenuState,
    thinking_label_from_cfg,
)
from sylliptor_agent_cli.config import (
    AppConfig,
    load_config,
    save_config,
    save_persisted_profile_key,
)
from sylliptor_agent_cli.profiles import (
    ProfileSpec,
    add_profile,
    get_active_profile,
    set_active_profile,
)


def test_state_tracks_dirty_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    state = ConfigMenuState.from_cfg(AppConfig(model="old"))

    state.set_field("model", "new")
    assert state.dirty is True

    state.reset()
    assert state.dirty is False


def test_commit_persists_model_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    cfg = load_config()
    cfg.model = "old"
    save_config(cfg)

    state = ConfigMenuState.from_cfg(load_config())
    state.set_field("model", "new")
    cfg_to_save = load_config()
    result = state.commit_to(cfg_to_save)
    save_config(cfg_to_save)

    assert result.saved is True
    assert load_config().model == "new"
    assert get_active_profile(load_config()).default_model == "new"


def test_commit_persists_default_model_section_to_active_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    cfg = AppConfig(model="old")
    cfg.extra_fields = {"profiles": {}, "active_profile": ""}
    add_profile(
        cfg,
        ProfileSpec(
            name="anthropic",
            base_url="https://api.anthropic.com/v1",
            default_model="old",
        ),
    )
    set_active_profile(cfg, "anthropic")
    save_config(cfg)

    state = ConfigMenuState.from_cfg(load_config())
    state.set_field("model", "claude-sonnet-4-6")
    state.set_field("base_url", "https://anthropic.example/v1")
    cfg_to_save = load_config()
    result = state.commit_to(cfg_to_save)
    save_config(cfg_to_save)

    loaded = load_config()
    profile = get_active_profile(loaded)
    assert result.saved is True
    assert loaded.model == "claude-sonnet-4-6"
    assert loaded.base_url == "https://anthropic.example/v1"
    assert profile.default_model == "claude-sonnet-4-6"
    assert profile.base_url == "https://anthropic.example/v1"


def test_commit_persists_subagent_role_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    state = ConfigMenuState.from_cfg(AppConfig(model="default"))
    state.set_role_model("coding", "anthropic/claude-sonnet-4-6")
    cfg_to_save = load_config()

    result = state.commit_to(cfg_to_save)
    save_config(cfg_to_save)

    assert result.saved is True
    assert load_config().extra_fields["role_models"]["coding"] == "anthropic/claude-sonnet-4-6"


def test_commit_clears_role_override_when_emptied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    cfg = load_config()
    cfg.extra_fields["role_models"] = {"coding": "x"}
    save_config(cfg)

    state = ConfigMenuState.from_cfg(load_config())
    state.set_role_model("coding", "")
    cfg_to_save = load_config()
    result = state.commit_to(cfg_to_save)
    save_config(cfg_to_save)

    assert result.saved is True
    assert "coding" not in load_config().extra_fields.get("role_models", {})


def test_commit_persists_forge_role_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    state = ConfigMenuState.from_cfg(AppConfig(model="default"))
    state.set_forge_role_model("planner", "anthropic/claude-opus-4-7")
    cfg_to_save = load_config()

    result = state.commit_to(cfg_to_save)
    save_config(cfg_to_save)

    assert result.saved is True
    assert load_config().extra_fields["forge_role_models"]["planner"] == "anthropic/claude-opus-4-7"


def test_commit_invalid_timeout_returns_validation_error() -> None:
    state = ConfigMenuState.from_cfg(AppConfig(model="default"))
    state.set_field("llm_timeout_s", "not-a-number")

    result = state.commit_to(AppConfig(model="default"))

    assert result.saved is False
    assert result.error == "Request timeout (seconds) must be a positive number."


def test_commit_invalid_base_url_returns_validation_error() -> None:
    state = ConfigMenuState.from_cfg(AppConfig(model="default"))
    state.set_field("base_url", "not-a-url")

    result = state.commit_to(AppConfig(model="default"))

    assert result.saved is False
    assert "Base URL" in str(result.error)


def test_state_from_cfg_tolerates_invalid_reasoning_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    monkeypatch.setenv("SYLLIPTOR_LLM_REASONING_EFFORT", "extreme")

    state = ConfigMenuState.from_cfg(AppConfig(model="default"))

    assert state.thinking_label == "auto"
    assert state.config_warning is not None
    assert "SYLLIPTOR_LLM_REASONING_EFFORT" in state.config_warning


def test_switching_profile_refreshes_api_key_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    monkeypatch.delenv("SYLLIPTOR_API_KEY", raising=False)
    cfg = AppConfig(model="gpt-test")
    cfg.extra_fields = {"profiles": {}, "active_profile": ""}
    add_profile(cfg, ProfileSpec(name="openai", base_url="https://api.openai.com/v1"))
    add_profile(cfg, ProfileSpec(name="anthropic", base_url="https://api.anthropic.com/v1"))
    set_active_profile(cfg, "openai")
    save_config(cfg)
    save_persisted_profile_key("anthropic", "sk-ant")

    state = ConfigMenuState.from_cfg(load_config())
    state.set_active_profile_name("anthropic")

    assert state.api_key_source == "stored:profile=anthropic"
    assert state.masked_api_key != "(not set)"


def test_default_model_rows_include_active_profile_preset_suggestions() -> None:
    cfg = AppConfig(model="deepseek-v4-pro")
    add_profile(
        cfg,
        ProfileSpec(
            name="deepseek",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            default_model="deepseek-v4-pro",
        ),
    )
    set_active_profile(cfg, "deepseek")
    state = ConfigMenuState.from_cfg(cfg)

    model_values = [
        value for value, _label, _description in config_menu_mod._default_model_rows(state)
    ]

    assert "deepseek-v4-pro" in model_values
    assert "deepseek-v4-flash" in model_values
    assert "deepseek-coder" not in model_values


def test_default_model_rows_fallback_to_base_url_provider() -> None:
    cfg = AppConfig(model="deepseek-v4-pro")
    add_profile(
        cfg,
        ProfileSpec(
            name="deepseek",
            base_url="https://api.deepseek.com",
            api_key_env="DEEPSEEK_API_KEY",
            default_model="deepseek-v4-pro",
        ),
    )
    add_profile(
        cfg,
        ProfileSpec(
            name="anthropic",
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
            default_model="claude-sonnet-4-6",
        ),
    )
    set_active_profile(cfg, "deepseek")
    state = ConfigMenuState.from_cfg(cfg)
    state.set_field("base_url", "https://api.anthropic.com/v1/")

    model_values = [
        value for value, _label, _description in config_menu_mod._default_model_rows(state)
    ]

    assert "claude-sonnet-4-6" in model_values
    assert "deepseek-v4-flash" not in model_values


def test_config_reload_updates_clients_with_effective_profile_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    monkeypatch.setenv("SYLLIPTOR_API_KEY", "sk-test")
    cfg = AppConfig(model="claude-sonnet-4-6", base_url="https://api.openai.com/v1")
    cfg.extra_fields = {"profiles": {}, "active_profile": ""}
    add_profile(
        cfg,
        ProfileSpec(
            name="anthropic",
            base_url="https://api.anthropic.com/v1",
            extra_headers={"anthropic-version": "2023-06-01"},
        ),
    )
    set_active_profile(cfg, "anthropic")
    cfg.base_url = "https://api.openai.com/v1"
    client = SimpleNamespace(
        base_url="",
        api_key="",
        model="",
        timeout_s=0.0,
        temperature=0.0,
        prompt_cache_key=None,
        prompt_cache_retention=None,
        enable_thinking=None,
        reasoning_effort=None,
        extra_headers={},
        provider_key=None,
        provider_concurrency_caps={},
        provider_retry_settings=None,
    )
    session = SimpleNamespace(
        cfg=AppConfig(),
        client=client,
        router_client=None,
        conversation_compactor=None,
        mode="review",
    )
    monkeypatch.setattr(
        chat_loop,
        "_rebuild_session_tools_for_mode",
        lambda **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        chat_loop,
        "refresh_session_environment_context_message",
        lambda _session: None,
        raising=False,
    )
    monkeypatch.setattr(
        chat_loop,
        "_refresh_chat_hud_context_cache",
        lambda _session: None,
        raising=False,
    )

    chat_loop._apply_config_menu_changes_to_session(session=session, cfg=cfg)

    assert client.base_url == "https://api.anthropic.com/v1"
    assert client.extra_headers == {"anthropic-version": "2023-06-01"}


@pytest.mark.parametrize("label", ["off", "minimal", "low", "medium", "high", "xhigh", "auto"])
def test_thinking_label_round_trip(label: str) -> None:
    cfg = AppConfig(model="default")
    state = ConfigMenuState.from_cfg(cfg)
    state.set_thinking_label(label)

    result = state.commit_to(cfg)

    assert result.saved is True
    assert thinking_label_from_cfg(cfg) == label
    if label == "off":
        assert cfg.llm_reasoning_effort == "none"
    elif label == "auto":
        assert cfg.llm_reasoning_effort is None
    else:
        assert cfg.llm_reasoning_effort == label


def test_commit_does_not_persist_env_reasoning_effort_for_unrelated_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_LLM_REASONING_EFFORT", "high")
    cfg = AppConfig(model="old")
    state = ConfigMenuState.from_cfg(cfg)

    state.set_field("model", "new")
    result = state.commit_to(cfg)

    assert result.saved is True
    assert cfg.model == "new"
    assert cfg.llm_reasoning_effort is None
    assert "llm_reasoning_effort" not in result.changes
    assert "llm_enable_thinking" not in result.changes


def test_commit_persists_explicit_env_reasoning_effort_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_LLM_REASONING_EFFORT", "high")
    cfg = AppConfig(model="default")
    state = ConfigMenuState.from_cfg(cfg)

    state.set_thinking_label("high")
    result = state.commit_to(cfg)

    assert result.saved is True
    assert cfg.llm_reasoning_effort == "high"
    assert result.changes["llm_reasoning_effort"] == "high"
