from __future__ import annotations

import os
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from sylliptor_agent_cli.cli import app as sylliptor_app
from sylliptor_agent_cli.cli_impl.commands import root as root_mod
from sylliptor_agent_cli.config import AppConfig, save_config
from sylliptor_agent_cli.llm.types import LLMResponse
from sylliptor_agent_cli.profiles import ProfileSpec, add_profile, set_active_profile
from sylliptor_agent_cli.provider_diagnostics import (
    ProviderLiveValidation,
    build_provider_diagnostics,
    validate_active_provider_live,
)


@pytest.fixture(autouse=True)
def _clear_generic_web_search_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SYLLIPTOR_WEB_SEARCH_API_KEY", raising=False)


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "SYLLIPTOR_CONFIG_DIR": os.fspath(tmp_path),
        "SYLLIPTOR_API_KEY": "",
        "OPENAI_API_KEY": "",
        "ANTHROPIC_API_KEY": "",
        "GEMINI_API_KEY": "",
        "TAVILY_API_KEY": "",
    }


def _cfg_with_profile(
    profile: ProfileSpec, *, stream: bool = False, web_search_mode: str = "auto"
) -> AppConfig:
    cfg = AppConfig(
        model=profile.default_model or "test-model", stream=stream, web_search_mode=web_search_mode
    )
    cfg.extra_fields = {"profiles": {}, "active_profile": ""}
    add_profile(cfg, profile)
    set_active_profile(cfg, profile.name)
    return cfg


def _render_provider_table(cfg: AppConfig) -> str:
    console = Console(record=True, width=180)
    console.print(root_mod._provider_doctor_table(cfg))
    return console.export_text()


def test_provider_diagnostics_redacts_api_key_and_shows_native_vs_compat(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret-value")
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="anthropic",
            protocol="anthropic_messages",
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
            default_model="claude-sonnet-4-6",
            web_search_adapter="anthropic_messages",
        ),
        web_search_mode="native",
    )

    output = _render_provider_table(cfg)

    assert "anthropic" in output
    assert "anthropic_messages" in output
    assert "native" in output
    assert "api.anthropic.com" in output
    assert "https://api.anthropic.com/v1" not in output
    assert "sk-ant-secret-value" not in output
    assert "env:ANTHROPIC_API_KEY (redacted)" in output
    assert "provider_hosted" not in output
    assert "native/provider-hosted" in output


def test_provider_diagnostics_shows_compatibility_protocol_and_external_search(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret-value")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-secret-value")
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai",
            protocol="openai_compat",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-5.5",
            web_search_adapter="tavily",
        ),
        web_search_mode="external",
    )

    output = _render_provider_table(cfg)

    assert "openai_compat" in output
    assert "compatibility" in output
    assert "external" in output
    assert "api.openai.com" in output
    assert "sk-openai-secret-value" not in output
    assert "tvly-secret-value" not in output


def test_provider_diagnostics_surfaces_effective_cache_policy() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai-responses",
            protocol="openai_responses",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-5.5",
            web_search_adapter="openai_responses",
        )
    )

    diagnostics = build_provider_diagnostics(cfg)
    rows = dict(diagnostics.rows())

    assert rows["cache_status"] == "available"
    assert rows["cache_strategy"] == "openai_prompt_cache"
    assert rows["cache_capability_source"] in {"preset", "protocol"}
    assert "prompt_cache_key" in rows["cache_allowed_fields"]
    assert rows["cache_emitted_fields"] == "none"


def test_provider_diagnostics_accepts_auth_backed_responses_endpoint() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="chatgpt-codex",
            protocol="openai_responses",
            base_url="https://chatgpt.com/backend-api/codex",
            auth_provider="openai-codex",
            default_model="gpt-5.4",
        ),
        web_search_mode="off",
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert not any("intended for the OpenAI Responses API" in issue for issue in diagnostics.issues)


def test_provider_diagnostics_marks_cache_policy_disabled_when_cache_mode_off() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai-responses",
            protocol="openai_responses",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-5.5",
            web_search_adapter="openai_responses",
        )
    )
    cfg.prompt_cache_mode = "off"

    diagnostics = build_provider_diagnostics(cfg)
    rows = dict(diagnostics.rows())

    assert rows["cache_status"] == "disabled"
    assert rows["cache_strategy"] == "openai_prompt_cache"
    assert "prompt_cache_key" in rows["cache_allowed_fields"]
    assert rows["cache_emitted_fields"] == "none"


def test_provider_diagnostics_allows_openai_responses_streaming() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai-responses",
            protocol="openai_responses",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-5.5",
            web_search_adapter="openai_responses",
        ),
        stream=True,
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert diagnostics.streaming_supported is True
    assert diagnostics.stream_enabled is True
    assert not any("does not support streaming yet" in issue for issue in diagnostics.issues)


def test_provider_diagnostics_allows_anthropic_native_streaming() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="anthropic-native",
            protocol="anthropic_messages",
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
            default_model="claude-sonnet-4-6",
            web_search_adapter="anthropic_messages",
        ),
        stream=True,
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert diagnostics.streaming_supported is True
    assert diagnostics.stream_enabled is True
    assert not any("does not support streaming yet" in issue for issue in diagnostics.issues)


def test_provider_diagnostics_allows_gemini_native_streaming() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="gemini-native",
            protocol="gemini_generate_content",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GEMINI_API_KEY",
            default_model="gemini-3-flash-preview",
            web_search_adapter="gemini_grounding",
        ),
        stream=True,
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert diagnostics.streaming_supported is True
    assert diagnostics.stream_enabled is True
    assert not any("does not support streaming yet" in issue for issue in diagnostics.issues)


def test_provider_diagnostics_marks_gemini_interactions_experimental(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SYLLIPTOR_EXPERIMENTAL_GEMINI_INTERACTIONS", raising=False)
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="gemini-interactions",
            protocol="gemini_interactions",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GEMINI_API_KEY",
            default_model="gemini-2.5-flash",
            web_search_adapter="gemini_grounding",
        )
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert diagnostics.protocol_kind == "native"
    assert diagnostics.streaming_supported is False
    assert any("experimental and disabled by default" in issue for issue in diagnostics.issues)
    assert any(
        "Gemini GenerateContent remains the stable native Gemini protocol" in quirk
        for quirk in diagnostics.quirks
    )


def test_provider_diagnostics_allows_enabled_gemini_interactions_experiment(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_EXPERIMENTAL_GEMINI_INTERACTIONS", "1")
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="gemini-interactions",
            protocol="gemini_interactions",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GEMINI_API_KEY",
            default_model="gemini-2.5-flash",
            web_search_adapter="gemini_grounding",
        )
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert not any("experimental and disabled by default" in issue for issue in diagnostics.issues)


def test_provider_diagnostics_reports_external_search_missing_credentials(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai",
            protocol="openai_compat",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-5.5",
            web_search_adapter="tavily",
        ),
        web_search_mode="external",
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert diagnostics.web_search_mode == "external"
    assert diagnostics.web_search_backend_kind == "external"
    assert diagnostics.web_search_registration_ready is False
    assert any("TAVILY_API_KEY" in issue for issue in diagnostics.issues)
    assert any("sylliptor config set web_search_mode auto" in issue for issue in diagnostics.issues)


def test_provider_diagnostics_reports_policy_disabled_registration(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai",
            protocol="openai_responses",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-5.5",
            web_search_adapter="openai_responses",
        )
    )
    cfg.web_search_policy = "off"

    diagnostics = build_provider_diagnostics(cfg)

    assert diagnostics.web_search_policy == "off"
    assert diagnostics.web_search_registration_ready is False
    assert any("prevents web_search tool registration" in note for note in diagnostics.notes)


def test_provider_diagnostics_reports_missing_active_profile_api_key(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    monkeypatch.delenv("SYLLIPTOR_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="anthropic-native",
            protocol="anthropic_messages",
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
            default_model="claude-sonnet-4-6",
            web_search_adapter="anthropic_messages",
        )
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert diagnostics.api_key_present is False
    assert diagnostics.api_key_source == "missing"
    assert any(
        "API key is missing" in issue
        and "ANTHROPIC_API_KEY" in issue
        and "sylliptor config set-api-key" in issue
        for issue in diagnostics.issues
    )


def test_provider_diagnostics_reports_custom_compat_native_search_mismatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    monkeypatch.delenv("SYLLIPTOR_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="custom",
            protocol="openai_compat",
            base_url="https://gateway.example.test/v1",
            default_model="gateway-model",
            web_search_adapter="auto",
        ),
        web_search_mode="native",
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert diagnostics.protocol_kind == "compatibility"
    assert diagnostics.web_search_registration_ready is False
    assert any("custom OpenAI-compatible profiles" in issue for issue in diagnostics.issues)
    assert any("web_search_mode=native" in issue for issue in diagnostics.issues)


def test_provider_diagnostics_reports_web_search_mode_adapter_mismatch_suggestions() -> None:
    native_mode_cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai",
            protocol="openai_compat",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-5.5",
            web_search_adapter="tavily",
        ),
        web_search_mode="native",
    )
    external_mode_cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai-responses",
            protocol="openai_responses",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-5.5",
            web_search_adapter="openai_responses",
        ),
        web_search_mode="external",
    )

    native_issues = build_provider_diagnostics(native_mode_cfg).issues
    external_issues = build_provider_diagnostics(external_mode_cfg).issues

    assert any("web_search_mode=native" in issue for issue in native_issues)
    assert any("sylliptor config set web_search_mode external" in issue for issue in native_issues)
    assert any("web_search_mode=external" in issue for issue in external_issues)
    assert any("sylliptor config set web_search_mode native" in issue for issue in external_issues)


def test_provider_diagnostics_reports_native_named_profile_using_compat_protocol() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="anthropic-native",
            protocol="openai_compat",
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
            default_model="claude-sonnet-4-6",
            web_search_adapter="anthropic_messages",
        )
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert any(
        "named like a native profile" in issue
        and "sylliptor profile convert anthropic-native --to native" in issue
        for issue in diagnostics.issues
    )


def test_provider_diagnostics_reports_anthropic_first_party_host_using_compat_protocol() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="anthropic",
            protocol="openai_compat",
            base_url="https://api.anthropic.com",
            api_key_env="ANTHROPIC_API_KEY",
            default_model="claude-sonnet-4-6",
            web_search_adapter="anthropic_messages",
        )
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert any("legacy compatibility semantics" in issue for issue in diagnostics.issues)
    assert any(
        "Anthropic first-party API using compatibility mode" in issue
        and "sylliptor profile convert anthropic --to native" in issue
        for issue in diagnostics.issues
    )


def test_provider_diagnostics_does_not_warn_for_explicit_anthropic_compat_profile() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="anthropic-compat",
            protocol="openai_compat",
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
            default_model="claude-sonnet-4-6",
            web_search_adapter="anthropic_messages",
        )
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert not any("legacy compatibility semantics" in issue for issue in diagnostics.issues)
    assert not any(
        "Anthropic first-party API using compatibility mode" in issue
        for issue in diagnostics.issues
    )


def test_provider_diagnostics_reports_legacy_gemini_profile_using_compat_protocol() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="gemini",
            protocol="openai_compat",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key_env="GEMINI_API_KEY",
            default_model="gemini-3-flash-preview",
            web_search_adapter="gemini_grounding",
        )
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert any(
        "legacy compatibility semantics" in issue
        and "sylliptor profile convert gemini --to native" in issue
        for issue in diagnostics.issues
    )


def test_provider_diagnostics_reports_gemini_native_with_openai_compatible_base_url() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="gemini",
            protocol="gemini_generate_content",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key_env="GEMINI_API_KEY",
            default_model="gemini-3-flash-preview",
            web_search_adapter="gemini_grounding",
        )
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert any(
        "Gemini OpenAI-compatible endpoint" in issue
        and "sylliptor profile convert gemini --to compatibility" in issue
        for issue in diagnostics.issues
    )


def test_provider_diagnostics_reports_openai_responses_with_incompatible_base_url() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai-responses",
            protocol="openai_responses",
            base_url="https://gateway.example.test/v1",
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-5.5",
            web_search_adapter="openai_responses",
        )
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert any(
        "protocol=openai_responses" in issue
        and "sylliptor profile convert openai-responses --to compatibility" in issue
        for issue in diagnostics.issues
    )


def test_provider_diagnostics_reports_empty_model() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai-responses",
            protocol="openai_responses",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            default_model="",
            web_search_adapter="openai_responses",
        )
    )
    cfg.model = ""

    diagnostics = build_provider_diagnostics(cfg)

    assert diagnostics.model == ""
    assert any("Model is empty" in issue for issue in diagnostics.issues)


def test_provider_diagnostics_reports_model_family_mismatch() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="anthropic",
            protocol="anthropic_messages",
            base_url="https://api.anthropic.com/v1",
            api_key_env="ANTHROPIC_API_KEY",
            default_model="gemini-2.5-flash",
            web_search_adapter="anthropic_messages",
        )
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert any("looks like a gemini model" in issue for issue in diagnostics.issues)
    assert any("profile/protocol is for anthropic" in issue for issue in diagnostics.issues)


def test_provider_diagnostics_reports_stale_model_alias() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="gemini",
            protocol="gemini_generate_content",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GEMINI_API_KEY",
            default_model="gemini-2.5-flash",
            web_search_adapter="gemini_grounding",
        )
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert any("known renamed/deprecated alias" in issue for issue in diagnostics.issues)
    assert any("gemini-3.5-flash" in issue for issue in diagnostics.issues)


def test_provider_diagnostics_reports_native_feature_model_risk() -> None:
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="gemini",
            protocol="gemini_generate_content",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GEMINI_API_KEY",
            default_model="gemini-2.5-flash-live-preview-native-audio",
            web_search_adapter="gemini_grounding",
        ),
        stream=True,
        web_search_mode="native",
    )

    diagnostics = build_provider_diagnostics(cfg)

    assert any("Native web_search is enabled" in issue for issue in diagnostics.issues)
    assert any("stream=true is enabled" in issue for issue in diagnostics.issues)


class _FakeClient:
    def __init__(self, response: LLMResponse | Exception) -> None:
        self._response = response

    def chat(self, **_kwargs):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _CapturingClient(_FakeClient):
    supports_tool_calling = True

    def __init__(self, response: LLMResponse | Exception) -> None:
        super().__init__(response)
        self.chat_kwargs: dict[str, object] = {}

    def chat(self, **kwargs):
        self.chat_kwargs = dict(kwargs)
        return super().chat(**kwargs)


def test_live_provider_validation_uses_mocked_client_without_printing_secret(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret-value")
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai-responses",
            protocol="openai_responses",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-5.4-mini",
            web_search_adapter="openai_responses",
        )
    )
    captured: dict[str, object] = {}
    client = _CapturingClient(LLMResponse(content="ok", tool_calls=[], raw={}))

    def factory(**kwargs):
        captured.update(kwargs)
        return client

    validation = validate_active_provider_live(cfg, client_factory=factory)

    assert validation.ok is True
    assert validation.model == "gpt-5.4-mini"
    assert validation.status == "passed"
    assert "sk-openai-secret-value" not in validation.message
    assert captured["api_key"] == "sk-openai-secret-value"
    assert captured["timeout_s"] == 15.0
    assert client.chat_kwargs["tools"]


def test_live_provider_validation_fails_when_tool_probe_is_rejected(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    monkeypatch.setenv("SYLLIPTOR_API_KEY", "secret-value")
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="tool-probe",
            base_url="https://gateway.example.test/v1",
            api_key_env="SYLLIPTOR_API_KEY",
            default_model="test-model",
            web_search_adapter="auto",
        )
    )
    response = LLMResponse(
        content="ok",
        tool_calls=[],
        raw={},
        provider_metadata={
            "transport": {
                "tools_omitted": True,
                "tools_omit_reason": "provider_rejected_tool_calling",
                "tools_retry_used": True,
            }
        },
    )

    validation = validate_active_provider_live(
        cfg,
        client_factory=lambda **_kwargs: _CapturingClient(response),
    )

    assert validation.status == "failed"
    assert "rejected tool calling" in validation.message
    assert "secret-value" not in validation.message


def test_live_provider_validation_classifies_model_availability_errors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret-key")
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="gemini",
            protocol="gemini_generate_content",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GEMINI_API_KEY",
            default_model="gemini-bad",
            web_search_adapter="gemini_grounding",
        )
    )

    validation = validate_active_provider_live(
        cfg,
        client_factory=lambda **_kwargs: _FakeClient(RuntimeError("404 model not found")),
    )

    assert validation.status == "failed"
    assert "could not use model 'gemini-bad'" in validation.message
    assert "gemini-secret-key" not in validation.message


def test_doctor_providers_cli_uses_redacted_diagnostics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="gemini",
            protocol="gemini_generate_content",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key_env="GEMINI_API_KEY",
            default_model="gemini-3-flash-preview",
            web_search_adapter="gemini_grounding",
        ),
        stream=True,
        web_search_mode="native",
    )
    save_config(cfg)

    result = CliRunner().invoke(
        sylliptor_app,
        ["doctor", "providers"],
        env={**_env(tmp_path), "GEMINI_API_KEY": "gemini-secret-key"},
    )

    assert result.exit_code == 0
    assert "sylliptor doctor providers" in result.output
    assert "gemini_generate_content" in result.output
    assert "generativelanguage.googleapis.com" in result.output
    assert "gemini-secret-key" not in result.output
    assert "env:GEMINI_API_KEY (redacted)" in result.output
    assert "stream_enabled" in result.output
    assert "streaming_supported" in result.output
    assert "stream=true" not in result.output


def test_doctor_providers_live_requires_confirmation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai-responses",
            protocol="openai_responses",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-5.4-mini",
            web_search_adapter="openai_responses",
        )
    )
    save_config(cfg)
    called = False

    def fake_validate(*_args, **_kwargs):
        nonlocal called
        called = True
        return ProviderLiveValidation(
            profile_name="openai-responses",
            provider_key="openai",
            protocol="openai_responses",
            model="gpt-5.4-mini",
            status="passed",
            message="ok",
        )

    monkeypatch.setattr(root_mod, "validate_active_provider_live", fake_validate)

    result = CliRunner().invoke(
        sylliptor_app,
        ["doctor", "providers", "--live"],
        input="n\n",
        env={**_env(tmp_path), "OPENAI_API_KEY": "sk-openai-secret-value"},
    )

    assert result.exit_code == 0
    assert "may incur provider cost" in result.output
    assert "cancelled" in result.output
    assert "sk-openai-secret-value" not in result.output
    assert called is False


def test_doctor_providers_live_yes_uses_redacted_validation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path))
    cfg = _cfg_with_profile(
        ProfileSpec(
            name="openai-responses",
            protocol="openai_responses",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            default_model="gpt-5.4-mini",
            web_search_adapter="openai_responses",
        )
    )
    save_config(cfg)

    def fake_validate(*_args, **_kwargs):
        return ProviderLiveValidation(
            profile_name="openai-responses",
            provider_key="openai",
            protocol="openai_responses",
            model="gpt-5.4-mini",
            status="passed",
            message="Minimal text request completed successfully.",
        )

    monkeypatch.setattr(root_mod, "validate_active_provider_live", fake_validate)

    result = CliRunner().invoke(
        sylliptor_app,
        ["doctor", "providers", "--live", "--yes"],
        env={**_env(tmp_path), "OPENAI_API_KEY": "sk-openai-secret-value"},
    )

    assert result.exit_code == 0
    assert "sylliptor doctor providers --live" in result.output
    assert "gpt-5.4-mini" in result.output
    assert "passed" in result.output
    assert "sk-openai-secret-value" not in result.output
