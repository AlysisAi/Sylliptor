from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from click import Abort
from rich.console import Console
from typer.testing import CliRunner

from sylliptor_agent_cli import cli as cli_mod
from sylliptor_agent_cli.cli import app as sylliptor_app
from sylliptor_agent_cli.cli_impl import config_menu as config_menu_mod
from sylliptor_agent_cli.cli_impl import setup_wizard as setup_wizard_mod
from sylliptor_agent_cli.config import (
    AppConfig,
    config_path,
    credentials_path,
    load_config,
    load_persisted_api_key,
    load_persisted_profile_keys,
    save_config,
    save_persisted_api_key,
)
from sylliptor_agent_cli.profile_presets import ProfilePreset, make_profile_from_preset
from sylliptor_agent_cli.profiles import ProfileSpec
from sylliptor_agent_cli.sandbox_doctor import (
    SandboxCheck,
    SandboxDiagnostic,
    SandboxImagePullResult,
    SandboxPullResult,
)

_ORIGINAL_VALIDATE_API_KEY = setup_wizard_mod._validate_api_key


class PromptScript:
    def __init__(self, answers: list[str | BaseException]) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, text: str, *args: Any, **kwargs: Any) -> str:
        del args
        self.calls.append((text, kwargs))
        if not self.answers:
            raise AssertionError(f"Unexpected prompt: {text}")
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class PickerScript:
    def __init__(self, answers: list[str | None], *, auto_router_inherit: bool = True) -> None:
        self.answers = list(answers)
        self.auto_router_inherit = auto_router_inherit
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> str | None:
        self.calls.append(kwargs)
        if kwargs.get("title") == "Router Model" and self.auto_router_inherit:
            if not self.answers:
                return setup_wizard_mod._INHERIT_DEFAULT_MODEL_VALUE
            next_answer = self.answers[0]
            if next_answer not in {
                None,
                setup_wizard_mod._CUSTOM_MODEL_VALUE,
                setup_wizard_mod._INHERIT_DEFAULT_MODEL_VALUE,
            }:
                return setup_wizard_mod._INHERIT_DEFAULT_MODEL_VALUE
        if not self.answers:
            raise AssertionError(f"Unexpected picker: {kwargs}")
        return self.answers.pop(0)


def _patch_console(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=120)
    monkeypatch.setattr(setup_wizard_mod, "_resolve_console", lambda: console)
    return output


def _config_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path / "config"))
    monkeypatch.setenv("SYLLIPTOR_DATA_DIR", os.fspath(tmp_path / "data"))
    monkeypatch.delenv("SYLLIPTOR_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def _sandbox_diagnostic(
    *,
    ready: bool,
    can_pull: bool = False,
    next_steps: tuple[str, ...] = ("Sandbox is ready.",),
) -> SandboxDiagnostic:
    return SandboxDiagnostic(
        ready=ready,
        status="ready" if ready else "not_ready",
        configured_mode="strict",
        configured_backend="auto",
        selected_backend="docker",
        docker_image="ghcr.io/alysisai/sylliptor-sandbox:dev",
        server_image="ghcr.io/alysisai/sylliptor-sandbox:server",
        checks=(
            SandboxCheck("Docker CLI", "ok", "/usr/bin/docker"),
            SandboxCheck("Docker daemon", "ok", "running"),
            SandboxCheck(
                "sandbox image",
                "ok" if ready else "missing",
                "ghcr.io/alysisai/sylliptor-sandbox:dev",
            ),
        ),
        next_steps=next_steps,
        can_pull=can_pull,
    )


@pytest.fixture(autouse=True)
def _setup_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        setup_wizard_mod,
        "diagnose_sandbox",
        lambda _cfg, *, include_smoke=False: _sandbox_diagnostic(ready=True),
    )
    monkeypatch.setattr(
        setup_wizard_mod,
        "pull_sandbox_images",
        Mock(side_effect=AssertionError("pull should not be called")),
    )
    monkeypatch.setattr(
        setup_wizard_mod,
        "_validate_api_key",
        lambda **_kwargs: setup_wizard_mod._ApiKeyValidationResult(status="validated"),
    )


def _run_basic_wizard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    provider: str = "openai",
    model: str = "gpt-5",
    api_key: str = "sk-test-1234",
    workspace: Path | None = None,
) -> tuple[bool, str, PickerScript, PromptScript]:
    _config_env(tmp_path, monkeypatch)
    output = _patch_console(monkeypatch)
    selected_workspace = workspace or tmp_path
    picker = PickerScript([provider, model])
    prompt = PromptScript(["", api_key, os.fspath(selected_workspace)])
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(setup_wizard_mod.typer, "prompt", prompt)
    return setup_wizard_mod.run_setup_wizard(), output.getvalue(), picker, prompt


def test_setup_wizard_full_flow_persists_everything(monkeypatch, tmp_path: Path) -> None:
    result, _output, _picker, _prompt = _run_basic_wizard(monkeypatch, tmp_path)

    assert result is True
    cfg = load_config()
    assert load_persisted_api_key() is None
    assert load_persisted_profile_keys()["openai"] == "sk-test-1234"
    assert cfg.model == "gpt-5"
    assert cfg.extra_fields["active_profile"] == "openai"
    assert cfg.extra_fields["default_workspace_path"] == os.fspath(tmp_path.resolve())
    assert "router" not in cfg.extra_fields.get("role_models", {})


def test_setup_wizard_router_model_inherits_and_clears_existing_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config_env(tmp_path, monkeypatch)
    cfg = AppConfig()
    cfg.extra_fields = {"role_models": {"coding": "coding-model", "router": "old-router-model"}}
    save_config(cfg)

    result, output, picker, _prompt = _run_basic_wizard(monkeypatch, tmp_path)

    assert result is True
    saved_cfg = load_config()
    assert saved_cfg.extra_fields["role_models"] == {"coding": "coding-model"}
    assert "Router model:" in output
    assert "inherits default" in output
    router_rows = picker.calls[2]["rows"]
    assert router_rows[0][0] == setup_wizard_mod._INHERIT_DEFAULT_MODEL_VALUE


def test_setup_wizard_persists_explicit_router_model_and_validates_it(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config_env(tmp_path, monkeypatch)
    _patch_console(monkeypatch)
    picker = PickerScript(
        ["openai", "gpt-5", "gpt-5.4-mini"],
        auto_router_inherit=False,
    )
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(
        setup_wizard_mod.typer,
        "prompt",
        PromptScript(["", "sk-test-1234", os.fspath(tmp_path)]),
    )
    validated_models: list[str | None] = []

    def validate(*, profile, api_key, model=None, **_kwargs):
        del profile, api_key
        validated_models.append(model)
        return setup_wizard_mod._ApiKeyValidationResult(status="validated")

    monkeypatch.setattr(setup_wizard_mod, "_validate_api_key", validate)

    assert setup_wizard_mod.run_setup_wizard() is True
    assert load_config().extra_fields["role_models"]["router"] == "gpt-5.4-mini"
    assert validated_models == [None, "gpt-5", "gpt-5.4-mini"]


def test_setup_wizard_abort_on_api_key_writes_nothing(monkeypatch, tmp_path: Path) -> None:
    _config_env(tmp_path, monkeypatch)
    output = _patch_console(monkeypatch)
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", PickerScript(["openai"]))
    monkeypatch.setattr(setup_wizard_mod.typer, "prompt", PromptScript(["", Abort(), "y"]))

    assert setup_wizard_mod.run_setup_wizard() is False
    assert "Setup cancelled. No changes saved." in output.getvalue()
    assert not config_path().exists()
    assert not credentials_path().exists()


def test_setup_wizard_abort_on_workspace_writes_nothing(monkeypatch, tmp_path: Path) -> None:
    _config_env(tmp_path, monkeypatch)
    output = _patch_console(monkeypatch)
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", PickerScript(["openai", "gpt-5"]))
    monkeypatch.setattr(
        setup_wizard_mod.typer,
        "prompt",
        PromptScript(["", "sk-test-1234", Abort(), "y"]),
    )

    assert setup_wizard_mod.run_setup_wizard() is False
    assert "Setup cancelled. No changes saved." in output.getvalue()
    assert not config_path().exists()
    assert not credentials_path().exists()


def test_setup_wizard_success_uses_one_config_save_and_one_profile_key_save(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config_env(tmp_path, monkeypatch)
    _patch_console(monkeypatch)
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", PickerScript(["openai", "gpt-5"]))
    monkeypatch.setattr(
        setup_wizard_mod.typer, "prompt", PromptScript(["", "sk-test-1234", os.fspath(tmp_path)])
    )
    save_calls: list[AppConfig] = []
    key_calls: list[tuple[str, str]] = []

    def fake_save_config(cfg: AppConfig) -> None:
        save_calls.append(cfg)

    def fake_save_profile_key(profile_name: str, api_key: str) -> None:
        key_calls.append((profile_name, api_key))

    monkeypatch.setattr(setup_wizard_mod, "save_config", fake_save_config)
    monkeypatch.setattr(setup_wizard_mod, "save_persisted_profile_key", fake_save_profile_key)

    assert setup_wizard_mod.run_setup_wizard() is True
    assert len(save_calls) == 1
    assert key_calls == [("openai", "sk-test-1234")]
    assert save_calls[0].model == "gpt-5"
    assert save_calls[0].extra_fields["default_workspace_path"] == os.fspath(tmp_path.resolve())


def test_setup_wizard_rollback_restores_config_when_key_save_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config_env(tmp_path, monkeypatch)
    old_cfg = AppConfig(model="existing")
    save_config(old_cfg)
    before = config_path().read_bytes()
    _patch_console(monkeypatch)
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", PickerScript(["openai", "gpt-5"]))
    monkeypatch.setattr(
        setup_wizard_mod.typer, "prompt", PromptScript(["", "sk-test-1234", os.fspath(tmp_path)])
    )

    def fail_key_save(_profile_name: str, _api_key: str) -> None:
        raise PermissionError("credential store locked")

    monkeypatch.setattr(setup_wizard_mod, "save_persisted_profile_key", fail_key_save)

    assert setup_wizard_mod.run_setup_wizard() is False
    assert config_path().read_bytes() == before
    assert load_config().model == "existing"
    assert not credentials_path().exists()


def test_setup_wizard_creates_anthropic_profile_with_chosen_model(
    monkeypatch,
    tmp_path: Path,
) -> None:
    result, _output, _picker, _prompt = _run_basic_wizard(
        monkeypatch,
        tmp_path,
        provider="anthropic",
        model="claude-sonnet-4-6",
        api_key="sk-ant-test",
    )

    assert result is True
    cfg = load_config()
    profile = cfg.extra_fields["profiles"]["anthropic"]
    assert cfg.extra_fields["active_profile"] == "anthropic"
    assert cfg.model == "claude-sonnet-4-6"
    assert profile["protocol"] == "anthropic_messages"
    assert profile["base_url"] == "https://api.anthropic.com/v1"
    assert profile["default_model"] == "claude-sonnet-4-6"
    assert profile["web_search_adapter"] == "anthropic_messages"
    assert profile["extra_headers"] == {}
    assert load_persisted_profile_keys()["anthropic"] == "sk-ant-test"


def test_provider_fallback_does_not_treat_api_key_like_input_as_openai(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config_env(tmp_path, monkeypatch)
    output = _patch_console(monkeypatch)
    rows = [
        (preset.key, preset.label, setup_wizard_mod._preset_description(preset))
        for preset in setup_wizard_mod._setup_presets()
    ]
    monkeypatch.setattr(
        setup_wizard_mod.typer,
        "prompt",
        PromptScript(["sk-this-is-a-key-not-a-provider", "2"]),
    )

    selected = setup_wizard_mod._run_wizard_picker_fallback(
        console=setup_wizard_mod._resolve_console(),
        title="Provider Profile",
        subtitle="Pick a provider.",
        rows=rows,
        current_value=rows[0][0],
        footer_hint="fallback",
        invalid_hint="Pick a provider first; you'll enter the key in the next step.",
    )

    assert selected == rows[1][0]
    assert selected != "openai"
    assert "Pick a provider first; you'll enter the key in the next step." in output.getvalue()


def test_setup_profile_picker_separates_compatibility_and_native_presets() -> None:
    presets = setup_wizard_mod._setup_presets()
    keys = [preset.key for preset in presets]
    advanced = setup_wizard_mod._advanced_setup_presets()
    advanced_keys = [preset.key for preset in advanced]

    assert keys == ["openai-responses", "anthropic", "gemini"]
    assert all(preset.protocol != "gemini_interactions" for preset in presets)
    assert all(preset.protocol != "gemini_interactions" for preset in advanced)
    assert "anthropic-compat" not in keys
    assert "gemini-compat" not in keys
    assert "anthropic-compat" in advanced_keys
    assert "gemini-compat" in advanced_keys
    assert all(
        "Compatibility protocol" in setup_wizard_mod._preset_description(preset)
        for preset in advanced
        if preset.key in {"anthropic-compat", "gemini-compat"}
    )
    assert all(
        "Native first-party protocol" in setup_wizard_mod._preset_description(preset)
        for preset in presets
        if preset.key in {"openai-responses", "anthropic", "gemini"}
    )


def test_setup_wizard_advanced_flow_can_choose_anthropic_compat(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config_env(tmp_path, monkeypatch)
    _patch_console(monkeypatch)
    picker = PickerScript(
        [
            setup_wizard_mod._ADVANCED_PROVIDER_PRESETS_VALUE,
            "anthropic-compat",
            "claude-sonnet-4-6",
        ]
    )
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(
        setup_wizard_mod.typer,
        "prompt",
        PromptScript(["", "sk-ant-test", os.fspath(tmp_path)]),
    )

    assert setup_wizard_mod.run_setup_wizard() is True
    profile = load_config().extra_fields["profiles"]["anthropic-compat"]
    assert profile["protocol"] == "openai_compat"
    assert profile["base_url"] == "https://api.anthropic.com/v1/"


def test_setup_wizard_surfaces_provider_diagnostic_warnings(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config_env(tmp_path, monkeypatch)
    cfg = AppConfig(web_search_mode="external")
    save_config(cfg)
    output = _patch_console(monkeypatch)
    picker = PickerScript(["openai-responses", "gpt-5.5"])
    prompt = PromptScript(["", "sk-test-1234", os.fspath(tmp_path)])
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(setup_wizard_mod.typer, "prompt", prompt)

    assert setup_wizard_mod.run_setup_wizard() is True

    rendered = output.getvalue()
    assert "Provider diagnostic:" in rendered
    assert "web_search_mode=external is incompatible" in rendered
    assert "web_search_adapter=openai_responses" in rendered


def test_model_picker_rows_include_preset_suggestions(monkeypatch, tmp_path: Path) -> None:
    _config_env(tmp_path, monkeypatch)
    _patch_console(monkeypatch)
    picker = PickerScript(["openai", "gpt-5.5"])
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(
        setup_wizard_mod.typer, "prompt", PromptScript(["", "sk-test-1234", os.fspath(tmp_path)])
    )

    assert setup_wizard_mod.run_setup_wizard() is True
    model_rows = picker.calls[1]["rows"]
    assert (
        "gpt-5.5",
        "gpt-5.5",
        "default - flagship model for complex coding and reasoning",
    ) in model_rows
    assert model_rows[-1][1] == "Type a custom model name"


def test_deepseek_model_picker_uses_v4_models(monkeypatch, tmp_path: Path) -> None:
    _config_env(tmp_path, monkeypatch)
    _patch_console(monkeypatch)
    picker = PickerScript(["deepseek", "deepseek-v4-flash"])
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(
        setup_wizard_mod.typer, "prompt", PromptScript(["", "sk-test-1234", os.fspath(tmp_path)])
    )

    assert setup_wizard_mod.run_setup_wizard() is True
    model_values = [value for value, _label, _description in picker.calls[1]["rows"]]
    assert "deepseek-v4-pro" in model_values
    assert "deepseek-v4-flash" in model_values
    assert "deepseek-coder" not in model_values


def test_gemini_model_picker_uses_stable_models(monkeypatch, tmp_path: Path) -> None:
    _config_env(tmp_path, monkeypatch)
    _patch_console(monkeypatch)
    picker = PickerScript(["gemini", "gemini-3.5-flash"])
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(
        setup_wizard_mod.typer, "prompt", PromptScript(["", "sk-test-1234", os.fspath(tmp_path)])
    )

    assert setup_wizard_mod.run_setup_wizard() is True
    model_values = [value for value, _label, _description in picker.calls[1]["rows"]]
    assert model_values[0] == "gemini-3.5-flash"
    assert "gemini-3.1-flash-lite" in model_values
    assert "gemini-2.5-pro" in model_values
    assert "gemini-2.5-flash" in model_values
    assert "gemini-3.1-preview" not in model_values
    assert "gemini-3-pro-preview" not in model_values
    assert "gemini-3.1-flash-lite-preview" not in model_values


def test_gemini_custom_stale_preview_alias_canonicalizes_before_validation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config_env(tmp_path, monkeypatch)
    _patch_console(monkeypatch)
    picker = PickerScript(["gemini", setup_wizard_mod._CUSTOM_MODEL_VALUE])
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(
        setup_wizard_mod.typer,
        "prompt",
        PromptScript(["", "sk-test-1234", "gemini-3.1-preview", os.fspath(tmp_path)]),
    )
    validated_models: list[str | None] = []

    def validate(*, profile, api_key, model=None, **_kwargs):
        del profile, api_key
        validated_models.append(model)
        return setup_wizard_mod._ApiKeyValidationResult(status="validated")

    monkeypatch.setattr(setup_wizard_mod, "_validate_api_key", validate)

    assert setup_wizard_mod.run_setup_wizard() is True
    assert validated_models == [None, "gemini-3.1-pro-preview"]
    assert load_config().model == "gemini-3.1-pro-preview"


def test_model_picker_without_suggestions_only_offers_custom() -> None:
    preset = ProfilePreset(
        key="empty",
        label="Empty",
        protocol="openai_compat",
        base_url="http://localhost:9999/v1",
        api_key_env=None,
        suggested_models=(),
    )
    profile = make_profile_from_preset(preset)
    rows = setup_wizard_mod._model_picker_rows(
        setup_wizard_mod._ProfileStepResult(profile=profile, label=preset.label, preset=preset)
    )

    assert rows == [
        (
            setup_wizard_mod._CUSTOM_MODEL_VALUE,
            "Type a custom model name",
            "Use any model supported by this provider",
        )
    ]
    assert setup_wizard_mod._FALLBACK_VALIDATION_MODEL not in {row[0] for row in rows}


def _validate_with_transport_response(
    response: httpx.Response,
    *,
    base_url: str = "https://api.example.test/v1",
) -> tuple[setup_wizard_mod._ApiKeyValidationResult, dict[str, Any]]:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return response

    result = _ORIGINAL_VALIDATE_API_KEY(
        profile=ProfileSpec(
            name="custom",
            base_url=base_url,
            extra_headers={"x-provider": "example"},
            default_model="example-model",
        ),
        api_key="sk-test",
        model="example-model",
        transport=httpx.MockTransport(handler),
    )
    return result, captured


def test_api_key_validation_posts_chat_completion_ping() -> None:
    result, captured = _validate_with_transport_response(
        httpx.Response(200, json={"choices": [{"message": {"content": "p"}}]})
    )

    assert result.status == "validated"
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.example.test/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer sk-test"
    assert captured["headers"]["x-provider"] == "example"
    assert captured["body"] == {
        "model": "example-model",
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.0,
    }


def test_api_key_validation_normalizes_trailing_slash_base_url() -> None:
    result, captured = _validate_with_transport_response(
        httpx.Response(200, json={"choices": [{"message": {"content": "p"}}]}),
        base_url="https://api.example.test/v1/",
    )

    assert result.status == "validated"
    assert captured["url"] == "https://api.example.test/v1/chat/completions"


def test_api_key_validation_401_is_failed() -> None:
    result, _captured = _validate_with_transport_response(httpx.Response(401, text="bad key"))

    assert result.status == "failed"
    assert result.message == "Provider rejected the API key (HTTP 401)."


def test_api_key_validation_400_model_not_found_is_distinct() -> None:
    result, _captured = _validate_with_transport_response(
        httpx.Response(400, json={"error": {"message": "model not_found"}})
    )

    assert result.status == "model_not_found"
    assert "Model 'example-model' not found" in result.message


def test_api_key_validation_400_max_tokens_unsupported_is_not_model_not_found() -> None:
    result, _captured = _validate_with_transport_response(
        httpx.Response(
            400,
            json={
                "error": {
                    "message": (
                        "Unsupported parameter: 'max_tokens' is not supported with this model. "
                        "Use 'max_completion_tokens' instead."
                    ),
                    "type": "invalid_request_error",
                    "param": "max_tokens",
                    "code": "unsupported_parameter",
                }
            },
        )
    )

    assert result.status == "inconclusive"
    assert "max_tokens" in result.message
    assert "not found" not in result.message


def test_api_key_validation_retries_when_gemini_rejects_temperature() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        captured.append(body)
        if "temperature" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "temperature cannot be set for this model.",
                        "type": "invalid_request_error",
                        "param": "temperature",
                        "code": "unsupported_parameter",
                    }
                },
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "p"}}]})

    result = _ORIGINAL_VALIDATE_API_KEY(
        profile=ProfileSpec(
            name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="gemini-2.5-flash",
        ),
        api_key="sk-test",
        model="gemini-2.5-flash",
        transport=httpx.MockTransport(handler),
    )

    assert result.status == "validated"
    assert len(captured) == 3
    assert captured[0]["temperature"] == 0.0
    assert captured[1]["temperature"] == 1.0
    assert "temperature" not in captured[2]


def test_gemini_api_key_validation_uses_fast_validation_model_before_model_choice() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"choices": [{"message": {"content": "p"}}]})

    result = _ORIGINAL_VALIDATE_API_KEY(
        profile=ProfileSpec(
            name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="gemini-2.5-flash",
        ),
        api_key="sk-test",
        suggested_models=(
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ),
        validation_model="gemini-2.5-flash",
        transport=httpx.MockTransport(handler),
    )

    assert result.status == "validated"
    assert result.message == (
        "Validated API key using 'gemini-2.5-flash'. The selected model will be verified next."
    )
    assert "gemini-2.5-flash" in result.message
    assert "fallback" not in result.message.casefold()
    assert captured["body"]["model"] == "gemini-2.5-flash"
    assert captured["body"]["reasoning_effort"] == "low"


def test_gemini_selected_model_validation_uses_low_reasoning_effort() -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json={"choices": [{"message": {"content": "p"}}]})

    result = _ORIGINAL_VALIDATE_API_KEY(
        profile=ProfileSpec(
            name="gemini",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="gemini-2.5-flash",
        ),
        api_key="sk-test",
        model="gemini-2.5-flash",
        transport=httpx.MockTransport(handler),
    )

    assert result.status == "validated"
    assert result.message == ""
    assert captured["body"]["model"] == "gemini-2.5-flash"
    assert captured["body"]["reasoning_effort"] == "low"


def test_gemini_validation_uses_extended_timeout() -> None:
    gemini_profile = ProfileSpec(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    custom_profile = ProfileSpec(name="custom", base_url="https://api.example.test/v1")

    assert (
        setup_wizard_mod._validation_timeout_s(gemini_profile)
        > setup_wizard_mod._VALIDATION_TIMEOUT_S
    )
    assert (
        setup_wizard_mod._validation_timeout_s(custom_profile)
        == setup_wizard_mod._VALIDATION_TIMEOUT_S
    )


def test_api_key_validation_404_is_endpoint_inconclusive_not_failed() -> None:
    result, _captured = _validate_with_transport_response(httpx.Response(404, text="not found"))

    assert result.status == "inconclusive"
    assert "Provider endpoint not found at https://api.example.test/v1" in result.message


def test_api_key_validation_429_is_rate_limited_inconclusive() -> None:
    result, _captured = _validate_with_transport_response(httpx.Response(429, text="slow down"))

    assert result.status == "inconclusive"
    assert "rate-limited" in result.message


def test_api_key_validation_timeout_is_inconclusive() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    result = _ORIGINAL_VALIDATE_API_KEY(
        profile=ProfileSpec(
            name="custom",
            base_url="https://api.example.test/v1",
            default_model="example-model",
        ),
        api_key="sk-test",
        model="example-model",
        transport=httpx.MockTransport(handler),
    )

    assert result.status == "inconclusive"
    assert "validation request timed out" in result.message


@pytest.mark.parametrize(
    ("validation_status", "validation_message", "expected"),
    [
        ("validated", "", "stored, validated"),
        (
            "inconclusive",
            "Provider endpoint not found at https://api.example.test/v1.",
            "stored, validation inconclusive",
        ),
        ("failed", "Provider rejected the API key (HTTP 401).", "stored, but provider rejected it"),
        ("skipped", "", "stored, validation skipped"),
        ("model_not_found", "Model 'bad-model' not found.", "stored, model validation failed"),
    ],
)
def test_setup_summary_renders_api_key_validation_status(
    monkeypatch: pytest.MonkeyPatch,
    validation_status: setup_wizard_mod._ValidationStatus,
    validation_message: str,
    expected: str,
) -> None:
    output = _patch_console(monkeypatch)

    setup_wizard_mod._print_setup_complete(
        console=setup_wizard_mod._resolve_console(),
        profile_result=setup_wizard_mod._ProfileStepResult(
            profile=ProfileSpec(name="custom", base_url="https://api.example.test/v1"),
            label="Custom",
            preset=None,
        ),
        api_key_result=setup_wizard_mod._ApiKeyStepResult(
            api_key="sk-test",
            validation_status=validation_status,
            validation_message=validation_message,
        ),
        model_result=setup_wizard_mod._ModelStepResult(model="example-model"),
        router_model_result=setup_wizard_mod._RouterModelStepResult(),
        workspace_result=setup_wizard_mod._WorkspaceStepResult(workspace="C:\\repo"),
        sandbox_result=setup_wizard_mod._SandboxStepResult(ready=True, status="docker"),
    )

    assert expected in output.getvalue()


def test_model_picker_reprompts_on_model_not_found(monkeypatch, tmp_path: Path) -> None:
    _config_env(tmp_path, monkeypatch)
    output = _patch_console(monkeypatch)
    picker = PickerScript(["openai", "gpt-5.5", "gpt-5.4-mini"])
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(
        setup_wizard_mod.typer, "prompt", PromptScript(["", "sk-test-1234", os.fspath(tmp_path)])
    )
    validated_models: list[str | None] = []

    def validate(*, profile, api_key, model=None, **_kwargs):
        del profile, api_key
        validated_models.append(model)
        if model == "gpt-5.5":
            return setup_wizard_mod._ApiKeyValidationResult(
                status="model_not_found",
                message="Model 'gpt-5.5' not found at this provider. Pick a different model in the next step.",
            )
        return setup_wizard_mod._ApiKeyValidationResult(status="validated")

    monkeypatch.setattr(setup_wizard_mod, "_validate_api_key", validate)

    assert setup_wizard_mod.run_setup_wizard() is True
    assert validated_models == [None, "gpt-5.5", "gpt-5.4-mini"]
    assert load_config().model == "gpt-5.4-mini"
    assert "Model 'gpt-5.5' not found" in output.getvalue()


def test_custom_model_not_found_can_be_used_after_explicit_warning(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config_env(tmp_path, monkeypatch)
    output = _patch_console(monkeypatch)
    picker = PickerScript(["openai", setup_wizard_mod._CUSTOM_MODEL_VALUE])
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(
        setup_wizard_mod.typer,
        "prompt",
        PromptScript(["", "sk-test-1234", "provider-private-model", "y", os.fspath(tmp_path)]),
    )

    def validate(*, profile, api_key, model=None, **_kwargs):
        del profile, api_key
        if model == "provider-private-model":
            return setup_wizard_mod._ApiKeyValidationResult(
                status="model_not_found",
                message=(
                    "Model 'provider-private-model' not found at this provider. "
                    "Pick a different model in the next step."
                ),
            )
        return setup_wizard_mod._ApiKeyValidationResult(status="validated")

    monkeypatch.setattr(setup_wizard_mod, "_validate_api_key", validate)

    assert setup_wizard_mod.run_setup_wizard() is True
    cfg = load_config()
    assert cfg.model == "provider-private-model"
    rendered = output.getvalue()
    assert "Model 'provider-private-model' not found" in rendered
    assert "stored, validation inconclusive" in rendered


def test_api_key_validation_failure_reprompts_then_accepts_last_key(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config_env(tmp_path, monkeypatch)
    _patch_console(monkeypatch)
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", PickerScript(["openai", "gpt-5"]))
    monkeypatch.setattr(
        setup_wizard_mod.typer,
        "prompt",
        PromptScript(["", "bad-1", "y", "bad-2", "y", "bad-3", os.fspath(tmp_path)]),
    )
    validations: list[str] = []

    def fail_validation(*, profile, api_key, **_kwargs):
        del profile
        validations.append(api_key)
        return setup_wizard_mod._ApiKeyValidationResult(
            status="failed",
            message="Provider rejected the key (HTTP 401).",
        )

    monkeypatch.setattr(setup_wizard_mod, "_validate_api_key", fail_validation)

    assert setup_wizard_mod.run_setup_wizard() is True
    assert validations == ["bad-1", "bad-2", "bad-3"]
    assert load_persisted_profile_keys()["openai"] == "bad-3"


def test_api_key_validation_network_failure_continues(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config_env(tmp_path, monkeypatch)
    output = _patch_console(monkeypatch)
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", PickerScript(["openai", "gpt-5"]))
    monkeypatch.setattr(
        setup_wizard_mod.typer, "prompt", PromptScript(["", "sk-test-1234", os.fspath(tmp_path)])
    )
    monkeypatch.setattr(
        setup_wizard_mod,
        "_validate_api_key",
        lambda **_kwargs: setup_wizard_mod._ApiKeyValidationResult(
            status="inconclusive",
            message="Could not reach https://api.openai.com/v1: timed out",
        ),
    )

    assert setup_wizard_mod.run_setup_wizard() is True
    assert "Could not reach https://api.openai.com/v1" in output.getvalue()
    assert load_persisted_profile_keys()["openai"] == "sk-test-1234"


def test_workspace_default_prefers_projects(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    projects = home / "projects"
    projects.mkdir(parents=True)
    monkeypatch.setattr(setup_wizard_mod.Path, "home", classmethod(lambda cls: home))

    assert setup_wizard_mod._suggest_workspace_default() == projects


def test_workspace_default_falls_back_to_home_and_not_cwd(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "home"
    downloads = tmp_path / "Downloads"
    home.mkdir()
    downloads.mkdir()
    monkeypatch.chdir(downloads)
    monkeypatch.setattr(setup_wizard_mod.Path, "home", classmethod(lambda cls: home))

    assert setup_wizard_mod._suggest_workspace_default() == home
    assert setup_wizard_mod._suggest_workspace_default() != downloads


def test_setup_wizard_invalid_workspace_reprompts(monkeypatch, tmp_path: Path) -> None:
    _config_env(tmp_path, monkeypatch)
    _patch_console(monkeypatch)
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", PickerScript(["openai", "gpt-5"]))
    monkeypatch.setattr(
        setup_wizard_mod.typer,
        "prompt",
        PromptScript(["", "sk-test-1234", "/does/not/exist", os.fspath(tmp_path)]),
    )

    assert setup_wizard_mod.run_setup_wizard() is True
    assert load_config().extra_fields["default_workspace_path"] == os.fspath(tmp_path.resolve())


def test_setup_wizard_skips_sandbox_pull_when_ready(monkeypatch, tmp_path: Path) -> None:
    pull = Mock(side_effect=AssertionError("pull should not be called"))
    monkeypatch.setattr(setup_wizard_mod, "pull_sandbox_images", pull)

    result, output, _picker, _prompt = _run_basic_wizard(monkeypatch, tmp_path)

    assert result is True
    pull.assert_not_called()
    assert "Sandbox ready" in output


def test_setup_wizard_offers_pull_when_image_missing(monkeypatch, tmp_path: Path) -> None:
    diagnose_calls: list[bool] = []

    def fake_diagnose(_cfg: AppConfig, *, include_smoke: bool = False) -> SandboxDiagnostic:
        diagnose_calls.append(include_smoke)
        if include_smoke:
            return _sandbox_diagnostic(ready=True)
        return _sandbox_diagnostic(
            ready=False,
            can_pull=True,
            next_steps=("Run `sylliptor sandbox pull`.",),
        )

    pull = Mock(return_value=SandboxPullResult(ok=True, results=()))
    monkeypatch.setattr(setup_wizard_mod, "diagnose_sandbox", fake_diagnose)
    monkeypatch.setattr(setup_wizard_mod, "pull_sandbox_images", pull)
    _config_env(tmp_path, monkeypatch)
    _patch_console(monkeypatch)
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", PickerScript(["openai", "gpt-5"]))
    monkeypatch.setattr(
        setup_wizard_mod.typer,
        "prompt",
        PromptScript(["", "sk-test-1234", os.fspath(tmp_path), "y"]),
    )

    assert setup_wizard_mod.run_setup_wizard() is True
    pull.assert_called_once_with(timeout_s=900)
    assert diagnose_calls == [False, True]


def test_setup_wizard_pull_failure_logs_raw_output_without_printing_it(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        setup_wizard_mod,
        "diagnose_sandbox",
        lambda _cfg, *, include_smoke=False: _sandbox_diagnostic(
            ready=False,
            can_pull=True,
            next_steps=("Run `sylliptor sandbox pull`.",),
        ),
    )
    monkeypatch.setattr(
        setup_wizard_mod,
        "pull_sandbox_images",
        Mock(
            return_value=SandboxPullResult(
                ok=False,
                error="docker pull failed",
                results=(
                    SandboxImagePullResult(
                        image="ghcr.io/alysisai/sylliptor-sandbox:dev",
                        ok=False,
                        output="RAW SUBPROCESS OUTPUT",
                    ),
                ),
            )
        ),
    )
    _config_env(tmp_path, monkeypatch)
    output = _patch_console(monkeypatch)
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", PickerScript(["openai", "gpt-5"]))
    monkeypatch.setattr(
        setup_wizard_mod.typer,
        "prompt",
        PromptScript(["", "sk-test-1234", os.fspath(tmp_path), "y"]),
    )

    assert setup_wizard_mod.run_setup_wizard() is True
    rendered = output.getvalue()
    assert "RAW SUBPROCESS OUTPUT" not in rendered
    assert "Raw pull output saved to:" in rendered
    logs = list((tmp_path / "data" / "logs").glob("sandbox_pull_*.log"))
    assert len(logs) == 1
    assert "RAW SUBPROCESS OUTPUT" in logs[0].read_text(encoding="utf-8")


def test_setup_wizard_unexpected_sandbox_exception_does_not_print_traceback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def boom(_cfg: AppConfig, *, include_smoke: bool = False) -> SandboxDiagnostic:
        raise RuntimeError("bang")

    monkeypatch.setattr(setup_wizard_mod, "diagnose_sandbox", boom)

    result, output, _picker, _prompt = _run_basic_wizard(monkeypatch, tmp_path)

    assert result is True
    assert "Sandbox check failed:" in output
    assert "Traceback" not in output


def test_setup_wizard_ctrl_c_after_model_back_can_cancel_without_saving(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config_env(tmp_path, monkeypatch)
    output = _patch_console(monkeypatch)
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", PickerScript(["openai", None]))
    monkeypatch.setattr(
        setup_wizard_mod.typer,
        "prompt",
        PromptScript(["", "sk-test-1234", KeyboardInterrupt(), "y"]),
    )

    assert setup_wizard_mod.run_setup_wizard() is False
    assert "Setup cancelled. No changes saved." in output.getvalue()
    assert not config_path().exists()
    assert not credentials_path().exists()


def test_setup_wizard_escape_at_welcome_confirm_cancel_writes_nothing(
    monkeypatch, tmp_path: Path
) -> None:
    _config_env(tmp_path, monkeypatch)
    output = _patch_console(monkeypatch)
    monkeypatch.setattr(
        setup_wizard_mod,
        "_esc_aware_text_input",
        PromptScript([setup_wizard_mod._GoBack(), "y"]),
    )

    assert setup_wizard_mod.run_setup_wizard() is False
    assert "Setup cancelled. No changes saved." in output.getvalue()
    assert not config_path().exists()
    assert not credentials_path().exists()


def test_setup_wizard_escape_at_welcome_decline_reshows_welcome(
    monkeypatch, tmp_path: Path
) -> None:
    _config_env(tmp_path, monkeypatch)
    output = _patch_console(monkeypatch)
    picker = PickerScript(["openai", "gpt-5"])
    text_input = PromptScript(
        [setup_wizard_mod._GoBack(), "n", "", "sk-test-1234", os.fspath(tmp_path)]
    )
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(setup_wizard_mod, "_esc_aware_text_input", text_input)

    assert setup_wizard_mod.run_setup_wizard() is True
    assert output.getvalue().count("Welcome to Sylliptor") == 2


def test_setup_wizard_escape_at_profile_returns_to_welcome(monkeypatch, tmp_path: Path) -> None:
    _config_env(tmp_path, monkeypatch)
    output = _patch_console(monkeypatch)
    picker = PickerScript([None, "openai", "gpt-5"])
    text_input = PromptScript(["", "", "sk-test-1234", os.fspath(tmp_path)])
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(setup_wizard_mod, "_esc_aware_text_input", text_input)

    assert setup_wizard_mod.run_setup_wizard() is True
    assert output.getvalue().count("Welcome to Sylliptor") == 2
    assert [call["title"] for call in picker.calls] == [
        "Provider Profile",
        "Provider Profile",
        "Default Model",
        "Router Model",
    ]


def test_setup_wizard_escape_at_api_key_returns_to_profile_with_preselection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config_env(tmp_path, monkeypatch)
    _patch_console(monkeypatch)
    picker = PickerScript(["openai", "openai", "gpt-5"])
    text_input = PromptScript(["", setup_wizard_mod._GoBack(), "sk-test-1234", os.fspath(tmp_path)])
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(setup_wizard_mod, "_esc_aware_text_input", text_input)

    assert setup_wizard_mod.run_setup_wizard() is True
    assert picker.calls[1]["title"] == "Provider Profile"
    assert picker.calls[1]["current_value"] == "openai"


def test_setup_wizard_escape_at_model_returns_to_api_key_with_previous_default(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config_env(tmp_path, monkeypatch)
    _patch_console(monkeypatch)
    picker = PickerScript(["openai", None, "gpt-5"])
    text_input = PromptScript(["", "sk-test-1234", "", os.fspath(tmp_path)])
    validations: list[str] = []

    def validate(*, profile, api_key, **_kwargs):
        del profile
        validations.append(api_key)
        return setup_wizard_mod._ApiKeyValidationResult(status="validated")

    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(setup_wizard_mod, "_esc_aware_text_input", text_input)
    monkeypatch.setattr(setup_wizard_mod, "_validate_api_key", validate)

    assert setup_wizard_mod.run_setup_wizard() is True
    assert validations == ["sk-test-1234", "sk-test-1234"]
    assert any("Enter to keep current" in call[0] for call in text_input.calls)


def test_setup_wizard_escape_at_workspace_returns_to_model_with_preselection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config_env(tmp_path, monkeypatch)
    _patch_console(monkeypatch)
    picker = PickerScript(
        [
            "openai",
            "gpt-5.5",
            setup_wizard_mod._INHERIT_DEFAULT_MODEL_VALUE,
            None,
            "gpt-5.5",
            setup_wizard_mod._INHERIT_DEFAULT_MODEL_VALUE,
        ],
        auto_router_inherit=False,
    )
    text_input = PromptScript(["", "sk-test-1234", setup_wizard_mod._GoBack(), os.fspath(tmp_path)])
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(setup_wizard_mod, "_esc_aware_text_input", text_input)

    assert setup_wizard_mod.run_setup_wizard() is True
    assert picker.calls[3]["title"] == "Router Model"
    assert picker.calls[3]["current_value"] == setup_wizard_mod._INHERIT_DEFAULT_MODEL_VALUE
    assert picker.calls[4]["title"] == "Default Model"
    assert picker.calls[4]["current_value"] == "gpt-5.5"


def test_setup_wizard_profile_change_invalidates_api_key_and_model_results(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _config_env(tmp_path, monkeypatch)
    _patch_console(monkeypatch)
    picker = PickerScript(
        [
            "openai",
            "gpt-5",
            setup_wizard_mod._INHERIT_DEFAULT_MODEL_VALUE,
            None,
            None,
            "anthropic",
            "claude-sonnet-4-6",
            setup_wizard_mod._INHERIT_DEFAULT_MODEL_VALUE,
        ],
        auto_router_inherit=False,
    )
    text_input = PromptScript(
        [
            "",
            "sk-openai",
            setup_wizard_mod._GoBack(),
            setup_wizard_mod._GoBack(),
            "sk-ant",
            os.fspath(tmp_path),
        ]
    )
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(setup_wizard_mod, "_esc_aware_text_input", text_input)

    assert setup_wizard_mod.run_setup_wizard() is True
    cfg = load_config()
    assert cfg.extra_fields["active_profile"] == "anthropic"
    assert load_persisted_profile_keys()["anthropic"] == "sk-ant"
    api_prompts = [call[0] for call in text_input.calls if call[0].startswith("Paste your API key")]
    assert api_prompts[-1] == "Paste your API key"
    assert picker.calls[-2]["title"] == "Default Model"
    assert any(row[0] == "claude-sonnet-4-6" for row in picker.calls[-2]["rows"])


def test_setup_wizard_ctrl_c_confirmation_decline_stays_on_same_step(
    monkeypatch, tmp_path: Path
) -> None:
    _config_env(tmp_path, monkeypatch)
    _patch_console(monkeypatch)
    picker = PickerScript(["openai", "gpt-5"])
    text_input = PromptScript(["", KeyboardInterrupt(), "n", "sk-test-1234", os.fspath(tmp_path)])
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(setup_wizard_mod, "_esc_aware_text_input", text_input)

    assert setup_wizard_mod.run_setup_wizard() is True
    api_prompts = [call[0] for call in text_input.calls if call[0].startswith("Paste your API key")]
    assert api_prompts == ["Paste your API key", "Paste your API key"]


def test_setup_wizard_ctrl_c_confirmation_accept_cancels_without_writes(
    monkeypatch, tmp_path: Path
) -> None:
    _config_env(tmp_path, monkeypatch)
    output = _patch_console(monkeypatch)
    picker = PickerScript(["openai"])
    text_input = PromptScript(["", KeyboardInterrupt(), "y"])
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(setup_wizard_mod, "_esc_aware_text_input", text_input)

    assert setup_wizard_mod.run_setup_wizard() is False
    assert "Setup cancelled. No changes saved." in output.getvalue()
    assert not config_path().exists()
    assert not credentials_path().exists()


def test_setup_wizard_non_tty_note_is_printed(monkeypatch, tmp_path: Path) -> None:
    _config_env(tmp_path, monkeypatch)
    output = _patch_console(monkeypatch)
    monkeypatch.setattr(
        setup_wizard_mod, "_escape_capture_unavailable_reason", lambda: "non-interactive terminal"
    )
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", PickerScript(["openai", "gpt-5"]))
    monkeypatch.setattr(
        setup_wizard_mod.typer, "prompt", PromptScript(["", "sk-test-1234", os.fspath(tmp_path)])
    )

    assert setup_wizard_mod.run_setup_wizard() is True
    assert "does not support Esc/back navigation (non-interactive terminal)" in output.getvalue()


def test_setup_wizard_back_and_forth_then_cancel_writes_nothing(
    monkeypatch, tmp_path: Path
) -> None:
    _config_env(tmp_path, monkeypatch)
    _patch_console(monkeypatch)
    picker = PickerScript(["openai", "gpt-5", None])
    text_input = PromptScript(
        [
            "",
            "sk-test-1234",
            setup_wizard_mod._GoBack(),
            KeyboardInterrupt(),
            "y",
        ]
    )
    monkeypatch.setattr(setup_wizard_mod, "_run_wizard_picker", picker)
    monkeypatch.setattr(setup_wizard_mod, "_esc_aware_text_input", text_input)

    assert setup_wizard_mod.run_setup_wizard() is False
    assert not config_path().exists()
    assert not credentials_path().exists()


def test_main_callback_triggers_wizard_on_first_run(monkeypatch, tmp_path: Path) -> None:
    _config_env(tmp_path, monkeypatch)
    calls: list[str] = []

    def fake_wizard() -> bool:
        calls.append("wizard")
        save_config(AppConfig(model="gpt-5"))
        save_persisted_api_key("sk-test-1234")
        return True

    def fake_chat() -> None:
        calls.append("chat")

    monkeypatch.setattr(cli_mod, "_is_non_interactive_terminal", lambda: False)
    monkeypatch.setattr(setup_wizard_mod, "run_setup_wizard", fake_wizard)
    monkeypatch.setattr(cli_mod, "_run_default_chat_action", fake_chat)

    result = CliRunner().invoke(
        sylliptor_app, [], env={"SYLLIPTOR_CONFIG_DIR": os.environ["SYLLIPTOR_CONFIG_DIR"]}
    )

    assert result.exit_code == 0
    assert calls == ["wizard", "chat"]


def test_main_callback_skips_wizard_on_partial_config(monkeypatch, tmp_path: Path) -> None:
    _config_env(tmp_path, monkeypatch)
    save_config(AppConfig(model=""))
    save_persisted_api_key("sk-test-1234")
    calls: list[tuple[str, str | None]] = []

    def fake_wizard() -> bool:
        calls.append(("wizard", None))
        return True

    def fake_menu(*, cfg: AppConfig | None = None, auto_focus: str | None = None):
        del cfg
        calls.append(("menu", auto_focus))

    def fake_chat() -> None:
        calls.append(("chat", None))

    monkeypatch.setattr(cli_mod, "_is_non_interactive_terminal", lambda: False)
    monkeypatch.setattr(setup_wizard_mod, "run_setup_wizard", fake_wizard)
    monkeypatch.setattr(config_menu_mod, "run_config_menu", fake_menu)
    monkeypatch.setattr(cli_mod, "_run_default_chat_action", fake_chat)

    result = CliRunner().invoke(
        sylliptor_app, [], env={"SYLLIPTOR_CONFIG_DIR": os.environ["SYLLIPTOR_CONFIG_DIR"]}
    )

    assert result.exit_code == 0
    assert calls == [("menu", "model"), ("chat", None)]


def test_setup_typer_command_runs_wizard(monkeypatch) -> None:
    called: list[bool] = []

    def fake_wizard() -> bool:
        called.append(True)
        return True

    monkeypatch.setattr(setup_wizard_mod, "run_setup_wizard", fake_wizard)

    result = CliRunner().invoke(sylliptor_app, ["setup"])

    assert result.exit_code == 0
    assert called == [True]
