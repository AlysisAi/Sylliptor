"""Tests for the full-screen TUI setup wizard.

The :class:`SetupFlow` is driven synchronously (no terminal) for the bulk of the
coverage; a couple of headless ``run_setup_tui`` smokes exercise the prompt_toolkit
application through a pipe input + dummy output with ``inline_busy`` so a
pre-loaded key sequence walks the whole flow deterministically.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import sylliptor_agent_cli.account_login as account_login
import sylliptor_agent_cli.sandbox_doctor as sandbox_doctor
from sylliptor_agent_cli.cli_impl.tui import setup_flow as flow_mod
from sylliptor_agent_cli.cli_impl.tui.setup_app import run_setup_tui
from sylliptor_agent_cli.cli_impl.tui.setup_flow import SetupFlow
from sylliptor_agent_cli.config import ConfigError, load_config, load_persisted_profile_keys

# --------------------------------------------------------------------------- helpers


def _config_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path / "config"))
    monkeypatch.setenv("SYLLIPTOR_DATA_DIR", os.fspath(tmp_path / "data"))
    for var in ("SYLLIPTOR_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def _fake_diag(*, ready: bool, can_pull: bool = False, backend: str = "docker") -> SimpleNamespace:
    return SimpleNamespace(
        ready=ready,
        status="ready" if ready else "not_ready",
        selected_backend=backend if ready else None,
        docker_image="img:dev",
        can_pull=can_pull,
    )


def _patch_validate(
    monkeypatch: pytest.MonkeyPatch, status: str = "validated", message: str = ""
) -> None:
    monkeypatch.setattr(
        flow_mod._wiz,
        "_validate_api_key",
        lambda **_k: flow_mod._wiz._ApiKeyValidationResult(status=status, message=message),
    )


def _drive_busy(flow: SetupFlow) -> None:
    """Run busy steps until the flow lands on an interactive screen."""
    guard = 0
    while flow.current_mode() == "busy":
        flow.run_busy()
        guard += 1
        assert guard < 20, "busy chain did not converge"


# --------------------------------------------------------------------------- flow: happy path


def test_flow_full_happy_path_persists(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    _patch_validate(monkeypatch)
    monkeypatch.setattr(
        sandbox_doctor, "diagnose_sandbox", lambda _cfg, **_k: _fake_diag(ready=True)
    )

    flow = SetupFlow()
    assert flow.screen().stage == "welcome"
    flow.advance_message()  # -> provider
    flow.choose("openai")  # compat OpenAI (requires a key)
    assert flow.stage == "api_key"
    flow.submit_input("sk-test-123")
    _drive_busy(flow)  # validate key -> model
    assert flow.stage == "model"
    flow.choose("gpt-5.5")
    _drive_busy(flow)  # validate model -> workspace
    assert flow.stage == "workspace"
    flow.submit_input(os.fspath(tmp_path))
    _drive_busy(flow)  # commit -> diagnose -> complete
    assert flow.stage == "complete"

    cfg = load_config()
    assert cfg.model == "gpt-5.5"
    assert load_persisted_profile_keys().get("openai") == "sk-test-123"
    # Summary reflects a validated key + ready sandbox.
    summary = " ".join(text for text, _tone in flow._summary_lines())
    assert "validated" in summary
    assert "ready" in summary

    # Enter on the complete screen finishes with success.
    flow.advance_message()
    assert flow.stage == "done"
    assert flow.success is True


def test_flow_optional_key_provider_skips_validation(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sandbox_doctor, "diagnose_sandbox", lambda _cfg, **_k: _fake_diag(ready=True)
    )

    flow = SetupFlow()
    flow.advance_message()
    flow.choose("ollama")  # local: api_key_env is None
    assert flow.stage == "api_key"
    flow.submit_input("")  # empty key is allowed for this provider
    assert flow.stage == "model"
    assert flow.api_key_result is not None and flow.api_key_result.validation_status == "skipped"


# --------------------------------------------------------------------------- flow: validation


def test_flow_key_validation_failure_retries_then_continues(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    _patch_validate(monkeypatch, status="failed", message="bad key")

    flow = SetupFlow()
    flow.advance_message()
    flow.choose("openai")
    for _ in range(flow_mod._MAX_KEY_ATTEMPTS - 1):
        flow.submit_input("sk-bad")
        flow.run_busy()
        assert flow.stage == "api_key"  # bounced back to retry
        assert flow.status_tone == "err"
    # Final attempt continues with the last key rather than blocking forever.
    flow.submit_input("sk-bad")
    flow.run_busy()
    assert flow.stage == "model"
    assert flow.api_key_result.validation_status == "failed"


def test_flow_custom_model_not_found_confirm(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)

    # Key validates, but the chosen model is reported missing.
    calls = {"n": 0}

    def _validate(**kwargs: Any):
        calls["n"] += 1
        if kwargs.get("model"):
            return flow_mod._wiz._ApiKeyValidationResult(
                status="model_not_found", message="no such model"
            )
        return flow_mod._wiz._ApiKeyValidationResult(status="validated")

    monkeypatch.setattr(flow_mod._wiz, "_validate_api_key", _validate)

    flow = SetupFlow()
    flow.advance_message()
    flow.choose("openai")
    flow.submit_input("sk-test")
    _drive_busy(flow)  # -> model
    flow.choose(flow_mod._wiz._CUSTOM_MODEL_VALUE)
    assert flow.stage == "custom_model"
    flow.submit_input("totally-made-up")
    _drive_busy(flow)
    assert flow.stage == "model_not_found_confirm"
    flow.confirm(True)  # use it anyway
    assert flow.stage == "workspace"
    assert flow.api_key_result.validation_status == "inconclusive"


def test_flow_workspace_invalid_path_stays(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    _patch_validate(monkeypatch)

    flow = SetupFlow()
    flow.advance_message()
    flow.choose("ollama")
    flow.submit_input("")  # skip key
    flow.choose("llama3.3")
    _drive_busy(flow)
    assert flow.stage == "workspace"
    target_file = tmp_path / "afile.txt"
    target_file.write_text("x", encoding="utf-8")
    flow.submit_input(os.fspath(target_file))  # not a directory
    assert flow.stage == "workspace"
    assert flow.status_tone == "err"


# --------------------------------------------------------------------------- flow: custom profile


def test_flow_custom_profile_builds_openai_compat(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    _patch_validate(monkeypatch)

    flow = SetupFlow()
    flow.advance_message()
    flow.choose("custom")
    assert flow.stage == "custom_name"
    flow.submit_input("myco")
    assert flow.stage == "custom_url"
    flow.submit_input("https://api.example.com/v1")
    assert flow.stage == "custom_headers"
    flow.submit_input("x-api-key=abc, x-org=acme")  # header-authenticated endpoint
    assert flow.stage == "api_key"
    profile = flow.profile_result.profile
    assert profile.name == "myco"
    assert profile.protocol == "openai_compat"
    assert profile.base_url == "https://api.example.com/v1"
    assert profile.extra_headers == {"x-api-key": "abc", "x-org": "acme"}
    # Back from api_key returns through the headers step to the URL step.
    flow.back()
    assert flow.stage == "custom_headers"
    flow.back()
    assert flow.stage == "custom_url"


def test_flow_custom_profile_malformed_headers_reprompt(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    flow = SetupFlow()
    flow.advance_message()
    flow.choose("custom")
    flow.submit_input("")  # name defaults to "custom"
    flow.submit_input("https://api.example.com/v1")
    assert flow.stage == "custom_headers"
    flow.submit_input("not-a-header")  # missing '=' -> re-prompt
    assert flow.stage == "custom_headers"
    assert flow.status_tone == "err"
    flow.submit_input("")  # empty -> no headers, proceed
    assert flow.stage == "api_key"
    assert flow.profile_result.profile.extra_headers == {}


# --------------------------------------------------------------------------- flow: sandbox branches


def _to_sandbox(flow: SetupFlow, tmp_path: Path) -> None:
    flow.advance_message()
    flow.choose("ollama")
    flow.submit_input("")
    flow.choose("llama3.3")
    _drive_busy(flow)
    flow.submit_input(os.fspath(tmp_path))
    _drive_busy(flow)  # commit -> diagnose (-> next sandbox screen)


def test_flow_sandbox_pull_confirm_yes(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    _patch_validate(monkeypatch)
    seq = [_fake_diag(ready=False, can_pull=True), _fake_diag(ready=True)]
    monkeypatch.setattr(sandbox_doctor, "diagnose_sandbox", lambda _cfg, **_k: seq.pop(0))
    monkeypatch.setattr(
        sandbox_doctor,
        "pull_sandbox_images",
        lambda **_k: SimpleNamespace(ok=True, error=None, results=[]),
    )

    _to_sandbox(flow := SetupFlow(), tmp_path)
    assert flow.stage == "sandbox_pull_confirm"
    flow.confirm(True)
    _drive_busy(flow)  # pull -> recheck -> complete
    assert flow.stage == "complete"
    assert flow.sandbox_result.ready is True


def test_flow_sandbox_no_backend_disable(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    _patch_validate(monkeypatch)
    monkeypatch.setattr(
        sandbox_doctor,
        "diagnose_sandbox",
        lambda _cfg, **_k: _fake_diag(ready=False, can_pull=False),
    )
    monkeypatch.setattr(sandbox_doctor, "detect_bubblewrap_install_plan", lambda: None)

    _to_sandbox(flow := SetupFlow(), tmp_path)
    assert flow.stage == "sandbox_choice"
    # With no install plan, only disable / later are offered.
    values = [r.value for r in flow.screen().rows]
    assert values == ["disable", "later"]
    flow.choose("disable")
    _drive_busy(flow)
    assert flow.stage == "complete"
    assert flow.sandbox_result.status == "disabled"
    # Both sandbox keys were written off (strict invariant) — config reloads clean.
    cfg = load_config()
    assert cfg is not None


# --------------------------------------------------------------------------- flow: hosted MiMo login


def test_flow_hosted_mimo_offers_login(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    _patch_validate(monkeypatch, status="skipped")
    monkeypatch.setattr(
        sandbox_doctor, "diagnose_sandbox", lambda _cfg, **_k: _fake_diag(ready=True)
    )
    monkeypatch.setattr(
        account_login, "login", lambda _cfg, **_k: SimpleNamespace(email="me@example.com")
    )

    flow = SetupFlow()
    flow.advance_message()
    flow.choose("sylliptor")  # hosted MiMo trial (no key)
    flow.submit_input("")  # optional key skipped
    flow.choose("mimo")
    _drive_busy(flow)  # validate (skipped) -> workspace
    flow.submit_input(os.fspath(tmp_path))
    _drive_busy(flow)  # commit -> diagnose -> login_confirm
    assert flow.stage == "login_confirm"
    flow.confirm(True)
    _drive_busy(flow)  # logging_in -> complete
    assert flow.stage == "complete"
    assert "me@example.com" in flow.login_summary


# --------------------------------------------------------------------------- flow: cancel / back


def test_flow_cancel_at_welcome(monkeypatch):
    flow = SetupFlow()
    flow.request_cancel()
    assert flow.stage == "cancel_confirm"
    flow.confirm(False)  # keep going
    assert flow.stage == "welcome"
    flow.request_cancel()
    flow.confirm(True)  # cancel for real
    assert flow.stage == "done"
    assert flow.success is False


def test_flow_back_navigation(monkeypatch):
    _patch_validate(monkeypatch)
    flow = SetupFlow()
    flow.advance_message()
    flow.choose("openai")
    assert flow.stage == "api_key"
    flow.back()
    assert flow.stage == "provider"
    flow.back()
    assert flow.stage == "welcome"


# --------------------------------------------------------------------------- headless application


def _headless(keys: str, **kwargs: Any) -> bool:
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        return run_setup_tui(
            owl_color=False, input=pipe, output=DummyOutput(), inline_busy=True, **kwargs
        )


def test_headless_cancel_returns_false(tmp_path, monkeypatch):
    _config_env(tmp_path, monkeypatch)
    # welcome Enter -> provider; Ctrl+C exits immediately (no confirm to get stuck on).
    assert _headless("\r\x03") is False


def test_headless_ctrl_c_exits_from_input_step(tmp_path, monkeypatch):
    _config_env(tmp_path, monkeypatch)
    # welcome -> provider(idx0) -> api_key input; Ctrl+C must still exit.
    assert _headless("\r\r\x03") is False


def test_headless_full_path_saves(tmp_path, monkeypatch):
    _config_env(tmp_path, monkeypatch)
    _patch_validate(monkeypatch)
    monkeypatch.setattr(
        sandbox_doctor, "diagnose_sandbox", lambda _cfg, **_k: _fake_diag(ready=True)
    )

    # welcome -> provider(idx0 openai-responses, key required) -> type key ->
    # model(idx0) -> workspace(default home, Enter) -> complete -> done.
    keys = "\r" + "\r" + "sk-xyz" + "\r" + "\r" + "\r" + "\r"
    assert _headless(keys) is True
    cfg = load_config()
    assert cfg.model  # a default model was persisted


# --------------------------------------------------------------------------- regression: more paths


def test_flow_preset_model_not_found_bounces_to_picker(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)

    def _validate(**kwargs: Any):
        if kwargs.get("model"):
            return flow_mod._wiz._ApiKeyValidationResult(status="model_not_found", message="gone")
        return flow_mod._wiz._ApiKeyValidationResult(status="validated")

    monkeypatch.setattr(flow_mod._wiz, "_validate_api_key", _validate)

    flow = SetupFlow()
    flow.advance_message()
    flow.choose("openai")
    flow.submit_input("sk-test")
    _drive_busy(flow)
    flow.choose("gpt-5.5")  # a PRESET model (custom=False) -> else branch
    _drive_busy(flow)
    assert flow.stage == "model"  # bounced back to the picker, not the confirm
    assert flow.status_tone == "err"
    assert flow.api_key_result.validation_status == "model_not_found"
    # The summary still renders sanely after a preset miss.
    summary = " ".join(t for t, _tone in flow._summary_lines())
    assert "model validation failed" in summary
    assert flow._api_key_tone() == "err"


def test_flow_fatal_commit_error(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    _patch_validate(monkeypatch)
    monkeypatch.setattr(
        sandbox_doctor, "diagnose_sandbox", lambda _cfg, **_k: _fake_diag(ready=True)
    )

    def _boom(**_k: Any):
        raise ConfigError("disk full")

    monkeypatch.setattr(flow_mod._wiz, "_commit_setup", _boom)

    flow = SetupFlow()
    flow.advance_message()
    flow.choose("ollama")
    flow.submit_input("")
    flow.choose("llama3.3")
    _drive_busy(flow)
    flow.submit_input(os.fspath(tmp_path))
    _drive_busy(flow)  # committing -> fatal (NOT complete)
    assert flow.stage == "fatal"
    assert "disk full" in flow.fatal_error
    flow.advance_message()
    assert flow.stage == "done"
    assert flow.success is False  # a save failure must not claim success


def test_flow_hosted_mimo_login_failure_is_non_fatal(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    _patch_validate(monkeypatch, status="skipped")
    monkeypatch.setattr(
        sandbox_doctor, "diagnose_sandbox", lambda _cfg, **_k: _fake_diag(ready=True)
    )

    def _login_boom(_cfg: Any, **_k: Any):
        raise RuntimeError("network down")

    monkeypatch.setattr(account_login, "login", _login_boom)

    flow = SetupFlow()
    flow.advance_message()
    flow.choose("sylliptor")
    flow.submit_input("")
    flow.choose("mimo")
    _drive_busy(flow)
    flow.submit_input(os.fspath(tmp_path))
    _drive_busy(flow)  # -> login_confirm
    assert flow.stage == "login_confirm"
    flow.confirm(True)
    _drive_busy(flow)  # logging_in raises internally -> complete, NOT fatal
    assert flow.stage == "complete"
    assert flow.login_ok is False
    assert "not connected" in flow.login_summary and "network down" in flow.login_summary
    acct_tone = next(tone for text, tone in flow._summary_lines() if text.startswith("Account"))
    assert acct_tone == "warn"  # a failed login is not shown in the green success tone
    assert flow.success is None  # setup is not marked failed


def test_flow_sandbox_diagnose_raises_is_contained(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    _patch_validate(monkeypatch)

    def _raise(*_a: Any, **_k: Any):
        raise RuntimeError("docker daemon down")

    monkeypatch.setattr(sandbox_doctor, "diagnose_sandbox", _raise)

    _to_sandbox(flow := SetupFlow(), tmp_path)  # commit -> diagnose (raises, contained)
    assert flow.stage == "complete"
    assert flow.sandbox_result.status == "check failed"
    assert flow.status_tone == "warn"


def test_flow_sandbox_pull_raises_is_contained(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    _patch_validate(monkeypatch)
    monkeypatch.setattr(
        sandbox_doctor,
        "diagnose_sandbox",
        lambda _cfg, **_k: _fake_diag(ready=False, can_pull=True),
    )

    def _raise(**_k: Any):
        raise RuntimeError("registry unreachable")

    monkeypatch.setattr(sandbox_doctor, "pull_sandbox_images", _raise)

    _to_sandbox(flow := SetupFlow(), tmp_path)
    assert flow.stage == "sandbox_pull_confirm"
    flow.confirm(True)
    _drive_busy(flow)  # pull raises internally -> contained, not fatal
    assert flow.stage == "complete"
    assert flow.sandbox_result.status == "pull failed"
    assert flow.status_tone == "warn"


def test_flow_sandbox_install_branch(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    _patch_validate(monkeypatch)
    seq = [_fake_diag(ready=False, can_pull=False), _fake_diag(ready=True)]
    monkeypatch.setattr(sandbox_doctor, "diagnose_sandbox", lambda _cfg, **_k: seq.pop(0))
    monkeypatch.setattr(
        sandbox_doctor,
        "detect_bubblewrap_install_plan",
        lambda: SimpleNamespace(display="apt-get install -y bubblewrap"),
    )
    monkeypatch.setattr(
        sandbox_doctor, "install_bubblewrap", lambda **_k: SimpleNamespace(ok=True, detail="")
    )

    _to_sandbox(flow := SetupFlow(), tmp_path)
    assert flow.stage == "sandbox_choice"
    assert [r.value for r in flow.screen().rows] == ["install_bwrap", "disable", "later"]
    flow.choose("install_bwrap")
    assert flow.stage == "installing_sandbox"
    assert flow.busy_kind() == "terminal"  # runs via run_in_terminal, not a worker
    _drive_busy(flow)  # install -> recheck(ready) -> complete
    assert flow.stage == "complete"
    assert flow.sandbox_result.ready is True


def test_flow_sandbox_install_failure_maps_to_not_ready(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    _patch_validate(monkeypatch)
    monkeypatch.setattr(
        sandbox_doctor,
        "diagnose_sandbox",
        lambda _cfg, **_k: _fake_diag(ready=False, can_pull=False),
    )
    monkeypatch.setattr(
        sandbox_doctor, "detect_bubblewrap_install_plan", lambda: SimpleNamespace(display="apt ...")
    )
    monkeypatch.setattr(
        sandbox_doctor,
        "install_bubblewrap",
        lambda **_k: SimpleNamespace(ok=False, detail="no package manager"),
    )

    _to_sandbox(flow := SetupFlow(), tmp_path)
    flow.choose("install_bwrap")
    _drive_busy(flow)
    assert flow.stage == "complete"
    assert flow.sandbox_result.status == "not ready"
    assert flow.status_tone == "warn"


def test_flow_api_key_keep_current_on_return(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    _patch_validate(monkeypatch)

    flow = SetupFlow()
    flow.advance_message()
    flow.choose("openai")  # key-required provider
    flow.submit_input("sk-keep")
    _drive_busy(flow)  # -> model
    flow.back()  # model -> api_key, with the validated key still held
    assert flow.stage == "api_key"
    attempts_before = flow._key_attempts
    flow.submit_input("")  # empty submit keeps the current key (no re-paste, no penalty)
    assert flow.stage == "model"
    assert flow._key_attempts == attempts_before
    assert flow.api_key_result.api_key == "sk-keep"


def test_flow_empty_required_key_does_not_consume_retry_budget(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    flow = SetupFlow()
    flow.advance_message()
    flow.choose("openai")  # key required, none entered yet
    flow.submit_input("")
    assert flow.stage == "api_key"
    flow.submit_input("")
    assert flow.stage == "api_key"
    assert flow._key_attempts == 0  # empty submits must not erode the failure budget


# --------------------------------------------------------------------------- wiring regression


def test_wiring_invokes_setup_tui_when_enabled_and_interactive(monkeypatch):
    """Regression for the import bug that left the setup TUI dead on arrival."""
    from sylliptor_agent_cli.cli_impl import tui as tui_pkg
    from sylliptor_agent_cli.cli_impl.commands import startup
    from sylliptor_agent_cli.cli_impl.tui import setup_app as setup_app_mod

    calls = {"n": 0}

    def _fake_run(**_k: Any) -> bool:
        calls["n"] += 1
        return True

    monkeypatch.setattr(tui_pkg, "is_tui_enabled", lambda: True)
    monkeypatch.setattr(startup, "_is_non_interactive_terminal", lambda: False)
    monkeypatch.setattr(setup_app_mod, "run_setup_tui", _fake_run)

    assert startup._try_setup_tui() is True
    assert calls["n"] == 1  # run_setup_tui was actually reached


def test_wiring_skips_setup_tui_when_non_interactive(monkeypatch):
    from sylliptor_agent_cli.cli_impl import tui as tui_pkg
    from sylliptor_agent_cli.cli_impl.commands import startup
    from sylliptor_agent_cli.cli_impl.tui import setup_app as setup_app_mod

    def _should_not_run(**_k: Any) -> bool:
        raise AssertionError("run_setup_tui must not run on a non-interactive terminal")

    monkeypatch.setattr(tui_pkg, "is_tui_enabled", lambda: True)
    monkeypatch.setattr(startup, "_is_non_interactive_terminal", lambda: True)
    monkeypatch.setattr(setup_app_mod, "run_setup_tui", _should_not_run)

    assert startup._try_setup_tui() is None  # falls back to the classic wizard


def test_wiring_setup_command_runs_tui_without_flag(monkeypatch):
    """`sylliptor setup` shows the interactive screens even when SYLLIPTOR_TUI is off."""
    from sylliptor_agent_cli.cli_impl import tui as tui_pkg
    from sylliptor_agent_cli.cli_impl.commands import startup
    from sylliptor_agent_cli.cli_impl.tui import setup_app as setup_app_mod

    calls = {"n": 0}

    def _fake_run(**_k: Any) -> bool:
        calls["n"] += 1
        return True

    monkeypatch.setattr(tui_pkg, "is_tui_enabled", lambda: False)  # flag OFF
    monkeypatch.setattr(startup, "_is_non_interactive_terminal", lambda: False)
    monkeypatch.setattr(setup_app_mod, "run_setup_tui", _fake_run)

    # First-run path stays gated (flag off -> classic).
    assert startup._try_setup_tui() is None
    assert calls["n"] == 0
    # The explicit setup command opts out of the gate.
    assert startup._try_setup_tui(require_flag=False) is True
    assert calls["n"] == 1


def test_wiring_announces_fallback_reason(monkeypatch, capsys):
    from sylliptor_agent_cli.cli_impl import tui as tui_pkg
    from sylliptor_agent_cli.cli_impl.commands import startup

    monkeypatch.setattr(tui_pkg, "is_tui_enabled", lambda: True)
    monkeypatch.setattr(startup, "_is_non_interactive_terminal", lambda: True)

    assert startup._try_setup_tui(require_flag=False, announce_fallback=True) is None
    out = capsys.readouterr().out
    assert "classic setup wizard" in out and "not interactive" in out
