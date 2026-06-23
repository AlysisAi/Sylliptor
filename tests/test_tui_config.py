"""Tests for the full-screen TUI configuration menu.

The :class:`ConfigFlow` is driven synchronously (no terminal) for the bulk of the
coverage — it reuses ``config_menu.ConfigMenuState`` so these assert the *menu
walk* (navigation + section mutations + save) rather than the config logic itself.
A single headless ``run_tui`` smoke checks the overlay is wired into the chat app:
bare ``/config`` opens it (and is not routed to the command runner), ``q`` closes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from sylliptor_agent_cli.cli_impl.tui import config_flow as flow_mod
from sylliptor_agent_cli.cli_impl.tui.config_flow import ConfigFlow
from sylliptor_agent_cli.config import (
    AppConfig,
    load_config,
    load_persisted_profile_keys,
)
from sylliptor_agent_cli.profile_presets import PROFILE_PRESETS, make_profile_from_preset

# --------------------------------------------------------------------------- helpers


def _config_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYLLIPTOR_CONFIG_DIR", os.fspath(tmp_path / "config"))
    monkeypatch.setenv("SYLLIPTOR_DATA_DIR", os.fspath(tmp_path / "data"))
    for var in (
        "SYLLIPTOR_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "SYLLIPTOR_SHELL_SANDBOX_MODE",
        "SYLLIPTOR_VERIFY_SANDBOX_MODE",
    ):
        monkeypatch.delenv(var, raising=False)


def _cfg(**overrides: Any) -> AppConfig:
    base = {"model": "gpt-4o", "base_url": "https://api.openai.com/v1"}
    base.update(overrides)
    return AppConfig(**base)


def _cfg_with_profiles(active: str = "openai") -> AppConfig:
    openai = make_profile_from_preset(
        next(p for p in PROFILE_PRESETS if p.key == "openai"), name="openai"
    )
    anthropic = make_profile_from_preset(
        next(p for p in PROFILE_PRESETS if p.key == "anthropic"), name="anthropic"
    )
    cfg = AppConfig(model="gpt-4o")
    cfg.extra_fields = {
        "profiles": {"openai": openai.to_dict(), "anthropic": anthropic.to_dict()},
        "active_profile": active,
    }
    return cfg


# --------------------------------------------------------------------------- menu


def test_menu_lists_sections_and_actions():
    flow = ConfigFlow(cfg=_cfg())
    scr = flow.screen()
    assert scr.stage == "menu" and scr.mode == "list"
    values = [r.value for r in scr.rows]
    assert values == [
        "workspace",
        "profile",
        "api_key",
        "default",
        "router",
        "sandbox",
        "advanced",
        "__save__",
        "__cancel__",
    ]
    assert not flow.state.dirty


def test_advanced_submenu_holds_subagent_and_forge():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("advanced")
    assert flow.stage == "advanced" and flow.current_mode() == "list"
    values = [r.value for r in flow.screen().rows]
    assert values == ["subagents", "forge", "back"]
    flow.choose("back")
    assert flow.stage == "menu"


def test_menu_choose_save_and_cancel_route():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("__cancel__")  # clean → exits immediately
    assert flow.stage == "done" and flow.success is False

    flow2 = ConfigFlow(cfg=_cfg())
    flow2.choose("__save__")
    assert flow2.stage == "saving" and flow2.current_mode() == "busy"


# --------------------------------------------------------------------------- sandbox


def test_sandbox_section_sets_field_and_returns_to_menu():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("sandbox")
    assert flow.stage == "sandbox"
    # The picker pre-selects the current mode (strict by default).
    assert [r.current for r in flow.screen().rows] == [True, False, False]
    flow.choose("off")
    assert flow.stage == "menu"
    assert flow.state.fields["sandbox_mode"] == "off"
    assert flow.status_tone == "warn"  # "off" warns
    assert flow.state.dirty


# --------------------------------------------------------------------------- default model


def test_default_model_full_flow():
    flow = ConfigFlow(cfg=_cfg(model="gpt-4o"))
    flow.choose("default")
    assert flow.stage == "model"
    flow.choose(flow.screen().rows[0].value)  # current model row
    assert flow.stage == "model_base_url"
    flow.submit_input("")  # keep current base_url
    assert flow.stage == "model_thinking"
    flow.choose("high")
    assert flow.stage == "model_timeout"
    flow.submit_input("45")
    assert flow.stage == "menu"
    assert flow.state.fields["llm_timeout_s"] == "45"
    assert flow.state.thinking_label == "high"


def test_default_model_custom_path():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("default")
    flow.choose(flow_mod._CUSTOM_MODEL_VALUE)
    assert flow.stage == "custom_model"
    flow.submit_input("my-custom-model")
    assert flow.stage == "model_base_url"
    assert flow.state.fields["model"] == "my-custom-model"


def test_model_timeout_rejects_non_positive():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("default")
    flow.choose(flow.screen().rows[0].value)
    flow.submit_input("")  # base_url
    flow.choose("auto")  # thinking
    flow.submit_input("0")  # invalid timeout
    assert flow.stage == "model_timeout"  # stayed
    assert flow.status_tone == "err"


# --------------------------------------------------------------------------- execution limits


def test_execution_limits_full_flow():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("router")
    assert flow.stage == "limits_routing"
    flow.choose("code_only")
    assert flow.stage == "limits_budget"
    flow.choose("fixed")
    assert flow.stage == "limits_max_steps"
    flow.submit_input("30")
    assert flow.stage == "limits_task_steps"
    flow.submit_input("120")
    assert flow.stage == "limits_subagent_steps"
    flow.submit_input("20")
    assert flow.stage == "menu"
    assert flow.state.fields["routing_mode"] == "code_only"
    assert flow.state.fields["step_budget_policy"] == "fixed"
    assert flow.state.fields["max_steps"] == "30"
    assert flow.state.fields["task_max_steps"] == "120"
    assert flow.state.fields["subagent_max_steps"] == "20"


def test_limits_rejects_bad_int_and_stays():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("router")
    flow.choose("auto")
    flow.choose("adaptive")
    flow.submit_input("notanint")
    assert flow.stage == "limits_max_steps"
    assert flow.status_tone == "err"


# --------------------------------------------------------------------------- subagent / forge


def test_subagent_overrides_sequence_and_field_key():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("advanced")
    flow.choose("subagents")
    assert flow.stage == "subagent_field" and flow.field_key() == "subagent_field:0"
    flow.submit_input("mini")  # coding model
    assert flow.field_key() == "subagent_field:1"
    assert flow.state.role_models.get("coding") == "mini"
    # Walk to the end (blank keeps each subsequent field).
    guard = 0
    while flow.stage == "subagent_field":
        flow.submit_input("")
        guard += 1
        assert guard < 30
    assert flow.stage == "advanced"  # completion returns to the Advanced sub-menu
    assert flow.state.role_models.get("coding") == "mini"


def test_subagent_temperature_validation():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("advanced")
    flow.choose("subagents")
    flow.submit_input("")  # coding model -> step to coding temp
    assert flow.field_key() == "subagent_field:1"
    flow.submit_input("-1")  # invalid temperature
    assert flow.stage == "subagent_field" and flow.field_key() == "subagent_field:1"
    assert flow.status_tone == "err"


def test_subagent_esc_reverts():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("advanced")
    flow.choose("subagents")
    flow.submit_input("changed")
    assert flow.state.role_models.get("coding") == "changed"
    flow.back()  # Esc cancels the section
    assert flow.stage == "advanced"
    assert flow.state.role_models.get("coding") == ""  # reverted


def test_forge_overrides_sequence():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("advanced")
    flow.choose("forge")
    assert flow.stage == "forge_field" and flow.field_key() == "forge_field:0"
    flow.submit_input("forge-model")
    guard = 0
    while flow.stage == "forge_field":
        flow.submit_input("")
        guard += 1
        assert guard < 30
    assert flow.stage == "advanced"
    assert flow.state.forge_role_models.get("coding") == "forge-model"


# --------------------------------------------------------------------------- api key


def test_api_key_set():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("api_key")
    assert flow.stage == "api_key"
    flow.submit_input("sk-secret-123")
    assert flow.stage == "menu"
    assert flow.state.new_api_key == "sk-secret-123"


def test_api_key_blank_keeps_current():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("api_key")
    flow.submit_input("")
    assert flow.stage == "menu"
    assert flow.state.new_api_key == ""


def test_api_key_clear_confirm():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("api_key")
    flow.submit_input("clear")
    assert flow.stage == "api_key_clear_confirm"
    flow.confirm(True)
    assert flow.stage == "menu"
    assert flow.state.clear_stored_key_confirmed is True


# --------------------------------------------------------------------------- provider profile


def test_provider_switch_changes_active():
    flow = ConfigFlow(cfg=_cfg_with_profiles(active="openai"))
    flow.choose("profile")
    flow.choose("switch")
    assert flow.stage == "provider_switch"
    flow.choose("anthropic")
    assert flow.stage == "provider"
    assert flow.state.active_profile == "anthropic"


def test_provider_add_preset_with_base_url():
    flow = ConfigFlow(cfg=AppConfig(model="x"))
    flow.choose("profile")
    flow.choose("add_preset")
    assert flow.stage == "provider_add_preset"
    flow.choose("openai")  # has a base_url → no URL prompt
    assert flow.stage == "provider_preset_name"
    flow.submit_input("")  # default name = preset key
    assert flow.stage == "provider"
    assert "openai" in flow.state.profiles
    assert flow.state.active_profile == "openai"


def test_provider_add_custom():
    flow = ConfigFlow(cfg=AppConfig(model="x"))
    flow.choose("profile")
    flow.choose("add_custom")
    assert flow.stage == "provider_custom_name"
    flow.submit_input("local")
    assert flow.stage == "provider_custom_url"
    flow.submit_input("http://localhost:1234/v1")
    assert flow.stage == "provider_custom_headers"
    flow.submit_input("")  # no headers
    assert flow.stage == "provider"
    assert "local" in flow.state.profiles
    assert flow.state.profiles["local"]["base_url"] == "http://localhost:1234/v1"


def test_provider_add_custom_rejects_empty_url():
    flow = ConfigFlow(cfg=AppConfig(model="x"))
    flow.choose("profile")
    flow.choose("add_custom")
    flow.submit_input("local")
    flow.submit_input("")  # url required
    assert flow.stage == "provider_custom_url"
    assert flow.status_tone == "err"


def test_provider_edit_secret_guard():
    flow = ConfigFlow(cfg=_cfg_with_profiles())
    flow.choose("profile")
    flow.choose("edit")
    assert flow.stage == "provider_edit" and flow.field_key() == "provider_edit:0"
    # base_url field: pasting a key-looking value is refused.
    flow.submit_input("sk-abcdef0123456789abcdef")
    assert flow.stage == "provider_edit" and flow.field_key() == "provider_edit:0"
    assert flow.status_tone == "warn"
    # 'force' keeps the flagged candidate and advances (edits accumulate locally
    # and are applied to the profile only when the whole sequence finalizes).
    flow.submit_input("force")
    assert flow.field_key() == "provider_edit:1"
    assert flow._edit_values["base_url"].startswith("sk-")


def test_provider_edit_full_walk_updates_profile():
    flow = ConfigFlow(cfg=_cfg_with_profiles())
    flow.choose("profile")
    flow.choose("edit")
    flow.submit_input("https://example.com/v1")  # base_url
    guard = 0
    while flow.stage == "provider_edit":
        flow.submit_input("")  # keep the rest
        guard += 1
        assert guard < 20
    assert flow.stage == "provider"
    assert flow.state.profiles[flow.state.active_profile]["base_url"] == "https://example.com/v1"


def test_provider_remove_confirm():
    flow = ConfigFlow(cfg=_cfg_with_profiles(active="openai"))
    flow.choose("profile")
    flow.choose("remove")
    assert flow.stage == "provider_remove"
    flow.choose("anthropic")
    assert flow.stage == "provider_remove_confirm"
    flow.confirm(True)
    assert flow.stage == "provider"
    assert "anthropic" not in flow.state.profiles


# --------------------------------------------------------------------------- save / cancel


def test_save_persists_sandbox_change(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("sandbox")
    flow.choose("off")
    assert flow.state.dirty
    flow.choose("__save__")
    assert flow.current_mode() == "busy"
    flow.run_busy()
    assert flow.stage == "done" and flow.success is True and flow.saved is True
    assert flow.changes_count >= 1
    from sylliptor_agent_cli.sandbox_settings import sandbox_mode_from_config

    assert sandbox_mode_from_config(load_config()) == "off"


def test_save_persists_profile_api_key(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    flow = ConfigFlow(cfg=_cfg_with_profiles(active="openai"))
    flow.choose("api_key")
    flow.submit_input("sk-persist-me")
    flow.choose("__save__")
    flow.run_busy()
    assert flow.stage == "done" and flow.saved is True
    assert load_persisted_profile_keys().get("openai") == "sk-persist-me"


def test_save_validation_failure_returns_to_menu(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    flow = ConfigFlow(cfg=_cfg())
    flow.state.fields["llm_timeout_s"] = "-5"  # invalid, bypassing the input guard
    flow._goto("saving")
    flow.run_busy()
    assert flow.stage == "menu"
    assert flow.status_tone == "err"
    assert flow.saved is False and flow.success is None


def test_cancel_when_dirty_confirms():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("sandbox")
    flow.choose("off")  # dirty
    flow.request_cancel()
    assert flow.stage == "cancel_confirm"
    flow.confirm(False)  # keep editing
    assert flow.stage == "menu"
    flow.request_cancel()
    flow.confirm(True)  # discard
    assert flow.stage == "done" and flow.success is False


def test_cancel_when_clean_exits_directly():
    flow = ConfigFlow(cfg=_cfg())
    flow.request_cancel()
    assert flow.stage == "done" and flow.success is False


def test_cancel_no_resumes_to_current_subflow_stage():
    # Ctrl+C inside a sub-flow → discard confirm; answering "No" returns to where
    # the user was (the sub-flow), not the top menu.
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("sandbox")
    flow.choose("off")  # make it dirty, back at menu
    flow.choose("default")  # enter the model sub-flow
    assert flow.stage == "model"
    flow.request_cancel()
    assert flow.stage == "cancel_confirm"
    flow.confirm(False)
    assert flow.stage == "model"  # resumed, not snapped to menu


# --------------------------------------------------------------------------- save split (worker/UI handoff)


def test_perform_save_records_outcome_without_transition(monkeypatch, tmp_path):
    # The overlay runs perform_save() on a worker (no renderer-visible stage write),
    # then apply_save_outcome() on the UI thread does the transition.
    _config_env(tmp_path, monkeypatch)
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("sandbox")
    flow.choose("off")
    flow._goto("saving")
    flow.perform_save()
    assert flow.stage == "saving"  # perform_save did NOT transition
    assert flow._save_outcome == ("saved", "")
    assert flow.saved is False
    flow.apply_save_outcome()
    assert flow.stage == "done" and flow.saved is True and flow.success is True


def test_perform_save_validation_failure_outcome(monkeypatch, tmp_path):
    _config_env(tmp_path, monkeypatch)
    flow = ConfigFlow(cfg=_cfg())
    flow.state.fields["llm_timeout_s"] = "-5"  # invalid
    flow._goto("saving")
    flow.perform_save()
    assert flow._save_outcome is not None and flow._save_outcome[0] == "error"
    assert flow.stage == "saving"  # not transitioned yet
    flow.apply_save_outcome()
    assert flow.stage == "menu" and flow.status_tone == "err" and flow.saved is False


def test_set_save_failure_surfaces_on_apply():
    flow = ConfigFlow(cfg=_cfg())
    flow._goto("saving")
    flow.set_save_failure("disk on fire")
    flow.apply_save_outcome()
    assert flow.stage == "menu" and flow.status_tone == "err"
    assert "disk on fire" in flow.status


# --------------------------------------------------------------------------- project / workspace


def test_menu_has_project_workspace_first():
    flow = ConfigFlow(cfg=_cfg(), current_workspace="/home/me/proj")
    rows = flow.screen().rows
    assert rows[0].value == "workspace" and rows[0].label == "Project / Workspace"
    assert "proj" in rows[0].description  # current workspace name shown


def test_workspace_screen_offers_type_path_and_back():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("workspace")
    assert flow.stage == "workspace"
    values = [r.value for r in flow.screen().rows]
    assert "__type_path__" in values and "back" in values


def test_workspace_set_default_stages_through_save(tmp_path, monkeypatch):
    _config_env(tmp_path, monkeypatch)
    project = tmp_path / "myproject"
    project.mkdir()
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("workspace")
    flow.choose("__type_path__")
    assert flow.stage == "workspace_path"
    flow.submit_input(os.fspath(project))
    assert flow.stage == "workspace_action"
    assert "myproject" in flow._pending_workspace
    flow.choose("set_default")
    assert flow.stage == "menu" and flow.state.dirty
    assert "myproject" in flow.state.default_workspace_path
    # Persists through the normal Save.
    flow._goto("saving")
    flow.run_busy()
    assert flow.stage == "done" and flow.saved is True
    saved = load_config()
    assert "myproject" in str((saved.extra_fields or {}).get("default_workspace_path"))


def test_workspace_switch_now_signals_switch_and_persists(tmp_path, monkeypatch):
    _config_env(tmp_path, monkeypatch)
    project = tmp_path / "other"
    project.mkdir()
    flow = ConfigFlow(cfg=_cfg(), current_workspace=os.fspath(tmp_path))
    flow.choose("workspace")
    flow.choose("__type_path__")
    flow.submit_input(os.fspath(project))
    assert flow.stage == "workspace_action"
    flow.choose("switch")
    assert flow.stage == "switching" and flow.current_mode() == "busy"
    flow.run_busy()
    assert flow.stage == "done" and flow.saved is True
    assert flow.switch_workspace and "other" in flow.switch_workspace
    saved = load_config()
    assert "other" in str((saved.extra_fields or {}).get("default_workspace_path"))


def test_workspace_empty_path_reprompts():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("workspace")
    flow.choose("__type_path__")
    flow.submit_input("   ")
    assert flow.stage == "workspace_path" and flow.status_tone == "err"


def test_workspace_back_returns_to_menu():
    flow = ConfigFlow(cfg=_cfg())
    flow.choose("workspace")
    flow.choose("back")
    assert flow.stage == "menu"


# --------------------------------------------------------------------------- headless wiring smoke


class _FakeSession:
    def __init__(self, surface: Any) -> None:
        self.surface = surface

    def run_turn(self, text: str, *, cancellation_token: Any = None) -> int:
        return 0

    def close(self) -> None:
        pass


def _fake_command_runner(calls: list) -> Any:
    def _runner(sess: Any, text: str, width: int):
        calls.append((text, width))
        if text.strip().lower() in {"/exit", "/quit"}:
            return ("exit", "", None, None)
        return ("handled", "", None, None)

    return _runner


def _run_headless(keys: str, **kwargs: Any):
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from sylliptor_agent_cli.cli_impl.tui import run_tui
    from sylliptor_agent_cli.cli_impl.tui.state import TuiState

    state = TuiState(model_name="m", username="t")
    with create_pipe_input() as pipe:
        pipe.send_text(keys)
        return run_tui(state, owl_color=False, input=pipe, output=DummyOutput(), **kwargs)


def test_headless_config_opens_overlay_not_routed_to_runner():
    # Bare /config is intercepted natively: it opens the overlay (the factory is
    # invoked) instead of being echoed as a user line or routed to the runner.
    # Pressing q closes it (clean menu → exits), then /exit leaves.
    calls: list = []
    opened = {"n": 0}

    def _factory():
        opened["n"] += 1
        return ConfigFlow(cfg=_cfg())

    _result, transcript = _run_headless(
        "/config\rq/exit\r",
        session_builder=_FakeSession,
        command_runner=_fake_command_runner(calls),
        config_flow_factory=_factory,
        background_turns=False,
    )
    assert opened["n"] == 1
    assert ("user", "/config") not in transcript
    assert all(text.strip().lower() != "/config" for text, _w in calls)
