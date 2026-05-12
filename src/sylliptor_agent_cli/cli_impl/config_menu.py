from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import typer
from click import Abort
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from ..config import (
    _ROLE_TEMPERATURE_FIELDS,
    AppConfig,
    ConfigError,
    clear_persisted_api_key,
    clear_persisted_profile_key,
    config_path,
    load_config,
    resolve_api_key,
    resolve_llm_reasoning_effort,
    save_config,
    save_persisted_api_key,
    save_persisted_profile_key,
    set_config_value,
)
from ..profile_presets import (
    PROFILE_PRESETS,
    ProfilePreset,
    find_preset_for_base_url,
    find_preset_for_profile,
    make_profile_from_preset,
    model_options_for_preset,
)
from ..profiles import (
    ProfileSpec,
    list_profiles,
    set_active_profile,
    validate_base_url,
)
from ..surface.console import make_console
from ..surface.styles import STYLE_CONTENT, STYLE_DIM, STYLE_EMPHASIS
from ..web_search_adapters import WEB_SEARCH_ADAPTER_CHOICES, normalize_web_search_adapter

ROLE_ORDER: tuple[str, ...] = (
    "coding",
    "planner",
    "review",
    "compactor",
    "conflict_review",
    "conflict_resolve",
)
SECTION_VALUES: tuple[tuple[str, str], ...] = (
    ("profile", "Provider Profile"),
    ("api_key", "API Key"),
    ("default", "Default Model"),
    ("router", "Execution Limits"),
    ("subagents", "Subagent model overrides"),
    ("forge", "Forge model overrides"),
)
THINKING_LABELS: tuple[str, ...] = ("off", "minimal", "low", "medium", "high", "xhigh", "auto")
ROUTING_MODES: tuple[str, ...] = ("auto", "code_only")
STEP_BUDGET_POLICIES: tuple[str, ...] = ("adaptive", "fixed")
_THINKING_LABEL_EXTRA_FIELD = "llm_thinking_label"
_TRUE_THINKING_LABELS = {"minimal", "low", "medium", "high", "xhigh"}
_MISSING_REQUIRED = "[yellow]missing · required[/yellow]"
_CLEAR_KEY_PENDING = "[yellow]will be cleared on save[/yellow]"
_WARNING_SUMMARIES = {_MISSING_REQUIRED, _CLEAR_KEY_PENDING}
_ROLE_DESCRIPTIONS: dict[str, str] = {
    "coding": "generating and editing code",
    "planner": "high-level task planning",
    "review": "reviewing diffs and changes",
    "compactor": "summarizing long conversations",
    "conflict_review": "reviewing merge conflicts",
    "conflict_resolve": "resolving merge conflicts",
}
_FIELD_LABELS: dict[str, str] = {
    "max_steps": "Max steps per response",
    "task_max_steps": "Max steps per task",
    "subagent_max_steps": "Max steps per subagent run",
}
_API_KEY_ENV_PROMPT = "API key env var name (NOT the key itself, e.g. 'ANTHROPIC_API_KEY')"
_API_KEY_ENV_FIELD_NAME = "API key env var name"
_SECRET_FORCE_TOKEN = "force"
_CUSTOM_MODEL_VALUE = "__custom_model__"


@dataclass(frozen=True)
class ConfigMenuResult:
    saved: bool
    changes: dict[str, Any]
    api_key_changed: bool
    error: str | None = None


@dataclass
class ConfigMenuState:
    fields: dict[str, str]
    thinking_label: str
    role_models: dict[str, str]
    forge_role_models: dict[str, str]
    role_temperatures: dict[str, str]
    profiles: dict[str, dict[str, Any]]
    active_profile: str
    api_key_source: str = "missing"
    masked_api_key: str = "(not set)"
    new_api_key: str = ""
    clear_stored_key_confirmed: bool = False
    clear_stored_key_profile: str | None = None
    config_warning: str | None = None
    thinking_label_explicitly_set: bool = field(default=False, repr=False)
    _original: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_cfg(cls, cfg: AppConfig) -> ConfigMenuState:
        warnings: list[str] = []
        try:
            resolved_key = resolve_api_key(cfg)
            api_key_source = resolved_key.source
            masked_api_key = mask_api_key(resolved_key.key)
        except ConfigError as exc:
            warnings.append(str(exc))
            api_key_source = "error"
            masked_api_key = "(not set)"
        try:
            thinking_label = thinking_label_from_cfg(cfg)
        except ConfigError as exc:
            warnings.append(str(exc))
            thinking_label = "auto"
        profiles = {profile.name: profile.to_dict() for profile in list_profiles(cfg)}
        active_profile = str(cfg.extra_fields.get("active_profile") or "")
        role_models = _normalized_role_model_values(
            cfg.extra_fields.get("role_models") if isinstance(cfg.extra_fields, dict) else None
        )
        forge_role_models = _normalized_role_model_values(
            cfg.extra_fields.get("forge_role_models")
            if isinstance(cfg.extra_fields, dict)
            else None
        )
        role_temperatures: dict[str, str] = {}
        default_temperature = _finite_float(getattr(cfg, "temperature", 0.2), fallback=0.2)
        for role in ROLE_ORDER:
            field_name = _ROLE_TEMPERATURE_FIELDS.get(role)
            if field_name is None:
                continue
            current = _finite_float(getattr(cfg, field_name, default_temperature), fallback=None)
            if current is None or current == default_temperature:
                role_temperatures[role] = ""
            else:
                role_temperatures[role] = _format_number(current)

        state = cls(
            fields={
                "model": str(getattr(cfg, "model", "") or ""),
                "base_url": str(getattr(cfg, "base_url", "") or ""),
                "llm_timeout_s": _format_number(getattr(cfg, "llm_timeout_s", 60.0)),
                "routing_mode": _normalize_routing_mode(
                    getattr(cfg, "routing_mode", "auto") or "auto"
                ),
                "step_budget_policy": _normalize_step_budget_policy(
                    getattr(cfg, "step_budget_policy", "adaptive") or "adaptive"
                ),
                "max_steps": _format_integer(getattr(cfg, "max_steps", 25)),
                "task_max_steps": _format_integer(getattr(cfg, "task_max_steps", 100)),
                "subagent_max_steps": _format_integer(getattr(cfg, "subagent_max_steps", 16)),
            },
            thinking_label=thinking_label,
            role_models=role_models,
            forge_role_models=forge_role_models,
            role_temperatures=role_temperatures,
            profiles=profiles,
            active_profile=active_profile,
            api_key_source=api_key_source,
            masked_api_key=masked_api_key,
            config_warning="; ".join(warnings) or None,
        )
        state._original = state.snapshot()
        return state

    @property
    def dirty(self) -> bool:
        return self.snapshot() != self._original

    def snapshot(self) -> dict[str, Any]:
        return {
            "fields": {key: str(value) for key, value in sorted(self.fields.items())},
            "thinking_label": self.thinking_label,
            "role_models": {role: str(self.role_models.get(role, "")) for role in ROLE_ORDER},
            "forge_role_models": {
                role: str(self.forge_role_models.get(role, "")) for role in ROLE_ORDER
            },
            "role_temperatures": {
                role: str(self.role_temperatures.get(role, ""))
                for role in ROLE_ORDER
                if role in _ROLE_TEMPERATURE_FIELDS
            },
            "profiles": {name: dict(data) for name, data in sorted(self.profiles.items())},
            "active_profile": self.active_profile,
            "new_api_key": self.new_api_key,
            "clear_stored_key_confirmed": self.clear_stored_key_confirmed,
            "clear_stored_key_profile": self.clear_stored_key_profile,
            "thinking_label_explicitly_set": self.thinking_label_explicitly_set,
        }

    def reset(self) -> None:
        original_fields = self._original.get("fields")
        if isinstance(original_fields, dict):
            self.fields = {str(key): str(value) for key, value in original_fields.items()}
        self.thinking_label = str(self._original.get("thinking_label") or "auto")
        self.role_models = _role_text_map_from_snapshot(self._original.get("role_models"))
        self.forge_role_models = _role_text_map_from_snapshot(
            self._original.get("forge_role_models")
        )
        self.role_temperatures = _role_text_map_from_snapshot(
            self._original.get("role_temperatures")
        )
        profiles = self._original.get("profiles")
        self.profiles = {
            str(name): dict(data)
            for name, data in (profiles.items() if isinstance(profiles, dict) else ())
            if isinstance(data, dict)
        }
        self.active_profile = str(self._original.get("active_profile") or "")
        self.new_api_key = str(self._original.get("new_api_key") or "")
        self.clear_stored_key_confirmed = bool(
            self._original.get("clear_stored_key_confirmed", False)
        )
        raw_clear_profile = self._original.get("clear_stored_key_profile")
        self.clear_stored_key_profile = str(raw_clear_profile) if raw_clear_profile else None
        self.thinking_label_explicitly_set = bool(
            self._original.get("thinking_label_explicitly_set", False)
        )
        self.refresh_api_key_status()

    def set_field(self, name: str, value: str) -> None:
        key = str(name).strip()
        if key == "new_api_key":
            self.new_api_key = str(value)
            self.clear_stored_key_confirmed = False
            self.clear_stored_key_profile = None
            return
        if key not in {
            "model",
            "base_url",
            "llm_timeout_s",
            "routing_mode",
            "step_budget_policy",
            "max_steps",
            "task_max_steps",
            "subagent_max_steps",
        }:
            raise KeyError(f"Unknown config menu field: {name}")
        self.fields[key] = str(value)
        if key == "base_url":
            self.refresh_api_key_status()

    def set_thinking_label(self, label: str) -> None:
        normalized = _normalize_thinking_label(label)
        self.thinking_label = normalized
        self.thinking_label_explicitly_set = True

    def set_routing_mode(self, value: str) -> None:
        self.fields["routing_mode"] = _normalize_routing_mode(value)

    def set_step_budget_policy(self, value: str) -> None:
        self.fields["step_budget_policy"] = _normalize_step_budget_policy(value)

    def set_max_steps(self, value: str) -> None:
        self.fields["max_steps"] = str(value)

    def set_task_max_steps(self, value: str) -> None:
        self.fields["task_max_steps"] = str(value)

    def set_subagent_max_steps(self, value: str) -> None:
        self.fields["subagent_max_steps"] = str(value)

    def set_role_model(self, role: str, model: str) -> None:
        role_key = _normalize_role(role)
        self.role_models[role_key] = str(model)

    def set_forge_role_model(self, role: str, model: str) -> None:
        role_key = _normalize_role(role)
        self.forge_role_models[role_key] = str(model)

    def set_role_temperature(self, role: str, value: str) -> None:
        role_key = _normalize_role(role)
        if role_key not in _ROLE_TEMPERATURE_FIELDS:
            raise KeyError(f"Role has no temperature setting: {role}")
        self.role_temperatures[role_key] = str(value)

    def mark_clear_stored_key_confirmed(self) -> None:
        self.clear_stored_key_confirmed = True
        self.clear_stored_key_profile = self.active_profile or None
        self.new_api_key = ""

    def set_active_profile_name(self, name: str) -> None:
        profile_name = str(name or "").strip().lower()
        if profile_name not in self.profiles:
            raise KeyError(f"Unknown profile: {name}")
        self.active_profile = profile_name
        profile = ProfileSpec.from_dict(profile_name, self.profiles[profile_name])
        if profile.base_url:
            self.fields["base_url"] = profile.base_url
        if profile.default_model:
            self.fields["model"] = profile.default_model
        self.refresh_api_key_status()

    def add_profile_spec(self, profile: ProfileSpec, *, make_active: bool = True) -> None:
        self.profiles[profile.name] = profile.to_dict()
        if make_active:
            self.set_active_profile_name(profile.name)

    def update_active_profile_spec(self, **fields: Any) -> None:
        if self.active_profile not in self.profiles:
            raise KeyError("No active profile configured.")
        profile = ProfileSpec.from_dict(self.active_profile, self.profiles[self.active_profile])
        values = {
            "name": profile.name,
            "protocol": profile.protocol,
            "base_url": profile.base_url,
            "api_key_env": profile.api_key_env,
            "extra_headers": dict(profile.extra_headers),
            "default_model": profile.default_model,
            "web_search_adapter": profile.web_search_adapter,
            "web_search_model": profile.web_search_model,
            "notes": profile.notes,
        }
        values.update(fields)
        self.profiles[self.active_profile] = ProfileSpec(**values).to_dict()
        self.set_active_profile_name(self.active_profile)

    def remove_profile_name(self, name: str) -> None:
        profile_name = str(name or "").strip().lower()
        if profile_name not in self.profiles:
            raise KeyError(f"Unknown profile: {name}")
        self.profiles.pop(profile_name, None)
        if self.active_profile == profile_name:
            next_profile = sorted(self.profiles)[0] if self.profiles else ""
            if next_profile:
                self.set_active_profile_name(next_profile)
            else:
                self.active_profile = ""
                self.refresh_api_key_status()

    def _resolution_cfg(self) -> AppConfig:
        cfg = AppConfig(
            model=str(self.fields.get("model", "") or ""),
            base_url=str(self.fields.get("base_url", "") or ""),
        )
        cfg.extra_fields = {"profiles": dict(self.profiles)}
        if self.active_profile:
            cfg.extra_fields["active_profile"] = self.active_profile
        return cfg

    def refresh_api_key_status(self) -> None:
        try:
            resolved_key = resolve_api_key(self._resolution_cfg())
        except ConfigError:
            self.api_key_source = "error"
            self.masked_api_key = "(not set)"
            return
        self.api_key_source = resolved_key.source
        self.masked_api_key = mask_api_key(resolved_key.key)

    def commit_to(self, cfg: AppConfig) -> ConfigMenuResult:
        validation_error = self.validate()
        if validation_error is not None:
            return ConfigMenuResult(
                saved=False,
                changes={},
                api_key_changed=False,
                error=validation_error,
            )

        changes: dict[str, Any] = {}

        original_profiles = self._original.get("profiles")
        if self.profiles != (original_profiles if isinstance(original_profiles, dict) else {}):
            cfg.extra_fields["profiles"] = dict(sorted(self.profiles.items()))
            changes["profiles"] = sorted(self.profiles)

        original_active_profile = str(self._original.get("active_profile") or "")
        if self.active_profile != original_active_profile:
            if self.active_profile:
                cfg.extra_fields["active_profile"] = self.active_profile
                set_active_profile(cfg, self.active_profile)
            else:
                cfg.extra_fields.pop("active_profile", None)
            changes["active_profile"] = self.active_profile

        desired_base_url = str(self.fields.get("base_url", ""))
        if str(getattr(cfg, "base_url", "") or "") != desired_base_url:
            set_config_value(cfg, "base_url", desired_base_url)
            changes["base_url"] = desired_base_url

        desired_model = str(self.fields.get("model", ""))
        if str(getattr(cfg, "model", "") or "") != desired_model:
            set_config_value(cfg, "model", desired_model)
            changes["model"] = desired_model

        desired_timeout = float(str(self.fields.get("llm_timeout_s", "")).strip())
        current_timeout = _finite_float(getattr(cfg, "llm_timeout_s", None), fallback=None)
        if current_timeout != desired_timeout:
            set_config_value(cfg, "llm_timeout_s", _format_number(desired_timeout))
            changes["llm_timeout_s"] = desired_timeout

        desired_routing_mode = _normalize_routing_mode(self.fields.get("routing_mode", "auto"))
        if str(getattr(cfg, "routing_mode", "") or "") != desired_routing_mode:
            set_config_value(cfg, "routing_mode", desired_routing_mode)
            changes["routing_mode"] = desired_routing_mode

        desired_step_budget_policy = _normalize_step_budget_policy(
            self.fields.get("step_budget_policy", "adaptive")
        )
        if str(getattr(cfg, "step_budget_policy", "") or "") != desired_step_budget_policy:
            set_config_value(cfg, "step_budget_policy", desired_step_budget_policy)
            changes["step_budget_policy"] = desired_step_budget_policy

        for key in ("max_steps", "task_max_steps", "subagent_max_steps"):
            desired_int = int(str(self.fields.get(key, "")).strip())
            current_int = _finite_int(getattr(cfg, key, None), fallback=None)
            if current_int != desired_int:
                set_config_value(cfg, key, str(desired_int))
                changes[key] = desired_int

        desired_thinking_value = thinking_label_to_config_value(self.thinking_label)
        desired_reasoning_effort = thinking_label_to_reasoning_effort(self.thinking_label)
        original_thinking_label = str(self._original.get("thinking_label") or "auto")
        should_write_thinking = (
            self.thinking_label_explicitly_set or self.thinking_label != original_thinking_label
        )
        if should_write_thinking:
            if (
                getattr(cfg, "llm_enable_thinking", None) != desired_thinking_value
                or getattr(cfg, "llm_reasoning_effort", None) != desired_reasoning_effort
            ):
                set_config_value(
                    cfg,
                    "llm_enable_thinking",
                    _thinking_config_text(desired_thinking_value),
                )
                set_config_value(cfg, "llm_reasoning_effort", desired_reasoning_effort or "auto")
                changes["llm_enable_thinking"] = desired_thinking_value
                changes["llm_reasoning_effort"] = desired_reasoning_effort
            _set_thinking_label_hint(cfg, self.thinking_label)

        role_models = _non_empty_role_values(self.role_models)
        current_role_models = _non_empty_role_values(
            _normalized_role_model_values(cfg.extra_fields.get("role_models"))
        )
        if current_role_models != role_models:
            if role_models:
                cfg.extra_fields["role_models"] = dict(role_models)
            else:
                cfg.extra_fields.pop("role_models", None)
            changes["role_models"] = dict(role_models)

        forge_role_models = _non_empty_role_values(self.forge_role_models)
        current_forge_role_models = _non_empty_role_values(
            _normalized_role_model_values(cfg.extra_fields.get("forge_role_models"))
        )
        if current_forge_role_models != forge_role_models:
            if forge_role_models:
                cfg.extra_fields["forge_role_models"] = dict(forge_role_models)
            else:
                cfg.extra_fields.pop("forge_role_models", None)
            changes["forge_role_models"] = dict(forge_role_models)

        default_temperature = _finite_float(getattr(cfg, "temperature", 0.2), fallback=0.2)
        for role in ROLE_ORDER:
            field_name = _ROLE_TEMPERATURE_FIELDS.get(role)
            if field_name is None:
                continue
            raw_value = str(self.role_temperatures.get(role, "")).strip()
            current_value = _finite_float(getattr(cfg, field_name, None), fallback=None)
            if raw_value:
                desired_temperature = float(raw_value)
            else:
                desired_temperature = default_temperature
            if current_value != desired_temperature:
                set_config_value(cfg, field_name, _format_number(desired_temperature))
                changes[field_name] = desired_temperature

        return ConfigMenuResult(
            saved=True,
            changes=changes,
            api_key_changed=bool(self.new_api_key.strip() or self.clear_stored_key_confirmed),
        )

    def validate(self) -> str | None:
        timeout_text = str(self.fields.get("llm_timeout_s", "")).strip()
        try:
            timeout = float(timeout_text)
        except ValueError:
            return "Request timeout (seconds) must be a positive number."
        if timeout <= 0 or not math.isfinite(timeout):
            return "Request timeout (seconds) must be a positive number."
        base_url_text = str(self.fields.get("base_url", "")).strip()
        try:
            validate_base_url(base_url_text, key="Base URL", allow_empty=True)
        except ConfigError as exc:
            return str(exc)
        try:
            _normalize_thinking_label(self.thinking_label)
        except ValueError as exc:
            return str(exc)
        for key in ("routing_mode", "step_budget_policy"):
            try:
                if key == "routing_mode":
                    _normalize_routing_mode(self.fields.get(key, "auto"))
                else:
                    _normalize_step_budget_policy(self.fields.get(key, "adaptive"))
            except ValueError as exc:
                return str(exc)
        for key in ("max_steps", "task_max_steps", "subagent_max_steps"):
            text = str(self.fields.get(key, "")).strip()
            label = _FIELD_LABELS.get(key, key)
            try:
                value = int(text)
            except ValueError:
                return f"{label} must be a positive integer."
            if value <= 0:
                return f"{label} must be a positive integer."
        for role, raw_value in self.role_temperatures.items():
            text = str(raw_value).strip()
            if not text:
                continue
            label = f"{_role_label(role)} temperature"
            try:
                temperature = float(text)
            except ValueError:
                return f"{label} must be a non-negative number."
            if temperature < 0 or not math.isfinite(temperature):
                return f"{label} must be a non-negative number."
        return None


def run_config_menu(
    *,
    cfg: AppConfig | None = None,
    auto_focus: str | None = None,
) -> ConfigMenuResult:
    """Open the inline configuration menu and persist changes on save."""
    effective_cfg = cfg or load_config()
    state = ConfigMenuState.from_cfg(effective_cfg)
    console = _resolve_console()

    if auto_focus == "api_key":
        _run_api_key_section(state, console)
    elif auto_focus == "model":
        _run_default_section(state, console)

    while True:
        action = _prompt_main_action(console, state)
        if action == "profile":
            _run_provider_section(state, console)
        elif action == "api_key":
            _run_api_key_section(state, console)
        elif action == "default":
            _run_default_section(state, console)
        elif action == "router":
            _run_router_section(state, console)
        elif action == "subagents":
            _run_subagent_section(state, console)
        elif action == "forge":
            _run_forge_section(state, console)
        elif action == "save":
            result = _save_and_exit(state, effective_cfg, console)
            if result.saved or result.error is None:
                return result
        elif action == "cancel" and _confirm_cancel_when_dirty(state, console):
            return ConfigMenuResult(saved=False, changes={}, api_key_changed=False)


def _resolve_console() -> Console:
    return make_console()


def _profile_summary_text(state: ConfigMenuState) -> str:
    if state.active_profile:
        return f"{state.active_profile} (active)"
    if state.profiles:
        return f"{len(state.profiles)} profiles, none active"
    return _MISSING_REQUIRED


def _api_key_summary_text(state: ConfigMenuState) -> str:
    if state.clear_stored_key_confirmed:
        return _CLEAR_KEY_PENDING
    pending_key = state.new_api_key.strip()
    if pending_key:
        return f"set ({mask_api_key(pending_key)})"
    if state.masked_api_key != "(not set)":
        return f"set ({state.masked_api_key})"
    return _MISSING_REQUIRED


def _default_model_summary_text(state: ConfigMenuState) -> str:
    model = state.fields.get("model", "").strip()
    if not model:
        return _MISSING_REQUIRED
    return f"{model} · thinking {state.thinking_label}"


def _limits_summary_text(state: ConfigMenuState) -> str:
    return (
        f"steps {state.fields['max_steps']}/{state.fields['task_max_steps']}/"
        f"{state.fields['subagent_max_steps']} · "
        f"routing {state.fields['routing_mode']} · "
        f"budget {state.fields['step_budget_policy']}"
    )


def _override_summary_text(values: dict[str, str]) -> str:
    count = sum(1 for value in values.values() if str(value).strip())
    if count <= 0:
        return "none"
    return _pluralize(count, "override")


def _top_level_menu_rows(state: ConfigMenuState) -> list[tuple[str, str, str]]:
    subagent_values = {
        **state.role_models,
        **{f"{role}_temperature": value for role, value in state.role_temperatures.items()},
    }
    return [
        ("profile", "Provider Profile", _profile_summary_text(state)),
        ("api_key", "API Key", _api_key_summary_text(state)),
        ("default", "Default Model", _default_model_summary_text(state)),
        ("router", "Execution Limits", _limits_summary_text(state)),
        ("subagents", "Subagent model overrides", _override_summary_text(subagent_values)),
        (
            "forge",
            "Forge model overrides",
            _override_summary_text(state.forge_role_models),
        ),
    ]


def _print_section_cancelled(console: Console, section_name: str) -> None:
    console.print(f'[dim]Section "{section_name}" cancelled.[/dim]')


def _summary_is_markup(summary: str) -> bool:
    return summary in _WARNING_SUMMARIES


def _pluralize(count: int, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {word}"


def _role_label(role: str) -> str:
    return str(role).replace("_", " ").capitalize()


def _role_description(role: str) -> str:
    return _ROLE_DESCRIPTIONS.get(str(role), "role-specific work")


def _print_role_explainer(console: Console) -> None:
    from rich.table import Table

    console.print("[dim]Roles:[/dim]")
    table = Table(show_header=False, box=None, padding=(0, 2), collapse_padding=True)
    table.add_column("role", no_wrap=True, style="dim")
    table.add_column("description", no_wrap=False, style="dim")
    for role in ROLE_ORDER:
        table.add_row(role, _role_description(role))
    console.print(table)


def _config_picker_hint(*, row_count: int, allow_save: bool = False) -> str:
    jump_limit = min(max(row_count, 1), 9)
    base = f"↑/↓ navigate  Enter select  1-{jump_limit} jump"
    if allow_save:
        return f"{base}  s save  q exit"
    return f"{base}  Esc back"


def _build_config_picker_panel(
    *,
    title: str,
    subtitle: str,
    rows: list[tuple[str, str, str]],
    selected_value: str | None,
    interactive: bool,
    footer_hint: str | None = None,
) -> Panel:
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    selected = str(selected_value or "").strip().casefold()
    table = Table(show_header=False, box=None, expand=True, padding=(0, 1), collapse_padding=True)
    table.add_column("option", no_wrap=True, ratio=3)
    table.add_column("description", no_wrap=False, ratio=5, overflow="fold")

    for index, (value, label, description) in enumerate(rows, start=1):
        row_selected = str(value).strip().casefold() == selected
        marker = "> " if row_selected else "  "
        option_style = "bold cyan" if row_selected else STYLE_CONTENT
        desc_style = STYLE_CONTENT if row_selected else STYLE_DIM
        table.add_row(
            Text(f"{marker}{index}) {label}", style=option_style),
            Text(str(description or ""), style=desc_style),
        )

    renderables: list[Any] = []
    if str(subtitle or "").strip():
        renderables.append(Text(str(subtitle), style="dim"))
        renderables.append(Text(""))
    renderables.append(table)
    renderables.append(Text(""))
    renderables.append(
        Text(
            footer_hint or _config_picker_hint(row_count=len(rows)),
            style="dim",
        )
    )
    return Panel(
        Group(*renderables),
        title=title,
        border_style="cyan" if interactive else "dim",
    )


def _build_config_top_level_panel(
    *,
    state: ConfigMenuState,
    selected_value: str | None,
    interactive: bool,
    unknown_key_message: str | None = None,
) -> Panel:
    from rich.console import Group
    from rich.table import Table
    from rich.text import Text

    rows = _top_level_menu_rows(state)
    selected = str(selected_value or "").strip().casefold()
    pending_changes = _pending_change_count(state)

    table = Table(show_header=False, box=None, expand=True, padding=(0, 1), collapse_padding=True)
    table.add_column("option", no_wrap=True, ratio=3)
    table.add_column("summary", no_wrap=False, ratio=5, overflow="fold")
    for index, (value, label, summary) in enumerate(rows, start=1):
        row_selected = str(value).strip().casefold() == selected
        marker = "> " if row_selected else "  "
        option_style = "bold cyan" if row_selected else STYLE_CONTENT
        summary_style = STYLE_CONTENT if row_selected else STYLE_DIM
        summary_text = (
            Text.from_markup(summary)
            if _summary_is_markup(summary)
            else Text(summary, style=summary_style)
        )
        table.add_row(
            Text(f"{marker}{index}) {label}", style=option_style),
            summary_text,
        )

    renderables: list[Any] = []
    if pending_changes > 0:
        renderables.append(Text(f"Pending changes: {pending_changes}", style="yellow"))
    else:
        renderables.append(Text("No pending changes.", style="dim"))
    if state.config_warning:
        renderables.append(Text(str(state.config_warning), style="yellow"))
    renderables.append(Text(""))
    renderables.append(table)
    renderables.append(Text(""))
    renderables.append(Text(_config_picker_hint(row_count=len(rows), allow_save=True), style="dim"))
    if unknown_key_message:
        renderables.append(Text(f"Unknown key: {unknown_key_message}", style="red"))
    return Panel(
        Group(*renderables),
        title="Sylliptor Configuration",
        border_style="cyan" if interactive else "dim",
    )


def _prompt_inline_choice_fallback(
    *,
    console: Console,
    title: str,
    text: str,
    choices: list[tuple[str, str]],
    default: str | None = None,
    prompt_label: str | None = None,
    fallback_reason: str | None = None,
) -> str | None:
    if not choices:
        return None
    error_message: str | None = None
    while True:
        console.print()
        console.print(f"[bold]{title}[/bold]")
        if fallback_reason:
            console.print(f"[dim]{fallback_reason}[/dim]")
            fallback_reason = None
        if error_message:
            console.print(f"[red]{error_message}[/red]")
        if text:
            console.print(f"[dim]{text}[/dim]")
        console.print()
        for index, (value, label) in enumerate(choices, start=1):
            suffix = " [dim](current)[/dim]" if default and value == default else ""
            console.print(f"  {index}) {label}{suffix}")
        footer = prompt_label or f"[1-{len(choices)}] choose  [Enter/q] back"
        console.print()
        console.print(f"[dim]{footer}[/dim]")
        try:
            raw = str(typer.prompt("Choice", default="", show_default=False)).strip()
        except (Abort, EOFError, KeyboardInterrupt):
            console.print("")
            return None
        if not raw:
            return None
        normalized = raw.casefold()
        if normalized in {"q", "quit", "cancel", "c"}:
            return None
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(choices):
                return choices[index][0]
        error_message = "Unknown choice."


def _prompt_inline_choice(
    *,
    console: Console,
    title: str,
    text: str,
    choices: list[tuple[str, str]],
    default: str | None = None,
    prompt_label: str | None = None,
) -> str | None:
    return _prompt_inline_choice_fallback(
        console=console,
        title=title,
        text=text,
        choices=choices,
        default=default,
        prompt_label=prompt_label,
    )


def _prompt_main_action_fallback(console: Console, state: ConfigMenuState) -> str:
    error_message: str | None = None
    while True:
        rows = _top_level_menu_rows(state)
        pending_changes = _pending_change_count(state)
        console.print()
        if error_message:
            console.print(f"[red]{error_message}[/red]")
        console.rule("[bold]Sylliptor Configuration[/bold]")
        if pending_changes > 0:
            console.print(f"[yellow]Pending changes: {pending_changes}[/yellow]")
        else:
            console.print("[dim]No pending changes.[/dim]")
        if state.config_warning:
            console.print(f"[yellow]{escape(str(state.config_warning))}[/yellow]")
        console.print()
        for index, (_value, label, summary) in enumerate(rows, start=1):
            summary_text = (
                summary if _summary_is_markup(summary) else f"[dim]{escape(summary)}[/dim]"
            )
            console.print(f"  {index}) [bold]{label}[/bold]  {summary_text}")
        console.print()
        if state.dirty:
            console.print("[dim][1-6] edit  [s] save  [q] cancel  [Enter] save & exit[/dim]")
        else:
            console.print("[dim][1-6] edit  [s] save  [q] cancel  [Enter] cancel[/dim]")
        try:
            raw = str(typer.prompt("Choice", default="", show_default=False)).strip()
        except (Abort, EOFError, KeyboardInterrupt):
            console.print("")
            return "cancel"
        if not raw:
            return "save" if state.dirty else "cancel"
        normalized = raw.casefold()
        if normalized in {"s", "save"}:
            return "save"
        if normalized in {"q", "quit", "cancel", "c"}:
            return "cancel"
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(rows):
                return rows[index][0]
        error_message = "Unknown choice."


def _try_run_config_live_menu(
    *,
    console: Console,
    rows: list[tuple[str, str, str]],
    current_value: str,
    panel_builder: Callable[[str | None, bool], Panel],
    unknown_key_panel_builder: Callable[[str | None, bool, str | None], Panel] | None = None,
    confirm_on_digit: bool = False,
    command_hotkeys: dict[str, str] | None = None,
    accept_right: bool = False,
    too_small_result: str | None = None,
) -> tuple[str | None, bool, str | None]:
    try:
        from ..cli import (
            _clear_terminal_screen,
            _is_non_interactive_terminal,
            _read_input_keys_with_timeout,
            _terminal_too_small,
            _terminal_too_small_panel,
            _watch_terminal_resize,
        )
    except Exception as exc:
        return None, False, str(exc)

    if _is_non_interactive_terminal():
        return None, False, "non-interactive terminal"

    try:
        from prompt_toolkit.input import create_input
        from prompt_toolkit.keys import Keys
        from rich.live import Live
    except Exception as exc:
        return None, False, str(exc)

    if not rows:
        return None, True, None

    if _terminal_too_small():
        console.print(_terminal_too_small_panel())
        return too_small_result, True, None

    selected_index = next(
        (idx for idx, (value, _label, _desc) in enumerate(rows) if value == current_value),
        0,
    )
    if selected_index < 0 or selected_index >= len(rows):
        selected_index = 0

    input_reader: Any | None = None
    try:
        input_reader = create_input()
    except Exception as exc:
        return None, False, str(exc)

    def _selected_value() -> str:
        return rows[selected_index][0]

    last_unknown_key: str | None = None

    def _panel() -> Panel:
        if unknown_key_panel_builder is not None:
            return unknown_key_panel_builder(_selected_value(), True, last_unknown_key)
        return panel_builder(_selected_value(), True)

    def _hotkey_for(key: Any, raw_data: Any) -> str | None:
        if isinstance(raw_data, str) and len(raw_data) == 1 and raw_data.isprintable():
            return raw_data.lower()
        if isinstance(key, str) and len(key) == 1 and key.isprintable():
            return key.lower()
        return None

    def _is_escape(key: Any) -> bool:
        return bool(key == Keys.Escape or key == Keys.ControlC or key in {"escape", "c-c"})

    try:
        with _watch_terminal_resize() as consume_resize:
            with Live(
                _panel(),
                console=console,
                auto_refresh=False,
                transient=True,
                screen=False,
            ) as live:
                with input_reader.raw_mode():
                    while True:
                        if _terminal_too_small():
                            console.print(_terminal_too_small_panel())
                            return too_small_result, True, None

                        resized = consume_resize()
                        if resized:
                            _clear_terminal_screen(console=console)
                            live.update(_panel(), refresh=True)

                        key_presses = _read_input_keys_with_timeout(
                            input_reader=input_reader,
                            timeout_s=0.12,
                        )
                        if not key_presses:
                            continue

                        panel_updated = False
                        for key_press in key_presses:
                            key = key_press.key
                            raw_data = key_press.data

                            if key == Keys.Up or key == "up":
                                last_unknown_key = None
                                selected_index = (selected_index - 1) % len(rows)
                                panel_updated = True
                                continue
                            if key == Keys.Down or key == "down":
                                last_unknown_key = None
                                selected_index = (selected_index + 1) % len(rows)
                                panel_updated = True
                                continue

                            digit_key: str | None = None
                            if isinstance(key, str) and key.isdigit():
                                digit_key = key
                            elif isinstance(raw_data, str) and raw_data.isdigit():
                                digit_key = raw_data
                            if digit_key is not None:
                                last_unknown_key = None
                                idx = int(digit_key) - 1
                                if 0 <= idx < len(rows):
                                    selected_index = idx
                                    if confirm_on_digit:
                                        return _selected_value(), True, None
                                    panel_updated = True
                                continue

                            if accept_right and (key == Keys.Right or key == "right"):
                                return _selected_value(), True, None

                            if (
                                key == Keys.Enter
                                or key == Keys.ControlM
                                or raw_data in {chr(13), chr(10)}
                            ):
                                return _selected_value(), True, None

                            if _is_escape(key):
                                if command_hotkeys and "escape" in command_hotkeys:
                                    return command_hotkeys["escape"], True, None
                                return None, True, None

                            hotkey = _hotkey_for(key, raw_data)
                            if hotkey and command_hotkeys and hotkey in command_hotkeys:
                                return command_hotkeys[hotkey], True, None
                            if (
                                hotkey
                                and not hotkey.isdigit()
                                and unknown_key_panel_builder is not None
                            ):
                                last_unknown_key = hotkey
                                panel_updated = True

                        if panel_updated:
                            live.update(_panel(), refresh=True)
    except Exception as exc:
        return None, False, str(exc)
    finally:
        if input_reader is not None:
            close = getattr(input_reader, "close", None)
            if callable(close):
                close()

    return None, True, None


def _run_config_picker(
    *,
    console: Console,
    title: str,
    subtitle: str,
    rows: list[tuple[str, str, str]],
    current_value: str,
    footer_hint: str | None = None,
) -> str | None:
    selected_value, interactive_available, reason = _try_run_config_live_menu(
        console=console,
        rows=rows,
        current_value=current_value,
        panel_builder=lambda selected, interactive: _build_config_picker_panel(
            title=title,
            subtitle=subtitle,
            rows=rows,
            selected_value=selected,
            interactive=interactive,
            footer_hint=footer_hint,
        ),
        confirm_on_digit=True,
        too_small_result=None,
    )
    if interactive_available:
        return selected_value

    detail = reason or "unknown reason"
    console.print(f"[dim]Interactive picker unavailable: {detail}. Using numeric input.[/dim]")
    return _prompt_inline_choice(
        console=console,
        title=title,
        text=subtitle,
        choices=[(value, label) for value, label, _description in rows],
        default=current_value,
        prompt_label=footer_hint,
    )


def _run_config_top_level(*, state: ConfigMenuState, console: Console) -> str:
    rows = _top_level_menu_rows(state)
    current_value = rows[0][0] if rows else "cancel"
    selected_value, interactive_available, reason = _try_run_config_live_menu(
        console=console,
        rows=rows,
        current_value=current_value,
        panel_builder=lambda selected, interactive: _build_config_top_level_panel(
            state=state,
            selected_value=selected,
            interactive=interactive,
        ),
        unknown_key_panel_builder=lambda selected, interactive, unknown_key: (
            _build_config_top_level_panel(
                state=state,
                selected_value=selected,
                interactive=interactive,
                unknown_key_message=unknown_key,
            )
        ),
        confirm_on_digit=True,
        command_hotkeys={
            "s": "save",
            "q": "cancel",
            "escape": "cancel",
        },
        accept_right=True,
        too_small_result="cancel",
    )
    if interactive_available and selected_value is not None:
        return selected_value
    if interactive_available:
        return "cancel"

    detail = reason or "unknown reason"
    console.print(f"[dim]Interactive picker unavailable: {detail}. Using numeric input.[/dim]")
    return _prompt_main_action_fallback(console, state)


def _prompt_main_action(console: Console, state: ConfigMenuState) -> str:
    return _run_config_top_level(state=state, console=console)


def _run_provider_section(state: ConfigMenuState, console: Console) -> None:
    console.print()
    console.rule("[bold]Provider Profile[/bold]")
    console.print(
        "[dim]A profile bundles provider URL, headers, and the API key env variable. "
        "The active profile is what Sylliptor uses for all model calls.[/dim]"
    )
    for profile_name in sorted(state.profiles):
        marker = "active" if profile_name == state.active_profile else ""
        profile = ProfileSpec.from_dict(profile_name, state.profiles[profile_name])
        console.print(f"{profile.name}: {profile.base_url} {marker}".rstrip())
    try:
        action = _run_config_picker(
            console=console,
            title="Provider Profile",
            subtitle=f"Active profile: {state.active_profile or '(none)'}",
            rows=[
                ("switch", "Switch active profile", "Choose another configured profile."),
                ("add_preset", "Add from preset", "Pick a known provider (OpenAI, Anthropic, ...)"),
                ("add_custom", "Add custom", "Use any API base URL"),
                ("edit", "Edit current", "Change URL, key env, default model, headers, notes"),
                ("remove", "Remove", "Delete a profile (applied on save)"),
                ("back", "Back", "Return to the menu"),
            ],
            current_value="switch" if state.profiles else "add_preset",
        )
        if action in {None, "back"}:
            return
        if action == "switch":
            _run_profile_switch(state, console)
        elif action == "add_preset":
            _run_profile_add_preset(state, console)
        elif action == "add_custom":
            _run_profile_add_custom(state, console)
        elif action == "edit":
            _run_profile_edit_current(state, console)
        elif action == "remove":
            _run_profile_remove(state, console)
    except (Abort, EOFError, KeyboardInterrupt):
        console.print("")
        _print_section_cancelled(console, "Provider Profile")
        return
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]")
        return


def _run_profile_switch(state: ConfigMenuState, console: Console) -> None:
    if not state.profiles:
        return
    rows = [(name, name, "") for name in sorted(state.profiles)]
    selected = _run_config_picker(
        console=console,
        title="Provider Profile",
        subtitle=f"Active: {state.active_profile or 'none'}",
        rows=rows,
        current_value=state.active_profile or rows[0][0],
    )
    if selected:
        state.set_active_profile_name(selected)


def _run_profile_add_preset(state: ConfigMenuState, console: Console) -> None:
    rows = [(preset.key, preset.label, _preset_description(preset)) for preset in PROFILE_PRESETS]
    selected = _run_config_picker(
        console=console,
        title="Provider Profile",
        subtitle="Pick a provider preset",
        rows=rows,
        current_value="openai",
    )
    if not selected:
        return
    preset = next(preset for preset in PROFILE_PRESETS if preset.key == selected)
    profile_name = _prompt_text("Profile name", preset.key)
    profile = make_profile_from_preset(preset, name=profile_name)
    if not profile.base_url:
        base_url = _prompt_text("Base URL", "")
        profile = ProfileSpec(
            name=profile.name,
            protocol=profile.protocol,
            base_url=base_url,
            api_key_env=profile.api_key_env,
            extra_headers=profile.extra_headers,
            default_model=profile.default_model,
            web_search_adapter=profile.web_search_adapter,
            web_search_model=profile.web_search_model,
            notes=profile.notes,
        )
    state.add_profile_spec(profile)
    console.print(f"[green]Profile {profile.name} added.[/green]")
    _print_preset_warning(console, preset)


def _preset_description(preset: ProfilePreset) -> str:
    if preset.notes:
        return preset.notes
    parsed = urlparse(preset.base_url)
    if parsed.netloc:
        return parsed.netloc
    return "Use any API base URL"


def _print_preset_warning(console: Console, preset: ProfilePreset) -> None:
    warning = str(preset.setup_warning or "").strip()
    if warning:
        console.print(f"[yellow]{warning}[/yellow]")


def _run_profile_add_custom(state: ConfigMenuState, console: Console) -> None:
    name = _prompt_text("Profile name", "custom")
    base_url = _prompt_text("Base URL", "")
    headers = _parse_header_text(_prompt_text("Extra headers (k=v, comma-separated)", ""))
    profile = ProfileSpec(
        name=name.lower(),
        protocol="openai_compat",
        base_url=base_url,
        extra_headers=headers,
        web_search_adapter="auto",
        web_search_model="",
        notes="Custom model API endpoint.",
    )
    state.add_profile_spec(profile)
    console.print(f"[green]Profile {profile.name} added.[/green]")


def _prompt_web_search_adapter(console: Console, current: str) -> str:
    for _attempt in range(3):
        value = _prompt_non_secret_text(
            console,
            "Web search adapter",
            current,
            "Web search adapter",
        )
        try:
            return normalize_web_search_adapter(value or "auto")
        except (ConfigError, ValueError) as e:
            allowed = ", ".join(WEB_SEARCH_ADAPTER_CHOICES)
            console.print(f"[red]{e}[/red]")
            console.print(f"[dim]Allowed adapters: {allowed}[/dim]")
    return normalize_web_search_adapter(current or "auto")


def _run_profile_edit_current(state: ConfigMenuState, console: Console) -> None:
    if not state.active_profile or state.active_profile not in state.profiles:
        console.print(
            "[yellow]No active profile selected. Switch to one first or add a new profile.[/yellow]"
        )
        return
    profile = ProfileSpec.from_dict(state.active_profile, state.profiles[state.active_profile])
    console.print(_build_profile_edit_current_panel(profile.name))
    console.print(
        "[dim]Note: the API key itself is stored separately (encrypted in keyring). "
        "This field is only the name of the env variable Sylliptor reads as a fallback.[/dim]"
    )
    base_url = _prompt_non_secret_text(console, "Base URL", profile.base_url, "Base URL")
    api_key_env = _prompt_non_secret_text(
        console, _API_KEY_ENV_PROMPT, profile.api_key_env or "", _API_KEY_ENV_FIELD_NAME
    )
    default_model = _prompt_non_secret_text(
        console, "Default model", profile.default_model, "Default model"
    )
    web_search_adapter = _prompt_web_search_adapter(console, profile.web_search_adapter)
    web_search_model = _prompt_non_secret_text(
        console,
        "Web search model",
        profile.web_search_model,
        "Web search model",
    )
    headers_default = ", ".join(f"{key}={value}" for key, value in profile.extra_headers.items())
    headers = _parse_header_text(
        _prompt_non_secret_text(console, "Extra headers", headers_default, "Extra headers")
    )
    notes = _prompt_non_secret_text(console, "Notes", profile.notes, "Notes")
    state.update_active_profile_spec(
        base_url=base_url,
        api_key_env=api_key_env or None,
        default_model=default_model,
        web_search_adapter=web_search_adapter or "auto",
        web_search_model=web_search_model,
        extra_headers=headers,
        notes=notes,
    )


def _build_profile_edit_current_panel(profile_name: str) -> Panel:
    from rich.console import Group
    from rich.text import Text

    return Panel(
        Group(
            Text("Editable here:", style=STYLE_EMPHASIS),
            Text("  Base URL", style="dim"),
            Text("  API key env var NAME", style="dim"),
            Text("  Default model", style="dim"),
            Text("  Web search adapter", style="dim"),
            Text("  Web search model", style="dim"),
            Text("  Extra headers", style="dim"),
            Text("  Notes", style="dim"),
            Text(""),
            Text("Stored separately:", style=STYLE_EMPHASIS),
            Text("  The actual API key", style="dim"),
            Text("   (use the API Key section to update)", style="dim"),
        ),
        title=f"Edit Profile: {profile_name}",
        border_style="cyan",
    )


def _looks_like_secret(value: str) -> bool:
    """Conservative pattern check for accidentally-pasted secrets."""
    text = str(value or "").strip()
    if len(text) < 16:
        return False
    if text.startswith(("sk-", "sk_", "anthropic-", "Bearer ", "ghp_", "gho_", "github_pat_")):
        return True
    if len(text) >= 40 and all(c.isalnum() or c in "_-" for c in text):
        return True
    return False


def _run_profile_remove(state: ConfigMenuState, console: Console) -> None:
    if not state.profiles:
        return
    rows = [(name, name, "") for name in sorted(state.profiles)]
    selected = _run_config_picker(
        console=console,
        title="Provider Profile",
        subtitle="Pick a profile to remove (applied on save)",
        rows=rows,
        current_value=state.active_profile or rows[0][0],
    )
    if selected and _prompt_yes_no(f"Remove profile {selected}? [y/N]"):
        state.remove_profile_name(selected)
        console.print(f"[yellow]Profile {selected} will be removed on save.[/yellow]")


def _run_api_key_section(state: ConfigMenuState, console: Console) -> None:
    console.print()
    console.rule("[bold]API Key[/bold]")
    console.print("[dim]Used by the active profile for all model calls.[/dim]")
    console.print(f"Stored: {state.masked_api_key} · source: {state.api_key_source}")
    try:
        value = str(
            typer.prompt(
                'New API key (Enter to keep current, "clear" to remove)',
                default="",
                hide_input=True,
                show_default=False,
            )
        )
    except (Abort, EOFError, KeyboardInterrupt):
        console.print("")
        _print_section_cancelled(console, "API Key")
        return
    if value.strip().casefold() == "clear":
        try:
            if _prompt_yes_no("Clear the stored API key? [y/N]"):
                state.mark_clear_stored_key_confirmed()
                console.print("[yellow]Stored API key will be cleared on save.[/yellow]")
        except (Abort, EOFError, KeyboardInterrupt):
            console.print("")
            _print_section_cancelled(console, "API Key")
            return
    elif value.strip():
        state.set_field("new_api_key", value.strip())


def _run_default_section(state: ConfigMenuState, console: Console) -> None:
    console.print()
    console.rule("[bold]Default Model[/bold]")
    console.print(
        "[dim]Used when you chat with the agent. Subagents and Forge roles fall back "
        "to this model unless overridden in the sections below.[/dim]"
    )
    try:
        model = _prompt_default_model(console, state)
        base_url = _prompt_text("Base URL", state.fields["base_url"])
        state.set_field("base_url", base_url)
        thinking_label = _prompt_thinking_label(console, state.thinking_label)
        timeout = _prompt_positive_float_text(
            console,
            "Request timeout (seconds)",
            state.fields["llm_timeout_s"],
        )
    except (Abort, EOFError, KeyboardInterrupt):
        console.print("")
        _print_section_cancelled(console, "Default Model")
        return

    state.set_field("model", model)
    state.set_thinking_label(thinking_label)
    state.set_field("llm_timeout_s", timeout)
    console.print(
        "[dim]Reasoning effort. Some providers ignore this until they add native "
        "reasoning support.[/dim]"
    )


def _prompt_default_model(console: Console, state: ConfigMenuState) -> str:
    rows = _default_model_rows(state)
    current_model = str(state.fields.get("model") or "").strip()
    row_values = {value for value, _label, _description in rows}
    current_value = current_model if current_model in row_values else rows[0][0]
    selected = _run_config_picker(
        console=console,
        title="Default Model",
        subtitle=_default_model_picker_subtitle(state),
        rows=rows,
        current_value=current_value,
    )
    if selected is None:
        raise Abort()
    if selected == _CUSTOM_MODEL_VALUE:
        return _prompt_text("Model", current_model)
    return selected


def _default_model_picker_subtitle(state: ConfigMenuState) -> str:
    preset = _active_preset(state)
    if preset is None:
        return "Pick the active profile model, or type a custom model ID."
    return f"Pick a supported model for {preset.label}."


def _default_model_rows(state: ConfigMenuState) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    current_model = str(state.fields.get("model") or "").strip()

    if current_model:
        rows.append((current_model, current_model, "current configured model"))
        seen.add(current_model)

    preset = _active_preset(state)
    if preset is not None:
        for value, label, description in model_options_for_preset(preset):
            if value in seen:
                continue
            seen.add(value)
            rows.append((value, label, description))

    rows.append(
        (
            _CUSTOM_MODEL_VALUE,
            "Type a custom model name",
            "Use any model supported by the active provider",
        )
    )
    return rows


def _active_preset(state: ConfigMenuState) -> ProfilePreset | None:
    preset = find_preset_for_base_url(state.fields.get("base_url", ""))
    if preset is not None:
        return preset

    if not state.active_profile or state.active_profile not in state.profiles:
        return None
    profile = ProfileSpec.from_dict(state.active_profile, state.profiles[state.active_profile])
    return find_preset_for_profile(profile)


def _run_router_section(state: ConfigMenuState, console: Console) -> None:
    console.print()
    console.rule("[bold]Execution Limits[/bold]")
    console.print("[dim]How the agent routes requests and how many tool steps it may take.[/dim]")
    try:
        routing_mode_rows = [
            ("auto", "Auto", "Pick code or chat path per request intent"),
            ("code_only", "Code-only", "Always treat the request as a code task"),
        ]
        routing_mode = _run_config_picker(
            console=console,
            title="Request routing",
            subtitle=f"Active: {state.fields['routing_mode']}",
            rows=routing_mode_rows,
            current_value=state.fields["routing_mode"],
        )
        if routing_mode is None:
            return
        state.set_routing_mode(routing_mode)
        step_budget_rows = [
            ("adaptive", "Adaptive", "Sylliptor adjusts the budget based on task complexity"),
            ("fixed", "Fixed", "Always use the configured limit, no adjustment"),
        ]
        step_budget_policy = _run_config_picker(
            console=console,
            title="Step budget policy",
            subtitle=f"Active: {state.fields['step_budget_policy']}",
            rows=step_budget_rows,
            current_value=state.fields["step_budget_policy"],
        )
        if step_budget_policy is None:
            return
        state.set_step_budget_policy(step_budget_policy)

        state.set_max_steps(
            _prompt_positive_int_text(console, "Max steps per response", state.fields["max_steps"])
        )
        state.set_task_max_steps(
            _prompt_positive_int_text(console, "Max steps per task", state.fields["task_max_steps"])
        )
        state.set_subagent_max_steps(
            _prompt_positive_int_text(
                console,
                "Max steps per subagent run",
                state.fields["subagent_max_steps"],
            )
        )
    except (Abort, EOFError, KeyboardInterrupt):
        console.print("")
        _print_section_cancelled(console, "Execution Limits")
        return


def _run_subagent_section(state: ConfigMenuState, console: Console) -> None:
    console.print()
    console.rule("[bold]Subagent model overrides[/bold]")
    console.print(
        "[dim]Override the model used by the agent's internal roles. Leave empty to inherit "
        "the default model.[/dim]"
    )
    _print_role_explainer(console)
    role_model_snapshot = dict(state.role_models)
    role_temp_snapshot = dict(state.role_temperatures)
    try:
        for role in ROLE_ORDER:
            role_label = _role_label(role)
            model = _prompt_override_text(
                f"  {role_label} model (Enter to skip)",
                state.role_models.get(role, ""),
            )
            state.set_role_model(role, model)
            if role in _ROLE_TEMPERATURE_FIELDS:
                temperature = _prompt_override_text(
                    f"  {role_label} temperature (Enter to skip)",
                    state.role_temperatures.get(role, ""),
                )
                state.set_role_temperature(role, temperature)
    except (Abort, EOFError, KeyboardInterrupt):
        state.role_models = dict(role_model_snapshot)
        state.role_temperatures = dict(role_temp_snapshot)
        console.print("")
        _print_section_cancelled(console, "Subagent model overrides")
        return

    overrides = _non_empty_role_values(state.role_models)
    if overrides:
        console.print(f"[dim]{_pluralize(len(overrides), 'override')} set.[/dim]")
    else:
        console.print("[dim]No subagent model overrides set.[/dim]")


def _run_forge_section(state: ConfigMenuState, console: Console) -> None:
    console.print()
    console.rule("[bold]Forge model overrides[/bold]")
    console.print(
        "[dim]Override the model used by Forge swarm roles. Leave empty to inherit the "
        "subagent override, then the default model.[/dim]"
    )
    _print_role_explainer(console)
    role_model_snapshot = dict(state.forge_role_models)
    try:
        for role in ROLE_ORDER:
            model = _prompt_override_text(
                f"  {_role_label(role)} model (Enter to skip)",
                state.forge_role_models.get(role, ""),
            )
            state.set_forge_role_model(role, model)
    except (Abort, EOFError, KeyboardInterrupt):
        state.forge_role_models = dict(role_model_snapshot)
        console.print("")
        _print_section_cancelled(console, "Forge model overrides")
        return

    overrides = _non_empty_role_values(state.forge_role_models)
    if overrides:
        console.print(f"[dim]{_pluralize(len(overrides), 'override')} set.[/dim]")
    else:
        console.print("[dim]No forge model overrides set.[/dim]")


def _save_and_exit(
    state: ConfigMenuState,
    cfg: AppConfig,
    console: Console,
) -> ConfigMenuResult:
    result = state.commit_to(cfg)
    if not result.saved:
        console.print(f"[red]{result.error or 'Config validation failed.'}[/red]")
        return result
    try:
        save_config(cfg)
        if state.new_api_key.strip():
            if state.active_profile:
                save_persisted_profile_key(state.active_profile, state.new_api_key.strip())
            else:
                save_persisted_api_key(state.new_api_key.strip())
        if state.clear_stored_key_confirmed:
            clear_profile = state.clear_stored_key_profile or state.active_profile
            if clear_profile:
                clear_persisted_profile_key(clear_profile)
            else:
                clear_persisted_api_key()
    except (ConfigError, OSError) as exc:
        message = str(exc)
        console.print(f"[red]Failed to save config:[/red] {message}")
        if isinstance(exc, OSError):
            try:
                path = config_path()
            except Exception as path_exc:  # noqa: BLE001
                console.print(
                    f"[dim]Could not resolve config path for permission hint: {path_exc}[/dim]"
                )
            else:
                console.print(f"[dim]Check write permission on {path}.[/dim]")
        return ConfigMenuResult(saved=False, changes={}, api_key_changed=False, error=message)

    change_count = len(result.changes) + (1 if result.api_key_changed else 0)
    console.print(f"[green]Saved {_pluralize(change_count, 'change')}.[/green]")
    return result


def _confirm_cancel_when_dirty(state: ConfigMenuState, console: Console) -> bool:
    if not state.dirty:
        return True
    try:
        if _prompt_yes_no("Discard pending changes? [y/N]"):
            return True
    except (Abort, EOFError, KeyboardInterrupt):
        console.print("")
    console.print("[dim]Returning to configuration menu.[/dim]")
    return False


def _prompt_text(prompt: str, default: str) -> str:
    value = str(typer.prompt(prompt, default=str(default), show_default=True))
    if value == "":
        return str(default)
    return value.strip()


def _prompt_non_secret_text(
    console: Console,
    prompt: str,
    default: str,
    field_name: str,
) -> str:
    while True:
        value = _prompt_text(prompt, default)
        guarded = _resolve_non_secret_value(console, value, field_name)
        if guarded is not None:
            return guarded


def _resolve_non_secret_value(console: Console, value: str, field_name: str) -> str | None:
    candidate = str(value)
    while _looks_like_secret(candidate):
        console.print(f"[red]That looks like an API key, not a {field_name}.[/red]")
        console.print(
            "[yellow]Tip: paste API keys in the API Key section, not here.\n"
            "Keys here are written to plaintext profile config.[/yellow]"
        )
        replacement = _prompt_override_text(
            f"Re-enter {field_name} (or 'force' to keep this value)",
            "",
        )
        if replacement == _SECRET_FORCE_TOKEN:
            console.print("[red]Confirmed: storing potentially-sensitive value in plaintext.[/red]")
            return candidate
        if replacement:
            candidate = replacement
            continue
        return None
    return candidate


def _prompt_override_text(prompt: str, default: str) -> str:
    value = str(typer.prompt(prompt, default=str(default), show_default=True))
    return value.strip()


def _prompt_thinking_label(console: Console, current: str) -> str:
    rows = [
        (
            label,
            label,
            {
                "off": "no extra reasoning tokens",
                "minimal": "minimal reasoning budget",
                "low": "small reasoning budget",
                "medium": "medium reasoning budget",
                "high": "large reasoning budget",
                "xhigh": "maximum reasoning budget when supported",
                "auto": "let the provider decide",
            }[label],
        )
        for label in THINKING_LABELS
    ]
    value = _run_config_picker(
        console=console,
        title="Default Model",
        subtitle=(
            "Reasoning effort. Some providers ignore this until they add native reasoning support."
        ),
        rows=rows,
        current_value=current,
    )
    if value is None:
        return current
    return _normalize_thinking_label(value)


def _prompt_positive_float_text(console: Console, prompt: str, current: str) -> str:
    for _attempt in range(3):
        value = _prompt_text(prompt, current)
        number = _finite_float(value, fallback=None)
        if number is not None and number > 0:
            return _format_number(number)
        console.print(f"[red]{prompt} must be a positive number.[/red]")
    return current


def _prompt_positive_int_text(console: Console, prompt: str, current: str) -> str:
    for _attempt in range(3):
        value = _prompt_text(prompt, current)
        number = _finite_int(value, fallback=None)
        if number is not None and number > 0:
            return str(number)
        console.print(f"[red]{prompt} must be a positive integer.[/red]")
    return current


def _prompt_yes_no(prompt: str) -> bool:
    clean_prompt = str(prompt).removesuffix(" [y/N]")
    return bool(typer.confirm(clean_prompt, default=False))


def _parse_header_text(raw: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    for item in str(raw or "").split(","):
        text = item.strip()
        if not text:
            continue
        if "=" not in text:
            raise ConfigError("Profile headers must use k=v syntax.")
        key, value = text.split("=", 1)
        if key.strip() and value.strip():
            headers[key.strip()] = value.strip()
    return headers


def _pending_change_count(state: ConfigMenuState) -> int:
    original = state._original
    current = state.snapshot()
    count = 0
    for key in ("fields", "role_models", "forge_role_models", "role_temperatures", "profiles"):
        original_values = original.get(key) if isinstance(original.get(key), dict) else {}
        current_values = current.get(key) if isinstance(current.get(key), dict) else {}
        for field_name in set(original_values) | set(current_values):
            if str(original_values.get(field_name, "")) != str(current_values.get(field_name, "")):
                count += 1
    for key in (
        "active_profile",
        "thinking_label",
        "thinking_label_explicitly_set",
        "new_api_key",
        "clear_stored_key_confirmed",
    ):
        if current.get(key) != original.get(key):
            count += 1
    return count


def mask_api_key(api_key: str | None) -> str:
    clean = str(api_key or "").strip()
    if not clean:
        return "(not set)"
    suffix = clean[-4:] if len(clean) >= 4 else clean
    if clean.startswith("sk-"):
        return f"sk-…{suffix}"
    return f"…{suffix}"


def thinking_label_to_config_value(label: str) -> bool | None:
    normalized = _normalize_thinking_label(label)
    if normalized == "off":
        return False
    if normalized == "auto":
        return None
    return True


def thinking_label_to_reasoning_effort(label: str) -> str | None:
    normalized = _normalize_thinking_label(label)
    if normalized == "off":
        return "none"
    if normalized == "auto":
        return None
    return normalized


def thinking_label_from_cfg(cfg: AppConfig) -> str:
    value = getattr(cfg, "llm_enable_thinking", None)
    reasoning_effort = resolve_llm_reasoning_effort(cfg)
    if reasoning_effort == "none":
        return "off"
    if reasoning_effort in _TRUE_THINKING_LABELS:
        return reasoning_effort
    if value is False:
        return "off"
    if value is None:
        return "auto"
    hint = ""
    if isinstance(cfg.extra_fields, dict):
        hint = str(cfg.extra_fields.get(_THINKING_LABEL_EXTRA_FIELD) or "").strip().lower()
    if hint in _TRUE_THINKING_LABELS:
        return hint
    return "medium"


def _set_thinking_label_hint(cfg: AppConfig, label: str) -> None:
    normalized = _normalize_thinking_label(label)
    if normalized in _TRUE_THINKING_LABELS:
        cfg.extra_fields[_THINKING_LABEL_EXTRA_FIELD] = normalized
        return
    cfg.extra_fields.pop(_THINKING_LABEL_EXTRA_FIELD, None)


def _normalize_role(role: str) -> str:
    role_key = str(role or "").strip().lower()
    if role_key not in ROLE_ORDER:
        raise KeyError(f"Unknown role: {role}")
    return role_key


def _normalize_thinking_label(label: str) -> str:
    normalized = str(label or "").strip().lower()
    if normalized not in THINKING_LABELS:
        allowed = ", ".join(THINKING_LABELS)
        raise ValueError(f"thinking must be one of: {allowed}")
    return normalized


def _normalize_routing_mode(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in ROUTING_MODES:
        allowed = ", ".join(ROUTING_MODES)
        raise ValueError(f"Request routing must be one of: {allowed}")
    return normalized


def _normalize_step_budget_policy(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in STEP_BUDGET_POLICIES:
        allowed = ", ".join(STEP_BUDGET_POLICIES)
        raise ValueError(f"Step budget policy must be one of: {allowed}")
    return normalized


def _thinking_config_text(value: bool | None) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "auto"


def _normalized_role_model_values(raw: Any) -> dict[str, str]:
    source = raw if isinstance(raw, dict) else {}
    values = {role: "" for role in ROLE_ORDER}
    for key, value in source.items():
        role = str(key).strip().lower()
        if role in values:
            values[role] = str(value).strip()
    return values


def _non_empty_role_values(values: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for role in ROLE_ORDER:
        value = str(values.get(role, "")).strip()
        if value:
            out[role] = value
    return out


def _role_text_map_from_snapshot(raw: Any) -> dict[str, str]:
    source = raw if isinstance(raw, dict) else {}
    return {role: str(source.get(role, "")) for role in ROLE_ORDER}


def _finite_float(raw: Any, *, fallback: float | None) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(value):
        return fallback
    return value


def _finite_int(raw: Any, *, fallback: int | None) -> int | None:
    try:
        text = str(raw).strip()
        value = int(text)
    except (TypeError, ValueError):
        return fallback
    return value


def _format_number(raw: Any) -> str:
    value = _finite_float(raw, fallback=None)
    if value is None:
        return str(raw or "")
    if value.is_integer():
        return str(int(value))
    text = f"{value:.12g}"
    return re.sub(r"\.0+$", "", text)


def _format_integer(raw: Any) -> str:
    value = _finite_int(raw, fallback=None)
    if value is None:
        return str(raw or "")
    return str(value)
