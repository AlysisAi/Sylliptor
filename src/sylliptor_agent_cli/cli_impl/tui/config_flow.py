"""Presentation-agnostic state machine for the full-screen TUI ``/config`` menu.

The classic configuration menu (:func:`cli_impl.config_menu.run_config_menu`) is a
Rich ``Live`` + ``create_input().raw_mode()`` flow. It cannot run inside the chat
alt-screen, so when ``SYLLIPTOR_TUI`` is on bare ``/config`` previously degraded to a
read-only panel. This module re-expresses the *same* menu — Provider Profile · API
Key · Default Model · Execution Limits · Subagent overrides · Forge overrides ·
Sandbox — as a step machine that holds state and renders a small :class:`Screen`
description. The prompt_toolkit overlay in :mod:`cli_impl.tui.config_overlay` reads
that description and routes key events back into the flow; nothing here imports
prompt_toolkit, so the whole thing is unit testable by driving the public methods.

The menu *logic* is reused wholesale from :mod:`cli_impl.config_menu`
(:class:`ConfigMenuState` and its row builders / validators / ``commit_to``), so
there is a single source of truth for "what config does"; this module only owns
"how the TUI walks through it". It mirrors the structure of
:mod:`cli_impl.tui.setup_flow`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ...config import (
    ConfigError,
    clear_persisted_api_key,
    clear_persisted_profile_key,
    load_config,
    save_config,
    save_persisted_api_key,
    save_persisted_profile_key,
)
from ...profile_presets import (
    PROFILE_PRESETS,
    make_profile_from_preset,
    preset_selection_label,
)
from ...profiles import ProfileSpec, validate_base_url
from ...provider_diagnostics import provider_diagnostic_warning_lines
from ...sandbox_settings import normalize_sandbox_mode
from ...web_search_adapters import normalize_web_search_adapter
from ...workspace_binding import (
    WorkspaceBindingError,
    discover_workspace_candidates,
    resolve_workspace_binding,
)
from ..config_menu import (
    _CUSTOM_MODEL_VALUE,
    _ROLE_TEMPERATURE_FIELDS,
    ROLE_ORDER,
    THINKING_LABELS,
    ConfigMenuResult,
    ConfigMenuState,
    _default_model_picker_subtitle,
    _default_model_rows,
    _finite_float,
    _format_number,
    _looks_like_secret,
    _ordered_profile_presets_for_setup,
    _override_summary_text,
    _parse_header_text,
    _pending_change_count,
    _preset_description,
    _role_description,
    _role_label,
    _sandbox_mode_env_override,
    _top_level_menu_rows,
)
from ..setup_wizard import _suggest_workspace_default
from .setup_flow import Mode, Row, Screen, Tone

# The literal a user types in a profile-edit field to keep a value that tripped the
# accidental-secret guard (mirrors ``config_menu._SECRET_FORCE_TOKEN``).
_SECRET_FORCE_TOKEN = "force"

# Sentinel row values on the top-level menu for the two trailing action rows.
_SAVE = "__save__"
_CANCEL = "__cancel__"

# stage → interaction mode, so the overlay's key-binding filters don't have to
# build a full :class:`Screen` on every repaint.
_STAGE_MODE: dict[str, Mode] = {
    "menu": "list",
    "advanced": "list",
    "workspace": "list",
    "workspace_path": "input",
    "workspace_action": "list",
    "switching": "busy",
    "sandbox": "list",
    "model": "list",
    "custom_model": "input",
    "model_base_url": "input",
    "model_thinking": "list",
    "model_timeout": "input",
    "limits_routing": "list",
    "limits_budget": "list",
    "limits_max_steps": "input",
    "limits_task_steps": "input",
    "limits_subagent_steps": "input",
    "subagent_field": "input",
    "forge_field": "input",
    "api_key": "input",
    "api_key_clear_confirm": "confirm",
    "provider": "list",
    "provider_switch": "list",
    "provider_add_preset": "list",
    "provider_preset_name": "input",
    "provider_preset_url": "input",
    "provider_custom_name": "input",
    "provider_custom_url": "input",
    "provider_custom_headers": "input",
    "provider_edit": "input",
    "provider_remove": "list",
    "provider_remove_confirm": "confirm",
    "saving": "busy",
    "cancel_confirm": "confirm",
    "done": "done",
}

# Esc/back: the stage to return to from non-sequence stages.
_PREV: dict[str, str] = {
    "advanced": "menu",
    "workspace": "menu",
    "workspace_path": "workspace",
    "workspace_action": "workspace",
    "sandbox": "menu",
    "model": "menu",
    "custom_model": "model",
    "model_base_url": "model",
    "model_thinking": "model_base_url",
    "model_timeout": "model_thinking",
    "limits_routing": "menu",
    "limits_budget": "limits_routing",
    "limits_max_steps": "limits_budget",
    "limits_task_steps": "limits_max_steps",
    "limits_subagent_steps": "limits_task_steps",
    "api_key": "menu",
    "api_key_clear_confirm": "api_key",
    "provider": "menu",
    "provider_switch": "provider",
    "provider_add_preset": "provider",
    "provider_preset_name": "provider_add_preset",
    "provider_preset_url": "provider_preset_name",
    "provider_custom_name": "provider",
    "provider_custom_url": "provider_custom_name",
    "provider_custom_headers": "provider_custom_url",
    "provider_edit": "provider",
    "provider_remove": "provider",
    "provider_remove_confirm": "provider_remove",
}

# stage → footer breadcrumb label.
_BREADCRUMB: dict[str, str] = {
    "menu": "configuration",
    "advanced": "advanced",
    "workspace": "project",
    "workspace_path": "project",
    "workspace_action": "project",
    "switching": "saving",
    "sandbox": "sandbox",
    "model": "default model",
    "custom_model": "default model",
    "model_base_url": "default model",
    "model_thinking": "default model",
    "model_timeout": "default model",
    "limits_routing": "execution limits",
    "limits_budget": "execution limits",
    "limits_max_steps": "execution limits",
    "limits_task_steps": "execution limits",
    "limits_subagent_steps": "execution limits",
    "subagent_field": "subagent overrides",
    "forge_field": "forge overrides",
    "api_key": "api key",
    "api_key_clear_confirm": "api key",
    "saving": "saving",
    "cancel_confirm": "configuration",
}

_ROUTING_ROWS = (
    ("auto", "Auto", "Pick code or chat path per request intent"),
    ("code_only", "Code-only", "Always treat the request as a code task"),
)
_BUDGET_ROWS = (
    ("adaptive", "Adaptive", "Sylliptor adjusts the budget based on task complexity"),
    ("fixed", "Fixed", "Always use the configured limit, no adjustment"),
)
_THINKING_DESCRIPTIONS = {
    "off": "no extra reasoning tokens",
    "minimal": "minimal reasoning budget",
    "low": "small reasoning budget",
    "medium": "medium reasoning budget",
    "high": "large reasoning budget",
    "xhigh": "maximum reasoning budget when supported",
    "auto": "let the provider decide",
}
_PROVIDER_ACTION_ROWS = (
    ("switch", "Switch active profile", "Choose another configured profile."),
    ("add_preset", "Add from preset", "Pick a known provider (OpenAI, Anthropic, …)"),
    ("add_custom", "Add custom", "Use any OpenAI-compatible base URL"),
    ("edit", "Edit current", "Change URL, key env, default model, headers, notes"),
    ("remove", "Remove", "Delete a profile (applied on save)"),
    ("back", "Back", "Return to the menu"),
)
_EDIT_FIELDS = (
    "base_url",
    "api_key_env",
    "default_model",
    "web_search_adapter",
    "web_search_model",
    "extra_headers",
    "notes",
)
_EDIT_LABELS = {
    "base_url": "Base URL",
    "api_key_env": "API key env var NAME (not the key itself)",
    "default_model": "Default model",
    "web_search_adapter": "Web search adapter",
    "web_search_model": "Web search model",
    "extra_headers": "Extra headers (k=v, comma-separated)",
    "notes": "Notes",
}


def _strip_markup(text: str) -> str:
    """Drop Rich ``[style]…[/style]`` tags so menu summaries render as plain text."""
    import re

    return re.sub(r"\[/?[a-z][a-z0-9 _#-]*\]", "", str(text)).strip()


class ConfigFlow:
    """Drives the TUI configuration menu one screen at a time.

    The overlay calls :meth:`screen` to render, the navigation/selection methods on
    key events, and — on the single ``saving`` busy stage — :meth:`run_busy` on a
    worker. Every transition is synchronous and side-effect-explicit so tests can
    walk the whole flow without a terminal.
    """

    def __init__(self, *, cfg: Any | None = None, current_workspace: str | None = None) -> None:
        self.cfg = cfg if cfg is not None else load_config()
        self.state = ConfigMenuState.from_cfg(self.cfg)
        # The live session's workspace root (display only); the running session is
        # bound to it and cannot change it in place — switching restarts the chat.
        self.current_workspace = str(current_workspace or "")
        self.stage = "menu"
        self.index = 0
        self.status = ""
        self.status_tone: Tone = "dim"
        # Terminal results.
        self.success: bool | None = None
        self.saved: bool = False
        self.result: ConfigMenuResult | None = None
        self.changes_count: int = 0
        # Set when the user chooses "Switch now": the overlay reads it on close and
        # asks the host to relaunch the chat bound to this folder (fresh session).
        self.switch_workspace: str | None = None
        self._switch_target: str | None = None
        self._pending_workspace = ""
        # Discovered project candidates are cached (filesystem scan); screen() is
        # pull-based and runs on every repaint, so we must not re-scan per frame.
        self._ws_candidates_cache: list[Any] | None = None
        # Transient sub-flow state.
        self._resume_stage = "menu"
        self._sub_steps: list[tuple[str, str]] = []
        self._sub_i = 0
        self._sub_model_snapshot: dict[str, str] = {}
        self._sub_temp_snapshot: dict[str, str] = {}
        self._forge_i = 0
        self._forge_snapshot: dict[str, str] = {}
        self._pending_preset: Any = None
        self._pending_preset_profile: ProfileSpec | None = None
        self._custom_name = ""
        self._custom_url = ""
        self._edit_profile: ProfileSpec | None = None
        self._edit_values: dict[str, str] = {}
        self._edit_i = 0
        self._edit_secret_candidate: str | None = None
        self._pending_remove = ""
        # Save outcome recorded by the blocking I/O step (:meth:`_perform_save`) and
        # applied as a stage transition by :meth:`apply_save_outcome`. Splitting the
        # two lets the overlay run the I/O on a worker while every renderer-visible
        # stage/status write stays on the UI thread.
        self._save_outcome: tuple[str, str] | None = None

    # ---------------------------------------------------------------- helpers

    def current_mode(self) -> Mode:
        return _STAGE_MODE.get(self.stage, "message")

    def busy_kind(self) -> str:
        return "thread"

    def field_key(self) -> str:
        """A key that changes whenever the active input field changes.

        The overlay seeds/clears its input box when this changes, so a single
        ``input`` stage that walks several fields (subagent/forge/edit sequences)
        re-seeds correctly between fields.
        """
        if self.stage == "subagent_field":
            return f"subagent_field:{self._sub_i}"
        if self.stage == "forge_field":
            return f"forge_field:{self._forge_i}"
        if self.stage == "provider_edit":
            return f"provider_edit:{self._edit_i}"
        return self.stage

    def _set_status(self, text: str, tone: Tone = "dim") -> None:
        self.status = text
        self.status_tone = tone

    def _goto(self, stage: str, *, index: int = 0, keep_status: bool = False) -> None:
        self.stage = stage
        self.index = index
        if not keep_status:
            self.status = ""
            self.status_tone = "dim"

    def _finish(self, success: bool) -> None:
        self.success = success
        self.stage = "done"

    def _breadcrumb(self) -> str:
        if self.stage.startswith("provider"):
            return "provider profile"
        return _BREADCRUMB.get(self.stage, "configuration")

    def _provider_diag_first(self) -> str:
        try:
            issues = list(provider_diagnostic_warning_lines(self.state._resolution_cfg()))
        except Exception:  # noqa: BLE001 - diagnostics are best-effort
            return ""
        return issues[0] if issues else ""

    # ------------------------------------------------------------- rendering

    def screen(self) -> Screen:
        builder = getattr(self, f"_screen_{self.stage}", None)
        if builder is None:
            scr = Screen(
                stage=self.stage,
                mode="message",
                title="Configuration",
                lines=[(f"(unknown stage {self.stage})", "err")],
            )
        else:
            scr = builder()
        scr.index = self.index
        if not scr.status:
            scr.status = self.status
            scr.status_tone = self.status_tone
        scr.progress = self._breadcrumb()
        scr.success = self.success
        return scr

    def _screen_menu(self) -> Screen:
        # The per-role model overrides (subagent + forge) are advanced/optional, so
        # they live under a single "Advanced" entry instead of cluttering the menu.
        rows = [
            Row(
                label="Project / Workspace",
                description=self._workspace_summary(),
                value="workspace",
            )
        ]
        rows.extend(
            Row(label=label, description=_strip_markup(summary), value=value)
            for value, label, summary in _top_level_menu_rows(self.state)
            if value not in ("subagents", "forge")
        )
        rows.append(Row(label="Advanced", description=self._advanced_summary(), value="advanced"))
        rows.append(
            Row(label="Save & exit", description="Write changes to disk and close", value=_SAVE)
        )
        rows.append(Row(label="Discard & close", description="Close without saving", value=_CANCEL))
        pending = _pending_change_count(self.state)
        subtitle = f"{pending} pending change(s)" if pending else "No pending changes."
        if self.state.config_warning:
            subtitle += f"  ·  {self.state.config_warning}"
        return Screen(
            stage="menu",
            mode="list",
            title="Sylliptor configuration",
            subtitle=subtitle,
            rows=rows,
            hint="↑↓ move · 1-9 jump · Enter open · s save · Esc/q close",
        )

    def _advanced_summary(self) -> str:
        sub = self._subagent_override_summary()
        forge = _override_summary_text(self.state.forge_role_models)
        if sub == "none" and forge == "none":
            return "per-role model overrides (none set)"
        return f"subagents {sub} · forge {forge}"

    def _workspace_summary(self) -> str:
        cur = Path(self.current_workspace).name if self.current_workspace else "—"
        default = self.state.default_workspace_path
        if default:
            return f"current: {cur} · default: {Path(default).name}"
        return f"current: {cur} · switch project or set a default"

    def _subagent_override_summary(self) -> str:
        values = {
            **self.state.role_models,
            **{
                f"{role}_temperature": value for role, value in self.state.role_temperatures.items()
            },
        }
        return _override_summary_text(values)

    def _screen_advanced(self) -> Screen:
        rows = [
            Row(
                label="Subagent model overrides",
                description=self._subagent_override_summary(),
                value="subagents",
            ),
            Row(
                label="Forge model overrides",
                description=_override_summary_text(self.state.forge_role_models),
                value="forge",
            ),
            Row(label="Back", description="Return to the menu", value="back"),
        ]
        return Screen(
            stage="advanced",
            mode="list",
            title="Advanced",
            subtitle="Per-role model overrides — leave empty to inherit the default model.",
            rows=rows,
            hint="↑↓ move · Enter select · Esc back",
        )

    # ----------------------------------------------------------- project / workspace

    def _workspace_candidates(self) -> list[Any]:
        if self._ws_candidates_cache is None:
            self._ws_candidates_cache = self._discover_workspace_candidates()
        return self._ws_candidates_cache

    def _discover_workspace_candidates(self) -> list[Any]:
        bases: list[Path] = []
        try:
            bases.append(_suggest_workspace_default())
        except Exception:  # noqa: BLE001 - discovery is best-effort
            pass
        if self.current_workspace:
            try:
                parent = Path(self.current_workspace).expanduser().parent
                if parent not in bases:
                    bases.append(parent)
            except Exception:  # noqa: BLE001
                pass
        out: list[Any] = []
        seen: set[str] = set()
        for base in bases:
            try:
                found = discover_workspace_candidates(base)
            except Exception:  # noqa: BLE001
                continue
            for cand in found:
                key = os.fspath(cand.path)
                if key in seen:
                    continue
                seen.add(key)
                out.append(cand)
        return out[:8]

    def _screen_workspace(self) -> Screen:
        rows: list[Row] = []
        for cand in self._workspace_candidates():
            rows.append(
                Row(
                    label=cand.path.name or os.fspath(cand.path),
                    description=getattr(cand, "summary", "") or os.fspath(cand.path),
                    value=os.fspath(cand.path),
                    current=os.fspath(cand.path) == self.current_workspace,
                )
            )
        rows.append(
            Row(label="Type a path…", description="Enter any folder path.", value="__type_path__")
        )
        rows.append(Row(label="Back", description="Return to the menu.", value="back"))
        cur = self.current_workspace or "—"
        default = self.state.default_workspace_path or "none"
        return Screen(
            stage="workspace",
            mode="list",
            title="Project / Workspace",
            subtitle=f"Current: {cur}  ·  default: {default}",
            rows=rows,
            hint="↑↓ move · 1-9 pick · Enter select · Esc back",
        )

    def _screen_workspace_path(self) -> Screen:
        return Screen(
            stage="workspace_path",
            mode="input",
            title="Project / Workspace",
            subtitle="Type the folder you want to work in.",
            input_label="Workspace folder",
            input_default=self.current_workspace,
            hint="Enter next · Esc back",
        )

    def _screen_workspace_action(self) -> Screen:
        rows = [
            Row(
                label="Switch now — restart chat here",
                description="Save changes and reopen the chat in this project (fresh conversation).",
                value="switch",
            ),
            Row(
                label="Set as default — next launch",
                description="Use this project next time you start Sylliptor. Current session stays.",
                value="set_default",
            ),
            Row(label="Back", description="Pick a different folder.", value="back"),
        ]
        return Screen(
            stage="workspace_action",
            mode="list",
            title="Project / Workspace",
            subtitle=f"Selected: {self._pending_workspace}",
            rows=rows,
            hint="↑↓ move · Enter select · Esc back",
        )

    def _screen_switching(self) -> Screen:
        return Screen(
            stage="switching",
            mode="busy",
            title="Switching project",
            busy_label="Saving and switching project…",
        )

    def _screen_sandbox(self) -> Screen:
        current = normalize_sandbox_mode(self.state.fields.get("sandbox_mode", "strict"))
        rows = [
            Row(
                label="Strict (recommended)",
                description="Run shell & verification commands in a sandbox (bubblewrap/Docker).",
                value="strict",
                current=current == "strict",
            ),
            Row(
                label="Warn",
                description="Try the sandbox; fail closed if no backend (no host fallback).",
                value="warn",
                current=current == "warn",
            ),
            Row(
                label="Off (less safe)",
                description="Run commands directly on the host shell. No isolation.",
                value="off",
                current=current == "off",
            ),
        ]
        subtitle = "How Sylliptor isolates shell and verification commands."
        override = _sandbox_mode_env_override()
        if override:
            subtitle += f"  ·  Note: {override} overrides this at runtime."
        return Screen(
            stage="sandbox",
            mode="list",
            title="Sandbox",
            subtitle=subtitle,
            rows=rows,
            hint="↑↓ move · Enter select · Esc back",
        )

    def _screen_model(self) -> Screen:
        rows = [
            Row(label=label, description=desc, value=value)
            for value, label, desc in _default_model_rows(self.state)
        ]
        return Screen(
            stage="model",
            mode="list",
            title="Default model",
            subtitle=_default_model_picker_subtitle(self.state),
            rows=rows,
            hint="↑↓ move · 1-9 pick · Enter select · Esc back",
        )

    def _screen_custom_model(self) -> Screen:
        return Screen(
            stage="custom_model",
            mode="input",
            title="Default model",
            subtitle="Type any model id supported by the active provider.",
            input_label="Model",
            input_default=str(self.state.fields.get("model", "")),
            hint="Enter next · Esc back",
        )

    def _screen_model_base_url(self) -> Screen:
        return Screen(
            stage="model_base_url",
            mode="input",
            title="Default model",
            subtitle="Base URL for the active provider (blank keeps the current value).",
            input_label="Base URL",
            input_default=str(self.state.fields.get("base_url", "")),
            hint="Enter next · Esc back",
        )

    def _screen_model_thinking(self) -> Screen:
        current = self.state.thinking_label
        rows = [
            Row(
                label=label,
                description=_THINKING_DESCRIPTIONS[label],
                value=label,
                current=label == current,
            )
            for label in THINKING_LABELS
        ]
        return Screen(
            stage="model_thinking",
            mode="list",
            title="Default model",
            subtitle="Reasoning effort. Some providers ignore this until they add native support.",
            rows=rows,
            hint="↑↓ move · Enter select · Esc back",
        )

    def _screen_model_timeout(self) -> Screen:
        return Screen(
            stage="model_timeout",
            mode="input",
            title="Default model",
            subtitle="How long to wait for a model response.",
            input_label="Request timeout (seconds)",
            input_default=str(self.state.fields.get("llm_timeout_s", "")),
            hint="Enter save · Esc back",
        )

    def _screen_limits_routing(self) -> Screen:
        current = self.state.fields["routing_mode"]
        rows = [
            Row(label=label, description=desc, value=value, current=value == current)
            for value, label, desc in _ROUTING_ROWS
        ]
        return Screen(
            stage="limits_routing",
            mode="list",
            title="Execution limits",
            subtitle="How the agent routes requests.",
            rows=rows,
            hint="↑↓ move · Enter select · Esc back",
        )

    def _screen_limits_budget(self) -> Screen:
        current = self.state.fields["step_budget_policy"]
        rows = [
            Row(label=label, description=desc, value=value, current=value == current)
            for value, label, desc in _BUDGET_ROWS
        ]
        return Screen(
            stage="limits_budget",
            mode="list",
            title="Execution limits",
            subtitle="Step budget policy.",
            rows=rows,
            hint="↑↓ move · Enter select · Esc back",
        )

    def _screen_limits_max_steps(self) -> Screen:
        return self._int_screen("limits_max_steps", "Max steps per response", "max_steps")

    def _screen_limits_task_steps(self) -> Screen:
        return self._int_screen("limits_task_steps", "Max steps per task", "task_max_steps")

    def _screen_limits_subagent_steps(self) -> Screen:
        return self._int_screen(
            "limits_subagent_steps", "Max steps per subagent run", "subagent_max_steps"
        )

    def _int_screen(self, stage: str, label: str, field: str) -> Screen:
        return Screen(
            stage=stage,
            mode="input",
            title="Execution limits",
            subtitle="A positive integer.",
            input_label=label,
            input_default=str(self.state.fields.get(field, "")),
            hint="Enter next · Esc back",
        )

    def _screen_subagent_field(self) -> Screen:
        role, kind = self._sub_steps[self._sub_i]
        label = _role_label(role)
        if kind == "model":
            field_label = f"{label} model (blank to keep / inherit)"
            default = self.state.role_models.get(role, "")
        else:
            field_label = f"{label} temperature (blank to keep / inherit)"
            default = self.state.role_temperatures.get(role, "")
        subtitle = f"{_role_description(role)}  ·  step {self._sub_i + 1}/{len(self._sub_steps)}"
        return Screen(
            stage="subagent_field",
            mode="input",
            title="Subagent model overrides",
            subtitle=subtitle,
            input_label=field_label,
            input_default=default,
            hint="Enter next · Esc cancel section",
        )

    def _screen_forge_field(self) -> Screen:
        role = ROLE_ORDER[self._forge_i]
        subtitle = f"{_role_description(role)}  ·  step {self._forge_i + 1}/{len(ROLE_ORDER)}"
        return Screen(
            stage="forge_field",
            mode="input",
            title="Forge model overrides",
            subtitle=subtitle,
            input_label=f"{_role_label(role)} model (blank to keep / inherit)",
            input_default=self.state.forge_role_models.get(role, ""),
            hint="Enter next · Esc cancel section",
        )

    def _screen_api_key(self) -> Screen:
        return Screen(
            stage="api_key",
            mode="input",
            title="API key",
            subtitle=f"Stored: {self.state.masked_api_key} · source: {self.state.api_key_source}",
            input_label='New API key (blank keeps current · "clear" to remove)',
            input_password=True,
            hint="Enter save · Esc back",
        )

    def _screen_api_key_clear_confirm(self) -> Screen:
        return Screen(
            stage="api_key_clear_confirm",
            mode="confirm",
            title="API key",
            lines=[("Clear the stored API key?", "warn")],
            hint="Y clear · N keep · Esc back",
            confirm_default=False,
        )

    def _screen_provider(self) -> Screen:
        active = self.state.active_profile or "(none)"
        rows = [
            Row(label=label, description=desc, value=value)
            for value, label, desc in _PROVIDER_ACTION_ROWS
        ]
        return Screen(
            stage="provider",
            mode="list",
            title="Provider profile",
            subtitle=f"Active profile: {active}",
            rows=rows,
            hint="↑↓ move · Enter select · Esc back",
        )

    def _screen_provider_switch(self) -> Screen:
        rows = [
            Row(
                label=name,
                description="active" if name == self.state.active_profile else "",
                value=name,
                current=name == self.state.active_profile,
            )
            for name in sorted(self.state.profiles)
        ]
        return Screen(
            stage="provider_switch",
            mode="list",
            title="Switch profile",
            subtitle=f"Active: {self.state.active_profile or 'none'}",
            rows=rows,
            hint="↑↓ move · Enter select · Esc back",
        )

    def _screen_provider_add_preset(self) -> Screen:
        rows = [
            Row(
                label=preset_selection_label(preset),
                description=_preset_description(preset),
                value=preset.key,
            )
            for preset in _ordered_profile_presets_for_setup()
        ]
        return Screen(
            stage="provider_add_preset",
            mode="list",
            title="Add provider preset",
            subtitle="Pick a provider preset.",
            rows=rows,
            hint="↑↓ move · 1-9 pick · Enter select · Esc back",
        )

    def _screen_provider_preset_name(self) -> Screen:
        return Screen(
            stage="provider_preset_name",
            mode="input",
            title="Add provider preset",
            subtitle="Name this profile (lowercase, no spaces).",
            input_label="Profile name",
            input_default=getattr(self._pending_preset, "key", "custom"),
            hint="Enter next · Esc back",
        )

    def _screen_provider_preset_url(self) -> Screen:
        return Screen(
            stage="provider_preset_url",
            mode="input",
            title="Add provider preset",
            subtitle="This preset needs an OpenAI-compatible base URL.",
            input_label="Base URL",
            input_default="",
            hint="Enter next · Esc back",
        )

    def _screen_provider_custom_name(self) -> Screen:
        return Screen(
            stage="provider_custom_name",
            mode="input",
            title="Add custom profile",
            subtitle="Name this profile (lowercase, no spaces).",
            input_label="Profile name",
            input_default=self._custom_name or "custom",
            hint="Enter next · Esc back",
        )

    def _screen_provider_custom_url(self) -> Screen:
        return Screen(
            stage="provider_custom_url",
            mode="input",
            title="Add custom profile",
            subtitle="The OpenAI-compatible base URL for this provider.",
            input_label="Base URL",
            input_default=self._custom_url,
            hint="Enter next · Esc back",
        )

    def _screen_provider_custom_headers(self) -> Screen:
        return Screen(
            stage="provider_custom_headers",
            mode="input",
            title="Add custom profile",
            subtitle="Extra request headers, if your endpoint needs them.",
            input_label="Extra headers (k=v, comma-separated)",
            input_default="",
            hint="Enter add · Esc back",
        )

    def _screen_provider_edit(self) -> Screen:
        field = _EDIT_FIELDS[self._edit_i]
        name = getattr(self._edit_profile, "name", "")
        return Screen(
            stage="provider_edit",
            mode="input",
            title=f"Edit profile: {name}",
            subtitle=f"Blank keeps the current value  ·  step {self._edit_i + 1}/{len(_EDIT_FIELDS)}",
            input_label=_EDIT_LABELS[field],
            input_default=self._edit_values.get(field, ""),
            hint="Enter next · Esc cancel",
        )

    def _screen_provider_remove(self) -> Screen:
        rows = [Row(label=name, value=name) for name in sorted(self.state.profiles)]
        return Screen(
            stage="provider_remove",
            mode="list",
            title="Remove profile",
            subtitle="Pick a profile to remove (applied on save).",
            rows=rows,
            hint="↑↓ move · Enter select · Esc back",
        )

    def _screen_provider_remove_confirm(self) -> Screen:
        return Screen(
            stage="provider_remove_confirm",
            mode="confirm",
            title="Remove profile",
            lines=[(f"Remove profile {self._pending_remove}?", "warn")],
            hint="Y remove · N keep · Esc back",
            confirm_default=False,
        )

    def _screen_saving(self) -> Screen:
        return Screen(
            stage="saving", mode="busy", title="Saving", busy_label="Saving configuration…"
        )

    def _screen_cancel_confirm(self) -> Screen:
        return Screen(
            stage="cancel_confirm",
            mode="confirm",
            title="Discard changes",
            lines=[("Discard pending changes and close?", "warn")],
            hint="Y discard · N keep editing",
            confirm_default=False,
        )

    def _screen_done(self) -> Screen:
        return Screen(stage="done", mode="done", title="")

    # ----------------------------------------------------------- navigation

    def move(self, delta: int) -> None:
        scr = self.screen()
        if scr.mode != "list" or not scr.rows:
            return
        self.index = (self.index + delta) % len(scr.rows)

    def choose_index(self, idx: int) -> None:
        scr = self.screen()
        if scr.mode != "list" or not scr.rows:
            return
        if 0 <= idx < len(scr.rows):
            self.index = idx
            self.choose(scr.rows[idx].value)

    def choose_current(self) -> None:
        scr = self.screen()
        if scr.mode != "list" or not scr.rows:
            return
        self.choose(scr.rows[self.index].value)

    def choose(self, value: str) -> None:
        handler = {
            "menu": self._choose_menu,
            "advanced": self._choose_advanced,
            "workspace": self._choose_workspace,
            "workspace_action": self._choose_workspace_action,
            "sandbox": self._choose_sandbox,
            "model": self._choose_model,
            "model_thinking": self._choose_thinking,
            "limits_routing": self._choose_routing,
            "limits_budget": self._choose_budget,
            "provider": self._choose_provider_action,
            "provider_switch": self._choose_provider_switch,
            "provider_add_preset": self._choose_preset,
            "provider_remove": self._choose_remove,
        }.get(self.stage)
        if handler is not None:
            handler(value)

    def submit_input(self, text: str) -> None:
        handler = getattr(self, f"_submit_{self.stage}", None)
        if handler is not None:
            handler(str(text))

    def confirm(self, yes: bool) -> None:
        if self.stage == "api_key_clear_confirm":
            if yes:
                self.state.mark_clear_stored_key_confirmed()
                self._goto("menu")
                self._set_status("Stored API key will be cleared on save.", "warn")
            else:
                self._goto("menu")
        elif self.stage == "provider_remove_confirm":
            if yes:
                self.state.remove_profile_name(self._pending_remove)
                self._goto("provider")
                self._set_status(f"Profile {self._pending_remove} will be removed on save.", "warn")
            else:
                self._goto("provider")
        elif self.stage == "cancel_confirm":
            if yes:
                self._finish(False)
            else:
                self._goto(self._resume_stage or "menu")

    def back(self) -> None:
        if self.stage == "menu":
            self.request_cancel()
            return
        if self.stage == "subagent_field":
            self.state.role_models = dict(self._sub_model_snapshot)
            self.state.role_temperatures = dict(self._sub_temp_snapshot)
            self._goto("advanced")
            return
        if self.stage == "forge_field":
            self.state.forge_role_models = dict(self._forge_snapshot)
            self._goto("advanced")
            return
        if self.stage == "cancel_confirm":
            self._goto(self._resume_stage or "menu")
            return
        prev = _PREV.get(self.stage)
        if prev is not None:
            self._goto(prev)

    def request_cancel(self) -> None:
        """Ctrl+C / the Discard row: confirm before discarding when dirty."""
        if self.stage in {"cancel_confirm", "done", "saving"}:
            return
        if self.state.dirty:
            # Answering "No" returns to wherever the user was (a sub-flow field, a
            # section list, or the menu) rather than always snapping back to the top.
            self._resume_stage = self.stage
            self._goto("cancel_confirm", keep_status=True)
        else:
            self._finish(False)

    # ------------------------------------------------------------- menu step

    def _choose_menu(self, value: str) -> None:
        if value == _SAVE:
            self._goto("saving")
            return
        if value == _CANCEL:
            self.request_cancel()
            return
        if value == "profile":
            self._goto("provider")
        elif value == "api_key":
            self._goto("api_key")
        elif value == "default":
            self._goto("model")
        elif value == "router":
            self._goto("limits_routing", index=self._routing_index())
        elif value == "workspace":
            self._goto("workspace")
        elif value == "advanced":
            self._goto("advanced")
        elif value == "sandbox":
            self._goto("sandbox", index=self._sandbox_index())

    def _choose_advanced(self, value: str) -> None:
        if value == "back":
            self._goto("menu")
        elif value == "subagents":
            self._enter_subagents()
        elif value == "forge":
            self._enter_forge()

    # ----------------------------------------------------- project / workspace

    def _resolve_workspace(self, raw: str) -> str | None:
        text = str(raw).strip()
        if not text:
            self._set_status("A folder path is required.", "err")
            return None
        try:
            selected = Path(text).expanduser().resolve()
            binding = resolve_workspace_binding(
                selected, allow_broad_workspace=True, source="config_menu"
            )
        except WorkspaceBindingError as exc:
            self._set_status(str(exc), "err")
            return None
        except Exception as exc:  # noqa: BLE001 - bad path / permission error
            self._set_status(f"Invalid path: {exc}", "err")
            return None
        return os.fspath(binding.requested_path)

    def _choose_workspace(self, value: str) -> None:
        if value == "back":
            self._goto("menu")
            return
        if value == "__type_path__":
            self._goto("workspace_path")
            return
        resolved = self._resolve_workspace(value)
        if resolved is None:
            return
        self._pending_workspace = resolved
        self._goto("workspace_action")

    def _submit_workspace_path(self, text: str) -> None:
        resolved = self._resolve_workspace(text)
        if resolved is None:
            return
        self._pending_workspace = resolved
        self._goto("workspace_action")

    def _choose_workspace_action(self, value: str) -> None:
        if value == "back" or not self._pending_workspace:
            self._goto("workspace")
            return
        if value == "set_default":
            self.state.set_default_workspace_path(self._pending_workspace)
            self._goto("menu")
            self._set_status("Default project set. Save to apply.", "ok")
            return
        if value == "switch":
            # Persist the new default + all pending edits, then signal the host to
            # relaunch the chat bound to this folder (the live root cannot change in
            # place). The full save runs on the busy stage; on success
            # apply_save_outcome promotes _switch_target -> switch_workspace.
            self.state.set_default_workspace_path(self._pending_workspace)
            self._switch_target = self._pending_workspace
            self._goto("switching")

    def _sandbox_index(self) -> int:
        order = ("strict", "warn", "off")
        current = normalize_sandbox_mode(self.state.fields.get("sandbox_mode", "strict"))
        return order.index(current) if current in order else 0

    # ----------------------------------------------------------- sandbox

    def _choose_sandbox(self, value: str) -> None:
        normalized = normalize_sandbox_mode(value)
        self.state.fields["sandbox_mode"] = normalized
        self._goto("menu")
        if normalized == "off":
            self._set_status("Sandbox off — commands run on the host shell. Save to apply.", "warn")
        else:
            self._set_status(f"Sandbox mode: {normalized}. Save to apply.", "ok")

    # ----------------------------------------------------------- default model

    def _choose_model(self, value: str) -> None:
        if value == _CUSTOM_MODEL_VALUE:
            self._goto("custom_model")
            return
        self.state.set_field("model", value)
        self._goto("model_base_url")

    def _submit_custom_model(self, text: str) -> None:
        model = text.strip() or str(self.state.fields.get("model", "")).strip()
        if not model:
            self._set_status("Model is required.", "err")
            return
        self.state.set_field("model", model)
        self._goto("model_base_url")

    def _submit_model_base_url(self, text: str) -> None:
        value = text.strip() or str(self.state.fields.get("base_url", ""))
        try:
            validate_base_url(value, key="Base URL", allow_empty=True)
        except ConfigError as exc:
            self._set_status(str(exc), "err")
            return
        self.state.set_field("base_url", value)
        self._goto("model_thinking", index=self._thinking_index())

    def _thinking_index(self) -> int:
        current = self.state.thinking_label
        return THINKING_LABELS.index(current) if current in THINKING_LABELS else 0

    def _choose_thinking(self, value: str) -> None:
        self.state.set_thinking_label(value)
        self._goto("model_timeout")

    def _submit_model_timeout(self, text: str) -> None:
        value = text.strip() or str(self.state.fields.get("llm_timeout_s", ""))
        number = _finite_float(value, fallback=None)
        if number is None or number <= 0:
            self._set_status("Request timeout must be a positive number.", "err")
            return
        self.state.set_field("llm_timeout_s", _format_number(number))
        self._goto("menu")
        self._set_status("Default model updated. Save to apply.", "ok")

    # ----------------------------------------------------------- limits

    def _choose_routing(self, value: str) -> None:
        self.state.set_routing_mode(value)
        self._goto("limits_budget", index=self._budget_index())

    def _choose_budget(self, value: str) -> None:
        self.state.set_step_budget_policy(value)
        self._goto("limits_max_steps")

    def _submit_limits_max_steps(self, text: str) -> None:
        n = self._parse_pos_int(text, self.state.fields["max_steps"])
        if n is None:
            self._set_status("Max steps per response must be a positive integer.", "err")
            return
        self.state.set_max_steps(str(n))
        self._goto("limits_task_steps")

    def _submit_limits_task_steps(self, text: str) -> None:
        n = self._parse_pos_int(text, self.state.fields["task_max_steps"])
        if n is None:
            self._set_status("Max steps per task must be a positive integer.", "err")
            return
        self.state.set_task_max_steps(str(n))
        self._goto("limits_subagent_steps")

    def _submit_limits_subagent_steps(self, text: str) -> None:
        n = self._parse_pos_int(text, self.state.fields["subagent_max_steps"])
        if n is None:
            self._set_status("Max steps per subagent run must be a positive integer.", "err")
            return
        self.state.set_subagent_max_steps(str(n))
        self._goto("menu")
        self._set_status("Execution limits updated. Save to apply.", "ok")

    def _routing_index(self) -> int:
        current = self.state.fields.get("routing_mode", "auto")
        order = [value for value, _label, _desc in _ROUTING_ROWS]
        return order.index(current) if current in order else 0

    def _budget_index(self) -> int:
        current = self.state.fields.get("step_budget_policy", "adaptive")
        order = [value for value, _label, _desc in _BUDGET_ROWS]
        return order.index(current) if current in order else 0

    @staticmethod
    def _parse_pos_int(text: str, default: str) -> int | None:
        value = text.strip() or str(default)
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    # ----------------------------------------------------- subagent overrides

    def _enter_subagents(self) -> None:
        self._sub_model_snapshot = dict(self.state.role_models)
        self._sub_temp_snapshot = dict(self.state.role_temperatures)
        steps: list[tuple[str, str]] = []
        for role in ROLE_ORDER:
            steps.append((role, "model"))
            if role in _ROLE_TEMPERATURE_FIELDS:
                steps.append((role, "temp"))
        self._sub_steps = steps
        self._sub_i = 0
        self._goto("subagent_field")

    def _submit_subagent_field(self, text: str) -> None:
        role, kind = self._sub_steps[self._sub_i]
        value = text.strip()
        if kind == "model":
            if value:
                self.state.set_role_model(role, value)
        else:
            if value:
                number = _finite_float(value, fallback=None)
                if number is None or number < 0:
                    self._set_status("Temperature must be a non-negative number.", "err")
                    return
                self.state.set_role_temperature(role, _format_number(number))
        self._advance_subagent()

    def _advance_subagent(self) -> None:
        self._sub_i += 1
        if self._sub_i >= len(self._sub_steps):
            self._goto("advanced")
            self._set_status("Subagent overrides updated. Save to apply.", "ok")
        else:
            self.status = ""
            self.status_tone = "dim"

    # -------------------------------------------------------- forge overrides

    def _enter_forge(self) -> None:
        self._forge_snapshot = dict(self.state.forge_role_models)
        self._forge_i = 0
        self._goto("forge_field")

    def _submit_forge_field(self, text: str) -> None:
        role = ROLE_ORDER[self._forge_i]
        value = text.strip()
        if value:
            self.state.set_forge_role_model(role, value)
        self._forge_i += 1
        if self._forge_i >= len(ROLE_ORDER):
            self._goto("advanced")
            self._set_status("Forge overrides updated. Save to apply.", "ok")
        else:
            self.status = ""
            self.status_tone = "dim"

    # ----------------------------------------------------------- api key

    def _submit_api_key(self, text: str) -> None:
        value = text.strip()
        if not value:
            self._goto("menu")
            return
        if value.casefold() == "clear":
            self._goto("api_key_clear_confirm")
            return
        self.state.set_field("new_api_key", value)
        self._goto("menu")
        self._set_status("API key set. Save to apply.", "ok")

    # -------------------------------------------------------- provider profile

    def _choose_provider_action(self, value: str) -> None:
        if value == "back":
            self._goto("menu")
        elif value == "switch":
            if not self.state.profiles:
                self._set_status("No profiles configured yet.", "warn")
                return
            self._goto("provider_switch", index=self._provider_switch_index())
        elif value == "add_preset":
            self._goto("provider_add_preset")
        elif value == "add_custom":
            self._goto("provider_custom_name")
        elif value == "edit":
            self._enter_provider_edit()
        elif value == "remove":
            if not self.state.profiles:
                self._set_status("No profiles to remove.", "warn")
                return
            self._goto("provider_remove")

    def _provider_switch_index(self) -> int:
        names = sorted(self.state.profiles)
        active = self.state.active_profile
        return names.index(active) if active in names else 0

    def _choose_provider_switch(self, value: str) -> None:
        self.state.set_active_profile_name(value)
        self._goto("provider")
        diag = self._provider_diag_first()
        if diag:
            self._set_status(f"Provider diagnostic: {diag}", "warn")
        else:
            self._set_status(f"Active profile: {value}.", "ok")

    def _choose_preset(self, value: str) -> None:
        self._pending_preset = next((p for p in PROFILE_PRESETS if p.key == value), None)
        if self._pending_preset is None:
            self._set_status("Unknown preset.", "err")
            return
        self._goto("provider_preset_name")

    def _submit_provider_preset_name(self, text: str) -> None:
        name = text.strip() or getattr(self._pending_preset, "key", "custom")
        profile = make_profile_from_preset(self._pending_preset, name=name)
        if not profile.base_url:
            self._pending_preset_profile = profile
            self._goto("provider_preset_url")
            return
        self._finalize_preset_profile(profile)

    def _submit_provider_preset_url(self, text: str) -> None:
        url = text.strip()
        if not url:
            self._set_status("Base URL is required.", "err")
            return
        base = self._pending_preset_profile
        profile = ProfileSpec(
            name=base.name,
            protocol=base.protocol,
            base_url=url,
            api_key_env=base.api_key_env,
            extra_headers=base.extra_headers,
            default_model=base.default_model,
            web_search_adapter=base.web_search_adapter,
            web_search_model=base.web_search_model,
            notes=base.notes,
        )
        self._finalize_preset_profile(profile)

    def _finalize_preset_profile(self, profile: ProfileSpec) -> None:
        self.state.add_profile_spec(profile)
        self._goto("provider")
        warning = (getattr(self._pending_preset, "setup_warning", "") or "").strip()
        diag = self._provider_diag_first()
        if warning:
            self._set_status(f"Profile {profile.name} added. {warning}", "warn")
        elif diag:
            self._set_status(f"Profile {profile.name} added. Provider diagnostic: {diag}", "warn")
        else:
            self._set_status(f"Profile {profile.name} added.", "ok")

    def _submit_provider_custom_name(self, text: str) -> None:
        self._custom_name = (text.strip() or "custom").lower()
        self._goto("provider_custom_url")

    def _submit_provider_custom_url(self, text: str) -> None:
        url = text.strip()
        if not url:
            self._set_status("Base URL is required.", "err")
            return
        self._custom_url = url
        self._goto("provider_custom_headers")

    def _submit_provider_custom_headers(self, text: str) -> None:
        try:
            headers = _parse_header_text(text)
        except ConfigError as exc:
            self._set_status(str(exc), "err")
            return
        profile = ProfileSpec(
            name=self._custom_name,
            protocol="openai_compat",
            base_url=self._custom_url,
            extra_headers=headers,
            web_search_adapter="auto",
            web_search_model="",
            notes="Custom OpenAI-compatible endpoint.",
        )
        self.state.add_profile_spec(profile)
        self._goto("provider")
        diag = self._provider_diag_first()
        if diag:
            self._set_status(f"Profile {profile.name} added. Provider diagnostic: {diag}", "warn")
        else:
            self._set_status(f"Profile {profile.name} added.", "ok")

    def _enter_provider_edit(self) -> None:
        if not self.state.active_profile or self.state.active_profile not in self.state.profiles:
            self._set_status("No active profile. Switch to one or add a new profile first.", "warn")
            return
        profile = ProfileSpec.from_dict(
            self.state.active_profile, self.state.profiles[self.state.active_profile]
        )
        self._edit_profile = profile
        self._edit_values = {
            "base_url": profile.base_url,
            "api_key_env": profile.api_key_env or "",
            "default_model": profile.default_model,
            "web_search_adapter": profile.web_search_adapter,
            "web_search_model": profile.web_search_model,
            "extra_headers": ", ".join(f"{k}={v}" for k, v in profile.extra_headers.items()),
            "notes": profile.notes,
        }
        self._edit_i = 0
        self._edit_secret_candidate = None
        self._goto("provider_edit")

    def _submit_provider_edit(self, text: str) -> None:
        field = _EDIT_FIELDS[self._edit_i]
        value = text.strip()
        if not value:
            value = self._edit_values.get(field, "")
        # Accidental-secret guard (mirrors config_menu._resolve_non_secret_value).
        if value and value != _SECRET_FORCE_TOKEN and _looks_like_secret(value):
            self._edit_secret_candidate = value
            self._set_status(
                "That looks like an API key — paste keys in the API Key section. "
                "Re-enter, or type 'force' to store it anyway.",
                "warn",
            )
            return
        if value == _SECRET_FORCE_TOKEN and self._edit_secret_candidate is not None:
            value = self._edit_secret_candidate
        self._edit_secret_candidate = None
        if field == "web_search_adapter":
            try:
                value = normalize_web_search_adapter(value or "auto")
            except (ConfigError, ValueError) as exc:
                self._set_status(str(exc), "err")
                return
        if field == "extra_headers":
            try:
                _parse_header_text(value)
            except ConfigError as exc:
                self._set_status(str(exc), "err")
                return
        self._edit_values[field] = value
        self._edit_i += 1
        if self._edit_i >= len(_EDIT_FIELDS):
            self._finalize_provider_edit()
        else:
            self.status = ""
            self.status_tone = "dim"

    def _finalize_provider_edit(self) -> None:
        values = self._edit_values
        self.state.update_active_profile_spec(
            base_url=values["base_url"],
            api_key_env=values["api_key_env"] or None,
            default_model=values["default_model"],
            web_search_adapter=values["web_search_adapter"] or "auto",
            web_search_model=values["web_search_model"],
            extra_headers=_parse_header_text(values["extra_headers"]),
            notes=values["notes"],
        )
        self._goto("provider")
        diag = self._provider_diag_first()
        if diag:
            self._set_status(f"Profile updated. Provider diagnostic: {diag}", "warn")
        else:
            self._set_status("Profile updated. Save to apply.", "ok")

    def _choose_remove(self, value: str) -> None:
        self._pending_remove = value
        self._goto("provider_remove_confirm")

    # ----------------------------------------------------------- busy / save

    def run_busy(self) -> None:
        # Synchronous driver (used by tests): perform the blocking I/O and apply the
        # resulting stage transition in one call. The overlay instead calls
        # :meth:`perform_save` on a worker and :meth:`apply_save_outcome` on the UI
        # thread, so the renderer never observes a stage write from another thread.
        handler = getattr(self, f"_run_{self.stage}", None)
        if handler is not None:
            handler()

    def _run_saving(self) -> None:
        self.perform_save()
        self.apply_save_outcome()

    def _run_switching(self) -> None:
        # "Switch now" persists everything (incl. the new default workspace) exactly
        # like a save; apply_save_outcome then promotes the pending switch target.
        self.perform_save()
        self.apply_save_outcome()

    def perform_save(self) -> None:
        """Blocking save I/O. Records the outcome; writes NO renderer-visible state.

        Safe to run on a worker thread: it only touches disk/keyring and stashes the
        result on :attr:`_save_outcome` / :attr:`result` / :attr:`changes_count`
        (none of which the render callbacks read). The stage/status transition is
        applied separately by :meth:`apply_save_outcome` on the UI thread.
        """
        result = self.state.commit_to(self.cfg)
        if not result.saved:
            self._save_outcome = ("error", result.error or "Config validation failed.")
            return
        try:
            save_config(self.cfg)
            new_key = self.state.new_api_key.strip()
            if new_key:
                if self.state.active_profile:
                    save_persisted_profile_key(self.state.active_profile, new_key)
                else:
                    save_persisted_api_key(new_key)
            if self.state.clear_stored_key_confirmed:
                clear_profile = self.state.clear_stored_key_profile or self.state.active_profile
                if clear_profile:
                    clear_persisted_profile_key(clear_profile)
                else:
                    clear_persisted_api_key()
        except (ConfigError, OSError) as exc:
            self._save_outcome = ("error", f"Failed to save config: {exc}")
            return
        self.result = result
        self.changes_count = len(result.changes) + (1 if result.api_key_changed else 0)
        self._save_outcome = ("saved", "")

    def set_save_failure(self, message: str) -> None:
        """Record an unexpected save failure (called by the overlay's worker)."""
        self._save_outcome = ("error", message or "Failed to save config.")

    def apply_save_outcome(self) -> None:
        """Apply the recorded save outcome as a stage transition (UI thread)."""
        outcome = self._save_outcome
        self._save_outcome = None
        if outcome is None:
            return
        kind, message = outcome
        if kind == "saved":
            self.saved = True
            if self._switch_target:
                self.switch_workspace = self._switch_target
            self._finish(True)
        else:
            self._switch_target = None
            self._goto("menu")
            self._set_status(message, "err")


__all__ = ["ConfigFlow"]
