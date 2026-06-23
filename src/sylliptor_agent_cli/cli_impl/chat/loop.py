# ruff: noqa: F821
# Dependencies are injected at runtime from sylliptor_agent_cli.cli to preserve monkeypatch surfaces.
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import typer

from ...compaction.conversation_compactor import CompactionState
from ...runtime_kind import RuntimeKind
from ...surface.console import safe_plain_error
from .state import _ChatExecutionRequest, _ChatPlanModeState, _ForgeChatState

_PROTECTED_GLOBAL_NAMES: set[str] = set()


def _sync_cli_globals(cli_mod: Any) -> None:
    module_globals = globals()
    if not _PROTECTED_GLOBAL_NAMES:
        for local_name, local_value in module_globals.items():
            if callable(local_value):
                _PROTECTED_GLOBAL_NAMES.add(local_name)
    for name, value in cli_mod.__dict__.items():
        if name.startswith("__") or name in _PROTECTED_GLOBAL_NAMES:
            continue
        module_globals[name] = value


def _path_binding_source(path_source: Any, path: Any) -> str:
    if path_source is not None and path_source is not ParameterSource.DEFAULT:
        return "explicit_path"
    if path not in (None, ".", Path(".")):
        return "explicit_path"
    return "cwd"


_SUBAGENT_DEFAULT_TASKS: dict[str, str] = {
    "explorer": "Summarize the repo structure, key entry points, and what to inspect next.",
    "reviewer": "Review current git diff/status and report risks, regressions, and missing tests.",
    "test-strategist": (
        "Propose the smallest high-value test plan for the current changes and edge cases."
    ),
}

_CHAT_WORKDIR_NAVIGATION_PREFIXES: tuple[str, ...] = (
    "from now on operate under ",
    "from now on operate in ",
    "from now on work under ",
    "from now on work in ",
    "switch to ",
    "work under ",
    "work in ",
    "operate under ",
    "operate in ",
    "go to ",
)
_CHAT_WORKDIR_TARGET_RE = re.compile(
    r"^(?P<path>\"[^\"]+\"|'[^']+'|[^,\s]+(?:[\\/][^,\s]+)*)(?:\s*(?:,?\s*(?:and|then)\s+)(?P<rest>.+))?$",
    re.IGNORECASE,
)
_CHAT_WORKDIR_FALSE_POSITIVE_TARGETS = {"definition"}
_STRICT_SHELL_SANDBOX_UNAVAILABLE_PREFIX = (
    "Shell sandbox strict mode is enabled, but no usable backend is available:"
)


def _is_default_shell_sandbox_startup_failure(*, cfg: Any, error: Exception) -> bool:
    if not str(error).startswith(_STRICT_SHELL_SANDBOX_UNAVAILABLE_PREFIX):
        return False
    if os.environ.get("SYLLIPTOR_SHELL_SANDBOX_MODE") is not None:
        return False
    extra_fields = getattr(cfg, "extra_fields", {})
    if not isinstance(extra_fields, dict):
        return True
    shell_cfg = extra_fields.get("shell_sandbox")
    if not isinstance(shell_cfg, dict):
        return True
    return shell_cfg.get("mode") is None


def _cfg_with_warn_shell_sandbox_mode(cfg: Any) -> Any:
    extra_fields = dict(getattr(cfg, "extra_fields", {}) or {})
    shell_cfg = dict(extra_fields.get("shell_sandbox") or {})
    shell_cfg["mode"] = "warn"
    extra_fields["shell_sandbox"] = shell_cfg
    return cfg.model_copy(update={"extra_fields": extra_fields}, deep=True)


def _resolve_forge_entry_root(*, session: Any, fallback_root: Path) -> Path:
    session_root = getattr(session, "root", None)
    if session_root is None:
        return Path(fallback_root).resolve()
    return Path(resolve_session_active_workdir_path(session)).resolve()


def _default_subagent_task(subagent_name: str) -> str:
    return _SUBAGENT_DEFAULT_TASKS.get(str(subagent_name or "").strip().lower(), "")


def _chat_plan_usage_lines() -> tuple[str, ...]:
    return (
        "[yellow]Usage:[/yellow] /plan <task>   default draft/review/approve flow; can execute after approval",
        "                 /plan mode     secondary persistent readonly planning overlay",
        "                 /plan approve  only inside Plan Mode; executes the stored draft",
        "                 /plan off",
        "                 /plan status",
        "[dim]Compatibility:[/dim] /plan draft <task>, /plan readonly, /plan on",
    )


def _chat_plan_already_on_message(*, plan_mode_escape_supported: bool) -> str:
    if plan_mode_escape_supported:
        return "Plan Mode already on. Press Esc at an empty prompt or use /plan off to leave."
    return "Plan Mode already on. Use /plan off to leave."


def _chat_plan_draft_blocked_by_mode_lines(*, plan_mode_escape_supported: bool) -> tuple[str, ...]:
    lines = [
        "Cannot start /plan while Plan Mode is on.",
        "Use /plan off first, then use /plan <task> for the default draft/review/approve flow.",
    ]
    if plan_mode_escape_supported:
        lines.append("Press Esc at an empty prompt to leave interactively.")
    return tuple(lines)


def _chat_plan_readonly_mode_guidance_lines() -> tuple[str, ...]:
    return (
        "Cannot start /plan in Read-Only mode.",
        "Switch to /mode review, /mode auto, or /mode fullaccess, then use /plan <task> for the default draft/review/approve flow.",
        "Use /plan mode only when you explicitly want persistent readonly planning.",
    )


_PLAN_MODE_EXECUTE_NOW_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|yes|yeah|yep|sure|please)\s+)?"
    r"(?:do it|go ahead|go for it|implement(?: it)?|execute(?: it)?|"
    r"start(?: implementing| coding)?|proceed|run it|apply it|ship it)"
    r"(?:\s+(?:now|then|please))?$",
    re.IGNORECASE,
)
_PLAN_MODE_NUMBERED_STEP_RE = re.compile(r"^\s{0,3}\d+[.)]\s+\S")
_PLAN_MODE_TASK_PREVIEW_CHARS = 96


def _apply_interactive_chat_step_budget_floor(
    effective: Any,
    *,
    max_steps_provided: bool,
) -> None:
    if max_steps_provided:
        return
    policy = str(getattr(effective, "step_budget_policy", "adaptive") or "adaptive").strip().lower()
    if policy != "adaptive":
        return
    default_chat_max_steps = int(globals().get("DEFAULT_CHAT_MAX_STEPS", 50) or 50)
    try:
        current_max_steps = int(getattr(effective, "max_steps", 0) or 0)
    except (TypeError, ValueError):
        current_max_steps = 0
    if current_max_steps < default_chat_max_steps:
        effective.max_steps = default_chat_max_steps


def _parameter_value_was_provided(value: Any, source: Any) -> bool:
    if value is None:
        return False
    if source is None:
        return True
    return source is not ParameterSource.DEFAULT


def _chat_plan_mode_latest_task(plan_mode_state: Any) -> str | None:
    task = getattr(plan_mode_state, "latest_task", None)
    clean = str(task or "").strip()
    return clean or None


def _chat_plan_mode_latest_draft(plan_mode_state: Any) -> str | None:
    draft = getattr(plan_mode_state, "latest_draft", None)
    clean = str(draft or "").strip()
    return clean or None


def _clear_chat_plan_mode_draft_state(plan_mode_state: Any) -> None:
    if hasattr(plan_mode_state, "latest_task"):
        plan_mode_state.latest_task = None
    if hasattr(plan_mode_state, "latest_draft"):
        plan_mode_state.latest_draft = None


def _store_chat_plan_mode_draft_state(
    plan_mode_state: Any,
    *,
    user_message: str,
    draft: str,
) -> bool:
    task = str(user_message or "").strip()
    clean_draft = str(draft or "").strip()
    previous_task = _chat_plan_mode_latest_task(plan_mode_state)
    previous_draft = _chat_plan_mode_latest_draft(plan_mode_state)
    if hasattr(plan_mode_state, "latest_task"):
        plan_mode_state.latest_task = task or None
    if hasattr(plan_mode_state, "latest_draft"):
        plan_mode_state.latest_draft = clean_draft or None
    return previous_task != (task or None) or previous_draft != (clean_draft or None)


def _chat_plan_task_preview(task: str | None) -> str:
    clean = re.sub(r"\s+", " ", str(task or "").strip())
    if len(clean) <= _PLAN_MODE_TASK_PREVIEW_CHARS:
        return clean
    return clean[: _PLAN_MODE_TASK_PREVIEW_CHARS - 3].rstrip() + "..."


def _looks_like_actionable_plan_mode_draft(text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return False
    steps = sum(1 for line in clean.splitlines() if _PLAN_MODE_NUMBERED_STEP_RE.match(line.strip()))
    return steps >= 2


def _latest_assistant_text_since(session: Any, *, start_index: int = 0) -> str | None:
    messages = getattr(session, "messages", None)
    if not isinstance(messages, list):
        return None
    start = max(int(start_index or 0), 0)
    for entry in reversed(messages[start:]):
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role") or "").strip().lower()
        if role != "assistant":
            continue
        text = str(entry.get("content") or "").strip()
        if text:
            return text
    return None


def _plan_mode_no_stored_draft_lines() -> tuple[str, ...]:
    return (
        "No stored actionable Plan Mode draft is available yet.",
        "Send the concrete implementation request as a normal chat message first.",
        "/plan <task> remains the default draft/review/approve path outside Plan Mode.",
        "Once the host captures a numbered draft, use exact /plan approve to execute it or /plan off to leave.",
    )


def _plan_mode_readonly_origin_lines() -> tuple[str, ...]:
    return (
        "This Plan Mode overlay was entered from plain Read-Only mode.",
        "Exact /plan approve cannot execute into a readonly session.",
        "Use /plan off to leave the overlay, then switch to /mode review, /mode auto, or /mode fullaccess and use /plan <task> for execution.",
    )


def _plan_mode_entry_guidance_lines(
    *,
    restore_mode: str,
) -> tuple[str, ...]:
    if restore_mode == "readonly":
        return (
            "Plan Mode is a persistent readonly planning overlay. It does not execute by itself.",
            "If you want readonly planning here, send the concrete implementation task as a normal chat message.",
            "For the default draft/review/approve flow, use /plan off, switch to /mode review, /mode auto, or /mode fullaccess, then use /plan <task>.",
        )
    return (
        "Plan Mode is a persistent readonly planning overlay. It does not execute by itself.",
        "For the default draft/review/approve flow, leave with /plan off and use /plan <task>.",
        "If you want readonly planning here, send the concrete implementation task as a normal chat message.",
        f"When the latest draft looks right, use exact /plan approve to leave Plan Mode, restore {_chat_mode_display(restore_mode)}, and execute it.",
        "Use /plan off to leave without execution.",
    )


def _plan_mode_execute_now_guidance_lines(
    *,
    plan_mode_state: Any,
    plan_mode_escape_supported: bool,
) -> tuple[str, ...]:
    restore_mode = _chat_plan_mode_restore_mode(plan_mode_state) or "readonly"
    latest_draft = _chat_plan_mode_latest_draft(plan_mode_state)
    lines = [
        "Plan Mode is still on and stays read-only.",
        "/plan <task> remains the default draft/review/approve path outside Plan Mode.",
    ]
    if latest_draft is None:
        lines.extend(_plan_mode_no_stored_draft_lines())
    elif restore_mode == "readonly":
        lines.append("A latest actionable draft is already stored for this session.")
        lines.extend(_plan_mode_readonly_origin_lines())
    else:
        lines.append("A latest actionable draft is already stored for this session.")
        lines.append(
            f"Use exact /plan approve to leave Plan Mode, restore {_chat_mode_display(restore_mode)}, and execute that draft."
        )
        lines.append("Use /plan off to leave without execution.")
    if plan_mode_escape_supported:
        lines.append("Press Esc at an empty prompt to leave interactively.")
    return tuple(lines)


def _is_plan_mode_execute_now_follow_up(user_message: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(user_message or "").strip())
    if not normalized:
        return False
    trimmed = normalized.strip(" \t\r\n.,;:!?")
    return bool(_PLAN_MODE_EXECUTE_NOW_RE.fullmatch(trimmed))


def _parse_chat_plan_command(raw_plan_arg: str) -> tuple[str, str]:
    clean = str(raw_plan_arg or "").strip()
    if not clean:
        return ("draft", "")
    lowered = clean.lower()
    if lowered in {"mode", "readonly", "on", "approve", "off", "status"}:
        return (lowered, "")
    if lowered == "draft":
        return ("draft", "")
    if lowered.startswith("draft "):
        return ("draft", clean[6:].strip())
    return ("draft", clean)


def _strip_wrapping_quotes(text: str) -> str:
    stripped = str(text or "").strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1].strip()
    return stripped


def _classify_chat_workdir_target(*, session: Any, raw_path: str) -> tuple[bool, str | None]:
    requested_path = str(raw_path or "").strip()
    if not requested_path:
        return (False, None)
    lowered = requested_path.casefold()
    if lowered in _CHAT_WORKDIR_FALSE_POSITIVE_TARGETS:
        return (False, None)
    pathlike = (
        "/" in requested_path
        or "\\" in requested_path
        or requested_path.startswith(".")
        or Path(requested_path).is_absolute()
    )
    workspace_root = Path(getattr(session, "root", Path("."))).resolve()
    current_workdir = Path(resolve_session_active_workdir_path(session))
    requested_obj = Path(requested_path)
    candidate = (
        requested_obj.resolve()
        if requested_obj.is_absolute()
        else (current_workdir / requested_obj).resolve()
    )
    try:
        candidate.relative_to(workspace_root)
    except ValueError:
        if pathlike:
            return (
                True,
                "Requested path escapes the bound workspace_root. Start a new session for another workspace.",
            )
        return (False, None)
    if not candidate.exists():
        if pathlike:
            return (True, f"Directory does not exist: {candidate}")
        return (False, None)
    if not candidate.is_dir():
        if pathlike:
            return (True, f"Path is not a directory: {candidate}")
        return (False, None)
    return (True, None)


def _parse_chat_workdir_navigation_request(
    *,
    input_text: str,
    session: Any,
) -> tuple[str, str] | tuple[str, str, str] | None:
    trimmed = str(input_text or "").strip()
    lowered = trimmed.casefold()
    prefix = next(
        (item for item in _CHAT_WORKDIR_NAVIGATION_PREFIXES if lowered.startswith(item)), None
    )
    if prefix is None:
        return None
    remainder = trimmed[len(prefix) :].strip()
    if not remainder:
        return None
    match = _CHAT_WORKDIR_TARGET_RE.match(remainder)
    if match is None:
        return None
    raw_requested_path = _strip_wrapping_quotes(match.group("path") or "")
    requested_path = (
        raw_requested_path
        if raw_requested_path in {".", ".."}
        else raw_requested_path.rstrip(".,;:!?")
    )
    if not requested_path:
        return None
    should_handle, error_message = _classify_chat_workdir_target(
        session=session,
        raw_path=requested_path,
    )
    if not should_handle:
        return None
    trailing_instruction = str(match.group("rest") or "").strip()
    if error_message:
        return (requested_path, trailing_instruction, error_message)
    return (requested_path, trailing_instruction)


def _forge_plan_command_guidance_lines(*args: Any, **kwargs: Any) -> Any:
    from . import rendering as _rendering

    _rendering._sync_rendering_globals(globals())
    return _rendering._forge_plan_command_guidance_lines(*args, **kwargs)


def _print_forge_plan_command_guidance(*args: Any, **kwargs: Any) -> Any:
    from . import rendering as _rendering

    _rendering._sync_rendering_globals(globals())
    return _rendering._print_forge_plan_command_guidance(*args, **kwargs)


def _open_assets_modal(*, session: Any, console: Console, run_paths: Any) -> None:
    cfg = getattr(session, "cfg", None)
    if not isinstance(cfg, AppConfig):
        cfg = load_config()
    try:
        from ..assets_modal import run_assets_modal

        run_assets_modal(cfg=cfg, run_paths=run_paths, console=console)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Assets modal failed:[/red] {exc}")


def _chat_skill_usage_lines(*args: Any, **kwargs: Any) -> Any:
    from . import rendering as _rendering

    _rendering._sync_rendering_globals(globals())
    return _rendering._chat_skill_usage_lines(*args, **kwargs)


def _artifact_display_ref(*args: Any, **kwargs: Any) -> Any:
    from . import rendering as _rendering

    _rendering._sync_rendering_globals(globals())
    return _rendering._artifact_display_ref(*args, **kwargs)


def _ensure_subagents_enabled_for_session(*, session: Any, console: Console) -> bool:
    if bool(getattr(session, "subagents_enabled", False)):
        return True
    session.subagents_enabled = True
    if hasattr(session, "cfg") and isinstance(session.cfg, AppConfig):
        session.cfg.subagents_enabled = True
    try:
        _rebuild_session_tools_for_mode(
            session=session,
            mode=str(getattr(session, "mode", "review")),
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Failed to enable subagents:[/red] {e}")
        return False
    console.print("[dim]Subagents enabled for this session.[/dim]")
    return True


def _render_explicit_subagent_panel(*args: Any, **kwargs: Any) -> Any:
    from . import rendering as _rendering

    _rendering._sync_rendering_globals(globals())
    return _rendering._render_explicit_subagent_panel(*args, **kwargs)


def _run_explicit_subagent(
    *,
    session: Any,
    console: Console,
    subagent_name: str,
    subagent_task: str,
) -> str:
    if not _ensure_subagents_enabled_for_session(session=session, console=console):
        return "handled"
    subagent_tool = getattr(session, "tools", {}).get("subagent_run")
    if subagent_tool is None:
        console.print("[red]subagent_run tool is unavailable in this session.[/red]")
        return "handled"
    try:
        result_raw = subagent_tool.run({"name": subagent_name, "task": subagent_task})
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Subagent failed:[/red] {e}")
        return "handled"
    result = result_raw if isinstance(result_raw, dict) else {"result": str(result_raw)}
    if "error" in result:
        err = str(result.get("error") or "Unknown error")
        available_obj = result.get("available_subagents")
        if isinstance(available_obj, list) and available_obj:
            available = ", ".join(str(item) for item in available_obj)
            console.print(f"[red]Subagent error:[/red] {err} [dim](available: {available})[/dim]")
        else:
            console.print(f"[red]Subagent error:[/red] {err}")
        return "handled"
    effective_name = str(result.get("subagent") or subagent_name)
    _render_explicit_subagent_panel(
        console=console,
        subagent_name=effective_name,
        result=result,
    )
    return "handled"


def _chat_subagent_rows(*args: Any, **kwargs: Any) -> Any:
    from . import rendering as _rendering

    _rendering._sync_rendering_globals(globals())
    return _rendering._chat_subagent_rows(*args, **kwargs)


def _chat_subagent_picker_panel(*args: Any, **kwargs: Any) -> Any:
    from . import rendering as _rendering

    _rendering._sync_rendering_globals(globals())
    return _rendering._chat_subagent_picker_panel(*args, **kwargs)


def _select_chat_subagent_interactive(
    *,
    registry: dict[str, Any],
    console: Console,
) -> tuple[str | None, bool]:
    rows = _chat_subagent_rows(registry=registry)
    if not rows:
        return None, True
    return _run_inline_option_selector(
        console=console,
        rows=rows,
        current_value=rows[0][0],
        panel_builder=lambda selected, interactive: _chat_subagent_picker_panel(
            registry=registry,
            selected_name=selected,
            interactive=interactive,
        ),
        unavailable_label="Subagent picker",
    )


def _chat_subagent_usage_panel(*args: Any, **kwargs: Any) -> Any:
    from . import rendering as _rendering

    _rendering._sync_rendering_globals(globals())
    return _rendering._chat_subagent_usage_panel(*args, **kwargs)


def _resolve_subagent_from_guided_flow(*, session: Any, console: Console) -> tuple[str, str] | None:
    registry_obj = getattr(session, "subagent_registry", None)
    registry = registry_obj if isinstance(registry_obj, dict) else {}
    if not registry:
        console.print("No subagents available.")
        return None
    selected_name, picker_available = _select_chat_subagent_interactive(
        registry=registry,
        console=console,
    )
    if not picker_available:
        console.print(_chat_subagent_usage_panel(registry=registry))
        return None
    if selected_name is None:
        return None
    default_task = _default_subagent_task(selected_name)
    prompt_label = f"Task for {selected_name}"
    try:
        task_value = typer.prompt(prompt_label, default=default_task).strip()
    except (EOFError, KeyboardInterrupt):
        console.print("")
        return None
    if not task_value:
        task_value = default_task
    if not task_value:
        console.print("[yellow]Task cannot be empty.[/yellow]")
        return None
    return selected_name, task_value


def _render_planner_reply(*args: Any, **kwargs: Any) -> Any:
    from . import rendering as _rendering

    _rendering._sync_rendering_globals(globals())
    return _rendering._render_planner_reply(*args, **kwargs)


def _render_labeled_chat_message(*args: Any, **kwargs: Any) -> Any:
    from . import rendering as _rendering

    _rendering._sync_rendering_globals(globals())
    return _rendering._render_labeled_chat_message(*args, **kwargs)


def _render_plan_draft(*args: Any, **kwargs: Any) -> Any:
    from . import rendering as _rendering

    _rendering._sync_rendering_globals(globals())
    return _rendering._render_plan_draft(*args, **kwargs)


def _chat_plan_mode_enabled(plan_mode_state: Any) -> bool:
    return bool(getattr(plan_mode_state, "enabled", False))


def _chat_plan_mode_restore_mode(plan_mode_state: Any) -> str | None:
    restore_mode = getattr(plan_mode_state, "restore_mode", None)
    if restore_mode is None:
        return None
    normalized = str(restore_mode).strip().lower()
    return normalized or None


def _render_chat_plan_mode_status(*args: Any, **kwargs: Any) -> Any:
    from . import rendering as _rendering

    _rendering._sync_rendering_globals(globals())
    return _rendering._render_chat_plan_mode_status(*args, **kwargs)


def _disable_chat_plan_mode(
    *,
    session: Any,
    console: Console,
    plan_mode_state: Any,
    clear_draft: bool = True,
) -> str | None:
    restore_mode = _chat_plan_mode_restore_mode(plan_mode_state) or "review"
    current_mode = str(getattr(session, "mode", "review")).strip().lower() or "review"
    previous_task = _chat_plan_mode_latest_task(plan_mode_state)
    previous_draft = _chat_plan_mode_latest_draft(plan_mode_state)
    plan_mode_state.enabled = False
    plan_mode_state.restore_mode = None
    if clear_draft:
        _clear_chat_plan_mode_draft_state(plan_mode_state)
    if current_mode != restore_mode:
        try:
            _apply_chat_effective_mode(
                session=session,
                next_mode=restore_mode,
                persist_default_mode=False,
            )
        except Exception as e:  # noqa: BLE001
            plan_mode_state.enabled = True
            plan_mode_state.restore_mode = restore_mode
            if hasattr(plan_mode_state, "latest_task"):
                plan_mode_state.latest_task = previous_task
            if hasattr(plan_mode_state, "latest_draft"):
                plan_mode_state.latest_draft = previous_draft
            console.print(f"[red]Failed to disable Plan Mode:[/red] {e}")
            return None
    console.print(
        f"Plan Mode set for this session: off (restored {_chat_mode_display(restore_mode)})"
    )
    return restore_mode


def _record_plan_mode_draft_from_turn(
    *,
    session: Any,
    console: Console,
    plan_mode_state: Any,
    user_message: str,
    start_index: int,
) -> None:
    latest_reply = _latest_assistant_text_since(session, start_index=start_index)
    if latest_reply is None or not _looks_like_actionable_plan_mode_draft(latest_reply):
        return
    changed = _store_chat_plan_mode_draft_state(
        plan_mode_state,
        user_message=user_message,
        draft=latest_reply,
    )
    if not changed:
        return
    restore_mode = _chat_plan_mode_restore_mode(plan_mode_state) or "readonly"
    task_preview = _chat_plan_task_preview(user_message)
    console.print(f"[dim]Stored latest Plan Mode draft for:[/dim] {task_preview}")
    if restore_mode == "readonly":
        console.print(
            "[dim]This overlay started from Read-Only mode, so exact /plan approve cannot execute it here.[/dim]"
        )
        return
    console.print(
        f"[dim]Use exact /plan approve to leave Plan Mode, restore {_chat_mode_display(restore_mode)}, and execute it.[/dim]"
    )


def _apply_chat_effective_mode(
    *,
    session: Any,
    next_mode: str,
    persist_default_mode: bool,
) -> None:
    _rebuild_session_tools_for_mode(session=session, mode=next_mode)
    session.mode = next_mode
    if persist_default_mode and hasattr(session, "cfg"):
        session.cfg.default_mode = next_mode
    refresh_session_environment_context_message(session)
    surface = getattr(session, "surface", None)
    emit_mode_changed = getattr(surface, "emit_mode_changed", None)
    if callable(emit_mode_changed):
        emit_mode_changed(next_mode)


def _apply_config_menu_changes_to_session(*, session: Any, cfg: AppConfig) -> None:
    from ...config import (
        ConfigError,
        clone_cfg,
        resolve_api_key,
        resolve_llm_enable_thinking,
        resolve_llm_reasoning_effort,
        resolve_llm_timeout_s,
        resolve_prompt_cache_key,
        resolve_prompt_cache_retention,
        resolve_role_temperature,
    )
    from ...llm.provider_limits import resolve_provider_retry_settings
    from ...model_registry import ModelRegistry, resolve_model_provider_key
    from ...model_router import ROLE_CODING, ROLE_COMPACTOR, resolve_model_for_role
    from ...profiles import get_active_profile, resolve_effective_base_url

    session.cfg = clone_cfg(cfg)
    active_profile = get_active_profile(session.cfg)
    effective_base_url = resolve_effective_base_url(cfg=session.cfg, profile=active_profile)
    resolved_key = resolve_api_key(session.cfg)
    session.api_key = str(resolved_key.key or "")
    session.api_key_source = resolved_key.source

    timeout_s = resolve_llm_timeout_s(session.cfg)
    enable_thinking = resolve_llm_enable_thinking(session.cfg)
    reasoning_effort = resolve_llm_reasoning_effort(session.cfg)
    prompt_cache_key = resolve_prompt_cache_key(session.cfg)
    prompt_cache_retention = resolve_prompt_cache_retention(session.cfg)
    coding_temperature = resolve_role_temperature(session.cfg, role=ROLE_CODING)
    provider_retry_settings = resolve_provider_retry_settings(session.cfg)

    def _apply_client_config(client: Any, *, model: str, temperature: float | None = None) -> None:
        client.base_url = effective_base_url
        client.api_key = session.api_key
        client.model = model
        client.timeout_s = timeout_s
        if temperature is not None:
            client.temperature = temperature
        client.prompt_cache_key = prompt_cache_key
        client.prompt_cache_retention = prompt_cache_retention
        client.enable_thinking = enable_thinking
        client.reasoning_effort = reasoning_effort
        if hasattr(client, "extra_headers"):
            client.extra_headers = dict(active_profile.extra_headers)
        if hasattr(client, "provider_key"):
            client.provider_key = resolve_model_provider_key(
                cfg=session.cfg,
                model_name=model,
                base_url=effective_base_url,
                profile_name=active_profile.name,
            )
        if hasattr(client, "provider_concurrency_caps"):
            client.provider_concurrency_caps = dict(session.cfg.provider_concurrency_caps)
        if hasattr(client, "provider_retry_settings"):
            client.provider_retry_settings = provider_retry_settings

    client = getattr(session, "client", None)
    if client is not None:
        _apply_client_config(
            client,
            model=str(session.cfg.model or ""),
            temperature=coding_temperature,
        )

    router_client = getattr(session, "router_client", None)
    if router_client is not None:
        _apply_client_config(router_client, model=str(session.cfg.model or ""))

    compactor = getattr(session, "conversation_compactor", None)
    compactor_client = getattr(compactor, "compactor_client", None)
    if compactor_client is not None:
        compactor_model = str(session.cfg.model or "")
        try:
            compactor_model = resolve_model_for_role(
                cfg=session.cfg,
                role=ROLE_COMPACTOR,
                plan=None,
            )
        except ConfigError:
            compactor_model = str(session.cfg.model or "")
        _apply_client_config(
            compactor_client,
            model=compactor_model,
            temperature=resolve_role_temperature(session.cfg, role=ROLE_COMPACTOR),
        )

    session.model_registry = ModelRegistry(cfg=session.cfg, api_key=session.api_key)
    _rebuild_session_tools_for_mode(
        session=session,
        mode=str(getattr(session, "mode", "review") or "review"),
    )
    refresh_session_environment_context_message(session)
    _refresh_chat_hud_context_cache(session)


def _resolve_interactive_plan_mode_request(
    *,
    session: Any,
    console: Console,
    plan_mode_state: Any,
    user_message: str,
    plan_mode_escape_supported: bool = False,
) -> str | _ChatExecutionRequest:
    _ = session
    if _is_plan_mode_execute_now_follow_up(user_message):
        for line in _plan_mode_execute_now_guidance_lines(
            plan_mode_state=plan_mode_state, plan_mode_escape_supported=plan_mode_escape_supported
        ):
            console.print(line)
        return "handled"
    from ...interactive_plan_mode import INTERACTIVE_PLAN_MODE_SYSTEM_PROMPT

    return _ChatExecutionRequest(
        instruction=user_message,
        routing_mode_override="code_only",
        ephemeral_system_messages=(INTERACTIVE_PLAN_MODE_SYSTEM_PROMPT,),
        plan_mode_capture_task=user_message,
    )


def _clone_chat_startup_messages(session: Any) -> list[dict[str, Any]]:
    startup_messages_obj = getattr(session, "startup_messages", None)
    if isinstance(startup_messages_obj, list) and startup_messages_obj:
        cloned = [dict(message) for message in startup_messages_obj if isinstance(message, dict)]
        if cloned:
            return cloned

    messages_obj = getattr(session, "messages", None)
    pinned_prefix_len = max(0, int(getattr(session, "pinned_prefix_len", 0) or 0))
    if not isinstance(messages_obj, list) or pinned_prefix_len <= 0:
        return []
    return [
        dict(message) for message in messages_obj[:pinned_prefix_len] if isinstance(message, dict)
    ]


def _clear_chat_conversation(*, session: Any, pending_images: list[str]) -> None:
    startup_messages = _clone_chat_startup_messages(session)
    if not startup_messages:
        raise RuntimeError("session startup messages unavailable")

    store = getattr(session, "store", None)
    if store is None or not hasattr(store, "append"):
        raise RuntimeError("session store unavailable")
    store.append("conversation_cleared", {"trigger": "user_command"})

    session.messages = startup_messages
    refresh_session_workspace_binding_context_message(session)
    refresh_session_environment_context_message(session)

    compactor = getattr(session, "conversation_compactor", None)
    if compactor is not None and hasattr(compactor, "state"):
        pinned_prefix_len = max(0, int(getattr(compactor.state, "pinned_prefix_len", 0) or 0))
        compactor.state = CompactionState(
            summary={},
            history_chunk_index=0,
            memory_message_index=None,
            pinned_prefix_len=pinned_prefix_len,
            pins=[],
            pins_message_index=None,
        )

    pending_images.clear()
    _refresh_chat_hud_context_cache(session)


def _handle_chat_command(*args: Any, **kwargs: Any) -> Any:
    from . import commands as _commands

    _commands._sync_command_globals(globals())
    return _commands._handle_chat_command(*args, **kwargs)


def _handle_forge_chat_command(*args: Any, **kwargs: Any) -> Any:
    from . import commands as _commands

    _commands._sync_command_globals(globals())
    return _commands._handle_forge_chat_command(*args, **kwargs)


def _planner_workspace_context_for_session(*, session: Any) -> dict[str, Any] | None:
    cached = getattr(session, "planner_workspace_context", None)
    if isinstance(cached, dict):
        return cached
    root_obj = getattr(session, "root", None)
    if root_obj is None:
        return None
    try:
        workspace_root = Path(root_obj)
        workspace_context = resolve_workspace_context(workspace_root)
        scan = scan_workspace(context=workspace_context)
        payload = scan.to_dict()
    except Exception:  # noqa: BLE001
        return None
    try:
        session.planner_workspace_context = payload
    except Exception:  # noqa: BLE001
        pass
    return payload


def _finish_chat_surface_activity(*, session: Any) -> None:
    surface = getattr(session, "surface", None)
    handler = getattr(surface, "on_assistant_message_done", None)
    if not callable(handler):
        return
    try:
        handler("")
    except Exception:  # noqa: BLE001
        return


def _run_plan_mode_approval_loop(
    *,
    session: Any,
    console: Console,
    user_message: str,
    max_iterations: int | None = None,
) -> str | None:
    limit = max_iterations if max_iterations is not None else MAX_PLAN_ITERATIONS
    if limit <= 0:
        return None

    try:
        from ...llm.types import LLMError as _PlanLLMError
    except Exception:  # noqa: BLE001
        _PlanLLMError = RuntimeError

    previous_plan: str | None = None
    feedback: str | None = None
    for _iteration in range(limit):
        console.print("")
        _render_labeled_chat_message(console=console, label="Request", message=user_message)
        if feedback:
            _render_labeled_chat_message(
                console=console,
                label="Revision feedback",
                message=feedback,
            )
        console.print("")

        if previous_plan and feedback:
            _emit_plan_mode_trace(
                session=session,
                message="Revising draft plan with your feedback.",
            )
        else:
            _emit_plan_mode_trace(
                session=session,
                message="Drafting execution plan for your request.",
            )
        _emit_plan_mode_trace(
            session=session,
            message="Collecting relevant conversation context for planning.",
            full_only=True,
        )
        _emit_plan_mode_trace(
            session=session,
            message="Tools stay disabled while planning this draft.",
            full_only=True,
        )

        session_stream = bool(getattr(session, "stream", False))
        on_text_delta = (
            _make_plan_mode_delta_trace_callback(session=session) if session_stream else None
        )
        stream_plan_draft = session_stream and on_text_delta is not None
        details: dict[str, Any] = {}

        try:
            draft = generate_plan_draft(
                client=getattr(session, "client", None),
                session_messages=list(getattr(session, "messages", []) or []),
                user_message=user_message,
                previous_plan=previous_plan,
                feedback=feedback,
                workspace_context=_planner_workspace_context_for_session(session=session),
                stream=stream_plan_draft,
                on_text_delta=on_text_delta if stream_plan_draft else None,
                details=details,
            )
        except KeyboardInterrupt:
            _finish_chat_surface_activity(session=session)
            console.print("Plan drafting interrupted. Back to chat.")
            return None
        except (RuntimeError, _PlanLLMError) as e:
            if not stream_plan_draft:
                _finish_chat_surface_activity(session=session)
                console.print(f"[red]{e}[/red]")
                return None

            _emit_plan_mode_trace(
                session=session,
                message="Planner stream failed; retrying once without streaming.",
                full_only=True,
            )
            details = {}
            try:
                draft = generate_plan_draft(
                    client=getattr(session, "client", None),
                    session_messages=list(getattr(session, "messages", []) or []),
                    user_message=user_message,
                    previous_plan=previous_plan,
                    feedback=feedback,
                    workspace_context=_planner_workspace_context_for_session(session=session),
                    stream=False,
                    on_text_delta=None,
                    details=details,
                )
            except KeyboardInterrupt:
                _finish_chat_surface_activity(session=session)
                console.print("Plan drafting interrupted. Back to chat.")
                return None
            except (RuntimeError, _PlanLLMError) as retry_error:
                _finish_chat_surface_activity(session=session)
                console.print(f"[red]{retry_error}[/red]")
                return None
        _emit_plan_mode_trace(session=session, message="Plan draft ready for review.")

        request_messages = details.get("request_messages")
        response = details.get("response")
        if isinstance(request_messages, list) and response is not None:
            record_plan_usage(
                session=session,
                request_messages=request_messages,
                response=response,
            )
            _refresh_chat_hud_context_cache(session)

        _finish_chat_surface_activity(session=session)
        _render_plan_draft(console=console, draft=draft)
        console.print("")

        action = _prompt_plan_mode_action(console=console)
        if action is None:
            return None
        if action == "approve":
            return instruction_with_approved_plan(user_message=user_message, approved_plan=draft)
        if action == "propose":
            feedback_input = _prompt_plan_mode_feedback(console=console)
            if feedback_input is None:
                return None
            if not feedback_input.strip():
                console.print("[yellow]Feedback cannot be empty.[/yellow]")
                continue
            previous_plan = draft
            feedback = feedback_input.strip()
            _emit_plan_mode_trace(
                session=session,
                message="Captured feedback and preparing a revised draft.",
                full_only=True,
            )
            console.print("[bold]Regenerating plan with your feedback...[/bold]")
            continue

        console.print("Discarded plan. What do you want to build next?")
        return None
    console.print("[yellow]Plan iteration limit reached. Returning to prompt.[/yellow]")
    return None


def _print_chat_context(*args: Any, **kwargs: Any) -> Any:
    from . import rendering as _rendering

    _rendering._sync_rendering_globals(globals())
    return _rendering._print_chat_context(*args, **kwargs)


def chat(
    path: Path = typer.Option(Path("."), "--path", help="Working directory/root."),
    create_path: bool = typer.Option(
        False,
        "--create-path",
        help="Create --path if it does not exist before binding the workspace.",
    ),
    allow_broad_workspace: bool = typer.Option(
        False,
        "--allow-broad-workspace",
        help="Allow guarded broad workspaces in non-interactive startup flows.",
    ),
    image: list[Path] | None = typer.Option(
        None,
        "--image",
        help="Queue image path(s) for the next message. Repeat --image for multiple files.",
    ),
    mode: Mode | None = typer.Option(None, "--mode", help="Mode override."),
    model: str | None = typer.Option(None, "--model", help="Model override."),
    base_url: str | None = typer.Option(None, "--base-url", help="Base URL override."),
    temperature: float | None = typer.Option(None, "--temperature", help="Sampling temperature."),
    stream: bool | None = typer.Option(
        None,
        "--stream/--no-stream",
        help="Enable streamed assistant output.",
    ),
    max_steps: int | None = typer.Option(
        None,
        "--max-steps",
        help="Max steps override (per user turn).",
    ),
    subagents: bool | None = typer.Option(
        None,
        "--subagents/--no-subagents",
        help="Enable or disable subagent delegation for this session.",
    ),
    no_log: bool = typer.Option(False, "--no-log", help="Disable JSONL session logging."),
    verify_cmd: list[str] | None = typer.Option(
        None,
        "--verify-cmd",
        help="Override verification command for this chat session (repeatable).",
    ),
    api_key_env: str | None = typer.Option(
        None,
        "--api-key-env",
        help=(
            "Read API key from this environment variable (overrides SYLLIPTOR_API_KEY/OPENAI_API_KEY)."
        ),
    ),
    api_key_stdin: bool = typer.Option(
        False,
        "--api-key-stdin",
        help="Prompt for API key (hidden input). Key is kept in memory for this run only.",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help=(
            "UNSAFE: Provide API key via CLI argument (may leak via shell history / process list). "
            "Prefer --api-key-stdin or --api-key-env."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="In auto mode, skip confirmations for sensitive commands.",
    ),
    cli_ctx: Any = None,
) -> None:
    from ...llm.types import LLMError

    console = _console()
    requested_path = path
    cfg = load_config()
    effective = clone_cfg(cfg)
    if base_url is not None:
        effective.base_url = base_url
    if model is not None:
        effective.model = model
    if temperature is not None:
        _apply_temperature_override(effective, temperature)
    current_ctx = cli_ctx if cli_ctx is not None else get_current_context(silent=True)
    non_interactive = _is_non_interactive_terminal()
    stream_source = current_ctx.get_parameter_source("stream") if current_ctx is not None else None
    path_source = current_ctx.get_parameter_source("path") if current_ctx is not None else None
    max_steps_source = (
        current_ctx.get_parameter_source("max_steps") if current_ctx is not None else None
    )
    stream_provided = stream is not None
    max_steps_provided = max_steps is not None
    if current_ctx is not None:
        stream_provided = stream_source is not None and stream_source is not ParameterSource.DEFAULT
        max_steps_provided = _parameter_value_was_provided(max_steps, max_steps_source)
    if stream_provided and stream is not None:
        effective.stream = stream
    else:
        # Interactive chat is more readable with incremental output by default.
        effective.stream = True
    if max_steps is not None:
        effective.max_steps = max_steps
    if subagents is not None:
        effective.subagents_enabled = subagents
    _apply_interactive_chat_step_budget_floor(
        effective,
        max_steps_provided=max_steps_provided,
    )

    effective_mode = (mode.value if mode else effective.default_mode) or "review"
    binding_source = _path_binding_source(path_source, requested_path)

    try:
        if not effective.model:
            raise ConfigError("Model is not set. Run: sylliptor config set model <MODEL>")
        api_key_override = _resolve_api_key_override(
            api_key=api_key,
            api_key_env=api_key_env,
            api_key_stdin=api_key_stdin,
        )
        workspace_binding = _resolve_startup_workspace_binding(
            requested_path=requested_path,
            console=console,
            interactive=not non_interactive,
            create_if_missing=create_path,
            allow_broad_workspace=allow_broad_workspace,
            source=binding_source,
            action=WorkspaceAction.CHAT,
        )
        session_root = workspace_binding.workspace_context.workspace_root
        focus_path = workspace_binding.workspace_context.focus_path
        # Full-screen TUI is the default interactive chat surface. Users can
        # set SYLLIPTOR_TUI=0 to fall back to the classic prompt loop below.
        if not non_interactive:
            from ..tui import is_tui_enabled as _tui_enabled

            if _tui_enabled():
                import getpass as _getpass

                from ...git_ops import current_branch as _current_branch
                from ..commands.welcome import _welcome_workspace_value
                from ..tui import TuiState as _TuiState
                from ..tui import run_tui as _run_tui

                try:
                    _username = _getpass.getuser()
                except Exception:
                    _username = ""
                _ws_path = focus_path if focus_path != session_root else session_root
                try:
                    _workspace = _welcome_workspace_value(_ws_path)
                except Exception:
                    _workspace = ""
                try:
                    _branch = _current_branch(session_root)
                except Exception:
                    _branch = ""
                _tui_state = _TuiState(
                    model_name=str(effective.model or ""),
                    username=_username,
                    workspace=_workspace,
                    branch=_branch,
                    exec_mode=str(effective_mode or "").strip(),
                    # Approval gate must default OFF (= classic behaviour): otherwise
                    # TuiSurface.request_approval auto-approves everything and the
                    # execution mode (safe/review prompts) is silently bypassed.
                    # --yes opts into approve-everything; Shift+Tab toggles it live.
                    auto_approve=bool(yes),
                )
                _tui_box: dict[str, Any] = {"session": None}

                def _tui_session_builder(surface: Any) -> Any:
                    built = create_session(
                        cfg=effective,
                        root=session_root,
                        mode=effective_mode,
                        runtime_kind=RuntimeKind.INTERACTIVE_CHAT,
                        yes=yes,
                        max_steps=effective.max_steps,
                        no_log=no_log,
                        api_key_override=api_key_override,
                        console=None,
                        surface=surface,
                        non_interactive=non_interactive,
                        enable_chat_turn_step_budget=True,
                        chat_turn_fixed_override=(
                            effective.max_steps if max_steps_provided else None
                        ),
                        verify_cmd=verify_cmd,
                        subagents_enabled=effective.subagents_enabled,
                        workspace_binding=workspace_binding,
                    )
                    _set_chat_usage_hud_enabled(built, _resolve_usage_hud_default(effective))
                    _refresh_chat_hud_context_cache(built)
                    _tui_box["session"] = built
                    return built

                def _tui_on_turn_complete() -> None:
                    built = _tui_box.get("session")
                    if built is None:
                        return
                    from ..commands.startup import (
                        _chat_context_percent_value,
                        _known_cost_value,
                    )

                    try:
                        _refresh_chat_hud_context_cache(built)
                        pct = _chat_context_percent_value(built)
                        if pct is not None:
                            _tui_state.context_pct = float(pct)
                    except Exception:
                        pass
                    try:
                        summary = getattr(built, "usage_summary", None)
                        if summary is not None:
                            totals = summary.totals()
                            _tui_state.tokens = int(totals.get("total_tokens") or 0)
                            cost = _known_cost_value(totals)
                            if cost is not None:
                                _tui_state.cost_usd = float(cost)
                    except Exception:
                        pass

                def _tui_config_flow_factory() -> Any:
                    # Built fresh each time bare /config opens, so it always reflects
                    # the on-disk config (and any edits applied since launch). The live
                    # session root is passed for display + so "switch project" knows
                    # what it's leaving.
                    from ..tui.config_flow import ConfigFlow

                    built = _tui_box.get("session")
                    root = str(getattr(built, "root", "") or "") if built is not None else ""
                    return ConfigFlow(current_workspace=root)

                def _tui_on_config_saved() -> bool:
                    # After the overlay saves, reload the live session from disk so the
                    # new model/keys/limits apply on the next turn (same reload the
                    # classic /config uses), and refresh the footer's model badge.
                    # Returns False if the reload failed so the overlay can warn the
                    # user the running session was NOT updated (parity with classic).
                    built = _tui_box.get("session")
                    if built is None:
                        return True
                    try:
                        reloaded = load_config()
                        _apply_config_menu_changes_to_session(session=built, cfg=reloaded)
                        _tui_state.model_name = str(getattr(reloaded, "model", "") or "")
                        new_mode = str(getattr(built, "mode", "") or "").strip()
                        if new_mode:
                            _tui_state.exec_mode = new_mode
                    except Exception:
                        return False
                    try:
                        _refresh_chat_hud_context_cache(built)
                    except Exception:
                        pass
                    return True

                _tui_forge_state = _ForgeChatState()
                _tui_plan_state = _ChatPlanModeState()

                def _tui_make_forge_execute(command_text: str):
                    # Build the callable the worker thread runs for "/execute plan":
                    # it opens the live forge view, runs the swarm (whose events
                    # stream into that view via TuiSurface.on_swarm_event), and
                    # surfaces the handler's captured warnings/summary at the end.
                    def _run(token: Any) -> None:
                        import io as _io

                        from rich.console import Console as _RichConsole

                        built = _tui_box.get("session")
                        if built is None:
                            return
                        surface = getattr(built, "surface", None)
                        paths = getattr(_tui_forge_state, "paths", None)
                        buf = _io.StringIO()
                        cap = _RichConsole(
                            file=buf,
                            force_terminal=False,
                            no_color=True,
                            highlight=False,
                            width=100,
                        )
                        if surface is not None and paths is not None:
                            try:
                                # Pass the worker token so the trace sink's own thread
                                # can drop events after a soft-interrupt.
                                surface.begin_forge(paths, token)
                            except Exception:
                                pass
                        try:
                            _handle_chat_command(
                                input_text=command_text,
                                root=focus_path,
                                session=built,
                                pending_images=[],
                                console=cap,
                                forge_state=_tui_forge_state,
                                plan_mode_state=_tui_plan_state,
                                plan_mode_escape_supported=False,
                            )
                        except Exception as _exec_exc:  # noqa: BLE001
                            if surface is not None:
                                try:
                                    surface.append_system(f"Forge execution failed: {_exec_exc}")
                                except Exception:
                                    pass
                        finally:
                            cancelled = bool(getattr(token, "is_cancelled", False))
                            if surface is not None:
                                out = buf.getvalue().rstrip("\n")
                                if out and not cancelled:
                                    try:
                                        surface.append_system(out)
                                    except Exception:
                                        pass
                                try:
                                    surface.end_forge()
                                except Exception:
                                    pass

                    return _run

                def _tui_command_runner(
                    sess: Any, text: str, width: int
                ) -> tuple[str, str, str | None, dict[str, Any] | None]:
                    # "/execute plan" inside Forge runs the swarm — defer it to the
                    # worker thread (not the synchronous runner, which would freeze
                    # the UI) with a live forge view instead of a flat captured dump.
                    _exec_parts = text.strip().lower().split()
                    if (
                        _tui_forge_state.ui_mode == "forge"
                        and len(_exec_parts) >= 2
                        and _exec_parts[0] == "/execute"
                        and _exec_parts[1] == "plan"
                    ):
                        return (
                            "run",
                            "",
                            text.strip(),
                            {"_forge_execute": _tui_make_forge_execute(text.strip())},
                        )
                    # Route a submission through the chat command handler with a
                    # capture console so its output renders into the TUI transcript
                    # instead of corrupting the alt-screen. stdin is swapped to EOF
                    # so any interactive command (e.g. /resume) cancels cleanly.
                    import io as _io
                    import sys as _sys

                    from rich.console import Console as _RichConsole

                    buf = _io.StringIO()
                    cap = _RichConsole(
                        file=buf,
                        force_terminal=False,
                        no_color=True,
                        highlight=False,
                        width=max(20, min(int(width or 100), 120)),
                    )
                    saved_stdin = _sys.stdin
                    try:
                        _sys.stdin = _io.StringIO("")
                        result = _handle_chat_command(
                            input_text=text,
                            root=focus_path,
                            session=sess,
                            pending_images=[],
                            console=cap,
                            forge_state=_tui_forge_state,
                            plan_mode_state=_tui_plan_state,
                            plan_mode_escape_supported=False,
                        )
                    except Exception as _cmd_exc:  # noqa: BLE001
                        return ("handled", f"Command error: {_cmd_exc}", None, None)
                    finally:
                        _sys.stdin = saved_stdin
                    output = buf.getvalue().rstrip("\n")
                    # Keep the footer mode badge in sync after any command that may
                    # have changed it (e.g. "/mode fast" applied inline).
                    _tui_state.exec_mode = str(getattr(sess, "mode", "") or "").strip()
                    # The forge state machine flips ui_mode inside _enter_forge_mode
                    # (and back to "chat" on /back / /done); this single sync drives
                    # the FORGE footer badge + the forge-specific input placeholder.
                    _tui_state.forge_mode = _tui_forge_state.ui_mode == "forge"
                    if _tui_state.forge_mode:
                        _forge_paths = getattr(_tui_forge_state, "paths", None)
                        _run_id = str(getattr(_forge_paths, "run_id", "") or "")
                        _tui_state.forge_run_id = (
                            _run_id if len(_run_id) <= 12 else _run_id[:11] + "…"
                        )
                    else:
                        _tui_state.forge_run_id = ""
                    if result == "exit":
                        return ("exit", output, None, None)
                    if result == "send":
                        return ("run", output, text, {})
                    if isinstance(result, _ChatExecutionRequest):
                        run_kwargs: dict[str, Any] = {}
                        if result.routing_mode_override:
                            run_kwargs["routing_mode_override"] = result.routing_mode_override
                        if result.ephemeral_system_messages:
                            run_kwargs["ephemeral_system_messages"] = list(
                                result.ephemeral_system_messages
                            )
                        if result.ephemeral_user_messages:
                            run_kwargs["ephemeral_user_messages"] = list(
                                result.ephemeral_user_messages
                            )
                        return ("run", output, result.instruction, run_kwargs)
                    return ("handled", output, None, None)

                from ..commands.chat_status import _chat_status_panel_spec
                from ..commands.chat_tui_panels import (
                    _chat_asset_detail_panel_spec,
                    _chat_assets_picker_spec,
                    _chat_config_panel_spec,
                    _chat_context_panel_spec,
                    _chat_forge_intro_panel_spec,
                    _chat_forge_markdown_panel_spec,
                    _chat_forge_plan_panel_spec,
                    _chat_model_info_panel_spec,
                    _chat_skill_listing_panel_spec,
                    _chat_terminals_panel_spec,
                    _chat_toolbar_panel_spec,
                    _chat_usage_panel_spec,
                    _short_subagent_desc,
                )

                def _tui_status_panel(arg: str = "") -> dict[str, Any] | None:
                    # TUI-native /status: render the live session snapshot as a
                    # centered popup instead of a flat gray table dump.
                    built = _tui_box.get("session")
                    if built is None:
                        return None
                    try:
                        return _chat_status_panel_spec(session=built, pending_images=[])
                    except Exception:  # noqa: BLE001 - never crash the UI
                        return None

                def _tui_usage_panel(arg: str = "") -> dict[str, Any] | None:
                    # Bare /usage opens the panel; "/usage hud …" falls through to
                    # the command runner so the HUD toggle still applies.
                    if arg.strip():
                        return None
                    built = _tui_box.get("session")
                    if built is None:
                        return None
                    try:
                        return _chat_usage_panel_spec(session=built)
                    except Exception:  # noqa: BLE001
                        return None

                def _tui_context_panel(arg: str = "") -> dict[str, Any] | None:
                    if arg.strip():
                        return None
                    built = _tui_box.get("session")
                    if built is None:
                        return None
                    try:
                        return _chat_context_panel_spec(session=built)
                    except Exception:  # noqa: BLE001
                        return None

                def _tui_model_info_panel(arg: str = "") -> dict[str, Any] | None:
                    built = _tui_box.get("session")
                    if built is None:
                        return None
                    try:
                        return _chat_model_info_panel_spec(session=built, model_ref=arg)
                    except Exception:  # noqa: BLE001
                        return None

                def _tui_config_panel(arg: str = "") -> dict[str, Any] | None:
                    # Bare /config now opens the full interactive menu overlay (handled
                    # in app._submit before this provider). This read-only model panel
                    # serves /config show|list|help (the tracked-model view) and acts as
                    # the fallback when the overlay is unavailable; set|clear|rm|delete
                    # fall through to the command runner so edits still apply.
                    normalized = arg.strip().lower()
                    if normalized and normalized not in {"show", "list", "help"}:
                        return None
                    built = _tui_box.get("session")
                    if built is None:
                        return None
                    try:
                        return _chat_config_panel_spec(session=built)
                    except Exception:  # noqa: BLE001
                        return None

                def _tui_toolbar_panel(arg: str = "") -> dict[str, Any] | None:
                    if arg.strip():
                        return None
                    built = _tui_box.get("session")
                    if built is None:
                        return None
                    try:
                        return _chat_toolbar_panel_spec(session=built)
                    except Exception:  # noqa: BLE001
                        return None

                def _tui_terminals_panel(arg: str = "") -> dict[str, Any] | None:
                    # Bare /terminals (or "list") opens the panel; show|kill|help fall
                    # through to the command runner.
                    normalized = arg.strip().lower()
                    if normalized and normalized != "list":
                        return None
                    built = _tui_box.get("session")
                    if built is None:
                        return None
                    try:
                        return _chat_terminals_panel_spec(session=built)
                    except Exception:  # noqa: BLE001
                        return None

                def _tui_skill_panel(arg: str = "") -> dict[str, Any] | None:
                    # Bare /skill lists skills; "/skill <name>" and "/skill <name>
                    # <task>" fall through to the runner (info text / attach turn).
                    if arg.strip():
                        return None
                    built = _tui_box.get("session")
                    if built is None:
                        return None
                    try:
                        return _chat_skill_listing_panel_spec(session=built)
                    except Exception:  # noqa: BLE001
                        return None

                def _tui_forge_intro_panel(arg: str = "") -> dict[str, Any] | None:
                    # Plain /forge opens a guidance popup explaining how Forge works;
                    # its Enter (confirm) routes "/forge" to the command runner to enter
                    # the planning session, Esc cancels. "/forge resume" and re-entry
                    # while already in Forge return None so they fall through to the
                    # runner and resume directly (no intro).
                    if arg.strip():
                        return None
                    if _tui_forge_state.ui_mode == "forge":
                        return None
                    try:
                        return _chat_forge_intro_panel_spec()
                    except Exception:  # noqa: BLE001 - never crash the UI on a panel
                        return None

                # --- Forge read-only panels (only fire inside a Forge session) ---
                def _tui_forge_plan_and_paths() -> tuple[Any, Any] | None:
                    if _tui_forge_state.ui_mode != "forge":
                        return None  # chat-mode /plan, /show handled elsewhere
                    plan = getattr(_tui_forge_state, "plan", None)
                    paths = getattr(_tui_forge_state, "paths", None)
                    if not isinstance(plan, dict) or paths is None:
                        return None
                    return plan, paths

                def _tui_forge_show_panel(arg: str = "") -> dict[str, Any] | None:
                    # Forge /show → the plan summary panel.
                    pair = _tui_forge_plan_and_paths()
                    if pair is None:
                        return None
                    plan, paths = pair
                    try:
                        return _chat_forge_plan_panel_spec(paths=paths, plan=plan)
                    except Exception:  # noqa: BLE001 - never crash the UI on a panel
                        return None

                def _tui_forge_plan_edit_spec(plan: Any, paths: Any) -> dict[str, Any]:
                    # /plan edit → open plan.json in the in-TUI editor; on save,
                    # validate the JSON shape, persist it, and reload the in-memory
                    # plan (no $EDITOR, which can't run under the alt-screen).
                    import json as _json

                    from ...forge import save_plan as _save_plan

                    try:
                        _save_plan(paths, plan)  # PLAN.md/json reflect the live plan
                    except Exception:
                        pass
                    try:
                        text = paths.plan_json_path.read_text(encoding="utf-8")
                    except Exception:
                        text = _json.dumps(plan, indent=2, ensure_ascii=False)

                    def _on_save(new_text: str) -> tuple[bool, str]:
                        try:
                            candidate = _json.loads(new_text)
                        except _json.JSONDecodeError as exc:
                            return (False, f"Invalid JSON: {exc}")
                        from ..commands.forge_helpers import _validate_forge_plan_shape

                        shape_error = _validate_forge_plan_shape(candidate)
                        if shape_error:
                            return (False, shape_error)
                        try:
                            _save_plan(paths, candidate)
                        except Exception as exc:  # noqa: BLE001
                            return (False, f"Failed to save plan: {exc}")
                        _tui_forge_state.plan = candidate
                        suffix = ""
                        try:
                            from ..commands.forge_helpers import (
                                _validate_forge_plan_for_paths,
                            )

                            warnings = _validate_forge_plan_for_paths(paths, candidate)
                            if warnings:
                                suffix = f" · {len(warnings)} validation warning(s)"
                        except Exception:
                            pass
                        return (True, f"Plan saved and reloaded.{suffix}")

                    return {
                        "editor": {
                            "title": f"Edit plan.json · {getattr(paths, 'run_id', '')}".rstrip(
                                " ·"
                            ),
                            "text": text,
                            "on_save": _on_save,
                        }
                    }

                def _tui_forge_plan_panel(arg: str = "") -> dict[str, Any] | None:
                    # Forge /plan: tasks|table|view → plan panel; markdown|md → PLAN.md
                    # doc panel; edit → in-TUI JSON editor. Bare /plan returns None so
                    # it falls through to the forge /plan picker.
                    pair = _tui_forge_plan_and_paths()
                    if pair is None:
                        return None
                    plan, paths = pair
                    sub = arg.strip().lower()
                    try:
                        if sub in {"tasks", "table", "view"}:
                            return _chat_forge_plan_panel_spec(paths=paths, plan=plan)
                        if sub in {"markdown", "md"}:
                            return _chat_forge_markdown_panel_spec(paths=paths, plan=plan)
                        if sub in {"edit", "edit-json"}:
                            return _tui_forge_plan_edit_spec(plan, paths)
                    except Exception:  # noqa: BLE001
                        return None
                    return None

                # --- Forge assets (picker + detail panel; replaces the stdin modal) ---
                def _tui_asset_context() -> tuple[Any, Any] | None:
                    built = _tui_box.get("session")
                    if built is None:
                        return None
                    cfg = getattr(built, "cfg", None)
                    if not isinstance(cfg, AppConfig):
                        cfg = effective if isinstance(effective, AppConfig) else None
                    asset_paths = getattr(_tui_forge_state, "paths", None)
                    if asset_paths is None:
                        # Outside a Forge session, resolve the workspace's current run.
                        try:
                            from ...agent.prompt_context import (
                                resolve_session_active_workdir_path,
                            )
                            from ...forge import load_current_run_paths

                            asset_paths = load_current_run_paths(
                                Path(resolve_session_active_workdir_path(built))
                            )
                        except Exception:
                            asset_paths = None
                    if cfg is None or asset_paths is None:
                        return None
                    return cfg, asset_paths

                def _tui_assets_panel(arg: str = "") -> dict[str, Any] | None:
                    ctx = _tui_asset_context()
                    if ctx is None:
                        return None
                    cfg_obj, asset_paths = ctx
                    asset_id = arg.strip()
                    if asset_id:
                        try:
                            return _chat_asset_detail_panel_spec(
                                cfg=cfg_obj, paths=asset_paths, asset_id=asset_id
                            )
                        except Exception:  # noqa: BLE001
                            return None
                    # Bare /assets: show a small panel only when there is nothing to
                    # pick; otherwise fall through to the selectable picker below.
                    try:
                        from ...assets.surface import build_asset_surface

                        count = len(
                            build_asset_surface(cfg=cfg_obj, run_paths=asset_paths).list_assets()
                        )
                    except Exception:  # noqa: BLE001
                        return None
                    if count == 0:
                        return {
                            "title": "Assets",
                            "sections": [
                                (
                                    "Assets",
                                    [("status", "No assets attached to this run yet.", "plain")],
                                )
                            ],
                            "hint": "Esc close",
                        }
                    return None

                def _tui_assets_picker() -> dict[str, Any] | None:
                    ctx = _tui_asset_context()
                    if ctx is None:
                        return None
                    cfg_obj, asset_paths = ctx
                    try:
                        return _chat_assets_picker_spec(cfg=cfg_obj, paths=asset_paths)
                    except Exception:  # noqa: BLE001
                        return None

                _tui_panel_providers = {
                    "/status": _tui_status_panel,
                    "/usage": _tui_usage_panel,
                    "/ctx": _tui_context_panel,
                    "/context": _tui_context_panel,
                    "/model-info": _tui_model_info_panel,
                    "/config": _tui_config_panel,
                    "/toolbar": _tui_toolbar_panel,
                    "/terminals": _tui_terminals_panel,
                    "/skill": _tui_skill_panel,
                    "/forge": _tui_forge_intro_panel,
                    "/show": _tui_forge_show_panel,
                    "/plan": _tui_forge_plan_panel,
                    "/assets": _tui_assets_panel,
                }

                # Slash-command dropdown: same completer the classic prompt uses, so
                # typing "/" lists commands (and /subagent|/skill complete names).
                from ..chat_slash_completer import ChatSlashCompleter

                def _tui_subagent_names() -> list[str]:
                    built = _tui_box.get("session")
                    return sorted(str(n) for n in (getattr(built, "subagent_registry", {}) or {}))

                def _tui_skill_names() -> list[str]:
                    built = _tui_box.get("session")
                    return sorted(str(n) for n in (getattr(built, "skill_registry", {}) or {}))

                _tui_completer = ChatSlashCompleter(
                    mode_provider=lambda: _tui_forge_state.ui_mode,
                    subagent_names_provider=_tui_subagent_names,
                    skill_names_provider=_tui_skill_names,
                )

                # TUI-native /mode picker: bare /mode opens a selectable popup;
                # "/mode <name>" still applies inline via the command runner.
                from ..commands.chat_terminal import (
                    _chat_mode_display,
                    _chat_mode_rows,
                    _chat_trace_rows,
                )
                from ..commands.startup import _chat_trace_level, _set_chat_trace_level

                _FULLACCESS_WARNING = (
                    "full (fullaccess) disables write/shell safety guards and approval prompts."
                )

                def _tui_mode_select(value: str) -> list[tuple[str, str]] | None:
                    built = _tui_box.get("session")
                    if built is None:
                        return None
                    current = str(getattr(built, "mode", "review") or "review").strip().lower()
                    if _chat_plan_mode_enabled(_tui_plan_state):
                        if value == "readonly":
                            return [("system", "Mode already set: Read-Only (Plan Mode is on)")]
                        return [
                            (
                                "warn",
                                "Cannot change execution mode while Plan Mode is on. "
                                "Use /plan off first.",
                            )
                        ]
                    if value == current:
                        msgs = [("system", f"Mode already set: {_chat_mode_display(value)}")]
                        if value == "fullaccess":
                            msgs.append(("warn", _FULLACCESS_WARNING))
                        return msgs
                    try:
                        _apply_chat_effective_mode(
                            session=built, next_mode=value, persist_default_mode=True
                        )
                    except Exception as exc:  # noqa: BLE001
                        return [("error", f"Failed to change mode: {exc}")]
                    _tui_state.exec_mode = value
                    msgs = [("system", f"Mode → {_chat_mode_display(value)}")]
                    if value == "fullaccess":
                        msgs.append(("warn", _FULLACCESS_WARNING))
                    return msgs

                def _tui_mode_picker() -> dict[str, Any] | None:
                    built = _tui_box.get("session")
                    current = str(getattr(built, "mode", "review") or "review").strip().lower()
                    rows: list[dict[str, Any]] = []
                    for value, label, desc in _chat_mode_rows():
                        clean = label.split(") ", 1)[-1] if ") " in label else label
                        rows.append(
                            {
                                "label": clean,
                                "description": desc,
                                "value": value,
                                "current": value == current,
                            }
                        )
                    return {"title": "Mode", "rows": rows, "on_select": _tui_mode_select}

                # TUI-native /trace picker: bare /trace opens a selectable popup;
                # "/trace <level>" still applies inline via the command runner.
                def _tui_trace_select(value: str) -> list[tuple[str, str]] | None:
                    built = _tui_box.get("session")
                    if built is None:
                        return None
                    current = _chat_trace_level(built)
                    if value == current:
                        return [("system", f"Reasoning trace already set: {value}")]
                    try:
                        applied = _set_chat_trace_level(session=built, level=value)
                    except Exception as exc:  # noqa: BLE001
                        return [("error", f"Failed to set trace level: {exc}")]
                    return [("system", f"Reasoning trace → {applied}")]

                def _tui_trace_picker() -> dict[str, Any] | None:
                    built = _tui_box.get("session")
                    current = _chat_trace_level(built) if built is not None else "compact"
                    rows: list[dict[str, Any]] = []
                    for value, label, desc in _chat_trace_rows():
                        clean = label.split(") ", 1)[-1] if ") " in label else label
                        rows.append(
                            {
                                "label": clean,
                                "description": desc,
                                "value": value,
                                "current": value == current,
                            }
                        )
                    return {
                        "title": "Reasoning Trace",
                        "rows": rows,
                        "on_select": _tui_trace_select,
                    }

                # TUI-native /subagent picker: bare /subagent opens a selectable
                # list of subagents to spawn; choosing one prefills the input with
                # "/subagent <name> " so the user types the task and Enter spawns it
                # (explicit "/subagent <name> <task>" routes to the runner). The
                # on|off|status toggles and the explicit form still go to the runner.
                def _tui_subagent_select(value: str) -> dict[str, Any]:
                    return {"prefill": f"/subagent {value} "}

                def _tui_subagent_picker() -> dict[str, Any] | None:
                    built = _tui_box.get("session")
                    registry_obj = (
                        getattr(built, "subagent_registry", None) if built is not None else None
                    )
                    registry = registry_obj if isinstance(registry_obj, dict) else {}
                    if not registry:
                        return None  # nothing to pick → runner prints guidance
                    rows: list[dict[str, Any]] = []
                    for value, _label, desc in _chat_subagent_rows(registry=registry):
                        rows.append(
                            {
                                "label": str(value),
                                # Full first-clause summary — the picker wraps it to
                                # a couple of lines rather than clipping on the right.
                                "description": _short_subagent_desc(desc, limit=120),
                                "value": str(value),
                                "current": False,
                            }
                        )
                    enabled = bool(getattr(built, "subagents_enabled", False))
                    hint = "↑↓ select · Enter to spawn · Esc cancel"
                    if not enabled:
                        # Own line so it never crowds / clips the keybinding hint.
                        hint += "\nauto-delegate off · enable with /subagent on"
                    return {
                        "title": "Spawn Subagent",
                        "rows": rows,
                        "on_select": _tui_subagent_select,
                        "hint": hint,
                    }

                # TUI-native forge pickers: bare /plan opens a tasks/markdown/edit
                # chooser; bare /assistant toggles the planner assistant. Each row's
                # on_select SUBMITS the explicit form (e.g. "/plan tasks") so the
                # panel/runner path applies with no extra Enter. They fire only in a
                # Forge session; in chat mode they return None and fall through.
                def _tui_forge_plan_picker() -> dict[str, Any] | None:
                    if _tui_forge_state.ui_mode != "forge":
                        return None
                    rows = [
                        {
                            "label": "tasks",
                            "description": "Show the plan summary — goal, requirements, task table.",
                            "value": "tasks",
                            "current": False,
                        },
                        {
                            "label": "markdown",
                            "description": "Preview PLAN.md for the current run.",
                            "value": "markdown",
                            "current": False,
                        },
                        {
                            "label": "edit",
                            "description": "Edit plan.json in an in-TUI editor, then reload.",
                            "value": "edit",
                            "current": False,
                        },
                    ]
                    return {
                        "title": "Plan",
                        "rows": rows,
                        "on_select": lambda value: {"submit": f"/plan {value}"},
                        "hint": "↑↓ select · Enter · Esc cancel",
                    }

                def _tui_forge_assistant_picker() -> dict[str, Any] | None:
                    if _tui_forge_state.ui_mode != "forge":
                        return None
                    enabled = bool(getattr(_tui_forge_state, "assistant_enabled", False))
                    rows = [
                        {
                            "label": "on",
                            "description": "Planner assistant drafts and refines the plan with you.",
                            "value": "on",
                            "current": enabled,
                        },
                        {
                            "label": "off",
                            "description": "Manual planning only — you drive /goal, /task, /plan edit.",
                            "value": "off",
                            "current": not enabled,
                        },
                    ]
                    return {
                        "title": "Planner Assistant",
                        "rows": rows,
                        "on_select": lambda value: {"submit": f"/assistant {value}"},
                        "hint": "↑↓ select · Enter · Esc cancel",
                    }

                _tui_picker_providers = {
                    "/mode": _tui_mode_picker,
                    "/trace": _tui_trace_picker,
                    "/subagent": _tui_subagent_picker,
                    "/plan": _tui_forge_plan_picker,
                    "/assistant": _tui_forge_assistant_picker,
                    "/assets": _tui_assets_picker,
                }

                _tui_ok = False
                _tui_result: Any = None
                try:
                    _tui_result, _ = _run_tui(
                        _tui_state,
                        session_builder=_tui_session_builder,
                        on_turn_complete=_tui_on_turn_complete,
                        on_hud_refresh=_tui_on_turn_complete,
                        command_runner=_tui_command_runner,
                        panel_providers=_tui_panel_providers,
                        picker_providers=_tui_picker_providers,
                        completer=_tui_completer,
                        config_flow_factory=_tui_config_flow_factory,
                        on_config_saved=_tui_on_config_saved,
                    )
                    _tui_ok = True
                except Exception as _tui_exc:  # pragma: no cover - defensive fallback
                    console.print(
                        f"[yellow]TUI unavailable ({_tui_exc}); using classic chat.[/yellow]"
                    )
                finally:
                    _tui_session = _tui_box.get("session")
                    if _tui_session is not None:
                        try:
                            _tui_session.close()
                        except Exception:
                            pass
                if _tui_ok:
                    # "Switch project" from /config exits with this sentinel; relaunch a
                    # fresh chat bound to the chosen folder (the old session is already
                    # closed above). We forward this session's execution-posture flags
                    # (mode/yes/no_log/verify_cmd/…) so switching project doesn't
                    # silently reset auto-approve/mode/logging. NOTE: this re-enters
                    # chat() (one frame per switch); switching is a deliberate,
                    # heavyweight manual action so the depth is bounded in practice.
                    if (
                        isinstance(_tui_result, tuple)
                        and len(_tui_result) == 2
                        and _tui_result[0] == "switch_workspace"
                    ):
                        from ..commands.startup import _run_default_chat_action

                        _run_default_chat_action(
                            path=Path(str(_tui_result[1])),
                            allow_broad_workspace=True,
                            mode=mode,
                            model=model,
                            base_url=base_url,
                            temperature=temperature,
                            stream=stream,
                            max_steps=max_steps,
                            subagents=subagents,
                            no_log=no_log,
                            verify_cmd=verify_cmd,
                            yes=yes,
                        )
                    return
        printWelcome(
            console=console,
            workspace=focus_path if focus_path != session_root else session_root,
            model=str(effective.model or "?"),
        )
        console.print("[dim]Starting chat session...[/dim]")
        create_session_kwargs = {
            "cfg": effective,
            "root": session_root,
            "mode": effective_mode,
            "runtime_kind": RuntimeKind.INTERACTIVE_CHAT,
            "yes": yes,
            "max_steps": effective.max_steps,
            "no_log": no_log,
            "api_key_override": api_key_override,
            "console": console,
            "surface": _make_rich_surface(console=console, show_status_line=False),
            "non_interactive": non_interactive,
            "enable_chat_turn_step_budget": True,
            "chat_turn_fixed_override": (effective.max_steps if max_steps_provided else None),
            "verify_cmd": verify_cmd,
            "subagents_enabled": effective.subagents_enabled,
            "workspace_binding": workspace_binding,
        }
        try:
            session = create_session(**create_session_kwargs)
        except ConfigError as e:
            if not _is_default_shell_sandbox_startup_failure(cfg=effective, error=e):
                raise
            console.print(
                "[yellow]Shell sandbox unavailable:[/yellow] starting chat with shell execution disabled. "
                "Run `sylliptor doctor sandbox` for setup help, or set "
                "SYLLIPTOR_SHELL_SANDBOX_MODE=off for explicit unsafe host execution."
            )
            create_session_kwargs["cfg"] = _cfg_with_warn_shell_sandbox_mode(effective)
            session = create_session(**create_session_kwargs)
        _set_chat_usage_hud_enabled(session, _resolve_usage_hud_default(effective))
        baseline_temperature = getattr(getattr(session, "client", None), "temperature", None)
        if baseline_temperature is None:
            baseline_temperature = getattr(effective, "chat_temperature", None)
        session._toolbar_default_temperature = baseline_temperature
        _refresh_chat_hud_context_cache(session)
        _ensure_session_summary_metadata(session=session, allow_model_summary=False)
    except ConfigError as e:
        console.print(f"[red]Config error:[/red] {e}")
        raise typer.Exit(code=2) from e
    except WorkspaceBindingError as e:
        console.print(f"[red]Workspace error:[/red] {e}")
        raise typer.Exit(code=1) from e
    try:
        if not bool(getattr(session, "subagents_enabled", False)):
            console.print("[dim]Tip: enable subagents with /subagent on.[/dim]")
        pending_images = [os.fspath(p) for p in (image or [])]
        forge_state = _ForgeChatState()
        plan_mode_state = _ChatPlanModeState()
        prompt_session = _maybe_make_chat_prompt_session(
            console=console,
            root=focus_path,
            pending_images=pending_images,
            forge_state=forge_state,
            session=session,
            plan_mode_state=plan_mode_state,
        )
        if pending_images:
            console.print(f"Queued {len(pending_images)} image(s) for your next message.")
        while True:
            try:
                if prompt_session:
                    prompt_session_erases = bool(
                        getattr(prompt_session, "_sylliptor_erase_when_done", True)
                    )

                    def _bottom_toolbar(
                        _pending_images: list[str] = pending_images,
                    ) -> str:
                        return _chat_bottom_toolbar(
                            session=session,
                            pending_images=_pending_images,
                            forge_state=forge_state,
                            plan_mode_enabled=_chat_plan_mode_enabled(plan_mode_state),
                        )

                    prompt_result = prompt_session.prompt(
                        _chat_prompt_label_formatted(
                            ui_mode=forge_state.ui_mode,
                            mode=str(getattr(session, "mode", "")),
                        ),
                        bottom_toolbar=_bottom_toolbar,
                    )
                    if prompt_result is _CHAT_PROMPT_RESULT_PLAN_MODE_OFF:
                        user_msg = "/plan off"
                    else:
                        user_msg = prompt_result
                    if not prompt_session_erases and isinstance(user_msg, str):
                        _clear_submitted_prompt_line(
                            submitted_text=user_msg,
                            prompt_label=_chat_prompt_label(
                                ui_mode=forge_state.ui_mode,
                                mode=str(getattr(session, "mode", "")),
                            ),
                        )
                else:
                    fallback_label = _chat_prompt_fallback_label(
                        ui_mode=forge_state.ui_mode,
                        mode=str(getattr(session, "mode", "")),
                    )
                    user_msg = typer.prompt(fallback_label, prompt_suffix=" ")
                    _clear_submitted_prompt_line(
                        submitted_text=user_msg,
                        prompt_label=f"{fallback_label} ",
                    )
            except (EOFError, KeyboardInterrupt):
                console.print("")
                return

            command_result = _handle_chat_command(
                input_text=user_msg,
                root=focus_path,
                session=session,
                pending_images=pending_images,
                console=console,
                forge_state=forge_state,
                plan_mode_state=plan_mode_state,
                plan_mode_escape_supported=prompt_session is not None,
            )
            if command_result == "exit":
                return
            if command_result == "handled":
                continue
            if command_result == "send" and _chat_plan_mode_enabled(plan_mode_state):
                command_result = _resolve_interactive_plan_mode_request(
                    session=session,
                    console=console,
                    plan_mode_state=plan_mode_state,
                    user_message=user_msg,
                    plan_mode_escape_supported=prompt_session is not None,
                )
                if command_result == "handled":
                    continue

            execution_instruction = (
                command_result.instruction
                if isinstance(command_result, _ChatExecutionRequest)
                else user_msg
            )
            routing_mode_override = (
                command_result.routing_mode_override
                if isinstance(command_result, _ChatExecutionRequest)
                else None
            )
            ephemeral_system_messages = (
                list(command_result.ephemeral_system_messages)
                if isinstance(command_result, _ChatExecutionRequest)
                and command_result.ephemeral_system_messages
                else None
            )
            ephemeral_user_messages = (
                list(command_result.ephemeral_user_messages)
                if isinstance(command_result, _ChatExecutionRequest)
                and command_result.ephemeral_user_messages
                else None
            )
            temporary_mode_override = (
                command_result.mode_override
                if isinstance(command_result, _ChatExecutionRequest)
                else None
            )
            restore_mode_after = (
                command_result.restore_mode_after
                if isinstance(command_result, _ChatExecutionRequest)
                else None
            )
            plan_mode_capture_task = (
                command_result.plan_mode_capture_task
                if isinstance(command_result, _ChatExecutionRequest)
                else None
            )

            images_for_turn = pending_images.copy()
            interrupted = False
            llm_failed = False
            restored_mode_after_turn = False
            turn_start_messages = (
                len(getattr(session, "messages", []) or [])
                if plan_mode_capture_task is not None
                else 0
            )
            if temporary_mode_override is not None:
                try:
                    _apply_chat_effective_mode(
                        session=session,
                        next_mode=temporary_mode_override,
                        persist_default_mode=False,
                    )
                except Exception as e:  # noqa: BLE001
                    console.print(f"[red]Failed to prepare approved execution:[/red] {e}")
                    continue
            try:
                with _chat_turn_interrupt_monitor():
                    run_turn_kwargs: dict[str, Any] = {"image_paths": images_for_turn or None}
                    if routing_mode_override is not None:
                        run_turn_kwargs["routing_mode_override"] = routing_mode_override
                    if ephemeral_system_messages:
                        run_turn_kwargs["ephemeral_system_messages"] = ephemeral_system_messages
                    if ephemeral_user_messages:
                        run_turn_kwargs["ephemeral_user_messages"] = ephemeral_user_messages
                    session.run_turn(execution_instruction, **run_turn_kwargs)
            except KeyboardInterrupt:
                interrupted = True
                _finish_chat_surface_activity(session=session)
                console.print(
                    "[yellow]Interrupted current turn.[/yellow] "
                    "You can send a new message or use /exit."
                )
            except LLMError as e:
                llm_failed = True
                _render_chat_llm_error(session=session, console=console, error=e)
            finally:
                if restore_mode_after is not None:
                    try:
                        _apply_chat_effective_mode(
                            session=session,
                            next_mode=restore_mode_after,
                            persist_default_mode=False,
                        )
                        restored_mode_after_turn = True
                    except Exception as e:  # noqa: BLE001
                        console.print(
                            f"[red]Failed to restore Plan Mode after execution:[/red] {e}"
                        )

            if llm_failed:
                # Keep queued images so the user can retry without re-attaching.
                pending_images = images_for_turn
                _refresh_chat_hud_context_cache(session)
                continue

            pending_images.clear()
            if restored_mode_after_turn:
                _refresh_chat_hud_context_cache(session)
            _refresh_chat_hud_context_cache(session)
            if interrupted:
                continue
            if plan_mode_capture_task is not None:
                _record_plan_mode_draft_from_turn(
                    session=session,
                    console=console,
                    plan_mode_state=plan_mode_state,
                    user_message=plan_mode_capture_task,
                    start_index=turn_start_messages,
                )
            _ensure_session_summary_metadata(session=session, allow_model_summary=False)
            usage_result = _chat_turn_usage_line(session)
            if usage_result is not None:
                usage_line, usage_warning_line = usage_result
                if usage_line:
                    console.print(usage_line, style="dim", highlight=False)
                if usage_warning_line:
                    console.print(
                        usage_warning_line,
                        style=_chat_turn_usage_style(session),
                        highlight=False,
                    )
    finally:
        session.close()


def run(
    instruction: str = typer.Argument(..., help="What you want the agent to do."),
    path: Path = typer.Option(Path("."), "--path", help="Working directory/root."),
    create_path: bool = typer.Option(
        False,
        "--create-path",
        help="Create --path if it does not exist before binding the workspace.",
    ),
    allow_broad_workspace: bool = typer.Option(
        False,
        "--allow-broad-workspace",
        help="Allow guarded broad workspaces in non-interactive startup flows.",
    ),
    image: list[Path] | None = typer.Option(
        None,
        "--image",
        help="Attach image path(s). Repeat --image for multiple files.",
    ),
    mode: Mode | None = typer.Option(None, "--mode", help="Mode override."),
    model: str | None = typer.Option(None, "--model", help="Model override."),
    base_url: str | None = typer.Option(None, "--base-url", help="Base URL override."),
    temperature: float | None = typer.Option(None, "--temperature", help="Sampling temperature."),
    stream: bool | None = typer.Option(
        None,
        "--stream/--no-stream",
        help="Enable streamed assistant output.",
    ),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Max steps override."),
    subagents: bool | None = typer.Option(
        None,
        "--subagents/--no-subagents",
        help="Enable or disable subagent delegation for this session.",
    ),
    no_log: bool = typer.Option(False, "--no-log", help="Disable JSONL session logging."),
    verify_cmd: list[str] | None = typer.Option(
        None,
        "--verify-cmd",
        help="Override verification command for this run (repeatable).",
    ),
    api_key_env: str | None = typer.Option(
        None,
        "--api-key-env",
        help=(
            "Read API key from this environment variable (overrides SYLLIPTOR_API_KEY/OPENAI_API_KEY)."
        ),
    ),
    api_key_stdin: bool = typer.Option(
        False,
        "--api-key-stdin",
        help="Prompt for API key (hidden input). Key is kept in memory for this run only.",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help=(
            "UNSAFE: Provide API key via CLI argument (may leak via shell history / process list). "
            "Prefer --api-key-stdin or --api-key-env."
        ),
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="In auto mode, skip confirmations for sensitive commands (hard blocks still apply).",
    ),
    cli_ctx: Any = None,
) -> None:
    console = _console()
    cfg = load_config()
    current_ctx = cli_ctx if cli_ctx is not None else get_current_context(silent=True)
    non_interactive = _is_non_interactive_terminal()
    path_source = current_ctx.get_parameter_source("path") if current_ctx is not None else None
    max_steps_source = (
        current_ctx.get_parameter_source("max_steps") if current_ctx is not None else None
    )
    max_steps_provided = max_steps is not None
    if current_ctx is not None:
        max_steps_provided = _parameter_value_was_provided(max_steps, max_steps_source)

    effective = clone_cfg(cfg)
    if base_url is not None:
        effective.base_url = base_url
    if model is not None:
        effective.model = model
    if temperature is not None:
        _apply_temperature_override(effective, temperature)
    if stream is not None:
        effective.stream = stream
    else:
        # Interactive chat is more readable with incremental output by default.
        effective.stream = True
    if max_steps is not None:
        effective.max_steps = max_steps
    if subagents is not None:
        effective.subagents_enabled = subagents

    effective_mode = (mode.value if mode else effective.default_mode) or "review"
    binding_source = _path_binding_source(path_source, path)

    try:
        if not effective.model:
            raise ConfigError("Model is not set. Run: sylliptor config set model <MODEL>")
        api_key_override = _resolve_api_key_override(
            api_key=api_key,
            api_key_env=api_key_env,
            api_key_stdin=api_key_stdin,
        )
        workspace_binding = _resolve_startup_workspace_binding(
            requested_path=path,
            console=console,
            interactive=not non_interactive,
            create_if_missing=create_path,
            allow_broad_workspace=allow_broad_workspace,
            source=binding_source,
            action=WorkspaceAction.CHAT,
        )
        code = run_agent(
            cfg=effective,
            root=workspace_binding.workspace_context.workspace_root,
            instruction=instruction,
            image_paths=[os.fspath(p) for p in (image or [])],
            mode=effective_mode,
            runtime_kind=RuntimeKind.ONE_SHOT,
            yes=yes,
            max_steps=effective.max_steps,
            no_log=no_log,
            api_key_override=api_key_override,
            console=console,
            surface=_make_rich_surface(console=console),
            non_interactive=non_interactive,
            usage_role="run",
            subagents_enabled=effective.subagents_enabled,
            one_shot_execution=True,
            enable_chat_turn_step_budget=True,
            chat_turn_fixed_override=(effective.max_steps if max_steps_provided else None),
            verify_cmd=verify_cmd,
            workspace_binding=workspace_binding,
        )
    except ConfigError as e:
        console.print(f"[red]Config error:[/red] {e}")
        raise typer.Exit(code=2) from e
    except WorkspaceBindingError as e:
        console.print(f"[red]Workspace error:[/red] {e}")
        raise typer.Exit(code=1) from e
    except Exception as e:  # noqa: BLE001
        # Prefer friendly Sylliptor MiMo trial copy (trial_expired, rate-limit, ...)
        # over a raw ``LLM error 402: {...}`` dump; any other error renders as-is.
        message = str(e)
        try:
            from ...llm.openai_compat import sylliptor_trial_error_message

            message = sylliptor_trial_error_message(e) or message
        except Exception:  # noqa: BLE001
            pass
        try:
            console.print(f"[red]Error:[/red] {message}")
        except Exception as render_exc:  # noqa: BLE001 - CLI error rendering must not double-crash.
            safe_plain_error(
                stream=getattr(console, "file", None),
                error_type=type(render_exc).__name__,
                message=str(e),
            )
        raise typer.Exit(code=1) from e

    raise typer.Exit(code=code)


def _handle_chat_command_impl(cli_mod: Any, *args: Any, **kwargs: Any) -> Any:
    _sync_cli_globals(cli_mod)
    return _handle_chat_command(*args, **kwargs)


def _handle_forge_chat_command_impl(cli_mod: Any, *args: Any, **kwargs: Any) -> Any:
    _sync_cli_globals(cli_mod)
    return _handle_forge_chat_command(*args, **kwargs)


def _run_plan_mode_approval_loop_impl(cli_mod: Any, *args: Any, **kwargs: Any) -> Any:
    _sync_cli_globals(cli_mod)
    return _run_plan_mode_approval_loop(*args, **kwargs)


def _print_chat_context_impl(cli_mod: Any, *args: Any, **kwargs: Any) -> Any:
    _sync_cli_globals(cli_mod)
    return _print_chat_context(*args, **kwargs)


def chat_impl(cli_mod: Any, *args: Any, **kwargs: Any) -> Any:
    _sync_cli_globals(cli_mod)
    return chat(*args, **kwargs)


def run_impl(cli_mod: Any, *args: Any, **kwargs: Any) -> Any:
    _sync_cli_globals(cli_mod)
    return run(*args, **kwargs)
