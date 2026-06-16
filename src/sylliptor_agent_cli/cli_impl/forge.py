# ruff: noqa: F821
# Dependencies are injected at runtime from sylliptor_agent_cli.cli to preserve monkeypatch surfaces.
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any

import typer
from click import get_current_context
from click.core import ParameterSource

from ..assets import AssetError
from ..assets.budget_allocator import (
    TaskAssetAllocation,
    allocate_task_assets,
    write_task_asset_allocation,
)
from ..assets.surface import build_asset_surface
from ..assets.usage_logger import AssetUsageLogger
from ..assets.worker_mirror import TaskAssetMirror, mirror_task_assets
from ..assets.worker_section import render_relevant_assets_section
from ..assets.worker_tools import build_worker_asset_mcp_manager, compose_worker_asset_mcp_manager
from ..error_text import sanitize_error_summary, sanitize_optional_error_summary
from ..model_registry import ModelRegistry
from ..plan_validation import PlannerFailedError, raise_for_execution_ready_plan
from ..runtime_kind import RuntimeKind
from ..swarm_orchestrator import acquire_swarm_mutation_guard
from ..swarm_scheduler import canonical_task_status
from ..task_readiness import is_clearly_non_mutating_task
from ..task_scope import (
    assess_scope_changes,
    is_non_material_untracked_path,
    normalize_scope_patterns,
    relocate_known_scratch_artifacts,
)

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


def _path_binding_source() -> str:
    current_ctx = get_current_context(silent=True)
    path_source = current_ctx.get_parameter_source("path") if current_ctx is not None else None
    if path_source is not None and path_source is not ParameterSource.DEFAULT:
        return "explicit_path"

    caller = inspect.currentframe()
    caller = caller.f_back if caller is not None else None
    path_value = caller.f_locals.get("path") if caller is not None else None
    if path_value not in (None, ".", Path(".")):
        return "explicit_path"

    if path_source is None or path_source is ParameterSource.DEFAULT:
        return "cwd"
    return "explicit_path"


def _missing_swarm_run_error(*, binding: WorkspaceBinding) -> ForgeError:
    requested = os.fspath(binding.requested_path)
    return ForgeError(
        "No current forge run was found for this workspace. "
        f"Start with `sylliptor forge plan --path {requested}` or enter Forge "
        "from chat after starting sylliptor inside a project folder."
    )


def _print_forge_lock_wait_notice(console: Any, info: dict[str, Any]) -> None:
    diagnostic = str(info.get("diagnostic") or "").strip()
    console.print(
        "[yellow]Forge execution queued:[/yellow] another execution is mutating this workspace; waiting for it to finish."
    )
    if diagnostic:
        console.print(f"[dim]{diagnostic}[/dim]")


def _render_planner_reply(
    *, console: Any, message: str, questions: list[str] | None = None
) -> None:
    console.print("[bold]Planner:[/bold]")
    console.print(message)
    if questions:
        console.print("[dim]Planner questions[/dim]")
        for question in questions:
            console.print(f"- {question}")


def _merge_changed_files(*path_groups: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in path_groups:
        for raw in group:
            value = str(raw).strip().replace("\\", "/")
            while value.startswith("./"):
                value = value[2:]
            if not value or value in seen:
                continue
            seen.add(value)
            merged.append(value)
    return merged


def _runtime_snapshot_changed_files(
    before_snapshot: dict[str, str],
    after_snapshot: dict[str, str],
) -> list[str]:
    changed: list[str] = []
    for path in sorted(set(before_snapshot) | set(after_snapshot)):
        if before_snapshot.get(path) != after_snapshot.get(path):
            changed.append(_normalize_changed_file_path(path))
    return [path for path in changed if path]


def _normalize_changed_file_path(path: str) -> str:
    value = str(path).strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return value.strip("/")


def _authorized_custom_tool_runtime_side_effects(
    *,
    sessions_dir: Path,
    session_id: str,
) -> set[str]:
    session_log = sessions_dir / f"{session_id}.jsonl"
    if not session_log.exists():
        return set()
    authorized: set[str] = set()
    try:
        lines = session_log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return set()
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "tool_result":
            continue
        payload = event.get("payload")
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict) or result.get("success") is not True:
            continue
        side_effects = result.get("side_effects")
        if not isinstance(side_effects, dict):
            continue
        writes = side_effects.get("workspace_writes")
        if not isinstance(writes, list):
            continue
        for item in writes:
            if not isinstance(item, dict):
                continue
            if str(item.get("scope") or "") != "tool_dir":
                continue
            rel_path = _normalize_changed_file_path(str(item.get("path") or ""))
            if rel_path == ".sylliptor/tools" or rel_path.startswith(".sylliptor/tools/"):
                authorized.add(rel_path)
    return authorized


def _drop_parent_directory_placeholders(paths: list[str]) -> list[str]:
    concrete_paths = [path for path in paths if path and not path.endswith("/")]
    filtered: list[str] = []
    for path in paths:
        if path.endswith("/") and any(other.startswith(path) for other in concrete_paths):
            continue
        filtered.append(path)
    return filtered


def _task_declares_explicit_write_scope(task: dict[str, Any]) -> bool:
    raw = task.get("write_scope")
    if isinstance(raw, list):
        return any(str(item or "").strip() for item in raw)
    if isinstance(raw, str):
        return bool(raw.strip())
    return False


def _task_is_analysis_only(task: dict[str, Any]) -> bool:
    raw_flag = task.get("analysis_only")
    if isinstance(raw_flag, bool):
        return raw_flag
    return is_clearly_non_mutating_task(
        title=str(task.get("title") or "").strip(),
        description=str(task.get("description") or "").strip(),
        acceptance_criteria=[
            str(item or "")
            for item in (task.get("acceptance_criteria") or [])
            if str(item or "").strip()
        ],
    )


def _empty_forge_exec_task_asset_mirror(
    workspace_path: Path,
    *,
    task_id: str = "",
) -> TaskAssetMirror:
    workspace = workspace_path.resolve()
    return TaskAssetMirror(
        workspace_path=workspace,
        manifest_path=workspace / ".sylliptor" / "task_assets" / "manifest.json",
        primary=[],
        may_need=[],
        pinned=[],
        task_id=task_id,
    )


def _combined_forge_exec_image_paths(
    *,
    legacy_paths: list[str],
    mirror: TaskAssetMirror,
    allocation: TaskAssetAllocation | None,
    cfg: Any,
    role_model: str,
    model_registry: ModelRegistry,
    usage_logger: AssetUsageLogger,
) -> list[str] | None:
    combined: list[str] = []
    seen: set[str] = set()

    def _append(path: str) -> None:
        try:
            normalized = os.fspath(Path(path).resolve())
        except OSError:
            return
        if normalized in seen:
            return
        seen.add(normalized)
        combined.append(normalized)

    for path in legacy_paths:
        _append(path)
    if cfg.assets.worker.inline_images and model_registry.get(role_model).supports_vision:
        decision_by_id = {
            decision.asset_id: decision.mode
            for decision in (allocation.decisions if allocation else [])
        }
        max_new = max(0, int(cfg.assets.worker.max_inline_images))
        added_new = 0
        for entry in mirror.primary:
            if entry.kind != "image" or entry.status != "mirrored":
                continue
            if decision_by_id.get(entry.asset_id) not in {"full_inline", "focused_extract"}:
                continue
            if entry.raw_workspace_path is None or not entry.raw_workspace_path.exists():
                continue
            before = len(combined)
            _append(os.fspath(entry.raw_workspace_path))
            if len(combined) > before:
                usage_logger.inline_injection(asset_id=entry.asset_id, kind=entry.kind)
                added_new += 1
                if added_new >= max_new:
                    break
    max_total = max(0, int(cfg.assets.worker.max_inline_images))
    if max_total > 0:
        combined = combined[:max_total]
    return combined or None


def _append_patch_debug_section(patch_path: Path, *, title: str, patch_text: str) -> None:
    existing = patch_path.read_text(encoding="utf-8") if patch_path.exists() else ""
    parts: list[str] = []
    if existing:
        parts.append(existing if existing.endswith("\n") else existing + "\n")
    parts.append(f"# {title}\n")
    if patch_text:
        parts.append(patch_text if patch_text.endswith("\n") else patch_text + "\n")
    patch_path.write_text("\n".join(part.rstrip("\n") for part in parts if part), encoding="utf-8")


def forge_plan(
    path: Path = typer.Option(
        Path("."),
        "--path",
        help="Workspace path or repository subdirectory.",
    ),
    create_path: bool = typer.Option(
        False,
        "--create-path",
        help="Create --path if it does not exist before binding the workspace.",
    ),
    allow_broad_workspace: bool = typer.Option(
        False,
        "--allow-broad-workspace",
        help="Allow guarded broad workspaces instead of choosing a narrower project folder.",
    ),
) -> None:
    console = _console()
    try:
        binding = _resolve_startup_workspace_binding(
            requested_path=path,
            console=console,
            interactive=not _is_non_interactive_terminal(),
            create_if_missing=create_path,
            allow_broad_workspace=allow_broad_workspace,
            source=_path_binding_source(),
            action=WorkspaceAction.FORGE_PLAN,
        )
        paths = create_plan_run(
            path,
            create_if_missing=create_path,
            allow_broad_workspace=allow_broad_workspace,
            workspace_binding=binding,
        )
        plan = load_plan(paths)
        workspace_scan = ensure_workspace_context_artifacts(paths)
    except ForgeError as e:
        console.print(f"[red]Forge error:[/red] {e}")
        raise typer.Exit(code=2) from e
    except WorkspaceBindingError as e:
        console.print(f"[red]Forge error:[/red] {e}")
        raise typer.Exit(code=2) from e

    console.rule("[bold cyan]forge plan[/bold cyan]")
    console.print(f"Run ID: {paths.run_id}")
    console.print(f"Plan directory: {paths.plan_dir}")
    for line in format_workspace_context_summary_lines(workspace_scan):
        console.print(line)
    console.print("Planning loop started. Type /help for commands. Type /done to finish.")
    assistant_enabled = False
    planning_suggested: set[str] = set()
    planner_state = _ForgePlannerSessionState(
        workspace_context=(
            _workspace_context_payload_for_paths(paths=paths) or workspace_scan.to_dict()
        )
    )

    def _emit_planner_meta(message: str) -> None:
        console.print(message)

    def _emit_planner_warning_group(label: str, warnings: list[str]) -> None:
        for warning in warnings:
            console.print(f"[yellow]{label}:[/yellow] {warning}")

    while True:
        try:
            line = typer.prompt("plan")
        except (EOFError, KeyboardInterrupt):
            console.print("")
            break
        text = line.strip()
        if not text:
            continue

        append_transcript_note(paths, role="user", message=text)
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in {"/done", "done", "/exit", "exit", "/quit", "quit"}:
            append_transcript_note(paths, role="system", message="Planning loop finished.")
            break
        if cmd in {"/help", "help"}:
            console.print(_planning_help_panel())
            append_transcript_note(paths, role="system", message="Displayed planning help.")
            continue
        if cmd == "/assistant":
            assistant_cmd = arg.lower()
            if not assistant_cmd:
                assistant_cmd, picker_available = _select_forge_assistant_interactive(
                    enabled=assistant_enabled,
                    console=console,
                )
                if not picker_available:
                    console.print("[yellow]Usage:[/yellow] /assistant on|off|status")
                    append_transcript_note(
                        paths,
                        role="system",
                        message="Rejected invalid /assistant usage.",
                    )
                    continue
                if assistant_cmd is None:
                    continue
            if assistant_cmd == "on":
                assistant_enabled = True
                console.print("Planner assistant: ON")
                append_transcript_note(paths, role="system", message="Planner assistant enabled.")
                continue
            if assistant_cmd == "off":
                assistant_enabled = False
                _set_forge_planner_follow_up_state(
                    planner_state=planner_state,
                    questions=[],
                    awaiting_clarification=False,
                )
                console.print("Planner assistant: OFF")
                append_transcript_note(paths, role="system", message="Planner assistant disabled.")
                continue
            if assistant_cmd == "status":
                state = "ON" if assistant_enabled else "OFF"
                console.print(f"Planner assistant: {state}")
                append_transcript_note(
                    paths,
                    role="system",
                    message=f"Planner assistant status requested ({state}).",
                )
                continue

            console.print("[yellow]Usage:[/yellow] /assistant on|off|status")
            append_transcript_note(
                paths,
                role="system",
                message="Rejected invalid /assistant usage.",
            )
            continue

        if assistant_enabled:
            _run_forge_planner_turn_controller(
                console=console,
                paths=paths,
                plan=plan,
                planner_state=planner_state,
                user_text=text,
                cfg_loader=load_config,
                unavailable_message_builder=(
                    lambda error: (
                        "Planner assistant is unavailable because config could not be loaded: "
                        f"{error}"
                    )
                ),
                emit_meta=_emit_planner_meta,
                emit_warning_group=_emit_planner_warning_group,
                api_key_override=None,
                render_reply=lambda message, questions: _render_planner_reply(
                    console=console,
                    message=message,
                    questions=questions,
                ),
                selection_label="planner",
                planning_relevant=True,
            )
            continue

        if cmd == "/goal":
            if not arg:
                console.print("[yellow]Usage:[/yellow] /goal <text>")
                append_transcript_note(paths, role="system", message="Rejected empty /goal.")
                continue
            goal = arg
            plan["project_goal"] = goal
            if not str(plan.get("summary") or "").strip():
                plan["summary"] = goal
            save_plan(paths, plan)
            console.print("Project goal updated.")
            if "goal" not in planning_suggested:
                planning_suggested.add("goal")
                console.print(
                    "[dim]Next: add tasks with /task <title>, or describe the work and "
                    "let the planner draft them.[/dim]"
                )
            append_transcript_note(paths, role="system", message="Updated project goal.")
            continue
        if cmd == "/task":
            if not arg:
                console.print("[yellow]Usage:[/yellow] /task <title>")
                append_transcript_note(paths, role="system", message="Rejected empty /task.")
                continue
            title = arg
            try:
                task = add_task(
                    plan,
                    title=title,
                    description=f"Manual planning chat task: {title}",
                )
            except ForgeError as e:
                console.print(f"[yellow]Task rejected:[/yellow] {e}")
                append_transcript_note(
                    paths,
                    role="system",
                    message=f"Rejected /task because it lacked runnable file scope: {e}",
                )
                continue
            save_plan(paths, plan)
            console.print(f"Added task: {task['id']} - {task['title']}")
            if "task" not in planning_suggested:
                planning_suggested.add("task")
                console.print(
                    "[dim]Next: add more tasks, or /done to save and validate the plan.[/dim]"
                )
            append_transcript_note(paths, role="system", message=f"Added task {task['id']}.")
            continue

        add_requirement(plan, text)
        save_plan(paths, plan)
        console.print("Captured requirement note.")
        append_transcript_note(paths, role="system", message="Captured requirement note.")

    finalize_plan(plan)
    reconciliation_result, _ = _reconcile_plan_for_paths(
        paths=paths,
        plan=plan,
        refresh_if_stale=True,
        transcript_tail=planner_state.transcript,
    )
    save_plan(paths, plan)
    validation_warnings = _validate_forge_plan_for_paths(paths, plan)
    if reconciliation_result.warnings:
        console.print("[yellow]Plan reconciliation warnings:[/yellow]")
        for warning in reconciliation_result.warnings:
            console.print(f"- {warning}")
            append_transcript_note(
                paths,
                role="system",
                message=f"Plan reconciliation warning: {warning}",
            )
    _write_plan_validation_artifact(
        paths=paths,
        reconciliation_result=reconciliation_result,
        validation_warnings=validation_warnings,
    )
    if validation_warnings:
        console.print("[yellow]Plan validation warnings:[/yellow]")
        for warning in validation_warnings:
            console.print(f"- {warning}")
            append_transcript_note(
                paths,
                role="system",
                message=f"Plan validation warning: {warning}",
            )
    console.print(f"Plan saved: {paths.plan_md_path}")
    console.print(f"Structured plan: {paths.plan_json_path}")


def forge_swarm(
    path: Path = typer.Option(
        Path("."),
        "--path",
        help="Workspace path or repository subdirectory.",
    ),
    allow_broad_workspace: bool = typer.Option(
        False,
        "--allow-broad-workspace",
        help="Allow guarded broad workspaces instead of requiring a narrower project path.",
    ),
    parallel: int = typer.Option(2, "--parallel", min=1, help="Parallel workers per batch."),
    base_branch: str | None = typer.Option(
        None,
        "--base-branch",
        help="Base branch (defaults to current checked out branch).",
    ),
    max_tasks: int | None = typer.Option(
        None,
        "--max-tasks",
        min=1,
        help="Maximum number of tasks to execute in this swarm run.",
    ),
    max_attempts: int | None = typer.Option(
        None,
        "--max-attempts",
        min=1,
        help="Maximum attempts allowed per task before scheduler skips it.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print schedule and exit."),
    keep_worktrees: bool = typer.Option(
        False,
        "--keep-worktrees",
        help=(
            "Keep failed task worktrees and branches for debugging; default cleanup removes "
            "successful worktrees and rejected failed branch state."
        ),
    ),
    retry_failed: bool = typer.Option(
        False,
        "--retry-failed",
        help="Include tasks currently marked failed.",
    ),
    retry_changes_requested: bool = typer.Option(
        False,
        "--retry-changes-requested",
        help="Include tasks currently marked changes_requested.",
    ),
    only: str | None = typer.Option(
        None,
        "--only",
        help="Comma-separated task ids to execute (still enforces dependencies).",
    ),
    retry_merge_conflicts: bool = typer.Option(
        False,
        "--retry-merge-conflicts",
        help="Retry tasks marked merge_conflict during merge phase.",
    ),
    scope: str = typer.Option(
        "strict",
        "--scope",
        help="Write-scope enforcement: strict by default; use warn or off to opt out.",
    ),
    verify: str = typer.Option(
        "warn",
        "--verify",
        help="Verification policy: off, warn, or strict.",
    ),
    verify_cmd: list[str] | None = typer.Option(
        None,
        "--verify-cmd",
        help="Override verify command for this run (repeatable).",
    ),
    integration_verify: str | None = typer.Option(
        None,
        "--integration-verify",
        help="Batch integration verification policy: off, warn, or strict (defaults to config: warn).",
    ),
    integration_verify_cmd: list[str] | None = typer.Option(
        None,
        "--integration-verify-cmd",
        help="Override integration verify command for this swarm run (repeatable).",
    ),
    replan: str | None = typer.Option(
        None,
        "--replan",
        help="Between-batch replanning mode: off, suggest, or apply.",
    ),
    review: bool = typer.Option(
        False,
        "--review",
        help="Run automated PR review gate before merging task branches.",
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
    no_log: bool = typer.Option(False, "--no-log", help="Disable JSONL session logging."),
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
) -> None:
    console = _console()
    cfg = load_config()
    effective = clone_cfg(cfg)
    current_ctx = get_current_context(silent=True)
    max_steps_source = (
        current_ctx.get_parameter_source("max_steps") if current_ctx is not None else None
    )
    max_steps_provided = max_steps is not None
    if current_ctx is not None:
        max_steps_provided = (
            max_steps_source is not None and max_steps_source is not ParameterSource.DEFAULT
        )
    if base_url is not None:
        effective.base_url = base_url
    if model is not None:
        effective.model = model
    if temperature is not None:
        _apply_temperature_override(effective, temperature)
    if stream is not None:
        effective.stream = stream
    if max_steps is not None:
        effective.max_steps = max_steps
    swarm_max_steps = effective.max_steps if max_steps_provided else None

    effective_mode = (mode.value if mode else effective.default_mode) or "review"
    scope_mode = "strict"
    verify_mode = "warn"
    integration_verify_mode = None
    replanning_mode = None

    try:
        scope_mode = _normalize_scope_mode(scope)
        verify_mode = _normalize_verify_mode(verify)
        integration_verify_mode = integration_verify
        replanning_mode = replan
        api_key_override = _resolve_api_key_override(
            api_key=api_key,
            api_key_env=api_key_env,
            api_key_stdin=api_key_stdin,
        )
        binding = resolve_workspace_binding(
            path,
            create_if_missing=False,
            allow_broad_workspace=allow_broad_workspace,
            source=_path_binding_source(),
        )
        ensure_workspace_policy(
            binding,
            action=WorkspaceAction.SWARM,
            allow_broad_workspace=allow_broad_workspace,
        )
        try:
            paths = load_current_run_paths(binding.workspace_context.focus_path)
        except ForgeError as e:
            if "current_run.json" in str(e):
                raise _missing_swarm_run_error(binding=binding) from e
            raise
        plan = load_plan(paths)
        run_mutation_guard = acquire_swarm_mutation_guard(
            paths,
            mode="forge_swarm:cli",
            on_wait=lambda info: _print_forge_lock_wait_notice(console, info),
        )
        try:
            if bool(getattr(run_mutation_guard, "acquired_after_wait", False)):
                plan = load_plan(paths)
            reconciliation_result, _ = _reconcile_plan_for_paths(
                paths=paths,
                plan=plan,
                refresh_if_stale=True,
            )
            if reconciliation_result.changed:
                save_plan(paths, plan)
            if reconciliation_result.warnings:
                console.print("[yellow]Plan reconciliation warnings:[/yellow]")
                for warning in reconciliation_result.warnings:
                    console.print(f"- {warning}")
            validation_warnings = _validate_forge_plan_for_paths(paths, plan)
            _write_plan_validation_artifact(
                paths=paths,
                reconciliation_result=reconciliation_result,
                validation_warnings=validation_warnings,
            )
            no_execution_ready_tasks_message = _forge_no_execution_ready_tasks_message(plan)
            if no_execution_ready_tasks_message is not None:
                raise ForgeError(no_execution_ready_tasks_message)
            try:
                raise_for_execution_ready_plan(
                    plan,
                    retry_failed=retry_failed,
                    retry_changes_requested=retry_changes_requested,
                    retry_merge_conflicts=retry_merge_conflicts,
                    only=only,
                )
            except PlannerFailedError as e:
                err = ForgeError(str(e))
                err.failure_category = e.failure_category  # type: ignore[attr-defined]
                raise err from e
            resolve_model_for_role(
                cfg=effective,
                role=ROLE_CODING,
                plan=plan,
                prefer_context="forge",
            )
            code = run_swarm(
                paths=paths,
                plan=plan,
                cfg=effective,
                mode=effective_mode,
                yes=yes,
                max_steps=swarm_max_steps,
                api_key_override=api_key_override,
                no_log=no_log,
                parallel=parallel,
                base_branch=base_branch,
                max_tasks=max_tasks,
                max_attempts=max_attempts,
                dry_run=dry_run,
                keep_worktrees=keep_worktrees,
                retry_failed=retry_failed,
                retry_changes_requested=retry_changes_requested,
                only=only,
                retry_merge_conflicts=retry_merge_conflicts,
                scope_mode=scope_mode,
                verify_mode=verify_mode,
                verify_cmd=verify_cmd,
                integration_mode=integration_verify_mode,
                integration_verify_cmd=integration_verify_cmd,
                replanning_mode=replanning_mode,
                review=review,
                console=console,
                workspace_binding=binding,
                run_mutation_guard=run_mutation_guard,
            )
        finally:
            run_mutation_guard.release()
    except (ConfigError, ForgeError, GitOpsError, WorkspaceBindingError) as e:
        console.print(f"[red]Forge error:[/red] {e}")
        raise typer.Exit(code=2) from e
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Forge error:[/red] {e}")
        raise typer.Exit(code=1) from e

    log_paths = sorted(paths.execution_logs_dir.glob("*.jsonl"))
    _print_usage_summary_from_logs(
        console=console,
        title="Swarm Usage Summary",
        log_paths=log_paths,
    )
    raise typer.Exit(code=code)


def forge_exec(
    task_id: str = typer.Argument(..., help="Task id from plan.json (for example T01)."),
    path: Path = typer.Option(
        Path("."),
        "--path",
        help="Workspace path or repository subdirectory.",
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
    no_log: bool = typer.Option(False, "--no-log", help="Disable JSONL session logging."),
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
    pr: bool = typer.Option(
        False,
        "--pr/--no-pr",
        help="Run task in PR-like git flow (branch, commit, patch, merge).",
    ),
    review: bool = typer.Option(
        False,
        "--review",
        help="Run automated PR review gate before merge (requires --pr).",
    ),
    base_branch: str | None = typer.Option(
        None,
        "--base-branch",
        help="Base branch for --pr mode (defaults to current branch).",
    ),
    keep_branch: bool = typer.Option(
        False,
        "--keep-branch",
        help="Keep task branch after successful merge in --pr mode.",
    ),
    scope: str = typer.Option(
        "strict",
        "--scope",
        help="Write-scope enforcement: strict by default; use warn or off to opt out.",
    ),
    verify: str = typer.Option(
        "warn",
        "--verify",
        help="Verification policy for PR flow: off, warn, or strict.",
    ),
    verify_cmd: list[str] | None = typer.Option(
        None,
        "--verify-cmd",
        help="Override verify command for this run (repeatable).",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="In auto mode, skip confirmations for sensitive commands (hard blocks still apply).",
    ),
) -> None:
    console = _console()
    cfg = load_config()
    effective = clone_cfg(cfg)
    current_ctx = get_current_context(silent=True)
    max_steps_source = (
        current_ctx.get_parameter_source("max_steps") if current_ctx is not None else None
    )
    max_steps_provided = max_steps is not None
    if current_ctx is not None:
        max_steps_provided = (
            max_steps_source is not None and max_steps_source is not ParameterSource.DEFAULT
        )
    if base_url is not None:
        effective.base_url = base_url
    if model is not None:
        effective.model = model
    if temperature is not None:
        _apply_temperature_override(effective, temperature)
    if stream is not None:
        effective.stream = stream
    if max_steps is not None:
        effective.max_steps = max_steps

    effective_mode = (mode.value if mode else effective.default_mode) or "review"
    scope_mode = "strict"
    verify_mode = "warn"
    verify_commands: list[str] = []
    verify_command_source: str | None = None
    run_cfg = clone_cfg(effective)

    try:
        scope_mode = _normalize_scope_mode(scope)
        verify_mode = _normalize_verify_mode(verify)
        api_key_override = _resolve_api_key_override(
            api_key=api_key,
            api_key_env=api_key_env,
            api_key_stdin=api_key_stdin,
        )
        paths = load_current_run_paths(path)
        plan = load_plan(paths)
        run_cfg.model = resolve_model_for_role(
            cfg=effective,
            role=ROLE_CODING,
            plan=plan,
            prefer_context="forge",
        )
    except (ConfigError, ForgeError) as e:
        console.print(f"[red]Forge error:[/red] {e}")
        raise typer.Exit(code=2) from e

    task = find_task(plan, task_id)
    if task is None:
        console.print(f"[red]Forge error:[/red] Task not found: {task_id}")
        raise typer.Exit(code=2)
    task_status = canonical_task_status(str(task.get("status") or ""))
    if task_status in {"superseded", "invalidated"}:
        console.print(
            "[red]Forge error:[/red] Task is non-executable obsolete work "
            f"({task_status}): {task_id}. Use an active planned replacement task instead."
        )
        raise typer.Exit(code=2)
    if verify_mode != "off":
        verify_selection = resolve_authoritative_task_verify_command_selection(
            cfg=effective,
            verify_cmd=verify_cmd,
            task=task,
            root=paths.root,
            plan_requirements=[
                str(item).strip() for item in (plan.get("requirements") or []) if str(item).strip()
            ],
        )
        verify_commands = list(verify_selection.commands)
        verify_command_source = verify_selection.source
        run_cfg.verify_commands = list(verify_commands)

    blockers = _task_dependency_blockers(plan, task)
    if blockers:
        console.print("[red]Forge error:[/red] Dependencies are not done: " + ", ".join(blockers))
        raise typer.Exit(code=2)
    if review and not pr:
        console.print("[red]Forge error:[/red] --review requires --pr.")
        raise typer.Exit(code=2)

    try:
        run_mutation_guard = acquire_swarm_mutation_guard(
            paths,
            mode=f"forge_exec:{task_id}",
            on_wait=lambda info: _print_forge_lock_wait_notice(console, info),
        )
    except ForgeError as e:
        console.print(f"[red]Forge error:[/red] {e}")
        raise typer.Exit(code=2) from e

    try:
        if bool(getattr(run_mutation_guard, "acquired_after_wait", False)):
            plan = load_plan(paths)
            task = find_task(plan, task_id)
            if task is None:
                console.print(
                    "[red]Forge error:[/red] Queued Forge exec revalidated the current plan "
                    f"and task no longer exists: {task_id}"
                )
                raise typer.Exit(code=2)
            task_status = canonical_task_status(str(task.get("status") or ""))
            if task_status in {"done", "superseded", "invalidated"}:
                console.print(
                    "[red]Forge error:[/red] Queued Forge exec revalidated the current plan "
                    f"and task is no longer executable ({task_status}): {task_id}."
                )
                raise typer.Exit(code=2)
            run_cfg.model = resolve_model_for_role(
                cfg=effective,
                role=ROLE_CODING,
                plan=plan,
                prefer_context="forge",
            )
            verify_commands = []
            verify_command_source = None
            if verify_mode != "off":
                verify_selection = resolve_authoritative_task_verify_command_selection(
                    cfg=effective,
                    verify_cmd=verify_cmd,
                    task=task,
                    root=paths.root,
                    plan_requirements=[
                        str(item).strip()
                        for item in (plan.get("requirements") or [])
                        if str(item).strip()
                    ],
                )
                verify_commands = list(verify_selection.commands)
                verify_command_source = verify_selection.source
                run_cfg.verify_commands = list(verify_commands)
            blockers = _task_dependency_blockers(plan, task)
            if blockers:
                console.print(
                    "[red]Forge error:[/red] Dependencies are not done: " + ", ".join(blockers)
                )
                raise typer.Exit(code=2)

        pr_base_branch: str | None = None
        pr_task_branch: str | None = None
        commit_hash: str | None = None
        merge_commit_hash: str | None = None
        merge_result: str | None = None
        allowed_scope = (
            normalize_scope_patterns(task, root=paths.root) if scope_mode != "off" else []
        )
        scope_warnings: list[str] = []
        review_blocked = False
        verify_blocked = False
        verify_summary: str | None = None
        verify_payload: dict[str, Any] | None = None
        merge_conflict_detected = False
        conflict_review_path: Path | None = None
        verify_path = paths.execution_verify_dir / f"{_safe_task_file_component(task_id)}.txt"
        remote_settings = load_remote_settings_from_env()
        remote_record: dict[str, Any] | None = None
        conflict_auto_settings = load_conflict_auto_resolve_settings(cfg=effective)

        if pr:
            try:
                ensure_git_available()
                ensure_git_repo(paths.root)
                ensure_clean_for_pr(paths.root)
                pr_base_branch = base_branch.strip() if base_branch else current_branch(paths.root)
                if not pr_base_branch:
                    raise GitOpsError("base branch is empty")
                pr_task_branch = str(task.get("branch") or "").strip()
                if not pr_task_branch:
                    pr_task_branch = generate_task_branch_name(
                        task_id,
                        str(task.get("title") or ""),
                    )
                    task["branch"] = pr_task_branch
                    save_plan(paths, plan)
                checkout_branch(paths.root, pr_task_branch, base_branch=pr_base_branch)
                merge_result = "not merged"
            except GitOpsError as e:
                console.print(f"[red]Forge error:[/red] {e}")
                raise typer.Exit(code=2) from e

        ensure_execution_dirs(paths)
        set_task_status(plan, task_id, "in_progress")
        save_plan(paths, plan)

        started_at = now_iso()
        run_non_interactive = _is_non_interactive_terminal()
        prompt_verification_enabled = verify_mode != "off" and bool(verify_commands)
        prepared_knowledge = _prepare_task_execution_knowledge(
            run_paths=paths,
            task=task,
            selection_label="execution",
        )
        runtime_session_id = _safe_task_file_component(task_id)
        task_mcp_scope, task_mcp_scope_warnings = normalize_task_mcp_scope(
            task.get("mcp_scope"),
            warning_prefix=f"Task {task_id}",
        )
        scope_warnings.extend(task_mcp_scope_warnings)
        instruction = ""
        task_image_paths: list[str] | None = None
        budget_artifact_path = paths.execution_budgets_dir / f"{runtime_session_id}.json"
        runtime_sessions_dir = _execution_private_sessions_dir(
            cfg=run_cfg,
            run_id=paths.run_id,
            task_id=task_id,
            workspace_root=paths.root,
        )
        _cleanup_execution_private_sessions_dir(runtime_sessions_dir)
        task_attempts_raw = task.get("attempts")
        try:
            task_attempt_count = (
                max(0, int(task_attempts_raw if task_attempts_raw is not None else 0)) + 1
            )
        except (TypeError, ValueError):
            task_attempt_count = 1
        task_step_budget = _resolve_managed_task_step_budget(
            cfg=run_cfg,
            plan=plan,
            task=task,
            kind="managed_task",
            mode=effective_mode,
            verification_enabled=verify_mode != "off",
            max_steps_override=(effective.max_steps if max_steps_provided else None),
            attempt_count=task_attempt_count,
            image_count=0,
        )
        head_before_run = head_commit(paths.root) if paths.has_head_commit else None
        before_runtime_snapshot: dict[str, str] | None = None
        reporting_baseline: Any | None = None
        recording_surface = RecordingSurface(_make_rich_surface(console=console))
        asset_setup_error: str | None = None
        asset_setup_warnings: list[str] = []
        asset_usage_logger = AssetUsageLogger(run_paths=paths, task_id=task_id)
        asset_model_registry = ModelRegistry(cfg=run_cfg)
        asset_surface = (
            build_asset_surface(
                cfg=run_cfg,
                run_paths=paths,
                model_registry=asset_model_registry,
            )
            if run_cfg.assets.enabled
            else None
        )
        task_asset_mirror = _empty_forge_exec_task_asset_mirror(paths.root, task_id=task_id)
        if asset_surface is not None:
            try:
                task_asset_mirror = mirror_task_assets(
                    task=task,
                    plan=plan,
                    surface=asset_surface,
                    workspace_path=paths.root,
                )
            except AssetError as exc:
                if run_cfg.assets.worker.fail_on_mirror_error:
                    asset_setup_error = (
                        f"forge exec asset mirror failed: {sanitize_error_summary(str(exc))}"
                    )
                else:
                    asset_setup_warnings.append(
                        "forge exec asset mirror skipped: "
                        f"{sanitize_optional_error_summary(str(exc))}"
                    )
        scope_warnings.extend(asset_setup_warnings)
        for entry in [
            *task_asset_mirror.primary,
            *task_asset_mirror.may_need,
            *task_asset_mirror.pinned,
        ]:
            asset_usage_logger.mirror(
                asset_id=entry.asset_id,
                kind=entry.kind,
                status=entry.status,
            )
        has_mirrored_task_assets = bool(
            task_asset_mirror.primary or task_asset_mirror.may_need or task_asset_mirror.pinned
        )

        run_code = 1
        run_err: str | None = asset_setup_error
        task_mcp_manager: Any | None = None
        asset_allocation: TaskAssetAllocation | None = None
        try:
            task_mcp_manager = _build_forge_task_scoped_mcp_manager(
                workspace_root=paths.root,
                session_id=runtime_session_id,
                task_scope=task_mcp_scope,
            )
            mcp_context_section = _build_forge_mcp_execution_context_section(
                task_scope=task_mcp_scope,
                mcp_manager=task_mcp_manager,
            )
            instruction_bundle = _build_forge_exec_instruction_bundle(
                plan=plan,
                task=task,
                root=paths.root,
                cfg=run_cfg,
                role_model=run_cfg.model,
                mode=effective_mode,
                yes=yes,
                deny_write_prefixes=[".sylliptor"],
                allow_write_globs=allowed_scope if scope_mode == "strict" else None,
                non_interactive=run_non_interactive,
                verification_enabled=prompt_verification_enabled,
                authoritative_verification_commands=(
                    verify_commands if prompt_verification_enabled else None
                ),
                api_key=api_key_override,
                subagents_enabled=False,
                leading_sections=[prepared_knowledge.prompt_section, mcp_context_section],
            )
            relevant_assets_section = ""
            if asset_surface is not None and task_asset_mirror.primary:
                asset_allocation = allocate_task_assets(
                    task=task,
                    plan=plan,
                    mirror=task_asset_mirror,
                    cfg=run_cfg,
                    model_registry=asset_model_registry,
                    instruction_token_budget=instruction_bundle.budget.final_instruction_budget,
                    api_key=api_key_override,
                )
                for decision in asset_allocation.decisions:
                    asset_usage_logger.allocation_decision(
                        asset_id=decision.asset_id,
                        mode=decision.mode,
                    )
                relevant_assets_section = render_relevant_assets_section(
                    mirror=task_asset_mirror,
                    allocation=asset_allocation,
                    cfg=run_cfg,
                    surface=asset_surface,
                    model_registry=asset_model_registry,
                    api_key=api_key_override,
                )
            elif asset_surface is not None and (
                task_asset_mirror.may_need or task_asset_mirror.pinned
            ):
                asset_allocation = TaskAssetAllocation(
                    task_id=task_id,
                    decisions=[],
                    elapsed_ms=0,
                    model=None,
                    tokens_used={},
                    fallback_used=False,
                    fallback_reason=None,
                )
                relevant_assets_section = render_relevant_assets_section(
                    mirror=task_asset_mirror,
                    allocation=asset_allocation,
                    cfg=run_cfg,
                    surface=asset_surface,
                    model_registry=asset_model_registry,
                    api_key=api_key_override,
                )
            if relevant_assets_section:
                instruction_bundle = _build_forge_exec_instruction_bundle(
                    plan=plan,
                    task=task,
                    root=paths.root,
                    cfg=run_cfg,
                    role_model=run_cfg.model,
                    mode=effective_mode,
                    yes=yes,
                    deny_write_prefixes=[".sylliptor"],
                    allow_write_globs=allowed_scope if scope_mode == "strict" else None,
                    non_interactive=run_non_interactive,
                    verification_enabled=prompt_verification_enabled,
                    authoritative_verification_commands=(
                        verify_commands if prompt_verification_enabled else None
                    ),
                    api_key=api_key_override,
                    subagents_enabled=False,
                    leading_sections=[prepared_knowledge.prompt_section, mcp_context_section],
                    relevant_assets_section=relevant_assets_section,
                )
            instruction = instruction_bundle.instruction
            _write_execution_context_artifact(
                paths=paths,
                task_id=task_id,
                context_text=instruction_bundle.artifact_text,
            )
            task_image_paths = _combined_forge_exec_image_paths(
                legacy_paths=list(instruction_bundle.image_paths),
                mirror=task_asset_mirror,
                allocation=asset_allocation,
                cfg=run_cfg,
                role_model=run_cfg.model,
                model_registry=asset_model_registry,
                usage_logger=asset_usage_logger,
            )
            task_step_budget = _resolve_managed_task_step_budget(
                cfg=run_cfg,
                plan=plan,
                task=task,
                kind="managed_task",
                mode=effective_mode,
                verification_enabled=verify_mode != "off",
                max_steps_override=(effective.max_steps if max_steps_provided else None),
                attempt_count=task_attempt_count,
                image_count=len(task_image_paths or []),
            )
            budget_artifact_payload = instruction_bundle.to_budget_artifact_payload()
            budget_artifact_payload["step_budget"] = task_step_budget.to_payload()
            budget_artifact_path = _write_execution_budget_artifact(
                paths=paths,
                task_id=task_id,
                payload=budget_artifact_payload,
            )
            before_runtime_snapshot = _snapshot_runtime_tree(paths.root)
            reporting_baseline = _capture_task_local_workspace_baseline(
                paths.root,
                before_commit=head_before_run,
            )
            if run_err is None:
                if asset_surface is not None and has_mirrored_task_assets:
                    task_mcp_manager = compose_worker_asset_mcp_manager(
                        base_manager=task_mcp_manager,
                        asset_manager=build_worker_asset_mcp_manager(
                            cfg=run_cfg,
                            surface=asset_surface,
                            model_registry=asset_model_registry,
                            mirror=task_asset_mirror,
                            usage_logger=asset_usage_logger,
                            api_key=api_key_override,
                        ),
                    )
                run_code = run_agent(
                    cfg=run_cfg,
                    root=paths.root,
                    instruction=instruction,
                    mode=effective_mode,
                    runtime_kind=RuntimeKind.FORGE_EXEC,
                    yes=yes,
                    max_steps=task_step_budget.resolved_max_steps,
                    no_log=no_log,
                    api_key_override=api_key_override,
                    console=console,
                    surface=recording_surface,
                    image_paths=task_image_paths,
                    deny_write_prefixes=[".sylliptor"],
                    allow_write_globs=allowed_scope if scope_mode == "strict" else None,
                    non_interactive=run_non_interactive,
                    session_log_dir_override=runtime_sessions_dir,
                    session_id_override=runtime_session_id,
                    usage_role=f"forge_exec:{task_id}",
                    enable_compaction=False,
                    enable_tool_output_offload=True,
                    enable_conversation_summarization=True,
                    compaction_profile="execution",
                    enable_chat_turn_step_budget=False,
                    one_shot_execution=True,
                    verification_enabled=prompt_verification_enabled,
                    authoritative_verification_commands=(
                        verify_commands if prompt_verification_enabled else None
                    ),
                    subagents_enabled=False,
                    enforce_explicit_subagent_requests=False,
                    mcp_manager=task_mcp_manager,
                )
        except Exception as e:  # noqa: BLE001
            run_code = 1
            run_err = str(e)
            if not instruction:
                _write_execution_context_artifact(
                    paths=paths,
                    task_id=task_id,
                    context_text=(
                        "# Task Context Pack\n\n"
                        "Task execution setup failed before the agent started.\n\n"
                        f"- Error: {run_err}\n"
                    ),
                )
            if not budget_artifact_path.exists():
                _write_execution_budget_artifact(
                    paths=paths,
                    task_id=task_id,
                    payload={
                        "error": run_err,
                        "step_budget": task_step_budget.to_payload(),
                    },
                )
            if before_runtime_snapshot is None:
                before_runtime_snapshot = _snapshot_runtime_tree(paths.root)
            if reporting_baseline is None:
                reporting_baseline = _capture_task_local_workspace_baseline(
                    paths.root,
                    before_commit=head_before_run,
                )
        finally:
            if task_mcp_manager is not None:
                task_mcp_manager.close()

        assert before_runtime_snapshot is not None
        assert reporting_baseline is not None
        after_runtime_snapshot = _snapshot_runtime_tree(paths.root)
        runtime_artifact_changes = _runtime_snapshot_changed_files(
            before_runtime_snapshot,
            after_runtime_snapshot,
        )
        authorized_runtime_side_effects = _authorized_custom_tool_runtime_side_effects(
            sessions_dir=runtime_sessions_dir,
            session_id=runtime_session_id,
        )
        runtime_artifact_changes = [
            path for path in runtime_artifact_changes if path not in authorized_runtime_side_effects
        ]
        runtime_artifacts_changed = bool(runtime_artifact_changes)
        if asset_allocation is not None:
            write_task_asset_allocation(
                run_paths=paths,
                allocation=asset_allocation,
                started_at=started_at,
            )
        asset_usage_logger.summary(
            primary_count=len(task_asset_mirror.primary),
            may_need_count=len(task_asset_mirror.may_need),
            pinned_count=len(task_asset_mirror.pinned),
        )
        try:
            exec_artifacts = _write_exec_log_artifacts(
                paths=paths,
                task_id=task_id,
                cfg=run_cfg,
                no_log=no_log,
                before_logs=None,
                sessions_dir=runtime_sessions_dir,
                expected_session_id=runtime_session_id,
            )
        finally:
            _cleanup_execution_private_sessions_dir(runtime_sessions_dir)

        safe_task_component = _safe_task_file_component(task_id)
        patch_path = paths.execution_patches_dir / f"{safe_task_component}.diff"
        scratch_artifact_dir = paths.execution_dir / "scratch" / safe_task_component
        scratch_artifact_dir.mkdir(parents=True, exist_ok=True)
        success = run_code == 0 and not runtime_artifacts_changed
        pr_report_state_upgraded = False
        head_after_run = head_commit(paths.root) if paths.has_head_commit else None
        try:
            report_diff = _build_task_local_workspace_reporting_diff(
                paths.root,
                baseline=reporting_baseline,
                after_commit=head_after_run,
            )
        finally:
            _cleanup_task_local_workspace_baseline(reporting_baseline)
        patch_path.write_text(report_diff.patch_text, encoding="utf-8")
        scratch_scope_diagnostics = relocate_known_scratch_artifacts(
            root=paths.root,
            artifact_dir=scratch_artifact_dir,
        )
        relocated_scratch_paths = {item.path for item in scratch_scope_diagnostics}
        changed_files = list(report_diff.changed_files)
        if relocated_scratch_paths:
            changed_files = [path for path in changed_files if path not in relocated_scratch_paths]
        agent_added_non_material_paths: list[str] = []
        if head_before_run and head_after_run and head_after_run != head_before_run:
            agent_added_non_material_paths = [
                path
                for path in added_files_since(
                    paths.root,
                    before_commit=head_before_run,
                    after_commit=head_after_run,
                )
                if is_non_material_untracked_path(path)
            ]
            if agent_added_non_material_paths:
                changed_files = [
                    path for path in changed_files if path not in agent_added_non_material_paths
                ]
        pr_material_changed_files = (
            _drop_parent_directory_placeholders(
                _merge_changed_files(
                    list(changed_files),
                    list_changed_files_including_untracked(paths.root),
                )
            )
            if pr
            else list(changed_files)
        )
        scope_changed_files = pr_material_changed_files if pr else changed_files
        scope_inspection_error = report_diff.inspection_error
        scope_violation_files: list[str] = []
        scope_diagnostics = [item.to_payload() for item in scratch_scope_diagnostics]
        for diagnostic in scratch_scope_diagnostics:
            scope_warnings.append(
                "Scope recovery: "
                f"{diagnostic.classification} for {diagnostic.path} "
                f"({diagnostic.reason_code}; action={diagnostic.recommended_action})."
            )
        material_changes_detected = bool(pr_material_changed_files)
        nonzero_agent_exit = run_code != 0 and run_err is None
        strict_scope_blocked = False
        can_attempt_pr_flow = False
        pr_nonzero_salvage_allowed = False
        pr_nonzero_salvage_attempted = False
        no_material_changes_blocked = False
        result_kind: str | None = None
        noop_reason: str | None = None
        analysis_only_noop_accepted = False
        if scope_mode in {"warn", "strict"}:
            if scope_inspection_error:
                if scope_mode == "strict":
                    strict_scope_blocked = True
                    success = False
                    run_err = (run_err + "; " if run_err else "") + scope_inspection_error
                else:
                    scope_warnings.append(scope_inspection_error)
            scope_assessment = assess_scope_changes(
                scope_changed_files,
                allowed_scope,
                task=task,
                root=paths.root,
                extra_diagnostics=scratch_scope_diagnostics,
            )
            scope_changed_files = list(scope_assessment.effective_changed_files)
            scope_diagnostics = [item.to_payload() for item in scope_assessment.diagnostics]
            for diagnostic in scope_assessment.diagnostics:
                if diagnostic.allowed:
                    warning = (
                        "Scope recovery: "
                        f"{diagnostic.classification} for {diagnostic.path} "
                        f"({diagnostic.reason_code}; action={diagnostic.recommended_action})."
                    )
                    if warning not in scope_warnings:
                        scope_warnings.append(warning)
            if not scope_assessment.ok:
                violations = scope_assessment.blocking_paths
                scope_violation_files = list(violations)
                preview = ", ".join(violations[:20])
                if len(violations) > 20:
                    preview += ", ..."
                classes = sorted(
                    {
                        str(item.get("classification") or "unknown")
                        for item in scope_diagnostics
                        if not bool(item.get("allowed"))
                    }
                )
                scope_msg = (
                    f"Out-of-scope file changes detected ({len(violations)}): {preview}. "
                    f"Allowed scope: {allowed_scope or ['(none)']}."
                )
                if classes:
                    scope_msg += f" Scope classifications: {', '.join(classes)}."
                if scope_mode == "strict":
                    strict_scope_blocked = True
                    success = False
                    run_err = (
                        (run_err + "; " if run_err else "")
                        + scope_msg
                        + " Task was blocked due to strict scope isolation."
                    )
                else:
                    scope_warnings.append(scope_msg)

        if pr and not runtime_artifacts_changed and not material_changes_detected:
            success = False

        if run_code == 0 and not runtime_artifacts_changed and not material_changes_detected:
            if _task_is_analysis_only(task):
                if pr:
                    if not pr_base_branch:
                        success = False
                        run_err = (run_err + "; " if run_err else "") + (
                            "missing PR base branch context for analysis-only no-op cleanup"
                        )
                        merge_result = "not merged: analysis-only no-op cleanup failed"
                    else:
                        try:
                            checkout_branch(
                                paths.root,
                                pr_base_branch,
                                base_branch=pr_base_branch,
                            )
                            if (
                                keep_branch
                                or not pr_task_branch
                                or pr_task_branch == pr_base_branch
                            ):
                                merge_result = (
                                    "no merge required (analysis-only no-op; branch kept)"
                                )
                            else:
                                try:
                                    delete_branch(paths.root, pr_task_branch)
                                    merge_result = (
                                        "no merge required (analysis-only no-op; branch deleted)"
                                    )
                                except GitOpsError as cleanup_err:
                                    scope_warnings.append(
                                        "Branch cleanup warning: "
                                        f"failed to delete {pr_task_branch}: {cleanup_err}"
                                    )
                                    merge_result = (
                                        "no merge required (analysis-only no-op; "
                                        f"branch delete warning: {cleanup_err})"
                                    )
                            success = True
                        except GitOpsError as e:
                            success = False
                            run_err = (run_err + "; " if run_err else "") + (
                                f"PR no-op cleanup failed: {e}"
                            )
                            merge_result = f"not merged: analysis-only no-op cleanup failed: {e}"
                else:
                    success = True
                if success:
                    result_kind = "success_noop"
                    noop_reason = "analysis_only"
                    analysis_only_noop_accepted = True
                    verify_summary = "verification skipped: analysis-only task made no changes"
            elif _task_declares_explicit_write_scope(task):
                no_material_changes_blocked = True
                success = False
                run_err = (run_err + "; " if run_err else "") + (
                    "No material file changes were detected for a task with explicit write_scope. "
                    "The task was rejected because its expected local file update was not produced."
                )

        pr_nonzero_salvage_allowed = (
            pr
            and nonzero_agent_exit
            and verify_mode == "strict"
            and bool(verify_commands)
            and not runtime_artifacts_changed
            and material_changes_detected
            and not strict_scope_blocked
        )

        if nonzero_agent_exit and pr_nonzero_salvage_allowed:
            pr_nonzero_salvage_attempted = True
            scope_warnings.append(
                f"PR flow attempted to salvage a non-zero agent exit ({run_code}); "
                "acceptance requires strict verification and PR gates."
            )
        elif nonzero_agent_exit:
            success = False
            run_err = (run_err + "; " if run_err else "") + (
                f"agent exited non-zero ({run_code}); refusing to accept partial task result"
            )

        can_attempt_pr_flow = (
            pr
            and not runtime_artifacts_changed
            and material_changes_detected
            and not strict_scope_blocked
            and (run_code == 0 or pr_nonzero_salvage_allowed)
        )

        if can_attempt_pr_flow:
            success = True
            if not pr_base_branch or not pr_task_branch:
                success = False
                run_err = (run_err + "; " if run_err else "") + "missing PR branch context"
                merge_result = "not merged"
            else:
                try:
                    non_material_untracked_paths = list_untracked_packaging_metadata_paths(
                        paths.root
                    )
                    stage_all(paths.root)
                    unstage_staged_prefixes(
                        paths.root,
                        [".sylliptor", ".sylliptor_images", "sylliptor-feedback"],
                    )
                    ensure_not_staged_prefixes(
                        paths.root,
                        [".sylliptor", ".sylliptor_images", "sylliptor-feedback"],
                    )
                    if non_material_untracked_paths:
                        unstage_staged_paths(paths.root, non_material_untracked_paths)
                        ensure_not_staged_paths(paths.root, non_material_untracked_paths)
                    if has_grounded_rust_target_runtime_artifacts(paths.root):
                        staged_now = staged_files(paths.root)
                        unstage_staged_runtime_artifacts(
                            paths.root,
                            current_paths=staged_now,
                        )
                        staged_now = staged_files(paths.root)
                        ensure_not_staged_runtime_artifacts(
                            paths.root,
                            current_paths=staged_now,
                        )
                    commit_title = str(task.get("title") or "").strip() or "task update"
                    commit_hash = commit_all(
                        paths.root,
                        message=f"{task_id}: {commit_title}",
                    )
                    patch_text = format_patch_stdout(paths.root, base_branch=pr_base_branch)
                    patch_path.write_text(
                        patch_text if patch_text else "(empty format-patch output)\n",
                        encoding="utf-8",
                    )
                    changed_files = changed_files_between(
                        paths.root,
                        revspec=f"{pr_base_branch}..HEAD",
                    )
                    pr_report_state_upgraded = True
                except GitOpsError as e:
                    success = False
                    run_err = (run_err + "; " if run_err else "") + f"PR flow failed: {e}"
                    merge_result = f"not merged: {e}"

                if success and remote_settings.enabled:
                    if not pr_task_branch:
                        success = False
                        run_err = (
                            run_err + "; " if run_err else ""
                        ) + "missing task branch for remote sync"
                        merge_result = "not merged: remote sync branch context missing"
                    else:
                        remote_name = remote_settings.remote_name
                        provider = "unknown"
                        remote_record = init_remote_record(
                            task_id=task_id,
                            remote=remote_name,
                            provider=provider,
                        )
                        remote_errors = remote_record["errors"]
                        assert isinstance(remote_errors, list)
                        try:
                            remote_url = get_remote_url(paths.root, remote_name)
                            provider = resolve_provider(
                                settings_provider=remote_settings.provider,
                                remote_url=remote_url,
                            )
                            remote_record["provider"] = provider
                        except RemoteSyncError as e:
                            msg = f"remote discovery failed: {e}"
                            remote_errors.append(msg)
                            if remote_settings.strict:
                                success = False
                                run_err = (run_err + "; " if run_err else "") + msg
                                merge_result = f"not merged: {msg}"
                            else:
                                scope_warnings.append(msg)

                        if success:
                            pushed_branch, branch_output = push_branch(
                                paths.root,
                                remote=remote_name,
                                branch=pr_task_branch,
                            )
                            remote_record["pushed_branch"] = pushed_branch
                            remote_record["branch_push_output"] = truncate_output(branch_output)
                            if not pushed_branch:
                                msg = (
                                    f"remote branch push failed: {branch_output or 'unknown error'}"
                                )
                                remote_errors.append(msg)
                                if remote_settings.strict:
                                    success = False
                                    run_err = (run_err + "; " if run_err else "") + msg
                                    merge_result = f"not merged: {msg}"
                                else:
                                    scope_warnings.append(msg)

                        if success and remote_settings.create_pr and remote_record is not None:
                            created_pr, pr_url, pr_id, pr_output = ensure_pr_or_mr(
                                paths.root,
                                provider=str(remote_record.get("provider") or "unknown"),
                                base_branch=pr_base_branch,
                                head_branch=pr_task_branch,
                                title=(
                                    f"{task_id}: "
                                    f"{str(task.get('title') or '').strip() or 'task update'}"
                                ),
                                body=instruction[:4000],
                            )
                            remote_record["created_pr"] = created_pr
                            remote_record["pr_url"] = pr_url
                            remote_record["pr_number_or_iid"] = pr_id
                            remote_record["pr_output"] = truncate_output(pr_output)
                            if created_pr and pr_url:
                                task["remote_pr_url"] = pr_url
                                task["remote_provider"] = str(
                                    remote_record.get("provider") or "unknown"
                                )
                                save_plan(paths, plan)
                            if not created_pr:
                                msg = (
                                    f"remote PR/MR creation failed: {pr_output or 'unknown error'}"
                                )
                                remote_errors.append(msg)
                                if remote_settings.strict:
                                    success = False
                                    run_err = (run_err + "; " if run_err else "") + msg
                                    merge_result = f"not merged: {msg}"
                                else:
                                    scope_warnings.append(msg)

                        if remote_record is not None:
                            write_remote_record(
                                execution_dir=paths.execution_dir,
                                task_id=task_id,
                                record=remote_record,
                            )

                if success and verify_mode != "off" and verify_commands:
                    before_verify_snapshot = _snapshot_workspace_tree(paths.root)
                    verify_result = run_task_verification(
                        root=paths.root,
                        commands=verify_commands,
                        artifact_path=verify_path,
                        cfg=effective,
                    )
                    verify_summary = verify_result.summary
                    verify_payload = verify_run_result_to_payload(
                        root=paths.root,
                        result=verify_result,
                    )
                    after_verify_snapshot = _snapshot_workspace_tree(paths.root)
                    verify_mutation_diff = _build_workspace_snapshot_reporting_diff(
                        paths.root,
                        before_snapshot=before_verify_snapshot,
                        after_snapshot=after_verify_snapshot,
                    )
                    verify_mutation_paths = list(verify_mutation_diff.changed_files)
                    if verify_mutation_paths:
                        preview = ", ".join(verify_mutation_paths[:20])
                        if len(verify_mutation_paths) > 20:
                            preview += ", ..."
                        verify_mutation_msg = (
                            "Verification commands modified repository state after the task commit "
                            f"({len(verify_mutation_paths)}): {preview}."
                        )
                        if scope_mode in {"warn", "strict"}:
                            scope_assessment = assess_scope_changes(
                                verify_mutation_paths,
                                allowed_scope,
                                task=task,
                                root=paths.root,
                            )
                            scope_diagnostics.extend(
                                item.to_payload() for item in scope_assessment.diagnostics
                            )
                            if not scope_assessment.ok:
                                verify_scope_violations = scope_assessment.blocking_paths
                                scope_violation_files = _merge_changed_files(
                                    scope_violation_files,
                                    verify_scope_violations,
                                )
                                classes = sorted(
                                    {
                                        item.classification
                                        for item in scope_assessment.diagnostics
                                        if not item.allowed
                                    }
                                )
                                scope_msg = (
                                    f"Out-of-scope file changes detected ({len(verify_scope_violations)}): "
                                    f"{', '.join(verify_scope_violations[:20])}"
                                )
                                if len(verify_scope_violations) > 20:
                                    scope_msg += ", ..."
                                scope_msg += (
                                    f". Allowed scope: {allowed_scope or ['(none)']}."
                                    " Task was blocked due to strict scope isolation."
                                    " Verification commands modified repository state after the task commit."
                                )
                                if classes:
                                    scope_msg += f" Scope classifications: {', '.join(classes)}."
                                if scope_mode == "strict":
                                    success = False
                                    commit_hash = None
                                    merge_result = "not merged: strict scope isolation blocked verification-time writes"
                                    run_err = (run_err + "; " if run_err else "") + scope_msg
                                    changed_files = _merge_changed_files(
                                        changed_files,
                                        verify_mutation_paths,
                                    )
                                    _append_patch_debug_section(
                                        patch_path,
                                        title="Post-verification workspace diff",
                                        patch_text=verify_mutation_diff.patch_text,
                                    )
                                else:
                                    scope_warnings.append(scope_msg)
                            elif scope_mode == "strict":
                                success = False
                                commit_hash = None
                                merge_result = (
                                    "not merged: verification commands modified repository state"
                                )
                                run_err = (run_err + "; " if run_err else "") + verify_mutation_msg
                                changed_files = _merge_changed_files(
                                    changed_files, verify_mutation_paths
                                )
                                _append_patch_debug_section(
                                    patch_path,
                                    title="Post-verification workspace diff",
                                    patch_text=verify_mutation_diff.patch_text,
                                )
                            else:
                                scope_warnings.append(verify_mutation_msg)
                    if not verify_result.all_passed:
                        success = False
                        verify_blocked = True
                        merge_result = (
                            "not merged: strict verification failed"
                            if verify_mode == "strict"
                            else "not merged: verification failed"
                        )
                        run_err = (run_err + "; " if run_err else "") + (
                            f"verification failed: {verify_result.summary}"
                        )
                elif verify_mode == "strict":
                    success = False
                    verify_blocked = True
                    merge_result = "not merged: strict verification unavailable"
                    run_err = (run_err + "; " if run_err else "") + (
                        "strict verification requires authoritative commands, but none were available"
                    )
                    verify_summary = "verification skipped: no authoritative commands available"
                elif verify_mode != "off":
                    verify_summary = "verification skipped: no authoritative commands available"
                elif verify_mode == "off":
                    verify_summary = "verification disabled (--verify off)"

                if success and review:
                    try:
                        review_outcome = review_task(
                            paths=paths,
                            plan=plan,
                            task=task,
                            cfg=effective,
                            api_key_override=api_key_override,
                            verification_payload_override=verify_payload,
                        )
                        if not review_outcome.approved:
                            success = False
                            review_blocked = True
                            merge_result = "not merged: review requested changes"
                    except ReviewError as e:
                        success = False
                        run_err = (run_err + "; " if run_err else "") + f"review failed: {e}"
                        merge_result = f"not merged: review failed: {e}"

                if success:
                    try:
                        merge_title = str(task.get("title") or "").strip()
                        merge_message = (
                            f"Merge {task_id}: {merge_title}" if merge_title else f"Merge {task_id}"
                        )
                        merge_commit_hash = merge_no_ff(
                            paths.root,
                            base_branch=pr_base_branch,
                            task_branch=pr_task_branch,
                            message=merge_message,
                        )
                        if keep_branch:
                            merge_result = f"merged into {pr_base_branch} (branch kept)"
                        else:
                            try:
                                delete_branch(paths.root, pr_task_branch)
                                merge_result = f"merged into {pr_base_branch} (branch deleted)"
                            except GitOpsError as cleanup_err:
                                scope_warnings.append(
                                    "Branch cleanup warning: "
                                    f"failed to delete {pr_task_branch}: {cleanup_err}"
                                )
                                merge_result = (
                                    f"merged into {pr_base_branch} "
                                    f"(branch delete warning: {cleanup_err})"
                                )
                    except GitOpsError as e:
                        success = False
                        run_err = (run_err + "; " if run_err else "") + f"PR flow failed: {e}"
                        unmerged = list_unmerged_files(paths.root)
                        if unmerged and pr_base_branch and pr_task_branch:
                            merge_conflict_detected = True
                            context = capture_merge_conflict_context(
                                paths.root,
                                base_branch=pr_base_branch,
                                task_branch=pr_task_branch,
                                merge_error=str(e),
                            )
                            review_outcome = review_merge_conflict(
                                paths=paths,
                                task=task,
                                cfg=effective,
                                api_key_override=api_key_override,
                                context=context,
                                plan=plan,
                            )
                            cleanup_ok, cleanup_log = try_abort_merge(
                                paths.root,
                                base_branch=pr_base_branch,
                            )
                            conflict_artifacts = write_conflict_artifacts(
                                paths=paths,
                                task_id=task_id,
                                context=context,
                                review_json=review_outcome.review_json,
                                review_md=review_outcome.review_markdown,
                                cleanup_log=cleanup_log,
                            )
                            conflict_review_path = conflict_artifacts.review_md_path
                            merge_result = (
                                f"not merged: conflict while merging {pr_task_branch} into "
                                f"{pr_base_branch}"
                            )
                            if review_outcome.skipped_reason:
                                scope_warnings.append(
                                    f"Conflict review note: {review_outcome.skipped_reason}"
                                )
                            if not cleanup_ok:
                                scope_warnings.append(
                                    "Merge cleanup warning: repository state may need manual recovery. "
                                    f"See {conflict_artifacts.cleanup_log_path}"
                                )
                            if can_attempt_conflict_auto_resolve(
                                task=task,
                                settings=conflict_auto_settings,
                            ):
                                bump_conflict_attempt(task)
                                save_plan(paths, plan)
                                auto_outcome = attempt_auto_resolve_conflict(
                                    paths=paths,
                                    plan=plan,
                                    task=task,
                                    cfg=effective,
                                    api_key_override=api_key_override,
                                    base_branch=pr_base_branch,
                                    task_branch=pr_task_branch,
                                    keep_worktrees=False,
                                    settings=conflict_auto_settings,
                                    verify_commands=(
                                        verify_commands if verify_mode != "off" else []
                                    ),
                                )
                                if auto_outcome.success:
                                    success = True
                                    run_err = None
                                    merge_conflict_detected = False
                                    merge_commit_hash = auto_outcome.merge_commit_hash
                                    merge_result = f"auto-resolved and merged into {pr_base_branch}"
                                    if auto_outcome.warnings:
                                        scope_warnings.extend(auto_outcome.warnings)
                                    conflict_review_path = auto_outcome.report_path
                                else:
                                    scope_warnings.append(
                                        "Conflict auto-resolve failed: "
                                        f"{auto_outcome.error or 'unknown error'}"
                                    )
                        else:
                            merge_result = f"not merged: {e}"

                if (
                    success
                    and remote_settings.enabled
                    and remote_record is not None
                    and pr_base_branch
                ):
                    pushed_base, base_output = push_base(
                        paths.root,
                        remote=str(remote_record.get("remote") or remote_settings.remote_name),
                        base_branch=pr_base_branch,
                    )
                    remote_record["pushed_base"] = pushed_base
                    remote_record["base_push_output"] = truncate_output(base_output)
                    if not pushed_base:
                        msg = f"remote base push failed: {base_output or 'unknown error'}"
                        raw_errors = remote_record.get("errors")
                        if isinstance(raw_errors, list):
                            raw_errors.append(msg)
                        # Local merge already happened; keep success and record warning.
                        scope_warnings.append(msg)
                    write_remote_record(
                        execution_dir=paths.execution_dir,
                        task_id=task_id,
                        record=remote_record,
                    )

        if pr and merge_result is None:
            merge_result = "not merged"

        if pr and not pr_report_state_upgraded:
            recovered_pr_report_state = False
            if commit_hash is not None and pr_base_branch:
                try:
                    patch_text = format_patch_stdout(paths.root, base_branch=pr_base_branch)
                    patch_path.write_text(
                        patch_text if patch_text else "(empty format-patch output)\n",
                        encoding="utf-8",
                    )
                    changed_files = changed_files_between(
                        paths.root,
                        revspec=f"{pr_base_branch}..HEAD",
                    )
                    recovered_pr_report_state = True
                except GitOpsError:
                    recovered_pr_report_state = False
            if not recovered_pr_report_state:
                patch_path.write_text(report_diff.patch_text, encoding="utf-8")
                changed_files = list(report_diff.changed_files)

        if runtime_artifacts_changed:
            summary = "Task failed: agent modified files under .sylliptor/ which is not allowed."
        elif scope_violation_files:
            summary = "Task blocked due to strict scope isolation."
        elif no_material_changes_blocked:
            summary = "Task failed: no material file changes were detected."
        elif verify_blocked:
            summary = "Task blocked by strict verification gate."
        elif review_blocked:
            summary = "Task blocked by review gate (changes requested)."
        elif analysis_only_noop_accepted:
            summary = "Analysis-only task completed successfully with no repository changes."
        elif run_code == 0 and success:
            summary = "Task execution completed successfully."
        elif pr and can_attempt_pr_flow and run_code == 0:
            summary = "Task execution failed during PR flow."
        else:
            summary = "Task execution failed."
        if run_err:
            summary += f" Error: {run_err}"
        if scope_warnings:
            summary += " Warnings: " + " | ".join(scope_warnings)
        if conflict_review_path is not None:
            summary += f" Conflict review: {conflict_review_path}"

        finished_at = now_iso()
        report_verify_commands = verify_commands if pr and verify_mode != "off" else []
        report_path = write_task_report(
            paths=paths,
            task=task,
            result="success" if success else "failure",
            result_kind=result_kind,
            summary=summary,
            started_at=started_at,
            finished_at=finished_at,
            changed_files=changed_files,
            verify_commands=report_verify_commands,
            patch_path=patch_path,
            budget_artifact_path=budget_artifact_path,
            execution_log_artifacts=exec_artifacts,
            verify_artifact_path=verify_path if verify_path.exists() else None,
            verify_summary=verify_summary,
            verify_payload=verify_payload,
            verify_command_source=verify_command_source,
            base_branch=pr_base_branch,
            task_branch=pr_task_branch,
            commit_hash=commit_hash,
            merge_commit_hash=merge_commit_hash,
            merge_result=merge_result,
            salvaged_nonzero_exit=bool(pr_nonzero_salvage_attempted and success),
            noop_reason=noop_reason,
            remote_lines=_remote_report_lines(remote_record),
        )
        persisted_capture = persist_execution_knowledge_capture(
            paths=paths,
            task=task,
            source="forge_exec",
            assistant_message=recording_surface.final_assistant_message,
            artifact_dir=(
                paths.execution_knowledge_capture_dir
                / _safe_task_file_component(task_id)
                / _safe_task_file_component(started_at)
            ),
            report_path=report_path,
            patch_path=patch_path,
            verify_artifact_path=verify_path if verify_path.exists() else None,
            budget_artifact_path=budget_artifact_path,
            session_artifact_dir=exec_artifacts.session_artifact_dir,
        )
        if success:
            promote_validated_knowledge_capture(
                paths=paths,
                task=task,
                artifact_dir=persisted_capture.artifact_dir,
            )
        else:
            mark_knowledge_capture_promotion_skipped(
                artifact_dir=persisted_capture.artifact_dir,
                reason="task execution outcome was not accepted",
            )

        write_task_attempt_entry(
            paths=paths,
            task=task,
            source="forge_exec",
            result="success" if success else "failure",
            summary=summary,
            changed_files=changed_files,
            verify_summary=verify_summary,
            report_path=report_path,
            patch_path=patch_path,
            verify_artifact_path=verify_path if verify_path.exists() else None,
            budget_artifact_path=budget_artifact_path,
            session_artifact_dir=exec_artifacts.session_artifact_dir,
            acceptance_state="accepted" if success else "rejected",
            extra_tags=[
                "execution",
                "sequential",
            ],
        )
        issue_paths = changed_files or list(allowed_scope)
        if runtime_artifacts_changed:
            write_issue_entry(
                paths=paths,
                task=task,
                source="forge_exec",
                title=f"{task_id}: protected .sylliptor mutation attempt",
                summary="Engineer execution attempted to modify protected .sylliptor runtime state.",
                paths_in_scope=issue_paths,
                report_path=report_path,
                patch_path=patch_path,
                verify_artifact_path=verify_path if verify_path.exists() else None,
                budget_artifact_path=budget_artifact_path,
                session_artifact_dir=exec_artifacts.session_artifact_dir,
                tags=["protected_runtime_mutation"],
            )
        elif verify_blocked:
            write_issue_entry(
                paths=paths,
                task=task,
                source="forge_exec",
                title=f"{task_id}: verification failed",
                summary=verify_summary or "Verification blocked task completion.",
                paths_in_scope=issue_paths,
                report_path=report_path,
                patch_path=patch_path,
                verify_artifact_path=verify_path if verify_path.exists() else None,
                budget_artifact_path=budget_artifact_path,
                session_artifact_dir=exec_artifacts.session_artifact_dir,
                tags=["verification_failure"],
            )
        elif review_blocked:
            write_issue_entry(
                paths=paths,
                task=task,
                source="forge_exec",
                title=f"{task_id}: review requested changes",
                summary=summary,
                paths_in_scope=issue_paths,
                report_path=report_path,
                patch_path=patch_path,
                verify_artifact_path=verify_path if verify_path.exists() else None,
                budget_artifact_path=budget_artifact_path,
                session_artifact_dir=exec_artifacts.session_artifact_dir,
                tags=["review_blocked"],
            )
        elif merge_conflict_detected:
            write_issue_entry(
                paths=paths,
                task=task,
                source="forge_exec",
                title=f"{task_id}: merge conflict remains unresolved",
                summary=summary,
                paths_in_scope=issue_paths,
                report_path=report_path,
                patch_path=patch_path,
                verify_artifact_path=verify_path if verify_path.exists() else None,
                budget_artifact_path=budget_artifact_path,
                session_artifact_dir=exec_artifacts.session_artifact_dir,
                tags=["merge_conflict"],
            )
        elif not success:
            write_issue_entry(
                paths=paths,
                task=task,
                source="forge_exec",
                title=f"{task_id}: task execution failed",
                summary=summary,
                paths_in_scope=issue_paths,
                report_path=report_path,
                patch_path=patch_path,
                verify_artifact_path=verify_path if verify_path.exists() else None,
                budget_artifact_path=budget_artifact_path,
                session_artifact_dir=exec_artifacts.session_artifact_dir,
                tags=(
                    ["execution_failure", "scope_violation"]
                    if scope_violation_files
                    else ["execution_failure"]
                ),
            )
        rebuild_knowledge_index(paths)

        if success:
            status = "done"
        elif merge_conflict_detected:
            status = "merge_conflict"
        elif verify_blocked:
            status = "verify_failed"
        elif review_blocked:
            status = "changes_requested"
        else:
            status = "failed"
        set_task_status(plan, task_id, status)
        save_plan(paths, plan)

        console.print(f"Task: {task_id} ({task.get('title', '')})")
        console.print(f"Result: {'success' if success else 'failure'}")
        console.print(f"Report: {report_path}")
        console.print(f"Patch: {patch_path}")
        _print_usage_summary_from_logs(
            console=console,
            title=f"Usage Summary ({task_id})",
            log_paths=([exec_artifacts.log_copy_path] if exec_artifacts.log_retained else []),
        )
        if conflict_review_path is not None:
            console.print(f"Conflict Review: {conflict_review_path}")

        if success:
            raise typer.Exit(code=0)
        raise typer.Exit(code=1)

    finally:
        run_mutation_guard.release()


def forge_plan_impl(cli_mod: Any, *args: Any, **kwargs: Any) -> Any:
    _sync_cli_globals(cli_mod)
    return forge_plan(*args, **kwargs)


def forge_swarm_impl(cli_mod: Any, *args: Any, **kwargs: Any) -> Any:
    _sync_cli_globals(cli_mod)
    return forge_swarm(*args, **kwargs)


def forge_exec_impl(cli_mod: Any, *args: Any, **kwargs: Any) -> Any:
    _sync_cli_globals(cli_mod)
    return forge_exec(*args, **kwargs)
