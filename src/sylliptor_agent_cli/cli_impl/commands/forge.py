from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import typer

from ...config import ConfigError, clone_cfg, load_config
from ...forge import (
    ForgeError,
    attach_asset,
    find_task,
    load_current_run_paths,
    load_plan,
)
from ...review_gate import ReviewError, review_task
from . import _patchable
from ._shared import Mode, _console, _Table
from .forge_asset_view import forge_asset_view_count, forge_asset_view_entries


def _cli_module() -> Any:
    module = sys.modules.get("sylliptor_agent_cli.cli")
    if module is not None:
        return module
    from ... import cli

    return cli


forge_app = typer.Typer(add_completion=False, help="Forge commands.")


@forge_app.command("plan")
def forge_plan(
    ctx: typer.Context = None,
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
    from ..forge import forge_plan_impl

    return forge_plan_impl(_cli_module(), path, create_path, allow_broad_workspace, cli_ctx=ctx)


@forge_app.command("attach")
def forge_attach(
    source_path: Path = typer.Argument(..., help="File path to attach."),
    path: Path = typer.Option(
        Path("."),
        "--path",
        help="Workspace path or repository subdirectory.",
    ),
) -> None:
    console = _console()
    typer.echo(
        "Deprecation: `sylliptor forge attach` is the legacy asset attachment flow.\n"
        "Use `/assets` from chat or `sylliptor forge assets add` for the new flow with\n"
        "LLM comprehension and per-task binding. The legacy command continues to work,\n"
        "and attached assets are migrated on the next plan load.",
        err=True,
    )
    try:
        paths, metadata = attach_asset(path, source_path)
    except ForgeError as e:
        console.print(f"[red]Forge error:[/red] {e}")
        raise typer.Exit(code=2) from e

    console.print(f"Attached to run: {paths.run_id}")
    console.print(f"- original: {metadata.get('original_path')}")
    console.print(f"- stored: {metadata.get('stored_path')}")
    console.print(f"- size: {metadata.get('size_bytes')} bytes")
    if metadata.get("text_copy_path"):
        console.print(f"- extracted text: {metadata.get('text_copy_path')}")


@forge_app.command("show")
def forge_show(
    path: Path = typer.Option(
        Path("."),
        "--path",
        help="Workspace path or repository subdirectory.",
    ),
) -> None:
    console = _console()
    try:
        paths = load_current_run_paths(path)
        plan = load_plan(paths)
    except ForgeError as e:
        console.print(f"[red]Forge error:[/red] {e}")
        raise typer.Exit(code=2) from e

    console.rule("[bold]forge show[/bold]")
    goal = str(plan.get("project_goal") or "").strip() or "(not set)"
    console.print(f"Run ID: {paths.run_id}")
    console.print(f"Project goal: {goal}")

    tasks = plan.get("tasks") or []
    task_table = _Table(title="Tasks")
    task_table.add_column("id")
    task_table.add_column("status")
    task_table.add_column("title")
    task_table.add_column("dependencies")
    if tasks:
        for task in tasks:
            deps = task.get("dependencies") or []
            task_table.add_row(
                str(task.get("id", "")),
                str(task.get("status", "")),
                str(task.get("title", "")),
                ", ".join(str(d) for d in deps) if deps else "-",
            )
    console.print(task_table)

    assets = forge_asset_view_entries(paths, plan)
    asset_table = _Table(title="Assets")
    asset_table.add_column("source")
    asset_table.add_column("stored_path")
    asset_table.add_column("size_bytes")
    if assets:
        for asset in assets:
            asset_table.add_row(
                asset.source,
                asset.stored_path,
                "" if asset.size_bytes is None else str(asset.size_bytes),
            )
    console.print(asset_table)
    if assets:
        names = [asset.display_name for asset in assets if asset.display_name]
        if names:
            console.print(f"Asset files: {', '.join(names)}")


@forge_app.command("status")
def forge_status(
    path: Path = typer.Option(
        Path("."),
        "--path",
        help="Workspace path or repository subdirectory.",
    ),
) -> None:
    console = _console()
    try:
        paths = load_current_run_paths(path)
        plan = load_plan(paths)
    except ForgeError as e:
        console.print(f"[red]Forge error:[/red] {e}")
        raise typer.Exit(code=2) from e

    table = _Table(title="forge status")
    table.add_column("field")
    table.add_column("value")
    table.add_row("run_id", paths.run_id)
    table.add_row("run_dir", os.fspath(paths.run_dir))
    table.add_row("plan_json", os.fspath(paths.plan_json_path))
    table.add_row("plan_md", os.fspath(paths.plan_md_path))
    table.add_row("tasks", str(len(plan.get("tasks") or [])))
    table.add_row("assets", str(forge_asset_view_count(paths, plan)))
    console.print(table)


@forge_app.command("review")
def forge_review(
    task_id: str = typer.Argument(..., help="Task id from plan.json (for example T01)."),
    path: Path = typer.Option(
        Path("."),
        "--path",
        help="Workspace path or repository subdirectory.",
    ),
    model: str | None = typer.Option(None, "--model", help="Model override."),
    base_url: str | None = typer.Option(None, "--base-url", help="Base URL override."),
    temperature: float | None = typer.Option(None, "--temperature", help="Sampling temperature."),
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
) -> None:
    console = _console()
    cfg = _patchable("load_config", load_config)()
    effective = clone_cfg(cfg)
    if base_url is not None:
        effective.base_url = base_url
    if model is not None:
        effective.model = model
    if temperature is not None:
        _cli_module()._apply_temperature_override(effective, temperature)

    try:
        if not effective.model:
            raise ConfigError("Model is not set. Run: sylliptor config set model <MODEL>")
        api_key_override = _cli_module()._resolve_api_key_override(
            api_key=api_key,
            api_key_env=api_key_env,
            api_key_stdin=api_key_stdin,
        )
        paths = load_current_run_paths(path)
        plan = load_plan(paths)
        task = find_task(plan, task_id)
        if task is None:
            raise ForgeError(f"Task not found: {task_id}")
        outcome = _patchable("review_task", review_task)(
            paths=paths,
            plan=plan,
            task=task,
            cfg=effective,
            api_key_override=api_key_override,
        )
    except (ConfigError, ForgeError, ReviewError) as e:
        console.print(f"[red]Forge error:[/red] {e}")
        raise typer.Exit(code=2) from e

    console.print(f"Task: {task_id} ({task.get('title', '')})")
    console.print(f"Approved: {'yes' if outcome.approved else 'no'}")
    console.print(f"Confidence: {outcome.confidence}")
    console.print(f"Summary: {outcome.summary}")
    console.print(f"Review JSON: {outcome.json_path}")
    console.print(f"Review Markdown: {outcome.markdown_path}")
    raise typer.Exit(code=0 if outcome.approved else 1)


@forge_app.command("swarm")
def forge_swarm(
    ctx: typer.Context = None,
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
    from ..forge import forge_swarm_impl

    return forge_swarm_impl(
        _cli_module(),
        path,
        allow_broad_workspace,
        parallel,
        base_branch,
        max_tasks,
        max_attempts,
        dry_run,
        keep_worktrees,
        retry_failed,
        retry_changes_requested,
        only,
        retry_merge_conflicts,
        scope,
        verify,
        verify_cmd,
        integration_verify,
        integration_verify_cmd,
        replan,
        review,
        mode,
        model,
        base_url,
        temperature,
        stream,
        max_steps,
        no_log,
        api_key_env,
        api_key_stdin,
        api_key,
        yes,
        cli_ctx=ctx,
    )


@forge_app.command("exec")
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
    from ..forge import forge_exec_impl

    return forge_exec_impl(
        _cli_module(),
        task_id,
        path,
        mode,
        model,
        base_url,
        temperature,
        stream,
        max_steps,
        no_log,
        api_key_env,
        api_key_stdin,
        api_key,
        pr,
        review,
        base_branch,
        keep_branch,
        scope,
        verify,
        verify_cmd,
        yes,
    )
