from __future__ import annotations

import io
import os
import re
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rich.console import Console

from .agent_loop import run_agent
from .assets import AssetError
from .assets.budget_allocator import (
    TaskAssetAllocation,
    allocate_task_assets,
    write_task_asset_allocation,
)
from .assets.surface import build_asset_surface
from .assets.usage_logger import AssetUsageLogger
from .assets.worker_mirror import TaskAssetMirror, mirror_task_assets
from .assets.worker_section import render_relevant_assets_section
from .assets.worker_tools import build_worker_asset_mcp_manager, compose_worker_asset_mcp_manager
from .config import AppConfig, clone_cfg
from .error_text import sanitize_error_summary, sanitize_optional_error_summary
from .execution_shared import (
    build_execution_reporting_diff_with_commit_range,
    build_task_execution_instruction_bundle,
    build_workspace_snapshot_reporting_diff,
    mirror_plan_into_worktree,
    mirror_selected_knowledge_into_worktree,
    prepare_task_execution_knowledge,
    resolve_managed_task_step_budget,
    safe_task_file_component,
    snapshot_runtime_tree,
    snapshot_workspace_tree,
    write_exec_log_artifacts,
    write_execution_budget_artifact,
    write_execution_context_artifact,
)
from .failure_category import (
    FailureCategory,
    failure_category_value,
    is_provider_throttling_error,
    is_provider_unavailable_error,
)
from .forge import RunPaths, ensure_execution_dirs, now_iso, write_task_report
from .git_ops import (
    GitOpsError,
    added_files_since,
    changed_files_between,
    changed_files_since,
    commit_all,
    ensure_not_staged_paths,
    ensure_not_staged_prefixes,
    ensure_not_staged_runtime_artifacts,
    ensure_runtime_artifact_excludes,
    format_patch_stdout,
    head_commit,
    reset_mixed,
    stage_all,
    staged_files,
    unstage_staged_paths,
    unstage_staged_prefixes,
    unstage_staged_runtime_artifacts,
)
from .knowledge_base import rebuild_knowledge_index, write_issue_entry, write_task_attempt_entry
from .knowledge_capture import (
    RecordingSurface,
    mark_knowledge_capture_promotion_skipped,
    persist_execution_knowledge_capture,
)
from .mcp.manager import create_mcp_manager
from .model_registry import ModelRegistry
from .model_router import ROLE_CODING, resolve_model_for_role
from .repo_scan import RepoScanResult, scan_workspace
from .runtime_artifacts import has_grounded_rust_target_runtime_artifacts
from .runtime_kind import RuntimeKind
from .surface.console import make_console
from .swarm_trace import (
    NoopSwarmTraceSink,
    SwarmTraceSink,
    SwarmWorkerTraceSurface,
    build_swarm_trace_event,
)
from .task_scope import (
    check_scope,
    is_non_material_untracked_path,
    list_changed_files_including_untracked,
    list_untracked_packaging_metadata_paths,
    normalize_scope_patterns,
)
from .verify_gate import (
    ResolvedVerifyCommands,
    VerifyRunResult,
    resolve_authoritative_task_verify_command_selection,
    run_task_verification,
    verify_run_result_to_payload,
)
from .workspace_context import WorkspaceContextError, resolve_workspace_context

_DIAGNOSTIC_TASK_TITLE_RE = re.compile(
    r"^\s*(?:explore|investigate|examine|inspect|locate|find|survey|map|analy[sz]e|diagnose|audit|understand|identify)\b",
    re.IGNORECASE,
)
_MUTATING_TASK_TEXT_RE = re.compile(
    r"\b(?:fix|implement|update|add|change|modify|create|delete|remove|refactor|write)\b",
    re.IGNORECASE,
)
_CONDITIONAL_NOOP_TASK_RE = re.compile(
    r"\b(?:if|when)\b.{0,100}\b(?:exists?|present|available|needed|applicable)\b"
    r"|\bif\s+needed\b|\bwhere\s+applicable\b",
    re.IGNORECASE | re.DOTALL,
)
_VERIFY_FAILED_TEST_RE = re.compile(r"^\s*(?:FAILED|ERROR)\s+([^\s]+)", re.MULTILINE)
_VERIFY_FAILURE_SECTION_RE = re.compile(r"^_{3,}\s+(.+?)\s+_{3,}\s*$", re.MULTILINE)
_VERIFY_ACTIONABLE_FAILURE_RE = re.compile(
    r"^\s*(?:E\s+|AssertionError|ImportError|ModuleNotFoundError|ValueError|TypeError|"
    r"SyntaxError|NameError|AttributeError)(.+)?$",
    re.MULTILINE,
)
_BASELINE_FAILURE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_BASELINE_FAILURE_STOPWORDS = {
    "assert",
    "assertion",
    "command",
    "error",
    "expected",
    "failed",
    "failure",
    "format",
    "formatting",
    "function",
    "got",
    "line",
    "module",
    "pytest",
    "render",
    "render_value",
    "return",
    "returns",
    "style",
    "test",
    "tests",
    "unknown",
    "value",
    "values",
}


@dataclass(frozen=True)
class TaskWorkerResult:
    task_id: str
    title: str
    branch: str
    worktree_path: str
    started_at: str
    finished_at: str
    success: bool
    summary: str
    commit_hash: str | None
    error: str | None
    report_path: str
    patch_path: str
    log_path: str
    log_pointer_path: str
    warnings: list[str]
    changed_files: list[str]
    verify_failed: bool
    verify_summary: str | None
    verify_artifact_path: str | None
    knowledge_capture_artifact_dir: str | None = None
    task_attempt_entry_id: str | None = None
    task_attempt_knowledge_file_path: str | None = None
    verify_payload: dict[str, Any] | None = None
    verify_command_source: str | None = None
    failure_reason: str | None = None
    failure_category: FailureCategory | str | None = None
    scope_violation_files: list[str] | None = None
    allowed_scope: list[str] | None = None
    agent_exit_code: int = 0
    salvaged_nonzero_exit: bool = False
    salvaged_agent_exception: bool = False
    agent_exception_summary: str | None = None
    result_kind: str | None = None
    noop_reason: str | None = None

    @property
    def effective_result_kind(self) -> str:
        if self.result_kind:
            return self.result_kind
        if self.success:
            return "success_commit"
        return "failure"

    @property
    def noop_success(self) -> bool:
        return self.effective_result_kind == "success_noop"

    def to_json(self) -> dict[str, Any]:
        clean_agent_exception_summary = sanitize_optional_error_summary(
            self.agent_exception_summary
        )
        return {
            "task_id": self.task_id,
            "title": self.title,
            "branch": self.branch,
            "worktree_path": self.worktree_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "success": self.success,
            "summary": self.summary,
            "commit_hash": self.commit_hash,
            "error": self.error,
            "report_path": self.report_path,
            "patch_path": self.patch_path,
            "log_path": self.log_path,
            "log_pointer_path": self.log_pointer_path,
            "warnings": self.warnings,
            "changed_files": self.changed_files,
            "verify_failed": self.verify_failed,
            "verify_summary": self.verify_summary,
            "verify_artifact_path": self.verify_artifact_path,
            "knowledge_capture_artifact_dir": self.knowledge_capture_artifact_dir,
            "task_attempt_entry_id": self.task_attempt_entry_id,
            "task_attempt_knowledge_file_path": self.task_attempt_knowledge_file_path,
            "verify_payload": self.verify_payload,
            "verify_command_source": self.verify_command_source,
            "failure_reason": self.failure_reason,
            "failure_category": None
            if self.success
            else failure_category_value(self.failure_category),
            "scope_violation_files": self.scope_violation_files,
            "allowed_scope": self.allowed_scope,
            "agent_exit_code": self.agent_exit_code,
            "salvaged_nonzero_exit": self.salvaged_nonzero_exit,
            "salvaged_agent_exception": self.salvaged_agent_exception,
            "agent_exception_summary": clean_agent_exception_summary,
            "result_kind": self.effective_result_kind,
            "noop_success": self.noop_success,
            "noop_reason": self.noop_reason,
        }


@dataclass(frozen=True)
class ResolvedWorkerVerifyContract:
    cfg: AppConfig
    commands: tuple[str, ...]
    selection: ResolvedVerifyCommands | None


def resolve_worker_verify_contract(
    *,
    cfg: AppConfig,
    verify_mode: str,
    verify_commands: list[str] | None,
    verify_command_selection: ResolvedVerifyCommands | None,
    task: dict[str, Any] | None,
    root: Path,
    repo_scan: RepoScanResult | None = None,
    plan_requirements: list[str] | None = None,
) -> ResolvedWorkerVerifyContract:
    run_cfg = clone_cfg(cfg)
    if verify_mode == "off":
        return ResolvedWorkerVerifyContract(
            cfg=run_cfg,
            commands=(),
            selection=None,
        )

    selection = resolve_authoritative_task_verify_command_selection(
        cfg=cfg,
        verify_cmd=verify_commands,
        task=task,
        root=root,
        repo_scan=repo_scan,
        plan_requirements=plan_requirements,
        selection=verify_command_selection,
        allow_empty_config=True,
    )
    run_cfg.verify_commands = list(selection.commands)
    return ResolvedWorkerVerifyContract(
        cfg=run_cfg,
        commands=selection.commands,
        selection=selection,
    )


def reject_abnormal_success_result(result: TaskWorkerResult) -> TaskWorkerResult:
    if not result.success:
        return result
    abnormal_exit = result.agent_exit_code not in (None, 0)
    salvage_label = result.salvaged_nonzero_exit or result.salvaged_agent_exception
    if not abnormal_exit and not salvage_label:
        return result
    reasons: list[str] = []
    if abnormal_exit:
        reasons.append(f"agent_exit_code={result.agent_exit_code}")
    if result.salvaged_agent_exception:
        reasons.append("salvaged_agent_exception=true")
    if result.salvaged_nonzero_exit:
        reasons.append("salvaged_nonzero_exit=true")
    reason = ", ".join(reasons)
    error = (
        "worker result reported success after abnormal agent termination "
        f"({reason}); refusing to accept partial worker result"
    )
    summary = f"Worker rejected abnormal success result. Error: {error}"
    return replace(
        result,
        success=False,
        summary=summary,
        error=error,
        failure_reason=result.failure_reason or "agent_abnormal_success",
        failure_category=result.failure_category or FailureCategory.IMPLEMENTATION_FAILED,
        salvaged_nonzero_exit=False,
        salvaged_agent_exception=False,
        result_kind=None,
        noop_reason=None,
    )


def _repo_rel(root: Path, path: Path) -> str:
    return os.fspath(path.resolve().relative_to(root.resolve()))


def _normalize_rel_prefix(value: str) -> str:
    cleaned = value.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned.rstrip("/")


def _matches_prefix(rel_path: str, prefix: str) -> bool:
    return rel_path == prefix or rel_path.startswith(prefix + "/")


def _protected_path_violations(paths: list[str], prefixes: list[str]) -> list[str]:
    normalized = [_normalize_rel_prefix(prefix) for prefix in prefixes if prefix.strip()]
    violations: list[str] = []
    for path in paths:
        rel = _normalize_rel_prefix(path)
        if any(_matches_prefix(rel, prefix) for prefix in normalized):
            violations.append(path)
    return violations


def _scope_violation_message(
    *, violations: list[str], allowed_scope: list[str], blocked: bool
) -> str:
    violations_preview = ", ".join(violations[:20])
    if len(violations) > 20:
        violations_preview += ", ..."
    message = (
        f"Out-of-scope file changes detected ({len(violations)}): {violations_preview}. "
        f"Allowed scope: {allowed_scope or ['(none)']}."
    )
    if blocked:
        message += " Task was blocked due to strict scope isolation."
    return message


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


def _task_is_diagnostic_analysis_only(task: dict[str, Any]) -> bool:
    raw_flag = task.get("analysis_only")
    if isinstance(raw_flag, bool):
        return raw_flag
    title = str(task.get("title") or "").strip()
    if not title or not _DIAGNOSTIC_TASK_TITLE_RE.search(title):
        return False
    return _MUTATING_TASK_TEXT_RE.search(title) is None


def _task_allows_conditional_noop(task: dict[str, Any]) -> bool:
    acceptance = task.get("acceptance_criteria") or []
    if not isinstance(acceptance, list):
        acceptance = []
    text = "\n".join(
        [
            str(task.get("title") or ""),
            str(task.get("description") or ""),
            *(str(item or "") for item in acceptance),
        ]
    )
    return bool(_CONDITIONAL_NOOP_TASK_RE.search(text))


def _verification_failure_fingerprints(result: VerifyRunResult) -> set[str]:
    fingerprints: set[str] = set()
    for command_result in result.command_results:
        if command_result.exit_code == 0:
            continue
        command = command_result.command or command_result.effective_command or "<unknown>"
        output = command_result.output or "\n".join(
            part for part in (command_result.stdout, command_result.stderr) if part
        )
        matched = False
        for match in _VERIFY_FAILED_TEST_RE.finditer(output):
            fingerprints.add(f"{command}::{match.group(1).strip()}")
            matched = True
        for match in _VERIFY_FAILURE_SECTION_RE.finditer(output):
            label = " ".join(match.group(1).strip().split())
            if label and label.lower() not in {"failures", "errors"}:
                fingerprints.add(f"{command}::{label}")
                matched = True
        if matched:
            continue
        fallback_match = _VERIFY_ACTIONABLE_FAILURE_RE.search(output)
        if fallback_match is not None:
            line = " ".join(fallback_match.group(0).strip().split())
            if line:
                fingerprints.add(f"{command}::{line[:200]}")
                continue
        fingerprints.add(f"{command}::exit:{command_result.exit_code}")
    return fingerprints


def _failure_fingerprints_by_command(fingerprints: set[str]) -> dict[str, set[str]]:
    by_command: dict[str, set[str]] = {}
    for fingerprint in fingerprints:
        command, _, _detail = fingerprint.partition("::")
        if not command:
            continue
        by_command.setdefault(command, set()).add(fingerprint)
    return by_command


def _baseline_failure_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in _BASELINE_FAILURE_TOKEN_RE.finditer(text):
        token = match.group(0).lower().strip("_")
        if not token or token in _BASELINE_FAILURE_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _task_focus_tokens(task: dict[str, Any] | None) -> set[str]:
    if not isinstance(task, dict):
        return set()
    acceptance = task.get("acceptance_criteria") or []
    if not isinstance(acceptance, list):
        acceptance = []
    text = "\n".join(
        [
            str(task.get("title") or ""),
            str(task.get("description") or ""),
            *(str(item or "") for item in acceptance),
        ]
    )
    return _baseline_failure_tokens(text)


def _residual_failure_is_unrelated_to_task(fingerprint: str, task_tokens: set[str]) -> bool:
    if not task_tokens:
        return False
    _command, _separator, detail = fingerprint.partition("::")
    failure_tokens = _baseline_failure_tokens(detail or fingerprint)
    if not failure_tokens:
        return False
    return failure_tokens.isdisjoint(task_tokens)


def _baseline_improved_failure_comparison(
    *,
    baseline: VerifyRunResult | None,
    current: VerifyRunResult,
    task: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if baseline is None or baseline.all_passed or current.all_passed:
        return None
    if (
        failure_category_value(current.failure_category_value)
        == FailureCategory.INFRA_UNAVAILABLE.value
    ):
        return None

    baseline_failures = _verification_failure_fingerprints(baseline)
    current_failures = _verification_failure_fingerprints(current)
    if not baseline_failures or not current_failures:
        return None
    if not current_failures <= baseline_failures:
        return None
    baseline_by_command = _failure_fingerprints_by_command(baseline_failures)
    current_by_command = _failure_fingerprints_by_command(current_failures)
    task_tokens = _task_focus_tokens(task)
    unchanged_residual_failures: set[str] = set()
    for command, command_failures in current_by_command.items():
        baseline_command_failures = baseline_by_command.get(command, set())
        if command_failures < baseline_command_failures:
            continue
        unchanged_residual_failures.update(command_failures)
    task_related_unchanged = [
        failure
        for failure in unchanged_residual_failures
        if not _residual_failure_is_unrelated_to_task(failure, task_tokens)
    ]
    if task_related_unchanged:
        return None
    resolved_failures = sorted(baseline_failures - current_failures)
    if not resolved_failures and not unchanged_residual_failures:
        return None
    reason = (
        "post-task verification failures are unchanged from baseline but unrelated to "
        "the task focus"
        if not resolved_failures
        else "post-task verification failures are a subset of baseline failures"
    )
    return {
        "accepted": True,
        "reason": reason,
        "baseline_failure_count": len(baseline_failures),
        "current_failure_count": len(current_failures),
        "resolved_failure_count": len(resolved_failures),
        "current_failures": sorted(current_failures),
        "resolved_failures": resolved_failures,
        "unchanged_unrelated_failures": sorted(unchanged_residual_failures),
    }


def _extract_git_archive_to_directory(
    *,
    repo: Path,
    commit: str,
    target_dir: Path,
) -> bool:
    try:
        cp = subprocess.run(
            ["git", "-C", os.fspath(repo), "archive", "--format=tar", commit],
            check=False,
            capture_output=True,
        )
    except OSError:
        return False
    if cp.returncode != 0:
        return False

    target_root = target_dir.resolve()
    with tarfile.open(fileobj=io.BytesIO(cp.stdout), mode="r:") as archive:
        members = archive.getmembers()
        for member in members:
            destination = (target_root / member.name).resolve()
            if not destination.is_relative_to(target_root):
                return False
        archive.extractall(target_root, members=members)
    return True


def _run_baseline_verification_snapshot(
    *,
    repo: Path,
    commit: str | None,
    commands: list[str],
    artifact_path: Path,
    cfg: AppConfig,
) -> VerifyRunResult | None:
    if not commit or not commands:
        return None
    try:
        with tempfile.TemporaryDirectory(prefix="sylliptor-verify-baseline-") as temp_root:
            baseline_root = Path(temp_root) / "repo"
            baseline_root.mkdir(parents=True, exist_ok=True)
            if not _extract_git_archive_to_directory(
                repo=repo,
                commit=commit,
                target_dir=baseline_root,
            ):
                return None
            return run_task_verification(
                root=baseline_root,
                commands=commands,
                artifact_path=artifact_path,
                cfg=cfg,
            )
    except (OSError, tarfile.TarError):
        return None


def _append_patch_debug_section(patch_path: Path, *, title: str, patch_text: str) -> None:
    existing = patch_path.read_text(encoding="utf-8") if patch_path.exists() else ""
    parts: list[str] = []
    if existing:
        parts.append(existing if existing.endswith("\n") else existing + "\n")
    parts.append(f"# {title}\n")
    if patch_text:
        parts.append(patch_text if patch_text.endswith("\n") else patch_text + "\n")
    patch_path.write_text("\n".join(part.rstrip("\n") for part in parts if part), encoding="utf-8")


def _format_agent_exception_summary(exc: Exception) -> str:
    name = exc.__class__.__name__
    message = str(exc).strip()
    return sanitize_error_summary(f"{name}: {message}" if message else name)


def _empty_task_asset_mirror(workspace_path: Path, *, task_id: str = "") -> TaskAssetMirror:
    workspace = workspace_path.resolve()
    return TaskAssetMirror(
        workspace_path=workspace,
        manifest_path=workspace / ".sylliptor" / "task_assets" / "manifest.json",
        primary=[],
        may_need=[],
        pinned=[],
        task_id=task_id,
    )


def _combined_task_image_paths(
    *,
    legacy_paths: list[str],
    mirror: TaskAssetMirror,
    allocation: TaskAssetAllocation | None,
    cfg: AppConfig,
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


def run_task_worker(
    *,
    task: dict[str, Any],
    plan: dict[str, Any],
    worktree_repo_path: Path,
    base_branch: str,
    run_paths: RunPaths,
    cfg: AppConfig,
    mode: str,
    yes: bool,
    max_steps: int | None,
    api_key_override: str | None,
    no_log: bool = False,
    console: Console | None = None,
    scope_mode: str = "strict",
    verify_mode: str = "warn",
    verify_commands: list[str] | None = None,
    verify_command_selection: ResolvedVerifyCommands | None = None,
    trace_sink: SwarmTraceSink | None = None,
    trace_level: str = "off",
) -> TaskWorkerResult:
    ensure_execution_dirs(run_paths)
    (run_paths.execution_dir / "worker_results").mkdir(parents=True, exist_ok=True)

    task_id = str(task.get("id") or "").strip()
    task_title = str(task.get("title") or "").strip()
    task_branch = str(task.get("branch") or "").strip()
    safe_task = safe_task_file_component(task_id)
    plan_requirements = (
        [str(item).strip() for item in (plan.get("requirements") or []) if str(item).strip()]
        if isinstance(plan, dict)
        else None
    )
    verify_contract = resolve_worker_verify_contract(
        cfg=cfg,
        verify_mode=verify_mode,
        verify_commands=verify_commands,
        verify_command_selection=verify_command_selection,
        task=task,
        root=worktree_repo_path,
        plan_requirements=plan_requirements,
    )
    verify_resolution = verify_contract.selection
    verify_cmds = list(verify_contract.commands)
    verify_command_source = verify_resolution.source if verify_resolution is not None else None
    run_cfg = verify_contract.cfg
    run_cfg.model = resolve_model_for_role(
        cfg=cfg,
        role=ROLE_CODING,
        plan=plan,
        prefer_context="forge",
    )
    worker_trace_sink = trace_sink or NoopSwarmTraceSink()
    worker_surface = SwarmWorkerTraceSurface(
        run_id=run_paths.run_id,
        task_id=task_id,
        trace_sink=worker_trace_sink,
        trace_level=trace_level,
    )
    if verify_mode != "off":
        resolved_verify_source = verify_command_source or "unknown"
        resolved_verify_commands = ", ".join(verify_cmds) if verify_cmds else "(none)"
        worker_trace_sink.emit(
            build_swarm_trace_event(
                run_id=run_paths.run_id,
                task_id=task_id,
                phase="verify.lifecycle",
                message=(
                    "Resolved authoritative verification commands "
                    f"from {resolved_verify_source}: {resolved_verify_commands}."
                ),
                verbosity="full",
            )
        )
    recording_surface = RecordingSurface(worker_surface)
    worker_console = console or make_console(
        file=io.StringIO(), force_terminal=False, no_color=True
    )

    started_at = now_iso()
    patch_path = run_paths.execution_patches_dir / f"{safe_task}.diff"
    verify_path = run_paths.execution_verify_dir / f"{safe_task}.txt"
    baseline_verify_path = run_paths.execution_verify_dir / f"{safe_task}.baseline.txt"
    allowed_scope = normalize_scope_patterns(task, root=worktree_repo_path)
    diagnostic_analysis_only = _task_is_diagnostic_analysis_only(task)
    conditional_noop_allowed = _task_allows_conditional_noop(task)
    prompt_verification_enabled = (
        verify_mode != "off" and bool(verify_cmds) and not diagnostic_analysis_only
    )
    mirror_plan_into_worktree(run_paths=run_paths, worktree_repo_path=worktree_repo_path)
    prepared_knowledge = prepare_task_execution_knowledge(
        run_paths=run_paths,
        task=task,
        selection_label="execution",
    )
    mirror_selected_knowledge_into_worktree(
        materialized=prepared_knowledge,
        run_paths=run_paths,
        worktree_repo_path=worktree_repo_path,
    )
    asset_setup_warnings: list[str] = []
    asset_setup_error: str | None = None
    asset_usage_logger = AssetUsageLogger(run_paths=run_paths, task_id=task_id)
    asset_model_registry = ModelRegistry(cfg=run_cfg)
    asset_surface = (
        build_asset_surface(
            cfg=run_cfg,
            run_paths=run_paths,
            model_registry=asset_model_registry,
        )
        if run_cfg.assets.enabled
        else None
    )
    task_asset_mirror = _empty_task_asset_mirror(worktree_repo_path, task_id=task_id)
    if asset_surface is not None:
        try:
            task_asset_mirror = mirror_task_assets(
                task=task,
                plan=plan,
                surface=asset_surface,
                workspace_path=worktree_repo_path,
            )
        except AssetError as exc:
            if run_cfg.assets.worker.fail_on_mirror_error:
                asset_setup_error = (
                    f"worker asset mirror failed: {sanitize_error_summary(str(exc))}"
                )
            else:
                warning = (
                    f"worker asset mirror skipped: {sanitize_optional_error_summary(str(exc))}"
                )
                asset_setup_warnings.append(warning)
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
    instruction_bundle = build_task_execution_instruction_bundle(
        plan=plan,
        task=task,
        root=worktree_repo_path,
        cfg=run_cfg,
        role_model=run_cfg.model,
        mode=mode,
        yes=yes,
        deny_write_prefixes=[".sylliptor"],
        allow_write_globs=allowed_scope if scope_mode == "strict" else None,
        non_interactive=True,
        verification_enabled=prompt_verification_enabled,
        authoritative_verification_commands=(verify_cmds if prompt_verification_enabled else None),
        api_key=api_key_override,
        subagents_enabled=False,
        leading_sections=[prepared_knowledge.prompt_section],
    )
    asset_allocation: TaskAssetAllocation | None = None
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
    elif asset_surface is not None and (task_asset_mirror.may_need or task_asset_mirror.pinned):
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
        instruction_bundle = build_task_execution_instruction_bundle(
            plan=plan,
            task=task,
            root=worktree_repo_path,
            cfg=run_cfg,
            role_model=run_cfg.model,
            mode=mode,
            yes=yes,
            deny_write_prefixes=[".sylliptor"],
            allow_write_globs=allowed_scope if scope_mode == "strict" else None,
            non_interactive=True,
            verification_enabled=prompt_verification_enabled,
            authoritative_verification_commands=(
                verify_cmds if prompt_verification_enabled else None
            ),
            api_key=api_key_override,
            subagents_enabled=False,
            leading_sections=[prepared_knowledge.prompt_section],
            relevant_assets_section=relevant_assets_section,
        )
    instruction = instruction_bundle.instruction
    write_execution_context_artifact(
        run_paths=run_paths,
        task_id=task_id,
        context_text=instruction_bundle.artifact_text,
    )
    task_image_paths = _combined_task_image_paths(
        legacy_paths=list(instruction_bundle.image_paths),
        mirror=task_asset_mirror,
        allocation=asset_allocation,
        cfg=run_cfg,
        role_model=run_cfg.model,
        model_registry=asset_model_registry,
        usage_logger=asset_usage_logger,
    )
    task_asset_mcp_manager: Any | None = None
    task_attempts_raw = task.get("attempts")
    try:
        task_attempt_count = max(1, int(task_attempts_raw if task_attempts_raw is not None else 1))
    except (TypeError, ValueError):
        task_attempt_count = 1
    task_step_budget = resolve_managed_task_step_budget(
        cfg=run_cfg,
        plan=plan,
        task=task,
        kind="managed_task",
        mode=mode,
        verification_enabled=verify_mode != "off",
        max_steps_override=max_steps,
        attempt_count=task_attempt_count,
        image_count=len(task_image_paths or []),
    )
    budget_artifact_payload = instruction_bundle.to_budget_artifact_payload()
    budget_artifact_payload["step_budget"] = task_step_budget.to_payload()
    budget_artifact_path = write_execution_budget_artifact(
        run_paths=run_paths,
        task_id=task_id,
        payload=budget_artifact_payload,
    )
    protected_prefixes = [".sylliptor", ".sylliptor_images", "sylliptor-feedback"]
    head_before_run = head_commit(worktree_repo_path)
    before_runtime_snapshot = snapshot_runtime_tree(worktree_repo_path)

    run_code = 1
    run_err: str | None = asset_setup_error
    agent_run_exception = False
    agent_exception_summary: str | None = None
    failure_category: FailureCategory | str | None = None
    worktree_git_marker = worktree_repo_path / ".git"
    if worktree_git_marker.exists():
        try:
            ensure_runtime_artifact_excludes(worktree_repo_path)
        except GitOpsError as e:
            run_err = f"worker setup failed: {e}"
    try:
        if run_err is None:
            if asset_surface is not None:
                task_asset_mcp_manager = compose_worker_asset_mcp_manager(
                    base_manager=create_mcp_manager(
                        workspace_root=worktree_repo_path,
                        runtime_kind=RuntimeKind.SWARM_WORKER,
                        session_id=safe_task,
                    ),
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
                root=worktree_repo_path,
                instruction=instruction,
                mode=mode,
                runtime_kind=RuntimeKind.SWARM_WORKER,
                yes=yes,
                max_steps=task_step_budget.resolved_max_steps,
                no_log=no_log,
                api_key_override=api_key_override,
                console=worker_console,
                surface=recording_surface,
                image_paths=task_image_paths,
                deny_write_prefixes=[".sylliptor"],
                session_log_dir_override=run_paths.execution_sessions_dir,
                session_id_override=safe_task,
                allow_write_globs=allowed_scope if scope_mode == "strict" else None,
                non_interactive=True,
                usage_role=f"swarm_worker:{task_id}",
                enable_compaction=False,
                enable_tool_output_offload=True,
                enable_conversation_summarization=True,
                compaction_profile="execution",
                enable_chat_turn_step_budget=False,
                one_shot_execution=True,
                verification_enabled=prompt_verification_enabled,
                authoritative_verification_commands=(
                    verify_cmds if prompt_verification_enabled else None
                ),
                subagents_enabled=False,
                mcp_manager=task_asset_mcp_manager,
            )
            task_asset_mcp_manager = None
    except Exception as e:  # noqa: BLE001
        run_code = 1
        agent_run_exception = True
        agent_exception_summary = _format_agent_exception_summary(e)
        if is_provider_throttling_error(e):
            failure_category = FailureCategory.PROVIDER_THROTTLED
        elif is_provider_unavailable_error(e):
            failure_category = FailureCategory.PROVIDER_UNAVAILABLE
    finally:
        if task_asset_mcp_manager is not None:
            task_asset_mcp_manager.close()

    after_runtime_snapshot = snapshot_runtime_tree(worktree_repo_path)
    runtime_artifacts_changed = before_runtime_snapshot != after_runtime_snapshot
    if asset_allocation is not None:
        write_task_asset_allocation(
            run_paths=run_paths,
            allocation=asset_allocation,
            started_at=started_at,
        )
    asset_usage_logger.summary(
        primary_count=len(task_asset_mirror.primary),
        may_need_count=len(task_asset_mirror.may_need),
        pinned_count=len(task_asset_mirror.pinned),
    )
    head_after_run = head_commit(worktree_repo_path)
    scope_inspection_error: str | None = None
    try:
        reporting_diff = build_execution_reporting_diff_with_commit_range(
            worktree_repo_path,
            before_commit=head_before_run,
            after_commit=head_after_run,
        )
    except GitOpsError as e:
        reporting_diff = build_execution_reporting_diff_with_commit_range(
            worktree_repo_path,
            before_commit=None,
            after_commit=None,
        )
        scope_inspection_error = f"scope inspection failed: {e}"
    patch_path.write_text(reporting_diff.patch_text, encoding="utf-8")
    changed_files = _merge_changed_files(
        list(reporting_diff.changed_files),
        list_changed_files_including_untracked(worktree_repo_path),
    )
    agent_added_non_material_paths: list[str] = []
    if head_before_run and head_after_run and head_after_run != head_before_run:
        agent_added_non_material_paths = [
            path
            for path in added_files_since(
                worktree_repo_path,
                before_commit=head_before_run,
                after_commit=head_after_run,
            )
            if is_non_material_untracked_path(path)
        ]
        if agent_added_non_material_paths:
            changed_files = [
                path for path in changed_files if path not in agent_added_non_material_paths
            ]
    warnings: list[str] = list(asset_setup_warnings)

    commit_hash: str | None = None
    verify_failed = False
    verify_summary: str | None = None
    verify_payload: dict[str, Any] | None = None
    failure_reason: str | None = None
    scope_violation_files: list[str] = []
    result_kind: str | None = None
    noop_reason: str | None = None
    material_changes_detected = bool(changed_files)
    nonzero_agent_exit = run_code != 0 and run_err is None and not agent_run_exception
    strict_scope_blocked = False
    success = False

    def _append_run_error(message: str) -> None:
        nonlocal run_err
        run_err = (run_err + "; " if run_err else "") + message

    if nonzero_agent_exit:
        failure_reason = failure_reason or "agent_nonzero_exit"
        failure_category = failure_category or FailureCategory.IMPLEMENTATION_FAILED
        _append_run_error(
            f"agent exited non-zero ({run_code}); refusing to accept partial worker result"
        )
    elif agent_run_exception:
        failure_reason = failure_reason or "agent_exception"
        failure_category = failure_category or FailureCategory.IMPLEMENTATION_FAILED

    def _refresh_authoritative_verification_after_material_changes() -> None:
        nonlocal run_cfg
        nonlocal verify_cmds
        nonlocal verify_command_source
        nonlocal verify_resolution

        if verify_mode == "off" or verify_cmds or diagnostic_analysis_only:
            return
        try:
            post_change_scan = scan_workspace(context=resolve_workspace_context(worktree_repo_path))
        except (WorkspaceContextError, OSError):
            return
        refreshed_contract = resolve_worker_verify_contract(
            cfg=cfg,
            verify_mode=verify_mode,
            verify_commands=None,
            verify_command_selection=None,
            task=task,
            root=worktree_repo_path,
            repo_scan=post_change_scan,
            plan_requirements=plan_requirements,
        )
        refreshed_cmds = list(refreshed_contract.commands)
        if not refreshed_cmds:
            return
        verify_resolution = refreshed_contract.selection
        verify_cmds = refreshed_cmds
        verify_command_source = verify_resolution.source if verify_resolution is not None else None
        run_cfg.verify_commands = list(refreshed_cmds)
        resolved_verify_source = verify_command_source or "unknown"
        worker_trace_sink.emit(
            build_swarm_trace_event(
                run_id=run_paths.run_id,
                task_id=task_id,
                phase="verify.lifecycle",
                message=(
                    "Refreshed authoritative verification commands after material changes "
                    f"from {resolved_verify_source}: {', '.join(refreshed_cmds)}."
                ),
                verbosity="full",
            )
        )

    _refresh_authoritative_verification_after_material_changes()

    def _run_authoritative_verification(
        *,
        fail_on_failed_verify: bool,
        fail_on_repo_mutation: bool,
        failed_verify_reason: str = "verification_failed",
        failed_verify_error_prefix: str = "verification failed",
        infra_unavailable_error_prefix: str = "verification infrastructure unavailable",
        mark_verify_failed: bool = True,
        allow_baseline_improved_failure: bool = False,
    ) -> bool:
        nonlocal success
        nonlocal commit_hash
        nonlocal verify_failed
        nonlocal verify_summary
        nonlocal verify_payload
        nonlocal failure_reason
        nonlocal failure_category
        nonlocal changed_files
        nonlocal scope_violation_files

        before_verify_snapshot = snapshot_workspace_tree(worktree_repo_path)
        verify_result = run_task_verification(
            root=worktree_repo_path,
            commands=verify_cmds,
            artifact_path=verify_path,
            cfg=run_cfg,
        )
        verify_summary = verify_result.summary
        verify_payload = verify_run_result_to_payload(
            root=run_paths.root,
            result=verify_result,
        )
        after_verify_snapshot = snapshot_workspace_tree(worktree_repo_path)
        verify_mutation_diff = build_workspace_snapshot_reporting_diff(
            worktree_repo_path,
            before_snapshot=before_verify_snapshot,
            after_snapshot=after_verify_snapshot,
        )
        verify_mutation_paths = list(verify_mutation_diff.changed_files)
        verification_ok = True

        if verify_mutation_paths:
            verify_mutation_msg = "Verification commands modified repository state"
            if commit_hash is not None:
                verify_mutation_msg += (
                    f" after the task commit: {', '.join(verify_mutation_paths[:20])}"
                )
            else:
                verify_mutation_msg += f": {', '.join(verify_mutation_paths[:20])}"
            if len(verify_mutation_paths) > 20:
                verify_mutation_msg += ", ..."
            verify_mutation_msg += "."
            if scope_mode in {"warn", "strict"}:
                scope_ok, verify_scope_violations = check_scope(
                    verify_mutation_paths,
                    allowed_scope,
                    root=worktree_repo_path,
                )
                if not scope_ok:
                    scope_violation_files = _merge_changed_files(
                        scope_violation_files,
                        verify_scope_violations,
                    )
                    scope_msg = _scope_violation_message(
                        violations=verify_scope_violations,
                        allowed_scope=allowed_scope,
                        blocked=scope_mode == "strict" or fail_on_repo_mutation,
                    )
                    if commit_hash is not None:
                        scope_msg += " Verification commands modified repository state after the task commit."
                    else:
                        scope_msg += " Verification commands modified repository state during already-satisfied verification."
                    if scope_mode == "strict" or fail_on_repo_mutation:
                        success = False
                        verification_ok = False
                        failure_reason = "scope_violation"
                        failure_category = failure_category or FailureCategory.IMPLEMENTATION_FAILED
                        commit_hash = None
                        _append_run_error(scope_msg)
                        changed_files = _merge_changed_files(changed_files, verify_mutation_paths)
                        _append_patch_debug_section(
                            patch_path,
                            title="Post-verification workspace diff",
                            patch_text=verify_mutation_diff.patch_text,
                        )
                    else:
                        warnings.append(scope_msg)
                elif scope_mode == "strict" or fail_on_repo_mutation:
                    success = False
                    verification_ok = False
                    failure_category = failure_category or FailureCategory.IMPLEMENTATION_FAILED
                    commit_hash = None
                    _append_run_error(verify_mutation_msg)
                    changed_files = _merge_changed_files(changed_files, verify_mutation_paths)
                    _append_patch_debug_section(
                        patch_path,
                        title="Post-verification workspace diff",
                        patch_text=verify_mutation_diff.patch_text,
                    )
                else:
                    warnings.append(verify_mutation_msg)
            elif fail_on_repo_mutation:
                success = False
                verification_ok = False
                failure_category = failure_category or FailureCategory.IMPLEMENTATION_FAILED
                commit_hash = None
                _append_run_error(verify_mutation_msg)
                changed_files = _merge_changed_files(changed_files, verify_mutation_paths)
                _append_patch_debug_section(
                    patch_path,
                    title="Post-verification workspace diff",
                    patch_text=verify_mutation_diff.patch_text,
                )
            else:
                warnings.append(verify_mutation_msg)

        if not verify_result.all_passed:
            verify_failure_category = (
                verify_result.failure_category_value or FailureCategory.VERIFICATION_FAILED
            )
            verify_infra_unavailable = (
                failure_category_value(verify_failure_category)
                == FailureCategory.INFRA_UNAVAILABLE.value
            )
            baseline_comparison: dict[str, Any] | None = None
            if (
                allow_baseline_improved_failure
                and verify_mode == "warn"
                and fail_on_failed_verify
                and not verify_infra_unavailable
            ):
                baseline_result = _run_baseline_verification_snapshot(
                    repo=worktree_repo_path,
                    commit=head_before_run,
                    commands=verify_cmds,
                    artifact_path=baseline_verify_path,
                    cfg=run_cfg,
                )
                baseline_comparison = _baseline_improved_failure_comparison(
                    baseline=baseline_result,
                    current=verify_result,
                    task=task,
                )
            if baseline_comparison is not None:
                if verify_payload is not None:
                    verify_payload["baseline_comparison"] = baseline_comparison
                    verify_payload["baseline_artifact_path"] = str(
                        baseline_verify_path.relative_to(run_paths.root)
                        if baseline_verify_path.is_relative_to(run_paths.root)
                        else baseline_verify_path
                    )
                if baseline_comparison["resolved_failure_count"]:
                    warnings.append(
                        "Verification warning: post-task verification still fails, but it "
                        "reduced pre-existing baseline failures "
                        f"({baseline_comparison['baseline_failure_count']} -> "
                        f"{baseline_comparison['current_failure_count']})."
                    )
                else:
                    warnings.append(
                        "Verification warning: post-task verification still fails, but the "
                        "remaining failures match the pre-task baseline and are unrelated to "
                        "this task's focus."
                    )
                return verification_ok
            if verify_mode == "strict":
                verify_failed = True
                failure_reason = failure_reason or (
                    "verification_infra_unavailable"
                    if verify_infra_unavailable
                    else "verification_failed"
                )
                failure_category = failure_category or verify_failure_category
                success = False
                verification_ok = False
                if verify_infra_unavailable:
                    _append_run_error(
                        f"strict verification infrastructure unavailable: {verify_result.summary}"
                    )
                else:
                    _append_run_error(f"strict verification failed: {verify_result.summary}")
            elif fail_on_failed_verify:
                if mark_verify_failed:
                    verify_failed = True
                failure_reason = failure_reason or (
                    "verification_infra_unavailable"
                    if verify_infra_unavailable
                    else failed_verify_reason
                )
                failure_category = failure_category or verify_failure_category
                success = False
                verification_ok = False
                if verify_infra_unavailable:
                    _append_run_error(f"{infra_unavailable_error_prefix}: {verify_result.summary}")
                else:
                    _append_run_error(f"{failed_verify_error_prefix}: {verify_result.summary}")
            else:
                warnings.append(f"Verification warning: {verify_result.summary}")

        return verification_ok and verify_result.all_passed

    if scope_inspection_error:
        if scope_mode == "strict":
            strict_scope_blocked = True
            failure_reason = "scope_violation"
            failure_category = failure_category or FailureCategory.IMPLEMENTATION_FAILED
            _append_run_error(scope_inspection_error)
        else:
            warnings.append(scope_inspection_error)
    if scope_mode in {"warn", "strict"}:
        scope_ok, violations = check_scope(changed_files, allowed_scope, root=worktree_repo_path)
        if not scope_ok:
            scope_violation_files = list(violations)
            scope_msg = _scope_violation_message(
                violations=violations,
                allowed_scope=allowed_scope,
                blocked=scope_mode == "strict",
            )
            if scope_mode == "strict":
                strict_scope_blocked = True
                failure_reason = "scope_violation"
                failure_category = failure_category or FailureCategory.IMPLEMENTATION_FAILED
                _append_run_error(scope_msg)
            else:
                warnings.append(scope_msg)

    can_attempt_commit_verify = (
        not runtime_artifacts_changed
        and material_changes_detected
        and not strict_scope_blocked
        and run_code == 0
    )

    if can_attempt_commit_verify:
        try:
            agent_commit_hash: str | None = None
            if head_before_run and head_after_run and head_after_run != head_before_run:
                agent_commit_hash = head_after_run
                agent_commit_paths = changed_files_since(
                    worktree_repo_path,
                    before_commit=head_before_run,
                    after_commit=head_after_run,
                )
                protected_violations = _protected_path_violations(
                    agent_commit_paths,
                    protected_prefixes,
                )
                if protected_violations:
                    preview = ", ".join(protected_violations[:20])
                    if len(protected_violations) > 20:
                        preview += ", ..."
                    raise GitOpsError(f"worker created commit touching protected paths: {preview}")
                agent_added_non_material_paths = [
                    path
                    for path in added_files_since(
                        worktree_repo_path,
                        before_commit=head_before_run,
                        after_commit=head_after_run,
                    )
                    if is_non_material_untracked_path(path)
                ]
                if agent_added_non_material_paths:
                    reset_mixed(worktree_repo_path, target=head_before_run)
                    agent_commit_hash = None
            non_material_untracked_paths = list_untracked_packaging_metadata_paths(
                worktree_repo_path
            )
            stage_all(worktree_repo_path)
            unstage_staged_prefixes(worktree_repo_path, protected_prefixes)
            ensure_not_staged_prefixes(worktree_repo_path, protected_prefixes)
            if non_material_untracked_paths:
                unstage_staged_paths(worktree_repo_path, non_material_untracked_paths)
                ensure_not_staged_paths(worktree_repo_path, non_material_untracked_paths)
            staged_now = staged_files(worktree_repo_path)
            if staged_now and has_grounded_rust_target_runtime_artifacts(worktree_repo_path):
                unstage_staged_runtime_artifacts(
                    worktree_repo_path,
                    current_paths=staged_now,
                )
                staged_now = staged_files(worktree_repo_path)
                ensure_not_staged_runtime_artifacts(
                    worktree_repo_path,
                    current_paths=staged_now,
                )
            if staged_now:
                commit_hash = commit_all(
                    worktree_repo_path,
                    message=f"{task_id}: {task_title or 'task update'}",
                )
            elif agent_commit_hash is not None:
                commit_hash = agent_commit_hash
            else:
                raise GitOpsError("no repository changes were staged for commit")
            success = True
            patch_text = format_patch_stdout(worktree_repo_path, base_branch=base_branch)
            patch_path.write_text(
                patch_text if patch_text else "(empty format-patch output)\n",
                encoding="utf-8",
            )
            changed_files = changed_files_between(
                worktree_repo_path,
                revspec=f"{base_branch}..HEAD",
            )
            if verify_mode != "off" and verify_cmds:
                _run_authoritative_verification(
                    fail_on_failed_verify=True,
                    fail_on_repo_mutation=False,
                    failed_verify_reason="verification_failed",
                    failed_verify_error_prefix="verification failed",
                    infra_unavailable_error_prefix="verification infrastructure unavailable",
                    mark_verify_failed=True,
                    allow_baseline_improved_failure=True,
                )
            elif verify_mode == "strict":
                success = False
                failure_reason = failure_reason or "verification_unavailable"
                failure_category = failure_category or FailureCategory.IMPLEMENTATION_FAILED
                _append_run_error(
                    "strict verification requires authoritative commands, but none were available"
                )
                verify_summary = "verification skipped: no authoritative commands available"
            elif verify_mode != "off":
                verify_summary = "verification skipped: no authoritative commands available"
            else:
                verify_summary = "verification disabled (--verify off)"
            if success:
                result_kind = "success_commit"
        except GitOpsError as e:
            success = False
            failure_category = failure_category or FailureCategory.IMPLEMENTATION_FAILED
            _append_run_error(f"worker commit/patch failed: {e}")
    elif (
        not runtime_artifacts_changed
        and not material_changes_detected
        and not strict_scope_blocked
        and run_err is None
        and scope_inspection_error is None
        and run_code == 0
    ):
        if (
            diagnostic_analysis_only
            and not agent_run_exception
            and failure_category != FailureCategory.PROVIDER_THROTTLED
        ):
            success = True
            result_kind = "success_noop"
            noop_reason = "diagnostic_analysis_only"
            verify_summary = "verification skipped: diagnostic analysis-only task made no changes"
            if run_code != 0:
                warnings.append(
                    "Diagnostic analysis-only task exited non-zero after making no repository "
                    "changes; accepted as a non-blocking diagnostic no-op."
                )
        elif (
            conditional_noop_allowed
            and not agent_run_exception
            and failure_category != FailureCategory.PROVIDER_THROTTLED
        ):
            success = True
            result_kind = "success_noop"
            noop_reason = "conditional_noop"
            verify_summary = "verification skipped: conditional task made no changes"
            if run_code != 0:
                warnings.append(
                    "Conditional task exited non-zero after making no repository changes; "
                    "accepted because the task text allowed no work when not applicable."
                )
        elif verify_mode == "off":
            success = False
            failure_reason = failure_reason or "verification_required_for_noop"
            failure_category = failure_category or FailureCategory.IMPLEMENTATION_FAILED
            _append_run_error(
                "zero-diff worker outcomes require passing authoritative verification; "
                "verification was disabled"
            )
            verify_summary = "verification disabled (--verify off)"
        elif not verify_cmds:
            success = False
            failure_reason = failure_reason or "verification_required_for_noop"
            failure_category = failure_category or FailureCategory.IMPLEMENTATION_FAILED
            _append_run_error(
                "zero-diff worker outcomes require passing authoritative verification; "
                "no verification commands were available"
            )
            verify_summary = "verification skipped: no commands"
        else:
            success = True
            if _run_authoritative_verification(
                fail_on_failed_verify=True,
                fail_on_repo_mutation=True,
                failed_verify_reason="noop_verification_failed",
                failed_verify_error_prefix="already-satisfied verification failed",
                infra_unavailable_error_prefix=(
                    "already-satisfied verification infrastructure unavailable"
                ),
                mark_verify_failed=False,
            ):
                result_kind = "success_noop"
                noop_reason = "already_satisfied"
            else:
                success = False

    salvaged_nonzero_exit = False
    salvaged_agent_exception = False
    if not success and agent_exception_summary:
        agent_exception_error = f"agent raised: {agent_exception_summary}"
        run_err = f"{agent_exception_error}; {run_err}" if run_err else agent_exception_error
    if not success and failure_category is None:
        # TODO(failure-category-audit): default=implementation_failed,
        # site=swarm_worker generic failure result, see failure_category.py
        failure_category = FailureCategory.IMPLEMENTATION_FAILED

    if runtime_artifacts_changed:
        summary = "Worker failed: attempted protected .sylliptor modifications."
    elif failure_reason == "scope_violation":
        summary = "Worker blocked due to strict scope isolation."
    elif success and result_kind == "success_noop" and noop_reason == "diagnostic_analysis_only":
        summary = "Worker completed diagnostic analysis with no repository changes."
    elif success and result_kind == "success_noop" and noop_reason == "conditional_noop":
        summary = "Worker completed conditional task with no repository changes."
    elif success and result_kind == "success_noop":
        summary = (
            "Worker completed successfully with no repository changes; "
            "authoritative verification passed and the task was already satisfied."
        )
    elif success:
        summary = "Worker completed successfully and produced a task commit."
    else:
        summary = "Worker failed."
    if run_err:
        summary += f" Error: {run_err}"
    if warnings:
        summary += " Warnings: " + " | ".join(warnings)

    finished_at = now_iso()
    exec_artifacts = write_exec_log_artifacts(
        paths=run_paths,
        task_id=task_id,
        cfg=run_cfg,
        no_log=no_log,
        before_logs=set(),
        sessions_dir=run_paths.execution_sessions_dir,
        expected_session_id=safe_task,
    )
    report_path = write_task_report(
        paths=run_paths,
        task=task,
        result="success" if success else "failure",
        result_kind=result_kind or ("success_commit" if success else "failure"),
        summary=summary,
        started_at=started_at,
        finished_at=finished_at,
        changed_files=changed_files,
        verify_commands=verify_cmds,
        patch_path=patch_path,
        budget_artifact_path=budget_artifact_path,
        execution_log_artifacts=exec_artifacts,
        verify_artifact_path=verify_path if verify_path.exists() else None,
        verify_summary=verify_summary,
        verify_payload=verify_payload,
        verify_command_source=verify_command_source,
        base_branch=base_branch,
        task_branch=task_branch,
        commit_hash=commit_hash,
        merge_commit_hash=None,
        merge_result=(
            "no merge required (analysis-only no-op)"
            if result_kind == "success_noop" and noop_reason == "diagnostic_analysis_only"
            else (
                "no merge required (conditional no-op)"
                if result_kind == "success_noop" and noop_reason == "conditional_noop"
                else (
                    "no merge required (verified no-op)"
                    if result_kind == "success_noop"
                    else "not merged"
                )
            )
        ),
        salvaged_nonzero_exit=salvaged_nonzero_exit,
        salvaged_agent_exception=salvaged_agent_exception,
        agent_exception_summary=agent_exception_summary,
        noop_reason=noop_reason,
    )
    persisted_capture = persist_execution_knowledge_capture(
        paths=run_paths,
        task=task,
        source="swarm_worker",
        assistant_message=recording_surface.final_assistant_message,
        artifact_dir=(
            run_paths.execution_knowledge_capture_dir
            / safe_task
            / safe_task_file_component(started_at)
        ),
        report_path=report_path,
        patch_path=patch_path,
        verify_artifact_path=verify_path if verify_path.exists() else None,
        budget_artifact_path=budget_artifact_path,
        session_artifact_dir=exec_artifacts.session_artifact_dir,
    )
    if not success:
        mark_knowledge_capture_promotion_skipped(
            artifact_dir=persisted_capture.artifact_dir,
            reason="worker execution outcome was not accepted",
        )

    task_attempt_entry = write_task_attempt_entry(
        paths=run_paths,
        task=task,
        source="swarm_worker",
        result="success" if success else "failure",
        summary=summary,
        changed_files=changed_files,
        verify_summary=verify_summary,
        report_path=report_path,
        patch_path=patch_path,
        verify_artifact_path=verify_path if verify_path.exists() else None,
        budget_artifact_path=budget_artifact_path,
        session_artifact_dir=exec_artifacts.session_artifact_dir,
        acceptance_state="pending" if success else "rejected",
        extra_tags=(
            ["execution", "worker", result_kind] if result_kind else ["execution", "worker"]
        ),
    )

    result = TaskWorkerResult(
        task_id=task_id,
        title=task_title,
        branch=task_branch,
        worktree_path=os.fspath(worktree_repo_path),
        started_at=started_at,
        finished_at=finished_at,
        success=success,
        summary=summary,
        commit_hash=commit_hash,
        error=run_err,
        report_path=_repo_rel(run_paths.root, report_path),
        patch_path=_repo_rel(run_paths.root, patch_path),
        log_path=_repo_rel(run_paths.root, exec_artifacts.log_copy_path),
        log_pointer_path=_repo_rel(run_paths.root, exec_artifacts.pointer_path),
        warnings=warnings,
        changed_files=changed_files,
        verify_failed=verify_failed,
        verify_summary=verify_summary,
        verify_artifact_path=(
            _repo_rel(run_paths.root, verify_path) if verify_path.exists() else None
        ),
        knowledge_capture_artifact_dir=_repo_rel(run_paths.root, persisted_capture.artifact_dir),
        task_attempt_entry_id=task_attempt_entry.id,
        task_attempt_knowledge_file_path=(
            _repo_rel(run_paths.root, task_attempt_entry.file_path)
            if task_attempt_entry.file_path is not None
            else None
        ),
        verify_payload=verify_payload,
        verify_command_source=verify_command_source,
        failure_reason=failure_reason,
        failure_category=failure_category,
        scope_violation_files=(list(scope_violation_files) if scope_violation_files else None),
        allowed_scope=list(allowed_scope),
        agent_exit_code=run_code,
        salvaged_nonzero_exit=salvaged_nonzero_exit,
        salvaged_agent_exception=salvaged_agent_exception,
        agent_exception_summary=agent_exception_summary,
        result_kind=result_kind,
        noop_reason=noop_reason,
    )
    issue_paths = changed_files or list(allowed_scope)
    if runtime_artifacts_changed:
        write_issue_entry(
            paths=run_paths,
            task=task,
            source="swarm_worker",
            title=f"{task_id}: protected .sylliptor mutation attempt",
            summary="Worker execution attempted to modify protected .sylliptor runtime state.",
            paths_in_scope=issue_paths,
            report_path=report_path,
            patch_path=patch_path,
            verify_artifact_path=verify_path if verify_path.exists() else None,
            budget_artifact_path=budget_artifact_path,
            session_artifact_dir=exec_artifacts.session_artifact_dir,
            tags=["protected_runtime_mutation"],
        )
    elif failure_reason == "noop_verification_failed":
        write_issue_entry(
            paths=run_paths,
            task=task,
            source="swarm_worker",
            title=f"{task_id}: already-satisfied verification failed",
            summary=verify_summary or "Authoritative verification failed for zero-diff success.",
            paths_in_scope=issue_paths,
            report_path=report_path,
            patch_path=patch_path,
            verify_artifact_path=verify_path if verify_path.exists() else None,
            budget_artifact_path=budget_artifact_path,
            session_artifact_dir=exec_artifacts.session_artifact_dir,
            tags=["verification_failure", "noop_verification_failure"],
        )
    elif failure_reason == "verification_infra_unavailable":
        write_issue_entry(
            paths=run_paths,
            task=task,
            source="swarm_worker",
            title=f"{task_id}: verification infrastructure unavailable",
            summary=verify_summary or "Worker verification could not run due to infrastructure.",
            paths_in_scope=issue_paths,
            report_path=report_path,
            patch_path=patch_path,
            verify_artifact_path=verify_path if verify_path.exists() else None,
            budget_artifact_path=budget_artifact_path,
            session_artifact_dir=exec_artifacts.session_artifact_dir,
            tags=["verification_failure", "infra_unavailable"],
        )
    elif verify_failed:
        write_issue_entry(
            paths=run_paths,
            task=task,
            source="swarm_worker",
            title=f"{task_id}: strict verification failed",
            summary=verify_summary or "Worker verification failed.",
            paths_in_scope=issue_paths,
            report_path=report_path,
            patch_path=patch_path,
            verify_artifact_path=verify_path if verify_path.exists() else None,
            budget_artifact_path=budget_artifact_path,
            session_artifact_dir=exec_artifacts.session_artifact_dir,
            tags=["verification_failure"],
        )
    elif not success:
        write_issue_entry(
            paths=run_paths,
            task=task,
            source="swarm_worker",
            title=f"{task_id}: worker execution failed",
            summary=summary,
            paths_in_scope=issue_paths,
            report_path=report_path,
            patch_path=patch_path,
            verify_artifact_path=verify_path if verify_path.exists() else None,
            budget_artifact_path=budget_artifact_path,
            session_artifact_dir=exec_artifacts.session_artifact_dir,
            tags=(
                ["execution_failure", "scope_violation"]
                if failure_reason == "scope_violation"
                else ["execution_failure"]
            ),
        )
    rebuild_knowledge_index(run_paths)
    return result
