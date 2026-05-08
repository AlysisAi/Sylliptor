from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import PurePosixPath
from typing import Any, Literal

from .assets.models import AssetError
from .assets.plan_binding import task_asset_briefing
from .direction_change import filter_obsolete_direction_paths
from .failure_category import FailureCategory
from .mcp.forge_scope import normalize_task_mcp_scope
from .plan_reconciliation import is_code_implementation_path
from .swarm_scheduler import canonical_task_status
from .task_readiness import (
    EXECUTION_UNREADY_SCOPE_WARNING,
    normalize_existing_task_scope_fields,
    normalized_text_list,
    status_is_execution_candidate,
    task_is_missing_runnable_scope,
    task_readiness_warning,
    task_requires_runnable_file_scope,
)
from .task_scope import (
    extract_forbidden_repo_path_hints,
    extract_repo_path_hints,
    is_agent_internal_scope_path,
    split_normalized_repo_path_list,
)

_NON_EXECUTABLE_OBSOLETE_STATUSES = frozenset({"superseded", "invalidated"})
PlanAcceptanceRuleId = Literal["R1", "R2", "R3", "R4"]


class PlannerFailedError(RuntimeError):
    failure_category = FailureCategory.PLANNER_FAILED


@dataclass(frozen=True)
class PlanAcceptanceIssue:
    rule_id: PlanAcceptanceRuleId
    observed: str
    task_id: str | None = None
    detail: str = ""


def _task_ids(tasks: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        task_id = str(task.get("id") or "").strip()
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        ids.append(task_id)
    return ids


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _raw_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def _parse_only_ids(only: str | None) -> set[str] | None:
    if only is None:
        return None
    ids = {part.strip() for part in only.split(",") if part.strip()}
    return ids or None


def _task_id_for_issue(task: dict[str, Any], index: int) -> str:
    return str(task.get("id") or "").strip() or f"task[{index}]"


def _candidate_execution_tasks(
    plan: dict[str, Any],
    *,
    retry_failed: bool,
    retry_changes_requested: bool,
    retry_merge_conflicts: bool,
    only: str | None,
) -> list[tuple[int, dict[str, Any], str, str]]:
    only_ids = _parse_only_ids(only)
    tasks_raw = plan.get("tasks")
    if not isinstance(tasks_raw, list):
        return []

    candidates: list[tuple[int, dict[str, Any], str, str]] = []
    for index, task in enumerate(tasks_raw):
        if not isinstance(task, dict):
            continue
        task_id = _task_id_for_issue(task, index)
        status = canonical_task_status(str(task.get("status") or ""))
        if (
            only_ids is not None
            and task_id not in only_ids
            and status != "ready_for_merge"
            and not (retry_merge_conflicts and status == "merge_conflict")
        ):
            continue
        if not status_is_execution_candidate(
            status,
            retry_failed=retry_failed,
            retry_changes_requested=retry_changes_requested,
            retry_merge_conflicts=retry_merge_conflicts,
        ):
            continue
        candidates.append((index, task, task_id, status))
    return candidates


def _mutating_task_scope(task: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
    estimated_files, write_scope = normalize_existing_task_scope_fields(
        estimated_files=task.get("estimated_files"),
        write_scope=task.get("write_scope"),
    )
    requires_runnable_scope = task_requires_runnable_file_scope(
        title=str(task.get("title") or "").strip(),
        description=str(task.get("description") or "").strip(),
        acceptance_criteria=normalized_text_list(task.get("acceptance_criteria")),
        estimated_files=estimated_files,
        write_scope=write_scope,
    )
    return requires_runnable_scope, estimated_files, write_scope


def _missing_field_issue(
    *,
    task_id: str,
    field_name: str,
) -> PlanAcceptanceIssue:
    detail = EXECUTION_UNREADY_SCOPE_WARNING if field_name == "write_scope" else ""
    return PlanAcceptanceIssue(
        rule_id="R4",
        task_id=task_id,
        observed=f"missing field: {field_name}",
        detail=detail,
    )


def _required_task_field_errors(
    task: dict[str, Any],
    *,
    task_id: str,
) -> list[PlanAcceptanceIssue]:
    issues: list[PlanAcceptanceIssue] = []
    if not str(task.get("id") or "").strip():
        issues.append(_missing_field_issue(task_id=task_id, field_name="id"))
    if not str(task.get("title") or "").strip():
        issues.append(_missing_field_issue(task_id=task_id, field_name="title"))
    raw_write_scope = task.get("write_scope")
    if not isinstance(raw_write_scope, list) or not _raw_text_list(raw_write_scope):
        issues.append(_missing_field_issue(task_id=task_id, field_name="write_scope"))
    return issues


def _empty_write_scope_observed(raw_write_scope: list[str]) -> str:
    normalized, _dropped = split_normalized_repo_path_list(raw_write_scope)
    if normalized and all(is_agent_internal_scope_path(path) for path in normalized):
        prefixes = sorted({path.split("/", 1)[0] for path in normalized})
        if prefixes == [".sylliptor"]:
            return "all write_scope paths under .sylliptor/"
        if prefixes:
            return "all write_scope paths under agent-internal dirs: " + ", ".join(prefixes)
    return "write_scope has no runnable user-code paths after filtering"


def _metadata_only_observed(write_scope: list[str]) -> str:
    lowered = [path.casefold() for path in write_scope]
    if lowered and all(
        path in {"readme", "readme.md"} or path.startswith("docs/") or path.endswith(".md")
        for path in lowered
    ):
        return "write_scope is README/docs only"
    return "write_scope has no code implementation paths"


_PRIMARY_IMPLEMENTATION_FILENAMES = frozenset(
    {
        "cargo.toml",
        "go.mod",
        "go.sum",
        "mix.exs",
        "package-lock.json",
        "package.json",
        "pnpm-lock.yaml",
        "poetry.lock",
        "pom.xml",
        "pyproject.toml",
        "requirements-dev.txt",
        "requirements.txt",
        "setup.cfg",
        "setup.py",
        "tox.ini",
        "yarn.lock",
    }
)
_PRIMARY_IMPLEMENTATION_EXTENSIONS = frozenset(
    {
        ".cfg",
        ".conf",
        ".ini",
        ".toml",
        ".yaml",
        ".yml",
    }
)
_SUPPORT_SCOPE_TEXT_RE = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:changelog|coverage|docs?|doctest|documentation|examples?|jest|markdown|"
    r"pytest|readme|regression|specs?|tests?|unittest|verification|verify|vitest)"
    r"(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_PRIMARY_SUPPORT_SCOPE_TEXT_RE = re.compile(
    r"^\s*(?:(?:add|create|write|update)\s+)?"
    r"(?:(?:[\w./-]+)\s+){0,8}"
    r"(?:tests?|test\s+case|coverage)\b"
    r"|^\s*(?:verify|validate|run|check)\b"
    r"|^\s*(?:(?:add|create|write|update|sync)\s+)?"
    r"(?:readme|docs?|documentation|changelog|manual)\b"
    r"|^\s*document\b",
    re.IGNORECASE,
)
_IMPLEMENTATION_INTENT_TEXT_RE = re.compile(
    r"\b(?:build|change|configure|edit|enable|fix|implement|improve|modify|patch|"
    r"refactor|rename|repair|support|wire)\b",
    re.IGNORECASE,
)


def _is_primary_implementation_path(path: str) -> bool:
    if is_code_implementation_path(path):
        return True
    pure = PurePosixPath(path)
    name = pure.name.casefold()
    if name in _PRIMARY_IMPLEMENTATION_FILENAMES:
        return True
    if pure.suffix.casefold() in _PRIMARY_IMPLEMENTATION_EXTENSIONS:
        return True
    return False


def _task_text(task: dict[str, Any]) -> str:
    parts = [
        str(task.get("title") or ""),
        str(task.get("description") or ""),
        *normalized_text_list(task.get("acceptance_criteria")),
    ]
    return " ".join(part for part in parts if part).casefold()


def _task_is_explicit_support_scope(task: dict[str, Any]) -> bool:
    text = _task_text(task)
    if not text:
        return False
    title = str(task.get("title") or "").strip()
    if _PRIMARY_SUPPORT_SCOPE_TEXT_RE.search(title):
        return True
    intent_text = " ".join(
        part
        for part in (
            title,
            str(task.get("description") or "").strip(),
        )
        if part
    )
    if _IMPLEMENTATION_INTENT_TEXT_RE.search(intent_text):
        return False
    return _SUPPORT_SCOPE_TEXT_RE.search(text) is not None


def _path_identity_key(value: str) -> str:
    normalized = str(value or "").strip().replace("\\", "/").rstrip("/")
    if normalized.casefold() in {"readme", "readme.md"}:
        return "__readme_alias__"
    return normalized.casefold()


def _explicit_task_path_hints(task: dict[str, Any]) -> list[str]:
    text = "\n".join(
        [
            str(task.get("title") or ""),
            str(task.get("description") or ""),
            *normalized_text_list(task.get("acceptance_criteria")),
        ]
    )
    forbidden = {_path_identity_key(path) for path in extract_forbidden_repo_path_hints(text)}
    hints, _obsolete = filter_obsolete_direction_paths(
        extract_repo_path_hints(text),
        latest_user_text=text,
        task_text=text,
    )
    normalized, _dropped = split_normalized_repo_path_list(hints)
    out: list[str] = []
    for path in normalized:
        if is_agent_internal_scope_path(path):
            continue
        if _path_identity_key(path) in forbidden:
            continue
        out.append(path)
    return list(dict.fromkeys(out))


def _scope_entry_covers_path(scope_entry: str, path: str) -> bool:
    normalized_scope = str(scope_entry or "").strip().replace("\\", "/").rstrip("/")
    normalized_path = str(path or "").strip().replace("\\", "/").rstrip("/")
    if not normalized_scope or not normalized_path:
        return False
    if _path_identity_key(normalized_scope) == _path_identity_key(normalized_path):
        return True
    if normalized_scope == normalized_path:
        return True
    if any(char in normalized_scope for char in "*?["):
        return fnmatchcase(normalized_path, normalized_scope)
    if normalized_scope.endswith("/"):
        return normalized_path.startswith(normalized_scope)
    return False


def _missing_explicit_scope_hints(
    *,
    task: dict[str, Any],
    estimated_files: list[str],
    write_scope: list[str],
) -> list[str]:
    explicit_paths = _explicit_task_path_hints(task)
    if not explicit_paths:
        return []
    scope_entries = [*estimated_files, *write_scope]
    missing = [
        path
        for path in explicit_paths
        if not any(_scope_entry_covers_path(scope, path) for scope in scope_entries)
    ]
    return missing


def _task_needs_primary_implementation_scope(
    task: dict[str, Any],
) -> bool:
    if _task_is_explicit_support_scope(task):
        return False
    return True


def find_plan_acceptance_issues(
    plan: dict[str, Any],
    *,
    retry_failed: bool = False,
    retry_changes_requested: bool = False,
    retry_merge_conflicts: bool = False,
    only: str | None = None,
) -> list[PlanAcceptanceIssue]:
    tasks_raw = plan.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        return [
            PlanAcceptanceIssue(
                rule_id="R1",
                observed="no mutating execution candidates",
                detail="plan.tasks is missing or empty",
            )
        ]

    candidates = _candidate_execution_tasks(
        plan,
        retry_failed=retry_failed,
        retry_changes_requested=retry_changes_requested,
        retry_merge_conflicts=retry_merge_conflicts,
        only=only,
    )
    mutating_seen = False
    issues: list[PlanAcceptanceIssue] = []
    for _index, task, task_id, _status in candidates:
        requires_runnable_scope, estimated_files, write_scope = _mutating_task_scope(task)
        if not requires_runnable_scope:
            continue
        mutating_seen = True
        issues.extend(_required_task_field_errors(task, task_id=task_id))

        raw_write_scope = _raw_text_list(task.get("write_scope"))
        if raw_write_scope and not write_scope:
            issues.append(
                PlanAcceptanceIssue(
                    rule_id="R2",
                    task_id=task_id,
                    observed=_empty_write_scope_observed(raw_write_scope),
                    detail=EXECUTION_UNREADY_SCOPE_WARNING,
                )
            )
            continue
        missing_explicit_scope = _missing_explicit_scope_hints(
            task=task,
            estimated_files=estimated_files,
            write_scope=write_scope,
        )
        if missing_explicit_scope:
            issues.append(
                PlanAcceptanceIssue(
                    rule_id="R3",
                    task_id=task_id,
                    observed=(
                        "scope omits explicit task path hints: "
                        + ", ".join(missing_explicit_scope[:8])
                    ),
                    detail="write_scope=" + ", ".join(write_scope[:8]),
                )
            )

        if (
            write_scope
            and not any(
                _is_primary_implementation_path(path) for path in [*estimated_files, *write_scope]
            )
            and _task_needs_primary_implementation_scope(task)
        ):
            issues.append(
                PlanAcceptanceIssue(
                    rule_id="R3",
                    task_id=task_id,
                    observed=_metadata_only_observed(write_scope),
                    detail="write_scope=" + ", ".join(write_scope[:8]),
                )
            )

    if not mutating_seen and candidates:
        issues.insert(
            0,
            PlanAcceptanceIssue(
                rule_id="R1",
                observed="no mutating execution candidates",
                detail="execution candidates are read-only/report-only",
            ),
        )
    return issues


def _format_plan_acceptance_block(issues: list[PlanAcceptanceIssue]) -> str:
    if not issues:
        return ""
    rendered: list[str] = []
    for issue in issues[:5]:
        parts = [issue.rule_id]
        if issue.task_id:
            parts.append(f"task={issue.task_id}")
        parts.append(f"observed={issue.observed}")
        if issue.detail:
            parts.append(f"detail={issue.detail}")
        rendered.append(" ".join(parts))
    if len(issues) > 5:
        rendered.append(f"+{len(issues) - 5} more")
    return "Execution blocked: " + "; ".join(rendered)


def _find_cycle(ids_in_order: list[str], deps_map: dict[str, list[str]]) -> list[str] | None:
    white = 0
    gray = 1
    black = 2
    state: dict[str, int] = {task_id: white for task_id in ids_in_order}
    stack: list[str] = []
    stack_index: dict[str, int] = {}

    def _dfs(task_id: str) -> list[str] | None:
        state[task_id] = gray
        stack_index[task_id] = len(stack)
        stack.append(task_id)
        for dep_id in deps_map.get(task_id, []):
            dep_state = state.get(dep_id, white)
            if dep_state == white:
                cycle = _dfs(dep_id)
                if cycle is not None:
                    return cycle
                continue
            if dep_state == gray:
                start = stack_index.get(dep_id, 0)
                return stack[start:] + [dep_id]

        stack.pop()
        stack_index.pop(task_id, None)
        state[task_id] = black
        return None

    for task_id in ids_in_order:
        if state.get(task_id, white) != white:
            continue
        cycle = _dfs(task_id)
        if cycle is not None:
            return cycle
    return None


def validate_plan(plan: dict[str, Any]) -> list[str]:
    _warn_legacy_schema(plan)
    tasks_raw = plan.get("tasks")
    if not isinstance(tasks_raw, list):
        return ["Plan validation warning: tasks field is missing or not an array."]

    tasks: list[dict[str, Any]] = [task for task in tasks_raw if isinstance(task, dict)]
    known_ids = _task_ids(tasks)
    known_set = set(known_ids)
    warnings: list[str] = []

    deps_map: dict[str, list[str]] = {}
    for task in tasks:
        task_id = str(task.get("id") or "").strip()
        if not task_id or task_id not in known_set:
            continue
        status = canonical_task_status(str(task.get("status") or ""))

        deps = _string_list(task.get("dependencies"))
        deps_map[task_id] = (
            []
            if status in _NON_EXECUTABLE_OBSOLETE_STATUSES
            else [dep for dep in deps if dep in known_set]
        )

        for dep_id in deps:
            if dep_id not in known_set:
                warnings.append(f"Task {task_id} has unknown dependency id: {dep_id}")
            else:
                dep_task = next(
                    (
                        candidate
                        for candidate in tasks
                        if str(candidate.get("id") or "").strip() == dep_id
                    ),
                    None,
                )
                dep_status = canonical_task_status(str((dep_task or {}).get("status") or ""))
                if (
                    status not in _NON_EXECUTABLE_OBSOLETE_STATUSES
                    and dep_status in _NON_EXECUTABLE_OBSOLETE_STATUSES
                ):
                    warnings.append(f"Task {task_id} depends on non-executable task id: {dep_id}")

        if status in _NON_EXECUTABLE_OBSOLETE_STATUSES:
            continue

        acceptance = _string_list(task.get("acceptance_criteria"))
        if not acceptance:
            warnings.append(f"Task {task_id} is missing acceptance_criteria")

        if task_is_missing_runnable_scope(task):
            warnings.append(
                task_readiness_warning(
                    task_id=task_id,
                    title=str(task.get("title") or "").strip(),
                )
            )

        _normalized_mcp_scope, mcp_scope_warnings = normalize_task_mcp_scope(
            task.get("mcp_scope"),
            warning_prefix=f"Task {task_id}",
        )
        warnings.extend(mcp_scope_warnings)
        warnings.extend(_validate_task_asset_briefing_shape(task, task_id=task_id))

    cycle = _find_cycle(known_ids, deps_map)
    if cycle is not None:
        warnings.append("Circular dependency detected: " + " -> ".join(cycle))

    return warnings


def _warn_legacy_schema(plan: dict[str, Any]) -> None:
    try:
        schema_version = int(plan.get("schema_version", 1) or 1)
    except (TypeError, ValueError):
        schema_version = 1
    if schema_version <= 1:
        warnings.warn(
            "Plan schema_version 1 is deprecated. Load the run to migrate it to schema_version 2.",
            DeprecationWarning,
            stacklevel=2,
        )
    elif schema_version != 2:
        warnings.warn(
            f"Plan schema_version {schema_version} is not explicitly supported; "
            "validation will apply the closest compatible rules.",
            RuntimeWarning,
            stacklevel=2,
        )


def validate_plan_against_assets(
    plan: dict[str, Any],
    assets_source: Any,
    *,
    max_primary_per_task: int = 8,
) -> list[str]:
    records = _asset_records_by_id(assets_source)
    warnings: list[str] = []
    for task in plan.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("id") or "").strip() or "(unknown)"
        try:
            briefing = task_asset_briefing(task)
        except AssetError as exc:
            warnings.append(f"Task {task_id} has invalid asset_briefing: {exc}")
            continue
        if briefing is None:
            continue
        primary_ids = [entry.asset_id for entry in briefing.primary]
        may_need_ids = [entry.asset_id for entry in briefing.may_need]
        if len(primary_ids) > max_primary_per_task:
            warnings.append(
                f"Task {task_id} references too many primary assets: "
                f"{len(primary_ids)} > {max_primary_per_task}"
            )
        overlap = sorted(set(primary_ids) & set(may_need_ids))
        if overlap:
            warnings.append(
                f"Task {task_id} references the same asset in primary and may_need: "
                + ", ".join(overlap)
            )
        for asset_id in [*primary_ids, *may_need_ids]:
            record = records.get(asset_id)
            if record is None:
                warnings.append(f"Task {task_id} references missing asset id: {asset_id}")
            elif getattr(record, "deleted_at", None) is not None:
                warnings.append(f"Task {task_id} references deleted asset id: {asset_id}")
    return warnings


def _validate_task_asset_briefing_shape(
    task: dict[str, Any],
    *,
    task_id: str,
    max_primary_per_task: int = 8,
) -> list[str]:
    if "asset_briefing" not in task:
        return []
    try:
        briefing = task_asset_briefing(task)
    except AssetError as exc:
        return [f"Task {task_id} has invalid asset_briefing: {exc}"]
    if briefing is None:
        return []
    warnings: list[str] = []
    primary_ids = [entry.asset_id for entry in briefing.primary]
    may_need_ids = [entry.asset_id for entry in briefing.may_need]
    if len(primary_ids) > max_primary_per_task:
        warnings.append(
            f"Task {task_id} references too many primary assets: "
            f"{len(primary_ids)} > {max_primary_per_task}"
        )
    overlap = sorted(set(primary_ids) & set(may_need_ids))
    if overlap:
        warnings.append(
            f"Task {task_id} references the same asset in primary and may_need: "
            + ", ".join(overlap)
        )
    return warnings


def _asset_records_by_id(assets_source: Any) -> dict[str, Any]:
    records_method = getattr(assets_source, "records", None)
    if callable(records_method):
        return {
            str(record.id): record
            for record in records_method(include_deleted=True)
            if getattr(record, "id", "")
        }
    index = getattr(assets_source, "index", None)
    index_records = getattr(index, "records", None)
    if callable(index_records):
        return {
            str(record.id): record
            for record in index_records(include_deleted=True)
            if getattr(record, "id", "")
        }
    if isinstance(assets_source, dict):
        return assets_source
    return {}


def raise_for_execution_ready_plan(
    plan: dict[str, Any],
    *,
    retry_failed: bool = False,
    retry_changes_requested: bool = False,
    retry_merge_conflicts: bool = False,
    only: str | None = None,
) -> None:
    acceptance_issues = find_plan_acceptance_issues(
        plan,
        retry_failed=retry_failed,
        retry_changes_requested=retry_changes_requested,
        retry_merge_conflicts=retry_merge_conflicts,
        only=only,
    )
    if acceptance_issues:
        raise PlannerFailedError(_format_plan_acceptance_block(acceptance_issues))
