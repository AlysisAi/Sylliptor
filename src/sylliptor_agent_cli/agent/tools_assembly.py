from __future__ import annotations

import copy
import importlib
import inspect
import ipaddress
import json
import os
import re
import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ..approval_scope import (
    exact_command_scope,
    exact_file_set_scope,
    exact_verify_command_set_scope,
)
from ..config import AppConfig, ConfigError, resolve_role_temperature, resolve_web_search_policy
from ..context.tool_schema_budgeter import (
    CUSTOM_MCP_SCHEMA_FAMILIES,
    DEFAULT_CUSTOM_MCP_DESCRIPTION_MAX_CHARS,
    compact_custom_mcp_tool_parameters,
)
from ..crash_diagnostics import CrashDiagnosticLogger
from ..custom_tools import (
    CustomToolDiscoveryResult,
    CustomToolSessionState,
    CustomToolSpec,
    build_custom_tool_session_state,
    run_custom_tool,
)
from ..diff_paths import iter_patch_paths
from ..durable_service_manager import DurableServiceManager, ProcessOwnership
from ..execution_deadline import (
    DEFAULT_DEADLINE_CLEANUP_RESERVE_SECONDS,
    MINIMUM_SUBAGENT_START_SECONDS,
    MINIMUM_TOOL_START_SECONDS,
    DeadlineExhausted,
    DeadlineOperation,
    DeadlinePhase,
    ExecutionDeadline,
    deadline_timeout_or_raise,
)
from ..extensions.activation import ActivationDecision
from ..extensions.models import normalize_extension_id
from ..mcp.manager import ForgeTaskScopedMcpManager, McpManager
from ..mcp.models import ResolvedMcpConfig, ResolvedMcpServer
from ..model_registry import ModelRegistry
from ..model_router import ROLE_CODING, resolve_model_for_role
from ..policy import evaluate_shell_command
from ..runtime_kind import RuntimeKind, normalize_runtime_kind
from ..session_store import SessionStore
from ..skills import SkillBundle, SkillReadError, read_skill_bundle_file, resolve_skill_by_name
from ..step_budget import StepBudgetRequest, resolve_step_budget
from ..subagents import (
    SubagentDefinition,
    allowed_subagent_tool_names,
    canonical_subagent_name,
    normalize_subagent_mode,
    resolve_subagent_model_role,
)
from ..surface import (
    ApprovalRequest,
    NestedSubagentSurface,
    NoopSurface,
    PatchEvent,
    SubagentEndEvent,
    SubagentStartEvent,
)
from ..surface.base import Surface
from ..task_scope import (
    ancestor_directory_scope_patterns,
    is_non_material_untracked_path,
    scope_path_matches_pattern,
)
from ..terminal_manager import ProcessOutputSnapshot, TerminalLimitError, TerminalManager
from ..tools.availability import mark_available, mark_unavailable, register_tool_availability
from ..tools.fs import (
    FsError,
    fs_copy,
    fs_delete,
    fs_list,
    fs_mkdir,
    fs_move,
    fs_read,
    fs_read_lines,
    fs_write,
    prepare_fs_edit,
    write_prepared_fs_edit,
)
from ..tools.git import git_apply_patch, git_diff, git_history, git_status
from ..tools.history import history_search
from ..tools.registry import (
    built_in_subagent_tool_names,
    copied_tool_parameters,
    iter_builtin_tool_metadata,
    require_builtin_tool_metadata,
)
from ..tools.repo_map import repo_map
from ..tools.search import search_rg
from ..tools.shell import shell_run
from ..tools.symbols import symbol_search
from ..tools.test_discovery import test_discover
from ..tools.web import web_fetch
from ..tools.web_search import WebSearchError, resolve_web_search_runtime_status, web_search
from ..usage_tracker import UsageRecord, UsageSummary
from ..verification_command_analysis import (
    VerificationCommandEvidentiaryCapability,
    analyze_verification_command,
)
from ..verify_gate import (
    ResolvedVerifyCommands,
    VerifyError,
    is_authoritative_verify_command_selection,
    resolve_verify_commands,
    run_task_verification,
    trusted_shell_expression_command_set,
    validation_errors_for_selection,
    verification_command_specs_payload,
    verification_selection_payload,
    verify_run_result_to_payload,
)
from ..web_research import (
    build_web_fetch_recovery_search_query,
    canonicalize_web_url_input,
    normalize_web_url,
)
from ..workspace_context import resolve_workspace_context
from . import _patchable
from .errors import AgentRuntimeError, ApprovalDeclinedError, SessionWorkdirError
from .mutation_classification import classify_mutation_paths
from .prompt_context import (
    _MODE_FULLACCESS,
    ALWAYS_PROTECTED_WRITE_PREFIXES,
    _component_plugin_allowed,
    _extract_repo_relative_paths_from_text,
    _normalize_rel_match_path,
    _normalize_workspace_relpath,
    _normalized_authoritative_verify_commands,
    _normalized_verify_commands,
    _paths_require_verification,
    _PluginActivationIndex,
    _workspace_relpath_for_path,
    resolve_workdir_relpath_within_workspace,
)
from .verification_commands import (
    _expand_simple_verify_command_chain,
    _has_disallowed_shell_control_flow,
    _verify_run_commands_match_effective_contract,
)
from .verification_evidence import (
    VerificationEvidence,
    VerificationEvidenceCategory,
    classify_verification_evidence,
)

_turn_snapshot = importlib.import_module("sylliptor_agent_cli.agent.turn.snapshot")
_SHELL_MUTATION_SNAPSHOT_METADATA_PREFIX = _turn_snapshot._SHELL_MUTATION_SNAPSHOT_METADATA_PREFIX
_detect_command_mutation_paths = _turn_snapshot._detect_command_mutation_paths
_list_git_workspace_snapshot_paths = _turn_snapshot._list_git_workspace_snapshot_paths
_normalize_snapshot_ignore_paths = _turn_snapshot._normalize_snapshot_ignore_paths
_path_matches_snapshot_ignore = _turn_snapshot._path_matches_snapshot_ignore
_run_with_command_mutation_detection = _turn_snapshot._run_with_command_mutation_detection
_snapshot_workspace_for_command_mutation_detection = (
    _turn_snapshot._snapshot_workspace_for_command_mutation_detection
)
_walk_workspace_snapshot_paths = _turn_snapshot._walk_workspace_snapshot_paths
_workspace_snapshot_signature = _turn_snapshot._workspace_snapshot_signature


def _call_with_optional_kwargs(
    func: Callable[..., Any],
    *,
    required_kwargs: dict[str, Any],
    optional_kwargs: dict[str, Any],
) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**required_kwargs, **optional_kwargs)
    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    accepted_kwargs = dict(required_kwargs)
    for key, value in optional_kwargs.items():
        if accepts_var_kwargs or key in signature.parameters:
            accepted_kwargs[key] = value
    return func(**accepted_kwargs)


_AUTHORITATIVE_SUBAGENT_FINAL_TEXT_SOURCES = frozenset({"store_final", "surface_assistant_done"})


def _latest_subagent_store_final_text(sub_session: Any) -> tuple[str, bool]:
    child_store = getattr(sub_session, "store", None)
    events_snapshot = getattr(child_store, "events_snapshot", None)
    if not callable(events_snapshot):
        return "", False
    try:
        events = events_snapshot()
    except Exception:  # noqa: BLE001 - result capture should fall back instead of failing.
        return "", False
    for event in reversed(events if isinstance(events, list) else []):
        if not isinstance(event, dict) or str(event.get("type") or "") != "final":
            continue
        payload = event.get("payload")
        if not isinstance(payload, dict):
            continue
        text = str(payload.get("content") or "").strip()
        if text:
            return text, True
    return "", True


def _latest_subagent_message_text(sub_session: Any) -> str:
    for message in reversed(getattr(sub_session, "messages", [])):
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "") != "assistant":
            continue
        text = str(message.get("content") or "").strip()
        if text:
            return text
    return ""


def _resolve_subagent_final_text(
    *,
    sub_session: Any,
    subagent_surface: NestedSubagentSurface,
) -> tuple[str, str]:
    store_text, store_checked = _latest_subagent_store_final_text(sub_session)
    if store_text:
        return store_text, "store_final"
    if not store_checked:
        surface_text = str(subagent_surface.last_assistant_message_done or "").strip()
        if surface_text:
            return surface_text, "surface_assistant_done"
    message_text = _latest_subagent_message_text(sub_session)
    if message_text:
        return message_text, "assistant_message"
    return "", "missing"


def _subagent_final_report_problem(*, text: str, source: str) -> str | None:
    if not str(text or "").strip():
        return "missing_final_report"
    if source not in _AUTHORITATIVE_SUBAGENT_FINAL_TEXT_SOURCES:
        return "missing_final_report_signal"
    return None


def _command_mutation_metadata(
    *,
    root: Path,
    touched_repo_paths: list[str],
    command_was_verification: bool = False,
) -> dict[str, Any]:
    classifications = classify_mutation_paths(
        touched_repo_paths,
        root=root,
        command_was_verification=command_was_verification,
    )
    material = [item.path for item in classifications if item.is_material]
    benign = [item.path for item in classifications if not item.is_material]
    out: dict[str, Any] = {
        "mutation_path_classifications": [item.as_payload() for item in classifications],
    }
    if material:
        out["material_touched_repo_paths"] = material
    if benign:
        out["benign_runtime_paths"] = benign
    return out


def _verification_relevant_material_paths(paths: list[str]) -> list[str]:
    if not paths or not _paths_require_verification(set(paths)):
        return []
    return list(paths)


def _aggregate_tool_evidence_payload(records: list[VerificationEvidence]) -> dict[str, Any]:
    if not records:
        return {
            "verification_evidence_category": VerificationEvidenceCategory.NOT_VERIFICATION.value,
            "verification_evidence_reason": "no_verification_evidence",
            "verification_evidence_allowed": False,
            "verification_evidence_supplemental_only": False,
        }
    priority = {
        VerificationEvidenceCategory.AUTHORITATIVE: 0,
        VerificationEvidenceCategory.REPO_NATIVE: 1,
        VerificationEvidenceCategory.TASK_ACCEPTANCE: 2,
        VerificationEvidenceCategory.NOT_VERIFICATION: 3,
    }
    primary = sorted(records, key=lambda item: priority[item.category])[0]
    allowed = all(
        item.allowed_to_satisfy_contract
        for item in records
        if item.category != VerificationEvidenceCategory.NOT_VERIFICATION
    )
    if any(item.category == VerificationEvidenceCategory.NOT_VERIFICATION for item in records):
        allowed = False
    return {
        "verification_evidence_category": primary.category.value,
        "verification_evidence_reason": (
            primary.reason
            if allowed
            else next(
                (item.reason for item in records if not item.allowed_to_satisfy_contract),
                primary.reason,
            )
        ),
        "verification_evidence_allowed": allowed,
        "verification_evidence_supplemental_only": all(item.supplemental_only for item in records),
        "verification_evidence_records": [item.as_payload() for item in records],
    }


def _custom_tool_plugin_id(
    tool: CustomToolSpec,
    index: _PluginActivationIndex,
) -> str | None:
    parts = PurePosixPath(tool.relative_tool_path).parts
    if len(parts) >= 2 and parts[0] == "plugins":
        return index.slug_to_plugin_id.get(parts[1])
    return None


def _filter_custom_tool_session_state_for_plugins(
    *,
    state: CustomToolSessionState,
    activation_decision: ActivationDecision,
    index: _PluginActivationIndex,
) -> tuple[CustomToolSessionState, Counter[str]]:
    dropped_counts: Counter[str] = Counter()
    keep_cache: dict[str, bool] = {}

    def keep(tool: CustomToolSpec) -> bool:
        cache_key = os.fspath(tool.source_path)
        if cache_key in keep_cache:
            return keep_cache[cache_key]
        allowed = _component_plugin_allowed(
            _custom_tool_plugin_id(tool, index),
            activation_decision,
            dropped_counts,
        )
        keep_cache[cache_key] = allowed
        return allowed

    filtered_discovery = CustomToolDiscoveryResult(
        global_tools=tuple(tool for tool in state.discovery.global_tools if keep(tool)),
        project_tools=tuple(tool for tool in state.discovery.project_tools if keep(tool)),
        effective_tools=tuple(tool for tool in state.discovery.effective_tools if keep(tool)),
        shadowed_tools=tuple(tool for tool in state.discovery.shadowed_tools if keep(tool)),
        issues=state.discovery.issues,
    )
    return (
        CustomToolSessionState(
            discovery=filtered_discovery,
            trust_state=state.trust_state,
            catalog_entries=tuple(
                entry for entry in state.catalog_entries if entry.spec is None or keep(entry.spec)
            ),
            effective_tools_by_name=filtered_discovery.effective_tools_by_name(),
            exposed_tools_by_name={
                name: tool for name, tool in state.exposed_tools_by_name.items() if keep(tool)
            },
        ),
        dropped_counts,
    )


def _mcp_server_plugin_id(server: ResolvedMcpServer) -> str | None:
    raw = str(server.id or "")
    if "/" not in raw:
        return None
    return normalize_extension_id(raw.split("/", 1)[0])


def _filter_mcp_config_for_plugins(
    *,
    config: ResolvedMcpConfig,
    activation_decision: ActivationDecision,
) -> tuple[ResolvedMcpConfig, Counter[str]]:
    dropped_counts: Counter[str] = Counter()
    servers = tuple(
        server
        for server in config.servers
        if _component_plugin_allowed(
            _mcp_server_plugin_id(server),
            activation_decision,
            dropped_counts,
        )
    )
    return replace(config, servers=servers), dropped_counts


_MODE_PERMISSIVENESS_ORDER = ("readonly", "review", "auto", "fullaccess")


_MODE_PERMISSIVENESS_RANK = {
    mode_name: idx for idx, mode_name in enumerate(_MODE_PERMISSIVENESS_ORDER)
}

FULLACCESS_DENYLIST_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\s+/\s*$",
    r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r)\s+/\*",
    r"\bgit\s+push\s+.*--force.*\b(main|master)\b",
    r"\bsudo\b",
    r"\bcurl\s+[^|]*\|\s*sh\b",
    r"\bwget\s+[^|]*\|\s*sh\b",
    r"\bdd\s+if=/dev/",
    r"\bmkfs\.",
    r":\(\)\s*\{\s*:\|:&\s*\}\s*;\s*:",
    r"\bchmod\s+-R\s+777\s+/",
    r">\s*/dev/sd[a-z]",
]


def _normalize_fullaccess_shell_command(cmd: str) -> str:
    return " ".join(str(cmd).strip().split())


def _fullaccess_denylist_match(cmd: str) -> str | None:
    normalized = _normalize_fullaccess_shell_command(cmd)
    for pattern in FULLACCESS_DENYLIST_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return pattern
    return None


def _fullaccess_shell_audit_ts() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]
    run: Callable[[dict[str, Any]], dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_openai_tool(self) -> dict[str, Any]:
        family = _model_schema_family(self.metadata)
        description_max_chars = self.metadata.get("model_description_max_chars")
        if family in CUSTOM_MCP_SCHEMA_FAMILIES:
            description_max_chars = _schema_description_max_chars(description_max_chars)
        description = str(self.metadata.get("model_description") or self.description)
        description = _model_facing_tool_description(
            description,
            max_chars=description_max_chars,
        )
        parameters = self.parameters
        if bool(self.metadata.get("compact_parameters_for_model")):
            if family in CUSTOM_MCP_SCHEMA_FAMILIES:
                parameters = compact_custom_mcp_tool_parameters(self.parameters)
            else:
                parameters = _drop_model_facing_schema_prose(self.parameters)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": description,
                "parameters": parameters,
            },
        }


_MODEL_FACING_SCHEMA_PROSE_KEYS = frozenset(
    {
        "$comment",
        "description",
        "example",
        "examples",
        "markdownDescription",
        "title",
    }
)


def _model_schema_family(metadata: dict[str, Any]) -> str:
    tool_type = str(metadata.get("tool_type") or "").strip().lower()
    if tool_type == "custom_tool":
        return "custom"
    if tool_type in {"mcp", "mcp_tool"}:
        return "mcp"
    return tool_type


def _schema_description_max_chars(value: Any) -> int:
    try:
        configured = int(value)
    except (TypeError, ValueError):
        configured = 0
    if configured <= 0:
        return DEFAULT_CUSTOM_MCP_DESCRIPTION_MAX_CHARS
    return configured


def _model_facing_tool_description(description: str, *, max_chars: Any) -> str:
    text = " ".join(str(description or "").split())
    try:
        limit = int(max_chars)
    except (TypeError, ValueError):
        limit = 0
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _drop_model_facing_schema_prose(value: Any) -> Any:
    if isinstance(value, dict):
        reduced: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key)
            if normalized_key in _MODEL_FACING_SCHEMA_PROSE_KEYS:
                continue
            if normalized_key in {"const", "default", "enum"}:
                reduced[key] = copy.deepcopy(item)
                continue
            reduced[key] = _drop_model_facing_schema_prose(item)
        return reduced
    if isinstance(value, list):
        return [_drop_model_facing_schema_prose(item) for item in value]
    return copy.deepcopy(value)


def _drop_schema_descriptions(value: Any) -> Any:
    return _drop_model_facing_schema_prose(value)


_BUILTIN_MODEL_DESCRIPTIONS: dict[str, str] = {
    "fs_read": "Read a text file under the workspace.",
    "fs_read_lines": "Read numbered lines from a text file.",
    "fs_edit": "Apply exact-text edits to one UTF-8 file.",
    "fs_move": "Move or rename one workspace file.",
    "fs_copy": "Copy one workspace file.",
    "fs_delete": "Delete one workspace file.",
    "fs_write": "Write a UTF-8 text file.",
    "fs_mkdir": "Create a workspace directory.",
    "fs_list": "List workspace files with optional filters.",
    "web_fetch": "Fetch one provided or web_search HTTP(S) URL.",
    "symbol_search": "Find Python, JS/TS, or Java symbols; request details/snippets when planning edits.",
    "test_discover": "Suggest likely focused tests and commands for paths or failures.",
    "repo_map": "Map related files, imports, symbols, and likely tests before broad exploration.",
    "search_rg": "Search workspace text with a ripgrep regex.",
    "history_search": "Search session history and tool artifacts.",
    "verify_run": "Run configured verifier; no pipes/filters or swapped tools.",
    "shell_run": "Run a policy-checked shell command.",
    "shell_background": (
        "Run a background command with session lifetime; killed when this session ends, "
        "but survives across chat turns. Use shell_service_start for durable servers."
    ),
    "shell_service_start": (
        "Start a durable service with durable lifetime: keeps running after this session "
        "ends. Use for servers/daemons that must outlive the session; manage with status/stop."
    ),
    "workspace_preview_start": (
        "Serve static workspace files without Docker. Choose semantic access (auto/local/lan); "
        "the runtime resolves interfaces and a free port. LAN access is approval-gated and "
        "temporarily authenticated. Manage the service with shell_service_status/stop."
    ),
    "shell_service_status": (
        "Check a durable service that outlives the session and re-run its readiness probe."
    ),
    "shell_service_stop": (
        "Stop a durable service by service_id. Durable services otherwise keep running "
        "after the session ends."
    ),
    "shell_output": "Read buffered output from a background shell process.",
    "shell_wait": "Wait for background process output or exit without busy polling.",
    "shell_kill": "Terminate a background shell process.",
    "shell_list": "List background shell processes.",
    "session_set_workdir": (
        "Change active_workdir inside workspace_root for directory-scoped work."
    ),
    "subagent_run": (
        "Run an isolated subagent. Provide name and a self-contained task with "
        "goal/paths/context/output; mode/max_steps are optional."
    ),
    "git_status": "Run git status porcelain.",
    "git_diff": "Run git diff.",
    "git_history": "Inspect git log, show, or blame.",
    "git_apply_patch": "Apply a unified diff with git apply.",
}


def _subagent_exact_tool_catalog_message(
    tools: dict[str, ToolDef],
    *,
    max_tools: int = 80,
    max_chars: int = 8000,
) -> str:
    lines = [
        "<available_tool_catalog>",
        "Use only these exact tool names. Do not invent aliases or call unavailable tools.",
        "If a tool call returns unknown_tool, inspect its structured recovery payload and retry once with an exact listed name.",
        "tools:",
    ]
    for index, name in enumerate(sorted(tools)):
        if index >= max_tools:
            lines.append("- ...(truncated)")
            break
        tool = tools[name]
        required = tool.parameters.get("required") if isinstance(tool.parameters, dict) else []
        required_args = [
            str(item)
            for item in (required if isinstance(required, list) else [])
            if str(item).strip()
        ]
        purpose = " ".join(
            str(tool.metadata.get("model_description") or tool.description or "").split()
        )
        if len(purpose) > 140:
            purpose = purpose[:137].rstrip() + "..."
        required_text = ", ".join(required_args) if required_args else "(none)"
        candidate = f"- {name}: {purpose or '-'} required_args={required_text}"
        projected = "\n".join([*lines, candidate, "</available_tool_catalog>\n"])
        if len(projected) > max_chars:
            lines.append("- ...(truncated)")
            break
        lines.append(candidate)
    lines.append("</available_tool_catalog>\n")
    return "\n".join(lines)


def _tool_event_metadata(tool: ToolDef | None) -> dict[str, Any]:
    if tool is None or not tool.metadata:
        return {}
    metadata = copy.deepcopy(tool.metadata)
    event_metadata: dict[str, Any] = {}
    tool_type = str(metadata.get("tool_type") or "").strip()
    if tool_type:
        event_metadata["tool_type"] = tool_type
    custom_tool = metadata.get("custom_tool")
    if isinstance(custom_tool, dict):
        event_metadata["custom_tool"] = {
            key: value for key, value in custom_tool.items() if key != "output_schema"
        }
    return event_metadata


def _custom_tool_capability_summary(spec: Any) -> str:
    capabilities = getattr(spec, "capabilities", None)
    if capabilities is None:
        return "capabilities: unspecified"
    secret_refs = getattr(capabilities, "secret_refs", ())
    secret_summary = ", ".join(secret_refs) if secret_refs else "-"
    network_hosts = getattr(capabilities, "network_hosts", ())
    network_hosts_summary = ", ".join(network_hosts) if network_hosts else "-"
    return (
        "capabilities: "
        f"read_only={bool(getattr(capabilities, 'read_only', False))}, "
        f"destructive={bool(getattr(capabilities, 'destructive', False))}, "
        f"network={getattr(capabilities, 'network_access', 'unspecified')}, "
        f"network_hosts={network_hosts_summary}, "
        f"fs_read={getattr(capabilities, 'filesystem_read_scope', 'unspecified')}, "
        f"fs_write={getattr(capabilities, 'filesystem_write_scope', 'unspecified')}, "
        f"process_spawn={getattr(capabilities, 'process_spawn', 'unspecified')}, "
        f"secrets={secret_summary}"
    )


_ROUTING_MODE_CODE_ONLY = "code_only"


_READONLY_MAIN_SESSION_BUILTIN_TOOL_NAMES = frozenset(
    built_in_subagent_tool_names(exposure="readonly")
)


_READONLY_TOP_LEVEL_WEB_TOOL_NAMES = frozenset({"web_fetch", "web_search"})


def _built_in_tool_exposed_in_mode(
    *,
    tool_name: str,
    mode: str,
    subagent_depth: int = 0,
) -> bool:
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode != "readonly":
        return True
    normalized_tool_name = str(tool_name or "").strip()
    if normalized_tool_name in _READONLY_MAIN_SESSION_BUILTIN_TOOL_NAMES:
        return True
    # Top-level Plan/readonly sessions can safely use bounded web discovery and
    # fetch tools; nested readonly subagents stay on the narrower catalog policy.
    return subagent_depth == 0 and normalized_tool_name in _READONLY_TOP_LEVEL_WEB_TOOL_NAMES


def _mcp_tool_exposed_in_mode(*, mode: str) -> bool:
    return str(mode or "").strip().lower() != "readonly"


def _custom_tools_write_scope_restricted(
    *,
    mode: str,
    deny_write_prefixes: list[str] | None,
    allow_write_globs: list[str] | None,
) -> bool:
    if str(mode or "").strip().lower() == _MODE_FULLACCESS:
        return False
    if allow_write_globs is not None:
        return True
    always_protected = {
        _normalize_rel_match_path(prefix).casefold()
        for prefix in ALWAYS_PROTECTED_WRITE_PREFIXES
        if _normalize_rel_match_path(prefix)
    }
    for raw in deny_write_prefixes or []:
        cleaned = _normalize_rel_match_path(str(raw))
        if cleaned and cleaned.casefold() not in always_protected:
            return True
    return False


def _unified_diff(old: str, new: str, path: str) -> str:
    import difflib

    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff)


def _create_session_for_subagent(
    *,
    create_session_factory: Callable[..., Any] | None,
    **kwargs: Any,
) -> Any:
    create_session = _patchable("create_session", create_session_factory)
    if callable(create_session):
        return create_session(**kwargs)
    raise AgentRuntimeError("create_session is unavailable for subagent execution")


def build_tools(
    *,
    root: Path,
    console: Any | None,
    surface: Surface | None = None,
    store: SessionStore,
    mode: str,
    yes: bool,
    cfg: AppConfig | None = None,
    api_key: str | None = None,
    max_steps: int | None = None,
    no_log: bool = False,
    usage_role: str = "main",
    usage_summary: UsageSummary | None = None,
    model_registry: ModelRegistry | None = None,
    deny_write_prefixes: list[str] | None = None,
    allow_write_globs: list[str] | None = None,
    non_interactive: bool = False,
    shell_runner: Any | None = None,
    terminal_manager: TerminalManager | None = None,
    durable_service_manager: DurableServiceManager | None = None,
    verification_enabled: bool = True,
    authoritative_verification_commands: list[str] | None = None,
    effective_verification_commands: list[str] | None = None,
    verify_command_selection: ResolvedVerifyCommands | None = None,
    get_verify_command_selection: Callable[[], ResolvedVerifyCommands | None] | None = None,
    one_shot_execution: bool = False,
    skills_enabled: bool = True,
    skill_registry: dict[str, SkillBundle] | None = None,
    subagents_enabled: bool = False,
    subagent_depth: int = 0,
    subagent_registry: dict[str, SubagentDefinition] | None = None,
    session_log_dir_override: Path | None = None,
    step_budget_runtime: Any | None = None,
    emit_web_search_runtime_diagnostics: bool = False,
    runtime_kind: RuntimeKind | str = RuntimeKind.ONE_SHOT,
    mcp_manager: McpManager | ForgeTaskScopedMcpManager | None = None,
    custom_tool_session_state: CustomToolSessionState | None = None,
    get_active_workdir_relpath: Callable[[], str] | None = None,
    set_active_workdir_callback: Callable[[str, str], dict[str, Any]] | None = None,
    create_session_factory: Callable[..., Any] | None = None,
    execution_deadline: ExecutionDeadline | None = None,
    crash_diagnostic_log_path: str | os.PathLike[str] | None = None,
    crash_diagnostics: CrashDiagnosticLogger | None = None,
) -> dict[str, ToolDef]:
    root = root.resolve()
    workspace_context = resolve_workspace_context(root)
    surface = surface or NoopSurface()
    host_managed_approvals = bool(getattr(surface, "host_managed_approvals", False))
    resolved_runtime_kind = normalize_runtime_kind(
        runtime_kind, fallback=RuntimeKind.INTERACTIVE_CHAT
    )
    authoritative_verify_commands = _normalized_authoritative_verify_commands(
        authoritative_verification_commands
    )
    static_verify_selection = verify_command_selection
    normalized_effective_verification_commands = _normalized_verify_commands(
        effective_verification_commands
        or (
            list(static_verify_selection.commands)
            if isinstance(static_verify_selection, ResolvedVerifyCommands)
            else []
        )
    )

    def _deadline_payload() -> dict[str, Any]:
        if execution_deadline is None:
            return {
                "failure_category": "deadline",
                "deadline_exhausted": False,
                "remaining_seconds": None,
                "deadline": None,
            }
        remaining = execution_deadline.remaining_seconds()
        return {
            "failure_category": "deadline",
            "deadline_exhausted": execution_deadline.is_exhausted(),
            "remaining_seconds": remaining,
            "deadline": execution_deadline.telemetry_snapshot(),
        }

    def _deadline_error(
        message: str,
        *,
        prevented_launch: bool = True,
        start_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "error": message,
            "deadline_prevented_launch": prevented_launch,
            **_deadline_payload(),
        }
        if start_decision is not None:
            payload["deadline_start_decision"] = start_decision
        if crash_diagnostics is not None:
            crash_diagnostics.event(
                "deadline_exhausted",
                {
                    "operation": "tool",
                    "deadline_exhausted": payload["deadline_exhausted"],
                    "remaining_seconds": payload["remaining_seconds"],
                    "deadline": payload["deadline"],
                    "deadline_start_decision": start_decision,
                },
                durable=True,
            )
        return payload

    def _deadline_warning_fields(
        message: str,
        *,
        start_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "deadline_warning": message,
            "deadline_prevented_launch": False,
            **_deadline_payload(),
        }
        if start_decision is not None:
            payload["deadline_start_decision"] = start_decision
        if crash_diagnostics is not None:
            crash_diagnostics.event(
                "deadline_exhausted",
                {
                    "operation": "tool",
                    "deadline_exhausted": payload["deadline_exhausted"],
                    "remaining_seconds": payload["remaining_seconds"],
                    "deadline": payload["deadline"],
                    "deadline_start_decision": start_decision,
                },
                durable=True,
            )
        return payload

    def _deadline_start_decision(
        operation: DeadlineOperation,
        *,
        minimum_remaining_seconds: float,
        configured_timeout_seconds: float | None = None,
        allow_during_finalization: bool = False,
    ) -> dict[str, Any] | None:
        if execution_deadline is None:
            return None
        return execution_deadline.start_decision(
            operation,
            minimum_remaining_seconds=minimum_remaining_seconds,
            configured_timeout_seconds=configured_timeout_seconds,
            allow_during_finalization=allow_during_finalization,
        ).telemetry_snapshot()

    def _deadline_timeout(
        configured_timeout_seconds: float,
        *,
        operation: str,
    ) -> float:
        timeout = deadline_timeout_or_raise(
            execution_deadline,
            configured_timeout_seconds,
            reserve_seconds=DEFAULT_DEADLINE_CLEANUP_RESERVE_SECONDS,
            operation=operation,
        )
        return float(configured_timeout_seconds if timeout is None else timeout)

    def _current_verify_selection() -> ResolvedVerifyCommands | None:
        if callable(get_verify_command_selection):
            try:
                current = get_verify_command_selection()
            except Exception:  # noqa: BLE001
                current = None
            if isinstance(current, ResolvedVerifyCommands):
                return current
        if isinstance(static_verify_selection, ResolvedVerifyCommands):
            return static_verify_selection
        if authoritative_verify_commands is not None:
            return ResolvedVerifyCommands(
                commands=tuple(authoritative_verify_commands),
                source="environment.authoritative_verification_commands",
                reason="managed runtime injected authoritative verification commands",
                contract_type="authoritative_override",
            )
        if normalized_effective_verification_commands:
            return ResolvedVerifyCommands(
                commands=tuple(normalized_effective_verification_commands),
                source="session.effective_verification_commands",
                reason="session already resolved an effective verification contract",
                contract_type="selected",
            )
        return None

    command_mutation_tracking_enabled = bool(
        subagent_depth == 0
        and (one_shot_execution or resolved_runtime_kind == RuntimeKind.INTERACTIVE_CHAT)
    )
    command_mutation_ignored_paths: list[Path] = []
    if command_mutation_tracking_enabled:
        command_mutation_ignored_paths = [
            candidate
            for candidate in [
                getattr(store, "path", None),
                getattr(store, "session_artifact_root", None),
            ]
            if isinstance(candidate, Path)
        ]
    history_artifact_persistence_available = bool(
        getattr(store, "enabled", False) or session_log_dir_override is not None
    )
    git_backed_workspace = workspace_context.git_root is not None
    resolved_skill_registry = dict(skill_registry or {})
    built_in_tool_names = {spec.name.casefold() for spec in iter_builtin_tool_metadata()}
    if custom_tool_session_state is None:
        custom_tool_session_state = build_custom_tool_session_state(
            workspace_root=root,
            custom_tools_enabled=bool(getattr(cfg, "custom_tools_enabled", True)) if cfg else True,
            mode=mode,
            runtime_kind=resolved_runtime_kind,
            built_in_tool_names=built_in_tool_names,
            write_scope_restricted=_custom_tools_write_scope_restricted(
                mode=mode,
                deny_write_prefixes=deny_write_prefixes,
                allow_write_globs=allow_write_globs,
            ),
        )

    is_full_access_mode = mode == _MODE_FULLACCESS

    deny_prefixes: list[str] = []
    if not is_full_access_mode:
        seen_deny_prefixes: set[str] = set()
        for raw in [
            *ALWAYS_PROTECTED_WRITE_PREFIXES,
            *(deny_write_prefixes or []),
        ]:
            cleaned = _normalize_rel_match_path(str(raw))
            if cleaned:
                normalized = cleaned.casefold()
                if normalized not in seen_deny_prefixes:
                    seen_deny_prefixes.add(normalized)
                    deny_prefixes.append(cleaned)
    deny_prefixes_cf = [pref.casefold() for pref in deny_prefixes]
    allow_patterns: list[str] | None = None
    allowed_ancestor_dirs_cf: set[str] = set()
    if not is_full_access_mode and allow_write_globs is not None:
        allow_patterns = []
        for raw in allow_write_globs:
            cleaned = _normalize_rel_match_path(str(raw))
            if cleaned:
                allow_patterns.append(cleaned)
        allowed_ancestor_dirs_cf = {
            _normalize_rel_match_path(path).casefold()
            for path in ancestor_directory_scope_patterns(allow_write_globs)
            if _normalize_rel_match_path(path)
        }

    def _is_denied_path(rel_path: str) -> bool:
        if not deny_prefixes_cf:
            return False
        rel_norm = _normalize_rel_match_path(rel_path)
        rel_cf = rel_norm.casefold()
        for pref_cf in deny_prefixes_cf:
            if rel_cf == pref_cf or rel_cf.startswith(pref_cf + "/"):
                return True
        return False

    def _resolve_rel_path(rel_path: str) -> str:
        root_abs = root.resolve()
        target = (root_abs / rel_path).resolve()
        try:
            normalized = target.relative_to(root_abs)
        except ValueError as e:
            raise AgentRuntimeError(f"Path escapes root: {rel_path}") from e
        return os.fspath(normalized)

    def _resolve_rel_write_path(rel_path: str) -> str:
        return _resolve_rel_path(rel_path)

    def _guard_write_path(rel_path: str) -> None:
        if is_full_access_mode:
            return
        if _is_denied_path(rel_path):
            raise AgentRuntimeError(f"Blocked write to protected path: {rel_path}")
        if allow_patterns is not None:
            rel_norm = _normalize_rel_match_path(rel_path)
            in_scope = any(
                scope_path_matches_pattern(rel_norm, pattern, root=root)
                for pattern in allow_patterns
            )
            if not in_scope:
                rel_cf = rel_norm.casefold()
                in_scope = any(
                    rel_cf == _normalize_rel_match_path(pattern).casefold()
                    for pattern in allow_patterns
                    if not any(ch in pattern for ch in ["*", "?", "["])
                )
            if not in_scope:
                raise AgentRuntimeError(f"Blocked write outside allowed scope: {rel_path}")

    def _is_allowed_ancestor_dir_creation(rel_path: str) -> bool:
        if is_full_access_mode or not allowed_ancestor_dirs_cf:
            return False
        rel_norm = _normalize_rel_match_path(rel_path).casefold()
        return rel_norm in allowed_ancestor_dirs_cf

    def guard_write(kind: str, preview: str, *, files: list[str] | None = None) -> None:
        if is_full_access_mode:
            return
        if mode == "readonly":
            raise AgentRuntimeError(f"Blocked in readonly mode: {kind}")
        if mode == "review":
            if non_interactive and not host_managed_approvals:
                raise AgentRuntimeError(
                    "Confirmation required for sensitive command. Re-run with --yes or adjust plan."
                )
            decision = surface.request_approval(
                ApprovalRequest(
                    kind=kind,
                    reason="review mode requires confirmation for write operations",
                    preview=preview,
                    files=files or [],
                    allow_for_session_scope=exact_file_set_scope(files or [], operation=kind)
                    if files
                    else None,
                )
            )
            if not decision.allow:
                raise ApprovalDeclinedError(kind)
        if mode == "auto" and kind == "fs_delete" and not yes:
            if non_interactive and not host_managed_approvals:
                raise AgentRuntimeError(
                    "Confirmation required for sensitive command. Re-run with --yes or adjust plan."
                )
            decision = surface.request_approval(
                ApprovalRequest(
                    kind=kind,
                    reason="file deletion requires confirmation",
                    preview=preview,
                    files=files or [],
                    allow_for_session_scope=exact_file_set_scope(files or [], operation=kind)
                    if files
                    else None,
                )
            )
            if not decision.allow:
                raise ApprovalDeclinedError(kind)

    def guard_shell(cmd: str, *, tool_name: str = "shell_run") -> None:
        if matched_pattern := _fullaccess_denylist_match(cmd):
            if tool_name != "shell_run" and not is_full_access_mode:
                raise AgentRuntimeError(f"Blocked command: denylist pattern {matched_pattern}")
            raise AgentRuntimeError(
                f"Blocked fullaccess shell command by denylist pattern: {matched_pattern}"
            )
        if is_full_access_mode:
            return
        if mode == "readonly":
            raise AgentRuntimeError(f"Blocked in readonly mode: {tool_name}")
        decision = evaluate_shell_command(cmd)
        if not decision.allowed:
            raise AgentRuntimeError(f"Blocked command: {decision.reason}")
        if mode == "review":
            if non_interactive and not host_managed_approvals:
                raise AgentRuntimeError(
                    "Confirmation required for sensitive command. Re-run with --yes or adjust plan."
                )
            decision = surface.request_approval(
                ApprovalRequest(
                    kind=tool_name,
                    reason="review mode requires confirmation for shell commands",
                    preview=cmd,
                    command=cmd,
                    allow_for_session_scope=exact_command_scope(cmd, kind=tool_name),
                )
            )
            if not decision.allow:
                raise ApprovalDeclinedError(tool_name)
            return
        # auto mode
        if decision.needs_confirm and not yes:
            if non_interactive and not host_managed_approvals:
                raise AgentRuntimeError(
                    "Confirmation required for sensitive command. Re-run with --yes or adjust plan."
                )
            choice = surface.request_approval(
                ApprovalRequest(
                    kind=tool_name,
                    reason=f"sensitive command: {decision.reason}",
                    preview=cmd,
                    command=cmd,
                    allow_for_session_scope=exact_command_scope(cmd, kind=tool_name),
                )
            )
            if not choice.allow:
                raise ApprovalDeclinedError(tool_name)

    def guard_terminal_op(op_name: str) -> None:
        if is_full_access_mode:
            return
        if mode == "readonly":
            raise AgentRuntimeError(f"Blocked in readonly mode: {op_name}")

    def guard_verify(commands: list[str]) -> None:
        if is_full_access_mode:
            return
        if mode == "readonly":
            raise AgentRuntimeError("Blocked in readonly mode: verify_run")

        sensitive_reason: str | None = None
        for command in commands:
            decision = evaluate_shell_command(command)
            if not decision.allowed:
                raise AgentRuntimeError(f"Blocked command: {decision.reason}")
            if sensitive_reason is None and decision.needs_confirm:
                sensitive_reason = decision.reason

        preview = "\n".join(f"$ {command}" for command in commands)
        command_label = (
            commands[0] if len(commands) == 1 else f"{len(commands)} verification commands"
        )

        if mode == "review":
            if non_interactive and not host_managed_approvals:
                raise AgentRuntimeError(
                    "Confirmation required for sensitive command. Re-run with --yes or adjust plan."
                )
            decision = surface.request_approval(
                ApprovalRequest(
                    kind="verify_run",
                    reason="review mode requires confirmation for verification commands",
                    preview=preview,
                    command=command_label,
                    allow_for_session_scope=exact_verify_command_set_scope(commands),
                )
            )
            if not decision.allow:
                raise ApprovalDeclinedError("verify_run")
            return

        if sensitive_reason and not yes:
            if non_interactive and not host_managed_approvals:
                raise AgentRuntimeError(
                    "Confirmation required for sensitive command. Re-run with --yes or adjust plan."
                )
            choice = surface.request_approval(
                ApprovalRequest(
                    kind="verify_run",
                    reason=f"sensitive command in verification set: {sensitive_reason}",
                    preview=preview,
                    command=command_label,
                    allow_for_session_scope=exact_verify_command_set_scope(commands),
                )
            )
            if not choice.allow:
                raise ApprovalDeclinedError("verify_run")

    parent_mode_normalized = normalize_subagent_mode(mode)

    def _clamp_subagent_mode(requested_mode: str) -> str:
        requested = normalize_subagent_mode(requested_mode)
        requested_rank = _MODE_PERMISSIVENESS_RANK.get(
            requested,
            _MODE_PERMISSIVENESS_RANK["auto"],
        )
        parent_rank = _MODE_PERMISSIVENESS_RANK.get(
            parent_mode_normalized,
            _MODE_PERMISSIVENESS_RANK["auto"],
        )
        effective_rank = min(requested_rank, parent_rank)
        if parent_mode_normalized != _MODE_FULLACCESS:
            effective_rank = min(effective_rank, _MODE_PERMISSIVENESS_RANK["auto"])
        return _MODE_PERMISSIVENESS_ORDER[effective_rank]

    def _replay_subagent_usage(*, sub_session: Any) -> int:
        child_usage_summary = getattr(sub_session, "usage_summary", None)
        records_fn = getattr(child_usage_summary, "records", None)
        if not callable(records_fn):
            return 0
        raw_records = records_fn()
        records = [record for record in raw_records if isinstance(record, UsageRecord)]
        if not records:
            return 0
        if usage_summary is not None:
            usage_summary.merge_records(records)
        for record in records:
            store.append("llm_usage", record.to_payload())
        return len(records)

    def _resolve_subagent_definition(raw_name: str) -> SubagentDefinition | None:
        if not subagent_registry:
            return None
        normalized = canonical_subagent_name(raw_name)
        if normalized is None:
            return None
        return subagent_registry.get(normalized)

    def _resolve_subagent_model(
        definition: SubagentDefinition, cfg_copy: AppConfig
    ) -> tuple[str, str]:
        role = resolve_subagent_model_role(definition.model_role)
        if definition.model:
            selected_model = str(definition.model).strip()
            temperature_role = role or ROLE_CODING
            return selected_model, temperature_role
        if role:
            selected_model = resolve_model_for_role(
                cfg=cfg_copy,
                role=role,
                plan=None,
            )
            return selected_model, role
        selected_model = resolve_model_for_role(
            cfg=cfg_copy,
            role=ROLE_CODING,
            plan=None,
        )
        return selected_model, ROLE_CODING

    def _subagent_run(args: dict[str, Any]) -> dict[str, Any]:
        if not subagents_enabled:
            return {"error": "Subagents are disabled for this session."}
        if subagent_depth > 0:
            return {"error": "Subagents cannot invoke subagents (nesting is blocked)."}
        provider_auth_available = False
        if cfg is not None and not api_key:
            try:
                from ..profiles import get_active_profile

                provider_auth_available = bool(get_active_profile(cfg).auth_provider)
            except Exception:
                provider_auth_available = False
        if cfg is None or (not api_key and not provider_auth_available):
            return {"error": "Subagent execution unavailable: missing session configuration."}

        raw_name = str(args.get("name", "")).strip()
        task = str(args.get("task", "")).strip()
        if not raw_name:
            return {"error": "Missing required argument: name"}
        if not task:
            return {"error": "Missing required argument: task"}

        definition = _resolve_subagent_definition(raw_name)
        if definition is None:
            available = sorted(subagent_registry.keys()) if subagent_registry else []
            return {
                "error": f"Unknown subagent: {raw_name}",
                "available_subagents": available,
            }
        deadline_decision = _deadline_start_decision(
            DeadlineOperation.SUBAGENT,
            minimum_remaining_seconds=MINIMUM_SUBAGENT_START_SECONDS,
        )
        if deadline_decision is not None and not bool(deadline_decision.get("allowed")):
            payload = _deadline_error(
                "Subagent launch skipped because the run deadline has too little remaining time.",
                prevented_launch=True,
                start_decision=deadline_decision,
            )
            payload.update(
                {
                    "subagent": definition.name,
                    "subagent_session_id": None,
                    "steps_completed": 0,
                    "elapsed_ms": 0,
                }
            )
            store.append(
                "subagent_end",
                {
                    "name": definition.name,
                    "subagent_session_id": None,
                    "status": "failed",
                    "failure_category": "deadline",
                    "deadline_exhausted": payload["deadline_exhausted"],
                    "deadline_prevented_launch": True,
                    "remaining_seconds": payload["remaining_seconds"],
                    "steps_completed": 0,
                    "elapsed_ms": 0,
                },
            )
            return payload

        requested_mode = normalize_subagent_mode(definition.mode)
        mode_override = str(args.get("mode", "") or "").strip()
        if mode_override:
            requested_mode = normalize_subagent_mode(mode_override)
        resolved_mode = _clamp_subagent_mode(requested_mode)

        subagent_cfg = cfg.model_copy(deep=True)
        try:
            selected_model, temperature_role = _resolve_subagent_model(definition, subagent_cfg)
        except ConfigError as e:
            return {"error": f"Subagent model resolution failed: {e}"}
        subagent_cfg.model = selected_model
        subagent_cfg.routing_mode = _ROUTING_MODE_CODE_ONLY
        resolved_temperature = resolve_role_temperature(subagent_cfg, role=temperature_role)
        subagent_cfg.temperature = resolved_temperature
        subagent_cfg.coding_temperature = resolved_temperature

        active_turn_budget = getattr(step_budget_runtime, "active_turn_budget", None)
        if type(active_turn_budget) is int and active_turn_budget > 0:
            parent_turn_budget = active_turn_budget
        elif type(max_steps) is int and max_steps > 0:
            parent_turn_budget = max_steps
        else:
            parent_turn_budget = max(1, int(cfg.max_steps))
        explicit_subagent_max_steps = args.get("max_steps") if "max_steps" in args else None
        resolution = resolve_step_budget(
            StepBudgetRequest(
                kind="subagent",
                policy=cfg.step_budget_policy,
                hard_cap=cfg.subagent_max_steps,
                fixed_override=(
                    explicit_subagent_max_steps if explicit_subagent_max_steps is not None else None
                ),
                mode=resolved_mode,
                subagent_name=definition.name,
                parent_turn_budget=parent_turn_budget,
                explicit_path_count=len(
                    _extract_repo_relative_paths_from_text(root=root, text=task)
                ),
            )
        )
        effective_subagent_max_steps = resolution.resolved_max_steps

        store.append(
            "subagent_start",
            {
                "name": definition.name,
                "subagent_session_id": None,
                "mode": resolved_mode,
                "model": selected_model,
                "temperature_role": temperature_role,
                "temperature": resolved_temperature,
                "max_steps": effective_subagent_max_steps,
                "parent_turn_budget": parent_turn_budget,
                "step_budget": resolution.to_payload(),
                "task": task,
                "deadline": (
                    execution_deadline.telemetry_snapshot()
                    if execution_deadline is not None
                    else None
                ),
            },
        )
        if crash_diagnostics is not None:
            crash_diagnostics.event(
                "subagent_started",
                {
                    "subagent": definition.name,
                    "subagent_role": temperature_role,
                    "model": selected_model,
                    "max_steps": effective_subagent_max_steps,
                    "deadline": (
                        execution_deadline.telemetry_snapshot()
                        if execution_deadline is not None
                        else None
                    ),
                },
            )
        subagent_surface = NestedSubagentSurface(
            surface,
            subagent_name=definition.name,
            subagent_mode=resolved_mode,
        )
        subagent_started_at = perf_counter()
        subagent_surface.on_subagent_start(
            SubagentStartEvent(
                name=definition.name,
                mode=resolved_mode,
            )
        )

        try:
            sub_session = _create_session_for_subagent(
                create_session_factory=create_session_factory,
                cfg=subagent_cfg,
                root=root,
                mode=resolved_mode,
                runtime_kind=RuntimeKind.SUBAGENT,
                yes=yes,
                max_steps=effective_subagent_max_steps,
                no_log=no_log,
                api_key_override=api_key or None,
                console=None,
                deny_write_prefixes=deny_write_prefixes,
                allow_write_globs=allow_write_globs,
                non_interactive=non_interactive,
                session_log_dir_override=session_log_dir_override,
                surface=subagent_surface,
                usage_role=f"{usage_role}:subagent:{definition.name}",
                trusted_system_prompt_append=(
                    definition.system_prompt if definition.prompt_trust == "trusted" else None
                ),
                untrusted_prompt_prelude=(
                    definition.system_prompt if definition.prompt_trust != "trusted" else None
                ),
                enable_compaction=False,
                enable_chat_turn_step_budget=False,
                verification_enabled=verification_enabled,
                authoritative_verification_commands=authoritative_verification_commands,
                one_shot_execution=False,
                subagents_enabled=False,
                subagent_depth=subagent_depth + 1,
                subagent_registry=subagent_registry,
                execution_deadline=execution_deadline,
                crash_diagnostic_log_path=crash_diagnostic_log_path,
            )
        except Exception as e:  # noqa: BLE001
            elapsed_ms = int((perf_counter() - subagent_started_at) * 1000)
            subagent_surface.on_subagent_end(
                SubagentEndEvent(
                    name=definition.name,
                    mode=resolved_mode,
                    status="failed",
                    elapsed_ms=elapsed_ms,
                    steps_completed=subagent_surface.steps_completed,
                    error=f"Failed to initialize subagent session: {e}",
                )
            )
            store.append(
                "subagent_end",
                {
                    "name": definition.name,
                    "subagent_session_id": None,
                    "status": "failed",
                    "error": f"Failed to initialize subagent session: {e}",
                    "elapsed_ms": elapsed_ms,
                    "steps_completed": subagent_surface.steps_completed,
                },
            )
            return {
                "error": f"Failed to initialize subagent session: {e}",
                "elapsed_ms": elapsed_ms,
                "steps_completed": subagent_surface.steps_completed,
            }

        subagent_session_id = str(
            getattr(getattr(sub_session, "store", None), "session_id", "") or ""
        )
        main_tool_names = list(sub_session.tools.keys())
        allowed_names = allowed_subagent_tool_names(
            tool_names=main_tool_names,
            allow_tools=definition.allow_tools,
            deny_tools=definition.deny_tools,
        )
        filtered_tools = {
            name: sub_session.tools[name] for name in allowed_names if name in sub_session.tools
        }
        sub_session.tools = filtered_tools
        sub_session.tool_list = [tool.as_openai_tool() for tool in filtered_tools.values()]
        if not filtered_tools:
            sub_session.close()
            elapsed_ms = int((perf_counter() - subagent_started_at) * 1000)
            subagent_surface.on_subagent_end(
                SubagentEndEvent(
                    name=definition.name,
                    mode=resolved_mode,
                    status="failed",
                    elapsed_ms=elapsed_ms,
                    steps_completed=subagent_surface.steps_completed,
                    subagent_session_id=subagent_session_id,
                    error="No tools available after allow/deny sandboxing.",
                )
            )
            store.append(
                "subagent_end",
                {
                    "name": definition.name,
                    "subagent_session_id": subagent_session_id,
                    "status": "failed",
                    "error": "No tools available after allow/deny sandboxing.",
                    "elapsed_ms": elapsed_ms,
                    "steps_completed": subagent_surface.steps_completed,
                },
            )
            return {
                "error": f"Subagent '{definition.name}' has no available tools after sandboxing.",
                "elapsed_ms": elapsed_ms,
                "steps_completed": subagent_surface.steps_completed,
            }

        tool_catalog_message = _subagent_exact_tool_catalog_message(filtered_tools)
        if isinstance(getattr(sub_session, "messages", None), list):
            sub_session.messages.append({"role": "system", "content": tool_catalog_message})
        store.append(
            "subagent_tool_catalog",
            {
                "name": definition.name,
                "subagent_session_id": subagent_session_id,
                "tool_names": sorted(filtered_tools),
                "tool_count": len(filtered_tools),
            },
        )

        final_text = ""
        final_text_source = "missing"
        usage_payload: dict[str, Any] = {}
        exit_code = 1
        usage_replayed = False

        def _try_replay_subagent_usage_once() -> None:
            nonlocal usage_replayed
            if usage_replayed:
                return
            usage_replayed = True
            try:
                _replay_subagent_usage(sub_session=sub_session)
            except Exception as e:  # noqa: BLE001
                store.append(
                    "warning",
                    {
                        "warning": "subagent_usage_replay_failed",
                        "name": definition.name,
                        "subagent_session_id": subagent_session_id,
                        "error": str(e),
                    },
                )

        try:
            exit_code = sub_session.run_turn(task)
            final_text, final_text_source = _resolve_subagent_final_text(
                sub_session=sub_session,
                subagent_surface=subagent_surface,
            )
            usage_payload = sub_session.usage_summary.totals()
        except Exception as e:  # noqa: BLE001
            usage_payload = sub_session.usage_summary.totals()
            _try_replay_subagent_usage_once()
            elapsed_ms = int((perf_counter() - subagent_started_at) * 1000)
            store.append(
                "subagent_end",
                {
                    "name": definition.name,
                    "subagent_session_id": subagent_session_id,
                    "status": "failed",
                    "exit_code": exit_code,
                    "error": f"Subagent execution failed: {e}",
                    "usage": usage_payload,
                    "elapsed_ms": elapsed_ms,
                    "steps_completed": subagent_surface.steps_completed,
                },
            )
            subagent_surface.on_subagent_end(
                SubagentEndEvent(
                    name=definition.name,
                    mode=resolved_mode,
                    status="failed",
                    elapsed_ms=elapsed_ms,
                    steps_completed=subagent_surface.steps_completed,
                    subagent_session_id=subagent_session_id,
                    error=f"Subagent execution failed: {e}",
                )
            )
            return {
                "error": f"Subagent '{definition.name}' execution failed: {e}",
                "subagent": definition.name,
                "subagent_session_id": subagent_session_id,
                "usage": usage_payload,
                "elapsed_ms": elapsed_ms,
                "steps_completed": subagent_surface.steps_completed,
            }
        finally:
            sub_session.close()

        if exit_code != 0:
            _try_replay_subagent_usage_once()
            elapsed_ms = int((perf_counter() - subagent_started_at) * 1000)
            error_payload = {
                "name": definition.name,
                "subagent_session_id": subagent_session_id,
                "status": "failed",
                "exit_code": exit_code,
                "usage": usage_payload,
                "elapsed_ms": elapsed_ms,
                "steps_completed": subagent_surface.steps_completed,
                "deadline_exhausted": (
                    execution_deadline.is_exhausted() if execution_deadline is not None else False
                ),
                "deadline_prevented_launch": False,
            }
            if final_text:
                error_payload["final_text"] = final_text
                error_payload["final_text_source"] = final_text_source
            subagent_surface.on_subagent_end(
                SubagentEndEvent(
                    name=definition.name,
                    mode=resolved_mode,
                    status="failed",
                    elapsed_ms=elapsed_ms,
                    steps_completed=subagent_surface.steps_completed,
                    subagent_session_id=subagent_session_id,
                    error=(final_text or f"Subagent '{definition.name}' failed."),
                )
            )
            store.append("subagent_end", error_payload)
            if crash_diagnostics is not None:
                crash_diagnostics.event(
                    "subagent_completed",
                    {
                        "subagent": definition.name,
                        "subagent_session_id": subagent_session_id,
                        "status": "failed",
                        "exit_code": exit_code,
                        "duration_ms": elapsed_ms,
                        "steps_completed": subagent_surface.steps_completed,
                        "deadline": (
                            execution_deadline.telemetry_snapshot()
                            if execution_deadline is not None
                            else None
                        ),
                    },
                )
            return {
                "error": f"Subagent '{definition.name}' failed.",
                "subagent": definition.name,
                "subagent_session_id": subagent_session_id,
                "exit_code": exit_code,
                "usage": usage_payload,
                "final_text": final_text,
                "final_text_source": final_text_source,
                "elapsed_ms": elapsed_ms,
                "steps_completed": subagent_surface.steps_completed,
                "deadline_exhausted": error_payload["deadline_exhausted"],
                "deadline_prevented_launch": False,
            }

        final_report_problem = _subagent_final_report_problem(
            text=final_text,
            source=final_text_source,
        )
        if final_report_problem is not None:
            _try_replay_subagent_usage_once()
            elapsed_ms = int((perf_counter() - subagent_started_at) * 1000)
            error_message = (
                f"Subagent '{definition.name}' did not produce a substantive final report "
                f"({final_report_problem})."
            )
            degraded_payload: dict[str, Any] = {
                "name": definition.name,
                "subagent_session_id": subagent_session_id,
                "status": "degraded",
                "failure_category": "final_report",
                "final_report_problem": final_report_problem,
                "final_text_source": final_text_source,
                "exit_code": exit_code,
                "usage": usage_payload,
                "elapsed_ms": elapsed_ms,
                "steps_completed": subagent_surface.steps_completed,
                "deadline_exhausted": (
                    execution_deadline.is_exhausted() if execution_deadline is not None else False
                ),
                "deadline_prevented_launch": False,
                "error": error_message,
            }
            if final_text:
                degraded_payload["final_text"] = final_text
            store.append("subagent_end", degraded_payload)
            subagent_surface.on_subagent_end(
                SubagentEndEvent(
                    name=definition.name,
                    mode=resolved_mode,
                    status="degraded",
                    elapsed_ms=elapsed_ms,
                    steps_completed=subagent_surface.steps_completed,
                    subagent_session_id=subagent_session_id,
                    error=error_message,
                )
            )
            if crash_diagnostics is not None:
                crash_diagnostics.event(
                    "subagent_completed",
                    {
                        "subagent": definition.name,
                        "subagent_session_id": subagent_session_id,
                        "status": "degraded",
                        "exit_code": exit_code,
                        "duration_ms": elapsed_ms,
                        "steps_completed": subagent_surface.steps_completed,
                        "final_report_problem": final_report_problem,
                        "deadline": (
                            execution_deadline.telemetry_snapshot()
                            if execution_deadline is not None
                            else None
                        ),
                    },
                )
            return {
                "error": error_message,
                "subagent": definition.name,
                "subagent_session_id": subagent_session_id,
                "status": "degraded",
                "failure_category": "final_report",
                "final_report_problem": final_report_problem,
                "usage": usage_payload,
                "final_text": final_text,
                "final_text_source": final_text_source,
                "elapsed_ms": elapsed_ms,
                "steps_completed": subagent_surface.steps_completed,
                "deadline_exhausted": degraded_payload["deadline_exhausted"],
                "deadline_prevented_launch": False,
                "sandbox": {
                    "mode": resolved_mode,
                    "tools": list(filtered_tools.keys()),
                },
            }

        _try_replay_subagent_usage_once()
        elapsed_ms = int((perf_counter() - subagent_started_at) * 1000)

        store.append(
            "subagent_end",
            {
                "name": definition.name,
                "subagent_session_id": subagent_session_id,
                "status": "success",
                "exit_code": exit_code,
                "usage": usage_payload,
                "elapsed_ms": elapsed_ms,
                "steps_completed": subagent_surface.steps_completed,
                "final_text_source": final_text_source,
                "deadline_exhausted": (
                    execution_deadline.is_exhausted() if execution_deadline is not None else False
                ),
                "deadline_prevented_launch": False,
            },
        )
        subagent_surface.on_subagent_end(
            SubagentEndEvent(
                name=definition.name,
                mode=resolved_mode,
                status="success",
                elapsed_ms=elapsed_ms,
                steps_completed=subagent_surface.steps_completed,
                subagent_session_id=subagent_session_id,
            )
        )
        if crash_diagnostics is not None:
            crash_diagnostics.event(
                "subagent_completed",
                {
                    "subagent": definition.name,
                    "subagent_session_id": subagent_session_id,
                    "status": "success",
                    "exit_code": exit_code,
                    "duration_ms": elapsed_ms,
                    "steps_completed": subagent_surface.steps_completed,
                    "deadline": (
                        execution_deadline.telemetry_snapshot()
                        if execution_deadline is not None
                        else None
                    ),
                },
            )
        return {
            "subagent": definition.name,
            "subagent_session_id": subagent_session_id,
            "result": final_text,
            "result_source": final_text_source,
            "usage": usage_payload,
            "elapsed_ms": elapsed_ms,
            "steps_completed": subagent_surface.steps_completed,
            "deadline_exhausted": (
                execution_deadline.is_exhausted() if execution_deadline is not None else False
            ),
            "deadline_prevented_launch": False,
            "sandbox": {
                "mode": resolved_mode,
                "tools": list(filtered_tools.keys()),
            },
        }

    tools: list[ToolDef] = []

    def _default_active_workdir_relpath() -> str:
        return (
            _normalize_workspace_relpath(get_active_workdir_relpath())
            if callable(get_active_workdir_relpath)
            else "."
        )

    def _normalize_tool_path_base(
        raw_value: Any,
        *,
        field_name: str,
        default: str = "active_workdir",
    ) -> str:
        if raw_value is None:
            return default
        text = str(raw_value).strip().lower()
        if not text:
            return default
        if text in {"active_workdir", "workspace_root"}:
            return text
        raise AgentRuntimeError(
            f"Invalid {field_name}: {raw_value!r}. Expected 'active_workdir' or 'workspace_root'."
        )

    def _resolve_workspace_relative_path(
        *,
        raw_path: Any,
        raw_base: Any = None,
        field_name: str,
        base_field_name: str,
        allow_empty: bool = False,
    ) -> str:
        workspace_root = root.resolve()
        base_kind = _normalize_tool_path_base(raw_base, field_name=base_field_name)
        if base_kind == "workspace_root":
            base_path = workspace_root
        else:
            base_path = resolve_workdir_relpath_within_workspace(
                workspace_root=workspace_root,
                relpath=_default_active_workdir_relpath(),
            )

        text = "" if raw_path is None else str(raw_path).strip()
        if not text:
            if allow_empty:
                return _workspace_relpath_for_path(workspace_root=workspace_root, path=base_path)
            raise AgentRuntimeError(f"Missing required argument: {field_name}")

        requested = Path(text)
        candidate = (
            requested.resolve() if requested.is_absolute() else (base_path / requested).resolve()
        )
        try:
            candidate.relative_to(workspace_root)
        except ValueError as e:
            raise AgentRuntimeError(f"Path escapes root ({field_name}): {text}") from e
        rel_path = _workspace_relpath_for_path(workspace_root=workspace_root, path=candidate)
        if rel_path == "README" and not (workspace_root / "README").exists():
            if (workspace_root / "README.md").exists():
                return "README.md"
        if rel_path == "README.md" and not (workspace_root / "README.md").exists():
            if (workspace_root / "README").exists():
                return "README"
        return rel_path

    def _make_tool_def(
        name: str,
        *,
        run: Callable[[dict[str, Any]], dict[str, Any]],
        parameters: dict[str, Any] | None = None,
    ) -> ToolDef:
        metadata = require_builtin_tool_metadata(name)
        return ToolDef(
            name=metadata.name,
            description=metadata.description,
            parameters=parameters if parameters is not None else copied_tool_parameters(name),
            run=run,
            metadata={
                "tool_type": "builtin",
                "compact_parameters_for_model": True,
                "model_description": _BUILTIN_MODEL_DESCRIPTIONS.get(
                    metadata.name, metadata.description
                ),
            },
        )

    def _custom_tool_requires_approval(spec: Any) -> bool:
        if mode == "review":
            return True
        return False

    def _run_custom_tool(spec: Any, args: dict[str, Any]) -> dict[str, Any]:
        if mode == "readonly":
            raise AgentRuntimeError(f"Blocked in readonly mode: custom tool '{spec.name}'")
        args_preview = json.dumps(args, ensure_ascii=True, indent=2, sort_keys=True)
        preview = (
            f"Run custom tool\n"
            f"name: {spec.name}\n"
            f"scope: {spec.source_scope}\n"
            f"path: {spec.source_path}\n"
            f"{_custom_tool_capability_summary(spec)}\n"
            f"args:\n{args_preview}"
        )
        if _custom_tool_requires_approval(spec):
            if non_interactive and not host_managed_approvals:
                raise AgentRuntimeError(
                    "Confirmation required for custom tool execution. Re-run with --yes or adjust plan."
                )
            decision = surface.request_approval(
                ApprovalRequest(
                    kind=f"custom_tool_run:{spec.name}",
                    reason="review mode requires confirmation for custom tools",
                    preview=preview,
                    files=[spec.relative_tool_path],
                    command=spec.name,
                    metadata={"custom_tool": spec.metadata(include_output_schema=True)},
                )
            )
            if not decision.allow:
                raise ApprovalDeclinedError(
                    f"custom tool '{spec.name}'",
                    message=f"User declined: custom tool '{spec.name}'",
                )
        artifact_dir: Path | None = None
        artifact_reference_prefix: str | None = None
        if store.artifact_persistence_enabled:
            artifact_dir = store.runtime_artifact_path("tool_logs")
            artifact_reference_prefix = store.session_artifact_layout.artifact_locator("tool_logs")
        return run_custom_tool(
            spec=spec,
            args=args,
            workspace_root=root,
            session_id=store.session_id,
            artifact_dir=artifact_dir,
            artifact_reference_prefix=artifact_reference_prefix,
        )

    def _append_builtin_tool(
        name: str,
        *,
        run: Callable[[dict[str, Any]], dict[str, Any]],
        parameters: dict[str, Any] | None = None,
    ) -> None:
        if not _built_in_tool_exposed_in_mode(
            tool_name=name,
            mode=mode,
            subagent_depth=subagent_depth,
        ):
            return
        tools.append(_make_tool_def(name, run=run, parameters=parameters))

    _append_builtin_tool(
        "fs_read",
        run=lambda args: _patchable("fs_read", fs_read)(
            root=root,
            path=_resolve_workspace_relative_path(
                raw_path=args.get("path"),
                raw_base=args.get("path_base"),
                field_name="path",
                base_field_name="path_base",
            ),
            max_bytes=int(args.get("max_bytes") or 20000),
        ),
    )

    _append_builtin_tool(
        "fs_read_lines",
        run=lambda args: _patchable("fs_read_lines", fs_read_lines)(
            root=root,
            path=_resolve_workspace_relative_path(
                raw_path=args.get("path"),
                raw_base=args.get("path_base"),
                field_name="path",
                base_field_name="path_base",
            ),
            start_line=int(args["start_line"]) if args.get("start_line") is not None else 0,
            end_line=(int(args["end_line"]) if args.get("end_line") is not None else None),
            max_lines=(int(args["max_lines"]) if args.get("max_lines") is not None else 200),
            include_line_numbers=bool(args.get("include_line_numbers", True)),
        ),
    )

    def _fs_edit(args: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_workspace_relative_path(
            raw_path=args.get("path"),
            raw_base=args.get("path_base"),
            field_name="path",
            base_field_name="path_base",
        )
        _guard_write_path(path)
        raw_edits = args.get("edits")
        if not isinstance(raw_edits, list):
            raise FsError("edits must be a non-empty array of edit objects")
        prepared = prepare_fs_edit(root=root, path=path, edits=raw_edits)
        diff = _unified_diff(prepared.original_content, prepared.updated_content, path)
        store.append("diff_preview", {"path": path, "diff": diff[:20000]})
        surface.on_patch_generated(
            PatchEvent(
                files=[path],
                diff=diff,
                summary=f"1 file changed via fs_edit ({path})",
            )
        )
        guard_write("fs_edit", diff[:20000] or f"(no diff) {path}", files=[path])
        return write_prepared_fs_edit(prepared)

    _append_builtin_tool("fs_edit", run=_fs_edit)

    def _fs_move(args: dict[str, Any]) -> dict[str, Any]:
        source_path = _resolve_workspace_relative_path(
            raw_path=args.get("source_path"),
            raw_base=args.get("source_path_base"),
            field_name="source_path",
            base_field_name="source_path_base",
        )
        destination_path = _resolve_workspace_relative_path(
            raw_path=args.get("destination_path"),
            raw_base=args.get("destination_path_base"),
            field_name="destination_path",
            base_field_name="destination_path_base",
        )
        _guard_write_path(source_path)
        _guard_write_path(destination_path)
        overwrite = bool(args.get("overwrite", False))
        preview = (
            "Move file\n"
            f"source: {source_path}\n"
            f"destination: {destination_path}\n"
            f"overwrite: {str(overwrite).lower()}"
        )
        guard_write("fs_move", preview, files=[source_path, destination_path])
        return fs_move(
            root=root,
            source_path=source_path,
            destination_path=destination_path,
            overwrite=overwrite,
        )

    _append_builtin_tool("fs_move", run=_fs_move)

    def _fs_copy(args: dict[str, Any]) -> dict[str, Any]:
        source_path = _resolve_workspace_relative_path(
            raw_path=args.get("source_path"),
            raw_base=args.get("source_path_base"),
            field_name="source_path",
            base_field_name="source_path_base",
        )
        destination_path = _resolve_workspace_relative_path(
            raw_path=args.get("destination_path"),
            raw_base=args.get("destination_path_base"),
            field_name="destination_path",
            base_field_name="destination_path_base",
        )
        _guard_write_path(destination_path)
        overwrite = bool(args.get("overwrite", False))
        preview = (
            "Copy file\n"
            f"source: {source_path}\n"
            f"destination: {destination_path}\n"
            f"overwrite: {str(overwrite).lower()}"
        )
        guard_write("fs_copy", preview, files=[source_path, destination_path])
        return fs_copy(
            root=root,
            source_path=source_path,
            destination_path=destination_path,
            overwrite=overwrite,
        )

    _append_builtin_tool("fs_copy", run=_fs_copy)

    def _fs_delete(args: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_workspace_relative_path(
            raw_path=args.get("path"),
            raw_base=args.get("path_base"),
            field_name="path",
            base_field_name="path_base",
        )
        try:
            _guard_write_path(path)
        except AgentRuntimeError as exc:
            if "outside allowed scope" not in str(exc) or not is_non_material_untracked_path(path):
                raise
        preview = f"Delete file\npath: {path}"
        guard_write("fs_delete", preview, files=[path])
        return fs_delete(root=root, path=path)

    _append_builtin_tool("fs_delete", run=_fs_delete)

    def _fs_write(args: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_workspace_relative_path(
            raw_path=args.get("path"),
            raw_base=args.get("path_base"),
            field_name="path",
            base_field_name="path_base",
        )
        _guard_write_path(path)
        content = str(args.get("content", ""))
        try:
            old = _patchable("fs_read", fs_read)(root=root, path=path, max_bytes=2_000_000)[
                "content"
            ]
        except (FsError, OSError):
            old = ""
        diff = _unified_diff(old, content, path)
        store.append("diff_preview", {"path": path, "diff": diff[:20000]})
        surface.on_patch_generated(
            PatchEvent(
                files=[path],
                diff=diff,
                summary=f"1 file changed via fs_write ({path})",
            )
        )
        guard_write("fs_write", diff[:20000] or f"(no diff) {path}", files=[path])
        return fs_write(root=root, path=path, content=content)

    _append_builtin_tool("fs_write", run=_fs_write)

    def _fs_mkdir(args: dict[str, Any]) -> dict[str, Any]:
        path = _resolve_workspace_relative_path(
            raw_path=args.get("path"),
            raw_base=args.get("path_base"),
            field_name="path",
            base_field_name="path_base",
        )
        if _is_allowed_ancestor_dir_creation(path):
            if _is_denied_path(path):
                raise AgentRuntimeError(f"Blocked write to protected path: {path}")
        else:
            _guard_write_path(path)
        parents = bool(args.get("parents", True))
        exist_ok = bool(args.get("exist_ok", True))
        preview = (
            "Create directory\n"
            f"path: {path}\n"
            f"parents: {str(parents).lower()}\n"
            f"exist_ok: {str(exist_ok).lower()}"
        )
        guard_write("fs_mkdir", preview, files=[path])
        return fs_mkdir(
            root=root,
            path=path,
            parents=parents,
            exist_ok=exist_ok,
        )

    _append_builtin_tool("fs_mkdir", run=_fs_mkdir)

    _append_builtin_tool(
        "fs_list",
        run=lambda args: _patchable("fs_list", fs_list)(
            root=root,
            root_path=_resolve_workspace_relative_path(
                raw_path=args.get("root_path"),
                raw_base=args.get("path_base"),
                field_name="root_path",
                base_field_name="path_base",
                allow_empty=True,
            ),
            globs=args.get("globs"),
            ignore=args.get("ignore"),
        ),
    )

    web_search_exposed_in_mode = (
        _built_in_tool_exposed_in_mode(
            tool_name="web_search",
            mode=mode,
            subagent_depth=subagent_depth,
        )
        and resolve_web_search_policy(cfg) != "off"
    )
    web_search_status = (
        resolve_web_search_runtime_status(cfg=cfg, api_key=api_key)
        if web_search_exposed_in_mode
        else None
    )

    def _web_fetch_recovery_is_public_candidate(raw_url: str) -> bool:
        normalized = normalize_web_url(raw_url)
        if normalized is None:
            return False
        try:
            split = urlsplit(normalized)
        except ValueError:
            return False
        host = (split.hostname or "").rstrip(".").lower()
        if not host or host == "localhost" or host.endswith(".localhost"):
            return False
        if split.username is not None or split.password is not None:
            return False
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return True
        return not (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_unspecified
            or ip.is_reserved
        )

    def _web_fetch_source_matches_requested(*, source_url: str, requested_url: str) -> bool:
        source = normalize_web_url(source_url)
        requested = normalize_web_url(requested_url)
        if source is None or requested is None:
            return False
        if source == requested:
            return True
        source_split = urlsplit(source)
        requested_split = urlsplit(requested)
        if (
            source_split.scheme,
            source_split.netloc,
            source_split.query,
        ) != (
            requested_split.scheme,
            requested_split.netloc,
            requested_split.query,
        ):
            return False
        return source_split.path.rstrip("/") == requested_split.path.rstrip("/")

    def _web_fetch_recovery_display_url(raw_url: str) -> str:
        normalized = normalize_web_url(raw_url)
        if normalized is not None:
            return normalized
        canonical = canonicalize_web_url_input(raw_url)
        if canonical is not None:
            return canonical
        try:
            split = urlsplit(str(raw_url or "").strip())
            port = split.port
        except ValueError:
            return "[invalid URL omitted]"
        scheme = str(split.scheme or "").lower()
        host = (split.hostname or "").rstrip(".").lower()
        if scheme not in {"http", "https"} or not host:
            return "[unsupported URL omitted]"
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            port = None
        netloc = host if port is None else f"{host}:{port}"
        path = split.path or "/"
        return urlunsplit((scheme, netloc, path, split.query, ""))

    def _web_fetch_recovery_payload(
        *,
        requested_url: str,
        raw_requested_url: str,
        finalization_suppressed: bool,
        automatic_attempted: bool = False,
        search_error: str = "",
    ) -> dict[str, Any]:
        display_url = _web_fetch_recovery_display_url(requested_url)
        query = build_web_fetch_recovery_search_query(display_url)
        payload: dict[str, Any] = {
            "error": (
                "web_fetch only allows a URL explicitly provided by the user or one returned "
                "by web_search earlier in this session."
            ),
            "error_code": "web_fetch_provenance_required",
            "url": display_url,
            "allowed_provenance": [
                "user_provided",
                "returned_by_web_search",
                "fetched_page_link",
                "trusted_local_file",
                "trusted_tool_output",
                "canonical_redirect",
                "search_mediated_recovery",
                "same_origin_derived_search_result",
            ],
            "provenance_recovery": {
                "suggested_search_query": query,
                "web_search_available": bool(
                    web_search_status is not None and web_search_status.registration_ready
                ),
                "automatic_recovery_attempted": automatic_attempted,
                "finalization_suppressed": finalization_suppressed,
                "search_error": search_error,
            },
        }
        canonical_raw = normalize_web_url(raw_requested_url)
        if canonical_raw is not None and canonical_raw != display_url:
            payload["raw_input_url"] = raw_requested_url
        return payload

    def _maybe_establish_web_fetch_provenance_via_search(
        *,
        requested_url: str,
        raw_requested_url: str,
    ) -> tuple[str | None, str | None, dict[str, Any] | None]:
        finalization_suppressed = (
            execution_deadline is not None
            and execution_deadline.phase() == DeadlinePhase.FINALIZATION_WINDOW
        )
        base_payload = _web_fetch_recovery_payload(
            requested_url=requested_url,
            raw_requested_url=raw_requested_url,
            finalization_suppressed=finalization_suppressed,
        )
        if finalization_suppressed:
            return None, None, base_payload
        if web_search_status is None or not web_search_status.registration_ready:
            return None, None, base_payload
        if not _web_fetch_recovery_is_public_candidate(requested_url):
            return None, None, base_payload
        query = str(base_payload["provenance_recovery"]["suggested_search_query"] or "").strip()
        if not query:
            return None, None, base_payload
        try:
            host = (
                urlsplit(normalize_web_url(requested_url) or requested_url).hostname or ""
            ).lower()
            search_result = web_search(
                query=query,
                cfg=cfg,
                api_key=api_key,
                allowed_domains=[host] if host else None,
                max_sources=5,
                external_web_access=True,
                session_id=str(getattr(store, "session_id", "") or "") or None,
            )
        except WebSearchError as exc:
            return (
                None,
                None,
                _web_fetch_recovery_payload(
                    requested_url=requested_url,
                    raw_requested_url=raw_requested_url,
                    finalization_suppressed=finalization_suppressed,
                    automatic_attempted=True,
                    search_error=str(exc),
                ),
            )
        matching_source_url = ""
        for source in list(search_result.get("sources") or []):
            if not isinstance(source, dict):
                continue
            source_url = str(source.get("url") or "").strip()
            if _web_fetch_source_matches_requested(
                source_url=source_url,
                requested_url=requested_url,
            ):
                matching_source_url = source_url
                break
        if not matching_source_url:
            payload = _web_fetch_recovery_payload(
                requested_url=requested_url,
                raw_requested_url=raw_requested_url,
                finalization_suppressed=finalization_suppressed,
                automatic_attempted=True,
            )
            payload["provenance_recovery"]["search_result_source_count"] = len(
                list(search_result.get("sources") or [])
            )
            return None, None, payload
        _changed, normalized = store.establish_search_mediated_web_fetch_url(
            raw_url=requested_url,
            query=query,
            source_url=matching_source_url,
        )
        store.append(
            "web_fetch_provenance_recovery",
            {
                "url": requested_url,
                "normalized_url": normalized,
                "query": query,
                "source_url": matching_source_url,
                "provenance_classification": "search_mediated_recovery",
            },
        )
        return store.resolve_web_fetch_url(requested_url)[0], normalized, None

    def _web_fetch_tool(args: dict[str, Any]) -> dict[str, Any]:
        raw_requested_url = str(args.get("url", "")).strip()
        provenance_classification, resolved_requested_url = store.resolve_web_fetch_url(
            raw_requested_url
        )
        requested_url = resolved_requested_url or raw_requested_url
        recovered_via_search = False
        if provenance_classification is None:
            (
                provenance_classification,
                recovered_url,
                recovery_result,
            ) = _maybe_establish_web_fetch_provenance_via_search(
                requested_url=requested_url,
                raw_requested_url=raw_requested_url,
            )
            if provenance_classification is None:
                rejection = recovery_result or _web_fetch_recovery_payload(
                    requested_url=requested_url,
                    raw_requested_url=raw_requested_url,
                    finalization_suppressed=False,
                )
                fetchable_urls = store.fetchable_web_fetch_urls()
                if fetchable_urls:
                    rejection["fetchable_urls"] = fetchable_urls
                    rejection["guidance"] = (
                        "Do not guess or restate URLs from memory. Retry web_fetch with one of "
                        "fetchable_urls (these came from prior trusted session evidence), or run "
                        "web_search again to find the page."
                    )
                else:
                    rejection["guidance"] = (
                        "No URLs are fetchable yet. Run web_search first, or ask the user for "
                        "the exact URL. Do not guess URLs."
                    )
                return rejection
            recovered_via_search = True
            if recovered_url:
                requested_url = recovered_url
        result = _patchable("web_fetch", web_fetch)(
            url=requested_url,
            max_chars=(args["max_chars"] if "max_chars" in args else 20000),
        )
        if recovered_via_search:
            result["provenance_classification"] = provenance_classification
        return result

    _append_builtin_tool("web_fetch", run=_web_fetch_tool)

    if web_search_exposed_in_mode and web_search_status is not None:
        if (
            emit_web_search_runtime_diagnostics
            and web_search_status.mode == "auto"
            and not web_search_status.registration_ready
        ):
            store.append("web_search_runtime_unavailable", web_search_status.to_payload())

        if web_search_status.registration_ready:
            _append_builtin_tool(
                "web_search",
                run=lambda args: web_search(
                    query=str(args.get("query", "")),
                    cfg=cfg,
                    api_key=api_key,
                    allowed_domains=args.get("allowed_domains"),
                    max_sources=(args["max_sources"] if "max_sources" in args else 8),
                    external_web_access=(
                        args["external_web_access"] if "external_web_access" in args else True
                    ),
                    session_id=str(getattr(store, "session_id", "") or "") or None,
                ),
            )

    _append_builtin_tool(
        "symbol_search",
        run=lambda args: _patchable("symbol_search", symbol_search)(
            root=root,
            query=str(args.get("query", "")),
            kind=str(args["kind"]) if args.get("kind") is not None else None,
            root_path=_resolve_workspace_relative_path(
                raw_path=args.get("root_path"),
                raw_base=args.get("path_base"),
                field_name="root_path",
                base_field_name="path_base",
                allow_empty=True,
            ),
            globs=args.get("globs"),
            max_results=(int(args["max_results"]) if args.get("max_results") is not None else 100),
            exact=bool(args.get("exact", False)),
        ),
    )

    _append_builtin_tool(
        "test_discover",
        run=lambda args: _patchable("test_discover", test_discover)(
            root=root,
            paths=args.get("paths"),
            symbols=args.get("symbols"),
            changed_only=bool(args.get("changed_only", False)),
            include_commands=bool(args.get("include_commands", True)),
            max_results=(int(args["max_results"]) if args.get("max_results") is not None else 20),
            failure_summary=(
                args.get("failure_summary")
                if isinstance(args.get("failure_summary"), dict)
                else None
            ),
        ),
    )

    _append_builtin_tool(
        "repo_map",
        run=lambda args: _patchable("repo_map", repo_map)(
            root=root,
            paths=args.get("paths"),
            symbols=args.get("symbols"),
            include_tests=bool(args.get("include_tests", True)),
            include_imports=bool(args.get("include_imports", True)),
            include_references=bool(args.get("include_references", False)),
            depth=(int(args["depth"]) if args.get("depth") is not None else 2),
            max_items=(int(args["max_items"]) if args.get("max_items") is not None else 80),
        ),
    )

    _append_builtin_tool(
        "search_rg",
        run=lambda args: _patchable("search_rg", search_rg)(
            root=root,
            pattern=str(args.get("pattern", "")),
            root_path=_resolve_workspace_relative_path(
                raw_path=args.get("root_path"),
                raw_base=args.get("path_base"),
                field_name="root_path",
                base_field_name="path_base",
                allow_empty=True,
            ),
            globs=args.get("globs"),
        ),
    )

    if history_artifact_persistence_available:
        _append_builtin_tool(
            "history_search",
            run=lambda args: history_search(
                root=root,
                session_id=store.session_id,
                session_artifact_root=store.session_artifact_root,
                pattern=str(args.get("pattern", "")),
                max_results=int(args.get("max_results") or 50),
                max_file_bytes=int(args.get("max_file_bytes") or 200000),
                include_history=bool(args.get("include_history", True)),
                include_tool_outputs=bool(args.get("include_tool_outputs", True)),
                include_memory=bool(args.get("include_memory", True)),
            ),
        )

    if skills_enabled and resolved_skill_registry:

        def _skill_read(args: dict[str, Any]) -> dict[str, Any]:
            raw_name = str(args.get("name", "")).strip()
            if not raw_name:
                return {"error": "Missing required argument: name"}
            skill = resolve_skill_by_name(resolved_skill_registry, raw_name)
            if skill is None:
                return {
                    "error": f"Unknown skill: {raw_name}",
                    "available_skills": sorted(
                        skill.name for skill in resolved_skill_registry.values()
                    ),
                }
            try:
                return read_skill_bundle_file(
                    skill,
                    path=(str(args["path"]) if args.get("path") is not None else None),
                )
            except SkillReadError as exc:
                return {
                    "error": str(exc),
                    "name": skill.name,
                    "source_path": skill.source_path.as_posix(),
                }

        _append_builtin_tool("skill_read", run=_skill_read)

    verify_artifact_counter = 0

    def _next_verify_artifact_path() -> Path:
        nonlocal verify_artifact_counter
        verify_artifact_counter += 1
        return store.runtime_artifact_path(
            "verify",
            f"step{verify_artifact_counter:03d}_verify_run.txt",
        )

    def _verify_run(args: dict[str, Any]) -> dict[str, Any]:
        deadline_decision = _deadline_start_decision(
            DeadlineOperation.VERIFICATION,
            minimum_remaining_seconds=MINIMUM_TOOL_START_SECONDS,
            allow_during_finalization=True,
        )
        if deadline_decision is not None and not bool(deadline_decision.get("allowed")):
            return _deadline_error(
                "verify_run skipped because the run deadline is exhausted or too close.",
                start_decision=deadline_decision,
            )
        try:
            verify_timeout_s = _deadline_timeout(900, operation="verify_run")
        except DeadlineExhausted:
            return _deadline_error(
                "verify_run skipped because the run deadline is exhausted or too close."
            )
        effective_cfg = cfg or AppConfig(model="")
        raw_commands = args.get("commands")
        verify_cmd: list[str] | None = None
        current_selection = _current_verify_selection()
        current_effective_verification_commands = _normalized_verify_commands(
            list(current_selection.commands) if current_selection is not None else []
        )
        unavailable_verification_contract = bool(
            current_selection is not None
            and str(current_selection.contract_type or "").strip() == "unavailable"
            and not current_effective_verification_commands
        )
        ignore_explicit_commands_for_unavailable_contract = (
            unavailable_verification_contract
            and authoritative_verify_commands is None
            and (
                (not one_shot_execution and resolved_runtime_kind == RuntimeKind.INTERACTIVE_CHAT)
                or not _normalized_verify_commands(
                    getattr(effective_cfg, "verify_commands", []) or []
                )
            )
        )
        trusted_shell_commands = trusted_shell_expression_command_set(current_selection)

        def _validate_explicit_verify_candidate(command: str) -> None:
            normalized_exact = " ".join(str(command or "").strip().split())
            trusted = normalized_exact in trusted_shell_commands
            analysis = analyze_verification_command(
                command,
                trusted=trusted,
                workspace_root=root,
            )
            if analysis.rejection_reason:
                raise VerifyError("verification command is invalid: " + analysis.rejection_reason)
            if _has_disallowed_shell_control_flow(command) and not trusted:
                raise VerifyError("verification command is invalid: disallowed_shell_control_flow")

        selection_metadata = verification_selection_payload(
            current_selection
            if current_selection is not None
            else ResolvedVerifyCommands(
                commands=tuple(current_effective_verification_commands),
                source="session.effective_verification_commands",
                reason="session already resolved an effective verification contract",
                contract_type="selected",
            ),
            authoritative=(
                is_authoritative_verify_command_selection(current_selection)
                if current_selection is not None
                else bool(authoritative_verify_commands is not None)
            ),
        )
        selection_metadata.update(verification_command_specs_payload(current_selection))
        if raw_commands is not None:
            if not isinstance(raw_commands, list):
                raise VerifyError("commands must be an array of command strings.")
            verify_cmd = []
            for item in raw_commands:
                text = str(item).strip()
                if not text:
                    raise VerifyError("commands cannot contain empty values.")
                expanded_commands = _expand_simple_verify_command_chain(
                    text,
                    workspace_root=root,
                )
                if not ignore_explicit_commands_for_unavailable_contract:
                    if len(expanded_commands) == 1 and expanded_commands[0] == text:
                        _validate_explicit_verify_candidate(text)
                    else:
                        for command in expanded_commands:
                            _validate_explicit_verify_candidate(command)
                verify_cmd.extend(expanded_commands)
            if not verify_cmd:
                raise VerifyError("commands cannot be empty.")

        ignored_model_verification_commands: list[str] = []
        if authoritative_verify_commands is not None:
            if verify_cmd is not None:
                requested_commands = _normalized_verify_commands(verify_cmd)
                if requested_commands != authoritative_verify_commands:
                    raise VerifyError(
                        "Managed verification commands are locked to the authoritative Forge command set."
                    )
            commands = list(authoritative_verify_commands)
        elif verify_cmd is not None and current_effective_verification_commands:
            requested_commands = _normalized_verify_commands(verify_cmd)
            incompatible_commands = _verify_run_commands_match_effective_contract(
                requested_commands=requested_commands,
                effective_verification_commands=current_effective_verification_commands,
            )
            if incompatible_commands:
                raise VerifyError(
                    "verify_run commands must stay within the session's effective verification contract."
                )
            commands = requested_commands
        elif verify_cmd is not None and unavailable_verification_contract:
            requested_commands = _normalized_verify_commands(verify_cmd)
            commands = []
            for command in requested_commands:
                analysis = analyze_verification_command(command, trusted=False, workspace_root=root)
                if (
                    analysis.evidentiary_capability
                    == VerificationCommandEvidentiaryCapability.ASSERTIVE
                    and not analysis.rejection_reason
                ):
                    commands.append(command)
                else:
                    ignored_model_verification_commands.append(command)
        elif verify_cmd is not None and current_selection is not None:
            commands = _normalized_verify_commands(verify_cmd)
        elif verify_cmd is None and current_selection is not None:
            commands = list(current_effective_verification_commands)
        else:
            commands = resolve_verify_commands(
                cfg=effective_cfg,
                verify_cmd=verify_cmd,
            )
        validation_errors = validation_errors_for_selection(current_selection)
        if validation_errors and (
            authoritative_verify_commands is not None
            or (
                current_selection is not None
                and is_authoritative_verify_command_selection(current_selection)
            )
        ):
            raise VerifyError(
                "authoritative verification command is invalid: " + "; ".join(validation_errors[:3])
            )
        for command in commands:
            normalized_exact = " ".join(str(command or "").strip().split())
            analysis = analyze_verification_command(
                command,
                trusted=normalized_exact in trusted_shell_commands,
                workspace_root=root,
            )
            if analysis.rejection_reason:
                raise VerifyError("verification command is invalid: " + analysis.rejection_reason)
            if (
                _has_disallowed_shell_control_flow(command)
                and normalized_exact not in trusted_shell_commands
            ):
                raise VerifyError(
                    "verification commands must be single commands without shell control flow or chaining."
                )
        guard_verify(commands)
        artifact_path = _next_verify_artifact_path()
        result, touched_repo_paths = _run_with_command_mutation_detection(
            root=root,
            enabled=command_mutation_tracking_enabled,
            ignored_paths=command_mutation_ignored_paths,
            operation=lambda: _call_with_optional_kwargs(
                _patchable("run_task_verification", run_task_verification),
                required_kwargs={
                    "root": root,
                    "commands": commands,
                    "artifact_path": artifact_path,
                    "cfg": effective_cfg,
                },
                optional_kwargs={"timeout_s": verify_timeout_s},
            ),
        )
        payload = verify_run_result_to_payload(root=root, result=result)
        if ignored_model_verification_commands:
            payload["ignored_model_verification_commands"] = ignored_model_verification_commands
            payload["verification_skip_reason"] = "verification_contract_unavailable"
        material_touched_repo_paths: list[str] = []
        if touched_repo_paths:
            payload = dict(payload)
            payload["touched_repo_paths"] = touched_repo_paths
            mutation_metadata = _command_mutation_metadata(
                root=root,
                touched_repo_paths=touched_repo_paths,
                command_was_verification=True,
            )
            payload.update(mutation_metadata)
            material_touched_repo_paths = list(
                mutation_metadata.get("material_touched_repo_paths") or []
            )
        verification_relevant_material_touched_paths = _verification_relevant_material_paths(
            material_touched_repo_paths
        )
        evidence_records: list[VerificationEvidence] = []
        command_results = payload.get("command_results")
        if isinstance(command_results, list):
            for item in command_results:
                if not isinstance(item, dict):
                    continue
                command = str(item.get("command") or item.get("effective_command") or "")
                if not command:
                    continue
                exit_code_raw = item.get("exit_code")
                evidence_records.append(
                    classify_verification_evidence(
                        command,
                        known_verification_commands=current_effective_verification_commands,
                        authoritative=bool(selection_metadata.get("verification_authoritative")),
                        material_touched_paths=verification_relevant_material_touched_paths,
                        exit_code=(exit_code_raw if isinstance(exit_code_raw, int) else None),
                        output=str(item.get("output_preview") or ""),
                        real_execution=(
                            item.get("real_execution")
                            if isinstance(item.get("real_execution"), bool)
                            or item.get("real_execution") is None
                            else None
                        ),
                        root=root,
                    )
                )
        payload.update(_aggregate_tool_evidence_payload(evidence_records))
        payload.update(selection_metadata)
        stored_artifact_path = (
            os.fspath(result.artifact_path.resolve())
            if result.artifact_path.exists()
            else os.fspath(result.artifact_path)
        )
        store.append(
            "verify_run",
            {
                "commands": commands,
                "all_passed": result.all_passed,
                "summary": result.summary,
                "fallback_used": payload.get("fallback_used"),
                "fallback_count": payload.get("fallback_count"),
                "fallback_details": payload.get("fallback_details"),
                "artifact_path": stored_artifact_path,
                "model_artifact_path": payload.get("artifact_path"),
                "artifact_saved": payload.get("artifact_saved"),
                "artifact_readable_via_fs": payload.get("artifact_readable_via_fs"),
                "artifact_location": payload.get("artifact_location"),
                "verification_evidence_category": payload.get("verification_evidence_category"),
                "verification_evidence_reason": payload.get("verification_evidence_reason"),
                "verification_evidence_allowed": payload.get("verification_evidence_allowed"),
                "verification_evidence_supplemental_only": payload.get(
                    "verification_evidence_supplemental_only"
                ),
                "ignored_model_verification_commands": payload.get(
                    "ignored_model_verification_commands", []
                ),
                "verification_skip_reason": payload.get("verification_skip_reason"),
                "material_touched_repo_paths": payload.get("material_touched_repo_paths", []),
                "benign_runtime_paths": payload.get("benign_runtime_paths", []),
                **selection_metadata,
            },
        )
        return payload

    if verification_enabled:
        _append_builtin_tool("verify_run", run=_verify_run)

    def _shell(args: dict[str, Any]) -> dict[str, Any]:
        deadline_decision = _deadline_start_decision(
            DeadlineOperation.SHELL_TOOL,
            minimum_remaining_seconds=MINIMUM_TOOL_START_SECONDS,
            allow_during_finalization=True,
        )
        if deadline_decision is not None and not bool(deadline_decision.get("allowed")):
            return _deadline_error(
                "shell_run skipped because the run deadline is exhausted or too close.",
                start_decision=deadline_decision,
            )
        try:
            shell_timeout_s = _deadline_timeout(60, operation="shell_run")
        except DeadlineExhausted:
            return _deadline_error(
                "shell_run skipped because the run deadline is exhausted or too close."
            )
        cmd = str(args.get("cmd", ""))
        effective_cwd = _resolve_workspace_relative_path(
            raw_path=args.get("cwd"),
            raw_base=args.get("cwd_base"),
            field_name="cwd",
            base_field_name="cwd_base",
            allow_empty=True,
        )
        guard_shell(cmd)
        store.append("cmd", {"cmd": cmd, "cwd": effective_cwd})
        started = perf_counter()
        result: dict[str, Any] | None = None
        try:
            result, touched_repo_paths = _run_with_command_mutation_detection(
                root=root,
                enabled=command_mutation_tracking_enabled,
                ignored_paths=command_mutation_ignored_paths,
                operation=lambda: _call_with_optional_kwargs(
                    _patchable("shell_run", shell_run),
                    required_kwargs={
                        "root": root,
                        "cmd": cmd,
                        "cwd": effective_cwd,
                        "runner": shell_runner,
                    },
                    optional_kwargs={"timeout_s": shell_timeout_s},
                ),
            )
        finally:
            if is_full_access_mode:
                duration_ms = int((perf_counter() - started) * 1000)
                store.append(
                    "fullaccess_shell",
                    {
                        "event": "fullaccess_shell",
                        "ts": _fullaccess_shell_audit_ts(),
                        "command": cmd,
                        "cwd": str((result or {}).get("cwd") or effective_cwd or root),
                        "exit_code": int((result or {}).get("exit_code", -1)),
                        "duration_ms": duration_ms,
                        "mode": "fullaccess",
                    },
                )
        if touched_repo_paths:
            result = dict(result)
            result["touched_repo_paths"] = touched_repo_paths
            result.update(
                _command_mutation_metadata(
                    root=root,
                    touched_repo_paths=touched_repo_paths,
                    command_was_verification=False,
                )
            )
        current_selection = _current_verify_selection()
        current_effective_verification_commands = _normalized_verify_commands(
            list(current_selection.commands) if current_selection is not None else []
        )
        shell_exit_code = result.get("exit_code") if isinstance(result, dict) else None
        shell_evidence = classify_verification_evidence(
            str(result.get("effective_cmd") or result.get("cmd") or cmd),
            known_verification_commands=current_effective_verification_commands,
            authoritative=(
                is_authoritative_verify_command_selection(current_selection)
                if current_selection is not None
                else bool(authoritative_verify_commands is not None)
            ),
            material_touched_paths=_verification_relevant_material_paths(
                result.get("material_touched_repo_paths", [])
                if isinstance(result.get("material_touched_repo_paths"), list)
                else [],
            ),
            exit_code=(shell_exit_code if isinstance(shell_exit_code, int) else None),
            output="\n".join(
                [
                    str(result.get("stdout") or "").strip(),
                    str(result.get("stderr") or "").strip(),
                ]
            ).strip(),
            root=root,
        )
        result["verification_evidence_category"] = shell_evidence.category.value
        result["verification_evidence_reason"] = shell_evidence.reason
        result["verification_evidence_allowed"] = shell_evidence.allowed_to_satisfy_contract
        result["verification_evidence_supplemental_only"] = shell_evidence.supplemental_only
        return result

    _append_builtin_tool("shell_run", run=_shell)

    def _require_terminal_manager() -> TerminalManager:
        if terminal_manager is None:
            raise AgentRuntimeError("Background shell tools are unavailable in this session.")
        return terminal_manager

    def _require_durable_service_manager() -> DurableServiceManager:
        if durable_service_manager is None:
            raise AgentRuntimeError("Durable service tools are unavailable in this session.")
        return durable_service_manager

    def _service_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
        readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
        return {
            "service_id": payload.get("service_id"),
            "ownership": payload.get("ownership") or ProcessOwnership.DURABLE_SERVICE.value,
            "status": payload.get("status"),
            "alive": bool(payload.get("alive")),
            "backend": payload.get("backend"),
            "readiness": {
                "type": readiness.get("type"),
                "status": readiness.get("status"),
                "host": readiness.get("host"),
                "port": readiness.get("port"),
                "path": readiness.get("path"),
            },
            "failure_category": payload.get("failure_category"),
            "log_paths": payload.get("log_paths"),
            "preview_url": payload.get("preview_url"),
            "startup_error": payload.get("startup_error"),
        }

    def _guard_service_readiness_spec(raw_readiness: Any) -> dict[str, Any] | None:
        if raw_readiness is None:
            return None
        if not isinstance(raw_readiness, dict):
            raise AgentRuntimeError("readiness must be an object when provided")
        readiness = dict(raw_readiness)
        if str(readiness.get("type") or "").strip().lower() == "command":
            command = str(readiness.get("command") or "").strip()
            if not command:
                raise AgentRuntimeError("readiness.command is required for command readiness")
            guard_shell(command, tool_name="shell_service_start")
        return readiness

    def _format_bg_snapshot(
        *,
        process_id: str,
        snapshot: ProcessOutputSnapshot,
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        lines: list[dict[str, Any]] = []
        output_truncated_by_max_bytes = False
        remaining_bytes = max_bytes if max_bytes is not None else None
        for line in snapshot.lines:
            text = line.text
            if remaining_bytes is not None:
                encoded = text.encode("utf-8", errors="replace")
                if remaining_bytes <= 0:
                    output_truncated_by_max_bytes = True
                    break
                if len(encoded) > remaining_bytes:
                    text = encoded[:remaining_bytes].decode("utf-8", errors="replace")
                    output_truncated_by_max_bytes = True
                    remaining_bytes = 0
                else:
                    remaining_bytes -= len(encoded)
            lines.append({"seq": line.seq, "stream": line.stream, "text": text})
        payload = {
            "process_id": process_id,
            "lifetime": "session",
            "status": snapshot.status,
            "exit_code": snapshot.exit_code,
            "failure_reason": snapshot.failure_reason,
            "lines": lines,
            "next_seq": snapshot.next_seq,
            "dropped_lines": snapshot.dropped_lines,
            "runtime_s": round(snapshot.runtime_s, 3),
            "total_bytes": snapshot.total_bytes,
        }
        if output_truncated_by_max_bytes:
            payload["output_truncated_by_max_bytes"] = True
            payload["max_bytes"] = max_bytes
        return payload

    def _format_bg_summaries(manager: TerminalManager) -> list[dict[str, Any]]:
        return [
            {
                "process_id": summary.process_id,
                "cmd": summary.cmd,
                "cwd": str(summary.cwd),
                "status": summary.status,
                "exit_code": summary.exit_code,
                "runtime_s": round(summary.runtime_s, 3),
                "started_at_wall": summary.started_at_wall,
            }
            for summary in manager.list()
        ]

    def _unknown_bg_process_payload(
        *,
        manager: TerminalManager,
        process_id: str,
        operation: str,
        since: int | None = None,
    ) -> dict[str, Any]:
        known_processes = _format_bg_summaries(manager)
        payload: dict[str, Any] = {
            "status": "unknown_process_id",
            "unknown_process_id": True,
            "process_id": process_id,
            "requested_process_id": process_id,
            "operation": operation,
            "exit_code": None,
            "failure_reason": "No background process with that process_id is tracked in this session.",
            "lines": [],
            "next_seq": since if since is not None else 0,
            "dropped_lines": 0,
            "runtime_s": 0.0,
            "total_bytes": 0,
            "known_processes": known_processes,
            "known_process_ids": [process["process_id"] for process in known_processes],
            "recovery": {
                "recommended_tool": "shell_list",
                "suggested_arguments": {},
                "reason": (
                    "The supplied process_id is not tracked. Use shell_list or the process_id "
                    "returned by shell_background; do not use a tool_call_id as process_id."
                ),
            },
        }
        if since is not None:
            payload["since"] = since
        store.append(
            "bg_unknown_process",
            {
                "operation": operation,
                "process_id": process_id,
                "known_process_count": len(known_processes),
            },
        )
        return payload

    shell_empty_poll_counts: dict[tuple[str, int, int, str], int] = {}

    def _coerce_shell_since(raw_since: Any) -> int:
        try:
            since = int(raw_since) if raw_since is not None else 0
        except (TypeError, ValueError) as exc:
            raise AgentRuntimeError(f"Invalid since value: {raw_since!r}") from exc
        if since < 0:
            raise AgentRuntimeError("since must be non-negative")
        return since

    def _coerce_shell_wait_seconds(raw_wait: Any) -> float:
        try:
            wait_seconds = float(raw_wait) if raw_wait is not None else 5.0
        except (TypeError, ValueError) as exc:
            raise AgentRuntimeError(f"Invalid wait_seconds value: {raw_wait!r}") from exc
        if wait_seconds < 0:
            raise AgentRuntimeError("wait_seconds must be non-negative")
        return min(wait_seconds, 60.0)

    def _coerce_shell_max_bytes(raw_max_bytes: Any) -> int | None:
        if raw_max_bytes is None:
            return None
        try:
            max_bytes = int(raw_max_bytes)
        except (TypeError, ValueError) as exc:
            raise AgentRuntimeError(f"Invalid max_bytes value: {raw_max_bytes!r}") from exc
        if max_bytes <= 0:
            raise AgentRuntimeError("max_bytes must be positive")
        return max_bytes

    def _coerce_shell_wait_until(raw_until: Any) -> str:
        until = str(raw_until or "either").strip().lower()
        if until not in {"output_available", "process_exited", "either"}:
            raise AgentRuntimeError(
                "until must be one of output_available, process_exited, or either"
            )
        return until

    def _clamp_shell_wait_seconds(wait_seconds: float) -> tuple[float, dict[str, Any] | None]:
        if execution_deadline is None:
            return wait_seconds, None
        decision = execution_deadline.start_decision(
            DeadlineOperation.SHELL_TOOL,
            minimum_remaining_seconds=MINIMUM_TOOL_START_SECONDS,
            configured_timeout_seconds=wait_seconds,
            allow_during_finalization=True,
        ).telemetry_snapshot()
        if not bool(decision.get("allowed")):
            return 0.0, decision
        clamped = execution_deadline.clamp_timeout(
            wait_seconds,
            reserve_seconds=DEFAULT_DEADLINE_CLEANUP_RESERVE_SECONDS,
        )
        if clamped is None:
            return 0.0, decision
        if execution_deadline.phase() == DeadlinePhase.FINALIZATION_WINDOW:
            clamped = min(float(clamped), 1.0)
        return float(clamped), decision

    def _maybe_add_empty_poll_guidance(
        *,
        payload: dict[str, Any],
        process_id: str,
        since: int,
        snapshot: ProcessOutputSnapshot,
    ) -> None:
        if snapshot.lines or snapshot.status != "running":
            shell_empty_poll_counts.pop(
                (process_id, since, snapshot.next_seq, snapshot.status), None
            )
            return
        key = (process_id, since, snapshot.next_seq, snapshot.status)
        count = shell_empty_poll_counts.get(key, 0) + 1
        shell_empty_poll_counts[key] = count
        payload["empty_poll_count"] = count
        if count >= 2:
            payload["wait_guidance"] = {
                "recommended_tool": "shell_wait",
                "reason": "No new output or process status change was observed for repeated immediate polls.",
                "process_id": process_id,
                "since": since,
                "suggested_arguments": {
                    "process_id": process_id,
                    "since": since,
                    "until": "either",
                    "wait_seconds": 5,
                },
            }

    def _shell_background(args: dict[str, Any]) -> dict[str, Any]:
        deadline_decision = _deadline_start_decision(
            DeadlineOperation.SHELL_BACKGROUND,
            minimum_remaining_seconds=MINIMUM_TOOL_START_SECONDS,
        )
        deadline_warning = None
        if deadline_decision is not None and not bool(deadline_decision.get("allowed")):
            deadline_warning = _deadline_warning_fields(
                "Deadline policy would normally block background work; start proceeded because "
                "this is advisory and not safety.",
                start_decision=deadline_decision,
            )
        manager = _require_terminal_manager()
        cmd = str(args.get("cmd", ""))
        effective_cwd_relpath = _resolve_workspace_relative_path(
            raw_path=args.get("cwd"),
            raw_base=args.get("cwd_base"),
            field_name="cwd",
            base_field_name="cwd_base",
            allow_empty=True,
        )
        guard_shell(cmd, tool_name="shell_background")
        cwd_path = root if not effective_cwd_relpath else (root / effective_cwd_relpath).resolve()
        store.append("bg_start", {"cmd": cmd, "cwd": effective_cwd_relpath})
        started = perf_counter()
        snapshot: ProcessOutputSnapshot | None = None
        try:
            try:
                process_id = manager.start(
                    cmd=cmd,
                    cwd=cwd_path,
                    root=root,
                )
            except TerminalLimitError as exc:
                raise AgentRuntimeError(str(exc)) from exc
            except ValueError as exc:
                raise AgentRuntimeError(f"Invalid background process request: {exc}") from exc
            except (ConfigError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
                raise AgentRuntimeError(f"Failed to start background process: {exc}") from exc
            snapshot = manager.read(process_id)
            payload = _format_bg_snapshot(process_id=process_id, snapshot=snapshot)
            if deadline_warning is not None:
                payload.update(deadline_warning)
            return payload
        finally:
            if is_full_access_mode:
                duration_ms = int((perf_counter() - started) * 1000)
                exit_code = snapshot.exit_code if snapshot is not None else None
                store.append(
                    "fullaccess_shell",
                    {
                        "event": "fullaccess_shell",
                        "ts": _fullaccess_shell_audit_ts(),
                        "command": cmd,
                        "cwd": str(cwd_path),
                        "exit_code": int(exit_code if exit_code is not None else -1),
                        "duration_ms": duration_ms,
                        "mode": "fullaccess",
                    },
                )

    def _shell_output(args: dict[str, Any]) -> dict[str, Any]:
        manager = _require_terminal_manager()
        guard_terminal_op("shell_output")
        process_id = str(args.get("process_id", "")).strip()
        if not process_id:
            raise AgentRuntimeError("Missing required argument: process_id")
        since = _coerce_shell_since(args.get("since"))
        try:
            snapshot = manager.read(process_id, since=since)
        except KeyError:
            return _unknown_bg_process_payload(
                manager=manager,
                process_id=process_id,
                operation="shell_output",
                since=since,
            )
        payload = _format_bg_snapshot(process_id=process_id, snapshot=snapshot)
        _maybe_add_empty_poll_guidance(
            payload=payload,
            process_id=process_id,
            since=since,
            snapshot=snapshot,
        )
        return payload

    def _shell_wait(args: dict[str, Any]) -> dict[str, Any]:
        manager = _require_terminal_manager()
        guard_terminal_op("shell_wait")
        process_id = str(args.get("process_id", "")).strip()
        if not process_id:
            raise AgentRuntimeError("Missing required argument: process_id")
        since = _coerce_shell_since(args.get("since"))
        wait_seconds = _coerce_shell_wait_seconds(args.get("wait_seconds"))
        until = _coerce_shell_wait_until(args.get("until"))
        max_bytes = _coerce_shell_max_bytes(args.get("max_bytes"))
        clamped_wait_seconds, deadline_decision = _clamp_shell_wait_seconds(wait_seconds)
        started = perf_counter()
        try:
            snapshot, timed_out = manager.wait_for_output(
                process_id,
                since=since,
                timeout_s=clamped_wait_seconds,
                until=until,  # type: ignore[arg-type]
            )
        except KeyError:
            payload = _unknown_bg_process_payload(
                manager=manager,
                process_id=process_id,
                operation="shell_wait",
                since=since,
            )
            payload.update(
                {
                    "waited": False,
                    "timed_out": False,
                    "wait_seconds_requested": wait_seconds,
                    "wait_seconds_effective": 0.0,
                    "until": until,
                    "elapsed_ms": int((perf_counter() - started) * 1000),
                }
            )
            if deadline_decision is not None:
                payload["deadline_start_decision"] = deadline_decision
                payload["deadline_clamped"] = clamped_wait_seconds < wait_seconds
            return payload
        elapsed_ms = int((perf_counter() - started) * 1000)
        payload = _format_bg_snapshot(
            process_id=process_id,
            snapshot=snapshot,
            max_bytes=max_bytes,
        )
        payload.update(
            {
                "waited": True,
                "timed_out": timed_out,
                "wait_seconds_requested": wait_seconds,
                "wait_seconds_effective": clamped_wait_seconds,
                "until": until,
                "elapsed_ms": elapsed_ms,
            }
        )
        if deadline_decision is not None:
            payload["deadline_start_decision"] = deadline_decision
            payload["deadline_clamped"] = clamped_wait_seconds < wait_seconds
        return payload

    def _shell_kill(args: dict[str, Any]) -> dict[str, Any]:
        manager = _require_terminal_manager()
        guard_terminal_op("shell_kill")
        process_id = str(args.get("process_id", "")).strip()
        if not process_id:
            raise AgentRuntimeError("Missing required argument: process_id")
        try:
            snapshot = manager.kill(process_id)
        except KeyError as exc:
            raise AgentRuntimeError(f"Unknown background process_id: {process_id}") from exc
        store.append(
            "bg_kill",
            {
                "process_id": process_id,
                "status": snapshot.status,
                "exit_code": snapshot.exit_code,
            },
        )
        return _format_bg_snapshot(process_id=process_id, snapshot=snapshot)

    def _shell_list(_args: dict[str, Any]) -> dict[str, Any]:
        manager = _require_terminal_manager()
        guard_terminal_op("shell_list")
        return {"processes": _format_bg_summaries(manager)}

    def _shell_service_start(args: dict[str, Any]) -> dict[str, Any]:
        deadline_decision = _deadline_start_decision(
            DeadlineOperation.SHELL_BACKGROUND,
            minimum_remaining_seconds=MINIMUM_TOOL_START_SECONDS,
        )
        deadline_warning = None
        if deadline_decision is not None and not bool(deadline_decision.get("allowed")):
            deadline_warning = _deadline_warning_fields(
                "Deadline policy would normally block service work; start proceeded because "
                "this is advisory and not safety.",
                start_decision=deadline_decision,
            )
        manager = _require_durable_service_manager()
        cmd = str(args.get("cmd", ""))
        guard_shell(cmd, tool_name="shell_service_start")
        readiness = _guard_service_readiness_spec(args.get("readiness"))
        effective_cwd_relpath = _resolve_workspace_relative_path(
            raw_path=args.get("cwd"),
            raw_base=args.get("cwd_base"),
            field_name="cwd",
            base_field_name="cwd_base",
            allow_empty=True,
        )
        cwd_path = root if not effective_cwd_relpath else (root / effective_cwd_relpath).resolve()
        try:
            started = manager.start(cmd=cmd, cwd=cwd_path, readiness=readiness)
        except ValueError as exc:
            raise AgentRuntimeError(f"Invalid durable service request: {exc}") from exc
        except (ConfigError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
            raise AgentRuntimeError(f"Failed to start durable service: {exc}") from exc
        payload = dict(started.payload)
        payload["lifetime"] = "durable"
        if deadline_warning is not None:
            payload.update(deadline_warning)
        store.append(
            "service_start",
            {
                **_service_event_payload(payload),
                "cwd": effective_cwd_relpath,
            },
        )
        return payload

    def _workspace_preview_start(args: dict[str, Any]) -> dict[str, Any]:
        deadline_decision = _deadline_start_decision(
            DeadlineOperation.SHELL_BACKGROUND,
            minimum_remaining_seconds=MINIMUM_TOOL_START_SECONDS,
        )
        deadline_warning = None
        if deadline_decision is not None and not bool(deadline_decision.get("allowed")):
            deadline_warning = _deadline_warning_fields(
                "Deadline policy would normally block preview work; start proceeded because "
                "this is advisory and not safety.",
                start_decision=deadline_decision,
            )
        manager = _require_durable_service_manager()
        guard_terminal_op("workspace_preview_start")
        requested_access = str(args.get("access") or "auto").strip().lower()
        try:
            effective_access = manager.resolve_preview_access(requested_access)
        except ValueError as exc:
            raise AgentRuntimeError(f"Invalid workspace preview request: {exc}") from exc
        if effective_access == "lan" and not yes and not is_full_access_mode:
            if non_interactive and not host_managed_approvals:
                raise AgentRuntimeError(
                    "LAN preview exposure requires interactive approval. Use local access or "
                    "re-run in an approval-capable session."
                )
            decision = surface.request_approval(
                ApprovalRequest(
                    kind="workspace_preview_lan",
                    reason=(
                        "LAN preview access exposes an authenticated workspace server to other "
                        "devices on the current network"
                    ),
                    preview="Start a temporary authenticated LAN workspace preview",
                )
            )
            if not decision.allow:
                raise ApprovalDeclinedError("workspace_preview_lan")
        raw_port = args.get("port")
        if raw_port is None or raw_port == "":
            port = None
        else:
            if isinstance(raw_port, bool):
                raise AgentRuntimeError("Preview port must be an integer")
            try:
                port = int(raw_port)
            except (TypeError, ValueError) as exc:
                raise AgentRuntimeError("Preview port must be an integer") from exc
        effective_cwd_relpath = _resolve_workspace_relative_path(
            raw_path=args.get("cwd"),
            raw_base=args.get("cwd_base"),
            field_name="cwd",
            base_field_name="cwd_base",
            allow_empty=True,
        )
        cwd_path = root if not effective_cwd_relpath else (root / effective_cwd_relpath).resolve()
        try:
            started = manager.start_preview(
                cwd=cwd_path,
                access=requested_access,
                port=port,
            )
        except ValueError as exc:
            raise AgentRuntimeError(f"Invalid workspace preview request: {exc}") from exc
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            raise AgentRuntimeError(f"Failed to start workspace preview: {exc}") from exc
        payload = dict(started.payload)
        payload["lifetime"] = "durable"
        if deadline_warning is not None:
            payload.update(deadline_warning)
        store.append(
            "service_start",
            {
                **_service_event_payload(payload),
                "cwd": effective_cwd_relpath,
                "service_kind": "workspace_preview",
            },
        )
        return payload

    def _shell_service_status(args: dict[str, Any]) -> dict[str, Any]:
        manager = _require_durable_service_manager()
        guard_terminal_op("shell_service_status")
        service_id = str(args.get("service_id", "")).strip()
        if not service_id:
            raise AgentRuntimeError("Missing required argument: service_id")
        try:
            payload = manager.status(service_id)
        except ValueError as exc:
            raise AgentRuntimeError(f"Invalid durable service_id: {exc}") from exc
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            raise AgentRuntimeError(f"Failed to inspect durable service: {exc}") from exc
        store.append("service_status", _service_event_payload(payload))
        return payload

    def _shell_service_stop(args: dict[str, Any]) -> dict[str, Any]:
        manager = _require_durable_service_manager()
        guard_terminal_op("shell_service_stop")
        service_id = str(args.get("service_id", "")).strip()
        if not service_id:
            raise AgentRuntimeError("Missing required argument: service_id")
        try:
            payload = manager.stop(service_id)
        except ValueError as exc:
            raise AgentRuntimeError(f"Invalid durable service_id: {exc}") from exc
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            raise AgentRuntimeError(f"Failed to stop durable service: {exc}") from exc
        store.append("service_stop", _service_event_payload(payload))
        return payload

    _append_builtin_tool("shell_background", run=_shell_background)
    _append_builtin_tool("shell_output", run=_shell_output)
    _append_builtin_tool("shell_wait", run=_shell_wait)
    _append_builtin_tool("shell_kill", run=_shell_kill)
    _append_builtin_tool("shell_list", run=_shell_list)
    _append_builtin_tool("shell_service_start", run=_shell_service_start)
    _append_builtin_tool("workspace_preview_start", run=_workspace_preview_start)
    _append_builtin_tool("shell_service_status", run=_shell_service_status)
    _append_builtin_tool("shell_service_stop", run=_shell_service_stop)

    def _session_set_workdir(args: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(args.get("path", "")).strip()
        if not raw_path:
            raise SessionWorkdirError("Missing required argument: path")
        if not callable(set_active_workdir_callback):
            raise SessionWorkdirError("session_set_workdir is unavailable in this session.")
        return set_active_workdir_callback(raw_path, "tool")

    _append_builtin_tool("session_set_workdir", run=_session_set_workdir)

    _append_builtin_tool(
        "git_status",
        run=lambda _args: git_status(root=root),
    )

    _append_builtin_tool(
        "git_diff",
        run=lambda _args: git_diff(root=root),
    )

    if git_backed_workspace:
        _append_builtin_tool(
            "git_history",
            run=lambda args: git_history(
                root=root,
                mode=str(args.get("mode", "")),
                path=str(args["path"]) if args.get("path") is not None else None,
                limit=int(args["limit"]) if args.get("limit") is not None else 10,
                ref=str(args["ref"]) if args.get("ref") is not None else None,
                grep=str(args["grep"]) if args.get("grep") is not None else None,
                author=str(args["author"]) if args.get("author") is not None else None,
                commit=str(args["commit"]) if args.get("commit") is not None else None,
                start_line=(
                    int(args["start_line"]) if args.get("start_line") is not None else None
                ),
                end_line=(int(args["end_line"]) if args.get("end_line") is not None else None),
            ),
        )

    def _git_apply(args: dict[str, Any]) -> dict[str, Any]:
        patch = str(args.get("patch", ""))
        patch_paths = iter_patch_paths(patch)
        for p in patch_paths:
            _guard_write_path(p)
        preview = patch[:20000]
        store.append("diff_preview", {"patch": preview})
        surface.on_patch_generated(
            PatchEvent(
                files=sorted(set(patch_paths)),
                diff=patch,
                summary=f"{len(set(patch_paths))} file(s) changed via git_apply_patch",
            )
        )
        guard_write("git_apply_patch", preview or "(empty patch)", files=sorted(set(patch_paths)))
        return git_apply_patch(root=root, patch=patch)

    _append_builtin_tool("git_apply_patch", run=_git_apply)

    if subagents_enabled and subagent_depth == 0:
        available_subagent_names = sorted(subagent_registry.keys()) if subagent_registry else []
        subagent_parameters = copied_tool_parameters("subagent_run")
        properties = subagent_parameters.get("properties")
        if not isinstance(properties, dict):
            raise AgentRuntimeError("subagent_run parameters must define properties")
        subagent_name_schema = properties.get("name")
        if not isinstance(subagent_name_schema, dict):
            raise AgentRuntimeError("subagent_run parameters must define a name property")
        if available_subagent_names:
            subagent_name_schema["enum"] = available_subagent_names

        _append_builtin_tool(
            "subagent_run",
            parameters=subagent_parameters,
            run=_subagent_run,
        )

    for custom_tool_spec in sorted(
        custom_tool_session_state.exposed_tools_by_name.values(),
        key=lambda spec: spec.name.casefold(),
    ):
        tools.append(
            ToolDef(
                name=custom_tool_spec.name,
                description=custom_tool_spec.description,
                parameters=copy.deepcopy(custom_tool_spec.input_schema),
                run=lambda args, spec=custom_tool_spec: _run_custom_tool(spec, args),
                metadata={
                    "tool_type": "custom_tool",
                    "compact_parameters_for_model": True,
                    "model_description_max_chars": 1200,
                    "custom_tool": custom_tool_spec.metadata(include_output_schema=True),
                },
            )
        )

    if mcp_manager is not None and _mcp_tool_exposed_in_mode(mode=mode):
        for binding in mcp_manager.tool_bindings:
            bound_binding = binding.bind_session_mode(mode)
            tools.append(
                ToolDef(
                    name=bound_binding.tool_alias,
                    description=bound_binding.description,
                    parameters=bound_binding.parameters,
                    run=bound_binding.run,
                    metadata={
                        "tool_type": "mcp",
                        "compact_parameters_for_model": True,
                        "model_description_max_chars": 1000,
                    },
                )
            )
    active_tools = {t.name: t for t in tools}
    for metadata in iter_builtin_tool_metadata():
        register_tool_availability(metadata.name, optional=metadata.optional)
        if metadata.name in active_tools:
            mark_available(metadata.name)
        elif metadata.optional:
            reason = metadata.optional_unavailable_reason or (
                "not registered in active tool registry "
                f"for mode={mode} runtime_kind={resolved_runtime_kind.value}"
            )
            mark_unavailable(metadata.name, reason)
    return active_tools
