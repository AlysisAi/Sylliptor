from __future__ import annotations

import copy
import json
import re
from collections.abc import Collection
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from ...config import resolve_role_temperature
from ...execution_deadline import (
    MINIMUM_LLM_START_SECONDS,
    MINIMUM_TOOL_START_SECONDS,
    DeadlineExhausted,
    DeadlineOperation,
    DeadlinePhase,
    temporarily_clamp_client_timeout,
)
from ...llm.metadata import assistant_message_from_response
from ...llm.types import LLMError
from ...runtime_kind import RuntimeKind
from ...step_budget import StepBudgetRequest, resolve_step_budget
from ...subagents import SubagentDefinition, canonical_subagent_name, normalize_subagent_mode
from ...surface import NoopSurface, ToolEndEvent, ToolOutputEvent, ToolStartEvent
from ...surface.base import Surface
from ...tools.availability import is_tool_unavailable_result, unavailable_tool_result
from ...tools.fs import FsError
from ...tools.git import GitError
from ...tools.history import HistorySearchError
from ...tools.registry import (
    build_unknown_tool_recovery_payload,
    compatibility_tool_alias_for,
    transform_compatibility_tool_alias,
)
from ...tools.search import SearchError
from ...tools.shell import ShellError
from ...tools.symbols import SymbolSearchError
from ...tools.web import WebFetchError
from ...tools.web_search import WebSearchError
from ...turn_intent import (
    classify_local_materialization_requirement,
)
from ...turn_intent import (
    classify_repo_execution_intent as _classify_one_shot_repo_turn_intent,
)
from ...turn_intent import normalize_turn_intent_text as _normalize_marker_text
from ...usage_tracker import build_usage_record
from ...verify_gate import VerifyError
from .. import _patchable
from ..acceptance_contract import (
    acceptance_contract_problem_payload,
    build_acceptance_contract,
    finalize_acceptance_contract,
)
from ..completion_gate import (
    NON_FINAL_PROGRESS_PROBLEM,
    NON_FINAL_PROGRESS_STAGE,
    CompletionGateDecision,
    CompletionGateDecisionKind,
    build_completion_gate_snapshot,
    completion_gate_decision_payload,
    decide_completion_gate,
    record_completion_gate_decision,
)
from ..errors import AgentRuntimeError
from ..prompt_context import (
    _IMAGE_ATTACHMENT_TURN_SYSTEM_HINT,
    MAX_POST_EXPLORE_ANCHOR_PATHS,
    _build_user_message,
    _extract_repo_relative_paths_from_text,
    _first_turn_repo_grounding_nudge_message,
    _plain_dir_workspace_route_override_reason,
    _recent_visible_non_repo_history,
    _repo_workspace_route_override_reason,
    _resolve_session_pinned_prefix_len,
    _session_has_active_workspace_task,
    _session_has_stable_workspace_grounding,
    _session_repo_scan,
    _session_task_brief_content,
    _session_workspace_grounding,
    _set_session_pinned_prefix_len,
    _turn_route_context,
    _workspace_kind_is_repo_backed,
    refresh_session_task_brief_message,
)
from ..routing import (
    _NON_REPO_TURN_SYSTEM_HINT,
    _ROUTING_MODE_AUTO,
    _build_turn_language_system_message,
    _fallback_route_decision,
    _is_stream_unsupported_error,
    _local_materialization_route_override_reason,
    _main_agent_chat,
    _managed_execution_route_override_reason,
    _non_repo_tool_assisted_tools,
    _normalize_routing_mode,
    _normalize_turn_language_name,
    _normalize_turn_script_name,
    _OneShotRepoTurnIntent,
    _registered_tool_schema_list,
    _request_messages_with_ephemeral_system_prompt_suffixes,
    _request_messages_with_ephemeral_system_prompts,
    _request_messages_with_ephemeral_user_messages,
    _resolve_degraded_route_execution_posture,
    _resolve_repo_turn_execution_intent,
    _respond_non_repo_turn,
    _route_reply_for_non_repo_turn,
    _route_turn,
    _safe_forced_tool_choice_for_recovery,
    _should_add_non_repo_turn_hint,
    _TurnRouteDecision,
)
from ..tools_assembly import _ROUTING_MODE_CODE_ONLY, ToolDef, _tool_event_metadata
from ..verification import (
    TurnExecutionState,
    _completion_gate_blocker_allows_final,
    _completion_gate_nudge_message,
    _completion_gate_problem_summary,
    _completion_gate_problems,
    _completion_gate_repair_stage,
    _completion_gate_step_budget_exhausted_message,
    _completion_gate_terminal_failure_message,
    _extract_touched_repo_paths,
    _record_tool_effect,
    _refresh_execute_turn_verification_selection,
    _runtime_message,
    _sorted_missing_verification_commands,
    _verification_expected_for_turn,
)
from .events import (
    _emit_assistant_message_events,
    _emit_message_delta_event,
    _emit_tool_call_completed_event,
    _emit_tool_call_progress_event,
    _emit_tool_call_started_event,
)
from .exploration import (
    _append_recent_exploration_path,
    _assistant_text_contains_progress_intent,
    _assistant_text_has_blocker_marker,
    _assistant_text_has_completion_marker,
    _assistant_text_has_well_formed_blocker,
    _build_post_explore_bootstrap_nudge,
    _edit_similarity_key,
    _exploration_attempt_outcome,
    _exploration_similarity_key,
    _extract_successful_exploration_paths,
    _is_action_progress_tool,
    _is_exploration_only_tool,
    _is_failed_edit_stagnation_tool,
    _is_successful_subagent_run,
    _one_shot_progress_fingerprint,
    _tool_call_retry_key,
)
from .read_cache import (
    _maybe_reuse_same_batch_read_result,
    _remember_same_batch_read_result,
    _same_batch_read_cache_should_invalidate,
    _SameBatchReadReuseCache,
)

MAX_IDENTICAL_TOOL_CALL_FAILURES = 2
MAX_NON_FINAL_CONTINUATIONS_PER_TURN = 2
MAX_EXPLORATION_ONLY_STEPS_BEFORE_NUDGE = 6
MAX_IDENTICAL_EXPLORATION_ATTEMPTS = 3
MAX_EXPLORATION_NUDGES_PER_TURN = 2
MAX_POST_EXPLORE_BOOTSTRAP_NUDGES_PER_TURN = 2
MAX_FAILED_EDIT_STEPS_BEFORE_NUDGE = 2
MAX_IDENTICAL_FAILED_EDIT_ATTEMPTS = 2
MAX_EDIT_NUDGES_PER_TURN = 2
MAX_EMPTY_RESPONSE_ANOMALY_RECOVERIES = 2
_ADAPTIVE_RETRY_TEMPERATURE = 0.5
_FORCED_FINAL_SUMMARY_SYSTEM_PROMPT_TEMPLATE = """The current turn is stopping now.

Stop reason: {termination_cause}

No more tool calls are allowed.
Respond with plain text only.
Give a concise summary of:
- what was completed
- what remains unfinished
- any known issues or risks in the current state
"""
_FINAL_TOOL_ENABLED_STEP_SYSTEM_PROMPT = """This is your last tool-enabled step for this turn.
If one high-value action remains, use this step for that action.
If the task is complete, provide the final answer.
Avoid low-value exploration or unnecessary extra detours.
If you still cannot finish cleanly, the runtime may ask for a final summary next."""
_LOW_STEP_BUDGET_SYSTEM_PROMPT_TEMPLATE = """Step budget pressure: {remaining_steps} tool-enabled step(s) remain after this one.
Prioritize finishing integration and verification over additional exploration.
Use tools only for decisive actions; if there is not enough context to finish safely, report the concrete blocker."""
_PHASE_BUDGET_EXPLORATION_SYSTEM_PROMPT_TEMPLATE = """Phase budget pressure: {exploration_steps} consecutive exploration-only step(s) have completed without material progress.
Use the next step to start implementation, delegate focused exploration if available, or report the concrete blocker."""
_PHASE_BUDGET_VERIFICATION_SYSTEM_PROMPT_TEMPLATE = """Phase budget pressure: edits have started and {remaining_steps} tool-enabled step(s) remain after this one.
Prioritize integration and verification now; avoid reopening broad exploration unless a concrete blocker requires it."""
_DEADLINE_FINALIZATION_SYSTEM_PROMPT_TEMPLATE = """Run deadline finalization window is active.
Do not start new subagents, broad exploration, optional dependency installs, speculative rewrites, provider retry sleeps, or optimization passes.
Materialize the best valid result now. Prefer syntactically valid artifacts, preserve existing inputs, and write required outputs before explaining anything.
If a bounded high-value verification check is already known and there is enough time, run it once. Otherwise skip verification and give a truthful final answer with known uncertainty.
{checkpoint_hint}"""
_SUBAGENT_EXPLORATION_NUDGE_TEMPLATE = """Subagent delegation check: {exploration_steps} consecutive read-only exploration step(s) have completed without a subagent_run call.
If more repository context is still needed, use the next tool-enabled step to call subagent_run with a focused, self-contained task brief. If you already have enough context, move to implementation or verification now.
Recent path anchors: {anchor_paths}"""
_SUBAGENT_REQUIRED_NUDGE_TEMPLATE = """The current user request explicitly asked for subagent or delegation behavior, but this turn has not attempted subagent_run yet.
Use the next tool-enabled step to call subagent_run with the best registered subagent and a self-contained task brief. If subagent_run is unavailable or fails, report that concrete blocker instead of finalizing as if delegation happened.
Available subagents: {available_subagents}"""
_SUBAGENT_REQUIRED_RETRY_EXHAUSTED_MESSAGE = (
    "The user explicitly requested subagent delegation, but the turn reached the retry limit "
    "without any subagent_run attempt."
)
_SUBAGENT_REQUEST_UNAVAILABLE_MESSAGE_TEMPLATE = (
    "Subagents were explicitly requested, but subagent_run is unavailable for this session "
    "({reason}). Enable subagents for a top-level session and retry."
)
_CLARIFICATION_ONLY_RE = re.compile(
    r"(?:\?|"
    r"\b(?:clarify|clarification|which|what|where|who|when|could you provide|"
    r"please provide|what would you like|need more info|need more information)\b|"
    r"\b(?:διευκρινιση|διευκρινισεις|ποιο|ποια|ποιος|τι|που|χρειαζομαι "
    r"περισσοτερες πληροφοριες)\b)",
    re.IGNORECASE,
)
_CLARIFICATION_VALID_BLOCKER_RE = re.compile(
    r"\b(?:credential|credentials|password|token|secret|api\s+key|login|auth|"
    r"authorization|access|permission|destructive|delete|remove|wipe|overwrite|"
    r"drop|production|external\s+input|unavailable|missing)\b",
    re.IGNORECASE,
)


@dataclass
class _EmptyResponseAnomalyRecoveryState:
    attempts: int = 0
    finalization_window_attempts: int = 0
    last_missing_action: str = ""
    last_tool_choice: dict[str, Any] | None = None


MAX_SUBAGENT_REQUIRED_NUDGES_PER_TURN = 2
MAX_SUBAGENT_EXPLORATION_NUDGES_PER_TURN = 1
MAX_PARALLEL_SUBAGENT_TOOL_CALLS = 4
_SUBAGENT_REQUEST_PATTERNS = (
    re.compile(
        r"\b(?:use|run|call|ask|spawn|start|invoke)\b(?:\s+\S+){0,8}\s+"
        r"\b(?:sub[\s-]?agents?|helper\s+agents?|speciali[sz]ed\s+agents?|"
        r"parallel\s+agents?|explorer|reviewer|test[\s-]?strategist|general[\s-]?purpose)\b"
    ),
    re.compile(
        r"\b(?:sub[\s-]?agents?|helper\s+agents?|speciali[sz]ed\s+agents?|"
        r"parallel\s+agents?|explorer|reviewer|test[\s-]?strategist|general[\s-]?purpose)\b"
        r"(?:\s+\S+){0,8}\s+\b(?:use|run|call|ask|spawn|start|invoke)\b"
    ),
    re.compile(
        r"\bdelegat(?:e|es|ed|ing|ion)\b(?:\s+\S+){0,10}\s+"
        r"\b(?:sub[\s-]?agents?|agents?|task|work|research|investigation|review|tests?|implementation)\b"
    ),
    re.compile(r"\bparallel\b(?:\s+\S+){0,5}\s+\bagents?\b"),
)
_SUBAGENT_REQUEST_OPT_OUT_PATTERNS = (
    re.compile(
        r"\b(?:no|without|avoid|disable|disabled|never|do\s+not|dont|don't)\b"
        r"(?:\s+\S+){0,6}\s+\b(?:sub[\s-]?agents?|agents?|delegat(?:e|es|ed|ing|ion))\b"
    ),
    re.compile(
        r"\b(?:sub[\s-]?agents?|agents?|delegat(?:e|es|ed|ing|ion))\b"
        r"(?:\s+\S+){0,6}\s+\b(?:off|disabled|disable|avoid|never)\b"
    ),
)
_DEADLINE_FINALIZATION_EXPLORATION_TOOL_NAMES = frozenset(
    {
        "fs_read",
        "fs_read_lines",
        "fs_list",
        "git_diff",
        "git_history",
        "git_status",
        "history_search",
        "search_rg",
        "skill_read",
        "symbol_search",
        "web_fetch",
        "web_search",
    }
)
_DEADLINE_FINALIZATION_MUTATION_TOOL_NAMES = frozenset(
    {
        "fs_copy",
        "fs_delete",
        "fs_edit",
        "fs_mkdir",
        "fs_move",
        "fs_write",
        "git_apply_patch",
        "shell_run",
        "verify_run",
    }
)


_SubagentTurnPolicyLevel = Literal[
    "off",
    "available",
    "recommended",
    "required_by_user",
    "unavailable",
]


@dataclass(frozen=True)
class _SubagentTurnPolicy:
    level: _SubagentTurnPolicyLevel
    reason: str
    available_subagents: tuple[str, ...] = ()

    @property
    def required_by_user(self) -> bool:
        return self.level == "required_by_user"

    @property
    def unavailable(self) -> bool:
        return self.level == "unavailable"

    @property
    def active(self) -> bool:
        return self.level in {"available", "recommended", "required_by_user"}


def _subagent_names_preview(names: Collection[str] | tuple[str, ...], *, limit: int = 8) -> str:
    clean_names = [str(name or "").strip() for name in names if str(name or "").strip()]
    clean_names = sorted(dict.fromkeys(clean_names))
    if not clean_names:
        return "-"
    shown = clean_names[:limit]
    suffix = ""
    if len(clean_names) > limit:
        suffix = f", +{len(clean_names) - limit} more"
    return ", ".join(shown) + suffix


def _instruction_explicitly_opts_out_subagent(instruction: str) -> bool:
    normalized = _normalize_marker_text(instruction)
    return bool(
        normalized
        and any(pattern.search(normalized) for pattern in _SUBAGENT_REQUEST_OPT_OUT_PATTERNS)
    )


def _instruction_explicitly_requests_subagent(
    instruction: str,
    *,
    subagent_names: Collection[str],
) -> bool:
    normalized = _normalize_marker_text(instruction)
    if not normalized:
        return False
    if _instruction_explicitly_opts_out_subagent(instruction):
        return False
    if any(pattern.search(normalized) for pattern in _SUBAGENT_REQUEST_PATTERNS):
        return True
    for raw_name in subagent_names:
        name = _normalize_marker_text(str(raw_name or "").replace("-", " "))
        if not name:
            continue
        if re.search(
            rf"\b(?:use|run|call|ask|spawn|start|invoke)\b"
            rf"(?:\s+\S+){{0,8}}\s+\b{re.escape(name)}\b",
            normalized,
        ):
            return True
    return False


def _resolve_subagent_turn_policy(
    *,
    instruction: str,
    subagents_enabled: bool,
    enforce_explicit_request: bool = True,
    subagent_depth: int,
    subagent_registry: dict[str, SubagentDefinition] | None,
    turn_tools: dict[str, ToolDef],
    repo_turn_execution_intent: _OneShotRepoTurnIntent,
) -> _SubagentTurnPolicy:
    available_names = tuple(sorted((subagent_registry or {}).keys()))
    if _instruction_explicitly_opts_out_subagent(instruction):
        return _SubagentTurnPolicy(level="off", reason="user_opt_out")
    explicit_request = _instruction_explicitly_requests_subagent(
        instruction,
        subagent_names=available_names,
    )
    if not explicit_request and not available_names:
        return _SubagentTurnPolicy(level="off", reason="no_registered_subagents")
    if subagent_depth > 0:
        reason = "nested_subagent_session"
        return (
            _SubagentTurnPolicy(
                level="unavailable",
                reason=reason,
                available_subagents=available_names,
            )
            if explicit_request and enforce_explicit_request
            else _SubagentTurnPolicy(level="off", reason=reason)
        )
    if not subagents_enabled:
        reason = "subagents_disabled"
        return (
            _SubagentTurnPolicy(
                level="unavailable",
                reason=reason,
                available_subagents=available_names,
            )
            if explicit_request and enforce_explicit_request
            else _SubagentTurnPolicy(level="off", reason=reason)
        )
    if "subagent_run" not in turn_tools:
        reason = "subagent_tool_not_exposed"
        return (
            _SubagentTurnPolicy(
                level="unavailable",
                reason=reason,
                available_subagents=available_names,
            )
            if explicit_request and enforce_explicit_request
            else _SubagentTurnPolicy(level="off", reason=reason)
        )
    if explicit_request:
        return _SubagentTurnPolicy(
            level="required_by_user",
            reason="explicit_user_request",
            available_subagents=available_names,
        )
    if repo_turn_execution_intent == "execute":
        return _SubagentTurnPolicy(
            level="recommended",
            reason="repo_execution_turn",
            available_subagents=available_names,
        )
    return _SubagentTurnPolicy(
        level="available",
        reason="repo_non_execution_turn",
        available_subagents=available_names,
    )


def _subagent_turn_context_message(policy: _SubagentTurnPolicy) -> str | None:
    if not policy.active:
        return None
    lines = [
        "<subagent_turn_context>",
        f"policy: {policy.level}",
        f"reason: {policy.reason}",
        f"available_subagents: {_subagent_names_preview(policy.available_subagents)}",
        "rules:",
    ]
    if policy.required_by_user:
        lines.append(
            "- The user explicitly asked for subagent/delegation behavior. Call subagent_run "
            "before finalizing unless the tool is unavailable or fails, in which case report "
            "the concrete blocker."
        )
    else:
        lines.append(
            "- Make an explicit delegation decision before broad repository exploration. Use "
            "subagent_run for multi-file, unfamiliar, review, or test-strategy work; use "
            "direct tools when one targeted read is enough."
        )
    lines.append(
        "- Subagent task briefs must be self-contained: goal, paths/symbols when known, "
        "current context, and expected answer shape."
    )
    lines.append("</subagent_turn_context>")
    return "\n".join(lines)


def _subagent_required_nudge_message(policy: _SubagentTurnPolicy) -> str:
    return _SUBAGENT_REQUIRED_NUDGE_TEMPLATE.format(
        available_subagents=_subagent_names_preview(policy.available_subagents),
    )


def _subagent_exploration_nudge_message(
    *,
    exploration_steps: int,
    anchor_paths: Collection[str],
) -> str:
    anchor_text = _subagent_names_preview(anchor_paths, limit=5)
    return _SUBAGENT_EXPLORATION_NUDGE_TEMPLATE.format(
        exploration_steps=max(1, int(exploration_steps)),
        anchor_paths=anchor_text,
    )


def _has_invalid_tool_call_json(tool_calls: list[Any]) -> bool:
    for tc in tool_calls:
        arguments = getattr(tc, "arguments", None)
        if not isinstance(arguments, dict):
            continue
        if set(arguments.keys()) != {"_raw_arguments"}:
            continue
        if isinstance(arguments.get("_raw_arguments"), str):
            return True
    return False


def _is_subagent_run_tool_call(tool_call: Any) -> bool:
    return str(getattr(tool_call, "name", "") or "").strip().lower() == "subagent_run"


def _deadline_operation_for_tool_name(tool_name: str) -> DeadlineOperation:
    normalized = str(tool_name or "").strip().lower()
    if normalized == "subagent_run":
        return DeadlineOperation.SUBAGENT
    if normalized == "verify_run":
        return DeadlineOperation.VERIFICATION
    if normalized == "shell_background":
        return DeadlineOperation.SHELL_BACKGROUND
    if normalized in _DEADLINE_FINALIZATION_EXPLORATION_TOOL_NAMES:
        return DeadlineOperation.EXPLORATION_TOOL
    if normalized in _DEADLINE_FINALIZATION_MUTATION_TOOL_NAMES:
        return DeadlineOperation.MUTATION_TOOL
    return DeadlineOperation.TOOL_DISPATCH


def _subagent_tool_call_resolves_readonly_mode(
    tool_call: Any,
    *,
    subagent_registry: dict[str, SubagentDefinition] | None,
) -> bool:
    arguments = getattr(tool_call, "arguments", None)
    if not isinstance(arguments, dict):
        return False
    mode_override = str(arguments.get("mode") or "").strip()
    if mode_override:
        return normalize_subagent_mode(mode_override) in {"readonly", "review"}
    requested_name = canonical_subagent_name(str(arguments.get("name") or ""))
    if requested_name is None:
        return subagent_registry is None
    if not subagent_registry:
        return True
    definition = subagent_registry.get(requested_name)
    if definition is None:
        return False
    return normalize_subagent_mode(definition.mode) in {"readonly", "review"}


def _can_prelaunch_parallel_subagent_batch(
    *,
    tool_calls: list[Any],
    turn_tools: dict[str, Any],
    subagent_registry: dict[str, SubagentDefinition] | None,
    failed_tool_call_counts: dict[str, int],
    hook_dispatcher: Any,
    subagent_policy_reason: str,
    deadline_can_start: bool,
) -> bool:
    if len(tool_calls) < 2 or not all(_is_subagent_run_tool_call(tc) for tc in tool_calls):
        return False
    if not all(
        _subagent_tool_call_resolves_readonly_mode(
            tc,
            subagent_registry=subagent_registry,
        )
        for tc in tool_calls
    ):
        return False
    if hook_dispatcher is not None or subagent_policy_reason == "user_opt_out":
        return False
    if not deadline_can_start:
        return False
    for tc in tool_calls:
        if turn_tools.get(tc.name) is None or unavailable_tool_result(tc.name) is not None:
            return False
        retry_key = _tool_call_retry_key(tc.name, tc.arguments)
        if failed_tool_call_counts.get(retry_key, 0) >= MAX_IDENTICAL_TOOL_CALL_FAILURES:
            return False
    return True


def _emit_surface_error(
    surface: Surface | object,
    code: str,
    message: str,
    recoverable: bool,
    *,
    worker_id: str | None = None,
    role: str | None = None,
) -> None:
    surface_cls = getattr(surface, "__class__", None)
    handler = getattr(surface, "emit_error", None)
    if callable(handler):
        cls_handler = getattr(surface_cls, "emit_error", None)
        if cls_handler is not getattr(NoopSurface, "emit_error", None):
            handler(code, message, recoverable, worker_id=worker_id, role=role)
            return
    fallback = getattr(surface, "on_error", None)
    if callable(fallback):
        fallback(message)


def run_turn(
    self,
    instruction: str,
    *,
    image_paths: list[str] | None = None,
    routing_mode_override: str | None = None,
    ephemeral_system_messages: list[str] | tuple[str, ...] | None = None,
    ephemeral_user_messages: list[str] | tuple[str, ...] | None = None,
    cancellation_token: Any | None = None,
) -> int:
    def _throw_if_cancelled() -> None:
        if cancellation_token is None:
            return
        throw_if_cancelled = getattr(cancellation_token, "throw_if_cancelled", None)
        if callable(throw_if_cancelled):
            throw_if_cancelled("cancelled_by_user")
            return
        if bool(getattr(cancellation_token, "is_cancelled", False)):
            raise RuntimeError("cancelled_by_user")

    def _phase_update(message: str) -> None:
        clean = message.strip()
        if not clean:
            return
        self.store.append("progress", {"message": clean})
        handler = getattr(self.surface, "on_progress_update", None)
        if callable(handler):
            handler(clean)

    instruction = str(instruction or "")
    image_paths = list(image_paths or [])
    assistant_message_emitted = False
    steps_attempted = 0
    deadline = getattr(self, "execution_deadline", None)
    diagnostics = getattr(self, "crash_diagnostics", None)

    def _deadline_snapshot() -> dict[str, Any] | None:
        if deadline is None:
            return None
        return deadline.telemetry_snapshot()

    def _deadline_decision_payload(
        operation: DeadlineOperation | str,
        *,
        minimum_remaining_seconds: float,
        estimated_duration_seconds: float | None = None,
        configured_timeout_seconds: float | None = None,
        allow_during_finalization: bool = False,
    ) -> dict[str, Any] | None:
        if deadline is None:
            return None
        decision = deadline.start_decision(
            operation,
            minimum_remaining_seconds=minimum_remaining_seconds,
            estimated_duration_seconds=estimated_duration_seconds,
            configured_timeout_seconds=configured_timeout_seconds,
            allow_during_finalization=allow_during_finalization,
        )
        return decision.telemetry_snapshot()

    def _deadline_allows(
        operation: DeadlineOperation | str,
        *,
        minimum_remaining_seconds: float,
        estimated_duration_seconds: float | None = None,
        configured_timeout_seconds: float | None = None,
        allow_during_finalization: bool = False,
    ) -> bool:
        payload = _deadline_decision_payload(
            operation,
            minimum_remaining_seconds=minimum_remaining_seconds,
            estimated_duration_seconds=estimated_duration_seconds,
            configured_timeout_seconds=configured_timeout_seconds,
            allow_during_finalization=allow_during_finalization,
        )
        if payload is None or bool(payload.get("allowed")):
            return True
        self.store.append(
            "deadline_operation_blocked",
            {
                "operation": payload.get("operation"),
                "reason": payload.get("reason"),
                "deadline": _deadline_snapshot(),
                "decision": payload,
            },
        )
        _diagnostic_event(
            "deadline_operation_blocked",
            {
                "operation": payload.get("operation"),
                "reason": payload.get("reason"),
                "deadline": _deadline_snapshot(),
                "deadline_start_decision": payload,
            },
        )
        return False

    def _record_deadline_duration(
        operation: DeadlineOperation | str,
        started_at_perf_counter: float,
    ) -> None:
        if deadline is None:
            return
        deadline.observe_duration(operation, max(0.0, perf_counter() - started_at_perf_counter))

    def _diagnostic_event(
        event_type: str,
        payload: dict[str, Any] | None = None,
        *,
        durable: bool = False,
    ) -> None:
        if diagnostics is None:
            return
        diagnostics.event(event_type, payload or {}, durable=durable)

    _throw_if_cancelled()

    runtime_kind_text_for_deadline = str(
        getattr(self.runtime_kind, "value", self.runtime_kind) or ""
    ).strip()
    if deadline is None and (
        self.one_shot_execution
        or runtime_kind_text_for_deadline in {"one_shot", "forge_exec", "swarm_worker"}
    ):
        payload = {
            "runtime_kind": runtime_kind_text_for_deadline,
            "deadline_config_source": "absent",
            "reason": "run deadline is not configured for this non-interactive run",
        }
        self.store.append("run_deadline_unconfigured", payload)
        _diagnostic_event("run_deadline_unconfigured", payload)

    def _finish_turn(code: int, *, reason: str, final_text: str = "") -> int:
        _diagnostic_event(
            "turn_finished",
            {
                "exit_code": code,
                "reason": reason,
                "steps_attempted": steps_attempted,
                "runtime_kind": str(getattr(self.runtime_kind, "value", self.runtime_kind)),
                "deadline": _deadline_snapshot(),
            },
            durable=True,
        )
        if self.hook_dispatcher is not None:
            cwd, active_workdir_relpath = self._hook_runtime_context()
            self._safe_dispatch_hooks(
                lambda: self.hook_dispatcher.fire_turn_complete(  # type: ignore[union-attr]
                    cwd=cwd,
                    active_workdir_relpath=active_workdir_relpath,
                    payload={
                        "exit_code": code,
                        "reason": reason,
                        "instruction": instruction,
                        "final_text": str(final_text or ""),
                        "steps_attempted": steps_attempted,
                        "assistant_message_emitted": assistant_message_emitted,
                        "messages_count": len(self.messages),
                        "workspace_touched_paths": sorted(self.workspace_touched_paths),
                    },
                )
            )
        return code

    routing_mode = _normalize_routing_mode(
        routing_mode_override if routing_mode_override is not None else self.routing_mode
    )
    hook_turn_system_messages: list[str] = []
    hook_turn_user_messages: list[str] = []
    prompt_hook_result = self._safe_dispatch_hooks(
        lambda: self.hook_dispatcher.fire_user_prompt_submit(  # type: ignore[union-attr]
            prompt=instruction,
            image_paths=image_paths,
            cwd=self._hook_runtime_context()[0],
            active_workdir_relpath=self._hook_runtime_context()[1],
        )
    )
    hook_turn_system_messages.extend(prompt_hook_result.additional_system_messages)
    hook_turn_user_messages.extend(prompt_hook_result.additional_user_messages)
    if prompt_hook_result.modified_prompt is not None:
        instruction = prompt_hook_result.modified_prompt
    if prompt_hook_result.blocked:
        message = (
            f"Prompt blocked by hook: {prompt_hook_result.reason}"
            if prompt_hook_result.reason
            else "Prompt blocked by hook."
        )
        self.store.append("error", {"error": message})
        _emit_surface_error(self.surface, "hook_error", message, True)
        return _finish_turn(1, reason="prompt_blocked")
    had_active_workspace_task_before_turn = _session_has_active_workspace_task(self)
    # Refreshing the task brief can insert or mutate pinned session messages in place,
    # so failed-turn rollback needs the full pre-turn message state.
    pre_turn_messages = copy.deepcopy(self.messages)
    pre_turn_pinned_prefix_len = _resolve_session_pinned_prefix_len(self)
    refresh_session_task_brief_message(
        self,
        pending_instruction=instruction,
    )
    turn_start_messages = len(self.messages)
    assistant_message_emitted = False
    last_visible_assistant_text = ""

    def _rollback_turn_after_llm_error() -> None:
        if assistant_message_emitted:
            return
        current_messages = getattr(self, "messages", None)
        current_pinned_prefix_len = _resolve_session_pinned_prefix_len(self)
        if (
            current_messages == pre_turn_messages
            and current_pinned_prefix_len == pre_turn_pinned_prefix_len
        ):
            return
        current_len = len(current_messages) if isinstance(current_messages, list) else 0
        rolled_back = max(0, current_len - len(pre_turn_messages))
        self.messages = copy.deepcopy(pre_turn_messages)
        _set_session_pinned_prefix_len(self, pre_turn_pinned_prefix_len)
        self.store.append(
            "warning",
            {
                "warning": "turn_rollback_after_llm_error",
                "rolled_back_messages": rolled_back,
            },
        )

    def _record_turn_llm_error(err: LLMError) -> None:
        self.store.append("error", {"error": str(err)})
        _rollback_turn_after_llm_error()

    ephemeral_turn_system_messages = [
        str(prompt or "").strip() for prompt in (ephemeral_system_messages or [])
    ]
    ephemeral_turn_system_messages = [prompt for prompt in ephemeral_turn_system_messages if prompt]
    if image_paths and _IMAGE_ATTACHMENT_TURN_SYSTEM_HINT not in ephemeral_turn_system_messages:
        ephemeral_turn_system_messages.append(_IMAGE_ATTACHMENT_TURN_SYSTEM_HINT)
        self.store.append(
            "system_note",
            {
                "message": "image_attachment_turn_hint",
                "image_count": len(image_paths),
            },
        )
    ephemeral_turn_system_messages.extend(hook_turn_system_messages)
    ephemeral_turn_user_messages = [
        str(prompt or "").strip() for prompt in (ephemeral_user_messages or [])
    ]
    ephemeral_turn_user_messages = [prompt for prompt in ephemeral_turn_user_messages if prompt]
    ephemeral_turn_user_messages.extend(hook_turn_user_messages)
    turn_language = ""
    turn_script = ""
    turn_language_explicit = False
    turn_language_source = "default"
    turn_language_failure_reason = ""

    def _runtime_text(key: str, **kwargs: Any) -> str:
        return _runtime_message(
            key,
            language=turn_language,
            explicit_language_override=turn_language_explicit,
            **kwargs,
        )

    def _phase_update_key(key: str, **kwargs: Any) -> None:
        _phase_update(_runtime_text(key, **kwargs))

    def _deadline_checkpoint_hint() -> str:
        paths = _extract_repo_relative_paths_from_text(root=self.root, text=instruction)
        if not paths:
            return ""
        path_text = ", ".join(str(path) for path in paths[:8])
        return f"Required or mentioned output paths to preserve/materialize: {path_text}"

    def _append_deadline_finalization_prompt(
        suffixes: list[str],
        *,
        step: int,
    ) -> None:
        if deadline is None:
            return
        entered = deadline.maybe_enter_finalization("reserve_reached")
        if entered:
            payload = {
                "step": step,
                "deadline": _deadline_snapshot(),
                "deadline_phase": DeadlinePhase.FINALIZATION_WINDOW.value,
                "deadline_finalization_reason": deadline.finalization_reason,
                "deadline_finalization_reserve_seconds": deadline.finalization_reserve_seconds(),
                "deadline_normal_work_remaining_seconds": (
                    deadline.normal_work_remaining_seconds()
                ),
            }
            self.store.append("deadline_finalization_started", payload)
            _diagnostic_event("deadline_finalization_started", payload)
        if deadline.phase() != DeadlinePhase.FINALIZATION_WINDOW:
            return
        if deadline.finalization_directive_sent:
            return
        checkpoint_hint = _deadline_checkpoint_hint()
        suffixes.append(
            _DEADLINE_FINALIZATION_SYSTEM_PROMPT_TEMPLATE.format(
                checkpoint_hint=checkpoint_hint,
            ).strip()
        )
        deadline.mark_finalization_directive_sent()
        self.store.append(
            "deadline_finalization_directive",
            {
                "step": step,
                "checkpoint_hint": checkpoint_hint,
                "deadline": _deadline_snapshot(),
            },
        )

    if self.step_budget_runtime is not None:
        self.step_budget_runtime.active_turn_budget = None

    turn_tools = self.tools
    turn_tool_list = _registered_tool_schema_list(turn_tools, self.tool_list)

    def _current_turn_step_limit() -> int:
        active_turn_budget = (
            self.step_budget_runtime.active_turn_budget
            if self.step_budget_runtime is not None
            else None
        )
        if isinstance(active_turn_budget, int) and active_turn_budget > 0:
            return active_turn_budget
        return max(1, int(self.max_steps))

    # Live reasoning channel shared by both the non-repo and repo turn paths:
    # surfaces that opt in (the TUI) receive the model's streamed thinking via
    # ``on_reasoning_token``; the classic RichSurface defines no such method, so
    # this stays a no-op there and the classic CLI is untouched.
    _reasoning_sink_exists = callable(getattr(self.surface, "on_reasoning_token", None))

    def _on_reasoning_delta(delta: str) -> None:
        # Per-token cancellation check: a long (streamed) reasoning phase must be
        # interruptible, not just at step boundaries. Raises KeyboardInterrupt,
        # which propagates out of the LLM stream (past ``except LLMError``).
        _throw_if_cancelled()
        sink = getattr(self.surface, "on_reasoning_token", None)
        if delta and callable(sink):
            try:
                sink(delta)
            except Exception:
                pass

    def _deadline_exhausted_result(operation: str, *, step: int | None = None) -> int:
        nonlocal assistant_message_emitted
        remaining_seconds = deadline.remaining_seconds() if deadline is not None else None
        payload = {
            "operation": operation,
            "step": step,
            "remaining_seconds": remaining_seconds,
            "deadline_exhausted": deadline.is_exhausted() if deadline is not None else False,
            "deadline": _deadline_snapshot(),
        }
        self.store.append("deadline_exhausted", payload)
        _diagnostic_event("deadline_exhausted", payload, durable=True)
        message = "The run deadline was exhausted before the turn could finish."
        _emit_surface_error(self.surface, "deadline_error", message, True)
        self._emit_forced_final_summary_before_termination(
            reason="deadline_exhausted",
            termination_cause="the run deadline is exhausted",
            termination_kind="deadline_exhausted",
            max_steps=_current_turn_step_limit(),
            language=turn_language,
            script=turn_script,
            explicit_language_override=turn_language_explicit,
            latest_assistant_text=last_visible_assistant_text,
            allow_llm_summary=False,
        )
        assistant_message_emitted = True
        return _finish_turn(1, reason="deadline_exhausted")

    if (
        routing_mode_override is None
        and routing_mode == _ROUTING_MODE_CODE_ONLY
        and _should_add_non_repo_turn_hint(
            instruction,
            image_paths=image_paths,
        )
    ):
        self.messages.append({"role": "system", "content": _NON_REPO_TURN_SYSTEM_HINT})
        self.store.append("system_note", {"message": "non_repo_turn_hint"})

    user_message, log_payload = _build_user_message(
        root=self.root,
        instruction=instruction,
        image_paths=image_paths,
    )
    display_instruction = log_payload.get("display_content")
    self.store.append("user_message", log_payload)
    self.messages.append(user_message)
    turn_user_message_index = len(self.messages) - 1
    if not isinstance(display_instruction, str) or not display_instruction.strip():
        display_instruction = instruction
    self.surface.on_user_message(display_instruction)
    _diagnostic_event(
        "turn_started",
        {
            "runtime_kind": str(getattr(self.runtime_kind, "value", self.runtime_kind)),
            "max_steps": self.max_steps,
            "deadline": _deadline_snapshot(),
        },
    )
    _phase_update_key("phase_understanding_request")

    local_materialization_requirement = classify_local_materialization_requirement(instruction)

    def _local_materialization_payload() -> dict[str, Any]:
        return {
            "local_materialization_required": local_materialization_requirement.required,
            "local_materialization_confidence": local_materialization_requirement.confidence,
            "local_materialization_output_paths": list(
                local_materialization_requirement.output_paths
            ),
            "local_materialization_action_verb": (local_materialization_requirement.action_verb),
            "local_materialization_evidence_span": (
                local_materialization_requirement.evidence_span
            ),
            "local_materialization_reason": local_materialization_requirement.reason,
        }

    recent_visible_non_repo_history = _recent_visible_non_repo_history(self.messages)
    route_client = self.router_client or self.client
    if routing_mode == _ROUTING_MODE_AUTO and not image_paths:
        route_context = _turn_route_context(
            self,
            had_active_workspace_task_before_turn=had_active_workspace_task_before_turn,
        )
        allow_implicit_repo_bugfix_override = _workspace_kind_is_repo_backed(
            self.store.workspace_kind
        )
        try:
            if not _deadline_allows(
                DeadlineOperation.ROUTING_LLM,
                minimum_remaining_seconds=MINIMUM_LLM_START_SECONDS,
            ):
                return _deadline_exhausted_result("routing_llm", step=0)
            _diagnostic_event(
                "llm_started",
                {"operation": "routing_llm", "step": 0, "deadline": _deadline_snapshot()},
            )
            route_turn = _patchable("_route_turn", _route_turn)
            operation_started = perf_counter()
            with temporarily_clamp_client_timeout(
                route_client,
                deadline,
                operation="routing_llm",
            ):
                original_route_decision = route_turn(
                    client=route_client,
                    instruction=instruction,
                    language=turn_language,
                    script=turn_script,
                    explicit_language_override=turn_language_explicit,
                    route_context=route_context,
                    recent_visible_history=recent_visible_non_repo_history,
                    allow_implicit_repo_bugfix_override=allow_implicit_repo_bugfix_override,
                )
            _record_deadline_duration(DeadlineOperation.ROUTING_LLM, operation_started)
            if (
                route_client is not self.client
                and str(getattr(original_route_decision, "route", "") or "").strip().lower()
                == "general"
                and str(getattr(original_route_decision, "decision_source", "") or "").startswith(
                    "fallback"
                )
            ):
                # The dedicated router-role model failed to produce a usable
                # decision and degraded to the static clarification path. Retry
                # once with the main model and use it for the non-repo response.
                if not _deadline_allows(
                    DeadlineOperation.ROUTING_LLM,
                    minimum_remaining_seconds=MINIMUM_LLM_START_SECONDS,
                ):
                    return _deadline_exhausted_result("routing_llm", step=0)
                operation_started = perf_counter()
                with temporarily_clamp_client_timeout(
                    self.client,
                    deadline,
                    operation="routing_llm",
                ):
                    retry_route_decision = route_turn(
                        client=self.client,
                        instruction=instruction,
                        language=turn_language,
                        script=turn_script,
                        explicit_language_override=turn_language_explicit,
                        route_context=route_context,
                        recent_visible_history=recent_visible_non_repo_history,
                        allow_implicit_repo_bugfix_override=allow_implicit_repo_bugfix_override,
                    )
                _record_deadline_duration(DeadlineOperation.ROUTING_LLM, operation_started)
                if not str(getattr(retry_route_decision, "decision_source", "") or "").startswith(
                    "fallback"
                ):
                    self.store.append(
                        "router_model_fallback_to_main",
                        {
                            "router_decision_source": str(
                                getattr(original_route_decision, "decision_source", "") or ""
                            ),
                            "retry_decision_source": str(
                                getattr(retry_route_decision, "decision_source", "") or ""
                            ),
                            "retry_route": retry_route_decision.route,
                        },
                    )
                    original_route_decision = retry_route_decision
                    route_client = self.client
            _diagnostic_event(
                "llm_completed",
                {"operation": "routing_llm", "step": 0, "deadline": _deadline_snapshot()},
            )
            if deadline is not None and deadline.is_exhausted():
                return _deadline_exhausted_result("routing_llm", step=0)
        except DeadlineExhausted:
            return _deadline_exhausted_result("routing_llm", step=0)
        except LLMError as err:
            _diagnostic_event(
                "llm_failed",
                {
                    "operation": "routing_llm",
                    "step": 0,
                    "failure_category": "llm_error",
                    "deadline": _deadline_snapshot(),
                },
            )
            _record_turn_llm_error(err)
            raise
        original_route_decision_source = str(
            getattr(original_route_decision, "decision_source", "router") or "router"
        )
        original_route = str(getattr(original_route_decision, "route", "") or "").strip().lower()
        original_route_execution_posture = str(
            getattr(original_route_decision, "execution_posture", "") or ""
        ).strip() or str(
            _resolve_degraded_route_execution_posture(
                instruction=instruction,
                route=original_route or "general",
            )
        )
        original_route_execution_posture_source = str(
            getattr(original_route_decision, "execution_posture_source", "") or ""
        ).strip() or (
            "router" if getattr(original_route_decision, "execution_posture", None) else "fallback"
        )
        route_override_reason = None
        route_override_execution_posture = original_route_execution_posture
        route_override_execution_posture_source = original_route_execution_posture_source
        if original_route_decision.route != "repo":
            route_override_reason = _managed_execution_route_override_reason(
                runtime_kind=self.runtime_kind,
                original_route=original_route_decision.route,
                route_execution_posture=original_route_execution_posture,
            )
            if route_override_reason is None:
                route_override_reason = _local_materialization_route_override_reason(
                    materialization=local_materialization_requirement,
                    original_route=original_route_decision.route,
                )
            if route_override_reason == "local_materialization_requires_repo_execution":
                route_override_execution_posture = "execute"
                route_override_execution_posture_source = "deterministic_override"
            if route_override_reason is None:
                route_override_reason = _plain_dir_workspace_route_override_reason(
                    self,
                    instruction,
                    had_active_workspace_task_before_turn=had_active_workspace_task_before_turn,
                )
            if route_override_reason is None:
                route_override_reason = _repo_workspace_route_override_reason(
                    self,
                    instruction,
                    had_active_workspace_task_before_turn=had_active_workspace_task_before_turn,
                )
        route_decision = original_route_decision
        if route_override_reason:
            route_decision = _TurnRouteDecision(
                route="repo",
                execution_posture=route_override_execution_posture,
                confidence=original_route_decision.confidence,
                reply="",
                language=original_route_decision.language,
                script=original_route_decision.script,
                explicit_language_override=original_route_decision.explicit_language_override,
                language_source=original_route_decision.language_source,
                decision_source=original_route_decision_source,
                execution_posture_source=route_override_execution_posture_source,
                tool_family=getattr(original_route_decision, "tool_family", "none"),
                tool_candidates=tuple(
                    getattr(original_route_decision, "tool_candidates", ()) or ()
                ),
            )
        route_selection_source = (
            "deterministic_override" if route_override_reason else original_route_decision_source
        )
        route_execution_posture = (
            str(
                getattr(route_decision, "execution_posture", "") or original_route_execution_posture
            ).strip()
            or original_route_execution_posture
        )
        route_execution_posture_source = (
            str(
                getattr(route_decision, "execution_posture_source", "")
                or original_route_execution_posture_source
            ).strip()
            or original_route_execution_posture_source
        )
        turn_language = _normalize_turn_language_name(getattr(route_decision, "language", "") or "")
        turn_script = _normalize_turn_script_name(getattr(route_decision, "script", "") or "")
        turn_language_explicit = bool(getattr(route_decision, "explicit_language_override", False))
        turn_language_source = (
            str(getattr(route_decision, "language_source", "") or "").strip() or "default"
        )
        turn_language_failure_reason = ""
        one_shot_turn_intent = _classify_one_shot_repo_turn_intent(instruction)
        classified_turn_intent_kind = (
            "mutating_execution" if one_shot_turn_intent == "execute" else "read_only"
        )
        if (
            route_decision.route != "repo"
            and local_materialization_requirement.required
            and local_materialization_requirement.confidence >= 0.8
        ):
            previous_route = route_decision.route
            previous_execution_posture = route_execution_posture
            route_override_reason = (
                route_override_reason or "local_materialization_non_repo_invariant"
            )
            route_decision = _TurnRouteDecision(
                route="repo",
                execution_posture="execute",
                confidence=route_decision.confidence,
                reply="",
                language=route_decision.language,
                script=route_decision.script,
                explicit_language_override=route_decision.explicit_language_override,
                language_source=route_decision.language_source,
                decision_source=route_decision.decision_source,
                execution_posture_source="deterministic_override",
                tool_family=getattr(route_decision, "tool_family", "none"),
                tool_candidates=tuple(getattr(route_decision, "tool_candidates", ()) or ()),
            )
            route_selection_source = "deterministic_override"
            route_execution_posture = "execute"
            route_execution_posture_source = "deterministic_override"
            self.store.append(
                "non_repo_materialization_invariant_reroute",
                {
                    "previous_route": previous_route,
                    "previous_execution_posture": previous_execution_posture,
                    "route_override_reason": route_override_reason,
                    **_local_materialization_payload(),
                },
            )
        self.store.append(
            "route_decision",
            {
                "route": route_decision.route,
                "original_route": original_route_decision.route,
                "execution_posture": route_execution_posture,
                "execution_posture_source": route_execution_posture_source,
                "router_execution_posture": original_route_execution_posture,
                "router_execution_posture_source": original_route_execution_posture_source,
                "router_decision_source": original_route_decision_source,
                "route_selection_source": route_selection_source,
                "confidence": route_decision.confidence,
                "language": turn_language,
                "script": turn_script,
                "explicit_language_override": turn_language_explicit,
                "language_source": turn_language_source,
                "router_language": route_decision.language,
                "router_script": route_decision.script,
                "router_explicit_language_override": route_decision.explicit_language_override,
                "tool_family": getattr(route_decision, "tool_family", "none"),
                "tool_candidates": list(getattr(route_decision, "tool_candidates", ()) or ()),
                "router_tool_family": getattr(original_route_decision, "tool_family", "none"),
                "router_tool_candidates": list(
                    getattr(original_route_decision, "tool_candidates", ()) or ()
                ),
                "classified_turn_intent": one_shot_turn_intent,
                "classified_turn_intent_kind": classified_turn_intent_kind,
                "route_override_reason": route_override_reason,
                **_local_materialization_payload(),
                "route_context": (
                    route_context.to_payload() if route_context is not None else None
                ),
            },
        )

        if route_decision.route != "repo":
            _phase_update_key("phase_drafting_response")
            non_repo_streamed_text_emitted = False

            def _on_non_repo_text_delta(delta: str) -> None:
                nonlocal non_repo_streamed_text_emitted
                _throw_if_cancelled()  # interruptible mid-stream (see _on_reasoning_delta)
                if delta:
                    _emit_message_delta_event(self.surface, delta)
                    non_repo_streamed_text_emitted = True
                self.surface.on_assistant_token(delta)

            final_assistant_message: dict[str, Any] | None = None
            final_text = _route_reply_for_non_repo_turn(
                route_decision,
                explicit_language_override=turn_language_explicit,
                recent_visible_history=recent_visible_non_repo_history,
            )
            if final_text:
                self.store.append(
                    "non_repo_router_reply_used",
                    {
                        "route": route_decision.route,
                        "decision_source": route_decision.decision_source,
                    },
                )
            else:
                non_repo_tools = (
                    _non_repo_tool_assisted_tools(
                        self.tools,
                        route_decision=route_decision,
                    )
                    if route_decision.route in {"general", "tool"}
                    else {}
                )
                non_repo_tool_list = [tool.as_openai_tool() for tool in non_repo_tools.values()]
                try:
                    respond_non_repo_turn = _patchable(
                        "_respond_non_repo_turn", _respond_non_repo_turn
                    )
                    non_repo_response = respond_non_repo_turn(
                        client=route_client,
                        instruction=instruction,
                        route=route_decision.route,
                        language=turn_language,
                        script=turn_script,
                        explicit_language_override=turn_language_explicit,
                        temperature=resolve_role_temperature(self.cfg, role="chat"),
                        recent_visible_history=recent_visible_non_repo_history,
                        tool_defs=non_repo_tools,
                        tool_list=non_repo_tool_list,
                        surface=self.surface,
                        store=self.store,
                        stream=self.stream,
                        on_text_delta=_on_non_repo_text_delta if self.stream else None,
                        on_reasoning_delta=(
                            _on_reasoning_delta
                            if (self.stream and _reasoning_sink_exists)
                            else None
                        ),
                    )
                    assistant_message_candidate = getattr(
                        non_repo_response,
                        "assistant_message",
                        None,
                    )
                    if isinstance(assistant_message_candidate, dict):
                        final_assistant_message = assistant_message_candidate
                    final_text = str(non_repo_response or "").strip()
                except LLMError as err:
                    _record_turn_llm_error(err)
                    raise
            if not final_text:
                try:
                    final_text = _fallback_route_decision(
                        instruction,
                        language=turn_language,
                        script=turn_script,
                        explicit_language_override=turn_language_explicit,
                        client=route_client,
                    ).reply
                except LLMError as err:
                    _record_turn_llm_error(err)
                    raise
            assistant_message_emitted = True
            if final_assistant_message is None:
                final_assistant_message = {"role": "assistant", "content": final_text}
            self.messages.append(final_assistant_message)
            self.store.append(
                "assistant_message",
                {"content": final_text, "message": final_assistant_message},
            )
            self.store.append("final", {"content": final_text})
            _emit_assistant_message_events(
                self.surface,
                final_text,
                streamed_text_emitted=non_repo_streamed_text_emitted,
            )
            self.surface.on_assistant_message_done(final_text)
            return _finish_turn(0, reason="non_repo_completed", final_text=final_text)
    else:
        one_shot_turn_intent = _classify_one_shot_repo_turn_intent(instruction)
        route_execution_posture = str(one_shot_turn_intent or "execute")
        self.store.append(
            "language_decision",
            {
                "language": turn_language,
                "script": turn_script,
                "confidence": 0.0,
                "explicit_language_override": turn_language_explicit,
                "language_source": turn_language_source,
                "failure_reason": turn_language_failure_reason,
            },
        )

    turn_language_system_message = _build_turn_language_system_message(
        turn_language,
        turn_script,
        explicit_language_override=turn_language_explicit,
    )
    if turn_language_system_message:
        self.messages.append({"role": "system", "content": turn_language_system_message})
        self.store.append(
            "system_note",
            {
                "message": "turn_language_script_directive",
                "language": turn_language,
                "script": turn_script,
                "explicit_language_override": turn_language_explicit,
                "language_source": turn_language_source,
            },
        )

    failed_tool_call_counts: dict[str, int] = {}
    adaptive_retry_keys_used: set[str] = set()
    repo_turn_execution_intent = _resolve_repo_turn_execution_intent(
        one_shot_execution=self.one_shot_execution,
        runtime_kind=self.runtime_kind,
        route_execution_posture=route_execution_posture,
        classified_turn_intent=one_shot_turn_intent,
    )
    execution_safeguards_enabled = repo_turn_execution_intent == "execute"
    resolved_turn_intent_kind = (
        "mutating_execution" if execution_safeguards_enabled else "read_only"
    )
    _refresh_execute_turn_verification_selection(
        self,
        instruction=instruction,
        route_execution_posture=repo_turn_execution_intent,
    )
    self.store.append(
        "turn_intent_resolved",
        {
            "classified_turn_intent": one_shot_turn_intent,
            "repo_turn_execution_intent": repo_turn_execution_intent,
            "turn_intent": resolved_turn_intent_kind,
            "request_intent": (
                "mutating_execution" if one_shot_turn_intent == "execute" else "read_only"
            ),
            "router_execution_posture": route_execution_posture,
            "execution_safeguards_enabled": execution_safeguards_enabled,
            **_local_materialization_payload(),
        },
    )
    turn_max_steps = max(1, int(self.max_steps))
    if self.enable_chat_turn_step_budget:
        turn_budget_resolution = resolve_step_budget(
            StepBudgetRequest(
                kind="chat_turn",
                policy=self.cfg.step_budget_policy,
                hard_cap=self.max_steps,
                fixed_override=self.chat_turn_fixed_override,
                mode=self.mode,
                route="repo",
                one_shot_execution=self.one_shot_execution,
                one_shot_turn_intent=repo_turn_execution_intent,
                verification_enabled=self.verification_enabled,
                subagents_enabled=self.subagents_enabled,
                explicit_path_count=len(
                    _extract_repo_relative_paths_from_text(
                        root=self.root,
                        text=instruction,
                    )
                ),
                image_count=len(image_paths or []),
            )
        )
        turn_max_steps = turn_budget_resolution.resolved_max_steps
        if self.step_budget_runtime is not None:
            self.step_budget_runtime.active_turn_budget = turn_max_steps
            self.step_budget_runtime.last_resolution = turn_budget_resolution
        self.store.append(
            "turn_step_budget_resolved",
            turn_budget_resolution.to_payload(),
        )
    known_verification_commands = list(self.effective_verification_commands)
    acceptance_contract = None
    if execution_safeguards_enabled:
        acceptance_contract = build_acceptance_contract(
            root=self.root,
            instruction=instruction,
            authoritative_verification_commands=(
                list(self.authoritative_verification_commands)
                if self.authoritative_verification_commands is not None
                else None
            ),
            effective_verification_commands=known_verification_commands,
            task_brief=_session_task_brief_content(self),
            repo_scan=_session_repo_scan(self),
            planning_constraints=getattr(self, "planning_scope_constraints", None),
        )
        self.store.append("acceptance_contract", acceptance_contract.as_payload())
    execution_state = TurnExecutionState(
        execution_requested=execution_safeguards_enabled,
        expected_verification_commands=set(known_verification_commands),
        acceptance_contract=acceptance_contract,
    )
    subagent_turn_policy = _resolve_subagent_turn_policy(
        instruction=instruction,
        subagents_enabled=self.subagents_enabled,
        enforce_explicit_request=self.enforce_explicit_subagent_requests,
        subagent_depth=self.subagent_depth,
        subagent_registry=self.subagent_registry,
        turn_tools=turn_tools,
        repo_turn_execution_intent=repo_turn_execution_intent,
    )
    subagent_required_nudges_sent = 0
    subagent_exploration_nudges_sent = 0
    subagent_attempt_count = 0
    if subagent_turn_policy.unavailable:
        final_text = _SUBAGENT_REQUEST_UNAVAILABLE_MESSAGE_TEMPLATE.format(
            reason=subagent_turn_policy.reason,
        )
        self.store.append(
            "subagent_request_unavailable",
            {
                "reason": subagent_turn_policy.reason,
                "available_subagents": list(subagent_turn_policy.available_subagents),
                "instruction": instruction,
            },
        )
        self._emit_final_assistant_text(
            final_text=final_text,
            language=turn_language,
            script=turn_script,
            explicit_language_override=turn_language_explicit,
        )
        assistant_message_emitted = True
        return _finish_turn(1, reason="subagent_request_unavailable", final_text=final_text)
    subagent_turn_context = _subagent_turn_context_message(subagent_turn_policy)
    if subagent_turn_context:
        ephemeral_turn_user_messages.append(subagent_turn_context)
        self.store.append(
            "subagent_turn_policy",
            {
                "level": subagent_turn_policy.level,
                "reason": subagent_turn_policy.reason,
                "available_subagents": list(subagent_turn_policy.available_subagents),
            },
        )
    execution_follow_through_enabled = (
        self.subagent_depth == 0
        and execution_safeguards_enabled
        and (
            self.one_shot_execution
            or (
                self.runtime_kind == RuntimeKind.INTERACTIVE_CHAT
                and self.enable_chat_turn_step_budget
            )
        )
    )
    interactive_step_budget_handoff_enabled = (
        execution_follow_through_enabled
        and not self.one_shot_execution
        and self.runtime_kind == RuntimeKind.INTERACTIVE_CHAT
    )
    completion_gate_enabled = execution_follow_through_enabled
    execution_phase_tracking_enabled = execution_follow_through_enabled
    completion_gate_failed_event = (
        "one_shot_completion_gate_failed"
        if self.one_shot_execution
        else "interactive_completion_gate_failed"
    )
    completion_gate_incomplete_event = (
        "one_shot_completion_gate_incomplete_after_retries"
        if self.one_shot_execution
        else "interactive_completion_gate_incomplete_after_retries"
    )
    no_material_edits_detected_event = (
        "one_shot_no_material_edits_detected"
        if self.one_shot_execution
        else "interactive_no_material_edits_detected"
    )
    no_material_edits_incomplete_event = (
        "one_shot_no_material_edits_incomplete_after_retries"
        if self.one_shot_execution
        else "interactive_no_material_edits_incomplete_after_retries"
    )
    completion_gate_nudge_prefix_key = (
        "completion_gate_nudge_prefix"
        if self.one_shot_execution
        else "interactive_completion_gate_nudge_prefix"
    )
    completion_gate_terminal_message_key = (
        "completion_gate_terminal_failure"
        if self.one_shot_execution
        else "interactive_completion_gate_terminal_failure"
    )
    completion_gate_step_budget_message_key = (
        "completion_gate_step_budget_exhausted"
        if self.one_shot_execution
        else "interactive_completion_gate_step_budget_exhausted"
    )
    non_final_progress_detected_event = (
        "one_shot_non_final_progress_detected"
        if self.one_shot_execution
        else "interactive_non_final_progress_detected"
    )
    non_final_incomplete_event = (
        "one_shot_incomplete_after_retries"
        if self.one_shot_execution
        else "interactive_incomplete_after_retries"
    )
    non_final_progress_stopped_key = (
        "one_shot_non_final_progress_stopped"
        if self.one_shot_execution
        else "interactive_non_final_progress_stopped"
    )
    continuation_nudge_key = (
        "one_shot_continuation_nudge"
        if self.one_shot_execution
        else "interactive_continuation_nudge"
    )
    one_shot_exploration_guard_enabled = (
        self.one_shot_execution and self.subagent_depth == 0 and execution_safeguards_enabled
    )
    one_shot_edit_guard_enabled = one_shot_exploration_guard_enabled
    consecutive_exploration_only_steps = 0
    exploration_nudges_sent = 0
    exploration_attempt_call_counts: dict[str, int] = {}
    exploration_attempt_similarity_counts: dict[str, int] = {}
    consecutive_exploration_success_count = 0
    consecutive_exploration_failed_count = 0
    last_exploration_stagnation_payload: dict[str, Any] | None = None
    subagent_success_count = 0
    post_explore_action_progress_started = False
    post_explore_bootstrap_nudges_sent = 0
    recent_exploration_paths: list[str] = []
    repo_tool_activity_observed = False
    repo_read_only_tool_activity_observed = False
    repo_action_tool_activity_observed = False
    repo_unknown_tool_activity_observed = False
    first_turn_repo_grounding_retry_sent = False
    last_post_explore_stagnation_payload: dict[str, Any] | None = None
    consecutive_failed_edit_steps = 0
    edit_nudges_sent = 0
    failed_edit_attempt_call_counts: dict[str, int] = {}
    failed_edit_similarity_counts: dict[str, int] = {}
    consecutive_failed_edit_attempt_count = 0
    last_edit_stagnation_payload: dict[str, Any] | None = None
    last_nudge_text_sent = ""
    empty_response_anomaly_state = _EmptyResponseAnomalyRecoveryState()
    forced_tool_choice_for_next_step: dict[str, Any] | None = None
    finalization_empty_anomaly_recovery_pending = False

    def _observed_repo_tool_intent() -> str:
        if not repo_tool_activity_observed:
            return "none"
        if (
            repo_read_only_tool_activity_observed
            and not repo_action_tool_activity_observed
            and not repo_unknown_tool_activity_observed
            and execution_state.material_edit_count <= 0
            and execution_state.verification_attempt_count <= 0
        ):
            return "read_only"
        return "mutating_or_execution"

    def _completion_gate_repo_turn_execution_intent(final_text: str) -> _OneShotRepoTurnIntent:
        observed_intent = _observed_repo_tool_intent()
        if (
            not self.one_shot_execution
            and repo_turn_execution_intent == "execute"
            and observed_intent == "read_only"
            and not _assistant_text_has_completion_marker(final_text)
        ):
            return "read_only"
        return repo_turn_execution_intent

    def _completion_gate_requires_material_edit_evidence(
        *,
        final_text: str,
        gate_turn_intent: _OneShotRepoTurnIntent,
    ) -> bool:
        if self.one_shot_execution:
            return True
        if _assistant_text_has_completion_marker(final_text):
            return True
        return gate_turn_intent == "execute" and repo_tool_activity_observed

    def _turn_intent_payload(
        *,
        completion_gate_turn_intent: _OneShotRepoTurnIntent | None = None,
    ) -> dict[str, Any]:
        payload = {
            "classified_turn_intent": one_shot_turn_intent,
            "repo_turn_execution_intent": repo_turn_execution_intent,
            "turn_intent": resolved_turn_intent_kind,
            "observed_tool_intent": _observed_repo_tool_intent(),
            "repo_tool_activity_observed": repo_tool_activity_observed,
            "repo_read_only_tool_activity_observed": repo_read_only_tool_activity_observed,
            "repo_action_tool_activity_observed": repo_action_tool_activity_observed,
            "repo_unknown_tool_activity_observed": repo_unknown_tool_activity_observed,
            **_local_materialization_payload(),
        }
        if completion_gate_turn_intent is not None:
            payload["completion_gate_turn_intent"] = completion_gate_turn_intent
        return payload

    def _nudge_would_repeat_without_progress(
        message: str,
        decision: CompletionGateDecision,
    ) -> bool:
        return bool(
            message
            and message == last_nudge_text_sent
            and not decision.meaningful_progress_since_previous_rejection
        )

    def _assistant_text_is_clarification_only(text: str) -> bool:
        raw_text = str(text or "").strip()
        if not raw_text:
            return False
        if _assistant_text_has_completion_marker(raw_text):
            return False
        if _assistant_text_contains_progress_intent(raw_text):
            return False
        normalized = _normalize_marker_text(raw_text)
        return bool(_CLARIFICATION_ONLY_RE.search(normalized))

    def _clarification_can_finalize(*, text: str) -> bool:
        if not self.one_shot_execution:
            if (
                repo_turn_execution_intent == "execute"
                and local_materialization_requirement.required
                and local_materialization_requirement.confidence >= 0.8
            ):
                return bool(_CLARIFICATION_VALID_BLOCKER_RE.search(_normalize_marker_text(text)))
            return True
        if repo_turn_execution_intent != "execute":
            return True
        if one_shot_turn_intent != "execute":
            return True
        if (
            local_materialization_requirement.required
            and local_materialization_requirement.confidence >= 0.8
        ):
            normalized = _normalize_marker_text(text)
            return bool(_CLARIFICATION_VALID_BLOCKER_RE.search(normalized))
        normalized = _normalize_marker_text(text)
        return bool(_CLARIFICATION_VALID_BLOCKER_RE.search(normalized))

    def _empty_response_missing_action() -> str:
        if execution_state.material_edit_count <= 0:
            if local_materialization_requirement.output_paths:
                return "implement required output"
            return "edit a relevant path"
        if _verification_expected_for_turn(
            turn_intent=repo_turn_execution_intent,
            blocked=False,
            touched_repo_paths=execution_state.touched_repo_paths,
            verification_contract_requires_execution=(
                self.verification_contract_type
                in {"authoritative_override", "explicit_override", "task_inferred"}
            ),
        ) and (
            execution_state.verification_attempt_count <= 0
            or execution_state.missing_verification_commands()
            or execution_state.verification_coverage_is_stale()
        ):
            return "run required verification"
        return "report a concrete blocker or final result"

    def _preferred_recovery_tool_names(missing_action: str) -> tuple[str, ...]:
        if missing_action == "run required verification":
            return ("verify_run",)
        if missing_action == "implement required output":
            return ("fs_write",)
        return tuple()

    def _empty_response_recovery_message(missing_action: str) -> str:
        anchor_paths = list(local_materialization_requirement.output_paths)
        if not anchor_paths:
            anchor_paths = recent_exploration_paths[-MAX_POST_EXPLORE_ANCHOR_PATHS:]
        anchor_text = ", ".join(anchor_paths[:MAX_POST_EXPLORE_ANCHOR_PATHS])
        if not anchor_text:
            anchor_text = "(none)"
        return (
            "Model-control recovery: the previous assistant response was empty after tool "
            "results. Do not provide hidden reasoning. Take exactly one concrete action now: "
            f"{missing_action}. Use the appropriate tool call if possible; otherwise report a "
            f"concrete blocker with evidence. Anchor paths: {anchor_text}."
        )

    def _build_completion_gate_decision(
        *,
        stage: str,
        problems: list[str],
        final_text: str,
        blocked_response: bool = False,
        blocked_response_allows_completion: bool = False,
        verification_expected: bool = False,
        budget_exhausted: bool = False,
    ) -> CompletionGateDecision:
        snapshot = build_completion_gate_snapshot(
            stage=stage,
            problems=problems,
            material_edit_count=execution_state.material_edit_count,
            material_edit_tools=execution_state.material_edit_tools,
            touched_repo_paths=execution_state.touched_repo_paths,
            verification_relevant_edit_generation=(
                execution_state.verification_relevant_edit_generation
            ),
            last_successful_verification_generation=(
                execution_state.last_successful_verification_generation
            ),
            expected_verification_commands=execution_state.expected_verification_commands,
            covered_verification_commands=execution_state.covered_verification_commands,
            missing_verification_commands=execution_state.missing_verification_commands(),
            failed_verification_command_snippets=(
                execution_state.failed_verification_command_snippets
            ),
            verification_coverage_stale=execution_state.verification_coverage_is_stale(),
            last_verification_passed=execution_state.last_verification_passed,
            last_verification_failure_category=(execution_state.last_verification_failure_category),
            accepted_blocker=blocked_response_allows_completion,
            blocked_response=blocked_response,
            blocked_response_allows_completion=blocked_response_allows_completion,
            verification_expected=verification_expected,
            final_text=final_text,
            repo_tool_activity_observed=repo_tool_activity_observed,
            acceptance_status_counts=(
                execution_state.acceptance_contract.status_counts()
                if execution_state.acceptance_contract is not None
                else {}
            ),
            acceptance_problems=(
                execution_state.acceptance_contract.problem_names()
                if execution_state.acceptance_contract is not None
                else []
            ),
            acceptance_failure_summaries=(
                execution_state.acceptance_contract.failure_summaries()
                if execution_state.acceptance_contract is not None
                else []
            ),
        )
        return decide_completion_gate(
            execution_state.completion_gate_controller_state,
            snapshot,
            budget_exhausted=budget_exhausted,
        )

    def _completion_gate_decision_fields(
        decision: CompletionGateDecision,
    ) -> dict[str, Any]:
        payload = completion_gate_decision_payload(decision)
        return {
            "decision": payload["decision"],
            "completion_gate_decision": payload["decision"],
            "completion_gate_episode_id": payload["episode_id"],
            "completion_gate_previous_episode_id": payload["previous_episode_id"],
            "completion_gate_stagnant_attempt_count": payload["stagnant_attempt_count"],
            "completion_gate_meaningful_progress_since_previous_rejection": payload[
                "meaningful_progress_since_previous_rejection"
            ],
            "completion_gate_decision_reason": payload["reason"],
            "completion_gate_nudge_reason": payload["reason"],
            "nudge_reason": payload["reason"],
            "completion_gate_consecutive_no_progress_rejections": payload[
                "consecutive_no_progress_rejections"
            ],
            "completion_gate_max_stagnant_attempts": payload["max_stagnant_attempts"],
            "completion_gate_controller": payload,
        }

    def _verification_evidence_fields() -> dict[str, Any]:
        return {
            "verification_evidence_category": (
                execution_state.latest_verification_evidence_category
            ),
            "verification_evidence_reason": execution_state.latest_verification_evidence_reason,
            "verification_evidence_counts": dict(execution_state.verification_evidence_counts),
            "verification_evidence_generation": execution_state.verification_evidence_generation,
        }

    def _acceptance_contract_fields() -> dict[str, Any]:
        return acceptance_contract_problem_payload(execution_state.acceptance_contract)

    for step in range(1, turn_max_steps + 1):
        _throw_if_cancelled()
        steps_attempted = step
        stream_used = self.stream
        step_ephemeral_suffix_system_messages: list[str] = []
        remaining_tool_steps_after_this = turn_max_steps - step
        if step == turn_max_steps:
            step_ephemeral_suffix_system_messages.append(_FINAL_TOOL_ENABLED_STEP_SYSTEM_PROMPT)
        elif 0 < remaining_tool_steps_after_this <= 3:
            step_ephemeral_suffix_system_messages.append(
                _LOW_STEP_BUDGET_SYSTEM_PROMPT_TEMPLATE.format(
                    remaining_steps=remaining_tool_steps_after_this
                )
            )
        if (
            execution_follow_through_enabled
            and consecutive_exploration_only_steps >= 3
            and execution_state.material_edit_count <= 0
        ):
            step_ephemeral_suffix_system_messages.append(
                _PHASE_BUDGET_EXPLORATION_SYSTEM_PROMPT_TEMPLATE.format(
                    exploration_steps=consecutive_exploration_only_steps
                )
            )
        elif (
            execution_follow_through_enabled
            and execution_state.material_edit_count > 0
            and execution_state.verification_attempt_count <= 0
            and 0 < remaining_tool_steps_after_this <= 5
        ):
            step_ephemeral_suffix_system_messages.append(
                _PHASE_BUDGET_VERIFICATION_SYSTEM_PROMPT_TEMPLATE.format(
                    remaining_steps=remaining_tool_steps_after_this
                )
            )
        _append_deadline_finalization_prompt(step_ephemeral_suffix_system_messages, step=step)

        def _request_messages_for_step(
            messages: list[dict[str, Any]],
            turn_prompts: tuple[str, ...] = tuple(ephemeral_turn_system_messages),
            user_context_messages: tuple[str, ...] = tuple(ephemeral_turn_user_messages),
            suffix_prompts: tuple[str, ...] = tuple(step_ephemeral_suffix_system_messages),
        ) -> list[dict[str, Any]]:
            # Keep turn-level wrappers anchored at the turn boundary, but
            # append step-dynamic nudges at the end so the reusable prefix
            # remains stable for prompt-cache matching.
            request_messages = _request_messages_with_ephemeral_system_prompts(
                messages=messages,
                insert_index=turn_start_messages,
                prompts=list(turn_prompts),
            )
            request_messages = _request_messages_with_ephemeral_user_messages(
                messages=request_messages,
                insert_index=(turn_user_message_index + len(turn_prompts)),
                contents=list(user_context_messages),
            )
            return _request_messages_with_ephemeral_system_prompt_suffixes(
                messages=request_messages,
                prompts=list(suffix_prompts),
            )

        streamed_text_emitted = False

        def _on_text_delta(delta: str) -> None:
            nonlocal streamed_text_emitted
            _throw_if_cancelled()  # interruptible mid-stream (see _on_reasoning_delta)
            if delta:
                _emit_message_delta_event(self.surface, delta)
                streamed_text_emitted = True
            self.surface.on_assistant_token(delta)

        # Reasoning sink (``_on_reasoning_delta``) is defined once before the route
        # split above; here we only gate it on this step's effective stream mode.
        _reasoning_enabled = stream_used and _reasoning_sink_exists

        request_messages = _request_messages_for_step(self.messages)
        try:
            if self.conversation_compactor is not None:
                if not _deadline_allows(
                    DeadlineOperation.COMPACTION_LLM,
                    minimum_remaining_seconds=MINIMUM_LLM_START_SECONDS,
                ):
                    return _deadline_exhausted_result("compaction_llm", step=step)
                pre_compact_message_count = len(self.messages)
                compactor_client = getattr(
                    self.conversation_compactor,
                    "compactor_client",
                    None,
                )
                try:
                    operation_started = perf_counter()
                    with temporarily_clamp_client_timeout(
                        compactor_client,
                        deadline,
                        operation="compaction_llm",
                    ):
                        compacted_messages, compacted = self.conversation_compactor.maybe_compact(
                            messages=self.messages,
                            tool_list=turn_tool_list,
                            main_model=self.client.model,
                            focus=instruction,
                        )
                    _record_deadline_duration(
                        DeadlineOperation.COMPACTION_LLM,
                        operation_started,
                    )
                except DeadlineExhausted:
                    return _deadline_exhausted_result("compaction_llm", step=step)
                self.messages = compacted_messages
                if deadline is not None and deadline.is_exhausted():
                    return _deadline_exhausted_result("compaction_llm", step=step)
                if compacted:
                    _phase_update_key("phase_compacted_history")
                    if self.hook_dispatcher is not None:
                        cwd, active_workdir_relpath = self._hook_runtime_context()
                        post_compact_message_count = len(compacted_messages)
                        self._safe_dispatch_hooks(
                            lambda hook_cwd=cwd, hook_relpath=active_workdir_relpath, pre_count=pre_compact_message_count, post_count=post_compact_message_count: (
                                self.hook_dispatcher.fire_pre_compact(  # type: ignore[union-attr]
                                    cwd=hook_cwd,
                                    active_workdir_relpath=hook_relpath,
                                    trigger="compaction_applied",
                                    message_count=pre_count,
                                    payload={
                                        "pre_compact_message_count": pre_count,
                                        "post_compact_message_count": post_count,
                                    },
                                )
                            )
                        )
            _append_deadline_finalization_prompt(step_ephemeral_suffix_system_messages, step=step)
            request_messages = _request_messages_for_step(
                self.messages,
                suffix_prompts=tuple(step_ephemeral_suffix_system_messages),
            )
            main_llm_in_finalization = (
                deadline is not None and deadline.phase() == DeadlinePhase.FINALIZATION_WINDOW
            )
            if (
                main_llm_in_finalization
                and deadline.finalization_llm_started
                and not finalization_empty_anomaly_recovery_pending
            ):
                return _deadline_exhausted_result("main_llm_finalization_spent", step=step)
            if not _deadline_allows(
                DeadlineOperation.MAIN_LLM,
                minimum_remaining_seconds=MINIMUM_LLM_START_SECONDS,
                allow_during_finalization=True,
            ):
                return _deadline_exhausted_result("main_llm", step=step)
            if main_llm_in_finalization and finalization_empty_anomaly_recovery_pending:
                finalization_empty_anomaly_recovery_pending = False
            elif main_llm_in_finalization:
                deadline.mark_finalization_llm_started()
            _diagnostic_event(
                "llm_started",
                {"operation": "main_llm", "step": step, "deadline": _deadline_snapshot()},
            )
            operation_started = perf_counter()
            request_tool_choice = forced_tool_choice_for_next_step
            forced_tool_choice_for_next_step = None
            with temporarily_clamp_client_timeout(
                self.client,
                deadline,
                operation="main_llm",
            ):
                resp = _main_agent_chat(
                    client=self.client,
                    messages=request_messages,
                    tools=turn_tool_list,
                    stream=stream_used,
                    on_text_delta=_on_text_delta if stream_used else None,
                    on_reasoning_delta=_on_reasoning_delta if _reasoning_enabled else None,
                    cancellation_token=cancellation_token,
                    tool_choice=request_tool_choice,
                )
            _record_deadline_duration(DeadlineOperation.MAIN_LLM, operation_started)
            _diagnostic_event(
                "llm_completed",
                {"operation": "main_llm", "step": step, "deadline": _deadline_snapshot()},
            )
            if deadline is not None and deadline.is_exhausted():
                return _deadline_exhausted_result("main_llm", step=step)
        except DeadlineExhausted:
            return _deadline_exhausted_result("main_llm", step=step)
        except LLMError as e:
            if stream_used and _is_stream_unsupported_error(e):
                self.store.append(
                    "warning",
                    {"warning": "stream_not_supported", "error": str(e)},
                )
                progress_handler = getattr(self.surface, "on_progress_update", None)
                if callable(progress_handler):
                    progress_handler("Streaming not supported; retrying without stream.")
                try:
                    if not _deadline_allows(
                        DeadlineOperation.MAIN_LLM_RETRY,
                        minimum_remaining_seconds=MINIMUM_LLM_START_SECONDS,
                    ):
                        return _deadline_exhausted_result("main_llm_retry", step=step)
                    _diagnostic_event(
                        "llm_started",
                        {
                            "operation": "main_llm_retry",
                            "step": step,
                            "deadline": _deadline_snapshot(),
                        },
                    )
                    operation_started = perf_counter()
                    with temporarily_clamp_client_timeout(
                        self.client,
                        deadline,
                        operation="main_llm_retry",
                    ):
                        resp = _main_agent_chat(
                            client=self.client,
                            messages=request_messages,
                            tools=turn_tool_list,
                            stream=False,
                            on_text_delta=None,
                            on_reasoning_delta=None,
                            cancellation_token=cancellation_token,
                            tool_choice=request_tool_choice,
                        )
                    _record_deadline_duration(
                        DeadlineOperation.MAIN_LLM_RETRY,
                        operation_started,
                    )
                    _diagnostic_event(
                        "llm_completed",
                        {
                            "operation": "main_llm_retry",
                            "step": step,
                            "deadline": _deadline_snapshot(),
                        },
                    )
                    if deadline is not None and deadline.is_exhausted():
                        return _deadline_exhausted_result("main_llm_retry", step=step)
                except DeadlineExhausted:
                    return _deadline_exhausted_result("main_llm_retry", step=step)
                except LLMError as retry_err:
                    _diagnostic_event(
                        "llm_failed",
                        {
                            "operation": "main_llm_retry",
                            "step": step,
                            "failure_category": "llm_error",
                            "deadline": _deadline_snapshot(),
                        },
                    )
                    self.store.append("error", {"error": str(retry_err)})
                    _rollback_turn_after_llm_error()
                    raise
                stream_used = False
            else:
                _diagnostic_event(
                    "llm_failed",
                    {
                        "operation": "main_llm",
                        "step": step,
                        "failure_category": "llm_error",
                        "deadline": _deadline_snapshot(),
                    },
                )
                self.store.append("error", {"error": str(e)})
                _rollback_turn_after_llm_error()
                raise

        usage = resp.usage
        usage_record = build_usage_record(
            role=self.usage_role,
            requested_model=self.client.model,
            response_model=resp.response_model,
            messages=request_messages,
            response_content=resp.content or "",
            response_tool_calls=[
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in resp.tool_calls
            ],
            api_prompt_tokens=(usage.prompt_tokens if usage else None),
            api_completion_tokens=(usage.completion_tokens if usage else None),
            api_total_tokens=(usage.total_tokens if usage else None),
            api_cached_prompt_tokens=(usage.cached_prompt_tokens if usage else None),
            tool_list=turn_tool_list,
            pinned_prefix_len=self.pinned_prefix_len,
            registry=self.model_registry,
        )
        self.usage_summary.add_record(usage_record)
        self.store.append("llm_usage", usage_record.to_payload())

        tool_calls = resp.tool_calls
        has_invalid_tool_arguments_json = _has_invalid_tool_call_json(tool_calls)
        adaptive_retry_keys = [
            key
            for key in (_tool_call_retry_key(tc.name, tc.arguments) for tc in tool_calls)
            if failed_tool_call_counts.get(key, 0) >= MAX_IDENTICAL_TOOL_CALL_FAILURES
            and key not in adaptive_retry_keys_used
        ]
        adaptive_retry_reason: str | None = None
        if has_invalid_tool_arguments_json:
            adaptive_retry_reason = "invalid_tool_arguments_json"
        elif adaptive_retry_keys:
            adaptive_retry_reason = "repeated_tool_error"
        if adaptive_retry_reason is not None:
            adaptive_tools = [
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in tool_calls
            ]
            self.store.append(
                "warning",
                {
                    "warning": "adaptive_temperature_retry",
                    "reason": adaptive_retry_reason,
                    "step": step,
                    "temperature": _ADAPTIVE_RETRY_TEMPERATURE,
                    "tool_calls": adaptive_tools,
                },
            )
            _phase_update_key("phase_retrying_step")
            try:
                if not _deadline_allows(
                    DeadlineOperation.ADAPTIVE_RETRY_LLM,
                    minimum_remaining_seconds=MINIMUM_LLM_START_SECONDS,
                ):
                    return _deadline_exhausted_result("adaptive_retry_llm", step=step)
                _diagnostic_event(
                    "llm_started",
                    {
                        "operation": "adaptive_retry_llm",
                        "step": step,
                        "deadline": _deadline_snapshot(),
                    },
                )
                operation_started = perf_counter()
                with temporarily_clamp_client_timeout(
                    self.client,
                    deadline,
                    operation="adaptive_retry_llm",
                ):
                    retry_resp = _main_agent_chat(
                        client=self.client,
                        messages=_request_messages_for_step(self.messages),
                        tools=turn_tool_list,
                        stream=False,
                        on_text_delta=None,
                        temperature=_ADAPTIVE_RETRY_TEMPERATURE,
                    )
                _record_deadline_duration(
                    DeadlineOperation.ADAPTIVE_RETRY_LLM,
                    operation_started,
                )
                _diagnostic_event(
                    "llm_completed",
                    {
                        "operation": "adaptive_retry_llm",
                        "step": step,
                        "deadline": _deadline_snapshot(),
                    },
                )
                if deadline is not None and deadline.is_exhausted():
                    return _deadline_exhausted_result("adaptive_retry_llm", step=step)
            except DeadlineExhausted:
                return _deadline_exhausted_result("adaptive_retry_llm", step=step)
            except LLMError as adaptive_retry_error:
                _diagnostic_event(
                    "llm_failed",
                    {
                        "operation": "adaptive_retry_llm",
                        "step": step,
                        "failure_category": "llm_error",
                        "deadline": _deadline_snapshot(),
                    },
                )
                self.store.append(
                    "warning",
                    {
                        "warning": "adaptive_temperature_retry_failed",
                        "step": step,
                        "error": str(adaptive_retry_error),
                    },
                )
            else:
                retry_usage = retry_resp.usage
                retry_usage_record = build_usage_record(
                    role=self.usage_role,
                    requested_model=self.client.model,
                    response_model=retry_resp.response_model,
                    messages=request_messages,
                    response_content=retry_resp.content or "",
                    response_tool_calls=[
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in retry_resp.tool_calls
                    ],
                    api_prompt_tokens=(retry_usage.prompt_tokens if retry_usage else None),
                    api_completion_tokens=(retry_usage.completion_tokens if retry_usage else None),
                    api_total_tokens=(retry_usage.total_tokens if retry_usage else None),
                    api_cached_prompt_tokens=(
                        retry_usage.cached_prompt_tokens if retry_usage else None
                    ),
                    tool_list=turn_tool_list,
                    pinned_prefix_len=self.pinned_prefix_len,
                    registry=self.model_registry,
                )
                self.usage_summary.add_record(retry_usage_record)
                self.store.append("llm_usage", retry_usage_record.to_payload())
                if adaptive_retry_reason == "repeated_tool_error":
                    adaptive_retry_keys_used.update(adaptive_retry_keys)
                resp = retry_resp
                tool_calls = retry_resp.tool_calls
        if tool_calls:
            repo_tool_activity_observed = True
            names = ", ".join(tc.name for tc in tool_calls[:3])
            if len(tool_calls) > 3:
                names += ", ..."
            assistant_message = assistant_message_from_response(resp)
            _phase_update_key(
                "phase_running_tool_steps",
                count=len(tool_calls),
                names=names,
            )
            last_visible_assistant_text = self._emit_assistant_message_if_changed(
                text=resp.content or "",
                prior_visible_text=last_visible_assistant_text,
                extra_payload={
                    "tool_calls": [tc.name for tc in tool_calls],
                    "message": assistant_message,
                },
                streamed_text_emitted=streamed_text_emitted,
            )
            assistant_message_emitted = True
            self.messages.append(assistant_message)

            step_had_action_progress = False
            step_had_successful_action_progress = False
            step_exploration_attempt_count = 0
            step_exploration_success_count = 0
            step_exploration_failed_count = 0
            step_repeated_exploration_pattern = False
            repeated_exploration_tool: str | None = None
            repeated_exploration_key: str | None = None
            step_failed_edit_attempt_count = 0
            step_successful_edit_attempt_count = 0
            step_repeated_failed_edit_pattern = False
            repeated_failed_edit_tool: str | None = None
            repeated_failed_edit_key: str | None = None
            step_failed_edit_errors: list[str] = []
            step_tool_names = [tc.name for tc in tool_calls]
            for step_tool_name in step_tool_names:
                if _is_exploration_only_tool(step_tool_name):
                    repo_read_only_tool_activity_observed = True
                elif _is_action_progress_tool(step_tool_name):
                    repo_action_tool_activity_observed = True
                else:
                    repo_unknown_tool_activity_observed = True
            same_batch_read_cache = _SameBatchReadReuseCache()
            parallel_subagent_executor: ThreadPoolExecutor | None = None
            parallel_subagent_futures: dict[str, Future[Any]] = {}

            def _shutdown_parallel_subagent_executor() -> None:
                nonlocal parallel_subagent_executor
                if parallel_subagent_executor is None:
                    return
                parallel_subagent_executor.shutdown(wait=True, cancel_futures=False)
                parallel_subagent_executor = None

            if _can_prelaunch_parallel_subagent_batch(
                tool_calls=tool_calls,
                turn_tools=turn_tools,
                subagent_registry=self.subagent_registry,
                failed_tool_call_counts=failed_tool_call_counts,
                hook_dispatcher=self.hook_dispatcher,
                subagent_policy_reason=subagent_turn_policy.reason,
                deadline_can_start=_deadline_allows(
                    DeadlineOperation.SUBAGENT,
                    minimum_remaining_seconds=MINIMUM_TOOL_START_SECONDS,
                ),
            ):
                parallel_subagent_executor = ThreadPoolExecutor(
                    max_workers=min(MAX_PARALLEL_SUBAGENT_TOOL_CALLS, len(tool_calls)),
                    thread_name_prefix="subagent-run",
                )
                parallel_subagent_futures = {
                    tc.id: parallel_subagent_executor.submit(
                        turn_tools[tc.name].run,
                        copy.deepcopy(tc.arguments),
                    )
                    for tc in tool_calls
                }

            for tc in tool_calls:
                if not _deadline_allows(
                    DeadlineOperation.TOOL_DISPATCH,
                    minimum_remaining_seconds=MINIMUM_TOOL_START_SECONDS,
                ):
                    _shutdown_parallel_subagent_executor()
                    return _deadline_exhausted_result("tool_dispatch", step=step)
                retry_key = _tool_call_retry_key(tc.name, tc.arguments)
                prior_failures = failed_tool_call_counts.get(retry_key, 0)
                tool = turn_tools.get(tc.name)
                effective_tool_name = tc.name
                alias_recovery_payload: dict[str, Any] | None = None
                alias = None
                if tool is None:
                    alias = compatibility_tool_alias_for(
                        requested_tool_name=tc.name,
                        arguments=tc.arguments,
                        available_tool_names=turn_tools.keys(),
                    )
                    if alias is not None:
                        effective_tool_name = alias.target
                        tool = turn_tools.get(effective_tool_name)
                        alias_recovery_payload = {
                            "requested_tool_name": tc.name,
                            "executed_tool_name": effective_tool_name,
                            "alias": alias.alias,
                            "target": alias.target,
                            "description": alias.description,
                        }
                tool_deadline_operation = _deadline_operation_for_tool_name(effective_tool_name)
                tool_call_payload: dict[str, Any] = {
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "tool_call_id": tc.id,
                    "step": step,
                }
                if alias_recovery_payload is not None:
                    tool_call_payload["compatibility_alias"] = alias_recovery_payload
                tool_call_payload.update(_tool_event_metadata(tool))
                if str(tc.name or "").strip().lower() == "subagent_run":
                    subagent_attempt_count += 1
                self.store.append(
                    "tool_call",
                    tool_call_payload,
                )
                _emit_tool_call_started_event(
                    self.surface,
                    call_id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments,
                )
                self.surface.on_tool_start(
                    ToolStartEvent(
                        tool_call_id=tc.id,
                        name=tc.name,
                        args=tc.arguments,
                        step=step,
                    )
                )
                _diagnostic_event(
                    "tool_started",
                    {
                        "tool_name": tc.name,
                        "step": step,
                        "deadline": _deadline_snapshot(),
                    },
                )
                t0 = perf_counter()
                effective_tool_arguments = (
                    transform_compatibility_tool_alias(alias, tc.arguments)
                    if alias is not None
                    else copy.deepcopy(tc.arguments)
                )
                hook_runtime_system_messages: list[str] = []
                hook_runtime_user_messages: list[str] = []
                pre_tool_blocked = False
                tool_executed_for_deadline_observation = False
                unavailable_result = unavailable_tool_result(effective_tool_name)
                subagent_blocked_by_turn_policy = (
                    str(tc.name or "").strip().lower() == "subagent_run"
                    and subagent_turn_policy.reason == "user_opt_out"
                )
                if subagent_blocked_by_turn_policy:
                    result = {
                        "error": (
                            "subagent_run is disabled for this turn because the user "
                            "explicitly requested no subagents."
                        )
                    }
                elif unavailable_result is not None:
                    result = unavailable_result
                elif prior_failures >= MAX_IDENTICAL_TOOL_CALL_FAILURES:
                    result = {
                        "error": (
                            "Blocked repeated tool call after "
                            f"{prior_failures} failures with identical arguments: {tc.name}. "
                            "Change strategy before retrying."
                        )
                    }
                    self.store.append(
                        "warning",
                        {
                            "warning": "repeated_tool_failure_guard",
                            "tool": tc.name,
                            "step": step,
                            "failures": prior_failures,
                        },
                    )
                elif not tool:
                    result = build_unknown_tool_recovery_payload(
                        requested_tool_name=tc.name,
                        arguments=tc.arguments,
                        available_tool_names=turn_tools.keys(),
                    )
                    hook_runtime_system_messages.append(str(result.get("guidance") or ""))
                    self.store.append(
                        "unknown_tool_recovery",
                        {
                            "step": step,
                            "tool_call_id": tc.id,
                            "requested_tool_name": tc.name,
                            "available_tool_names": result.get("available_tool_names", []),
                            "nearest_tool_suggestions": result.get(
                                "nearest_tool_suggestions",
                                [],
                            ),
                            "safe_compatibility_alias": bool(
                                result.get("safe_compatibility_alias")
                            ),
                            "alias_ambiguous": bool(result.get("alias_ambiguous")),
                        },
                    )
                else:
                    deadline_decision = _deadline_decision_payload(
                        tool_deadline_operation,
                        minimum_remaining_seconds=MINIMUM_TOOL_START_SECONDS,
                        allow_during_finalization=tool_deadline_operation
                        in {
                            DeadlineOperation.MUTATION_TOOL,
                            DeadlineOperation.VERIFICATION,
                        },
                    )
                    if deadline_decision is not None and not bool(deadline_decision.get("allowed")):
                        result = {
                            "error": (
                                f"{tc.name} skipped because the run deadline policy blocked "
                                f"{deadline_decision.get('operation')}: "
                                f"{deadline_decision.get('reason')}"
                            ),
                            "deadline_prevented_launch": True,
                            "deadline_start_decision": deadline_decision,
                            "deadline": _deadline_snapshot(),
                            "failure_category": "deadline",
                            "remaining_seconds": (
                                deadline.remaining_seconds() if deadline is not None else None
                            ),
                            "deadline_exhausted": (
                                deadline.is_exhausted() if deadline is not None else False
                            ),
                        }
                        self.store.append(
                            "deadline_operation_blocked",
                            {
                                "tool": tc.name,
                                "operation": deadline_decision.get("operation"),
                                "reason": deadline_decision.get("reason"),
                                "step": step,
                                "decision": deadline_decision,
                                "deadline": _deadline_snapshot(),
                            },
                        )
                        _diagnostic_event(
                            "deadline_operation_blocked",
                            {
                                "tool_name": tc.name,
                                "operation": deadline_decision.get("operation"),
                                "reason": deadline_decision.get("reason"),
                                "step": step,
                                "deadline": _deadline_snapshot(),
                                "deadline_start_decision": deadline_decision,
                            },
                        )
                    else:
                        tool_executed_for_deadline_observation = True
                        result = None
                if result is None:
                    cwd, active_workdir_relpath = self._hook_runtime_context()
                    pre_tool_hook_result = self._safe_dispatch_hooks(
                        lambda tool_name=effective_tool_name, tool_input=copy.deepcopy(effective_tool_arguments), hook_cwd=cwd, hook_relpath=active_workdir_relpath, hook_step=step: (
                            self.hook_dispatcher.fire_pre_tool_use(  # type: ignore[union-attr]
                                tool_name=tool_name,
                                tool_input=tool_input,
                                cwd=hook_cwd,
                                active_workdir_relpath=hook_relpath,
                                step=hook_step,
                            )
                        )
                    )
                    hook_runtime_system_messages.extend(
                        pre_tool_hook_result.additional_system_messages
                    )
                    hook_runtime_user_messages.extend(pre_tool_hook_result.additional_user_messages)
                    if pre_tool_hook_result.modified_input is not None:
                        effective_tool_arguments = copy.deepcopy(
                            pre_tool_hook_result.modified_input
                        )
                    if pre_tool_hook_result.blocked:
                        pre_tool_blocked = True
                        blocked_reason = pre_tool_hook_result.reason or f"{tc.name} blocked by hook"
                        result = {"error": f"Blocked by hook: {blocked_reason}"}
                        if self.hook_dispatcher is not None:
                            self._safe_dispatch_hooks(
                                lambda tool_name=effective_tool_name, reason=blocked_reason, hook_cwd=cwd, hook_relpath=active_workdir_relpath: (
                                    self.hook_dispatcher.fire_notification(  # type: ignore[union-attr]
                                        cwd=hook_cwd,
                                        active_workdir_relpath=hook_relpath,
                                        message=f"Tool blocked: {tool_name}",
                                        level="warning",
                                        cause="pre_tool_use_blocked",
                                        payload={
                                            "tool_name": tool_name,
                                            "reason": reason,
                                        },
                                    )
                                )
                            )
                    if not pre_tool_blocked:
                        reused_result = _maybe_reuse_same_batch_read_result(
                            root=self.root,
                            cache=same_batch_read_cache,
                            tool_name=effective_tool_name,
                            arguments=effective_tool_arguments,
                        )
                        if reused_result is not None:
                            result = reused_result
                        else:
                            try:
                                future = parallel_subagent_futures.get(tc.id)
                                if future is not None:
                                    result = future.result()
                                else:
                                    result = tool.run(effective_tool_arguments)
                            except (
                                FsError,
                                SearchError,
                                SymbolSearchError,
                                HistorySearchError,
                                ShellError,
                                GitError,
                                VerifyError,
                                WebFetchError,
                                WebSearchError,
                                AgentRuntimeError,
                            ) as e:
                                result = {"error": str(e)}
                            except Exception as e:  # noqa: BLE001
                                result = {"error": f"Tool failed: {e}"}
                        post_tool_hook_result = self._safe_dispatch_hooks(
                            lambda tool_name=effective_tool_name, tool_input=copy.deepcopy(effective_tool_arguments), tool_response=copy.deepcopy(result if isinstance(result, dict) else {}), hook_cwd=cwd, hook_relpath=active_workdir_relpath, hook_step=step: (
                                self.hook_dispatcher.fire_post_tool_use(  # type: ignore[union-attr]
                                    tool_name=tool_name,
                                    tool_input=tool_input,
                                    tool_response=tool_response,
                                    cwd=hook_cwd,
                                    active_workdir_relpath=hook_relpath,
                                    step=hook_step,
                                )
                            )
                        )
                        hook_runtime_system_messages.extend(
                            post_tool_hook_result.additional_system_messages
                        )
                        hook_runtime_user_messages.extend(
                            post_tool_hook_result.additional_user_messages
                        )
                        if (
                            self.hook_dispatcher is not None
                            and isinstance(result, dict)
                            and "subagent" in result
                        ):
                            subagent_name_val = str(result.get("subagent") or "")
                            subagent_session_id_val = str(result.get("subagent_session_id") or "")
                            subagent_exit_code = result.get("exit_code")
                            subagent_status = "failed" if "error" in result else "success"
                            self._safe_dispatch_hooks(
                                lambda tool_name=effective_tool_name, s_name=subagent_name_val, s_id=subagent_session_id_val, s_status=subagent_status, s_exit=subagent_exit_code, hook_cwd=cwd, hook_relpath=active_workdir_relpath: (
                                    self.hook_dispatcher.fire_subagent_stop(  # type: ignore[union-attr]
                                        cwd=hook_cwd,
                                        active_workdir_relpath=hook_relpath,
                                        tool_name=tool_name,
                                        subagent_name=s_name,
                                        subagent_session_id=s_id,
                                        status=s_status,
                                        exit_code=(
                                            int(s_exit) if isinstance(s_exit, int | float) else None
                                        ),
                                    )
                                )
                            )
                elapsed_ms = int((perf_counter() - t0) * 1000)
                if tool_executed_for_deadline_observation:
                    _record_deadline_duration(tool_deadline_operation, t0)
                result_preview = json.dumps(result, ensure_ascii=True)
                _emit_tool_call_progress_event(
                    self.surface,
                    call_id=tc.id,
                    text=result_preview,
                )
                self.surface.on_tool_output(
                    ToolOutputEvent(
                        tool_call_id=tc.id,
                        name=tc.name,
                        chunk=result_preview,
                    )
                )
                status = "failed" if isinstance(result, dict) and "error" in result else "done"
                tool_unavailable = is_tool_unavailable_result(result)
                meta: dict[str, Any] = {}
                if alias_recovery_payload is not None:
                    meta["executed_tool_name"] = effective_tool_name
                    meta["compatibility_alias"] = alias_recovery_payload
                if status == "failed":
                    meta["error"] = str(result.get("error"))
                    if prior_failures >= MAX_IDENTICAL_TOOL_CALL_FAILURES:
                        failed_tool_call_counts[retry_key] = prior_failures
                    else:
                        failed_tool_call_counts[retry_key] = prior_failures + 1
                else:
                    failed_tool_call_counts.pop(retry_key, None)
                    if not tool_unavailable and _is_action_progress_tool(effective_tool_name):
                        step_had_successful_action_progress = True
                    touched_workspace_paths = (
                        set()
                        if tool_unavailable
                        else _extract_touched_repo_paths(
                            root=self.root,
                            tool_name=effective_tool_name,
                            arguments=effective_tool_arguments,
                            result=result if isinstance(result, dict) else {},
                        )
                    )
                    if touched_workspace_paths:
                        self.workspace_touched_paths.update(touched_workspace_paths)
                    if not tool_unavailable:
                        _remember_same_batch_read_result(
                            root=self.root,
                            cache=same_batch_read_cache,
                            tool_name=effective_tool_name,
                            arguments=effective_tool_arguments,
                            result=result if isinstance(result, dict) else {},
                        )
                if _same_batch_read_cache_should_invalidate(effective_tool_name, tool):
                    same_batch_read_cache.clear()
                _record_tool_effect(
                    root=self.root,
                    state=execution_state,
                    tool_name=effective_tool_name,
                    arguments=effective_tool_arguments,
                    status=status,
                    result=result if isinstance(result, dict) else {"error": "invalid_result"},
                    known_verification_commands=known_verification_commands,
                    verification_authoritative=bool(self.verification_authoritative),
                )

                is_successful_subagent_run = _is_successful_subagent_run(
                    tool_name=effective_tool_name,
                    arguments=effective_tool_arguments,
                    status=status,
                    result=result if isinstance(result, dict) else {},
                )
                if execution_phase_tracking_enabled and not tool_unavailable:
                    if _is_action_progress_tool(effective_tool_name):
                        step_had_action_progress = True
                        if is_successful_subagent_run:
                            subagent_success_count += 1
                            extracted_subagent_paths = _extract_successful_exploration_paths(
                                root=self.root,
                                tool_name=effective_tool_name,
                                arguments=effective_tool_arguments,
                                result=result
                                if isinstance(result, dict)
                                else {"error": "invalid_result"},
                                max_items=MAX_POST_EXPLORE_ANCHOR_PATHS,
                            )
                            for candidate in extracted_subagent_paths:
                                _append_recent_exploration_path(
                                    paths=recent_exploration_paths,
                                    candidate=candidate,
                                )
                        else:
                            post_explore_action_progress_started = True
                    elif _is_exploration_only_tool(effective_tool_name):
                        step_exploration_attempt_count += 1
                        if status == "failed":
                            step_exploration_failed_count += 1
                        else:
                            step_exploration_success_count += 1
                            extracted_paths = _extract_successful_exploration_paths(
                                root=self.root,
                                tool_name=effective_tool_name,
                                arguments=effective_tool_arguments,
                                result=result
                                if isinstance(result, dict)
                                else {"error": "invalid_result"},
                                max_items=MAX_POST_EXPLORE_ANCHOR_PATHS,
                            )
                            for candidate in extracted_paths:
                                _append_recent_exploration_path(
                                    paths=recent_exploration_paths,
                                    candidate=candidate,
                                )
                        attempt_count = exploration_attempt_call_counts.get(retry_key, 0) + 1
                        exploration_attempt_call_counts[retry_key] = attempt_count
                        similarity_key = _exploration_similarity_key(
                            effective_tool_name,
                            effective_tool_arguments,
                        )
                        similarity_count = (
                            exploration_attempt_similarity_counts.get(similarity_key, 0) + 1
                        )
                        exploration_attempt_similarity_counts[similarity_key] = similarity_count
                        if (
                            attempt_count >= MAX_IDENTICAL_EXPLORATION_ATTEMPTS
                            or similarity_count >= MAX_IDENTICAL_EXPLORATION_ATTEMPTS
                        ):
                            step_repeated_exploration_pattern = True
                            if repeated_exploration_tool is None:
                                repeated_exploration_tool = effective_tool_name
                            if repeated_exploration_key is None:
                                repeated_exploration_key = similarity_key
                if (
                    not tool_unavailable
                    and one_shot_edit_guard_enabled
                    and _is_failed_edit_stagnation_tool(effective_tool_name)
                ):
                    if status == "failed":
                        step_failed_edit_attempt_count += 1
                        if isinstance(result, dict):
                            error_text = str(result.get("error") or "")
                            if error_text:
                                step_failed_edit_errors.append(error_text[:240])
                        attempt_count = failed_edit_attempt_call_counts.get(retry_key, 0) + 1
                        failed_edit_attempt_call_counts[retry_key] = attempt_count
                        similarity_key = _edit_similarity_key(
                            effective_tool_name,
                            effective_tool_arguments,
                        )
                        similarity_count = failed_edit_similarity_counts.get(similarity_key, 0) + 1
                        failed_edit_similarity_counts[similarity_key] = similarity_count
                        if (
                            attempt_count >= MAX_IDENTICAL_FAILED_EDIT_ATTEMPTS
                            or similarity_count >= MAX_IDENTICAL_FAILED_EDIT_ATTEMPTS
                        ):
                            step_repeated_failed_edit_pattern = True
                            if repeated_failed_edit_tool is None:
                                repeated_failed_edit_tool = effective_tool_name
                            if repeated_failed_edit_key is None:
                                repeated_failed_edit_key = similarity_key
                    else:
                        step_successful_edit_attempt_count += 1
                _emit_tool_call_completed_event(
                    self.surface,
                    call_id=tc.id,
                    success=status == "done",
                    result=result,
                )
                self.surface.on_tool_end(
                    ToolEndEvent(
                        tool_call_id=tc.id,
                        name=tc.name,
                        status=status,
                        elapsed_ms=elapsed_ms,
                        meta=meta,
                    )
                )
                diagnostic_tool_payload = {
                    "tool_name": tc.name,
                    "step": step,
                    "status": status,
                    "success": status == "done",
                    "duration_ms": elapsed_ms,
                    "deadline": _deadline_snapshot(),
                }
                if alias_recovery_payload is not None:
                    diagnostic_tool_payload["executed_tool_name"] = effective_tool_name
                    diagnostic_tool_payload["compatibility_alias"] = alias_recovery_payload
                _diagnostic_event(
                    "tool_completed",
                    diagnostic_tool_payload,
                )
                content_for_message = json.dumps(
                    result,
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                if self.tool_output_offloader is not None:
                    offload_result = self.tool_output_offloader.maybe_offload(
                        tool_name=tc.name,
                        tool_call_id=tc.id,
                        step=step,
                        result=result,
                        content_json=content_for_message,
                    )
                    content_for_message = offload_result.content_for_message
                    if offload_result.offloaded:
                        self.store.append(
                            "tool_output_offloaded",
                            {
                                "tool": tc.name,
                                "tool_call_id": tc.id,
                                "artifact_locator": offload_result.artifact_locator,
                                "artifact_fs_path": offload_result.artifact_fs_path,
                                "artifact_readable_via_fs": (
                                    offload_result.artifact_readable_via_fs
                                ),
                                "artifact_location": offload_result.artifact_location,
                                "original_chars": offload_result.original_chars,
                                "preview_chars": offload_result.preview_chars,
                                "step": step,
                            },
                        )
                    if offload_result.error:
                        self.store.append(
                            "warning",
                            {
                                "warning": "tool_output_offload_failed",
                                "tool": tc.name,
                                "tool_call_id": tc.id,
                                "step": step,
                                "error": offload_result.error,
                            },
                        )
                tool_result_payload = {
                    "name": tc.name,
                    "result": result,
                    "content": content_for_message,
                    "tool_call_id": tc.id,
                    "step": step,
                }
                if alias_recovery_payload is not None:
                    tool_result_payload["executed_tool_name"] = effective_tool_name
                    tool_result_payload["compatibility_alias"] = alias_recovery_payload
                self.store.append("tool_result", tool_result_payload)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": content_for_message,
                    }
                )
                self._append_hook_messages(
                    event_name="tool_hook_context",
                    system_messages=hook_runtime_system_messages,
                    user_messages=hook_runtime_user_messages,
                )
                if deadline is not None and deadline.is_exhausted():
                    _shutdown_parallel_subagent_executor()
                    return _deadline_exhausted_result("tool_dispatch", step=step)

            _shutdown_parallel_subagent_executor()

            if execution_phase_tracking_enabled:
                if step_had_action_progress:
                    consecutive_exploration_only_steps = 0
                    exploration_attempt_call_counts.clear()
                    exploration_attempt_similarity_counts.clear()
                    consecutive_exploration_success_count = 0
                    consecutive_exploration_failed_count = 0
                    last_exploration_stagnation_payload = None
                    if post_explore_action_progress_started:
                        post_explore_bootstrap_nudges_sent = 0
                        last_post_explore_stagnation_payload = None
                elif step_exploration_attempt_count > 0:
                    consecutive_exploration_only_steps += 1
                    consecutive_exploration_success_count += step_exploration_success_count
                    consecutive_exploration_failed_count += step_exploration_failed_count
                else:
                    consecutive_exploration_only_steps = 0
                    exploration_attempt_call_counts.clear()
                    exploration_attempt_similarity_counts.clear()
                    consecutive_exploration_success_count = 0
                    consecutive_exploration_failed_count = 0
                    last_exploration_stagnation_payload = None

                step_exploration_attempt_outcome = _exploration_attempt_outcome(
                    step_exploration_success_count,
                    step_exploration_failed_count,
                )
                exploration_attempt_outcome = _exploration_attempt_outcome(
                    consecutive_exploration_success_count,
                    consecutive_exploration_failed_count,
                )

                should_nudge_for_exploration = (
                    consecutive_exploration_only_steps >= MAX_EXPLORATION_ONLY_STEPS_BEFORE_NUDGE
                    or step_repeated_exploration_pattern
                )
                should_nudge_for_subagent_exploration = (
                    not self.one_shot_execution
                    and subagent_turn_policy.active
                    and subagent_attempt_count <= 0
                    and subagent_exploration_nudges_sent < MAX_SUBAGENT_EXPLORATION_NUDGES_PER_TURN
                    and consecutive_exploration_only_steps >= 2
                    and step_exploration_attempt_count > 0
                    and execution_state.material_edit_count <= 0
                )
                if should_nudge_for_subagent_exploration:
                    subagent_exploration_nudges_sent += 1
                    anchor_paths = recent_exploration_paths[-MAX_POST_EXPLORE_ANCHOR_PATHS:]
                    nudge = _subagent_exploration_nudge_message(
                        exploration_steps=consecutive_exploration_only_steps,
                        anchor_paths=anchor_paths,
                    )
                    self.messages.append({"role": "system", "content": nudge})
                    self.store.append(
                        "subagent_exploration_nudge",
                        {
                            "step": step,
                            "attempt": subagent_exploration_nudges_sent,
                            "consecutive_exploration_only_steps": (
                                consecutive_exploration_only_steps
                            ),
                            "exploration_attempt_count": step_exploration_attempt_count,
                            "exploration_success_count": step_exploration_success_count,
                            "exploration_failed_count": step_exploration_failed_count,
                            "anchor_paths": anchor_paths,
                            "message": nudge,
                        },
                    )
                if one_shot_exploration_guard_enabled and should_nudge_for_exploration:
                    post_explore_mode = (
                        subagent_success_count > 0 and not post_explore_action_progress_started
                    )
                    reason = (
                        f"repeated_{exploration_attempt_outcome}_exploration_loop"
                        if step_repeated_exploration_pattern
                        else "consecutive_exploration_steps"
                    )
                    stagnation_payload = {
                        "step": step,
                        "reason": reason,
                        "exploration_attempt_outcome": exploration_attempt_outcome,
                        "step_exploration_attempt_outcome": step_exploration_attempt_outcome,
                        "consecutive_exploration_only_steps": consecutive_exploration_only_steps,
                        "exploration_attempt_count": step_exploration_attempt_count,
                        "exploration_success_count": step_exploration_success_count,
                        "exploration_failed_count": step_exploration_failed_count,
                        "consecutive_exploration_success_count": (
                            consecutive_exploration_success_count
                        ),
                        "consecutive_exploration_failed_count": (
                            consecutive_exploration_failed_count
                        ),
                        "tool_names": step_tool_names,
                        "repeated_tool": repeated_exploration_tool,
                        "repeated_retry_key": repeated_exploration_key,
                        "nudge_attempt": exploration_nudges_sent + 1,
                    }
                    if post_explore_mode:
                        anchored_targets = recent_exploration_paths[-MAX_POST_EXPLORE_ANCHOR_PATHS:]
                        post_payload = dict(stagnation_payload)
                        post_payload.update(
                            {
                                "post_explore": True,
                                "subagent_success_count": (subagent_success_count),
                                "action_progress_started": post_explore_action_progress_started,
                                "anchor_paths": anchored_targets,
                                "nudge_attempt": post_explore_bootstrap_nudges_sent + 1,
                            }
                        )
                        last_post_explore_stagnation_payload = dict(post_payload)
                        self.store.append(
                            "one_shot_post_explore_stagnation_detected",
                            post_payload,
                        )
                        if (
                            post_explore_bootstrap_nudges_sent
                            >= MAX_POST_EXPLORE_BOOTSTRAP_NUDGES_PER_TURN
                        ):
                            message = _runtime_text("one_shot_post_explore_retry_exhausted")
                            self.store.append(
                                "one_shot_post_explore_incomplete_after_retries",
                                {
                                    "step": step,
                                    "reason": reason,
                                    "exploration_attempt_outcome": exploration_attempt_outcome,
                                    "step_exploration_attempt_outcome": (
                                        step_exploration_attempt_outcome
                                    ),
                                    "nudge_attempts": post_explore_bootstrap_nudges_sent,
                                    "consecutive_exploration_only_steps": (
                                        consecutive_exploration_only_steps
                                    ),
                                    "exploration_attempt_count": step_exploration_attempt_count,
                                    "exploration_success_count": step_exploration_success_count,
                                    "exploration_failed_count": step_exploration_failed_count,
                                    "consecutive_exploration_success_count": (
                                        consecutive_exploration_success_count
                                    ),
                                    "consecutive_exploration_failed_count": (
                                        consecutive_exploration_failed_count
                                    ),
                                    "tool_names": step_tool_names,
                                    "repeated_tool": repeated_exploration_tool,
                                    "anchor_paths": anchored_targets,
                                },
                            )
                            _emit_surface_error(
                                self.surface,
                                "execution_guard_error",
                                message,
                                True,
                            )
                            self._emit_forced_final_summary_before_termination(
                                reason="post_explore_retry_exhausted",
                                termination_cause="post-explore bootstrap retries are exhausted",
                                termination_kind="execution_guard_stagnation",
                                max_steps=_current_turn_step_limit(),
                                language=turn_language,
                                script=turn_script,
                                explicit_language_override=turn_language_explicit,
                            )
                            assistant_message_emitted = True
                            return _finish_turn(1, reason="post_explore_retry_exhausted")

                        post_explore_bootstrap_nudges_sent += 1
                        nudge_message = _build_post_explore_bootstrap_nudge(
                            anchor_paths=anchored_targets,
                            language=turn_language,
                            explicit_language_override=turn_language_explicit,
                        )
                        self.messages.append({"role": "system", "content": nudge_message})
                        self.store.append(
                            "implementation_bootstrap_nudge",
                            {
                                "step": step,
                                "attempt": post_explore_bootstrap_nudges_sent,
                                "message": nudge_message,
                                "reason": reason,
                                "exploration_attempt_outcome": exploration_attempt_outcome,
                                "step_exploration_attempt_outcome": (
                                    step_exploration_attempt_outcome
                                ),
                                "anchor_paths": anchored_targets,
                            },
                        )
                        _phase_update_key("phase_post_explore_bootstrap")
                    else:
                        last_exploration_stagnation_payload = dict(stagnation_payload)
                        self.store.append(
                            "one_shot_exploration_stagnation_detected",
                            stagnation_payload,
                        )
                        if exploration_nudges_sent >= MAX_EXPLORATION_NUDGES_PER_TURN:
                            message = _runtime_text("one_shot_exploration_retry_exhausted")
                            self.store.append(
                                "one_shot_exploration_incomplete_after_retries",
                                {
                                    "step": step,
                                    "reason": reason,
                                    "exploration_attempt_outcome": exploration_attempt_outcome,
                                    "step_exploration_attempt_outcome": (
                                        step_exploration_attempt_outcome
                                    ),
                                    "nudge_attempts": exploration_nudges_sent,
                                    "consecutive_exploration_only_steps": (
                                        consecutive_exploration_only_steps
                                    ),
                                    "exploration_attempt_count": step_exploration_attempt_count,
                                    "exploration_success_count": step_exploration_success_count,
                                    "exploration_failed_count": step_exploration_failed_count,
                                    "consecutive_exploration_success_count": (
                                        consecutive_exploration_success_count
                                    ),
                                    "consecutive_exploration_failed_count": (
                                        consecutive_exploration_failed_count
                                    ),
                                    "tool_names": step_tool_names,
                                    "repeated_tool": repeated_exploration_tool,
                                },
                            )
                            _emit_surface_error(
                                self.surface,
                                "execution_guard_error",
                                message,
                                True,
                            )
                            self._emit_forced_final_summary_before_termination(
                                reason="exploration_retry_exhausted",
                                termination_cause="exploration retries are exhausted",
                                termination_kind="execution_guard_stagnation",
                                max_steps=_current_turn_step_limit(),
                                language=turn_language,
                                script=turn_script,
                                explicit_language_override=turn_language_explicit,
                            )
                            assistant_message_emitted = True
                            return _finish_turn(1, reason="exploration_retry_exhausted")

                        exploration_nudges_sent += 1
                        exploration_nudge = _runtime_text("one_shot_exploration_nudge")
                        self.messages.append({"role": "system", "content": exploration_nudge})
                        self.store.append(
                            "exploration_nudge",
                            {
                                "step": step,
                                "attempt": exploration_nudges_sent,
                                "message": exploration_nudge,
                                "reason": reason,
                                "exploration_attempt_outcome": exploration_attempt_outcome,
                                "step_exploration_attempt_outcome": (
                                    step_exploration_attempt_outcome
                                ),
                            },
                        )
                        _phase_update_key("phase_exploration_stagnation")

            if one_shot_edit_guard_enabled:
                if step_had_successful_action_progress:
                    consecutive_failed_edit_steps = 0
                    edit_nudges_sent = 0
                    failed_edit_attempt_call_counts.clear()
                    failed_edit_similarity_counts.clear()
                    consecutive_failed_edit_attempt_count = 0
                    last_edit_stagnation_payload = None
                elif step_failed_edit_attempt_count > 0:
                    consecutive_failed_edit_steps += 1
                    consecutive_failed_edit_attempt_count += step_failed_edit_attempt_count
                else:
                    consecutive_failed_edit_steps = 0
                    failed_edit_attempt_call_counts.clear()
                    failed_edit_similarity_counts.clear()
                    consecutive_failed_edit_attempt_count = 0
                    last_edit_stagnation_payload = None

                should_nudge_for_failed_edits = (
                    consecutive_failed_edit_steps >= MAX_FAILED_EDIT_STEPS_BEFORE_NUDGE
                    or step_repeated_failed_edit_pattern
                )
                if should_nudge_for_failed_edits:
                    reason = (
                        "repeated_failed_edit_loop"
                        if step_repeated_failed_edit_pattern
                        else "consecutive_failed_edit_steps"
                    )
                    stagnation_payload = {
                        "step": step,
                        "reason": reason,
                        "consecutive_failed_edit_steps": consecutive_failed_edit_steps,
                        "step_failed_edit_attempt_count": step_failed_edit_attempt_count,
                        "step_successful_edit_attempt_count": step_successful_edit_attempt_count,
                        "consecutive_failed_edit_attempt_count": (
                            consecutive_failed_edit_attempt_count
                        ),
                        "tool_names": step_tool_names,
                        "repeated_tool": repeated_failed_edit_tool,
                        "repeated_similarity_key": repeated_failed_edit_key,
                        "error_samples": step_failed_edit_errors[:3],
                        "nudge_attempt": edit_nudges_sent + 1,
                    }
                    last_edit_stagnation_payload = dict(stagnation_payload)
                    self.store.append("one_shot_edit_stagnation_detected", stagnation_payload)
                    if edit_nudges_sent >= MAX_EDIT_NUDGES_PER_TURN:
                        message = _runtime_text("one_shot_edit_retry_exhausted")
                        self.store.append(
                            "one_shot_edit_incomplete_after_retries",
                            {
                                "step": step,
                                "reason": reason,
                                "nudge_attempts": edit_nudges_sent,
                                "consecutive_failed_edit_steps": (consecutive_failed_edit_steps),
                                "step_failed_edit_attempt_count": (step_failed_edit_attempt_count),
                                "consecutive_failed_edit_attempt_count": (
                                    consecutive_failed_edit_attempt_count
                                ),
                                "tool_names": step_tool_names,
                                "repeated_tool": repeated_failed_edit_tool,
                                "error_samples": step_failed_edit_errors[:3],
                            },
                        )
                        _emit_surface_error(
                            self.surface,
                            "execution_guard_error",
                            message,
                            True,
                        )
                        self._emit_forced_final_summary_before_termination(
                            reason="edit_retry_exhausted",
                            termination_cause="failed edit retries are exhausted",
                            termination_kind="execution_guard_stagnation",
                            max_steps=_current_turn_step_limit(),
                            language=turn_language,
                            script=turn_script,
                            explicit_language_override=turn_language_explicit,
                        )
                        assistant_message_emitted = True
                        return _finish_turn(1, reason="edit_retry_exhausted")

                    edit_nudges_sent += 1
                    edit_nudge = _runtime_text("one_shot_edit_strategy_nudge")
                    self.messages.append({"role": "system", "content": edit_nudge})
                    self.store.append(
                        "edit_strategy_nudge",
                        {
                            "step": step,
                            "attempt": edit_nudges_sent,
                            "message": edit_nudge,
                            "reason": reason,
                        },
                    )
                    _phase_update_key("phase_failed_edit_loop")

            continue

        final_text = resp.content.strip() if resp.content else ""

        if final_text and subagent_turn_policy.required_by_user and subagent_attempt_count <= 0:
            if subagent_required_nudges_sent >= MAX_SUBAGENT_REQUIRED_NUDGES_PER_TURN:
                self.store.append(
                    "subagent_required_incomplete_after_retries",
                    {
                        "step": step,
                        "attempts": subagent_required_nudges_sent,
                        "content": final_text,
                        "available_subagents": list(subagent_turn_policy.available_subagents),
                    },
                )
                _emit_surface_error(
                    self.surface,
                    "execution_guard_error",
                    _SUBAGENT_REQUIRED_RETRY_EXHAUSTED_MESSAGE,
                    True,
                )
                self._emit_forced_final_summary_before_termination(
                    reason="subagent_required_retry_exhausted",
                    termination_cause="required subagent_run was not attempted",
                    termination_kind="execution_guard_stagnation",
                    max_steps=_current_turn_step_limit(),
                    language=turn_language,
                    script=turn_script,
                    explicit_language_override=turn_language_explicit,
                    latest_assistant_text=final_text,
                )
                assistant_message_emitted = True
                return _finish_turn(1, reason="subagent_required_retry_exhausted")
            subagent_required_nudges_sent += 1
            assistant_message = assistant_message_from_response(resp, content=final_text)
            self.messages.append(assistant_message)
            self.store.append(
                "assistant_message",
                {"content": final_text, "message": assistant_message},
            )
            nudge = _subagent_required_nudge_message(subagent_turn_policy)
            self.messages.append({"role": "system", "content": nudge})
            self.store.append(
                "subagent_required_nudge",
                {
                    "step": step,
                    "attempt": subagent_required_nudges_sent,
                    "content": final_text,
                    "available_subagents": list(subagent_turn_policy.available_subagents),
                    "message": nudge,
                },
            )
            _phase_update("Subagent delegation requested; retrying with subagent_run.")
            continue

        should_retry_first_turn_repo_grounding = (
            self.runtime_kind == RuntimeKind.INTERACTIVE_CHAT
            and not self.one_shot_execution
            and self.subagent_depth == 0
            and routing_mode == _ROUTING_MODE_AUTO
            and step == 1
            and _workspace_kind_is_repo_backed(self.store.workspace_kind)
            and not had_active_workspace_task_before_turn
            and not first_turn_repo_grounding_retry_sent
            and repo_turn_execution_intent == "execute"
            and bool(final_text)
            and not repo_tool_activity_observed
            and _session_has_stable_workspace_grounding(self)
        )
        if should_retry_first_turn_repo_grounding:
            first_turn_repo_grounding_retry_sent = True
            grounding_nudge, grounding_targets = _first_turn_repo_grounding_nudge_message(
                self,
                instruction,
            )
            self.messages.append({"role": "system", "content": grounding_nudge})
            self.store.append(
                "normal_chat_first_turn_repo_execute_retry",
                {
                    "step": step,
                    "final_text": final_text,
                    "had_tool_calls": False,
                    "repo_tool_activity_observed": repo_tool_activity_observed,
                    "workspace_grounding": (
                        _session_workspace_grounding(self).to_payload()
                        if _session_workspace_grounding(self) is not None
                        else None
                    ),
                    "targets": grounding_targets,
                },
            )
            self.store.append(
                "system_note",
                {
                    "message": "first_turn_repo_execute_retry",
                    "targets": grounding_targets,
                },
            )
            continue

        should_continue_execution_progress = (
            execution_follow_through_enabled
            and bool(final_text)
            and _assistant_text_contains_progress_intent(final_text)
            and not _assistant_text_has_completion_marker(final_text)
            and not _assistant_text_has_blocker_marker(final_text)
        )
        if should_continue_execution_progress:
            fingerprint = _one_shot_progress_fingerprint(final_text)
            decision = _build_completion_gate_decision(
                stage=NON_FINAL_PROGRESS_STAGE,
                problems=[NON_FINAL_PROGRESS_PROBLEM],
                final_text=final_text,
            )
            self.store.append(
                non_final_progress_detected_event,
                {
                    "step": step,
                    "attempt": decision.stagnant_attempt_count,
                    "fingerprint": fingerprint,
                    "content": final_text,
                    "runtime_kind": self.runtime_kind.value,
                    **_completion_gate_decision_fields(decision),
                },
            )
            if decision.kind == CompletionGateDecisionKind.TERMINATE_STAGNANT:
                record_completion_gate_decision(
                    execution_state.completion_gate_controller_state,
                    decision,
                )
                reason = "stagnant_progress"
                message = _runtime_text(non_final_progress_stopped_key)
                self.store.append(
                    non_final_incomplete_event,
                    {
                        "step": step,
                        "attempts": decision.stagnant_attempt_count,
                        "reason": reason,
                        "content": final_text,
                        "runtime_kind": self.runtime_kind.value,
                        **_completion_gate_decision_fields(decision),
                    },
                )
                _emit_surface_error(
                    self.surface,
                    "execution_guard_error",
                    message,
                    True,
                )
                self._emit_forced_final_summary_before_termination(
                    reason="non_final_progress_retry_exhausted",
                    termination_cause=(
                        "non-final progress stagnated without implementation or verification progress"
                    ),
                    termination_kind="completion_gate_stagnation",
                    max_steps=_current_turn_step_limit(),
                    language=turn_language,
                    script=turn_script,
                    explicit_language_override=turn_language_explicit,
                    latest_assistant_text=final_text,
                )
                assistant_message_emitted = True
                return _finish_turn(
                    1,
                    reason="non_final_progress_retry_exhausted",
                    final_text=final_text,
                )

            record_completion_gate_decision(
                execution_state.completion_gate_controller_state,
                decision,
            )
            continuation_nudge = _runtime_text(continuation_nudge_key)
            if _nudge_would_repeat_without_progress(continuation_nudge, decision):
                self.store.append(
                    "nudge_stall_detected",
                    {
                        "step": step,
                        "stage": NON_FINAL_PROGRESS_STAGE,
                        "reason": "duplicate_continuation_nudge",
                        "message": continuation_nudge,
                        "runtime_kind": self.runtime_kind.value,
                        "content": final_text,
                        **_turn_intent_payload(),
                        **_completion_gate_decision_fields(decision),
                    },
                )
            assistant_message = assistant_message_from_response(resp, content=final_text)
            self.messages.append(assistant_message)
            self.store.append(
                "assistant_message",
                {"content": final_text, "message": assistant_message},
            )
            self.messages.append({"role": "system", "content": continuation_nudge})
            last_nudge_text_sent = continuation_nudge
            self.store.append(
                "continuation_nudge",
                {
                    "step": step,
                    "attempt": decision.stagnant_attempt_count,
                    "message": continuation_nudge,
                    "runtime_kind": self.runtime_kind.value,
                    **_turn_intent_payload(),
                    **_completion_gate_decision_fields(decision),
                },
            )
            _phase_update_key(
                "phase_continuing_one_shot"
                if self.one_shot_execution
                else "phase_continuing_execution"
            )
            continue

        if (
            completion_gate_enabled
            and not final_text
            and repo_tool_activity_observed
            and not tool_calls
        ):
            next_anomaly_attempt = empty_response_anomaly_state.attempts + 1
            in_finalization_window = (
                deadline is not None and deadline.phase() == DeadlinePhase.FINALIZATION_WINDOW
            )
            finalization_recovery_allowed = (
                not in_finalization_window
                or empty_response_anomaly_state.finalization_window_attempts < 1
            )
            step_recovery_allowed = step < turn_max_steps
            deadline_recovery_allowed = finalization_recovery_allowed and _deadline_allows(
                DeadlineOperation.MAIN_LLM,
                minimum_remaining_seconds=MINIMUM_LLM_START_SECONDS,
                allow_during_finalization=True,
            )
            missing_action = _empty_response_missing_action()
            should_terminate_empty_anomaly = (
                next_anomaly_attempt > MAX_EMPTY_RESPONSE_ANOMALY_RECOVERIES
                or not step_recovery_allowed
                or not deadline_recovery_allowed
            )
            if should_terminate_empty_anomaly:
                reason = (
                    "empty_response_anomaly_retry_exhausted"
                    if next_anomaly_attempt > MAX_EMPTY_RESPONSE_ANOMALY_RECOVERIES
                    else "empty_response_anomaly_budget_exhausted"
                )
                self.store.append(
                    "empty_model_response_anomaly_incomplete_after_retries",
                    {
                        "step": step,
                        "runtime_kind": self.runtime_kind.value,
                        "attempt": next_anomaly_attempt,
                        "max_attempts": MAX_EMPTY_RESPONSE_ANOMALY_RECOVERIES,
                        "missing_action": missing_action,
                        "step_recovery_allowed": step_recovery_allowed,
                        "deadline_recovery_allowed": deadline_recovery_allowed,
                        "finalization_window": in_finalization_window,
                        "finalization_window_attempts": (
                            empty_response_anomaly_state.finalization_window_attempts
                        ),
                        "repo_tool_activity_observed": repo_tool_activity_observed,
                        "state": execution_state.as_payload(),
                        **_turn_intent_payload(),
                        **_local_materialization_payload(),
                    },
                )
                _emit_surface_error(
                    self.surface,
                    "model_control_error",
                    (
                        "The model repeatedly returned empty responses after tool results; "
                        "stopping locally without another summary call."
                    ),
                    True,
                )
                local_summary = (
                    "The turn stopped because the model repeatedly returned empty responses "
                    "after tool results.\n\n"
                    "Completed work:\n"
                    f"- Material actions recorded: {execution_state.material_edit_count}.\n"
                    f"- Verification attempts recorded: {execution_state.verification_attempt_count}.\n\n"
                    "Remaining work:\n"
                    f"- {missing_action}.\n\n"
                    "Known issues or risks:\n"
                    "- The final model response was empty, so this is a local runtime summary."
                )
                self._emit_final_assistant_text(
                    final_text=local_summary,
                    language=turn_language,
                    script=turn_script,
                    explicit_language_override=turn_language_explicit,
                    prior_visible_text=last_visible_assistant_text,
                    streamed_text_emitted=streamed_text_emitted,
                )
                assistant_message_emitted = True
                return _finish_turn(1, reason=reason, final_text=local_summary)

            empty_response_anomaly_state.attempts = next_anomaly_attempt
            empty_response_anomaly_state.last_missing_action = missing_action
            if in_finalization_window:
                empty_response_anomaly_state.finalization_window_attempts += 1
                finalization_empty_anomaly_recovery_pending = True
            forced_tool_choice_for_next_step = _safe_forced_tool_choice_for_recovery(
                client=self.client,
                tools=turn_tool_list,
                preferred_tool_names=_preferred_recovery_tool_names(missing_action),
            )
            empty_response_anomaly_state.last_tool_choice = forced_tool_choice_for_next_step
            recovery_message = _empty_response_recovery_message(missing_action)
            self.messages.append({"role": "system", "content": recovery_message})
            recovery_payload = {
                "step": step,
                "runtime_kind": self.runtime_kind.value,
                "stage": "empty_response_model_control",
                "attempt": next_anomaly_attempt,
                "stage_limit": MAX_EMPTY_RESPONSE_ANOMALY_RECOVERIES,
                "problems": ["empty_final_response"],
                "missing_action": missing_action,
                **_turn_intent_payload(),
            }
            self.store.append("empty_model_response_recovery", recovery_payload)
            self.store.append(
                "completion_gate_nudge",
                {
                    **recovery_payload,
                    "message": recovery_message,
                    "stage_attempt": next_anomaly_attempt,
                    "problem_summary": _completion_gate_problem_summary(["empty_final_response"]),
                    "repo_tool_activity_observed": repo_tool_activity_observed,
                },
            )
            self.store.append(
                "empty_model_response_model_control_anomaly",
                {
                    "step": step,
                    "runtime_kind": self.runtime_kind.value,
                    "attempt": next_anomaly_attempt,
                    "max_attempts": MAX_EMPTY_RESPONSE_ANOMALY_RECOVERIES,
                    "missing_action": missing_action,
                    "message": recovery_message,
                    "forced_tool_choice": forced_tool_choice_for_next_step,
                    "forced_tool_choice_supported": (forced_tool_choice_for_next_step is not None),
                    "finalization_window": in_finalization_window,
                    "finalization_window_attempts": (
                        empty_response_anomaly_state.finalization_window_attempts
                    ),
                    "repo_tool_activity_observed": repo_tool_activity_observed,
                    "state": execution_state.as_payload(),
                    **_turn_intent_payload(),
                    **_local_materialization_payload(),
                },
            )
            _phase_update_key("phase_completion_gate_repair")
            continue

        if completion_gate_enabled:
            finalize_acceptance_contract(
                contract=execution_state.acceptance_contract,
                root=self.root,
                touched_paths=execution_state.touched_repo_paths,
                durable_service_status=(
                    self.durable_service_manager.status
                    if self.durable_service_manager is not None
                    else None
                ),
            )
            blocked_response = _assistant_text_has_well_formed_blocker(final_text)
            blocked_response_allows_completion = _completion_gate_blocker_allows_final(
                state=execution_state,
                blocked_response=blocked_response,
            )
            clarification_response = _assistant_text_is_clarification_only(final_text)
            clarification_allows_completion = (
                clarification_response and _clarification_can_finalize(text=final_text)
            )
            if clarification_allows_completion:
                blocked_response = True
                blocked_response_allows_completion = True
            completion_gate_turn_intent = _completion_gate_repo_turn_execution_intent(final_text)
            verification_expected = _verification_expected_for_turn(
                turn_intent=completion_gate_turn_intent,
                blocked=blocked_response_allows_completion,
                touched_repo_paths=execution_state.touched_repo_paths,
                verification_contract_requires_execution=(
                    self.verification_contract_type
                    in {"authoritative_override", "explicit_override", "task_inferred"}
                ),
            )
            gate_problems = _completion_gate_problems(
                state=execution_state,
                final_text=final_text,
                blocked=blocked_response_allows_completion,
                verification_expected=verification_expected,
                require_material_edit_evidence=_completion_gate_requires_material_edit_evidence(
                    final_text=final_text,
                    gate_turn_intent=completion_gate_turn_intent,
                ),
            )
            if (
                clarification_response
                and not clarification_allows_completion
                and "clarification_requested" not in gate_problems
            ):
                gate_problems.append("clarification_requested")
            if gate_problems:
                gate_stage = _completion_gate_repair_stage(gate_problems)
                decision = _build_completion_gate_decision(
                    stage=gate_stage,
                    problems=gate_problems,
                    final_text=final_text,
                    blocked_response=blocked_response,
                    blocked_response_allows_completion=blocked_response_allows_completion,
                    verification_expected=verification_expected,
                )
                record_completion_gate_decision(
                    execution_state.completion_gate_controller_state,
                    decision,
                )
                decision_fields = _completion_gate_decision_fields(decision)
                stage_limit = decision.max_stagnant_attempts
                stage_attempts = decision.stagnant_attempt_count
                if "empty_final_response" in gate_problems:
                    self.store.append(
                        "empty_model_response_recovery",
                        {
                            "step": step,
                            "runtime_kind": self.runtime_kind.value,
                            "stage": gate_stage,
                            "attempt": stage_attempts,
                            "stage_limit": stage_limit,
                            "problems": gate_problems,
                            **_turn_intent_payload(
                                completion_gate_turn_intent=completion_gate_turn_intent,
                            ),
                            **decision_fields,
                        },
                    )
                failure_snippet = (
                    execution_state.last_verification_failure_snippet
                    or execution_state.first_failed_verification_snippet()
                    if gate_stage == "verification_failed"
                    else ""
                )
                no_material_anchor_paths = (
                    recent_exploration_paths[-MAX_POST_EXPLORE_ANCHOR_PATHS:]
                    if gate_stage == "no_material_edits"
                    else []
                )
                if gate_stage == "no_material_edits":
                    self.store.append(
                        no_material_edits_detected_event,
                        {
                            "step": step,
                            "runtime_kind": self.runtime_kind.value,
                            "repo_tool_activity_observed": repo_tool_activity_observed,
                            "anchor_paths": no_material_anchor_paths,
                            "state": execution_state.as_payload(),
                            "content": final_text,
                            **_turn_intent_payload(
                                completion_gate_turn_intent=completion_gate_turn_intent,
                            ),
                            **_verification_evidence_fields(),
                            **_acceptance_contract_fields(),
                            **decision_fields,
                        },
                    )
                self.store.append(
                    completion_gate_failed_event,
                    {
                        "step": step,
                        "runtime_kind": self.runtime_kind.value,
                        "problems": gate_problems,
                        "problem_summary": _completion_gate_problem_summary(gate_problems),
                        "stage": gate_stage,
                        "stage_attempt": stage_attempts,
                        "stage_limit": stage_limit,
                        "blocked_response": blocked_response,
                        "blocked_response_allows_completion": (blocked_response_allows_completion),
                        "clarification_response": clarification_response,
                        "clarification_allows_completion": clarification_allows_completion,
                        "verification_expected": verification_expected,
                        "verification_failure_snippet": failure_snippet,
                        "repo_tool_activity_observed": repo_tool_activity_observed,
                        "anchor_paths": no_material_anchor_paths,
                        "missing_verification_commands": _sorted_missing_verification_commands(
                            execution_state
                        ),
                        "verification_coverage_stale": (
                            execution_state.verification_coverage_is_stale()
                        ),
                        "state": execution_state.as_payload(),
                        "content": final_text,
                        "attempt": stage_attempts,
                        **_turn_intent_payload(
                            completion_gate_turn_intent=completion_gate_turn_intent,
                        ),
                        **_verification_evidence_fields(),
                        **_acceptance_contract_fields(),
                        **decision_fields,
                    },
                )
                if decision.kind == CompletionGateDecisionKind.TERMINATE_STAGNANT:
                    problem_summary = _completion_gate_problem_summary(gate_problems)
                    message = _completion_gate_terminal_failure_message(
                        problem_summary=problem_summary,
                        stage=gate_stage,
                        message_key=completion_gate_terminal_message_key,
                        verification_failure_snippet=failure_snippet,
                        language=turn_language,
                        explicit_language_override=turn_language_explicit,
                    )
                    if gate_stage == "no_material_edits":
                        self.store.append(
                            no_material_edits_incomplete_event,
                            {
                                "step": step,
                                "runtime_kind": self.runtime_kind.value,
                                "problems": gate_problems,
                                "problem_summary": problem_summary,
                                "stage": gate_stage,
                                "stage_attempts": stage_attempts,
                                "stage_limit": stage_limit,
                                "repair_nudges_sent": (
                                    execution_state.completion_gate_repair_attempts
                                ),
                                "repo_tool_activity_observed": repo_tool_activity_observed,
                                "anchor_paths": no_material_anchor_paths,
                                "state": execution_state.as_payload(),
                                "content": final_text,
                                "attempts": stage_attempts,
                                **_turn_intent_payload(
                                    completion_gate_turn_intent=completion_gate_turn_intent,
                                ),
                                **_verification_evidence_fields(),
                                **_acceptance_contract_fields(),
                                **decision_fields,
                            },
                        )
                    self.store.append(
                        completion_gate_incomplete_event,
                        {
                            "step": step,
                            "runtime_kind": self.runtime_kind.value,
                            "problems": gate_problems,
                            "problem_summary": problem_summary,
                            "stage": gate_stage,
                            "stage_attempts": stage_attempts,
                            "stage_limit": stage_limit,
                            "repair_nudges_sent": (execution_state.completion_gate_repair_attempts),
                            "blocked_response": blocked_response,
                            "blocked_response_allows_completion": (
                                blocked_response_allows_completion
                            ),
                            "verification_expected": verification_expected,
                            "verification_failure_snippet": failure_snippet,
                            "repo_tool_activity_observed": repo_tool_activity_observed,
                            "anchor_paths": no_material_anchor_paths,
                            "missing_verification_commands": _sorted_missing_verification_commands(
                                execution_state
                            ),
                            "verification_coverage_stale": (
                                execution_state.verification_coverage_is_stale()
                            ),
                            "state": execution_state.as_payload(),
                            "content": final_text,
                            "attempts": stage_attempts,
                            **_turn_intent_payload(
                                completion_gate_turn_intent=completion_gate_turn_intent,
                            ),
                            **_verification_evidence_fields(),
                            **_acceptance_contract_fields(),
                            **decision_fields,
                        },
                    )
                    _emit_surface_error(
                        self.surface,
                        "completion_gate_error",
                        message,
                        True,
                    )
                    self._emit_forced_final_summary_before_termination(
                        reason="completion_gate_terminal_failure",
                        termination_cause=(
                            "completion-gate stagnation after repeated invalid "
                            "finalization attempts without new implementation or "
                            "verification progress"
                        ),
                        termination_kind="completion_gate_stagnation",
                        max_steps=_current_turn_step_limit(),
                        language=turn_language,
                        script=turn_script,
                        explicit_language_override=turn_language_explicit,
                        latest_assistant_text=final_text,
                    )
                    assistant_message_emitted = True
                    return _finish_turn(1, reason="completion_gate_terminal_failure")

                execution_state.increment_repair_attempts_for_stage(gate_stage)
                if final_text:
                    assistant_message = assistant_message_from_response(resp, content=final_text)
                    self.messages.append(assistant_message)
                    self.store.append(
                        "assistant_message",
                        {"content": final_text, "message": assistant_message},
                    )
                nudge = _completion_gate_nudge_message(
                    gate_problems,
                    prefix_key=completion_gate_nudge_prefix_key,
                    verification_failure_snippet=failure_snippet,
                    missing_verification_commands=_sorted_missing_verification_commands(
                        execution_state
                    ),
                    verification_coverage_stale=execution_state.verification_coverage_is_stale(),
                    anchor_paths=no_material_anchor_paths,
                    language=turn_language,
                    explicit_language_override=turn_language_explicit,
                )
                if _nudge_would_repeat_without_progress(nudge, decision):
                    self.store.append(
                        "nudge_stall_detected",
                        {
                            "step": step,
                            "stage": gate_stage,
                            "reason": "duplicate_completion_gate_nudge",
                            "message": nudge,
                            "runtime_kind": self.runtime_kind.value,
                            "problems": gate_problems,
                            "problem_summary": _completion_gate_problem_summary(gate_problems),
                            "content": final_text,
                            **_turn_intent_payload(
                                completion_gate_turn_intent=completion_gate_turn_intent,
                            ),
                            **_verification_evidence_fields(),
                            **_acceptance_contract_fields(),
                            **decision_fields,
                        },
                    )
                self.messages.append({"role": "system", "content": nudge})
                last_nudge_text_sent = nudge
                self.store.append(
                    "completion_gate_nudge",
                    {
                        "step": step,
                        "runtime_kind": self.runtime_kind.value,
                        "attempt": execution_state.completion_gate_repair_attempts,
                        "stage": gate_stage,
                        "stage_attempt": execution_state.repair_attempts_for_stage(gate_stage),
                        "stage_limit": stage_limit,
                        "problems": gate_problems,
                        "problem_summary": _completion_gate_problem_summary(gate_problems),
                        "verification_failure_snippet": failure_snippet,
                        "repo_tool_activity_observed": repo_tool_activity_observed,
                        "anchor_paths": no_material_anchor_paths,
                        "verification_coverage_stale": (
                            execution_state.verification_coverage_is_stale()
                        ),
                        "language": turn_language,
                        "explicit_language_override": turn_language_explicit,
                        "message": nudge,
                        **_turn_intent_payload(
                            completion_gate_turn_intent=completion_gate_turn_intent,
                        ),
                        **_verification_evidence_fields(),
                        **_acceptance_contract_fields(),
                        **decision_fields,
                    },
                )
                if gate_stage == "no_material_edits":
                    self.store.append(
                        "no_material_edits_bootstrap_nudge",
                        {
                            "step": step,
                            "attempt": execution_state.repair_attempts_for_stage(gate_stage),
                            "stage_limit": stage_limit,
                            "repo_tool_activity_observed": repo_tool_activity_observed,
                            "anchor_paths": no_material_anchor_paths,
                            "message": nudge,
                            **_turn_intent_payload(
                                completion_gate_turn_intent=completion_gate_turn_intent,
                            ),
                            **decision_fields,
                        },
                    )
                if gate_stage == "verification_failed":
                    self.store.append(
                        "failed_verification_repair_attempt",
                        {
                            "step": step,
                            "attempt": execution_state.repair_attempts_for_stage(gate_stage),
                            "stage_limit": stage_limit,
                            "snippet": failure_snippet,
                            "message": nudge,
                            **_turn_intent_payload(
                                completion_gate_turn_intent=completion_gate_turn_intent,
                            ),
                            **decision_fields,
                        },
                    )
                _phase_update_key("phase_completion_gate_repair")
                continue
            if blocked_response_allows_completion:
                self.store.append(
                    "completion_gate_blocker_accepted",
                    {
                        "step": step,
                        "runtime_kind": self.runtime_kind.value,
                        "blocked_response": blocked_response,
                        "blocked_response_allows_completion": True,
                        "verification_expected": verification_expected,
                        "state": execution_state.as_payload(),
                        "content": final_text,
                        **_turn_intent_payload(
                            completion_gate_turn_intent=completion_gate_turn_intent,
                        ),
                        **_verification_evidence_fields(),
                        **_acceptance_contract_fields(),
                    },
                )
                if not stream_used:
                    _phase_update_key("phase_writing_final_response")
                self._emit_final_assistant_text(
                    final_text=final_text,
                    assistant_response=resp,
                    language=turn_language,
                    script=turn_script,
                    explicit_language_override=turn_language_explicit,
                    prior_visible_text=last_visible_assistant_text,
                    streamed_text_emitted=streamed_text_emitted,
                )
                assistant_message_emitted = True
                return _finish_turn(0, reason="blocked", final_text=final_text)

        if not stream_used:
            _phase_update_key("phase_writing_final_response")
        self.store.append(
            "turn_intent_finalized",
            {
                "runtime_kind": self.runtime_kind.value,
                "state": execution_state.as_payload(),
                **_turn_intent_payload(
                    completion_gate_turn_intent=_completion_gate_repo_turn_execution_intent(
                        final_text
                    ),
                ),
                **_acceptance_contract_fields(),
            },
        )
        self._emit_final_assistant_text(
            final_text=final_text,
            assistant_response=resp,
            language=turn_language,
            script=turn_script,
            explicit_language_override=turn_language_explicit,
            prior_visible_text=last_visible_assistant_text,
            streamed_text_emitted=streamed_text_emitted,
        )
        assistant_message_emitted = True
        return _finish_turn(0, reason="completed", final_text=final_text)

    if (
        one_shot_exploration_guard_enabled
        and subagent_success_count > 0
        and not post_explore_action_progress_started
        and (
            post_explore_bootstrap_nudges_sent > 0
            or consecutive_exploration_only_steps > 0
            or last_post_explore_stagnation_payload is not None
        )
    ):
        payload: dict[str, Any] = {
            "step": _current_turn_step_limit(),
            "max_steps": _current_turn_step_limit(),
            "reason": "post_explore_step_budget_exhausted",
            "subagent_success_count": subagent_success_count,
            "nudge_attempts": post_explore_bootstrap_nudges_sent,
            "consecutive_exploration_only_steps": consecutive_exploration_only_steps,
            "consecutive_exploration_success_count": consecutive_exploration_success_count,
            "consecutive_exploration_failed_count": consecutive_exploration_failed_count,
            "anchor_paths": recent_exploration_paths[-MAX_POST_EXPLORE_ANCHOR_PATHS:],
        }
        if last_post_explore_stagnation_payload is not None:
            payload["last_stagnation"] = last_post_explore_stagnation_payload
        self.store.append("one_shot_post_explore_incomplete_after_retries", payload)
        _emit_surface_error(
            self.surface,
            "step_budget_error",
            _runtime_text("one_shot_post_explore_step_budget_exhausted"),
            True,
        )
        self._emit_forced_final_summary_before_termination(
            reason="post_explore_step_budget_exhausted",
            termination_cause="the overall step budget is exhausted",
            termination_kind="step_budget_exhausted",
            max_steps=_current_turn_step_limit(),
            language=turn_language,
            script=turn_script,
            explicit_language_override=turn_language_explicit,
        )
        assistant_message_emitted = True
        return _finish_turn(1, reason="post_explore_step_budget_exhausted")

    if one_shot_exploration_guard_enabled and (
        exploration_nudges_sent > 0
        or consecutive_exploration_only_steps > 0
        or last_exploration_stagnation_payload is not None
    ):
        reason = "exploration_step_budget_exhausted"
        exploration_attempt_outcome = _exploration_attempt_outcome(
            consecutive_exploration_success_count,
            consecutive_exploration_failed_count,
        )
        payload: dict[str, Any] = {
            "step": _current_turn_step_limit(),
            "max_steps": _current_turn_step_limit(),
            "reason": reason,
            "exploration_attempt_outcome": exploration_attempt_outcome,
            "nudge_attempts": exploration_nudges_sent,
            "consecutive_exploration_only_steps": consecutive_exploration_only_steps,
            "consecutive_exploration_success_count": consecutive_exploration_success_count,
            "consecutive_exploration_failed_count": consecutive_exploration_failed_count,
        }
        if last_exploration_stagnation_payload is not None:
            payload["last_stagnation"] = last_exploration_stagnation_payload
        self.store.append("one_shot_exploration_incomplete_after_retries", payload)
        _emit_surface_error(
            self.surface,
            "step_budget_error",
            _runtime_text("one_shot_exploration_step_budget_exhausted"),
            True,
        )
        self._emit_forced_final_summary_before_termination(
            reason=reason,
            termination_cause="the overall step budget is exhausted",
            termination_kind="step_budget_exhausted",
            max_steps=_current_turn_step_limit(),
            language=turn_language,
            script=turn_script,
            explicit_language_override=turn_language_explicit,
        )
        assistant_message_emitted = True
        return _finish_turn(1, reason="exploration_step_budget_exhausted")

    if one_shot_edit_guard_enabled and (
        edit_nudges_sent > 0
        or consecutive_failed_edit_steps > 0
        or last_edit_stagnation_payload is not None
    ):
        payload: dict[str, Any] = {
            "step": _current_turn_step_limit(),
            "max_steps": _current_turn_step_limit(),
            "reason": "edit_step_budget_exhausted",
            "nudge_attempts": edit_nudges_sent,
            "consecutive_failed_edit_steps": consecutive_failed_edit_steps,
            "consecutive_failed_edit_attempt_count": consecutive_failed_edit_attempt_count,
        }
        if last_edit_stagnation_payload is not None:
            payload["last_stagnation"] = last_edit_stagnation_payload
        self.store.append("one_shot_edit_incomplete_after_retries", payload)
        _emit_surface_error(
            self.surface,
            "step_budget_error",
            _runtime_text("one_shot_edit_step_budget_exhausted"),
            True,
        )
        self._emit_forced_final_summary_before_termination(
            reason="edit_step_budget_exhausted",
            termination_cause="the overall step budget is exhausted",
            termination_kind="step_budget_exhausted",
            max_steps=_current_turn_step_limit(),
            language=turn_language,
            script=turn_script,
            explicit_language_override=turn_language_explicit,
        )
        assistant_message_emitted = True
        return _finish_turn(1, reason="edit_step_budget_exhausted")

    if completion_gate_enabled and execution_state.completion_gate_repair_attempts > 0:
        exhausted_stage = "generic"
        exhausted_stage_attempts = execution_state.completion_gate_repair_attempts
        reason = "completion_gate_step_budget_exhausted"
        failure_snippet = ""
        if execution_state.completion_gate_failed_verify_repair_attempts > 0:
            exhausted_stage = "verification_failed"
            exhausted_stage_attempts = execution_state.completion_gate_failed_verify_repair_attempts
            reason = "completion_gate_failed_verification_step_budget_exhausted"
            failure_snippet = execution_state.last_verification_failure_snippet
        elif execution_state.completion_gate_missing_verify_repair_attempts > 0:
            exhausted_stage = (
                "verification_incomplete"
                if execution_state.expected_verification_commands
                and execution_state.missing_verification_commands()
                else "verification_not_attempted"
            )
            exhausted_stage_attempts = (
                execution_state.completion_gate_missing_verify_repair_attempts
            )
            reason = (
                "completion_gate_incomplete_verification_step_budget_exhausted"
                if exhausted_stage == "verification_incomplete"
                else "completion_gate_missing_verification_step_budget_exhausted"
            )
        exhausted_problems = [exhausted_stage] if exhausted_stage != "generic" else ["generic"]
        decision = _build_completion_gate_decision(
            stage=exhausted_stage,
            problems=exhausted_problems,
            final_text="",
            budget_exhausted=True,
        )
        record_completion_gate_decision(
            execution_state.completion_gate_controller_state,
            decision,
        )
        decision_fields = _completion_gate_decision_fields(decision)
        exhausted_stage_limit = decision.max_stagnant_attempts
        self.store.append(
            completion_gate_incomplete_event,
            {
                "step": _current_turn_step_limit(),
                "max_steps": _current_turn_step_limit(),
                "runtime_kind": self.runtime_kind.value,
                "reason": reason,
                "attempts": execution_state.completion_gate_repair_attempts,
                "stage": exhausted_stage,
                "stage_attempts": exhausted_stage_attempts,
                "stage_limit": exhausted_stage_limit,
                "verification_failure_snippet": failure_snippet,
                "missing_verification_commands": _sorted_missing_verification_commands(
                    execution_state
                ),
                "verification_coverage_stale": (execution_state.verification_coverage_is_stale()),
                "state": execution_state.as_payload(),
                **decision_fields,
            },
        )
        exit_code = 1
        if interactive_step_budget_handoff_enabled:
            exit_code = 0
            _phase_update_key("phase_step_budget_handoff")
            self.store.append(
                "interactive_step_budget_handoff",
                {
                    "step": _current_turn_step_limit(),
                    "max_steps": _current_turn_step_limit(),
                    "reason": reason,
                    "stage": exhausted_stage,
                },
            )
        else:
            _emit_surface_error(
                self.surface,
                "completion_gate_error",
                _completion_gate_step_budget_exhausted_message(
                    stage=exhausted_stage,
                    message_key=completion_gate_step_budget_message_key,
                    verification_failure_snippet=failure_snippet,
                    language=turn_language,
                    explicit_language_override=turn_language_explicit,
                ),
                True,
            )
        self._emit_forced_final_summary_before_termination(
            reason=reason,
            termination_cause="the overall step budget is exhausted during completion-gate repair",
            termination_kind="step_budget_exhausted",
            max_steps=_current_turn_step_limit(),
            language=turn_language,
            script=turn_script,
            explicit_language_override=turn_language_explicit,
        )
        assistant_message_emitted = True
        return _finish_turn(exit_code, reason=reason)

    max_steps_message = _runtime_text("max_steps_exceeded")
    if interactive_step_budget_handoff_enabled:
        self.store.append(
            "interactive_step_budget_handoff",
            {
                "step": _current_turn_step_limit(),
                "max_steps": _current_turn_step_limit(),
                "reason": "max_steps_exhausted",
            },
        )
        _phase_update_key("phase_step_budget_handoff")
        self._emit_forced_final_summary_before_termination(
            reason="max_steps_exhausted",
            termination_cause="the overall step budget is exhausted",
            termination_kind="step_budget_exhausted",
            max_steps=_current_turn_step_limit(),
            language=turn_language,
            script=turn_script,
            explicit_language_override=turn_language_explicit,
        )
        assistant_message_emitted = True
        return _finish_turn(0, reason="max_steps_exhausted")
    self.store.append(
        "error",
        {
            "error": max_steps_message,
            "max_steps": _current_turn_step_limit(),
        },
    )
    _emit_surface_error(self.surface, "step_budget_error", max_steps_message, True)
    self._emit_forced_final_summary_before_termination(
        reason="max_steps_exceeded",
        termination_cause="the overall step budget is exhausted",
        termination_kind="step_budget_exhausted",
        max_steps=_current_turn_step_limit(),
        language=turn_language,
        script=turn_script,
        explicit_language_override=turn_language_explicit,
    )
    assistant_message_emitted = True
    return _finish_turn(1, reason="max_steps_exceeded")
