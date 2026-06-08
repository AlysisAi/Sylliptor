from __future__ import annotations

import copy
import json
import re
from collections.abc import Collection
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

from ...config import resolve_role_temperature
from ...llm.metadata import assistant_message_from_response
from ...llm.types import LLMError
from ...runtime_kind import RuntimeKind
from ...step_budget import StepBudgetRequest, resolve_step_budget
from ...subagents import SubagentDefinition
from ...surface import NoopSurface, ToolEndEvent, ToolOutputEvent, ToolStartEvent
from ...surface.base import Surface
from ...tools.availability import is_tool_unavailable_result, unavailable_tool_result
from ...tools.fs import FsError
from ...tools.git import GitError
from ...tools.history import HistorySearchError
from ...tools.search import SearchError
from ...tools.shell import ShellError
from ...tools.symbols import SymbolSearchError
from ...tools.web import WebFetchError
from ...tools.web_search import WebSearchError
from ...turn_intent import (
    classify_repo_execution_intent as _classify_one_shot_repo_turn_intent,
)
from ...turn_intent import normalize_turn_intent_text as _normalize_marker_text
from ...usage_tracker import build_usage_record
from ...verify_gate import VerifyError
from .. import _patchable
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
    _completion_gate_stage_attempt_limit,
    _completion_gate_step_budget_exhausted_message,
    _completion_gate_terminal_failure_message,
    _extract_touched_repo_paths,
    _record_tool_effect,
    _refresh_interactive_turn_verification_selection,
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
MAX_SUBAGENT_REQUIRED_NUDGES_PER_TURN = 2
MAX_SUBAGENT_EXPLORATION_NUDGES_PER_TURN = 1
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
) -> int:
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

    def _finish_turn(code: int, *, reason: str, final_text: str = "") -> int:
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
    _phase_update_key("phase_understanding_request")

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
            route_turn = _patchable("_route_turn", _route_turn)
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
        except LLMError as err:
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
        if original_route_decision.route != "repo":
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
            if route_override_reason is None:
                route_override_reason = _managed_execution_route_override_reason(
                    runtime_kind=self.runtime_kind,
                    original_route=original_route_decision.route,
                    route_execution_posture=original_route_execution_posture,
                )
        route_decision = original_route_decision
        if route_override_reason:
            route_decision = _TurnRouteDecision(
                route="repo",
                execution_posture=original_route_execution_posture,
                confidence=original_route_decision.confidence,
                reply="",
                language=original_route_decision.language,
                script=original_route_decision.script,
                explicit_language_override=original_route_decision.explicit_language_override,
                language_source=original_route_decision.language_source,
                decision_source=original_route_decision_source,
                execution_posture_source=original_route_execution_posture_source,
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
                "route_override_reason": route_override_reason,
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
        route_execution_posture = str(_classify_one_shot_repo_turn_intent(instruction) or "execute")
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

    _refresh_interactive_turn_verification_selection(
        self,
        instruction=instruction,
        route_execution_posture=route_execution_posture,
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
    one_shot_non_final_continuations = 0
    non_final_progress_fingerprints: set[str] = set()
    one_shot_turn_intent = _classify_one_shot_repo_turn_intent(instruction)
    repo_turn_execution_intent = _resolve_repo_turn_execution_intent(
        one_shot_execution=self.one_shot_execution,
        runtime_kind=self.runtime_kind,
        route_execution_posture=route_execution_posture,
        classified_turn_intent=one_shot_turn_intent,
    )
    execution_safeguards_enabled = repo_turn_execution_intent == "execute"
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
    execution_state = TurnExecutionState(
        execution_requested=execution_safeguards_enabled,
        expected_verification_commands=set(known_verification_commands),
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
    first_turn_repo_grounding_retry_sent = False
    last_post_explore_stagnation_payload: dict[str, Any] | None = None
    consecutive_failed_edit_steps = 0
    edit_nudges_sent = 0
    failed_edit_attempt_call_counts: dict[str, int] = {}
    failed_edit_similarity_counts: dict[str, int] = {}
    consecutive_failed_edit_attempt_count = 0
    last_edit_stagnation_payload: dict[str, Any] | None = None

    for step in range(1, turn_max_steps + 1):
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
            if delta:
                _emit_message_delta_event(self.surface, delta)
                streamed_text_emitted = True
            self.surface.on_assistant_token(delta)

        request_messages = _request_messages_for_step(self.messages)
        try:
            if self.conversation_compactor is not None:
                pre_compact_message_count = len(self.messages)
                compacted_messages, compacted = self.conversation_compactor.maybe_compact(
                    messages=self.messages,
                    tool_list=turn_tool_list,
                    main_model=self.client.model,
                    focus=instruction,
                )
                self.messages = compacted_messages
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
            request_messages = _request_messages_for_step(self.messages)
            resp = _main_agent_chat(
                client=self.client,
                messages=request_messages,
                tools=turn_tool_list,
                stream=stream_used,
                on_text_delta=_on_text_delta if stream_used else None,
            )
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
                    resp = _main_agent_chat(
                        client=self.client,
                        messages=request_messages,
                        tools=turn_tool_list,
                        stream=False,
                        on_text_delta=None,
                    )
                except LLMError as retry_err:
                    self.store.append("error", {"error": str(retry_err)})
                    _rollback_turn_after_llm_error()
                    raise
                stream_used = False
            else:
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
                retry_resp = _main_agent_chat(
                    client=self.client,
                    messages=_request_messages_for_step(self.messages),
                    tools=turn_tool_list,
                    stream=False,
                    on_text_delta=None,
                    temperature=_ADAPTIVE_RETRY_TEMPERATURE,
                )
            except LLMError as adaptive_retry_error:
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
            same_batch_read_cache = _SameBatchReadReuseCache()

            for tc in tool_calls:
                retry_key = _tool_call_retry_key(tc.name, tc.arguments)
                prior_failures = failed_tool_call_counts.get(retry_key, 0)
                tool = turn_tools.get(tc.name)
                tool_call_payload: dict[str, Any] = {
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "tool_call_id": tc.id,
                    "step": step,
                }
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
                t0 = perf_counter()
                effective_tool_arguments = copy.deepcopy(tc.arguments)
                hook_runtime_system_messages: list[str] = []
                hook_runtime_user_messages: list[str] = []
                pre_tool_blocked = False
                unavailable_result = unavailable_tool_result(tc.name)
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
                    result = {"error": f"Unknown tool: {tc.name}"}
                else:
                    cwd, active_workdir_relpath = self._hook_runtime_context()
                    pre_tool_hook_result = self._safe_dispatch_hooks(
                        lambda tool_name=tc.name, tool_input=copy.deepcopy(effective_tool_arguments), hook_cwd=cwd, hook_relpath=active_workdir_relpath, hook_step=step: (
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
                                lambda tool_name=tc.name, reason=blocked_reason, hook_cwd=cwd, hook_relpath=active_workdir_relpath: (
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
                            tool_name=tc.name,
                            arguments=effective_tool_arguments,
                        )
                        if reused_result is not None:
                            result = reused_result
                        else:
                            try:
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
                            lambda tool_name=tc.name, tool_input=copy.deepcopy(effective_tool_arguments), tool_response=copy.deepcopy(result if isinstance(result, dict) else {}), hook_cwd=cwd, hook_relpath=active_workdir_relpath, hook_step=step: (
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
                                lambda tool_name=tc.name, s_name=subagent_name_val, s_id=subagent_session_id_val, s_status=subagent_status, s_exit=subagent_exit_code, hook_cwd=cwd, hook_relpath=active_workdir_relpath: (
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
                if status == "failed":
                    meta["error"] = str(result.get("error"))
                    if prior_failures >= MAX_IDENTICAL_TOOL_CALL_FAILURES:
                        failed_tool_call_counts[retry_key] = prior_failures
                    else:
                        failed_tool_call_counts[retry_key] = prior_failures + 1
                else:
                    failed_tool_call_counts.pop(retry_key, None)
                    if not tool_unavailable and _is_action_progress_tool(tc.name):
                        step_had_successful_action_progress = True
                    touched_workspace_paths = (
                        set()
                        if tool_unavailable
                        else _extract_touched_repo_paths(
                            root=self.root,
                            tool_name=tc.name,
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
                            tool_name=tc.name,
                            arguments=effective_tool_arguments,
                            result=result if isinstance(result, dict) else {},
                        )
                if _same_batch_read_cache_should_invalidate(tc.name, tool):
                    same_batch_read_cache.clear()
                _record_tool_effect(
                    root=self.root,
                    state=execution_state,
                    tool_name=tc.name,
                    arguments=effective_tool_arguments,
                    status=status,
                    result=result if isinstance(result, dict) else {"error": "invalid_result"},
                    known_verification_commands=known_verification_commands,
                )

                is_successful_subagent_run = _is_successful_subagent_run(
                    tool_name=tc.name,
                    arguments=effective_tool_arguments,
                    status=status,
                    result=result if isinstance(result, dict) else {},
                )
                if execution_phase_tracking_enabled and not tool_unavailable:
                    if _is_action_progress_tool(tc.name):
                        step_had_action_progress = True
                        if is_successful_subagent_run:
                            subagent_success_count += 1
                            extracted_subagent_paths = _extract_successful_exploration_paths(
                                root=self.root,
                                tool_name=tc.name,
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
                    elif _is_exploration_only_tool(tc.name):
                        step_exploration_attempt_count += 1
                        if status == "failed":
                            step_exploration_failed_count += 1
                        else:
                            step_exploration_success_count += 1
                            extracted_paths = _extract_successful_exploration_paths(
                                root=self.root,
                                tool_name=tc.name,
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
                            tc.name,
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
                                repeated_exploration_tool = tc.name
                            if repeated_exploration_key is None:
                                repeated_exploration_key = similarity_key
                if (
                    not tool_unavailable
                    and one_shot_edit_guard_enabled
                    and _is_failed_edit_stagnation_tool(tc.name)
                ):
                    if status == "failed":
                        step_failed_edit_attempt_count += 1
                        if isinstance(result, dict):
                            error_text = str(result.get("error") or "")
                            if error_text:
                                step_failed_edit_errors.append(error_text[:240])
                        attempt_count = failed_edit_attempt_call_counts.get(retry_key, 0) + 1
                        failed_edit_attempt_call_counts[retry_key] = attempt_count
                        similarity_key = _edit_similarity_key(tc.name, effective_tool_arguments)
                        similarity_count = failed_edit_similarity_counts.get(similarity_key, 0) + 1
                        failed_edit_similarity_counts[similarity_key] = similarity_count
                        if (
                            attempt_count >= MAX_IDENTICAL_FAILED_EDIT_ATTEMPTS
                            or similarity_count >= MAX_IDENTICAL_FAILED_EDIT_ATTEMPTS
                        ):
                            step_repeated_failed_edit_pattern = True
                            if repeated_failed_edit_tool is None:
                                repeated_failed_edit_tool = tc.name
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
                self.store.append(
                    "tool_result",
                    {
                        "name": tc.name,
                        "result": result,
                        "content": content_for_message,
                        "tool_call_id": tc.id,
                        "step": step,
                    },
                )
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
            and route_execution_posture == "execute"
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
            repeated_progress = bool(fingerprint) and (
                fingerprint in non_final_progress_fingerprints
            )
            continuation_cap_reached = (
                one_shot_non_final_continuations >= MAX_NON_FINAL_CONTINUATIONS_PER_TURN
            )
            self.store.append(
                non_final_progress_detected_event,
                {
                    "step": step,
                    "attempt": one_shot_non_final_continuations + 1,
                    "fingerprint": fingerprint,
                    "content": final_text,
                    "runtime_kind": self.runtime_kind.value,
                },
            )
            if repeated_progress or continuation_cap_reached:
                reason = "repeated_progress" if repeated_progress else "continuation_cap"
                message = _runtime_text(non_final_progress_stopped_key)
                self.store.append(
                    non_final_incomplete_event,
                    {
                        "step": step,
                        "attempts": one_shot_non_final_continuations,
                        "reason": reason,
                        "content": final_text,
                        "runtime_kind": self.runtime_kind.value,
                    },
                )
                _emit_surface_error(
                    self.surface,
                    "execution_guard_error",
                    message,
                    True,
                )
                self._emit_forced_final_summary_before_termination(
                    reason=(
                        "non_final_progress_retry_exhausted"
                        if repeated_progress
                        else "non_final_progress_continuation_cap_reached"
                    ),
                    termination_cause=(
                        "repeated non-final progress is detected"
                        if repeated_progress
                        else "the non-final progress continuation limit is reached"
                    ),
                    max_steps=_current_turn_step_limit(),
                    language=turn_language,
                    script=turn_script,
                    explicit_language_override=turn_language_explicit,
                    latest_assistant_text=final_text,
                )
                assistant_message_emitted = True
                return _finish_turn(
                    1,
                    reason=(
                        "non_final_progress_retry_exhausted"
                        if repeated_progress
                        else "non_final_progress_continuation_cap_reached"
                    ),
                    final_text=final_text,
                )

            one_shot_non_final_continuations += 1
            if fingerprint:
                non_final_progress_fingerprints.add(fingerprint)
            assistant_message = assistant_message_from_response(resp, content=final_text)
            self.messages.append(assistant_message)
            self.store.append(
                "assistant_message",
                {"content": final_text, "message": assistant_message},
            )
            continuation_nudge = _runtime_text(continuation_nudge_key)
            self.messages.append({"role": "system", "content": continuation_nudge})
            self.store.append(
                "continuation_nudge",
                {
                    "step": step,
                    "attempt": one_shot_non_final_continuations,
                    "message": continuation_nudge,
                    "runtime_kind": self.runtime_kind.value,
                },
            )
            _phase_update_key(
                "phase_continuing_one_shot"
                if self.one_shot_execution
                else "phase_continuing_execution"
            )
            continue

        if completion_gate_enabled:
            blocked_response = _assistant_text_has_blocker_marker(final_text)
            blocked_response_allows_completion = _completion_gate_blocker_allows_final(
                state=execution_state,
                blocked_response=blocked_response,
            )
            verification_expected = _verification_expected_for_turn(
                turn_intent=repo_turn_execution_intent,
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
                require_material_edit_evidence=(
                    self.one_shot_execution
                    or repo_tool_activity_observed
                    or _assistant_text_has_completion_marker(final_text)
                ),
            )
            if gate_problems:
                gate_stage = _completion_gate_repair_stage(gate_problems)
                stage_limit = _completion_gate_stage_attempt_limit(gate_stage)
                if not self.one_shot_execution and gate_stage in {
                    "verification_failed",
                    "verification_incomplete",
                }:
                    stage_limit += 1
                stage_attempts = execution_state.repair_attempts_for_stage(gate_stage)
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
                        "stage_attempt": stage_attempts + 1,
                        "stage_limit": stage_limit,
                        "blocked_response": blocked_response,
                        "blocked_response_allows_completion": (blocked_response_allows_completion),
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
                        "attempt": execution_state.completion_gate_repair_attempts + 1,
                    },
                )
                if stage_attempts >= stage_limit:
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
                                "repo_tool_activity_observed": repo_tool_activity_observed,
                                "anchor_paths": no_material_anchor_paths,
                                "state": execution_state.as_payload(),
                                "content": final_text,
                                "attempts": execution_state.completion_gate_repair_attempts,
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
                            "attempts": execution_state.completion_gate_repair_attempts,
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
                        termination_cause="completion-gate repair attempts are exhausted",
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
                self.messages.append({"role": "system", "content": nudge})
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
                        },
                    )
                _phase_update_key("phase_completion_gate_repair")
                continue

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
        exhausted_stage_limit = _completion_gate_stage_attempt_limit(exhausted_stage)
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
        max_steps=_current_turn_step_limit(),
        language=turn_language,
        script=turn_script,
        explicit_language_override=turn_language_explicit,
    )
    assistant_message_emitted = True
    return _finish_turn(1, reason="max_steps_exceeded")
