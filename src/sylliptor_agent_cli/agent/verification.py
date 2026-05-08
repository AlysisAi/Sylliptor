from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..diff_paths import iter_patch_paths
from ..failure_category import FailureCategory, is_infra_unavailable_error
from ..language_policy import normalize_language_name
from ..runtime_kind import RuntimeKind
from ..tools.availability import is_tool_unavailable_result
from ..verify_gate import (
    assess_verification_command_execution,
    extract_actionable_failure_snippet,
    extract_verification_failure_snippet,
    is_authoritative_verify_command_selection,
    is_toolchain_unavailable_verification_output,
    resolve_task_aware_verify_command_selection,
    verification_selection_payload,
)
from ..verify_gate import run_task_verification as run_task_verification
from .prompt_context import (
    MAX_POST_EXPLORE_ANCHOR_PATHS,
    _extract_workspace_relation_paths_from_text,
    _normalize_repo_relative_hint_path,
    _paths_require_verification,
    _session_repo_scan,
    _session_task_brief_content,
    _session_verify_command_selection,
    _task_brief_lines_from_text,
    refresh_session_environment_context_message,
)
from .verification_commands import (
    _has_disallowed_shell_control_flow,
    _matching_effective_verification_commands,
    _shell_command_is_verification_attempt,
)

if TYPE_CHECKING:
    from .routing import _OneShotRepoTurnIntent


_COMMAND_LIKE_MUTATION_TOOL_NAMES = {"verify_run", "shell_run"}
_MATERIAL_EDIT_TOOL_NAMES = {
    "fs_write",
    "fs_edit",
    "git_apply_patch",
    "fs_move",
    "fs_copy",
    "fs_delete",
    "fs_mkdir",
}
_VERIFICATION_SHELL_MARKERS = (
    "pytest",
    "py.test",
    "unittest",
    "tox",
    "nox",
    "go test",
    "cargo test",
    "npm test",
    "pnpm test",
    "yarn test",
    "vitest",
    "jest",
    "ruff check",
    "mypy",
    "flake8",
    "pylint",
    "make test",
    "make check",
)
_COMPLETION_GATE_PROBLEM_LABELS = {
    "empty_final_response": "empty final response",
    "no_material_edits": "no material edits",
    "verification_not_attempted": "verification not attempted",
    "verification_incomplete": "verification coverage incomplete",
    "verification_failed": "verification failing",
}
_COMPLETION_GATE_REPAIR_MESSAGES = {
    "empty_final_response": (
        "Provide a non-empty final response describing exactly what changed or the concrete blocker."
    ),
    "no_material_edits": (
        "You have not made any material repository edits yet. Do not finalize or summarize yet. "
        "Your next step must be a real action/progress tool call that starts implementation now "
        "(for example fs_edit, git_apply_patch, fs_write, verify_run, or shell_run), or a "
        "concrete blocker report."
    ),
    "verification_not_attempted": (
        "Run the session's configured verification now. Prefer verify_run with no arguments. "
        "Do not use piped, grepped, help/list/build-only, or otherwise filtered shell commands as verification."
    ),
    "verification_incomplete": (
        "Run the remaining commands required by the session's effective verification contract exactly. "
        "Do not substitute a different build system or a different test command."
    ),
    "verification_failed": (
        "Your latest verification attempt failed. Fix issues and rerun verification until it passes, "
        "or report a concrete blocker such as a missing wrapper or incompatible host toolchain. "
        "Do not claim success based on an alternate command that is outside the configured verification contract."
    ),
}
_COMPLETION_GATE_REPAIR_MESSAGES_BY_LOCALE = {
    "english": _COMPLETION_GATE_REPAIR_MESSAGES,
}
_COMPLETION_GATE_REPAIR_STAGE_LIMITS = {
    "generic": 1,
    "no_material_edits": 1,
    "verification_not_attempted": 1,
    "verification_incomplete": 1,
    "verification_failed": 1,
}
_ONE_SHOT_COMPLETION_GATE_NUDGE_PREFIX = (
    "Completion gate: this one-shot execution run cannot finalize yet."
)
_RUNTIME_DEFAULT_LANGUAGE = "english"
_RUNTIME_MESSAGE_CATALOG: dict[str, dict[str, str]] = {
    "english": {
        "phase_understanding_request": "Understanding your request.",
        "phase_drafting_response": "Drafting response.",
        "phase_compacted_history": "Compacted conversation history.",
        "phase_retrying_step": "Retrying with higher temperature for this step.",
        "phase_running_tool_steps": "Running {count} tool step(s): {names}.",
        "phase_post_explore_bootstrap": (
            "Detected post-explore stagnation; nudging implementation bootstrap."
        ),
        "phase_exploration_stagnation": (
            "Detected exploration stagnation; nudging toward implementation."
        ),
        "phase_failed_edit_loop": "Detected failed edit loop; nudging strategy switch.",
        "phase_continuing_one_shot": (
            "Continuing one-shot execution after non-final progress update."
        ),
        "phase_continuing_execution": "Continuing execution after non-final progress update.",
        "phase_completion_gate_repair": (
            "Completion gate detected missing execution evidence; requesting one repair pass."
        ),
        "phase_step_budget_handoff": (
            "Step budget exhausted; preparing a concise handoff so the chat can continue."
        ),
        "phase_writing_final_response": "Writing the final response.",
        "one_shot_continuation_nudge": (
            "Continue execution now. Do not stop at planning/progress updates. "
            "Implement the requested changes and run relevant verification, "
            "or explain a concrete blocker."
        ),
        "interactive_continuation_nudge": (
            "Continue execution now. Do not stop at a planning/progress update. "
            "Use tools to make progress, run relevant verification, or explain a concrete blocker."
        ),
        "one_shot_exploration_nudge": (
            "Avoid repeated read-only exploration. Start implementing now, delegate once "
            "to a suitable available subagent if more investigation is genuinely needed, "
            "or explain a concrete blocker."
        ),
        "one_shot_post_explore_bootstrap_nudge": (
            "A subagent already returned useful context in this one-shot turn. You now have enough "
            "context to start implementation. Do not call the same research subagent again "
            "in this turn. Do not use more read-only tools unless there is a concrete blocker. "
            "Your next step must be an action/progress tool (for example fs_edit, "
            "git_apply_patch, fs_write, verify_run, or shell_run) or a concrete blocker report."
        ),
        "one_shot_post_explore_bootstrap_targets": ("Likely repo-root-relative targets: {joined}."),
        "one_shot_edit_strategy_nudge": (
            "Edit strategy is stuck. Switch approach now: fs_edit ops are replace_exact, "
            "insert_before_exact, insert_after_exact, append, prepend (replace is tolerated as "
            "alias of replace_exact). Re-read the target file before retrying. If localized "
            "fs_edit is a poor fit, use git_apply_patch or fs_write. Do not repeat the same "
            "failing edit call."
        ),
        "one_shot_non_final_progress_stopped": (
            "One-shot run stopped: model returned repeated/non-final progress text "
            "without continuing implementation."
        ),
        "interactive_non_final_progress_stopped": (
            "Execution turn stopped: model returned repeated/non-final progress text "
            "without continuing implementation."
        ),
        "one_shot_post_explore_retry_exhausted": (
            "One-shot run stopped: post-explore stagnation persisted after bounded "
            "implementation-bootstrap nudges. Start implementing now or report a concrete blocker."
        ),
        "one_shot_exploration_retry_exhausted": (
            "One-shot run stopped: exploration stagnation persisted after bounded nudges. "
            "Start implementing, delegate once to a suitable available subagent if more "
            "investigation is genuinely needed, or report a concrete blocker."
        ),
        "one_shot_edit_retry_exhausted": (
            "One-shot run stopped: failed edit/write loop persisted after bounded strategy "
            "nudges. Switch to exact-match fs_edit ops, or use git_apply_patch/fs_write, "
            "or report a concrete blocker."
        ),
        "one_shot_post_explore_step_budget_exhausted": (
            "One-shot run stopped: post-explore stagnation consumed the step budget. "
            "Start implementing now or report a concrete blocker."
        ),
        "one_shot_exploration_step_budget_exhausted": (
            "One-shot run stopped: exploration stagnation consumed the step budget. "
            "Start implementing, delegate once to a suitable available subagent if more "
            "investigation is genuinely needed, or report a concrete blocker."
        ),
        "one_shot_edit_step_budget_exhausted": (
            "One-shot run stopped: failed edit/write loop consumed the step budget. "
            "Switch to exact-match fs_edit ops, or use git_apply_patch/fs_write, "
            "or report a concrete blocker."
        ),
        "completion_gate_nudge_prefix": _ONE_SHOT_COMPLETION_GATE_NUDGE_PREFIX,
        "interactive_completion_gate_nudge_prefix": (
            "Completion gate: this interactive execution turn cannot finalize yet."
        ),
        "completion_gate_fallback_suffix": "Use tools to complete the requested work.",
        "completion_gate_first_reported_error": "First reported error: {snippet}.",
        "completion_gate_fix_then_rerun": "Fix that, rerun verification, then summarize.",
        "completion_gate_terminal_failure": (
            "One-shot run stopped: completion gate requirements were not met ({problem_summary})."
        ),
        "interactive_completion_gate_terminal_failure": (
            "Execution turn stopped: completion gate requirements were not met ({problem_summary})."
        ),
        "completion_gate_step_budget_exhausted": (
            "One-shot run stopped: completion gate repair attempt consumed the step budget."
        ),
        "interactive_completion_gate_step_budget_exhausted": (
            "Execution turn stopped: completion gate repair attempt consumed the step budget."
        ),
        "max_steps_exceeded": "max_steps exceeded",
    },
}


@dataclass
class TurnExecutionState:
    execution_requested: bool
    expected_verification_commands: set[str] = field(default_factory=set)
    covered_verification_commands: set[str] = field(default_factory=set)
    covered_verification_command_generations: dict[str, int] = field(default_factory=dict)
    material_edit_count: int = 0
    material_edit_tools: set[str] = field(default_factory=set)
    touched_repo_paths: set[str] = field(default_factory=set)
    verification_attempt_count: int = 0
    verification_tools: set[str] = field(default_factory=set)
    last_verification_passed: bool | None = None
    last_verification_failure_snippet: str = ""
    last_verification_failure_category: str = ""
    failed_verification_command_snippets: dict[str, str] = field(default_factory=dict)
    verification_relevant_edit_generation: int = 0
    last_successful_verification_generation: int | None = None
    completion_gate_repair_attempts: int = 0
    completion_gate_no_material_edits_repair_attempts: int = 0
    completion_gate_missing_verify_repair_attempts: int = 0
    completion_gate_failed_verify_repair_attempts: int = 0

    def refresh_verification_coverage(self) -> None:
        self.covered_verification_commands = {
            command
            for command, generation in self.covered_verification_command_generations.items()
            if generation == self.verification_relevant_edit_generation
        }

    def note_verification_relevant_edit(self) -> None:
        self.verification_relevant_edit_generation += 1
        self.refresh_verification_coverage()

    def record_verification_coverage(self, commands: set[str]) -> None:
        if not commands:
            return
        for command in commands:
            self.covered_verification_command_generations[command] = (
                self.verification_relevant_edit_generation
            )
            self.failed_verification_command_snippets.pop(command, None)
        self.last_successful_verification_generation = self.verification_relevant_edit_generation
        self.refresh_verification_coverage()

    def record_verification_failures(self, failures: dict[str, str]) -> None:
        for command, snippet in failures.items():
            clean_command = str(command or "").strip()
            if not clean_command:
                continue
            clean_snippet = str(snippet or "").strip()
            self.failed_verification_command_snippets[clean_command] = clean_snippet

    def missing_verification_commands(self) -> set[str]:
        return self.expected_verification_commands - self.covered_verification_commands

    def failed_verification_commands(self) -> set[str]:
        return set(self.failed_verification_command_snippets) & self.expected_verification_commands

    def first_failed_verification_snippet(self) -> str:
        for command in sorted(self.failed_verification_commands()):
            snippet = self.failed_verification_command_snippets.get(command, "")
            if snippet:
                return snippet
        return ""

    def verification_coverage_is_stale(self) -> bool:
        return (
            bool(self.expected_verification_commands)
            and self.last_successful_verification_generation is not None
            and self.last_successful_verification_generation
            < self.verification_relevant_edit_generation
        )

    def repair_attempts_for_stage(self, stage: str) -> int:
        if stage == "no_material_edits":
            return self.completion_gate_no_material_edits_repair_attempts
        if stage == "verification_not_attempted":
            return self.completion_gate_missing_verify_repair_attempts
        if stage == "verification_incomplete":
            return self.completion_gate_missing_verify_repair_attempts
        if stage == "verification_failed":
            return self.completion_gate_failed_verify_repair_attempts
        return self.completion_gate_repair_attempts

    def increment_repair_attempts_for_stage(self, stage: str) -> None:
        self.completion_gate_repair_attempts += 1
        if stage == "no_material_edits":
            self.completion_gate_no_material_edits_repair_attempts += 1
        elif stage == "verification_not_attempted":
            self.completion_gate_missing_verify_repair_attempts += 1
        elif stage == "verification_incomplete":
            self.completion_gate_missing_verify_repair_attempts += 1
        elif stage == "verification_failed":
            self.completion_gate_failed_verify_repair_attempts += 1

    def as_payload(self) -> dict[str, Any]:
        return {
            "execution_requested": self.execution_requested,
            "expected_verification_commands": sorted(self.expected_verification_commands),
            "covered_verification_commands": sorted(self.covered_verification_commands),
            "missing_verification_commands": sorted(self.missing_verification_commands()),
            "material_edit_count": self.material_edit_count,
            "material_edit_tools": sorted(self.material_edit_tools),
            "touched_repo_paths": sorted(self.touched_repo_paths),
            "verification_attempt_count": self.verification_attempt_count,
            "verification_tools": sorted(self.verification_tools),
            "last_verification_passed": self.last_verification_passed,
            "last_verification_failure_snippet": self.last_verification_failure_snippet,
            "last_verification_failure_category": self.last_verification_failure_category,
            "failed_verification_commands": sorted(self.failed_verification_commands()),
            "verification_relevant_edit_generation": self.verification_relevant_edit_generation,
            "last_successful_verification_generation": self.last_successful_verification_generation,
            "verification_coverage_stale": self.verification_coverage_is_stale(),
            "completion_gate_repair_attempts": self.completion_gate_repair_attempts,
            "completion_gate_no_material_edits_repair_attempts": self.completion_gate_no_material_edits_repair_attempts,
            "completion_gate_missing_verify_repair_attempts": self.completion_gate_missing_verify_repair_attempts,
            "completion_gate_failed_verify_repair_attempts": self.completion_gate_failed_verify_repair_attempts,
        }


def _runtime_message_locale(
    *,
    language: str = "",
    explicit_language_override: bool = False,
) -> str:
    if not explicit_language_override:
        return _RUNTIME_DEFAULT_LANGUAGE
    normalized = normalize_language_name(language).casefold()
    if normalized in _RUNTIME_MESSAGE_CATALOG:
        return normalized
    return _RUNTIME_DEFAULT_LANGUAGE


def _runtime_message(
    key: str,
    *,
    language: str = "",
    explicit_language_override: bool = False,
    **kwargs: Any,
) -> str:
    locale = _runtime_message_locale(
        language=language,
        explicit_language_override=explicit_language_override,
    )
    template = _RUNTIME_MESSAGE_CATALOG.get(locale, {}).get(key)
    if template is None:
        template = _RUNTIME_MESSAGE_CATALOG[_RUNTIME_DEFAULT_LANGUAGE].get(key, key)
    try:
        return template.format(**kwargs)
    except Exception:  # noqa: BLE001
        return template


def _extract_touched_repo_paths(
    *,
    root: Path,
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> set[str]:
    normalized_tool = tool_name.strip().lower()
    raw_paths: list[str] = []

    if normalized_tool in {"fs_write", "fs_edit", "fs_delete", "fs_mkdir"}:
        raw_path = result.get("path", arguments.get("path"))
        if isinstance(raw_path, str):
            raw_paths.append(raw_path)
    elif normalized_tool in {"fs_move", "fs_copy"}:
        for key in ("source_path", "destination_path"):
            raw_path = result.get(key, arguments.get(key))
            if isinstance(raw_path, str):
                raw_paths.append(raw_path)
    elif normalized_tool == "git_apply_patch":
        patch = str(arguments.get("patch") or "")
        raw_paths.extend(iter_patch_paths(patch))
    elif normalized_tool in _COMMAND_LIKE_MUTATION_TOOL_NAMES:
        touched_paths = result.get("touched_repo_paths")
        if isinstance(touched_paths, list):
            raw_paths.extend(str(item) for item in touched_paths if isinstance(item, str))

    touched: set[str] = set()
    for raw_path in raw_paths:
        normalized = _normalize_repo_relative_hint_path(root=root, raw=raw_path)
        if normalized:
            touched.add(normalized)
    return touched


def _verification_attempt_passed(
    *,
    tool_name: str,
    status: str,
    result: dict[str, Any],
) -> bool:
    if status == "failed":
        return False
    normalized_tool = tool_name.strip().lower()
    touched_repo_paths = result.get("touched_repo_paths")
    normalized_touched = (
        {str(item) for item in touched_repo_paths if isinstance(item, str) and str(item).strip()}
        if isinstance(touched_repo_paths, list)
        else set()
    )
    if normalized_tool == "verify_run":
        commands = result.get("commands")
        if isinstance(commands, list) and any(
            _has_disallowed_shell_control_flow(str(command)) for command in commands
        ):
            return False
        if normalized_touched and _paths_require_verification(normalized_touched):
            return False
        all_passed = result.get("all_passed")
        if isinstance(all_passed, bool):
            return all_passed
        command_results = result.get("command_results")
        if isinstance(command_results, list):
            checks: list[bool] = []
            for item in command_results:
                if not isinstance(item, dict):
                    checks.append(False)
                    continue
                real_execution = item.get("real_execution")
                if real_execution is False:
                    checks.append(False)
                    continue
                ok = item.get("ok")
                if isinstance(ok, bool):
                    checks.append(ok)
                    continue
                exit_code = item.get("exit_code")
                checks.append(isinstance(exit_code, int) and exit_code == 0)
            return bool(checks) and all(checks)
        return False
    if normalized_tool == "shell_run":
        exit_code = result.get("exit_code")
        if not (isinstance(exit_code, int) and exit_code == 0):
            return False
        if normalized_touched and _paths_require_verification(normalized_touched):
            return False
        output = "\n".join(
            [
                str(result.get("stdout") or "").strip(),
                str(result.get("stderr") or "").strip(),
            ]
        ).strip()
        assessment = assess_verification_command_execution(
            command=str(result.get("effective_cmd") or result.get("cmd") or ""),
            exit_code=exit_code,
            output=output,
        )
        return assessment.real_execution is not False
    return False


def _verification_command_result_passed(item: dict[str, Any]) -> bool:
    real_execution = item.get("real_execution")
    if real_execution is False:
        return False
    ok = item.get("ok")
    if isinstance(ok, bool):
        return ok
    exit_code = item.get("exit_code")
    return isinstance(exit_code, int) and exit_code == 0


def _verification_command_result_snippet(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("output_preview") or "").strip(),
        str(item.get("output") or "").strip(),
        str(item.get("stderr") or "").strip(),
        str(item.get("stdout") or "").strip(),
    ]
    text = "\n".join(part for part in parts if part)
    snippet = extract_actionable_failure_snippet(text)
    return snippet or (text[:240].rstrip() if text else "")


def _verification_failure_category_for_tool_result(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    result: dict[str, Any],
) -> str:
    normalized_tool = tool_name.strip().lower()
    if normalized_tool == "verify_run":
        category = str(result.get("failure_category") or "").strip()
        return category or FailureCategory.VERIFICATION_FAILED.value

    if normalized_tool == "shell_run":
        output = "\n".join(
            [
                str(result.get("stdout") or "").strip(),
                str(result.get("stderr") or "").strip(),
            ]
        ).strip()
        command = str(
            result.get("effective_cmd") or result.get("cmd") or arguments.get("cmd") or ""
        )
        exit_code_raw = result.get("exit_code")
        exit_code = exit_code_raw if isinstance(exit_code_raw, int) else 1
        assessment = assess_verification_command_execution(
            command=command,
            exit_code=exit_code,
            output=output,
        )
        if (
            assessment.non_execution_reason == "execution_layer_failure"
            or is_infra_unavailable_error(output)
            or is_toolchain_unavailable_verification_output(output)
        ):
            return FailureCategory.INFRA_UNAVAILABLE.value

    return FailureCategory.VERIFICATION_FAILED.value


def _record_verify_run_command_outcomes(
    *,
    state: TurnExecutionState,
    result: dict[str, Any],
    known_verification_commands: list[str] | None,
) -> None:
    command_results = result.get("command_results")
    if not isinstance(command_results, list):
        if result.get("all_passed") is True:
            commands = result.get("commands")
            if isinstance(commands, list):
                covered: set[str] = set()
                for command in commands:
                    covered.update(
                        _matching_effective_verification_commands(
                            observed_command=str(command),
                            effective_verification_commands=known_verification_commands,
                        )
                    )
                state.record_verification_coverage(covered)
        return

    covered: set[str] = set()
    failures: dict[str, str] = {}
    for item in command_results:
        if not isinstance(item, dict):
            continue
        matches: set[str] = set()
        observed_candidates = [
            str(item.get("command") or ""),
            str(item.get("effective_command") or ""),
        ]
        for observed in observed_candidates:
            if not observed:
                continue
            matches.update(
                _matching_effective_verification_commands(
                    observed_command=observed,
                    effective_verification_commands=known_verification_commands,
                )
            )
        if not matches:
            continue
        if _verification_command_result_passed(item):
            covered.update(matches)
            continue
        snippet = _verification_command_result_snippet(item)
        for command in matches:
            failures[command] = f"{command}: {snippet}" if snippet else command

    state.record_verification_coverage(covered)
    state.record_verification_failures(failures)


def _record_shell_verification_command_outcome(
    *,
    state: TurnExecutionState,
    arguments: dict[str, Any],
    result: dict[str, Any],
    known_verification_commands: list[str] | None,
    passed: bool,
) -> None:
    matches = _matching_effective_verification_commands(
        observed_command=str(result.get("effective_cmd") or arguments.get("cmd") or ""),
        effective_verification_commands=known_verification_commands,
    )
    if passed:
        state.record_verification_coverage(matches)
        return
    output = "\n".join(
        [
            str(result.get("stdout") or "").strip(),
            str(result.get("stderr") or "").strip(),
        ]
    ).strip()
    snippet = extract_actionable_failure_snippet(output) or output[:240].rstrip()
    state.record_verification_failures(
        {command: f"{command}: {snippet}" if snippet else command for command in matches}
    )


def _record_tool_effect(
    *,
    root: Path,
    state: TurnExecutionState,
    tool_name: str,
    arguments: dict[str, Any],
    status: str,
    result: dict[str, Any],
    known_verification_commands: list[str] | None,
) -> None:
    if is_tool_unavailable_result(result):
        return
    normalized_tool = tool_name.strip().lower()
    touched_paths: set[str] = set()
    if normalized_tool in _COMMAND_LIKE_MUTATION_TOOL_NAMES:
        touched_paths = _extract_touched_repo_paths(
            root=root,
            tool_name=normalized_tool,
            arguments=arguments,
            result=result,
        )
    elif status != "failed" and normalized_tool in _MATERIAL_EDIT_TOOL_NAMES:
        touched_paths = _extract_touched_repo_paths(
            root=root,
            tool_name=normalized_tool,
            arguments=arguments,
            result=result,
        )
    if status != "failed" and normalized_tool in _MATERIAL_EDIT_TOOL_NAMES:
        state.material_edit_count += 1
        state.material_edit_tools.add(normalized_tool)
        state.touched_repo_paths.update(touched_paths)
        if _paths_require_verification(touched_paths):
            state.note_verification_relevant_edit()
    elif normalized_tool in _COMMAND_LIKE_MUTATION_TOOL_NAMES and touched_paths:
        state.material_edit_count += 1
        state.material_edit_tools.add(normalized_tool)
        state.touched_repo_paths.update(touched_paths)
        if _paths_require_verification(touched_paths):
            state.note_verification_relevant_edit()

    verification_attempt = False
    if normalized_tool == "verify_run":
        verification_attempt = True
    elif normalized_tool == "shell_run":
        verification_attempt = _shell_command_is_verification_attempt(
            str(arguments.get("cmd") or ""),
            known_verification_commands=known_verification_commands,
        )
    if not verification_attempt:
        return

    state.verification_attempt_count += 1
    state.verification_tools.add(normalized_tool)
    state.last_verification_passed = _verification_attempt_passed(
        tool_name=normalized_tool,
        status=status,
        result=result,
    )
    if normalized_tool == "verify_run":
        _record_verify_run_command_outcomes(
            state=state,
            result=result,
            known_verification_commands=known_verification_commands,
        )
    elif normalized_tool == "shell_run":
        _record_shell_verification_command_outcome(
            state=state,
            arguments=arguments,
            result=result,
            known_verification_commands=known_verification_commands,
            passed=state.last_verification_passed is True,
        )

    if state.last_verification_passed is True:
        state.last_verification_failure_category = ""
        if not state.failed_verification_commands():
            state.last_verification_failure_snippet = ""
    else:
        state.last_verification_failure_category = _verification_failure_category_for_tool_result(
            tool_name=normalized_tool,
            arguments=arguments,
            result=result,
        )
        state.last_verification_failure_snippet = (
            extract_verification_failure_snippet(
                tool_name=normalized_tool,
                result=result,
            )
            or state.first_failed_verification_snippet()
        )


def _verification_expected_for_turn(
    *,
    turn_intent: _OneShotRepoTurnIntent,
    blocked: bool,
    touched_repo_paths: set[str],
    verification_contract_requires_execution: bool = False,
) -> bool:
    if turn_intent != "execute":
        return False
    if verification_contract_requires_execution:
        return True
    if blocked:
        return False
    return _paths_require_verification(touched_repo_paths)


def _completion_gate_blocker_allows_final(
    *,
    state: TurnExecutionState,
    blocked_response: bool,
) -> bool:
    if not blocked_response:
        return False
    if not state.touched_repo_paths or not _paths_require_verification(state.touched_repo_paths):
        return True
    if state.verification_attempt_count <= 0:
        return False
    if state.last_verification_passed is True:
        return True
    return state.last_verification_failure_category == FailureCategory.INFRA_UNAVAILABLE.value


def _completion_gate_problems(
    *,
    state: TurnExecutionState,
    final_text: str,
    blocked: bool,
    verification_expected: bool,
    require_material_edit_evidence: bool = True,
) -> list[str]:
    problems: list[str] = []
    if not final_text.strip():
        problems.append("empty_final_response")
    if require_material_edit_evidence and not blocked and state.material_edit_count <= 0:
        problems.append("no_material_edits")
    if verification_expected:
        if state.verification_attempt_count <= 0:
            problems.append("verification_not_attempted")
        elif state.failed_verification_commands():
            problems.append("verification_failed")
        elif state.last_verification_passed is not True:
            problems.append("verification_failed")
        elif state.expected_verification_commands and state.missing_verification_commands():
            problems.append("verification_incomplete")
    return problems


def _sorted_missing_verification_commands(state: TurnExecutionState) -> list[str]:
    return sorted(state.missing_verification_commands())


def _completion_gate_problem_summary(problems: list[str]) -> str:
    labels = [_COMPLETION_GATE_PROBLEM_LABELS.get(item, item) for item in problems]
    return ", ".join(labels) if labels else "unknown completion gate failure"


def _completion_gate_repair_stage(problems: list[str]) -> str:
    if "no_material_edits" in problems:
        return "no_material_edits"
    if "verification_failed" in problems:
        return "verification_failed"
    if "verification_incomplete" in problems:
        return "verification_incomplete"
    if "verification_not_attempted" in problems:
        return "verification_not_attempted"
    return "generic"


def _completion_gate_stage_attempt_limit(stage: str) -> int:
    return int(_COMPLETION_GATE_REPAIR_STAGE_LIMITS.get(stage, 1))


def _completion_gate_repair_message(
    problem: str,
    *,
    language: str = "",
    explicit_language_override: bool = False,
) -> str:
    locale = _runtime_message_locale(
        language=language,
        explicit_language_override=explicit_language_override,
    )
    messages = _COMPLETION_GATE_REPAIR_MESSAGES_BY_LOCALE.get(
        locale,
        _COMPLETION_GATE_REPAIR_MESSAGES,
    )
    return messages.get(problem, "")


def _completion_gate_nudge_message(
    problems: list[str],
    *,
    prefix_key: str = "completion_gate_nudge_prefix",
    verification_failure_snippet: str = "",
    missing_verification_commands: list[str] | None = None,
    verification_coverage_stale: bool = False,
    anchor_paths: list[str] | None = None,
    language: str = "",
    explicit_language_override: bool = False,
) -> str:
    details: list[str] = []
    for item in problems:
        detail = _completion_gate_repair_message(
            item,
            language=language,
            explicit_language_override=explicit_language_override,
        )
        if detail:
            details.append(detail)
    suffix = (
        " ".join(details)
        if details
        else _runtime_message(
            "completion_gate_fallback_suffix",
            language=language,
            explicit_language_override=explicit_language_override,
        )
    )
    snippet = extract_actionable_failure_snippet(verification_failure_snippet)
    if snippet and "verification_failed" in problems:
        suffix = " ".join(
            [
                suffix,
                _runtime_message(
                    "completion_gate_first_reported_error",
                    language=language,
                    explicit_language_override=explicit_language_override,
                    snippet=snippet,
                ),
                _runtime_message(
                    "completion_gate_fix_then_rerun",
                    language=language,
                    explicit_language_override=explicit_language_override,
                ),
            ]
        )
    if missing_verification_commands and "verification_incomplete" in problems:
        locale = _runtime_message_locale(
            language=language,
            explicit_language_override=explicit_language_override,
        )
        stale_detail = (
            "Later verification-relevant edits invalidated earlier verification coverage."
            if verification_coverage_stale and locale == "english"
            else (
                "Μεταγενέστερες αλλαγές που απαιτούν verification ακύρωσαν το προηγούμενο verification."
                if verification_coverage_stale
                else ""
            )
        )
        missing_label = (
            "Missing verification commands:"
            if locale == "english"
            else "Λείπουν verification commands:"
        )
        suffix = " ".join(
            [
                item
                for item in (
                    suffix,
                    stale_detail,
                    missing_label,
                    ", ".join(missing_verification_commands) + ".",
                )
                if item
            ]
        )
    if anchor_paths and "no_material_edits" in problems:
        suffix = " ".join(
            [
                suffix,
                _runtime_message(
                    "one_shot_post_explore_bootstrap_targets",
                    language=language,
                    explicit_language_override=explicit_language_override,
                    joined=", ".join(anchor_paths[:MAX_POST_EXPLORE_ANCHOR_PATHS]),
                ),
            ]
        )
    prefix = _runtime_message(
        prefix_key,
        language=language,
        explicit_language_override=explicit_language_override,
    )
    return f"{prefix} {suffix}"


def _completion_gate_terminal_failure_message(
    *,
    problem_summary: str,
    stage: str,
    message_key: str = "completion_gate_terminal_failure",
    verification_failure_snippet: str = "",
    language: str = "",
    explicit_language_override: bool = False,
) -> str:
    snippet = extract_actionable_failure_snippet(verification_failure_snippet)
    message = _runtime_message(
        message_key,
        language=language,
        explicit_language_override=explicit_language_override,
        problem_summary=problem_summary,
    )
    if snippet and stage == "verification_failed":
        message += " " + _runtime_message(
            "completion_gate_first_reported_error",
            language=language,
            explicit_language_override=explicit_language_override,
            snippet=snippet,
        )
    return message


def _completion_gate_step_budget_exhausted_message(
    *,
    stage: str,
    message_key: str = "completion_gate_step_budget_exhausted",
    verification_failure_snippet: str = "",
    language: str = "",
    explicit_language_override: bool = False,
) -> str:
    snippet = extract_actionable_failure_snippet(verification_failure_snippet)
    message = _runtime_message(
        message_key,
        language=language,
        explicit_language_override=explicit_language_override,
    )
    if snippet and stage == "verification_failed":
        message += " " + _runtime_message(
            "completion_gate_first_reported_error",
            language=language,
            explicit_language_override=explicit_language_override,
            snippet=snippet,
        )
    return message


def _build_interactive_turn_verify_task(
    *,
    session: Any,
    instruction: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    task_paths = _extract_workspace_relation_paths_from_text(root=session.root, text=instruction)
    task_brief = _session_task_brief_content(session)
    if task_brief:
        for path in _extract_workspace_relation_paths_from_text(root=session.root, text=task_brief):
            if path not in task_paths:
                task_paths.append(path)
    task_texts = [str(instruction or "").strip()]
    if task_brief:
        task_texts.extend(_task_brief_lines_from_text(task_brief, max_lines=6))
    task_texts = [text for text in task_texts if text]
    if not task_paths and not task_texts:
        return None, []
    task: dict[str, Any] = {}
    if task_paths:
        task["estimated_files"] = list(task_paths)
        task["write_scope"] = list(task_paths)
    if task_texts:
        task["acceptance_criteria"] = list(task_texts)
    return task, task_texts


def _refresh_interactive_turn_verification_selection(
    session: Any,
    *,
    instruction: str,
    route_execution_posture: str,
) -> None:
    if not bool(getattr(session, "verification_enabled", True)):
        return
    if (
        getattr(session, "runtime_kind", RuntimeKind.INTERACTIVE_CHAT)
        != RuntimeKind.INTERACTIVE_CHAT
    ):
        return
    if getattr(session, "authoritative_verification_commands", None) is not None:
        return
    if str(route_execution_posture or "").strip().lower() != "execute":
        return

    repo_scan = _session_repo_scan(session)
    task, plan_requirements = _build_interactive_turn_verify_task(
        session=session,
        instruction=instruction,
    )
    current = _session_verify_command_selection(session)
    resolved = resolve_task_aware_verify_command_selection(
        cfg=session.cfg,
        verify_cmd=None,
        task=task,
        root=session.root,
        repo_scan=repo_scan,
        plan_requirements=plan_requirements,
        selection=current,
    )
    if (
        current is not None
        and current.commands == resolved.commands
        and current.source == resolved.source
        and current.reason == resolved.reason
        and current.contract_type == resolved.contract_type
    ):
        return

    previous_payload = (
        verification_selection_payload(
            current,
            authoritative=is_authoritative_verify_command_selection(current),
        )
        if current is not None
        else None
    )
    session.effective_verification_commands = list(resolved.commands)
    session.verification_selection_source = resolved.source
    session.verification_selection_reason = resolved.reason
    session.verification_contract_type = resolved.contract_type
    session.verification_authoritative = is_authoritative_verify_command_selection(resolved)
    refresh_session_environment_context_message(session)
    payload: dict[str, Any] = {
        "instruction_paths": list(task.get("estimated_files", []))
        if isinstance(task, dict)
        else [],
        "route_execution_posture": route_execution_posture,
        **verification_selection_payload(
            resolved,
            authoritative=is_authoritative_verify_command_selection(resolved),
        ),
    }
    if previous_payload is not None:
        payload["previous"] = previous_payload
    session.store.append("verification_contract_updated", payload)
