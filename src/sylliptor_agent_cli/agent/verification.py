from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..diff_paths import iter_patch_paths
from ..failure_category import FailureCategory, is_infra_unavailable_error
from ..language_policy import normalize_language_name
from ..runtime_kind import RuntimeKind
from ..tools.availability import is_tool_unavailable_result
from ..verification_command_analysis import (
    analyze_verification_command,
    is_benign_non_execution_reason,
)
from ..verify_gate import (
    ResolvedVerifyCommands,
    assess_verification_command_execution,
    extract_actionable_failure_snippet,
    extract_verification_failure_snippet,
    is_authoritative_verify_command_selection,
    is_toolchain_unavailable_verification_output,
    resolve_task_aware_verify_command_selection,
    verification_selection_payload,
)
from ..verify_gate import run_task_verification as run_task_verification
from .acceptance_contract import (
    AcceptanceContract,
    acceptance_contract_problem_payload,
    extract_explicit_acceptance_commands,
    record_acceptance_tool_effect,
)
from .completion_certificate import (
    CompletionCertificateInput,
    evaluate_completion_certificate,
)
from .completion_gate import CompletionGateControllerState
from .mutation_classification import classify_mutation_paths, material_mutation_paths
from .prompt_context import (
    _extract_workspace_relation_paths_from_text,
    _normalize_repo_relative_hint_path,
    _paths_require_verification,
    _session_repo_scan,
    _session_task_brief_content,
    _session_verify_command_selection,
    _task_brief_lines_from_text,
    _verification_commands_apply_to_paths,
    refresh_session_environment_context_message,
)
from .verification_commands import (
    _matching_effective_verification_commands,
    _normalize_shell_command_for_match,
)
from .verification_evidence import (
    VerificationEvidence,
    VerificationEvidenceCategory,
    classify_verification_evidence,
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
    "shell_service_start",
    "workspace_preview_start",
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
_TEST_EXECUTION_COMMAND_RE = re.compile(
    r"(?:^|\s)(?:pytest|py\.test|tox|nox)(?:\s|$)|"
    r"\b(?:python(?:3)?|py)\s+-m\s+(?:pytest|unittest)\b|"
    r"\b(?:python(?:3)?|py)\b[^\n]*\bmanage\.py\s+test\b|"
    r"\b(?:python(?:3)?|py)\b[^\n]*\b(?:runtests?|test_[^\s/]+|[^\s/]+_test)\.py\b|"
    r"(?:^|\s)(?:\./)?(?:bin/)?(?:runtests?|test)(?:\s|$)|"
    r"\b(?:go|cargo)\s+test\b|"
    r"\b(?:npm|pnpm|yarn)\s+test\b|"
    r"\b(?:vitest|jest|rspec|phpunit|ctest)\b|"
    r"\b(?:mvn|mvnw|maven|gradle|gradlew|dotnet|bazel|mix)\b[^\n]*\btest\b|"
    r"\b(?:make|just)\s+(?:test|check)\b",
    re.IGNORECASE,
)
_TEST_SUCCESS_CLAIM_RE = re.compile(
    r"\b(?:all\s+)?(?:\d+\s+)?tests?(?:\s+suite)?\s+"
    r"(?:(?:is|are|was|were)\s+)?"
    r"(?:pass(?:ed|es|ing)?|succeed(?:ed|s)?|green)\b|"
    r"\btests?\s*:\s*[^\n]{0,120}\b(?:pass(?:ed|es|ing)?|succeed(?:ed|s)?)\b|"
    r"\b(?:pass(?:ed|es|ing)?|green)\s+(?:all\s+)?tests?\b",
    re.IGNORECASE,
)
_GENERIC_VERIFICATION_SUCCESS_CLAIM_RE = re.compile(
    r"\bverified\b|"
    r"\bverification\s+(?:passed|succeeded|completed|was\s+successful)\b|"
    r"\b(?:validation|checks?)\s+(?:passed|succeeded)\b",
    re.IGNORECASE,
)
_NEGATED_CLAIM_PREFIX_RE = re.compile(
    r"(?:\b(?:not|never|no|without)\b[^.!?\n]{0,32}|"
    r"\b(?:cannot|can't|could\s+not|couldn't|did\s+not|didn't|wasn't|isn't|"
    r"unable\s+to|failed\s+to)(?:\s+be)?)\s*$",
    re.IGNORECASE,
)
_SAFE_LEADING_CD_RE = re.compile(
    r"^\s*cd(?:\s+/d)?\s+(?:\"[^\"]*\"|'[^']*'|[^\s]+)\s*&&\s*",
    re.IGNORECASE,
)
_UNSAFE_CLAIM_EVIDENCE_SHELL_RE = re.compile(
    r"\|\||(?<![&])\|(?![&])|;|[\r\n]|&&|(?:^|\s)&(?:\s|$)",
)
_SHELL_REDIRECTION_RE = re.compile(
    r"\s+(?:\d*>&\d+|\d*(?:>>?|<)\s*[^\s]+)(?=\s|$)",
)
SUPPLEMENTAL_VERIFICATION_ADVISORY = (
    "Note: every passing check so far was authored during this session. "
    "Self-written tests verify your interpretation, not the task's. Re-read the "
    "task's exact requirements (output path, format, names, values) and confirm "
    "your deliverable against the spec itself before finalizing."
)
_COMPLETION_GATE_PROBLEM_LABELS = {
    "empty_final_response": "empty final response",
    "clarification_requested": "clarification requested instead of action",
    "no_material_edits": "no material edits",
    "verification_not_attempted": "verification not attempted",
    "verification_incomplete": "verification coverage incomplete",
    "verification_failed": "verification failing",
    "acceptance_criteria_unverified": "acceptance criteria unverified",
    "acceptance_criteria_failed": "acceptance criteria failed",
    "acceptance_evidence_insufficient": "acceptance evidence insufficient",
    "unexpected_scope_changes": "unexpected scope changes",
}
_ONE_SHOT_COMPLETION_GATE_NUDGE_PREFIX = (
    "Completion gate: this one-shot execution run cannot finalize yet."
)
_RUNTIME_DEFAULT_LANGUAGE = "english"
_RUNTIME_MESSAGE_CATALOG: dict[str, dict[str, str]] = {
    "english": {
        "phase_understanding_request": "Understanding your request.",
        "phase_drafting_response": "Contacting model provider.",
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
            "Completion gate detected missing execution evidence; requesting action-oriented repair."
        ),
        "phase_step_budget_handoff": (
            "Step budget exhausted; preparing a concise handoff so the chat can continue."
        ),
        "phase_writing_final_response": "Writing the final response.",
        "one_shot_continuation_nudge": (
            "Continue execution now. A text-only plan or progress update is incomplete "
            "for this one-shot run. Use the next required tool action to implement or "
            "create the requested deliverable, run an implementation-producing command, "
            "verify only after material work exists or when the implementation already "
            "exists, or explain a concrete evidence-backed blocker."
        ),
        "interactive_continuation_nudge": (
            "Continue execution now. Do not stop at a planning/progress update. "
            "Use tools to make progress, run relevant verification, or explain a concrete blocker."
        ),
        "one_shot_exploration_nudge": (
            "Avoid repeated read-only exploration. Start implementing or creating the "
            "requested deliverable now, delegate once to a suitable available subagent "
            "if more investigation is genuinely needed, or explain a concrete "
            "evidence-backed blocker."
        ),
        "one_shot_post_explore_bootstrap_nudge": (
            "A subagent already returned useful context in this one-shot turn. You now have enough "
            "context to start implementation. Do not call the same research subagent again "
            "in this turn. Do not use more read-only tools unless there is a concrete blocker. "
            "Your next step must be an implementation or deliverable-creation action "
            "(for example fs_edit, fs_write, git_apply_patch, fs_move, fs_copy, or "
            "shell_run only when it actually performs implementation or creates the "
            "requested deliverable) or a concrete evidence-backed blocker report. "
            "Verification comes after material work exists."
        ),
        "one_shot_post_explore_bootstrap_targets": ("Likely repo-root-relative targets: {joined}."),
        "one_shot_edit_strategy_nudge": (
            "Edit strategy is stuck. Switch approach now: re-read the target lines, then use "
            "fs_edit replace_lines/insert_before_line/insert_after_line with expected_old when "
            "possible, or exact ops replace_exact/insert_before_exact/insert_after_exact when "
            "matching known text. If localized fs_edit is a poor fit, use git_apply_patch or "
            "fs_write. Do not repeat the same failing edit call."
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
            "implementation-bootstrap nudges. Start implementing or creating the requested "
            "deliverable now or report a concrete blocker."
        ),
        "one_shot_exploration_retry_exhausted": (
            "One-shot run stopped: exploration stagnation persisted after bounded nudges. "
            "Start implementing or creating the requested deliverable, delegate once to a "
            "suitable available subagent if more "
            "investigation is genuinely needed, or report a concrete blocker."
        ),
        "one_shot_edit_retry_exhausted": (
            "One-shot run stopped: failed edit/write loop persisted after bounded strategy "
            "nudges. Switch to exact-match fs_edit ops, or use git_apply_patch/fs_write, "
            "or report a concrete blocker."
        ),
        "one_shot_post_explore_step_budget_exhausted": (
            "One-shot run stopped: post-explore stagnation consumed the step budget. "
            "Start implementing or creating the requested deliverable now or report a concrete blocker."
        ),
        "one_shot_exploration_step_budget_exhausted": (
            "One-shot run stopped: exploration stagnation consumed the step budget. "
            "Start implementing or creating the requested deliverable, delegate once to a "
            "suitable available subagent if more "
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
    material_edit_generation: int = 0
    material_edit_tools: set[str] = field(default_factory=set)
    touched_repo_paths: set[str] = field(default_factory=set)
    last_diff_review_generation: int | None = None
    verification_attempt_count: int = 0
    verification_tools: set[str] = field(default_factory=set)
    last_verification_passed: bool | None = None
    last_verification_failure_snippet: str = ""
    last_verification_failure_category: str = ""
    failed_verification_command_snippets: dict[str, str] = field(default_factory=dict)
    verification_relevant_edit_generation: int = 0
    last_successful_verification_generation: int | None = None
    verification_evidence_counts: dict[str, int] = field(default_factory=dict)
    latest_verification_evidence_category: str = ""
    latest_verification_evidence_reason: str = ""
    accepted_verification_evidence: list[dict[str, Any]] = field(default_factory=list)
    supplemental_verification_evidence: list[dict[str, Any]] = field(default_factory=list)
    rejected_verification_evidence: list[dict[str, Any]] = field(default_factory=list)
    executed_verification_evidence: list[dict[str, Any]] = field(default_factory=list)
    verification_evidence_generation: int = 0
    completion_gate_repair_attempts: int = 0
    completion_gate_no_material_edits_repair_attempts: int = 0
    completion_gate_missing_verify_repair_attempts: int = 0
    completion_gate_failed_verify_repair_attempts: int = 0
    completion_gate_controller_state: CompletionGateControllerState = field(
        default_factory=CompletionGateControllerState
    )
    acceptance_contract: AcceptanceContract | None = None
    latest_completion_certificate: dict[str, Any] = field(default_factory=dict)

    def refresh_verification_coverage(self) -> None:
        self.covered_verification_commands = {
            command
            for command, generation in self.covered_verification_command_generations.items()
            if generation == self.verification_relevant_edit_generation
        }

    def note_verification_relevant_edit(self) -> None:
        self.verification_relevant_edit_generation += 1
        self.refresh_verification_coverage()

    def note_material_edit(self) -> None:
        self.material_edit_count += 1
        self.material_edit_generation += 1

    def record_diff_review(self) -> None:
        self.last_diff_review_generation = self.material_edit_generation

    def diff_review_is_stale(self) -> bool:
        return self.material_edit_count > 0 and (
            self.last_diff_review_generation is None
            or self.last_diff_review_generation < self.material_edit_generation
        )

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

    def record_verification_evidence(
        self,
        evidence: VerificationEvidence,
        *,
        accepted: bool,
        observed_exit_code: int | None = None,
        observed_output: bool = False,
    ) -> None:
        category = evidence.category.value
        self.verification_evidence_counts[category] = (
            self.verification_evidence_counts.get(category, 0) + 1
        )
        self.latest_verification_evidence_category = category
        self.latest_verification_evidence_reason = evidence.reason
        payload = evidence.as_payload()
        payload["accepted"] = bool(accepted)
        payload["generation"] = self.verification_relevant_edit_generation
        payload["observed_exit_code"] = observed_exit_code
        payload["observed_output"] = bool(observed_output)
        if evidence.real_execution is True:
            self.executed_verification_evidence.append(payload)
            self.executed_verification_evidence[:] = self.executed_verification_evidence[-20:]
        if accepted:
            self.verification_evidence_generation += 1
            self.accepted_verification_evidence.append(payload)
            self.accepted_verification_evidence[:] = self.accepted_verification_evidence[-10:]
        elif evidence.supplemental_only:
            self.supplemental_verification_evidence.append(payload)
            self.supplemental_verification_evidence[:] = self.supplemental_verification_evidence[
                -10:
            ]
        else:
            self.rejected_verification_evidence.append(payload)
            self.rejected_verification_evidence[:] = self.rejected_verification_evidence[-10:]

    def record_executed_command_evidence(
        self,
        *,
        normalized_command: str,
        observed_exit_code: int,
        observed_output: bool,
    ) -> None:
        payload: dict[str, Any] = {
            "evidence_category": "COMMAND_EXECUTION",
            "normalized_command": normalized_command,
            "matched_command": None,
            "real_execution": True,
            "allowed_to_satisfy_contract": False,
            "reason": "observed_shell_verification_execution",
            "covered_verification_commands": [],
            "supplemental_only": False,
            "accepted": False,
            "generation": self.verification_relevant_edit_generation,
            "observed_exit_code": observed_exit_code,
            "observed_output": bool(observed_output),
        }
        self.executed_verification_evidence.append(payload)
        self.executed_verification_evidence[:] = self.executed_verification_evidence[-20:]

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
            "material_edit_generation": self.material_edit_generation,
            "material_edit_tools": sorted(self.material_edit_tools),
            "touched_repo_paths": sorted(self.touched_repo_paths),
            "last_diff_review_generation": self.last_diff_review_generation,
            "diff_review_stale": self.diff_review_is_stale(),
            "verification_attempt_count": self.verification_attempt_count,
            "verification_tools": sorted(self.verification_tools),
            "last_verification_passed": self.last_verification_passed,
            "last_verification_failure_snippet": self.last_verification_failure_snippet,
            "last_verification_failure_category": self.last_verification_failure_category,
            "failed_verification_commands": sorted(self.failed_verification_commands()),
            "verification_relevant_edit_generation": self.verification_relevant_edit_generation,
            "last_successful_verification_generation": self.last_successful_verification_generation,
            "verification_coverage_stale": self.verification_coverage_is_stale(),
            "verification_evidence_counts": dict(sorted(self.verification_evidence_counts.items())),
            "latest_verification_evidence_category": (self.latest_verification_evidence_category),
            "latest_verification_evidence_reason": self.latest_verification_evidence_reason,
            "accepted_verification_evidence": list(self.accepted_verification_evidence),
            "supplemental_verification_evidence": list(self.supplemental_verification_evidence),
            "rejected_verification_evidence": list(self.rejected_verification_evidence),
            "executed_verification_evidence": list(self.executed_verification_evidence),
            "verification_evidence_generation": self.verification_evidence_generation,
            "completion_gate_repair_attempts": self.completion_gate_repair_attempts,
            "completion_gate_no_material_edits_repair_attempts": self.completion_gate_no_material_edits_repair_attempts,
            "completion_gate_missing_verify_repair_attempts": self.completion_gate_missing_verify_repair_attempts,
            "completion_gate_failed_verify_repair_attempts": self.completion_gate_failed_verify_repair_attempts,
            "completion_gate_controller": self.completion_gate_controller_state.as_payload(),
            "completion_gate_last_decision_kind": self.completion_gate_controller_state.last_decision_kind,
            "completion_certificate": dict(self.latest_completion_certificate),
            **acceptance_contract_problem_payload(self.acceptance_contract),
        }

    def acceptance_problem_names(self) -> list[str]:
        if self.acceptance_contract is None:
            return []
        return self.acceptance_contract.problem_names()

    def acceptance_requires_execution(self) -> bool:
        if self.acceptance_contract is None:
            return False
        return any(
            criterion.required_for_finalization
            and criterion.required
            and (
                criterion.commands
                or criterion.thresholds
                or criterion.ports
                or criterion.kind.value
                in {
                    "explicit_command_io",
                    "functional_api_protocol",
                    "persistent_service",
                    "explicit_host_user_verification_command",
                }
            )
            for criterion in self.acceptance_contract.criteria
        )


def _successful_verification_claim_kind(final_text: str) -> str | None:
    text = str(final_text or "")
    for kind, pattern in (
        ("tests", _TEST_SUCCESS_CLAIM_RE),
        ("verification", _GENERIC_VERIFICATION_SUCCESS_CLAIM_RE),
    ):
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start() - 48) : match.start()]
            if _NEGATED_CLAIM_PREFIX_RE.search(prefix):
                continue
            return kind
    return None


def _fresh_executed_evidence_for_claim(
    state: TurnExecutionState,
    *,
    claim_kind: str,
) -> list[dict[str, Any]]:
    required_generation = state.verification_relevant_edit_generation
    evidence: list[dict[str, Any]] = []
    for raw_item in state.executed_verification_evidence:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        try:
            generation = int(item.get("generation"))
        except (TypeError, ValueError):
            continue
        if generation < required_generation:
            continue
        if item.get("real_execution") is not True:
            continue
        if item.get("reason") == "mutated_material_paths":
            continue
        if item.get("observed_exit_code") != 0 or item.get("observed_output") is not True:
            continue
        command = str(item.get("normalized_command") or "").strip()
        if not command:
            continue
        if claim_kind == "tests":
            analysis = analyze_verification_command(command, trusted=True)
            family = str(analysis.command_family or "").casefold()
            if "test" not in family and _TEST_EXECUTION_COMMAND_RE.search(command) is None:
                continue
        evidence.append(item)
    return evidence


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
    if normalized_tool in _COMMAND_LIKE_MUTATION_TOOL_NAMES:
        return set(material_mutation_paths(touched, root=root))
    return touched


def _verification_attempt_passed(
    *,
    tool_name: str,
    status: str,
    result: dict[str, Any],
    evidence: VerificationEvidence | None = None,
) -> bool:
    if status == "failed":
        return False
    if evidence is not None and not evidence.allowed_to_satisfy_contract:
        return False
    normalized_tool = tool_name.strip().lower()
    touched_repo_paths = result.get("material_touched_repo_paths", result.get("touched_repo_paths"))
    normalized_touched = (
        {str(item) for item in touched_repo_paths if isinstance(item, str) and str(item).strip()}
        if isinstance(touched_repo_paths, list)
        else set()
    )
    if normalized_tool == "verify_run":
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
                if real_execution is not True:
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
        return assessment.real_execution is True
    return False


def _verification_relevant_material_paths(paths: set[str]) -> set[str]:
    if not paths or not _paths_require_verification(paths):
        return set()
    return set(paths)


def _verification_command_result_is_benign_skip(item: dict[str, Any]) -> bool:
    return (
        item.get("status") == "skipped"
        and item.get("ok") is True
        and is_benign_non_execution_reason(str(item.get("non_execution_reason") or ""))
    )


def _verification_command_result_passed(item: dict[str, Any]) -> bool:
    if _verification_command_result_is_benign_skip(item):
        return True
    real_execution = item.get("real_execution")
    if real_execution is not True:
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
    evidence: VerificationEvidence | None = None,
) -> None:
    matches = (
        set(evidence.covered_verification_commands)
        if evidence is not None and evidence.allowed_to_satisfy_contract
        else set()
    )
    if not matches:
        matches = _matching_effective_verification_commands(
            observed_command=str(result.get("effective_cmd") or arguments.get("cmd") or ""),
            effective_verification_commands=known_verification_commands,
        )
    if not matches:
        return
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


def _verification_output_text(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(result.get("stdout") or "").strip(),
            str(result.get("stderr") or "").strip(),
            str(result.get("output") or "").strip(),
            str(result.get("output_preview") or "").strip(),
        ]
    ).strip()


def _aggregate_verification_evidence(
    records: list[VerificationEvidence],
    *,
    fallback_command: str = "",
) -> VerificationEvidence:
    if not records:
        return VerificationEvidence(
            category=VerificationEvidenceCategory.NOT_VERIFICATION,
            normalized_command=fallback_command,
            reason="no_verification_evidence",
        )
    priority = {
        VerificationEvidenceCategory.AUTHORITATIVE: 0,
        VerificationEvidenceCategory.REPO_NATIVE: 1,
        VerificationEvidenceCategory.TASK_ACCEPTANCE: 2,
        VerificationEvidenceCategory.NOT_VERIFICATION: 3,
    }
    primary = sorted(records, key=lambda item: priority[item.category])[0]
    covered = sorted(
        {command for item in records for command in item.covered_verification_commands if command}
    )
    allowed = bool(records) and all(
        item.allowed_to_satisfy_contract
        for item in records
        if item.category != VerificationEvidenceCategory.NOT_VERIFICATION
    )
    if any(item.category == VerificationEvidenceCategory.NOT_VERIFICATION for item in records):
        allowed = False
    return VerificationEvidence(
        category=primary.category,
        normalized_command=primary.normalized_command,
        matched_command=primary.matched_command,
        real_execution=primary.real_execution,
        allowed_to_satisfy_contract=allowed,
        reason=primary.reason
        if allowed
        else next(
            (item.reason for item in records if not item.allowed_to_satisfy_contract),
            primary.reason,
        ),
        covered_verification_commands=tuple(covered),
        supplemental_only=all(item.supplemental_only for item in records),
    )


def _verification_evidence_note(
    evidence: VerificationEvidence,
    *,
    result: dict[str, Any] | None = None,
) -> str:
    if evidence.category == VerificationEvidenceCategory.NOT_VERIFICATION:
        return ""
    if evidence.supplemental_only:
        return (
            "evidence origin: SELF_AUTHORED "
            "(supplemental - cannot independently confirm spec compliance)"
        )
    result_payload = result if isinstance(result, dict) else {}
    command_specs = result_payload.get("verification_command_specs")
    if isinstance(command_specs, list) and any(
        isinstance(item, dict) and item.get("provenance") == "PREEXISTING_REPO_NATIVE"
        for item in command_specs
    ):
        return "evidence origin: PREEXISTING_REPO_NATIVE (independent)"
    if result_payload.get("verification_contract_type") == "repo_native":
        return "evidence origin: PREEXISTING_REPO_NATIVE (independent)"
    if evidence.category == VerificationEvidenceCategory.AUTHORITATIVE:
        return "evidence origin: USER_EXPLICIT (independent)"
    if evidence.category == VerificationEvidenceCategory.REPO_NATIVE:
        return "evidence origin: PREEXISTING_REPO_NATIVE (independent)"
    if evidence.category == VerificationEvidenceCategory.TASK_ACCEPTANCE:
        return "evidence origin: DIRECT_BLACK_BOX (independent)"
    return ""


def _verify_run_evidence_records(
    *,
    result: dict[str, Any],
    known_verification_commands: list[str] | None,
    verification_authoritative: bool,
    material_touched_paths: set[str],
    root: Path,
) -> list[VerificationEvidence]:
    command_results = result.get("command_results")
    verification_relevant_touched_paths = _verification_relevant_material_paths(
        material_touched_paths
    )
    records: list[VerificationEvidence] = []
    if isinstance(command_results, list):
        for item in command_results:
            if not isinstance(item, dict):
                continue
            command = str(item.get("command") or item.get("effective_command") or "")
            if not command:
                continue
            exit_code_raw = item.get("exit_code")
            exit_code = exit_code_raw if isinstance(exit_code_raw, int) else None
            record = classify_verification_evidence(
                command,
                known_verification_commands=known_verification_commands,
                authoritative=verification_authoritative,
                material_touched_paths=verification_relevant_touched_paths,
                exit_code=exit_code,
                output=_verification_output_text(item),
                real_execution=(
                    item.get("real_execution")
                    if isinstance(item.get("real_execution"), bool)
                    or item.get("real_execution") is None
                    else None
                ),
                root=root,
            )
            if (
                _verification_command_result_is_benign_skip(item)
                and record.category != VerificationEvidenceCategory.NOT_VERIFICATION
                and record.covered_verification_commands
            ):
                record = replace(
                    record,
                    allowed_to_satisfy_contract=True,
                    reason=str(item.get("non_execution_reason") or "verification_skipped"),
                )
            records.append(record)
        return records

    commands = result.get("commands")
    if isinstance(commands, list):
        all_passed = result.get("all_passed")
        exit_code = 0 if all_passed is True else 1 if all_passed is False else None
        for command in commands:
            records.append(
                classify_verification_evidence(
                    str(command),
                    known_verification_commands=known_verification_commands,
                    authoritative=verification_authoritative,
                    material_touched_paths=verification_relevant_touched_paths,
                    exit_code=exit_code,
                    output=_verification_output_text(result),
                    root=root,
                )
            )
    return records


def _shell_verification_evidence(
    *,
    root: Path,
    state: TurnExecutionState,
    arguments: dict[str, Any],
    result: dict[str, Any],
    known_verification_commands: list[str] | None,
    verification_authoritative: bool,
    material_touched_paths: set[str],
) -> VerificationEvidence:
    exit_code_raw = result.get("exit_code")
    exit_code = exit_code_raw if isinstance(exit_code_raw, int) else None
    command = str(result.get("effective_cmd") or result.get("cmd") or arguments.get("cmd") or "")
    return classify_verification_evidence(
        command,
        known_verification_commands=known_verification_commands,
        authoritative=verification_authoritative,
        changed_paths=state.touched_repo_paths,
        material_touched_paths=_verification_relevant_material_paths(material_touched_paths),
        exit_code=exit_code,
        output=_verification_output_text(result),
        root=root,
    )


def _verification_evidence_observation(
    *,
    tool_name: str,
    evidence: VerificationEvidence,
    result: dict[str, Any],
) -> tuple[int | None, bool]:
    def _has_output(payload: dict[str, Any]) -> bool:
        if any(
            str(payload.get(key) or "").strip()
            for key in ("output", "output_preview", "stdout", "stderr")
        ):
            return True
        output_chars = payload.get("output_chars")
        return isinstance(output_chars, int) and output_chars > 0

    normalized_tool = tool_name.strip().casefold()
    if normalized_tool == "shell_run":
        exit_code = result.get("exit_code")
        return (
            exit_code if isinstance(exit_code, int) else None,
            _has_output(result),
        )

    if normalized_tool == "verify_run":
        command_results = result.get("command_results")
        if isinstance(command_results, list):
            evidence_command = _normalize_shell_command_for_match(evidence.normalized_command)
            for raw_item in command_results:
                if not isinstance(raw_item, dict):
                    continue
                command = str(raw_item.get("command") or raw_item.get("effective_command") or "")
                effective_command = str(
                    raw_item.get("effective_command") or raw_item.get("command") or ""
                )
                normalized_candidates = {
                    _normalize_shell_command_for_match(command),
                    _normalize_shell_command_for_match(effective_command),
                }
                if evidence_command not in normalized_candidates:
                    continue
                exit_code = raw_item.get("exit_code")
                return (
                    exit_code if isinstance(exit_code, int) else None,
                    _has_output(raw_item),
                )
        all_passed = result.get("all_passed")
        exit_code = 0 if all_passed is True else 1 if all_passed is False else None
        return (
            exit_code,
            _has_output(result),
        )

    return None, False


def _unmasked_shell_verification_command(command: str) -> str:
    candidate = str(command or "").strip()
    while match := _SAFE_LEADING_CD_RE.match(candidate):
        candidate = candidate[match.end() :].strip()
    if not candidate or _UNSAFE_CLAIM_EVIDENCE_SHELL_RE.search(candidate):
        return ""
    analysis_candidate = _SHELL_REDIRECTION_RE.sub("", candidate).strip()
    analysis = analyze_verification_command(analysis_candidate, trusted=True)
    if analysis.command_family is None:
        return ""
    return _normalize_shell_command_for_match(analysis_candidate)


def _record_tool_effect(
    *,
    root: Path,
    state: TurnExecutionState,
    tool_name: str,
    arguments: dict[str, Any],
    status: str,
    result: dict[str, Any],
    known_verification_commands: list[str] | None,
    verification_authoritative: bool = False,
) -> None:
    if is_tool_unavailable_result(result):
        return
    normalized_tool = tool_name.strip().lower()
    touched_paths: set[str] = set()
    benign_runtime_paths: set[str] = set()
    if normalized_tool in _COMMAND_LIKE_MUTATION_TOOL_NAMES:
        raw_touched_paths = result.get("touched_repo_paths")
        if isinstance(raw_touched_paths, list):
            classifications = classify_mutation_paths(
                [str(item) for item in raw_touched_paths if isinstance(item, str)],
                root=root,
                command_was_verification=normalized_tool == "verify_run",
            )
            touched_paths = {item.path for item in classifications if item.is_material}
            benign_runtime_paths = {item.path for item in classifications if not item.is_material}
            if touched_paths:
                result["material_touched_repo_paths"] = sorted(touched_paths)
            if benign_runtime_paths:
                result["benign_runtime_paths"] = sorted(benign_runtime_paths)
        else:
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
        state.note_material_edit()
        state.material_edit_tools.add(normalized_tool)
        state.touched_repo_paths.update(touched_paths)
        if _paths_require_verification(touched_paths):
            state.note_verification_relevant_edit()
    elif normalized_tool in _COMMAND_LIKE_MUTATION_TOOL_NAMES and touched_paths:
        state.note_material_edit()
        state.material_edit_tools.add(normalized_tool)
        state.touched_repo_paths.update(touched_paths)
        if _paths_require_verification(touched_paths):
            state.note_verification_relevant_edit()
    elif status != "failed" and normalized_tool == "git_diff":
        state.record_diff_review()

    verification_attempt = False
    evidence_records: list[VerificationEvidence] = []
    evidence = VerificationEvidence(
        category=VerificationEvidenceCategory.NOT_VERIFICATION,
        normalized_command=str(arguments.get("cmd") or ""),
        reason="not_checked",
    )
    if normalized_tool == "verify_run":
        verification_attempt = True
        evidence_records = _verify_run_evidence_records(
            result=result,
            known_verification_commands=known_verification_commands,
            verification_authoritative=verification_authoritative,
            material_touched_paths=touched_paths,
            root=root,
        )
        evidence = _aggregate_verification_evidence(evidence_records)
    elif normalized_tool == "shell_run":
        evidence = _shell_verification_evidence(
            root=root,
            state=state,
            arguments=arguments,
            result=result,
            known_verification_commands=known_verification_commands,
            verification_authoritative=verification_authoritative,
            material_touched_paths=touched_paths,
        )
        evidence_records = [evidence]
        verification_attempt = evidence.category != VerificationEvidenceCategory.NOT_VERIFICATION
    if normalized_tool in _COMMAND_LIKE_MUTATION_TOOL_NAMES:
        result["verification_evidence_category"] = evidence.category.value
        result["verification_evidence_reason"] = evidence.reason
        result["verification_evidence_allowed"] = evidence.allowed_to_satisfy_contract
        result["verification_evidence_supplemental_only"] = evidence.supplemental_only
        verification_note = _verification_evidence_note(evidence, result=result)
        if verification_note:
            result["verification_note"] = verification_note
        if evidence.supplemental_only:
            result["verification_supplemental_only_note"] = SUPPLEMENTAL_VERIFICATION_ADVISORY
    record_acceptance_tool_effect(
        contract=state.acceptance_contract,
        root=root,
        tool_name=normalized_tool,
        arguments=arguments,
        status=status,
        result=result,
        touched_paths=touched_paths,
        known_verification_commands=known_verification_commands,
        verification_authoritative=verification_authoritative,
        evidence_category=evidence.category.value,
        evidence_allowed=evidence.allowed_to_satisfy_contract,
    )
    if normalized_tool == "shell_run" and not verification_attempt:
        raw_command = str(
            result.get("effective_cmd") or result.get("cmd") or arguments.get("cmd") or ""
        )
        normalized_command = _unmasked_shell_verification_command(raw_command)
        observed_exit_code, observed_output = _verification_evidence_observation(
            tool_name=normalized_tool,
            evidence=evidence,
            result=result,
        )
        if normalized_command and observed_exit_code == 0 and observed_output:
            state.record_executed_command_evidence(
                normalized_command=normalized_command,
                observed_exit_code=observed_exit_code,
                observed_output=observed_output,
            )
    if not verification_attempt:
        return

    state.verification_attempt_count += 1
    state.verification_tools.add(normalized_tool)
    state.last_verification_passed = _verification_attempt_passed(
        tool_name=normalized_tool,
        status=status,
        result=result,
        evidence=evidence,
    )
    for record in evidence_records:
        observed_exit_code, observed_output = _verification_evidence_observation(
            tool_name=normalized_tool,
            evidence=record,
            result=result,
        )
        state.record_verification_evidence(
            record,
            accepted=(
                state.last_verification_passed is True and record.allowed_to_satisfy_contract
            ),
            observed_exit_code=observed_exit_code,
            observed_output=observed_output,
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
            evidence=evidence,
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
    verification_contract_available: bool = True,
    effective_verification_commands: list[str] | tuple[str, ...] | set[str] | None = None,
) -> bool:
    if turn_intent != "execute":
        return False
    if verification_contract_requires_execution:
        return True
    if blocked:
        return False
    if not verification_contract_available:
        return False
    return _verification_commands_apply_to_paths(
        touched_repo_paths,
        effective_verification_commands,
    )


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
    certificate = evaluate_completion_certificate(
        CompletionCertificateInput(
            contract=state.acceptance_contract,
            final_text=final_text,
            blocked=blocked,
            blocker_valid=blocked,
            material_edit_count=state.material_edit_count,
            require_material_result=require_material_edit_evidence,
            verification_expected=verification_expected,
            verification_attempt_count=state.verification_attempt_count,
            last_verification_passed=state.last_verification_passed,
            failed_verification_commands=state.failed_verification_commands(),
            expected_verification_commands=set(state.expected_verification_commands),
            missing_verification_commands=state.missing_verification_commands(),
            verification_coverage_stale=state.verification_coverage_is_stale(),
            accepted_verification_evidence=list(state.accepted_verification_evidence),
        )
    )
    state.latest_completion_certificate = certificate.as_payload()
    return list(certificate.problems)


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
    if "acceptance_criteria_failed" in problems or "unexpected_scope_changes" in problems:
        return "acceptance_failed"
    if (
        "acceptance_criteria_unverified" in problems
        or "acceptance_evidence_insufficient" in problems
    ):
        return "acceptance_unverified"
    if "clarification_requested" in problems:
        return "clarification_requested"
    if "empty_final_response" in problems:
        return "empty_final_response"
    return "generic"


_LIVE_BACKGROUND_PROCESS_FINALIZATION_LINE = (
    "- You have {n} background process(es) started with shell_background; they are "
    "terminated when this run ends. If the task requires a server/daemon to still "
    "be running after you finish, start it with shell_service_start (durable) instead, "
    "and re-verify."
)


def _live_background_process_finalization_advisory_line(
    *,
    one_shot_execution: bool,
    live_background_processes: int = 0,
) -> str:
    try:
        count = int(live_background_processes)
    except (TypeError, ValueError):
        count = 0
    if not one_shot_execution or count <= 0:
        return ""
    return _LIVE_BACKGROUND_PROCESS_FINALIZATION_LINE.format(n=count)


def _completion_gate_nudge_message(
    problems: list[str],
    *,
    prefix_key: str = "completion_gate_nudge_prefix",
    verification_failure_snippet: str = "",
    missing_verification_commands: list[str] | None = None,
    verification_coverage_stale: bool = False,
    anchor_paths: list[str] | None = None,
    has_material_edits: bool = False,
    all_verification_evidence_self_authored: bool = False,
    diff_review_stale: bool = False,
    language: str = "",
    explicit_language_override: bool = False,
    one_shot_execution: bool = False,
    live_background_processes: int = 0,
) -> str:
    _ = (
        prefix_key,
        verification_coverage_stale,
        anchor_paths,
        language,
        explicit_language_override,
    )
    problem_set = set(problems)
    lines = ["Finalization check - one pass before you finish:"]
    if "no_material_edits" in problem_set:
        lines.append(
            "- No file changes are recorded yet. If the task required creating/modifying "
            "something, do it now; if you concluded no change is needed, say so explicitly "
            "with your reasoning."
        )
    snippet = extract_actionable_failure_snippet(verification_failure_snippet)
    if "verification_failed" in problem_set:
        failure_detail = snippet or "the latest verification attempt did not pass"
        lines.append(
            f"- Your last verification failed: {failure_detail}. Fix and re-run, or explain "
            "why the failure is expected/out of scope."
        )
    if missing_verification_commands and (
        "verification_not_attempted" in problem_set or "verification_incomplete" in problem_set
    ):
        lines.append(
            "- Expected verification not yet run: "
            + ", ".join(missing_verification_commands)
            + ". Run them, or state why they don't apply."
        )
    elif "verification_not_attempted" in problem_set or "verification_incomplete" in problem_set:
        lines.append(
            "- Expected verification has not been completed. Run it, or state why it does not apply."
        )
    if all_verification_evidence_self_authored:
        lines.append(f"- {SUPPLEMENTAL_VERIFICATION_ADVISORY}")
    if has_material_edits and diff_review_stale:
        lines.append(
            "- Consider reviewing the current diff for accidental scope or quality issues before "
            "finalizing."
        )
    live_background_process_line = _live_background_process_finalization_advisory_line(
        one_shot_execution=one_shot_execution,
        live_background_processes=live_background_processes,
    )
    if live_background_process_line:
        lines.append(live_background_process_line)
    lines.append(
        "- Re-read the task statement once and confirm every explicitly named output "
        "(paths, formats, values) exists exactly as requested."
    )
    lines.append(
        "Then give your final answer. If you are confident the work is complete as-is, "
        "finalize - this checklist is advisory."
    )
    return "\n".join(lines)


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


def _refresh_execute_turn_verification_selection(
    session: Any,
    *,
    instruction: str,
    route_execution_posture: str,
) -> None:
    if not bool(getattr(session, "verification_enabled", True)):
        return
    runtime_kind = getattr(session, "runtime_kind", RuntimeKind.INTERACTIVE_CHAT)
    one_shot_execution = bool(getattr(session, "one_shot_execution", False))
    if runtime_kind != RuntimeKind.INTERACTIVE_CHAT and not one_shot_execution:
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
    explicit_commands = extract_explicit_acceptance_commands(
        instruction,
        *[str(item) for item in plan_requirements],
    )
    if (
        explicit_commands
        and not is_authoritative_verify_command_selection(current)
        and resolved.contract_type in {"generic_fallback", "unavailable", ""}
    ):
        resolved = ResolvedVerifyCommands(
            commands=tuple(explicit_commands),
            source="task_refinement.explicit_user_command",
            reason="explicit user command is the task-native verification contract",
            contract_type="task_acceptance",
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


def _refresh_interactive_turn_verification_selection(
    session: Any,
    *,
    instruction: str,
    route_execution_posture: str,
) -> None:
    _refresh_execute_turn_verification_selection(
        session,
        instruction=instruction,
        route_execution_posture=route_execution_posture,
    )
