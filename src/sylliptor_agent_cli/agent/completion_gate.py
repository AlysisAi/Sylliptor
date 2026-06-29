from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class CompletionGateDecisionKind(StrEnum):
    ALLOW_FINAL = "ALLOW_FINAL"
    NUDGE_AND_CONTINUE = "NUDGE_AND_CONTINUE"
    TERMINATE_STAGNANT = "TERMINATE_STAGNANT"
    TERMINATE_BUDGET_EXHAUSTED = "TERMINATE_BUDGET_EXHAUSTED"


DEFAULT_COMPLETION_GATE_STAGNANT_NUDGE_LIMIT = 2
DEFAULT_COMPLETION_GATE_CONSECUTIVE_NO_PROGRESS_LIMIT = 5
DEFAULT_COMPLETION_GATE_STAGE_STAGNANT_NUDGE_LIMITS: dict[str, int] = {
    "generic": 2,
    "empty_final_response": 2,
    "clarification_requested": 2,
    "no_material_edits": 2,
    "verification_not_attempted": 2,
    "verification_incomplete": 2,
    "verification_failed": 2,
    "acceptance_failed": 2,
    "acceptance_unverified": 2,
    "non_final_progress": 2,
}
NON_FINAL_PROGRESS_STAGE = "non_final_progress"
NON_FINAL_PROGRESS_PROBLEM = "non_final_progress"


_ABS_TEMP_PATH_RE = re.compile(
    r"(?:/private)?/(?:tmp|var/folders)/[^\s:;,\]\)]+|"
    r"[A-Za-z]:\\[^\s:;,\]\)]*\\(?:Temp|tmp)\\[^\s:;,\]\)]+"
)
_ISO_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ][0-2]\d:[0-5]\d:[0-5]\d(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_CLOCK_TIME_RE = re.compile(r"\b[0-2]?\d:[0-5]\d(?::[0-5]\d(?:\.\d+)?)?\b")
_DURATION_RE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:ms|s|sec|secs|seconds|msec)\b", re.I)
_UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
    re.I,
)
_HEX_ADDRESS_RE = re.compile(r"\b0x[0-9a-f]{6,}\b", re.I)
_LONG_NUMBER_RE = re.compile(r"\b\d{5,}\b")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_command(command: str) -> str:
    return " ".join(str(command or "").strip().split())


def normalize_completion_gate_failure_signature(text: str) -> str:
    clean = str(text or "").strip()
    if not clean:
        return ""
    clean = _ABS_TEMP_PATH_RE.sub("<temp-path>", clean)
    clean = _ISO_TIMESTAMP_RE.sub("<timestamp>", clean)
    clean = _CLOCK_TIME_RE.sub("<time>", clean)
    clean = _DURATION_RE.sub("<duration>", clean)
    clean = _UUID_RE.sub("<uuid>", clean)
    clean = _HEX_ADDRESS_RE.sub("<hex>", clean)
    clean = _LONG_NUMBER_RE.sub("<number>", clean)
    clean = _WHITESPACE_RE.sub(" ", clean)
    return clean.strip()


@dataclass(frozen=True)
class CompletionGateRepairPolicy:
    default_stagnant_nudge_limit: int = DEFAULT_COMPLETION_GATE_STAGNANT_NUDGE_LIMIT
    max_consecutive_no_progress_rejections: int = (
        DEFAULT_COMPLETION_GATE_CONSECUTIVE_NO_PROGRESS_LIMIT
    )
    stage_stagnant_nudge_limits: Mapping[str, int] = field(
        default_factory=lambda: dict(DEFAULT_COMPLETION_GATE_STAGE_STAGNANT_NUDGE_LIMITS)
    )

    def stagnant_nudge_limit(self, stage: str) -> int:
        raw_limit = self.stage_stagnant_nudge_limits.get(
            str(stage or "generic"),
            self.default_stagnant_nudge_limit,
        )
        return max(1, int(raw_limit))


DEFAULT_COMPLETION_GATE_REPAIR_POLICY = CompletionGateRepairPolicy()


@dataclass(frozen=True)
class CompletionGateEvidenceSnapshot:
    stage: str
    problems: tuple[str, ...] = tuple()
    material_edit_count: int = 0
    material_edit_tools: tuple[str, ...] = tuple()
    touched_repo_paths: tuple[str, ...] = tuple()
    verification_relevant_edit_generation: int = 0
    last_successful_verification_generation: int | None = None
    expected_verification_commands: tuple[str, ...] = tuple()
    covered_verification_commands: tuple[str, ...] = tuple()
    missing_verification_commands: tuple[str, ...] = tuple()
    failed_verification_signatures: tuple[str, ...] = tuple()
    verification_coverage_stale: bool = False
    last_verification_passed: bool | None = None
    last_verification_failure_category: str = ""
    accepted_blocker: bool = False
    blocked_response: bool = False
    blocked_response_allows_completion: bool = False
    verification_expected: bool = False
    final_text_present: bool = False
    repo_tool_activity_observed: bool = False
    acceptance_status_counts: dict[str, int] = field(default_factory=dict)
    acceptance_problems: tuple[str, ...] = tuple()
    acceptance_failure_signatures: tuple[str, ...] = tuple()

    def episode_payload(self) -> dict[str, Any]:
        verification_payload: dict[str, Any]
        if self.stage == "no_material_edits":
            verification_payload = {
                "verification_relevant_edit_generation": 0,
                "last_successful_verification_generation": None,
                "expected_verification_commands": [],
                "covered_verification_commands": [],
                "missing_verification_commands": [],
                "failed_verification_signatures": [],
                "verification_coverage_stale": False,
                "last_verification_passed": None,
                "last_verification_failure_category": "",
            }
        else:
            verification_payload = {
                "verification_relevant_edit_generation": (
                    self.verification_relevant_edit_generation
                ),
                "last_successful_verification_generation": (
                    self.last_successful_verification_generation
                ),
                "expected_verification_commands": list(self.expected_verification_commands),
                "covered_verification_commands": list(self.covered_verification_commands),
                "missing_verification_commands": list(self.missing_verification_commands),
                "failed_verification_signatures": list(self.failed_verification_signatures),
                "verification_coverage_stale": self.verification_coverage_stale,
                "last_verification_passed": self.last_verification_passed,
                "last_verification_failure_category": self.last_verification_failure_category,
            }
        return {
            "stage": self.stage,
            "material_edit_count": self.material_edit_count,
            "material_edit_tools": list(self.material_edit_tools),
            "touched_repo_paths": list(self.touched_repo_paths),
            **verification_payload,
            "accepted_blocker": self.accepted_blocker,
            "blocked_response_allows_completion": self.blocked_response_allows_completion,
            "verification_expected": self.verification_expected,
            "repo_tool_activity_observed": self.repo_tool_activity_observed,
            "acceptance_status_counts": dict(sorted(self.acceptance_status_counts.items())),
            "acceptance_problems": list(self.acceptance_problems),
            "acceptance_failure_signatures": list(self.acceptance_failure_signatures),
        }

    def episode_id(self) -> str:
        encoded = json.dumps(
            self.episode_payload(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


@dataclass
class CompletionGateControllerState:
    last_episode_id: str = ""
    stagnant_attempt_count: int = 0
    consecutive_no_progress_rejections: int = 0
    total_rejected_finalizations: int = 0
    last_decision_kind: str = ""
    last_reason: str = ""
    last_stage: str = ""
    last_problems: tuple[str, ...] = tuple()
    last_meaningful_progress_since_previous_rejection: bool = False
    last_episode_payload: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "last_episode_id": self.last_episode_id,
            "stagnant_attempt_count": self.stagnant_attempt_count,
            "consecutive_no_progress_rejections": self.consecutive_no_progress_rejections,
            "total_rejected_finalizations": self.total_rejected_finalizations,
            "last_decision_kind": self.last_decision_kind,
            "last_reason": self.last_reason,
            "last_stage": self.last_stage,
            "last_problems": list(self.last_problems),
            "last_meaningful_progress_since_previous_rejection": (
                self.last_meaningful_progress_since_previous_rejection
            ),
            "last_episode_payload": dict(self.last_episode_payload),
        }


@dataclass(frozen=True)
class CompletionGateDecision:
    kind: CompletionGateDecisionKind
    stage: str
    problems: tuple[str, ...]
    episode_id: str
    stagnant_attempt_count: int
    meaningful_progress_since_previous_rejection: bool
    reason: str
    max_stagnant_attempts: int
    consecutive_no_progress_rejections: int
    max_consecutive_no_progress_rejections: int
    previous_episode_id: str = ""
    episode_payload: dict[str, Any] = field(default_factory=dict)

    def as_payload(self) -> dict[str, Any]:
        return {
            "decision": self.kind.value,
            "stage": self.stage,
            "problems": list(self.problems),
            "episode_id": self.episode_id,
            "previous_episode_id": self.previous_episode_id,
            "stagnant_attempt_count": self.stagnant_attempt_count,
            "meaningful_progress_since_previous_rejection": (
                self.meaningful_progress_since_previous_rejection
            ),
            "reason": self.reason,
            "max_stagnant_attempts": self.max_stagnant_attempts,
            "consecutive_no_progress_rejections": (self.consecutive_no_progress_rejections),
            "max_consecutive_no_progress_rejections": (self.max_consecutive_no_progress_rejections),
            "episode_payload": dict(self.episode_payload),
        }


def completion_gate_decision_payload(decision: CompletionGateDecision) -> dict[str, Any]:
    return decision.as_payload()


def build_completion_gate_snapshot(
    *,
    stage: str,
    problems: list[str] | tuple[str, ...],
    material_edit_count: int,
    material_edit_tools: set[str] | list[str] | tuple[str, ...],
    touched_repo_paths: set[str] | list[str] | tuple[str, ...],
    verification_relevant_edit_generation: int,
    last_successful_verification_generation: int | None,
    expected_verification_commands: set[str] | list[str] | tuple[str, ...],
    covered_verification_commands: set[str] | list[str] | tuple[str, ...],
    missing_verification_commands: set[str] | list[str] | tuple[str, ...],
    failed_verification_command_snippets: dict[str, str],
    verification_coverage_stale: bool,
    last_verification_passed: bool | None,
    last_verification_failure_category: str = "",
    accepted_blocker: bool = False,
    blocked_response: bool = False,
    blocked_response_allows_completion: bool = False,
    verification_expected: bool = False,
    final_text: str = "",
    repo_tool_activity_observed: bool = False,
    acceptance_status_counts: dict[str, int] | None = None,
    acceptance_problems: list[str] | tuple[str, ...] = tuple(),
    acceptance_failure_summaries: list[str] | tuple[str, ...] = tuple(),
) -> CompletionGateEvidenceSnapshot:
    failed_signatures = []
    for command, snippet in sorted(failed_verification_command_snippets.items()):
        normalized_command = _normalize_command(command)
        normalized_snippet = normalize_completion_gate_failure_signature(snippet)
        if normalized_command or normalized_snippet:
            failed_signatures.append(f"{normalized_command}: {normalized_snippet}".strip(": "))
    return CompletionGateEvidenceSnapshot(
        stage=str(stage or "generic"),
        problems=tuple(str(item) for item in problems),
        material_edit_count=max(0, int(material_edit_count)),
        material_edit_tools=tuple(sorted(str(item) for item in material_edit_tools if str(item))),
        touched_repo_paths=tuple(sorted(str(item) for item in touched_repo_paths if str(item))),
        verification_relevant_edit_generation=max(0, int(verification_relevant_edit_generation)),
        last_successful_verification_generation=last_successful_verification_generation,
        expected_verification_commands=tuple(
            sorted(_normalize_command(item) for item in expected_verification_commands if str(item))
        ),
        covered_verification_commands=tuple(
            sorted(_normalize_command(item) for item in covered_verification_commands if str(item))
        ),
        missing_verification_commands=tuple(
            sorted(_normalize_command(item) for item in missing_verification_commands if str(item))
        ),
        failed_verification_signatures=tuple(sorted(failed_signatures)),
        verification_coverage_stale=bool(verification_coverage_stale),
        last_verification_passed=last_verification_passed,
        last_verification_failure_category=str(last_verification_failure_category or ""),
        accepted_blocker=bool(accepted_blocker),
        blocked_response=bool(blocked_response),
        blocked_response_allows_completion=bool(blocked_response_allows_completion),
        verification_expected=bool(verification_expected),
        final_text_present=bool(str(final_text or "").strip()),
        repo_tool_activity_observed=bool(repo_tool_activity_observed),
        acceptance_status_counts=dict(acceptance_status_counts or {}),
        acceptance_problems=tuple(sorted(str(item) for item in acceptance_problems if str(item))),
        acceptance_failure_signatures=tuple(
            sorted(
                normalize_completion_gate_failure_signature(str(item))
                for item in acceptance_failure_summaries
                if str(item).strip()
            )
        ),
    )


def _payload_set(payload: Mapping[str, Any], key: str) -> set[str]:
    value = payload.get(key)
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(item) for item in value if str(item)}


def _payload_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _payload_status_count(payload: Mapping[str, Any], status: str) -> int:
    raw_counts = payload.get("acceptance_status_counts")
    if not isinstance(raw_counts, Mapping):
        return 0
    value = raw_counts.get(status)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def _semantic_meaningful_progress(
    previous_payload: Mapping[str, Any],
    current_payload: Mapping[str, Any],
) -> bool:
    if not previous_payload:
        return False
    if _payload_int(current_payload, "material_edit_count") > _payload_int(
        previous_payload, "material_edit_count"
    ):
        return True
    if _payload_set(current_payload, "covered_verification_commands") > _payload_set(
        previous_payload, "covered_verification_commands"
    ):
        return True
    if _payload_set(current_payload, "missing_verification_commands") < _payload_set(
        previous_payload, "missing_verification_commands"
    ):
        return True
    current_success_generation = current_payload.get("last_successful_verification_generation")
    previous_success_generation = previous_payload.get("last_successful_verification_generation")
    if (
        current_payload.get("last_verification_passed") is True
        and current_success_generation is not None
        and current_success_generation != previous_success_generation
    ):
        return True
    if _payload_status_count(current_payload, "PASSED") > _payload_status_count(
        previous_payload, "PASSED"
    ):
        return True
    for status in ("FAILED", "BLOCKED", "UNVERIFIED"):
        if _payload_status_count(current_payload, status) < _payload_status_count(
            previous_payload, status
        ):
            return True
    if _payload_set(current_payload, "acceptance_problems") < _payload_set(
        previous_payload, "acceptance_problems"
    ):
        return True
    if _payload_set(current_payload, "acceptance_failure_signatures") < _payload_set(
        previous_payload, "acceptance_failure_signatures"
    ):
        return True
    current_generation = _payload_int(current_payload, "verification_relevant_edit_generation")
    previous_generation = _payload_int(previous_payload, "verification_relevant_edit_generation")
    if current_generation > previous_generation:
        if _payload_set(current_payload, "failed_verification_signatures") != _payload_set(
            previous_payload, "failed_verification_signatures"
        ):
            return True
        if _payload_set(current_payload, "acceptance_failure_signatures") != _payload_set(
            previous_payload, "acceptance_failure_signatures"
        ):
            return True
    return False


def decide_completion_gate(
    controller_state: CompletionGateControllerState,
    snapshot: CompletionGateEvidenceSnapshot,
    *,
    budget_exhausted: bool = False,
    max_stagnant_attempts: int = DEFAULT_COMPLETION_GATE_STAGNANT_NUDGE_LIMIT,
    max_consecutive_no_progress_rejections: int = (
        DEFAULT_COMPLETION_GATE_CONSECUTIVE_NO_PROGRESS_LIMIT
    ),
    repair_policy: CompletionGateRepairPolicy | None = None,
) -> CompletionGateDecision:
    if repair_policy is None:
        if (
            max_stagnant_attempts == DEFAULT_COMPLETION_GATE_STAGNANT_NUDGE_LIMIT
            and max_consecutive_no_progress_rejections
            == DEFAULT_COMPLETION_GATE_CONSECUTIVE_NO_PROGRESS_LIMIT
        ):
            repair_policy = DEFAULT_COMPLETION_GATE_REPAIR_POLICY
        else:
            repair_policy = CompletionGateRepairPolicy(
                default_stagnant_nudge_limit=max_stagnant_attempts,
                max_consecutive_no_progress_rejections=(max_consecutive_no_progress_rejections),
                stage_stagnant_nudge_limits={},
            )
    stage_stagnant_limit = repair_policy.stagnant_nudge_limit(snapshot.stage)
    consecutive_no_progress_limit = max(
        1,
        int(repair_policy.max_consecutive_no_progress_rejections),
    )
    episode_id = snapshot.episode_id()
    episode_payload = snapshot.episode_payload()
    previous_episode_id = controller_state.last_episode_id
    meaningful_progress = bool(
        previous_episode_id
        and _semantic_meaningful_progress(
            controller_state.last_episode_payload,
            episode_payload,
        )
    )

    if budget_exhausted:
        return CompletionGateDecision(
            kind=CompletionGateDecisionKind.TERMINATE_BUDGET_EXHAUSTED,
            stage=snapshot.stage,
            problems=snapshot.problems,
            episode_id=episode_id,
            previous_episode_id=previous_episode_id,
            stagnant_attempt_count=controller_state.stagnant_attempt_count,
            meaningful_progress_since_previous_rejection=meaningful_progress,
            reason="step_budget_exhausted",
            max_stagnant_attempts=stage_stagnant_limit,
            consecutive_no_progress_rejections=(
                controller_state.consecutive_no_progress_rejections
            ),
            max_consecutive_no_progress_rejections=consecutive_no_progress_limit,
            episode_payload=episode_payload,
        )

    if not snapshot.problems:
        return CompletionGateDecision(
            kind=CompletionGateDecisionKind.ALLOW_FINAL,
            stage=snapshot.stage,
            problems=tuple(),
            episode_id=episode_id,
            previous_episode_id=previous_episode_id,
            stagnant_attempt_count=0,
            meaningful_progress_since_previous_rejection=meaningful_progress,
            reason="requirements_satisfied",
            max_stagnant_attempts=stage_stagnant_limit,
            consecutive_no_progress_rejections=0,
            max_consecutive_no_progress_rejections=consecutive_no_progress_limit,
            episode_payload=episode_payload,
        )

    if meaningful_progress or not previous_episode_id:
        stagnant_attempt_count = 1
        consecutive_no_progress_rejections = 1
    else:
        stagnant_attempt_count = controller_state.stagnant_attempt_count + 1
        consecutive_no_progress_rejections = controller_state.consecutive_no_progress_rejections + 1

    if stagnant_attempt_count > stage_stagnant_limit:
        kind = CompletionGateDecisionKind.TERMINATE_STAGNANT
        reason = "episode_stagnant"
    elif consecutive_no_progress_rejections > consecutive_no_progress_limit:
        kind = CompletionGateDecisionKind.TERMINATE_STAGNANT
        reason = "consecutive_no_progress_rejections"
    else:
        kind = CompletionGateDecisionKind.NUDGE_AND_CONTINUE
        reason = "meaningful_progress_recheck" if meaningful_progress else "requirements_missing"

    return CompletionGateDecision(
        kind=kind,
        stage=snapshot.stage,
        problems=snapshot.problems,
        episode_id=episode_id,
        previous_episode_id=previous_episode_id,
        stagnant_attempt_count=stagnant_attempt_count,
        meaningful_progress_since_previous_rejection=meaningful_progress,
        reason=reason,
        max_stagnant_attempts=stage_stagnant_limit,
        consecutive_no_progress_rejections=consecutive_no_progress_rejections,
        max_consecutive_no_progress_rejections=consecutive_no_progress_limit,
        episode_payload=episode_payload,
    )


def record_completion_gate_decision(
    controller_state: CompletionGateControllerState,
    decision: CompletionGateDecision,
) -> None:
    controller_state.last_decision_kind = decision.kind.value
    controller_state.last_reason = decision.reason
    controller_state.last_stage = decision.stage
    controller_state.last_problems = decision.problems
    controller_state.last_meaningful_progress_since_previous_rejection = (
        decision.meaningful_progress_since_previous_rejection
    )
    controller_state.last_episode_payload = dict(decision.episode_payload)
    if decision.kind == CompletionGateDecisionKind.ALLOW_FINAL:
        return
    if decision.kind == CompletionGateDecisionKind.TERMINATE_BUDGET_EXHAUSTED:
        return
    controller_state.last_episode_id = decision.episode_id
    controller_state.stagnant_attempt_count = decision.stagnant_attempt_count
    controller_state.consecutive_no_progress_rejections = (
        decision.consecutive_no_progress_rejections
    )
    controller_state.total_rejected_finalizations += 1
