from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .acceptance_contract import (
    AcceptanceContract,
    AcceptanceCriterion,
    AcceptanceCriterionEnforcement,
    AcceptanceCriterionKind,
    AcceptanceCriterionStatus,
)


class CompletionCertificateStatus(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    CONTRADICTED = "CONTRADICTED"


_EVIDENCE_RANK = {
    "HOST_AUTHORITATIVE": 0,
    "USER_EXPLICIT": 1,
    "PREEXISTING_TASK_CHECKER": 2,
    "DIRECT_BLACK_BOX": 3,
    "PREEXISTING_REPO_NATIVE": 4,
    "SELF_AUTHORED": 5,
    "AD_HOC_OBSERVATION": 6,
}


@dataclass(frozen=True)
class CompletionCertificate:
    status: CompletionCertificateStatus
    problems: tuple[str, ...] = tuple()
    hard_criterion_ids: tuple[str, ...] = tuple()
    covered_hard_criterion_ids: tuple[str, ...] = tuple()
    failed_hard_criterion_ids: tuple[str, ...] = tuple()
    evidence_hierarchy: tuple[str, ...] = tuple()
    reason: str = ""

    def as_payload(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "problems": list(self.problems),
            "hard_criterion_ids": list(self.hard_criterion_ids),
            "covered_hard_criterion_ids": list(self.covered_hard_criterion_ids),
            "failed_hard_criterion_ids": list(self.failed_hard_criterion_ids),
            "evidence_hierarchy": list(self.evidence_hierarchy),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CompletionCertificateInput:
    contract: AcceptanceContract | None
    final_text: str
    blocked: bool
    blocker_valid: bool
    material_edit_count: int
    require_material_result: bool
    verification_expected: bool
    verification_attempt_count: int
    last_verification_passed: bool | None
    failed_verification_commands: set[str] = field(default_factory=set)
    expected_verification_commands: set[str] = field(default_factory=set)
    missing_verification_commands: set[str] = field(default_factory=set)
    verification_coverage_stale: bool = False
    accepted_verification_evidence: list[dict[str, Any]] = field(default_factory=list)


def evaluate_completion_certificate(
    certificate_input: CompletionCertificateInput,
) -> CompletionCertificate:
    problems: list[str] = []
    failed_hard: list[str] = []
    covered_hard: list[str] = []

    if not str(certificate_input.final_text or "").strip():
        problems.append("empty_final_response")
    if certificate_input.blocked:
        if not certificate_input.blocker_valid:
            problems.append("acceptance_evidence_insufficient")
    elif certificate_input.require_material_result and certificate_input.material_edit_count <= 0:
        problems.append("no_material_edits")

    if certificate_input.verification_expected:
        if certificate_input.verification_attempt_count <= 0:
            problems.append("verification_not_attempted")
        elif certificate_input.failed_verification_commands:
            problems.append("verification_failed")
        elif certificate_input.last_verification_passed is not True:
            problems.append("verification_failed")
        elif certificate_input.missing_verification_commands:
            problems.append("verification_incomplete")
        elif certificate_input.verification_coverage_stale:
            problems.append("verification_incomplete")

    hard_criteria = _hard_criteria(certificate_input.contract)
    blocker_covers_missing = certificate_input.blocked and certificate_input.blocker_valid
    for criterion in hard_criteria:
        if criterion.status == AcceptanceCriterionStatus.PASSED:
            covered_hard.append(criterion.criterion_id)
            continue
        if criterion.status == AcceptanceCriterionStatus.FAILED:
            failed_hard.append(criterion.criterion_id)
            problems.append("acceptance_criteria_failed")
            if criterion.kind == AcceptanceCriterionKind.PRESERVATION_UNCHANGED_PATH:
                problems.append("unexpected_scope_changes")
            continue
        if criterion.status == AcceptanceCriterionStatus.BLOCKED:
            if blocker_covers_missing:
                continue
            problems.append("acceptance_evidence_insufficient")
            continue
        if criterion.status == AcceptanceCriterionStatus.UNVERIFIED:
            if blocker_covers_missing:
                continue
            problems.append("acceptance_criteria_unverified")

    status = CompletionCertificateStatus.SUFFICIENT
    reason = "requirements_satisfied"
    deduped_problems = tuple(dict.fromkeys(problems))
    if failed_hard or "verification_failed" in deduped_problems:
        status = CompletionCertificateStatus.CONTRADICTED
        reason = "hard_requirement_failed"
    elif deduped_problems:
        status = CompletionCertificateStatus.INSUFFICIENT
        reason = "requirements_missing"

    return CompletionCertificate(
        status=status,
        problems=deduped_problems,
        hard_criterion_ids=tuple(criterion.criterion_id for criterion in hard_criteria),
        covered_hard_criterion_ids=tuple(covered_hard),
        failed_hard_criterion_ids=tuple(failed_hard),
        evidence_hierarchy=_evidence_hierarchy(certificate_input.accepted_verification_evidence),
        reason=reason,
    )


def _hard_criteria(contract: AcceptanceContract | None) -> list[AcceptanceCriterion]:
    if contract is None:
        return []
    return [
        criterion
        for criterion in contract.criteria
        if criterion.enforcement == AcceptanceCriterionEnforcement.HARD
        and criterion.required
        and criterion.required_for_finalization
    ]


def _evidence_hierarchy(evidence: list[dict[str, Any]]) -> tuple[str, ...]:
    origins = set()
    for item in evidence:
        if not isinstance(item, dict):
            continue
        raw = str(
            item.get("origin")
            or item.get("evidence_origin")
            or item.get("evidence_category")
            or item.get("category")
            or ""
        )
        if not raw:
            continue
        origins.add(
            {
                "AUTHORITATIVE": "HOST_AUTHORITATIVE",
                "TASK_ACCEPTANCE": "DIRECT_BLACK_BOX",
                "REPO_NATIVE": "PREEXISTING_REPO_NATIVE",
            }.get(raw, raw)
        )
    return tuple(
        sorted(
            (origin for origin in origins if origin),
            key=lambda origin: (_EVIDENCE_RANK.get(origin, 99), origin),
        )
    )
