from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_CHAT_MAX_STEPS = 50
DEFAULT_TASK_MAX_STEPS = 100
DEFAULT_SUBAGENT_MAX_STEPS = 16

VALID_STEP_BUDGET_POLICIES = {"fixed", "adaptive"}


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return max(1, int(default))
    return max(1, parsed)


def _non_negative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _clamp(value: int, *, minimum: int, maximum: int) -> int:
    if maximum < minimum:
        minimum = maximum
    return max(minimum, min(maximum, value))


def normalize_step_budget_policy(raw_value: Any) -> str:
    normalized = str(raw_value or "").strip().lower()
    if normalized in VALID_STEP_BUDGET_POLICIES:
        return normalized
    return "adaptive"


def resolve_subagent_step_profile(subagent_name: Any) -> str | None:
    normalized = str(subagent_name or "").strip().lower()
    if not normalized:
        return None
    if normalized in {"explorer", "reviewer", "test-strategist"}:
        return normalized
    return "default"


@dataclass(frozen=True)
class StepBudgetRequest:
    kind: str
    policy: str
    hard_cap: int
    fixed_override: int | None = None
    mode: str | None = None
    route: str | None = None
    one_shot_execution: bool = False
    one_shot_turn_intent: str | None = None
    verification_enabled: bool = False
    subagents_enabled: bool = False
    subagent_name: str | None = None
    parent_turn_budget: int | None = None
    attempt_count: int = 1
    image_count: int = 0
    explicit_path_count: int = 0
    acceptance_criteria_count: int = 0
    estimated_files_count: int = 0
    write_scope_count: int = 0
    dependency_count: int = 0
    asset_count: int = 0
    conflict_file_count: int = 0


@dataclass(frozen=True)
class StepBudgetResolution:
    kind: str
    policy: str
    hard_cap: int
    resolved_max_steps: int
    override_applied: bool
    reason: str
    signals_used: dict[str, Any]
    base_steps: int
    profile: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "policy": self.policy,
            "hard_cap": self.hard_cap,
            "resolved_max_steps": self.resolved_max_steps,
            "override_applied": self.override_applied,
            "reason": self.reason,
            "signals_used": dict(self.signals_used),
            "base_steps": self.base_steps,
            "profile": self.profile,
        }


@dataclass
class StepBudgetRuntime:
    active_turn_budget: int | None = None
    last_resolution: StepBudgetResolution | None = None


def _resolve_hard_cap(request: StepBudgetRequest) -> int:
    default_cap = {
        "chat_turn": DEFAULT_CHAT_MAX_STEPS,
        "managed_task": DEFAULT_TASK_MAX_STEPS,
        "conflict_resolution": DEFAULT_TASK_MAX_STEPS,
        "subagent": DEFAULT_SUBAGENT_MAX_STEPS,
    }.get(str(request.kind), DEFAULT_CHAT_MAX_STEPS)
    hard_cap = _positive_int(request.hard_cap, default=default_cap)
    if request.kind == "subagent" and request.parent_turn_budget is not None:
        hard_cap = min(
            hard_cap,
            _positive_int(request.parent_turn_budget, default=hard_cap),
        )
    return max(1, hard_cap)


def _fixed_resolution(
    request: StepBudgetRequest,
    *,
    hard_cap: int,
    policy: str,
    reason: str,
    fixed_value: int,
    profile: str | None,
    signals_used: dict[str, Any] | None = None,
) -> StepBudgetResolution:
    resolved = _clamp(
        _positive_int(fixed_value, default=hard_cap),
        minimum=1,
        maximum=hard_cap,
    )
    return StepBudgetResolution(
        kind=request.kind,
        policy=policy,
        hard_cap=hard_cap,
        resolved_max_steps=resolved,
        override_applied=(reason == "fixed_override"),
        reason=reason,
        signals_used=signals_used or {},
        base_steps=resolved,
        profile=profile,
    )


def resolve_step_budget(request: StepBudgetRequest) -> StepBudgetResolution:
    policy = normalize_step_budget_policy(request.policy)
    hard_cap = _resolve_hard_cap(request)
    profile = (
        resolve_subagent_step_profile(request.subagent_name) if request.kind == "subagent" else None
    )
    if request.fixed_override is not None:
        return _fixed_resolution(
            request,
            hard_cap=hard_cap,
            policy=policy,
            reason="fixed_override",
            fixed_value=request.fixed_override,
            profile=profile,
            signals_used={
                "policy": policy,
                "hard_cap": hard_cap,
                "fixed_override": _positive_int(request.fixed_override, default=hard_cap),
            },
        )
    if policy == "fixed":
        return _fixed_resolution(
            request,
            hard_cap=hard_cap,
            policy=policy,
            reason="fixed_policy",
            fixed_value=hard_cap,
            profile=profile,
            signals_used={"policy": policy, "hard_cap": hard_cap},
        )

    kind = str(request.kind or "").strip().lower()
    if kind == "chat_turn":
        base_steps = 10
        route = str(request.route or "").strip().lower()
        intent = str(request.one_shot_turn_intent or "").strip().lower()
        repo_route = route == "repo"
        execute_like_repo_turn = repo_route and (
            bool(request.one_shot_execution) or intent == "execute"
        )
        resolved = base_steps
        execution_reserve_steps = 0
        if repo_route:
            if request.one_shot_execution:
                resolved += 4
            if intent == "execute":
                resolved += 8
            elif intent == "explain":
                resolved += 2
            if execute_like_repo_turn:
                execution_reserve_steps = 8
                resolved += execution_reserve_steps
            if str(request.mode or "").strip().lower() in {"auto", "fullaccess"}:
                resolved += 4
            if request.verification_enabled and execute_like_repo_turn:
                resolved += 4
            if request.subagents_enabled:
                resolved += 2
            resolved += min(_non_negative_int(request.explicit_path_count), 6)
            resolved += min(_non_negative_int(request.image_count), 3)
        resolved = _clamp(resolved, minimum=6, maximum=hard_cap)
        return StepBudgetResolution(
            kind=kind,
            policy=policy,
            hard_cap=hard_cap,
            resolved_max_steps=resolved,
            override_applied=False,
            reason="adaptive_chat_turn" if repo_route else "adaptive_chat_turn_non_repo",
            signals_used={
                "route": route,
                "mode": request.mode,
                "one_shot_execution": bool(request.one_shot_execution),
                "one_shot_turn_intent": intent or None,
                "execution_reserve_steps": execution_reserve_steps,
                "verification_enabled": bool(request.verification_enabled),
                "subagents_enabled": bool(request.subagents_enabled),
                "explicit_path_count": _non_negative_int(request.explicit_path_count),
                "image_count": _non_negative_int(request.image_count),
            },
            base_steps=base_steps,
        )

    if kind == "managed_task":
        base_steps = 24
        verification_repair_reserve_steps = 14 if request.verification_enabled else 0
        file_scope_count = max(
            _non_negative_int(request.estimated_files_count),
            _non_negative_int(request.write_scope_count),
        )
        attempt_extra = max(_non_negative_int(request.attempt_count) - 1, 0)
        resolved = base_steps
        resolved += 2 * min(_non_negative_int(request.acceptance_criteria_count), 8)
        resolved += 2 * min(file_scope_count, 8)
        resolved += min(_non_negative_int(request.dependency_count), 4)
        resolved += min(
            _non_negative_int(request.asset_count) + _non_negative_int(request.image_count),
            6,
        )
        resolved += verification_repair_reserve_steps
        resolved += 6 * min(attempt_extra, 3)
        resolved = _clamp(resolved, minimum=20, maximum=hard_cap)
        return StepBudgetResolution(
            kind=kind,
            policy=policy,
            hard_cap=hard_cap,
            resolved_max_steps=resolved,
            override_applied=False,
            reason="adaptive_managed_task",
            signals_used={
                "mode": request.mode,
                "verification_enabled": bool(request.verification_enabled),
                "verification_repair_reserve_steps": verification_repair_reserve_steps,
                "attempt_count": _non_negative_int(request.attempt_count),
                "acceptance_criteria_count": _non_negative_int(request.acceptance_criteria_count),
                "estimated_files_count": _non_negative_int(request.estimated_files_count),
                "write_scope_count": _non_negative_int(request.write_scope_count),
                "dependency_count": _non_negative_int(request.dependency_count),
                "asset_count": _non_negative_int(request.asset_count),
                "image_count": _non_negative_int(request.image_count),
            },
            base_steps=base_steps,
        )

    if kind == "conflict_resolution":
        base_steps = 18
        attempt_extra = max(_non_negative_int(request.attempt_count) - 1, 0)
        resolved = base_steps
        resolved += 3 * min(_non_negative_int(request.conflict_file_count), 8)
        if request.verification_enabled:
            resolved += 4
        resolved += 4 * min(attempt_extra, 3)
        resolved = _clamp(resolved, minimum=12, maximum=hard_cap)
        return StepBudgetResolution(
            kind=kind,
            policy=policy,
            hard_cap=hard_cap,
            resolved_max_steps=resolved,
            override_applied=False,
            reason="adaptive_conflict_resolution",
            signals_used={
                "mode": request.mode,
                "verification_enabled": bool(request.verification_enabled),
                "attempt_count": _non_negative_int(request.attempt_count),
                "conflict_file_count": _non_negative_int(request.conflict_file_count),
            },
            base_steps=base_steps,
        )

    if kind == "subagent":
        base_steps = 8
        resolved = base_steps
        if profile == "explorer":
            resolved += 4
        elif profile == "reviewer":
            resolved += 2
        elif profile == "test-strategist":
            resolved += 3
        resolved += min(_non_negative_int(request.explicit_path_count), 4)
        if str(request.mode or "").strip().lower() != "readonly":
            resolved += 2
        resolved = _clamp(resolved, minimum=4, maximum=hard_cap)
        return StepBudgetResolution(
            kind=kind,
            policy=policy,
            hard_cap=hard_cap,
            resolved_max_steps=resolved,
            override_applied=False,
            reason="adaptive_subagent",
            signals_used={
                "mode": request.mode,
                "subagent_name": request.subagent_name,
                "parent_turn_budget": (
                    _positive_int(request.parent_turn_budget, default=hard_cap)
                    if request.parent_turn_budget is not None
                    else None
                ),
                "explicit_path_count": _non_negative_int(request.explicit_path_count),
            },
            base_steps=base_steps,
            profile=profile,
        )

    base_steps = hard_cap
    return StepBudgetResolution(
        kind=kind or "chat_turn",
        policy=policy,
        hard_cap=hard_cap,
        resolved_max_steps=hard_cap,
        override_applied=False,
        reason="unknown_kind_fallback",
        signals_used={"requested_kind": request.kind},
        base_steps=base_steps,
        profile=profile,
    )
