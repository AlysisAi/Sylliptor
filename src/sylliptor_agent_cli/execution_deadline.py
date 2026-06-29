from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

DEFAULT_DEADLINE_CLEANUP_RESERVE_SECONDS = 1.0
MINIMUM_OPERATION_TIMEOUT_SECONDS = 0.05
MINIMUM_LLM_START_SECONDS = 0.25
MINIMUM_TOOL_START_SECONDS = 0.05
MINIMUM_FORCED_SUMMARY_SECONDS = 2.0
MINIMUM_SUBAGENT_START_SECONDS = 2.0
DEFAULT_FINALIZATION_MINIMUM_SECONDS = 1.0
DEFAULT_FINALIZATION_MAX_SECONDS = 120.0
DEFAULT_FINALIZATION_MAX_FRACTION = 0.25
_MISSING = object()


class DeadlinePhase(StrEnum):
    NORMAL = "normal"
    FINALIZATION_WINDOW = "finalization_window"
    EXHAUSTED = "exhausted"


class DeadlineSource(StrEnum):
    EXPLICIT_CLI = "explicit_cli"
    ENVIRONMENT = "environment"
    CONFIG = "config"
    INHERITED_PARENT = "inherited_parent"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class DeadlineOperation(StrEnum):
    ROUTING_LLM = "routing_llm"
    MAIN_LLM = "main_llm"
    MAIN_LLM_RETRY = "main_llm_retry"
    COMPACTION_LLM = "compaction_llm"
    ADAPTIVE_RETRY_LLM = "adaptive_retry_llm"
    SUBAGENT = "subagent"
    VERIFICATION = "verification"
    SHELL_TOOL = "shell_tool"
    SHELL_BACKGROUND = "shell_background"
    EXPLORATION_TOOL = "exploration_tool"
    MUTATION_TOOL = "mutation_tool"
    PROVIDER_RETRY_SLEEP = "provider_retry_sleep"
    LOCAL_FINAL_SUMMARY = "local_final_summary"
    TOOL_DISPATCH = "tool_dispatch"


_FINALIZATION_BLOCKED_OPERATIONS = frozenset(
    {
        DeadlineOperation.ROUTING_LLM.value,
        DeadlineOperation.MAIN_LLM_RETRY.value,
        DeadlineOperation.COMPACTION_LLM.value,
        DeadlineOperation.ADAPTIVE_RETRY_LLM.value,
        DeadlineOperation.SUBAGENT.value,
        DeadlineOperation.SHELL_BACKGROUND.value,
        DeadlineOperation.EXPLORATION_TOOL.value,
        DeadlineOperation.PROVIDER_RETRY_SLEEP.value,
    }
)


class DeadlineExhausted(RuntimeError):
    """Internal control-flow marker for run deadline exhaustion."""


def validate_deadline_seconds(value: Any, *, key: str = "run_deadline_seconds") -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a finite number > 0") from exc
    if parsed <= 0 or not math.isfinite(parsed):
        raise ValueError(f"{key} must be a finite number > 0")
    return parsed


@dataclass(frozen=True)
class DeadlineFinalizationPolicy:
    minimum_reserve_seconds: float = DEFAULT_FINALIZATION_MINIMUM_SECONDS
    cleanup_reserve_seconds: float = DEFAULT_DEADLINE_CLEANUP_RESERVE_SECONDS
    max_reserve_seconds: float = DEFAULT_FINALIZATION_MAX_SECONDS
    max_reserve_fraction: float = DEFAULT_FINALIZATION_MAX_FRACTION
    llm_latency_multiplier: float = 1.5
    verification_latency_multiplier: float = 1.25
    tool_latency_multiplier: float = 1.1
    local_cleanup_seconds: float = DEFAULT_DEADLINE_CLEANUP_RESERVE_SECONDS

    def normalized(self) -> DeadlineFinalizationPolicy:
        return DeadlineFinalizationPolicy(
            minimum_reserve_seconds=max(0.0, float(self.minimum_reserve_seconds)),
            cleanup_reserve_seconds=max(0.0, float(self.cleanup_reserve_seconds)),
            max_reserve_seconds=max(0.0, float(self.max_reserve_seconds)),
            max_reserve_fraction=max(0.0, min(1.0, float(self.max_reserve_fraction))),
            llm_latency_multiplier=max(0.0, float(self.llm_latency_multiplier)),
            verification_latency_multiplier=max(0.0, float(self.verification_latency_multiplier)),
            tool_latency_multiplier=max(0.0, float(self.tool_latency_multiplier)),
            local_cleanup_seconds=max(0.0, float(self.local_cleanup_seconds)),
        )


@dataclass
class DeadlineDurationObservation:
    count: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0

    def record(self, duration_seconds: float) -> None:
        duration = max(0.0, float(duration_seconds))
        self.count += 1
        self.total_seconds += duration
        self.max_seconds = max(self.max_seconds, duration)

    @property
    def average_seconds(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.total_seconds / float(self.count)

    def estimate_seconds(self) -> float:
        if self.count <= 0:
            return 0.0
        return max(self.average_seconds, self.max_seconds)

    def telemetry_snapshot(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "average_seconds": round(self.average_seconds, 6),
            "max_seconds": round(self.max_seconds, 6),
        }


@dataclass(frozen=True)
class DeadlineStartDecision:
    operation: str
    allowed: bool
    phase: DeadlinePhase
    reason: str
    remaining_seconds: float | None
    normal_work_remaining_seconds: float | None
    finalization_reserve_seconds: float
    minimum_required_seconds: float
    estimated_duration_seconds: float | None

    def telemetry_snapshot(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "allowed": self.allowed,
            "phase": self.phase.value,
            "reason": self.reason,
            "remaining_seconds": (
                None if self.remaining_seconds is None else round(self.remaining_seconds, 6)
            ),
            "normal_work_remaining_seconds": (
                None
                if self.normal_work_remaining_seconds is None
                else round(self.normal_work_remaining_seconds, 6)
            ),
            "finalization_reserve_seconds": round(self.finalization_reserve_seconds, 6),
            "minimum_required_seconds": round(max(0.0, self.minimum_required_seconds), 6),
            "estimated_duration_seconds": (
                None
                if self.estimated_duration_seconds is None
                else round(max(0.0, self.estimated_duration_seconds), 6)
            ),
        }


@dataclass
class ExecutionDeadline:
    started_at_monotonic: float
    deadline_monotonic: float | None
    configured_duration_seconds: float | None = None
    source: DeadlineSource | str = DeadlineSource.UNKNOWN
    finalization_policy: DeadlineFinalizationPolicy = field(
        default_factory=DeadlineFinalizationPolicy,
        repr=False,
        compare=False,
    )
    _clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)
    _duration_observations: dict[str, DeadlineDurationObservation] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _finalization_reason: str | None = field(default=None, repr=False, compare=False)
    _finalization_entered_at_monotonic: float | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _finalization_directive_sent: bool = field(default=False, repr=False, compare=False)
    _finalization_llm_started: bool = field(default=False, repr=False, compare=False)

    @classmethod
    def from_duration(
        cls,
        duration_seconds: float | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        source: DeadlineSource | str = DeadlineSource.UNKNOWN,
        finalization_policy: DeadlineFinalizationPolicy | None = None,
    ) -> ExecutionDeadline:
        started = float(clock())
        if duration_seconds is None:
            return cls(
                started_at_monotonic=started,
                deadline_monotonic=None,
                configured_duration_seconds=None,
                source=source,
                finalization_policy=finalization_policy or DeadlineFinalizationPolicy(),
                _clock=clock,
            )
        duration = validate_deadline_seconds(duration_seconds)
        return cls(
            started_at_monotonic=started,
            deadline_monotonic=started + duration,
            configured_duration_seconds=duration,
            source=source,
            finalization_policy=finalization_policy or DeadlineFinalizationPolicy(),
            _clock=clock,
        )

    @classmethod
    def from_absolute(
        cls,
        *,
        started_at_monotonic: float,
        deadline_monotonic: float | None,
        configured_duration_seconds: float | None = None,
        clock: Callable[[], float] = time.monotonic,
        source: DeadlineSource | str = DeadlineSource.INHERITED_PARENT,
        finalization_policy: DeadlineFinalizationPolicy | None = None,
    ) -> ExecutionDeadline:
        if deadline_monotonic is not None and not math.isfinite(float(deadline_monotonic)):
            raise ValueError("deadline_monotonic must be finite when provided")
        if configured_duration_seconds is not None:
            validate_deadline_seconds(configured_duration_seconds)
        return cls(
            started_at_monotonic=float(started_at_monotonic),
            deadline_monotonic=(
                float(deadline_monotonic) if deadline_monotonic is not None else None
            ),
            configured_duration_seconds=configured_duration_seconds,
            source=source,
            finalization_policy=finalization_policy or DeadlineFinalizationPolicy(),
            _clock=clock,
        )

    @property
    def enabled(self) -> bool:
        return self.deadline_monotonic is not None

    def elapsed_seconds(self) -> float:
        return max(0.0, float(self._clock()) - self.started_at_monotonic)

    def remaining_seconds(self) -> float | None:
        if self.deadline_monotonic is None:
            return None
        return max(0.0, self.deadline_monotonic - float(self._clock()))

    def is_exhausted(self) -> bool:
        remaining = self.remaining_seconds()
        return remaining is not None and remaining <= 0.0

    def can_start(self, minimum_remaining_seconds: float = 0.0) -> bool:
        remaining = self.remaining_seconds()
        if remaining is None:
            return True
        return remaining >= max(0.0, float(minimum_remaining_seconds))

    def clamp_timeout(
        self,
        configured_timeout_seconds: float | None,
        *,
        reserve_seconds: float = DEFAULT_DEADLINE_CLEANUP_RESERVE_SECONDS,
        minimum_timeout_seconds: float = MINIMUM_OPERATION_TIMEOUT_SECONDS,
    ) -> float | None:
        remaining = self.remaining_seconds()
        if remaining is None:
            return configured_timeout_seconds
        safe_remaining = remaining - max(0.0, float(reserve_seconds))
        if safe_remaining < max(0.0, float(minimum_timeout_seconds)):
            return None
        if configured_timeout_seconds is None:
            return safe_remaining
        configured = validate_deadline_seconds(
            configured_timeout_seconds,
            key="configured_timeout_seconds",
        )
        return max(minimum_timeout_seconds, min(configured, safe_remaining))

    def observe_duration(self, operation: str | DeadlineOperation, duration_seconds: float) -> None:
        key = _normalize_operation(operation)
        observation = self._duration_observations.setdefault(
            key,
            DeadlineDurationObservation(),
        )
        observation.record(duration_seconds)

    def estimated_duration_seconds(
        self,
        operation: str | DeadlineOperation,
        *,
        configured_timeout_seconds: float | None = None,
        default_seconds: float | None = None,
    ) -> float | None:
        key = _normalize_operation(operation)
        observed = self._duration_observations.get(key)
        candidates: list[float] = []
        if observed is not None and observed.count > 0:
            candidates.append(observed.estimate_seconds())
        if default_seconds is not None:
            candidates.append(max(0.0, float(default_seconds)))
        if configured_timeout_seconds is not None:
            configured = validate_deadline_seconds(
                configured_timeout_seconds,
                key="configured_timeout_seconds",
            )
            clamped = self.clamp_timeout(configured)
            candidates.append(configured if clamped is None else clamped)
        if not candidates:
            return None
        return max(candidates)

    def finalization_reserve_seconds(
        self,
        policy: DeadlineFinalizationPolicy | None = None,
    ) -> float:
        if not self.enabled:
            return 0.0
        effective = (policy or self.finalization_policy).normalized()
        observed_llm = max(
            (
                observation.estimate_seconds()
                for name, observation in self._duration_observations.items()
                if "llm" in name and observation.count > 0
            ),
            default=0.0,
        )
        observed_verification = max(
            (
                observation.estimate_seconds()
                for name, observation in self._duration_observations.items()
                if ("verification" in name or "verify" in name) and observation.count > 0
            ),
            default=0.0,
        )
        observed_tool = max(
            (
                observation.estimate_seconds()
                for name, observation in self._duration_observations.items()
                if "llm" not in name
                and "verification" not in name
                and "verify" not in name
                and observation.count > 0
            ),
            default=0.0,
        )
        raw_reserve = max(
            effective.minimum_reserve_seconds,
            effective.cleanup_reserve_seconds,
            effective.local_cleanup_seconds,
            observed_llm * effective.llm_latency_multiplier,
            observed_verification * effective.verification_latency_multiplier,
            observed_tool * effective.tool_latency_multiplier,
        )
        if self.configured_duration_seconds is None:
            relative_cap = effective.max_reserve_seconds
        else:
            relative_cap = max(
                MINIMUM_OPERATION_TIMEOUT_SECONDS,
                self.configured_duration_seconds * effective.max_reserve_fraction,
            )
        cap = max(
            MINIMUM_OPERATION_TIMEOUT_SECONDS,
            min(effective.max_reserve_seconds, relative_cap),
        )
        return max(0.0, min(raw_reserve, cap))

    def normal_work_remaining_seconds(self) -> float | None:
        remaining = self.remaining_seconds()
        if remaining is None:
            return None
        return max(0.0, remaining - self.finalization_reserve_seconds())

    def phase(self) -> DeadlinePhase:
        if self.is_exhausted():
            return DeadlinePhase.EXHAUSTED
        remaining = self.remaining_seconds()
        if remaining is None:
            return DeadlinePhase.NORMAL
        if remaining <= self.finalization_reserve_seconds():
            return DeadlinePhase.FINALIZATION_WINDOW
        return DeadlinePhase.NORMAL

    def maybe_enter_finalization(self, reason: str = "reserve_reached") -> bool:
        phase = self.phase()
        if phase != DeadlinePhase.FINALIZATION_WINDOW:
            return False
        if self._finalization_reason is None:
            self._finalization_reason = str(reason or "reserve_reached")
            self._finalization_entered_at_monotonic = float(self._clock())
            return True
        return False

    @property
    def finalization_reason(self) -> str | None:
        if self.phase() == DeadlinePhase.FINALIZATION_WINDOW:
            return self._finalization_reason or "reserve_reached"
        return self._finalization_reason

    @property
    def finalization_directive_sent(self) -> bool:
        return self._finalization_directive_sent

    def mark_finalization_directive_sent(self) -> None:
        self._finalization_directive_sent = True

    @property
    def finalization_llm_started(self) -> bool:
        return self._finalization_llm_started

    def mark_finalization_llm_started(self) -> None:
        self._finalization_llm_started = True

    def start_decision(
        self,
        operation: str | DeadlineOperation,
        *,
        minimum_remaining_seconds: float = 0.0,
        estimated_duration_seconds: float | None = None,
        configured_timeout_seconds: float | None = None,
        allow_during_finalization: bool = False,
    ) -> DeadlineStartDecision:
        operation_name = _normalize_operation(operation)
        remaining = self.remaining_seconds()
        reserve = self.finalization_reserve_seconds()
        normal_remaining = self.normal_work_remaining_seconds()
        phase = self.phase()
        minimum = max(0.0, float(minimum_remaining_seconds))
        estimate = estimated_duration_seconds
        if estimate is None:
            estimate = self.estimated_duration_seconds(
                operation_name,
                configured_timeout_seconds=configured_timeout_seconds,
                default_seconds=minimum,
            )
        if remaining is None:
            return DeadlineStartDecision(
                operation=operation_name,
                allowed=True,
                phase=phase,
                reason="deadline_unconfigured",
                remaining_seconds=None,
                normal_work_remaining_seconds=None,
                finalization_reserve_seconds=reserve,
                minimum_required_seconds=minimum,
                estimated_duration_seconds=estimate,
            )
        if phase == DeadlinePhase.EXHAUSTED:
            return DeadlineStartDecision(
                operation=operation_name,
                allowed=False,
                phase=phase,
                reason="deadline_exhausted",
                remaining_seconds=remaining,
                normal_work_remaining_seconds=normal_remaining,
                finalization_reserve_seconds=reserve,
                minimum_required_seconds=minimum,
                estimated_duration_seconds=estimate,
            )
        if remaining < minimum:
            return DeadlineStartDecision(
                operation=operation_name,
                allowed=False,
                phase=phase,
                reason="insufficient_hard_remaining",
                remaining_seconds=remaining,
                normal_work_remaining_seconds=normal_remaining,
                finalization_reserve_seconds=reserve,
                minimum_required_seconds=minimum,
                estimated_duration_seconds=estimate,
            )
        if phase == DeadlinePhase.FINALIZATION_WINDOW:
            blocked = operation_name in _FINALIZATION_BLOCKED_OPERATIONS
            if blocked and not allow_during_finalization:
                return DeadlineStartDecision(
                    operation=operation_name,
                    allowed=False,
                    phase=phase,
                    reason="finalization_disallows_operation",
                    remaining_seconds=remaining,
                    normal_work_remaining_seconds=normal_remaining,
                    finalization_reserve_seconds=reserve,
                    minimum_required_seconds=minimum,
                    estimated_duration_seconds=estimate,
                )
            return DeadlineStartDecision(
                operation=operation_name,
                allowed=True,
                phase=phase,
                reason="finalization_allowed",
                remaining_seconds=remaining,
                normal_work_remaining_seconds=normal_remaining,
                finalization_reserve_seconds=reserve,
                minimum_required_seconds=minimum,
                estimated_duration_seconds=estimate,
            )
        if estimate is not None and normal_remaining is not None and estimate > normal_remaining:
            return DeadlineStartDecision(
                operation=operation_name,
                allowed=False,
                phase=phase,
                reason="insufficient_normal_work_remaining",
                remaining_seconds=remaining,
                normal_work_remaining_seconds=normal_remaining,
                finalization_reserve_seconds=reserve,
                minimum_required_seconds=minimum,
                estimated_duration_seconds=estimate,
            )
        return DeadlineStartDecision(
            operation=operation_name,
            allowed=True,
            phase=phase,
            reason="normal_work_allowed",
            remaining_seconds=remaining,
            normal_work_remaining_seconds=normal_remaining,
            finalization_reserve_seconds=reserve,
            minimum_required_seconds=minimum,
            estimated_duration_seconds=estimate,
        )

    def telemetry_snapshot(self) -> dict[str, Any]:
        remaining = self.remaining_seconds()
        phase = self.phase()
        normal_work_remaining = self.normal_work_remaining_seconds()
        return {
            "enabled": self.enabled,
            "configured_seconds": self.configured_duration_seconds,
            "source": _normalize_source(self.source),
            "deadline_monotonic": self.deadline_monotonic,
            "elapsed_seconds": round(self.elapsed_seconds(), 6),
            "remaining_seconds": None if remaining is None else round(remaining, 6),
            "normal_work_remaining_seconds": (
                None if normal_work_remaining is None else round(normal_work_remaining, 6)
            ),
            "finalization_reserve_seconds": round(self.finalization_reserve_seconds(), 6),
            "phase": phase.value,
            "finalization_reason": self.finalization_reason,
            "finalization_directive_sent": self.finalization_directive_sent,
            "finalization_llm_started": self.finalization_llm_started,
            "exhausted": self.is_exhausted(),
            "duration_observations": {
                key: observation.telemetry_snapshot()
                for key, observation in sorted(self._duration_observations.items())
            },
        }


def _normalize_operation(operation: str | DeadlineOperation) -> str:
    raw = getattr(operation, "value", operation)
    return str(raw or "operation").strip().lower() or "operation"


def _normalize_source(source: DeadlineSource | str) -> str:
    raw = getattr(source, "value", source)
    cleaned = str(raw or "").strip().lower()
    return cleaned or DeadlineSource.UNKNOWN.value


def deadline_timeout_or_raise(
    deadline: ExecutionDeadline | None,
    configured_timeout_seconds: float | None,
    *,
    reserve_seconds: float = DEFAULT_DEADLINE_CLEANUP_RESERVE_SECONDS,
    minimum_timeout_seconds: float = MINIMUM_OPERATION_TIMEOUT_SECONDS,
    operation: str = "operation",
) -> float | None:
    if deadline is None:
        return configured_timeout_seconds
    timeout = deadline.clamp_timeout(
        configured_timeout_seconds,
        reserve_seconds=reserve_seconds,
        minimum_timeout_seconds=minimum_timeout_seconds,
    )
    if timeout is None:
        raise DeadlineExhausted(f"run deadline exhausted before {operation}")
    return timeout


@contextmanager
def temporarily_clamp_client_timeout(
    client: Any,
    deadline: ExecutionDeadline | None,
    *,
    reserve_seconds: float = DEFAULT_DEADLINE_CLEANUP_RESERVE_SECONDS,
    minimum_timeout_seconds: float = MINIMUM_OPERATION_TIMEOUT_SECONDS,
    operation: str = "llm_call",
) -> Iterator[None]:
    if deadline is None or not hasattr(client, "timeout_s"):
        yield
        return
    original = client.timeout_s
    original_retry_deadline = getattr(client, "_provider_retry_deadline_allows", _MISSING)
    original_finalization_retry_used = getattr(
        client,
        "_provider_finalization_retry_used",
        _MISSING,
    )
    timeout = deadline_timeout_or_raise(
        deadline,
        float(original) if original is not None else None,
        reserve_seconds=reserve_seconds,
        minimum_timeout_seconds=minimum_timeout_seconds,
        operation=operation,
    )

    def _provider_retry_deadline_allows(wait_seconds: float) -> bool:
        retry_window_seconds = max(0.0, float(wait_seconds)) + max(
            0.0,
            float(minimum_timeout_seconds),
        )
        in_finalization = deadline.phase() == DeadlinePhase.FINALIZATION_WINDOW
        finalization_retry_available = in_finalization and not bool(
            getattr(client, "_provider_finalization_retry_used", False)
        )
        decision = deadline.start_decision(
            DeadlineOperation.PROVIDER_RETRY_SLEEP,
            minimum_remaining_seconds=retry_window_seconds,
            estimated_duration_seconds=retry_window_seconds,
            allow_during_finalization=finalization_retry_available,
        )
        if not decision.allowed:
            return False
        if finalization_retry_available:
            client._provider_finalization_retry_used = True
        return True

    client.timeout_s = timeout
    client._provider_retry_deadline_allows = _provider_retry_deadline_allows
    try:
        yield
    finally:
        finalization_retry_used = bool(getattr(client, "_provider_finalization_retry_used", False))
        client.timeout_s = original
        if original_retry_deadline is _MISSING:
            try:
                delattr(client, "_provider_retry_deadline_allows")
            except AttributeError:
                pass
        else:
            client._provider_retry_deadline_allows = original_retry_deadline
        if original_finalization_retry_used is _MISSING:
            if finalization_retry_used:
                client._provider_finalization_retry_used = True
            else:
                try:
                    delattr(client, "_provider_finalization_retry_used")
                except AttributeError:
                    pass
        else:
            client._provider_finalization_retry_used = (
                bool(original_finalization_retry_used) or finalization_retry_used
            )
