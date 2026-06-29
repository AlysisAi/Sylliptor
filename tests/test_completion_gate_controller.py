from __future__ import annotations

import pytest

from sylliptor_agent_cli.agent_loop import (
    CompletionGateControllerState,
    CompletionGateDecisionKind,
    CompletionGateRepairPolicy,
    TurnExecutionState,
    build_completion_gate_snapshot,
    decide_completion_gate,
    normalize_completion_gate_failure_signature,
    record_completion_gate_decision,
)


def _snapshot(
    *,
    stage: str = "no_material_edits",
    material_edit_count: int = 0,
    verification_generation: int = 0,
    covered: set[str] | None = None,
    missing: set[str] | None = None,
    failures: dict[str, str] | None = None,
):
    return build_completion_gate_snapshot(
        stage=stage,
        problems=[stage],
        material_edit_count=material_edit_count,
        material_edit_tools={"fs_write"} if material_edit_count else set(),
        touched_repo_paths={"src/app.py"} if material_edit_count else set(),
        verification_relevant_edit_generation=verification_generation,
        last_successful_verification_generation=None,
        expected_verification_commands={"pytest -q"},
        covered_verification_commands=covered or set(),
        missing_verification_commands=missing if missing is not None else {"pytest -q"},
        failed_verification_command_snippets=failures or {},
        verification_coverage_stale=False,
        last_verification_passed=None,
        verification_expected=True,
        final_text="Implemented it.",
        repo_tool_activity_observed=True,
    )


def test_controller_allows_final_when_no_problems() -> None:
    state = CompletionGateControllerState()
    snapshot = build_completion_gate_snapshot(
        stage="complete",
        problems=[],
        material_edit_count=1,
        material_edit_tools={"fs_write"},
        touched_repo_paths={"src/app.py"},
        verification_relevant_edit_generation=1,
        last_successful_verification_generation=1,
        expected_verification_commands={"pytest -q"},
        covered_verification_commands={"pytest -q"},
        missing_verification_commands=set(),
        failed_verification_command_snippets={},
        verification_coverage_stale=False,
        last_verification_passed=True,
        verification_expected=True,
        final_text="Implemented and verified.",
        repo_tool_activity_observed=True,
    )

    decision = decide_completion_gate(state, snapshot)

    assert decision.kind == CompletionGateDecisionKind.ALLOW_FINAL
    assert decision.reason == "requirements_satisfied"


def test_repeated_no_progress_rejections_terminate_as_stagnant() -> None:
    state = CompletionGateControllerState()
    snapshot = _snapshot()

    first = decide_completion_gate(state, snapshot)
    record_completion_gate_decision(state, first)
    second = decide_completion_gate(state, snapshot)
    record_completion_gate_decision(state, second)
    third = decide_completion_gate(state, snapshot)

    assert first.kind == CompletionGateDecisionKind.NUDGE_AND_CONTINUE
    assert second.kind == CompletionGateDecisionKind.NUDGE_AND_CONTINUE
    assert third.kind == CompletionGateDecisionKind.TERMINATE_STAGNANT
    assert third.reason == "episode_stagnant"
    assert second.stagnant_attempt_count == 2
    assert third.stagnant_attempt_count == 3


def test_repeated_no_progress_rejection_permits_second_repair_nudge() -> None:
    state = CompletionGateControllerState()
    snapshot = _snapshot(stage="verification_not_attempted", material_edit_count=1)

    first = decide_completion_gate(state, snapshot)
    record_completion_gate_decision(state, first)
    second = decide_completion_gate(state, snapshot)
    record_completion_gate_decision(state, second)
    third = decide_completion_gate(state, snapshot)

    assert first.kind == CompletionGateDecisionKind.NUDGE_AND_CONTINUE
    assert second.kind == CompletionGateDecisionKind.NUDGE_AND_CONTINUE
    assert second.meaningful_progress_since_previous_rejection is False
    assert third.kind == CompletionGateDecisionKind.TERMINATE_STAGNANT


def test_policy_object_controls_stage_and_global_liveness_bounds() -> None:
    state = CompletionGateControllerState()
    snapshot = _snapshot(stage="verification_failed", material_edit_count=1)
    policy = CompletionGateRepairPolicy(
        default_stagnant_nudge_limit=3,
        max_consecutive_no_progress_rejections=2,
        stage_stagnant_nudge_limits={"verification_failed": 4},
    )

    first = decide_completion_gate(state, snapshot, repair_policy=policy)
    record_completion_gate_decision(state, first)
    second = decide_completion_gate(state, snapshot, repair_policy=policy)
    record_completion_gate_decision(state, second)
    third = decide_completion_gate(state, snapshot, repair_policy=policy)

    assert first.kind == CompletionGateDecisionKind.NUDGE_AND_CONTINUE
    assert second.kind == CompletionGateDecisionKind.NUDGE_AND_CONTINUE
    assert third.kind == CompletionGateDecisionKind.TERMINATE_STAGNANT
    assert third.reason == "consecutive_no_progress_rejections"
    assert third.max_stagnant_attempts == 4
    assert third.max_consecutive_no_progress_rejections == 2


def test_material_progress_starts_a_fresh_episode() -> None:
    state = CompletionGateControllerState()
    first = decide_completion_gate(state, _snapshot())
    record_completion_gate_decision(state, first)

    progressed = decide_completion_gate(
        state,
        _snapshot(
            stage="verification_not_attempted",
            material_edit_count=1,
            verification_generation=1,
        ),
    )

    assert progressed.kind == CompletionGateDecisionKind.NUDGE_AND_CONTINUE
    assert progressed.meaningful_progress_since_previous_rejection is True
    assert progressed.stagnant_attempt_count == 1


def test_identical_verification_failure_does_not_refresh_episode() -> None:
    state = CompletionGateControllerState()
    failure = {
        "pytest -q": (
            "FAILED /tmp/pytest-123/test_app.py::test_cli at 2026-06-18T10:11:12Z after 1.23s"
        )
    }
    equivalent_failure = {
        "pytest -q": (
            "FAILED /tmp/pytest-999/test_app.py::test_cli at 2026-06-18T10:12:13Z after 2.34s"
        )
    }
    first_snapshot = _snapshot(stage="verification_failed", failures=failure)
    second_snapshot = _snapshot(stage="verification_failed", failures=equivalent_failure)

    first = decide_completion_gate(state, first_snapshot)
    record_completion_gate_decision(state, first)
    second = decide_completion_gate(state, second_snapshot)

    assert first_snapshot.episode_id() == second_snapshot.episode_id()
    assert second.meaningful_progress_since_previous_rejection is False
    assert second.stagnant_attempt_count == 2


def test_distinct_failure_signature_without_material_change_does_not_count_as_progress() -> None:
    state = CompletionGateControllerState()
    first_snapshot = _snapshot(
        stage="verification_failed",
        failures={"pytest -q": "NameError: missing"},
    )
    second_snapshot = _snapshot(
        stage="verification_failed",
        failures={"pytest -q": "AssertionError: wrong"},
    )
    first = decide_completion_gate(
        state,
        first_snapshot,
    )
    record_completion_gate_decision(state, first)

    second = decide_completion_gate(
        state,
        second_snapshot,
    )

    assert first_snapshot.episode_id() != second_snapshot.episode_id()
    assert second.meaningful_progress_since_previous_rejection is False
    assert second.stagnant_attempt_count == 2


def test_read_only_or_unknown_tool_repetition_does_not_count_as_progress() -> None:
    state = CompletionGateControllerState()
    first = decide_completion_gate(state, _snapshot(stage="no_material_edits"))
    record_completion_gate_decision(state, first)

    second = decide_completion_gate(state, _snapshot(stage="no_material_edits"))

    assert second.meaningful_progress_since_previous_rejection is False
    assert second.stagnant_attempt_count == 2


def test_failure_signature_after_material_generation_change_counts_as_progress() -> None:
    state = CompletionGateControllerState()
    first = decide_completion_gate(
        state,
        _snapshot(
            stage="verification_failed",
            material_edit_count=1,
            verification_generation=1,
            failures={"pytest -q": "NameError: missing"},
        ),
    )
    record_completion_gate_decision(state, first)

    second = decide_completion_gate(
        state,
        _snapshot(
            stage="verification_failed",
            material_edit_count=1,
            verification_generation=2,
            failures={"pytest -q": "AssertionError: wrong"},
        ),
    )

    assert second.meaningful_progress_since_previous_rejection is True
    assert second.stagnant_attempt_count == 1


def test_required_output_creation_counts_as_progress() -> None:
    state = CompletionGateControllerState()
    missing_output = build_completion_gate_snapshot(
        stage="acceptance_unverified",
        problems=["acceptance_criteria_unverified"],
        material_edit_count=1,
        material_edit_tools={"fs_write"},
        touched_repo_paths={"result.txt"},
        verification_relevant_edit_generation=1,
        last_successful_verification_generation=None,
        expected_verification_commands=set(),
        covered_verification_commands=set(),
        missing_verification_commands=set(),
        failed_verification_command_snippets={},
        verification_coverage_stale=False,
        last_verification_passed=None,
        verification_expected=False,
        final_text="Created result.txt.",
        repo_tool_activity_observed=True,
        acceptance_status_counts={"UNVERIFIED": 1, "PASSED": 0},
        acceptance_problems=["acceptance_criteria_unverified"],
        acceptance_failure_summaries=["ac001: Required output path is missing: result.txt"],
    )
    created_output = build_completion_gate_snapshot(
        stage="verification_not_attempted",
        problems=["verification_not_attempted"],
        material_edit_count=1,
        material_edit_tools={"fs_write"},
        touched_repo_paths={"result.txt"},
        verification_relevant_edit_generation=1,
        last_successful_verification_generation=None,
        expected_verification_commands={"pytest -q"},
        covered_verification_commands=set(),
        missing_verification_commands={"pytest -q"},
        failed_verification_command_snippets={},
        verification_coverage_stale=False,
        last_verification_passed=None,
        verification_expected=True,
        final_text="Created result.txt.",
        repo_tool_activity_observed=True,
        acceptance_status_counts={"UNVERIFIED": 0, "PASSED": 1},
        acceptance_problems=[],
        acceptance_failure_summaries=[],
    )

    first = decide_completion_gate(state, missing_output)
    record_completion_gate_decision(state, first)
    second = decide_completion_gate(state, created_output)

    assert second.meaningful_progress_since_previous_rejection is True
    assert second.stagnant_attempt_count == 1


def test_budget_exhaustion_has_explicit_decision() -> None:
    state = CompletionGateControllerState()
    decision = decide_completion_gate(state, _snapshot(), budget_exhausted=True)

    assert decision.kind == CompletionGateDecisionKind.TERMINATE_BUDGET_EXHAUSTED
    assert decision.reason == "step_budget_exhausted"


@pytest.mark.parametrize(
    ("second_snapshot", "expected_progress", "expected_stagnant_attempts"),
    [
        (_snapshot(), False, 2),
        (_snapshot(material_edit_count=1, verification_generation=1), True, 1),
        (
            _snapshot(
                stage="verification_incomplete",
                material_edit_count=1,
                verification_generation=1,
                covered={"pytest -q"},
                missing={"ruff check ."},
            ),
            True,
            1,
        ),
        (
            _snapshot(
                stage="verification_failed",
                material_edit_count=1,
                verification_generation=1,
                failures={"pytest -q": "AssertionError: changed failure"},
            ),
            True,
            1,
        ),
    ],
)
def test_episode_transitions_are_progress_aware(
    second_snapshot,
    expected_progress: bool,
    expected_stagnant_attempts: int,
) -> None:
    state = CompletionGateControllerState()
    first = decide_completion_gate(state, _snapshot())
    record_completion_gate_decision(state, first)

    second = decide_completion_gate(state, second_snapshot)

    assert second.meaningful_progress_since_previous_rejection is expected_progress
    assert second.stagnant_attempt_count == expected_stagnant_attempts


def test_turn_execution_state_payload_keeps_existing_keys_and_adds_controller() -> None:
    state = TurnExecutionState(
        execution_requested=True,
        expected_verification_commands={"pytest -q"},
    )

    payload = state.as_payload()

    assert payload["execution_requested"] is True
    assert payload["completion_gate_repair_attempts"] == 0
    assert payload["completion_gate_no_material_edits_repair_attempts"] == 0
    assert payload["completion_gate_missing_verify_repair_attempts"] == 0
    assert payload["completion_gate_failed_verify_repair_attempts"] == 0
    assert payload["completion_gate_controller"]["total_rejected_finalizations"] == 0
    assert payload["completion_gate_episode_id"] == ""


def test_failure_signature_normalization_keeps_distinct_errors() -> None:
    first = normalize_completion_gate_failure_signature(
        "FAILED /tmp/run-123/test.py at 2026-06-18T10:11:12Z after 1.23s: NameError"
    )
    second = normalize_completion_gate_failure_signature(
        "FAILED /tmp/run-999/test.py at 2026-06-18T10:12:13Z after 2.34s: AssertionError"
    )

    assert "<temp-path>" in first
    assert "<timestamp>" in first
    assert "<duration>" in first
    assert first != second
