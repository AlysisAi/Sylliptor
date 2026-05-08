from __future__ import annotations

from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.execution_shared import resolve_managed_task_step_budget
from sylliptor_agent_cli.step_budget import StepBudgetRequest, resolve_step_budget


def test_resolve_step_budget_fixed_policy_uses_hard_cap() -> None:
    resolution = resolve_step_budget(
        StepBudgetRequest(
            kind="managed_task",
            policy="fixed",
            hard_cap=17,
            acceptance_criteria_count=9,
        )
    )

    assert resolution.resolved_max_steps == 17
    assert resolution.reason == "fixed_policy"
    assert resolution.override_applied is False


def test_resolve_step_budget_fixed_override_is_clamped_to_hard_cap() -> None:
    resolution = resolve_step_budget(
        StepBudgetRequest(
            kind="managed_task",
            policy="adaptive",
            hard_cap=12,
            fixed_override=99,
        )
    )

    assert resolution.resolved_max_steps == 12
    assert resolution.reason == "fixed_override"
    assert resolution.override_applied is True


def test_resolve_step_budget_fixed_override_wins_over_fixed_policy_for_managed_task() -> None:
    resolution = resolve_step_budget(
        StepBudgetRequest(
            kind="managed_task",
            policy="fixed",
            hard_cap=17,
            fixed_override=7,
        )
    )

    assert resolution.resolved_max_steps == 7
    assert resolution.reason == "fixed_override"
    assert resolution.override_applied is True


def test_resolve_step_budget_fixed_override_wins_over_fixed_policy_for_chat_turn() -> None:
    resolution = resolve_step_budget(
        StepBudgetRequest(
            kind="chat_turn",
            policy="fixed",
            hard_cap=17,
            fixed_override=7,
            route="repo",
        )
    )

    assert resolution.resolved_max_steps == 7
    assert resolution.reason == "fixed_override"
    assert resolution.override_applied is True


def test_resolve_step_budget_fixed_override_wins_over_fixed_policy_for_subagent() -> None:
    resolution = resolve_step_budget(
        StepBudgetRequest(
            kind="subagent",
            policy="fixed",
            hard_cap=16,
            fixed_override=7,
            subagent_name="reviewer",
            parent_turn_budget=9,
        )
    )

    assert resolution.resolved_max_steps == 7
    assert resolution.reason == "fixed_override"
    assert resolution.override_applied is True


def test_resolve_step_budget_fixed_override_still_clamps_when_it_wins_over_fixed_policy() -> None:
    resolution = resolve_step_budget(
        StepBudgetRequest(
            kind="managed_task",
            policy="fixed",
            hard_cap=5,
            fixed_override=7,
        )
    )

    assert resolution.resolved_max_steps == 5
    assert resolution.hard_cap == 5
    assert resolution.reason == "fixed_override"
    assert resolution.override_applied is True


def test_resolve_step_budget_chat_turn_scores_repo_execution_signals() -> None:
    resolution = resolve_step_budget(
        StepBudgetRequest(
            kind="chat_turn",
            policy="adaptive",
            hard_cap=50,
            route="repo",
            one_shot_execution=True,
            one_shot_turn_intent="execute",
            mode="auto",
            verification_enabled=True,
            subagents_enabled=True,
            explicit_path_count=10,
            image_count=5,
        )
    )

    assert resolution.resolved_max_steps == 49
    assert resolution.reason == "adaptive_chat_turn"
    assert resolution.signals_used["execution_reserve_steps"] == 8
    assert resolution.signals_used["explicit_path_count"] == 10
    assert resolution.signals_used["image_count"] == 5


def test_resolve_step_budget_chat_turn_low_hard_cap_clamps_to_hard_cap() -> None:
    resolution = resolve_step_budget(
        StepBudgetRequest(
            kind="chat_turn",
            policy="adaptive",
            hard_cap=4,
            route="repo",
        )
    )

    assert resolution.resolved_max_steps == 4
    assert resolution.hard_cap == 4


def test_resolve_step_budget_managed_task_scores_expected_signals() -> None:
    resolution = resolve_step_budget(
        StepBudgetRequest(
            kind="managed_task",
            policy="adaptive",
            hard_cap=100,
            verification_enabled=True,
            attempt_count=4,
            image_count=3,
            acceptance_criteria_count=9,
            estimated_files_count=10,
            write_scope_count=2,
            dependency_count=5,
            asset_count=4,
        )
    )

    assert resolution.resolved_max_steps == 98
    assert resolution.reason == "adaptive_managed_task"
    assert resolution.signals_used["verification_repair_reserve_steps"] == 14


def test_resolve_step_budget_managed_task_low_hard_cap_clamps_to_hard_cap() -> None:
    resolution = resolve_step_budget(
        StepBudgetRequest(
            kind="managed_task",
            policy="adaptive",
            hard_cap=10,
            verification_enabled=True,
            attempt_count=3,
            acceptance_criteria_count=4,
            estimated_files_count=6,
        )
    )

    assert resolution.resolved_max_steps == 10
    assert resolution.hard_cap == 10


def test_resolve_step_budget_conflict_resolution_scores_expected_signals() -> None:
    resolution = resolve_step_budget(
        StepBudgetRequest(
            kind="conflict_resolution",
            policy="adaptive",
            hard_cap=60,
            verification_enabled=True,
            attempt_count=3,
            conflict_file_count=5,
        )
    )

    assert resolution.resolved_max_steps == 45
    assert resolution.reason == "adaptive_conflict_resolution"


def test_resolve_step_budget_conflict_resolution_low_hard_cap_clamps_to_hard_cap() -> None:
    resolution = resolve_step_budget(
        StepBudgetRequest(
            kind="conflict_resolution",
            policy="adaptive",
            hard_cap=8,
            verification_enabled=True,
            attempt_count=2,
            conflict_file_count=3,
        )
    )

    assert resolution.resolved_max_steps == 8
    assert resolution.hard_cap == 8


def test_resolve_step_budget_subagent_profile_and_parent_cap_clamp() -> None:
    resolution = resolve_step_budget(
        StepBudgetRequest(
            kind="subagent",
            policy="adaptive",
            hard_cap=16,
            parent_turn_budget=11,
            subagent_name="explorer",
            explicit_path_count=9,
            mode="auto",
        )
    )

    assert resolution.hard_cap == 11
    assert resolution.resolved_max_steps == 11
    assert resolution.profile == "explorer"
    assert resolution.reason == "adaptive_subagent"


def test_resolve_step_budget_payload_serialization_is_structured() -> None:
    resolution = resolve_step_budget(
        StepBudgetRequest(
            kind="subagent",
            policy="adaptive",
            hard_cap=16,
            parent_turn_budget=12,
            subagent_name="reviewer",
            explicit_path_count=2,
            mode="readonly",
        )
    )

    payload = resolution.to_payload()

    assert payload["kind"] == "subagent"
    assert payload["policy"] == "adaptive"
    assert payload["resolved_max_steps"] == 12
    assert payload["profile"] == "reviewer"
    assert isinstance(payload["signals_used"], dict)


def test_execution_shared_resolve_managed_task_step_budget_uses_real_task_schema() -> None:
    cfg = AppConfig(step_budget_policy="adaptive", task_max_steps=100)
    plan = {
        "assets": [
            {"stored_path": "docs/api-contract.md"},
            {"stored_path": "design/mockup.png"},
            {"stored_path": "notes/irrelevant.txt"},
        ]
    }
    task = {
        "title": "Implement auth flow from api-contract and mockup",
        "description": "Use the api-contract and mockup to wire the login flow.",
        "acceptance_criteria": ["login succeeds", "errors render"],
        "estimated_files": ["src/auth.py", "src/ui.py"],
        "write_scope": ["src/**"],
        "dependencies": ["T01"],
    }

    resolution = resolve_managed_task_step_budget(
        cfg=cfg,
        plan=plan,
        task=task,
        verification_enabled=True,
        image_count=1,
    )

    assert resolution.resolved_max_steps == 50
    assert resolution.signals_used["acceptance_criteria_count"] == 2
    assert resolution.signals_used["estimated_files_count"] == 2
    assert resolution.signals_used["write_scope_count"] == 1
    assert resolution.signals_used["dependency_count"] == 1
    assert resolution.signals_used["asset_count"] == 2
    assert resolution.signals_used["verification_repair_reserve_steps"] == 14


def test_execution_shared_resolve_conflict_resolution_fixed_policy_uses_task_cap() -> None:
    cfg = AppConfig(step_budget_policy="fixed", task_max_steps=14)

    resolution = resolve_managed_task_step_budget(
        cfg=cfg,
        plan={},
        task={},
        kind="conflict_resolution",
        mode="auto",
        verification_enabled=True,
        attempt_count=3,
        conflict_file_count=5,
    )

    assert resolution.resolved_max_steps == 14
    assert resolution.hard_cap == 14
    assert resolution.reason == "fixed_policy"
    assert resolution.override_applied is False
