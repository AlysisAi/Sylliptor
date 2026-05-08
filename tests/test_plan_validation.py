from __future__ import annotations

import pytest

from sylliptor_agent_cli.failure_category import FailureCategory
from sylliptor_agent_cli.plan_validation import (
    PlannerFailedError,
    find_plan_acceptance_issues,
    raise_for_execution_ready_plan,
    validate_plan,
    validate_plan_against_assets,
)
from sylliptor_agent_cli.task_readiness import find_execution_unready_mutating_tasks


def test_validate_plan_warns_unknown_deps_and_missing_fields() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Fix task 1",
                "dependencies": ["T99"],
                "acceptance_criteria": [],
                "estimated_files": [],
                "write_scope": [],
            },
            {
                "id": "T02",
                "title": "Task 2",
                "dependencies": ["T01"],
                "acceptance_criteria": ["ok"],
                "estimated_files": ["src/t2.py"],
            },
        ]
    }

    warnings = validate_plan(plan)
    assert "Task T01 has unknown dependency id: T99" in warnings
    assert "Task T01 is missing acceptance_criteria" in warnings
    assert any(
        "runnable or ambiguous task lacks runnable estimated_files/write_scope" in w
        for w in warnings
    )


def test_validate_plan_accepts_v1_and_v2_schema_versions() -> None:
    plan = {
        "schema_version": 1,
        "tasks": [
            {
                "id": "T01",
                "title": "Report",
                "description": "Read-only report.",
                "dependencies": [],
                "acceptance_criteria": ["Done."],
                "estimated_files": [],
                "write_scope": [],
            }
        ],
    }

    with pytest.warns(DeprecationWarning):
        assert validate_plan(plan) == []
    plan["schema_version"] = 2
    assert validate_plan(plan) == []


def test_validate_plan_accepts_valid_asset_briefing_shape() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Use asset",
                "description": "Read-only report.",
                "dependencies": [],
                "acceptance_criteria": ["Done."],
                "estimated_files": [],
                "write_scope": [],
                "asset_briefing": {
                    "primary": [
                        {
                            "asset_id": "ast_aaaaaaaa",
                            "rationale": "Primary evidence",
                            "expected_use": "Use in analysis",
                        }
                    ],
                    "may_need": [],
                },
            }
        ]
    }

    assert not any("asset_briefing" in warning for warning in validate_plan(plan))


def test_validate_plan_asset_briefing_shape_errors() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Use asset",
                "description": "Read-only report.",
                "dependencies": [],
                "acceptance_criteria": ["Done."],
                "estimated_files": [],
                "write_scope": [],
                "asset_briefing": {
                    "primary": [
                        {
                            "asset_id": "ast_aaaaaaaa",
                            "rationale": "Primary evidence",
                            "expected_use": "Use in analysis",
                        }
                    ],
                    "may_need": [
                        {
                            "asset_id": "ast_aaaaaaaa",
                            "rationale": "Duplicate",
                            "expected_use": "Use again",
                        }
                    ],
                },
            }
        ]
    }

    warnings = validate_plan(plan)

    assert any("same asset in primary and may_need" in warning for warning in warnings)


def test_validate_plan_against_assets_reports_missing_and_deleted_refs() -> None:
    class Record:
        def __init__(self, asset_id: str, deleted_at: str | None) -> None:
            self.id = asset_id
            self.deleted_at = deleted_at

    plan = {
        "tasks": [
            {
                "id": "T01",
                "asset_briefing": {
                    "primary": [
                        {
                            "asset_id": "ast_deleted",
                            "rationale": "Deleted",
                            "expected_use": "Use it",
                        }
                    ],
                    "may_need": [
                        {
                            "asset_id": "ast_missing",
                            "rationale": "Missing",
                            "expected_use": "Use it",
                        }
                    ],
                },
            }
        ]
    }

    warnings = validate_plan_against_assets(
        plan,
        {"ast_deleted": Record("ast_deleted", "2026-05-03T00:00:00+00:00")},
    )

    assert "Task T01 references deleted asset id: ast_deleted" in warnings
    assert "Task T01 references missing asset id: ast_missing" in warnings


def test_validate_plan_against_assets_reports_malformed_briefing_without_raising() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "asset_briefing": {"primary": {"asset_id": "ast_bad"}},
            }
        ]
    }

    warnings = validate_plan_against_assets(plan, {})

    assert any("Task T01 has invalid asset_briefing" in warning for warning in warnings)


def test_validate_plan_allows_non_mutating_scope_free_task() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Investigate login issue",
                "description": "Read-only analysis only; report findings.",
                "dependencies": [],
                "acceptance_criteria": ["Findings documented."],
                "estimated_files": [],
                "write_scope": [],
            }
        ]
    }

    warnings = validate_plan(plan)
    assert not any("runnable estimated_files/write_scope" in warning for warning in warnings)


def test_validate_plan_allows_report_only_task_with_likely_runnable_noun() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Compare build options",
                "description": "Report findings only.",
                "dependencies": [],
                "acceptance_criteria": ["Findings documented."],
                "estimated_files": [],
                "write_scope": [],
                "status": "planned",
            }
        ]
    }

    warnings = validate_plan(plan)
    assert not any("runnable estimated_files/write_scope" in warning for warning in warnings)


def test_validate_plan_reports_mutating_task_with_invalid_scope_entries() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Fix login bug",
                "description": "Update auth flow.",
                "dependencies": [],
                "acceptance_criteria": ["Login works."],
                "estimated_files": ["fix login bug"],
                "write_scope": [".sylliptor/current_run.json"],
            }
        ]
    }

    warnings = validate_plan(plan)

    assert any(
        "runnable or ambiguous task lacks runnable estimated_files/write_scope" in w
        for w in warnings
    )


def test_missing_write_scope_raises_planner_failed_before_execution() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Fix calculator division",
                "description": "Update implementation code.",
                "dependencies": [],
                "acceptance_criteria": ["Division by zero raises ValueError."],
                "estimated_files": ["fix calculator division"],
                "write_scope": [],
                "status": "planned",
            }
        ]
    }

    with pytest.raises(PlannerFailedError) as exc_info:
        raise_for_execution_ready_plan(plan)

    assert exc_info.value.failure_category == FailureCategory.PLANNER_FAILED
    assert "Execution blocked:" in str(exc_info.value)
    assert "R4" in str(exc_info.value)
    assert "write_scope" in str(exc_info.value)


def test_execution_readiness_rejects_no_mutating_tasks_as_planner_failed() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Investigate login issue",
                "description": "Read-only analysis only; report findings.",
                "dependencies": [],
                "acceptance_criteria": ["Findings documented."],
                "estimated_files": [],
                "write_scope": [],
                "status": "planned",
            }
        ]
    }

    with pytest.raises(PlannerFailedError) as exc_info:
        raise_for_execution_ready_plan(plan)

    assert exc_info.value.failure_category == FailureCategory.PLANNER_FAILED
    assert "R1" in str(exc_info.value)
    assert "no mutating execution candidates" in str(exc_info.value)


def test_execution_readiness_rejects_internal_only_write_scope_fixture() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Fix calculator seed behavior",
                "description": "Update implementation code.",
                "dependencies": [],
                "acceptance_criteria": ["Calculator seed behavior is implemented."],
                "estimated_files": [".sylliptor/something.json"],
                "write_scope": [".sylliptor/something.json"],
                "status": "planned",
            }
        ]
    }

    with pytest.raises(PlannerFailedError) as exc_info:
        raise_for_execution_ready_plan(plan)

    assert exc_info.value.failure_category == FailureCategory.PLANNER_FAILED
    assert "R2" in str(exc_info.value)
    assert "all write_scope paths under .sylliptor/" in str(exc_info.value)


def test_execution_readiness_rejects_code_task_with_docs_only_write_scope() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Fix calculator behavior",
                "description": "Update the implementation code.",
                "dependencies": [],
                "acceptance_criteria": ["Calculator behavior is fixed."],
                "estimated_files": ["README.md"],
                "write_scope": ["README.md"],
                "status": "planned",
            }
        ]
    }

    with pytest.raises(PlannerFailedError) as exc_info:
        raise_for_execution_ready_plan(plan)

    assert exc_info.value.failure_category == FailureCategory.PLANNER_FAILED
    assert "R3" in str(exc_info.value)
    assert "write_scope is README/docs only" in str(exc_info.value)


def test_execution_readiness_allows_explicit_docs_only_mutation() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Document calculator behavior",
                "description": "Update README documentation.",
                "dependencies": [],
                "acceptance_criteria": ["Documentation is updated."],
                "estimated_files": ["README.md"],
                "write_scope": ["README.md"],
                "status": "planned",
            }
        ]
    }

    raise_for_execution_ready_plan(plan)


def test_execution_readiness_does_not_treat_substring_test_as_support_marker() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Fix latest calculator behavior",
                "description": "Update implementation code.",
                "dependencies": [],
                "acceptance_criteria": ["Latest behavior is fixed."],
                "estimated_files": ["README.md"],
                "write_scope": ["README.md"],
                "status": "planned",
            }
        ]
    }

    with pytest.raises(PlannerFailedError) as exc_info:
        raise_for_execution_ready_plan(plan)

    assert "R3" in str(exc_info.value)


def test_execution_readiness_allows_support_tasks_with_primary_implementation_scope() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Implement calculator behavior",
                "description": "Update implementation code.",
                "dependencies": [],
                "acceptance_criteria": ["Calculator behavior is fixed."],
                "estimated_files": ["calc.py"],
                "write_scope": ["calc.py"],
                "status": "planned",
            },
            {
                "id": "T02",
                "title": "Add pytest coverage",
                "description": "Add regression tests for calculator behavior.",
                "dependencies": ["T01"],
                "acceptance_criteria": ["Tests cover the fixed behavior."],
                "estimated_files": ["test_calc.py"],
                "write_scope": ["test_calc.py"],
                "status": "planned",
            },
            {
                "id": "T03",
                "title": "Update README examples",
                "description": "Document the new calculator behavior.",
                "dependencies": ["T01"],
                "acceptance_criteria": ["README examples match behavior."],
                "estimated_files": ["README.md"],
                "write_scope": ["README.md"],
                "status": "planned",
            },
        ]
    }

    raise_for_execution_ready_plan(plan)


def test_execution_readiness_allows_described_support_task_after_implementation_task() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Implement percent style in formatting.py",
                "description": "Add the style dispatch implementation.",
                "dependencies": [],
                "acceptance_criteria": ["Percent style renders expected output."],
                "estimated_files": ["formatting.py"],
                "write_scope": ["formatting.py"],
                "status": "planned",
            },
            {
                "id": "T02",
                "title": "Add focused percent-style tests in dedicated test file",
                "description": "Create tests/test_percent.py with focused pytest cases.",
                "dependencies": ["T01"],
                "acceptance_criteria": ["pytest -q tests/test_percent.py passes."],
                "estimated_files": ["tests/test_percent.py"],
                "write_scope": ["tests/test_percent.py"],
                "status": "planned",
            },
        ]
    }

    raise_for_execution_ready_plan(plan)


def test_execution_readiness_rejects_behavior_task_with_tests_only_scope() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Implement requested repository change",
                "description": (
                    "Make the smallest sane parser patch and cover blank entries explicitly."
                ),
                "dependencies": [],
                "acceptance_criteria": [
                    "Add or update regression coverage for the requested behavior."
                ],
                "estimated_files": ["tests/test_parser.py"],
                "write_scope": ["tests/test_parser.py"],
                "status": "planned",
            }
        ]
    }

    with pytest.raises(PlannerFailedError) as exc_info:
        raise_for_execution_ready_plan(plan)

    assert "R3" in str(exc_info.value)
    assert "write_scope has no code implementation paths" in str(exc_info.value)


def test_execution_readiness_treats_packaging_manifest_as_primary_scope() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Add console script entry point",
                "description": "Wire the package CLI entry point through pyproject metadata.",
                "dependencies": [],
                "acceptance_criteria": ["Console script is installable and verified."],
                "estimated_files": ["pyproject.toml", "tests/test_entry_points.py"],
                "write_scope": ["pyproject.toml", "tests/test_entry_points.py"],
                "status": "planned",
            }
        ]
    }

    raise_for_execution_ready_plan(plan)


def test_execution_readiness_rejects_missing_executor_required_field() -> None:
    plan = {
        "tasks": [
            {
                "title": "Fix calculator division",
                "description": "Update implementation code.",
                "dependencies": [],
                "acceptance_criteria": ["Division by zero raises ValueError."],
                "estimated_files": ["src/calc.py"],
                "write_scope": ["src/calc.py"],
                "status": "planned",
            }
        ]
    }

    with pytest.raises(PlannerFailedError) as exc_info:
        raise_for_execution_ready_plan(plan)

    assert exc_info.value.failure_category == FailureCategory.PLANNER_FAILED
    assert "R4" in str(exc_info.value)
    assert "missing field: id" in str(exc_info.value)


def test_validate_plan_flags_ambiguous_scope_empty_execution_candidate() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Task A",
                "description": "Manual planning chat task: Task A",
                "dependencies": [],
                "acceptance_criteria": ["Task is complete."],
                "estimated_files": [],
                "write_scope": [],
                "status": "planned",
            }
        ]
    }

    warnings = validate_plan(plan)

    assert any(
        "runnable or ambiguous task lacks runnable estimated_files/write_scope" in w
        for w in warnings
    )


def test_execution_readiness_blocks_ambiguous_scope_empty_execution_candidate() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Do the work",
                "description": "Manual planning chat task: Do the work",
                "dependencies": [],
                "acceptance_criteria": ["Work is done."],
                "estimated_files": [],
                "write_scope": [],
                "status": "planned",
            }
        ]
    }

    issues = find_execution_unready_mutating_tasks(plan)

    assert len(issues) == 1
    assert issues[0].task_id == "T01"
    assert (
        "runnable or ambiguous task lacks runnable estimated_files/write_scope" in issues[0].warning
    )


def test_plan_acceptance_rejects_scope_missing_explicit_path_hint() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Fix RTL grid alignment in src/service.ts",
                "description": "The implementation change belongs in src/service.ts.",
                "dependencies": [],
                "acceptance_criteria": ["The calendar starts on the correct weekday."],
                "estimated_files": ["package.json"],
                "write_scope": ["package.json"],
                "status": "planned",
            }
        ]
    }

    issues = find_plan_acceptance_issues(plan)

    assert any(
        issue.task_id == "T01"
        and "scope omits explicit task path hints: src/service.ts" in issue.observed
        for issue in issues
    )
    with pytest.raises(PlannerFailedError):
        raise_for_execution_ready_plan(plan)


def test_plan_validation_accepts_superseded_and_invalidated_task_statuses() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Implement obsolete TOML settings",
                "description": "Update settings.toml behavior.",
                "dependencies": [],
                "acceptance_criteria": [],
                "estimated_files": [],
                "write_scope": [],
                "status": "superseded",
            },
            {
                "id": "T02",
                "title": "Implement obsolete CLI flag settings",
                "description": "Update CLI flag behavior.",
                "dependencies": [],
                "acceptance_criteria": [],
                "estimated_files": [],
                "write_scope": [],
                "status": "invalidated",
            },
        ]
    }

    warnings = validate_plan(plan)

    assert warnings == []
    assert find_execution_unready_mutating_tasks(plan) == []


def test_validate_plan_detects_cycle_with_path() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Task 1",
                "dependencies": ["T02"],
                "acceptance_criteria": ["a"],
                "estimated_files": ["src/a.py"],
            },
            {
                "id": "T02",
                "title": "Task 2",
                "dependencies": ["T03"],
                "acceptance_criteria": ["b"],
                "estimated_files": ["src/b.py"],
            },
            {
                "id": "T03",
                "title": "Task 3",
                "dependencies": ["T01"],
                "acceptance_criteria": ["c"],
                "estimated_files": ["src/c.py"],
            },
        ]
    }

    warnings = validate_plan(plan)
    cycle_warnings = [w for w in warnings if w.startswith("Circular dependency detected: ")]
    assert len(cycle_warnings) == 1
    assert "T01" in cycle_warnings[0]
    assert "T02" in cycle_warnings[0]
    assert "T03" in cycle_warnings[0]


def test_validate_plan_warns_for_invalid_task_mcp_scope() -> None:
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Task 1",
                "dependencies": [],
                "acceptance_criteria": ["ok"],
                "estimated_files": ["src/task.py"],
                "mcp_scope": {
                    "allow_resources": "yes",
                    "allowed_tools": [{"server_id": "", "tool_name": "create_issue"}],
                },
            }
        ]
    }

    warnings = validate_plan(plan)

    assert any("mcp_scope.allow_resources" in warning for warning in warnings)
    assert any("server_id cannot be empty" in warning for warning in warnings)
