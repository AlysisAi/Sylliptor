from __future__ import annotations

from pathlib import Path

from sylliptor_agent_cli.plan_reconciliation import reconcile_plan_with_workspace
from sylliptor_agent_cli.repo_scan import scan_workspace
from sylliptor_agent_cli.workspace_context import resolve_workspace_context


def _workspace_context_payload(root: Path) -> dict[str, object]:
    return scan_workspace(context=resolve_workspace_context(root)).to_dict()


def test_reconciliation_fills_missing_estimated_files_from_strong_hints(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("hello\n", encoding="utf-8")
    workspace_context = _workspace_context_payload(root)
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Update README.md",
                "description": "Document the new workflow in README.md.",
                "acceptance_criteria": [],
                "estimated_files": [],
                "write_scope": [],
            }
        ]
    }

    result = reconcile_plan_with_workspace(
        plan,
        workspace_root=root,
        workspace_context=workspace_context,
    )

    assert result.changed is True
    assert plan["tasks"][0]["estimated_files"] == ["README.md"]
    assert plan["tasks"][0]["write_scope"] == ["README.md"]
    assert result.task_updates["T01"]["estimated_files"] == ["README.md"]
    assert any("inferred estimated_files from task text" in warning for warning in result.warnings)


def test_reconciliation_adds_explicit_task_paths_to_existing_wrong_scope(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "service.ts").write_text("export function render() {}\n", encoding="utf-8")
    (root / "package.json").write_text('{"scripts":{"test":"vitest"}}\n', encoding="utf-8")
    workspace_context = _workspace_context_payload(root)
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Fix RTL grid alignment in src/service.ts",
                "description": "The implementation change belongs in src/service.ts.",
                "acceptance_criteria": [],
                "estimated_files": ["package.json"],
                "write_scope": ["package.json"],
            }
        ]
    }

    result = reconcile_plan_with_workspace(
        plan,
        workspace_root=root,
        workspace_context=workspace_context,
    )

    assert result.changed is True
    assert plan["tasks"][0]["estimated_files"] == ["package.json", "src/service.ts"]
    assert plan["tasks"][0]["write_scope"] == ["package.json", "src/service.ts"]
    joined = " | ".join(result.warnings)
    assert "added explicit task path hints to estimated_files: src/service.ts" in joined
    assert "added explicit task path hints to write_scope: src/service.ts" in joined


def test_reconciliation_seeds_write_scope_and_drops_internal_paths(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    workspace_context = _workspace_context_payload(root)
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Update app",
                "description": "Touch src/app.py and internal state.",
                "acceptance_criteria": [],
                "estimated_files": [".sylliptor/runs/run_1/plan/plan.json", "src/app.py"],
                "write_scope": [],
            }
        ]
    }

    result = reconcile_plan_with_workspace(
        plan,
        workspace_root=root,
        workspace_context=workspace_context,
    )

    assert result.changed is True
    assert plan["tasks"][0]["estimated_files"] == ["src/app.py"]
    assert plan["tasks"][0]["write_scope"] == ["src/app.py"]
    joined = " | ".join(result.warnings)
    assert "dropped protected estimated_files entries" in joined
    assert "seeded write_scope from estimated_files" in joined


def test_reconciliation_warns_for_suspicious_missing_paths_without_crashing(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    workspace_context = _workspace_context_payload(root)
    plan = {
        "tasks": [
            {
                "id": "T99",
                "title": "Investigate missing path",
                "description": "Check missing/nested/file.py handling.",
                "acceptance_criteria": [],
                "estimated_files": ["missing/nested/file.py"],
                "write_scope": ["missing/nested/file.py"],
            }
        ]
    }

    result = reconcile_plan_with_workspace(
        plan,
        workspace_root=root,
        workspace_context=workspace_context,
    )

    assert result.changed is False
    assert plan["tasks"][0]["estimated_files"] == ["missing/nested/file.py"]
    assert plan["tasks"][0]["write_scope"] == ["missing/nested/file.py"]
    assert any("may be suspicious or missing" in warning for warning in result.warnings)


def test_reconcile_plan_warns_when_mutating_task_still_lacks_runnable_scope(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    workspace_context = _workspace_context_payload(root)
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Fix login bug",
                "description": "Update auth flow.",
                "acceptance_criteria": ["Login works."],
                "estimated_files": [],
                "write_scope": [],
            }
        ]
    }

    result = reconcile_plan_with_workspace(
        plan,
        workspace_root=root,
        workspace_context=workspace_context,
    )

    assert result.changed is False
    assert plan["tasks"][0]["estimated_files"] == []
    assert plan["tasks"][0]["write_scope"] == []
    assert any(
        "runnable or ambiguous task lacks runnable estimated_files/write_scope" in warning
        for warning in result.warnings
    )


def test_reconciliation_preserves_explicit_user_paths_even_when_missing(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    workspace_context = _workspace_context_payload(root)
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Update exact user-requested files",
                "description": (
                    "Touch src/planner/release_planner.py and tests/test_release_planner.py."
                ),
                "acceptance_criteria": [],
                "estimated_files": [
                    "src/planner/release_planner.py",
                    "tests/test_release_planner.py",
                ],
                "write_scope": [
                    "src/planner/release_planner.py",
                    "tests/test_release_planner.py",
                ],
            }
        ]
    }

    result = reconcile_plan_with_workspace(
        plan,
        workspace_root=root,
        workspace_context=workspace_context,
        user_text=(
            "Update src/planner/release_planner.py and tests/test_release_planner.py exactly."
        ),
    )

    assert result.changed is False
    assert plan["tasks"][0]["estimated_files"] == [
        "src/planner/release_planner.py",
        "tests/test_release_planner.py",
    ]
    assert plan["tasks"][0]["write_scope"] == [
        "src/planner/release_planner.py",
        "tests/test_release_planner.py",
    ]
    assert not any("may be suspicious or missing" in warning for warning in result.warnings)


def test_reconciliation_drops_forbidden_user_anchor_and_restores_implementation_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "todo_export.py").write_text("def export():\n    return []\n", encoding="utf-8")
    workspace_context = _workspace_context_payload(root)
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Fix todo_export.py to exclude USER_NOTES.md",
                "description": "Modify todo_export.py to exclude USER_NOTES.md from processing.",
                "acceptance_criteria": ["USER_NOTES.md is not modified."],
                "estimated_files": ["USER_NOTES.md"],
                "write_scope": ["USER_NOTES.md"],
            }
        ]
    }

    result = reconcile_plan_with_workspace(
        plan,
        workspace_root=root,
        workspace_context=workspace_context,
        user_text="Preserve the untracked USER_NOTES.md file.",
    )

    assert result.changed is True
    assert plan["tasks"][0]["estimated_files"] == ["todo_export.py"]
    assert plan["tasks"][0]["write_scope"] == ["todo_export.py"]
    joined = " | ".join(result.warnings)
    assert "dropped forbidden estimated_files entries: USER_NOTES.md" in joined
    assert "dropped forbidden write_scope entries: USER_NOTES.md" in joined


def test_reconciliation_drops_suspicious_off_request_paths_when_user_named_exact_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    workspace_context = _workspace_context_payload(root)
    plan = {
        "tasks": [
            {
                "id": "T02",
                "title": "Update release planner files",
                "description": (
                    "Edit src/planner/release_planner.py and tests/test_release_planner.py."
                ),
                "acceptance_criteria": [],
                "estimated_files": [
                    "src/planner/planner.py",
                    "tests/test_planner.py",
                ],
                "write_scope": [
                    "src/planner/planner.py",
                    "tests/test_planner.py",
                ],
            }
        ]
    }

    result = reconcile_plan_with_workspace(
        plan,
        workspace_root=root,
        workspace_context=workspace_context,
        user_text=(
            "Update src/planner/release_planner.py and tests/test_release_planner.py exactly."
        ),
    )

    assert result.changed is True
    assert plan["tasks"][0]["estimated_files"] == [
        "src/planner/release_planner.py",
        "tests/test_release_planner.py",
    ]
    assert plan["tasks"][0]["write_scope"] == [
        "src/planner/release_planner.py",
        "tests/test_release_planner.py",
    ]
    joined = " | ".join(result.warnings)
    assert (
        "dropped suspicious estimated_files entries not grounded in the latest user request"
        in joined
    )
    assert (
        "dropped suspicious write_scope entries not grounded in the latest user request" in joined
    )
    assert "restored grounded estimated_files from task text" in joined


def test_reconciliation_restores_explicit_paths_from_prior_user_turn_when_task_text_is_generic(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    workspace_context = _workspace_context_payload(root)
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Implement parser fix",
                "description": "Fix the parser regression.",
                "acceptance_criteria": ["Regression is covered by tests."],
                "estimated_files": ["src/planner/planner.py"],
                "write_scope": ["src/planner/planner.py"],
            }
        ]
    }

    result = reconcile_plan_with_workspace(
        plan,
        workspace_root=root,
        workspace_context=workspace_context,
        user_text="yes",
        transcript_tail=[
            {
                "role": "user",
                "content": (
                    "Update src/planner/release_planner.py and "
                    "tests/test_release_planner.py exactly."
                ),
            },
            {"role": "assistant", "content": "Need one confirmation."},
            {"role": "user", "content": "yes"},
        ],
    )

    assert result.changed is True
    assert plan["tasks"][0]["estimated_files"] == [
        "src/planner/release_planner.py",
        "tests/test_release_planner.py",
    ]
    assert plan["tasks"][0]["write_scope"] == [
        "src/planner/release_planner.py",
        "tests/test_release_planner.py",
    ]
    joined = " | ".join(result.warnings)
    assert "explicit user grounding" in joined


def test_reconciliation_does_not_use_obsolete_direction_path_as_anchor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "settings.py").write_text("# settings\n", encoding="utf-8")
    workspace_context = _workspace_context_payload(root)
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Use APP_TIMEOUT_SECONDS env var",
                "description": "Read APP_TIMEOUT_SECONDS in src/settings.py.",
                "acceptance_criteria": ["APP_TIMEOUT_SECONDS controls timeout."],
                "estimated_files": ["src/settings.py"],
                "write_scope": ["src/settings.py"],
                "status": "planned",
            }
        ]
    }

    result = reconcile_plan_with_workspace(
        plan,
        workspace_root=root,
        workspace_context=workspace_context,
        user_text=(
            "drop TOML from the plan entirely; use APP_TIMEOUT_SECONDS instead of settings.toml"
        ),
        target_task_ids={"T01"},
    )

    assert result.changed is False
    assert result.task_updates == {}
    assert plan["tasks"][0]["estimated_files"] == ["src/settings.py"]
    assert plan["tasks"][0]["write_scope"] == ["src/settings.py"]
    assert "settings.toml" not in " | ".join(result.warnings)


def test_reconciliation_maps_prior_turn_task_id_anchors_to_matching_tasks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    workspace_context = _workspace_context_payload(root)
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Planner task",
                "description": "Implement planner change.",
                "acceptance_criteria": [],
                "estimated_files": ["src/planner/planner.py"],
                "write_scope": ["src/planner/planner.py"],
            },
            {
                "id": "T02",
                "title": "Test task",
                "description": "Implement test change.",
                "acceptance_criteria": [],
                "estimated_files": ["tests/test_planner.py"],
                "write_scope": ["tests/test_planner.py"],
            },
        ]
    }

    result = reconcile_plan_with_workspace(
        plan,
        workspace_root=root,
        workspace_context=workspace_context,
        user_text="yes",
        transcript_tail=[
            {
                "role": "user",
                "content": (
                    "T01 update src/planner/release_planner.py\n"
                    "T02 update tests/test_release_planner.py"
                ),
            },
            {"role": "assistant", "content": "Need one confirmation."},
            {"role": "user", "content": "yes"},
        ],
    )

    assert result.changed is True
    assert plan["tasks"][0]["estimated_files"] == ["src/planner/release_planner.py"]
    assert plan["tasks"][0]["write_scope"] == ["src/planner/release_planner.py"]
    assert plan["tasks"][1]["estimated_files"] == ["tests/test_release_planner.py"]
    assert plan["tasks"][1]["write_scope"] == ["tests/test_release_planner.py"]


def test_reconciliation_can_limit_updates_to_selected_tasks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "app.py").write_text("print('hi')\n", encoding="utf-8")
    (root / "README.md").write_text("hello\n", encoding="utf-8")
    workspace_context = _workspace_context_payload(root)
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Document README.md",
                "description": "Update README.md for the shipped change.",
                "acceptance_criteria": [],
                "estimated_files": [],
                "write_scope": [],
            },
            {
                "id": "T02",
                "title": "Update src/app.py",
                "description": "Touch src/app.py for the follow-up.",
                "acceptance_criteria": [],
                "estimated_files": [],
                "write_scope": [],
            },
        ]
    }

    result = reconcile_plan_with_workspace(
        plan,
        workspace_root=root,
        workspace_context=workspace_context,
        target_task_ids={"T02"},
    )

    assert result.changed is True
    assert result.updated_task_ids == ["T02"]
    assert plan["tasks"][0]["estimated_files"] == []
    assert plan["tasks"][0]["write_scope"] == []
    assert plan["tasks"][1]["estimated_files"] == ["src/app.py"]
    assert plan["tasks"][1]["write_scope"] == ["src/app.py"]


def test_reconciliation_replaces_wrong_file_scope_with_symbol_definition(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "stats.py").write_text(
        "def median(values):\n    return values[0]\n",
        encoding="utf-8",
    )
    (root / "formatting.py").write_text(
        "def render_summary(stats):\n    return str(stats)\n",
        encoding="utf-8",
    )
    workspace_context = _workspace_context_payload(root)
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Add optional unit suffix to render_summary",
                "description": "Modify render_summary in stats.py.",
                "acceptance_criteria": ["Existing empty unit output remains byte-for-byte."],
                "estimated_files": ["stats.py"],
                "write_scope": ["stats.py"],
            }
        ]
    }

    result = reconcile_plan_with_workspace(
        plan,
        workspace_root=root,
        workspace_context=workspace_context,
    )

    assert result.changed is True
    assert plan["tasks"][0]["estimated_files"] == ["formatting.py"]
    assert plan["tasks"][0]["write_scope"] == ["formatting.py"]
    joined = " | ".join(result.warnings)
    assert "replaced symbol-mismatched estimated_files entries" in joined
    assert "replaced symbol-mismatched write_scope entries" in joined


def test_reconciliation_adds_unique_named_code_file_for_behavior_task_with_tests_only_scope(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "parser.py").write_text(
        "def normalize_tags(raw):\n    return []\n",
        encoding="utf-8",
    )
    (root / "tests").mkdir()
    (root / "tests" / "test_parser.py").write_text(
        "from parser import normalize_tags\n",
        encoding="utf-8",
    )
    workspace_context = _workspace_context_payload(root)
    plan = {
        "tasks": [
            {
                "id": "T01",
                "title": "Implement requested repository change",
                "description": (
                    "Don't do a broad parser rewrite. Make the smallest sane patch and "
                    "cover blank entries explicitly."
                ),
                "acceptance_criteria": [
                    "Add or update regression coverage for the requested behavior."
                ],
                "estimated_files": ["tests/**/*.py"],
                "write_scope": ["tests/**/*.py"],
            }
        ]
    }

    result = reconcile_plan_with_workspace(
        plan,
        workspace_root=root,
        workspace_context=workspace_context,
    )

    assert result.changed is True
    assert plan["tasks"][0]["estimated_files"] == ["tests/**/*.py", "parser.py"]
    assert plan["tasks"][0]["write_scope"] == ["tests/**/*.py", "parser.py"]
    joined = " | ".join(result.warnings)
    assert "added repository symbol definition path(s) to estimated_files: parser.py" in joined
    assert "added repository symbol definition path(s) to write_scope: parser.py" in joined
