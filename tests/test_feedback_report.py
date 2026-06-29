from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from rich.console import Console
from typer.testing import CliRunner

from sylliptor_agent_cli import cli as cli_mod
from sylliptor_agent_cli.cli import app as sylliptor_app
from sylliptor_agent_cli.config import AppConfig, ConfigError
from sylliptor_agent_cli.failure_category import FailureCategory
from sylliptor_agent_cli.feedback_report import (
    FeedbackBundleResult,
    FeedbackGithubIssueResult,
    FeedbackReportError,
    create_feedback_bundle,
    create_feedback_github_issue_draft,
)
from sylliptor_agent_cli.forge import attach_asset, create_plan_run
from sylliptor_agent_cli.session_store import SessionStore
from sylliptor_agent_cli.web_research import build_web_research_artifact_from_events


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "SYLLIPTOR_CONFIG_DIR": os.fspath(tmp_path / "cfg"),
        "SYLLIPTOR_DATA_DIR": os.fspath(tmp_path / "data"),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_retained_session(
    *,
    sessions_dir: Path,
    session_id: str,
    extra_events: list[dict[str, Any]] | None = None,
) -> tuple[Path, Path]:
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path = sessions_dir / f"{session_id}.jsonl"
    events = [
        {
            "type": "session_start",
            "session_id": session_id,
            "payload": {
                "system_prompt_sha256": "abc123",
                "verification_selection_source": "config.verify_commands_fallback",
                "verification_selection_reason": (
                    "using the configured generic fallback because repo scan found no repo-native command"
                ),
                "verification_contract_type": "generic_fallback",
                "verification_authoritative": False,
            },
        },
        {
            "type": "tool_call",
            "session_id": session_id,
            "payload": {"name": "fs_read", "step": 1, "arguments": {"path": "README.md"}},
        },
        {
            "type": "tool_call",
            "session_id": session_id,
            "payload": {"name": "shell_run", "step": 2, "arguments": {"cmd": "pytest -q"}},
        },
        {
            "type": "llm_usage",
            "session_id": session_id,
            "payload": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost_usd": 0.01,
            },
        },
        {
            "type": "verify_run",
            "session_id": session_id,
            "payload": {
                "all_passed": False,
                "verification_authoritative": False,
            },
        },
    ]
    if extra_events:
        events.extend(extra_events)
    log_path.write_text(
        "".join(json.dumps(event, ensure_ascii=True) + "\n" for event in events),
        encoding="utf-8",
    )
    artifact_root = sessions_dir / session_id
    (artifact_root / "tool_outputs").mkdir(parents=True, exist_ok=True)
    (artifact_root / "tool_outputs" / "latest.txt").write_text("artifact\n", encoding="utf-8")
    return log_path, artifact_root


def test_create_feedback_bundle_collects_retained_session_and_current_run(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")

    sessions_dir = tmp_path / "sessions"
    log_path, artifact_root = _write_retained_session(
        sessions_dir=sessions_dir,
        session_id="sess_demo",
    )
    cfg = AppConfig(session_log_dir=os.fspath(sessions_dir))

    paths = create_plan_run(workspace)
    paths.execution_reports_dir.mkdir(parents=True, exist_ok=True)
    paths.execution_logs_dir.mkdir(parents=True, exist_ok=True)
    paths.execution_sessions_dir.mkdir(parents=True, exist_ok=True)
    paths.execution_verify_dir.mkdir(parents=True, exist_ok=True)
    (paths.execution_dir / "trace").mkdir(parents=True, exist_ok=True)
    (paths.execution_dir / "worker_results").mkdir(parents=True, exist_ok=True)
    (paths.execution_integration_dir / "batch_001").mkdir(parents=True, exist_ok=True)
    (paths.knowledge_selected_dir).mkdir(parents=True, exist_ok=True)
    (paths.execution_reports_dir / "T01.md").write_text("report\n", encoding="utf-8")
    (paths.execution_logs_dir / "run.log").write_text(
        f"error at {workspace / 'src' / 'app.py'}\n",
        encoding="utf-8",
    )
    (paths.execution_sessions_dir / "worker_01").mkdir(parents=True, exist_ok=True)
    (paths.execution_sessions_dir / "worker_01" / "artifact.txt").write_text(
        "session artifact\n",
        encoding="utf-8",
    )
    (paths.execution_verify_dir / "verify.txt").write_text("verify\n", encoding="utf-8")
    (paths.execution_dir / "trace" / "swarm_trace.jsonl").write_text("{}\n", encoding="utf-8")
    category_payloads = {
        "implementation": FailureCategory.IMPLEMENTATION_FAILED.value,
        "provider": FailureCategory.PROVIDER_THROTTLED.value,
        "planner": FailureCategory.PLANNER_FAILED.value,
    }
    for name, category in category_payloads.items():
        (paths.execution_dir / "worker_results" / f"{name}.json").write_text(
            json.dumps(
                {
                    "task_id": name,
                    "success": False,
                    "verify_failed": False,
                    "failure_reason": "failed",
                    "result_kind": "failure",
                    "failure_category": category,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    (paths.execution_integration_dir / "batch_001" / "result.json").write_text(
        json.dumps(
            {
                "passed": False,
                "summary": "integration verify failed",
                "failure_category": FailureCategory.VERIFICATION_FAILED.value,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (paths.execution_integration_dir / "batch_001" / "infra.json").write_text(
        json.dumps(
            {
                "passed": False,
                "summary": "docker socket missing",
                "failure_category": FailureCategory.INFRA_UNAVAILABLE.value,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (paths.knowledge_selected_dir / "fact.md").write_text("fact\n", encoding="utf-8")
    (paths.run_dir / "worktrees" / "task_01").mkdir(parents=True, exist_ok=True)
    (paths.run_dir / "worktrees" / "task_01" / "ignored.txt").write_text(
        "ignore me\n",
        encoding="utf-8",
    )

    result = create_feedback_bundle(
        workspace_root=workspace,
        cfg=cfg,
        latest=True,
        feedback_text="weird output in beta",
    )

    assert result.output_root == workspace / "sylliptor-feedback"
    assert result.bundle_dir.exists()
    assert result.zip_path.exists()
    assert result.session_id == "sess_demo"
    assert result.run_id == paths.run_id

    assert (result.bundle_dir / "manifest.json").exists()
    assert (result.bundle_dir / "feedback.md").exists()
    assert (result.bundle_dir / "summary.md").exists()
    assert (result.bundle_dir / "session_score.json").exists()
    assert _read_jsonl(result.bundle_dir / "session" / "log.jsonl") == _read_jsonl(log_path)
    assert (result.bundle_dir / "session" / "artifacts" / "tool_outputs" / "latest.txt").read_text(
        encoding="utf-8"
    ) == (artifact_root / "tool_outputs" / "latest.txt").read_text(encoding="utf-8")
    assert (result.bundle_dir / "forge" / "current_run.json").exists()
    assert (result.bundle_dir / "forge" / "run" / "plan" / "PLAN.md").exists()
    assert (result.bundle_dir / "forge" / "run" / "execution" / "reports" / "T01.md").exists()
    assert (result.bundle_dir / "forge" / "run" / "execution" / "logs" / "run.log").exists()
    assert (
        result.bundle_dir
        / "forge"
        / "run"
        / "execution"
        / "sessions"
        / "worker_01"
        / "artifact.txt"
    ).exists()
    assert (
        result.bundle_dir / "forge" / "run" / "execution" / "trace" / "swarm_trace.jsonl"
    ).exists()
    assert (result.bundle_dir / "forge" / "run" / "knowledge" / "selected" / "fact.md").exists()
    assert not (result.bundle_dir / "forge" / "run" / "worktrees").exists()

    manifest = json.loads((result.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    copied_pointer = json.loads(
        (result.bundle_dir / "forge" / "current_run.json").read_text(encoding="utf-8")
    )
    assert manifest["session"]["included_log"] is True
    assert manifest["session"]["included_artifacts"] is True
    assert manifest["session"]["included_snapshot"] is False
    assert manifest["run"]["included_execution"] is True
    assert manifest["run"]["included_knowledge"] is True
    assert manifest["run"]["failure_category_counts"] == {
        FailureCategory.INFRA_UNAVAILABLE.value: 1,
        FailureCategory.PROVIDER_UNAVAILABLE.value: 0,
        FailureCategory.PROVIDER_THROTTLED.value: 1,
        FailureCategory.PLANNER_FAILED.value: 1,
        FailureCategory.IMPLEMENTATION_FAILED.value: 1,
        FailureCategory.VERIFICATION_FAILED.value: 1,
    }
    assert manifest["run"]["excluded_worktrees"] is True
    assert manifest["workspace_root"] == "<workspace-root>"
    assert (
        manifest["bundle_dir"]
        == result.bundle_dir.resolve().relative_to(workspace.resolve()).as_posix()
    )
    assert (
        manifest["zip_path"]
        == result.zip_path.resolve().relative_to(workspace.resolve()).as_posix()
    )
    assert manifest["session"]["log_path"] == "[redacted host path: sess_demo.jsonl]"
    assert manifest["session"]["artifact_root"] == "[redacted host path: sess_demo]"
    assert (
        manifest["run"]["run_dir"]
        == paths.run_dir.resolve().relative_to(workspace.resolve()).as_posix()
    )
    assert manifest["run"]["current_run_pointer_path"] == ".sylliptor/current_run.json"
    assert copied_pointer["workspace_root"] == "<workspace-root>"
    assert copied_pointer["focus_path"] == "."

    session_score = json.loads(
        (result.bundle_dir / "session_score.json").read_text(encoding="utf-8")
    )
    assert session_score["session_id"] == "sess_demo"
    assert session_score["test_shell_runs"] == 1
    assert session_score["path"] == "[redacted host path: sess_demo.jsonl]"
    assert session_score["verification_selection_source"] == "config.verify_commands_fallback"
    assert session_score["verification_contract_type"] == "generic_fallback"
    assert session_score["verification_authoritative"] is False
    assert session_score["last_verification_failure_kind"] == "non_authoritative"

    summary_md = (result.bundle_dir / "summary.md").read_text(encoding="utf-8")
    workspace_summary_md = (
        result.bundle_dir / "forge" / "run" / "plan" / "context" / "workspace_summary.md"
    ).read_text(encoding="utf-8")
    exported_run_log = (
        result.bundle_dir / "forge" / "run" / "execution" / "logs" / "run.log"
    ).read_text(encoding="utf-8")
    assert "- Output Root: `sylliptor-feedback`" in summary_md
    assert "- Included Session Log: `session/log.jsonl`" in summary_md
    assert "- Included Current Run Pointer: `forge/current_run.json`" in summary_md
    assert "- Verification Selection Source: `config.verify_commands_fallback`" in summary_md
    assert "- Verification Contract Type: `generic_fallback`" in summary_md
    assert "- Verification Authoritative: no" in summary_md
    assert "- Verification Failure Kind: `non_authoritative`" in summary_md
    assert "- Workspace Root: `.`" in workspace_summary_md
    assert "- Git Root: `(none)`" in workspace_summary_md
    assert "error at src/app.py" in exported_run_log
    assert os.fspath(tmp_path) not in json.dumps(manifest, sort_keys=True)
    assert os.fspath(tmp_path) not in json.dumps(copied_pointer, sort_keys=True)
    assert os.fspath(tmp_path) not in summary_md
    assert os.fspath(tmp_path) not in workspace_summary_md
    assert os.fspath(tmp_path) not in exported_run_log

    with zipfile.ZipFile(result.zip_path) as zf:
        names = set(zf.namelist())
    assert "manifest.json" in names
    assert "feedback.md" in names
    assert "summary.md" in names
    assert "session/log.jsonl" in names
    assert "session/artifacts/tool_outputs/latest.txt" in names
    assert "session/web_research_sources.json" not in names
    assert "forge/current_run.json" in names
    assert "forge/run/execution/trace/swarm_trace.jsonl" in names
    assert "forge/run/worktrees/task_01/ignored.txt" not in names


def test_create_feedback_bundle_includes_canonical_web_research_artifact(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_dir = tmp_path / "sessions"
    _write_retained_session(
        sessions_dir=sessions_dir,
        session_id="sess_web",
        extra_events=[
            {
                "type": "user_message",
                "session_id": "sess_web",
                "payload": {
                    "content": "Please inspect https://docs.example.com/start and the redirect chain."
                },
            },
            {
                "type": "tool_call",
                "session_id": "sess_web",
                "payload": {
                    "name": "web_search",
                    "step": 3,
                    "arguments": {"query": "docs example", "external_web_access": True},
                },
            },
            {
                "type": "tool_result",
                "session_id": "sess_web",
                "payload": {
                    "name": "web_search",
                    "step": 3,
                    "result": {
                        "query": "docs example",
                        "backend": "openai_responses",
                        "allowed_domains": ["docs.example.com"],
                        "sources": [
                            {
                                "title": "Start",
                                "url": "https://docs.example.com/start",
                                "snippet": "alpha",
                            },
                            {
                                "title": "Start duplicate",
                                "url": "https://docs.example.com/start",
                                "snippet": "duplicate",
                            },
                        ],
                    },
                },
            },
            {
                "type": "tool_call",
                "session_id": "sess_web",
                "payload": {
                    "name": "web_search",
                    "step": 4,
                    "arguments": {"query": "docs   example", "external_web_access": True},
                },
            },
            {
                "type": "tool_result",
                "session_id": "sess_web",
                "payload": {
                    "name": "web_search",
                    "step": 4,
                    "result": {
                        "query": "docs example",
                        "backend": "openai_responses",
                        "sources": [
                            {
                                "title": "Guide",
                                "url": "https://docs.example.com/guide",
                                "snippet": "beta",
                            }
                        ],
                    },
                },
            },
            {
                "type": "tool_call",
                "session_id": "sess_web",
                "payload": {
                    "name": "web_fetch",
                    "step": 5,
                    "arguments": {"url": "https://docs.example.com/start"},
                },
            },
            {
                "type": "tool_result",
                "session_id": "sess_web",
                "payload": {
                    "name": "web_fetch",
                    "step": 5,
                    "result": {
                        "url": "https://docs.example.com/start",
                        "final_url": "https://docs.example.com/final",
                        "status_code": 200,
                        "content_type": "text/html",
                        "title": "Final",
                        "backend": "httpx",
                    },
                },
            },
            {
                "type": "tool_call",
                "session_id": "sess_web",
                "payload": {
                    "name": "web_fetch",
                    "step": 6,
                    "arguments": {"url": "https://docs.example.com/start"},
                },
            },
            {
                "type": "tool_result",
                "session_id": "sess_web",
                "payload": {
                    "name": "web_fetch",
                    "step": 6,
                    "result": {
                        "url": "https://docs.example.com/start",
                        "final_url": "https://docs.example.com/final",
                        "status_code": 200,
                        "content_type": "text/html",
                        "title": "Final",
                        "backend": "httpx",
                    },
                },
            },
        ],
    )
    cfg = AppConfig(session_log_dir=os.fspath(sessions_dir))

    result = create_feedback_bundle(
        workspace_root=workspace,
        cfg=cfg,
        latest=True,
    )

    manifest = json.loads((result.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    exported = json.loads(
        (result.bundle_dir / "session" / "web_research_sources.json").read_text(encoding="utf-8")
    )
    session_score = json.loads(
        (result.bundle_dir / "session_score.json").read_text(encoding="utf-8")
    )
    summary_md = (result.bundle_dir / "summary.md").read_text(encoding="utf-8")

    assert manifest["session"]["included_web_research_artifact"] is True
    assert exported["schema_version"] == 2
    assert exported["deduped_normalized_queries"] == ["docs example"]
    assert exported["deduped_normalized_user_urls"] == ["https://docs.example.com/start"]
    assert exported["deduped_normalized_search_source_urls"] == [
        "https://docs.example.com/start",
        "https://docs.example.com/guide",
    ]
    assert exported["deduped_normalized_fetch_urls"] == ["https://docs.example.com/start"]
    assert exported["deduped_normalized_final_fetch_urls"] == ["https://docs.example.com/final"]
    assert exported["fetches"][0]["requested_url"] == "https://docs.example.com/start"
    assert exported["fetches"][0]["final_url"] == "https://docs.example.com/final"
    assert exported["fetches"][0]["provenance_classification"] == "user_provided"
    assert exported["searches"][0]["backend"] == "openai_responses"
    assert exported["searches"][0]["returned_sources"][0]["domain"] == "docs.example.com"
    assert session_score["web_search_calls"] == 2
    assert session_score["web_fetch_calls"] == 2
    assert session_score["duplicate_web_queries"] == 1
    assert session_score["duplicate_web_fetches"] == 1
    assert session_score["total_web_sources_returned"] == 3
    assert session_score["total_web_sources_fetched"] == 2
    assert "- Included Web Research Artifact: `session/web_research_sources.json`" in summary_md

    with zipfile.ZipFile(result.zip_path) as zf:
        assert "session/web_research_sources.json" in set(zf.namelist())


def test_create_feedback_bundle_merges_newer_web_artifact_ahead_of_log(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_dir = tmp_path / "sessions"
    session_id = "sess_web_artifact_ahead"
    log_path, artifact_root = _write_retained_session(
        sessions_dir=sessions_dir,
        session_id=session_id,
        extra_events=[
            {
                "type": "tool_result",
                "session_id": session_id,
                "payload": {
                    "name": "web_search",
                    "step": 3,
                    "result": {
                        "query": "docs example start",
                        "backend": "openai_responses",
                        "sources": [
                            {
                                "title": "Start",
                                "url": "https://docs.example.com/start",
                                "snippet": "alpha",
                            }
                        ],
                    },
                },
            }
        ],
    )
    retained_events = _read_jsonl(log_path)
    artifact_payload = build_web_research_artifact_from_events(
        retained_events
        + [
            {
                "type": "tool_result",
                "session_id": session_id,
                "payload": {
                    "name": "web_search",
                    "step": 4,
                    "result": {
                        "query": "docs example guide",
                        "backend": "openai_responses",
                        "sources": [
                            {
                                "title": "Guide",
                                "url": "https://docs.example.com/guide",
                                "snippet": "artifact only",
                            }
                        ],
                    },
                },
            }
        ]
    )
    (artifact_root / "web_research_sources.json").write_text(
        json.dumps(artifact_payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    cfg = AppConfig(session_log_dir=os.fspath(sessions_dir))

    result = create_feedback_bundle(
        workspace_root=workspace,
        cfg=cfg,
        latest=True,
    )

    exported = json.loads(
        (result.bundle_dir / "session" / "web_research_sources.json").read_text(encoding="utf-8")
    )
    session_score = json.loads(
        (result.bundle_dir / "session_score.json").read_text(encoding="utf-8")
    )

    assert exported["deduped_normalized_search_source_urls"] == [
        "https://docs.example.com/start",
        "https://docs.example.com/guide",
    ]
    assert [entry["normalized_query"] for entry in exported["searches"]] == [
        "docs example start",
        "docs example guide",
    ]
    assert session_score["web_search_calls"] == 2
    assert session_score["unique_web_queries"] == 2
    assert session_score["total_web_sources_returned"] == 2


def test_create_feedback_bundle_uses_in_memory_snapshot_when_logging_disabled(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace_note = workspace / "notes.txt"
    workspace_note.write_text("note\n", encoding="utf-8")
    active_session = SimpleNamespace(
        store=SimpleNamespace(
            enabled=False,
            session_id="active_no_log",
            path=None,
            session_artifact_root=workspace / "missing-artifacts",
        ),
        usage_summary=SimpleNamespace(totals=lambda: {"total_tokens": 9, "cost_usd": 0.0}),
        messages=[
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": (f"hello {workspace_note} and /etc/hosts sk-secret-1234567890abcdef"),
            },
            {"role": "assistant", "content": "world"},
        ],
        root=workspace,
        mode="review",
        client=SimpleNamespace(model="gpt-test"),
        stream=False,
        no_log=True,
        effective_verification_commands=["pytest -q"],
        authoritative_verification_commands=None,
        verification_selection_source="config.verify_commands_fallback",
        verification_selection_reason=(
            "using the configured generic fallback because repo scan found no repo-native command"
        ),
        verification_contract_type="generic_fallback",
        verification_authoritative=False,
    )

    result = create_feedback_bundle(
        workspace_root=workspace,
        active_session=active_session,
        pending_images=[os.fspath(workspace / "clipboard.png")],
        feedback_text="no retained log Authorization: Bearer secret-token-12345678",
    )

    snapshot = json.loads(
        (result.bundle_dir / "session" / "session_snapshot.json").read_text(encoding="utf-8")
    )
    score = json.loads((result.bundle_dir / "session_score.json").read_text(encoding="utf-8"))
    manifest = json.loads((result.bundle_dir / "manifest.json").read_text(encoding="utf-8"))

    assert snapshot["session_id"] == "active_no_log"
    assert snapshot["no_log"] is True
    assert snapshot["message_count"] == 3
    assert [message["role"] for message in snapshot["messages"]] == ["user", "assistant"]
    assert "[REDACTED]" in snapshot["messages"][0]["content"]
    assert "sk-secret-1234567890abcdef" not in snapshot["messages"][0]["content"]
    assert "notes.txt" in snapshot["messages"][0]["content"]
    assert "[redacted host path: hosts]" in snapshot["messages"][0]["content"]
    assert os.fspath(workspace) not in snapshot["messages"][0]["content"]
    assert snapshot["workspace_root"] == "<workspace-root>"
    assert snapshot["pending_images"] == ["clipboard.png"]
    assert snapshot["verification_selection_source"] == "config.verify_commands_fallback"
    assert snapshot["verification_contract_type"] == "generic_fallback"
    assert snapshot["verification_authoritative"] is False
    assert score["available"] is False
    assert manifest["session"]["included_log"] is False
    assert manifest["session"]["included_snapshot"] is True
    feedback_md = (result.bundle_dir / "feedback.md").read_text(encoding="utf-8")
    assert "[REDACTED]" in feedback_md
    assert "secret-token-12345678" not in feedback_md


def test_create_feedback_bundle_active_resumed_session_keeps_cumulative_web_history(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_dir = tmp_path / "sessions"
    session_id = "sess_resumed_web"
    _write_retained_session(
        sessions_dir=sessions_dir,
        session_id=session_id,
        extra_events=[
            {
                "type": "user_message",
                "session_id": session_id,
                "payload": {"content": "Please inspect https://docs.example.com/spec"},
            },
            {
                "type": "tool_result",
                "session_id": session_id,
                "payload": {
                    "name": "web_search",
                    "step": 3,
                    "result": {
                        "query": "docs example",
                        "backend": "openai_responses",
                        "sources": [
                            {
                                "title": "Guide",
                                "url": "https://docs.example.com/guide",
                                "snippet": "Official guide",
                            }
                        ],
                    },
                },
            },
        ],
    )

    store = SessionStore(
        enabled=False,
        artifact_persistence_enabled=True,
        sessions_dir=sessions_dir,
        session_id=session_id,
        cwd=os.fspath(workspace),
        repo_root=os.fspath(workspace),
    )
    store.append(
        "tool_call",
        {
            "name": "web_fetch",
            "step": 4,
            "arguments": {"url": "https://docs.example.com/guide"},
        },
    )
    store.append(
        "tool_result",
        {
            "name": "web_fetch",
            "step": 4,
            "result": {
                "url": "https://docs.example.com/guide",
                "final_url": "https://docs.example.com/final",
                "status_code": 200,
                "content_type": "text/html",
                "title": "Final",
                "backend": "httpx",
            },
        },
    )

    active_session = SimpleNamespace(
        store=store,
        usage_summary=SimpleNamespace(totals=lambda: {"total_tokens": 9, "cost_usd": 0.0}),
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Please inspect https://docs.example.com/spec"},
            {"role": "assistant", "content": "Loaded docs."},
        ],
        root=workspace,
        mode="review",
        client=SimpleNamespace(model="gpt-test"),
        stream=False,
        no_log=True,
        verification_contract_type="generic_fallback",
        verification_authoritative=False,
    )

    result = create_feedback_bundle(
        workspace_root=workspace,
        active_session=active_session,
    )

    exported = json.loads(
        (result.bundle_dir / "session" / "web_research_sources.json").read_text(encoding="utf-8")
    )
    snapshot = json.loads(
        (result.bundle_dir / "session" / "session_snapshot.json").read_text(encoding="utf-8")
    )
    score = json.loads((result.bundle_dir / "session_score.json").read_text(encoding="utf-8"))

    assert exported["deduped_normalized_user_urls"] == ["https://docs.example.com/spec"]
    assert exported["deduped_normalized_search_source_urls"] == ["https://docs.example.com/guide"]
    assert exported["deduped_normalized_fetch_urls"] == ["https://docs.example.com/guide"]
    assert exported["deduped_normalized_final_fetch_urls"] == ["https://docs.example.com/final"]
    assert snapshot["web_research_sources"]["deduped_normalized_search_source_urls"] == [
        "https://docs.example.com/guide"
    ]
    assert score["web_search_calls"] == 1
    assert score["web_fetch_calls"] == 1
    assert score["total_web_sources_fetched"] == 1


def test_create_feedback_bundle_redacts_secrets_in_exported_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_dir = tmp_path / "sessions"
    session_id = "sess_secret"
    log_path = sessions_dir / f"{session_id}.jsonl"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(
            {
                "type": "assistant_message",
                "session_id": session_id,
                "payload": {"content": "token sk-secret-1234567890abcdef"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_root = sessions_dir / session_id
    (artifact_root / "tool_outputs").mkdir(parents=True, exist_ok=True)
    (artifact_root / "tool_outputs" / "payload.txt").write_text(
        "Authorization: Bearer secret-token-12345678\n",
        encoding="utf-8",
    )
    cfg = AppConfig(session_log_dir=os.fspath(sessions_dir))

    paths = create_plan_run(workspace)
    paths.execution_reports_dir.mkdir(parents=True, exist_ok=True)
    (paths.execution_reports_dir / "T01.md").write_text(
        "OPENAI_API_KEY=super-secret-value\n",
        encoding="utf-8",
    )

    result = create_feedback_bundle(
        workspace_root=workspace,
        cfg=cfg,
        latest=True,
        feedback_text='api_key: "sk-secret-1234567890abcdef"',
    )

    exported_log = (result.bundle_dir / "session" / "log.jsonl").read_text(encoding="utf-8")
    exported_artifact = (
        result.bundle_dir / "session" / "artifacts" / "tool_outputs" / "payload.txt"
    ).read_text(encoding="utf-8")
    exported_report = (
        result.bundle_dir / "forge" / "run" / "execution" / "reports" / "T01.md"
    ).read_text(encoding="utf-8")
    feedback_md = (result.bundle_dir / "feedback.md").read_text(encoding="utf-8")

    assert "sk-secret-1234567890abcdef" not in exported_log
    assert "secret-token-12345678" not in exported_artifact
    assert "super-secret-value" not in exported_report
    assert "sk-secret-1234567890abcdef" not in feedback_md
    assert "[REDACTED]" in exported_log
    assert "[REDACTED]" in exported_artifact
    assert "[REDACTED]" in exported_report
    assert "[REDACTED]" in feedback_md


def test_create_feedback_bundle_sanitizes_freeform_paths_in_exported_session_jsonl(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    target = workspace / "src" / "app.py"
    target.write_text("print('ok')\n", encoding="utf-8")
    sessions_dir = tmp_path / "sessions"
    session_id = "sess_paths"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / f"{session_id}.jsonl").write_text(
        json.dumps(
            {
                "type": "assistant_message",
                "session_id": session_id,
                "payload": {
                    "content": (
                        f"see {target} for the workspace repro, /etc/hosts "
                        "for the host repro, and POST /v1/chat/completions "
                        "for the API route repro"
                    )
                },
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = AppConfig(session_log_dir=os.fspath(sessions_dir))

    result = create_feedback_bundle(
        workspace_root=workspace,
        cfg=cfg,
        latest=True,
    )

    exported_events = _read_jsonl(result.bundle_dir / "session" / "log.jsonl")
    assert len(exported_events) == 1
    content = exported_events[0]["payload"]["content"]

    assert "src/app.py" in content
    assert "[redacted host path: hosts]" in content
    assert "POST /v1/chat/completions" in content
    assert os.fspath(workspace) not in content
    assert "/etc/hosts" not in content
    assert "[redacted host path: completions]" not in content


def test_create_feedback_bundle_sanitizes_freeform_paths_in_copied_json_artifacts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    readme_path = workspace / "README.md"
    readme_path.write_text("demo\n", encoding="utf-8")
    sessions_dir = tmp_path / "sessions"
    session_id = "sess_json_paths"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / f"{session_id}.jsonl").write_text(
        json.dumps(
            {
                "type": "session_start",
                "session_id": session_id,
                "payload": {},
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_root = sessions_dir / session_id
    (artifact_root / "tool_outputs").mkdir(parents=True, exist_ok=True)
    (artifact_root / "tool_outputs" / "payload.json").write_text(
        json.dumps(
            {
                "message": f"see {readme_path} and /etc/hosts",
                "note": f"workspace file is {readme_path}",
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = AppConfig(session_log_dir=os.fspath(sessions_dir))

    result = create_feedback_bundle(
        workspace_root=workspace,
        cfg=cfg,
        latest=True,
    )

    exported_payload = json.loads(
        (result.bundle_dir / "session" / "artifacts" / "tool_outputs" / "payload.json").read_text(
            encoding="utf-8"
        )
    )

    assert exported_payload["message"] == "see README.md and [redacted host path: hosts]"
    assert exported_payload["note"] == "workspace file is README.md"
    assert os.fspath(workspace) not in json.dumps(exported_payload, sort_keys=True)
    assert "/etc/hosts" not in json.dumps(exported_payload, sort_keys=True)


def test_create_feedback_bundle_sanitizes_windows_freeform_paths_in_exported_session_jsonl(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_dir = tmp_path / "sessions"
    session_id = "sess_windows_paths"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / f"{session_id}.jsonl").write_text(
        json.dumps(
            {
                "type": "assistant_message",
                "session_id": session_id,
                "payload": {
                    "content": (
                        r"see C:\Users\alice\Desktop\win-log.txt, "
                        r"C:\tmp, "
                        r"C:/short-log.txt, "
                        r"\\server\share\ops\unc-log.txt, "
                        r"\\server/share/mixed-unc-log.txt, and "
                        r"\\server\share/mixed-separators-log.txt "
                        r"in the exported trace"
                    )
                },
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = AppConfig(session_log_dir=os.fspath(sessions_dir))

    result = create_feedback_bundle(
        workspace_root=workspace,
        cfg=cfg,
        latest=True,
    )

    exported_events = _read_jsonl(result.bundle_dir / "session" / "log.jsonl")
    assert len(exported_events) == 1
    content = exported_events[0]["payload"]["content"]

    assert "[redacted host path: win-log.txt]" in content
    assert "[redacted host path: tmp]" in content
    assert "[redacted host path: short-log.txt]" in content
    assert "[redacted host path: unc-log.txt]" in content
    assert "[redacted host path: mixed-unc-log.txt]" in content
    assert "[redacted host path: mixed-separators-log.txt]" in content
    assert r"C:\Users\alice\Desktop\win-log.txt" not in content
    assert r"C:\tmp" not in content
    assert r"C:/short-log.txt" not in content
    assert r"\\server\share\ops\unc-log.txt" not in content
    assert r"\\server/share/mixed-unc-log.txt" not in content
    assert r"\\server\share/mixed-separators-log.txt" not in content


def test_create_feedback_bundle_sanitizes_windows_freeform_paths_in_copied_json_artifacts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_dir = tmp_path / "sessions"
    session_id = "sess_windows_json"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / f"{session_id}.jsonl").write_text(
        json.dumps(
            {
                "type": "session_start",
                "session_id": session_id,
                "payload": {},
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    artifact_root = sessions_dir / session_id
    (artifact_root / "tool_outputs").mkdir(parents=True, exist_ok=True)
    (artifact_root / "tool_outputs" / "windows.json").write_text(
        json.dumps(
            {
                "message": (
                    r"check C:\Users\alice\Desktop\artifact.txt, "
                    r"C:\tmp, and C:/artifact-short.txt"
                ),
                "note": (
                    r"see \\server\share\ops\unc-note.txt, "
                    r"\\server/share/mixed-unc-note.txt, and "
                    r"\\server\share/mixed-separators-note.txt too"
                ),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = AppConfig(session_log_dir=os.fspath(sessions_dir))

    result = create_feedback_bundle(
        workspace_root=workspace,
        cfg=cfg,
        latest=True,
    )

    exported_payload = json.loads(
        (result.bundle_dir / "session" / "artifacts" / "tool_outputs" / "windows.json").read_text(
            encoding="utf-8"
        )
    )

    assert exported_payload["message"] == (
        "check [redacted host path: artifact.txt], "
        "[redacted host path: tmp], and [redacted host path: artifact-short.txt]"
    )
    assert exported_payload["note"] == (
        "see [redacted host path: unc-note.txt], "
        "[redacted host path: mixed-unc-note.txt], and "
        "[redacted host path: mixed-separators-note.txt] too"
    )
    assert r"C:\Users\alice\Desktop\artifact.txt" not in json.dumps(
        exported_payload, sort_keys=True
    )
    assert r"C:\tmp" not in json.dumps(exported_payload, sort_keys=True)
    assert r"C:/artifact-short.txt" not in json.dumps(exported_payload, sort_keys=True)
    assert r"\\server\share\ops\unc-note.txt" not in json.dumps(exported_payload, sort_keys=True)
    assert r"\\server/share/mixed-unc-note.txt" not in json.dumps(exported_payload, sort_keys=True)
    assert r"\\server\share/mixed-separators-note.txt" not in json.dumps(
        exported_payload, sort_keys=True
    )


def test_create_feedback_bundle_sanitizes_exported_asset_original_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_dir = tmp_path / "sessions"
    _write_retained_session(sessions_dir=sessions_dir, session_id="sess_assets")
    cfg = AppConfig(session_log_dir=os.fspath(sessions_dir))

    create_plan_run(workspace)
    asset_source = workspace / "brief.txt"
    asset_source.write_text("brief\n", encoding="utf-8")
    attach_asset(workspace, asset_source)

    result = create_feedback_bundle(
        workspace_root=workspace,
        cfg=cfg,
        latest=True,
    )

    exported_plan = json.loads(
        (result.bundle_dir / "forge" / "run" / "plan" / "plan.json").read_text(encoding="utf-8")
    )
    asset = exported_plan["assets"][0]

    assert asset["original_path"] == "brief.txt"
    assert asset["stored_path"].startswith(".sylliptor/runs/")
    assert os.fspath(tmp_path) not in json.dumps(exported_plan, sort_keys=True)


def test_create_feedback_bundle_redacts_foreign_session_workspace_metadata(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace_a"
    workspace.mkdir()
    foreign_workspace = tmp_path / "workspace_b"
    foreign_workspace.mkdir()
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    (sessions_dir / "sess_foreign.jsonl").write_text(
        json.dumps(
            {
                "type": "session_start",
                "session_id": "sess_foreign",
                "cwd": os.fspath(foreign_workspace),
                "workspace_root": os.fspath(foreign_workspace),
                "payload": {},
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = AppConfig(session_log_dir=os.fspath(sessions_dir))

    create_plan_run(workspace)

    result = create_feedback_bundle(
        workspace_root=workspace,
        cfg=cfg,
        latest=True,
    )

    exported_events = _read_jsonl(result.bundle_dir / "session" / "log.jsonl")
    assert len(exported_events) == 1
    exported = exported_events[0]

    assert exported["workspace_root"] == "[redacted host path: workspace_b]"
    assert exported["cwd"] == "[redacted host path: workspace_b]"
    assert exported["workspace_root"] != "<workspace-root>"
    assert os.fspath(tmp_path) not in json.dumps(exported, sort_keys=True)


def test_create_feedback_bundle_sanitizes_paths_in_feedback_markdown(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    issue_path = workspace / "src" / "bug.py"
    issue_path.parent.mkdir(parents=True, exist_ok=True)
    issue_path.write_text("raise RuntimeError\n", encoding="utf-8")
    sessions_dir = tmp_path / "sessions"
    _write_retained_session(sessions_dir=sessions_dir, session_id="sess_feedback_paths")
    cfg = AppConfig(session_log_dir=os.fspath(sessions_dir))

    result = create_feedback_bundle(
        workspace_root=workspace,
        cfg=cfg,
        latest=True,
        feedback_text=(
            f"bug hits at {issue_path}, also touches /etc/hosts, and still leaks /etc plus /mnt"
        ),
    )

    feedback_md = (result.bundle_dir / "feedback.md").read_text(encoding="utf-8")

    assert "src/bug.py" in feedback_md
    assert "[redacted host path: hosts]" in feedback_md
    assert "[redacted host path: etc]" in feedback_md
    assert "[redacted host path: mnt]" in feedback_md
    assert os.fspath(workspace) not in feedback_md
    assert "/etc/hosts" not in feedback_md
    assert " /etc " not in feedback_md
    assert " /mnt" not in feedback_md


def test_create_feedback_bundle_sanitizes_windows_paths_in_feedback_markdown(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_dir = tmp_path / "sessions"
    _write_retained_session(sessions_dir=sessions_dir, session_id="sess_feedback_windows")
    cfg = AppConfig(session_log_dir=os.fspath(sessions_dir))

    result = create_feedback_bundle(
        workspace_root=workspace,
        cfg=cfg,
        latest=True,
        feedback_text=(
            r"windows repro C:\Users\alice\Desktop\feedback.txt, "
            r"C:\tmp, "
            r"C:/feedback-short.txt, "
            r"\\server\share\ops\feedback-unc.txt, "
            r"\\server/share/feedback-mixed-unc.txt, and "
            r"\\server\share/feedback-mixed-separators.txt"
        ),
    )

    feedback_md = (result.bundle_dir / "feedback.md").read_text(encoding="utf-8")

    assert "[redacted host path: feedback.txt]" in feedback_md
    assert "[redacted host path: tmp]" in feedback_md
    assert "[redacted host path: feedback-short.txt]" in feedback_md
    assert "[redacted host path: feedback-unc.txt]" in feedback_md
    assert "[redacted host path: feedback-mixed-unc.txt]" in feedback_md
    assert "[redacted host path: feedback-mixed-separators.txt]" in feedback_md
    assert r"C:\Users\alice\Desktop\feedback.txt" not in feedback_md
    assert r"C:\tmp" not in feedback_md
    assert r"C:/feedback-short.txt" not in feedback_md
    assert r"\\server\share\ops\feedback-unc.txt" not in feedback_md
    assert r"\\server/share/feedback-mixed-unc.txt" not in feedback_md
    assert r"\\server\share/feedback-mixed-separators.txt" not in feedback_md


def test_create_feedback_bundle_rejects_unsafe_explicit_session_id(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    cfg = AppConfig(session_log_dir=os.fspath(tmp_path / "sessions"))

    with pytest.raises(FeedbackReportError, match="Invalid session id"):
        create_feedback_bundle(
            workspace_root=workspace,
            cfg=cfg,
            session_id="../../outside/demo",
        )


def test_create_feedback_bundle_rejects_unsafe_explicit_run_id(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(FeedbackReportError, match="Invalid run id"):
        create_feedback_bundle(
            workspace_root=workspace,
            run_id="../../src",
        )


def test_create_feedback_github_issue_draft_builds_sanitized_prefill(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_dir = tmp_path / "sessions"
    _write_retained_session(sessions_dir=sessions_dir, session_id="sess_issue")
    feedback_text = (
        f"failure at {workspace / 'src' / 'app.py'} Authorization: Bearer secret-token-12345678"
    )
    bundle = create_feedback_bundle(
        workspace_root=workspace,
        cfg=AppConfig(
            session_log_dir=os.fspath(sessions_dir),
            feedback_github_repo="https://github.com/acme/sylliptor.git",
        ),
        feedback_text=feedback_text,
        latest=True,
    )
    opened_urls: list[str] = []

    result = create_feedback_github_issue_draft(
        bundle_result=bundle,
        feedback_text=feedback_text,
        cfg=AppConfig(feedback_github_repo="https://github.com/acme/sylliptor.git"),
        browser_open=lambda url, **_kwargs: opened_urls.append(str(url)) is None or True,
    )

    assert result.repo == "acme/sylliptor"
    assert result.opened is True
    assert opened_urls == [result.issue_url]
    parsed = urlsplit(str(result.issue_url))
    query = parse_qs(parsed.query)
    body = query["body"][0]

    assert parsed.scheme == "https"
    assert parsed.netloc == "github.com"
    assert parsed.path == "/acme/sylliptor/issues/new"
    assert query["title"][0].startswith("Sylliptor feedback: failure at src/app.py")
    assert "src/app.py" in body
    assert os.fspath(workspace) not in body
    assert "secret-token-12345678" not in body
    assert "[REDACTED]" in body
    assert "not uploaded automatically" in body
    assert "raw logs" in body


def test_create_feedback_github_issue_draft_can_be_disabled_or_not_opened(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bundle = FeedbackBundleResult(
        bundle_dir=workspace / "sylliptor-feedback" / "bundle",
        zip_path=workspace / "sylliptor-feedback" / "bundle.zip",
        output_root=workspace / "sylliptor-feedback",
        workspace_root=workspace,
        session_id="sid",
        run_id=None,
    )
    opened_urls: list[str] = []

    disabled = create_feedback_github_issue_draft(
        bundle_result=bundle,
        cfg=AppConfig(feedback_github_enabled=False),
        browser_open=lambda url, **_kwargs: opened_urls.append(str(url)) is None or True,
    )
    no_open = create_feedback_github_issue_draft(
        bundle_result=bundle,
        cfg=AppConfig(feedback_github_repo="acme/sylliptor"),
        open_browser=False,
        browser_open=lambda url, **_kwargs: opened_urls.append(str(url)) is None or True,
    )

    assert disabled.issue_url is None
    assert disabled.disabled_reason == "disabled"
    assert no_open.issue_url is not None
    assert no_open.opened is False
    assert no_open.open_attempted is False
    assert no_open.disabled_reason == "browser_open_disabled"
    assert opened_urls == []


def test_create_feedback_github_issue_draft_caps_large_prefill_urls(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_dir = tmp_path / "sessions"
    _write_retained_session(sessions_dir=sessions_dir, session_id="sess_issue")
    bundle = create_feedback_bundle(
        workspace_root=workspace,
        cfg=AppConfig(session_log_dir=os.fspath(sessions_dir)),
        feedback_text="short local feedback",
        latest=True,
    )
    long_feedback = "very long feedback " + ("\u03bb" * 12000)

    result = create_feedback_github_issue_draft(
        bundle_result=bundle,
        feedback_text=long_feedback,
        cfg=AppConfig(feedback_github_repo="acme/sylliptor"),
        open_browser=False,
    )

    assert result.issue_url is not None
    assert len(result.issue_url) <= 8000
    body = parse_qs(urlsplit(result.issue_url).query)["body"][0]
    assert "too large for a stable GitHub prefill URL" in body
    assert "not uploaded automatically" in body


def test_chat_report_command_creates_bundle_host_side(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []
    issue_calls: list[dict[str, Any]] = []

    def fake_create_feedback_bundle(**kwargs: Any) -> FeedbackBundleResult:
        calls.append(dict(kwargs))
        return FeedbackBundleResult(
            bundle_dir=tmp_path / "sylliptor-feedback" / "bundle",
            zip_path=tmp_path / "sylliptor-feedback" / "bundle.zip",
            output_root=tmp_path / "sylliptor-feedback",
            workspace_root=tmp_path,
            session_id="sid",
            run_id=None,
        )

    def fake_create_feedback_github_issue_draft(**kwargs: Any) -> FeedbackGithubIssueResult:
        issue_calls.append(dict(kwargs))
        return FeedbackGithubIssueResult(
            repo="AlysisAi/Sylliptor",
            issue_url="https://github.com/AlysisAi/Sylliptor/issues/new?title=x",
            opened=False,
            open_attempted=False,
        )

    monkeypatch.setattr(cli_mod, "create_feedback_bundle", fake_create_feedback_bundle)
    monkeypatch.setattr(
        cli_mod,
        "create_feedback_github_issue_draft",
        fake_create_feedback_github_issue_draft,
    )

    session = SimpleNamespace(
        cfg=AppConfig(model="test-model"),
        store=SimpleNamespace(session_id="sid"),
        mode="review",
    )
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=False)

    result = cli_mod._handle_chat_command(
        input_text="/report weird behavior",
        root=tmp_path,
        session=session,
        pending_images=["queued.png"],
        console=console,
        forge_state=cli_mod._ForgeChatState(),
        plan_mode_state=cli_mod._ChatPlanModeState(),
    )

    assert result == "handled"
    assert calls
    assert calls[0]["workspace_root"] == tmp_path
    assert calls[0]["feedback_text"] == "weird behavior"
    assert calls[0]["active_session"] is session
    assert calls[0]["active_run_paths"] is None
    assert calls[0]["pending_images"] == ["queued.png"]
    assert issue_calls[0]["bundle_result"] == FeedbackBundleResult(
        bundle_dir=tmp_path / "sylliptor-feedback" / "bundle",
        zip_path=tmp_path / "sylliptor-feedback" / "bundle.zip",
        output_root=tmp_path / "sylliptor-feedback",
        workspace_root=tmp_path,
        session_id="sid",
        run_id=None,
    )
    assert issue_calls[0]["feedback_text"] == "weird behavior"
    rendered = stream.getvalue()
    assert "Feedback bundle directory:" in rendered
    assert "Feedback bundle archive:" in rendered
    assert "GitHub issue draft URL:" in rendered


def test_chat_report_still_reports_bundle_when_github_draft_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_create_feedback_bundle(**_kwargs: Any) -> FeedbackBundleResult:
        return FeedbackBundleResult(
            bundle_dir=tmp_path / "sylliptor-feedback" / "bundle",
            zip_path=tmp_path / "sylliptor-feedback" / "bundle.zip",
            output_root=tmp_path / "sylliptor-feedback",
            workspace_root=tmp_path,
            session_id="sid",
            run_id=None,
        )

    def fail_create_feedback_github_issue_draft(**_kwargs: Any) -> FeedbackGithubIssueResult:
        raise ConfigError("feedback_github_repo must be a GitHub repo in owner/name form")

    monkeypatch.setattr(cli_mod, "create_feedback_bundle", fake_create_feedback_bundle)
    monkeypatch.setattr(
        cli_mod,
        "create_feedback_github_issue_draft",
        fail_create_feedback_github_issue_draft,
    )

    session = SimpleNamespace(
        cfg=AppConfig(model="test-model"),
        store=SimpleNamespace(session_id="sid"),
        mode="review",
    )
    stream = io.StringIO()
    result = cli_mod._handle_chat_command(
        input_text="/report weird behavior",
        root=tmp_path,
        session=session,
        pending_images=[],
        console=Console(file=stream, force_terminal=False),
        forge_state=cli_mod._ForgeChatState(),
        plan_mode_state=cli_mod._ChatPlanModeState(),
    )

    rendered = stream.getvalue()
    assert result == "handled"
    assert "Feedback bundle directory:" in rendered
    assert "Feedback bundle archive:" in rendered
    assert "GitHub issue draft skipped:" in rendered
    assert "Feedback report failed" not in rendered


def test_chat_report_command_in_forge_includes_active_run_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []
    issue_calls: list[dict[str, Any]] = []

    def fake_create_feedback_bundle(**kwargs: Any) -> FeedbackBundleResult:
        calls.append(dict(kwargs))
        return FeedbackBundleResult(
            bundle_dir=tmp_path / "sylliptor-feedback" / "bundle",
            zip_path=tmp_path / "sylliptor-feedback" / "bundle.zip",
            output_root=tmp_path / "sylliptor-feedback",
            workspace_root=tmp_path,
            session_id="sid",
            run_id="run_01",
        )

    def fake_create_feedback_github_issue_draft(**kwargs: Any) -> FeedbackGithubIssueResult:
        issue_calls.append(dict(kwargs))
        return FeedbackGithubIssueResult(
            repo="AlysisAi/Sylliptor",
            issue_url="https://github.com/AlysisAi/Sylliptor/issues/new?title=x",
            opened=True,
            open_attempted=True,
        )

    monkeypatch.setattr(cli_mod, "create_feedback_bundle", fake_create_feedback_bundle)
    monkeypatch.setattr(
        cli_mod,
        "create_feedback_github_issue_draft",
        fake_create_feedback_github_issue_draft,
    )

    run_paths = SimpleNamespace(root=tmp_path, run_id="run_01")
    forge_state = cli_mod._ForgeChatState(ui_mode="forge", paths=run_paths, plan={})
    session = SimpleNamespace(
        cfg=AppConfig(model="test-model"),
        store=SimpleNamespace(session_id="sid"),
        mode="review",
    )
    console = Console(file=io.StringIO(), force_terminal=False)

    result = cli_mod._handle_chat_command(
        input_text="/report odd planner output",
        root=tmp_path,
        session=session,
        pending_images=[],
        console=console,
        forge_state=forge_state,
        plan_mode_state=cli_mod._ChatPlanModeState(),
    )

    assert result == "handled"
    assert calls
    assert calls[0]["active_run_paths"] is run_paths
    assert calls[0]["feedback_text"] == "odd planner output"
    assert issue_calls
    assert issue_calls[0]["feedback_text"] == "odd planner output"


def test_chat_feedback_alias_uses_report_flow(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_create_feedback_bundle(**kwargs: Any) -> FeedbackBundleResult:
        calls.append(dict(kwargs))
        return FeedbackBundleResult(
            bundle_dir=tmp_path / "sylliptor-feedback" / "bundle",
            zip_path=tmp_path / "sylliptor-feedback" / "bundle.zip",
            output_root=tmp_path / "sylliptor-feedback",
            workspace_root=tmp_path,
            session_id="sid",
            run_id=None,
        )

    def fake_create_feedback_github_issue_draft(**_kwargs: Any) -> FeedbackGithubIssueResult:
        return FeedbackGithubIssueResult(
            repo="AlysisAi/Sylliptor",
            issue_url=None,
            opened=False,
            open_attempted=False,
            disabled_reason="disabled",
        )

    monkeypatch.setattr(cli_mod, "create_feedback_bundle", fake_create_feedback_bundle)
    monkeypatch.setattr(
        cli_mod,
        "create_feedback_github_issue_draft",
        fake_create_feedback_github_issue_draft,
    )

    session = SimpleNamespace(
        cfg=AppConfig(model="test-model"),
        store=SimpleNamespace(session_id="sid"),
        mode="review",
    )
    result = cli_mod._handle_chat_command(
        input_text="/feedback issue text",
        root=tmp_path,
        session=session,
        pending_images=[],
        console=Console(file=io.StringIO(), force_terminal=False),
        forge_state=cli_mod._ForgeChatState(),
        plan_mode_state=cli_mod._ChatPlanModeState(),
    )

    assert result == "handled"
    assert calls[0]["feedback_text"] == "issue text"


def test_report_create_command_exports_latest_retained_bundle(tmp_path: Path) -> None:
    runner = CliRunner()
    env = _env(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "README.md").write_text("demo\n", encoding="utf-8")
    sessions_dir = tmp_path / "data" / "sessions"
    _write_retained_session(sessions_dir=sessions_dir, session_id="sess_cli")

    result = runner.invoke(
        sylliptor_app,
        ["report", "create", "beta note", "--path", os.fspath(workspace), "--latest"],
        env=env,
    )

    assert result.exit_code == 0
    assert "Feedback bundle directory:" in result.output
    assert "Feedback bundle archive:" in result.output
    assert "GitHub issue draft URL:" in result.output
    assert (workspace / "sylliptor-feedback").exists()


def test_report_create_command_defaults_to_url_only_and_open_flag_opts_in(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    env = _env(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_dir = tmp_path / "data" / "sessions"
    _write_retained_session(sessions_dir=sessions_dir, session_id="sess_cli")
    open_values: list[bool | None] = []

    def fake_create_feedback_github_issue_draft(**kwargs: Any) -> FeedbackGithubIssueResult:
        open_values.append(kwargs.get("open_browser"))
        return FeedbackGithubIssueResult(
            repo="AlysisAi/Sylliptor",
            issue_url="https://github.com/AlysisAi/Sylliptor/issues/new?title=x",
            opened=bool(kwargs.get("open_browser")),
            open_attempted=bool(kwargs.get("open_browser")),
        )

    monkeypatch.setattr(
        cli_mod,
        "create_feedback_github_issue_draft",
        fake_create_feedback_github_issue_draft,
    )

    first = runner.invoke(
        sylliptor_app,
        ["report", "create", "beta note", "--path", os.fspath(workspace), "--latest"],
        env=env,
    )
    second = runner.invoke(
        sylliptor_app,
        ["report", "create", "beta note", "--path", os.fspath(workspace), "--latest", "--open"],
        env=env,
    )

    assert first.exit_code == 0
    assert second.exit_code == 0
    assert open_values == [False, True]
    assert "GitHub issue draft URL:" in first.output
    assert "GitHub issue draft opened:" in second.output


def test_report_create_command_reports_bundle_when_github_draft_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    env = _env(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_dir = tmp_path / "data" / "sessions"
    _write_retained_session(sessions_dir=sessions_dir, session_id="sess_cli")

    def fail_create_feedback_github_issue_draft(**_kwargs: Any) -> FeedbackGithubIssueResult:
        raise ConfigError("feedback_github_repo must be a GitHub repo in owner/name form")

    monkeypatch.setattr(
        cli_mod,
        "create_feedback_github_issue_draft",
        fail_create_feedback_github_issue_draft,
    )

    result = runner.invoke(
        sylliptor_app,
        ["report", "create", "beta note", "--path", os.fspath(workspace), "--latest"],
        env=env,
    )

    assert result.exit_code == 0
    assert "Feedback bundle directory:" in result.output
    assert "Feedback bundle archive:" in result.output
    assert "GitHub issue draft skipped:" in result.output
    assert "Feedback report failed" not in result.output


def test_report_create_command_local_only_skips_github_issue(tmp_path: Path) -> None:
    runner = CliRunner()
    env = _env(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sessions_dir = tmp_path / "data" / "sessions"
    _write_retained_session(sessions_dir=sessions_dir, session_id="sess_cli")

    result = runner.invoke(
        sylliptor_app,
        [
            "report",
            "create",
            "beta note",
            "--path",
            os.fspath(workspace),
            "--latest",
            "--local-only",
        ],
        env=env,
    )

    assert result.exit_code == 0
    assert "Feedback bundle directory:" in result.output
    assert "GitHub issue draft" not in result.output


def test_report_create_rejects_conflicting_github_flags(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        sylliptor_app,
        [
            "report",
            "create",
            "beta note",
            "--path",
            os.fspath(tmp_path),
            "--github",
            "--local-only",
        ],
        env=_env(tmp_path),
    )

    assert result.exit_code == 2
    assert "Use either --github or --local-only" in result.output
