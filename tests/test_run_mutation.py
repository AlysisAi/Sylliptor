from __future__ import annotations

import importlib
import json
import os
import socket
from pathlib import Path

import pytest
from rich.console import Console

import sylliptor_agent_cli.atomic_io as atomic_io_mod
import sylliptor_agent_cli.run_lock as run_lock_mod
from sylliptor_agent_cli.atomic_io import atomic_write_json, atomic_write_text
from sylliptor_agent_cli.config import AppConfig
from sylliptor_agent_cli.forge import create_plan_run
from sylliptor_agent_cli.run_lock import (
    RunMutationConflictError,
    acquire_run_mutation_guard,
    inspect_run_mutation_lock,
    write_run_mutation_lock_metadata,
)
from sylliptor_agent_cli.swarm_orchestrator import (
    MergeOutcome,
    _append_remote_report_update,
    _persist_worker_result,
    _write_merge_result,
    _write_swarm_summary,
    acquire_swarm_mutation_guard,
    run_swarm,
)
from sylliptor_agent_cli.swarm_worker import TaskWorkerResult


def test_atomic_io_imports_cleanly_from_fresh_checkout() -> None:
    atomic_io_mod = importlib.import_module("sylliptor_agent_cli.atomic_io")
    run_lock_import = importlib.import_module("sylliptor_agent_cli.run_lock")

    assert hasattr(atomic_io_mod, "atomic_write_text")
    assert hasattr(atomic_io_mod, "atomic_write_json")
    assert hasattr(run_lock_import, "acquire_run_mutation_guard")


def test_atomic_write_text_replaces_existing_file_without_temp_leaks(tmp_path: Path) -> None:
    target = tmp_path / "artifact.txt"
    target.write_text("before\n", encoding="utf-8")

    atomic_write_text(target, "after\n")

    assert target.read_text(encoding="utf-8") == "after\n"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_atomic_write_json_overwrites_existing_file_atomically(tmp_path: Path) -> None:
    target = tmp_path / "artifact.json"
    target.write_text('{"before": true}\n', encoding="utf-8")

    atomic_write_json(target, {"after": True, "value": 3})

    assert target.read_text(encoding="utf-8") == '{\n  "after": true,\n  "value": 3\n}\n'
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_atomic_write_helpers_create_parent_dirs(tmp_path: Path) -> None:
    text_target = tmp_path / "nested" / "artifacts" / "artifact.txt"
    json_target = tmp_path / "nested" / "artifacts" / "artifact.json"

    atomic_write_text(text_target, "created\n")
    atomic_write_json(json_target, {"ok": True})

    assert text_target.read_text(encoding="utf-8") == "created\n"
    assert json_target.read_text(encoding="utf-8") == '{\n  "ok": true\n}\n'
    assert list(text_target.parent.glob(f".{text_target.name}.*.tmp")) == []
    assert list(json_target.parent.glob(f".{json_target.name}.*.tmp")) == []


def test_atomic_write_text_cleans_temp_file_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "artifact.txt"

    def fail_replace(_src: Path, _dst: Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("sylliptor_agent_cli.atomic_io.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        atomic_write_text(target, "after\n")

    assert target.exists() is False
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_atomic_write_text_preserves_newlines_exactly(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "artifact.txt"
    original_fdopen = atomic_io_mod.os.fdopen
    captured: dict[str, str | None] = {"newline": None}

    def recording_fdopen(*args, **kwargs):  # type: ignore[no-untyped-def]
        captured["newline"] = kwargs.get("newline")
        return original_fdopen(*args, **kwargs)

    monkeypatch.setattr("sylliptor_agent_cli.atomic_io.os.fdopen", recording_fdopen)

    atomic_write_text(target, "line1\r\nline2\n")

    assert captured["newline"] == ""
    assert target.read_bytes() == b"line1\r\nline2\n"


def test_atomic_write_text_raises_on_real_directory_fsync_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name == "nt":
        pytest.skip("directory fsync is not attempted on Windows")

    target = tmp_path / "artifact.txt"
    real_open = atomic_io_mod.os.open
    real_fsync = atomic_io_mod.os.fsync
    sentinel_dir_fd = 999_999

    def open_wrapper(path, flags, mode=0o777):  # type: ignore[no-untyped-def]
        if Path(path) == tmp_path and flags == atomic_io_mod.os.O_RDONLY:
            return sentinel_dir_fd
        return real_open(path, flags, mode)

    def fsync_wrapper(fd: int) -> None:
        if fd == sentinel_dir_fd:
            raise OSError(5, "directory fsync failed")
        real_fsync(fd)

    monkeypatch.setattr("sylliptor_agent_cli.atomic_io.os.open", open_wrapper)
    monkeypatch.setattr("sylliptor_agent_cli.atomic_io.os.fsync", fsync_wrapper)
    monkeypatch.setattr("sylliptor_agent_cli.atomic_io.os.close", lambda _fd: None)

    with pytest.raises(OSError, match="directory fsync failed"):
        atomic_write_text(target, "after\n")

    assert target.read_text(encoding="utf-8") == "after\n"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []


def test_swarm_artifact_writes_publish_atomically(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    paths = create_plan_run(repo)

    worker_result = TaskWorkerResult(
        task_id="T01",
        title="Task A",
        branch="feat/t01-a",
        worktree_path=os.fspath(paths.run_dir / "worktrees" / "T01" / "repo"),
        started_at="2026-03-26T00:00:00+00:00",
        finished_at="2026-03-26T00:01:00+00:00",
        success=True,
        summary="ok",
        commit_hash="abc123",
        error=None,
        report_path=".sylliptor/runs/x/execution/reports/T01.md",
        patch_path=".sylliptor/runs/x/execution/patches/T01.diff",
        log_path=".sylliptor/runs/x/execution/logs/T01.jsonl",
        log_pointer_path=".sylliptor/runs/x/execution/logs/T01.log.json",
        warnings=[],
        changed_files=["src/example.py"],
        verify_failed=False,
        verify_summary="verification passed (1/1)",
        verify_artifact_path=".sylliptor/runs/x/execution/verify/T01.txt",
    )
    worker_result_path = _persist_worker_result(paths, worker_result)
    assert json.loads(worker_result_path.read_text(encoding="utf-8"))["task_id"] == "T01"
    assert list(worker_result_path.parent.glob(f".{worker_result_path.name}.*.tmp")) == []

    merge_result_path = _write_merge_result(
        paths,
        MergeOutcome(
            task_id="T01",
            branch="feat/t01-a",
            success=True,
            merge_commit_hash="def456",
            error=None,
            backend_name="snapshot_workspace",
            action="applied",
        ),
    )
    assert json.loads(merge_result_path.read_text(encoding="utf-8"))["action"] == "applied"
    assert list(merge_result_path.parent.glob(f".{merge_result_path.name}.*.tmp")) == []

    summary_path = _write_swarm_summary(
        paths=paths,
        backend_name="snapshot_workspace",
        base_branch="main",
        executed=["T01"],
        merge_outcomes=[],
        integration_results=[],
        replanning_results=[],
        skipped={},
        recovered={},
        startup_warnings=[],
        dry_run=False,
        schedule_preview=[["T01"]],
        workspace_summary_lines=["Workspace root: `repo`"],
        binding_summary_lines=["Requested Path: `repo`"],
    )
    summary_text = summary_path.read_text(encoding="utf-8")
    assert "# Swarm Summary" in summary_text
    assert "T01" in summary_text
    assert list(summary_path.parent.glob(f".{summary_path.name}.*.tmp")) == []

    report_path = repo / "report.md"
    report_path.write_text("# Task Report\n", encoding="utf-8")
    _append_remote_report_update(
        paths=paths,
        report_path_raw=os.fspath(report_path.relative_to(paths.root)),
        record={
            "provider": "github",
            "branch": "feat/t01-a",
            "branch_push_status": "pushed",
            "pr_status": "created",
            "pr_url": "https://example.invalid/pr/1",
        },
    )
    report_text = report_path.read_text(encoding="utf-8")
    assert "## Remote Sync Update" in report_text
    assert "provider=github" in report_text
    assert list(report_path.parent.glob(f".{report_path.name}.*.tmp")) == []


def test_run_swarm_setup_failure_preserves_caller_owned_lock(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    paths = create_plan_run(repo)
    guard = acquire_swarm_mutation_guard(paths, mode="forge_swarm:test")

    import sylliptor_agent_cli.swarm_orchestrator as swarm_orchestrator_mod

    monkeypatch.setattr(
        swarm_orchestrator_mod,
        "ensure_swarm_dirs",
        lambda _paths: (_ for _ in ()).throw(RuntimeError("boom-after-lock")),
    )

    with pytest.raises(RuntimeError, match="boom-after-lock"):
        run_swarm(
            paths=paths,
            plan={"run_id": paths.run_id, "tasks": []},
            cfg=AppConfig(model="test-model"),
            mode="auto",
            yes=False,
            max_steps=1,
            api_key_override="k",
            no_log=True,
            parallel=1,
            base_branch=None,
            max_tasks=None,
            max_attempts=None,
            dry_run=False,
            keep_worktrees=False,
            retry_failed=False,
            only=None,
            retry_merge_conflicts=False,
            console=Console(),
            run_mutation_guard=guard,
        )

    metadata = inspect_run_mutation_lock(paths.run_dir)
    assert metadata is not None
    assert metadata["owner_token"] == guard.owner_token

    guard.release()
    assert not (paths.run_dir / "active_execution.lock.json").exists()


def test_run_mutation_guard_recovers_definitely_stale_lock(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    lock_path = run_dir / "active_execution.lock.json"
    write_run_mutation_lock_metadata(
        lock_path,
        {
            "schema_version": 1,
            "run_id": "run-1",
            "mode": "forge_swarm",
            "kind": "lock",
            "pid": 9_999_999,
            "hostname": socket.gethostname(),
            "acquired_at": "2026-03-26T00:00:00+00:00",
            "owner_token": "stale-owner",
            "workspace_root": os.fspath(tmp_path),
            "run_dir": os.fspath(run_dir),
        },
    )

    with acquire_run_mutation_guard(
        run_id="run-1",
        mode="forge_swarm",
        run_dir=run_dir,
        workspace_root=tmp_path,
    ):
        metadata = inspect_run_mutation_lock(run_dir)
        assert metadata is not None
        assert metadata["run_id"] == "run-1"
        assert metadata["mode"] == "forge_swarm"
        assert metadata["pid"] == os.getpid()
        assert metadata["hostname"] == socket.gethostname()

    assert not lock_path.exists()


def test_run_mutation_guard_fails_closed_for_ambiguous_or_active_lock(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    lock_path = run_dir / "active_execution.lock.json"
    write_run_mutation_lock_metadata(
        lock_path,
        {
            "schema_version": 1,
            "run_id": "run-1",
            "mode": "forge_swarm",
            "kind": "lock",
            "pid": 9_999_999,
            "hostname": "different-host",
            "acquired_at": "2026-03-26T00:00:00+00:00",
            "owner_token": "ambiguous-owner",
            "workspace_root": os.fspath(tmp_path),
            "run_dir": os.fspath(run_dir),
        },
    )

    with pytest.raises(
        RunMutationConflictError,
        match="Another Forge execution is already mutating this run",
    ):
        acquire_run_mutation_guard(
            run_id="run-1",
            mode="forge_swarm",
            run_dir=run_dir,
            workspace_root=tmp_path,
        )

    assert inspect_run_mutation_lock(run_dir) is not None


def test_windows_stale_lock_probe_does_not_send_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    metadata = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
    }

    def fail_kill(_pid: int, _signal: int) -> None:
        raise AssertionError("Windows stale-lock probes must not call os.kill")

    monkeypatch.setattr(run_lock_mod.os, "name", "nt", raising=False)
    monkeypatch.setattr(run_lock_mod.os, "kill", fail_kill)
    monkeypatch.setattr(run_lock_mod, "_windows_process_is_running", lambda _pid: True)

    assert run_lock_mod._definitely_stale_reason(metadata) is None


def test_windows_stale_lock_probe_recovers_missing_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = {
        "pid": 9_999_999,
        "hostname": socket.gethostname(),
    }

    monkeypatch.setattr(run_lock_mod.os, "name", "nt", raising=False)
    monkeypatch.setattr(run_lock_mod, "_windows_process_is_running", lambda _pid: False)

    assert run_lock_mod._definitely_stale_reason(metadata) == "owner process is no longer running"


def test_run_mutation_guard_recovery_handoff_returns_conflict(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_dir = tmp_path / "run"
    lock_path = run_dir / "active_execution.lock.json"
    write_run_mutation_lock_metadata(
        lock_path,
        {
            "schema_version": 1,
            "run_id": "run-1",
            "mode": "forge_swarm",
            "kind": "lock",
            "pid": 9_999_999,
            "hostname": socket.gethostname(),
            "acquired_at": "2026-03-26T00:00:00+00:00",
            "owner_token": "stale-owner",
            "workspace_root": os.fspath(tmp_path),
            "run_dir": os.fspath(run_dir),
        },
    )

    original_write_exclusive = run_lock_mod._write_exclusive
    call_count = 0

    def fake_write_exclusive(path: Path, text: str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            payload = json.loads(text)
            payload["owner_token"] = "other-owner"
            write_run_mutation_lock_metadata(path, payload)
            raise FileExistsError(path)
        original_write_exclusive(path, text)

    monkeypatch.setattr(run_lock_mod, "_write_exclusive", fake_write_exclusive)

    with pytest.raises(
        RunMutationConflictError,
        match="another execution claimed the run while stale-lock recovery was finalizing",
    ):
        acquire_run_mutation_guard(
            run_id="run-1",
            mode="forge_swarm",
            run_dir=run_dir,
            workspace_root=tmp_path,
        )

    metadata = inspect_run_mutation_lock(run_dir)
    assert metadata is not None
    assert metadata["owner_token"] == "other-owner"
