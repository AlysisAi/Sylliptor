from __future__ import annotations

import io
from pathlib import Path

import pytest
from rich.console import Console

from sylliptor_agent_cli.agent_loop import AgentRuntimeError, build_tools
from sylliptor_agent_cli.session_store import SessionStore
from sylliptor_agent_cli.surface import ApprovalDecision, ApprovalRequest
from sylliptor_agent_cli.surface.hidden_surface import HiddenApprovalSurface
from sylliptor_agent_cli.surface.noop_surface import NoopSurface


def _store(root: Path) -> SessionStore:
    return SessionStore(
        enabled=False,
        sessions_dir=root / "sessions",
        session_id="review-test",
        cwd=str(root),
        repo_root=str(root),
    )


def test_review_mode_fs_write_requires_approval_and_does_not_write(tmp_path: Path) -> None:
    tools = build_tools(
        root=tmp_path,
        console=Console(file=io.StringIO()),
        surface=NoopSurface(),
        store=_store(tmp_path),
        mode="review",
        yes=False,
    )

    with pytest.raises(AgentRuntimeError, match="User declined: fs_write"):
        tools["fs_write"].run({"path": "blocked.txt", "content": "x"})

    assert (tmp_path / "blocked.txt").exists() is False


def test_review_mode_fs_edit_requires_approval_and_does_not_write(tmp_path: Path) -> None:
    target = tmp_path / "blocked.txt"
    target.write_text("alpha\n", encoding="utf-8")
    tools = build_tools(
        root=tmp_path,
        console=Console(file=io.StringIO()),
        surface=NoopSurface(),
        store=_store(tmp_path),
        mode="review",
        yes=False,
    )

    with pytest.raises(AgentRuntimeError, match="User declined: fs_edit"):
        tools["fs_edit"].run(
            {
                "path": "blocked.txt",
                "edits": [{"op": "replace_exact", "target": "alpha", "replacement": "beta"}],
            }
        )

    assert target.read_text(encoding="utf-8") == "alpha\n"


def test_review_mode_fs_delete_requires_approval_and_does_not_delete(tmp_path: Path) -> None:
    target = tmp_path / "blocked.txt"
    target.write_text("alpha\n", encoding="utf-8")
    tools = build_tools(
        root=tmp_path,
        console=Console(file=io.StringIO()),
        surface=NoopSurface(),
        store=_store(tmp_path),
        mode="review",
        yes=False,
        non_interactive=False,
    )

    with pytest.raises(AgentRuntimeError, match="User declined: fs_delete"):
        tools["fs_delete"].run({"path": "blocked.txt"})

    assert target.exists() is True
    assert target.read_text(encoding="utf-8") == "alpha\n"


def test_review_mode_shell_run_requires_approval(tmp_path: Path) -> None:
    tools = build_tools(
        root=tmp_path,
        console=Console(file=io.StringIO()),
        surface=NoopSurface(),
        store=_store(tmp_path),
        mode="review",
        yes=False,
    )

    with pytest.raises(AgentRuntimeError, match="User declined: shell_run"):
        tools["shell_run"].run({"cmd": "echo hi"})


def test_review_mode_non_interactive_can_use_host_managed_approval(tmp_path: Path) -> None:
    class HostApprovalSurface(NoopSurface):
        host_managed_approvals = True

        def __init__(self) -> None:
            self.requests: list[ApprovalRequest] = []

        def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
            self.requests.append(request)
            return ApprovalDecision(allow=True)

    surface = HostApprovalSurface()
    tools = build_tools(
        root=tmp_path,
        console=Console(file=io.StringIO()),
        surface=surface,
        store=_store(tmp_path),
        mode="review",
        yes=False,
        non_interactive=True,
    )

    tools["fs_write"].run({"path": "allowed.txt", "content": "x"})

    assert (tmp_path / "allowed.txt").read_text(encoding="utf-8") == "x"
    assert len(surface.requests) == 1
    assert surface.requests[0].kind == "fs_write"


def test_review_mode_fs_write_escape_path_rejected_before_approval(tmp_path: Path) -> None:
    class RecordingSurface(NoopSurface):
        def __init__(self) -> None:
            self.approval_calls = 0
            self.patch_events = 0

        def request_approval(self, request):  # type: ignore[no-untyped-def]
            self.approval_calls += 1
            return super().request_approval(request)

        def on_patch_generated(self, event):  # type: ignore[no-untyped-def]
            self.patch_events += 1
            super().on_patch_generated(event)

    surface = RecordingSurface()
    tools = build_tools(
        root=tmp_path,
        console=Console(file=io.StringIO()),
        surface=surface,
        store=_store(tmp_path),
        mode="review",
        yes=False,
    )

    with pytest.raises(AgentRuntimeError, match="Path escapes root"):
        tools["fs_write"].run({"path": "../evil.txt", "content": "x"})

    assert surface.approval_calls == 0
    assert surface.patch_events == 0


def test_hidden_approval_surface_forwards_only_approval_requests() -> None:
    class RecordingSurface(NoopSurface):
        def __init__(self) -> None:
            self.approval_calls = 0
            self.progress_updates = 0
            self.errors = 0

        def on_progress_update(self, message: str) -> None:
            _ = message
            self.progress_updates += 1

        def on_error(self, err: str) -> None:
            _ = err
            self.errors += 1

        def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
            self.approval_calls += 1
            assert request.kind == "fs_write"
            return ApprovalDecision(allow=True)

    parent_surface = RecordingSurface()
    hidden_surface = HiddenApprovalSurface(parent_surface)

    decision = hidden_surface.request_approval(
        ApprovalRequest(
            kind="fs_write",
            reason="nested review write",
            preview="write file",
            files=["demo.txt"],
        )
    )
    hidden_surface.on_progress_update("nested progress")
    hidden_surface.on_error("nested error")

    assert decision.allow is True
    assert parent_surface.approval_calls == 1
    assert parent_surface.progress_updates == 0
    assert parent_surface.errors == 0
