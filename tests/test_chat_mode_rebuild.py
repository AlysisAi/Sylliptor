from __future__ import annotations

import io
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from rich.console import Console

import sylliptor_agent_cli.agent_loop as agent_loop_mod
import sylliptor_agent_cli.verify_gate as verify_gate_mod
from sylliptor_agent_cli import cli as cli_mod
from sylliptor_agent_cli.config import AppConfig, clone_cfg
from sylliptor_agent_cli.session_store import SessionStore
from sylliptor_agent_cli.verify_gate import VerifyError


def _store(root: Path, *, session_id: str = "chat-rebuild-test") -> SessionStore:
    return SessionStore(
        enabled=False,
        sessions_dir=root / "sessions",
        session_id=session_id,
        cwd=str(root),
        repo_root=str(root),
    )


class _FakeMcpBinding:
    def __init__(self, *, session_mode: str | None = None) -> None:
        self.tool_alias = "mcp__alpha__echo"
        self.description = "Echo via MCP"
        self.parameters = {"type": "object", "properties": {}, "required": []}
        self.session_mode = session_mode

    def bind_session_mode(self, session_mode: str | None) -> _FakeMcpBinding:
        return _FakeMcpBinding(session_mode=str(session_mode or "").strip().lower() or None)

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "tool_alias": self.tool_alias,
            "session_mode": self.session_mode,
            "arguments": dict(arguments),
        }


class _DummyMcpManager:
    def __init__(self, *bindings: _FakeMcpBinding) -> None:
        self.tool_bindings = tuple(bindings)


class _FakeShellRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(
        self, *, root: Path, cwd: Path, cmd: str, timeout_s: int
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(
            {
                "root": root,
                "cwd": cwd,
                "cmd": cmd,
                "timeout_s": timeout_s,
            }
        )
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="runner ok\n",
            stderr="",
        )


def _host_verify_cfg(cfg: AppConfig) -> AppConfig:
    effective = clone_cfg(cfg)
    extra_fields = dict(effective.extra_fields)
    verify_sandbox = dict(extra_fields.get("verify_sandbox") or {})
    verify_sandbox.setdefault("mode", "off")
    extra_fields["verify_sandbox"] = verify_sandbox
    effective.extra_fields = extra_fields
    return effective


def _make_session(tmp_path: Path, **overrides: Any) -> SimpleNamespace:
    cfg = _host_verify_cfg(AppConfig(model="test-model", web_search_mode="off"))
    defaults: dict[str, Any] = {
        "cfg": cfg,
        "root": tmp_path,
        "mode": "review",
        "yes": True,
        "max_steps": 4,
        "console": Console(file=io.StringIO(), force_terminal=False),
        "surface": None,
        "store": _store(tmp_path),
        "api_key": "",
        "shell_runner": None,
        "no_log": True,
        "non_interactive": True,
        "one_shot_execution": False,
        "verification_enabled": True,
        "effective_verification_commands": [],
        "authoritative_verification_commands": None,
        "deny_write_prefixes": None,
        "allow_write_globs": None,
        "usage_summary": None,
        "usage_role": "main",
        "model_registry": None,
        "subagents_enabled": False,
        "subagent_depth": 0,
        "subagent_registry": {},
        "session_log_dir_override": None,
        "step_budget_runtime": None,
        "runtime_kind": "interactive_chat",
        "mcp_manager": None,
        "tools": {},
        "tool_list": [],
    }
    defaults.update(overrides)
    defaults["cfg"] = _host_verify_cfg(defaults["cfg"])
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize("mode", ["review", "auto", "fullaccess"])
def test_rebuild_session_tools_preserves_mcp_bindings_in_write_capable_modes(
    tmp_path: Path,
    mode: str,
) -> None:
    session = _make_session(
        tmp_path,
        mcp_manager=_DummyMcpManager(_FakeMcpBinding()),
    )
    try:
        cli_mod._rebuild_session_tools_for_mode(session=session, mode=mode)

        assert "mcp__alpha__echo" in session.tools
    finally:
        session.store.close()


def test_rebuild_session_tools_preserves_verification_disabled_flag(tmp_path: Path) -> None:
    session = _make_session(tmp_path, verification_enabled=False)
    try:
        cli_mod._rebuild_session_tools_for_mode(session=session, mode="review")

        assert "verify_run" not in session.tools
    finally:
        session.store.close()


def test_rebuild_session_tools_preserves_effective_verification_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []

    def fake_run(cmd, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(verify_gate_mod.subprocess, "run", fake_run)

    cfg = AppConfig(model="test-model", web_search_mode="off")
    cfg.verify_commands = ["ruff check ."]
    session = _make_session(
        tmp_path,
        cfg=cfg,
        effective_verification_commands=["pytest -q"],
    )
    try:
        cli_mod._rebuild_session_tools_for_mode(session=session, mode="auto")
        result = session.tools["verify_run"].run({})

        # Verification commands execute as shell strings; the verify pipeline
        # may additionally shell out to git (argv lists) for its internal
        # workspace scans, which must not replace or add verify commands.
        executed_verify_commands = [cmd for cmd in calls if isinstance(cmd, str)]
        internal_calls = [cmd for cmd in calls if not isinstance(cmd, str)]
        assert executed_verify_commands == ["pytest -q"]
        assert all(cmd and cmd[0] == "git" for cmd in internal_calls)
        assert result["commands"] == ["pytest -q"]
    finally:
        session.store.close()


def test_rebuild_session_tools_preserves_authoritative_verification_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        agent_loop_mod,
        "run_task_verification",
        lambda **_kwargs: pytest.fail("verify engine should not run for rejected overrides"),
    )

    cfg = AppConfig(model="test-model", web_search_mode="off")
    cfg.verify_commands = ["ruff check ."]
    session = _make_session(
        tmp_path,
        cfg=cfg,
        authoritative_verification_commands=["pytest -q"],
    )
    try:
        cli_mod._rebuild_session_tools_for_mode(session=session, mode="review")

        with pytest.raises(
            VerifyError,
            match="Managed verification commands are locked to the authoritative Forge command set.",
        ):
            session.tools["verify_run"].run({"commands": ["ruff check ."]})
    finally:
        session.store.close()


def test_rebuild_session_tools_preserves_custom_shell_runner(tmp_path: Path) -> None:
    runner = _FakeShellRunner()
    session = _make_session(tmp_path, shell_runner=runner)
    try:
        cli_mod._rebuild_session_tools_for_mode(session=session, mode="auto")
        result = session.tools["shell_run"].run({"cmd": "echo hi"})

        assert result["stdout"] == "runner ok\n"
        assert runner.calls
        assert runner.calls[0]["cmd"] == "echo hi"
        assert runner.calls[0]["root"] == tmp_path.resolve()
    finally:
        session.store.close()


def test_rebuild_session_tools_preserves_active_workdir_defaults(tmp_path: Path) -> None:
    runner = _FakeShellRunner()
    nested = tmp_path / "packages" / "app"
    nested.mkdir(parents=True, exist_ok=True)
    session = _make_session(
        tmp_path,
        shell_runner=runner,
        focus_dir=tmp_path,
        focus_relpath=".",
        workspace_kind="plain_dir",
        active_workdir_relpath="packages/app",
    )
    try:
        cli_mod._rebuild_session_tools_for_mode(session=session, mode="auto")
        session.tools["shell_run"].run({"cmd": "echo hi"})

        assert runner.calls
        assert runner.calls[0]["cwd"] == nested.resolve()
    finally:
        session.store.close()
